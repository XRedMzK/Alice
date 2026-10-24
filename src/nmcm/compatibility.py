"""Behavioral provider admission kept separate from structural graph validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor

from .neural import NeuralModule
from .synthetic import accuracy, latent_metrics, module_digest


@dataclass(frozen=True)
class BehavioralDecision:
    module_id: str
    module_version: str
    role: str
    admitted: bool
    negative_control_pass: bool
    criteria: dict[str, bool]
    metrics: dict[str, float]
    provider_artifact_digest: str
    consumer_artifact_digest: str
    fixture_id: str
    fixture_digest: str
    thresholds_digest: str
    evidence_digest: str


def _decision_digest(
    *,
    module_id: str,
    module_version: str,
    role: str,
    criteria: Mapping[str, bool],
    metrics: Mapping[str, float],
    provider_artifact_digest: str,
    consumer_artifact_digest: str,
    fixture_id: str,
    fixture_digest: str,
    thresholds_digest: str,
) -> str:
    digest_payload = {
        "module_id": module_id,
        "module_version": module_version,
        "role": role,
        "criteria": dict(criteria),
        "metrics": dict(metrics),
        "provider_artifact_digest": provider_artifact_digest,
        "consumer_artifact_digest": consumer_artifact_digest,
        "fixture_id": fixture_id,
        "fixture_digest": fixture_digest,
        "thresholds_digest": thresholds_digest,
    }
    return hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_behavioral_decision(decision: BehavioralDecision) -> None:
    if decision.role not in {"baseline", "replacement", "negative_control"}:
        raise ValueError("behavioral decision has an unknown role")
    if not decision.criteria or not all(
        isinstance(value, bool) for value in decision.criteria.values()
    ):
        raise ValueError("behavioral decision criteria must be non-empty booleans")
    if not decision.metrics or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in decision.metrics.values()
    ):
        raise ValueError("behavioral decision metrics must be finite numbers")
    expected_digest = _decision_digest(
        module_id=decision.module_id,
        module_version=decision.module_version,
        role=decision.role,
        criteria=decision.criteria,
        metrics=decision.metrics,
        provider_artifact_digest=decision.provider_artifact_digest,
        consumer_artifact_digest=decision.consumer_artifact_digest,
        fixture_id=decision.fixture_id,
        fixture_digest=decision.fixture_digest,
        thresholds_digest=decision.thresholds_digest,
    )
    if decision.evidence_digest != expected_digest:
        raise ValueError("behavioral decision digest mismatch")
    for name in (
        "provider_artifact_digest",
        "consumer_artifact_digest",
        "fixture_digest",
        "thresholds_digest",
    ):
        value = getattr(decision, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"behavioral decision has invalid {name}")
    if not decision.fixture_id:
        raise ValueError("behavioral decision fixture ID must be non-empty")
    admission_keys = {
        "local_cosine",
        "local_nrmse",
        "downstream_accuracy",
    }
    if decision.role != "baseline":
        admission_keys.add("replacement_difference")
    if not admission_keys.issubset(decision.criteria):
        raise ValueError("behavioral decision omits mandatory admission criteria")
    expected_admitted = all(decision.criteria[key] for key in admission_keys)
    if decision.admitted != expected_admitted:
        raise ValueError("behavioral admission flag contradicts its criteria")
    expected_negative = (
        decision.role == "negative_control"
        and not expected_admitted
        and decision.criteria.get("required_accuracy_drop") is True
    )
    if decision.negative_control_pass != expected_negative:
        raise ValueError("negative-control flag contradicts its criteria")


def evaluate_provider(
    *,
    module: NeuralModule,
    role: str,
    stimulus: Tensor,
    canonical_latent: Tensor,
    consumer: NeuralModule,
    reference_logits: Tensor | None,
    labels: Tensor,
    thresholds: Mapping[str, float],
    fixture_id: str,
) -> BehavioralDecision:
    """Run and bind the real provider/consumer artifacts before applying gates."""

    if role not in {"baseline", "replacement", "negative_control"}:
        raise ValueError(f"unsupported behavioral role {role}")
    if not fixture_id:
        raise ValueError("behavioral fixture ID must be non-empty")
    with torch.no_grad():
        candidate_run = module.forward_sequence(
            {"stimulus": stimulus}, module.initial_state(stimulus.shape[0])
        )
        candidate_latent = candidate_run.tensors["dense_latent"]
        consumer_run = consumer.forward_sequence(
            {"dense_latent": candidate_latent},
            consumer.initial_state(candidate_latent.shape[0]),
        )
        candidate_logits = consumer_run.tensors["task_logits"]
    if reference_logits is None:
        reference_logits = candidate_logits
    local = latent_metrics(candidate_latent, canonical_latent)
    candidate_accuracy = accuracy(candidate_logits, labels)
    reference_accuracy = accuracy(reference_logits, labels)
    metrics = {
        **local,
        "candidate_accuracy": candidate_accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_difference": abs(candidate_accuracy - reference_accuracy),
        "accuracy_drop": reference_accuracy - candidate_accuracy,
    }
    criteria = {
        "local_cosine": local["mean_cosine"]
        >= thresholds["a1_a2_mean_cosine_min"],
        "local_nrmse": local["normalized_rmse"]
        <= thresholds["a1_a2_normalized_rmse_max"],
    }
    if role == "baseline":
        criteria["downstream_accuracy"] = (
            candidate_accuracy >= thresholds["b_a1_accuracy_min"]
        )
        admitted = all(criteria.values())
        negative_pass = False
    else:
        criteria["downstream_accuracy"] = (
            candidate_accuracy >= thresholds["b_a2_accuracy_min"]
        )
        criteria["replacement_difference"] = (
            abs(candidate_accuracy - reference_accuracy)
            <= thresholds["b_a1_a2_accuracy_absolute_difference_max"]
        )
        admitted = all(criteria.values())
        if role == "negative_control":
            criteria["required_accuracy_drop"] = (
                reference_accuracy - candidate_accuracy
                >= thresholds["a3_b_accuracy_drop_min"]
            )
            negative_pass = not admitted and criteria["required_accuracy_drop"]
        else:
            negative_pass = False
    provider_digest = module_digest(module)
    consumer_digest = module_digest(consumer)
    fixture_hasher = hashlib.sha256()
    for name, tensor in (
        ("stimulus", stimulus),
        ("canonical_latent", canonical_latent),
        ("reference_logits", reference_logits),
        ("labels", labels),
    ):
        contiguous = tensor.detach().cpu().contiguous()
        fixture_hasher.update(name.encode("utf-8"))
        fixture_hasher.update(str(contiguous.dtype).encode("ascii"))
        fixture_hasher.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        fixture_hasher.update(contiguous.view(torch.uint8).numpy().tobytes())
    fixture_digest = fixture_hasher.hexdigest()
    thresholds_digest = hashlib.sha256(
        json.dumps(dict(thresholds), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    evidence_digest = _decision_digest(
        module_id=module.module_id,
        module_version=module.module_version,
        role=role,
        criteria=criteria,
        metrics=metrics,
        provider_artifact_digest=provider_digest,
        consumer_artifact_digest=consumer_digest,
        fixture_id=fixture_id,
        fixture_digest=fixture_digest,
        thresholds_digest=thresholds_digest,
    )
    decision = BehavioralDecision(
        module_id=module.module_id,
        module_version=module.module_version,
        role=role,
        admitted=admitted,
        negative_control_pass=negative_pass,
        criteria=criteria,
        metrics=metrics,
        provider_artifact_digest=provider_digest,
        consumer_artifact_digest=consumer_digest,
        fixture_id=fixture_id,
        fixture_digest=fixture_digest,
        thresholds_digest=thresholds_digest,
        evidence_digest=evidence_digest,
    )
    validate_behavioral_decision(decision)
    return decision


def decision_payload(decision: BehavioralDecision) -> dict:
    return asdict(decision)
