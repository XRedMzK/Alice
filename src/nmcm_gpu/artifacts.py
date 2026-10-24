"""Read-only binding to the accepted FOUNDATION_0 CPU v2 artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from nmcm.checkpoint import load_module_shard, load_whole_checkpoint
from nmcm.compatibility import BehavioralDecision, validate_behavioral_decision
from nmcm.neural import NeuralModule, TensorMap
from nmcm.synthetic import (
    GatedRecurrentA2,
    IncompatibleA3,
    RecurrentA1,
    RecurrentB,
    RecurrentC,
    SyntheticSplit,
    Teacher,
    build_splits,
    build_teacher,
)


MODULE_NAMES = ("a1", "a2", "a3", "b", "c")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_module(name: str, teacher: Teacher) -> NeuralModule:
    factories = {
        "a1": lambda: RecurrentA1(teacher),
        "a2": lambda: GatedRecurrentA2(teacher),
        "a3": lambda: IncompatibleA3(RecurrentA1(teacher)),
        "b": RecurrentB,
        "c": RecurrentC,
    }
    try:
        return factories[name]()
    except KeyError as error:
        raise KeyError(f"unknown accepted module {name}") from error


@dataclass(frozen=True)
class AcceptedCpuArtifacts:
    root: Path
    profile: dict
    cpu_results: dict
    teacher: Teacher
    splits: dict[str, SyntheticSplit]
    modules: dict[str, NeuralModule]
    shard_states: dict[str, TensorMap]
    shard_records: dict[str, dict]
    behavioral_decisions: dict[str, BehavioralDecision]

    def clone_module(self, name: str) -> NeuralModule:
        return copy.deepcopy(self.modules[name])


def _verify_bound_file(root: Path, path: str, expected: str) -> None:
    target = root / path
    if not target.is_file() or sha256_file(target) != expected:
        raise RuntimeError(f"accepted CPU binding mismatch: {path}")


def load_accepted_cpu_artifacts(root: str | Path, profile: Mapping) -> AcceptedCpuArtifacts:
    root_path = Path(root).resolve()
    cpu = profile["accepted_cpu_baseline"]
    for path_key, digest_key in (
        ("acceptance_path", "acceptance_sha256"),
        ("results_path", "results_sha256"),
        ("junit_path", "junit_sha256"),
    ):
        _verify_bound_file(root_path, cpu[path_key], cpu[digest_key])
    cpu_results = json.loads((root_path / cpu["results_path"]).read_text(encoding="utf-8"))
    if cpu_results.get("status") != "PASS":
        raise RuntimeError("accepted CPU result is not PASS")
    if cpu_results.get("baseline", {}).get("baseline_id") != cpu["baseline_id"]:
        raise RuntimeError("accepted CPU baseline identity mismatch")

    teacher = build_teacher()
    splits = build_splits(teacher)
    records = cpu_results["module_shards"]["records"]
    if set(records) != set(MODULE_NAMES):
        raise RuntimeError("accepted CPU result does not contain exactly five shards")
    shard_root = root_path / profile["checkpoint_inputs"]["module_shards_root"]
    modules: dict[str, NeuralModule] = {}
    states: dict[str, TensorMap] = {}
    for name in MODULE_NAMES:
        module = fresh_module(name, teacher)
        state = load_module_shard(shard_root, records[name], module)
        modules[name] = module
        states[name] = state

    decisions = {
        name: BehavioralDecision(**payload)
        for name, payload in cpu_results["behavioral_decisions"].items()
    }
    for decision in decisions.values():
        validate_behavioral_decision(decision)
    return AcceptedCpuArtifacts(
        root=root_path,
        profile=dict(profile),
        cpu_results=cpu_results,
        teacher=teacher,
        splits=splits,
        modules=modules,
        shard_states=states,
        shard_records=records,
        behavioral_decisions=decisions,
    )


def module_input_port(name: str) -> str:
    return "stimulus" if name in {"a1", "a2", "a3"} else "dense_latent"


def accepted_inputs(artifacts: AcceptedCpuArtifacts, *, batch: int = 64) -> dict[str, Tensor]:
    stimulus = artifacts.splits["heldout"].stimulus[:batch].contiguous()
    a1 = artifacts.clone_module("a1").cpu()
    with torch.no_grad():
        result = a1.forward_sequence(
            {"stimulus": stimulus}, a1.initial_state(batch, device="cpu")
        )
    latent = result.tensors["dense_latent"].detach().cpu().contiguous()
    return {
        "a1": stimulus,
        "a2": stimulus,
        "a3": stimulus,
        "b": latent,
        "c": latent,
    }


def load_fresh_shard(
    artifacts: AcceptedCpuArtifacts, name: str
) -> tuple[NeuralModule, TensorMap]:
    module = fresh_module(name, artifacts.teacher)
    state = load_module_shard(
        artifacts.root / artifacts.profile["checkpoint_inputs"]["module_shards_root"],
        artifacts.shard_records[name],
        module,
    )
    return module, state


def load_accepted_whole_checkpoint(
    artifacts: AcceptedCpuArtifacts,
) -> tuple[dict[str, NeuralModule], dict[str, TensorMap], dict]:
    modules: dict[str, NeuralModule] = {
        "a": RecurrentA1(artifacts.teacher),
        "b": RecurrentB(),
        "c": RecurrentC(),
    }
    checkpoint = artifacts.root / artifacts.profile["checkpoint_inputs"]["whole_checkpoint"]
    states, manifest = load_whole_checkpoint(
        checkpoint,
        graph_id="foundation_a1bc",
        graph_version="2.0.0",
        modules=modules,
    )
    return modules, states, manifest


def tensor_map_exact(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left
    )


def tensor_map_close(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor], *, atol: float, rtol: float
) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        actual = left[name].detach().cpu()
        expected = right[name].detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            return False
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            return False
    return True
