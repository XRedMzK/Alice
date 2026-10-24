"""Materialize compact FOUNDATION_0 CPU acceptance evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import (
    load_module_shard,
    load_whole_checkpoint,
    save_module_shard,
    save_whole_checkpoint,
)
from .compatibility import decision_payload, evaluate_provider
from .concurrency import run_two_candidate_processes
from .foundation import build_trained_bundle
from .graph import GraphDefinition, GraphRunner
from .synthetic import (
    GatedRecurrentA2,
    IncompatibleA3,
    RecurrentA1,
    RecurrentB,
    RecurrentC,
    accuracy,
    latent_distribution,
    latent_metrics,
    module_digest,
    run_a,
    run_b,
    tensor_map_equal,
)
from .telemetry import TelemetryCollector


EXPECTED_TEST_NAMES = (
    "test_01_manifest_and_schema_validation",
    "test_02_registry_graph_and_semantics_blind_routing",
    "test_03_private_state_ownership_and_isolation",
    "test_04_a1_local_learning",
    "test_05_a2_local_learning_and_interface",
    "test_06_a1_to_b_task_and_freeze",
    "test_07_frozen_b_a1_to_a2_replacement",
    "test_08_a3_structural_positive_behavioral_negative",
    "test_09_targeted_ablation_with_non_target_guard",
    "test_10_add_remove_c_without_hidden_mutation",
    "test_11_independent_module_shards_and_standalone_restore",
    "test_12_whole_checkpoint_exact_continuation_and_fail_closed",
    "test_13_telemetry_noninterference_and_schema",
    "test_14_full_serial_training_determinism",
    "test_15_two_concurrent_isolated_candidate_bundles",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_facts(root: Path) -> dict[str, Any]:
    def invoke(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()

    try:
        return {
            "revision": invoke("rev-parse", "HEAD"),
            "branch": invoke("branch", "--show-current"),
            "working_tree_dirty": bool(invoke("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def _evaluator_binding_files(root: Path) -> set[Path]:
    files = set((root / "src/nmcm").glob("*.py"))
    files.update((root / "tests").glob("*.py"))
    files.update((root / "configs").glob("*.json"))
    files.update((root / "configs/modules").glob("*.json"))
    files.update((root / "configs/graphs").glob("*.json"))
    files.discard(root / "configs/foundation0_cpu_acceptance.json")
    for relative in (
        "temp.md",
        "pyproject.toml",
        "requirements-cpu.in",
        "requirements-cpu.lock",
    ):
        files.add(root / relative)
    return {path for path in files if path.is_file()}


def _binding_digests(root: Path, files: set[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path) for path in sorted(files)
    }


def _junit_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "NOT_PROVIDED",
            "expected_test_names": list(EXPECTED_TEST_NAMES),
        }
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    elapsed = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
    cases = []
    for suite in suites:
        for case in suite.findall("testcase"):
            cases.append(
                {
                    "name": case.attrib.get("name"),
                    "classname": case.attrib.get("classname"),
                    "time_seconds": float(case.attrib.get("time", 0.0)),
                    "status": "PASS"
                    if not list(case.findall("failure"))
                    and not list(case.findall("error"))
                    and not list(case.findall("skipped"))
                    else "NON_PASS",
                }
            )
    names = [case["name"] for case in cases]
    expected = set(EXPECTED_TEST_NAMES)
    actual = set(names)
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    classnames_exact = all(
        case["classname"] == "tests.test_foundation0_cpu" for case in cases
    )
    exact_suite = (
        tests == len(EXPECTED_TEST_NAMES)
        and len(cases) == len(EXPECTED_TEST_NAMES)
        and actual == expected
        and not duplicate_names
        and classnames_exact
        and failures == 0
        and errors == 0
        and skipped == 0
        and all(case["status"] == "PASS" for case in cases)
    )
    return {
        "status": "PASS" if exact_suite else "FAIL",
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "time_seconds": elapsed,
        "cases": cases,
        "expected_test_names": list(EXPECTED_TEST_NAMES),
        "missing_test_names": sorted(expected - actual),
        "unexpected_test_names": sorted(actual - expected),
        "duplicate_test_names": duplicate_names,
        "classnames_exact": classnames_exact,
        "path": str(path),
        "sha256": _sha256(path),
    }


def qualify(root: Path, output_root: Path, junit_path: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    binding_files_before = _evaluator_binding_files(root)
    binding_digests_before = _binding_digests(root, binding_files_before)
    baseline_path = root / "configs/foundation0_cpu.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    thresholds = baseline["thresholds"]
    junit = _junit_summary(junit_path)
    junit_cases = {case["name"]: case["status"] for case in junit.get("cases", [])}

    bundle = build_trained_bundle()
    heldout = bundle.splits["heldout"]
    with torch.no_grad():
        z_a1, _ = run_a(bundle.a1, heldout)
        z_a2, _ = run_a(bundle.a2, heldout)
        z_a3, _ = run_a(bundle.a3, heldout)
        logits_a1, _ = run_b(bundle.b, z_a1)
        logits_a2, _ = run_b(bundle.b, z_a2)
        logits_a3, _ = run_b(bundle.b, z_a3)
        logits_zero, _ = run_b(bundle.b, torch.zeros_like(z_a1))
    accuracy_a1 = accuracy(logits_a1, heldout.labels)
    accuracy_a2 = accuracy(logits_a2, heldout.labels)
    accuracy_a3 = accuracy(logits_a3, heldout.labels)
    accuracy_zero = accuracy(logits_zero, heldout.labels)
    a1_metrics = latent_metrics(z_a1, heldout.latent)
    a2_metrics = latent_metrics(z_a2, heldout.latent)
    a3_metrics = latent_metrics(z_a3, heldout.latent)
    distribution_a1 = latent_distribution(z_a1)
    distribution_a3 = latent_distribution(z_a3)

    decisions = {
        "a1": evaluate_provider(
            module=bundle.a1,
            role="baseline",
            stimulus=heldout.stimulus,
            canonical_latent=heldout.latent,
            consumer=bundle.b,
            reference_logits=None,
            labels=heldout.labels,
            thresholds=thresholds,
            fixture_id="foundation0-heldout-seed4203/v1",
        ),
        "a2": evaluate_provider(
            module=bundle.a2,
            role="replacement",
            stimulus=heldout.stimulus,
            canonical_latent=heldout.latent,
            consumer=bundle.b,
            reference_logits=logits_a1,
            labels=heldout.labels,
            thresholds=thresholds,
            fixture_id="foundation0-heldout-seed4203/v1",
        ),
        "a3": evaluate_provider(
            module=bundle.a3,
            role="negative_control",
            stimulus=heldout.stimulus,
            canonical_latent=heldout.latent,
            consumer=bundle.b,
            reference_logits=logits_a1,
            labels=heldout.labels,
            thresholds=thresholds,
            fixture_id="foundation0-heldout-seed4203/v1",
        ),
    }

    lesioned_b = copy.deepcopy(bundle.b)
    with torch.no_grad():
        lesioned_b.input_projection.weight.zero_()
        lesioned_b.recurrence.weight.zero_()
        logits_b_core_lesion, _ = run_b(lesioned_b, z_a1)
        b_prefix = bundle.b.forward_sequence(
            {"dense_latent": z_a1[:64, :4].contiguous()}, bundle.b.initial_state(64)
        )
        b_carried = bundle.b.forward_sequence(
            {"dense_latent": z_a1[:64, 4:].contiguous()}, b_prefix.state
        )
        b_reset = bundle.b.forward_sequence(
            {"dense_latent": z_a1[:64, 4:].contiguous()}, bundle.b.initial_state(64)
        )
        logits_zero_a2, _ = run_b(bundle.b, torch.zeros_like(z_a2))
    b_core_lesion_drop = accuracy_a1 - accuracy(logits_b_core_lesion, heldout.labels)
    a2_ablation_drop = accuracy_a2 - accuracy(logits_zero_a2, heldout.labels)
    b_state_is_causal = not torch.equal(
        b_carried.tensors["task_logits"], b_reset.tensors["task_logits"]
    )

    batch = 64
    stimulus = heldout.stimulus[:batch]
    graph_ab = GraphDefinition.from_path(root / "configs/graphs/foundation_a1b.json")
    graph_abc = GraphDefinition.from_path(root / "configs/graphs/foundation_a1bc.json")
    modules_ab = {"a": copy.deepcopy(bundle.a1), "b": copy.deepcopy(bundle.b)}
    modules_abc = {
        "a": copy.deepcopy(bundle.a1),
        "b": copy.deepcopy(bundle.b),
        "c": copy.deepcopy(bundle.c),
    }
    runner_ab = GraphRunner(
        graph_ab,
        modules_ab,
        telemetry=TelemetryCollector(
            enabled=False, experiment_id="qualify-ab", snapshot_id="q1"
        ),
        behavioral_decisions={"a": decisions["a1"]},
    )
    runner_abc = GraphRunner(
        graph_abc,
        modules_abc,
        telemetry=TelemetryCollector(
            enabled=False, experiment_id="qualify-abc", snapshot_id="q1"
        ),
        behavioral_decisions={"a": decisions["a1"]},
    )
    states_ab = {key: module.initial_state(batch) for key, module in modules_ab.items()}
    states_abc = {
        key: module.initial_state(batch) for key, module in modules_abc.items()
    }
    with torch.no_grad():
        result_ab = runner_ab.run({"x": stimulus}, states_ab)
        result_abc = runner_abc.run({"x": stimulus}, states_abc)
    extension_exact = (
        torch.equal(
            result_ab.outputs["dense_latent"], result_abc.outputs["dense_latent"]
        )
        and torch.equal(
            result_ab.outputs["task_logits"], result_abc.outputs["task_logits"]
        )
        and tensor_map_equal(result_ab.states["a"], result_abc.states["a"])
        and tensor_map_equal(result_ab.states["b"], result_abc.states["b"])
        and module_digest(modules_ab["a"]) == module_digest(modules_abc["a"])
        and module_digest(modules_ab["b"]) == module_digest(modules_abc["b"])
    )
    c_probe = copy.deepcopy(bundle.c)
    with torch.no_grad():
        c_intact = c_probe.forward_sequence(
            {"dense_latent": z_a1[:batch]}, c_probe.initial_state(batch)
        )
        c_lesioned = c_probe.forward_sequence(
            {"dense_latent": torch.zeros_like(z_a1[:batch])},
            c_probe.initial_state(batch),
        )
        c_prefix = c_probe.forward_sequence(
            {"dense_latent": z_a1[:batch, :4].contiguous()},
            c_probe.initial_state(batch),
        )
        c_carried = c_probe.forward_sequence(
            {"dense_latent": z_a1[:batch, 4:].contiguous()}, c_prefix.state
        )
        c_reset = c_probe.forward_sequence(
            {"dense_latent": z_a1[:batch, 4:].contiguous()},
            c_probe.initial_state(batch),
        )
    c_input_is_causal = not torch.equal(
        c_intact.tensors["side_latent"], c_lesioned.tensors["side_latent"]
    )
    c_state_is_causal = not torch.equal(
        c_carried.tensors["side_latent"], c_reset.tensors["side_latent"]
    )

    shard_root = output_root / "module_shards"
    module_shard_records: dict[str, dict[str, Any]] = {}
    module_shard_exact: dict[str, bool] = {}
    shard_modules = {
        "a1": copy.deepcopy(bundle.a1),
        "a2": copy.deepcopy(bundle.a2),
        "a3": copy.deepcopy(bundle.a3),
        "b": copy.deepcopy(bundle.b),
        "c": copy.deepcopy(bundle.c),
    }
    shard_inputs = {
        "a1": ("stimulus", stimulus),
        "a2": ("stimulus", stimulus),
        "a3": ("stimulus", stimulus),
        "b": ("dense_latent", z_a1[:batch]),
        "c": ("dense_latent", z_a1[:batch]),
    }
    fresh_modules = {
        "a1": RecurrentA1(bundle.teacher),
        "a2": GatedRecurrentA2(bundle.teacher),
        "a3": IncompatibleA3(RecurrentA1(bundle.teacher)),
        "b": RecurrentB(),
        "c": RecurrentC(),
    }
    for name, module in shard_modules.items():
        port, value = shard_inputs[name]
        with torch.no_grad():
            shard_prefix = module.forward_sequence(
                {port: value[:, :4].contiguous()}, module.initial_state(batch)
            )
            shard_expected = module.forward_sequence(
                {port: value[:, 4:].contiguous()}, shard_prefix.state
            )
        record = save_module_shard(
            shard_root,
            name,
            module,
            shard_prefix.state,
            snapshot_id="qualification-modules",
            logical_step=4,
        )
        restored_state = load_module_shard(shard_root, record, fresh_modules[name])
        with torch.no_grad():
            shard_actual = fresh_modules[name].forward_sequence(
                {port: value[:, 4:].contiguous()}, restored_state
            )
        module_shard_records[name] = record
        module_shard_exact[name] = tensor_map_equal(
            shard_expected.tensors, shard_actual.tensors
        ) and tensor_map_equal(shard_expected.state, shard_actual.state)

    checkpoint_root = output_root / "checkpoint"
    checkpoint_modules = {
        "a": copy.deepcopy(bundle.a1),
        "b": copy.deepcopy(bundle.b),
        "c": copy.deepcopy(bundle.c),
    }
    checkpoint_runner = GraphRunner(
        graph_abc,
        checkpoint_modules,
        telemetry=TelemetryCollector(
            enabled=False, experiment_id="checkpoint", snapshot_id="q1"
        ),
        behavioral_decisions={"a": decisions["a1"]},
    )
    checkpoint_initial = {
        key: module.initial_state(batch) for key, module in checkpoint_modules.items()
    }
    with torch.no_grad():
        prefix = checkpoint_runner.run(
            {"x": stimulus[:, :4].contiguous()}, checkpoint_initial
        )
    checkpoint_path = save_whole_checkpoint(
        checkpoint_root,
        graph_id=graph_abc.graph_id,
        graph_version=graph_abc.graph_version,
        modules=checkpoint_modules,
        states=prefix.states,
        snapshot_id="qualification-1",
        logical_step=4,
    )
    with torch.no_grad():
        expected_continuation = checkpoint_runner.run(
            {"x": stimulus[:, 4:].contiguous()}, prefix.states
        )
    restored_modules = {
        "a": RecurrentA1(bundle.teacher),
        "b": RecurrentB(),
        "c": RecurrentC(),
    }
    restored_states, checkpoint_manifest = load_whole_checkpoint(
        checkpoint_path,
        graph_id=graph_abc.graph_id,
        graph_version=graph_abc.graph_version,
        modules=restored_modules,
    )
    restored_runner = GraphRunner(
        graph_abc,
        restored_modules,
        telemetry=TelemetryCollector(
            enabled=False, experiment_id="restore", snapshot_id="q1"
        ),
        behavioral_decisions={"a": decisions["a1"]},
    )
    with torch.no_grad():
        actual_continuation = restored_runner.run(
            {"x": stimulus[:, 4:].contiguous()}, restored_states
        )
    checkpoint_exact = tensor_map_equal(
        expected_continuation.outputs, actual_continuation.outputs
    ) and all(
        tensor_map_equal(
            expected_continuation.states[key], actual_continuation.states[key]
        )
        for key in expected_continuation.states
    )

    telemetry_off_modules = {
        "a": copy.deepcopy(bundle.a1),
        "b": copy.deepcopy(bundle.b),
    }
    telemetry_on_modules = copy.deepcopy(telemetry_off_modules)
    telemetry_off = TelemetryCollector(
        enabled=False,
        experiment_id="qualification-telemetry",
        snapshot_id="q1",
        candidate_id="synthetic.a1",
        local_status="PASS",
    )
    telemetry_on = TelemetryCollector(
        enabled=True,
        experiment_id="qualification-telemetry",
        snapshot_id="q1",
        candidate_id="synthetic.a1",
        local_status="PASS",
    )
    telemetry_off_runner = GraphRunner(
        graph_ab,
        telemetry_off_modules,
        telemetry=telemetry_off,
        behavioral_decisions={"a": decisions["a1"]},
    )
    telemetry_on_runner = GraphRunner(
        graph_ab,
        telemetry_on_modules,
        telemetry=telemetry_on,
        behavioral_decisions={"a": decisions["a1"]},
    )
    telemetry_off_states = {
        key: module.initial_state(batch) for key, module in telemetry_off_modules.items()
    }
    telemetry_on_states = {
        key: module.initial_state(batch) for key, module in telemetry_on_modules.items()
    }
    torch.manual_seed(991)
    off_rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        telemetry_off_result = telemetry_off_runner.run(
            {"x": stimulus}, telemetry_off_states
        )
    off_rng_after = torch.get_rng_state().clone()
    torch.manual_seed(991)
    on_rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        telemetry_on_result = telemetry_on_runner.run(
            {"x": stimulus}, telemetry_on_states
        )
    on_rng_after = torch.get_rng_state().clone()
    telemetry_path = output_root / "telemetry.jsonl"
    telemetry_on.write_jsonl(telemetry_path)
    telemetry_exact = (
        tensor_map_equal(telemetry_off_result.outputs, telemetry_on_result.outputs)
        and all(
            tensor_map_equal(
                telemetry_off_result.states[key], telemetry_on_result.states[key]
            )
            for key in telemetry_off_result.states
        )
        and {
            key: module_digest(module) for key, module in telemetry_off_modules.items()
        }
        == {key: module_digest(module) for key, module in telemetry_on_modules.items()}
        and torch.equal(off_rng_before, off_rng_after)
        and torch.equal(on_rng_before, on_rng_after)
        and torch.equal(off_rng_after, on_rng_after)
        and not telemetry_off.events
        and len(telemetry_on.events) == 2
    )

    concurrency = run_two_candidate_processes(
        bundle.a1,
        bundle.a2,
        bundle.b,
        heldout.stimulus[:32],
        output_root / "concurrency",
        repetitions=20,
    )
    concurrency_summary = {
        "distinct_pids": concurrency["distinct_pids"],
        "distinct_artifact_roots": concurrency["distinct_artifact_roots"],
        "payloads_immutable": concurrency["payloads_immutable"],
        "serial_parity": concurrency["serial_parity"],
        "overlap_seconds": concurrency["overlap_seconds"],
        "max_live": concurrency["max_live"],
        "worker_pids": {
            kind: record["pid"] for kind, record in concurrency["workers"].items()
        },
    }
    concurrency_exact = (
        concurrency_summary["distinct_pids"]
        and concurrency_summary["distinct_artifact_roots"]
        and concurrency_summary["payloads_immutable"]
        and all(concurrency_summary["serial_parity"].values())
        and concurrency_summary["overlap_seconds"] > 0
        and concurrency_summary["max_live"] == 2
    )

    def gate(test_name: str, checks: Mapping[str, bool]) -> dict[str, Any]:
        complete_checks = {
            "junit_case_pass": junit_cases.get(test_name) == "PASS",
            **checks,
        }
        return {
            "test_name": test_name,
            "status": "PASS" if all(complete_checks.values()) else "FAIL",
            "checks": complete_checks,
        }

    gate_matrix = {
        "CPU-01": gate(EXPECTED_TEST_NAMES[0], {}),
        "CPU-02": gate(
            EXPECTED_TEST_NAMES[1],
            {
                "fixed_semantics_blind_trace": result_ab.trace == ("a", "b"),
                "typed_execution_envelope": result_ab.envelopes[
                    "dense_latent"
                ].interface_contract
                == "dense_latent/v1",
            },
        ),
        "CPU-03": gate(EXPECTED_TEST_NAMES[2], {}),
        "CPU-04": gate(
            EXPECTED_TEST_NAMES[3],
            {
                "optimizer_steps_positive": bundle.training_reports["a1"].steps > 0,
                "parameters_changed": module_digest(bundle.a1)
                != bundle.initial_parameter_digests["a1"],
                "canonical_cosine": a1_metrics["mean_cosine"]
                >= thresholds["a1_a2_mean_cosine_min"],
                "canonical_nrmse": a1_metrics["normalized_rmse"]
                <= thresholds["a1_a2_normalized_rmse_max"],
            },
        ),
        "CPU-05": gate(
            EXPECTED_TEST_NAMES[4],
            {
                "optimizer_steps_positive": bundle.training_reports["a2"].steps > 0,
                "parameters_changed": module_digest(bundle.a2)
                != bundle.initial_parameter_digests["a2"],
                "canonical_cosine": a2_metrics["mean_cosine"]
                >= thresholds["a1_a2_mean_cosine_min"],
                "canonical_nrmse": a2_metrics["normalized_rmse"]
                <= thresholds["a1_a2_normalized_rmse_max"],
            },
        ),
        "CPU-06": gate(
            EXPECTED_TEST_NAMES[5],
            {
                "a1_downstream_accuracy": accuracy_a1
                >= thresholds["b_a1_accuracy_min"],
                "b_training_improved": bundle.training_reports["b"].final_loss
                < bundle.training_reports["b"].initial_loss,
                "b_frozen": all(
                    not parameter.requires_grad for parameter in bundle.b.parameters()
                ),
                "b_recurrent_core_is_causal": b_core_lesion_drop
                >= thresholds["targeted_ablation_accuracy_drop_min"],
                "b_private_state_is_causal": b_state_is_causal,
            },
        ),
        "CPU-07": gate(
            EXPECTED_TEST_NAMES[6],
            {
                "a2_behaviorally_admitted": decisions["a2"].admitted,
                "frozen_b_zero_replacement_steps": bundle.replacement_b_optimizer_steps
                == 0,
                "frozen_b_digest_unchanged": module_digest(bundle.b)
                == bundle.frozen_b_digest,
            },
        ),
        "CPU-08": gate(
            EXPECTED_TEST_NAMES[7],
            {
                "a3_exact_orthogonal_negative": torch.equal(z_a3, -z_a1),
                "a3_distribution_preserved": distribution_a1 == distribution_a3,
                "a3_not_admitted": not decisions["a3"].admitted,
                "a3_negative_control_pass": decisions["a3"].negative_control_pass,
            },
        ),
        "CPU-09": gate(
            EXPECTED_TEST_NAMES[8],
            {
                "a1_edge_ablation_drop": accuracy_a1 - accuracy_zero
                >= thresholds["targeted_ablation_accuracy_drop_min"],
                "a2_edge_ablation_drop": a2_ablation_drop
                >= thresholds["targeted_ablation_accuracy_drop_min"],
                "balanced_zero_baseline": accuracy_zero == 0.5,
            },
        ),
        "CPU-10": gate(
            EXPECTED_TEST_NAMES[9],
            {
                "ab_unchanged_by_c": extension_exact,
                "c_input_is_causal": c_input_is_causal,
                "c_private_state_is_causal": c_state_is_causal,
            },
        ),
        "CPU-11": gate(
            EXPECTED_TEST_NAMES[10],
            {"all_module_shards_exact": all(module_shard_exact.values())},
        ),
        "CPU-12": gate(
            EXPECTED_TEST_NAMES[11],
            {"whole_checkpoint_exact_continuation": checkpoint_exact},
        ),
        "CPU-13": gate(
            EXPECTED_TEST_NAMES[12],
            {"telemetry_noninterference_exact": telemetry_exact},
        ),
        "CPU-14": gate(EXPECTED_TEST_NAMES[13], {}),
        "CPU-15": gate(
            EXPECTED_TEST_NAMES[14],
            {"isolated_concurrent_exact": concurrency_exact},
        ),
    }

    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ("torch", "numpy", "safetensors", "jsonschema", "pytest")
    }
    binding_files_after = _evaluator_binding_files(root)
    binding_digests = _binding_digests(root, binding_files_after)
    evaluator_binding_stable = binding_digests_before == binding_digests
    binding_latest_mtime_ns = max(
        path.stat().st_mtime_ns for path in binding_files_after
    )
    junit_fresh = bool(
        junit_path is not None
        and junit_path.is_file()
        and junit_path.stat().st_mtime_ns >= binding_latest_mtime_ns
    )
    junit["fresh_for_evaluator_binding"] = junit_fresh
    junit["binding_latest_mtime_ns"] = binding_latest_mtime_ns
    junit["junit_mtime_ns"] = (
        junit_path.stat().st_mtime_ns
        if junit_path is not None and junit_path.is_file()
        else None
    )
    source_digests = {
        name: digest
        for name, digest in binding_digests.items()
        if name.startswith("src/nmcm/")
    }
    manifest_digests = {
        name: digest
        for name, digest in binding_digests.items()
        if name.startswith("configs/modules/")
    }
    all_gates_pass = all(item["status"] == "PASS" for item in gate_matrix.values())
    status = (
        "PASS"
        if junit.get("status") == "PASS"
        and junit_fresh
        and evaluator_binding_stable
        and all_gates_pass
        else "FAIL"
    )
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "nmcm.foundation0.cpu_evidence/v2",
        "status": status,
        "scope": "CPU_ONLY_STOP_BEFORE_GPU",
        "baseline": {
            "baseline_id": baseline["baseline_id"],
            "authority": baseline["authority"],
            "sha256": _sha256(baseline_path),
            "thresholds": thresholds,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": versions,
            "torch_device": "cpu",
            "torch_cuda_available": torch.cuda.is_available(),
            "gpu_qualification": "NOT_RUN",
            "git": _git_facts(root),
        },
        "gate_matrix": gate_matrix,
        "training": {
            key: asdict(value) for key, value in bundle.training_reports.items()
        },
        "behavioral_decisions": {
            name: decision_payload(decision) for name, decision in decisions.items()
        },
        "metrics": {
            "a1_vs_teacher": a1_metrics,
            "a2_vs_teacher": a2_metrics,
            "a2_vs_a1": latent_metrics(z_a2, z_a1),
            "a3_vs_teacher": a3_metrics,
            "a1_distribution": distribution_a1,
            "a3_distribution": distribution_a3,
            "a3_exact_negative_of_a1": torch.equal(z_a3, -z_a1),
            "b_accuracy_a1": accuracy_a1,
            "b_accuracy_a2": accuracy_a2,
            "b_accuracy_a3": accuracy_a3,
            "b_accuracy_zero_ablation": accuracy_zero,
            "b_accuracy_recurrent_core_lesion": accuracy(
                logits_b_core_lesion, heldout.labels
            ),
            "replacement_accuracy_difference": abs(accuracy_a1 - accuracy_a2),
            "a3_accuracy_drop": accuracy_a1 - accuracy_a3,
            "a1_edge_ablation_accuracy_drop": accuracy_a1 - accuracy_zero,
            "a2_edge_ablation_accuracy_drop": a2_ablation_drop,
            "b_recurrent_core_ablation_accuracy_drop": b_core_lesion_drop,
            "b_private_state_is_causal": b_state_is_causal,
            "frozen_b_optimizer_steps": bundle.replacement_b_optimizer_steps,
            "frozen_b_digest": bundle.frozen_b_digest,
        },
        "extension": {
            "ab_vs_abc_exact_for_a_b": extension_exact,
            "c_input_is_causal": c_input_is_causal,
            "c_private_state_is_causal": c_state_is_causal,
        },
        "module_shards": {
            "exact_continuation": module_shard_exact,
            "records": module_shard_records,
            "total_bytes": sum(
                path.stat().st_size for path in shard_root.iterdir() if path.is_file()
            ),
        },
        "checkpoint": {
            "exact_continuation": checkpoint_exact,
            "path": str(checkpoint_path),
            "manifest_sha256": _sha256(checkpoint_path),
            "tensor_bytes": sum(
                record["tensor_bytes"] for record in checkpoint_manifest["modules"]
            ),
            "total_bytes": sum(
                path.stat().st_size
                for path in checkpoint_root.iterdir()
                if path.is_file()
            ),
        },
        "telemetry": {
            "noninterference_exact": telemetry_exact,
            "events": len(telemetry_on.events),
            "path": str(telemetry_path),
            "sha256": _sha256(telemetry_path),
        },
        "concurrency": concurrency_summary,
        "performance": {
            "bundle_wall_time_seconds": bundle.total_wall_time_seconds,
            "bundle_cpu_time_seconds": bundle.total_cpu_time_seconds,
            "qualification_wall_time_seconds": elapsed,
            "process_peak_memory_bytes": bundle.process_peak_memory_bytes,
            "module_parameter_bytes": {
                name: module.parameter_bytes(module)
                for name, module in {
                    "a1": bundle.a1,
                    "a2": bundle.a2,
                    "a3": bundle.a3,
                    "b": bundle.b,
                    "c": bundle.c,
                }.items()
            },
            "module_state_bytes_batch_1": {
                name: sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in module.initial_state(1).values()
                )
                for name, module in {
                    "a1": bundle.a1,
                    "a2": bundle.a2,
                    "a3": bundle.a3,
                    "b": bundle.b,
                    "c": bundle.c,
                }.items()
            },
        },
        "junit": junit,
        "digests": {
            "evaluator_binding_stable_during_qualification": evaluator_binding_stable,
            "evaluator_binding_at_start": binding_digests_before,
            "evaluator_binding": binding_digests,
            "source": source_digests,
            "manifests": manifest_digests,
        },
        "non_goals": {
            "alice": "NOT_STARTED",
            "qwen_download": "NONE",
            "language_module": "NOT_IMPLEMENTED",
            "day0": "NOT_STARTED",
            "cuda_execution": "NOT_RUN",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_root.mkdir(parents=True, exist_ok=False)
    evidence = qualify(root, args.output_root, args.junit)
    result_path = args.output_root / "results.json"
    _atomic_json(result_path, evidence)
    print(json.dumps({"status": evidence["status"], "results": str(result_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
