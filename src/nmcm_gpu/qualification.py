"""Materialize and validate one immutable FOUNDATION_GPU_0 official attempt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from jsonschema import Draft202012Validator

from nmcm.synthetic import accuracy, latent_metrics

from .artifacts import (
    MODULE_NAMES,
    accepted_inputs,
    load_accepted_cpu_artifacts,
    load_accepted_whole_checkpoint,
    load_fresh_shard,
    module_input_port,
    sha256_file,
    tensor_map_close,
)
from .concurrency import run_two_candidate_streams
from .controller import (
    AdmissionRequest,
    DecisionAction,
    ResourceController,
    validate_fail_closed_semantics,
)
from .environment import ExpectedGpuEnvironment, ResourcePolicy, load_gpu_profile, probe_environment
from .monitor import BackgroundResourceSampler, NvmlResourceMonitor, summarize_resource_window
from .placement import (
    canonical_cpu_tensor_map,
    module_digest,
    move_module_and_state,
    output_to_cpu,
    tensor_map_exact_equal,
    verify_hot_warm_hot_continuation,
)
from .telemetry import GpuTelemetryCollector


EXPECTED_TEST_NAMES = tuple(
    f"test_g{index:02d}_{suffix}"
    for index, suffix in enumerate(
        (
            "cuda_runtime_device_identity",
            "all_five_modules_execute_on_gpu",
            "cpu_gpu_cpu_module_state_round_trip",
            "hot_warm_hot_offload_continuation",
            "independent_shards_restore_to_gpu",
            "whole_checkpoint_gpu_continuation",
            "gpu_telemetry_noninterference",
            "per_module_cuda_telemetry_schema",
            "two_gpu_candidate_bundles_overlap",
            "concurrent_bundle_isolation",
            "resource_controller_boundaries",
            "actual_active_vram_below_limit",
            "monitor_unknown_is_fail_closed",
            "cpu_gpu_functional_equivalence",
            "local_conjunction_candidate",
        ),
        1,
    )
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _binding_files(root: Path) -> set[Path]:
    files = set((root / "src/nmcm_gpu").glob("*.py"))
    files.update((root / "gpu_tests").glob("*.py"))
    files.update((root / "gpu_configs").glob("*.json"))
    for relative in (
        "temp.md", "RESOURCE_POLICY.md", "requirements-gpu.in",
        "requirements-gpu.lock", "pytest-gpu.ini",
        "configs/foundation0_cpu_acceptance.json",
    ):
        files.add(root / relative)
    return {path for path in files if path.is_file()}


def _digests(root: Path, paths: set[Path]) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}


def _junit(path: Path) -> dict[str, Any]:
    xml_root = ET.parse(path).getroot()
    suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
    cases = []
    for suite in suites:
        for case in suite.findall("testcase"):
            children = list(case)
            status = "PASS" if not any(child.tag in {"failure", "error", "skipped"} for child in children) else "NON_PASS"
            cases.append({"name": case.attrib.get("name"), "classname": case.attrib.get("classname"), "status": status, "time_seconds": float(case.attrib.get("time", 0.0))})
    names = [case["name"] for case in cases]
    exact = (
        len(cases) == 15 and set(names) == set(EXPECTED_TEST_NAMES)
        and len(set(names)) == len(names)
        and all(case["status"] == "PASS" for case in cases)
        and all(case["classname"] == "gpu_tests.test_foundation0_gpu" for case in cases)
    )
    return {
        "status": "PASS" if exact else "FAIL", "cases": cases,
        "tests": len(cases), "passed": sum(case["status"] == "PASS" for case in cases),
        "failed_or_skipped": sum(case["status"] != "PASS" for case in cases),
        "missing": sorted(set(EXPECTED_TEST_NAMES) - set(names)),
        "unexpected": sorted(set(names) - set(EXPECTED_TEST_NAMES)),
        "path": str(path), "sha256": sha256_file(path),
    }


def _run(module, port: str, value: torch.Tensor, state):
    with torch.no_grad():
        return module.forward_sequence({port: value}, state)


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    return {"mean": statistics.fmean(ordered), "p50": at(0.50), "p95": at(0.95), "max": max(ordered)}


def qualify(root: Path, output_root: Path, junit_path: Path, *, attempt_started_ns: int) -> dict[str, Any]:
    profile_path = root / "gpu_configs/foundation0_gpu.json"
    profile = load_gpu_profile(profile_path)
    policy = ResourcePolicy.from_mapping(profile["resource_policy"])
    expected_environment = ExpectedGpuEnvironment.from_profile(profile)
    device = torch.device("cuda", expected_environment.device_index)
    deadline_ns = attempt_started_ns + int(policy.hard_wall_seconds * 1e9)
    binding_before = _digests(root, _binding_files(root))
    environment = probe_environment(profile)
    if not environment.passed:
        raise RuntimeError(f"GPU_ENVIRONMENT_BLOCKED:{environment.to_dict()}")

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("TF32 override contradicts frozen runtime policy")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(device)
    torch.manual_seed(6001)
    torch.cuda.manual_seed_all(6002)
    artifacts = load_accepted_cpu_artifacts(root, profile)
    batch = int(profile["workloads"]["correctness_batch"])
    inputs = accepted_inputs(artifacts, batch=batch)
    atol = float(profile["numerical_policy"]["cpu_gpu_atol"])
    rtol = float(profile["numerical_policy"]["cpu_gpu_rtol"])

    controller = ResourceController(
        policy, expected_device_index=expected_environment.device_index,
        expected_device_uuid=expected_environment.device_uuid,
        qualification_started_monotonic_ns=attempt_started_ns,
    )
    monitor = NvmlResourceMonitor(
        device_index=expected_environment.device_index,
        expected_device_uuid=expected_environment.device_uuid,
        expected_total_memory_mib=expected_environment.total_memory_mib,
    ).open()
    initial_sample = monitor.strict_sample({os.getpid()})
    request = AdmissionRequest(
        bundle_id="official-foundation-gpu0", worker_ids=(f"pid:{os.getpid()}",),
        mutable_module_instance_ids=("a1", "a2", "a3", "b", "c", "a1.concurrent", "b1.concurrent", "a2.concurrent", "b2.concurrent"),
        estimated_bundle_utilization_percent=float(profile["resource_policy"]["estimated_bundle_utilization_percent"]),
    )
    admission = controller.admit(request, initial_sample)
    if admission.action != DecisionAction.ADMIT:
        raise RuntimeError(f"resource admission denied: {admission.to_dict()}")

    sampler = BackgroundResourceSampler(
        monitor, period_seconds=policy.sample_period_seconds,
        admitted_pids=lambda: {os.getpid()}, on_sample=controller.should_continue,
    )
    sampler.start()
    active_started_ns = time.monotonic_ns()
    telemetry = GpuTelemetryCollector(
        enabled=True, experiment_id="foundation0-gpu-official", snapshot_id=output_root.name,
        candidate_id="all-modules", device=device, local_status="PASS",
    )
    placement_evidence: dict[str, Any] = {}
    offload_evidence: dict[str, Any] = {}
    shard_evidence: dict[str, Any] = {}
    parity_evidence: dict[str, Any] = {}
    latest_outputs: dict[str, Any] = {}
    telemetry_samples: dict[str, Any] = {}

    try:
        for name in MODULE_NAMES:
            port = module_input_port(name)
            cpu_module = copy.deepcopy(artifacts.modules[name]).cpu().eval()
            prefix_cpu = _run(cpu_module, port, inputs[name][:, :4], cpu_module.initial_state(batch, device="cpu"))
            cpu_suffix = output_to_cpu(_run(cpu_module, port, inputs[name][:, 4:], canonical_cpu_tensor_map(prefix_cpu.state)))
            live_module = copy.deepcopy(artifacts.modules[name]).cpu().eval()
            live_state = canonical_cpu_tensor_map(prefix_cpu.state)
            to_gpu = move_module_and_state(live_module, live_state, device)
            gpu_suffix_input = {port: inputs[name][:, 4:].to(device)}
            gpu_suffix = _run(to_gpu.module, port, gpu_suffix_input[port], to_gpu.state)
            gpu_snapshot = output_to_cpu(gpu_suffix)
            back_state = canonical_cpu_tensor_map(prefix_cpu.state)
            roundtrip_module = copy.deepcopy(artifacts.modules[name]).cpu().eval()
            rt1 = move_module_and_state(roundtrip_module, back_state, device)
            rt2 = move_module_and_state(rt1.module, rt1.state, "cpu")
            roundtrip_suffix = output_to_cpu(_run(rt2.module, port, inputs[name][:, 4:], rt2.state))
            placement_evidence[name] = {
                "cpu_to_gpu": to_gpu.transfer.to_dict(), "round_trip_to_gpu": rt1.transfer.to_dict(),
                "round_trip_to_cpu": rt2.transfer.to_dict(),
                "cpu_continuation_exact": tensor_map_exact_equal(cpu_suffix.tensors, roundtrip_suffix.tensors) and tensor_map_exact_equal(cpu_suffix.state, roundtrip_suffix.state),
            }
            offload_module = copy.deepcopy(artifacts.modules[name]).to(device).eval()
            prefix_gpu = _run(offload_module, port, inputs[name][:, :4].to(device), offload_module.initial_state(batch, device=device))
            offload = verify_hot_warm_hot_continuation(offload_module, prefix_gpu.state, gpu_suffix_input, hot_device=device)
            offload_evidence[name] = offload.to_dict()
            restored_module, restored_state = load_fresh_shard(artifacts, name)
            cpu_restored_module = copy.deepcopy(restored_module).cpu()
            cpu_restored_state = canonical_cpu_tensor_map(restored_state)
            cpu_restored_suffix = output_to_cpu(
                _run(
                    cpu_restored_module,
                    port,
                    inputs[name][:, 4:],
                    cpu_restored_state,
                )
            )
            restored = move_module_and_state(restored_module, restored_state, device)
            restored_suffix = _run(restored.module, port, gpu_suffix_input[port], restored.state)
            direct_module = copy.deepcopy(restored.module)
            direct_state = {key: value.clone() for key, value in restored.state.items()}
            direct_suffix = _run(direct_module, port, gpu_suffix_input[port], direct_state)
            shard_evidence[name] = {
                "on_gpu": all(t.device == device for t in list(restored.module.state_dict().values()) + list(restored.state.values())),
                "exact_same_gpu_continuation": tensor_map_exact_equal(restored_suffix.tensors, direct_suffix.tensors) and tensor_map_exact_equal(restored_suffix.state, direct_suffix.state),
                "accepted_cpu_gpu_continuation_close": tensor_map_close(
                    cpu_restored_suffix.tensors,
                    output_to_cpu(restored_suffix).tensors,
                    atol=atol,
                    rtol=rtol,
                )
                and tensor_map_close(
                    cpu_restored_suffix.state,
                    output_to_cpu(restored_suffix).state,
                    atol=atol,
                    rtol=rtol,
                ),
            }
            parity_evidence[name] = {
                "outputs_close": tensor_map_close(cpu_suffix.tensors, gpu_snapshot.tensors, atol=atol, rtol=rtol),
                "state_close": tensor_map_close(cpu_suffix.state, gpu_snapshot.state, atol=atol, rtol=rtol),
            }
            full_module = copy.deepcopy(artifacts.modules[name]).to(device).eval()
            full_result = _run(
                full_module,
                port,
                inputs[name].to(device),
                full_module.initial_state(batch, device=device),
            )
            measured_module = copy.deepcopy(artifacts.modules[name]).to(device).eval()
            measured_state = measured_module.initial_state(1, device=device)
            measured_input = {port: accepted_inputs(artifacts, batch=1)[name].to(device)}
            wall_values: list[float] = []
            cuda_values: list[float] = []
            last_event = None
            for repetition in range(5):
                result, last_event = telemetry.measure_invocation(
                    name, measured_module, measured_input,
                    lambda m=measured_module, p=port, v=measured_input[port], s={k: t.clone() for k,t in measured_state.items()}: _run(m, p, v, s),
                    logical_step=repetition,
                )
                assert last_event is not None
                wall_values.append(last_event.wall_time_ms)
                cuda_values.append(last_event.cuda_time_ms)
            telemetry_samples[name] = {
                "wall_ms": _percentiles(wall_values), "cuda_ms": _percentiles(cuda_values),
                "allocated_peak_bytes": last_event.memory.allocated_peak_bytes,
                "reserved_peak_bytes": last_event.memory.reserved_peak_bytes,
                "state_bytes": last_event.private_state_bytes,
                "owned_tensor_bytes": last_event.owned_tensor_bytes,
                "transfer_wall_ms": offload.offload.wall_time_ms + offload.restore.wall_time_ms,
                "transfer_cuda_ms": (offload.offload.cuda_time_ms or 0) + (offload.restore.cuda_time_ms or 0),
            }
            latest_outputs[name] = output_to_cpu(full_result)
            if sampler.stop_requested or time.monotonic_ns() >= deadline_ns:
                raise RuntimeError(f"official stop condition: {sampler.fatal_reason or 'deadline'}")

        whole_modules, whole_states, whole_manifest = load_accepted_whole_checkpoint(artifacts)
        cpu_whole_modules = {name: copy.deepcopy(module).cpu() for name, module in whole_modules.items()}
        cpu_whole_states = {name: canonical_cpu_tensor_map(state) for name, state in whole_states.items()}
        suffix_cpu = artifacts.splits["heldout"].stimulus[:batch, 4:].contiguous()
        a_cpu = _run(cpu_whole_modules["a"], "stimulus", suffix_cpu, cpu_whole_states["a"])
        b_cpu = _run(cpu_whole_modules["b"], "dense_latent", a_cpu.tensors["dense_latent"], cpu_whole_states["b"])
        c_cpu = _run(cpu_whole_modules["c"], "dense_latent", a_cpu.tensors["dense_latent"], cpu_whole_states["c"])
        for name in whole_modules:
            placed = move_module_and_state(whole_modules[name], whole_states[name], device)
            whole_modules[name], whole_states[name] = placed.module, placed.state
        suffix_gpu = suffix_cpu.to(device)
        a_gpu = _run(whole_modules["a"], "stimulus", suffix_gpu, whole_states["a"])
        b_gpu = _run(whole_modules["b"], "dense_latent", a_gpu.tensors["dense_latent"], whole_states["b"])
        c_gpu = _run(whole_modules["c"], "dense_latent", a_gpu.tensors["dense_latent"], whole_states["c"])
        whole_checkpoint = {
            "manifest": whole_manifest,
            "cpu_gpu_continuation_close": all(
                tensor_map_close(cpu.tensors, output_to_cpu(gpu).tensors, atol=atol, rtol=rtol)
                and tensor_map_close(cpu.state, output_to_cpu(gpu).state, atol=atol, rtol=rtol)
                for cpu, gpu in ((a_cpu, a_gpu), (b_cpu, b_gpu), (c_cpu, c_gpu))
            ),
        }

        off_module = copy.deepcopy(artifacts.modules["a1"]).to(device).eval()
        on_module = copy.deepcopy(artifacts.modules["a1"]).to(device).eval()
        stimulus8 = artifacts.splits["heldout"].stimulus[:8].to(device)
        torch.manual_seed(7101); torch.cuda.manual_seed_all(7102)
        cpu_rng_before = torch.get_rng_state().clone(); cuda_rng_before = torch.cuda.get_rng_state_all()
        off_result = _run(off_module, "stimulus", stimulus8, off_module.initial_state(8, device=device))
        cpu_rng_off = torch.get_rng_state().clone(); cuda_rng_off = torch.cuda.get_rng_state_all()
        torch.manual_seed(7101); torch.cuda.manual_seed_all(7102)
        on_collector = GpuTelemetryCollector(enabled=True, experiment_id="gpu-noninterference", snapshot_id=output_root.name, candidate_id="a1", device=device, local_status="PASS")
        on_result, _ = on_collector.measure_invocation("a1", on_module, {"stimulus": stimulus8}, lambda: _run(on_module, "stimulus", stimulus8, on_module.initial_state(8, device=device)), logical_step=0)
        telemetry_noninterference = {
            "outputs_exact": tensor_map_exact_equal(off_result.tensors, on_result.tensors),
            "state_exact": tensor_map_exact_equal(off_result.state, on_result.state),
            "parameters_exact": module_digest(off_module) == module_digest(on_module),
            "cpu_rng_exact": torch.equal(cpu_rng_before, cpu_rng_off) and torch.equal(cpu_rng_off, torch.get_rng_state()),
            "cuda_rng_exact": all(torch.equal(x,y) for x,y in zip(cuda_rng_before, cuda_rng_off)) and all(torch.equal(x,y) for x,y in zip(cuda_rng_off, torch.cuda.get_rng_state_all())),
        }

        concurrency = run_two_candidate_streams(
            artifacts.modules["a1"], artifacts.modules["a2"], artifacts.modules["b"],
            artifacts.splits["heldout"].stimulus[:8], output_root / "concurrency",
            iterations=int(profile["workloads"]["concurrency_iterations"]), device=device,
            stop_requested=lambda: sampler.stop_requested,
        )

        # G-14 is official CUDA work, so keep it inside the live NVML/controller
        # window and honor a resource stop at each module boundary.
        thresholds = artifacts.cpu_results["baseline"]["thresholds"]
        heldout_labels = artifacts.splits["heldout"].labels[:batch]
        gpu_science: dict[str, Any] = {}
        provider_logits: dict[str, torch.Tensor] = {}
        for provider in ("a1", "a2", "a3"):
            latent = latest_outputs[provider].tensors["dense_latent"]
            b_module = copy.deepcopy(artifacts.modules["b"]).to(device)
            b_run = _run(
                b_module,
                "dense_latent",
                latent.to(device),
                b_module.initial_state(batch, device=device),
            )
            torch.cuda.synchronize(device)
            if sampler.stop_requested or time.monotonic_ns() >= deadline_ns:
                raise RuntimeError(
                    f"official stop condition: {sampler.fatal_reason or 'deadline'}"
                )
            provider_logits[provider] = output_to_cpu(b_run).tensors["task_logits"]
            gpu_science[provider] = {
                "latent_metrics": latent_metrics(
                    latent, artifacts.splits["heldout"].latent[:batch]
                ),
                "accuracy": accuracy(provider_logits[provider], heldout_labels),
            }
        gpu_science["decisions_exact"] = {
            "a1_pass": gpu_science["a1"]["latent_metrics"]["mean_cosine"] >= thresholds["a1_a2_mean_cosine_min"] and gpu_science["a1"]["latent_metrics"]["normalized_rmse"] <= thresholds["a1_a2_normalized_rmse_max"] and gpu_science["a1"]["accuracy"] >= thresholds["b_a1_accuracy_min"],
            "a2_pass": gpu_science["a2"]["latent_metrics"]["mean_cosine"] >= thresholds["a1_a2_mean_cosine_min"] and gpu_science["a2"]["latent_metrics"]["normalized_rmse"] <= thresholds["a1_a2_normalized_rmse_max"] and gpu_science["a2"]["accuracy"] >= thresholds["b_a2_accuracy_min"] and abs(gpu_science["a1"]["accuracy"] - gpu_science["a2"]["accuracy"]) <= thresholds["b_a1_a2_accuracy_absolute_difference_max"],
            "a3_rejected": gpu_science["a1"]["accuracy"] - gpu_science["a3"]["accuracy"] >= thresholds["a3_b_accuracy_drop_min"],
        }
        accepted_decisions = artifacts.cpu_results["behavioral_decisions"]
        gpu_science["accepted_decision_parity"] = {
            "a1_admitted": gpu_science["decisions_exact"]["a1_pass"]
            == accepted_decisions["a1"]["admitted"],
            "a2_admitted": gpu_science["decisions_exact"]["a2_pass"]
            == accepted_decisions["a2"]["admitted"],
            "a3_admitted": (not gpu_science["decisions_exact"]["a3_rejected"])
            == accepted_decisions["a3"]["admitted"],
            "a3_negative_control": gpu_science["decisions_exact"]["a3_rejected"]
            == accepted_decisions["a3"]["negative_control_pass"],
        }

        # Keep the declared active resource window real and non-idle: repeat tiny
        # module work at low duty, synchronizing before each one-second sample.
        pacing_module = copy.deepcopy(artifacts.modules["a1"]).to(device).eval()
        pacing_input = artifacts.splits["heldout"].stimulus[:1].to(device)
        while (
            (time.monotonic_ns() - active_started_ns) / 1e9 < policy.minimum_active_window_seconds
            or sum(
                sample.monotonic_ns >= active_started_ns
                for sample in sampler.samples()
            )
            < policy.minimum_valid_samples
        ):
            _run(pacing_module, "stimulus", pacing_input, pacing_module.initial_state(1, device=device))
            torch.cuda.synchronize(device)
            if sampler.stop_requested or time.monotonic_ns() >= deadline_ns:
                raise RuntimeError(f"official stop condition: {sampler.fatal_reason or 'deadline'}")
            time.sleep(0.05)
        active_ended_ns = time.monotonic_ns()
    finally:
        samples = sampler.stop()
        monitor.close()

    resource_window = summarize_resource_window(
        samples, policy, active_start_monotonic_ns=active_started_ns,
        active_end_monotonic_ns=active_ended_ns,
    )
    resource_schema = json.loads((root / "gpu_configs/resource_sample.schema.json").read_text())
    for sample in samples:
        Draft202012Validator(resource_schema).validate(sample.to_dict())
    resource_path = output_root / "resource_samples.jsonl"
    resource_path.write_text("".join(json.dumps(sample.to_dict(), sort_keys=True) + "\n" for sample in samples), encoding="utf-8")
    telemetry_path = output_root / "telemetry.jsonl"
    telemetry.write_jsonl(telemetry_path)
    telemetry_schema = json.loads((root / "gpu_configs/gpu_telemetry_event.schema.json").read_text())
    for event in telemetry.event_dicts():
        Draft202012Validator(telemetry_schema).validate(event)

    junit = _junit(junit_path)
    binding_after = _digests(root, _binding_files(root))
    binding_stable = binding_before == binding_after
    latest_binding_mtime = max(path.stat().st_mtime_ns for path in _binding_files(root))
    junit_fresh = junit_path.stat().st_mtime_ns >= latest_binding_mtime
    runtime_seconds = (time.monotonic_ns() - attempt_started_ns) / 1e9
    injected = validate_fail_closed_semantics(policy, expected_device_index=expected_environment.device_index, expected_device_uuid=expected_environment.device_uuid)
    checks = {
        "G-01": environment.passed,
        "G-02": all(item["outputs_close"] and item["state_close"] for item in parity_evidence.values()),
        "G-03": all(item["cpu_continuation_exact"] and item["round_trip_to_gpu"]["values_preserved"] and item["round_trip_to_cpu"]["values_preserved"] for item in placement_evidence.values()),
        "G-04": all(item["passed"] and item["warm_owned_cuda_bytes"] == 0 for item in offload_evidence.values()),
        "G-05": all(item["on_gpu"] and item["exact_same_gpu_continuation"] and item["accepted_cpu_gpu_continuation_close"] for item in shard_evidence.values()),
        "G-06": whole_checkpoint["cpu_gpu_continuation_close"],
        "G-07": all(telemetry_noninterference.values()),
        "G-08": len(telemetry_samples) == 5 and all(
            item["wall_ms"]["max"] >= 0
            and item["cuda_ms"]["max"] >= 0
            and item["allocated_peak_bytes"] >= item["owned_tensor_bytes"]
            and item["reserved_peak_bytes"] >= item["allocated_peak_bytes"]
            and item["state_bytes"] > 0
            and item["transfer_wall_ms"] >= 0
            and item["transfer_cuda_ms"] >= 0
            for item in telemetry_samples.values()
        ),
        "G-09": concurrency["overlap_pass"] and concurrency["serial_parity_pass"],
        "G-10": concurrency["isolation_pass"],
        "G-11": all(injected.values()) and admission.action == DecisionAction.ADMIT,
        "G-12": resource_window.project_active_vram_peak_mib is not None and resource_window.project_active_vram_peak_mib <= policy.maximum_active_vram_mib,
        "G-13": all(injected.values()) and resource_window.passed and sampler.fatal_reason is None,
        "G-14": all(item["outputs_close"] and item["state_close"] for item in parity_evidence.values()) and all(gpu_science["decisions_exact"].values()) and all(gpu_science["accepted_decision_parity"].values()),
    }
    gate_matrix = {gate: {"status": "PASS" if passed else "FAIL", "checks": {"computed_condition": passed, "junit_case_pass": next((case["status"] == "PASS" for case in junit["cases"] if case["name"] == EXPECTED_TEST_NAMES[int(gate[-2:]) - 1]), False)}} for gate, passed in checks.items()}
    pre_g15_pass = all(item["status"] == "PASS" and all(item["checks"].values()) for item in gate_matrix.values())
    g15_checks = {
        "g01_g14_conjunction": pre_g15_pass, "exact_junit": junit["status"] == "PASS",
        "junit_fresh": junit_fresh, "binding_stable": binding_stable,
        "resource_window_pass": resource_window.passed, "deadline_met": runtime_seconds <= policy.hard_wall_seconds,
    }
    gate_matrix["G-15"] = {"status": "PASS" if all(g15_checks.values()) else "FAIL", "checks": g15_checks}
    status = "PASS" if gate_matrix["G-15"]["status"] == "PASS" else "FAIL"
    result = {
        "schema_version": "nmcm.foundation0.gpu_evidence/v1", "status": status,
        "attempt_id": output_root.name,
        "profile": {"id": profile["profile_id"], "path": str(profile_path), "sha256": sha256_file(profile_path)},
        "environment": environment.to_dict(), "gate_matrix": gate_matrix,
        "parity": {"modules": parity_evidence, "gpu_scientific_results": gpu_science, "policy": profile["numerical_policy"]},
        "placement": {"cpu_gpu_cpu": placement_evidence, "hot_warm_hot": offload_evidence},
        "module_shards": shard_evidence, "checkpoint": whole_checkpoint,
        "telemetry": {"noninterference": telemetry_noninterference, "per_module": telemetry_samples, "aggregate": telemetry.aggregate(), "path": str(telemetry_path), "sha256": sha256_file(telemetry_path)},
        "concurrency": concurrency,
        "resources": {"admission": admission.to_dict(), "injected_fail_closed": injected, "window": resource_window.to_dict(), "samples_path": str(resource_path), "samples_sha256": sha256_file(resource_path), "samples": [sample.to_dict() for sample in samples]},
        "timing": {"official_wall_seconds": runtime_seconds, "hard_limit_seconds": policy.hard_wall_seconds},
        "digests": {"binding_stable": binding_stable, "binding_before": binding_before, "binding_after": binding_after, "junit": junit, "junit_fresh": junit_fresh},
        "non_goals": {"training": "NONE", "alice": "NOT_STARTED", "qwen_download": "NONE", "day0": "NOT_STARTED", "cold_paging": "NOT_IMPLEMENTED"},
    }
    evidence_schema = json.loads((root / "gpu_configs/gpu_evidence.schema.json").read_text())
    Draft202012Validator(evidence_schema).validate(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output_root.exists():
        raise SystemExit("official attempt root must not already exist")
    args.output_root.mkdir(parents=True, exist_ok=False)
    attempt_started_ns = time.monotonic_ns()
    try:
        evidence = qualify(root, args.output_root, args.junit.resolve(), attempt_started_ns=attempt_started_ns)
    except Exception as error:
        failure = {
            "schema_version": "nmcm.foundation0.gpu_attempt_failure/v1",
            "status": "FAIL", "attempt_id": args.output_root.name,
            "error_type": type(error).__name__, "error": str(error),
            "wall_seconds": (time.monotonic_ns() - attempt_started_ns) / 1e9,
            "reusable": False,
        }
        _atomic_json(args.output_root / "failure.json", failure)
        raise
    _atomic_json(args.output_root / "results.json", evidence)
    print(json.dumps({"status": evidence["status"], "results": str(args.output_root / "results.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
