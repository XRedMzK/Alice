"""Bounded two-candidate CUDA-stream concurrency qualification.

This is deliberately a one-process design.  Each candidate owns a different
module graph, explicit recurrent states, CUDA generator, input storage, stream,
and artifact root.  CUDA events on the device's shared clock provide an
overlapping-work-window measurement; the result does not overclaim profiler-
level proof of simultaneous kernel residency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from nmcm.neural import NeuralModule, TensorMap
from nmcm.synthetic import GatedRecurrentA2, RecurrentA1, RecurrentB

from .placement import module_digest, tensor_digest, tensor_map_digest


OVERLAP_POLICY_ID = "cuda-event-work-window-and-makespan/v1"
MIN_EVENT_OVERLAP_MS = 0.01
MAX_CONCURRENCY_ITERATIONS = 128
MAX_CONCURRENCY_BATCH = 256
MAX_CONCURRENCY_SEQUENCE_LENGTH = 64


Candidate = RecurrentA1 | GatedRecurrentA2


@dataclass
class _CandidateRuntime:
    kind: str
    candidate: Candidate
    consumer: RecurrentB
    stimulus: Tensor
    candidate_state: TensorMap
    consumer_state: TensorMap
    generator: torch.Generator
    generator_seed: int
    stream: torch.cuda.Stream
    artifact_root: Path | None
    last_logits: Tensor | None = None
    rng_probe: Tensor | None = None
    host_started_ns: int | None = None
    host_enqueued_ns: int | None = None


def _normalise_cuda_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("GPU concurrency requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA concurrency requested but CUDA is unavailable")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {index}")
    return torch.device("cuda", index)


def _storage_keys(tensors: list[Tensor] | tuple[Tensor, ...]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for tensor in tensors:
        storage = tensor.untyped_storage()
        if storage.nbytes():
            result.add((tensor.device.index or 0, int(storage.data_ptr())))
    return result


def _module_storage_keys(module: NeuralModule) -> set[tuple[int, int]]:
    tensors = tuple(module.parameters(recurse=True)) + tuple(
        module.buffers(recurse=True)
    )
    return _storage_keys(tensors)


def _state_storage_keys(runtime: _CandidateRuntime) -> set[tuple[int, int]]:
    return _storage_keys(
        tuple(runtime.candidate_state.values())
        + tuple(runtime.consumer_state.values())
    )


def _build_runtime(
    kind: str,
    candidate: Candidate,
    consumer: RecurrentB,
    stimulus: Tensor,
    *,
    device: torch.device,
    generator_seed: int,
    artifact_root: Path | None,
) -> _CandidateRuntime:
    candidate_copy = copy.deepcopy(candidate).eval().to(device=device)
    consumer_copy = copy.deepcopy(consumer).eval().to(device=device)
    stimulus_copy = stimulus.detach().to(device=device, copy=True).contiguous()
    candidate_state = candidate_copy.initial_state(
        stimulus_copy.shape[0], device=device
    )
    consumer_state = consumer_copy.initial_state(
        stimulus_copy.shape[0], device=device
    )
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    return _CandidateRuntime(
        kind=kind,
        candidate=candidate_copy,
        consumer=consumer_copy,
        stimulus=stimulus_copy,
        candidate_state=candidate_state,
        consumer_state=consumer_state,
        generator=generator,
        generator_seed=generator_seed,
        stream=torch.cuda.Stream(device=device),
        artifact_root=artifact_root,
    )


def _launch_workload(
    runtime: _CandidateRuntime,
    *,
    iterations: int,
    stop_requested: Callable[[], bool] | None,
) -> None:
    # RNG is independent and observable even though the frozen neural modules
    # themselves contain no stochastic operation at inference time.
    first_probe = torch.rand(
        (8,), device=runtime.stimulus.device, generator=runtime.generator
    )
    last_logits: Tensor | None = None
    candidate_state = runtime.candidate_state
    consumer_state = runtime.consumer_state
    with torch.no_grad():
        for _ in range(iterations):
            if stop_requested is not None and stop_requested():
                raise RuntimeError("GPU resource controller requested an immediate stop")
            candidate_output = runtime.candidate.forward_sequence(
                {"stimulus": runtime.stimulus}, candidate_state
            )
            candidate_state = candidate_output.state
            consumer_output = runtime.consumer.forward_sequence(
                {"dense_latent": candidate_output.tensors["dense_latent"]},
                consumer_state,
            )
            consumer_state = consumer_output.state
            last_logits = consumer_output.tensors["task_logits"]
    second_probe = torch.rand(
        (8,), device=runtime.stimulus.device, generator=runtime.generator
    )
    if last_logits is None:  # guarded by the public iterations check
        raise AssertionError("concurrency workload produced no result")
    runtime.candidate_state = candidate_state
    runtime.consumer_state = consumer_state
    runtime.last_logits = last_logits
    runtime.rng_probe = torch.cat((first_probe, second_probe)).contiguous()


def _runtime_result(runtime: _CandidateRuntime) -> dict[str, Any]:
    if runtime.last_logits is None or runtime.rng_probe is None:
        raise RuntimeError(f"candidate {runtime.kind} did not complete")
    return {
        "kind": runtime.kind,
        "candidate_parameter_digest": module_digest(runtime.candidate),
        "consumer_parameter_digest": module_digest(runtime.consumer),
        "candidate_state_digest": tensor_map_digest(runtime.candidate_state),
        "consumer_state_digest": tensor_map_digest(runtime.consumer_state),
        "output_digest": tensor_digest(runtime.last_logits, name="task_logits"),
        "generator_seed": runtime.generator_seed,
        "generator_state_digest": tensor_digest(
            runtime.generator.get_state(), name="cuda_generator_state"
        ),
        "rng_probe_digest": tensor_digest(runtime.rng_probe, name="rng_probe"),
    }


def _run_serial_reference(
    runtime: _CandidateRuntime,
    *,
    iterations: int,
    stop_requested: Callable[[], bool] | None,
) -> tuple[dict[str, Any], float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(runtime.stream):
        start.record(runtime.stream)
        _launch_workload(
            runtime, iterations=iterations, stop_requested=stop_requested
        )
        end.record(runtime.stream)
    end.synchronize()
    return _runtime_result(runtime), float(start.elapsed_time(end))


def _prepare_artifact_roots(parent: Path) -> dict[str, Path]:
    roots = {kind: parent / f"candidate_{kind}" for kind in ("a1", "a2")}
    existing = [str(path) for path in roots.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "candidate artifacts must use fresh roots; already present: "
            + ", ".join(existing)
        )
    parent.mkdir(parents=True, exist_ok=True)
    for root in roots.values():
        root.mkdir(parents=False, exist_ok=False)
    return roots


def _write_candidate_artifact(root: Path, result: Mapping[str, Any]) -> None:
    (root / "result.json").write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_two_candidate_streams(
    a1: RecurrentA1,
    a2: GatedRecurrentA2,
    consumer: RecurrentB,
    stimulus: Tensor,
    artifact_parent: str | Path,
    *,
    iterations: int = 64,
    device: torch.device | str = "cuda:0",
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run A1+B and A2+B on two explicit CUDA streams in one process.

    The bounded workload is run serially first and then with two stream-owned
    clones.  Exact digest parity is required for each candidate.  Overlap is
    measured from per-stream CUDA event intervals relative to one origin event;
    the preregistered pass condition is both positive interval intersection and
    a concurrent event makespan shorter than the sum of its two stream windows.
    """

    cuda_device = _normalise_cuda_device(device)
    if not 2 <= iterations <= MAX_CONCURRENCY_ITERATIONS:
        raise ValueError(
            f"iterations must be in [2, {MAX_CONCURRENCY_ITERATIONS}] for low-duty qualification"
        )
    if stimulus.ndim != 3:
        raise ValueError("stimulus must have [batch,time,feature] shape")
    if stimulus.shape[0] > MAX_CONCURRENCY_BATCH:
        raise ValueError("concurrency batch exceeds the bounded low-duty profile")
    if stimulus.shape[1] > MAX_CONCURRENCY_SEQUENCE_LENGTH:
        raise ValueError("concurrency sequence exceeds the bounded low-duty profile")
    if stimulus.device.type != "cpu":
        # A host fixture gives each candidate an unambiguous independent H→D copy.
        raise ValueError("concurrency stimulus must be supplied from canonical CPU storage")

    parent = Path(artifact_parent)
    artifact_roots = _prepare_artifact_roots(parent)
    source_digests_before = {
        "a1": module_digest(a1),
        "a2": module_digest(a2),
        "consumer": module_digest(consumer),
        "stimulus": tensor_digest(stimulus, name="stimulus"),
    }

    serial_runtimes = {
        "a1": _build_runtime(
            "a1",
            a1,
            consumer,
            stimulus,
            device=cuda_device,
            generator_seed=6101,
            artifact_root=None,
        ),
        "a2": _build_runtime(
            "a2",
            a2,
            consumer,
            stimulus,
            device=cuda_device,
            generator_seed=6102,
            artifact_root=None,
        ),
    }
    torch.cuda.synchronize(cuda_device)
    serial: dict[str, dict[str, Any]] = {}
    serial_cuda_ms: dict[str, float] = {}
    for kind in ("a1", "a2"):
        serial[kind], serial_cuda_ms[kind] = _run_serial_reference(
            serial_runtimes[kind],
            iterations=iterations,
            stop_requested=stop_requested,
        )

    concurrent = {
        "a1": _build_runtime(
            "a1",
            a1,
            consumer,
            stimulus,
            device=cuda_device,
            generator_seed=6101,
            artifact_root=artifact_roots["a1"],
        ),
        "a2": _build_runtime(
            "a2",
            a2,
            consumer,
            stimulus,
            device=cuda_device,
            generator_seed=6102,
            artifact_root=artifact_roots["a2"],
        ),
    }
    module_keys = {
        kind: _module_storage_keys(runtime.candidate)
        | _module_storage_keys(runtime.consumer)
        for kind, runtime in concurrent.items()
    }
    state_keys = {
        kind: _state_storage_keys(runtime) for kind, runtime in concurrent.items()
    }
    input_keys = {
        kind: _storage_keys((runtime.stimulus,))
        for kind, runtime in concurrent.items()
    }
    isolation = {
        "distinct_module_instances": len(
            {
                id(concurrent["a1"].candidate),
                id(concurrent["a1"].consumer),
                id(concurrent["a2"].candidate),
                id(concurrent["a2"].consumer),
            }
        )
        == 4,
        "disjoint_module_storage": module_keys["a1"].isdisjoint(module_keys["a2"]),
        "disjoint_state_storage": state_keys["a1"].isdisjoint(state_keys["a2"]),
        "disjoint_input_storage": input_keys["a1"].isdisjoint(input_keys["a2"]),
        "distinct_generators": (
            concurrent["a1"].generator is not concurrent["a2"].generator
            and concurrent["a1"].generator_seed
            != concurrent["a2"].generator_seed
        ),
        "distinct_streams": concurrent["a1"].stream != concurrent["a2"].stream,
        "distinct_artifact_roots": (
            artifact_roots["a1"].resolve() != artifact_roots["a2"].resolve()
        ),
    }

    torch.cuda.synchronize(cuda_device)
    origin = torch.cuda.Event(enable_timing=True)
    starts = {
        kind: torch.cuda.Event(enable_timing=True) for kind in concurrent
    }
    ends = {kind: torch.cuda.Event(enable_timing=True) for kind in concurrent}
    origin.record(torch.cuda.current_stream(cuda_device))
    origin.synchronize()
    barrier = threading.Barrier(3)
    failures: list[tuple[str, BaseException]] = []
    failure_lock = threading.Lock()

    def worker(kind: str) -> None:
        runtime = concurrent[kind]
        try:
            with torch.cuda.device(cuda_device), torch.cuda.stream(runtime.stream):
                runtime.stream.wait_event(origin)
                barrier.wait(timeout=30)
                runtime.host_started_ns = time.perf_counter_ns()
                starts[kind].record(runtime.stream)
                _launch_workload(
                    runtime,
                    iterations=iterations,
                    stop_requested=stop_requested,
                )
                ends[kind].record(runtime.stream)
                runtime.host_enqueued_ns = time.perf_counter_ns()
        except BaseException as error:  # delivered to the controlling thread
            with failure_lock:
                failures.append((kind, error))

    threads = [
        threading.Thread(
            target=worker,
            args=(kind,),
            name=f"nmcm-gpu-{kind}",
            daemon=False,
        )
        for kind in ("a1", "a2")
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=30)
    for thread in threads:
        thread.join(timeout=60)
        if thread.is_alive():
            raise TimeoutError(f"CUDA candidate thread did not exit: {thread.name}")
    torch.cuda.synchronize(cuda_device)
    if failures:
        kind, error = failures[0]
        raise RuntimeError(f"CUDA candidate {kind} failed") from error

    records = {kind: _runtime_result(runtime) for kind, runtime in concurrent.items()}
    for kind in ("a1", "a2"):
        _write_candidate_artifact(artifact_roots[kind], records[kind])

    intervals_ms = {
        kind: {
            "start_offset_ms": float(origin.elapsed_time(starts[kind])),
            "end_offset_ms": float(origin.elapsed_time(ends[kind])),
            "duration_ms": float(starts[kind].elapsed_time(ends[kind])),
        }
        for kind in ("a1", "a2")
    }
    device_overlap_ms = max(
        0.0,
        min(interval["end_offset_ms"] for interval in intervals_ms.values())
        - max(interval["start_offset_ms"] for interval in intervals_ms.values()),
    )
    concurrent_makespan_ms = max(
        interval["end_offset_ms"] for interval in intervals_ms.values()
    ) - min(interval["start_offset_ms"] for interval in intervals_ms.values())
    concurrent_window_sum_ms = sum(
        interval["duration_ms"] for interval in intervals_ms.values()
    )
    overlap_pass = (
        device_overlap_ms >= MIN_EVENT_OVERLAP_MS
        and concurrent_makespan_ms + MIN_EVENT_OVERLAP_MS
        <= concurrent_window_sum_ms
    )

    host_intervals = {
        kind: {
            "started_ns": runtime.host_started_ns,
            "enqueued_ns": runtime.host_enqueued_ns,
        }
        for kind, runtime in concurrent.items()
    }
    if any(
        value is None
        for interval in host_intervals.values()
        for value in interval.values()
    ):
        raise RuntimeError("host launch interval evidence is incomplete")
    host_overlap_ms = max(
        0.0,
        (
            min(int(interval["enqueued_ns"]) for interval in host_intervals.values())
            - max(int(interval["started_ns"]) for interval in host_intervals.values())
        )
        / 1_000_000.0,
    )

    serial_parity = {kind: records[kind] == serial[kind] for kind in records}
    source_digests_after = {
        "a1": module_digest(a1),
        "a2": module_digest(a2),
        "consumer": module_digest(consumer),
        "stimulus": tensor_digest(stimulus, name="stimulus"),
    }
    source_immutable = source_digests_before == source_digests_after
    isolation_pass = all(isolation.values()) and source_immutable
    return {
        "policy_id": OVERLAP_POLICY_ID,
        "device": str(cuda_device),
        "iterations": iterations,
        "candidate_count": 2,
        "execution_model": "one_process_two_explicit_cuda_streams",
        "serial": serial,
        "concurrent": records,
        "serial_parity": serial_parity,
        "serial_parity_pass": all(serial_parity.values()),
        "isolation": isolation,
        "source_immutable": source_immutable,
        "isolation_pass": isolation_pass,
        "artifact_roots": {
            kind: str(path) for kind, path in artifact_roots.items()
        },
        "serial_cuda_ms": serial_cuda_ms,
        "serial_cuda_sum_ms": sum(serial_cuda_ms.values()),
        "concurrent_cuda_intervals_ms": intervals_ms,
        "concurrent_window_sum_ms": concurrent_window_sum_ms,
        "concurrent_makespan_ms": concurrent_makespan_ms,
        "device_window_overlap_ms": device_overlap_ms,
        "host_enqueue_overlap_ms": host_overlap_ms,
        "minimum_event_overlap_ms": MIN_EVENT_OVERLAP_MS,
        "overlap_pass": overlap_pass,
        "overlap_claim": (
            "CUDA event work windows overlap on the shared device clock; "
            "no claim of profiler-proven simultaneous kernel residency"
        ),
        "passed": all(serial_parity.values()) and isolation_pass and overlap_pass,
        "evidence_digest": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
