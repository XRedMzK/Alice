"""CPU correctness check for two isolated neural candidate bundles."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load, save

from .synthetic import (
    GatedRecurrentA2,
    RecurrentA1,
    RecurrentB,
    build_teacher,
    configure_determinism,
    module_digest,
    state_digest,
)


def candidate_payload(
    candidate: RecurrentA1 | GatedRecurrentA2,
    consumer: RecurrentB,
    stimulus: torch.Tensor,
) -> bytes:
    tensors = {f"candidate.{name}": value.detach().cpu().contiguous() for name, value in candidate.state_dict().items()}
    tensors.update(
        {f"consumer.{name}": value.detach().cpu().contiguous() for name, value in consumer.state_dict().items()}
    )
    tensors["fixture.stimulus"] = stimulus.detach().cpu().contiguous()
    return save(tensors)


def _execute_payload(
    candidate_kind: str,
    payload: bytes,
    *,
    repetitions: int,
    midpoint: Any | None = None,
) -> dict[str, Any]:
    configure_determinism(5101 if candidate_kind == "a1" else 5102)
    tensors = load(payload)
    teacher = build_teacher()
    if candidate_kind == "a1":
        candidate: RecurrentA1 | GatedRecurrentA2 = RecurrentA1(teacher)
    elif candidate_kind == "a2":
        candidate = GatedRecurrentA2(teacher)
    else:
        raise ValueError(f"unknown candidate kind {candidate_kind}")
    consumer = RecurrentB()
    candidate.load_state_dict(
        {
            name.removeprefix("candidate."): tensor
            for name, tensor in tensors.items()
            if name.startswith("candidate.")
        },
        strict=True,
    )
    consumer.load_state_dict(
        {
            name.removeprefix("consumer."): tensor
            for name, tensor in tensors.items()
            if name.startswith("consumer.")
        },
        strict=True,
    )
    stimulus = tensors["fixture.stimulus"]
    a_state = candidate.initial_state(stimulus.shape[0])
    b_state = consumer.initial_state(stimulus.shape[0])
    last_logits: torch.Tensor | None = None
    for iteration in range(repetitions):
        with torch.no_grad():
            a_result = candidate.forward_sequence({"stimulus": stimulus}, a_state)
            a_state = a_result.state
            b_result = consumer.forward_sequence(
                {"dense_latent": a_result.tensors["dense_latent"]}, b_state
            )
            b_state = b_result.state
            last_logits = b_result.tensors["task_logits"]
        if midpoint is not None and iteration + 1 == repetitions // 2:
            midpoint.wait(timeout=30)
    assert last_logits is not None
    return {
        "candidate_parameter_digest": module_digest(candidate),
        "consumer_parameter_digest": module_digest(consumer),
        "candidate_state_digest": state_digest(a_state),
        "consumer_state_digest": state_digest(b_state),
        "output_digest": hashlib.sha256(
            bytes(last_logits.contiguous().reshape(-1).view(torch.uint8).tolist())
        ).hexdigest(),
    }


def _worker(
    candidate_kind: str,
    payload: bytes,
    start_event: Any,
    midpoint: Any,
    ready_queue: Any,
    result_queue: Any,
    artifact_root: str,
    repetitions: int,
) -> None:
    try:
        torch.set_num_threads(1)
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=False)
        ready_queue.put({"kind": candidate_kind, "pid": os.getpid()})
        if not start_event.wait(timeout=30):
            raise TimeoutError("candidate start barrier timed out")
        started = time.perf_counter()
        result = _execute_payload(
            candidate_kind, payload, repetitions=repetitions, midpoint=midpoint
        )
        ended = time.perf_counter()
        artifact = root / "worker_result.json"
        artifact.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        result_queue.put(
            {
                "ok": True,
                "kind": candidate_kind,
                "pid": os.getpid(),
                "started": started,
                "ended": ended,
                "artifact_root": str(root),
                "result": result,
            }
        )
    except Exception as error:  # pragma: no cover - returned to the parent process
        result_queue.put(
            {"ok": False, "kind": candidate_kind, "pid": os.getpid(), "error": repr(error)}
        )


def run_two_candidate_processes(
    a1: RecurrentA1,
    a2: GatedRecurrentA2,
    consumer: RecurrentB,
    stimulus: torch.Tensor,
    artifact_parent: str | Path,
    *,
    repetitions: int = 20,
) -> dict[str, Any]:
    """Run A1+B and A2+B in overlapping spawned processes, then verify serial parity."""

    parent = Path(artifact_parent)
    parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        "a1": candidate_payload(a1, consumer, stimulus),
        "a2": candidate_payload(a2, consumer, stimulus),
    }
    payload_digests_before = {
        kind: hashlib.sha256(payload).hexdigest() for kind, payload in payloads.items()
    }
    serial = {
        kind: _execute_payload(kind, payload, repetitions=repetitions)
        for kind, payload in payloads.items()
    }

    context = mp.get_context("spawn")
    start_event = context.Event()
    midpoint = context.Barrier(2)
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    for kind in ("a1", "a2"):
        process = context.Process(
            target=_worker,
            args=(
                kind,
                payloads[kind],
                start_event,
                midpoint,
                ready_queue,
                result_queue,
                str(parent / f"candidate_{kind}"),
                repetitions,
            ),
        )
        process.start()
        processes.append(process)
    ready = [ready_queue.get(timeout=30) for _ in processes]
    start_event.set()
    concurrent = [result_queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError("candidate worker did not exit")
        if process.exitcode != 0:
            raise RuntimeError(f"candidate worker exited with {process.exitcode}")
    failures = [record for record in concurrent if not record["ok"]]
    if failures:
        raise RuntimeError(f"candidate worker failure: {failures}")
    records = {record["kind"]: record for record in concurrent}
    payload_digests_after = {
        kind: hashlib.sha256(payload).hexdigest() for kind, payload in payloads.items()
    }
    overlap_seconds = min(record["ended"] for record in records.values()) - max(
        record["started"] for record in records.values()
    )
    return {
        "ready": ready,
        "workers": records,
        "serial": serial,
        "serial_parity": {
            kind: records[kind]["result"] == serial[kind] for kind in records
        },
        "distinct_pids": len({record["pid"] for record in records.values()}) == 2,
        "distinct_artifact_roots": len(
            {record["artifact_root"] for record in records.values()}
        )
        == 2,
        "overlap_seconds": overlap_seconds,
        "payloads_immutable": payload_digests_before == payload_digests_after,
        "max_live": 2 if overlap_seconds > 0 else 1,
    }
