"""Observational telemetry that never participates in neural decisions."""

from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .neural import NeuralModule, state_bytes


def _tensor_bytes(tensors: Mapping[str, Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _statistics(tensors: Mapping[str, Tensor]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    with torch.no_grad():
        for name, tensor in tensors.items():
            flat = tensor.detach().float().reshape(-1)
            result[name] = {
                "count": flat.numel(),
                "finite_count": int(torch.isfinite(flat).sum().item()),
                "density": float((flat.abs() > 1e-6).float().mean().item()),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "norm": float(torch.linalg.vector_norm(flat).item()),
                "minimum": float(flat.min().item()),
                "maximum": float(flat.max().item()),
            }
    return result


@dataclass(frozen=True)
class TelemetryEvent:
    schema_version: str
    timestamp_utc: str
    module_instance_id: str
    module_id: str
    module_version: str
    compatibility_version: str
    lifecycle_state: str
    current_experiment: str
    candidate_id: str
    logical_step: int
    latency_ms: float
    wall_time_ms: float
    cpu_time_ms: float
    gpu_time_ms: None
    active_vram_mib: None
    active_vram_peak_mib: None
    gpu_utilization_mean_percent: None
    gpu_utilization_peak_percent: None
    gpu_measurement_method: str
    process_peak_memory_bytes: int
    parameter_bytes: int
    private_state_bytes: int
    activation_bytes: int
    output_bytes: int
    state_statistics: dict[str, dict[str, float | int]]
    activation_statistics: dict[str, dict[str, float | int]]
    output_statistics: dict[str, dict[str, float | int]]
    checkpoint_schema_version: str
    state_schema_version: str
    snapshot_id: str
    local_status: str
    transfer_statistics: dict[str, int | float | str]
    current_placement: str
    sampling_policy: str
    dropped_samples: int
    collection_overhead_ms: float
    gpu_status: str


class TelemetryCollector:
    def __init__(
        self,
        *,
        enabled: bool,
        experiment_id: str,
        snapshot_id: str,
        candidate_id: str = "baseline",
        local_status: str = "NOT_RUN",
        sampling_policy: str = "every_module_execution",
    ) -> None:
        if local_status not in {"PASS", "FAIL", "NOT_RUN"}:
            raise ValueError("local telemetry status must be PASS, FAIL, or NOT_RUN")
        self.enabled = enabled
        self.experiment_id = experiment_id
        self.snapshot_id = snapshot_id
        self.candidate_id = candidate_id
        self.local_status = local_status
        self.sampling_policy = sampling_policy
        self.events: list[TelemetryEvent] = []

    def begin(self) -> tuple[float, float]:
        return time.perf_counter(), time.process_time()

    def observe(
        self,
        instance_id: str,
        module: NeuralModule,
        started: tuple[float, float],
        inputs: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
        state: Mapping[str, Tensor],
        *,
        logical_step: int,
    ) -> None:
        if not self.enabled:
            return
        observer_started = time.perf_counter()
        wall_started, cpu_started = started
        wall_time_ms = (time.perf_counter() - wall_started) * 1000.0
        cpu_time_ms = (time.process_time() - cpu_started) * 1000.0
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        state_statistics = _statistics(state)
        activation_statistics = _statistics(inputs)
        output_statistics = _statistics(outputs)
        overhead_ms = (time.perf_counter() - observer_started) * 1000.0
        self.events.append(
            TelemetryEvent(
                schema_version="nmcm.telemetry/v2",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                module_instance_id=instance_id,
                module_id=module.module_id,
                module_version=module.module_version,
                compatibility_version=module.compatibility_version,
                lifecycle_state="active",
                current_experiment=self.experiment_id,
                candidate_id=self.candidate_id,
                logical_step=logical_step,
                latency_ms=wall_time_ms,
                wall_time_ms=wall_time_ms,
                cpu_time_ms=cpu_time_ms,
                gpu_time_ms=None,
                active_vram_mib=None,
                active_vram_peak_mib=None,
                gpu_utilization_mean_percent=None,
                gpu_utilization_peak_percent=None,
                gpu_measurement_method="NOT_RUN_CPU_ONLY",
                process_peak_memory_bytes=int(peak_kib * 1024),
                parameter_bytes=module.parameter_bytes(module),
                private_state_bytes=state_bytes(state),
                activation_bytes=_tensor_bytes(inputs),
                output_bytes=_tensor_bytes(outputs),
                state_statistics=state_statistics,
                activation_statistics=activation_statistics,
                output_statistics=output_statistics,
                checkpoint_schema_version="nmcm.shard/v2",
                state_schema_version=module.state_schema_version,
                snapshot_id=self.snapshot_id,
                local_status=self.local_status,
                transfer_statistics={
                    "count": 0,
                    "bytes": 0,
                    "latency_ms": 0.0,
                    "status": "NOT_RUN",
                },
                current_placement="cpu",
                sampling_policy=self.sampling_policy,
                dropped_samples=0,
                collection_overhead_ms=overhead_ms,
                gpu_status="NOT_RUN",
            )
        )

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
