"""Pinned, fail-closed NVML sampling for the GPU qualification window."""

from __future__ import annotations

import importlib.metadata
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence

from .environment import PINNED_NVML_DISTRIBUTION, PINNED_NVML_VERSION, ResourcePolicy


MIB = 1024 * 1024
RESOURCE_SAMPLE_SCHEMA_VERSION = "nmcm.gpu.resource_sample/v1"
RESOURCE_WINDOW_SCHEMA_VERSION = "nmcm.gpu.resource_window/v1"
NVML_BACKEND = f"NVML_DIRECT_PINNED:{PINNED_NVML_DISTRIBUTION}=={PINNED_NVML_VERSION}"


class MonitorError(RuntimeError):
    """NVML could not provide a complete, trustworthy resource sample."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MonitorError(f"NVML {name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MonitorError(f"NVML {name} is invalid: {value!r}")
    return result


@dataclass(frozen=True)
class ResourceSample:
    schema_version: str
    timestamp_utc: str
    monotonic_ns: int
    device_index: int
    device_uuid: str
    status: str
    valid: bool
    monitor_backend: str
    gpu_utilization_percent: float | None
    device_memory_used_mib: float | None
    device_memory_free_mib: float | None
    device_memory_total_mib: float | None
    project_active_vram_mib: float | None
    admitted_pids: tuple[int, ...]
    observed_project_pids: tuple[int, ...]
    missing_admitted_pids: tuple[int, ...]
    external_compute_pids: tuple[int, ...]
    external_compute_memory_mib: float | None
    process_memory_mib: tuple[tuple[int, float], ...]
    error: str | None

    @classmethod
    def unknown(
        cls,
        *,
        device_index: int,
        device_uuid: str,
        admitted_pids: Iterable[int] = (),
        error: str,
        status: str = "ERROR",
        monotonic_ns: int | None = None,
    ) -> "ResourceSample":
        pids = tuple(sorted(set(int(pid) for pid in admitted_pids)))
        return cls(
            schema_version=RESOURCE_SAMPLE_SCHEMA_VERSION,
            timestamp_utc=_utc_now(),
            monotonic_ns=time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            device_index=device_index,
            device_uuid=device_uuid,
            status=status,
            valid=False,
            monitor_backend=NVML_BACKEND,
            gpu_utilization_percent=None,
            device_memory_used_mib=None,
            device_memory_free_mib=None,
            device_memory_total_mib=None,
            project_active_vram_mib=None,
            admitted_pids=pids,
            observed_project_pids=(),
            missing_admitted_pids=pids,
            external_compute_pids=(),
            external_compute_memory_mib=None,
            process_memory_mib=(),
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
            "monotonic_ns": self.monotonic_ns,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "status": self.status,
            "valid": self.valid,
            "monitor_backend": self.monitor_backend,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "device_memory_used_mib": self.device_memory_used_mib,
            "device_memory_free_mib": self.device_memory_free_mib,
            "device_memory_total_mib": self.device_memory_total_mib,
            "project_active_vram_mib": self.project_active_vram_mib,
            "admitted_pids": list(self.admitted_pids),
            "observed_project_pids": list(self.observed_project_pids),
            "missing_admitted_pids": list(self.missing_admitted_pids),
            "external_compute_pids": list(self.external_compute_pids),
            "external_compute_memory_mib": self.external_compute_memory_mib,
            "process_memory_mib": {
                str(pid): memory for pid, memory in self.process_memory_mib
            },
            "error": self.error,
        }


class NvmlResourceMonitor:
    """Query one expected physical device through the pinned pynvml API.

    Any incomplete counter set becomes an invalid sample.  Consumers must pass
    samples through :class:`nmcm_gpu.controller.ResourceController`, which
    latches an invalid sample as STOP.
    """

    def __init__(
        self,
        *,
        device_index: int,
        expected_device_uuid: str,
        expected_total_memory_mib: int | None = None,
        pynvml_module: Any | None = None,
    ) -> None:
        self.device_index = device_index
        self.expected_device_uuid = expected_device_uuid
        self.expected_total_memory_mib = expected_total_memory_mib
        self._pynvml = pynvml_module
        self._handle: Any | None = None
        self._started = False
        self._lock = threading.RLock()

    def open(self) -> "NvmlResourceMonitor":
        with self._lock:
            if self._started:
                return self
            installed = importlib.metadata.version(PINNED_NVML_DISTRIBUTION)
            if installed != PINNED_NVML_VERSION:
                raise MonitorError(
                    f"{PINNED_NVML_DISTRIBUTION} version drift: "
                    f"expected {PINNED_NVML_VERSION}, observed {installed}"
                )
            if self._pynvml is None:
                import pynvml

                self._pynvml = pynvml
            try:
                self._pynvml.nvmlInit()
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                observed_uuid = _decode(self._pynvml.nvmlDeviceGetUUID(handle))
                if observed_uuid != self.expected_device_uuid:
                    raise MonitorError(
                        f"device UUID mismatch: expected {self.expected_device_uuid}, "
                        f"observed {observed_uuid}"
                    )
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                observed_total = int(memory.total) // MIB
                if (
                    self.expected_total_memory_mib is not None
                    and observed_total != self.expected_total_memory_mib
                ):
                    raise MonitorError(
                        f"device memory mismatch: expected {self.expected_total_memory_mib} MiB, "
                        f"observed {observed_total} MiB"
                    )
                self._handle = handle
                self._started = True
            except Exception:
                try:
                    self._pynvml.nvmlShutdown()
                except Exception:
                    pass
                self._handle = None
                self._started = False
                raise
            return self

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._pynvml.nvmlShutdown()
            finally:
                self._handle = None
                self._started = False

    def __enter__(self) -> "NvmlResourceMonitor":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # Explicit lifecycle aliases keep the qualification harness readable and
    # make the monitor's start/stop/snapshot contract independent of context
    # manager syntax.
    def start(self) -> "NvmlResourceMonitor":
        return self.open()

    def stop(self) -> None:
        self.close()

    def _process_memory(self) -> dict[int, float]:
        processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses_v3(self._handle)
        result: dict[int, float] = {}
        for process in processes:
            pid = int(process.pid)
            used_bytes = getattr(process, "usedGpuMemory", None)
            used_mib = _finite_number(used_bytes, f"process {pid} usedGpuMemory") / MIB
            # A PID should appear once, but max is conservative without double-counting
            # if a driver reports duplicate compute-instance records.
            result[pid] = max(result.get(pid, 0.0), used_mib)
        return result

    def sample(self, admitted_pids: Iterable[int] | None = None) -> ResourceSample:
        sampled_ns = time.monotonic_ns()
        # A direct snapshot always accounts for the qualification process by
        # default. Multi-worker harnesses pass their complete admitted PID set.
        source_pids = (os.getpid(),) if admitted_pids is None else admitted_pids
        pids = tuple(sorted(set(int(pid) for pid in source_pids)))
        try:
            with self._lock:
                if not self._started or self._handle is None:
                    raise MonitorError("NVML monitor is not open")
                utilization = _finite_number(
                    self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu,
                    "GPU utilization",
                )
                if utilization > 100.0:
                    raise MonitorError(f"NVML GPU utilization exceeds 100: {utilization}")
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mib = _finite_number(memory.used, "device memory used") / MIB
                free_mib = _finite_number(memory.free, "device memory free") / MIB
                total_mib = _finite_number(memory.total, "device memory total") / MIB
                if used_mib + free_mib > total_mib + 1.0:
                    raise MonitorError("NVML memory counters are internally inconsistent")
                process_memory = self._process_memory()
                observed = tuple(pid for pid in pids if pid in process_memory)
                missing = tuple(pid for pid in pids if pid not in process_memory)
                external = tuple(sorted(set(process_memory).difference(pids)))
                project_mib = sum(process_memory[pid] for pid in observed)
                external_mib = sum(process_memory[pid] for pid in external)
                if project_mib > used_mib + 1.0:
                    raise MonitorError(
                        "project process memory exceeds device-wide used memory"
                    )
                return ResourceSample(
                    schema_version=RESOURCE_SAMPLE_SCHEMA_VERSION,
                    timestamp_utc=_utc_now(),
                    monotonic_ns=sampled_ns,
                    device_index=self.device_index,
                    device_uuid=self.expected_device_uuid,
                    status="VALID",
                    valid=True,
                    monitor_backend=NVML_BACKEND,
                    gpu_utilization_percent=utilization,
                    device_memory_used_mib=used_mib,
                    device_memory_free_mib=free_mib,
                    device_memory_total_mib=total_mib,
                    project_active_vram_mib=project_mib,
                    admitted_pids=pids,
                    observed_project_pids=observed,
                    missing_admitted_pids=missing,
                    external_compute_pids=external,
                    external_compute_memory_mib=external_mib,
                    process_memory_mib=tuple(sorted(process_memory.items())),
                    error=None,
                )
        except Exception as exc:
            return ResourceSample.unknown(
                device_index=self.device_index,
                device_uuid=self.expected_device_uuid,
                admitted_pids=pids,
                monotonic_ns=sampled_ns,
                error=f"{type(exc).__name__}: {exc}",
            )

    def snapshot(self, admitted_pids: Iterable[int] | None = None) -> ResourceSample:
        return self.sample(admitted_pids)

    def strict_sample(self, admitted_pids: Iterable[int] | None = None) -> ResourceSample:
        sample = self.sample(admitted_pids)
        if not sample.valid:
            raise MonitorError(sample.error or "unknown NVML sample failure")
        return sample


class BackgroundResourceSampler:
    """Periodic NVML sampler that stops itself on any unknown/error sample."""

    def __init__(
        self,
        monitor: NvmlResourceMonitor,
        *,
        period_seconds: float,
        admitted_pids: Callable[[], Iterable[int]],
        on_sample: Callable[[ResourceSample], bool | None] | None = None,
    ) -> None:
        if not math.isfinite(period_seconds) or period_seconds <= 0:
            raise ValueError("period_seconds must be finite and positive")
        self.monitor = monitor
        self.period_seconds = period_seconds
        self.admitted_pids = admitted_pids
        self.on_sample = on_sample
        self._samples: list[ResourceSample] = []
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fatal_reason: str | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def samples(self) -> tuple[ResourceSample, ...]:
        with self._samples_lock:
            return tuple(self._samples)

    def snapshot(self) -> tuple[ResourceSample, ...]:
        """Return an immutable snapshot of all samples collected so far."""

        return self.samples()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("resource sampler is already running")
        self._stop.clear()
        self.fatal_reason = None
        self._thread = threading.Thread(
            target=self._run,
            name="nmcm-gpu-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            sample = self.monitor.sample(self.admitted_pids())
            with self._samples_lock:
                self._samples.append(sample)
            if not sample.valid:
                self.fatal_reason = sample.error or "resource monitor returned unknown state"
                self._stop.set()
                break
            if self.on_sample is not None:
                try:
                    keep_running = self.on_sample(sample)
                except Exception as exc:
                    self.fatal_reason = f"sample callback failed: {type(exc).__name__}: {exc}"
                    self._stop.set()
                    break
                if keep_running is False:
                    self.fatal_reason = "sample callback requested a safe-boundary stop"
                    self._stop.set()
                    break
            deadline += self.period_seconds
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Do not silently skip an arbitrarily long monitor gap.
                deadline = time.monotonic()
                remaining = 0
            self._stop.wait(remaining)

    def stop(self, timeout_seconds: float = 5.0) -> tuple[ResourceSample, ...]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise MonitorError("resource sampler did not stop within timeout")
        return self.samples()


@dataclass(frozen=True)
class ResourceWindowSummary:
    schema_version: str
    status: str
    active_window_seconds: float
    valid_samples: int
    total_samples: int
    utilization_time_weighted_mean_percent: float | None
    utilization_peak_percent: float | None
    project_active_vram_peak_mib: float | None
    external_compute_memory_peak_mib: float | None
    maximum_sampling_gap_seconds: float | None
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "active_window_seconds": self.active_window_seconds,
            "valid_samples": self.valid_samples,
            "total_samples": self.total_samples,
            "utilization_time_weighted_mean_percent": self.utilization_time_weighted_mean_percent,
            "utilization_peak_percent": self.utilization_peak_percent,
            "project_active_vram_peak_mib": self.project_active_vram_peak_mib,
            "external_compute_memory_peak_mib": self.external_compute_memory_peak_mib,
            "maximum_sampling_gap_seconds": self.maximum_sampling_gap_seconds,
            "violations": list(self.violations),
        }


def summarize_resource_window(
    samples: Sequence[ResourceSample],
    policy: ResourcePolicy,
    *,
    active_start_monotonic_ns: int | None = None,
    active_end_monotonic_ns: int | None = None,
) -> ResourceWindowSummary:
    """Apply every frozen sample/window gate to an already active-only window."""

    ordered = tuple(sorted(samples, key=lambda sample: sample.monotonic_ns))
    violations: list[str] = []
    if not ordered:
        return ResourceWindowSummary(
            schema_version=RESOURCE_WINDOW_SCHEMA_VERSION,
            status="FAIL",
            active_window_seconds=0.0,
            valid_samples=0,
            total_samples=0,
            utilization_time_weighted_mean_percent=None,
            utilization_peak_percent=None,
            project_active_vram_peak_mib=None,
            external_compute_memory_peak_mib=None,
            maximum_sampling_gap_seconds=None,
            violations=("NO_RESOURCE_SAMPLES",),
        )

    start_ns = ordered[0].monotonic_ns if active_start_monotonic_ns is None else active_start_monotonic_ns
    default_end_ns = ordered[-1].monotonic_ns + int(policy.sample_period_seconds * 1e9)
    end_ns = default_end_ns if active_end_monotonic_ns is None else active_end_monotonic_ns
    duration = max(0.0, (end_ns - start_ns) / 1e9)
    selected = tuple(
        sample for sample in ordered if start_ns <= sample.monotonic_ns < end_ns
    )
    valid = tuple(sample for sample in selected if sample.valid and sample.status == "VALID")
    if len(valid) != len(selected):
        violations.append("UNKNOWN_OR_MONITOR_ERROR")
    if len(valid) < policy.minimum_valid_samples:
        violations.append("INSUFFICIENT_VALID_SAMPLES")
    if duration < policy.minimum_active_window_seconds:
        violations.append("ACTIVE_WINDOW_TOO_SHORT")

    utilization_values = [sample.gpu_utilization_percent for sample in valid]
    vram_values = [sample.project_active_vram_mib for sample in valid]
    external_values = [sample.external_compute_memory_mib for sample in valid]
    if any(value is None for value in utilization_values + vram_values + external_values):
        violations.append("INCOMPLETE_VALID_SAMPLE")
        numeric_utilization: list[float] = []
        numeric_vram: list[float] = []
        numeric_external: list[float] = []
    else:
        numeric_utilization = [float(value) for value in utilization_values if value is not None]
        numeric_vram = [float(value) for value in vram_values if value is not None]
        numeric_external = [float(value) for value in external_values if value is not None]

    gaps = [
        (right.monotonic_ns - left.monotonic_ns) / 1e9
        for left, right in zip(selected, selected[1:])
    ]
    maximum_gap = max(gaps) if gaps else (duration if selected else None)
    if maximum_gap is not None and maximum_gap > policy.maximum_sample_age_seconds:
        violations.append("SAMPLING_GAP_TOO_LARGE")

    weighted_mean: float | None = None
    if numeric_utilization and len(numeric_utilization) == len(valid):
        weights: list[float] = []
        for index, sample in enumerate(valid):
            next_ns = valid[index + 1].monotonic_ns if index + 1 < len(valid) else end_ns
            interval_start = max(start_ns, sample.monotonic_ns)
            interval_end = min(end_ns, next_ns)
            weights.append(max(0.0, (interval_end - interval_start) / 1e9))
        total_weight = sum(weights)
        weighted_mean = (
            sum(value * weight for value, weight in zip(numeric_utilization, weights))
            / total_weight
            if total_weight > 0
            else fmean(numeric_utilization)
        )
        if max(numeric_utilization) > policy.maximum_utilization_percent:
            violations.append("UTILIZATION_SAMPLE_LIMIT_EXCEEDED")
        if weighted_mean > policy.maximum_utilization_percent:
            violations.append("UTILIZATION_WINDOW_MEAN_LIMIT_EXCEEDED")

    vram_peak = max(numeric_vram) if numeric_vram else None
    if vram_peak is not None and vram_peak > policy.maximum_active_vram_mib:
        violations.append("ACTIVE_VRAM_LIMIT_EXCEEDED")
    if numeric_external and max(numeric_external) > policy.maximum_external_compute_memory_mib:
        violations.append("EXTERNAL_COMPUTE_MEMORY_CONTAMINATION")

    utilization_peak = max(numeric_utilization) if numeric_utilization else None
    return ResourceWindowSummary(
        schema_version=RESOURCE_WINDOW_SCHEMA_VERSION,
        status="PASS" if not violations else "FAIL",
        active_window_seconds=duration,
        valid_samples=len(valid),
        total_samples=len(selected),
        utilization_time_weighted_mean_percent=weighted_mean,
        utilization_peak_percent=utilization_peak,
        project_active_vram_peak_mib=vram_peak,
        external_compute_memory_peak_mib=max(numeric_external) if numeric_external else None,
        maximum_sampling_gap_seconds=maximum_gap,
        violations=tuple(dict.fromkeys(violations)),
    )
