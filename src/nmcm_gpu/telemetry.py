"""CUDA telemetry for the isolated FOUNDATION_GPU_0 qualification harness.

The collector is deliberately separate from :mod:`nmcm.telemetry`: the accepted
CPU telemetry contract remains unchanged.  CUDA duration is measured with CUDA
events on the stream that was current at ``begin`` time.  Completing an interval
synchronizes its end event before reading either elapsed time or allocator
counters, so ``cuda_time_ms`` is never inferred from Python wall-clock time.

``GpuTelemetryCollector`` intentionally mirrors the small ``begin``/``observe``
surface used by ``GraphRunner``.  It can therefore also be used directly around
a module call, which is the preferred form when the measured interval must cover
only ``forward_sequence`` and not runner-side validation.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor, nn


SCHEMA_VERSION = "nmcm.gpu_telemetry/v1"
_VALID_STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN"})
_VALID_PLACEMENTS = frozenset({"cpu", "gpu"})


class GpuTelemetryError(RuntimeError):
    """Raised when a CUDA measurement cannot be made reliably."""


def tensor_bytes(tensors: Mapping[str, Tensor] | Iterable[Tensor]) -> int:
    """Return logical tensor bytes without copying or running a CUDA kernel."""

    values = tensors.values() if isinstance(tensors, Mapping) else tensors
    return sum(tensor.numel() * tensor.element_size() for tensor in values)


def module_owned_tensor_bytes(module: nn.Module) -> tuple[int, int, int]:
    """Return parameter, buffer, and combined logical bytes owned by ``module``."""

    parameter_bytes = tensor_bytes(module.parameters())
    buffer_bytes = tensor_bytes(module.buffers())
    return parameter_bytes, buffer_bytes, parameter_bytes + buffer_bytes


@dataclass(frozen=True)
class CudaDeviceIdentity:
    type: str
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    uuid: str | None
    torch_cuda_build: str | None


@dataclass(frozen=True)
class CudaMemoryCounters:
    allocated_before_bytes: int
    allocated_after_bytes: int
    allocated_peak_bytes: int
    reserved_before_bytes: int
    reserved_after_bytes: int
    reserved_peak_bytes: int
    peak_scope: str


@dataclass(frozen=True)
class TransferRecord:
    """Compact reference that can be attached to a module execution event."""

    transfer_id: str
    direction: str
    bytes: int
    wall_time_ms: float
    cuda_time_ms: float
    status: str


@dataclass(frozen=True)
class TransferStatistics:
    count: int
    bytes: int
    wall_time_ms: float
    cuda_time_ms: float
    transfer_ids: tuple[str, ...]


@dataclass(frozen=True)
class GpuModuleTelemetryEvent:
    schema_version: str
    event_kind: str
    event_id: str
    timestamp_utc: str
    experiment_id: str
    snapshot_id: str
    candidate_id: str
    logical_step: int
    status: str
    sampling_policy: str
    dropped_samples: int
    collection_overhead_ms: float
    device: CudaDeviceIdentity
    module_instance_id: str
    module_id: str
    module_version: str
    compatibility_version: str
    state_schema_version: str
    lifecycle_state: str
    current_placement: str
    wall_time_ms: float
    cuda_time_ms: float
    memory: CudaMemoryCounters
    parameter_bytes: int
    buffer_bytes: int
    owned_tensor_bytes: int
    private_state_bytes: int
    activation_bytes: int
    output_bytes: int
    transfer_statistics: TransferStatistics


@dataclass(frozen=True)
class GpuTransferTelemetryEvent:
    schema_version: str
    event_kind: str
    event_id: str
    timestamp_utc: str
    experiment_id: str
    snapshot_id: str
    candidate_id: str
    logical_step: int
    status: str
    sampling_policy: str
    dropped_samples: int
    collection_overhead_ms: float
    device: CudaDeviceIdentity
    module_instance_id: str
    module_id: str
    module_version: str
    transfer_id: str
    direction: str
    source_placement: str
    target_placement: str
    bytes: int
    wall_time_ms: float
    cuda_time_ms: float
    memory: CudaMemoryCounters


GpuTelemetryEvent = GpuModuleTelemetryEvent | GpuTransferTelemetryEvent
_InvocationResult = TypeVar("_InvocationResult")


@dataclass(frozen=True)
class CudaInterval:
    """Opaque token returned by ``begin`` and consumed exactly once."""

    collector_id: str
    interval_id: int
    device_index: int
    stream: Any
    start_event: Any
    wall_started_ns: int
    allocated_before_bytes: int
    reserved_before_bytes: int
    peak_stats_reset: bool


@dataclass(frozen=True)
class _FinishedInterval:
    wall_time_ms: float
    cuda_time_ms: float
    memory: CudaMemoryCounters


class GpuTelemetryCollector:
    """Collect per-module and placement-transfer CUDA evidence.

    When disabled, ``begin`` returns ``None`` and all observation methods are
    no-ops.  This makes telemetry-OFF runs free of CUDA events, synchronizations,
    allocator resets, tensor copies, and RNG activity.

    ``reset_peak_memory_stats=True`` gives interval-local allocator peaks for a
    sequential harness.  PyTorch peak allocator counters are process/device
    global; callers measuring overlapping streams should set it to ``False`` and
    treat the reported peak as a conservative process peak since the last reset.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        experiment_id: str,
        snapshot_id: str,
        candidate_id: str = "baseline",
        device: torch.device | str = "cuda:0",
        local_status: str = "NOT_RUN",
        sampling_policy: str = "every_module_execution",
        reset_peak_memory_stats: bool = True,
        strict_device: bool = True,
    ) -> None:
        if local_status not in _VALID_STATUSES:
            raise ValueError("local telemetry status must be PASS, FAIL, or NOT_RUN")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("GPU telemetry requires a CUDA device")
        if not experiment_id or not snapshot_id or not candidate_id:
            raise ValueError("experiment, snapshot, and candidate IDs must be non-empty")

        self.enabled = enabled
        self.experiment_id = experiment_id
        self.snapshot_id = snapshot_id
        self.candidate_id = candidate_id
        self.local_status = local_status
        self.sampling_policy = sampling_policy
        self.reset_peak_memory_stats = reset_peak_memory_stats
        self.strict_device = strict_device
        self.events: list[GpuTelemetryEvent] = []

        self._collector_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._next_interval = 0
        self._next_event = 0
        self._finished_intervals: set[int] = set()
        self._device = requested_device
        self._device_index: int | None = None
        self._device_identity_cache: CudaDeviceIdentity | None = None

        # Disabled telemetry deliberately does not initialize the CUDA runtime.
        if enabled:
            self._initialize_device()

    @property
    def device(self) -> torch.device:
        self._initialize_device()
        assert self._device_index is not None
        return torch.device("cuda", self._device_index)

    @property
    def device_identity(self) -> CudaDeviceIdentity:
        self._initialize_device()
        assert self._device_identity_cache is not None
        return self._device_identity_cache

    def _initialize_device(self) -> None:
        if self._device_identity_cache is not None:
            return
        if not torch.cuda.is_available():
            raise GpuTelemetryError("CUDA_UNAVAILABLE: cannot initialize GPU telemetry")
        index = self._device.index
        if index is None:
            index = torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise GpuTelemetryError(f"CUDA_DEVICE_UNAVAILABLE: cuda:{index}")
        try:
            properties = torch.cuda.get_device_properties(index)
        except Exception as error:
            raise GpuTelemetryError("CUDA_DEVICE_IDENTITY_UNKNOWN") from error
        raw_uuid = getattr(properties, "uuid", None)
        self._device_index = index
        self._device = torch.device("cuda", index)
        self._device_identity_cache = CudaDeviceIdentity(
            type="cuda",
            index=index,
            name=str(properties.name),
            compute_capability=(int(properties.major), int(properties.minor)),
            total_memory_bytes=int(properties.total_memory),
            uuid=None if raw_uuid is None else str(raw_uuid),
            torch_cuda_build=getattr(torch.version, "cuda", None),
        )

    def _new_interval_id(self) -> int:
        with self._lock:
            interval_id = self._next_interval
            self._next_interval += 1
        return interval_id

    def _new_event_id(self, kind: str) -> str:
        with self._lock:
            sequence = self._next_event
            self._next_event += 1
        return f"{self.candidate_id}:{kind}:{sequence:06d}"

    def _begin_interval(self, *, reset_peak: bool) -> CudaInterval:
        self._initialize_device()
        assert self._device_index is not None
        try:
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(self._device_index)
            allocated = int(torch.cuda.memory_allocated(self._device_index))
            reserved = int(torch.cuda.memory_reserved(self._device_index))
            stream = torch.cuda.current_stream(self._device_index)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
        except Exception as error:
            raise GpuTelemetryError("CUDA_TELEMETRY_BEGIN_FAILED") from error
        return CudaInterval(
            collector_id=self._collector_id,
            interval_id=self._new_interval_id(),
            device_index=self._device_index,
            stream=stream,
            start_event=start_event,
            wall_started_ns=time.perf_counter_ns(),
            allocated_before_bytes=allocated,
            reserved_before_bytes=reserved,
            peak_stats_reset=reset_peak,
        )

    def begin(self) -> CudaInterval | None:
        """Begin a module execution interval on the current CUDA stream."""

        if not self.enabled:
            return None
        return self._begin_interval(reset_peak=self.reset_peak_memory_stats)

    def begin_transfer(self) -> CudaInterval | None:
        """Begin a CPU/GPU placement transfer interval."""

        if not self.enabled:
            return None
        return self._begin_interval(reset_peak=self.reset_peak_memory_stats)

    def _finish_interval(self, started: CudaInterval | None) -> _FinishedInterval:
        if started is None:
            raise GpuTelemetryError("CUDA_TELEMETRY_INTERVAL_MISSING")
        if started.collector_id != self._collector_id:
            raise GpuTelemetryError("CUDA_TELEMETRY_INTERVAL_WRONG_COLLECTOR")
        if started.device_index != self._device_index:
            raise GpuTelemetryError("CUDA_TELEMETRY_INTERVAL_WRONG_DEVICE")
        with self._lock:
            if started.interval_id in self._finished_intervals:
                raise GpuTelemetryError("CUDA_TELEMETRY_INTERVAL_ALREADY_FINISHED")
            self._finished_intervals.add(started.interval_id)

        try:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(started.stream)
            # Event synchronization is required before elapsed_time is read.  It
            # also makes the wall measurement include completion of the measured
            # CUDA work rather than merely its asynchronous kernel launches.
            end_event.synchronize()
            wall_time_ms = (time.perf_counter_ns() - started.wall_started_ns) / 1e6
            cuda_time_ms = float(started.start_event.elapsed_time(end_event))
            allocated_after = int(torch.cuda.memory_allocated(started.device_index))
            reserved_after = int(torch.cuda.memory_reserved(started.device_index))
            allocated_peak = int(torch.cuda.max_memory_allocated(started.device_index))
            reserved_peak = int(torch.cuda.max_memory_reserved(started.device_index))
        except Exception as error:
            raise GpuTelemetryError("CUDA_TELEMETRY_FINISH_FAILED") from error

        if not math.isfinite(wall_time_ms) or not math.isfinite(cuda_time_ms):
            raise GpuTelemetryError("CUDA_TELEMETRY_NONFINITE_TIMING")
        if wall_time_ms < 0 or cuda_time_ms < 0:
            raise GpuTelemetryError("CUDA_TELEMETRY_NEGATIVE_TIMING")
        peak_scope = (
            "execution_interval_process_allocator"
            if started.peak_stats_reset
            else "process_allocator_since_last_external_reset"
        )
        return _FinishedInterval(
            wall_time_ms=wall_time_ms,
            cuda_time_ms=cuda_time_ms,
            memory=CudaMemoryCounters(
                allocated_before_bytes=started.allocated_before_bytes,
                allocated_after_bytes=allocated_after,
                allocated_peak_bytes=allocated_peak,
                reserved_before_bytes=started.reserved_before_bytes,
                reserved_after_bytes=reserved_after,
                reserved_peak_bytes=reserved_peak,
                peak_scope=peak_scope,
            ),
        )

    def _validate_tensor_placement(
        self,
        module: nn.Module,
        inputs: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
        state: Mapping[str, Tensor],
    ) -> str:
        assert self._device_index is not None
        tensors = (
            list(module.parameters())
            + list(module.buffers())
            + list(inputs.values())
            + list(outputs.values())
            + list(state.values())
        )
        expected = torch.device("cuda", self._device_index)
        all_expected = all(tensor.device == expected for tensor in tensors)
        if self.strict_device and not all_expected:
            raise GpuTelemetryError(
                f"GPU_TENSOR_PLACEMENT_MISMATCH: expected {expected} for all observed tensors"
            )
        return str(expected) if all_expected else "mixed"

    @staticmethod
    def _module_identity(module: nn.Module) -> tuple[str, str, str, str]:
        return (
            str(getattr(module, "module_id", type(module).__qualname__)),
            str(getattr(module, "module_version", "unknown")),
            str(getattr(module, "compatibility_version", "unknown")),
            str(getattr(module, "state_schema_version", "unknown")),
        )

    @staticmethod
    def _summarize_transfers(records: Sequence[TransferRecord]) -> TransferStatistics:
        for record in records:
            if record.status not in _VALID_STATUSES:
                raise ValueError("transfer status must be PASS, FAIL, or NOT_RUN")
        return TransferStatistics(
            count=len(records),
            bytes=sum(record.bytes for record in records),
            wall_time_ms=sum(record.wall_time_ms for record in records),
            cuda_time_ms=sum(record.cuda_time_ms for record in records),
            transfer_ids=tuple(record.transfer_id for record in records),
        )

    def observe(
        self,
        instance_id: str,
        module: nn.Module,
        started: CudaInterval | None,
        inputs: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
        state: Mapping[str, Tensor],
        *,
        logical_step: int,
        transfers: Sequence[TransferRecord] = (),
    ) -> GpuModuleTelemetryEvent | None:
        """Finish and record one module execution interval.

        The positional and keyword arguments intentionally match the accepted
        CPU collector's ``observe`` method; ``transfers`` is an optional GPU-only
        extension for a direct qualification harness.
        """

        if not self.enabled:
            return None
        if logical_step < 0:
            raise ValueError("logical_step must be non-negative")
        if not instance_id:
            raise ValueError("module instance ID must be non-empty")

        observer_started_ns = time.perf_counter_ns()
        measurement = self._finish_interval(started)
        placement = self._validate_tensor_placement(module, inputs, outputs, state)
        module_id, module_version, compatibility_version, state_schema_version = (
            self._module_identity(module)
        )
        parameter_bytes, buffer_bytes, owned_bytes = module_owned_tensor_bytes(module)
        transfer_statistics = self._summarize_transfers(transfers)
        overhead_ms = (time.perf_counter_ns() - observer_started_ns) / 1e6
        event = GpuModuleTelemetryEvent(
            schema_version=SCHEMA_VERSION,
            event_kind="module_execution",
            event_id=self._new_event_id("module"),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            snapshot_id=self.snapshot_id,
            candidate_id=self.candidate_id,
            logical_step=logical_step,
            status=self.local_status,
            sampling_policy=self.sampling_policy,
            dropped_samples=0,
            collection_overhead_ms=overhead_ms,
            device=self.device_identity,
            module_instance_id=instance_id,
            module_id=module_id,
            module_version=module_version,
            compatibility_version=compatibility_version,
            state_schema_version=state_schema_version,
            lifecycle_state="active",
            current_placement=placement,
            wall_time_ms=measurement.wall_time_ms,
            cuda_time_ms=measurement.cuda_time_ms,
            memory=measurement.memory,
            parameter_bytes=parameter_bytes,
            buffer_bytes=buffer_bytes,
            owned_tensor_bytes=owned_bytes,
            private_state_bytes=tensor_bytes(state),
            activation_bytes=tensor_bytes(inputs),
            output_bytes=tensor_bytes(outputs),
            transfer_statistics=transfer_statistics,
        )
        with self._lock:
            self.events.append(event)
        return event

    def measure_invocation(
        self,
        instance_id: str,
        module: nn.Module,
        inputs: Mapping[str, Tensor],
        invoke: Callable[[], _InvocationResult],
        *,
        logical_step: int,
        transfers: Sequence[TransferRecord] = (),
    ) -> tuple[_InvocationResult, GpuModuleTelemetryEvent | None]:
        """Invoke a module callback exactly once and record its CUDA interval.

        The callback result must expose mapping-valued ``tensors`` and ``state``
        attributes, as NMCM's ``ModuleOutput`` does.  The method depends only on
        that structural surface and does not import or alter the CPU package.
        """

        started = self.begin()
        result = invoke()
        outputs = getattr(result, "tensors", None)
        next_state = getattr(result, "state", None)
        if not isinstance(outputs, Mapping) or not isinstance(next_state, Mapping):
            raise TypeError(
                "invocation result must expose mapping-valued tensors and state"
            )
        event = self.observe(
            instance_id,
            module,
            started,
            inputs,
            outputs,
            next_state,
            logical_step=logical_step,
            transfers=transfers,
        )
        return result, event

    def finish_transfer(
        self,
        started: CudaInterval | None,
        *,
        module_instance_id: str,
        module: nn.Module,
        logical_step: int,
        source_placement: str,
        target_placement: str,
        bytes_transferred: int,
        status: str = "PASS",
    ) -> TransferRecord | None:
        """Finish a measured placement transfer and append its own JSONL event."""

        if not self.enabled:
            return None
        if source_placement not in _VALID_PLACEMENTS:
            raise ValueError("source placement must be cpu or gpu")
        if target_placement not in _VALID_PLACEMENTS:
            raise ValueError("target placement must be cpu or gpu")
        if source_placement == target_placement:
            raise ValueError("a placement transfer must change placement")
        if status not in _VALID_STATUSES:
            raise ValueError("transfer status must be PASS, FAIL, or NOT_RUN")
        if logical_step < 0 or bytes_transferred < 0:
            raise ValueError("logical_step and transfer bytes must be non-negative")
        if not module_instance_id:
            raise ValueError("module instance ID must be non-empty")

        observer_started_ns = time.perf_counter_ns()
        measurement = self._finish_interval(started)
        module_id, module_version, _, _ = self._module_identity(module)
        direction = f"{source_placement}_to_{target_placement}"
        transfer_id = self._new_event_id("transfer")
        record = TransferRecord(
            transfer_id=transfer_id,
            direction=direction,
            bytes=bytes_transferred,
            wall_time_ms=measurement.wall_time_ms,
            cuda_time_ms=measurement.cuda_time_ms,
            status=status,
        )
        overhead_ms = (time.perf_counter_ns() - observer_started_ns) / 1e6
        event = GpuTransferTelemetryEvent(
            schema_version=SCHEMA_VERSION,
            event_kind="placement_transfer",
            event_id=transfer_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
            snapshot_id=self.snapshot_id,
            candidate_id=self.candidate_id,
            logical_step=logical_step,
            status=status,
            sampling_policy=self.sampling_policy,
            dropped_samples=0,
            collection_overhead_ms=overhead_ms,
            device=self.device_identity,
            module_instance_id=module_instance_id,
            module_id=module_id,
            module_version=module_version,
            transfer_id=transfer_id,
            direction=direction,
            source_placement=source_placement,
            target_placement=target_placement,
            bytes=bytes_transferred,
            wall_time_ms=measurement.wall_time_ms,
            cuda_time_ms=measurement.cuda_time_ms,
            memory=measurement.memory,
        )
        with self._lock:
            self.events.append(event)
        return record

    def measure_transfer(
        self,
        transfer: Callable[[], _InvocationResult],
        *,
        module_instance_id: str,
        module: nn.Module,
        logical_step: int,
        source_placement: str,
        target_placement: str,
        bytes_transferred: int,
        status: str = "PASS",
    ) -> tuple[_InvocationResult, TransferRecord | None]:
        """Invoke one placement-transfer callback and return its reference record."""

        started = self.begin_transfer()
        result = transfer()
        record = self.finish_transfer(
            started,
            module_instance_id=module_instance_id,
            module=module,
            logical_step=logical_step,
            source_placement=source_placement,
            target_placement=target_placement,
            bytes_transferred=bytes_transferred,
            status=status,
        )
        return result, record

    def event_dicts(self) -> list[dict[str, Any]]:
        """Return a stable plain-data snapshot suitable for schema validation."""

        with self._lock:
            snapshot = tuple(self.events)
        # Normalize tuples to JSON arrays and reject NaN/Infinity here, before an
        # evidence file could be written.
        return [
            json.loads(json.dumps(asdict(event), allow_nan=False)) for event in snapshot
        ]

    def aggregate(self) -> dict[str, Any]:
        """Return deterministic aggregate and per-instance locality metrics."""

        with self._lock:
            snapshot = tuple(self.events)
        module_events = tuple(
            event for event in snapshot if isinstance(event, GpuModuleTelemetryEvent)
        )
        transfer_events = tuple(
            event for event in snapshot if isinstance(event, GpuTransferTelemetryEvent)
        )
        by_instance: dict[str, dict[str, int | float | str]] = {}
        for event in module_events:
            bucket = by_instance.setdefault(
                event.module_instance_id,
                {
                    "module_id": event.module_id,
                    "module_version": event.module_version,
                    "invocations": 0,
                    "wall_time_ms": 0.0,
                    "cuda_time_ms": 0.0,
                    "allocated_peak_bytes": 0,
                    "reserved_peak_bytes": 0,
                    "state_bytes_max": 0,
                    "owned_tensor_bytes": event.owned_tensor_bytes,
                    "transfer_count": 0,
                    "transfer_bytes": 0,
                    "transfer_wall_time_ms": 0.0,
                    "transfer_cuda_time_ms": 0.0,
                },
            )
            bucket["invocations"] = int(bucket["invocations"]) + 1
            bucket["wall_time_ms"] = float(bucket["wall_time_ms"]) + event.wall_time_ms
            bucket["cuda_time_ms"] = float(bucket["cuda_time_ms"]) + event.cuda_time_ms
            bucket["allocated_peak_bytes"] = max(
                int(bucket["allocated_peak_bytes"]), event.memory.allocated_peak_bytes
            )
            bucket["reserved_peak_bytes"] = max(
                int(bucket["reserved_peak_bytes"]), event.memory.reserved_peak_bytes
            )
            bucket["state_bytes_max"] = max(
                int(bucket["state_bytes_max"]), event.private_state_bytes
            )
            bucket["transfer_count"] = (
                int(bucket["transfer_count"]) + event.transfer_statistics.count
            )
            bucket["transfer_bytes"] = (
                int(bucket["transfer_bytes"]) + event.transfer_statistics.bytes
            )
            bucket["transfer_wall_time_ms"] = (
                float(bucket["transfer_wall_time_ms"])
                + event.transfer_statistics.wall_time_ms
            )
            bucket["transfer_cuda_time_ms"] = (
                float(bucket["transfer_cuda_time_ms"])
                + event.transfer_statistics.cuda_time_ms
            )

        # Standalone transfer events are the authoritative transfer totals.  The
        # references attached to module events are useful for locality lookup but
        # intentionally are not added a second time here.
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "snapshot_id": self.snapshot_id,
            "candidate_id": self.candidate_id,
            "event_count": len(snapshot),
            "module_event_count": len(module_events),
            "transfer_event_count": len(transfer_events),
            "wall_time_ms": sum(event.wall_time_ms for event in module_events),
            "cuda_time_ms": sum(event.cuda_time_ms for event in module_events),
            "allocated_peak_bytes": max(
                (event.memory.allocated_peak_bytes for event in module_events), default=0
            ),
            "reserved_peak_bytes": max(
                (event.memory.reserved_peak_bytes for event in module_events), default=0
            ),
            "transfer_bytes": sum(event.bytes for event in transfer_events),
            "transfer_wall_time_ms": sum(
                event.wall_time_ms for event in transfer_events
            ),
            "transfer_cuda_time_ms": sum(
                event.cuda_time_ms for event in transfer_events
            ),
            "by_module_instance": {
                key: by_instance[key] for key in sorted(by_instance)
            },
        }

    def write_jsonl(self, path: str | Path) -> None:
        """Atomically replace ``path`` with all collected events as JSON Lines."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        records = self.event_dicts()
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            record,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            # Persist the directory entry as well as the file contents on POSIX.
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
