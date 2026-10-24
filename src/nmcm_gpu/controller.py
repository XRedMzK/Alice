"""Fail-closed GPU resource admission and runtime threshold enforcement."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .environment import ResourcePolicy
from .monitor import ResourceSample


class DecisionAction(str, Enum):
    ADMIT = "ADMIT"
    QUEUE = "QUEUE"
    CONTINUE = "CONTINUE"
    STOP = "STOP"
    RELEASE = "RELEASE"


@dataclass(frozen=True)
class AdmissionRequest:
    bundle_id: str
    worker_ids: tuple[str, ...]
    mutable_module_instance_ids: tuple[str, ...]
    estimated_bundle_utilization_percent: float
    extra_active_vram_mib: float = 0.0

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("bundle_id must be non-empty")
        if not self.worker_ids or any(not value for value in self.worker_ids):
            raise ValueError("worker_ids must contain non-empty identifiers")
        if len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("worker_ids must be unique within a request")
        if not self.mutable_module_instance_ids or any(
            not value for value in self.mutable_module_instance_ids
        ):
            raise ValueError("mutable_module_instance_ids must be non-empty")
        if len(set(self.mutable_module_instance_ids)) != len(
            self.mutable_module_instance_ids
        ):
            raise ValueError("mutable module instance IDs must be unique")
        for name, value in (
            ("estimated_bundle_utilization_percent", self.estimated_bundle_utilization_percent),
            ("extra_active_vram_mib", self.extra_active_vram_mib),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ReservationBreakdown:
    worker_context_mib: float
    telemetry_and_transport_mib: float
    bundle_workspace_mib: float
    mutable_module_instances_mib: float
    extra_active_vram_mib: float
    safety_margin_mib: float
    total_mib: float

    def to_dict(self) -> dict[str, float]:
        return {
            "worker_context_mib": self.worker_context_mib,
            "telemetry_and_transport_mib": self.telemetry_and_transport_mib,
            "bundle_workspace_mib": self.bundle_workspace_mib,
            "mutable_module_instances_mib": self.mutable_module_instances_mib,
            "extra_active_vram_mib": self.extra_active_vram_mib,
            "safety_margin_mib": self.safety_margin_mib,
            "total_mib": self.total_mib,
        }


@dataclass(frozen=True)
class ResourceDecision:
    action: DecisionAction
    reason: str
    sample_monotonic_ns: int | None
    observed_utilization_percent: float | None
    observed_project_active_vram_mib: float | None
    projected_utilization_percent: float | None = None
    projected_active_vram_mib: float | None = None
    reservation: ReservationBreakdown | None = None

    @property
    def allowed(self) -> bool:
        return self.action in {DecisionAction.ADMIT, DecisionAction.CONTINUE}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "sample_monotonic_ns": self.sample_monotonic_ns,
            "observed_utilization_percent": self.observed_utilization_percent,
            "observed_project_active_vram_mib": self.observed_project_active_vram_mib,
            "projected_utilization_percent": self.projected_utilization_percent,
            "projected_active_vram_mib": self.projected_active_vram_mib,
            "reservation": self.reservation.to_dict() if self.reservation else None,
        }


def _stop(sample: ResourceSample | None, reason: str) -> ResourceDecision:
    return ResourceDecision(
        action=DecisionAction.STOP,
        reason=reason,
        sample_monotonic_ns=sample.monotonic_ns if sample else None,
        observed_utilization_percent=sample.gpu_utilization_percent if sample else None,
        observed_project_active_vram_mib=sample.project_active_vram_mib if sample else None,
    )


def evaluate_sample(
    sample: ResourceSample | None,
    policy: ResourcePolicy,
    *,
    expected_device_index: int,
    expected_device_uuid: str,
    now_monotonic_ns: int | None = None,
) -> ResourceDecision:
    """Pure fail-closed evaluation used for real and injected samples."""

    if sample is None:
        return _stop(None, "RESOURCE_MONITOR_UNKNOWN")
    if not sample.valid or sample.status != "VALID":
        return _stop(sample, "RESOURCE_MONITOR_ERROR_OR_UNKNOWN")
    if sample.device_index != expected_device_index or sample.device_uuid != expected_device_uuid:
        return _stop(sample, "RESOURCE_MONITOR_DEVICE_IDENTITY_MISMATCH")
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    age_seconds = (now_ns - sample.monotonic_ns) / 1e9
    if age_seconds < 0 or age_seconds > policy.maximum_sample_age_seconds:
        return _stop(sample, "RESOURCE_SAMPLE_STALE_OR_FUTURE")

    utilization = sample.gpu_utilization_percent
    active_vram = sample.project_active_vram_mib
    device_free = sample.device_memory_free_mib
    external_memory = sample.external_compute_memory_mib
    if (
        utilization is None
        or active_vram is None
        or device_free is None
        or external_memory is None
    ):
        return _stop(sample, "RESOURCE_SAMPLE_INCOMPLETE")
    if any(
        not math.isfinite(value) or value < 0
        for value in (utilization, active_vram, device_free, external_memory)
    ):
        return _stop(sample, "RESOURCE_SAMPLE_INVALID_NUMERIC_VALUE")
    if utilization > policy.maximum_utilization_percent:
        return _stop(sample, "GPU_UTILIZATION_LIMIT_EXCEEDED")
    if active_vram > policy.maximum_active_vram_mib:
        return _stop(sample, "ACTIVE_VRAM_LIMIT_EXCEEDED")
    if external_memory > policy.maximum_external_compute_memory_mib:
        return _stop(sample, "EXTERNAL_COMPUTE_MEMORY_CONTAMINATION")
    if sample.missing_admitted_pids:
        return _stop(sample, "ADMITTED_PROCESS_MISSING_FROM_NVML_ACCOUNTING")
    return ResourceDecision(
        action=DecisionAction.CONTINUE,
        reason="RESOURCE_SAMPLE_WITHIN_LIMITS",
        sample_monotonic_ns=sample.monotonic_ns,
        observed_utilization_percent=utilization,
        observed_project_active_vram_mib=active_vram,
    )


def _reservation(
    requests: Iterable[AdmissionRequest], policy: ResourcePolicy
) -> ReservationBreakdown:
    request_list = tuple(requests)
    workers = {worker for request in request_list for worker in request.worker_ids}
    module_instances = {
        module
        for request in request_list
        for module in request.mutable_module_instance_ids
    }
    if len(module_instances) != sum(
        len(request.mutable_module_instance_ids) for request in request_list
    ):
        raise ValueError("mutable module instances cannot be shared across bundles")
    ledger = policy.reservation_ledger
    worker_mib = len(workers) * ledger.cuda_context_mib
    telemetry_mib = len(request_list) * ledger.telemetry_and_transport_mib
    workspace_mib = len(request_list) * ledger.bundle_workspace_mib
    module_mib = len(module_instances) * ledger.per_mutable_module_instance_mib
    extra_mib = sum(request.extra_active_vram_mib for request in request_list)
    safety_mib = ledger.safety_margin_mib if request_list else 0.0
    total = worker_mib + telemetry_mib + workspace_mib + module_mib + extra_mib + safety_mib
    return ReservationBreakdown(
        worker_context_mib=worker_mib,
        telemetry_and_transport_mib=telemetry_mib,
        bundle_workspace_mib=workspace_mib,
        mutable_module_instances_mib=module_mib,
        extra_active_vram_mib=extra_mib,
        safety_margin_mib=safety_mib,
        total_mib=total,
    )


class ResourceController:
    """Thread-safe controller with a persistent fail-closed STOP latch."""

    def __init__(
        self,
        policy: ResourcePolicy,
        *,
        expected_device_index: int,
        expected_device_uuid: str,
        qualification_started_monotonic_ns: int | None = None,
    ) -> None:
        self.policy = policy
        self.expected_device_index = expected_device_index
        self.expected_device_uuid = expected_device_uuid
        self.qualification_started_monotonic_ns = (
            time.monotonic_ns()
            if qualification_started_monotonic_ns is None
            else qualification_started_monotonic_ns
        )
        self._active: dict[str, AdmissionRequest] = {}
        self._decisions: list[ResourceDecision] = []
        self._stop_reason: str | None = None
        self._lock = threading.RLock()

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stop_reason is not None

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    def active_bundle_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))

    def decisions(self) -> tuple[ResourceDecision, ...]:
        with self._lock:
            return tuple(self._decisions)

    def _record(self, decision: ResourceDecision) -> ResourceDecision:
        self._decisions.append(decision)
        if decision.action == DecisionAction.STOP and self._stop_reason is None:
            self._stop_reason = decision.reason
        return decision

    def observe(
        self, sample: ResourceSample | None, *, now_monotonic_ns: int | None = None
    ) -> ResourceDecision:
        with self._lock:
            if self._stop_reason is not None:
                return self._record(_stop(sample, f"STOP_LATCHED:{self._stop_reason}"))
            now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            elapsed_seconds = (
                now_ns - self.qualification_started_monotonic_ns
            ) / 1e9
            if elapsed_seconds < 0:
                return self._record(_stop(sample, "CONTROLLER_CLOCK_MOVED_BACKWARD"))
            if elapsed_seconds > self.policy.hard_wall_seconds:
                return self._record(_stop(sample, "HARD_WALL_TIME_LIMIT_EXCEEDED"))
            decision = evaluate_sample(
                sample,
                self.policy,
                expected_device_index=self.expected_device_index,
                expected_device_uuid=self.expected_device_uuid,
                now_monotonic_ns=now_ns,
            )
            return self._record(decision)

    def admit(
        self,
        request: AdmissionRequest,
        sample: ResourceSample | None,
        *,
        now_monotonic_ns: int | None = None,
    ) -> ResourceDecision:
        with self._lock:
            if request.bundle_id in self._active:
                return self._record(_stop(sample, "DUPLICATE_ACTIVE_BUNDLE_ID"))
            observation = self.observe(sample, now_monotonic_ns=now_monotonic_ns)
            if observation.action == DecisionAction.STOP:
                return observation
            assert sample is not None
            decision_now_ns = (
                time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            )
            if (
                decision_now_ns - self.qualification_started_monotonic_ns
            ) / 1e9 > self.policy.soft_wall_seconds:
                return self._record(
                    ResourceDecision(
                        action=DecisionAction.QUEUE,
                        reason="SOFT_WALL_TIME_BLOCKS_NEW_ADMISSION",
                        sample_monotonic_ns=sample.monotonic_ns,
                        observed_utilization_percent=sample.gpu_utilization_percent,
                        observed_project_active_vram_mib=sample.project_active_vram_mib,
                    )
                )
            combined = tuple(self._active.values()) + (request,)
            try:
                reservation = _reservation(combined, self.policy)
                current_reservation = _reservation(self._active.values(), self.policy)
            except ValueError as exc:
                return self._record(_stop(sample, f"STATE_ISOLATION_VIOLATION:{exc}"))
            incremental_mib = reservation.total_mib - current_reservation.total_mib
            projected_vram = max(
                reservation.total_mib,
                float(sample.project_active_vram_mib) + incremental_mib,
            )
            projected_utilization = (
                float(sample.gpu_utilization_percent)
                + request.estimated_bundle_utilization_percent
            )
            if request.estimated_bundle_utilization_percent > self.policy.maximum_utilization_percent:
                return self._record(
                    ResourceDecision(
                        action=DecisionAction.QUEUE,
                        reason="BUNDLE_UTILIZATION_ESTIMATE_EXCEEDS_LIMIT",
                        sample_monotonic_ns=sample.monotonic_ns,
                        observed_utilization_percent=sample.gpu_utilization_percent,
                        observed_project_active_vram_mib=sample.project_active_vram_mib,
                        projected_utilization_percent=projected_utilization,
                        projected_active_vram_mib=projected_vram,
                        reservation=reservation,
                    )
                )
            if projected_utilization > self.policy.maximum_utilization_percent:
                return self._record(
                    ResourceDecision(
                        action=DecisionAction.QUEUE,
                        reason="PROJECTED_AGGREGATE_UTILIZATION_EXCEEDS_LIMIT",
                        sample_monotonic_ns=sample.monotonic_ns,
                        observed_utilization_percent=sample.gpu_utilization_percent,
                        observed_project_active_vram_mib=sample.project_active_vram_mib,
                        projected_utilization_percent=projected_utilization,
                        projected_active_vram_mib=projected_vram,
                        reservation=reservation,
                    )
                )
            if projected_vram > self.policy.maximum_active_vram_mib:
                return self._record(
                    ResourceDecision(
                        action=DecisionAction.QUEUE,
                        reason="PROJECTED_ACTIVE_VRAM_EXCEEDS_LIMIT",
                        sample_monotonic_ns=sample.monotonic_ns,
                        observed_utilization_percent=sample.gpu_utilization_percent,
                        observed_project_active_vram_mib=sample.project_active_vram_mib,
                        projected_utilization_percent=projected_utilization,
                        projected_active_vram_mib=projected_vram,
                        reservation=reservation,
                    )
                )
            if float(sample.device_memory_free_mib) < incremental_mib:
                return self._record(
                    ResourceDecision(
                        action=DecisionAction.QUEUE,
                        reason="INSUFFICIENT_DEVICE_FREE_MEMORY_FOR_RESERVATION",
                        sample_monotonic_ns=sample.monotonic_ns,
                        observed_utilization_percent=sample.gpu_utilization_percent,
                        observed_project_active_vram_mib=sample.project_active_vram_mib,
                        projected_utilization_percent=projected_utilization,
                        projected_active_vram_mib=projected_vram,
                        reservation=reservation,
                    )
                )
            self._active[request.bundle_id] = request
            return self._record(
                ResourceDecision(
                    action=DecisionAction.ADMIT,
                    reason="RESOURCE_RESERVATION_ADMITTED",
                    sample_monotonic_ns=sample.monotonic_ns,
                    observed_utilization_percent=sample.gpu_utilization_percent,
                    observed_project_active_vram_mib=sample.project_active_vram_mib,
                    projected_utilization_percent=projected_utilization,
                    projected_active_vram_mib=projected_vram,
                    reservation=reservation,
                )
            )

    def release(self, bundle_id: str) -> ResourceDecision:
        with self._lock:
            if bundle_id not in self._active:
                decision = ResourceDecision(
                    action=DecisionAction.STOP,
                    reason="RELEASE_OF_UNKNOWN_BUNDLE",
                    sample_monotonic_ns=None,
                    observed_utilization_percent=None,
                    observed_project_active_vram_mib=None,
                )
                return self._record(decision)
            del self._active[bundle_id]
            return self._record(
                ResourceDecision(
                    action=DecisionAction.RELEASE,
                    reason="RESOURCE_RESERVATION_RELEASED",
                    sample_monotonic_ns=None,
                    observed_utilization_percent=None,
                    observed_project_active_vram_mib=None,
                    reservation=_reservation(self._active.values(), self.policy),
                )
            )

    def should_continue(self, sample: ResourceSample) -> bool:
        """Callback adapter for :class:`BackgroundResourceSampler`."""

        return self.observe(sample).action == DecisionAction.CONTINUE


def evaluate_injected_samples(
    samples: Sequence[ResourceSample | None],
    policy: ResourcePolicy,
    *,
    expected_device_index: int,
    expected_device_uuid: str,
    now_monotonic_ns: int,
) -> tuple[ResourceDecision, ...]:
    """Pure-data G-11/G-13 helper; never opens CUDA or NVML."""

    controller = ResourceController(
        policy,
        expected_device_index=expected_device_index,
        expected_device_uuid=expected_device_uuid,
        qualification_started_monotonic_ns=now_monotonic_ns,
    )
    decisions: list[ResourceDecision] = []
    for sample in samples:
        decision = controller.observe(sample, now_monotonic_ns=now_monotonic_ns)
        decisions.append(decision)
        if decision.action == DecisionAction.STOP:
            break
    return tuple(decisions)


def validate_fail_closed_semantics(
    policy: ResourcePolicy,
    *,
    expected_device_index: int,
    expected_device_uuid: str,
    now_monotonic_ns: int | None = None,
) -> dict[str, bool]:
    """Self-contained injected boundary checks without stressing a real GPU."""

    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns

    def valid(utilization: float, vram: float, *, at_ns: int = now_ns) -> ResourceSample:
        return ResourceSample(
            schema_version="nmcm.gpu.resource_sample/v1",
            timestamp_utc="1970-01-01T00:00:00Z",
            monotonic_ns=at_ns,
            device_index=expected_device_index,
            device_uuid=expected_device_uuid,
            status="VALID",
            valid=True,
            monitor_backend="INJECTED_TEST_SAMPLE",
            gpu_utilization_percent=utilization,
            device_memory_used_mib=vram,
            device_memory_free_mib=max(0.0, policy.maximum_active_vram_mib - vram),
            device_memory_total_mib=policy.maximum_active_vram_mib,
            project_active_vram_mib=vram,
            admitted_pids=(),
            observed_project_pids=(),
            missing_admitted_pids=(),
            external_compute_pids=(),
            external_compute_memory_mib=0.0,
            process_memory_mib=(),
            error=None,
        )

    epsilon = 1e-6
    evaluate = lambda sample: evaluate_sample(
        sample,
        policy,
        expected_device_index=expected_device_index,
        expected_device_uuid=expected_device_uuid,
        now_monotonic_ns=now_ns,
    ).action
    unknown = ResourceSample.unknown(
        device_index=expected_device_index,
        device_uuid=expected_device_uuid,
        error="injected monitor failure",
        monotonic_ns=now_ns,
    )
    stale_ns = now_ns - int((policy.maximum_sample_age_seconds + epsilon) * 1e9)
    return {
        "utilization_at_limit_continues": evaluate(
            valid(policy.maximum_utilization_percent, 0.0)
        )
        == DecisionAction.CONTINUE,
        "utilization_above_limit_stops": evaluate(
            valid(policy.maximum_utilization_percent + epsilon, 0.0)
        )
        == DecisionAction.STOP,
        "vram_at_limit_continues": evaluate(
            valid(0.0, policy.maximum_active_vram_mib)
        )
        == DecisionAction.CONTINUE,
        "vram_above_limit_stops": evaluate(
            valid(0.0, policy.maximum_active_vram_mib + epsilon)
        )
        == DecisionAction.STOP,
        "unknown_stops": evaluate(unknown) == DecisionAction.STOP,
        "missing_sample_stops": evaluate(None) == DecisionAction.STOP,
        "stale_sample_stops": evaluate(valid(0.0, 0.0, at_ns=stale_ns))
        == DecisionAction.STOP,
    }
