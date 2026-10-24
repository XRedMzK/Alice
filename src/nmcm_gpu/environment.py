"""Strict loading and read-only probing of the frozen GPU environment.

No CUDA or NVML work happens at import time.  This matters because the CPU
qualification environment must remain usable even where NVIDIA devices are
hidden by the execution sandbox.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


GPU_PROFILE_SCHEMA_VERSION = "nmcm.foundation0.gpu_profile/v1"
GPU_ENVIRONMENT_SCHEMA_VERSION = "nmcm.gpu.environment/v1"
PINNED_NVML_DISTRIBUTION = "nvidia-ml-py"
PINNED_NVML_VERSION = "13.610.43"
SUPPORTED_UTILIZATION_SEMANTICS = (
    "each valid active sample and whole-window mean must be <=80; "
    "a higher sample stops at the safe boundary"
)


class GpuProfileError(ValueError):
    """The frozen profile is absent, malformed, or has unsupported semantics."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GpuProfileError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GpuProfileError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GpuProfileError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise GpuProfileError(f"{name} must be finite and >= {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GpuProfileError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ReservationLedgerPolicy:
    """Fixed conservative reservations from the frozen qualification profile."""

    cuda_context_mib: float
    telemetry_and_transport_mib: float
    bundle_workspace_mib: float
    per_mutable_module_instance_mib: float
    safety_margin_mib: float


@dataclass(frozen=True)
class ResourcePolicy:
    """Validated resource semantics used by both monitor and controller."""

    hard_wall_seconds: float
    soft_wall_seconds: float
    maximum_utilization_percent: float
    maximum_active_vram_mib: float
    sample_period_seconds: float
    minimum_active_window_seconds: float
    minimum_valid_samples: int
    maximum_sample_age_seconds: float
    maximum_external_compute_memory_mib: float
    warmup_seconds: float
    offload_context_baseline_tolerance_mib: float
    utilization_interpretation: str
    unknown_or_monitor_error: str
    zero_gpu_estimates_in_cpu_manifests: str
    reservation_ledger: ReservationLedgerPolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourcePolicy":
        raw = _mapping(value, "resource_policy")
        interpretation = _string(
            raw.get("utilization_interpretation"),
            "resource_policy.utilization_interpretation",
        )
        if interpretation != SUPPORTED_UTILIZATION_SEMANTICS:
            raise GpuProfileError(
                "unsupported utilization semantics; refusing to weaken the frozen policy"
            )
        fail_mode = _string(
            raw.get("unknown_or_monitor_error"),
            "resource_policy.unknown_or_monitor_error",
        )
        if fail_mode != "fail_closed":
            raise GpuProfileError("resource monitor failure policy must be fail_closed")

        ledger_raw = _mapping(
            raw.get("reservation_ledger_mib"),
            "resource_policy.reservation_ledger_mib",
        )
        ledger = ReservationLedgerPolicy(
            cuda_context_mib=_number(
                ledger_raw.get("cuda_context"),
                "reservation_ledger_mib.cuda_context",
            ),
            telemetry_and_transport_mib=_number(
                ledger_raw.get("telemetry_and_transport"),
                "reservation_ledger_mib.telemetry_and_transport",
            ),
            bundle_workspace_mib=_number(
                ledger_raw.get("bundle_workspace"),
                "reservation_ledger_mib.bundle_workspace",
            ),
            per_mutable_module_instance_mib=_number(
                ledger_raw.get("per_mutable_module_instance"),
                "reservation_ledger_mib.per_mutable_module_instance",
            ),
            safety_margin_mib=_number(
                ledger_raw.get("safety_margin"),
                "reservation_ledger_mib.safety_margin",
            ),
        )
        policy = cls(
            hard_wall_seconds=_number(
                raw.get("hard_wall_seconds"), "resource_policy.hard_wall_seconds", minimum=0.001
            ),
            soft_wall_seconds=_number(
                raw.get("soft_wall_seconds"), "resource_policy.soft_wall_seconds", minimum=0.001
            ),
            maximum_utilization_percent=_number(
                raw.get("maximum_utilization_percent"),
                "resource_policy.maximum_utilization_percent",
            ),
            maximum_active_vram_mib=_number(
                raw.get("maximum_active_vram_mib"),
                "resource_policy.maximum_active_vram_mib",
                minimum=0.001,
            ),
            sample_period_seconds=_number(
                raw.get("sample_period_seconds"),
                "resource_policy.sample_period_seconds",
                minimum=0.001,
            ),
            minimum_active_window_seconds=_number(
                raw.get("minimum_active_window_seconds"),
                "resource_policy.minimum_active_window_seconds",
                minimum=0.001,
            ),
            minimum_valid_samples=_integer(
                raw.get("minimum_valid_samples"),
                "resource_policy.minimum_valid_samples",
                minimum=1,
            ),
            maximum_sample_age_seconds=_number(
                raw.get("maximum_sample_age_seconds"),
                "resource_policy.maximum_sample_age_seconds",
                minimum=0.001,
            ),
            maximum_external_compute_memory_mib=_number(
                raw.get("maximum_external_compute_memory_mib"),
                "resource_policy.maximum_external_compute_memory_mib",
            ),
            warmup_seconds=_number(
                raw.get("warmup_seconds"), "resource_policy.warmup_seconds"
            ),
            offload_context_baseline_tolerance_mib=_number(
                raw.get("offload_context_baseline_tolerance_mib"),
                "resource_policy.offload_context_baseline_tolerance_mib",
            ),
            utilization_interpretation=interpretation,
            unknown_or_monitor_error=fail_mode,
            zero_gpu_estimates_in_cpu_manifests=_string(
                raw.get("zero_gpu_estimates_in_cpu_manifests"),
                "resource_policy.zero_gpu_estimates_in_cpu_manifests",
            ),
            reservation_ledger=ledger,
        )
        if policy.maximum_utilization_percent > 100.0:
            raise GpuProfileError("maximum_utilization_percent cannot exceed 100")
        if policy.soft_wall_seconds > policy.hard_wall_seconds:
            raise GpuProfileError("soft_wall_seconds cannot exceed hard_wall_seconds")
        if policy.maximum_sample_age_seconds < policy.sample_period_seconds:
            raise GpuProfileError(
                "maximum_sample_age_seconds cannot be shorter than sample_period_seconds"
            )
        return policy


@dataclass(frozen=True)
class ExpectedGpuEnvironment:
    python: str
    torch: str
    cuda_build: str
    device_index: int
    device_name: str
    compute_capability: tuple[int, int]
    driver_version: str
    device_uuid: str
    total_memory_mib: int
    nvml_distribution: str = PINNED_NVML_DISTRIBUTION
    nvml_version: str = PINNED_NVML_VERSION

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]) -> "ExpectedGpuEnvironment":
        env = _mapping(profile.get("environment"), "environment")
        capability = env.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in capability)
        ):
            raise GpuProfileError("environment.compute_capability must be [major, minor]")
        monitor = _string(env.get("monitor"), "environment.monitor")
        expected_monitor = f"NVML via {PINNED_NVML_DISTRIBUTION}=={PINNED_NVML_VERSION}"
        if monitor != expected_monitor:
            raise GpuProfileError(
                f"environment.monitor must remain pinned to {expected_monitor!r}"
            )
        return cls(
            python=_string(env.get("python"), "environment.python"),
            torch=_string(env.get("torch"), "environment.torch"),
            cuda_build=_string(env.get("cuda_build"), "environment.cuda_build"),
            device_index=_integer(env.get("device_index"), "environment.device_index"),
            device_name=_string(env.get("device_name"), "environment.device_name"),
            compute_capability=(capability[0], capability[1]),
            driver_version=_string(env.get("driver_version"), "environment.driver_version"),
            device_uuid=_string(env.get("device_uuid"), "environment.device_uuid"),
            total_memory_mib=_integer(
                env.get("total_memory_mib"), "environment.total_memory_mib", minimum=1
            ),
        )


@dataclass(frozen=True)
class EnvironmentReport:
    schema_version: str
    status: str
    expected: Mapping[str, Any]
    observed: Mapping[str, Any]
    checks: Mapping[str, bool]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["errors"] = list(self.errors)
        expected = dict(result["expected"])
        if isinstance(expected.get("compute_capability"), tuple):
            expected["compute_capability"] = list(expected["compute_capability"])
        result["expected"] = expected
        return result


def load_gpu_profile(path: str | Path) -> dict[str, Any]:
    """Load the frozen profile and reject unknown control semantics."""

    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuProfileError(f"cannot load GPU profile {profile_path}: {exc}") from exc
    profile = dict(_mapping(raw, "profile"))
    if profile.get("schema_version") != GPU_PROFILE_SCHEMA_VERSION:
        raise GpuProfileError(
            f"unsupported GPU profile schema_version {profile.get('schema_version')!r}"
        )
    if profile.get("status") != "FROZEN_BEFORE_OFFICIAL_QUALIFICATION":
        raise GpuProfileError("GPU profile must be frozen before qualification")
    ExpectedGpuEnvironment.from_profile(profile)
    ResourcePolicy.from_mapping(_mapping(profile.get("resource_policy"), "resource_policy"))
    return profile


def load_resource_policy(path: str | Path) -> ResourcePolicy:
    profile = load_gpu_profile(path)
    return ResourcePolicy.from_mapping(
        _mapping(profile.get("resource_policy"), "resource_policy")
    )


def _decoded(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def probe_environment(profile_or_path: Mapping[str, Any] | str | Path) -> EnvironmentReport:
    """Probe the real runtime without modifying drivers, CUDA, or system state.

    An unavailable CUDA device is explicitly reported as
    ``GPU_ENVIRONMENT_BLOCKED``.  Dependency or identity drift is ``FAIL``;
    neither state can be interpreted as a GPU PASS.
    """

    profile = (
        load_gpu_profile(profile_or_path)
        if isinstance(profile_or_path, (str, Path))
        else dict(profile_or_path)
    )
    expected = ExpectedGpuEnvironment.from_profile(profile)
    expected_dict = asdict(expected)
    expected_dict["compute_capability"] = list(expected.compute_capability)
    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "device_index": None,
        "device_name": None,
        "compute_capability": None,
        "driver_version": None,
        "device_uuid": None,
        "total_memory_mib": None,
        "nvml_distribution": PINNED_NVML_DISTRIBUTION,
        "nvml_version": None,
    }
    errors: list[str] = []
    blocked = False

    try:
        observed["nvml_version"] = importlib.metadata.version(PINNED_NVML_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"missing {PINNED_NVML_DISTRIBUTION} package")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on candidate environment
        errors.append(f"cannot import torch: {type(exc).__name__}: {exc}")
        torch = None

    if torch is not None:
        observed["torch"] = str(torch.__version__)
        observed["cuda_build"] = torch.version.cuda
        try:
            observed["cuda_available"] = bool(torch.cuda.is_available())
            observed["device_count"] = int(torch.cuda.device_count())
        except Exception as exc:  # pragma: no cover - runtime dependent
            blocked = True
            errors.append(f"CUDA discovery failed: {type(exc).__name__}: {exc}")
        if not observed["cuda_available"]:
            blocked = True
            errors.append("CUDA device is unavailable in this execution context")
        elif observed["device_count"] <= expected.device_index:
            blocked = True
            errors.append(
                f"CUDA device index {expected.device_index} is not visible "
                f"(count={observed['device_count']})"
            )
        else:
            try:
                props = torch.cuda.get_device_properties(expected.device_index)
                observed["device_index"] = expected.device_index
                observed["device_name"] = str(props.name)
                observed["compute_capability"] = [int(props.major), int(props.minor)]
            except Exception as exc:  # pragma: no cover - runtime dependent
                blocked = True
                errors.append(f"CUDA property query failed: {type(exc).__name__}: {exc}")

    nvml_started = False
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml_started = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(expected.device_index)
        observed["driver_version"] = _decoded(pynvml.nvmlSystemGetDriverVersion())
        observed["device_uuid"] = _decoded(pynvml.nvmlDeviceGetUUID(handle))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        observed["total_memory_mib"] = int(memory.total) // (1024 * 1024)
    except Exception as exc:  # pragma: no cover - runtime dependent
        blocked = True
        errors.append(f"NVML device query failed: {type(exc).__name__}: {exc}")
    finally:
        if nvml_started:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:  # pragma: no cover - runtime dependent
                errors.append(f"NVML shutdown failed: {type(exc).__name__}: {exc}")

    checks = {
        "python": observed["python"] == expected.python,
        "torch": observed["torch"] == expected.torch,
        "cuda_build": observed["cuda_build"] == expected.cuda_build,
        "cuda_available": observed["cuda_available"] is True,
        "device_index": observed["device_index"] == expected.device_index,
        "device_name": observed["device_name"] == expected.device_name,
        "compute_capability": observed["compute_capability"]
        == list(expected.compute_capability),
        "driver_version": observed["driver_version"] == expected.driver_version,
        "device_uuid": observed["device_uuid"] == expected.device_uuid,
        "total_memory_mib": observed["total_memory_mib"] == expected.total_memory_mib,
        "nvml_version": observed["nvml_version"] == expected.nvml_version,
    }
    for name, passed in checks.items():
        if not passed and not (blocked and name in {"cuda_available", "device_index"}):
            errors.append(
                f"environment identity mismatch for {name}: "
                f"expected={expected_dict.get(name)!r}, observed={observed.get(name)!r}"
            )
    status = "GPU_ENVIRONMENT_BLOCKED" if blocked else ("PASS" if all(checks.values()) else "FAIL")
    return EnvironmentReport(
        schema_version=GPU_ENVIRONMENT_SCHEMA_VERSION,
        status=status,
        expected=expected_dict,
        observed=observed,
        checks=checks,
        errors=tuple(dict.fromkeys(errors)),
    )
