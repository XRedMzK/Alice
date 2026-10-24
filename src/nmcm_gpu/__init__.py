"""GPU control-plane support for the bounded FOUNDATION_0 qualification.

The package is deliberately separate from :mod:`nmcm`: importing it does not
initialise CUDA or mutate the accepted CPU baseline.
"""

from .controller import (
    AdmissionRequest,
    DecisionAction,
    ResourceController,
    ResourceDecision,
    evaluate_sample,
    evaluate_injected_samples,
    validate_fail_closed_semantics,
)
from .environment import (
    EnvironmentReport,
    ExpectedGpuEnvironment,
    GpuProfileError,
    ResourcePolicy,
    load_gpu_profile,
    load_resource_policy,
    probe_environment,
)
from .monitor import (
    BackgroundResourceSampler,
    NvmlResourceMonitor,
    ResourceSample,
    ResourceWindowSummary,
    summarize_resource_window,
)

__all__ = [
    "AdmissionRequest",
    "BackgroundResourceSampler",
    "DecisionAction",
    "EnvironmentReport",
    "ExpectedGpuEnvironment",
    "GpuProfileError",
    "NvmlResourceMonitor",
    "ResourceController",
    "ResourceDecision",
    "ResourcePolicy",
    "ResourceSample",
    "ResourceWindowSummary",
    "evaluate_injected_samples",
    "evaluate_sample",
    "load_gpu_profile",
    "load_resource_policy",
    "probe_environment",
    "summarize_resource_window",
    "validate_fail_closed_semantics",
]
