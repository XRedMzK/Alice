# Resource policy

Status: **FROZEN; GPU measurements remain NOT_RUN**

Reference hardware: NVIDIA GeForce RTX 5090, 32 GiB VRAM

This policy governs runtime admission, measurement, placement, and reporting.
The resource controller is part of the non-cognitive control plane: it never
interprets latent values or chooses a cognitive mechanism by semantic rule.

## 1. Binding limits

- **Target aggregate GPU utilization:** `<= 80%` over the declared active
  experiment window.
- **Hard active VRAM peak:** `<= 28672 MiB` (`28 GiB`).
- **Optimization objective:** minimum research wall-clock subject to safety,
  isolation, and evidence quality—not maximum GPU utilization.

The utilization number is a target/admission objective, not an instantaneous
hard ceiling. The VRAM number is a hard runtime limit. All concurrent NMCM work
on the device is considered together.

## 2. Measurement definitions

### Active VRAM

`project_active_vram_mib(t)` is the sum of device memory attributable to every
admitted NMCM process across all concurrently active bundles, including
parameters, activations, owned state, framework allocator cache/reservations,
CUDA contexts, candidate workers, and monitoring overhead. The reported peak is
the conservative maximum of process-level device sampling and framework
high-water marks where both are available.

The hard evidence metric is:

```text
observed_project_peak_mib = max_t(sum over admitted NMCM PIDs p of
                                  NVML_used_memory_mib(p, t))
```

Synchronized framework high-water marks cross-check the sampled aggregate; any
higher attributable framework value becomes the reported conservative peak.
PID set, timestamps, sampling gaps, and external processes are retained.

The scheduler admits reservations, not optimistic tensor-only estimates. It
also checks device-wide free memory so unrelated processes cannot make an
otherwise valid reservation unsafe. Exact tensor bytes are reported separately
from allocator/process VRAM; they are not interchangeable.

Every reserved byte has one ledger scope—worker, shared immutable allocation,
module instance, or bundle workspace—to prevent omission and double counting.
For all concurrently admitted bundles:

```text
pre_margin = sum(unique worker context/runtime overhead)
           + sum(unique shared read-only GPU allocations)
           + sum(HOT module-instance incremental reservations)
           + sum(bundle live activation/workspace + transport + telemetry)

safety_margin = max(512 MiB, ceil(0.05 * pre_margin))
project_reservation = pre_margin + safety_margin
```

A new worker pays its calibrated context/runtime term once. A genuinely shared
immutable allocation is reserved once; every mutable candidate instance is
reserved separately. Bundle activation/workspace is derived from the declared
schedule, including simultaneous workers, not from a serial maximum.

An admission reservation that could exceed `28672 MiB` is queued or rejected.
An observed overrun stops new admissions, ends the offending work at the next
safe boundary when possible, records `RESOURCE_FAIL`, and does not publish a
scientific PASS.

### Aggregate GPU utilization

On the single reference GPU, hardware-reported device busy percentage already
aggregates concurrent kernels; worker percentages MUST NOT be arithmetically
summed. The primary value is the time-weighted mean over the registered active
window after any separately reported setup/warm-up interval. Peak, sampling
period, missing samples, and external-load baseline are also recorded.

The initial proposed sampler period is one second. A qualifying active window
lasts at least 30 seconds and contains at least 30 valid hardware samples; setup
and a declared warm-up (proposed: five seconds) are reported separately. A
rolling 10-sample mean above 80% for three consecutive complete windows blocks
new admissions. Admission resumes only after two consecutive complete windows
at or below 75%; running work is not killed merely for this target breach. The
final gate still requires whole-active-window mean utilization at or below 80%.

These windows remain approval items because the target runtime/driver stack is
not yet selected. An experiment contaminated by material unaccounted external
GPU load is invalid for resource qualification rather than silently adjusted.

### Latency and GPU time

Per-module latency is wall time from structural invocation to completed output
at a synchronized measurement boundary. GPU time uses device events when
available. Reports include count, mean, p50, p95, and maximum plus warm-up
policy. End-to-end bundle wall-clock includes setup, queueing, checkpoint,
evaluation, and transfer time.

Per-module mean/peak utilization and scoped peak VRAM are measured in an
isolated qualification window for that module under its named workload.
Concurrent runs report exact owned tensor bytes and aggregate process/device
VRAM/utilization; they do not fabricate per-module utilization by apportioning
device samples.

### Transfer cost

Every placement transition records source/destination tier, bytes, wall time,
GPU time if applicable, synchronization cost, and reason. Transfer metrics are
attributed to a module instance and experiment.

## 3. Per-module declaration

Each manifest declares estimates or limits for:

- parameter count and bytes;
- persistent and private state bytes;
- isolated total active-VRAM peak (including its calibrated context baseline)
  and incremental module reservation above that baseline;
- expected GPU utilization contribution under a named workload (used only as
  a scheduling estimate, never naïvely summed as measured utilization);
- p95 latency budget;
- offload bytes, expected latency, allowed tiers, and safe boundary;
- measurement method and fixture/batch assumptions.

Runtime telemetry reports measured values separately. Estimates are versioned
when the module or workload changes.

The exact manifest field `incremental_active_vram_reservation_mib` is the
module-instance ledger term, not isolated process total. Context, live
activation/transport workspace, telemetry, shared allocation, and safety margin
are separate bundle/controller terms.

## 4. Admission and concurrency

Before launch, the scheduler evaluates:

1. bundle type and mutable-state isolation;
2. active VRAM reservations plus controller safety margin;
3. bundle-level estimated utilization and expected wall-clock benefit;
4. module placement and transfer requirements;
5. current device capacity and external load;
6. telemetry availability needed for the requested gate.

Several independent 20% candidates may be preferable to one 60% job when they
fit the VRAM limit and reduce total wall-clock. Same-live-state causal arms are
coordinated as one bundle. Work is not marked independent merely because it has
different process IDs.

Bundle utilization is estimated from an isolated profile of the declared graph
workload or a preregistered conservative bundle model. Sequential module
percentages are never added, and per-module isolated utilization is diagnostic,
not an additive reservation. Measured acceptance always uses aggregate device
samples from the real bundle window.

The scheduler exposes queue/admit/start/finish/offload events and the numeric
reason for every resource decision. It cannot inspect prompt text, intent,
ACT/TARGET, latent values, or evaluator semantics.

## 5. HOT/WARM/COLD placement

| Tier | Medium | Intended use |
|---|---|---|
| **HOT** | GPU VRAM | Active parameters, activations, and state needed for immediate execution. |
| **WARM** | System RAM | Inactive module parameters/state expected to return soon; pinned RAM only when justified. |
| **COLD** | RAM-backed or persistent storage | Durable shards and infrequently used blocks. |

The future total Alice footprint may exceed 28 GiB because the hard limit
applies to the active GPU working set. Offload is acceptable only at declared
safe boundaries with preserved promised state.

The later GPU qualification requires two distinct cycles: CPU→GPU→CPU verifies
initial placement and return parity, while GPU→RAM→GPU verifies a claimed
HOT→WARM→HOT offload/restore. The latter is required for every module claiming
that support. Both are outside current CPU acceptance and remain `NOT_RUN`.
Live automatic COLD paging is outside FOUNDATION_0; persistent checkpoint
restore is tested separately on CPU.

## 6. Offload and anti-thrashing rules

A valid round-trip:

- begins from a captured module state and input;
- transfers the declared parameters and state to the allowed tier;
- releases the corresponding HOT working set, leaving no owned tensor on the
  GPU and reclaiming allocator cache at the declared WARM boundary;
- restores the same module/version/state;
- reproduces state and continuation output within registered tolerance;
- records bytes, latency, synchronization, and post-restore VRAM;
- leaves checkpoint and compatibility identities unchanged.

At WARM, `owned_gpu_tensor_bytes` is exactly zero. Framework active/reserved
bytes and process VRAM are compared with a separately measured CUDA-context
baseline; the report must show the owned allocation was released rather than
duplicated in RAM while remaining resident on GPU. Counter resolution and
baseline tolerance are preregistered.

Each module declares maximum round trips or transfers per minute and a latency
budget. The controller reports transfer-time/wall-time ratio. Repeated movement
that exceeds either declaration is a resource-policy failure; it cannot be
justified by lower apparent VRAM. Universal thresholds are not invented before
the synthetic workload and runtime are approved.

A “transfer count” in the foundation gate means one lifecycle transition, not
the number of tensor-copy operations inside that transition. State-component
coverage, round-trip latency, and rate limits are all checked against the
manifest.

## 7. Required telemetry

At minimum, machine-readable telemetry can be grouped by module instance and
shows:

- module/version/compatibility, experiment, candidate, and lifecycle state;
- wall latency, GPU time, active VRAM, and utilization sample provenance;
- parameter/state bytes and state norm/finite statistics;
- activation/output mean, standard deviation, norm, extrema, density, and
  finite statistics where defined;
- transfer count/bytes/time and current placement;
- checkpoint/state schema and snapshot identity;
- local PASS/FAIL/NOT_RUN status;
- dropped samples, collection overhead, and timestamps.

JSON Lines is the foundation stream and the compact acceptance report is the
human summary. A GUI and a new observability service are out of scope.
Telemetry-on versus telemetry-off equivalence is a mandatory non-interference
test.

## 8. Environment and fail-closed behavior

Resource qualification requires working hardware counters and the approved
runtime. A missing NVIDIA driver, unavailable device, absent profiler/counter,
or skipped CUDA test yields `NOT_RUN`/`INFRA_FAIL`, never PASS. Deterministic
CPU tests may close contract stages but cannot substitute for the GPU VRAM,
utilization, GPU-time, or offload gates.

No workload may download model weights or start language/Alice training as part
of resource discovery. Any dependency installation, bounded target-GPU window,
or synthetic optimization requires its own approval recorded in
[STATUS.md](STATUS.md).
