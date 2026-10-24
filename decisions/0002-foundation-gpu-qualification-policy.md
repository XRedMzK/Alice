# Decision 0002: bounded FOUNDATION_GPU_0 policy

Status: **APPROVED AND FROZEN BEFORE OFFICIAL GPU QUALIFICATION**

Date: **2026-08-12**

Authority: `temp.md` SHA-256
`3ccedc4d2e8db82cdfe4e6463755f1d7060d9cd94a955f1704e7f31560c076b9`.

## Decision

The accepted `nmcm.foundation0.cpu/v2` artifacts remain immutable inputs. GPU
qualification uses a separate Python environment and an additive `nmcm_gpu`
control-plane package. It performs no training and does not change synthetic
equations, A1/A2/A3/B/C semantics, CPU thresholds, CPU schemas, or CPU tests.

For this bounded work package, the newer authority deliberately strengthens
the utilization rule in `RESOURCE_POLICY.md`: every valid active hardware
sample and the whole-window mean must be at most `80%`. A sample above `80%`
causes a safe-boundary stop and `RESOURCE_FAIL`. The hard active-VRAM limit is
`28672 MiB`. Missing, stale, inconsistent, or failed monitoring is
`GPU_ENVIRONMENT_BLOCKED`/`INFRA_FAIL`, never a skipped PASS.

The predeclared float32 CPU/GPU arithmetic policy is
`abs(gpu-cpu) <= 1e-6 + 1e-5*abs(cpu)`. Shape, dtype, finiteness, predictions,
accuracies, and scientific gate decisions remain exact. Pure placement copies
and same-GPU deterministic replay must be bitwise exact after canonical CPU
copy. CPU scientific thresholds are unchanged.

The official attempt is immutable and non-resumable, starts before its first
official CUDA operation, and has a `840 s` soft cutoff and `900 s` hard limit.
No Alice V1, language/Qwen work, DAY0, system driver, system CUDA, kernel, or
system-package change is authorized.

Machine-readable details are frozen in
`gpu_configs/foundation0_gpu.json` before the official run.

## Outcome

The accepted fresh attempt `foundation0_gpu_v1_20260812_official2` completed
on 2026-08-12 with all G-01…G-15 gates PASS. Total official wall time was
`32.141692678 s`; the continuously monitored workload window was
`31.041609896 s`, active utilization peaked at `3%`, and project VRAM peaked
at `874 MiB`. The frozen policy and all evaluator bindings remained unchanged
before and after the run.

The prior `official1` evidence is retained but not accepted: a final audit
found G-14 CUDA scoring occurred after its monitor stopped. The bounded bug was
fixed and a fresh JUnit plus distinct official attempt were run; no `official1`
evidence was reused.

The accepted outcome is recorded in
`configs/foundation0_gpu_acceptance.json` and
`docs/FOUNDATION_0_GPU_ACCEPTANCE.md`. This consumes the bounded execution
authority in this decision; it does not authorize a repeat, Alice, Qwen,
language work, training, or DAY0.
