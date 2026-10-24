> Public export: the host GPU identifier is redacted. The original report is
> unchanged; SOURCE_PROVENANCE.json records both original and public hashes.

# FOUNDATION_0 GPU acceptance

Status: **15/15 PASS — NMCM_V2_FOUNDATION_0 CLOSED**

Date: **2026-08-12**

Baseline: `nmcm.foundation0.gpu/v1`, qualifying the accepted immutable CPU
baseline `nmcm.foundation0.cpu/v2`.

The official attempt performed no training, architecture search, synthetic
equation change, CPU-threshold change, model download, Alice work, or DAY0
work. The predeclared policy is in
[foundation0_gpu.json](../gpu_configs/foundation0_gpu.json), and the compact
machine-readable acceptance record is
[foundation0_gpu_acceptance.json](../configs/foundation0_gpu_acceptance.json).

## 1. Exact GPU environment

The separate project-local `.venv-gpu` used Python `3.12.3`, PyTorch
`2.13.0+cu130`, CUDA build/runtime `13.0`, NumPy `2.5.2`, safetensors `0.8.0`,
jsonschema `4.26.0`, pytest `9.1.1`, and `nvidia-ml-py 13.610.43`. The full
resolved dependency set is captured in
[requirements-gpu.lock](../requirements-gpu.lock). The CPU environment,
driver, system CUDA, kernel, and system packages were not changed.

Runtime was eager float32 with deterministic algorithms enabled, autocast and
compile disabled, highest float32 matmul precision, cuDNN benchmark disabled,
and TF32 disabled.

## 2. Device, driver, and runtime identity

| Field | Observed |
|---|---|
| Device | NVIDIA GeForce RTX 5090, `cuda:0` |
| UUID | `<REDACTED_GPU_UUID>` |
| Compute capability | `12.0` |
| Physical memory | `32607 MiB` |
| NVIDIA driver | `595.84` |
| PyTorch CUDA build/runtime | `13.0` |
| Monitor | direct pinned NVML via `nvidia-ml-py==13.610.43` |

PyTorch and NVML identity checks agreed, and the UUID remained stable. The
earlier no-device result inside the default sandbox was not treated as a GPU
failure or simulated PASS; official execution used real device access.

## 3. G-01 through G-15

| Gate | Required evidence | Result |
|---|---|---|
| G-01 | CUDA runtime and device identity | **PASS** |
| G-02 | A1/A2/A3/B/C execute on `cuda:0` | **PASS** |
| G-03 | CPU → GPU → CPU module/state round-trip | **PASS** |
| G-04 | HOT → WARM → HOT continuation | **PASS** |
| G-05 | Five independent shards restore to GPU | **PASS** |
| G-06 | Whole A1-B-C checkpoint continuation | **PASS** |
| G-07 | Telemetry OFF/ON non-interference | **PASS** |
| G-08 | Per-module wall/CUDA/VRAM/state/transfer telemetry | **PASS** |
| G-09 | Two independent candidate bundles overlap | **PASS** |
| G-10 | Module/state/RNG/stream/artifact isolation | **PASS** |
| G-11 | Resource-controller boundary enforcement | **PASS** |
| G-12 | Actual active VRAM ≤ `28672 MiB` | **PASS** |
| G-13 | Unknown/missing/stale monitoring fails closed | **PASS** |
| G-14 | CPU/GPU functional equivalence | **PASS** |
| G-15 | Exact conjunctive qualification | **PASS** |

JUnit contains exactly 15 unique expected cases: 15 passed, 0 failed, 0
errors, 0 skipped, and 0 XFAIL. G-15 additionally recomputed the material
conditions from evidence rather than trusting the JUnit count alone. JUnit was
newer than every bound evaluator input, and all 22 pre/post/current binding
digests matched.

## 4. CPU/GPU parity

The policy was frozen before official execution: float32 tensors must have the
same shape and dtype, be finite, and satisfy
`abs(gpu-cpu) <= 1e-6 + 1e-5 * abs(cpu)`. Predictions, accuracies, and gate
decisions must be exact. Pure transfers and deterministic same-GPU replay must
be bitwise exact.

All five modules' outputs and carried states met the numerical policy, and
task predictions and accepted behavioral decisions matched exactly. On the
official held-out GPU workload:

| Candidate | Accuracy | Mean cosine to canonical | Normalized RMSE | Decision |
|---|---:|---:|---:|---|
| A1 | 1.000000 | 1.000000 | 0.00006681 | admitted |
| A2 | 1.000000 | 0.999935 | 0.01219726 | admitted |
| A3 | 0.000000 | -1.000000 | 1.99999905 | rejected; negative control PASS |

The accepted CPU scientific thresholds were unchanged.

A final read-only comparison against the accepted CPU evaluator binding found
all 31 CPU implementation/config/test/lock inputs bit-for-bit unchanged. The
only differing historical binding input is `temp.md`, which is the later GPU
work-package authority rather than CPU implementation state. The accepted CPU
descriptor, results, and JUnit hashes match the frozen GPU profile.

## 5. Offload round-trip

All A1/A2/A3/B/C HOT → WARM → HOT tests passed. Parameters, buffers, private
state, next output, and final state were exact. Each WARM phase reported zero
module/state-owned CUDA bytes. Measured device-to-host wall time ranged from
`0.099` to `0.189 ms`; host-to-device restore ranged from `0.137` to
`0.325 ms`.

## 6. Checkpoint and shard restore

All five independent accepted CPU shards restored to GPU, produced an exact
same-GPU continuation, and matched the accepted CPU continuation under the
frozen tolerance. The whole `nmcm.whole_checkpoint/v2` snapshot
`qualification-1` restored graph `foundation_a1bc@2.0.0` at logical step 4;
its CPU/GPU continuation passed the same policy.

## 7. Telemetry non-interference

Telemetry OFF and ON runs had exact outputs, states, parameter digests, CPU RNG
state, and CUDA RNG state. CUDA time came from synchronized CUDA events, not
from Python wall time. Allocator allocated/reserved high-water, owned tensor
bytes, private-state bytes, and transfer timing were recorded separately.

## 8. Per-module latency, memory, and transfer telemetry

Each row summarizes five isolated invocations with batch 1 and sequence length
8. `alloc/reserved` is the process allocator high-water in that module's
measurement scope; it is not mislabeled as module-owned memory. Exact owned
tensor bytes are reported separately.

| Module | Wall mean/max ms | CUDA mean/max ms | Alloc/reserved peak MiB | Owned tensor KiB | State B | Transfer wall ms |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 1.319 / 4.667 | 1.315 / 4.665 | 32.312 / 34.000 | 6.125 | 128 | 0.238 |
| A2 | 1.230 / 1.256 | 1.226 / 1.255 | 32.377 / 34.000 | 16.375 | 128 | 0.446 |
| A3 | 0.557 / 0.560 | 0.552 / 0.554 | 32.315 / 34.000 | 6.129 | 128 | 0.437 |
| B | 0.748 / 0.791 | 0.742 / 0.789 | 32.164 / 34.000 | 8.125 | 128 | 0.300 |
| C | 0.495 / 0.518 | 0.489 / 0.514 | 32.185 / 34.000 | 3.062 | 64 | 0.294 |

## 9. Concurrency and isolation

Independent A1+B and A2+B bundles ran in one process on two explicit,
non-default CUDA streams for eight iterations. Their CUDA event work windows
overlapped by `15.211 ms`; concurrent makespan was `21.147 ms`. Both candidates
matched their serial GPU baselines exactly.

Module instances, parameter/buffer storage, mutable state, input storage, CUDA
generators, streams, and artifact roots were distinct. The result proves
correct controllable overlap; it does not claim a speedup or CUPTI-proven
simultaneous kernel residency.

## 10. Resource-controller evidence

The active monitor window retained 31 valid 1-second samples over `31.042 s`
(32 records including the pre-active sample), with no missing admitted PID,
unknown counter, sample error, UUID change, or violation.

| Resource | Limit | Observed |
|---|---:|---:|
| Peak utilization | 80% | 3% |
| Time-weighted mean utilization | 80% | 3.000% |
| Project active VRAM peak | 28672 MiB | 874 MiB |
| Maximum active sample gap | policy-valid | 1.00001 s |
| External compute memory peak | 256 MiB contamination guard | 62.523 MiB |

Injected tests proved that utilization at 80% and VRAM at 28672 MiB remain
admissible, while values above either limit stop. Missing, stale, unknown, and
monitor-error states also stop/deny admission. The real device was not stressed
to test failure behavior.

## 11. Wall-clock

The total official attempt took `32.141692678 s`, below the `900 s` hard
limit. Its continuously monitored qualification-workload window took
`31.041609896 s` and included the final G-14 CUDA scoring. The separately
generated, fresh bound JUnit suite took `0.971 s`. Environment installation,
implementation, audits, and explicitly labeled development smokes were
excluded from the official-attempt timer and were not promoted into PASS
evidence.

## 12. Remaining limitations and boundary

- Qualification covers one host, one RTX 5090, eager float32, and the bounded
  synthetic A1/A2/A3/B/C workload; it is not a production performance claim.
- Concurrency evidence is shared-device CUDA-event window overlap, not a
  profiler claim about simultaneous kernel residency, and no speedup is
  required or claimed.
- Allocator high-water is process-scoped; exact module-owned bytes are the
  separate ownership counters in the evidence.
- HOT/WARM movement is explicit. Automatic paging and COLD/disk paging were not
  implemented.
- Per-module transfer timing is machine-readable in `results.json` and backed
  by G-04 CUDA-event transfer records; the invocation-only `telemetry.jsonl`
  contains no separate placement-transfer event rows.
- One external compute process used at most `62.523 MiB`, below the frozen
  `256 MiB` contamination guard; the host was therefore controlled, not
  completely idle.
- `RESOURCE_POLICY.md` is the frozen CPU-era baseline and consequently retains
  historical GPU `NOT_RUN` wording. Decision 0002 and the hash-bound GPU
  profile explicitly strengthened and governed the later target-GPU attempt;
  this acceptance report records the resulting current status.
- Any hardware, driver, dependency, profile, evaluator, test, or module change
  requires a new bounded qualification. This result authorizes no Qwen/model
  download, Alice V1 work, production training, or DAY0.
- Several embedded evidence/profile/JUnit paths are absolute to this local
  workspace. Their hashes are valid here, but moving the bundle requires an
  explicit path rebinding plus integrity revalidation.
- The top-level evidence schema is closed, while some nested evidence objects
  remain permissive. The accepted result compensates with evaluator-side
  semantic checks, child-artifact hashes, exact JUnit names, and two independent
  audits; deeper closed schemas are a future hardening option, not a missing
  FOUNDATION_0 gate.

## Evidence and closure

| Artifact | SHA-256 |
|---|---|
| [results.json](../runs/foundation0_gpu_v1_20260812_official2/results.json) | `f8ed364244fff3da2edb047a4c0392c8c606c5f7a25e1829da9a0d61a8e9e068` |
| [junit.xml](../runs/foundation0_gpu_v1_20260812_official2_tests/junit.xml) | `e5b04b5fc3d1b356ffbb404c1803ef391b6d3fe11566c8ac4a1754de9bcb6ada` |
| [telemetry.jsonl](../runs/foundation0_gpu_v1_20260812_official2/telemetry.jsonl) | `93514e2a3120775cb8a2636cdff8f97f686e18193669d454ab31d6544a16a351` |
| [resource_samples.jsonl](../runs/foundation0_gpu_v1_20260812_official2/resource_samples.jsonl) | `d4e9df5fbb9acc2fa494019dff7167561547a9761863d19502347cdc32efb311` |
| [frozen GPU profile](../gpu_configs/foundation0_gpu.json) | `681fe3d5caf861fa4c480604c9e721f0fcabd2405a8f2be96292aa4c59f463be` |

The earlier `official1` result is retained but not accepted: a final review
found its G-14 CUDA scoring occurred after the monitor stopped. The lifecycle
was fixed and all bound tests and qualification work were rerun from scratch as
the distinct `official2` attempt; no `official1` evidence was reused.

An independent read-only `official2` post-evidence audit recomputed 45 grouped
checks with zero failures; a second independent audit was also clean. It
validated the evidence/profile schemas, 25 telemetry records, 32 resource
records, all 22 current pre/post bindings, exact fresh JUnit, every raw gate,
and the G-14 monitor lifecycle. No High, Medium, or Low finding remained.

Therefore:

```text
NMCM_V2_FOUNDATION_0 = CLOSED
FOUNDATION_CPU = PASS
FOUNDATION_GPU = PASS
```

Work stops here. Canonical Alice does not exist, DAY0 has not started, and any
next work package requires separate user authorization.
