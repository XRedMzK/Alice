# FOUNDATION_0 CPU acceptance report

Status: **CPU PASS — GPU NOT_RUN / STOP BEFORE GPU**

> Historical boundary note: `GPU NOT_RUN` records the required stop point of
> this immutable CPU acceptance report. The later separately authorized GPU
> package is now 15/15 PASS; current closure is documented in
> [FOUNDATION_0_GPU_ACCEPTANCE.md](FOUNDATION_0_GPU_ACCEPTANCE.md).

- Date: **2026-08-12**
- Baseline: `nmcm.foundation0.cpu/v2`
- Module revision: B `1.1.0`
- Graph/shard/whole-checkpoint/telemetry formats: `2.0.0` / `v2`

Compact result:
[configs/foundation0_cpu_acceptance.json](../configs/foundation0_cpu_acceptance.json)

Full result:
`runs/foundation0_cpu_v2_20260812_evidence3/results.json`
(`sha256:ac64db7e52771de032f52850f5ad0195bbb53093130686bb4d235177b91c220b`)

The earlier v1/B-1.0 evidence and the first pre-publication v2 probe are
invalidated. They exposed a decorative B recurrent core and missing evidence/
restore bindings during review; none is an accepted result. Acceptance
thresholds were not weakened. This report names only the final `final3`/
`evidence3` artifacts.

## 1. Documentation changes

The architecture now explicitly freezes
`STRUCTURAL MODULARITY != FUNCTIONAL SPECIALIZATION`, the semantics-blind
control boundary, dense neural interfaces, owned state, local/causal/
replacement/integration evidence, and separate CPU versus future GPU gates.
The Qwen donor remains OPEN; `Qwen/Qwen3.5-0.8B-Base` is only the exact future
candidate. PARTNER EVALUATION SIGNAL and ACTIVE EPISTEMIC LEARNING appear only
as future Alice V1 roadmap mechanisms.

The v2 bounded revision removes B's direct latent-to-logit shortcut. B logits
now come only from its recurrent state, and core/state lesions are mandatory.
Behavioral admission is enforced at graph execution against the real provider,
the frozen B digest, the exact held-out fixture digest, and the frozen threshold
digest. Registry resolution remains structural and exact-versioned.

No Alice, language module, Transformers dependency, Qwen weights, DAY0, or GPU
qualification was added.

## 2. Exact environment and dependencies

Python is `3.12.3`; the project-local `.venv` contains:

| Package | Version |
|---|---:|
| PyTorch CPU | `2.13.0+cpu` |
| NumPy | `2.5.2` |
| safetensors | `0.8.0` |
| jsonschema | `4.26.0` |
| pytest | `9.1.1` |

`requirements-cpu.in` fixes top-level requirements and
`requirements-cpu.lock` fixes the resolved environment. `pip check` passed.
The accepted environment is CPU-only. PyTorch's official
[2.13 release](https://pytorch.org/blog/pytorch-2-13-release-blog/) and
[wheel indexes](https://download.pytorch.org/whl/) support a later, separately
approved profile; they do not authorize installing or running it now.

## 3. NVIDIA visibility diagnosis

The host NVIDIA stack is healthy. The reported failure came from Codex sandbox
device filtering, not a broken driver:

- in the default sandbox, `/dev/nvidia*` and `/dev/dri` are absent and
  `/usr/bin/nvidia-smi` exits `9`;
- the identical read-only command outside that sandbox sees a GeForce RTX 5090,
  driver `595.84`, CUDA driver capability `13.2`, and `32607 MiB` VRAM;
- PCI device `10de:2b85` is bound to `nvidia`; `nvidia`, `nvidia_modeset`,
  `nvidia_drm`, and `nvidia_uvm` are loaded consistently at `595.84`;
- `libcuda.so.1` and `libnvidia-ml.so.1` resolve to matching `595.84` libraries;
- no Xid/NVRM error was observed; no system CUDA toolkit is installed.

No driver, OS package, or GPU runtime was changed. `nvidia-smi`'s CUDA `13.2`
is driver API capability, not proof of a local toolkit or a GPU PASS.

## 4. Synthetic architecture actually implemented

- A1: Elman recurrent `R16→R32`, 1,568 parameters, owned R32 state.
- A2: distinct gated recurrent implementation, 4,192 parameters, the exact
  same external `dense_latent/v1` contract, owned R32 state.
- A3: independent A1-core copy plus fixed orthogonal `R=-I32`, 1,568
  parameters; structurally/statistically plausible and intentionally rejected.
- B `1.1.0`: bias-free recurrent R32 consumer, 2,080 parameters, owned R32
  state; trained only on frozen A1 latents, then frozen. Its two logits are read
  only from recurrent state.
- C: independent recurrent A-branch consumer, 784 parameters, owned R16 state;
  graph fan-out adds `A→C` without editing A or B.

The deterministic teacher is
`z*_t=tanh(Px_t+Qz*_(t-1))`, with `x∈R16`, `z*∈R32`, frozen seeds and stable Q.
The task labels come from a separate recurrent oracle over `z*`; antipodal
pairs and a frozen absolute score margin `0.15` make the balanced lesion oracle
well-defined. The runner validates input/output/state shape, dtype, contiguity,
finiteness, interface/envelope identity, graph admission evidence, and fixed
topology without inspecting latent values for routing.

## 5. CPU acceptance table

Formal command:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_foundation0_cpu.py \
  --junitxml=runs/foundation0_cpu_v2_20260812_final3/junit.xml
```

Result: **15 passed in 24.928 s; zero failures/errors/skips/xfails**.
The JUnit contains exactly the required names, is newer than every evaluator-
bound source/config/test file, and its SHA-256 is
`f1a43ee72406e2a2e7506c284e89ad1c9e6363e01f03e462b2903f86c8fc025b`.
Qualification recomputed every material condition and verified identical
pre/post evaluator bindings.

| Gate | Evidence | Result |
|---|---|---:|
| CPU-01 | manifest and Draft 2020-12 schemas | PASS |
| CPU-02 | exact registry, graph, envelopes, routing invariance | PASS |
| CPU-03 | private-state ownership/non-aliasing | PASS |
| CPU-04 | isolated A1 local learning | PASS |
| CPU-05 | isolated A2 local learning/interface | PASS |
| CPU-06 | A1→B task, B core/state causality, freeze | PASS |
| CPU-07 | artifact-bound frozen-B A1→A2 replacement | PASS |
| CPU-08 | A3 structural positive/behavioral negative | PASS |
| CPU-09 | A1/A2 targeted A→B edge ablation and guards | PASS |
| CPU-10 | add/remove C and C input/state causality | PASS |
| CPU-11 | five independent shards/standalone continuation | PASS |
| CPU-12 | atomic whole continuation and fail-closed restore | PASS |
| CPU-13 | v2 telemetry schema/non-interference | PASS |
| CPU-14 | full serial data/training determinism | PASS |
| CPU-15 | two isolated concurrent candidate bundles | PASS |

## 6. Replacement A1→A2 result

Against canonical held-out `z*`, A1 has cosine `1.0`, NRMSE
`0.0000677914`; A2 has cosine `0.9999360442`, NRMSE `0.0120792221`.
Both clear the frozen `>=0.98` / `<=0.10` gates.

Frozen-B accuracy is `0.998046875` for both A1 and A2: absolute difference
`0.0`, versus requirements A2 `>=0.94` and difference `<=0.02`. Replacement
performs exactly zero B optimizer steps and preserves B digest
`1c135fcc4f9cd270e643f7957276ff46e1c717d81bc3f7bde3a2089d124432ee`.
The decision is bound to the real A2 artifact, that B artifact, held-out fixture
digest, and threshold digest; a changed B or spoofed fixture is rejected before
execution.

## 7. A3 negative-control result

`z_A3=-z_A1` holds exactly. A1/A3 both have density `1.0`, effective rank
`16.17723534`, and maximum coordinate variance share `0.04217719523`, with the
same shape, dtype and structural representation claim.

Behavior rejects A3: cosine to canonical `-1.0`, NRMSE `1.999999285`, frozen-B
accuracy `0.001953125` versus A1 `0.998046875`; absolute drop `0.99609375`
(required `>=0.20`). The negative gate passes, while admission is false. Merely
changing the A3 graph mode or supplying another fixture cannot bypass the
authoritative fixture/consumer/threshold digests.

## 8. Causal ablation result

The complete A→B edge lesion produces balanced accuracy `0.5`. Capability loss
is `0.498046875` for A1 and A2 (required `>=0.25`). Lesioning B's input and
recurrent transforms also loses `0.498046875`, and carried-state versus reset-
state logits differ, proving B's private recurrence is functional rather than
decorative.

Clone/sham equals intact exactly, B's parameter digest is unchanged, and the C
branch is an exact non-target guard. C separately changes under input lesion
and under carried-versus-reset state. Focused B/A3/C fixtures never load or
train unrelated A2.

## 9. Checkpoint and restore result

Each module owns an immutable `nmcm.shard/v2` safetensors payload plus a Draft
2020-12 JSON sidecar. All five standalone continuations are exact. The
`nmcm.whole_checkpoint/v2` manifest is published last and records graph version,
snapshot/step, module/version, compatibility/state schema, file size and hashes.

Whole ABC continuation is exact for every output and private-state tensor.
Restore prepares and validates all shards before committing any module, with
rollback defense. Tests reject interrupted publication, corrupt hashes,
self-consistently rehashed mixed snapshots, invalid state shape/dtype/finiteness,
extra JSON fields, cross-module batch mismatch, and partial active mutation.

Tensor payloads total `39,264` bytes; all whole-checkpoint files total `43,120`
bytes. Five independent shard pairs total `83,732` bytes. Manifest SHA-256:
`b5812f30d979236cf0f99737bdea90cdc3a89b17b5a64076b540a63e5294c0b2`.

## 10. Telemetry non-interference result

Telemetry OFF versus ON preserves all outputs, states, parameter digests and
Torch RNG bytes exactly. The two v2 events identify module instance, candidate,
experiment, timestamp/logical step, local PASS, CPU wall/CPU time, collection
overhead, placement, exact byte counts, and strict nested tensor statistics.
GPU time/VRAM/utilization are explicit `null` with `gpu_status=NOT_RUN`.

Telemetry SHA-256:
`4169fd78ec5735555433172ca49f25492804b99028ba2fb9251ee6336188c4bc`.
Evidence-process peak memory was `408,797,184` bytes. Owned parameters total
`40,768` bytes; batch-1 private states total `576` bytes.

## 11. Concurrency correctness result

Two `spawn` processes reconstructed A1+B and A2+B from immutable payloads,
with separate module/state/RNG objects and artifact roots. Both overlapped for
`0.687944911 s`, `max_live=2`, had distinct PIDs and matched their serial
baselines bitwise. No CPU speedup claim is made or required.

## 12. Wall-clock for meaningful local experiments

| Experiment/evidence step | Wall s | CPU s when recorded |
|---|---:|---:|
| A1 local optimizer, 180 steps | 1.7030 | 1.7029 |
| A2 local optimizer, 180 steps | 3.5920 | 3.5915 |
| B local optimizer, 240 steps | 1.9272 | 1.9270 |
| Complete deterministic training bundle | 7.3090 | 7.3082 |
| CPU-04 isolated A1 pytest case | 2.440 | — |
| CPU-05 isolated A2 pytest case | 3.577 | — |
| CPU-06 local A1→B/core-state case | 2.007 | — |
| CPU-08 A3 negative case | 0.022 | — |
| CPU-09 edge ablation case | 0.012 | — |
| CPU-10 C extension/lesion case | 0.010 | — |
| CPU-11 independent shards | 0.047 | — |
| CPU-12 whole checkpoint/fail-closed | 0.045 | — |
| CPU-13 telemetry | 0.014 | — |
| CPU-15 concurrency | 1.988 | — |
| Formal 15-test suite | 24.928 (`26.48` external) | `28.69` user + `2.24` system |
| Evidence materialization | 10.059 (`11.29` external) | `14.11` user + `1.18` system |

Formal suite plus evidence materialization consumed `37.77 s` external wall
time. Formal-suite peak RSS was `435,688 KiB`. Editing, review, dependency
download, and invalidated pre-final probes are deliberately not misreported as
scientific experiment time.

## 13. Remaining blockers before bounded GPU stage

CPU acceptance is complete; GPU/resource gates remain `NOT_RUN`:

1. explicit user authorization for the proposed `<=15` minute, no-training
   window is absent;
2. the future run needs unsandboxed NVIDIA device access;
3. a separate GPU environment is not yet approved or created; the recommended
   current proposal is PyTorch `2.13.0+cu130`, not a selected runtime;
4. real CPU→GPU→CPU placement parity and HOT GPU→RAM→GPU offload parity,
   per-module VRAM, resource controller, concurrent candidates, aggregate
   utilization `<=80%`, and active VRAM `<=28 GiB` are unmeasured.

The driver should not be changed. NVIDIA MPS is optional. No Alice V1, Qwen
download, language module, DAY0, or CUDA work follows automatically from this
CPU PASS.
