# FOUNDATION_0 CPU baseline and qualification plan

Status: **FROZEN preregistration fulfilled by the CPU package; STOP BEFORE GPU**

> Historical plan note: this document preserves the CPU preregistration and
> its mandatory stop boundary. The later separately authorized GPU package has
> completed; see
> [FOUNDATION_0_GPU_ACCEPTANCE.md](FOUNDATION_0_GPU_ACCEPTANCE.md).

Observed evidence is reported separately in
[FOUNDATION_0_CPU_ACCEPTANCE.md](FOUNDATION_0_CPU_ACCEPTANCE.md). Baseline `v2`
supersedes the invalidated pre-final `v1` probe without changing acceptance
thresholds: B `1.1.0` removes the decorative direct task path and makes its
recurrent state causally necessary.

Plan version: `1.0`

Authority: the current `temp.md`, SHA-256
`35bfe4fb3c83ac23be5a2a46f1d8a0e73af328b41f1cdfbfda5fd76b8247c1a6`

Machine-readable baseline: [configs/foundation0_cpu.json](../configs/foundation0_cpu.json)

## 1. Purpose and boundary

FOUNDATION_0 qualifies a small modular neural research platform on CPU. It is
not an Alice prototype and does not claim cognition. The work package ends
after deterministic CPU evidence for interfaces, local learning, functional
replacement, causal ablation, state ownership, graph extension, checkpoints,
telemetry, determinism, and candidate isolation.

The controlling invariant is:

> `STRUCTURAL MODULARITY != FUNCTIONAL SPECIALIZATION`

Separate files, manifests, states, and checkpoints prove structural
modularity. Functional specialization additionally requires local behavioral
evidence, a controlled causal intervention, a replacement/isolation test, and
bounded integration. A module is a neural population, not a microservice. The
runner can interpret schemas and graph edges but not latent meaning.

This stage includes no service, network, database, GUI, production cognitive
module, distributed framework, language model, Transformers import, or model
download. `LANGUAGE_DONOR` remains `OPEN`; the exact future candidate
`Qwen/Qwen3.5-0.8B-Base` is documentation only.

## 2. Frozen synthetic task

The topology is:

```text
x_t in R16 -> A -> dense distributed z_A in R32 -> recurrent B -> task logits
                  |
                  +-> recurrent C (graph fan-out added later)
```

The canonical target is:

```text
z*_t = tanh(P x_t + Q z*_(t-1))
```

`P` and `Q` are generated once from seed `4101`; `Q` is scaled to spectral
radius `0.45`. All tensors are contiguous `float32`. Sequence length is `8`.
Train/dev/held-out splits use seeds `4201/4202/4203`, sizes `4096/512/1024`,
and antipodal `(x,-x)` pairs. A separate frozen recurrent task oracle consumes
`z*`; labels are the sign of its final score after deterministic filtering to
absolute score margin `0.15`. This preserves exact class balance and makes B's
recurrent state causally necessary. Labels are an evaluator concern and never
cross a neural graph edge.

The trainers receive the train split only. The held-out split is used only by
the qualifier. Optimizers, step counts, learning rates, dimensions, seeds, and
thresholds are frozen in the machine-readable baseline before formal
qualification.

## 3. Neural modules

### A1

A1 is a simple Elman-style recurrent population:

```text
h_t = tanh(W_x x_t + b_x + W_h h_(t-1))
z_t = h_t
```

It has 1,568 trainable parameters and an owned `hidden[batch,32]` private
state. It trains locally against `z*`; it receives no B loss.

### A2

A2 is a distinct gated recurrent population. It separately computes a
candidate and learned update gate, combines candidate and prior private state,
and applies an output projection. It has 4,192 parameters. It trains locally
against the same `z*` and exposes exactly the same stimulus and latent
interfaces as A1. Replacement occurs only at the declared zero-state boundary.

### A3 negative control

A3 owns an independent copy of the trained A1 core and applies the fixed
orthogonal transform `R=-I_32` to its output. Therefore `z_A3=-z_A1` exactly:
shape, dtype, density, covariance spectrum, norms, and coordinate variance
shares are preserved, while the canonical behavior is incompatible. A3
deliberately repeats the same structural/schema claim inside a marked negative
harness; behavioral qualification must reject it.

### B

B `1.1.0` is a bias-free odd recurrent downstream population with 2,080
parameters and owned `hidden[batch,32]` state. Inputs reach a recurrent core;
task logits are read only from that owned state, with no direct latent-to-logit
bypass. It trains only on frozen A1 latents against the recurrent task oracle,
then all B parameters are frozen and the optimizer is discarded. Replacement,
A3, and ablation runs perform exactly zero B optimizer steps. A core lesion and
a carried-state versus reset-state comparison are mandatory local causal gates.

The antipodal fixture and odd B construction make the negative controls
auditable: `B(-z)=-B(z)`, and an all-zero A→B rollout from zero B state yields
zero logits and exactly balanced `0.5` accuracy.

### C

C is a 784-parameter recurrent side population consuming the declared A
latent. It owns `hidden[batch,16]` and emits a side latent. Adding C changes
only graph configuration. There is no C→A/B feedback and no change to A/B
implementation or state schema.

## 4. Frozen numerical gates

Metric definitions:

```text
mean cosine = mean(cos(z_candidate[b,t], z_reference[b,t]))
NRMSE = sqrt(mean((z_candidate-z_reference)^2)) / sqrt(mean(z_reference^2))
```

Required gates, unchanged from the approved task:

| Gate | PASS rule |
|---|---:|
| A1/A2 held-out latent | cosine `>=0.98`; NRMSE `<=0.10` |
| B after A1 | accuracy `>=0.95` |
| A1→A2 with frozen B | A2 accuracy `>=0.94`; absolute A1/A2 difference `<=0.02` |
| A3 | behavioral gate rejects; B accuracy loss versus A1 `>=0.20` |
| complete A→B lesion | task accuracy loss `>=0.25` |
| add/remove C | A/B parameters, outputs, and private state exactly unchanged |
| deterministic CPU continuation | `torch.equal` for every promised tensor |
| telemetry OFF/ON | exact output/state/parameter/RNG equality |

No threshold can be relaxed after result inspection. A mathematical
contradiction discovered before a run would require a new versioned baseline,
not an in-place edit.

## 5. Platform boundary

The bounded package implements only:

- Draft 2020-12 `ModuleManifest` schema plus cross-field semantic validation;
- exact-version registry and fixed acyclic graph definitions;
- a neural interface with explicit external private-state bundles;
- a topological runner whose route is invariant to tensor values;
- independent safetensors module shards;
- an atomic whole-checkpoint JSON manifest over shard hashes;
- per-module JSONL telemetry emitted after forward computation;
- deterministic fixture/training/evaluation helpers;
- spawned-process CPU candidate isolation.

Tensor payloads use `safetensors==0.8.0`. JSON is ordinary Draft 2020-12
metadata. A module shard includes its parameters/buffers and private state; the
whole manifest records graph version, snapshot/step, module identity/version,
compatibility version, state schema version, paths, sizes, and SHA-256 hashes.
Tensor and JSON files are written to same-directory temporary paths, flushed,
and atomically renamed. This is not a forensic artifact DAG.

## 6. Required CPU test conjunction

All 15 tests are mandatory; no required `SKIP` or `XFAIL` can close the CPU
gate.

| ID | Test |
|---|---|
| CPU-01 | manifest, schema, cross-field, and parameter-count validation |
| CPU-02 | registry, fixed graph, interface rejection, routing invariance |
| CPU-03 | private-state ownership, non-aliasing, no in-place mutation |
| CPU-04 | A1 local learning and held-out latent capability |
| CPU-05 | A2 local learning, distinct implementation, same interface |
| CPU-06 | A1→B held-out task and B freeze |
| CPU-07 | frozen-B A1→A2 replacement with zero B updates |
| CPU-08 | A3 structural/statistical positive and behavioral negative |
| CPU-09 | complete targeted A→B ablation with sham/non-target guards |
| CPU-10 | C add/remove and A/B non-mutation |
| CPU-11 | five independent module shards and standalone continuation |
| CPU-12 | atomic whole checkpoint, exact continuation, corrupt-digest rejection |
| CPU-13 | telemetry schema and exact OFF/ON non-interference |
| CPU-14 | complete serial training/data determinism replay |
| CPU-15 | two overlapping spawned candidate bundles with serial parity and no shared mutable payload/artifact root |

Concurrency is a correctness test, not a speedup claim. Each worker reconstructs
A/B and state from immutable bytes, owns a distinct process/RNG/artifact root,
and synchronizes at start and midpoint so overlap is observed.

## 7. Evidence and stop condition

The CPU bundle records:

- exact environment/dependency versions and lock;
- all training step/loss/time records;
- latent, replacement, A3, and ablation metrics;
- module and checkpoint sizes/digests;
- telemetry events with wall/CPU time and peak process memory;
- JUnit results for exactly 15 tests;
- concurrency PIDs, overlap, immutable-payload status, and serial parity.

The work package stops after this evidence and documentation are written.
Failures stay explicit; no result is converted to PASS by omission.

## 8. GPU stage as prepared at the CPU stop boundary

At this historical CPU boundary, real GPU qualification was explicitly
`NOT_RUN`. Read-only diagnosis showed the
host RTX 5090 and driver are healthy but the default Codex sandbox hides device
nodes. No driver or toolkit change is needed or permitted here.

A later, separately approved GPU window (proposed maximum 15 minutes) may use a
separate PyTorch `2.13.0+cu130` environment and must verify:

- CPU→GPU→CPU initial placement/return parity;
- HOT GPU→RAM→GPU offload/restore parity;
- per-module peak VRAM and GPU time;
- controller decisions under actual load;
- concurrent candidate isolation;
- aggregate utilization `<=80%` and project active VRAM `<=28 GiB`.

Missing device access is a blocked GPU gate, never a required-test skip. MPS is
optional research only and not a project dependency.
