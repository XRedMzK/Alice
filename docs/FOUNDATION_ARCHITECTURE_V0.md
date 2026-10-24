# NMCM v2 foundation architecture

Status: **APPROVED AND FROZEN for the FOUNDATION_0 CPU work package**

Qualification: **CPU 15/15 PASS; GPU 15/15 PASS; FOUNDATION_0 CLOSED**. The
architecture baseline itself remains unchanged.

Baseline: `NMCM_V2_FOUNDATION_0 / architecture v0.1`

Compatibility namespace: `nmcm.module/v1alpha1`

This document is normative for the foundation. “MUST”, “MUST NOT”, “SHOULD”,
and “MAY” describe conformance requirements.

## 0. Methodological applicability

The frozen `v0.1` FOUNDATION_0 architecture and all accepted numerical gates
remain unchanged. Project-wide interpretation is governed by
[ALICE PROJECT METHODOLOGICAL SCOPE](PROJECT.md#alice-project-methodological-scope):
Alice is an engineering implementation and technical-validation project, and
philosophical proof of consciousness, subjectivity, personhood, or biological
equivalence is not an architecture PASS criterion.

`Brain-inspired` describes a source of useful organization, mechanisms,
learning principles, memory structures, or constraints; it does not mean a
whole-brain simulation or mandatory biological fidelity. Architecture sources
must distinguish:

- **ARCHITECTURAL MOTIVATION**, which suggests a design direction;
- **IMPLEMENTATION CONSTRAINT**, which constrains a selected realization for a
  stated engineering reason; and
- **HARD FUNCTIONAL REQUIREMENT**, which defines observable behavior required
  by an applicable contract.

A neuroscience or cognitive-science observation does not become a hard
requirement without explicit promotion in a versioned contract. Biological
fidelity is mandatory only when that contract identifies the particular
biological property as necessary. Validation remains proportional to the claim:
implementation correctness, uncertainty-resolving engineering experiments, and
architecture research do not automatically require the same procedure.

## 1. Architectural thesis

NMCM v2 is a graph of specialized, interacting neural modules/populations.
Cognitive content is produced by trainable neural mechanisms, their owned
state, and learned connections expressed as dense/distributed
representations. The architecture is not identified with any single language
model.

The platform surrounding the graph is deliberately non-cognitive. It may
schedule declared dependencies, allocate devices and memory, transfer state,
save checkpoints, run evaluators, and collect telemetry. It MUST NOT infer or
assign semantic meaning, determine intent/ACT/TARGET, select a cognitive
mechanism using symbolic semantic rules, or implement cognition through
Python `if/else` logic.

## 2. Frozen foundation invariants

These invariants form the approved CPU design baseline. Changing one requires
an explicit architecture decision and bounded impact qualification. This
freeze authorizes the bounded CPU implementation and its synthetic training;
it does not authorize CUDA qualification, Alice V1, or model downloads.

### I-01 — Neural modularity

Every cognitive mechanism is represented, where scientifically feasible, by
a neural module with its own trainable parameters, owned private/recurrent
state, optional persistent state, learned tensor interfaces, resource cost,
local tests, causal ablation, independent checkpoint shard, and version.
Exceptions require an architecture decision explaining why the mechanism is
not cognitive or why a neural realization is currently infeasible.

### I-02 — Distributed interfaces

Inter-module cognitive content crosses versioned dense/distributed latent
interfaces. Structural metadata such as tensor shape, dtype, device,
compatibility version, mask, sequence position, and provenance MAY be handled
by the control plane. Semantic labels such as intent, ACT, TARGET, emotion, or
chosen mechanism MUST NOT be the control-plane routing protocol.

### I-03 — Replaceability behind compatibility

A compatible implementation `A2` can replace `A` without retraining `B` when
the frozen input/output interface, state assumptions, numerical tolerances,
and compatibility version are satisfied. Compatibility is demonstrated by
consumer contract tests and bounded integration regression; it is never
inferred from matching tensor shape alone.

### I-04 — State ownership

Each mutable neural state has exactly one owning module instance. Private
state cannot be mutated by another module or concurrent candidate. Persistent
state is schema-versioned and checkpointed with its owner. Shared information
is communicated through declared interfaces, immutable snapshots, or an
explicitly modeled neural shared-state module.

### I-05 — Explicit, diagnosable coupling

Recurrent and bidirectional learned connections are allowed, but every edge,
update cadence, causal dependency, and state boundary is declared. Hidden
global state and undeclared mutation are forbidden.

### I-06 — Local scientific closure

A local module hypothesis has a local experiment capable of returning an
unambiguous PASS or FAIL without default full-system qualification. Upstream
representations and unaffected modules may be frozen or replayed. Full-system
runs are reserved for integration milestones, interface changes, blocker
qualification, or plausible system-wide effects.

### I-07 — Causal evidence

A claim that a cognitive mechanism is causally responsible for a capability
requires a controlled causal intervention or ablation against matched
state/input conditions, an expected effect, explicit non-target guard metrics,
and a recorded result. Correlation or aggregate end-to-end improvement alone is
insufficient for that attribution. A deterministic implementation-correctness
claim or non-cognitive plumbing change does not acquire an ablation requirement
unless it also makes a causal capability claim. The frozen FOUNDATION_0
A1/A2/A3/B/C ablations remain mandatory and unchanged.

### I-08 — Parallelism by construction

Independent experiment bundles share no mutable neural state and are
schedulable concurrently from the first implementation. Same-live-state
comparisons are packaged as causal bundles with controlled snapshots.
Scheduler decisions use declared dependencies and numeric resources, never
semantic interpretation of latent content.

### I-09 — Bounded active resources

On the reference RTX 5090, aggregate target GPU utilization is at most 80%
and the hard active VRAM peak is 28 GiB (28672 MiB), including concurrent
jobs and runtime overhead. Total future model size is not constrained to
VRAM: HOT/WARM/COLD placement is permitted under the transfer and anti-thrash
rules in [RESOURCE_POLICY.md](RESOURCE_POLICY.md).

### I-10 — Independent artifacts and restore

Every module has an independently addressable checkpoint/state shard.
Tensor payloads use `safetensors`; metadata and whole-system manifests use JSON
validated against Draft 2020-12 schemas. Whole-system checkpoints are
manifests over versioned shards plus graph and snapshot metadata, not an opaque
monolith. Restore verifies compatibility and integrity before exposing a
consistent graph state.

### I-11 — Control/evaluation separation

Scientific contracts and evaluator logic are independent of artifact paths,
devices, process IDs, and launch details. The runtime resource controller is
independent of cognition. Changing an execution binding cannot silently
change the scientific question or PASS thresholds.

### I-12 — Non-interfering observability

Telemetry exposes module lifecycle, version, experiment, latency, GPU time,
VRAM, state and activation statistics, checkpoint version, and local status.
Instrumentation MUST NOT feed decisions or values back into cognition unless
it is later introduced as an explicit neural interface and separately tested.

### I-13 — Extension without rewrites

A new compatible module `C` is added through registry/graph composition and
new edges. Existing implementations `A` and `B` do not require source edits.
If an interface must change, that is a versioned migration, not an implicit
rewrite.

### I-14 — Language base is replaceable

Decision 0005 records user-selected `Qwen/Qwen3.5-2B-Base` as the frozen active
text-only donor. It remains replaceable behind a canonical NMCM language
latent; no downstream M1 interface exposes Qwen hidden size, layer count, or
model class. The old 2B-versus-4B comparison is cancelled pre-inference and
supports no quality conclusion. The retained 4B snapshot is outside M1.

A future donor may provide pretrained language capability and initial
pretrained world knowledge. It is not the NMCM v2 foundation, Alice identity,
autobiographical memory, temporal context, SELF/PARTNER, epistemic system,
goals, planning, value system, or runtime-learning mechanism. Reusing a ready
component is allowed only when it does not replace the cognitive mechanism
under research.

### I-15 — Synthetic evidence is not DAY0

FOUNDATION_0 tests platform properties only. Passing them cannot instantiate
canonical Alice, satisfy an Alice capability milestone, or start DAY0.

### I-16 — No legacy inheritance by default

NMCM v1 is read-only research reference. Code, checkpoints, architecture,
APIs, and unfinished run state are not migrated automatically. Any later reuse
is an individually justified decision with provenance and local evidence.

### I-17 — Structural modularity is not functional specialization

`STRUCTURAL MODULARITY != FUNCTIONAL SPECIALIZATION`. A separate class,
module, or process is not a standalone functional neural mechanism merely
because it is structurally isolated. Functional specialization requires all
of the following evidence:

- a local behavioral test;
- a causal ablation;
- a replacement/isolation test;
- integration behavior.

Modules are specialized neural populations, not software microservices. The
orchestration around them remains semantics-blind.

This invariant applies when a component is claimed or promoted as a
functionally specialized cognitive mechanism. It does not reclassify
control-plane plumbing, artifact handling, or deterministic helper code as a
cognitive module merely to impose module-level experimental ceremony.

## 3. Logical planes

```text
                experiment definition (hypothesis, metrics, thresholds)
                                   |
                                   v
   +------------------------ non-cognitive control plane -------------------+
   | graph lifecycle | scheduler | resource admission | checkpoint | monitor|
   +-----------+---------------------------------------------------+--------+
               | structural invocation / opaque tensor handles     |
               v                                                   v
   +-------------------------- cognition plane -----------------------------+
   | [neural module A] <== dense latent edges ==> [neural module B]         |
   |       ^                                      ^                         |
   |       +======== optional recurrent/bidirectional learned edges ========+|
   |                         [future module C]                               |
   +------------------------------------------------------------------------+
               |
               v
   immutable artifacts: module shards, graph manifest, metrics, telemetry
```

The scheduler receives closed structural descriptors and opaque tensor handles,
not tensor values. A narrow executor/transport passes payloads directly between
declared module endpoints and may inspect only structural metadata needed for
validation/device movement. A runtime may answer “dependency X is ready” or
“budget Y is available”; it may not answer “the user intends Z, therefore
invoke module Q.” Conditional cognitive routing, if needed later, must itself
be learned and represented inside the cognition plane. The scheduler may then
execute the resulting declared graph work without interpreting its semantics.

### 3.1 Frozen synthetic CPU topology

The minimum FOUNDATION_0 graph is:

```text
x_t -> A -> distributed latent z_A -> B -> synthetic task output
       |
       +-> C  (added later through graph composition)
```

All neural interfaces are dense/distributed; no semantic symbolic routing is
permitted. The canonical synthetic process is

```text
x_t in R^16
z*_t in R^32
z*_t = tanh(P x_t + Q z*_(t-1))
```

`P` and `Q` are fixed from a frozen RNG seed, and `Q` is stabilized so the
trajectory remains bounded. Data is generated from frozen seeds with explicit
train/dev/held-out splits and remains small enough for fast CPU iteration.

The frozen module roles are:

- `A1`: a simple recurrent neural module;
- `A2`: a different gated recurrent implementation with the exact same
  external neural interface contract;
- `A3`: an intentional same-shape, same-dtype behavioral negative control,
  using a frozen orthogonal transform or permutation of the canonical latent;
- `B`: a downstream recurrent neural module with private state, trained with
  `A1` and then frozen for replacement tests;
- `C`: an additional compatible neural module attached later without changing
  the implementation or state schema of `A` or `B`.

### 3.2 Pre-registered CPU acceptance boundary

Before any result is observed, the following thresholds are frozen:

- `A1` and `A2` held-out latent quality: mean cosine `>= 0.98` and normalized
  RMSE `<= 0.10`;
- `B(A1)` held-out accuracy: `>= 0.95`;
- replacement: `B(A2) >= 0.94` and
  `abs(B(A2) - B(A1)) <= 0.02`;
- negative compatibility: the behavioral gate rejects `A3`, and `B(A3)` drops
  by `>= 0.20` absolute relative to `B(A1)`;
- targeted causal ablation: expected capability drop `>= 0.25` absolute;
- adding or removing `C` does not mutate `A`/`B` parameters or private state
  outside declared graph communication;
- deterministic CPU checkpoint continuation has exact tensor equality;
- telemetry OFF versus ON has exact model outputs and state.

Thresholds MUST NOT be weakened after results are seen. A mathematical
contradiction discovered before results may justify only a bounded,
documented, pre-experiment revision. The CPU parallelism gate proves
concurrent state isolation and correctness, not wall-clock speedup.

## 4. Core abstractions

### 4.1 Module instance

A module instance is the tuple of module implementation/version, parameters,
owned state, declared interfaces, compatibility version, resource policy, and
experiment identity. The normative fields and lifecycle are defined in
[MODULE_CONTRACT.md](MODULE_CONTRACT.md).

Synthetic test doubles used by FOUNDATION_0 are real small neural modules for
platform purposes: they contain trainable parameters and recurrent/private
state, but their task is deliberately non-cognitive and bounded.

### 4.2 Latent edge

A latent edge binds a producer output interface to a compatible consumer input
interface. Its envelope contains only:

- one or more dense tensors;
- structural masks/positions when required;
- schema and compatibility versions;
- producer instance/version and step provenance;
- immutable snapshot/causal-bundle identity where applicable.

Graph, envelope, and experiment-bundle schemas are closed
(`additionalProperties: false`) during FOUNDATION_0. They contain no predicate,
condition, semantic label, value-dependent edge mask, or output-rewrite field.
Only module code receives latent values; scheduler/resource APIs receive opaque
handles and byte/shape/dtype metadata.

The foundation does not standardize human-readable semantic dimensions.
Adapters that transform one latent schema to another are themselves versioned
neural modules when the transformation carries learned semantic content.

### 4.3 Module graph

The graph declares module instances, typed edges, recurrence boundaries,
update frequencies, causal dependencies, and placement eligibility. Graph
configuration is data, so adding `C` does not edit `A` or `B`. Cycles execute
through explicit steps/state transitions; accidental call-stack recurrence is
forbidden.

### 4.4 Registry and compatibility

The registry resolves `(module_id, module_version)` to an implementation and
manifest. Resolution is exact and structural; it does not independently claim
behavioral admission. At the execution boundary, the graph runner accepts a
substitute only when its interface matches and an evaluator decision is bound
to the actual provider artifact, frozen consumer, authoritative fixture, and
threshold set. Neither registry nor runner chooses from user intent or latent
content.

### 4.5 State and checkpoint coordinator

The coordinator asks each module to produce or restore its own `safetensors`
shard. A Draft 2020-12-validated whole-system JSON manifest records graph
version, module IDs and versions, shard hashes, state schema versions, and
interface compatibility versions. At a graph-step barrier, the coordinator
captures a consistent module/state snapshot, stages and verifies the shards, and
publishes the whole manifest last with an atomic rename. Restore exposes the
graph only after every declared shard, hash, schema version, compatibility
version, and snapshot identifier passes. A partial save or restore is invalid.
This is normal bounded atomic-write engineering; FOUNDATION_0 MUST NOT grow a
recursive forensic checkpoint DAG. Migration is per shard and explicit, with
no migration solely for v1 compatibility.

### 4.6 Scheduler and resource controller

The scheduler admits work from declared resource budgets, dependencies,
state-sharing constraints, and placement. It distinguishes:

- **causal bundles**, whose paired arms consume the same immutable captured
  state and therefore need coordinated execution; and
- **independent bundles**, which own or clone all mutable state and may run
  concurrently.

Bundle identity, state isolation, leases, worker intervals, and cancellation-
safe release are foundation runner primitives, not a serial runner retrofit.
Scheduler traces contain structural node/edge invocations, state-write
destinations, numeric resource decisions, and timestamps, but no latent values.

It optimizes research wall-clock under the resource policy, not nominal GPU
saturation. It may queue, pause at safe boundaries, or offload modules for
resource reasons. It must not inspect latent values to make cognitive choices.

### 4.7 Monitor

Telemetry is emitted per module instance and experiment in a machine-readable
stream. A CLI/TUI summary is a projection of that stream, not a separate source
of truth. Collection overhead and dropped samples are themselves measured.

### 4.8 Experiment runner and evaluator

An experiment definition names a hypothesis, frozen dependencies,
intervention, inputs, metrics, thresholds, and required evidence. A separate
execution binding names paths, device, seed, process/concurrency settings, and
artifact destinations. Evaluators consume declared outputs/artifacts and do
not discover meaning from directory names.

## 5. Failure containment

- A module failure invalidates its local run and any dependent graph step, not
  unrelated independent bundles.
- Resource-limit violation fails the offending execution and records peak
  evidence; it cannot be reclassified as a scientific FAIL.
- Interface/schema mismatch fails before neural execution.
- Non-finite state/output fails the module locally and prevents checkpoint
  publication.
- A missing shard, digest mismatch, or inconsistent snapshot ID fails restore
  closed.
- Telemetry loss is reported explicitly. It cannot silently become a PASS
  where the missing metric is required by a gate.

## 6. Allowed and forbidden examples

Allowed operational branching includes dtype/shape validation, dependency
readiness, memory admission, retry of an idempotent artifact write, fixed graph
traversal, and selection of an explicitly requested module version for an
experiment.

Forbidden control logic includes parsing a prompt to choose a cognitive module,
mapping text labels to ACT/TARGET, editing module outputs according to hand
written semantic rules, value-dependent edge masking or branchless selection,
treating a transcript as autobiographical state, or using evaluator results as
a hidden cognitive feedback channel.

## 7. Alice V1 modular mechanism boundaries

M1 current-turn semantics remains the only executed Alice V1 mechanism scope.
Its custom donor-boundary hypotheses failed, Boundary V3 was cancelled before
execution, and the successor ready-encoder E1 bake-off failed mandatory sealed
contrasts. No M1 implementation is frozen. The following mechanisms remain
future requirements and were not E1 scope.

### 7.1 PARTNER EVALUATION SIGNAL

The future input vocabulary is `LIKE / DISLIKE / NONE`: the partner reports a
reaction to an Alice response. This is social evidence, not a truth label,
command, global reward, or forced belief update. A future mechanism MAY affect
salience, learning strength, the partner preference model, and the probability
of review or clarification. It MUST NOT directly rewrite factual beliefs,
goals, values, or decisions.

### 7.2 ACTIVE EPISTEMIC LEARNING

This future mechanism replaces the old `UNKNOWN -> immediate "I don't know"`
rule. It must support epistemic states `KNOWN`, `UNCERTAIN`, `HYPOTHESIS`,
`UNKNOWN`, and `UNRESOLVED_GAP`, with actions `ANSWER`,
`QUALIFIED_HYPOTHESIS`, `CLARIFY`, `SEEK`, and `ABSTAIN`. A hypothesis is not a
belief.

Future partner experience must be able to follow the path `novel concept ->
episodic acquisition -> clarification -> consolidation -> later
generalization`. Neither future mechanism is implemented in FOUNDATION_0.

## 8. Change control

The user-approved corrections froze this `v0.1` CPU design and its
pre-registered thresholds. Later changes use an ADR, identify affected
interfaces/modules/evidence, and run proportionate local or bounded
integration regression. A full-system run is required only under the triggers
in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). The original CPU work package
stopped before actual CUDA qualification as required. The separately approved
[Decision 0002](decisions/0002-foundation-gpu-qualification-policy.md) later
authorized and completed the bounded GPU qualification; it authorizes no repeat
or expanded GPU work.
