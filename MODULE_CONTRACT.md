# Neural module contract

Status: **FROZEN for FOUNDATION_0 CPU baseline**

Qualification: **accepted on CPU and target GPU; FOUNDATION_0 CLOSED**. The
contract remains frozen at the version below.

Human-readable contract version: `v1alpha1`

Machine-readable schema: [configs/module_manifest.schema.json](configs/module_manifest.schema.json)

This contract specifies the smallest independently testable and replaceable
unit in NMCM v2. It defines observable boundaries, not the internal neural
architecture of future cognitive modules.

## 0. Applicability and methodological interpretation

This `v1alpha1` contract is the frozen special-case contract accepted for the
FOUNDATION_0 cognitive neural modules. Its schema, including nonempty local,
causal-ablation, and integration test-reference fields, and all accepted
FOUNDATION_0 evidence remain unchanged.

The contract applies to components claimed as cognitive neural mechanisms, not
to control-plane plumbing, artifact utilities, or deterministic helper code.
Those components require engineering validation proportional to their actual
correctness and integration risks; they must not be relabeled as cognitive
modules merely to manufacture module-level experimental ceremony. For a future
cognitive module, causal intervention is necessary when claiming that the
module is responsible for a functional property. Direct deterministic
correctness alone does not create that attribution claim. Any future contract
that changes the `v1alpha1` manifest requirements needs its own versioned
schema and explicit change control.

Module purpose and PASS/FAIL remain functional and engineering claims. Neither
a module name nor conformance to this contract proves consciousness,
subjectivity, personhood, genuine feeling, or biological equivalence.

## 1. Conformance rule

A conforming module MUST provide:

1. a machine-readable manifest validated against the canonical schema;
2. a human-readable specification that explains the manifest, bounded
   functional purpose, and any architecture-research rationale;
3. owned trainable parameters and private/recurrent state for cognitive
   mechanisms (synthetic foundation modules MUST include both; a later
   stateless exception needs an ADR recorded in `exception_decision`);
4. versioned dense latent inputs and outputs;
5. independent save/restore and explicit compatibility metadata;
6. declared resources, telemetry, offload behavior, and execution safety;
7. local acceptance, causal ablation, and bounded integration tests.

A manifest describes a module *version*, not a mutable global singleton.
Runtime instance identity, current device, experiment, and PASS/FAIL status are
telemetry fields and are not written back into the versioned manifest.

## 2. Required manifest fields

The following fields are mandatory. The JSON schema is authoritative for
syntax; this section is authoritative for meaning.

| Field | Meaning |
|---|---|
| `module_id` | Stable, globally unique logical module name. It does not encode a path, device, run, or candidate slot. |
| `module_version` | Immutable implementation/parameter behavior version using semantic versioning. |
| `purpose` | Bounded functional/engineering claim and explicit non-goals, with architecture-research motivation where relevant. It is documentation, not runtime routing input. |
| `input_interfaces` | Named structural endpoints consumed by the module. |
| `output_interfaces` | Named structural endpoints produced by the module. |
| `input_latent_schema` | Per-input dense tensor shape/dtype/mask and compatibility rules. |
| `output_latent_schema` | Per-output dense tensor shape/dtype/mask and compatibility rules. |
| `private_state_schema` | Owned recurrent/transient state, reset and clone semantics. It MUST be enabled for foundation neural modules. |
| `persistent_state_schema` | State surviving a run boundary, or an explicit `enabled: false` declaration. |
| `trainable_parameters` | Parameter groups, count/bytes estimate, and which groups are trainable in which policy. |
| `frozen_dependencies` | Dependency roles/interfaces that local experiments must be able to freeze. Exact resolved versions/snapshots belong to the experiment binding. |
| `update_frequency` | Declared logical cadence; no hidden scheduler-driven semantic cadence. |
| `causal_dependencies` | Structural prerequisites and state/snapshot requirements. |
| `plasticity_policy` | When and how parameters/state may update; default-deny outside the declared mode. |
| `checkpoint_schema` | Independent shard contents, versions, format, integrity, and restore rules. |
| `resource_budget` | Parameter/state sizes, active VRAM, utilization, latency, and transfer limits/estimates. |
| `local_acceptance_tests` | Test IDs, metrics, and deterministic PASS criteria local to this module. |
| `causal_ablation_tests` | Paired intervention tests, expected loss, and non-target guard criteria. |
| `integration_tests` | Consumer/provider contract and bounded graph regressions. |
| `parallel_execution_safety` | Mutable-state ownership and whether independent, cloned, causal-bundle, or serial execution is safe. |
| `offload_policy` | Allowed HOT/WARM/COLD transitions, safe boundaries, transfer budget, and round-trip tolerance. |
| `compatibility_version` | Version of the externally observable tensor/state assumptions used for substitution. |

The foundation also requires `manifest_schema`, `telemetry`, and `provenance`
fields so manifests are self-identifying, observable, and attributable.

## 3. Interface rules

### 3.1 Structural contract

Each endpoint has a stable `interface_id`, multiplicity, and one latent schema;
its containing input/output collection establishes direction. Every tensor
declares at least name, rank, symbolic/fixed shape, dtype family,
representation-contract ID, and normalization. Variable axes use named symbols
with bounds defined in the human specification. The conformance
validator MUST additionally enforce unique interface IDs and exact equality
between interface IDs and latent-schema-map keys; JSON Schema cannot express
that cross-field invariant by itself. It also verifies that each tensor’s
declared rank equals its shape length and that an enabled persistent state is
included in the checkpoint contract (the latter is also constrained by the
canonical schema).

The structural envelope MAY include masks, positions, module/version IDs,
logical step, snapshot ID, and experiment provenance. It MUST NOT include
control-plane intent/ACT/TARGET labels used to select cognitive behavior.

Representation-contract IDs are assigned by the approved contract/registry,
not trusted as a producer’s self-assertion. Compatibility tests include a
shape/statistics-preserving rotated-basis negative case: a different ID is
rejected structurally, and a falsely reused ID fails the frozen-consumer
behavioral test.

### 3.2 Dense/distributed representation

Foundation interfaces use dense tensor representations. A discrete dataset
label may be used by a test evaluator, but it does not travel as a semantic
routing command between modules. A learned adapter is a module with its own
contract and evidence; a handwritten semantic mapping is not an adapter.

### 3.3 Compatibility

Matching `compatibility_version` is necessary but not sufficient. Replacement
requires all of the following:

- provider output and consumer input schemas are structurally compatible;
- state/reset/update assumptions remain valid;
- numerical ranges and tolerances meet the human-readable contract;
- the replacement passes its local tests;
- frozen-consumer contract tests pass without updating consumer parameters;
- the bounded integration regression passes.

Any consumer retraining means the “replace without retraining” gate has not
been demonstrated for that substitution.

## 4. State ownership and lifecycle

### Private state

Private state belongs to one module instance. The module declares
initialization, reset, detach/truncation, clone, snapshot, and numerical
validity rules. No other module receives a mutable reference to it.
FOUNDATION_0 A/B/C manifests require at least one state tensor and a causal
state-use test. The general contract permits `enabled: false` only with an
explicit architecture-decision reference in `exception_decision`.

### Persistent state

Persistent state either has a named versioned schema or is explicitly disabled.
It declares serialization, migration, retention, and restore validation. A
transcript, artifact path, or language-model cache is not automatically
autobiographical memory.

### Runtime lifecycle

The control plane may move an instance through:

```text
registered -> inactive -> loading -> active -> offloading -> offloaded
                                  \-> error
```

Transitions occur only at declared safe boundaries. `offloaded` retains the
state promised by `offload_policy`; `inactive` need not occupy GPU memory.
Lifecycle state is operational metadata and cannot determine cognitive
meaning.

### Concurrent candidates

Candidate instances may share immutable frozen parameter snapshots. They MUST
NOT share mutable private state, optimizer state, random-number generator
state, or writable artifacts. If cloning is required, the snapshot and clone
method are part of `parallel_execution_safety`.

## 5. Parameters and plasticity

`trainable_parameters` enumerates parameter groups and distinguishes existence
from permission to update. `plasticity_policy` specifies permitted modes, for
example frozen evaluation, local candidate training, or a future bounded
online update. Contract `v1alpha1` always defaults to `frozen`; only an explicit
experiment binding can enable another declared policy.

FOUNDATION_0 authorizes only the bounded synthetic local updates recorded in
`configs/foundation0_cpu.json`: A1 and A2 train independently against the
canonical local latent target, and B trains on frozen A1 outputs before being
frozen. A3 and C are not optimized. Replacement, negative-control, ablation,
restore, telemetry, and concurrency runs perform zero B optimizer steps. This
does not authorize language-model, production-module, or online training.

## 6. Checkpoint contract

Every shard MUST contain or reference:

- `module_id`, `module_version`, and `compatibility_version`;
- parameter and private/persistent state schema versions;
- parameter tensors and promised restorable state;
- logical step and snapshot identity;
- provenance sufficient to identify the producing experiment;
- an integrity digest appropriate to the artifact risk;
- explicit format/version and migration entry point.

Saving module `A` cannot require serializing module `B`. A whole-system
checkpoint is a manifest over independently addressable shards. It is valid
only when all required shards share a consistent snapshot and pass integrity
and compatibility checks. Failed or partial attempts remain identifiable and
are never silently overwritten as valid.

The `v1alpha1` shard digest algorithm is SHA-256. A future algorithm change is a
checkpoint-schema version change, not the ambiguous claim “stronger.”

## 7. Resource and offload contract

The manifest separates estimates/reservations from observed measurements. At
minimum it covers parameter bytes, private/persistent-state bytes, estimated
isolated active-VRAM peak, incremental module reservation above the calibrated
worker baseline, estimated GPU utilization contribution under a named isolated
workload, latency budget, and GPU↔RAM transfer cost. Runtime measurements are
telemetry; observed violations do not mutate the declared budget. Worker,
workspace, transport, telemetry, shared-allocation, and safety-margin terms are
added once at bundle/controller scope under [RESOURCE_POLICY.md](RESOURCE_POLICY.md).

Per-module GPU mean/peak utilization, GPU time, and scoped peak VRAM use a named
isolated workload/window/method. Concurrent evidence reports aggregate
device/process utilization plus exact owned tensor bytes; it does not copy the
aggregate value onto every module.

The offload policy declares permitted destinations, safe points, state included,
maximum transfer amount/latency, and comparison tolerance after reload. A
module claiming GPU→RAM→GPU support must pass an output/state round-trip test
from the same captured input/state and must not rely on constant PCIe churn.

## 8. Required evidence

A manifest test reference is not evidence by itself. Before a module version is
conforming/promotable, every local, causal, and integration test ID resolves to
an immutable result with evaluator version, artifact digest, and explicit PASS;
dangling, stale, NOT_RUN, SKIP, or XFAIL references fail conformance.

### Local acceptance

Local tests isolate the module, use fixed/replayable latent fixtures, validate
shape/dtype/state transitions and the bounded synthetic capability, and report
explicit metrics and thresholds. They do not launch whole-system
qualification.

### Causal ablation

An ablation test names the intervention, matched snapshot/input, expected
target metric loss, non-target guard metrics, seeds/repeats, and PASS rule.
Disabling, zeroing, lesioning, or replacing a module is acceptable only when it
tests the registered causal hypothesis rather than merely breaking the graph.

### Integration

Integration tests verify declared provider/consumer contracts, graph
composition, compatible replacement, checkpoint composition, and affected
capabilities. They are bounded by the changed interface/dependency surface.

## 9. Telemetry contract

Each active experiment MUST be able to emit, per module instance. GPU fields
are explicit `null`/`NOT_RUN` in the CPU stage rather than fabricated zeros:

- module ID/version/compatibility and lifecycle state;
- current experiment and candidate identity;
- wall latency and GPU time;
- isolated-workload mean/peak GPU utilization and scoped peak VRAM;
- current/peak active VRAM and transfer counts/bytes/time;
- state norm and finite-value statistics;
- activation/output mean, standard deviation, norm, extrema, and finite count;
- checkpoint/state schema and snapshot version;
- latest relevant local PASS/FAIL/NOT_RUN status;
- timestamp, logical step, and telemetry schema version.

Telemetry collection cannot modify inputs, outputs, parameters, state, graph
edges, or scheduling according to semantic values. Sampling policy and
measurement overhead are recorded.

## 10. Versioning and migration

Version axes are intentionally separate:

| Axis | Changes when |
|---|---|
| `module_version` | Implementation, parameters, or behavior changes. |
| `compatibility_version` | Observable interface/state assumptions change. |
| latent schema version | Tensor structure/range contract changes. |
| state/checkpoint schema version | Serialized representation changes. |
| manifest schema version | Metadata syntax changes. |

Migrations operate on one module shard and declare source, target, lossiness,
validation, and rollback/retention behavior. A whole-system migration is not
the default. There is no implicit migration path from NMCM v1.

## 11. Human-readable module specification template

Each future module specification should contain:

1. identity, purpose, non-goals, owner, and version history;
2. functional claim or, where uncertainty requires research, an architecture
   hypothesis and expected causal contribution;
3. input/output latent semantics for researchers (never a runtime router);
4. internal parameter/state overview and declared coupling;
5. update/plasticity and frozen-dependency policy;
6. checkpoint/migration and offload behavior;
7. resource table with estimation method;
8. local, ablation, integration, and replacement evidence links;
9. known limitations and failure modes;
10. exact manifest path and digest/version.

No production module specification is created during the documentation phase.
