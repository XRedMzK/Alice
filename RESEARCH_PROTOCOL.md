# Research protocol

Status: **FROZEN for FOUNDATION_0 CPU baseline**

Qualification: **CPU 15/15 PASS; GPU 15/15 PASS; FOUNDATION_0 CLOSED**. The
protocol remains the frozen scientific baseline for those results.

## 0. Preservation and forward applicability

Sections 1–11 below preserve the accepted FOUNDATION_0 baseline, including its
scientific terminology, causal gates, evidence requirements, and exact
PASS/FAIL classification. Nothing in this forward-applicability note
retroactively changes those results or weakens a work-package-specific frozen
contract.

For future work, first follow the task classification and authority rules in
[CODEX.md](CODEX.md):

- **IMPLEMENTATION TASK:** build specified behavior; normally close with
  unit/regression/integration evidence and no research experiment;
- **ENGINEERING EXPERIMENT:** resolve uncertainty about mechanism/design
  behavior with claim-proportional controls;
- **ARCHITECTURE RESEARCH:** decide what mechanism Alice should use, potentially
  using neuroscience/cognitive-science motivation and formal comparison.

> **Use science to design the architecture; use engineering evidence to
> determine whether Alice works.**

Validation rigor for future work is proportional to uncertainty,
irreversibility, integration risk, and importance. Use the lowest level that
closes realistic failure modes:

1. **LEVEL 1 — Implementation validation:** unit, negative, schema,
   deterministic-fixture, and regression tests.
2. **LEVEL 2 — Module functional validation:** functional and bounded
   integration evidence, failure modes, persistence/recovery, and causal
   ablation when causal contribution matters.
3. **LEVEL 3 — Generalization / learning qualification:** held-out evidence,
   distribution/support controls, relevant seeds, baselines, and leakage or
   memorization checks.
4. **LEVEL 4 — Irreversible / adaptive high-stakes qualification:** pre-freeze,
   sealed holdout, publish-once evidence, independent review, and strict
   provenance when the claim requires them.

The phrase “every experiment” below means every registered experiment governed
by this frozen protocol or by a future contract that explicitly adopts it. It
does not convert an ordinary deterministic implementation task into an
experiment or make LEVEL 4 ceremony the universal minimum.

## 1. Governing loop

For the registered FOUNDATION_0 research/qualification work, and for future
engineering or architecture-research packages that explicitly adopt this
higher-rigor loop:

```text
LOCAL HYPOTHESIS
  -> preregistered local experiment
  -> local implementation/change
  -> local test with frozen/replayed dependencies
  -> controlled causal ablation
  -> PASS or FAIL with evidence
  -> freeze a passing module version
  -> bounded integration regression
```

The smallest experiment capable of answering the scientific question is the
default. Research wall-clock, including queueing, setup, evaluation, integrity
work, and retries, is a first-class metric. The local-question rule exists for
debuggability, causal attribution, replaceability, and reduced ambiguity. For
ordinary implementation work with no unresolved behavior/design uncertainty,
the default is implementation plus affected tests, not this full loop.

## 2. Separation of concerns

When a registered experiment/qualification uses them, three artifacts are
distinct and versioned independently:

1. **Scientific contract:** hypothesis, intervention, controls, dataset/fixture
   identity, metrics, thresholds, seeds/repeats policy, expected failure, and
   non-target guards.
2. **Evaluator:** path-agnostic code that consumes a typed evidence manifest
   and implements the registered metrics/decision rule.
3. **Execution binding:** module versions, immutable snapshots, hardware,
   device, artifact paths, process/concurrency settings, and concrete seeds.

Changing a path, device, or runner cannot silently change the hypothesis or
thresholds. Evaluators do not infer scientific meaning from filenames or run
directories. Runtime telemetry is evidence input, never a cognitive input.

This separation remains mandatory for the frozen FOUNDATION_0 evidence and any
future contract that adopts it. A LEVEL 1 implementation task need not invent
all three artifacts when direct tests fully close its claim.

## 3. Experiment registration under this frozen baseline

Before execution, every experiment governed by this frozen protocol records:

- stable experiment ID and version;
- one falsifiable question and why a local test is sufficient or insufficient;
- target module/version and exact frozen dependencies;
- interface, fixture/dataset, and captured-state versions/digests;
- independent or causal-bundle classification;
- intervention and control arms;
- primary metric, direction, threshold, and aggregation rule;
- non-target guard metrics and tolerances;
- seeds/repeats and stopping rule fixed before results are inspected;
- expected resource reservation and wall-clock bound;
- evaluator version and output evidence schema;
- escalation trigger for bounded integration or full-system qualification.

Threshold changes after observing results create a new experiment version. A
failed attempt is retained as compact immutable metadata/evidence; it does not
force the next version to preserve a flawed architecture.

For future work, preregistration and pre-execution freeze are required when
seeing a result could influence a later choice: final qualification, benchmark
or model selection, frozen PASS gates, adaptive testing, or consumption of a
holdout. They are not mandatory for deterministic implementation work with no
adaptive scientific or engineering choice. An explicit work-package authority
may always require stricter registration.

## 4. Experiment classes

### Local module experiment

Loads only the target module and minimal tensor fixtures/test doubles. It
checks the local hypothesis, interface, state ownership, numerical validity,
checkpoint shard, and declared resource behavior. Its evidence records loaded
modules so locality itself is testable.

### Independent candidate bundle

Runs candidate module versions from the same immutable input/upstream snapshot.
Each candidate owns all mutable neural state, RNG state, optimizer state (if
any), telemetry stream, and artifact root. Immutable parameters/fixtures may be
content-addressed and shared read-only. Independent bundles are eligible for
concurrent scheduling.

### Causal bundle

Contains paired control/intervention arms derived from the same immutable live
state, input, versions, and seeds. Arms may execute sequentially or in isolated
deep clones, but their state provenance must match and mutable tensor storage
must not alias. A no-op/sham arm verifies that snapshot cloning and the
intervention harness do not themselves change output/state. A causal bundle is
not split into unrelated scheduler work merely to increase utilization.

### Bounded integration regression

Exercises the changed module’s declared neighbors, interfaces, checkpoint
composition, and affected capabilities. Unaffected graph regions remain frozen
or absent. Its scope is derived from explicit edges and compatibility changes.

### Full-system qualification

Reserved for an integration milestone, shared interface/contract change,
blocker qualification, or a modification with plausible system-wide effects.
The reason must be recorded. Convenience, lack of a local evaluator, or a
monolithic runner is not sufficient justification.

## 5. Causal ablation standard when causal responsibility is claimed

The accepted FOUNDATION_0 causal gates remain mandatory and unchanged. For
future work, use this standard when the engineering/research claim is that a
module or mechanism is responsible for property X. Do not require an ablation
for every plumbing/helper component; a deterministic exact transformation may
be closed by direct correctness, negative, regression, and integration tests.

A valid causal test specifies:

- the module/function claim being tested;
- an intervention such as disable, lesion, zero, state reset, or compatible
  substitute;
- matched input and pre-intervention snapshot;
- a sham/no-op intervention from an independent deep clone;
- a target metric expected to degrade by a preregistered amount;
- non-target metrics expected to remain within preregistered tolerances;
- repeated seeds when nondeterminism is material;
- checks that the intervention did not corrupt the interface or evaluator.

“Ablation changes only the expected capability” means the observed changes
remain inside a preregistered causal effect surface and non-target guard band.
It does not claim that a recurrent graph has no downstream numerical response.
Ablating a module by creating invalid shapes, NaNs, or missing dependencies is
an infrastructure failure, not causal evidence.

Sequential arms restore distinct deep clones of the initial state; they never
reuse an object already mutated by an earlier arm. Sham equivalence is a
mandatory guard before interpreting the ablated arm.

## 6. PASS, FAIL, and invalid results

- **PASS:** every mandatory primary and guard criterion passes and required
  evidence is complete.
- **SCIENTIFIC_FAIL:** the valid run falsifies or fails to support the
  registered hypothesis.
- **CONTRACT_FAIL:** manifest/interface/state assumptions are violated before
  the question can be tested.
- **RESOURCE_FAIL:** admission or measured resource bounds are violated.
- **INFRA_FAIL:** environment, process, artifact, or telemetry failure makes
  the scientific result invalid.
- **NOT_RUN:** no valid attempt exists.

Only PASS and SCIENTIFIC_FAIL answer the scientific question. Required
`SKIP`, `XFAIL`, missing telemetry, or unavailable GPU evidence cannot be
counted as PASS. The CPU milestone is conjunctive: all `CPU-01`–`CPU-15` gates
must pass. GPU/resource qualification uses a separately named gate set, and
`NOT_RUN` hardware evidence never inherits the CPU result. The accepted
FOUNDATION_0 bundles independently passed both conjunctions.

The `SCIENTIFIC_FAIL` name above is the exact frozen FOUNDATION_0/research
classification. A future valid implementation or engineering execution that
fails its functional contract may instead use `VALIDATION_FAIL` (or its
work-package-defined equivalent). This is distinct from `CONTRACT_FAIL`,
`RESOURCE_FAIL`, and `INFRA_FAIL`, where execution cannot validly answer the
question. Do not retroactively rename accepted terminals.

## 7. Freeze and promotion

After local and causal PASS:

1. assign an immutable module version;
2. freeze manifest, human specification, parameters/state fixture, evaluator,
   and evidence digests;
3. run consumer compatibility tests;
4. run the bounded integration regression;
5. publish the version to the local registry only if all required checks pass.

Failed candidates remain traceable but are not promoted. A later candidate can
replace them without maintaining a bad internal design. Interface changes use
a new compatibility version and impact review.

This post-PASS promotion freeze is not the same as pre-execution
preregistration. For future work, independent review is required when
qualification is irreversible, a holdout becomes consumed, architecture state
is frozen, a high-impact lifecycle transition occurs, ambiguity could invalidate
future evidence, or the work package explicitly requires it. Ordinary LEVEL 1
or LEVEL 2 development has no universal review requirement.

## 8. Reproducibility and integrity

Evidence contains source revision/diff state, environment, exact manifests,
graph config, fixture/dataset digest, snapshot/shard digests, seeds, evaluator
version, timings, resources, metrics, and decision. Artifact writes are atomic
and run roots are isolated.

Integrity work is proportional to risk:

- hash on import, creation, mutation, migration, checkpoint publication, and
  promotion boundaries;
- validate size/schema/digest of directly consumed evidence;
- reuse an immutable content-addressed digest inside read-only local runs;
- do not repeatedly stream-hash every large untouched artifact at each stage
  without a concrete threat or mutation boundary.

Exact duplicates and hash separation do not prove semantic independence. Any
future dataset qualification must also test template/near-duplicate leakage
using task-appropriate methods.

Statistics are claim-dependent. Use statistical methods for stochastic
training, noisy measurements, seed sensitivity, variable human evaluation, or
uncertain distributional performance. Deterministic exact-contract checks do
not gain confidence from automatic statistical ceremony; repeated seeds remain
required only when nondeterminism is material or an explicit contract requires
them.

Fail closed when corrupted evidence, ambiguous frozen inputs, unsafe
persistence, leakage, invalid model selection, impossible replay, violated
resource constraints, unknown semantic mapping, or destructive integration
ambiguity could create a false PASS or damage authoritative state. Fail-closed
behavior protects engineering state; it is not a requirement to maximize
procedural complexity.

Do not add experimental machinery, artifact layers, reviews, frozen phases, or
statistical procedures unless they reduce a concrete engineering or evidential
risk for the claim. This does not remove unit/regression/integration tests,
negative controls, held-out evaluation, deterministic replay, artifact hashes,
crash/recovery checks, resource gates, causality checks, or one-shot/sealed tests
when those methods close a real risk.

The E2PL branch is the maximum-rigor path for its adaptive generalization claim:
adaptive benchmark design, data-support confounds, model selection,
future-holdout freshness, and one-shot qualification justified its stricter
custody. It is not the default minimum ceremony for every future module or
implementation task, and this statement does not modify any E2PL artifact.

## 9. Parallel research policy

Parallelism is declared before execution. The scheduler uses resource
reservations and state ownership, not semantic content. It may co-schedule
several low-utilization independent candidates when that reduces wall-clock.
It must serialize, snapshot, or clone any work with live-state coupling.

Parallel equivalence is tested against serial results within registered
tolerances. Failure to prove isolation causes a CONTRACT_FAIL and serial
fallback for that bundle; it does not permit undeclared shared state.

## 10. Test locality and escalation

Every change lists the narrowest affected tests. A local test MUST be runnable
without invoking the full qualification command and SHOULD assert that no
unrequested modules were loaded. Escalation occurs only when:

- the compatibility version or shared interface changes;
- explicit dependency analysis reaches multiple graph regions;
- global scheduler/checkpoint semantics change;
- a blocker requires milestone-level evidence; or
- local results expose an interaction the local fixture cannot answer.

The escalation reason and expanded test boundary are recorded before the
larger run starts. No multi-hour experiment is approved when a bounded test can
answer the same question.

## 11. FOUNDATION_0 application

The frozen synthetic plan, metrics, and gate-to-evidence mapping are in
[docs/FOUNDATION_0_IMPLEMENTATION_PLAN.md](docs/FOUNDATION_0_IMPLEMENTATION_PLAN.md).
That CPU work package authorized its small deterministic CPU training and
tests only. Decision 0002 separately authorized the one now-complete bounded
GPU qualification. Neither accepted package authorizes another GPU run,
model/language-weight download, production cognition, Alice work, or DAY0.
