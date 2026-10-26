<!-- CURRENT: PRE_DAY0_CANDIDATE_REPAIRS_20261026; earlier scopes below are historical. -->

The owner now authorizes a separate PRE-DAY0 candidate branch/PR to repair state
export, response boundaries, current context with existing M, and recovery and
resource limits. See docs/PRE_DAY0_CANDIDATE.md. Keep Qwen/Bridge/M architecture
and published memory schemas; no training, donor replacement or new cognitive
module. If Base remains unsuitable after the bounded technical fix, stop that
stage with concrete evidence and a minimal owner decision. Do not merge this PR,
accept Day0 or appoint a canonical instance without separate review/confirmation.

---

<!-- CURRENT: PUBLIC_V0_PRE_DAY0_20261026; older task scopes below are historical. -->

Current owner authority: sequential publication/merge of Stage1 PR2, Memory PR3,
and Dialogue PR4 after current CI and dependency/review checks, followed by the
PRE-DAY0 GAP REPORT. See STATUS.md and docs/PRE_DAY0_V0_GAP_REPORT_RU.md.
No new training or architecture is authorized for publication. No new cognitive
module before this gap analysis; Day0/canonical instance need separate owner
confirmation. Day0 -> close V1 -> cortical Alice v2. Keep exact source, external
asset and capacity/context limitations explicit. The prior records below do
not restart older work or unused fitting budgets.

---

# Alice working policy — Engineering First v2

Authority: owner decision of 25 October 2026. This is the active policy for new
Alice development. The owner approved the roadmap at PR #1 commit `9c29e08`;
it was merged as `641d40e`. Stage 1 technical runtime is authorized. Memory/Bridge
algorithms and training remain separate future decisions.

**ENGINEERING FIRST. REAL BEHAVIOR FIRST. MINIMAL NECESSARY COMPLEXITY.**

## Scope and authority

Follow the current explicit owner task, this policy, and the active engineering
rules in [ENGINEERING_MODULE_CONTRACT.md](ENGINEERING_MODULE_CONTRACT.md) and
[ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md). Read the current
[architecture status](ARCHITECTURE.md), relevant roadmap scope and Git status
before changing a component. Preserve unrelated user changes.

New development does not inherit mandatory M1-style qualification,
TRAIN/DEV/FINAL gates, near-perfect research scores, repeated seed/origin studies,
large synthetic corpora, exhaustive intermediate retention, or proof of novelty.
Science supplies useful technical solutions; a separate scientific publication
is not a development objective. Use additional measurement or controls only when
they resolve a concrete engineering uncertainty.

[MODULE_CONTRACT.md](MODULE_CONTRACT.md),
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), the
[Foundation architecture baseline](docs/FOUNDATION_ARCHITECTURE_V0.md), historical
ADRs and accepted evidence remain historical, frozen Foundation contracts in
their original scope. Their historical wording is not the default workflow for
new Alice functions. They still govern existing Foundation implementations and
any future work that explicitly adopts that contract. Do not silently change
their schemas, interfaces, thresholds, test meanings or results. An intentional
Foundation contract change needs explicit versioning and affected checks.
M1 remains archived, NOT_FROZEN, and separate from active Alice development.

## Build useful abilities

For every proposed cognitive function, state what Alice must now be able to do.
Check whether Qwen or an existing mechanism already supplies it. Before writing
a new mechanism, search existing engineering and neural solutions, check their
applicability, licenses, resource cost and interface fit, then integrate only the
needed functions and dependencies. Prefer reuse, merging or removal when a
separate module adds no necessary ability. Do not adopt a heavy architecture or
another model for a small function already available elsewhere.

The historical M0–M13 list is a functional decomposition, not a requirement for
thirteen independently trained networks. A separate class, valid tensor,
different embedding, passing unit test or successful training run does not
establish useful behavior. Test each new cognitive module with **MODULE OFF**
and **MODULE ON** on the same relevant real-runtime scenarios. PASS requires a
useful task improvement and no unacceptable regressions; a changed answer alone
does not qualify. Revise, merge or remove ineffective mechanisms.

## Keep the approved boundaries

- Qwen3.5-2B-Base forms responses with frozen, replaceable weights.
- Neural Memory is planned as independent plastic memory with its own mutable
  parameters and lifecycle.
- Memory Bridge is planned as a replaceable trainable interface through which
  memory influences Qwen computation and responses.
- Further cognitive mechanisms exist only when a necessary behavior justifies
  them. Support code may be deterministic; it does not need a new neural model.
- ComfyUI is the selected future external graph/UI/API layer. Cognitive state
  must outlive interface sessions.

Memory and Bridge algorithms remain unselected. Do not present these planned
mechanisms, a Qwen dialogue runtime or ComfyUI integration as implemented. A
transcript, prompt history or retrieval store alone does not implement the
approved plastic Neural Memory. Reuse existing Foundation interfaces without
an unjustified redesign; compatibility must be checked rather than assumed.

## Work, verification and storage

Prioritize an early minimal dialogue runtime, then useful integrated abilities.
Use real messages, sequential conversations and context changes; memory work
also needs acquisition, later use, restart persistence and replacement checks.
Use critical unit tests, integration, save/load/recovery, state-integrity,
disable/replace and VRAM/RAM/disk checks in the affected scope. Keep the set small
and meaningful. Metrics and repetitions serve concrete correctness or usefulness
questions; do not introduce arbitrary ideal thresholds.

When a mechanism fails: reproduce the failure, identify its technical cause,
fix or replace the affected component, repeat relevant engineering checks, and
verify the result in the real runtime. Do not automatically start multi-day
training or a new collection of experimental architectures. Reuse unchanged
Foundation evidence instead of requalifying it for unrelated changes.

Retain working checkpoints, code/configuration, necessary test results, short
failure reports and artifacts needed to recover important results. Estimate
disk cost before large computation and identify disposable outputs and their
cleanup point. Temporary feature banks, tensor dumps and repeated intermediate
results are not retained indefinitely by default. This policy does not authorize
deletion of historical M1/Foundation evidence or unknown user data.

## Current work boundary and reporting

This source-only continuation is prepared from Stage 1 `626b6f8` and accepted
local runtime/memory integration `4cecf57`, under the owner's subsequent explicit
authorization. The subsequent bounded dialogue continuation reuses verified
production code `87d7524`: explicit RU/EN object-label intent, transient
speaker/session-bound confirmation and answer-blind RU query adaptation. It is
deterministic support code, not a new neural language architecture or free NLU.
It contains production dependency closure, asset-free tests and
portable usage documentation only. Stage 1 PR #2 must be reviewed by the owner;
preparation is not an approval, publication or merge. No remote mutation or model
fitting is authorized by this package. Preserve separate Bridge/M lifecycle,
structured one-token acquisition, question-only memory framing and the ordinary
Stage 1 context mode when memory is absent. See
[runtime memory](docs/NEURAL_MEMORY_RUNTIME.md). Statements below are the original
Stage 1 task boundary and do not erase later accepted local work.

The current work package implements Stage 1: local frozen-Qwen generation,
participant/source IDs, sessions, errors and a typed optional Bridge connection.
No training, new large-model downloads, Neural Memory implementation or trained
Memory Bridge is authorized here. Report the short technical result and Collab
options for the next cognitive milestone; do not optimize dialog metrics.

Preserve upstream licenses/provenance. Do not publish credentials, private
conversations, model assets, caches or historical runs. Report what changed,
what checks actually ran, what remains planned or untested, and the next concrete
decision. A policy/documentation change cannot claim a new Alice capability.
