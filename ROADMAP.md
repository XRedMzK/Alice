# Alice functional roadmap — Public v0 / PRE-DAY0

26 October 2026. Engineering First v2 and the functional roadmap were approved
at PR #1 `9c29e08`, merged as `641d40e`. The owner authorized sequential public
integration of Stage 1, Memory and Dialogue on 26 October 2026.

## Delivered in the bounded Public v0 source

- Stage 1 frozen local Qwen runtime, participant/source IDs, sessions and typed
  embedding connection: [PR #2](https://github.com/XRedMzK/Alice/pull/2).
- Independent persistent Neural Memory and compatible sealed embedding Bridge:
  [PR #3](https://github.com/XRedMzK/Alice/pull/3). Only M changes on acquisition.
- Explicit supported RU/EN object-label intent, exact proposal confirmation,
  restart/correction recall and public installation/reproduction instructions:
  [PR #4](https://github.com/XRedMzK/Alice/pull/4).

This is one-token memory over literal IDs and explicit query forms, not an
unrestricted assistant. External Qwen and the exact compatible Bridge remain
required; the [CPU reconstruction recipe](docs/PAYLOAD_BRIDGE_RECONSTRUCTION.md)
is verified only on its recorded profile and fails closed on a different digest.
See [STATUS](STATUS.md), [architecture](ARCHITECTURE.md) and
[observed validation](docs/PUBLIC_V0_VALIDATION.md).

## Next work: PRE-DAY0 gaps

Follow the [PRE-DAY0 v0 GAP REPORT](docs/PRE_DAY0_V0_GAP_REPORT_RU.md): ordinary
dialogue quality, current-context use, memory/error recovery, long interaction,
state protection, and launch/shutdown. Current-question-only memory mode and
unsafe transcript export are concrete limits. Do not start new cognitive modules
or training before addressing the owner's decision on these gaps.

Public v0 is not Day0 acceptance. Day0 v0 and a canonical working instance need
separate owner confirmation. The sequence is Day0 -> close Alice V1 -> later
cortical Alice v2 systems, without skipping V1 closure.

## Historical approved functional crosswalk

The 25 October table/sequence below preserves the approved functional reasoning,
not a queue of networks or a current completion checklist. Earlier unselected
mechanism/future-step wording belongs to that snapshot. Current implementation
and remaining gaps are stated above and in STATUS. Foundation/M1 contracts,
results and source remain unchanged. M1 stays RESEARCH ARCHIVED / NOT_FROZEN;
its qualification is not an Alice prerequisite.

## Functional priorities

**CORE** means necessary for the approved Alice concept. **SUPPORT** enables its
operation and interaction. **LATER** does not block the first useful capability.
**REMOVE/MERGE** removes a mandatory standalone implementation while preserving
any useful function. A category does not require a new network.

The M0–M13 labels below are a review crosswalk for the historical cognitive
decomposition, not a new contract or a sequence of thirteen networks. Historical
Foundation milestone numbers used a different namespace; some older plans also
grouped functions differently. Function names and decisions govern this proposal.
No prior scientific result or qualification status is reclassified.

Each row addresses the owner's seven questions: necessary ability; existing
coverage; need for custom neural code; reusable solution; independence or merging;
real behavior check; consequence of omitting the function.

| Historical function | Disposition | Existing coverage, reuse and implementation decision | Real behavior check; consequence of omission |
| --- | --- | --- | --- |
| M0 — Foundation | SUPPORT, retain | Reuse existing registry/graph, owned state, shards, compatibility and resource control in `src/nmcm` / `src/nmcm_gpu`. Qwen does not supply these. No new network or speculative redesign. | Disable/replace a component, restore and reject corruption without damaging neighbors. Infrastructure needs correctness tests, not artificial answer degradation. Omission duplicates working infrastructure and risks state loss. |
| M1 — Current-message semantics | REMOVE/MERGE standalone requirement | First test Qwen's RU/EN understanding in runtime; sufficiency is unproven. Reuse its representations. Only address a reproduced gap with a narrow ready mechanism or interface change. Archived M1 is not required. | Negation, participant/possessor changes and meaning contrasts affect answers appropriately. Additions need OFF/ON. Omitting a separate network is acceptable; omitting comprehension makes dialog unusable. |
| M2 — Source / SELF / PARTNER | CORE, minimal identity from first runtime; richer participant cognition LATER | Represent Agent / User / Another Person #N and carry a stable `source_id` with messages/events from the first runtime. Reuse deterministic identity/provenance plumbing; bind acquired knowledge to its source in the first memory/Bridge implementation. No separate source network or advanced participant model is required now. | Different people keep distinct IDs; the same person's ID survives relevant session/save/restore boundaries. Later knowledge use and a source-specific correction preserve attribution. IDs alone do not prove cognitive source reasoning. Omission mixes participants and acquired knowledge. |
| M3 — Temporal conversational state | CORE, merge with context and memory | Reuse message ordering and Qwen context first. Cross-session order/currentness belongs in memory/Bridge behavior. No separate temporal network selected. | Reverse events, correct an obsolete fact and ask about current versus earlier state; later repeat across sessions. Omission makes currentness and correction unreliable. |
| M4 — Context integration | REMOVE/MERGE into runtime and Bridge | Qwen handles visible context. A trainable Bridge must integrate independent memory with its computation. Reuse compatible inputs/conditioning primitives; no extra fusion model by default. | Same question with different histories; memory ON improves the relevant answer without irrelevant interference. A separate M4 may disappear; integration cannot, or memory remains unused. |
| M5 — Natural-language production | SUPPORT, use Qwen | Frozen replaceable Qwen3.5-2B-Base is the approved response model. Reuse generation, stops and bounded context; no second language network. Start with a brief technical smoke check and Memory Bridge connection feasibility, not a separate dialog-quality qualification. | Runtime can generate, carry participant/source IDs and attach/bypass a candidate Bridge boundary without updating Qwen. Record dialog limitations and propose a bounded remedy; useful use of acquired knowledge is tested in the first cognitive milestone. Without production Alice cannot express that capability. |
| M6 — Epistemics / curiosity | Minimal epistemics CORE; active curiosity LATER | Test Qwen with source/currentness and memory first. Reuse adequate clarification/uncertainty behavior; only fill demonstrated gaps. Merge basic handling with memory/Bridge; no separate network yet. | Unknown term, explanation and conflicting correction cause appropriate clarification/qualification and later use. Omission allows guesses to masquerade as acquired knowledge; autonomous curiosity can wait. |
| M7 — Partner evaluation | LATER | Qwen may respond to feedback in current context. Later reuse explicit feedback and memory/preference updates if useful; no separate learner now. | LIKE/DISLIKE affects an agreed preference or salience channel, not factual truth; NONE invents no signal. Omission does not block first dialog or memory. |
| M8 — Runtime plasticity | CORE, part of Neural Memory | Qwen stays frozen. Foundation state handling is not experience acquisition. Reuse a suitable neural associative/fast-weights mechanism if applicable. Independent mutable memory parameters are required, custom invention is not. Merge with M9 and preserve `source_id` on acquisition. | New knowledge changes memory parameters; frozen Qwen uses it through the trainable Bridge after prior context is cleared and the process restarts. MEMORY OFF loses or degrades the ability. Omission removes independent acquisition, a central Alice requirement. |
| M9 — Episodic memory / consolidation | Memory CORE; elaborate consolidation LATER | Combine acquisition, readout and retention in independent plastic Neural Memory, connected through a trainable replaceable Bridge. Bind knowledge to `source_id` from the first memory implementation. Pretrained knowledge, transcript retrieval and KV cache are insufficient; reuse minimal neural-memory primitives, not a default consolidation network. | Source-bound knowledge is used by frozen Qwen after context clearance and fresh process restore; MEMORY OFF/ON demonstrates its contribution. Omission removes acquired experience; elaborate consolidation need not block small useful memory. |
| M10 — Persistence | SUPPORT, early and essential | Reuse Foundation shards, integrity/schema checks and checkpoint primitives; no neural code needed. Save memory and Bridge separately with explicit versions/compatibility. | Save, terminate, fresh restore and use learned information without old transcript; corruption cannot expose partial state. Omission loses experience on restart. |
| M11 — Vision | LATER | Initial scope is text. Later assess an existing visual encoder and minimal memory interface; preserve modality/source and adopt only needed dependencies. No custom vision network assumed. | New image changes later conversation and restored experience; modified/removed image changes the target result. Omission does not block text; old full-V1 vision acceptance cannot be claimed. |
| M12 — Integrated Alice | CORE, continuous integration | One runtime composes abilities. Reuse Foundation/model APIs; this is an outcome, not another network. Integrate from the first technical runtime, with explicit participants and `source_id`. | One continuous conversation acquires, corrects, changes context, uses experience later and restores without operator transfer. Omission leaves isolated demonstrations. |
| M13 — Human acceptance / DAY0 | SUPPORT, owner milestone | Owner review of useful behavior, not a model. Remove from the queue of networks. Runtime acceptance and DAY0 remain separate explicit owner actions. | Owner evaluates actual scenarios and limitations. Roadmap approval, synthetic PASS and Foundation acceptance imply neither runtime acceptance nor DAY0. |

The **Memory Bridge** is CORE: independent memory must influence Qwen computation
and answers through a replaceable trainable interface. It absorbs integration
work formerly spread across functions; its exact algorithm remains unselected.
**ComfyUI** is SUPPORT, the selected external graph/UI/API client. Implement it
after a working runtime; it must not own the memory lifetime.

## Minimal implementation order after approval

| Step | Useful deliverable and acceptance boundary | Dependencies |
| --- | --- | --- |
| 1 — Minimal frozen-Qwen technical runtime | Load/generate with bounded session context, stop/error paths, session isolation, basic resource measurement and frozen Qwen parameters. Agent / User / Another Person #N and `source_id` are present immediately. Check one candidate Memory Bridge connection/bypass boundary. This is technical preparation, not a dialog-quality or cognitive qualification. | Final roadmap approval; verify compatible existing local assets when available. No automatic downloads. A brief smoke check records Base limitations and a bounded proposed remedy; no endless dialog optimization. |
| 2 — Minimal memory/Bridge choice | Select the smallest applicable independent plastic memory and trainable Bridge for the first cognitive milestone. Briefly check reuse, license, dependencies, resources, separate state ownership, `source_id` binding and intervention compatibility. | Technical runtime/connection findings and [Collab shortlist](docs/COLLAB_CANDIDATES.md), not a broad dialog-quality PASS. Algorithms remain open until this decision; no novelty target or open-ended search. |
| 3 — First main cognitive milestone: Neural Memory + trainable Memory Bridge | New source-bound knowledge changes memory parameters. Frozen Qwen uses it through the Bridge after prior transcript/KV context is cleared and after process restart. Matched MEMORY OFF/ON shows useful contribution. Save/restore and correct `source_id` binding belong here from the start. | Selected mechanism/Bridge and scoped future implementation after final approval. This documentation update starts no training. Qwen remains frozen; independent memory and Bridge remain replaceable behind checked compatibility. |
| 4 — Reliable updates and replacement | Corrections, richer participant/source reasoning, temporal handling, retention after interference, unknown/conflicting input, disable/re-enable and a compatible Bridge replacement. Preserve memory; separately test behavioral compatibility. Minimal participant IDs and source-bound memory are already required in steps 1 and 3. | One working acquired capability; address actual defects rather than start another research programme. |
| 5 — External ComfyUI integration | Minimal API/custom node addresses the same runtime/state; reconnecting or restarting UI preserves experience. | Working runtime and persistence. Reuse ComfyUI API; do not rebuild cognitive state inside UI. |
| 6 — Further justified abilities | Feedback, curiosity, consolidation, vision, planning, affect or motivation only for a necessary use case existing components cannot meet. Reclassify, merge or remove as evidence requires. | Real behavior gap and bounded OFF/ON scenario. No automatic M-number sequence. |

Steps 1–2 are minimal preparation for the first main cognitive milestone in
step 3, not separate large qualification programmes. Use short technical decisions
and Collab. Optional UI, vision and secondary functions must not delay independent
memory that usefully changes a frozen model's behavior and survives restart.
Neither the old M0–M13 structure nor M1 scientific qualification is a reason to
create a module or add gates.

## First technical result: runtime and Bridge connection feasibility

After final approval, build the smallest direct text harness around frozen
Qwen3.5-2B-Base. It needs explicit sessions, Agent / User / Another Person #N,
`source_id` transport and a candidate point through which a future Memory Bridge
can influence model computation. This is proposed work, not code supplied here.

A brief technical smoke check is enough:

1. Load Qwen and process a few actual messages with bounded context, explicit
   participant identities and unchanged Qwen parameters; check stops/errors,
   session isolation and basic latency/RAM/VRAM/disk cost.
2. Check one candidate Bridge boundary for compatible input/output, device/dtype
   and state ownership. Show that a compatible test signal can be attached,
   bypassed and detached without corrupting runtime/state. Record feasibility
   and any technical blocker; a test signal proves wiring, not a learned Bridge
   or useful memory. It does not select the memory algorithm.
3. Verify `source_id` passes with the message/event to the future Bridge/memory
   boundary. Keep a stable mapping: Agent, User and distinct Another Person #N
   identities must not collapse into one source. Preserve that mapping through
   relevant session, save and restore boundaries when persistence is added.

Participant identity and provenance transport are ordinary support logic from the
first runtime, not a new neural participant module. Carry the source binding into
memory writes, persisted experience and later readout; do not confuse a person
mentioned in text with its actual source. Acquired knowledge must retain the
correct `source_id` from the first cognitive milestone. Complex participant
modeling, inferred identities and advanced source reasoning can come later.

This stage does not require a broad dialog benchmark, perfect conversational
behavior or a separate context OFF/ON quality gate before memory work. Base may
be unsuitable for direct dialog; its official
[model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base) records its pretrained
status. If the brief check exposes poor dialog behavior, record a concrete
example and propose the smallest bounded solution for owner review. Stop that
line of optimization rather than repeatedly tuning the runtime. Technical
readiness and dialog limitations are reported separately; they cannot be passed
off as a successful cognitive milestone. Do not silently switch model, fine-tune,
download another model or restart M1.

## First main cognitive milestone: Neural Memory + trainable Memory Bridge

Demonstrate one useful acquired capability with independent plastic Neural
Memory and a replaceable trainable Memory Bridge, while Qwen remains frozen:

1. Supply previously unknown knowledge from an identified Agent / User /
   Another Person #N source. Acquisition must change the separate memory's
   parameters and preserve the correct `source_id`; changing only a transcript,
   metadata or a cache is insufficient.
2. End the teaching conversation and clear its transcript, prompt history and
   KV/session context. Save the independent memory and Bridge state, terminate
   the process, then restore into a fresh runtime without replaying the teaching
   text. Preserve participant/source identity mappings.
3. Ask a related question in a new context. Frozen Qwen must use the acquired
   knowledge through the trainable Bridge and retain the correct source binding.
   An in-process check before restart can diagnose faults but cannot replace
   this post-restart behavior.
4. Compare matched **MEMORY OFF / MEMORY ON** on the same later query, with
   isolated copies of restored state and identical model/generation settings.
   ON must use the acquired knowledge; OFF must lose or degrade that ability
   without breaking the rest of the runtime. Account for knowledge Qwen already
   has or can infer. Changed wording alone is not a benefit.

Use a small relevant set, including source-specific knowledge, a correction and
unrelated behavior for regressions. A useful result needs no M1-style scientific
qualification. Memory and Bridge algorithms remain open; Collab reuse is the
default before custom implementation.

Bridge replacement preserves the independent memory artifact/state. That proves
storage preservation only: separately check read/write compatibility and later
behavior. Never silently reset/retrain memory or promise universal compatibility
between unrelated memory architectures.

Retain working checkpoints, code/configs, compact relevant results and artifacts
needed to recover important results. Estimate disk cost and cleanup scope before
computation. Follow [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md).

**Current stop point:** Stage 1 delivers the technical runtime and short results;
present [Collab memory/Bridge options](docs/NEURAL_MEMORY_BRIDGE_OPTIONS.md) for
the first main cognitive milestone. PR #1 is already merged with owner approval.
Do not infer permission for training, new large-model downloads or implementing
the unselected Neural Memory/learned Bridge from Stage 1 authorization.
