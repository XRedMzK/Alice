# Alice functional roadmap — proposal

25 October 2026. **PROPOSED — awaiting owner approval.**
[Engineering First Policy v2](docs/ENGINEERING_FIRST_POLICY_V2_RU.md) is approved;
this implementation order and the function dispositions below are for review.
This PR performs no training, model download, runtime or memory implementation.

Progress answers: **What useful task can Alice now perform that it could not
perform before?** Cognitive additions need relevant real-runtime MODULE OFF /
MODULE ON evidence and acceptable regressions, not merely changed output.

## Starting point

Foundation CPU and target-GPU acceptance remain historical PASS in their original
scope. Preserve source, interfaces, schemas, tests and evidence. The public tree
contains Foundation, not a Qwen dialog runtime, plastic neural memory, Memory
Bridge or ComfyUI integration. No new milestone below is complete. Synthetic
recurrent state is not acquired episodic memory.

M1 is RESEARCH ARCHIVED / NOT_FROZEN in its
[separate project](https://github.com/XRedMzK/M1-Event-Understanding). Its
qualification is not an Alice prerequisite and is not resumed here. Memory and
Bridge algorithms, intervention site, tensor contracts, capacity, learning
objective and update protocol remain open.

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
| M2 — Source / SELF / PARTNER | CORE, merge with context and memory | Qwen may use visible sources; lasting attribution is unproven. Reuse sender/event metadata and memory/Bridge bindings. No separate source network justified yet; labels alone do not supply the ability. | Same proposition from different sources, later attribution and correction of only one source, including new session/restore. Omission confuses reported information, observations and Alice's conclusions. |
| M3 — Temporal conversational state | CORE, merge with context and memory | Reuse message ordering and Qwen context first. Cross-session order/currentness belongs in memory/Bridge behavior. No separate temporal network selected. | Reverse events, correct an obsolete fact and ask about current versus earlier state; later repeat across sessions. Omission makes currentness and correction unreliable. |
| M4 — Context integration | REMOVE/MERGE into runtime and Bridge | Qwen handles visible context. A trainable Bridge must integrate independent memory with its computation. Reuse compatible inputs/conditioning primitives; no extra fusion model by default. | Same question with different histories; memory ON improves the relevant answer without irrelevant interference. A separate M4 may disappear; integration cannot, or memory remains unused. |
| M5 — Natural-language production | SUPPORT, use Qwen | Frozen replaceable Qwen3.5-2B-Base is the approved response model. Reuse Transformers generation, stops and bounded context; no second language network. Base dialog suitability must be tested. | Sequential relevant responses using available information and uncertainty; Qwen parameters unchanged. Without production Alice cannot demonstrate abilities through conversation. |
| M6 — Epistemics / curiosity | Minimal epistemics CORE; active curiosity LATER | Test Qwen with source/currentness and memory first. Reuse adequate clarification/uncertainty behavior; only fill demonstrated gaps. Merge basic handling with memory/Bridge; no separate network yet. | Unknown term, explanation and conflicting correction cause appropriate clarification/qualification and later use. Omission allows guesses to masquerade as acquired knowledge; autonomous curiosity can wait. |
| M7 — Partner evaluation | LATER | Qwen may respond to feedback in current context. Later reuse explicit feedback and memory/preference updates if useful; no separate learner now. | LIKE/DISLIKE affects an agreed preference or salience channel, not factual truth; NONE invents no signal. Omission does not block first dialog or memory. |
| M8 — Runtime plasticity | CORE, part of Neural Memory | Qwen stays frozen. Foundation state handling is not experience acquisition. Reuse a suitable neural associative/fast-weights mechanism if applicable. Independent mutable memory parameters are required, custom invention is not. Merge with M9. | New information/correction affects a related later question after the original session ends; memory OFF loses or degrades the ability. Omission removes independent acquisition, a central Alice requirement. |
| M9 — Episodic memory / consolidation | Memory CORE; elaborate consolidation LATER | Combine acquisition, readout and retention in independent plastic Neural Memory. Pretrained knowledge, transcript retrieval and KV cache are insufficient. Reuse minimal neural-memory primitives after applicability checks; no consolidation network by default. | Several new episodes with needed source/time binding, later relevant use after interference and OFF/ON. Omission removes acquired experience; elaborate consolidation need not block small useful memory. |
| M10 — Persistence | SUPPORT, early and essential | Reuse Foundation shards, integrity/schema checks and checkpoint primitives; no neural code needed. Save memory and Bridge separately with explicit versions/compatibility. | Save, terminate, fresh restore and use learned information without old transcript; corruption cannot expose partial state. Omission loses experience on restart. |
| M11 — Vision | LATER | Initial scope is text. Later assess an existing visual encoder and minimal memory interface; preserve modality/source and adopt only needed dependencies. No custom vision network assumed. | New image changes later conversation and restored experience; modified/removed image changes the target result. Omission does not block text; old full-V1 vision acceptance cannot be claimed. |
| M12 — Integrated Alice | CORE, continuous integration | One runtime composes abilities. Reuse Foundation/model APIs; this is an outcome, not another network. Integrate from the first dialog baseline. | One continuous conversation acquires, corrects, changes context, uses experience later and restores without operator transfer. Omission leaves isolated demonstrations. |
| M13 — Human acceptance / DAY0 | SUPPORT, owner milestone | Owner review of useful behavior, not a model. Remove from the queue of networks. Runtime acceptance and DAY0 remain separate explicit owner actions. | Owner evaluates actual scenarios and limitations. Roadmap approval, synthetic PASS and Foundation acceptance imply neither runtime acceptance nor DAY0. |

The **Memory Bridge** is CORE: independent memory must influence Qwen computation
and answers through a replaceable trainable interface. It absorbs integration
work formerly spread across functions; its exact algorithm remains unselected.
**ComfyUI** is SUPPORT, the selected external graph/UI/API client. Implement it
after a working runtime; it must not own the memory lifetime.

## Minimal implementation order after approval

| Step | Useful deliverable and acceptance boundary | Dependencies |
| --- | --- | --- |
| 1 — Frozen-Qwen dialog baseline | Real sequential messages, bounded session context, context OFF/ON, session isolation, stop/error paths, measured resources and unchanged model parameters. No long-term memory claim. | Roadmap approval; verify compatible existing local assets when available. No automatic model downloads. |
| 2 — Minimal memory/Bridge choice | Select the smallest applicable read/write mechanism and trainable interface for one behavior. Briefly check existing solutions, license, dependencies, resources, state ownership and intervention compatibility. | Observed baseline and [Collab shortlist](docs/COLLAB_CANDIDATES.md). Algorithms remain open until this decision. No novelty target or open-ended architecture search. |
| 3 — One persistent acquired capability | A new fact/definition updates independent neural-memory parameters, is used in a new conversation after original context is cleared, survives process restart and improves memory OFF/ON. Save/restore belongs here from the start. | Selected mechanism/Bridge; scoped implementation and necessary memory/Bridge training only after approval. Qwen stays frozen. |
| 4 — Reliable updates and replacement | Corrections, source/time handling, retention after interference, unknown/conflicting input, disable/re-enable and a compatible Bridge replacement. Preserve memory; separately test behavioral compatibility. | One working acquired capability; address actual defects rather than start another research programme. |
| 5 — External ComfyUI integration | Minimal API/custom node addresses the same runtime/state; reconnecting or restarting UI preserves experience. | Working runtime and persistence. Reuse ComfyUI API; do not rebuild cognitive state inside UI. |
| 6 — Further justified abilities | Feedback, curiosity, consolidation, vision, planning, affect or motivation only for a necessary use case existing components cannot meet. Reclassify, merge or remove as evidence requires. | Real behavior gap and bounded OFF/ON scenario. No automatic M-number sequence. |

Steps 2–4 need short technical decisions, not a qualification ladder. Optional UI,
vision and secondary functions must not delay the central difficulty: independent
memory that usefully changes a frozen model's behavior and survives restart.

## Nearest practically verifiable result: usable dialog baseline

After approval, build the smallest direct text harness around frozen
Qwen3.5-2B-Base with explicit session boundaries and a future function enable/
disable point. This is proposed work, not code supplied by this PR. Base is
pretrained, not automatically suitable for direct dialog, as its official
[model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base) explains. Test suitability
early; do not silently switch to Instruct or fine-tune.

Use a small set of actual RU/EN conversations:

1. Explain a new term or state an object's location; ask a related next-turn
   question. Compare context OFF/ON on the same final question.
2. Correct the information, switch topic, then ask for the current fact and
   source. ON should use the correction where relevant, not merely change the
   answer. Include an unrelated question to detect interference.
3. End the session and create another; verify isolation. At this baseline,
   losing session-only information is expected, not a neural-memory defect or
   successful persistence demonstration.
4. Exercise stop/error behavior and the future function boundary without
   corrupting another session. Record latency, RAM/VRAM/disk and confirm no Qwen
   optimizer/update path. A stub does not count as a cognitive module.

Keep model version, generation settings and relevant starting state matched.
Use matched seeds or a few repeats only if decoding variation obscures the
practical conclusion. PASS means useful context-sensitive dialog without
unacceptable regressions in these scenarios. Record concrete outcomes and
limitations rather than an arbitrary perfect F1 threshold.

If Base cannot support useful dialog under these constraints, report the
reproduced limitation and smallest proposed remedy for the owner. Generation
alone is not milestone completion. Do not automatically train, download another
model or restart M1.

## First result demonstrating independent Alice memory

Previously unknown user fact/definition → update separate neural-memory
parameters → end the original conversation and discard its transcript/KV context
→ use in a new conversation with a related new phrasing → save and terminate
→ fresh restore and use again → matched memory OFF/ON. Include a correction,
one interfering episode and unrelated behavior. The advantage must come from
memory; replaying the original text into Qwen is not this result. Pretrained
knowledge must not already answer the new fact.

Bridge replacement preserves the independent memory artifact/state. That proves
storage preservation only: separately check read/write compatibility and later
behavior. Never silently reset/retrain memory or promise universal compatibility
between unrelated memory architectures.

Retain working checkpoints, code/configs, compact relevant results and artifacts
needed to recover important results. Estimate disk cost and cleanup scope before
computation. Follow [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md).

**Current stop point:** submit this proposal for owner review. Roadmap approval
is the next decision; no implementation or training begins in this PR.
