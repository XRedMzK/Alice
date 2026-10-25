# Alice architecture direction

Status: owner-approved Engineering First v2 principles, 25 October 2026. Implemented Foundation and
unimplemented cognitive mechanisms are explicitly separated below. The original
Foundation contract is preserved; this document does not retroactively rewrite
its accepted tests or prove biological equivalence, subjectivity or consciousness.

## Purpose and ownership

Alice is an engineering project for a functioning modular cognitive system.
Research on the human brain and existing technical solutions informs design.
The central goal is its own changeable cognitive
state and acquired experience, independent of a particular language backbone.
Progress means a useful capability in Alice's real behavior. A tensor, successful
training run or isolated classifier is insufficient. New cognitive functions
need matched MODULE OFF / MODULE ON behavior checks and relevant regression
checks under the [active engineering rules](ENGINEERING_MODULE_CONTRACT.md).
See [policy scope](docs/POLICY_SCOPE.md) for the historical contract boundary.

## Approved principles

- **Language and response model:** Qwen3.5-2B-Base, with frozen weights, processes
  language and forms responses. It remains replaceable. Runtime learning of
  Qwen's parameters is not planned. A model name does not implement this runtime.
- **Neural memory, planned:** a separate mechanism with changeable neural
  parameters is intended to acquire/retrieve experience, persist between runs,
  and support later cognitive expansion independently of a Qwen version.
  These properties have not been implemented or qualified here.
- **Memory Bridge, planned:** a replaceable trainable interface should form
  queries to memory, receive neural representations, influence Qwen computation,
  and transmit experience to the memory-learning mechanism. Its exact architecture
  and training/update protocol have **not been selected**.
- **Collab:** development through functional composition of existing solutions.
  A needed function may be reused without adopting a whole model. Candidate
  mechanisms are checked for practical applicability, dependencies, licensing,
  resources and interface compatibility. Reuse Qwen or an existing mechanism
  before making another model. Separate functions may merge or disappear;
  neither scientific novelty nor the historical module count determines design.
- **ComfyUI:** selected as the future external visual interaction and graph-control
  layer, using custom nodes and an API. A minimal direct text harness is proposed
  first so UI work does not delay real model interaction. Telegram and Discord
  remain later clients. There is no ComfyUI/custom-node/client integration here.
- **State lifecycle:** Alice's cognitive state must remain independent of the
  ComfyUI/UI process lifecycle. Restarting an interface must not define a reset
  of experience. No such persistent cognitive state is implemented yet.
- **Expansion and migration:** emotions, motivation, planning and other functions
  are possible additional mechanisms. Adding them should not automatically reset
  memory. Replacing the memory architecture requires a separately verified
  migration; universal compatibility of weights or preserved experience is not
  promised.

The intended functional relationship is Qwen ↔ Memory Bridge ↔ neural memory,
with ComfyUI outside the cognitive-state lifetime. This is a conceptual boundary,
not an executable graph, selected tensor schema or implemented memory algorithm.

Within a session, ordinary message ordering, explicit sender IDs and bounded
context transport are support functions. They may reuse deterministic code and
Qwen. They do not demonstrate acquisition into independent plastic neural memory.
The resource scheduler remains semantics-blind; knowledge use and cognitive
behavior must be demonstrated in the model/runtime rather than asserted by a
router's label. No separate network is required for each historical function.

## What exists and what evidence covers

| Area | Implementation | Evidence and limit |
| --- | --- | --- |
| Foundation CPU | `src/nmcm`: contracts, registry/graph, owned recurrent state, shards/whole checkpoints, behavioral compatibility, telemetry | Historical synthetic CPU-01…15 passed; current basic check covers schemas and routing only. |
| Foundation GPU control plane | `src/nmcm_gpu`: admission, monitoring, placement, independent coordination and qualification support | Historical target-GPU G-01…15 passed. Host-bound profile and original run/checkpoint inputs are excluded from this clean source release. No GPU requalification or portable GPU execution is claimed. |
| Synthetic A1/A2/A3/B/C | Small test neural modules and original test protocol | Demonstrate bounded module replaceability and controls; are not a language model, Memory Bridge or memory of lived experience. |
| Frozen Qwen language/response integration | Planned | External donor identified; no weights or Alice Qwen runtime shipped. |
| Plastic neural memory / Memory Bridge | Planned; exact mechanisms open | No implementation, training or cognitive qualification. |
| ComfyUI and future clients | Selected integration direction; implementation planned | No custom nodes, client adapters or lifecycle persistence integration. |

## Foundation compatibility boundary

The existing frozen [MODULE_CONTRACT.md](MODULE_CONTRACT.md), machine-readable schemas,
manifests, fixed synthetic graphs, source, thresholds and original tests are copied
without semantic changes. Structural modularity alone is not functional
specialization. An interface match alone does not establish behavioral compatibility.
The orchestration layer remains semantics-blind; cognitive state belongs to its
module, not the scheduler.

Foundation's current neural API uses contiguous float32 sequence tensors
`[batch, time, feature]` and owned `[batch, feature]` private state. This is not
proof of arbitrary Qwen/Bridge compatibility. Start with existing interfaces;
introduce an adapter or versioned successor only for a demonstrated need. Do not
redesign Foundation in anticipation of an unselected memory algorithm.

The [original architecture baseline](docs/FOUNDATION_ARCHITECTURE_V0.md), historical
ADRs and acceptance reports retain their dates and scope. Their references to old
work-package authority, not-yet-run operations or future models describe the
historical package. They are not orders to resume M1 or implement a new mechanism.
Any future contract change needs explicit versioning and affected evidence.

## Open engineering choices and next review

No memory rule, plasticity mechanism, capacity, consolidation scheme, Bridge
topology, Qwen intervention site, memory tensor schema, training objective or
new module decomposition is selected by this publication. Foundation's synthetic
state/checkpoint utilities are reusable engineering infrastructure; they are not
proof that the planned neural memory already works. [ROADMAP.md](ROADMAP.md)
proposes a frozen-Qwen dialog baseline first, then one persistent memory
capability with a trainable Bridge. The owner must approve that sequence before
implementation. No model loading, downloads, training or new memory work is
performed by this policy PR.
