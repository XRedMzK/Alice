# Engineering contract for new Alice functions

Status: active rules under the owner's Engineering First v2 decision,
25 October 2026. The owner approved [ROADMAP.md](ROADMAP.md) at `9c29e08` and
authorized Stage 1. Memory/Bridge algorithm selection and training remain future
scoped decisions; Stage 1 supplies a technical connection, not learned memory.

This contract governs new functions. The unchanged
[MODULE_CONTRACT.md](MODULE_CONTRACT.md) remains the accepted Foundation
`v1alpha1` contract; this document neither changes its schema nor declares new
components compatible with it.

## Choose a function before a module

Record a short answer to each question:

1. What necessary ability does Alice gain?
2. Can Qwen or an existing mechanism already perform it?
3. What, if anything, requires a custom neural implementation?
4. Which existing engineering or neural solution can supply the missing part?
5. Should this remain independent, or be merged with another mechanism?
6. Which real-runtime MODULE OFF/ON scenarios demonstrate useful influence?
7. What ability or reliability would be lost if the function were omitted?

Classify the function as **CORE**, **SUPPORT**, **LATER**, or **REMOVE/MERGE**.
Prioritize critical missing abilities. A historical module number is not a reason
to create a network. A reused component, deterministic helper, existing Qwen
ability or merged mechanism can satisfy a function. Novelty is not a selection
criterion. Check applicability, resource cost, licenses and interface compatibility
before custom development or additional dependencies.

## Define an engineering boundary

For an admitted implementation, keep a concise specification of:

- useful behavior, examples, non-goals and expected behavior with the function off;
- inputs/outputs, versions, consumers, error behavior and replacement boundary;
- state/parameter ownership, update permissions and independent lifecycle;
- save/restore and compatibility requirements wherever state exists;
- bounded VRAM/RAM/disk and runtime cost for its intended workload;
- affected tests and useful-behavior acceptance criteria.

The specification can live beside the implementation. A new schema, formal
experiment registry or scientific qualification is not mandatory. Existing
Foundation consumers still require their actual structural and behavioral
compatibility rules; matching shapes alone is insufficient. Do not redesign
unchanged Foundation to accommodate an unselected future mechanism.

Own trainable parameters are required for the approved independent plastic
Neural Memory and trainable Memory Bridge. Other custom neural mechanisms need
them when their chosen design requires learning. Support functions need no
separate trainable parameters, recurrent state or checkpoint unless their actual
behavior requires them. Any mutable
state that does exist must have a clear owner and safe update/restore behavior.

Neural Memory owns its changeable parameters and persisted experience independently
of Qwen and the UI. The Bridge is independently replaceable; changing it must not
silently reset memory. Compatibility or migration is verified separately, and
universal interchangeability is not promised. Qwen weights stay frozen; no silent
fine-tuning or LoRA. Memory and Bridge algorithms remain open choices.

## Accept behavior in the system

Use [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md). New cognitive modules
need useful influence on Alice in matched MODULE OFF/ON scenarios, including
regression checks. Tensor conformance, isolated classification, training loss and
unit tests support diagnosis but cannot substitute for that result. If the
function produces no useful benefit, revise, merge or remove it.

Ordinary support changes use direct correctness and affected integration checks.
They do not become cognitive experiments merely because they connect modules.
Preserve historical Foundation and M1 results exactly in their original scope.
