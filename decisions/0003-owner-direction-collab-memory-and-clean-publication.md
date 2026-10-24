# ADR-0003: owner direction and clean Alice publication

Status: **OWNER-APPROVED PRINCIPLES; IMPLEMENTATIONS PLANNED**.
Date: 2026-10-25. Authority: the owner's final M1 cleanup/cold-storage/publication instruction.

M1 research is archived, NOT_FROZEN, and continues separately on GitHub. It is no
longer mandatory active Alice work. Preserve its published provenance and results.

Alice adopts Collab as functional composition of existing solutions. The planned
language/response model is frozen, replaceable Qwen3.5-2B-Base. Independent plastic
neural memory and a replaceable trainable Memory Bridge are future mechanisms;
their architectures and training protocols remain unselected. ComfyUI is selected
as the future external visual UI/graph/API layer, with custom nodes and later
Telegram/Discord/CLI. Cognitive state must outlive UI sessions. Expansion must
avoid automatic memory reset; a new memory architecture requires verified
migration, not a promise of universal weight compatibility.

Publish a clean source tree and new history rather than the private workspace or
old Git history. Keep accepted Foundation source/contracts, tests and evidence
scope intact. Exclude foreign weights, datasets, caches, runs, private logs/chats,
host-bound GPU records and secrets. This decision authorizes no new cognitive
implementation, training, M2 or architecture experiment. See [ARCHITECTURE.md](../ARCHITECTURE.md).
