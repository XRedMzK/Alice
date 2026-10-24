# ADR-0001: NMCM v2 foundation boundary

- Status: **ACCEPTED FOR THE BOUNDED CPU WORK PACKAGE**
- Date: 2026-08-12
- Decision owner: user/project owner
- Implementation authorized: **yes, CPU-only; GPU requires a separate decision**

## Context

NMCM v1 produced useful scientific and operational lessons but also exposed
high wall-clock from monolithic qualification, hidden process/state coupling,
serial execution, and binding between evaluators and artifact paths. NMCM v2
must establish replaceable neural modules and local evidence before Alice work.

## Decision

Adopt [ARCHITECTURE.md](../ARCHITECTURE.md) invariants I-01–I-17,
[MODULE_CONTRACT.md](../MODULE_CONTRACT.md),
[RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md), and
[RESOURCE_POLICY.md](../RESOURCE_POLICY.md) as the `v0.1` review baseline.

Qualify it using only the bounded synthetic topology and CPU-01–CPU-15 plan in
[docs/FOUNDATION_0_IMPLEMENTATION_PLAN.md](../docs/FOUNDATION_0_IMPLEMENTATION_PLAN.md).
Keep orchestration semantics-blind, module state/shards independently owned,
interfaces dense/versioned, experiments local-first, and resources explicitly
admitted/measured. Treat NMCM v1 as read-only evidence, never the v2 foundation.

## Consequences

- The platform can be rejected or revised before expensive Alice work.
- Foundation implementation requires explicit architecture and runtime
  approvals.
- Compatibility is behavioral/statistical as well as structural.
- Some control-plane convenience features are deliberately forbidden when they
  would encode cognition.
- Synthetic PASS will not select a language model, instantiate Alice, or start
  DAY0.

## Approval record

Accepted by the user's 2026-08-12 CPU work-package instruction in `temp.md`.
This approval freezes the synthetic CPU design and its bounded training/tests;
it does not authorize CUDA qualification, Alice V1, DAY0, or model download.
