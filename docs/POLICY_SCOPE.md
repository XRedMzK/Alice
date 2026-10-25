# Active rules and preserved historical contracts

Effective decision: 25 October 2026,
[Engineering First Policy v2](ENGINEERING_FIRST_POLICY_V2_RU.md).
The policy is approved by the owner; the replacement roadmap is a proposal for
owner review. This documentation change implements no cognitive mechanism.

| Document | Scope after this decision |
| --- | --- |
| [CODEX.md](../CODEX.md) | Active agent working policy for new Alice work. |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Approved principles and actual implementation boundaries. |
| [ROADMAP.md](../ROADMAP.md) | Proposed functional priorities and next observable result; pending owner approval. |
| [ENGINEERING_MODULE_CONTRACT.md](../ENGINEERING_MODULE_CONTRACT.md) | Active requirements for new functions, reuse and integration. |
| [ENGINEERING_VALIDATION.md](../ENGINEERING_VALIDATION.md) | Active practical behavior, regression, state and resource checks. |
| [MODULE_CONTRACT.md](../MODULE_CONTRACT.md) | Unchanged frozen Foundation v1alpha1 contract. Applies to Foundation conformance, not a requirement to build a neural network for each future function. |
| [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md) | Unchanged historical Foundation research protocol. New work does not inherit its qualification loop by default, including its old forward-applicability wording. |
| [RESOURCE_POLICY.md](../RESOURCE_POLICY.md) | Preserved Foundation resource baseline. Existing hard resource limits remain; historical run authorization and evidence requirements do not authorize a new run. |
| [FOUNDATION_ARCHITECTURE_V0.md](FOUNDATION_ARCHITECTURE_V0.md), Foundation ADRs, plans and acceptance reports | Historical baseline and accepted evidence, with original scope, statuses, numerical gates and hashes. |
| [M1 history](M1_HISTORY.md) | Archived research in a separate project; no compulsory dependency, resumed training or retroactive PASS. |

Historical references to `temp.md`, scientific closure, module-owned trainable
parameters, frozen seeds or local qualification remain meaningful within their
original accepted packages. They do not override Engineering First v2 for new
development. A future work package may use a specific research method when a
concrete engineering uncertainty warrants it; it must name that scope rather
than import the whole M1/Foundation programme.

Preserve the existing Foundation interfaces when using Foundation. This policy
does not make an incompatible component conform to v1alpha1. Use an explicit
adapter or a separately versioned contract only when a demonstrated integration
need requires one. Neither a new manifest schema nor Foundation redesign is part
of this decision.

The frozen files, implementation, schemas, tests, original reports and
`SOURCE_PROVENANCE.json` remain byte-for-byte unchanged. The provenance manifest
records the initial source publication; new policy/roadmap documents are new
authored work tracked by this PR's Git history.
