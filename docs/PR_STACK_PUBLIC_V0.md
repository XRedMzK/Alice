# Public v0 review and merge order

The source release is prepared as a dependency stack. No merge is automatic.
The owner reviews each layer against its stated base; acceptance of local
behavior is not acceptance of a public PR. Historical prototype and checkpoint
references remain separate from publication branches.

| Order | Review unit | Base | Scope |
| --- | --- | --- | --- |
| 1 | Existing [Stage 1 PR #2](https://github.com/XRedMzK/Alice/pull/2) | Public `main` after PR #1 | Local frozen Qwen runtime, participant/source IDs, sessions, typed optional Bridge port and technical smoke checks. |
| 2 | Neural Memory runtime integration | Exact Stage 1 head `626b6f81cabc732cea1b193e964548fc4a61d9dc` | Accepted independent full-rank M/Bridge path, persistent runtime ownership, source isolation, integrity/recovery, portable source packaging and asset-free tests. |
| 3 | Dialogue Acquisition and public reproduction | Exact reviewed head of memory integration | Bounded explicit RU/EN object-label intents, session/source/content-bound confirmation, answer-blind RU query adaptation, install guide and opt-in real-model reproduction. |

The memory layer is prepared from the existing source-only memory publication
commit `ee62eb6`; the final dialogue source baseline is `b499b19`. The combined
public stabilization preserves that baseline and adds narrowly scoped publication
hardening. Exact branch heads, diff counts and CI evidence must be recorded in
the review handoff; this document is not a CI result or a merge approval.

Review Stage 1 independently first. If its reviewed source changes, update and
recheck the dependent layers before merging them. Merge in order 1 → 2 → 3 and
retarget the next PR only after its dependency is merged. Preserve the original
accepted references and prepare continuation commits rather than rewriting them.

Each layer contains production source, necessary tests and portable documentation.
Model/Bridge weights, acquired M, private conversations, machine-specific paths,
experiment corpora, calibration tools, temporary tensors and run artifacts are
excluded. The accepted external Bridge remains an explicit installation
dependency; the code release does not silently train or synthesize a replacement.

Asset-free CPU checks cover routing, runtime contracts, confirmation ownership,
source isolation, corrected state, integrity and interrupted publication. The
opt-in [public smoke](../tools/public_memory_smoke.py) separately exercises the
installed CLI with local Qwen and Bridge assets. A CPU test count is not a claim
of real-model neural recall. See [installation and limits](PUBLIC_V0_INSTALL.md)
for the exact external-asset contract and reproducible command.
