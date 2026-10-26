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

Prepared review branches:

- Stage 1 keeps the existing PR #2 branch
  `work/qwen-runtime-stage1-20261025` at
  `626b6f81cabc732cea1b193e964548fc4a61d9dc`, against `main`
  `641d40e6ac6cc7bbbe95df98bf5414aa63c6e6ff`.
- Memory uses `review/public-memory-v0-20261026` at
  `21cb07e9b34b5261aa062378a34b1e69dcd43e5e`, based on that exact Stage 1 head.
- Dialogue uses `review/public-dialogue-v0-20261026`, based on the exact Memory
  head above. Its merge of `b499b19ad028c74f1ae613d26986e96609444225` preserves
  the previously accepted source history. The review handoff pins its final
  commit; a moving branch name is not a reproducibility identity.

The two continuation branches are prepared locally for owner review; naming
these branches does not claim they have been pushed or opened as remote PRs.

The memory layer is prepared from the existing source-only memory publication
commit `ee62eb6`; the final dialogue source baseline is `b499b19`. The combined
public stabilization preserves that baseline and adds narrowly scoped publication
hardening. Exact branch heads, diff counts and CI evidence must be recorded in
the review handoff; this document is not a CI result or a merge approval.

Review Stage 1 independently first. If its reviewed source changes, update and
recheck the dependent layers before merging them. Merge in order 1 → 2 → 3 and
retarget the next PR only after its dependency is merged. Preserve the original
accepted references and prepare continuation commits rather than rewriting them.
Use merge commits to preserve the reviewed ancestry; do not force-push or rewrite
accepted references. If the owner instead chooses a squash merge, prepare the
dependent continuation against the actual merged result and repeat its affected
checks before retargeting or merging it. The previous base is not automatically
equivalent to a new squash commit. This choice requires an explicit owner decision.

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

[Observed public-v0 validation and its scope](PUBLIC_V0_VALIDATION.md) records
the separate asset-free and installed real-model checks.
