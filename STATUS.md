# Alice Public v0 status

26 October 2026. **Public source integration is not Day0 acceptance.** No
canonical working instance is appointed by this release.

| Layer | Source/publication identity | Implemented boundary |
| --- | --- | --- |
| Stage 1 | Approved `626b6f8`, [PR #2](https://github.com/XRedMzK/Alice/pull/2), merge `74ae50a` | Local frozen Qwen generation, identities/sessions/errors and typed optional embedding connection. |
| Neural Memory | Approved `21cb07e`, test-only CI continuation `b3c982f`, [PR #3](https://github.com/XRedMzK/Alice/pull/3), merge `a5e9d46` | Independent source-bound M, separate sealed Bridge, correction/restore, synced publication and failure controls. |
| Dialogue | Approved `3abcc5d`, [PR #4](https://github.com/XRedMzK/Alice/pull/4), public reproduction source `880c59c` | Explicit bounded RU/EN object-label requests, exact confirmation ownership, query adapter, portable install and existing-Bridge reconstruction. |

Accepted histories are preserved. `b3c982f` corrects a numerical test against a
roundoff-bounded FP64 oracle; it changes no production tensor, architecture or
checkpoint. The Bridge reconstruction tool uses fixed seeds and an already
recorded scalar, not a new fit. No new Qwen/Bridge training was run for publication.
Foundation/M1 source, contracts and historical evidence remain unchanged.

The public source was freshly fetched, normally installed and run outside the
checkout with `PYTHONPATH` removed. Existing compatible dependencies and local
Qwen were reused. A newly reconstructed Bridge, without the former checkpoint,
supported five real-Qwen processes in 46.69 s: confirmed acquisition, restart,
source/correction/retention, OFF/EMPTY/ON, same-Bridge memory swap and unrelated
controls. The recorded CPU/MKL identity limitation remains an external-asset
boundary; green asset-free CI is not a universal reproduction claim.

Two additional short sessions expose concrete ordinary-dialogue and context
limitations. [PRE-DAY0 v0 GAP REPORT](docs/PRE_DAY0_V0_GAP_REPORT_RU.md): **NOT READY**.
State export protection, normal turn boundaries, current context with M,
recovery/retention and launch/quit usability precede further cognitive work.
Long-duration stability, arbitrary language, unlimited retention and arbitrary
donor replacement are unproven. The current prototype must not be the sole copy
of valuable acquired state while the protected-export gap remains unresolved.

[Install and exact assets](docs/PUBLIC_V0_INSTALL.md) ·
[Bridge recipe](docs/PAYLOAD_BRIDGE_RECONSTRUCTION.md) ·
[Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md).
Day0 and the canonical instance require a separate owner decision; close Alice
V1 before cortical Alice v2. No new module, architecture or training starts here.
