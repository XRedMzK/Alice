# Public v0 validation and limits

## Publication reproduction and PRE-DAY0 observation

The public GitHub source at `880c59c4d7d4996e2203d8c4aa1c1ba381899547` was
freshly cloned and installed normally into a new virtual environment. It ran
outside checkout with `PYTHONPATH` removed; Alice imports came from the new
installation. Compatible dependencies and Qwen assets were reused locally;
independent network dependency resolution was not claimed.

The existing sealed Bridge was reconstructed from the public utility, fixed
constructor and recorded scalar, without the old checkpoint or fitting. Exact
state identity was verified on the recorded native CPU profile. The resulting
Bridge supported the same five-process installed real-Qwen check in **46.69 s**:
confirmed RU/EN acquisition, restart, source/correction/retention, OFF/EMPTY/ON,
same-Bridge M swap and unrelated controls. This closes the concrete local
reproduction route from public source plus supplied Qwen. Different CPU/MKL
dispatch may fail strict reconstruction; no universal external-asset portability
or hosted checkpoint fallback is claimed. See [the recipe](PAYLOAD_BRIDGE_RECONSTRUCTION.md).

Two short additional sessions found Base role continuation/repetition without
memory and loss of the preceding call-sign context in memory mode. They are
negative product observations, not training data or a scientific benchmark.
[PRE-DAY0 GAP REPORT](PRE_DAY0_V0_GAP_REPORT_RU.md) records the concrete remaining
issues and **NOT READY** decision. No new architecture, Qwen/Bridge fitting or
canonical instance was introduced. The evidence below retains the earlier
prepublication checks with their own source identities and timings.

## Earlier prepared-source evidence

On 26 October 2026, the source at
`bb07f40d83f38a45ac4be87847a63766e91fdff4` was exported from Git, installed as an
ordinary wheel into a fresh virtual environment and exercised outside the
checkout with `PYTHONPATH` removed. Compatible dependencies were reused from an
existing local environment without a download. Imports resolved to the new
installation; `pip check` and both module/console help succeeded. This checks
packaging and behavior on that environment, not a fresh network resolution of
all dependencies. The subsequent documentation-only commit preserves the tested
production source, packaging, requirements and CI inputs byte for byte.

## Asset-free CPU checks

The exact CPU profile uses CPython 3.12, Torch `2.13.0+cpu` and the dependencies in
[requirements-cpu.lock](../requirements-cpu.lock).

| Scope | Result | Meaning |
| --- | --- | --- |
| Existing Stage 1 head `626b6f8`, original workflow scope | 11 PASS, 0.29 s | Sessions, source IDs, Bridge tensor boundary and two Foundation basic checks. |
| Memory review head `21cb07e`, its workflow scope | 179 PASS, 27.29 s | Parameter/source isolation, corrected writes, restore, integrity, durable publication and runtime integration. |
| Dialogue public source, final workflow scope | 266 PASS, 30.41 s | Memory checks plus bounded intents, exact confirmation ownership and CLI transport. |
| Complete source CPU suite on equivalent production source | 279 PASS, 54.87 s | The workflow scope plus additional historical Foundation fixtures. Some unchanged Foundation fixtures fit tiny synthetic modules. |

The tests cover failed writes, indeterminate commits, file/directory sync errors,
real process termination at publication boundaries, corrupt/unknown-state
rejection, stale confirmation races and isolation across sessions/sources. A
failed save never receives a saved acknowledgement. Pre-commit failure preserves
the prior committed snapshot. An indeterminate publication closes the memory
session and requires restore; it does not roll RAM back behind a possibly
committed manifest. These Linux/POSIX fault and process-death checks do not
establish physical power-loss durability on every filesystem or storage device.

CPU tests use no Qwen or accepted Bridge assets. Their count does not establish
neural recall. No Qwen or Bridge fitting was run for this release.

## Installed real-model check

The separate opt-in [public checker](../tools/public_memory_smoke.py) ran the
installed CLI in five fresh processes with local frozen Qwen3.5-2B-Base and the
accepted sealed payload Bridge. It used CUDA/BF16 with the tested runtime profile
in [the installation guide](PUBLIC_V0_INSTALL.md). Model and Bridge weights are
external dependencies and are not distributed with this source. An arbitrary
newly initialized Bridge does not reproduce this result.

The bounded check completed **PASS in 43.18 seconds**:

- Ordinary RU/EN messages proposed `window` and `museum` for source `user`, and
  `helmet` for `person:1`. Each write required confirmation; unconfirmed proposals
  left memory unchanged.
- A new process recalled those words from independently restored M through model
  generation. No teaching transcript or prior KV cache was restored, and the
  saved session histories contained no teaching/confirmation messages.
- OFF and EMPTY did not apply memory or return the acquired value. ON recovered
  it again on supported RU/EN question forms.
- A confirmed correction changed `window` to `dragon`. A later process recalled
  `dragon`, retained `museum`, and retained `helmet` for the other source. Only
  the corrected source plane changed.
- Restoring the original M with the same unchanged Bridge returned `window`
  again, while the separate corrected state stayed intact. This supports causal
  influence of acquired memory content.
- Unrelated arithmetic remained `4` with memory OFF and ON; `Paris` and `7` also
  remained correct without a memory intervention.

Qwen and Bridge were fixed during acquisition. Receipts reported only persistent
M updates, Bridge files remained unchanged, and read-only recall did not change
M. The checker used no structured `/remember` command, calibration or training.
Expected answers stayed in the external checker; runtime query input contained
no answer oracle. This is a small functional check rather than a generalization
benchmark or a new scientific qualification.

Across the five processes, observed peak RSS was **4760.65 MiB**; Torch peak
allocated/reserved VRAM was **3641.36 / 3660 MiB**. Torch figures exclude the
driver and other processes. These local observations are not resource guarantees.
Raw evidence, acquired M and Bridge copies remain outside the source tree.

## Demonstrated boundary

The current interface accepts a small explicit RU/EN object-label grammar,
literal object IDs and one-model-token values. Queries support the documented
literal forms; arbitrary paraphrases, other relations through free dialogue,
multi-token values, pronoun resolution and unlimited capacity are not established.
Memory answers use the current question only, including OFF/EMPTY; ordinary
Stage 1 context behavior remains available when memory is absent. The payload
codec and Bridge are bound to the accepted donor geometry. Arbitrary Qwen or
Bridge replacement needs a separate compatibility check and never authorizes
resetting acquired memory. ComfyUI is not required and does not own M.

Source and model evidence are separate from owner review and merge approval.
The [PR stack](PR_STACK_PUBLIC_V0.md) defines the required review order; no merge
or remote CI result for an unpublished continuation is implied here.
