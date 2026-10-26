# PRE-DAY0 v0 candidate — NOT READY / draft review

Owner authority: 26 October 2026, from public `faa03d3` and the
[PRE-DAY0 gap report](PRE_DAY0_V0_GAP_REPORT_RU.md). This branch is not Day0
acceptance or a canonical instance. No architecture, donor or training change
is authorized. The complete goal requires all four blocks and actual frozen-Qwen
behavior; a passing security test alone does not complete it.

| Priority | Observed result |
| --- | --- |
| 1 — State protection | PASS in the stated Linux boundary: create-only export, pinned descriptors, no replacement, temporary-copy attack/failure tests. |
| 2 — One answer | Technical turn boundary works. Short RU/EN context/correction examples work, but Base still answers a topic change with an earlier greeting. Ordinary dialogue remains unreliable. |
| 3 — Context with M | NOT PASS. An explicit experimental profile retains bounded active history, including OFF/EMPTY. In real Qwen, history can override correct neural content and contaminate unrelated replies. |
| 4 — Recovery/resources | Explicit terminal restart state, paired backup/restore and bounded disposable history implemented. A 91-response resource run completed with the expected input refusal and unchanged persisted M/Bridge. Recovery evidence is recorded below. |

The complete goal is unfinished. No training, new model, memory expansion,
architecture change, Day0 acceptance or canonical instance was performed.
The published question-only memory profile remains the CLI/API default.
`--memory-context conversation` is opt-in diagnostic code with known failures;
it must not be presented as accepted context+memory integration.

Source checkpoint `2bc4551` passed the exact 409-test CPU profile in an isolated
checkout (37.88 s), including the 269 published tests. A normal offline,
noneditable installation outside the checkout, with `PYTHONPATH` removed,
passed the five-process published real-Qwen memory regression in 43.52 s.
Existing dependencies and the public-recipe Bridge were reused. The following
backup FIFO hardening adds two refusal tests; it does not alter the neural path.
Neither the CPU count nor the preserved memory regression makes P2/P3 pass.

## Transcript export contract

Start with a separate `--transcript-dir /path/to/transcripts`. Without it,
transcript export is disabled. `/save chat.json` accepts a new leaf JSON filename,
not an arbitrary path. Existing destinations are never replaced; choose another
name. `SessionStore.save(name, exporter=TranscriptExportDirectory(...))` uses the
same restriction; the former arbitrary-path API deliberately fails with a clear
configuration error. Memory/Bridge checkpoint schemas and acquired state do not
change because of this export hardening.

The exporter opens every directory component without following symlinks, pins
directory descriptors, checks actual directory ancestry and mount metadata,
writes an anonymous `O_TMPFILE`, syncs it and publishes a no-replace hard link
from its live FD. Existing files, symlinks and hardlinks are not opened for write,
replaced or deleted. There is no mutable temporary pathname cleanup or unsafe
fallback. Publication/directory-sync uncertainty is reported without a success
acknowledgement; a complete export may exist and must be inspected before retry.

This targets Linux local whole-filesystem mounts (`mountinfo.root=/`), `/proc`,
`O_TMPFILE` and directory fsync. Subtree/bind/subvolume destinations or unknown
metadata fail before writing. Codex's private filesystem sandbox uses subtree
self-bind mounts; positive checks run in the ordinary native namespace on
temporary copies. No privileged/same-UID process is prevented from independently
altering files outside these APIs. Actual `bind` alias tests use an unprivileged
private user/mount namespace and never modify host mounts.

See Linux [open(2)](https://man7.org/linux/man-pages/man2/open.2.html) and
[link(2)](https://man7.org/linux/man-pages/man2/link.2.html) for the descriptor and
anonymous-file publication primitives.

## Actual dialogue and the blocking behavior

The no-memory role profile with turn stopping produced these real responses:

```text
User: Мой любимый цвет — бирюзовый.
Alice: Хорошо, бирюзовый — это очень красивый и успокаивающий цвет.
User: Какой цвет я назвал?
Alice: Вы назвали цвет бирюзовый.
User: Исправление: теперь мой любимый цвет — синий.
Alice: Хорошо, синий — это тоже очень красивый и успокаивающий цвет.
User: Какой мой любимый цвет теперь?
Alice: Ваш любимый цвет теперь — синий.
User: Новая тема: сколько будет 3+4?
Alice: 3+4=7.
```

Turn headers are stopped before another speaker is committed. Inline quotations
and fenced code are not speaker boundaries. `finish_reason` distinguishes a
turn boundary from the output budget; 32 tokens can truncate an explanation.
With 64 tokens the ice-melting explanation completed, but after the valid
`Neris → Lumen` correction, `Новая тема: сколько будет 2+2?` instead produced
`Hello, Lumen.` This is a semantic failure despite a clean turn boundary.

The experimental memory conversation profile has a stronger counterexample:

```text
Confirmed acquisition, source user: candidate-a → window
Confirmed acquisition, source person:1: candidate-a → helmet
Clear current context; MEMORY ON; ask user candidate-a: window
User: My call sign is Neris.
Alice: window                 # memory_applied=false; no history dropped
User: What is my call sign?
Alice: window                 # memory_applied=false; no history dropped
Ask person:1 candidate-a: helmet
Ask user candidate-a: helmet  # wrong; M/Bridge/readout unchanged
```

The source-bound neural state still contains the expected information. Without
history the same readout yields `window`; with all role history it yields
`helmet`. No archive/transcript reload or KV cache was involved. Exact input-ID
versus embedding logits agree (maximum difference 0); the `key_for` prepass also
changes logits by 0. A full QA-history control restores this one `window` reply,
but a QA-framed ordinary `My call sign is Neris.` produces repeated digits.
Manually supplied correct acknowledgements in capability controls do not count
as end-to-end success. This establishes failure of the tested Base/interface
combination, not a proof that every possible format is impossible.

Further prompt searches are stopped. The minimal owner decision proposed for
continuation is a bounded compatibility check of a frozen dialogue-capable
donor, using copies of M/Bridge. The existing codec is donor-bound: replacement
is not presumed compatible and must neither reset acquired state nor silently
authorize retraining. No alternate model is loaded by this branch.

## Launch for review

Install the package and obtain the exact compatible assets using
[PUBLIC_V0_INSTALL.md](PUBLIC_V0_INSTALL.md) and the
[Bridge reconstruction recipe](PAYLOAD_BRIDGE_RECONSTRUCTION.md).
These dependencies remain required; the repository contains no weights.
Use expendable copies and set the paths explicitly:

```bash
alice --model-path "$ALICE_MODEL_PATH" --device cuda:0 --dtype bfloat16 \
  --memory-bridge "$ALICE_BRIDGE_PATH" --memory-state "$ALICE_TRIAL_MEMORY" \
  --memory-context question --max-new-tokens 16 \
  --transcript-dir "$ALICE_TRANSCRIPT_DIRECTORY"
```

The default question profile preserves bounded public-v0 recall and omits current
conversation history, even with MEMORY OFF. Omit memory flags to inspect ordinary
dialogue with history. To reproduce the unaccepted combined profile, explicitly
choose `--memory-context conversation --max-new-tokens 32`.
`/clear` removes active context; `/new` closes the prior UI session and creates
another. Neither erases M. Only explicit supported acquisition changes M.

## Restart, backup and restore

An indeterminate parameter publication has no Saved acknowledgement. It sets
`restart_required=true`, error code `memory_restart_required`, closes the memory
writer and terminates the CLI with exit status 2. Stop using that process.
Missing acknowledgement does not prove absence of a commit. Reopen the same
checkpoint in a fresh process and inspect the restored content before retrying
the original acquisition. Never delete the manifest/lock to force an empty reset.

For a running healthy process, use `/memory backup NEW_DIRECTORY`. The service
holds its existing writer lock and copies only the committed M plus its matching
sealed Bridge. It does not republish or alter the active checkpoint.
For a stopped checkpoint, the equivalent offline command is:

```bash
python -m alice_runtime.memory_backup create \
  --memory-state "$ALICE_TRIAL_MEMORY" --bridge "$ALICE_BRIDGE_PATH" \
  --output "$ALICE_NEW_BACKUP" --expected-bridge-digest "$ALICE_TRUSTED_BRIDGE_DIGEST" \
  --protected-root "$ALICE_MODEL_PATH"
python -m alice_runtime.memory_backup restore \
  --backup "$ALICE_NEW_BACKUP" --output "$ALICE_NEW_RESTORE" \
  --expected-bridge-digest "$ALICE_TRUSTED_BRIDGE_DIGEST" \
  --protected-root "$ALICE_MODEL_PATH"
```

Both destination roots must be new. Restore validates manifests, checksums,
finite parameters, source registry, latent basis and the exact Bridge pairing
before success. It returns `memory_path` and `bridge_path`; start a new Alice
process using those two returned paths. An existing memory is never overwritten
or silently reset. The restored memory obtains its writer lock on first attach.
Bundles contain neither transcript nor Qwen weights. Partial failed copies are
left for inspection and are not acknowledged as valid backups. Keep the trusted
Bridge digest from an independently checked checkpoint/recipe, not from an
untrusted bundle. The same Linux/filesystem restrictions as export apply.
Nonblocking input opens reject FIFOs before waiting for a writer; create/restore
CLI timeout tests verify refusal without a success receipt or destination.

Measured recovery: offline create plus two fresh pair restores completed in
3.81 s. Three fresh frozen-Qwen processes (23.40 s total) returned corrected
`dragon`, retained `museum`, source `person:1` value `helmet`; clear OFF/EMPTY
omitted `dragon`, ON returned it, and a restored before-correction pair returned
`window` with the same Bridge. All checkpoint hashes stayed unchanged. This used
the explicit legacy question profile, without teaching-transcript restoration.
The external reporter failed after its behavior assertions while aggregating a
nested resources dictionary; the raw failure is retained. A corrected offline
assessment rechecked the saved events and hashes without repeating GPU work.

## Bounded resource evidence and retained limits

The short real run generated 91 responses, rejected one 65,537-character input,
then continued normally. It exercised 70 turns in one session and 20 subsequent
sessions: 64 retained messages per session, 16 retained sessions, 5 closed-session
evictions. Active sessions and participant IDs are never evicted/reassigned.
Other caps are 262,144 characters per session, 32 participants and a 20 MiB
transcript restore. Oversized old snapshots are rejected, not silently truncated;
this does not change the independent published M schema.

The combined ordinary/resource check took 115.17 s. For the 91-response process,
median latency was 1.026 s (range 0.972–1.618 s), peak RSS 4,759.52 MiB, peak
allocated VRAM 3,643.95 MiB, reserved VRAM 3,666 MiB. Allocated VRAM returned to
3,626.13 MiB at every measured snapshot. RSS is the process high-water mark,
not a sampled current-memory trace. M and Bridge files stayed byte-identical.
This is a short stability check, not evidence of multi-hour reliability or
semantic correctness. One-token values, supported RU/EN object-label grammar,
bounded capacity/interference and donor/Bridge compatibility limits remain.

Session confirmation is also bound to a RAM-only incarnation of the session:
evicting a closed session and reusing its textual ID cannot revive an old pending
proposal or confirmation token. This adds no fields to the v1 transcript or M.

## Personal review sequence

1. Use copied assets/state and a new export directory. Verify `/save chat.json`
   works once; a repeated filename, traversal or protected destination refuses
   without changing the checkpoint. Exit with `/quit` and restart.
2. In the default question profile, confirm one supported object-label fact,
   clear context, compare OFF/EMPTY/ON, correct it, and restart for recall.
3. Back up the pair, restore to a fresh directory, and query the restored pair
   without loading any teaching transcript. Preserve the original pair.
4. Inspect ordinary context and the explicit conversation profile using the
   counterexamples above. They are expected blockers, not acceptance examples.
5. Review resource/error output and restart instructions. Do not merge this
   draft or accept Day0 while the ordinary/context+M blockers remain.
