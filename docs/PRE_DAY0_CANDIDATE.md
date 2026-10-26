# PRE-DAY0 v0 candidate — repairs in progress

Owner authority: 26 October 2026, from public `faa03d3` and the
[PRE-DAY0 gap report](PRE_DAY0_V0_GAP_REPORT_RU.md). This branch is not Day0
acceptance or a canonical instance. No architecture, donor or training change
is authorized. The complete goal requires all four blocks and actual frozen-Qwen
behavior; a passing security test alone does not complete it.

| Priority | Current scope |
| --- | --- |
| 1 — State protection | Create-only transcript capability, explicit export directory, pinned Linux FDs and no-replace publication; temporary-copy tests and caller migration. |
| 2 — One answer | Pending bounded technical correction and real RU/EN check; no automatic Qwen replacement. |
| 3 — Context with M | Pending current-context/OFF/ON/clear/restart evidence; old transcripts may not become hidden long-term storage. |
| 4 — Recovery/resources | Pending integrated restart-required, safe M/Bridge backup/restore and bounded-retention/live-resource evidence. |

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
anonymous-file publication primitives. Remaining priorities and personal
acceptance instructions will be completed from measured results, without
redefining the goal around an easier subset.
