# Engineering validation for new Alice behavior

Status: active rules under Engineering First v2, 25 October 2026. This is the
default for new work; the unchanged [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)
records the frozen Foundation protocol and applies in its explicit scope.

## Small, meaningful checks

Start with the concrete ability and realistic failure modes. Use critical unit
tests, affected integration checks, loading/saving/recovery, state-integrity,
resource measurement and disable/replace checks as relevant. Do not require a
TRAIN/DEV/FINAL ladder, repeated seed/origin studies, thousands of synthetic cases,
near-perfect F1/accuracy or proof of novelty by default. Use metrics, additional
controls and repetitions when they help answer a concrete engineering question.

Real messages, sequential conversations and changes of context must test a
human-facing ability in the actual Alice runtime. Memory additionally needs
experience updates and later use. Controlled fixtures are useful for diagnosis;
they do not replace all real-runtime evidence. Documentation-only work does not
claim runtime validation.

## MODULE OFF / MODULE ON

For each new cognitive module, define the useful task outcome and unacceptable
regressions before comparing:

1. Use the same relevant messages, task context, initial state snapshots and
   generation settings in both arms. Isolate mutable state so one arm cannot
   teach or corrupt the other. Describe any unavoidable difference.
2. Run Alice with the function disabled (**MODULE OFF**), using a declared bypass
   or baseline that leaves the rest of the system operational.
3. Run Alice with the function connected (**MODULE ON**).
4. Inspect task success, relevant regressions and resource cost. Record a short
   paired result with the implementation/configuration identity and limitations.

PASS means the function helps Alice perform its necessary task and introduces no
unacceptable regression. A different wording, changed embedding or perturbed
answer is not itself an improvement. If the difference is absent or useless,
revise, merge or remove the mechanism. If generation variability obscures the
answer, repeat only enough matched cases to resolve that uncertainty; do not
launch a qualification programme automatically.

If an execution is invalid because of corrupted state, failed loading or an
environment/resource failure, record that separately from a working mechanism
that fails its behavioral purpose. Neither counts as PASS.

## Example: future memory acceptance

These are proposed behavioral checks, not claims of implemented memory:

1. Give Alice new information that is absent from the later query and initial
   evaluation context; verify the intended independent neural memory update.
2. End the current conversation. In a fresh conversation, ask a relevant question
   and check that Alice can use the acquired information.
3. Save, stop and restart the runtime. Check restored experience and subsequent
   use, without replaying the teaching transcript into the prompt.
4. Compare the same later query with memory OFF and ON. The intended ability must
   disappear or worsen when memory is disabled; account for information Qwen
   already knows or can infer without it.
5. Check a small set of unrelated behaviors and a concrete context/fact update
   for regressions, stale answers and state corruption.
6. Replace the Bridge through its declared interface. Preserve the memory
   checkpoint, verify compatibility and later recall separately, and reject an
   incompatible replacement without overwriting the retained memory.

Qwen parameters remain frozen. A transcript, prompt cache or retrieval database
alone does not establish the approved plastic neural-memory update. Save/restore
tests of Foundation synthetic state do not establish this capability either.

## Failure handling and retained evidence

Reproduce a specific failure, identify its cause, fix or replace the affected
component, repeat its relevant checks, then check the result in the real runtime.
Do not default to multi-day retraining or dozens of new architectures.

Keep working checkpoints, code/configuration, necessary results, short technical
failure reports and artifacts required to recover important outcomes. Before
large computation, estimate disk usage and declare disposable outputs and cleanup
timing. Do not retain every temporary bank, tensor dump or repeated intermediate
result indefinitely. Preserve historical protected evidence; this retention rule
does not authorize deleting it. Report unrun checks honestly and reuse unchanged
Foundation evidence within its original scope.
