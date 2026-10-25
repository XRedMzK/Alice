# Alice runtime with persistent parameter memory

This source-only integration reuses the bounded functional memory prototype
`244957a` and runtime integration `4cecf57`. It is prepared separately from
Stage 1 PR #2, whose review/merge decision remains with the owner. No model,
Bridge checkpoint, experience state, generated run log or calibration tool is
distributed in this source package.

## Run with existing compatible assets

Use CPython 3.12 on Linux and the existing optional runtime requirements.
Configure these paths for your machine:

```bash
export ALICE_PYTHON=/path/to/compatible/environment/bin/python
export ALICE_MODEL_PATH=/path/to/local/qwen3.5-2b-base
export ALICE_BRIDGE_PATH=/path/to/sealed/payload-bridge
export ALICE_MEMORY_STATE=/path/to/writable/alice-memory

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  "$ALICE_PYTHON" -B -m alice_runtime --model-path "$ALICE_MODEL_PATH" \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 8 \
  --memory-bridge "$ALICE_BRIDGE_PATH" --memory-state "$ALICE_MEMORY_STATE"
```

The Bridge must be the compatible sealed full-rank payload checkpoint, schema
`alice.payload_embedding_bridge/v1`. Its model/tokenizer bindings and latent
basis are checked. The writable memory directory is independent of Bridge/model
assets. A new directory starts empty; an existing compatible M is restored.
Invalid or incompatible state is rejected, never silently reset.

## Structured acquisition and restart

The initial interface deliberately requires an explicit source/cue/value fact:

```text
/remember user "What is the assigned label of object demo-0?" "window"
/remember person:1 "What is the assigned label of object demo-0?" "helmet"
/quit
```

Successful acknowledgement means only the memory parameters have changed and
the new state is saved. Start the same command again, without restoring any
sessions or teaching transcript:

```text
Which label was assigned to object demo-0?
window
/knowledge-source person:1
Tell me the assigned label of object demo-0.
helmet
```

The example uses the supported question forms; word reproduction depends on
the compatible checkpoint and the declared small-capacity memory scope. The
accepted local integration verified window/helmet/museum/cotton after restart,
correction of window to dragon, source separation, OFF/EMPTY and unrelated
4/Paris/7. This package preparation adds no new model training or vocabulary.

Corrections use `/remember` with the same source and cue. `/memory on|off|empty`
changes reads; EMPTY temporarily uses zeros and does not erase acquired M.
Explicit `/remember` still writes while reads are OFF. `/memory status` shows
schema/basis, source IDs and content digests; `/memory save` explicitly publishes
the current M. The default memory registry is agent/user/person:1/person:2.
`/source` selects the speaker, while `/knowledge-source` selects whose knowledge
is queried. The persisted registry is fixed; adding a UI participant does not
silently add a neural source plane.

`/new` and `/clear` affect only the session. `/save PATH` writes a transcript,
not memory. Acquisition is not appended to session history. EOF, Ctrl-C and
`/quit` release the single-writer lock. Concurrent writers, bad checksums and
overlapping state/Bridge/model directories fail explicitly.

`--message TEXT` uses the same runtime in batch mode. `--json` emits JSONL
events for interactive callers. The Python API exposes `attach_memory`,
`remember`, `respond`, `set_memory_mode`, `save_memory` and `close` on
`alice_runtime.runtime.AliceRuntime`.

## Ownership and limits

Memory owns FP32 parameters `[sources,17,128,128]` and versioned metadata. Values
are continuously encoded/read/decoded through an independent orthogonal basis;
they are never reconstructed as text/token IDs and appended to the prompt.
There is no fact dictionary, transcript or KV cache in the memory directory.
Qwen and Bridge remain fixed during acquisition; only M changes. The same
loaded Qwen instance serves keys, acquisition and generation.

Only one-token values from the bound Qwen vocabulary and three explicit English
question forms are supported. Multi-token acquisition is rejected before write.
Ordinary messages are not automatically learned. With memory attached, the
proven QA profile answers the current question without conversation history,
including OFF/EMPTY. Without memory flags the previous Stage 1 dialogue-context
profile remains available. Pronoun-based follow-up interpretation, arbitrary
paraphrases, unlimited retention and arbitrary donor compatibility are unproven.
Associative interference remains a capacity limit. ComfyUI is not required.

## Checks and resource scope

Asset-free CPU tests cover parameter/source isolation, corrected writes,
integrity, fresh-process restore, writer ownership, rollback, runtime/CLI
dispatch and ordinary Stage 1 behavior. CI uses the existing exact CPU profile;
it neither loads a donor nor performs Bridge fitting. Full real-model evidence
belongs to the separately retained local integration checks, not CI.

The prior accepted CUDA/BF16 integration used about 4.65 GiB peak RSS and 3660
MiB Torch reserved VRAM. Four-source memory parameter data uses 4.25 MiB;
Bridge parameters/buffers use about 576 KiB. These are bounded local observations,
not portable resource guarantees or scientific qualification thresholds.
