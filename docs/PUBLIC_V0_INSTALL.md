# Install and reproduce Alice public v0

This release provides the Stage 1 runtime, independent parameter memory and the
bounded RU/EN acquisition interface. It needs separately supplied local Qwen and
sealed Bridge assets. The repository contains no model weights, trained Bridge,
acquired experience, private conversation, calibration corpus or run artifacts.
Source alone can run the asset-free checks; it cannot recreate the accepted
Bridge checkpoint. Supplying an arbitrary newly initialized Bridge does not
reproduce the demonstrated behavior.

## Source and environment

Use Linux x86_64 and CPython 3.12. The observed real-model profile is Torch
`2.13.0+cu130`, Transformers `5.15.0`, Tokenizers `0.22.2`, Safetensors `0.8.0`
and Accelerate `1.14.0`, with a CUDA/BF16-capable NVIDIA device. The direct pins
are in [requirements-runtime-tested.txt](../requirements-runtime-tested.txt).
They are not a complete transitive lock. Other backends, dependency versions and
donors need their own compatibility check. CPU loading requires `float32` and
substantially more RAM; the real memory reproduction below uses CUDA/BF16.

Start from the owner-approved commit in the public PR stack. Set `ALICE_PUBLIC_REF`
to that exact commit rather than a moving branch:

```bash
git clone https://github.com/XRedMzK/Alice.git Alice
cd Alice
git checkout "$ALICE_PUBLIC_REF"
python3.12 -m venv .venv-runtime
export ALICE_PYTHON="$PWD/.venv-runtime/bin/python"
```

Install the pinned runtime and build requirements from your compatible local
wheelhouse, then install Alice as an ordinary package:

```bash
"$ALICE_PYTHON" -m pip install --no-index --find-links "$ALICE_WHEELHOUSE" \
  -r requirements-runtime-tested.txt
"$ALICE_PYTHON" -m pip install --no-index --no-deps --no-build-isolation .
"$ALICE_PYTHON" -m pip check
"$ALICE_PYTHON" -m alice_runtime --help
.venv-runtime/bin/alice --help
```

The wheelhouse must also contain compatible transitive dependencies. For a new
network-backed environment, select a matching Torch build using the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
before resolving the remaining requirements. The
[Transformers installation guide](https://huggingface.co/docs/transformers/installation)
describes its environment and offline configuration. Downloading or resolving a
different stack is not covered by the local verification. No model download or
training is part of these commands.

The installation check for this release uses an ordinary, non-editable source
install in a fresh virtual environment and runs outside the checkout with
`PYTHONPATH` removed. Available dependencies are reused from an existing
compatible environment without a download. This verifies source packaging and
runtime imports, not a fresh network resolution of every dependency. To reproduce
that offline reuse on a machine with a compatible environment:

```bash
export ALICE_EXISTING_PYTHON=/path/to/compatible/environment/bin/python
export ALICE_PYTHON="$PWD/.venv-runtime/bin/python"
export ALICE_DEPENDENCY_SITE="$("$ALICE_EXISTING_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
"$ALICE_PYTHON" -c 'from pathlib import Path; import os, sysconfig; Path(sysconfig.get_path("purelib"), "alice-existing-dependencies.pth").write_text(os.environ["ALICE_DEPENDENCY_SITE"] + "\n")'
"$ALICE_PYTHON" -m pip install --no-index --no-deps --no-build-isolation .
```

This deliberately shares dependencies, not Alice source. A child virtual
environment with `--system-site-packages` alone does not reliably inherit another
virtual environment's packages. Keep the dependency environment available.
The packaged entrypoint is `alice`; `python -m alice_runtime` remains supported.
The old Foundation checks still run from the source checkout and resolve their
existing repository resources there; this release does not claim a standalone
Foundation wheel/resource API.

## External asset contract

Configure paths for your machine. Keep all three directories separate; memory
must be writable and must not be inside the model or Bridge directory.

```bash
export ALICE_MODEL_PATH=/path/to/local/qwen3.5-2b-base
export ALICE_BRIDGE_PATH=/path/to/sealed/payload-bridge
export ALICE_MEMORY_STATE=/path/to/writable/alice-memory
```

The model directory must be a complete compatible local Qwen3.5-2B-Base snapshot:
`config.json`, tokenizer assets, `model.safetensors.index.json` and every shard
referenced by that index. Loading is text-only, local-only and frozen. Qwen's
official FP32 recurrent parameters are preserved during the BF16 load.

The accepted full-rank Bridge consists of `bridge.json` and the safetensors file
referenced by its `tensor_file`. Its public compatibility identity is:

| Field | Accepted value |
| --- | --- |
| Schema | `alice.payload_embedding_bridge/v1` |
| `state_digest` | `d2b7a0572fef22b1cb8a7d91d32990ee529b9ad0338905793467bd894d750cde` |
| Tensor SHA-256 | `cd422575c9a5a57f17542a07db890dd914abf06334c1dd9d467c0bb8a0439012` |
| Payload | 17 independent 128-dimensional value heads, shared 128-dimensional key |
| Model hidden size / slots | 2048 / 1 |
| Value input / encoding | `token_embedding_mean_v1` / `orthogonal_block_payload_v1` |
| Query normalization / layout | `bounded_relation_entity_v1` / `repeat_qa_turn_stop_v1` |

The runtime checks Bridge schema, tensor checksum, parameter digest, latent
basis, dimensions, semantics and SHA-256 bindings of model `config.json` and
`tokenizer_config.json`. Those two bindings do not fingerprint every model or
tokenizer shard. Preserve the provenance and checksums of the full external
snapshot independently. A manifest's self-consistent checksum alone does not
prove its origin. The smoke command below additionally compares the trusted
Bridge `state_digest` above before launch.

Memory uses schema `alice.memory_payload/v1`, FP32 shape `[sources,17,128,128]`,
and the Bridge's latent basis. The default registry is
`agent`, `user`, `person:1`, `person:2`. A new memory directory starts empty;
existing compatible memory is restored. Incompatible or corrupt state fails
explicitly. Replacing a Bridge does not authorize resetting or migrating memory.
The Bridge and Qwen stay fixed during acquisition; only M changes. Do not publish
your memory directory as a source artifact.

## Ordinary messages, restart and correction

Run from any directory after package installation, with no `PYTHONPATH`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  "$ALICE_PYTHON" -B -m alice_runtime --model-path "$ALICE_MODEL_PATH" \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 8 \
  --memory-bridge "$ALICE_BRIDGE_PATH" --memory-state "$ALICE_MEMORY_STATE"
```

Example input and expected control/recall behavior:

```text
Запомни: метка объекта atlas-0 — window.
  → Alice asks whether to save that exact fact for source user.
да
  → Alice acknowledges only after publishing the changed memory.
Remember that the label of object atlas-1 is museum.
yes
/quit
```

Start the same command again. Do not restore any session or teaching transcript:

```text
Какая метка у объекта atlas-0?
  → window
Which label was assigned to object atlas-1?
  → museum
Исправь: метка объекта atlas-0 — dragon.
да
/quit
```

After another restart, `Скажи метку объекта atlas-0.` should yield `dragon`, while
the other fact remains `museum`. Pending confirmation is temporary RAM state;
exiting before `да`/`yes` discards it. The actual speaker supplies `source_id`.
`/source person:1` selects another speaker, while `/knowledge-source person:1`
selects whose stored knowledge is being queried. Neither a source claim in the
message text nor another speaker's confirmation grants a write. Each proposal is
bound to its session, source and exact content.

`/memory off`, `/memory empty` and `/memory on` change reads without erasing M.
Explicit confirmed acquisition still writes while reads are OFF. `/new` and
`/clear` affect sessions only. `/save PATH` saves a transcript, not neural memory.
The [runtime guide](NEURAL_MEMORY_RUNTIME.md) describes the API and manual
structured-acquisition fallback. Successful recall goes through frozen Qwen;
only clarification, confirmation and saved acknowledgements use templates.

## Two separate verification paths

Asset-free CI runs the original two Foundation basic checks and the production
runtime/memory tests using [requirements-cpu.lock](../requirements-cpu.lock).
Those tests do not load Qwen or the accepted Bridge asset and are not evidence
that neural recall works on a real donor. Run the same CPU scope from source:

```bash
python3.12 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.lock
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=2 .venv-cpu/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_foundation0_cpu.py::test_01_manifest_and_schema_validation \
  tests/test_foundation0_cpu.py::test_02_registry_graph_and_semantics_blind_routing \
  tests/test_memory_*.py tests/test_runtime_*.py tests/test_confirmation_binding.py \
  tests/test_fact_intent.py tests/test_dialogue_acquisition.py
```

For the opt-in real-model check, choose a **new** output directory outside the
checkout and run the portable external checker from the public source tree:

```bash
"$ALICE_PYTHON" tools/public_memory_smoke.py --python "$ALICE_PYTHON" \
  --model-path "$ALICE_MODEL_PATH" --bridge "$ALICE_BRIDGE_PATH" \
  --device cuda:0 --dtype bfloat16 --output "$ALICE_SMOKE_OUTPUT" \
  --expected-bridge-digest d2b7a0572fef22b1cb8a7d91d32990ee529b9ad0338905793467bd894d750cde
```

The checker removes `PYTHONPATH`, runs the installed CLI in an empty working
directory and uses five fresh processes/sessions. It writes three small
demonstration facts using ordinary RU/EN messages and confirmation, then checks
fresh recall, OFF/EMPTY/ON, source separation, correction, retention of a second
fact, and restoring the earlier memory with the same Bridge. It also compares
unrelated arithmetic with memory OFF/ON, checks Paris/7, validates state checksums
and source-plane hashes, and checks that Bridge files do not change. Expected
answers exist only in the external checker, not in runtime assets or query input.
No `/remember` command, teaching transcript restore, calibration or training is
used. Per-process timeout is 90 seconds by default. A failed or inconclusive
control returns a nonzero exit and writes `result.json` with the failing cause.

Evidence stays in the chosen output directory: JSONL events, stderr, CLI input,
original/corrected M and a small Bridge copy. Do not commit it. GPU availability,
per-process wall time, peak RSS and Torch allocated/reserved VRAM are reported
separately from asset-free CPU test results. Torch allocation figures exclude
the driver and other processes.

The prior compatible CUDA/BF16 runtime used roughly 4.65 GiB peak RSS and 3660
MiB Torch reserved VRAM. Four-source M is 4.25 MiB and the Bridge is about 576
KiB; this five-process check retains a few small copies rather than a feature
bank. These are observations, not universal resource guarantees.

## Precise behavior boundary

This is one-token payload memory with small associative capacity. Natural
acquisition accepts only the documented explicit object-label grammar, literal
IDs and one model-token word values. Queries use three explicit English forms
or two literal Russian label forms. Other relations may be passed through the
structured API but are not a claim of free natural acquisition. No free-form
extraction, arbitrary paraphrase, multi-token answer, pronoun-based follow-up,
unlimited retention or arbitrary donor replacement is claimed.

With memory attached, each answer uses the current question only, including
OFF/EMPTY. Conversation history is UI state, not an additional fact channel.
Without memory flags, the existing Stage 1 context profile remains available.
Source selection is explicit; it is not inferred speaker recognition. ComfyUI
is not required and owns neither M nor Bridge lifecycle. No training, new neural
architecture or grammar expansion is introduced by this release.
