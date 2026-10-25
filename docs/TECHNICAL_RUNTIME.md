# Stage 1: local frozen-Qwen technical runtime

The owner approved PR #1 at `9c29e08`; it was merged as `641d40e` before this
implementation. `src/alice_runtime` provides technical text generation, explicit
participants/sessions and a removable input-embedding connection. It implements
no plastic Neural Memory, trained Memory Bridge or ComfyUI integration.

## Run with existing local resources

Use CPython 3.12 and an existing compatible Torch/Transformers environment.
The observed environment is Torch 2.13.0+cu130, Transformers 5.15.0,
Tokenizers 0.22.2, Safetensors 0.8.0 and Accelerate 1.14.0.
[requirements-runtime.in](../requirements-runtime.in) records direct optional
dependencies; it is not a resolved portable lock. The Foundation CPU lock is
unchanged. No packages or model assets were downloaded for this stage.

Set `ALICE_PYTHON` to the existing environment's Python and `ALICE_MODEL_PATH` to
the existing Qwen3.5-2B-Base snapshot directory. From this source checkout:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$ALICE_PYTHON" -m alice_runtime --model-path "$ALICE_MODEL_PATH" \
  --device cuda:0 --dtype bfloat16 --message 'Hello. My name is Mira.'
```

Omit `--message` for the synchronous terminal loop. Commands: `/new`, `/clear`,
`/person N`, `/source ID`, `/save PATH`, `/quit`. `--person 1` and
`--source-id person:1` select another explicit source. `--restore-sessions PATH`
loads a validated transcript/identity snapshot; `--session ID` then selects an
existing session. Omitting it creates a new isolated session. Session save files
contain transcripts; keep them outside published source history.

CPU mode is explicit `--device cpu --dtype float32`. It is supported by the
loader but the real-model run reported below used CUDA/BF16. There is no silent
device fallback. Stage 1 defaults to 256 input tokens and 16 output tokens;
the generation cap is 64. Context eviction drops whole old messages, preserving
their source headers; an oversized current message fails without committing it.

Loading uses native text-only `Qwen3_5ForCausalLM`, local files only, no remote
code and no Hub kernels. It excludes outer vision/MTP weights, preserves official
DeltaNet FP32 parameters, enters eval mode and disables parameter gradients.
Generation uses native greedy `generate`, `torch.inference_mode` and
`use_cache=False`; the latter avoids persistent hybrid-cache state in this
small technical stage. No optimizer or model-save path exists.

## Identity, sessions and the candidate Bridge boundary

Agent (`agent`), User (`user`) and Another Person #N (`person:N`) have explicit
stable IDs supplied by the caller. The runtime does not infer real-world
identity from text. IDs accompany messages and reach `BridgeContext.source_ids`
in the order of selected context messages. Generated text remains Agent content,
even when Base prints another speaker label; text cannot register a participant.
Save/restore retains the registry and sessions atomically. These transcripts are
ordinary support state, not independent acquired knowledge.

`runtime.bridge.attach(callable)` enables a candidate intervention at input
embeddings. Its callable receives `(embedding_copy, BridgeContext)` and must
return a contiguous finite tensor with exactly the original shape, device and
dtype: here `[1, sequence, 2048]`, CUDA/BF16. `.disable()`, `.enable()` and
`.detach()` permit bypass/replacement. The temporary hook is always removed,
including after errors. Failed generation commits no partial conversation.

This port is inference-only, not a memory algorithm or training path. A future
learned interface will need an explicitly scoped gradient/update path while
Qwen stays frozen. Passing the tensor check does not prove useful memory or
compatibility with Foundation's distinct float32 latent contract.

## Short result and limits

Eleven affected CPU tests passed, including two unchanged Foundation basic tests.
The real local-model smoke passed in **7.37 s**, including **3.61 s** loading:

- 1,881,825,088 text parameters; 2,592 official FP32 parameters preserved.
- No-op and disabled Bridge logits exactly matched baseline. A test signal
  changed logits (maximum absolute difference 15.18); detaching restored exact
  baseline. This demonstrates wiring, not a useful cognitive improvement.
- Wrong shape/device/dtype and nonfinite output were rejected without session
  changes; source IDs, session isolation, clear/close and snapshot roundtrip passed.
- Parameter version counters, frozen/eval status and checkpoint size/mtime stayed
  unchanged. This is an inference check, not a fresh full checkpoint hash audit.
- PyTorch peak allocated VRAM **3,637.9 MiB**, peak reserved **3,644 MiB**;
  full process VRAM snapshot after checks **4,266 MiB**. These are different
  measurements; the process snapshot is not a sampled peak. Peak RSS **4,759.3 MiB**.

The Base model produced `Hello, Mira. How can I help you?` followed by a speaker
continuation; a short Russian sequence prompt was repeated. Technical generation
works, but reliable chat behavior is not established. A bounded next formatting
remedy is explicit turn stopping/completion framing; if insufficient, report the
backbone limitation for a separate replacement decision. No dialog optimization,
model substitution, training or new download was performed here.

The first run reached report writing but exposed a JSON set/list mismatch in
Transformers loading metadata. Normalizing that field fixed it; the short run
then passed. No repeated architecture search or training was involved.

To reproduce only this short real-model check deliberately:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$ALICE_PYTHON" tools/runtime_smoke.py --model-path "$ALICE_MODEL_PATH" \
  --device cuda:0 --dtype bfloat16 --output /tmp/alice-runtime-smoke.json
```

It runs a few capped completions and probes, with no training, downloads or
dialogue score. Do not use the full historical Foundation qualification command.
Next: review [Neural Memory + Bridge options](NEURAL_MEMORY_BRIDGE_OPTIONS.md)
against source-bound acquisition → context clear → fresh process restore →
frozen-Qwen use → MEMORY OFF/ON. That cognitive milestone is still unimplemented.
