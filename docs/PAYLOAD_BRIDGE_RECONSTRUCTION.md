# Reconstruct the tested payload Bridge without training

The source distribution ships no Qwen weights, Bridge checkpoints or acquired
memory. The ordinary memory runtime requires a compatible sealed Bridge.
This repository has no published checkpoint download URL. For the one recorded
profile below, the supplied source can reconstruct its exact neural state on CPU.
This is reconstruction of an existing artifact, not a new calibration run.

Supply an existing local `Qwen/Qwen3.5-2B-Base` snapshot. The tested upstream
revision is `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`; Qwen assets remain governed
by their upstream Apache-2.0 license. Alice source and this reconstruction utility
use the repository MIT license. No Qwen embedding vectors are distributed here.
Install the runtime first using [the public installation guide](PUBLIC_V0_INSTALL.md).
The exact CPU state was checked using Python 3.12 and Torch `2.13.0+cu130`, with
two CPU threads; the utility itself neither loads Qwen weights nor uses CUDA.
The checked host was an AMD Ryzen 7 5700X with native CPU/MKL dispatch and no
`MKL_CBWR` override. This describes the observed environment, not a promise that
another processor or backend will produce the same tensor bytes.

```bash
"$ALICE_PYTHON" tools/reconstruct_payload_bridge.py \
  --model-path /path/to/local/Qwen3.5-2B-Base \
  --output /path/to/new/payload-bridge
```

Use the resulting new directory as `--memory-bridge` and keep the writable
`--memory-state` directory separate. Output must not already exist or overlap
model assets. The tool never downloads, fits, optimizes, acquires facts or changes
Qwen. CPU construction takes only the Bridge tensors; generation still requires
the separately supplied full Qwen snapshot and its runtime resources.

The utility checks exact SHA-256 bindings of `config.json` and
`tokenizer_config.json`. These checks do not hash or certify all Qwen weight
shards; the reader must supply the stated model snapshot. Construction uses the
existing `PayloadEmbeddingBridge` implementation with its fixed CPU seeds:
Fourier frequencies `1729`, orthogonal rotation `314159`, hidden size `2048`,
128-wide keys, 17 payload heads, one slot, bandwidth `0.003`, confidence threshold
`0.6`, and layout `repeat_qa_turn_stop_v1`. The recorded paired scale is
`0.6056226491928101`.

The original scalar was assigned once to the median L2 norm of 16 generic,
frozen BF16 input embeddings converted to FP32. There were zero optimizer steps
and zero backward calls. No final source/object/value assignments were used.
Reconstruction reuses that sealed scalar; it does not repeat the calibration or
need its historical tokenizer-selection helpers, budget ledger or fixtures.
The accepted state has only `R`, `keys.frequencies`, `keys.bandwidth` and
`value_scale`; it contains no Qwen embedding table, value palette or fact table.

Before saving and again after restore, the utility requires state digest:

```text
d2b7a0572fef22b1cb8a7d91d32990ee529b9ad0338905793467bd894d750cde
```

Its latent basis is `alice.memory_payload/17x128/` followed by that digest.
Saved manifest/tensor filenames contain fresh UUIDs; identical filenames or
manifest bytes are not required. The state digest verifies all Bridge tensor
bytes and configuration. CPU QR/RNG bit identity across every Torch version,
CPU backend and platform is not promised: a mismatch fails before a checkpoint
is written. Do not replace the expected digest or silently recalibrate. Report
the environment and request the verified original artifact from the maintainer
if exact reconstruction is unavailable; no public hosted artifact is claimed.
Different MKL compatibility modes have already been observed to change QR tensor
bytes by a few parts in a million and therefore change the required digest.
Asset-free CI explicitly skips the positive identity test on such a backend;
the separate tests still require wrong state/bindings to be rejected before
writing. A green CPU CI does not mean this external asset was reproduced there.

This reconstruction provides the concrete external-Bridge prerequisite for the
opt-in [installed CLI smoke](PUBLIC_V0_INSTALL.md#two-separate-verification-paths). Passing that smoke with
the supplied Qwen establishes the bounded real behavior. A successful Bridge
reconstruction alone proves state identity, not cognitive recall or broader
memory capabilities.
