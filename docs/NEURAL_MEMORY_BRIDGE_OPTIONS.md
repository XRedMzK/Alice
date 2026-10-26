# Neural Memory and Memory Bridge — Collab options

25 October 2026. **Candidates for the next decision; no algorithm selected.**
Stage 1 demonstrates a local frozen-Qwen runtime and an input-embedding connection:
a test signal can affect logits and be disabled or detached. This is technical
wiring, not implemented Neural Memory, learned acquisition or a trained Bridge.
No memory implementation or memory/Bridge training is chosen or started here.

## 1. Small fast-weight associative matrix + learned embedding projection

Use a separately owned neural parameter matrix for key/value associations, with
a bounded delta-rule write and a learned read/write projection. A small trainable
Bridge projects the memory readout into compatible Qwen input embeddings.
The starting mechanisms are described in the [author paper](https://proceedings.mlr.press/v139/schlag21a.html)
and [official MIT code repository](https://github.com/ischlag/fast-weight-transformers).
Reuse only necessary operations; do not import its old experiment toolkit.

This offers a small state, explicit updates and straightforward checkpointing.
Capacity and interference are the main risks: correcting one association must
not erase unrelated knowledge. Writes and reads need correct `source_id` binding;
the same fact from different participants must not collapse accidentally.
Persist the actual memory parameters and identity mapping, not only the latest
session or a renamed Qwen cache. Resetting a session must not reset this memory.
Successful latent retrieval would still not prove useful Qwen knowledge use.

## 2. Small nonlinear parameter memory + learned embedding projection

Assess a compact MLP that learns associative key/value mappings through updates
to its own parameters, taking only a minimal read/write mechanism from
[Titans](https://arxiv.org/abs/2501.00663). Keep the full Titans architecture out
of scope. Use the same external embedding Bridge boundary as option 1.

Nonlinear memory is a candidate when a simple matrix cannot represent the needed
associations; it is not assumed to work better for Alice. Gradient updates,
forgetting, update latency and retention make it more complex to diagnose.
Check a correction and one unrelated retained item before increasing capacity.
Source-conditioned acquisition and later attribution remain necessary.
Save memory parameters and any required update state independently of the Bridge;
neither a process restart nor a ComfyUI reconnect may silently initialize them.
The paper is a mechanism source, not a software license or an Alice integration.

## 3. Either memory + gated residual Bridge at a hidden layer

Instead of conditioning only input embeddings, a small trainable projection/gate
could inject memory readout into one declared hidden-layer representation.
[PyTorch forward hooks](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
provide removable output intervention primitives; the Alice mechanism is an
engineering proposal, not a ready integration supplied by that API.

This may offer more direct control but adds layer/version coupling and more
opportunities for dtype, device, state and cache mismatches. A gated residual
must not overwrite a source binding or modify Qwen parameters. The Bridge must
remain separately checkpointed and replaceable without resetting the memory;
preserved bytes alone do not establish behavioral compatibility of a replacement.
Defer this option unless the simpler embedding connection has a concrete limit.

## Shared reuse and selection boundary

[Prompt tuning](https://huggingface.co/docs/peft/package_reference/prompt_tuning)
illustrates learned embedding conditioning with frozen backbone parameters.
A task-specific soft prompt is not dynamic memory reading: projecting an
independent memory readout is our adaptation and requires practical validation.
Ordinary [KV-prefix tuning](https://huggingface.co/docs/peft/package_reference/prefix_tuning)
must not be assumed compatible with every Qwen3.5 hybrid attention/cache layer.

Prefer a small comparison of options 1 and 2 behind the same embedding boundary,
starting with the simplest source-bound acquisition/correction case. Compare
useful knowledge use, interference, update cost and recoverability; stop at the
first adequate mechanism. This is a recommendation for a future bounded task,
not authorization to implement or train either candidate.

Reuse existing PyTorch and Foundation state/checkpoint primitives where suitable.
No upstream code has been copied for this shortlist. Before adopting code, pin
the necessary files/version and check their applicable licenses and notices;
a paper license does not automatically license a separate implementation.
The Stage 1 inference-only port is not a training path. Any future authorized
Bridge learning needs the required gradient flow while Qwen parameters stay frozen.

The first cognitive result remains: new knowledge with correct `source_id`
changes independent memory parameters → teaching transcript/prompt/KV context
is cleared → process terminates and restores memory → frozen Qwen uses that
knowledge through the trainable Bridge → matched **MEMORY OFF / MEMORY ON**
shows useful contribution without unacceptable regression. Participant mappings
and source binding must survive restore. Memory remains independent of Qwen,
Bridge replacement and the ComfyUI lifecycle; no M1 qualification ladder applies.
