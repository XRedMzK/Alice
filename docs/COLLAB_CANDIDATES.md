# Minimal Collab candidate check

25 October 2026. Documentation review only, not a selected memory architecture,
benchmark or implementation. These primary sources suggest reusable primitives;
their suitability for Alice remains an engineering question after roadmap approval.

| Need | Existing primary source | Smallest useful part to assess | Limit / applicability check |
| --- | --- | --- | --- |
| Frozen text generation | [Qwen3.5-2B-Base model card](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) | Compatible text-only causal model and generation APIs; do not adopt multimodal serving infrastructure for a text baseline. | The card identifies a pretrained Base model, not a ready dialog assistant. Verify local model/config/library versions and practical in-context dialog suitability; no implicit Instruct substitution or fine-tuning. |
| Parameter/state lifecycle and intervention | [PyTorch Module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) | Existing module state dictionaries, buffers, freezing and forward hooks, alongside Foundation's independent checkpoint utilities. | A state dictionary contains tensor references, not an independent snapshot. Hooks and save/load primitives provide no memory algorithm, atomic whole restore or behavioral guarantee. Preserve state ownership and validate actual tensors at the chosen model boundary. |
| Independent plastic neural memory | [Titans, author paper](https://arxiv.org/abs/2501.00663) | Candidate associative read/write mechanism that stores information through memory-parameter updates; evaluate a minimal mechanism instead of importing a whole architecture. | A paper result is not evidence of an external Alice memory working with frozen Qwen. Cross-session persistence, correction, capacity, resource cost and Bridge compatibility must be checked separately. No implementation or algorithm is selected here; the paper is not a software license. |
| Small trainable conditioning interface | [PEFT prefix tuning documentation](https://huggingface.co/docs/peft/package_reference/prefix_tuning) | Candidate pattern for learned continuous conditioning while backbone weights remain frozen. Investigate only needed projection/injection primitives. | A task-specific prefix is not dynamic memory reading. Adapting memory output into embeddings/prefixes is an Alice hypothesis. Qwen3.5 uses hybrid attention/cache behavior; do not assume ordinary KV-prefix compatibility at every layer. |

Default reuse decisions: generation and serialization should reuse existing
primitives. Neural-memory rules and memory-conditioned trainable interfaces are
candidates, not chosen implementations. A transcript/RAG store may serve an
explicit engineering purpose but does not replace the approved plastic memory.

Before adopting code or a dependency, record the exact version, applicable
license/model terms, the function actually needed, added install/runtime cost,
interface and state owner, and one relevant runtime check. A short comparison of
the minimum viable options is sufficient; do not start a broad architecture
search or full-framework integration merely because a paper uses it.
