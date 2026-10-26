"""Dense-only lexical memory assembly and exact relevance bypass, without Qwen assets."""

import pytest
import torch

from alice_memory.checkpoints import restore_bridge
from alice_memory.lexical_bridge import LexicalEmbeddingBridge
from alice_memory.matrix import AssociativeMemory
from alice_memory.runtime import Fact, MemoryRuntime, Question
from alice_runtime.loading import LoadedQwen
from test_memory_runtime import FrozenLM, Tokenizer


def make_lexical_runtime():
    torch.manual_seed(16)
    model, tokenizer = FrozenLM(), Tokenizer()
    bridge = LexicalEmbeddingBridge(hidden_size=12).freeze()
    memory = AssociativeMemory(latent_basis_id=bridge.basis_id)
    return MemoryRuntime(LoadedQwen(model, tokenizer, {}), bridge, memory)


def capture_one(app, question, mode):
    seen = []

    def record(_module, _args, kwargs):
        assert kwargs.get("labels") is None
        assert kwargs.get("use_cache") is False
        assert "past_key_values" not in kwargs
        seen.append(kwargs["inputs_embeds"].detach().clone())

    handle = app.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = app.answer(question, mode, max_new_tokens=1)
    finally:
        handle.remove()
    assert len(seen) == 1
    return result, seen[0]


def test_off_empty_base_and_weak_on_use_exact_unconditioned_input():
    app = make_lexical_runtime()
    request = Question("Which room holds the object?")
    base, base_embeddings = capture_one(app, request, "base")
    for mode in ("off", "empty", "on"):
        result, embeddings = capture_one(app, request, mode)
        assert not result["memory_applied"]
        assert torch.equal(embeddings, base_embeddings)
        assert result["text"] == base["text"]
    key = app.bridge.key(app._feature(app._question_ids(request.text)))
    value = app.bridge.value(torch.arange(1., 13.))
    with torch.no_grad():
        app.memory.matrix[app.memory.source_index("user")].copy_(.4 * torch.outer(value, key))
    before = app.memory.content_digest()
    result, weak_embeddings = capture_one(app, request, "on")
    assert result["readout_norm"] < .6 and not result["memory_applied"]
    assert torch.equal(weak_embeddings, base_embeddings)
    assert app.memory.content_digest() == before


def test_acquired_content_reaches_only_dense_slot_with_fixed_state_and_source():
    app = make_lexical_runtime()
    cue = "Where is the parcel?"
    app.acquire(Fact("user", cue, "private-red"))
    app.acquire(Fact("person:1", cue, "private-blue"))
    request = Question(cue, "user", "person:1")
    q = app.bridge.key(app._feature(app._question_ids(cue)))
    readout = app.memory.read("person:1", q)
    expected_slot = app.bridge.Wo(readout).detach()
    before_bridge = app.bridge.parameter_digest()
    before_memory = app.memory.content_digest()
    before_qwen = {name: p.detach().clone() for name, p in app.model.named_parameters()}
    p = app._prefix(app._ids(f"Question: {cue}\nKnown answer:"))
    tail = app._prefix(app._ids("\nAnswer:"))
    app.tokenizer.seen.clear()

    def forbidden_value_path(_text):
        raise AssertionError("Query must not re-open acquired values")

    app._value_feature = forbidden_value_path
    result, embeddings = capture_one(app, request, "on")
    assert result["memory_applied"] and result["knowledge_source_id"] == "person:1"
    assert result["speaker_source_id"] == "user"
    assert embeddings.shape[1] == p.shape[1] + 1 + tail.shape[1]
    assert torch.equal(embeddings[:, :p.shape[1]], p)
    assert torch.equal(embeddings[0, p.shape[1]], expected_slot)
    assert torch.equal(embeddings[:, -tail.shape[1]:], tail)
    assert all("private-red" not in text and "private-blue" not in text for text in app.tokenizer.seen)
    assert app.memory.content_digest() == before_memory
    assert app.bridge.parameter_digest() == before_bridge
    assert all(p.grad is None and torch.equal(before_qwen[name], p)
               for name, p in app.model.named_parameters())
    app.bridge.assert_frozen()
    app.assert_qwen_frozen()


def test_format_only_control_has_zero_slot_independent_of_memory_content():
    app = make_lexical_runtime()
    request = Question("What is the object color?")
    first, empty_format = capture_one(app, request, "format_only")
    app.acquire(Fact("user", request.text, "private-color"))
    before = app.memory.content_digest()
    second, acquired_format = capture_one(app, request, "format_only")
    _, base_embeddings = capture_one(app, request, "base")
    assert torch.equal(empty_format, acquired_format)
    assert first["text"] == second["text"]
    assert not first["memory_applied"] and not second["memory_applied"]
    assert first["readout_norm"] == second["readout_norm"] == 0
    assert empty_format.shape[1] > base_embeddings.shape[1]
    slot_offset = len(app.tokenizer.encode(f"Question: {request.text}\nKnown answer:"))
    assert torch.count_nonzero(empty_format[:, slot_offset]) == 0
    assert app.memory.content_digest() == before


def test_schema_dispatch_restores_lexical_state_and_old_training_path_is_refused(tmp_path):
    app = make_lexical_runtime()
    app.acquire(Fact("user", "Where is the package?", "private-room"))
    app.bridge.save(tmp_path / "bridge")
    app.memory.save(tmp_path / "memory")
    bridge = restore_bridge(tmp_path / "bridge", latent_basis_id=app.bridge.basis_id,
                            expected_bindings=app.bridge.bindings)
    assert isinstance(bridge, LexicalEmbeddingBridge)
    memory = AssociativeMemory.load(tmp_path / "memory", latent_basis_id=bridge.basis_id)
    assert bridge.parameter_digest() == app.bridge.parameter_digest()
    assert memory.content_digest() == app.memory.content_digest()
    restored = MemoryRuntime(LoadedQwen(app.model, Tokenizer(), {}), bridge, memory)
    request = Question("Where is the package?")
    _, original_payload = capture_one(app, request, "on")
    _, restored_payload = capture_one(restored, request, "on")
    assert torch.equal(original_payload, restored_payload)
    memory_before = memory.content_digest()
    bridge.train().requires_grad_(True)
    with pytest.raises(RuntimeError, match="separate bounded fitting"):
        restored.training_loss([Fact("user", request.text, "private-room")], request, "private-room")
    assert memory.content_digest() == memory_before
