"""Full-rank value transport into a frozen fake backbone; no model assets."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alice_memory.bridge import BridgeStateError
from alice_memory.checkpoints import memory_class_for, restore_bridge, runtime_class_for
from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_memory.payload_matrix import ParametricPayloadMemory
from alice_memory.payload_runtime import PayloadMemoryRuntime
from alice_memory.runtime import Fact, Question
from alice_runtime.loading import LoadedQwen


class Tokenizer:
    VALUES = {"quartz": 3, "ivory": 4, "copper": 5, "saffron": 6}

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        if text.startswith(" ") and text[1:] in self.VALUES:
            return [self.VALUES[text[1:]]]
        return [1 + ord(character) % 30 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"token-{token}" for token in ids)


class Encoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=x.cumsum(1))


class FrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 2048)
        self.model = Encoder(self.embedding)
        self.head = nn.Linear(2048, 32)
        self.config = SimpleNamespace(eos_token_id=0)
        self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, labels=None, logits_to_keep=0, **kwargs):
        assert labels is None  # Runtime tests never teacher-force an answer.
        logits = self.head(self.model(inputs_embeds=inputs_embeds).last_hidden_state)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(loss=None, logits=logits)


def make_runtime():
    torch.manual_seed(71)
    model = FrozenLM()
    bridge = PayloadEmbeddingBridge(bindings={"backbone": "fake-frozen-2048"}).freeze()
    memory = ParametricPayloadMemory(latent_basis_id=bridge.basis_id)
    return PayloadMemoryRuntime(LoadedQwen(model, Tokenizer(), {}), bridge, memory)


def capture(app, request, mode="on"):
    seen = []

    def record(_module, _args, kwargs):
        assert kwargs.get("labels") is None
        assert kwargs.get("use_cache") is False
        assert "past_key_values" not in kwargs
        seen.append(kwargs["inputs_embeds"].detach().clone())

    hook = app.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = app.answer(request, mode, max_new_tokens=1)
    finally:
        hook.remove()
    assert len(seen) == 1
    return result, seen[0]


def expected_embedding(app, value):
    return app.model.get_input_embeddings().weight[Tokenizer.VALUES[value]].detach()


def payload_slot(app, request, embeddings):
    prefix = f"Question: {request.text}\nAnswer:"
    offset = len(app.tokenizer.encode(prefix, add_special_tokens=False))
    return embeddings[0, offset]


def test_payload_acquisition_changes_only_memory_and_query_does_not_reopen_value():
    app = make_runtime()
    cue = "What is the assigned hue of item K-73?"
    before_theta = app.bridge.parameter_digest()
    before_qwen = {name: p.detach().clone() for name, p in app.model.named_parameters()}
    before_memory = app.memory.content_digest()
    app.acquire(Fact("user", cue, "quartz"))
    assert app.memory.content_digest() != before_memory
    request = Question("Which hue was assigned to item K-73?")
    key = app.key_for(request.text)
    readout = app.memory.read("user", key)
    assert readout.shape == (17, 128)
    assert torch.all(readout.norm(dim=-1) > 0)
    app.tokenizer.seen.clear()

    def forbidden_value_path(_text):
        raise AssertionError("Query must not tokenize/retrieve acquired value text")

    app._value_feature = forbidden_value_path
    after_write = app.memory.content_digest()
    result, embeddings = capture(app, request)
    assert result["memory_applied"]
    assert result["speaker_source_id"] == result["knowledge_source_id"] == "user"
    assert torch.allclose(payload_slot(app, request, embeddings), expected_embedding(app, "quartz"),
                          atol=5e-5, rtol=5e-6)
    assert all(value not in text for text in app.tokenizer.seen for value in Tokenizer.VALUES)
    assert app.memory.content_digest() == after_write
    assert app.bridge.parameter_digest() == before_theta
    assert all(p.grad is None and torch.equal(before_qwen[name], p)
               for name, p in app.model.named_parameters())


def test_payload_off_empty_base_bypass_and_multi_token_rejection_are_state_safe():
    app = make_runtime()
    request = Question("What is the assigned hue of item B-19?")
    base, baseline = capture(app, request, "base")
    for mode in ("off", "empty", "on"):
        result, embeddings = capture(app, request, mode)
        assert not result["memory_applied"]
        assert torch.equal(embeddings, baseline)
        assert result["text"] == base["text"]
    app.acquire(Fact("user", request.text, "ivory"))
    before = app.memory.content_digest()
    for mode in ("off", "empty", "base"):
        result, embeddings = capture(app, request, mode)
        assert not result["memory_applied"]
        assert torch.equal(embeddings, baseline)
    with pytest.raises(ValueError, match="one-token"):
        app.acquire(Fact("user", request.text, "two words"))
    assert app.memory.content_digest() == before
    assert app.bridge.assert_frozen() is None
    app.assert_qwen_frozen()


def test_payload_source_swap_correction_and_frozen_bridge_controls():
    app = make_runtime()
    cue = "What is the assigned hue of item S-42?"
    app.acquire(Fact("user", cue, "quartz"))
    app.acquire(Fact("person:1", cue, "ivory"))
    theta = app.bridge.parameter_digest()
    user, other = Question(cue), Question(cue, "user", "person:1")
    first, original = capture(app, user)
    second, other_input = capture(app, other)
    assert first["knowledge_source_id"] == "user"
    assert second["knowledge_source_id"] == "person:1"
    assert torch.allclose(payload_slot(app, user, original), expected_embedding(app, "quartz"),
                          atol=5e-5, rtol=5e-6)
    assert torch.allclose(payload_slot(app, other, other_input), expected_embedding(app, "ivory"),
                          atol=5e-5, rtol=5e-6)
    state = app.memory.matrix.detach().clone()
    a, b = app.memory.source_index("user"), app.memory.source_index("person:1")
    with torch.no_grad():
        app.memory.matrix[a].copy_(state[b])
        app.memory.matrix[b].copy_(state[a])
    _, swapped = capture(app, user)
    assert not torch.equal(payload_slot(app, user, original), payload_slot(app, user, swapped))
    assert torch.allclose(payload_slot(app, user, swapped), expected_embedding(app, "ivory"),
                          atol=5e-5, rtol=5e-6)
    before_other = app.memory.matrix[b].detach().clone()
    app.acquire(Fact("user", cue, "copper"))
    _, corrected = capture(app, user)
    assert torch.allclose(payload_slot(app, user, corrected), expected_embedding(app, "copper"),
                          atol=5e-5, rtol=5e-6)
    assert torch.equal(app.memory.matrix[b], before_other)
    assert app.bridge.parameter_digest() == theta
    before = app.memory.content_digest()
    app.bridge.train().requires_grad_(True)
    with pytest.raises(BridgeStateError, match="frozen"):
        app.acquire(Fact("user", cue, "saffron"))
    assert app.memory.content_digest() == before


def test_payload_schema_factories_restore_all_heads_and_refuse_legacy_training(tmp_path):
    app = make_runtime()
    cue = "What is the assigned hue of item P-82?"
    app.acquire(Fact("user", cue, "saffron"))
    app.bridge.save(tmp_path / "bridge")
    app.memory.save(tmp_path / "memory")
    bridge = restore_bridge(tmp_path / "bridge", latent_basis_id=app.bridge.basis_id,
                            expected_bindings=app.bridge.bindings)
    assert isinstance(bridge, PayloadEmbeddingBridge)
    assert memory_class_for(bridge) is ParametricPayloadMemory
    assert runtime_class_for(bridge) is PayloadMemoryRuntime
    memory = memory_class_for(bridge).load(tmp_path / "memory", latent_basis_id=bridge.basis_id)
    assert memory.matrix.shape == (4, 17, 128, 128)
    assert torch.equal(memory.matrix, app.memory.matrix)
    assert torch.all(memory.matrix[memory.source_index("user")].flatten(1).norm(dim=-1) > 0)
    restored = runtime_class_for(bridge)(LoadedQwen(app.model, Tokenizer(), {}), bridge, memory)
    request = Question(cue)
    _, original = capture(app, request)
    _, after_restore = capture(restored, request)
    assert torch.equal(original, after_restore)
    assert restored.bridge.parameter_digest() == app.bridge.parameter_digest()
    before = restored.memory.content_digest()
    restored.bridge.train().requires_grad_(True)
    with pytest.raises(RuntimeError, match="separate bounded fitting"):
        restored.training_loss([Fact("user", cue, "saffron")], request, "saffron")
    assert restored.memory.content_digest() == before
