"""Public AliceRuntime memory lifecycle and inference wiring, without model assets.

The tiny frozen model does not establish cognitive recall. These tests inspect
the continuous slot reaching generation; real CLI/Qwen checks provide behavior.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_runtime.loading import LoadedQwen
from alice_runtime.runtime import AliceRuntime


CUE = "What is the assigned label of box Q?"
PARAPHRASE = "Which label was assigned to box Q?"


class Tokenizer:
    VALUES = {"quartz": 3, "ivory": 4, "copper": 5}
    pad_token_id = 0

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        if text.startswith(" ") and text[1:] in self.VALUES:
            return [self.VALUES[text[1:]]]
        return [1 + ord(c) % 30 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(f"token-{token}" for token in tokens)


class Encoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
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

    def forward(self, input_ids=None, inputs_embeds=None, labels=None,
                logits_to_keep=0, **kwargs):
        assert labels is None
        assert kwargs.get("use_cache") is False
        assert "past_key_values" not in kwargs
        h = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds).last_hidden_state
        logits = self.head(h)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    def generate(self, input_ids, **kwargs):
        token = self.forward(input_ids=input_ids, use_cache=False).logits[:, -1].argmax(-1)
        return torch.cat((input_ids, token[:, None]), dim=1)


@pytest.fixture
def bundle(tmp_path):
    bridge_path = tmp_path / "bridge"
    bridge = PayloadEmbeddingBridge()
    bridge.basis_id = "alice.memory_payload/17x128/" + bridge.parameter_digest()
    bridge.freeze().save(bridge_path)
    apps = []

    def create(*, state="experience", attached=True):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            loaded = LoadedQwen(FrozenLM(), Tokenizer(), {})
        app = AliceRuntime(loaded, max_context_tokens=256)
        apps.append(app)
        if attached:
            app.attach_memory(bridge_path, tmp_path / state)
        return app

    yield create, bridge_path, tmp_path / "experience"
    for app in apps:
        app.close()


def capture(app, session, question=CUE, *, target="user"):
    inputs = []

    def record(_module, _args, kwargs):
        assert kwargs.get("labels") is None
        assert kwargs["use_cache"] is False
        assert "past_key_values" not in kwargs
        inputs.append(kwargs["inputs_embeds"].detach().clone())

    hook = app.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = app.respond(session, "user", question, 1, knowledge_source_id=target)
    finally:
        hook.remove()
    assert len(inputs) == 1
    return result, inputs[0]


def slot(app, embeddings, question=CUE):
    index = len(app.tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False))
    return embeddings[0, index]


def value_embedding(app, value):
    return app.model.get_input_embeddings().weight[Tokenizer.VALUES[value]].detach()


def test_public_remember_persists_only_neural_state_and_reopens_independent_of_sessions(bundle):
    create, _bridge_path, state_path = bundle
    app = create()
    session = app.store.new_session("first")
    neural = app.neural_memory
    before_qwen = {n: p.detach().clone() for n, p in app.model.named_parameters()}
    before_bridge = neural.bridge.parameter_digest()
    before_memory = neural.memory.content_digest()
    receipt = app.remember("user", CUE, "quartz")
    assert receipt["saved"] and receipt["persistent_m_changed"]
    assert receipt["before"] == before_memory
    assert receipt["after"] == app.memory_status()["memory_digest"] != before_memory
    assert app.store.messages(session) == ()
    assert app.memory_status()["persisted"]
    assert neural.bridge.parameter_digest() == before_bridge
    assert all(p.grad is None and torch.equal(before_qwen[n], p)
               for n, p in app.model.named_parameters())
    metadata = json.loads((state_path / "memory.json").read_text())
    assert metadata["schema"] == "alice.memory_payload/v1"
    assert set(metadata).isdisjoint({"facts", "messages", "transcript", "token_ids", "values"})
    assert CUE not in json.dumps(metadata) and "quartz" not in json.dumps(metadata)
    saved_digest = neural.memory.content_digest()
    app.close()

    restored = create()
    fresh_session = restored.store.new_session("fresh")
    assert restored.memory_status()["memory_digest"] == saved_digest
    assert restored.store.messages(fresh_session) == ()
    result, embeddings = capture(restored, fresh_session, PARAPHRASE)
    assert result.memory_info["memory_applied"]
    assert result.memory_info["knowledge_source_id"] == "user"
    assert torch.allclose(slot(restored, embeddings, PARAPHRASE),
                          value_embedding(restored, "quartz"), atol=5e-5, rtol=5e-6)


def test_memory_queries_use_current_message_without_transcript_or_value_text(bundle, monkeypatch):
    create, _bridge_path, _state_path = bundle
    app = create()
    session = app.store.new_session("history")
    app.remember("user", CUE, "quartz")
    private = "PREVIOUS_SESSION_SECRET_that_must_not_reach_the_model"
    app.store.add_message(session, "user", private * 40)
    app.tokenizer.seen.clear()
    monkeypatch.setattr(app.neural_memory.engine, "_value_feature", lambda _text:
                        (_ for _ in ()).throw(AssertionError("Query reopened acquired value")))
    result, embeddings = capture(app, session, PARAPHRASE)
    assert result.memory_info["mode"] == "on"
    assert result.memory_info["memory_applied"]
    assert torch.allclose(slot(app, embeddings, PARAPHRASE), value_embedding(app, "quartz"),
                          atol=5e-5, rtol=5e-6)
    assert not any(private in text or "quartz" in text for text in app.tokenizer.seen)
    digest = app.memory_status()["memory_digest"]
    app.store.clear_session(session)
    second = app.store.new_session("new-session")
    again, _ = capture(app, second)
    assert again.memory_info["memory_digest"] == digest


def test_explicit_source_correction_and_off_empty_are_independent_and_state_safe(bundle):
    create, _bridge_path, _state_path = bundle
    app = create()
    session = app.store.new_session()
    assert "person:1" in app.store.participants
    app.remember("user", CUE, "quartz")
    app.remember("person:1", CUE, "ivory")
    person_plane = app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("person:1")].clone()
    other, embeddings = capture(app, session, target="person:1")
    assert other.memory_info["knowledge_source_id"] == "person:1"
    assert torch.allclose(slot(app, embeddings), value_embedding(app, "ivory"),
                          atol=5e-5, rtol=5e-6)
    digest = app.memory_status()["memory_digest"]
    for mode in ("off", "empty"):
        app.set_memory_mode(mode)
        result, baseline = capture(app, session)
        assert result.memory_info["mode"] == mode
        assert not result.memory_info["memory_applied"]
        assert result.memory_info["memory_digest"] == digest
        expected_length = len(app.tokenizer.encode(f"Question: {CUE}\nAnswer:", add_special_tokens=False))
        assert baseline.shape[1] == expected_length
    app.set_memory_mode("on")
    app.remember("user", CUE, "copper")
    corrected, embeddings = capture(app, session)
    assert corrected.memory_info["memory_applied"]
    assert torch.allclose(slot(app, embeddings), value_embedding(app, "copper"),
                          atol=5e-5, rtol=5e-6)
    assert torch.equal(person_plane, app.neural_memory.memory.matrix[
        app.neural_memory.memory.source_index("person:1")])


def test_failed_write_rolls_back_parameters_and_never_acknowledges_or_adds_transcript(bundle, monkeypatch):
    create, _bridge_path, state_path = bundle
    app = create()
    session = app.store.new_session()
    app.remember("user", CUE, "quartz")
    before = app.memory_status()["memory_digest"]
    manifest = (state_path / "memory.json").read_bytes()
    matrix = app.neural_memory.memory.matrix.detach().clone()
    monkeypatch.setattr(app.neural_memory.memory, "save", lambda *_:
                        (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises((OSError, RuntimeError, ValueError), match="disk full"):
        app.remember("user", CUE, "copper")
    assert app.memory_status()["memory_digest"] == before
    assert torch.equal(matrix, app.neural_memory.memory.matrix)
    assert (state_path / "memory.json").read_bytes() == manifest
    assert app.store.messages(session) == ()


def test_invalid_request_and_generation_failure_do_not_commit_session_or_change_memory(bundle, monkeypatch):
    create, _bridge_path, _state_path = bundle
    app = create()
    session = app.store.new_session()
    app.remember("user", CUE, "quartz")
    before = app.memory_status()["memory_digest"]
    for operation in (
        lambda: app.remember("unknown", CUE, "ivory"),
        lambda: app.remember("user", CUE, "two words"),
        lambda: app.respond(session, "user", CUE, 17),
        lambda: app.respond(session, "user", CUE, 1, knowledge_source_id="unknown"),
        lambda: app.set_memory_mode("reset"),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            operation()
        assert app.store.messages(session) == ()
        assert app.memory_status()["memory_digest"] == before
    monkeypatch.setattr(app.neural_memory, "respond", lambda *_a, **_k:
                        (_ for _ in ()).throw(RuntimeError("generation failed")))
    with pytest.raises(RuntimeError, match="generation failed"):
        app.respond(session, "user", CUE, 1)
    assert app.store.messages(session) == ()
    assert app.memory_status()["memory_digest"] == before


def test_memory_freeze_and_external_embedding_port_are_enforced(bundle):
    create, bridge_path, state_path = bundle
    app = create(attached=False)
    app.bridge.attach(lambda x, _ctx: x)
    with pytest.raises((ValueError, RuntimeError), match="[Bb]ridge|port|external wiring"):
        app.attach_memory(bridge_path, state_path)
    assert app.neural_memory is None
    app.bridge.detach()
    app.attach_memory(bridge_path, state_path)
    session = app.store.new_session()
    before = app.memory_status()["memory_digest"]
    app.model.train()
    with pytest.raises(RuntimeError, match="frozen"):
        app.remember("user", CUE, "quartz")
    with pytest.raises(RuntimeError, match="frozen"):
        app.respond(session, "user", CUE, 1)
    assert app.store.messages(session) == ()
    assert app.memory_status()["memory_digest"] == before
    app.model.eval()
    app.neural_memory.bridge.train()
    with pytest.raises((ValueError, RuntimeError), match="frozen"):
        app.remember("user", CUE, "quartz")
    assert app.memory_status()["memory_digest"] == before


def test_without_memory_the_existing_runtime_still_responds_and_has_no_memory_mode(bundle):
    create, _bridge_path, _state_path = bundle
    app = create(attached=False)
    session = app.store.new_session()
    result = app.respond(session, "user", "Hi", 32)
    assert result.memory_info is None
    assert app.store.messages(session)[0].text == "Hi"
    with pytest.raises((ValueError, RuntimeError)):
        app.set_memory_mode("on")
    with pytest.raises((ValueError, RuntimeError)):
        app.remember("user", CUE, "quartz")
