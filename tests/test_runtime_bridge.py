"""Critical typed-boundary and transactional checks; no model assets needed."""

from types import SimpleNamespace

import pytest
import torch

from alice_runtime.bridge import BridgeContext, BridgeError, EmbeddingBridgePort
from alice_runtime.loading import LoadedQwen, LoadingError, load_qwen
from alice_runtime.runtime import AliceRuntime, RuntimeInputError
from alice_runtime.sessions import SessionError, SessionStore


def test_bridge_checks_and_hook_cleanup():
    layer = torch.nn.Embedding(12, 4)
    tokens = torch.tensor([[1, 2]])
    baseline = layer(tokens).detach().clone()
    port = EmbeddingBridgePort()
    context = BridgeContext("s", ("user", "person:1"))
    seen = []

    def signal(x, ctx):
        seen.append(ctx)
        x[..., 0] += 1
        return x

    port.attach(signal)
    with port.activation(layer, context):
        assert not torch.equal(layer(tokens), baseline)
    assert seen == [context]
    assert torch.equal(layer(tokens), baseline)
    port.disable()
    with port.activation(layer, context):
        assert torch.equal(layer(tokens), baseline)
    port.enable()
    port.detach()
    assert not layer._forward_hooks
    with pytest.raises(BridgeError, match="No Bridge"):
        port.enable()

    for fn, error in [
        (lambda x, _: x[..., :-1], "shape"),
        (lambda x, _: x.double(), "dtype"),
        (lambda x, _: x.to("meta"), "device"),
        (lambda x, _: x.fill_(float("nan")), "finite"),
    ]:
        port.attach(fn)
        with pytest.raises(BridgeError, match=error):
            with port.activation(layer, context):
                layer(tokens)
        assert not layer._forward_hooks
        assert torch.equal(layer(tokens), baseline)


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(c) % 10 for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "generated" if tokens else ""


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.head = torch.nn.Linear(4, 12)
        self.config = SimpleNamespace(eos_token_id=0)
        self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def generate(self, input_ids, **kwargs):
        logits = self.forward(input_ids).logits[:, -1, :]
        return torch.cat([input_ids, logits.argmax(-1, keepdim=True)], dim=1)


def runtime():
    return AliceRuntime(LoadedQwen(TinyLM(), TinyTokenizer(), {}), max_context_tokens=128)


def test_failed_generation_keeps_session_and_provenance_intact():
    app = runtime()
    person = app.store.add_person(1)
    session = app.store.new_session("main")
    other = app.store.new_session("other")
    seen = []
    app.bridge.attach(lambda x, ctx: seen.append(ctx.source_ids) or x)
    result = app.respond(session, person.source_id, "Hi")
    assert result.source_ids == ("person:1",)
    assert seen == [("person:1",)]
    assert [m.source_id for m in app.store.messages(session)] == ["person:1", "agent"]
    assert not app.store.messages(other)
    before = app.store.messages(session)
    app.bridge.attach(lambda x, _: x.double())
    with pytest.raises(BridgeError):
        app.respond(session, "user", "Hello")
    assert app.store.messages(session) == before
    assert not app.model.get_input_embeddings()._forward_hooks
    with pytest.raises(RuntimeInputError):
        app.respond(session, "user", "x" * 300)
    with pytest.raises(SessionError):
        app.respond(session, "not-a-person", "Hello")
    assert app.store.messages(session) == before
    app.bridge.detach()
    app.model.train()
    with pytest.raises(RuntimeError, match="frozen"):
        app.respond(session, "user", "Hello")
    assert app.store.messages(session) == before


def test_context_eviction_keeps_whole_source_bound_messages():
    app = runtime()
    session = app.store.new_session()
    app.store.add_message(session, "user", "a" * 60)
    result = app.respond(session, "user", "new")
    assert result.context_messages_dropped == 1
    assert result.source_ids == ("user",)
    assert len(app.store.messages(session)) == 3  # Eviction affects input, not history.


def test_missing_snapshot_fails_locally(tmp_path):
    with pytest.raises(LoadingError, match="Local model directory"):
        load_qwen(tmp_path / "absent")
