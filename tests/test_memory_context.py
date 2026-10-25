"""Explicit source provenance at complete message token boundaries."""

from dataclasses import FrozenInstanceError
import pytest
import torch

from alice_runtime.bridge import BridgeContext, BridgeError, MessageSpan
from alice_runtime.loading import LoadedQwen
from alice_runtime.runtime import AliceRuntime


class CharacterTokenizer:
    def __init__(self):
        self.segments = []

    def encode(self, text, add_special_tokens=False):
        self.segments.append(text)
        return [ord(character) for character in text]


class EmbeddingsOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(256, 4)
        self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.embeddings


def app(max_context_tokens=256):
    return AliceRuntime(LoadedQwen(EmbeddingsOnly(), CharacterTokenizer(), {}),
                        max_context_tokens=max_context_tokens)


def test_context_is_backward_compatible_and_spans_are_immutable():
    legacy = BridgeContext("session", ("user",))
    assert legacy.current_source_id is None
    assert legacy.knowledge_source_id is None
    assert legacy.message_spans == ()
    span = MessageSpan("user", 0, 12)
    with pytest.raises(FrozenInstanceError):
        span.source_id = "person:1"
    with pytest.raises(BridgeError):
        MessageSpan("user", 12, 12)


def test_spans_identify_actual_selected_messages_and_not_generated_prefix():
    runtime = app()
    runtime.store.add_person(1)
    session = runtime.store.new_session()
    runtime.store.add_message(session, "user", "Question mentions person:1")
    runtime.store.add_message(session, "person:1", "An independent report")
    messages = runtime.store.messages(session)
    inputs, context, dropped = runtime._prepare(session, messages)
    assert not dropped
    assert context.source_ids == ("user", "person:1")
    assert context.current_source_id == "person:1"
    assert context.knowledge_source_id is None  # Not inferred from mentions or speaker.
    rendered = [
        "User [source_id=user]: Question mentions person:1\n",
        "Another Person #1 [source_id=person:1]: An independent report\n",
    ]
    ids = inputs[0].tolist()
    assert context.message_spans[0].start_token == 0
    assert context.message_spans[0].end_token == context.message_spans[1].start_token
    for span, text in zip(context.message_spans, rendered, strict=True):
        assert "".join(map(chr, ids[span.start_token:span.end_token])) == text
    end = context.message_spans[-1].end_token
    assert "".join(map(chr, ids[end:])) == "Agent [source_id=agent]:"
    assert "".join(map(chr, ids)) == "".join(rendered) + "Agent [source_id=agent]:"


def test_eviction_rebases_spans_and_new_session_has_no_old_source():
    runtime = app(max_context_tokens=128)
    runtime.store.add_person(2)
    session = runtime.store.new_session()
    runtime.store.add_message(session, "person:2", "x" * 60)
    runtime.store.add_message(session, "user", "Current")
    inputs, context, dropped = runtime._prepare(session, runtime.store.messages(session))
    assert dropped == 1
    assert context.current_source_id == "user"
    assert context.source_ids == ("user",)
    assert context.message_spans == (MessageSpan("user", 0, len("User [source_id=user]: Current\n")),)
    assert inputs.shape[1] <= 128

    fresh = runtime.store.new_session()
    runtime.store.add_message(fresh, "user", "Fresh")
    _, fresh_context, _ = runtime._prepare(fresh, runtime.store.messages(fresh))
    assert fresh_context.session_id != context.session_id
    assert all(span.source_id == "user" for span in fresh_context.message_spans)
    assert fresh_context.knowledge_source_id is None


def test_query_source_is_distinct_from_the_current_speaker():
    context = BridgeContext("new", ("user",), current_source_id="user",
                            knowledge_source_id="person:1",
                            message_spans=(MessageSpan("user", 0, 10),))
    assert context.current_source_id == "user"
    assert context.knowledge_source_id == "person:1"
