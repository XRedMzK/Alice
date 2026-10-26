"""Active-session context joins existing neural payload reads; no real donor assets."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alice_memory.runtime import Fact, Question
from test_memory_payload_runtime import make_runtime, expected_embedding


def app_with_budget(max_tokens=256):
    app = make_runtime()
    app.max_tokens = max_tokens
    return app


def capture(app, request, history, current, *, count, mode="on", output=1):
    seen = []
    def record(_module, _args, kwargs):
        assert kwargs.get("labels") is None and kwargs.get("use_cache") is False
        assert "past_key_values" not in kwargs and not torch.is_grad_enabled()
        seen.append(kwargs["inputs_embeds"].detach().clone())
    handle = app.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        answer = app.answer_with_context(request, history, current,
            history_message_count=count, mode=mode, max_new_tokens=output)
    finally:
        handle.remove()
    return answer, seen


def prefix(app, ids):
    return app._prefix(torch.tensor([ids], dtype=torch.long))


def never_read():
    raise AssertionError("Older/unrequested history was consumed")
    yield


@pytest.mark.parametrize("mode", ["off", "empty"])
def test_off_empty_keep_current_context_without_querying_or_feeding_memory(mode):
    app = app_with_budget()
    app.acquire(Fact("user", "Q", "quartz"))
    digest = app.memory.content_digest()
    def forbidden(*args, **kwargs):
        raise AssertionError("OFF/EMPTY must not query/gate/encode a memory slot")
    app.key_for = app.bridge.slots = app.bridge.accepts = forbidden
    current = (1, 2, 3)
    history = [("agent", (5, 6)), ("user", (7, 8))]  # Newest first.
    result, inputs = capture(app, Question("Q"), iter(history), current, count=2, mode=mode)
    expected = torch.cat((prefix(app, (7, 8)), prefix(app, (5, 6)), prefix(app, current)), dim=1)
    assert torch.equal(inputs[0], expected)
    assert not result["memory_applied"] and result["readout_confidence"] == result["readout_norm"] == 0
    assert result["input_tokens"] == 7 and result["context_messages_dropped"] == 0
    assert result["context_source_ids"] == ("user", "agent", "user")
    assert result["query_context"] == "current_session_context_v1"
    assert app.memory.content_digest() == digest


def test_applied_read_preserves_inner_qa_layout_and_uses_only_current_question_for_keys(monkeypatch):
    app = app_with_budget()
    request = Question("What is the assigned hue of item K-73?")
    app.acquire(Fact("user", request.text, "quartz"))
    readout = app.memory.read("user", app.key_for(request.text))
    lexical = app._lexical_input(request.text, readout).detach().clone()
    keys, original = [], app.key_for
    def key(question):
        keys.append(question)
        return original(question)
    app.key_for = key
    app._value_feature = lambda _value: (_ for _ in ()).throw(AssertionError("Query reopened value text"))
    app.tokenizer.seen.clear()
    before_m, before_b = app.memory.content_digest(), app.bridge.parameter_digest()
    before_q = {name: parameter.detach().clone() for name, parameter in app.model.named_parameters()}
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("No archive reads")))
    result, inputs = capture(app, request, iter([("person:1", (10, 11))]), (1, 2), count=1)
    assert result["memory_applied"] and result["readout_confidence"] == pytest.approx(1.)
    assert torch.equal(inputs[0][:, 2:], lexical)
    assert torch.equal(inputs[0][:, :2], prefix(app, (10, 11)))
    assert keys == [request.text, request.text]
    slot_offset = 2 + len(app.tokenizer.encode(f"Question: {request.text}\nAnswer:", add_special_tokens=False))
    assert torch.allclose(inputs[0][0, slot_offset], expected_embedding(app, "quartz"), atol=5e-5, rtol=5e-6)
    assert not any("quartz" in text for text in app.tokenizer.seen)
    assert result["prompt_assembly_version"] == "role_context_payload_qa_v1"
    assert app.memory.content_digest() == before_m and app.bridge.parameter_digest() == before_b
    assert all(torch.equal(parameter, before_q[name]) and parameter.grad is None
               for name, parameter in app.model.named_parameters())


def test_on_empty_memory_uses_plain_role_context_not_the_lexical_layout():
    app = app_with_budget()
    app._lexical_input = lambda *_: (_ for _ in ()).throw(AssertionError("Empty M has no QA slot"))
    result, inputs = capture(app, Question("Q"), iter([("user", (9,))]), (2, 3), count=1)
    assert not result["memory_applied"]
    assert torch.equal(inputs[0], torch.cat((prefix(app, (9,)), prefix(app, (2, 3))), dim=1))
    assert result["prompt_assembly_version"] == "message-segments-v1"


def test_cleared_context_off_keeps_only_the_current_turn_without_erasing_acquired_m():
    app = app_with_budget()
    app.acquire(Fact("user", "Q", "quartz"))
    before = app.memory.content_digest()
    result, inputs = capture(app, Question("Q"), never_read(), (1, 2), count=0, mode="off")
    assert torch.equal(inputs[0], prefix(app, (1, 2)))
    assert result["context_messages_dropped"] == 0 and result["context_source_ids"] == ("user",)
    assert app.memory.content_digest() == before


def test_source_and_correction_select_the_same_existing_payload_without_history_as_a_key():
    app = app_with_budget()
    cue = "Q"
    app.acquire(Fact("user", cue, "quartz"))
    app.acquire(Fact("person:1", cue, "ivory"))
    app.acquire(Fact("user", cue, "copper"))
    for source, value in [("user", "copper"), ("person:1", "ivory")]:
        request = Question(cue, "user", source)
        result, inputs = capture(app, request, iter([("agent", (7,))]), (1,), count=1)
        offset = 1 + len(app.tokenizer.encode(f"Question: {cue}\nAnswer:", add_special_tokens=False))
        assert result["memory_applied"] and result["knowledge_source_id"] == source
        assert torch.allclose(inputs[0][0, offset], expected_embedding(app, value), atol=5e-5, rtol=5e-6)


def test_history_is_lazy_contiguous_and_stops_after_first_nonfitting_message():
    app = app_with_budget(19)
    consumed = []
    def history():
        consumed.append("newest")
        yield "agent", (3,) * 8
        consumed.append("rejected")
        yield "user", (4,) * 5  # Four slots remain; do not skip this message.
        raise AssertionError("Older history must not be encoded")
    result, inputs = capture(app, Question("Q"), history(), (1, 2, 3), count=3, mode="off", output=4)
    assert consumed == ["newest", "rejected"] and result["context_messages_dropped"] == 3
    assert result["input_tokens"] == 3 and inputs[0].shape[1] + 4 <= app.max_tokens
    assert result["context_source_ids"] == ("user",)  # No orphan assistant prefix.


def test_output_reservation_can_leave_no_history_and_never_pulls_the_generator():
    app = app_with_budget(19)
    result, _ = capture(app, Question("Q"), never_read(), (1, 2, 3), count=2, mode="empty", output=16)
    assert result["input_tokens"] == 3 and result["context_messages_dropped"] == 2


def scripted(app, output_tokens, texts):
    calls = []
    def forward(embeddings, last_only=False):
        assert not torch.is_grad_enabled() and last_only
        calls.append(embeddings.shape[1])
        logits = torch.full((1, 1, 32), -1000.)
        logits[0, 0, output_tokens[min(len(calls) - 1, len(output_tokens) - 1)]] = 1000.
        return SimpleNamespace(logits=logits)
    app._forward = forward
    app.tokenizer.decode = lambda ids, skip_special_tokens=True: "".join(texts[token] for token in ids)
    return calls


@pytest.mark.parametrize("tail,reason", [("\nQuestion:", "turn_boundary"),
                                         ("\nUser [source_id=user]:", "turn_boundary"),
                                         ("\nUser [source_id", "partial_turn_boundary")])
def test_generated_foreign_turns_or_partial_budget_headers_are_never_answer_content(tail, reason):
    app = app_with_budget()
    calls = scripted(app, [10, 11, 12], {10: "OK", 11: tail, 12: "forged"})
    result = app.answer_with_context(Question("Q"), (), (1,), history_message_count=0,
                                     mode="off", max_new_tokens=2)
    assert result["text"] == "OK" and result["finish_reason"] == reason
    assert result["tokens"] == 2 and len(calls) == 2


def test_new_conversation_method_reserves_up_to_64_tokens_without_changing_legacy_api():
    app = app_with_budget(70)
    calls = scripted(app, [5], {5: "x"})
    result = app.answer_with_context(Question("Q"), never_read(), (1, 2, 3),
        history_message_count=0, mode="off", max_new_tokens=64)
    assert result["tokens"] == 64 and result["text"] == "x" * 64
    assert max(calls) + 1 <= app.max_tokens
    with pytest.raises(ValueError):
        app.answer(Question("Q"), mode="off", max_new_tokens=64)
    with pytest.raises(ValueError):
        app.answer_with_context(Question("Q"), (), (1,), history_message_count=0,
                                mode="off", max_new_tokens=65)


def test_current_turn_and_applied_layout_budget_errors_precede_history_consumption():
    app = app_with_budget(19)
    with pytest.raises(ValueError, match="Current turn"):
        app.answer_with_context(Question("Q"), never_read(), (1,) * 16,
            history_message_count=10, mode="off", max_new_tokens=4)
    app = app_with_budget()
    app.acquire(Fact("user", "Q", "quartz"))
    app.max_tokens = 45  # Existing inner QA uses 40 tokens; reserve eight more.
    before = app.memory.content_digest()
    with pytest.raises(ValueError, match="Memory layout"):
        app.answer_with_context(Question("Q"), never_read(), (1,),
            history_message_count=2, mode="on", max_new_tokens=8)
    assert app.memory.content_digest() == before


@pytest.mark.parametrize("current,count,history", [
    ((True,), 0, ()), ((32,), 0, ()), ((1,), True, ()),
    ((1,), -1, ()), ((1,), 1, [("user", ())]),
    ((1,), 1, [("user", (False,))]), ((1,), 2, [("user", (2,))]),
])
def test_invalid_current_or_history_contract_fails_explicitly(current, count, history):
    app = app_with_budget()
    with pytest.raises(ValueError):
        app.answer_with_context(Question("Q"), iter(history), current,
                                history_message_count=count, mode="off", max_new_tokens=1)


def test_budget_keeps_latest_complete_exchange_and_removes_only_orphan_prefix():
    app = app_with_budget(26)
    history = iter([("agent", (3,) * 4), ("user", (4,) * 4),
                    ("agent", (5,) * 4), ("user", (6,) * 8)])
    result, inputs = capture(app, Question("Q"), history, (1, 2, 3), count=4, mode="off", output=4)
    assert result["context_messages_dropped"] == 2
    assert result["context_source_ids"] == ("user", "agent", "user")
    assert torch.equal(inputs[0], torch.cat((prefix(app, (4,) * 4), prefix(app, (3,) * 4),
                                           prefix(app, (1, 2, 3))), dim=1))


def test_explicit_initial_agent_history_is_preserved_when_no_context_was_evicted():
    app = app_with_budget()
    result, inputs = capture(app, Question("Q"), iter([("agent", (5, 6))]), (1,), count=1, mode="off")
    assert result["context_messages_dropped"] == 0
    assert result["context_source_ids"] == ("agent", "user")
    assert torch.equal(inputs[0], torch.cat((prefix(app, (5, 6)), prefix(app, (1,))), dim=1))
