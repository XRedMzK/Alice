"""Bounded conversational acquisition through public AliceRuntime APIs.

The small frozen backbone verifies state ownership and continuous wiring only.
Real-model CLI restart conversations supply the cognitive behavior evidence.
"""

import json

import pytest
import torch

from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_runtime.loading import LoadedQwen
from alice_runtime.runtime import AliceRuntime
from test_runtime_memory_integration import FrozenLM, Tokenizer as BaseTokenizer


CUE = "What is the assigned label of object a-0?"
EN = "Remember that the label of object a-0 is window."
RU = "Запомни: метка объекта a-0 — window."


class Tokenizer(BaseTokenizer):
    VALUES = {"window": 3, "helmet": 4, "dragon": 5}


@pytest.fixture
def dialogue(tmp_path):
    bridge_path = tmp_path / "bridge"
    bridge = PayloadEmbeddingBridge()
    bridge.basis_id = "alice.memory_payload/17x128/" + bridge.parameter_digest()
    bridge.freeze().save(bridge_path)
    apps = []

    def create():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            loaded = LoadedQwen(FrozenLM(), Tokenizer(), {})
        app = AliceRuntime(loaded, max_context_tokens=256)
        apps.append(app)
        app.attach_memory(bridge_path, tmp_path / "memory")
        return app

    yield create, tmp_path
    for app in apps:
        app.close()


def ask(app, session, text, source="user", **kwargs):
    return app.respond(session, source, text, 1, **kwargs)


def action(result):
    assert result.memory_info is not None
    return result.memory_info.get("dialogue_action")


def digest(app):
    return app.memory_status()["memory_digest"]


def dense_value(app, source="user"):
    with torch.no_grad():
        key = app.neural_memory.engine.key_for(CUE)
        readout = app.neural_memory.memory.read(source, key)
        return app.neural_memory.bridge.slots(key, readout, 1.0)[0]


def expected_value(app, word):
    return app.model.get_input_embeddings().weight[Tokenizer.VALUES[word]].detach()


@pytest.mark.parametrize("sentence,yes", [(EN, "Yes"), (RU, "Да")])
def test_natural_fact_requires_confirmation_then_saves_only_memory(dialogue, sentence, yes):
    create, root = dialogue
    app = create()
    session = app.store.new_session("conversation")
    before = digest(app)
    bridge_before = app.neural_memory.bridge.parameter_digest()
    qwen_before = {name: p.detach().clone() for name, p in app.model.named_parameters()}

    proposal = ask(app, session, sentence)
    assert action(proposal) == "confirmation_required"
    assert proposal.generated_tokens == 0 and proposal.bridge_calls == 0
    assert "window" in proposal.text and "a-0" in proposal.text
    assert digest(app) == before
    assert not (root / "memory" / "memory.json").exists()
    assert app.store.messages(session) == ()

    saved = ask(app, session, yes)
    assert action(saved) == "remembered"
    assert saved.generated_tokens == 0
    assert digest(app) != before
    assert app.memory_status()["persisted"]
    assert (root / "memory" / "memory.json").is_file()
    assert app.store.messages(session) == ()
    assert app.neural_memory.bridge.parameter_digest() == bridge_before
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, qwen_before[name])
               for name, p in app.model.named_parameters())
    assert torch.allclose(dense_value(app), expected_value(app, "window"), atol=2e-5, rtol=5e-6)


def test_acquisition_uses_actual_speaker_not_queried_source_or_embedded_claim(dialogue):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    ask(app, session, EN, knowledge_source_id="person:1")
    ask(app, session, "yes", knowledge_source_id="person:1")
    assert torch.allclose(dense_value(app, "user"), expected_value(app, "window"), atol=2e-5, rtol=5e-6)
    assert torch.count_nonzero(dense_value(app, "person:1")) == 0
    user_plane = app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("user")].clone()

    proposal = ask(app, session, EN.replace("window", "helmet"), source="person:1",
                   knowledge_source_id="user")
    assert action(proposal) == "confirmation_required"
    ask(app, session, "yes", source="person:1", knowledge_source_id="user")
    assert torch.allclose(dense_value(app, "person:1"), expected_value(app, "helmet"), atol=2e-5, rtol=5e-6)
    assert torch.equal(user_plane, app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("user")])


def test_another_source_or_session_cannot_confirm_someone_elses_proposal(dialogue):
    create, _root = dialogue
    app = create()
    first, second = app.store.new_session("first"), app.store.new_session("second")
    before = digest(app)
    ask(app, first, EN)
    wrong_source = ask(app, first, "yes", source="person:1")
    assert action(wrong_source) == "no_pending" and digest(app) == before
    ask(app, first, EN)
    wrong_session = ask(app, second, "yes")
    assert action(wrong_session) == "no_pending" and digest(app) == before
    ask(app, first, EN)
    assert action(ask(app, first, "yes")) == "remembered"


def test_ui_participant_without_memory_plane_gets_source_limitation(dialogue):
    create, _root = dialogue
    app = create()
    app.store.add_person(3)
    session = app.store.new_session()
    before = digest(app)
    result = ask(app, session, EN, source="person:3")
    assert action(result) == "unsupported" and "source" in result.text.casefold()
    assert digest(app) == before
    assert action(ask(app, session, "yes", source="person:3")) == "no_pending"
    assert digest(app) == before


def test_ambiguity_does_not_guess_and_rephrased_fact_still_needs_confirmation(dialogue):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    before = digest(app)
    unclear = ask(app, session, "Запомни: метка объекта a-0 — window или helmet.")
    assert action(unclear) == "clarification_required" and unclear.text
    assert digest(app) == before
    assert action(ask(app, session, "Да")) == "no_pending"
    assert digest(app) == before
    precise = ask(app, session, RU)
    assert action(precise) == "confirmation_required" and digest(app) == before
    assert action(ask(app, session, "Да")) == "remembered"


def test_no_and_intervening_ordinary_message_cancel_unconfirmed_facts(dialogue):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    before = digest(app)
    ask(app, session, EN)
    assert action(ask(app, session, "No")) == "cancelled"
    assert action(ask(app, session, "Yes")) == "no_pending"
    assert digest(app) == before
    ask(app, session, EN)
    ordinary = ask(app, session, "What is two plus two?")
    assert action(ordinary) is None
    assert action(ask(app, session, "yes")) == "no_pending"
    assert digest(app) == before


def test_ordinary_fact_is_not_automatically_acquired_and_multitoken_value_is_explicit(dialogue):
    create, root = dialogue
    app = create()
    session = app.store.new_session()
    before = digest(app)
    ordinary = ask(app, session, "The label of object a-0 is window.")
    assert action(ordinary) is None and digest(app) == before
    unsupported = ask(app, session, 'Remember that the label of object a-0 is "bright window".')
    assert action(unsupported) == "unsupported"
    assert "token" in unsupported.text.casefold() or "токен" in unsupported.text.casefold()
    assert digest(app) == before
    assert not (root / "memory" / "memory.json").exists()
    assert action(ask(app, session, "yes")) == "no_pending"


def test_pending_fact_is_not_serialized_and_does_not_survive_restart(dialogue):
    create, root = dialogue
    app = create()
    session = app.store.new_session("pending")
    before = digest(app)
    ask(app, session, EN)
    snapshot = root / "sessions.json"
    app.store.save(snapshot)
    data = json.loads(snapshot.read_text())
    assert all(not item["messages"] for item in data["sessions"])
    assert "window" not in snapshot.read_text() and "a-0" not in snapshot.read_text()
    app.close()
    restored = create()
    fresh = restored.store.new_session()
    assert action(ask(restored, fresh, "yes")) == "no_pending"
    assert digest(restored) == before


def test_clear_and_explicit_remember_invalidate_pending_without_erasing_acquired_memory(dialogue):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    ask(app, session, EN)
    app.clear_session(session)
    assert action(ask(app, session, "yes")) == "no_pending"
    ask(app, session, EN)
    app.clear_pending_memory(session)
    assert action(ask(app, session, "yes")) == "no_pending"
    ask(app, session, EN)
    app.remember("user", CUE, "helmet")
    before = digest(app)
    assert action(ask(app, session, "yes")) == "no_pending"
    assert digest(app) == before
    assert torch.allclose(dense_value(app), expected_value(app, "helmet"), atol=2e-5, rtol=5e-6)


def test_confirmed_correction_and_off_empty_keep_independent_sources(dialogue):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    ask(app, session, EN)
    ask(app, session, "yes")
    app.remember("person:1", CUE, "helmet")
    other = app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("person:1")].clone()
    before = digest(app)
    proposal = ask(app, session, "Исправь: метка объекта a-0 — dragon.")
    assert action(proposal) == "confirmation_required" and digest(app) == before
    assert action(ask(app, session, "Подтверждаю")) == "remembered"
    assert torch.allclose(dense_value(app), expected_value(app, "dragon"), atol=2e-5, rtol=5e-6)
    assert torch.equal(other, app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("person:1")])
    corrected = digest(app)
    for mode in ("off", "empty"):
        app.set_memory_mode(mode)
        result = ask(app, session, CUE)
        assert not result.memory_info["memory_applied"]
        assert digest(app) == corrected
    app.set_memory_mode("on")
    assert ask(app, session, CUE).memory_info["memory_applied"]


def test_russian_query_uses_existing_answer_blind_cue_without_reopening_value(dialogue, monkeypatch):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    ask(app, session, RU)
    ask(app, session, "Да")
    app.tokenizer.seen.clear()
    monkeypatch.setattr(app.neural_memory.engine, "_value_feature", lambda _text:
                        (_ for _ in ()).throw(AssertionError("Query reopened acquired value")))
    seen = []
    original = app.neural_memory.respond

    def record(request, *args, **kwargs):
        seen.append(request)
        return original(request, *args, **kwargs)

    monkeypatch.setattr(app.neural_memory, "respond", record)
    for question in ("Какая метка у объекта a-0?", "Скажи метку объекта a-0."):
        result = ask(app, session, question)
        assert result.memory_info["memory_applied"]
        assert seen[-1].text == CUE
        assert seen[-1].knowledge_source_id == "user"
    assert not any("window" in text or RU in text for text in app.tokenizer.seen)


def test_failed_confirmation_save_rolls_back_and_can_be_retried(dialogue, monkeypatch):
    create, root = dialogue
    app = create()
    session = app.store.new_session()
    app.remember("user", CUE, "window")
    before = digest(app)
    manifest = (root / "memory" / "memory.json").read_bytes()
    ask(app, session, EN.replace("window", "dragon"))
    original = app.neural_memory.memory.save
    monkeypatch.setattr(app.neural_memory.memory, "save", lambda *_:
                        (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises((OSError, ValueError, RuntimeError), match="disk full"):
        ask(app, session, "yes")
    assert digest(app) == before
    assert (root / "memory" / "memory.json").read_bytes() == manifest
    assert app.store.messages(session) == ()
    monkeypatch.setattr(app.neural_memory.memory, "save", original)
    assert action(ask(app, session, "yes")) == "remembered"
    assert torch.allclose(dense_value(app), expected_value(app, "dragon"), atol=2e-5, rtol=5e-6)
