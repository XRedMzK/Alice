"""Adversarial confirmation checks with an asset-free frozen backbone.

These check which exact proposed fact can be saved, not cognitive recall by the
small fake language model. No confirmation token, fact or pending proposal is
persisted alongside the independent neural parameters.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading

import pytest
import torch

from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_runtime.loading import LoadedQwen
from alice_runtime.runtime import AliceRuntime
from alice_runtime.sessions import SessionStore
from test_runtime_memory_integration import FrozenLM, Tokenizer as BaseTokenizer


class Tokenizer(BaseTokenizer):
    VALUES = {"window": 3, "helmet": 4, "dragon": 5}


def cue(object_id):
    return f"What is the assigned label of object {object_id}?"


def proposal(object_id, value):
    return f"Remember that the label of object {object_id} is {value}."


def ask(app, session, text, source="user", **kwargs):
    return app.respond(session, source, text, 1, **kwargs)


def action(result):
    return result.memory_info.get("dialogue_action")


@pytest.fixture
def runtime(tmp_path):
    bridge_path = tmp_path / "bridge"
    bridge = PayloadEmbeddingBridge()
    bridge.basis_id = "alice.memory_payload/17x128/" + bridge.parameter_digest()
    bridge.freeze().save(bridge_path)
    apps = []

    def create(store=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            loaded = LoadedQwen(FrozenLM(), Tokenizer(), {})
        app = AliceRuntime(loaded, store=store, max_context_tokens=256)
        apps.append(app)
        app.attach_memory(bridge_path, tmp_path / "memory")
        return app

    yield create, tmp_path
    for app in apps:
        app.close()


def record_writes(app, monkeypatch):
    writes = []
    original = app.neural_memory.remember

    def record(source, question, value):
        writes.append((source, question, value))
        return original(source, question, value)

    monkeypatch.setattr(app.neural_memory, "remember", record)
    return writes


def assert_value(app, source, object_id, value):
    with torch.no_grad():
        key = app.neural_memory.engine.key_for(cue(object_id))
        readout = app.neural_memory.memory.read(source, key)
        slot = app.neural_memory.bridge.slots(key, readout, 1.0)[0]
        expected = app.model.get_input_embeddings().weight[Tokenizer.VALUES[value]]
        assert torch.allclose(slot, expected, atol=2e-5, rtol=5e-6)


def observe_confirmation_lock_entry(app, monkeypatch):
    """Signal after respond's entry snapshot, before its RLock acquisition."""
    original = app._lock
    thread = []
    entering = threading.Event()

    class ObservedLock:
        def __enter__(self):
            if thread and thread[0] == threading.get_ident():
                entering.set()
            return original.__enter__()

        def __exit__(self, *args):
            return original.__exit__(*args)

    monkeypatch.setattr(app, "_lock", ObservedLock())
    return thread, entering


def test_new_content_replaces_old_proposal_and_only_shown_fact_is_confirmed(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    before = app.memory_status()["memory_digest"]
    writes = record_writes(app, monkeypatch)
    first = ask(app, session, proposal("a-0", "window"))
    second = ask(app, session, proposal("b-0", "helmet"))
    assert action(first) == action(second) == "confirmation_required"
    assert "a-0" in first.text and "window" in first.text
    assert "b-0" in second.text and "helmet" in second.text
    assert "a-0" not in second.text and "window" not in second.text
    assert not writes and app.memory_status()["memory_digest"] == before
    with pytest.raises(FrozenInstanceError):
        app._pending_memory.value = "dragon"

    saved = ask(app, session, "yes")
    assert action(saved) == "remembered"
    assert writes == [("user", cue("b-0"), "helmet")]
    assert "b-0" in saved.text and "helmet" in saved.text
    assert "a-0" not in saved.text and "window" not in saved.text
    assert_value(app, "user", "b-0", "helmet")
    assert action(ask(app, session, "yes")) == "no_pending"
    assert len(writes) == 1


def test_explicit_confirmation_id_binds_only_the_exact_displayed_content(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    writes = record_writes(app, monkeypatch)
    old = ask(app, session, proposal("a-0", "window")).memory_info["confirmation_id"]
    latest = ask(app, session, proposal("b-0", "helmet")).memory_info["confirmation_id"]
    assert old != latest
    assert old.split(":")[1] != latest.split(":")[1]
    before = app.memory_status()["memory_digest"]
    stale = ask(app, session, "yes", confirmation_id=old)
    assert action(stale) == "stale_confirmation"
    assert "b-0" in stale.text and "helmet" in stale.text
    assert "a-0" not in stale.text and "window" not in stale.text
    assert stale.memory_info["confirmation_id"] == latest
    assert not writes and app.memory_status()["memory_digest"] == before
    assert app._pending_memory.confirmation_id == latest
    assert action(ask(app, session, "yes", confirmation_id=latest)) == "remembered"
    assert writes == [("user", cue("b-0"), "helmet")]


def test_identical_fact_reproposal_requires_its_new_confirmation_id(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    writes = record_writes(app, monkeypatch)
    old = ask(app, session, proposal("a-0", "window")).memory_info["confirmation_id"]
    latest = ask(app, session, proposal("a-0", "window")).memory_info["confirmation_id"]
    assert old != latest and old.split(":")[1] == latest.split(":")[1]
    stale = ask(app, session, "yes", confirmation_id=old)
    assert action(stale) == "stale_confirmation"
    assert "a-0" in stale.text and "window" in stale.text
    assert stale.memory_info["confirmation_id"] == latest
    assert not writes
    assert action(ask(app, session, "yes", confirmation_id=latest)) == "remembered"
    assert writes == [("user", cue("a-0"), "window")]


def test_proposal_in_another_session_and_source_cannot_be_confirmed_by_old_owner(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    first = app.store.new_session("first")
    second = app.store.new_session("second")
    writes = record_writes(app, monkeypatch)
    ask(app, first, proposal("a-0", "window"))
    shown = ask(app, second, proposal("b-0", "helmet"), source="person:1",
                knowledge_source_id="user")
    assert "person:1" in shown.text
    confirmation_id = shown.memory_info["confirmation_id"]
    for session, source in ((first, "user"), (second, "user"), (first, "person:1")):
        result = ask(app, session, "yes", source=source, confirmation_id=confirmation_id)
        assert action(result) == "no_pending"
        assert "b-0" not in result.text and "helmet" not in result.text
        assert "confirmation_id" not in result.memory_info
    assert not writes
    saved = ask(app, second, "yes", source="person:1", knowledge_source_id="user",
                confirmation_id=confirmation_id)
    assert action(saved) == "remembered"
    assert writes == [("person:1", cue("b-0"), "helmet")]
    assert_value(app, "person:1", "b-0", "helmet")
    user_plane = app.neural_memory.memory.matrix[app.neural_memory.memory.source_index("user")]
    assert torch.count_nonzero(user_plane) == 0


@pytest.mark.parametrize("replacement,expected_action", [
    (proposal("b-0", "window or helmet"), "clarification_required"),
    (proposal("b-0", "bright window"), "unsupported"),
    (proposal("b-0", "helmet") + " Source: person:1.", "clarification_required"),
])
def test_unadmitted_replacement_does_not_leave_a_confirmable_old_fact(
        runtime, monkeypatch, replacement, expected_action):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    before = app.memory_status()["memory_digest"]
    writes = record_writes(app, monkeypatch)
    ask(app, session, proposal("a-0", "window"))
    assert action(ask(app, session, replacement)) == expected_action
    assert action(ask(app, session, "yes")) == "no_pending"
    assert not writes and app.memory_status()["memory_digest"] == before


def test_restoring_same_session_and_source_does_not_restore_confirmation(runtime):
    create, root = runtime
    app = create()
    session = app.store.new_session("same-session-id")
    confirmation_id = ask(app, session, proposal("a-0", "window")).memory_info["confirmation_id"]
    app.store.save(root / "sessions.json")
    before = app.memory_status()["memory_digest"]
    app.close()
    restored = create(SessionStore.load(root / "sessions.json"))
    assert restored.store.messages(session) == ()
    assert action(ask(restored, session, "yes", confirmation_id=confirmation_id)) == "no_pending"
    assert restored.memory_status()["memory_digest"] == before
    assert not (root / "memory" / "memory.json").exists()


def test_failed_save_preserves_exact_proposal_until_retry_without_saved_ack(runtime, monkeypatch):
    create, root = runtime
    app = create()
    session = app.store.new_session()
    app.remember("user", cue("a-0"), "window")
    writes = record_writes(app, monkeypatch)
    ask(app, session, proposal("b-0", "dragon"))
    pending = app._pending_memory
    digest = app.memory_status()["memory_digest"]
    manifest = (root / "memory" / "memory.json").read_bytes()
    controls = []
    original_control = app._dialogue_result

    def record_control(*args, **kwargs):
        result = original_control(*args, **kwargs)
        controls.append(result)
        return result

    monkeypatch.setattr(app, "_dialogue_result", record_control)
    original_save = app.neural_memory.memory.save

    def fail_save(*_args, **_kwargs):
        raise OSError("confirmation write failed")

    monkeypatch.setattr(app.neural_memory.memory, "save", fail_save)
    with pytest.raises((OSError, RuntimeError, ValueError), match="confirmation write failed"):
        ask(app, session, "yes", confirmation_id=pending.confirmation_id)
    assert not controls  # No templated Saved/remembered control result was emitted.
    assert app._pending_memory is pending
    assert (pending.session_id, pending.source_id, pending.cue, pending.value) == (
        session, "user", cue("b-0"), "dragon")
    assert app.memory_status()["memory_digest"] == digest
    assert (root / "memory" / "memory.json").read_bytes() == manifest
    assert app.store.messages(session) == ()

    monkeypatch.setattr(app.neural_memory.memory, "save", original_save)
    saved = ask(app, session, "yes", confirmation_id=pending.confirmation_id)
    assert action(saved) == "remembered" and saved.memory_info["acquisition"]["saved"]
    assert writes == [("user", cue("b-0"), "dragon")] * 2
    assert_value(app, "user", "b-0", "dragon")


def test_new_proposal_after_failed_save_cannot_retry_the_failed_content(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    writes = record_writes(app, monkeypatch)
    ask(app, session, proposal("a-0", "window"))
    original_save = app.neural_memory.memory.save

    def fail_save(*_args, **_kwargs):
        raise OSError("out of space")

    monkeypatch.setattr(app.neural_memory.memory, "save", fail_save)
    with pytest.raises((OSError, RuntimeError, ValueError), match="out of space"):
        ask(app, session, "yes")
    monkeypatch.setattr(app.neural_memory.memory, "save", original_save)
    shown = ask(app, session, proposal("b-0", "helmet"), source="person:1")
    assert action(shown) == "confirmation_required" and "helmet" in shown.text
    assert action(ask(app, session, "yes")) == "no_pending"
    assert action(ask(app, session, "yes", source="person:1")) == "remembered"
    assert writes == [("user", cue("a-0"), "window"), ("person:1", cue("b-0"), "helmet")]
    assert_value(app, "person:1", "b-0", "helmet")
    assert torch.count_nonzero(app.neural_memory.memory.matrix[
        app.neural_memory.memory.source_index("user")]) == 0


@pytest.mark.parametrize("use_explicit_id", [False, True], ids=["legacy-yes", "displayed-id"])
def test_confirmation_queued_before_new_proposal_is_shown_cannot_save_it(
        runtime, monkeypatch, use_explicit_id):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    writes = record_writes(app, monkeypatch)
    old_id = ask(app, session, proposal("a-0", "window")).memory_info["confirmation_id"]
    validating, release, confirmed = (threading.Event() for _ in range(3))
    thread, entering_lock = observe_confirmation_lock_entry(app, monkeypatch)
    original_validate = app.neural_memory.validate_fact

    def delayed_validate(source, question, value):
        if question == cue("b-0"):
            validating.set()
            assert release.wait(3), "Test failed to release proposal validation"
        return original_validate(source, question, value)

    def confirm():
        thread.append(threading.get_ident())
        try:
            return ask(app, session, "yes", confirmation_id=old_id if use_explicit_id else None)
        finally:
            confirmed.set()

    monkeypatch.setattr(app.neural_memory, "validate_fact", delayed_validate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        proposed = pool.submit(ask, app, session, proposal("b-0", "helmet"))
        try:
            assert validating.wait(3)
            saved = pool.submit(confirm)
            assert entering_lock.wait(3)
            assert not confirmed.is_set()
        finally:
            release.set()
        shown = proposed.result(timeout=3)
        assert action(shown) == "confirmation_required"
        stale = saved.result(timeout=3)
        assert action(stale) == "stale_confirmation"
        assert "b-0" in stale.text and "helmet" in stale.text
        assert "a-0" not in stale.text and "window" not in stale.text
        assert stale.memory_info["confirmation_id"] == shown.memory_info["confirmation_id"]
    assert not writes
    assert action(ask(app, session, "yes", confirmation_id=shown.memory_info["confirmation_id"])) == "remembered"
    assert writes == [("user", cue("b-0"), "helmet")]


def test_concurrent_yes_replies_save_once_and_second_has_no_pending_fact(runtime, monkeypatch):
    create, _root = runtime
    app = create()
    session = app.store.new_session()
    ask(app, session, proposal("a-0", "window"))
    writes = record_writes(app, monkeypatch)
    saving, release, confirmed = (threading.Event() for _ in range(3))
    thread, entering_lock = observe_confirmation_lock_entry(app, monkeypatch)
    original_save = app.neural_memory.memory.save

    def delayed_save(*args, **kwargs):
        saving.set()
        assert release.wait(3), "Test failed to release memory save"
        return original_save(*args, **kwargs)

    def confirm():
        thread.append(threading.get_ident())
        try:
            return ask(app, session, "yes")
        finally:
            confirmed.set()

    monkeypatch.setattr(app.neural_memory.memory, "save", delayed_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ask, app, session, "yes")
        try:
            assert saving.wait(3)
            second = pool.submit(confirm)
            assert entering_lock.wait(3)
            assert not confirmed.is_set()
        finally:
            release.set()
        assert action(first.result(timeout=3)) == "remembered"
        assert action(second.result(timeout=3)) == "no_pending"
    assert writes == [("user", cue("a-0"), "window")]
