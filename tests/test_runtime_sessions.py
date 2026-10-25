"""Session/provenance support checks; no model, training, or neural-memory claim."""

import json

import pytest

from alice_runtime.sessions import Message, SessionError, SessionStore


def test_participant_identity_and_transcript_survive_restore(tmp_path):
    store = SessionStore()
    person = store.add_person(2)
    assert person.source_id == "person:2"
    assert person.label == "Another Person #2"
    assert store.add_person(2) == person
    session = store.new_session("conversation")
    store.add_message(session, "user", "Ключ на столе.")
    store.add_message(session, person.source_id, "The key is in the drawer.")
    path = tmp_path / "state.json"
    store.save(path)
    restored = SessionStore.load(path)
    assert restored.participants == store.participants
    assert restored.messages(session) == store.messages(session)
    restored.add_message(session, "agent", "Two distinct sources reported different locations.")
    assert len(store.messages(session)) == 2
    with pytest.raises(TypeError):
        restored.participants["new"] = person


def test_sessions_are_isolated_clearable_and_closed_state_persists(tmp_path):
    store = SessionStore()
    first = store.new_session("first")
    second = store.new_session("second")
    store.add_message(first, "user", "Private to this session")
    assert store.messages(second) == ()
    view = store.messages(first)
    store.clear_session(first)
    assert store.messages(first) == ()
    assert view == (Message("user", "Private to this session"),)
    store.close_session(first)
    path = tmp_path / "state.json"
    store.save(path)
    restored = SessionStore.load(path)
    for operation in (lambda: restored.messages(first),
                      lambda: restored.add_message(first, "user", "No"),
                      lambda: restored.clear_session(first)):
        with pytest.raises(SessionError, match="closed"):
            operation()
    assert restored.messages(second) == ()


def test_invalid_sources_inputs_and_sessions_are_rejected_without_mutation():
    store = SessionStore()
    session = store.new_session("active")
    for source, text in (("person:1", "Unknown"), ("user", "  "),
                         ("user", "x" * (store.MAX_MESSAGE_CHARS + 1))):
        with pytest.raises(SessionError):
            store.add_message(session, source, text)
    assert store.messages(session) == ()
    with pytest.raises(SessionError):
        store.add_person(True)
    with pytest.raises(SessionError):
        store.new_session(session)
    with pytest.raises(SessionError):
        store.new_session("../outside")
    with pytest.raises(SessionError):
        store.messages("missing")


def test_exchange_commit_is_atomic_and_allows_immediate_eos():
    store = SessionStore()
    session = store.new_session("transaction")
    incoming = store.validate_message("user", "Ready?")
    assert store.messages(session) == ()
    with pytest.raises(SessionError):
        store.append_result(session, incoming, "x" * (store.MAX_MESSAGE_CHARS + 1))
    with pytest.raises(SessionError):
        store.append_result(session, Message("unknown", "Forged"), "No")
    assert store.messages(session) == ()
    assert store.append_result(session, incoming, "Yes") == (
        incoming, Message("agent", "Yes"),
    )
    after_first = store.messages(session)
    next_message = store.validate_message("user", "End")
    assert store.append_result(session, next_message, "") == (next_message,)
    assert store.messages(session) == after_first + (next_message,)


def test_corrupt_snapshot_is_rejected_and_atomic_save_preserves_old_file(tmp_path, monkeypatch):
    store = SessionStore()
    session = store.new_session()
    store.add_message(session, "user", "Original")
    path = tmp_path / "state.json"
    store.save(path)
    original = path.read_bytes()
    import alice_runtime.sessions as module
    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(SessionError, match="disk error"):
        store.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
    payload = json.loads(original)
    payload["participants"].append(payload["participants"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(SessionError, match="duplicate participant"):
        SessionStore.load(path)
    payload = json.loads(original)
    payload["sessions"][0]["messages"][0]["source_id"] = "unknown"
    path.write_text(json.dumps(payload))
    with pytest.raises(SessionError, match="source_id"):
        SessionStore.load(path)
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(SessionError, match="Duplicate JSON key"):
        SessionStore.load(path)
