"""Finite disposable history, stable identities and bounded v1 imports; no model fitting."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alice_runtime import sessions as module
from alice_runtime.export import TranscriptExportDirectory
from alice_runtime.sessions import Message, SessionError, SessionStore


class CaptureExport(TranscriptExportDirectory):
    """Serialization-only sink: this fixture never opens or writes a filesystem."""
    def __init__(self):
        self.content = None

    def write(self, name, content):
        self.content = content
        return Path(name)


def snapshot(store):
    sink = CaptureExport()
    store.save("snapshot.json", exporter=sink)
    return sink.content


def write_snapshot(tmp_path, payload):
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_caps_and_stable_participant_registry():
    store = SessionStore()
    assert store.retention_status()["limits"] == {
        "sessions": 16, "messages_per_session": 64, "chars_per_session": 262144, "participants": 32}
    known = store.add_person(1)
    for number in range(2, 31):
        store.add_person(number)
    assert len(store.participants) == 32 and store.add_person(1) is known
    with pytest.raises(SessionError, match="Participant limit"):
        store.add_person(31)
    assert store.participants["person:1"] is known
    for invalid in (True, 0, 10 ** 10000):
        with pytest.raises(SessionError, match="Person number"):
            store.add_person(invalid)


def test_session_cap_refuses_all_open_then_evicts_oldest_closed_without_changing_active_ids():
    store = SessionStore()
    ids = [store.new_session(f"s{number}") for number in range(16)]
    store.add_message(ids[0], "user", "Active fact in this chat")
    before = snapshot(store)
    with pytest.raises(SessionError, match="Session limit"):
        store.new_session("extra")
    assert snapshot(store) == before
    store.close_session("s1")
    store.close_session("s5")
    store.new_session("extra")
    with pytest.raises(SessionError, match="Unknown session"):
        store.messages("s1")
    assert store.messages("s0") == (Message("user", "Active fact in this chat"),)
    with pytest.raises(SessionError, match="closed"):
        store.messages("s5")
    assert store.retention_status()["sessions"] == 16
    assert store.retention_status()["drops"]["sessions_evicted"] == 1


def test_invalid_or_duplicate_new_session_never_evicts_a_closed_session():
    store = SessionStore()
    for number in range(16):
        store.new_session(f"s{number}")
        store.close_session(f"s{number}")
    before = snapshot(store)
    for invalid in ("s0", "../outside"):
        with pytest.raises(SessionError):
            store.new_session(invalid)
    assert snapshot(store) == before and store.retention_status()["drops"]["sessions_evicted"] == 0


def test_message_cap_discards_whole_exchanges_not_an_orphan_assistant():
    store = SessionStore()
    session = store.new_session()
    for number in range(33):
        store.append_result(session, Message("user", f"question-{number}"), f"answer-{number}")
    messages = store.messages(session)
    assert len(messages) == 64 and messages[0] == Message("user", "question-1")
    assert messages[1] == Message("agent", "answer-1")
    assert store.retention_status()["drops"]["messages_dropped"] == 2


def test_character_cap_also_discards_a_whole_exchange():
    store = SessionStore()
    session = store.new_session()
    for marker in ("a", "b", "c"):
        store.append_result(session, Message("user", marker * 65536), marker.upper() * 65536)
    messages = store.messages(session)
    assert len(messages) == 4 and messages[0].text == "b" * 65536
    assert sum(len(message.text) for message in messages) == 262144
    assert store.retention_status()["drops"] == {
        "sessions_evicted": 0, "messages_dropped": 2, "chars_dropped": 131072}


def test_low_level_agent_continuations_are_retained_with_their_question():
    class Small(SessionStore):
        MAX_MESSAGES_PER_SESSION = 4
    store = Small()
    session = store.new_session()
    store.add_message(session, "user", "first")
    store.add_message(session, "agent", "part one")
    store.add_message(session, "agent", "part two")
    store.append_result(session, Message("user", "second"), "response")
    assert store.messages(session) == (Message("user", "second"), Message("agent", "response"))
    assert store.retention_status()["drops"]["messages_dropped"] == 3


def test_exchange_that_cannot_fit_is_refused_atomically_with_unchanged_drop_counters():
    class Small(SessionStore):
        MAX_SESSION_CHARS = 10
    store = Small()
    session = store.new_session()
    store.append_result(session, Message("user", "old"), "yes")
    before, status = store.messages(session), store.retention_status()
    with pytest.raises(SessionError, match="One exchange"):
        store.append_result(session, Message("user", "123456"), "123456")
    assert store.messages(session) == before and store.retention_status() == status


def test_drop_counters_saturate_and_caller_cannot_mutate_their_snapshot():
    store = SessionStore()
    session = store.new_session()
    store._drops["messages_dropped"] = store.MAX_DROP_COUNTER - 1
    for _ in range(33):
        store.append_result(session, Message("user", "q"), "a")
    assert store.retention_status()["drops"]["messages_dropped"] == store.MAX_DROP_COUNTER
    status = store.retention_status()
    status["drops"]["messages_dropped"] = 0
    assert store.retention_status()["drops"]["messages_dropped"] == store.MAX_DROP_COUNTER


def test_small_v1_roundtrip_preserves_source_ids_messages_and_closed_flags(tmp_path):
    store = SessionStore()
    store.add_person(2)
    store.new_session("active")
    store.append_result("active", Message("person:2", "Привет"), "Hello")
    store.new_session("closed")
    store.close_session("closed")
    path = tmp_path / "v1.json"
    path.write_bytes(snapshot(store))
    restored = SessionStore.load(path)
    assert restored.participants == store.participants and restored.messages("active") == store.messages("active")
    with pytest.raises(SessionError, match="closed"):
        restored.messages("closed")
    assert set(json.loads(path.read_bytes())) == {"version", "participants", "sessions"}
    assert restored.retention_status()["drops"] == {"sessions_evicted": 0, "messages_dropped": 0, "chars_dropped": 0}


@pytest.mark.parametrize("which", ["sessions", "participants", "messages", "characters"])
def test_oversized_v1_collections_are_rejected_before_population_not_trimmed(tmp_path, monkeypatch, which):
    store = SessionStore()
    store.new_session("one")
    payload = json.loads(snapshot(store))
    if which == "sessions":
        payload["sessions"] = [{"session_id": f"s{i}", "closed": True, "messages": []} for i in range(17)]
    elif which == "participants":
        payload["participants"] += [{"source_id": f"person:{i}", "label": f"Another Person #{i}"} for i in range(1, 32)]
    elif which == "messages":
        payload["sessions"][0]["messages"] = [{"source_id": "user", "text": "x"}] * 65
    else:
        payload["sessions"][0]["messages"] = [{"source_id": "user", "text": "x" * 65536}] * 5
    path = write_snapshot(tmp_path, payload)
    def no_population(*args, **kwargs):
        raise AssertionError("Restore limits must be checked before populating a session")
    monkeypatch.setattr(SessionStore, "new_session", no_population)
    with pytest.raises(SessionError, match="limit"):
        SessionStore.load(path)


def test_oversized_source_identifier_is_rejected_without_large_int_parsing(tmp_path):
    store = SessionStore()
    payload = json.loads(snapshot(store))
    payload["participants"].append({"source_id": "person:" + "9" * 10000, "label": "invalid"})
    with pytest.raises(SessionError, match="participant source_id"):
        SessionStore.load(write_snapshot(tmp_path, payload))


def test_oversized_sparse_file_is_rejected_before_read_or_json_parse(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    with path.open("wb") as handle:
        handle.truncate(SessionStore.MAX_SNAPSHOT_BYTES + 1)
    monkeypatch.setattr(module.json, "loads", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not parse")))
    with pytest.raises(SessionError, match="byte limit"):
        SessionStore.load(path)


def test_import_reads_at_most_the_byte_cap_plus_one_and_rejects_growth(tmp_path, monkeypatch):
    class Small(SessionStore):
        MAX_SNAPSHOT_BYTES = 256
    path = tmp_path / "growing.json"
    path.write_text('{"version":1,"participants":[],"sessions":[]}', encoding="utf-8")
    original = module.os.fdopen
    requests = []
    class GrowingReader:
        def __init__(self, fd, mode):
            self.handle = original(fd, mode)
        def __enter__(self): return self
        def __exit__(self, *args): self.handle.close()
        def read(self, size):
            requests.append(size)
            with path.open("ab") as writer: writer.write(b" " * 300)
            return self.handle.read(size)
    monkeypatch.setattr(module.os, "fdopen", GrowingReader)
    with pytest.raises(SessionError, match="byte limit"):
        Small.load(path)
    assert requests == [257]


def test_fifo_import_is_refused_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    command = [sys.executable, "-c", "from alice_runtime.sessions import SessionStore; SessionStore.load(__import__('sys').argv[1])", str(path)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
    assert result.returncode != 0 and "regular local file" in result.stderr


def test_save_refuses_control_character_escaping_that_would_exceed_restore_byte_cap():
    class Small(SessionStore):
        MAX_SNAPSHOT_BYTES = 512
    store = Small()
    session = store.new_session()
    store.add_message(session, "user", "\x00" * 100)
    sink = CaptureExport()
    before = store.messages(session)
    with pytest.raises(SessionError, match="restore limit"):
        store.save("too-large.json", exporter=sink)
    assert sink.content is None and store.messages(session) == before


def test_retention_clear_and_v1_restore_do_not_reset_independent_neural_parameters(tmp_path):
    from alice_memory.payload_matrix import ParametricPayloadMemory
    memory = ParametricPayloadMemory()
    with torch.no_grad(): memory.matrix[1, 1, 0, 0] = 0.75
    before = memory.content_digest()
    store = SessionStore()
    session = store.new_session()
    for number in range(40):
        store.append_result(session, Message("user", f"question {number}"), "answer")
    path = tmp_path / "transcript.json"
    path.write_bytes(snapshot(store))
    restored = SessionStore.load(path)
    store.clear_session(session)
    assert restored.messages(session) and not store.messages(session)
    assert memory.content_digest() == before
    assert "memory" not in json.loads(path.read_bytes())


def test_small_legacy_v1_can_begin_with_an_explicit_agent_message(tmp_path):
    payload = {"version": 1, "participants": [
        {"source_id": "agent", "label": "Agent"}, {"source_id": "user", "label": "User"}],
        "sessions": [{"session_id": "legacy", "closed": False, "messages": [
            {"source_id": "agent", "text": "An explicit initial greeting"},
            {"source_id": "user", "text": "Hello"}, {"source_id": "agent", "text": "Welcome"}]}]}
    restored = SessionStore.load(write_snapshot(tmp_path, payload))
    assert [message.source_id for message in restored.messages("legacy")] == ["agent", "user", "agent"]
    assert json.loads(snapshot(restored)) == payload
