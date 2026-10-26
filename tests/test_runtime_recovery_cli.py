"""Public runtime/CLI recovery states; fake frozen Qwen, no external assets."""

from pathlib import Path

import pytest

from alice_memory.payload_matrix import MemoryCommitUncertainError
from alice_runtime.neural_memory import MemoryRestartRequiredError
from test_dialogue_acquisition import EN, action, ask, dialogue
from test_runtime_memory_cli import RuntimeSpy, events, harness


MEMORY_FLAGS = ["--json", "--memory-bridge", "/bridge", "--memory-state", "/state"]


def test_runtime_restart_is_system_error_and_close_remains_idempotent(dialogue, monkeypatch):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    proposal = ask(app, session, EN)
    assert action(proposal) == "confirmation_required"
    real_save = app.neural_memory.memory.save

    def commit_without_ack(path):
        real_save(path)
        raise OSError("save acknowledgement interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(app.neural_memory.memory, "save", commit_without_ack)
        with pytest.raises(MemoryCommitUncertainError):
            ask(app, session, "yes")
    state = app.memory_status()
    assert state["state"] == "restart_required"
    assert state["restart_required"] and state["closed"]
    assert app.store.messages(session) == ()
    # A new fact must not become an unsupported-value reply after this failure.
    for text in (EN, "yes", "What is the assigned label of object a-0?"):
        with pytest.raises(MemoryRestartRequiredError, match="restart required"):
            ask(app, session, text)
    assert app.memory_status()["memory_digest"] == state["memory_digest"]
    app.close()
    app.close()
    restored = create()
    assert restored.memory_status()["state"] == "open"
    assert restored.memory_status()["memory_digest"] == state["memory_digest"]


def test_system_failure_during_admission_is_not_converted_to_unsupported(dialogue, monkeypatch):
    create, _root = dialogue
    app = create()
    session = app.store.new_session()
    before = app.memory_status()["memory_digest"]

    def restart_during_validation(*_args):
        raise MemoryRestartRequiredError("restart required during admission")

    monkeypatch.setattr(app.neural_memory, "validate_fact", restart_during_validation)
    with pytest.raises(MemoryRestartRequiredError, match="during admission"):
        ask(app, session, EN)
    assert app.memory_status()["memory_digest"] == before
    assert app.store.messages(session) == ()


@pytest.mark.parametrize("failure_type", [MemoryCommitUncertainError, MemoryRestartRequiredError])
@pytest.mark.parametrize("method,first_line", [
    ("respond", "yes"),
    ("remember", '/remember user "What is the assigned label of object a-0?" "window"'),
])
def test_interactive_terminal_recovery_error_stops_commands_without_saved_ack(
        harness, capsys, monkeypatch, failure_type, method, first_line):
    run, _loads = harness

    def fail(self, *_args, **_kwargs):
        self.calls.append((method, "failed"))
        raise failure_type("publication needs a fresh restore")

    monkeypatch.setattr(RuntimeSpy, method, fail)
    assert run(MEMORY_FLAGS, [first_line, "must not be processed", "/memory save", "/quit"]) == 2
    output, stderr = events(capsys)
    assert not stderr
    assert [event["event"] for event in output] == ["ready", "error", "resources"]
    error = output[1]
    assert error["error_code"] == "memory_restart_required"
    assert error["restart_required"] is True
    assert "restart" in error["error"].lower()
    assert not {"remembered", "response", "memory_backup", "memory_status"} & {
        event["event"] for event in output}
    app = RuntimeSpy.instances[-1]
    assert app.closed
    assert [call for call in app.calls if call[0] in {"respond", "remember", "save_memory"}] == [
        (method, "failed")]


@pytest.mark.parametrize("failure_type", [MemoryCommitUncertainError, MemoryRestartRequiredError])
def test_batch_terminal_recovery_error_has_exit_two_and_skips_later_messages(
        harness, capsys, monkeypatch, failure_type):
    run, _loads = harness

    def fail(self, *_args, **_kwargs):
        self.calls.append(("respond", "failed"))
        raise failure_type("checkpoint outcome is indeterminate")

    monkeypatch.setattr(RuntimeSpy, "respond", fail)
    assert run([*MEMORY_FLAGS, "--message", "first", "--message", "must not run"]) == 2
    output, stderr = events(capsys)
    assert not stderr and len(output) == 1
    assert output[0]["event"] == "error"
    assert output[0]["error_code"] == "memory_restart_required"
    assert output[0]["restart_required"] is True
    app = RuntimeSpy.instances[-1]
    assert app.closed
    assert [call for call in app.calls if call[0] == "respond"] == [("respond", "failed")]


def test_ordinary_input_error_is_nonterminal_and_next_request_can_succeed(
        harness, capsys, monkeypatch):
    run, _loads = harness

    def reject_value(self, *_args):
        self.calls.append(("remember", "rejected"))
        raise ValueError("The value must contain one model token")

    monkeypatch.setattr(RuntimeSpy, "remember", reject_value)
    assert run(MEMORY_FLAGS, [
        '/remember user "question" "two words"', "Hello", "/quit",
    ]) == 0
    output, stderr = events(capsys)
    assert not stderr
    assert [event["event"] for event in output] == ["ready", "error", "response", "resources"]
    assert "restart_required" not in output[1] and "error_code" not in output[1]
    assert output[2]["text"] == "window"
    app = RuntimeSpy.instances[-1]
    assert app.closed
    assert sum(call[0] == "respond" for call in app.calls) == 1


def test_cli_memory_backup_dispatches_path_and_reports_only_backup_ack(
        harness, capsys, monkeypatch, tmp_path):
    run, _loads = harness
    destination = tmp_path / "backup with spaces"

    def backup(self, path):
        self.calls.append(("backup_memory", path))
        return {"backup_path": str(Path(path)), "memory_digest": "memory-digest",
                "bridge_digest": "matching-bridge-digest"}

    monkeypatch.setattr(RuntimeSpy, "backup_memory", backup, raising=False)
    assert run(MEMORY_FLAGS, [f'/memory backup "{destination}"', "/quit"]) == 0
    output, stderr = events(capsys)
    assert not stderr
    assert [event["event"] for event in output] == ["ready", "memory_backup", "resources"]
    assert output[1]["backup"]["backup_path"] == str(destination)
    app = RuntimeSpy.instances[-1]
    assert app.closed and ("backup_memory", str(destination)) in app.calls
    assert not any(call[0] in {"respond", "remember", "save_memory"} for call in app.calls)
    assert not destination.exists()  # Dispatch fixture never touches a real checkpoint.
