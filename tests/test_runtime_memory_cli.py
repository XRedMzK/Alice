"""CLI dispatch/cleanup checks with no model loading or memory implementation mockup."""

from dataclasses import dataclass
import json
import sys

import pytest

import alice_runtime.__main__ as cli


@dataclass
class Answer:
    text: str
    session_id: str
    memory_info: dict | None = None


class RuntimeSpy:
    instances = []

    def __init__(self, loaded, store, budget):
        self.store, self.budget = store, budget
        self.calls, self.closed, self.attached = [], False, False
        self.mode = "on"
        self.instances.append(self)

    def attach_memory(self, bridge, state, model_path=None):
        self.calls.append(("attach", bridge, state, model_path))
        self.attached = True
        self.store.add_person(1)
        self.store.add_person(2)

    def set_memory_mode(self, mode):
        if not self.attached:
            raise ValueError("Neural Memory is not attached")
        self.calls.append(("mode", mode))
        self.mode = mode

    def remember(self, source, cue, value):
        if not self.attached:
            raise ValueError("Neural Memory is not attached")
        self.calls.append(("remember", source, cue, value))
        return {"source_id": source, "persisted": True}

    def memory_status(self):
        if not self.attached:
            raise ValueError("Neural Memory is not attached")
        return {"source_ids": ["agent", "user", "person:1", "person:2"], "mode": self.mode}

    def save_memory(self):
        self.calls.append(("save_memory",))
        return self.memory_status()

    def respond(self, session, source, text, limit, **kwargs):
        self.calls.append(("respond", source, text, limit, kwargs))
        return Answer("window", session)

    def resources(self):
        return {"process_peak_rss_mib": 12}

    def close(self):
        self.closed = True

    def clear_pending_memory(self, session=None):
        pass

    def clear_session(self, session):
        self.store.clear_session(session)


@pytest.fixture
def harness(monkeypatch):
    RuntimeSpy.instances = []
    loads = []
    monkeypatch.setattr(cli, "AliceRuntime", RuntimeSpy)
    monkeypatch.setattr(cli, "load_qwen", lambda *args: loads.append(args) or object())

    def run(args=(), lines=()):
        monkeypatch.setattr(sys, "argv", ["alice_runtime", "--model-path", "/local/qwen", *args])
        iterator = iter(lines)

        def input_line(prompt=""):
            try:
                return next(iterator)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr("builtins.input", input_line)
        return cli.main()

    return run, loads


def events(capsys):
    output = capsys.readouterr()
    return [json.loads(line) for line in output.out.splitlines()], output.err


def test_structured_remember_modes_and_independent_source_selection(harness, capsys, tmp_path, monkeypatch):
    run, loads = harness
    transcript = tmp_path / "transcripts" / "session.json"
    configured = []
    real_exporter = cli.TranscriptExportDirectory

    def configure(path, *, protected_paths):
        configured.append(tuple(str(p) for p in protected_paths))
        # This dispatch fixture supplies no real model/state assets. Real
        # protected-directory and alias behavior is tested on temporary copies.
        return real_exporter(path)

    monkeypatch.setattr(cli, "TranscriptExportDirectory", configure)
    assert run(["--json", "--memory-bridge", "/bridge", "--memory-state", "/state",
                "--transcript-dir", str(transcript.parent)], [
        '/remember user "What is the assigned label of object demo?" "window"',
        '/remember person:1 "What is the assigned label of object demo?" "helmet"',
        "/memory off", "/memory empty", "/memory on", "/memory status", "/memory save",
        "/source person:1", "/knowledge-source user",
        "Which label was assigned to object demo?", "/clear", "/new",
        '/save "session.json"', "/quit",
    ]) == 0
    output, err = events(capsys)
    assert not err and loads == [("/local/qwen", "cpu", "float32")]
    assert configured == [("/local/qwen", "/bridge", "/state")]
    app = RuntimeSpy.instances[0]
    assert app.closed
    assert app.calls[0] == ("attach", "/bridge", "/state", "/local/qwen")
    assert ("remember", "user", "What is the assigned label of object demo?", "window") in app.calls
    assert ("remember", "person:1", "What is the assigned label of object demo?", "helmet") in app.calls
    assert ("respond", "person:1", "Which label was assigned to object demo?", 16,
            {"knowledge_source_id": "user"}) in app.calls
    assert [c for c in app.calls if c[0] == "mode"] == [
        ("mode", "on"), ("mode", "off"), ("mode", "empty"), ("mode", "on")]
    assert sum(c[0] == "attach" for c in app.calls) == 1  # new/clear never reset memory.
    assert sum(x["event"] == "remembered" for x in output) == 2
    assert output[0]["event"] == "ready" and output[-1]["event"] == "resources"
    assert json.loads(transcript.read_text())["sessions"]  # /save still uses SessionStore.
    assert all(not s["messages"] for s in json.loads(transcript.read_text())["sessions"])


def test_malformed_commands_are_errors_and_do_not_reach_generation(harness, capsys):
    run, _ = harness
    assert run(["--json", "--memory-bridge", "/bridge", "--memory-state", "/state"], [
        '/remember user "missing value"', '/remember user "bad quote',
        "/memory reset", "/knowledge-source person:999", "/source missing",
        "/unknown command", "/quit extra", "/quit",
    ]) == 0
    output, err = events(capsys)
    assert not err
    assert sum(x["event"] == "error" for x in output) == 7
    app = RuntimeSpy.instances[0]
    assert not any(c[0] in {"respond", "remember"} for c in app.calls)
    assert app.closed


def test_legacy_batch_output_and_default_arguments_are_preserved(harness, capsys):
    run, loads = harness
    assert run(["--message", "Hello", "--message", "Again"]) == 0
    output, err = events(capsys)
    assert not err and len(output) == 3
    assert output[0]["text"] == output[1]["text"] == "window"
    assert "memory_info" not in output[0] and "memory_info" not in output[1]
    assert "event" not in output[0] and "resources" in output[-1]
    app = RuntimeSpy.instances[0]
    assert app.calls == [("respond", "user", "Hello", 16, {}), ("respond", "user", "Again", 16, {})]
    assert app.closed and loads == [("/local/qwen", "cpu", "float32")]


def test_interactive_confirmation_carries_last_displayed_content_id(harness, capsys, monkeypatch):
    run, _loads = harness
    calls = []

    def respond(self, session, source, text, limit, **kwargs):
        calls.append((text, kwargs))
        actions = [("confirmation_required", "fact-A"),
                   ("stale_confirmation", "fact-B"), ("remembered", None)]
        action, identity = actions[len(calls) - 1]
        return Answer("Shown fact", session, {"dialogue_action": action,
                                             "confirmation_id": identity})

    monkeypatch.setattr(RuntimeSpy, "respond", respond)
    assert run(["--json", "--memory-bridge", "/bridge", "--memory-state", "/state"],
               ["Remember that object a-0 has label window.", "yes", "yes"]) == 0
    events(capsys)
    assert "confirmation_id" not in calls[0][1]
    assert calls[1][1]["confirmation_id"] == "fact-A"
    assert calls[2][1]["confirmation_id"] == "fact-B"


def test_memory_batch_dispatches_target_and_mode(harness, capsys):
    run, _ = harness
    assert run(["--memory-bridge", "/bridge", "--memory-state", "/state",
                "--source-id", "person:1", "--knowledge-source-id", "user",
                "--memory-mode", "empty", "--message", "Question"]) == 0
    output, _ = events(capsys)
    assert output[0]["text"] == "window"
    app = RuntimeSpy.instances[0]
    assert ("mode", "empty") in app.calls
    assert ("respond", "person:1", "Question", 16, {"knowledge_source_id": "user"}) in app.calls
    assert app.closed


@pytest.mark.parametrize("arguments", [
    ["--memory-state", "/state"], ["--memory-bridge", "/bridge"],
    ["--memory-mode", "off"], ["--knowledge-source-id", "user"],
    ["--memory-bridge", "/bridge", "--memory-state", "/state", "--max-new-tokens", "17"],
])
def test_invalid_startup_flags_rejected_before_loading(harness, arguments):
    run, loads = harness
    with pytest.raises(SystemExit) as error:
        run(arguments)
    assert error.value.code == 2 and not loads
    assert not RuntimeSpy.instances


def test_startup_failure_and_eof_both_close_runtime(harness, capsys, monkeypatch):
    run, _ = harness
    original = RuntimeSpy.attach_memory

    def fail(self, *args, **kwargs):
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(RuntimeSpy, "attach_memory", fail)
    assert run(["--json", "--memory-bridge", "/bridge", "--memory-state", "/state"]) == 1
    output, err = events(capsys)
    assert output == [{"event": "error", "error": "checkpoint unavailable"}] and not err
    assert RuntimeSpy.instances[-1].closed
    monkeypatch.setattr(RuntimeSpy, "attach_memory", original)
    assert run(["--json"], []) == 0
    output, err = events(capsys)
    assert [x["event"] for x in output] == ["ready", "resources"]
    assert RuntimeSpy.instances[-1].closed
