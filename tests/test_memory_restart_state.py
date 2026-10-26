"""An indeterminate save remains a system/restart condition after its error."""

import pytest

from alice_memory.matrix import MemoryStateError
from alice_memory.payload_matrix import MemoryCommitUncertainError
from alice_memory.runtime import Question
from alice_runtime.neural_memory import MemoryRestartRequiredError, NeuralMemorySession
from test_runtime_memory_store import setup


def test_indeterminate_publication_has_explicit_restart_state_and_system_error(tmp_path, monkeypatch):
    loaded, bridge, state = setup(tmp_path)
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge, state) as app:
        app.remember("user", cue, "quartz")
        save = app.memory.save

        def committed_but_no_ack(path):
            save(path)
            raise OSError("ACK interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(app.memory, "save", committed_but_no_ack)
            with pytest.raises(MemoryCommitUncertainError) as failure:
                app.remember("user", cue, "copper")
        assert failure.value.restart_required
        assert failure.value.error_code == "memory_restart_required"
        status = app.status()
        assert status["state"] == "restart_required"
        assert status["restart_required"] and status["closed"]
        assert status["error_code"] == "memory_restart_required"
        assert "publication may have committed" in status["restart_reason"]
        for operation in (
                lambda: app.respond(Question(cue)),
                lambda: app.validate_fact("user", cue, "quartz"),
                lambda: app.remember("user", cue, "quartz"),
                app.save, lambda: app.backup(tmp_path / "backup")):
            with pytest.raises(MemoryRestartRequiredError, match="restart required") as blocked:
                operation()
            assert isinstance(blocked.value, RuntimeError)
        assert not (tmp_path / "backup").exists()
    with NeuralMemorySession(loaded, bridge, state) as restored:
        assert restored.status()["state"] == "open"
        assert not restored.status()["restart_required"]
        assert restored.memory.content_digest() == app.memory.content_digest()


def test_normal_close_is_distinct_from_indeterminate_commit(tmp_path):
    loaded, bridge, state = setup(tmp_path)
    app = NeuralMemorySession(loaded, bridge, state)
    assert app.status()["state"] == "open"
    app.close()
    status = app.status()
    assert status["state"] == "closed" and not status["restart_required"]
    assert "restart_reason" not in status and "error_code" not in status
    with pytest.raises(MemoryStateError, match="closed") as error:
        app.save()
    assert not isinstance(error.value, MemoryRestartRequiredError)
