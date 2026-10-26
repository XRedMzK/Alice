"""Checkpoint commit boundaries, real process death, and indeterminate saves.

These tests use only tiny CPU parameters; no Qwen/Bridge assets or training.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alice_memory import payload_matrix
from alice_memory.matrix import MemoryStateError
from alice_memory.payload_matrix import MemoryCommitUncertainError, ParametricPayloadMemory
from alice_memory.runtime import Question
from alice_runtime.neural_memory import NeuralMemorySession
from test_runtime_memory_store import setup, tree_bytes


def changed(memory):
    with torch.no_grad():
        memory.matrix[1, 0, 0, 0] = 7
    return memory


def test_failed_file_fsync_precedes_commit_and_preserves_old_state(tmp_path, monkeypatch):
    memory = ParametricPayloadMemory()
    memory.save(tmp_path)
    before = tree_bytes(tmp_path)
    old_digest = memory.content_digest()
    changed(memory)
    original = os.fsync

    def fail_payload_sync(descriptor):
        if os.readlink(f"/proc/self/fd/{descriptor}").endswith(".safetensors.tmp"):
            raise OSError("payload fsync failed")
        return original(descriptor)

    monkeypatch.setattr(payload_matrix.os, "fsync", fail_payload_sync)
    with pytest.raises(OSError, match="payload fsync failed"):
        memory.save(tmp_path)
    assert tree_bytes(tmp_path) == before
    assert ParametricPayloadMemory.load(tmp_path).content_digest() == old_digest


@pytest.mark.parametrize("boundary", ["manifest-file", "payload-directory"])
def test_other_precommit_fsync_failures_leave_old_checkpoint(tmp_path, monkeypatch, boundary):
    root = tmp_path / "state"
    memory = ParametricPayloadMemory()
    memory.save(root)
    before = tree_bytes(root)
    changed(memory)
    sync = os.fsync

    def fail_precommit_sync(descriptor):
        name = os.readlink(f"/proc/self/fd/{descriptor}")
        if ((boundary == "manifest-file" and name.endswith(".json.tmp"))
                or (boundary == "payload-directory" and name == str(root))):
            raise OSError("precommit fsync failed")
        return sync(descriptor)

    monkeypatch.setattr(payload_matrix.os, "fsync", fail_precommit_sync)
    with pytest.raises(OSError, match="precommit fsync failed"):
        memory.save(root)
    assert tree_bytes(root) == before


@pytest.mark.parametrize("replace_then_fail", [False, True])
def test_manifest_switch_error_never_acknowledges_or_uses_stale_ram(
        tmp_path, monkeypatch, replace_then_fail):
    loaded, bridge_path, state_path = setup(tmp_path)
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", cue, "quartz")
        old_digest = app.memory.content_digest()
        bridge_digest = app.bridge.parameter_digest()
        original = os.replace

        def fail_switch(source, destination):
            if Path(destination).name == "memory.json":
                if replace_then_fail:
                    original(source, destination)
                raise OSError("manifest publication interrupted")
            return original(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(payload_matrix.os, "replace", fail_switch)
            with pytest.raises(MemoryCommitUncertainError, match="may have committed"):
                app.remember("user", cue, "copper")
        assert app.status()["closed"]
        with pytest.raises(MemoryStateError, match="closed"):
            app.respond(Question(cue))
        assert app.bridge.parameter_digest() == bridge_digest
        # A possible committed write is never rolled back to old RAM.
        assert app.memory.content_digest() != old_digest
    with NeuralMemorySession(loaded, bridge_path, state_path) as restored:
        assert (restored.memory.content_digest() != old_digest) == replace_then_fail
        assert len(list(state_path.glob("memory-payload-*.safetensors"))) == 1


def test_final_directory_fsync_failure_closes_session_without_false_ack(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", cue, "quartz")
        old_digest = app.memory.content_digest()
        old_manifest = (state_path / "memory.json").read_bytes()
        sync = payload_matrix._sync_directory

        def fail_after_commit(directory):
            if directory == state_path and (state_path / "memory.json").read_bytes() != old_manifest:
                raise OSError("directory fsync failed")
            return sync(directory)

        with monkeypatch.context() as patch:
            patch.setattr(payload_matrix, "_sync_directory", fail_after_commit)
            with pytest.raises(MemoryCommitUncertainError):
                app.remember("user", cue, "copper")
        assert app.status()["closed"]
        assert app.memory.content_digest() != old_digest
        assert ParametricPayloadMemory.load(
            state_path, latent_basis_id=app.bridge.basis_id).content_digest() == app.memory.content_digest()
    with NeuralMemorySession(loaded, bridge_path, state_path) as restored:
        assert restored.memory.content_digest() != old_digest


@pytest.mark.parametrize("failure", ["save-wrapper", "receipt"])
def test_error_after_successful_save_cannot_roll_ram_back(tmp_path, monkeypatch, failure):
    loaded, bridge_path, state_path = setup(tmp_path)
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", cue, "quartz")
        old_digest = app.memory.content_digest()
        save, digest = app.memory.save, app.memory.content_digest
        published = False

        def wrapped_save(directory):
            nonlocal published
            result = save(directory)
            published = True
            if failure == "save-wrapper":
                raise OSError("error immediately after durable save")
            return result

        def receipt_digest():
            if published:
                raise RuntimeError("receipt construction interrupted")
            return digest()

        with monkeypatch.context() as patch:
            patch.setattr(app.memory, "save", wrapped_save)
            if failure == "receipt":
                patch.setattr(app.memory, "content_digest", receipt_digest)
            with pytest.raises(MemoryCommitUncertainError):
                app.remember("user", cue, "copper")
        assert app.status()["closed"]
        assert app.memory.content_digest() != old_digest
        restored = ParametricPayloadMemory.load(state_path, latent_basis_id=app.bridge.basis_id)
        assert restored.content_digest() == app.memory.content_digest()


def test_payload_retirement_error_keeps_successful_commit_recoverable(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", cue, "quartz")
        old_payload = state_path / json.loads((state_path / "memory.json").read_text())["tensor_file"]
        unlink = Path.unlink

        def fail_retirement(path, *args, **kwargs):
            if path == old_payload:
                raise OSError("old snapshot cleanup failed")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_retirement)
            result = app.remember("user", cue, "copper")
        assert result["saved"]
        assert "cleanup_warning" in app.status()
        assert len(list(state_path.glob("memory-payload-*.safetensors"))) == 2
        expected = app.memory.content_digest()
    with NeuralMemorySession(loaded, bridge_path, state_path) as restored:
        assert restored.memory.content_digest() == expected
        assert len(list(state_path.glob("memory-payload-*.safetensors"))) == 1


def test_nested_directory_creation_syncs_ancestor_entries_before_publication(tmp_path, monkeypatch):
    state_path = tmp_path / "new-parent" / "new-child" / "state"
    synchronized = []
    sync = payload_matrix._sync_directory

    def record(directory):
        synchronized.append(directory)
        return sync(directory)

    monkeypatch.setattr(payload_matrix, "_sync_directory", record)
    ParametricPayloadMemory().save(state_path)
    assert synchronized.index(tmp_path) < synchronized.index(state_path.parent.parent)
    assert synchronized.index(state_path.parent.parent) < synchronized.index(state_path.parent)
    assert synchronized.index(state_path.parent) < synchronized.index(state_path)


_KILL_WRITER = r'''
import json, os, signal, sys, torch
from pathlib import Path
from alice_memory import payload_matrix as module
from alice_memory.payload_matrix import ParametricPayloadMemory
root, stage = Path(sys.argv[1]), sys.argv[2]
old_manifest = (root / "memory.json").read_bytes() if (root / "memory.json").exists() else None
memory = ParametricPayloadMemory.load(root) if (root / "memory.json").exists() else ParametricPayloadMemory()
with torch.no_grad(): memory.matrix[1, 0, 0, 0] = 7
def kill(): os.kill(os.getpid(), signal.SIGKILL)
save_file, dump, replace, sync, prepare_bytes = module.save_file, module.json.dump, module.os.replace, module._sync_directory, module._prepare_bytes
def prepare(snapshot):
    if stage == "empty_prepare": kill()
    return prepare_bytes(snapshot)
def save(tensors, path):
    if stage == "partial_payload":
        Path(path).write_bytes(b"interrupted safetensors")
        kill()
    save_file(tensors, path)
def write_json(value, handle, **kwargs):
    if stage == "partial_manifest":
        handle.write('{"schema":')
        handle.flush()
        kill()
    dump(value, handle, **kwargs)
def switch(source, destination):
    replace(source, destination)
    if Path(destination).name == "memory.json":
        if stage == "manifest_replaced": kill()
    elif Path(destination).name.endswith(".prepare") and stage == "prepare_published": kill()
    elif Path(destination).name.endswith(".safetensors") and stage == "payload_published": kill()
def sync_directory(path):
    sync(path)
    if (path == root and stage == "publication_fsynced" and (root / "memory.json").exists()
            and (root / "memory.json").read_bytes() != old_manifest): kill()
module.save_file, module.json.dump = save, write_json
module.os.replace, module._sync_directory = switch, sync_directory
module._prepare_bytes = prepare
memory.save(root)
raise AssertionError("requested kill boundary was not reached")
'''

_FRESH_RESTORE = r'''
import json, sys
from pathlib import Path
from alice_memory.payload_matrix import ParametricPayloadMemory
root = Path(sys.argv[1])
ParametricPayloadMemory.recover_uncommitted(root)
memory = ParametricPayloadMemory.load(root) if (root / "memory.json").exists() else ParametricPayloadMemory()
print(json.dumps({"digest":memory.content_digest(), "manifest":(root / "memory.json").exists()}))
'''


_CRASH_STAGES = ["partial_payload", "payload_published", "partial_manifest",
                 "manifest_replaced", "publication_fsynced"]


@pytest.mark.parametrize("existing,stage", [
    *[(existing, stage) for stage in _CRASH_STAGES for existing in (False, True)],
    (False, "empty_prepare"), (False, "prepare_published"),
])
def test_sigkill_fresh_restore_has_only_complete_old_or_new_memory(tmp_path, existing, stage):
    old = ParametricPayloadMemory()
    old_digest = old.content_digest()
    if existing:
        old.save(tmp_path)
    new_digest = changed(old).content_digest()
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    dead = subprocess.run([sys.executable, "-B", "-c", _KILL_WRITER, str(tmp_path), stage],
                          capture_output=True, text=True, env=environment, timeout=20)
    assert dead.returncode == -9, dead.stderr
    restored = subprocess.run([sys.executable, "-B", "-c", _FRESH_RESTORE, str(tmp_path)],
                              check=True, capture_output=True, text=True, env=environment, timeout=20)
    state = json.loads(restored.stdout)
    committed = stage in {"manifest_replaced", "publication_fsynced"}
    assert state == {"digest": new_digest if committed else old_digest,
                     "manifest": existing or committed}
    assert not list(tmp_path.glob(".memory-payload-*"))
    assert len(list(tmp_path.glob("memory-payload-*.safetensors"))) == int(existing or committed)


def test_recovery_never_resets_unknown_or_corrupt_finalized_data(tmp_path):
    for name in ("private.txt", ".memory-payload-not-a-transaction.json.tmp",
                 "memory-payload-" + "a" * 32 + ".safetensors"):
        directory = tmp_path / str(len(list(tmp_path.iterdir())))
        directory.mkdir()
        (directory / name).write_bytes(b"keep these bytes")
        before = tree_bytes(directory)
        with pytest.raises(MemoryStateError):
            ParametricPayloadMemory.recover_uncommitted(directory)
        assert tree_bytes(directory) == before
    # A valid live M does not authorize deleting a corrupt orphan or unknown data.
    committed = tmp_path / "committed"
    ParametricPayloadMemory().save(committed)
    (committed / "private.txt").write_bytes(b"keep")
    (committed / ("memory-payload-" + "b" * 32 + ".safetensors")).write_bytes(b"bad")
    before = tree_bytes(committed)
    with pytest.raises(MemoryStateError, match="orphan"):
        ParametricPayloadMemory.recover_uncommitted(committed)
    assert tree_bytes(committed) == before


def test_missing_committed_manifest_and_corrupt_prepare_are_never_reset(tmp_path):
    lost_manifest = tmp_path / "lost-manifest"
    changed(ParametricPayloadMemory()).save(lost_manifest)
    (lost_manifest / "memory.json").unlink()
    before = tree_bytes(lost_manifest)
    with pytest.raises(MemoryStateError, match="Missing committed manifest"):
        ParametricPayloadMemory.recover_uncommitted(lost_manifest)
    assert tree_bytes(lost_manifest) == before
    bad_prepare = tmp_path / "bad-prepare"
    bad_prepare.mkdir()
    (bad_prepare / (".memory-payload-" + "a" * 32 + ".prepare")).write_bytes(b"corrupt")
    before = tree_bytes(bad_prepare)
    with pytest.raises(MemoryStateError, match="prepare marker"):
        ParametricPayloadMemory.recover_uncommitted(bad_prepare)
    assert tree_bytes(bad_prepare) == before


def test_recovery_retires_owned_artifacts_but_preserves_unknown_files(tmp_path):
    memory = ParametricPayloadMemory()
    memory.save(tmp_path)
    memory.save(tmp_path)  # An older, valid finalized snapshot is unreferenced.
    (tmp_path / (".memory-payload-" + "a" * 32 + ".json.tmp")).write_bytes(b"partial")
    unknown = tmp_path / "private.txt"
    unknown.write_bytes(b"keep")
    ParametricPayloadMemory.recover_uncommitted(tmp_path)
    assert unknown.read_bytes() == b"keep"
    assert len(list(tmp_path.glob("memory-payload-*.safetensors"))) == 1
    assert ParametricPayloadMemory.load(tmp_path).content_digest() == memory.content_digest()


def test_recovery_syncs_visible_manifest_before_retiring_any_snapshot(tmp_path, monkeypatch):
    memory = ParametricPayloadMemory()
    memory.save(tmp_path)
    changed(memory).save(tmp_path)
    before = tree_bytes(tmp_path)

    def fail_recovery_sync(_directory):
        raise OSError("cannot make the visible manifest durable")

    monkeypatch.setattr(payload_matrix, "_sync_directory", fail_recovery_sync)
    with pytest.raises(OSError, match="visible manifest"):
        ParametricPayloadMemory.recover_uncommitted(tmp_path)
    assert tree_bytes(tmp_path) == before
