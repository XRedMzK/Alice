"""Independent parameter/Bridge backup and safe fresh restore; CPU assets only."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alice_memory.payload_matrix import ParametricPayloadMemory
from alice_runtime import memory_backup
from alice_runtime.memory_backup import (
    MemoryBackupError, create_memory_backup, restore_memory_backup,
)
from alice_runtime.neural_memory import NeuralMemorySession
from test_runtime_memory_store import setup, tree_bytes


CUE = "What is the assigned hue of item C-42?"


def test_live_snapshot_restore_keeps_compatible_schemas_and_original_files(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        app.remember("person:1", CUE, "ivory")
        before_state, before_bridge = tree_bytes(state_path), tree_bytes(bridge_path)
        original_digest = app.memory.content_digest()
        snapshot = app.backup(tmp_path / "snapshot")
        assert snapshot["memory_digest"] == original_digest
        assert snapshot["bridge_digest"] == app.bridge.parameter_digest()
        assert tree_bytes(state_path) == before_state
        assert tree_bytes(bridge_path) == before_bridge
        assert sorted(Path(snapshot["backup_path"]).iterdir()) == sorted([
            tmp_path / "snapshot" / name for name in ("backup.json", "memory", "bridge")])
        metadata = json.loads((tmp_path / "snapshot" / "backup.json").read_text())
        assert not {"cue", "value", "messages", "facts", "paths"} & metadata.keys()
        assert tree_bytes(snapshot["memory_path"]) == {
            name: value for name, value in before_state.items() if name != ".alice-memory.lock"}
        assert tree_bytes(snapshot["bridge_path"]) == before_bridge
        app.remember("user", CUE, "copper")
        corrected_digest = app.memory.content_digest()
    source_after = tree_bytes(state_path)
    restored = restore_memory_backup(tmp_path / "snapshot", tmp_path / "restored",
                                     expected_bridge_digest=snapshot["bridge_digest"])
    assert restored["restored"] and restored["memory_digest"] == original_digest
    assert tree_bytes(state_path) == source_after
    assert tree_bytes(bridge_path) == before_bridge
    assert original_digest != corrected_digest
    with NeuralMemorySession(loaded, restored["bridge_path"], restored["memory_path"]) as reopened:
        assert reopened.status()["memory_digest"] == original_digest
        assert reopened.status()["bridge_digest"] == snapshot["bridge_digest"]
        torch.testing.assert_close(reopened.memory.matrix,
            ParametricPayloadMemory.load(snapshot["memory_path"], reopened.bridge.basis_id).matrix)
    source_payload = next(state_path.glob("memory-payload-*.safetensors"))
    restored_payload = next(Path(restored["memory_path"]).glob("memory-payload-*.safetensors"))
    assert os.stat(source_payload).st_ino != os.stat(restored_payload).st_ino


def test_closed_backup_uses_existing_lock_and_refuses_active_writer(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before = tree_bytes(state_path)
        with pytest.raises(MemoryBackupError, match="active"):
            create_memory_backup(state_path, bridge_path, tmp_path / "active")
        assert tree_bytes(state_path) == before
        assert not (tmp_path / "active").exists()
    before = tree_bytes(state_path)
    saved = create_memory_backup(state_path, bridge_path, tmp_path / "closed")
    assert saved["memory_digest"] == app.memory.content_digest()
    assert tree_bytes(state_path) == before


def test_backup_restores_in_a_fresh_process_without_session_or_model_objects(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        snapshot = app.backup(tmp_path / "snapshot")
    code = """
import json,sys
from alice_runtime.memory_backup import restore_memory_backup
print(json.dumps(restore_memory_backup(sys.argv[1],sys.argv[2],expected_bridge_digest=sys.argv[3])))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path / "snapshot"),
         str(tmp_path / "restored"), snapshot["bridge_digest"]],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        capture_output=True, text=True, check=True, timeout=20,
    )
    restored = json.loads(result.stdout)
    assert restored["restored"]
    assert restored["memory_digest"] == snapshot["memory_digest"]
    assert restored["bridge_digest"] == snapshot["bridge_digest"]


def test_backup_requires_committed_memory_and_does_not_create_lock_in_old_assets(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        before = tree_bytes(state_path)
        with pytest.raises(ValueError, match="Save memory"):
            app.backup(tmp_path / "uncommitted")
        assert tree_bytes(state_path) == before
    historical = tmp_path / "old-checkpoint"
    ParametricPayloadMemory().save(historical)
    before = tree_bytes(historical)
    with pytest.raises(MemoryBackupError, match="closed memory checkpoint"):
        create_memory_backup(historical, bridge_path, tmp_path / "old-backup")
    assert tree_bytes(historical) == before


def test_backup_refuses_active_roots_existing_targets_symlinks_and_hardlinks(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    bindings = {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                for name in ("config.json", "tokenizer_config.json")}
    loaded, bridge_path, state_path = setup(tmp_path, bindings)
    with NeuralMemorySession(loaded, bridge_path, state_path, model_path=model) as app:
        app.remember("user", CUE, "quartz")
        before = {str(path): tree_bytes(path) for path in (state_path, bridge_path, model)}
        occupied = tmp_path / "existing"
        occupied.mkdir()
        (occupied / "keep").write_bytes(b"keep")
        alias = tmp_path / "alias"
        alias.symlink_to(state_path, target_is_directory=True)
        hardlink = tmp_path / "hardlink"
        os.link(state_path / "memory.json", hardlink)
        for destination in (state_path / "backup", bridge_path / "backup", model / "backup",
                            occupied, alias / "backup", hardlink, state_path):
            with pytest.raises(MemoryBackupError):
                app.backup(destination)
        assert (occupied / "keep").read_bytes() == b"keep"
        assert hardlink.read_bytes() == (state_path / "memory.json").read_bytes()
        assert all(tree_bytes(path) == before[str(path)] for path in (state_path, bridge_path, model))


@pytest.mark.parametrize("corruption", ["payload", "metadata", "unknown", "bridge"])
def test_corrupt_or_mismatched_backup_never_creates_restore_destination(tmp_path, corruption):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        snapshot = app.backup(tmp_path / "snapshot")
    before = tree_bytes(state_path)
    if corruption == "payload":
        payload = next(Path(snapshot["memory_path"]).glob("*.safetensors"))
        with payload.open("ab") as handle:
            handle.write(b"corrupt")
    elif corruption == "metadata":
        manifest = tmp_path / "snapshot" / "backup.json"
        values = json.loads(manifest.read_text())
        values["memory_digest"] = "0" * 64
        manifest.write_text(json.dumps(values))
    elif corruption == "bridge":
        manifest = Path(snapshot["bridge_path"]) / "bridge.json"
        values = json.loads(manifest.read_text())
        values["latent_basis_id"] = "different-basis"
        manifest.write_text(json.dumps(values))
    else:
        (tmp_path / "snapshot" / "unknown.txt").write_text("keep")
    corrupt_before = tree_bytes(tmp_path / "snapshot")
    with pytest.raises(MemoryBackupError):
        restore_memory_backup(tmp_path / "snapshot", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert tree_bytes(tmp_path / "snapshot") == corrupt_before
    assert tree_bytes(state_path) == before


def test_restore_refuses_existing_state_and_wrong_trusted_bridge(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        app.backup(tmp_path / "snapshot")
        before = tree_bytes(state_path)
        with pytest.raises(MemoryBackupError):
            restore_memory_backup(tmp_path / "snapshot", state_path)
        for root in (state_path, bridge_path):
            with pytest.raises(MemoryBackupError, match="separate"):
                restore_memory_backup(tmp_path / "snapshot", root / "nested-restore")
        with pytest.raises(MemoryBackupError, match="expected immutable Bridge"):
            restore_memory_backup(tmp_path / "snapshot", tmp_path / "wrong",
                                  expected_bridge_digest="0" * 64)
        assert tree_bytes(state_path) == before


def test_child_substitution_prevents_ack_and_never_writes_through_alias(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before = tree_bytes(state_path)
        pair = memory_backup._pair
        destination = tmp_path / "snapshot"
        substituted = False

        def substitute_child(*args, **kwargs):
            nonlocal substituted
            result = pair(*args, **kwargs)
            if not substituted:
                substituted = True
                (destination / "memory").rename(destination / "owned-memory")
                (destination / "memory").symlink_to(state_path, target_is_directory=True)
            return result

        monkeypatch.setattr(memory_backup, "_pair", substitute_child)
        with pytest.raises(MemoryBackupError, match="child moved"):
            app.backup(destination)
        assert tree_bytes(state_path) == before
        assert (destination / "memory").is_symlink()
        with pytest.raises(MemoryBackupError):
            restore_memory_backup(destination, tmp_path / "restore")


def test_parent_swap_cannot_redirect_backup_writes_into_memory(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    parent, moved = tmp_path / "backups", tmp_path / "moved-backups"
    parent.mkdir()
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before = tree_bytes(state_path)
        mkdir = os.mkdir
        swapped = False

        def swap_parent(name, *args, **kwargs):
            nonlocal swapped
            if name == "snapshot" and not swapped:
                swapped = True
                parent.rename(moved)
                parent.symlink_to(state_path, target_is_directory=True)
            return mkdir(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(memory_backup.os, "mkdir", swap_parent)
            with pytest.raises(MemoryBackupError):
                app.backup(parent / "snapshot")
        assert tree_bytes(state_path) == before
        assert not (state_path / "snapshot").exists()
        assert (moved / "snapshot").is_dir()  # Failed copies are retained safely.


def test_raced_hardlink_is_never_truncated_and_unknown_destination_entry_is_kept(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before = tree_bytes(state_path)
        write = memory_backup._write_file
        injected = False

        def insert_hardlink(directory, name, data, created):
            nonlocal injected
            if name == "memory.json" and not injected:
                injected = True
                os.link(state_path / "memory.json", name, dst_dir_fd=directory)
            return write(directory, name, data, created)

        monkeypatch.setattr(memory_backup, "_write_file", insert_hardlink)
        with pytest.raises(MemoryBackupError):
            app.backup(tmp_path / "snapshot")
        assert tree_bytes(state_path) == before
        retained = tmp_path / "snapshot" / "memory" / "memory.json"
        assert retained.read_bytes() == (state_path / "memory.json").read_bytes()
        assert not (tmp_path / "snapshot" / "backup.json").exists()


def test_failed_copy_cannot_change_source_or_acknowledge_a_bundle(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before, before_bridge = tree_bytes(state_path), tree_bytes(bridge_path)

        def failed_sync(_descriptor):
            raise OSError("copy device full")

        monkeypatch.setattr(memory_backup.os, "fsync", failed_sync)
        with pytest.raises(MemoryBackupError, match="device full"):
            app.backup(tmp_path / "snapshot")
        assert tree_bytes(state_path) == before
        assert tree_bytes(bridge_path) == before_bridge
        assert (tmp_path / "snapshot").is_dir()
        assert not (tmp_path / "snapshot" / "backup.json").exists()


def test_unsupported_mount_is_refused_before_any_destination_is_created(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", CUE, "quartz")
        before = tree_bytes(state_path)

        def subtree_mount(_descriptor):
            raise ValueError("unsupported subtree mount")

        monkeypatch.setattr(memory_backup, "assert_whole_filesystem_mount", subtree_mount)
        with pytest.raises(MemoryBackupError, match="subtree mount"):
            app.backup(tmp_path / "snapshot")
        assert not (tmp_path / "snapshot").exists()
        assert tree_bytes(state_path) == before


@pytest.mark.parametrize("command", ["create", "restore"])
def test_offline_cli_refuses_fifo_input_without_waiting_for_a_writer(tmp_path, command):
    source, bridge, output = (tmp_path / name for name in ("source", "bridge", "not-created"))
    source.mkdir()
    bridge.mkdir()
    arguments = [sys.executable, "-B", "-m", "alice_runtime.memory_backup", command,
                 "--output", str(output), "--expected-bridge-digest", "0" * 64]
    if command == "create":
        fifo = source / ".alice-memory.lock"
        os.mkfifo(fifo)
        arguments.extend(["--memory-state", str(source), "--bridge", str(bridge)])
    else:
        fifo = source / "backup.json"
        os.mkfifo(fifo)
        (source / "memory").mkdir()
        (source / "bridge").mkdir()
        arguments.extend(["--backup", str(source)])
    identity = fifo.stat().st_dev, fifo.stat().st_ino, fifo.stat().st_mode
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=8,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                 "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
                                 "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert result.returncode == 1 and result.stdout == ""
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["error_code"] == "memory_backup_failed" and "regular file" in error["error"]
    assert not output.exists()
    assert (fifo.stat().st_dev, fifo.stat().st_ino, fifo.stat().st_mode) == identity
