"""Validated M + matching Bridge snapshots; never replace a live checkpoint.

Backups contain neural checkpoint files and small compatibility metadata only.
Restore always creates a new directory; it does not reset or migrate existing M.
Linux/POSIX directory descriptors pin every directory used for writes.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
import torch

from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_memory.payload_matrix import ParametricPayloadMemory, _metadata_digest

from .export import (
    assert_separate_directory, assert_whole_filesystem_mount, open_directory_fd,
)


SCHEMA = "alice.memory_backup/v1"
_PAYLOAD_LIMIT = 64 * 17 * 128 * 128 * 4 + 65536
_NOFOLLOW = os.O_NOFOLLOW
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW
_CHECKPOINT_MARKERS = ("memory.json", ".alice-memory.lock", "bridge.json", "config.json",
                       "tokenizer_config.json", "model.safetensors.index.json")


class MemoryBackupError(RuntimeError):
    """A backup/restore could not be safely validated or acknowledged."""

    error_code = "memory_backup_error"


def _identity(value):
    return value.st_dev, value.st_ino


def _leaf(name):
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise MemoryBackupError("Checkpoint files must have plain local filenames")
    return name


def _read_file(directory_fd, name, limit):
    descriptor = os.open(_leaf(name), os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise MemoryBackupError("Checkpoint input must be a bounded regular file")
        parts, remaining = [], limit + 1
        while remaining:
            part = os.read(descriptor, min(remaining, 2**20))
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
        result = b"".join(parts)
        if len(result) > limit:
            raise MemoryBackupError("Checkpoint grew beyond its allowed size")
        return result
    finally:
        os.close(descriptor)


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MemoryBackupError("Duplicate checkpoint metadata key")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise MemoryBackupError("Checkpoint metadata must be an object")
    return value


def _write_file(directory_fd, name, content, created):
    descriptor = os.open(_leaf(name), os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW,
                         0o600, dir_fd=directory_fd)
    created.append((directory_fd, name, _identity(os.fstat(descriptor))))
    try:
        data = memoryview(content)
        while data:
            written = os.write(descriptor, data)
            if written <= 0:
                raise MemoryBackupError("Incomplete checkpoint write")
            data = data[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _refuse_checkpoint_ancestry(directory_fd):
    """Keep standalone restores out of recognizable assets, even without hints."""
    current = os.dup(directory_fd)
    try:
        for _ in range(1024):
            for name in _CHECKPOINT_MARKERS:
                try:
                    os.stat(name, dir_fd=current, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise MemoryBackupError("Backup destination must be separate from memory, Bridge and model assets")
            parent = os.open("..", _DIRECTORY, dir_fd=current)
            if _identity(os.fstat(parent)) == _identity(os.fstat(current)):
                os.close(parent)
                return
            os.close(current)
            current = parent
        raise MemoryBackupError("Destination ancestry exceeds the supported bound")
    finally:
        os.close(current)


def _pair(memory_fd, bridge_fd, expected_memory_digest=None, expected_bridge_digest=None):
    # Read the actual pinned files into bounded bytes. Path.resolve() in the
    # ordinary checkpoint loaders would discard the directory descriptor pin.
    bmeta = _json(_read_file(bridge_fd, "bridge.json", 32768))
    filename = _leaf(bmeta.get("tensor_file"))
    if (bmeta.get("schema") != PayloadEmbeddingBridge.SCHEMA
            or not filename.startswith("payload-bridge-") or not filename.endswith(".safetensors")
            or type(bmeta.get("rotation_seed")) is not int
            or not isinstance(bmeta.get("payload_shape"), list)
            or any(type(size) is not int for size in bmeta["payload_shape"])):
        raise MemoryBackupError("Invalid payload Bridge manifest")
    if set(os.listdir(bridge_fd)) != {"bridge.json", filename}:
        raise MemoryBackupError("Backup Bridge directory contains unexpected files")
    payload = _read_file(bridge_fd, filename, 2**20)
    if hashlib.sha256(payload).hexdigest() != bmeta.get("sha256"):
        raise MemoryBackupError("Payload Bridge tensor digest mismatch")
    bridge = PayloadEmbeddingBridge(bmeta["hidden_size"], bmeta["key_dim"], bmeta["slots"],
                                    bmeta["latent_basis_id"], bmeta["bindings"],
                                    bandwidth=bmeta["bandwidth"], value_scale=bmeta["value_scale"])
    config = bridge._config()
    if (set(bmeta) != set(config) | {"latent_basis_id", "tensor_file", "sha256", "state_digest"}
            or any(bmeta[key] != value for key, value in config.items())):
        raise MemoryBackupError("Incompatible payload Bridge metadata")
    tensors = load_tensors(payload)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in tensors.values()):
        raise MemoryBackupError("Bridge tensors must remain finite FP32")
    bridge.load_state_dict(tensors, strict=True)
    bridge._validate_state()
    if any(bmeta[key] != value for key, value in bridge._config().items()):
        raise MemoryBackupError("Payload Bridge tensor semantics/configuration mismatch")
    if bridge.parameter_digest() != bmeta["state_digest"]:
        raise MemoryBackupError("Payload Bridge state digest mismatch")
    mmeta = _json(_read_file(memory_fd, "memory.json", 32768))
    filename = _leaf(mmeta.get("tensor_file"))
    fields = {"schema", "update_rule", "latent_basis_id", "source_ids", "shape",
              "dtype", "snapshot_id", "tensor_file", "sha256", "metadata_sha256"}
    if (set(mmeta) != fields or mmeta["schema"] != ParametricPayloadMemory.SCHEMA
            or mmeta["update_rule"] != "parallel-normalized-delta/v1"
            or mmeta["dtype"] != "float32" or mmeta["latent_basis_id"] != bridge.basis_id
            or not isinstance(mmeta["source_ids"], list) or not isinstance(mmeta["shape"], list)
            or not isinstance(mmeta["snapshot_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", mmeta["snapshot_id"])
            or filename != f"memory-payload-{mmeta['snapshot_id']}.safetensors"
            or _metadata_digest(mmeta) != mmeta["metadata_sha256"]):
        raise MemoryBackupError("Incompatible or corrupt payload memory manifest")
    if set(os.listdir(memory_fd)) != {"memory.json", filename}:
        raise MemoryBackupError("Backup memory directory contains unexpected files")
    payload = _read_file(memory_fd, filename, _PAYLOAD_LIMIT)
    if hashlib.sha256(payload).hexdigest() != mmeta["sha256"]:
        raise MemoryBackupError("Payload memory tensor digest mismatch")
    memory = ParametricPayloadMemory(mmeta["source_ids"], bridge.basis_id)
    tensors = load_tensors(payload)
    value = tensors.get("matrix")
    if (set(tensors) != {"matrix"} or not isinstance(value, torch.Tensor)
            or list(value.shape) != mmeta["shape"] or value.shape != memory.matrix.shape
            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
        raise MemoryBackupError("Invalid payload memory tensor")
    with torch.no_grad():
        memory.matrix.copy_(value)
    memory_digest, bridge_digest = memory.content_digest(), bridge.parameter_digest()
    if expected_memory_digest is not None and memory_digest != expected_memory_digest:
        raise MemoryBackupError("Committed memory differs from the owned in-memory state")
    if expected_bridge_digest is not None and bridge_digest != expected_bridge_digest:
        raise MemoryBackupError("The checkpoint Bridge differs from the expected immutable Bridge")
    return {"schema": SCHEMA, "memory_digest": memory_digest,
            "bridge_digest": bridge_digest, "latent_basis_id": bridge.basis_id,
            "source_ids": list(memory.source_ids)}


@contextmanager
def _fresh_directory(destination, protected_roots):
    path = Path(destination).expanduser().absolute()
    parent_fd = open_directory_fd(path.parent)
    protected_fds, child_fds, created = [], [], []
    root_fd = None
    try:
        for root in protected_roots:
            protected_fds.append(open_directory_fd(root))
        assert_whole_filesystem_mount(parent_fd)
        assert_separate_directory(parent_fd, protected_fds, reject_ancestor=False)
        _refuse_checkpoint_ancestry(parent_fd)
        # Parent must already exist. mkdir is exclusive and never follows or
        # replaces a destination symlink, hardlink, file or existing directory.
        os.mkdir(_leaf(path.name), 0o700, dir_fd=parent_fd)
        root_fd = os.open(path.name, _DIRECTORY, dir_fd=parent_fd)
        assert_whole_filesystem_mount(root_fd)
        assert_separate_directory(root_fd, protected_fds)
        if os.listdir(root_fd):
            raise MemoryBackupError("Fresh checkpoint destination unexpectedly contains files")
        for name in ("memory", "bridge"):
            os.mkdir(name, 0o700, dir_fd=root_fd)
            descriptor = os.open(name, _DIRECTORY, dir_fd=root_fd)
            child_fds.append(descriptor)
            assert_whole_filesystem_mount(descriptor)
            assert_separate_directory(descriptor, protected_fds)
            if os.listdir(descriptor):
                raise MemoryBackupError("Fresh checkpoint child unexpectedly contains files")
        yield path, root_fd, child_fds[0], child_fds[1], created
        for descriptor in (*child_fds, root_fd, parent_fd):
            os.fsync(descriptor)
        # A moved/replaced parent must never produce an ACK for a different path.
        check_fd = open_directory_fd(path.parent)
        try:
            if (_identity(os.fstat(check_fd)) != _identity(os.fstat(parent_fd))
                    or _identity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False))
                    != _identity(os.fstat(root_fd))):
                raise MemoryBackupError("Checkpoint destination moved; no backup was acknowledged")
            for name, descriptor in zip(("memory", "bridge"), child_fds):
                if (_identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False))
                        != _identity(os.fstat(descriptor))):
                    raise MemoryBackupError("Checkpoint child moved; no backup was acknowledged")
                assert_whole_filesystem_mount(descriptor)
                assert_separate_directory(descriptor, protected_fds)
        finally:
            os.close(check_fd)
    finally:
        # Failed fresh destinations are retained, never cleaned through mutable
        # names. Inspect incomplete copies before removing them manually. There
        # is no successful ACK if publication/durability validation failed.
        for descriptor in reversed(child_fds + ([] if root_fd is None else [root_fd])
                                   + protected_fds + [parent_fd]):
            os.close(descriptor)


def _copy_pair(memory_fd, bridge_fd, target_memory_fd, target_bridge_fd, created):
    for source, target, manifest_name, limit in (
            (memory_fd, target_memory_fd, "memory.json", _PAYLOAD_LIMIT),
            (bridge_fd, target_bridge_fd, "bridge.json", 2**20)):
        manifest_bytes = _read_file(source, manifest_name, 32768)
        filename = _leaf(_json(manifest_bytes).get("tensor_file"))
        payload = _read_file(source, filename, limit)
        _write_file(target, filename, payload, created)
        _write_file(target, manifest_name, manifest_bytes, created)


def _backup_locked(state_path, bridge_path, destination, *, expected_memory_digest=None,
                   expected_bridge_digest=None, protected_roots=(), held_lock_fd=None,
                   expected_state_identity=None):
    """Internal entry: the caller already owns the source's writer lock."""
    state_fd = bridge_fd = None
    try:
        state_fd, bridge_fd = open_directory_fd(state_path), open_directory_fd(bridge_path)
        if (expected_state_identity is not None
                and _identity(os.fstat(state_fd)) != expected_state_identity):
            raise MemoryBackupError("Source memory directory changed while taking its snapshot")
        if (held_lock_fd is not None
                and _identity(os.stat(".alice-memory.lock", dir_fd=state_fd, follow_symlinks=False))
                != _identity(os.fstat(held_lock_fd))):
            raise MemoryBackupError("Source memory directory no longer has the owned process lock")
        protected = (state_path, bridge_path, *protected_roots)
        with _fresh_directory(destination, protected) as (path, root, memory, bridge, created):
            _copy_pair(state_fd, bridge_fd, memory, bridge, created)
            metadata = _pair(memory, bridge, expected_memory_digest, expected_bridge_digest)
            _write_file(root, "backup.json", (json.dumps(metadata, sort_keys=True, indent=2)
                                             + "\n").encode(), created)
        return {**metadata, "backup_path": str(path), "memory_path": str(path / "memory"),
                "bridge_path": str(path / "bridge")}
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, SafetensorError) as exc:
        if isinstance(exc, MemoryBackupError):
            raise
        raise MemoryBackupError(f"Cannot back up memory checkpoint: {exc}") from exc
    finally:
        for descriptor in (state_fd, bridge_fd):
            if descriptor is not None:
                os.close(descriptor)


def create_memory_backup(state_path, bridge_path, destination, *, expected_bridge_digest=None,
                         protected_roots=()):
    """Back up a closed runtime checkpoint without modifying its files.

    The persistent runtime lock file must already exist. Active runtimes must use
    their NeuralMemorySession.backup method, which reuses their held writer lock.
    """
    root_fd = lock_fd = None
    try:
        root_fd = open_directory_fd(state_path)
        lock_fd = os.open(".alice-memory.lock", os.O_RDONLY | _NOFOLLOW, dir_fd=root_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise MemoryBackupError("Memory lock must be a regular file")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MemoryBackupError("Memory is active; use its owning session to create the backup") from exc
        return _backup_locked(state_path, bridge_path, destination,
                              expected_bridge_digest=expected_bridge_digest,
                              protected_roots=protected_roots, held_lock_fd=lock_fd,
                              expected_state_identity=_identity(os.fstat(root_fd)))
    except (OSError, ValueError) as exc:
        raise MemoryBackupError(f"Cannot open closed memory checkpoint for backup: {exc}") from exc
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)


def restore_memory_backup(backup_path, destination, *, expected_bridge_digest=None,
                          protected_roots=()):
    """Restore a validated compatible pair to a fresh directory, never in place."""
    source = memory = bridge = None
    try:
        source = open_directory_fd(backup_path)
        if set(os.listdir(source)) != {"backup.json", "memory", "bridge"}:
            raise MemoryBackupError("Backup must contain exactly its manifest, memory and Bridge")
        metadata = _json(_read_file(source, "backup.json", 16384))
        memory = os.open("memory", _DIRECTORY, dir_fd=source)
        bridge = os.open("bridge", _DIRECTORY, dir_fd=source)
        actual = _pair(memory, bridge, expected_bridge_digest=expected_bridge_digest)
        if metadata != actual:
            raise MemoryBackupError("Backup identity or memory/Bridge compatibility mismatch")
        with _fresh_directory(destination, (backup_path, *protected_roots)) as (
                path, root, target_memory, target_bridge, created):
            _copy_pair(memory, bridge, target_memory, target_bridge, created)
            if _pair(target_memory, target_bridge) != metadata:
                raise MemoryBackupError("Copied backup changed during restore")
            _write_file(root, "backup.json", (json.dumps(metadata, sort_keys=True, indent=2)
                                             + "\n").encode(), created)
        return {**metadata, "restored": True, "backup_path": str(path),
                "memory_path": str(path / "memory"), "bridge_path": str(path / "bridge")}
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, SafetensorError) as exc:
        if isinstance(exc, MemoryBackupError):
            raise
        raise MemoryBackupError(f"Cannot restore memory backup: {exc}") from exc
    finally:
        for descriptor in (source, memory, bridge):
            if descriptor is not None:
                os.close(descriptor)


def main(argv=None):
    """Offline checkpoint utility; no Qwen load, fitting or in-place restore."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Copy a closed M and matching Bridge to a fresh bundle")
    create.add_argument("--memory-state", required=True)
    create.add_argument("--bridge", required=True)
    restore = commands.add_parser("restore", help="Restore a verified bundle into a fresh directory")
    restore.add_argument("--backup", required=True)
    for command in (create, restore):
        command.add_argument("--output", required=True, help="New destination; existing paths are refused")
        command.add_argument("--expected-bridge-digest", required=True,
                             help="Trusted compatible sealed Bridge parameter digest")
        command.add_argument("--protected-root", action="append", default=[],
                             help="Additional immutable root such as the Qwen model directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            receipt = create_memory_backup(args.memory_state, args.bridge, args.output,
                expected_bridge_digest=args.expected_bridge_digest, protected_roots=args.protected_root)
        else:
            receipt = restore_memory_backup(args.backup, args.output,
                expected_bridge_digest=args.expected_bridge_digest, protected_roots=args.protected_root)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "error_code": "memory_backup_failed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
