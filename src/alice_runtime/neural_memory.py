"""Persistent ownership of the existing frozen-Bridge parametric memory path.

This service does not parse natural-language facts or keep their text. The
explicit caller supplies source/cue/value only for acquisition. Queries use the
existing dense payload readout, with no session history or token lookup.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.checkpoints import restore_bridge
from alice_memory.matrix import MemoryStateError
from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_memory.payload_matrix import ParametricPayloadMemory
from alice_memory.payload_runtime import PayloadMemoryRuntime
from alice_memory.runtime import Fact, Question


class NeuralMemorySession:
    """One writable memory directory, one Qwen instance, and an immutable Bridge.

    The process lock lasts until close. A successful remember acknowledgement
    follows atomic publication of M; a failed publication restores the previous
    in-memory parameters. Bridge files are read only and have another lifecycle.
    """

    LOCK_FILE = ".alice-memory.lock"

    def __init__(self, loaded, bridge_path, state_path, model_path=None,
                 max_tokens: int = 128):
        self._lock = threading.RLock()
        self._lock_fd = None
        self._closed = False
        self._persisted = False
        self._cleanup_warning = None
        self.bridge_path = Path(bridge_path).expanduser().resolve()
        raw_state = Path(state_path).expanduser().absolute()
        if any(path.is_symlink() for path in (raw_state, *raw_state.parents)):
            raise MemoryStateError("Memory directory must not traverse symlinks")
        self.state_path = raw_state.resolve()
        if (self.state_path == self.bridge_path
                or self.state_path.is_relative_to(self.bridge_path)
                or self.bridge_path.is_relative_to(self.state_path)):
            raise MemoryStateError("Memory and immutable Bridge directories must be separate")
        model_path = model_path or loaded.metadata.get("model_path")
        expected_bindings = None
        if model_path is not None:
            model_root = Path(model_path).expanduser().resolve()
            if (self.state_path == model_root or self.state_path.is_relative_to(model_root)
                    or model_root.is_relative_to(self.state_path)):
                raise MemoryStateError("Memory directory must be separate from model assets")
            expected_bindings = {
                name: hashlib.sha256((model_root / name).read_bytes()).hexdigest()
                for name in ("config.json", "tokenizer_config.json")
            }
        if type(max_tokens) is not int or not 8 <= max_tokens <= 1024:
            raise MemoryStateError("Memory context budget must be 8..1024 tokens")
        device = loaded.model.get_input_embeddings().weight.device
        self.bridge = restore_bridge(self.bridge_path, device=device,
                                     expected_bindings=expected_bindings)
        if not isinstance(self.bridge, PayloadEmbeddingBridge):
            raise BridgeStateError("Interactive memory requires the versioned full-rank payload Bridge")
        try:
            self._validate_directory()
            self.state_path.mkdir(parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
            self._lock_fd = os.open(self.state_path / self.LOCK_FILE, flags, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MemoryStateError("Memory directory is already open in another runtime") from exc
            # Revalidate after taking ownership; never initialize over an invalid
            # or partly present checkpoint.
            self._validate_directory()
            manifest = self.state_path / "memory.json"
            if manifest.exists():
                self.memory = ParametricPayloadMemory.load(
                    self.state_path, latent_basis_id=self.bridge.basis_id, device=device)
                self._persisted = True
            else:
                self.memory = ParametricPayloadMemory(
                    latent_basis_id=self.bridge.basis_id, device=device)
            if not {"agent", "user"}.issubset(self.memory.source_ids):
                raise MemoryStateError("Interactive memory requires stable agent and user source IDs")
            self.engine = PayloadMemoryRuntime(loaded, self.bridge, self.memory, max_tokens)
        except BaseException:
            self.close()
            raise

    def _validate_directory(self):
        if not self.state_path.exists():
            return
        if not self.state_path.is_dir():
            raise MemoryStateError("Memory state path must be a directory")
        entries = tuple(self.state_path.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in entries):
            raise MemoryStateError("Memory directory must contain regular local files only")
        manifest = self.state_path / "memory.json"
        if not manifest.exists() and any(p.name != self.LOCK_FILE for p in entries):
            raise MemoryStateError("Nonempty directory has no memory manifest; refusing to reset it")

    def _assert_open(self):
        if self._closed or self._lock_fd is None:
            raise MemoryStateError("Neural Memory session is closed")

    def _publish(self):
        previous = None
        if self._persisted:
            metadata = json.loads((self.state_path / "memory.json").read_text())
            # ParametricPayloadMemory.save validates this checkpoint before its
            # atomic switch. Only the formerly referenced, same-directory neural
            # snapshot may subsequently be retired; unknown files are untouched.
            previous = metadata.get("tensor_file")
        manifest = self.memory.save(self.state_path)
        self._persisted = True
        if (isinstance(previous, str)
                and re.fullmatch(r"memory-payload-[0-9a-f]{32}\.safetensors", previous)
                and previous != manifest["tensor_file"]):
            try:
                (self.state_path / previous).unlink(missing_ok=True)
            except OSError as exc:
                # Publication already succeeded. A duplicate snapshot is a
                # recoverable cleanup warning, never a false failed-write ACK.
                self._cleanup_warning = str(exc)
        return manifest

    def remember(self, source_id: str, cue: str, value: str) -> dict:
        """Acquire an explicit one-token value, then durably acknowledge M."""
        with self._lock:
            self._assert_open()
            before = self.memory.matrix.detach().clone()
            bridge_digest = self.bridge.parameter_digest()
            try:
                receipt = self.engine.acquire(Fact(source_id, cue, value))
                self._publish()
            except BaseException:
                with torch.no_grad():
                    self.memory.matrix.copy_(before)
                raise
            return {**receipt, "saved": True,
                    "persistent_m_changed": receipt["after"] != receipt["before"],
                    "memory_digest": self.memory.content_digest(),
                    "bridge_digest": bridge_digest, "bridge_unchanged": True,
                    "qwen_frozen": True, "state_path": str(self.state_path)}

    def respond(self, request: Question, mode: str = "on", max_new_tokens: int = 8) -> dict:
        with self._lock:
            self._assert_open()
            if mode not in {"on", "off", "empty"}:
                raise MemoryStateError("Interactive memory mode must be on, off or empty")
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 16:
                raise MemoryStateError("Memory generation budget must be 1..16 tokens")
            return self.engine.answer(request, mode, max_new_tokens)

    def save(self) -> dict:
        """Explicitly publish current M, including an intentionally empty state."""
        with self._lock:
            self._assert_open()
            self.bridge.assert_frozen()
            self.engine.assert_qwen_frozen()
            self._publish()
            return {**self.status(), "saved": True}

    def status(self) -> dict:
        with self._lock:
            result = {
                "schema": self.memory.SCHEMA,
                "latent_basis_id": self.memory.latent_basis_id,
                "source_ids": list(self.memory.source_ids),
                "memory_digest": self.memory.content_digest(),
                "bridge_digest": self.bridge.parameter_digest(),
                "state_path": str(self.state_path), "bridge_path": str(self.bridge_path),
                "persisted": self._persisted, "closed": self._closed,
            }
            if self._cleanup_warning:
                result["cleanup_warning"] = self._cleanup_warning
            return result

    def close(self):
        with self._lock:
            if self._lock_fd is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = None
            self._closed = True

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self, *_exc):
        self.close()
