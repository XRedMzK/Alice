"""Independent source-bound fast weights; no transcript or Qwen-sized state."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from torch import nn

DIM = 128
LATENT_BASIS_ID = "alice.memory-v0/128/v1"
DEFAULT_SOURCES = ("agent", "user", "person:1", "person:2")


class MemoryStateError(ValueError):
    """Invalid source, tensor or independently persisted memory state."""


def _vector(value: torch.Tensor, matrix: torch.Tensor, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.shape != (DIM,)
            or value.dtype != torch.float32 or value.device != matrix.device):
        raise MemoryStateError(f"{name} must be float32[{DIM}] on {matrix.device}")
    if not torch.isfinite(value).all():
        raise MemoryStateError(f"{name} must be finite")
    return value


def _key(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    value = _vector(value, matrix, "key/query")
    norm = torch.linalg.vector_norm(value)
    if norm <= 1e-12:
        raise MemoryStateError("key/query cannot have zero norm")
    return value / norm


def functional_delta(matrix: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     source_index: int, beta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Functional write for disposable training episodes; preserves all gradients."""
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
            or matrix.shape[1:] != (DIM, DIM) or matrix.dtype != torch.float32
            or not torch.isfinite(matrix).all()):
        raise MemoryStateError(f"matrix must be finite float32[S,{DIM},{DIM}]")
    if type(source_index) is not int or not 0 <= source_index < matrix.shape[0]:
        raise MemoryStateError("Invalid source index")
    if (not isinstance(beta, (int, float)) or isinstance(beta, bool)
            or not math.isfinite(beta) or not 0 < beta <= 1
            or not isinstance(eps, (int, float)) or isinstance(eps, bool)
            or not math.isfinite(eps) or eps <= 0):
        raise MemoryStateError("Require 0 < beta <= 1 and finite eps > 0")
    key = _key(key, matrix)
    value = _vector(value, matrix, "value")
    plane = matrix[source_index]
    updated = plane + beta * torch.outer(value - plane @ key, key) / (eps + key @ key)
    if not torch.isfinite(updated).all():
        raise MemoryStateError("Write would create nonfinite memory")
    result = matrix.clone()
    result[source_index] = updated
    return result


class AssociativeMemory(nn.Module):
    """Owned FP32 parameters. Acquisition changes only these fast weights.

    Identity registry/basis are fixed before experience acquisition. Read is pure;
    save/restore carries no fact text, per-fact lookup or language-model tensors.
    """

    def __init__(self, source_ids=DEFAULT_SOURCES, latent_basis_id=LATENT_BASIS_ID,
                 device="cpu") -> None:
        super().__init__()
        try:
            sources = tuple(source_ids)
        except TypeError as exc:
            raise MemoryStateError("source_ids must be an iterable of explicit IDs") from exc
        if (not sources or len(sources) > 64
                or any(not isinstance(s, str) or not re.fullmatch(
                    r"agent|user|person:[1-9][0-9]*", s) for s in sources)
                or len(set(sources)) != len(sources)):
            raise MemoryStateError("Require unique explicit source IDs (maximum 64)")
        if not isinstance(latent_basis_id, str) or not 1 <= len(latent_basis_id) <= 128:
            raise MemoryStateError("Invalid latent_basis_id")
        self.source_ids = sources
        self.latent_basis_id = latent_basis_id
        self.matrix = nn.Parameter(torch.zeros(len(sources), DIM, DIM,
                                               dtype=torch.float32, device=device),
                                   requires_grad=False)

    def source_index(self, source_id: str) -> int:
        try:
            return self.source_ids.index(source_id)
        except ValueError as exc:
            raise MemoryStateError(f"Unknown source_id: {source_id}") from exc

    def read(self, source_id: str, query: torch.Tensor) -> torch.Tensor:
        if (self.matrix.dtype != torch.float32 or self.matrix.requires_grad
                or not torch.isfinite(self.matrix).all()):
            raise MemoryStateError("Persistent memory must remain frozen FP32 outside writes")
        return self.matrix[self.source_index(source_id)] @ _key(query, self.matrix)

    def content_digest(self) -> str:
        payload = self.matrix.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @torch.no_grad()
    def acquire(self, source_id: str, key: torch.Tensor, value: torch.Tensor,
                beta: float = 1.0) -> dict:
        if self.matrix.requires_grad:
            raise MemoryStateError("Acquisition cannot optimize persistent memory by autograd")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise MemoryStateError("Acquisition requires tensor key/value")
        before = self.content_digest()
        updated = functional_delta(self.matrix, key.detach(), value.detach(),
                                   self.source_index(source_id), beta)
        delta_norm = float(torch.linalg.vector_norm(updated - self.matrix).item())
        self.matrix.copy_(updated)
        return {"source_id": source_id, "before": before,
                "after": self.content_digest(), "delta_norm": delta_norm}

    def save(self, directory: str | Path) -> dict:
        """Publish immutable payload then atomically switch the small manifest."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if self.matrix.dtype != torch.float32 or not torch.isfinite(self.matrix).all():
            raise MemoryStateError("Cannot save nonfinite or non-FP32 memory")
        snapshot = uuid4().hex
        filename = f"memory-{snapshot}.safetensors"
        payload = root / filename
        temporary = None
        committed = False
        try:
            save_file({"matrix": self.matrix.detach().cpu().contiguous()}, str(payload))
            with payload.open("rb") as handle:
                os.fsync(handle.fileno())
            manifest = {"schema": "alice.memory/v0", "update_rule": "normalized-delta/v1",
                        "latent_basis_id": self.latent_basis_id,
                        "source_ids": list(self.source_ids), "shape": list(self.matrix.shape),
                        "dtype": "float32", "snapshot_id": snapshot, "tensor_file": filename,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root,
                                             prefix=".memory-", suffix=".json", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "memory.json")
            committed = True
            return manifest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not committed:
                payload.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, latent_basis_id=LATENT_BASIS_ID,
             device="cpu") -> AssociativeMemory:
        """Validate before exposing a new owner; never reset an existing instance."""
        root = Path(directory).resolve()
        try:
            metadata_path = root / "memory.json"
            if metadata_path.stat().st_size > 32768:
                raise MemoryStateError("Oversized memory metadata")
            manifest = json.loads(metadata_path.read_text())
            fields = {"schema", "update_rule", "latent_basis_id", "source_ids", "shape",
                      "dtype", "snapshot_id", "tensor_file", "sha256"}
            if (not isinstance(manifest, dict) or set(manifest) != fields
                    or manifest["schema"] != "alice.memory/v0"
                    or manifest["update_rule"] != "normalized-delta/v1"
                    or manifest["dtype"] != "float32"
                    or not isinstance(manifest["source_ids"], list)
                    or not isinstance(manifest["shape"], list)
                    or manifest["latent_basis_id"] != latent_basis_id
                    or not re.fullmatch(r"[0-9a-f]{32}", manifest["snapshot_id"])):
                raise MemoryStateError("Incompatible memory schema/basis")
            filename = f"memory-{manifest['snapshot_id']}.safetensors"
            if manifest["tensor_file"] != filename:
                raise MemoryStateError("Invalid memory payload reference")
            payload = root / filename
            if (payload.is_symlink() or not payload.is_file()
                    or payload.stat().st_size > 64 * DIM * DIM * 4 + 65536):
                raise MemoryStateError("Memory payload must be a regular local file")
            if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["sha256"]:
                raise MemoryStateError("Memory payload integrity mismatch")
            memory = cls(manifest["source_ids"], latent_basis_id, device)
            tensors = load_file(str(payload), device="cpu")
            value = tensors.get("matrix")
            if (set(tensors) != {"matrix"} or not isinstance(value, torch.Tensor)
                    or list(value.shape) != manifest["shape"]
                    or value.shape != memory.matrix.shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise MemoryStateError("Invalid memory tensor")
            with torch.no_grad():
                memory.matrix.copy_(value.to(device))
            return memory
        except MemoryStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
            raise MemoryStateError(f"Cannot restore memory: {exc}") from exc
