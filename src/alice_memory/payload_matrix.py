"""Versioned parallel value payload in independently mutable neural parameters."""

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

from .matrix import AssociativeMemory, DEFAULT_SOURCES, DIM, MemoryStateError, _key

PAYLOAD_HEADS = 17
PAYLOAD_BASIS_ID = "alice.memory_payload/17x128/v1"


def _metadata_digest(metadata: dict) -> str:
    content = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def functional_payload_delta(matrix: torch.Tensor, query: torch.Tensor,
                             value: torch.Tensor, source_index: int,
                             beta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Update every value head for one source without mutating input tensors."""
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 4
            or matrix.shape[1:] != (PAYLOAD_HEADS, DIM, DIM)
            or matrix.dtype != torch.float32 or not torch.isfinite(matrix).all()):
        raise MemoryStateError("Payload matrix must be finite float32[S,17,128,128]")
    if type(source_index) is not int or not 0 <= source_index < matrix.shape[0]:
        raise MemoryStateError("Invalid payload source index")
    if (not isinstance(beta, (int, float)) or isinstance(beta, bool)
            or not math.isfinite(beta) or not 0 < beta <= 1
            or not isinstance(eps, (int, float)) or isinstance(eps, bool)
            or not math.isfinite(eps) or eps <= 0):
        raise MemoryStateError("Require 0 < beta <= 1 and finite eps > 0")
    query = _key(query, matrix)
    if (not isinstance(value, torch.Tensor) or value.shape != (PAYLOAD_HEADS, DIM)
            or value.dtype != torch.float32 or value.device != matrix.device
            or not torch.isfinite(value).all()):
        raise MemoryStateError(f"Payload value must be finite float32[17,128] on {matrix.device}")
    plane = matrix[source_index]
    error = value - plane @ query
    updated = plane + beta * error.unsqueeze(-1) * query / (eps + query @ query)
    if not torch.isfinite(updated).all():
        raise MemoryStateError("Payload write would create nonfinite memory")
    result = matrix.clone()
    result[source_index] = updated
    return result


class ParametricPayloadMemory(AssociativeMemory):
    """Seventeen independent 128-wide value heads, with a shared address key.

    The inherited source registry, basis guards and digest do not depend on Qwen.
    No text, token IDs, per-fact table or encoded-value cache is retained here.
    """

    SCHEMA = "alice.memory_payload/v1"

    def __init__(self, source_ids=DEFAULT_SOURCES, latent_basis_id=PAYLOAD_BASIS_ID,
                 device="cpu") -> None:
        super().__init__(source_ids, latent_basis_id, device)
        self.matrix = nn.Parameter(torch.zeros(len(self.source_ids), PAYLOAD_HEADS, DIM, DIM,
                                               dtype=torch.float32, device=device),
                                   requires_grad=False)

    def read(self, source_id: str, query: torch.Tensor) -> torch.Tensor:
        if (self.matrix.dtype != torch.float32 or self.matrix.requires_grad
                or self.matrix.shape != (len(self.source_ids), PAYLOAD_HEADS, DIM, DIM)
                or not torch.isfinite(self.matrix).all()):
            raise MemoryStateError("Persistent payload must remain finite frozen FP32[S,17,128,128]")
        return self.matrix[self.source_index(source_id)] @ _key(query, self.matrix)

    @torch.no_grad()
    def acquire(self, source_id: str, query: torch.Tensor, value: torch.Tensor,
                beta: float = 1.0) -> dict:
        if self.matrix.requires_grad:
            raise MemoryStateError("Acquisition cannot optimize payload memory by autograd")
        if not isinstance(query, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise MemoryStateError("Acquisition requires tensor query/value")
        before = self.content_digest()
        updated = functional_payload_delta(self.matrix, query.detach(), value.detach(),
                                           self.source_index(source_id), beta)
        delta_norm = float(torch.linalg.vector_norm(updated - self.matrix).item())
        self.matrix.copy_(updated)
        return {"source_id": source_id, "before": before,
                "after": self.content_digest(), "delta_norm": delta_norm}

    def save(self, directory: str | Path) -> dict:
        """Publish an immutable safetensors payload via one atomic manifest switch."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if (self.matrix.dtype != torch.float32 or self.matrix.requires_grad
                or self.matrix.shape != (len(self.source_ids), PAYLOAD_HEADS, DIM, DIM)
                or not torch.isfinite(self.matrix).all()):
            raise MemoryStateError("Cannot save invalid payload memory")
        manifest_path = root / "memory.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            previous = type(self).load(root, latent_basis_id=self.latent_basis_id)
            if previous.source_ids != self.source_ids:
                raise MemoryStateError("Refusing to overwrite a different source registry")
        snapshot = uuid4().hex
        filename = f"memory-payload-{snapshot}.safetensors"
        payload, temporary, committed = root / filename, None, False
        try:
            save_file({"matrix": self.matrix.detach().cpu().contiguous()}, str(payload))
            with payload.open("rb") as handle:
                os.fsync(handle.fileno())
            manifest = {"schema": self.SCHEMA, "update_rule": "parallel-normalized-delta/v1",
                        "latent_basis_id": self.latent_basis_id,
                        "source_ids": list(self.source_ids), "shape": list(self.matrix.shape),
                        "dtype": "float32", "snapshot_id": snapshot, "tensor_file": filename,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            manifest["metadata_sha256"] = _metadata_digest(manifest)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root,
                                             prefix=".memory-payload-", suffix=".json", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            committed = True
            return manifest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not committed:
                payload.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, latent_basis_id=PAYLOAD_BASIS_ID,
             device="cpu") -> ParametricPayloadMemory:
        root = Path(directory).resolve()
        try:
            metadata_path = root / "memory.json"
            if metadata_path.stat().st_size > 32768:
                raise MemoryStateError("Oversized payload metadata")
            manifest = json.loads(metadata_path.read_text())
            fields = {"schema", "update_rule", "latent_basis_id", "source_ids", "shape",
                      "dtype", "snapshot_id", "tensor_file", "sha256", "metadata_sha256"}
            if (not isinstance(manifest, dict) or set(manifest) != fields
                    or manifest["schema"] != cls.SCHEMA
                    or manifest["update_rule"] != "parallel-normalized-delta/v1"
                    or manifest["dtype"] != "float32"
                    or not isinstance(manifest["source_ids"], list)
                    or not isinstance(manifest["shape"], list)
                    or manifest["latent_basis_id"] != latent_basis_id
                    or not re.fullmatch(r"[0-9a-f]{32}", manifest["snapshot_id"])):
                raise MemoryStateError("Incompatible payload schema/basis")
            if _metadata_digest(manifest) != manifest["metadata_sha256"]:
                raise MemoryStateError("Payload metadata integrity mismatch")
            filename = f"memory-payload-{manifest['snapshot_id']}.safetensors"
            if manifest["tensor_file"] != filename:
                raise MemoryStateError("Invalid payload reference")
            payload = root / filename
            if (payload.is_symlink() or not payload.is_file()
                    or payload.stat().st_size > 64 * PAYLOAD_HEADS * DIM * DIM * 4 + 65536):
                raise MemoryStateError("Payload must be a regular bounded local file")
            if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["sha256"]:
                raise MemoryStateError("Payload integrity mismatch")
            memory = cls(manifest["source_ids"], latent_basis_id, device)
            tensors = load_file(str(payload), device="cpu")
            value = tensors.get("matrix")
            if (set(tensors) != {"matrix"} or not isinstance(value, torch.Tensor)
                    or list(value.shape) != manifest["shape"]
                    or value.shape != memory.matrix.shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise MemoryStateError("Invalid payload memory tensor")
            with torch.no_grad():
                memory.matrix.copy_(value.to(device))
            return memory
        except MemoryStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
            raise MemoryStateError(f"Cannot restore payload memory: {exc}") from exc
