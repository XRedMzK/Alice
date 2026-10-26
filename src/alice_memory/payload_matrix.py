"""Versioned parallel value payload in independently mutable neural parameters."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from torch import nn

from .matrix import AssociativeMemory, DEFAULT_SOURCES, DIM, MemoryStateError, _key

PAYLOAD_HEADS = 17
PAYLOAD_BASIS_ID = "alice.memory_payload/17x128/v1"
_PAYLOAD_NAME = re.compile(r"memory-payload-([0-9a-f]{32})\.safetensors")
_STAGING_NAME = re.compile(r"\.memory-payload-([0-9a-f]{32})\.(safetensors|json|prepare)\.tmp")
_PREPARE_NAME = re.compile(r"\.memory-payload-([0-9a-f]{32})\.prepare")


def _prepare_bytes(snapshot: str) -> bytes:
    return f"alice.memory_payload/initial-publication/v1\n{snapshot}\n".encode("ascii")


class MemoryCommitUncertainError(MemoryStateError):
    """Publication was attempted, but its durable completion was not confirmed.

    The caller must not acknowledge a save or roll back to an older RAM state.
    Reopen the checkpoint to discover which complete state is committed.
    """


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        """Durably publish M with one atomic manifest switch as the commit point.

        Before that switch, all generated files are disposable and the previous
        checkpoint is untouched. After a switch attempt, retain every payload:
        an error may mean that the complete new checkpoint is already visible.
        Only a successful final directory fsync permits a save acknowledgement.
        The caller owns serialization of writers (the runtime holds a flock).
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if (self.matrix.dtype != torch.float32 or self.matrix.requires_grad
                or self.matrix.shape != (len(self.source_ids), PAYLOAD_HEADS, DIM, DIM)
                or not torch.isfinite(self.matrix).all()):
            raise MemoryStateError("Cannot save invalid payload memory")
        manifest_path = root / "memory.json"
        initial = not (manifest_path.exists() or manifest_path.is_symlink())
        if not initial:
            previous = type(self).load(root, latent_basis_id=self.latent_basis_id)
            if previous.source_ids != self.source_ids:
                raise MemoryStateError("Refusing to overwrite a different source registry")
        elif any(path.name != ".alice-memory.lock" for path in root.iterdir()):
            raise MemoryStateError("Nonempty directory has no memory manifest; refusing to reset it")
        snapshot = uuid4().hex
        filename = f"memory-payload-{snapshot}.safetensors"
        payload = root / filename
        staging = root / f".memory-payload-{snapshot}.safetensors.tmp"
        temporary = root / f".memory-payload-{snapshot}.json.tmp"
        prepare = root / f".memory-payload-{snapshot}.prepare"
        prepare_staging = root / f".memory-payload-{snapshot}.prepare.tmp"
        switch_attempted = False
        created = []
        try:
            # Also persist newly created ancestors, even if mkdir was interrupted
            # during an earlier process. The checkpoint directory is synced below.
            for ancestor in reversed(root.absolute().parents):
                _sync_directory(ancestor)
            if initial:
                # A UUID-only prepare marker distinguishes an initial unfinished
                # save from a committed checkpoint whose manifest was lost.
                descriptor = os.open(prepare_staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                created.append(prepare_staging)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(_prepare_bytes(snapshot))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(prepare_staging, prepare)
                created.append(prepare)
                _sync_directory(root)
            descriptor = os.open(staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(descriptor)
            created.append(staging)
            save_file({"matrix": self.matrix.detach().cpu().contiguous()}, str(staging))
            with staging.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(staging, payload)
            created.append(payload)
            # The new manifest must never become durable before its payload name.
            _sync_directory(root)
            manifest = {"schema": self.SCHEMA, "update_rule": "parallel-normalized-delta/v1",
                        "latent_basis_id": self.latent_basis_id,
                        "source_ids": list(self.source_ids), "shape": list(self.matrix.shape),
                        "dtype": "float32", "snapshot_id": snapshot, "tensor_file": filename,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            manifest["metadata_sha256"] = _metadata_digest(manifest)
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            created.append(temporary)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            switch_attempted = True
            os.replace(temporary, manifest_path)
            _sync_directory(root)
            if initial:
                prepare.unlink()
                _sync_directory(root)
            return manifest
        except BaseException as exc:
            if switch_attempted:
                raise MemoryCommitUncertainError(
                    "Memory publication may have committed; reopen the checkpoint before retrying"
                ) from exc
            raise
        finally:
            if not switch_attempted:
                # These names were exclusively created by this save attempt.
                for path in reversed(created):
                    if path != prepare:
                        path.unlink(missing_ok=True)
                if prepare in created:
                    # Keep durable ownership proof until payload retirement is
                    # durable too, so recovery cannot mistake it for lost state.
                    _sync_directory(root)
                    prepare.unlink(missing_ok=True)
                    _sync_directory(root)

    @classmethod
    def is_uncommitted_name(cls, name: str) -> bool:
        """Names reserved solely for uncommitted payload publication artifacts."""
        return bool(_PAYLOAD_NAME.fullmatch(name) or _STAGING_NAME.fullmatch(name)
                    or _PREPARE_NAME.fullmatch(name))

    @classmethod
    def recover_uncommitted(cls, directory: str | Path, latent_basis_id=PAYLOAD_BASIS_ID) -> None:
        """Retire known orphans under the caller's exclusive directory lock.

        A manifest, when present, must fully validate before any cleanup. Without
        one, finalized payloads require a matching durable initial prepare marker;
        a missing committed manifest must never silently reset acquired M.
        Finalized orphan tensors must be valid bounded payloads. Staging files
        are never committed and may contain an interrupted partial write.
        All validation precedes deletion, and the live snapshot is never removed.
        """
        root = Path(directory)
        entries = tuple(root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise MemoryStateError("Memory directory must contain regular local files only")
        manifest_path = root / "memory.json"
        referenced = None
        if manifest_path.exists():
            cls.load(root, latent_basis_id=latent_basis_id)
            referenced = json.loads(manifest_path.read_text())["tensor_file"]
        elif any(path.name != ".alice-memory.lock" and not cls.is_uncommitted_name(path.name)
                 for path in entries):
            raise MemoryStateError("Nonempty directory has no memory manifest; refusing to reset it")
        prepared = set()
        for path in entries:
            match = _PREPARE_NAME.fullmatch(path.name)
            if match:
                if path.stat().st_size > 256 or path.read_bytes() != _prepare_bytes(match[1]):
                    raise MemoryStateError("Invalid initial publication prepare marker")
                prepared.add(match[1])
        if referenced is None and len(prepared) > 1:
            raise MemoryStateError("Ambiguous initial publication; refusing to reset it")
        retired = []
        for path in entries:
            staging_match = _STAGING_NAME.fullmatch(path.name)
            payload_match = _PAYLOAD_NAME.fullmatch(path.name)
            if _PREPARE_NAME.fullmatch(path.name):
                continue  # Remove ownership proof last, after every payload.
            if staging_match:
                if (referenced is None and staging_match[2] != "prepare"
                        and staging_match[1] not in prepared):
                    raise MemoryStateError("Unowned staging payload; refusing to reset it")
                retired.append(path)
            elif payload_match and path.name != referenced:
                if referenced is None and payload_match[1] not in prepared:
                    raise MemoryStateError("Missing committed manifest; refusing to reset it")
                try:
                    if path.stat().st_size > 64 * PAYLOAD_HEADS * DIM * DIM * 4 + 65536:
                        raise MemoryStateError("Oversized orphan payload")
                    tensors = load_file(str(path), device="cpu")
                    value = tensors.get("matrix")
                    if (set(tensors) != {"matrix"} or not isinstance(value, torch.Tensor)
                            or value.ndim != 4 or not 1 <= value.shape[0] <= 64
                            or value.shape[1:] != (PAYLOAD_HEADS, DIM, DIM)
                            or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                        raise MemoryStateError("Invalid orphan payload tensor")
                except (OSError, RuntimeError, SafetensorError) as exc:
                    raise MemoryStateError(f"Cannot validate orphan payload: {exc}") from exc
                retired.append(path)
        markers = [path for path in entries if _PREPARE_NAME.fullmatch(path.name)]
        if retired or markers:
            # A process may have died just after manifest replacement, before
            # its directory fsync. Make the visible commit durable before an
            # older payload can be retired during recovery.
            _sync_directory(root)
        for path in retired:
            path.unlink()
        if markers:
            _sync_directory(root)
            for path in markers:
                path.unlink()
        if retired or markers:
            _sync_directory(root)

    @classmethod
    def load(cls, directory: str | Path, latent_basis_id=PAYLOAD_BASIS_ID,
             device="cpu") -> ParametricPayloadMemory:
        root = Path(directory).resolve()
        try:
            metadata_path = root / "memory.json"
            if (metadata_path.is_symlink() or not metadata_path.is_file()
                    or metadata_path.stat().st_size > 32768):
                raise MemoryStateError("Payload metadata must be a regular bounded local file")
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
