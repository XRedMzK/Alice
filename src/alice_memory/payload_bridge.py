"""Full-rank neural value payload, with independent 128-dimensional addresses.

Sixteen rotated blocks transport a 2048-vector continuously. An additional head
contains a fixed confidence coordinate. No token IDs, vocabulary search or text
decoder is used, and the deterministic rotation is independent of all values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from torch import nn

from .addressing import FixedFourierKeys
from .bridge import BridgeStateError
from .query_normalization import VERSION as NORMALIZER_VERSION


class PayloadEmbeddingBridge(nn.Module):
    SCHEMA = "alice.payload_embedding_bridge/v1"
    ROTATION_SEED = 314159
    value_input = "token_embedding_mean_v1"
    value_encoder = "orthogonal_block_payload_v1"
    key_input = NORMALIZER_VERSION
    input_layout = "repeat_qa_turn_stop_v1"
    acceptance_threshold = 0.6
    payload_shape = (17, 128)

    def __init__(self, hidden_size: int = 2048, key_dim: int = 128,
                 slots: int = 1, basis_id: str | None = None,
                 bindings: dict[str, str] | None = None, *, bandwidth: float = 0.003,
                 value_scale: float = 1.0) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size != 2048:
            raise BridgeStateError("Payload v1 requires hidden_size=2048")
        if type(key_dim) is not int or key_dim != 128 or type(slots) is not int or slots != 1:
            raise BridgeStateError("Payload v1 requires key_dim=128 and slots=1")
        if (isinstance(value_scale, bool) or not isinstance(value_scale, (int, float))
                or not math.isfinite(value_scale) or value_scale <= 0):
            raise BridgeStateError("Value scale must be finite and positive")
        self.hidden_size, self.latent_dim, self.num_slots = hidden_size, key_dim, slots
        self.basis_id = "alice.memory-payload-v1/17x128" if basis_id is None else basis_id
        self.bindings = {} if bindings is None else dict(bindings) if isinstance(bindings, dict) else bindings
        self.keys = FixedFourierKeys(hidden_size, bandwidth=bandwidth)
        generator = torch.Generator(device="cpu").manual_seed(self.ROTATION_SEED)
        random = torch.randn(128, 128, generator=generator, dtype=torch.float32)
        rotation, triangular = torch.linalg.qr(random)
        signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
        self.R = nn.Parameter((rotation * signs.unsqueeze(0)).contiguous())
        self.register_buffer("value_scale", torch.tensor(value_scale, dtype=torch.float32))
        self._frozen_digest: str | None = None
        self._frozen_basis: str | None = None
        self._validate_state()

    def _config(self) -> dict:
        return {
            "schema": self.SCHEMA, "hidden_size": self.hidden_size,
            "key_dim": self.latent_dim, "slots": self.num_slots,
            "payload_shape": list(self.payload_shape), "bindings": self.bindings,
            "value_input": self.value_input, "value_encoder": self.value_encoder,
            "key_input": self.key_input, "input_layout": self.input_layout,
            "rotation_seed": self.ROTATION_SEED,
            "rotation_mode": "shared_row_orthogonal_v1",
            "acceptance_threshold": self.acceptance_threshold,
            "bandwidth": float(self.keys.bandwidth.detach().cpu()),
            "value_scale": float(self.value_scale.detach().cpu()),
        }

    def _validate_state(self) -> None:
        if (type(self.hidden_size) is not int or self.hidden_size != 2048
                or type(self.latent_dim) is not int or self.latent_dim != 128
                or type(self.num_slots) is not int or self.num_slots != 1
                or self.payload_shape != (17, 128)):
            raise BridgeStateError("Unsupported payload dimensions")
        if not isinstance(self.basis_id, str) or not 1 <= len(self.basis_id) <= 128:
            raise BridgeStateError("A stable payload basis ID is required")
        if not isinstance(self.bindings, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in self.bindings.items()):
            raise BridgeStateError("Backbone/tokenizer bindings must be string pairs")
        if (self.value_input != "token_embedding_mean_v1"
                or self.value_encoder != "orthogonal_block_payload_v1"
                or self.key_input != NORMALIZER_VERSION
                or self.input_layout != "repeat_qa_turn_stop_v1"
                or self.acceptance_threshold != 0.6 or self.ROTATION_SEED != 314159):
            raise BridgeStateError("Unsupported payload Bridge semantics")
        if self.R.shape != (128, 128) or self.value_scale.shape != torch.Size([]):
            raise BridgeStateError("Invalid rotation/scale shape")
        try:
            self.keys._validate_buffers()
        except ValueError as exc:
            raise BridgeStateError(str(exc)) from exc
        device = self.R.device
        if any(t.dtype != torch.float32 or t.device != device or not torch.isfinite(t).all()
               for t in self.state_dict().values()):
            raise BridgeStateError("Payload Bridge state must be finite FP32 on one device")
        if self.value_scale.requires_grad or self.value_scale.item() <= 0:
            raise BridgeStateError("Value scale must remain a fixed positive buffer")
        with torch.no_grad():
            error = (self.R.T @ self.R - torch.eye(128, device=device)).abs().max()
            if error > 1e-4:
                raise BridgeStateError("Payload rotation is not orthogonal within 1e-4")

    def _input(self, value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        self._validate_state()
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width:
            raise BridgeStateError(f"{name} must have last dimension {width}")
        if not value.is_floating_point() or value.device != self.R.device:
            raise BridgeStateError(f"{name} must be floating point on the Bridge device")
        value = value.float()
        if not torch.isfinite(value).all():
            raise BridgeStateError(f"{name} must be finite in FP32")
        return value

    def key(self, features: torch.Tensor) -> torch.Tensor:
        try:
            return self.keys(self._input(features, 2048, "key/query features"))
        except ValueError as exc:
            raise BridgeStateError(str(exc)) from exc

    def value(self, features: torch.Tensor) -> torch.Tensor:
        features = self._input(features, 2048, "value features")
        blocks = features.reshape(*features.shape[:-1], 16, 128)
        data = (blocks @ self.R) / self.value_scale
        confidence = torch.zeros(*features.shape[:-1], 1, 128,
                                 device=features.device, dtype=torch.float32)
        confidence[..., 0, 0] = 1
        return torch.cat((confidence, data), dim=-2).contiguous()

    def _readout(self, value: torch.Tensor) -> torch.Tensor:
        result = self._input(value, 128, "memory readout")
        if result.ndim < 2 or result.shape[-2:] != (17, 128):
            raise BridgeStateError("Memory readout must end with [17,128]")
        return result

    def slots(self, query: torch.Tensor, readout: torch.Tensor,
              embedding_rms: float | torch.Tensor) -> torch.Tensor:
        q = self._input(query, 128, "query")
        r = self._readout(readout)
        if q.shape != (*r.shape[:-2], 128):
            raise BridgeStateError("Query and payload batch shapes must match")
        rms = torch.as_tensor(embedding_rms, dtype=torch.float32, device=r.device)
        if rms.numel() != 1 or not torch.isfinite(rms).all() or rms.item() <= 0:
            raise BridgeStateError("Reference embedding RMS must be finite and positive")
        # Row-vector inverse: (E_blocks @ R) @ R.T. No normalization,
        # clipping, confidence division, token choice or vocabulary lookup.
        decoded = (r[..., 1:, :] @ self.R.T) * self.value_scale
        return decoded.reshape(*r.shape[:-2], 1, 2048).contiguous()

    def accepts(self, readout: torch.Tensor) -> bool:
        r = self._readout(readout)
        if r.shape != (17, 128):
            raise BridgeStateError("Acceptance expects one [17,128] payload")
        return bool(r[0, 0] >= self.acceptance_threshold)

    @torch.no_grad()
    def project_orthogonal_(self) -> PayloadEmbeddingBridge:
        """Optional retraction after an authorized R step; never used on frozen state."""
        if self._frozen_digest is not None or not self.R.requires_grad:
            raise BridgeStateError("Cannot retract a frozen payload Bridge")
        if self.R.shape != (128, 128) or not torch.isfinite(self.R).all():
            raise BridgeStateError("Cannot retract an invalid rotation")
        rotation, triangular = torch.linalg.qr(self.R)
        signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
        self.R.copy_(rotation * signs.unsqueeze(0))
        return self

    def parameter_digest(self) -> str:
        self._validate_state()
        digest = hashlib.sha256(json.dumps(self._config(), sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode())
        for name, tensor in sorted(self.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def freeze(self) -> PayloadEmbeddingBridge:
        self.eval().requires_grad_(False)
        self._frozen_digest = self.parameter_digest()
        self._frozen_basis = self.basis_id
        return self

    def assert_frozen(self) -> None:
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise BridgeStateError("Acquisition requires an eval/frozen payload Bridge")
        if (self._frozen_digest is None or self.parameter_digest() != self._frozen_digest
                or self.basis_id != self._frozen_basis):
            raise BridgeStateError("Frozen payload Bridge state or semantics changed")

    def save(self, directory: str | Path) -> None:
        self._validate_state()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "bridge.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (not isinstance(previous, dict) or previous.get("schema") != self.SCHEMA
                    or previous.get("latent_basis_id") != self.basis_id):
                raise BridgeStateError("Cannot replace another Bridge schema/basis checkpoint")
        filename = f"payload-bridge-{uuid4().hex}.safetensors"
        path, temporary = root / filename, root / f".payload-{uuid4().hex}.tmp"
        committed = False
        try:
            save_file({k: v.detach().cpu().contiguous().clone() for k, v in self.state_dict().items()}, str(path))
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata = {**self._config(), "latent_basis_id": self.basis_id,
                        "tensor_file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "state_digest": self.parameter_digest()}
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "bridge.json")
            committed = True
        finally:
            temporary.unlink(missing_ok=True)
            if not committed:
                path.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, *, latent_basis_id: str | None = None,
             device: str | torch.device = "cpu", expected_bindings: dict[str, str] | None = None,
             freeze: bool = True) -> PayloadEmbeddingBridge:
        root = Path(directory).resolve()
        try:
            metadata_path = root / "bridge.json"
            if metadata_path.stat().st_size > 32768:
                raise BridgeStateError("Oversized payload Bridge metadata")
            metadata = json.loads(metadata_path.read_text())
            config_fields = {"schema", "hidden_size", "key_dim", "slots", "payload_shape", "bindings",
                             "value_input", "value_encoder", "key_input", "input_layout", "rotation_seed",
                             "rotation_mode", "acceptance_threshold", "bandwidth", "value_scale"}
            if (not isinstance(metadata, dict) or set(metadata) != config_fields | {
                    "latent_basis_id", "tensor_file", "sha256", "state_digest"}
                    or metadata["schema"] != cls.SCHEMA):
                raise BridgeStateError("Invalid payload Bridge checkpoint manifest")
            if (type(metadata["rotation_seed"]) is not int
                    or not isinstance(metadata["payload_shape"], list)
                    or any(type(size) is not int for size in metadata["payload_shape"])):
                raise BridgeStateError("Invalid payload Bridge configuration types")
            if latent_basis_id is not None and metadata["latent_basis_id"] != latent_basis_id:
                raise BridgeStateError("Incompatible payload basis")
            if expected_bindings is not None and metadata["bindings"] != expected_bindings:
                raise BridgeStateError("Backbone/tokenizer binding mismatch")
            filename = metadata["tensor_file"]
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or not filename.startswith("payload-bridge-") or not filename.endswith(".safetensors")):
                raise BridgeStateError("Invalid payload Bridge tensor filename")
            path = (root / filename).resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or path.stat().st_size > 2**20):
                raise BridgeStateError("Missing, oversized or outside-checkpoint Bridge tensors")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise BridgeStateError("Payload Bridge tensor digest mismatch")
            model = cls(metadata["hidden_size"], metadata["key_dim"], metadata["slots"],
                        metadata["latent_basis_id"], metadata["bindings"],
                        bandwidth=metadata["bandwidth"], value_scale=metadata["value_scale"])
            state = load_file(str(path), device="cpu")
            if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
                raise BridgeStateError("Payload Bridge tensors must be finite FP32")
            model.load_state_dict(state, strict=True)
            if any(metadata[k] != value for k, value in model._config().items()):
                raise BridgeStateError("Payload Bridge semantics/configuration mismatch")
            if model.parameter_digest() != metadata["state_digest"]:
                raise BridgeStateError("Payload Bridge state digest mismatch")
            model.to(device=device)
            return model.freeze() if freeze else model
        except BridgeStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
            raise BridgeStateError(f"Cannot restore payload Bridge: {exc}") from exc
