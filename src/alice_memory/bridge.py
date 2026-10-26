"""Small trainable neural interface; it stores no acquired fact or memory matrix."""

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
from torch.nn import functional as F


class BridgeStateError(ValueError):
    """Invalid tensor boundary or incompatible/faulty independent Bridge state."""


class ParametricBridge(nn.Module):
    """Shared key/query encoder, value encoder and four neural embedding slots.

    Input features must already come from an answer-blind or value-only frozen
    backbone pass as appropriate. This module never tokenizes or decodes text.
    Training uses the ordinary autograd graph; acquisition calls ``freeze`` and
    uses detached features while only the separate memory matrix is updated.
    """

    SCHEMA = "alice.memory_bridge/v0"
    LEGACY_VALUE_ENCODER = "legacy_tanh_v0"
    NORMALIZED_VALUE_ENCODER = "layernorm_radial_v1"

    def __init__(self, hidden_size: int = 2048, latent_dim: int = 128,
                 slots: int = 4, basis_id: str | None = None,
                 bindings: dict[str, str] | None = None,
                 value_encoder: str = NORMALIZED_VALUE_ENCODER) -> None:
        super().__init__()
        for name, size in (("hidden_size", hidden_size), ("latent_dim", latent_dim), ("slots", slots)):
            if type(size) is not int or size < 1:
                raise BridgeStateError(f"{name} must be a positive integer")
        if bindings is not None and (not isinstance(bindings, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in bindings.items())):
            raise BridgeStateError("Backbone/tokenizer bindings must be string pairs")
        if (not isinstance(value_encoder, str)
                or value_encoder not in {self.LEGACY_VALUE_ENCODER, self.NORMALIZED_VALUE_ENCODER}):
            raise BridgeStateError("Unknown value encoder mode")
        if basis_id is None:
            basis_id = ("alice.memory-v0/128/v1" if value_encoder == self.LEGACY_VALUE_ENCODER
                        else "alice.memory-v0/128/radial-v1")
        if not isinstance(basis_id, str) or not basis_id:
            raise BridgeStateError("A stable latent basis ID is required")
        self.hidden_size, self.latent_dim, self.num_slots = hidden_size, latent_dim, slots
        self.basis_id, self.bindings = basis_id, dict(bindings or {})
        self.value_encoder = value_encoder
        self.Wk = nn.Linear(hidden_size, latent_dim, bias=False, dtype=torch.float32)
        self.Wv = nn.Linear(hidden_size, latent_dim, bias=False, dtype=torch.float32)
        self.Wo = nn.Linear(latent_dim, slots * hidden_size, bias=False, dtype=torch.float32)
        self.gate = nn.Linear(2 * latent_dim + 1, 1, dtype=torch.float32)
        nn.init.normal_(self.Wo.weight, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, -2.0)
        self._frozen_digest: str | None = None

    def _input(self, value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width:
            raise BridgeStateError(f"{name} must have last dimension {width}")
        if not value.is_floating_point() or value.device != self.Wk.weight.device:
            raise BridgeStateError(f"{name} must be floating point on the Bridge device")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise BridgeStateError("Bridge parameters must remain FP32")
        if not torch.isfinite(value).all():
            raise BridgeStateError(f"{name} contains nonfinite values")
        return value.float()

    def key(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.Wk(self._input(features, self.hidden_size, "key/query features"))
        return F.normalize(projected, p=2, dim=-1, eps=1e-6)

    def value_preactivation(self, features: torch.Tensor) -> torch.Tensor:
        value = self._input(features, self.hidden_size, "value features")
        if self.value_encoder == self.NORMALIZED_VALUE_ENCODER:
            value = F.layer_norm(value, (self.hidden_size,), eps=1e-5)
        elif self.value_encoder != self.LEGACY_VALUE_ENCODER:
            raise BridgeStateError("Unknown value encoder mode")
        return self.Wv(value)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.value_preactivation(features)
        if self.value_encoder == self.LEGACY_VALUE_ENCODER:
            return projected.tanh() / math.sqrt(self.latent_dim)
        # Smooth radial bound preserves directions and a finite Jacobian at zero;
        # unlike elementwise tanh it cannot flatten every large coordinate to ±1.
        return projected / (self.latent_dim + projected.square().sum(-1, keepdim=True)).sqrt()

    def slots(self, query: torch.Tensor, readout: torch.Tensor,
              embedding_rms: float | torch.Tensor) -> torch.Tensor:
        q = self._input(query, self.latent_dim, "query")
        r = self._input(readout, self.latent_dim, "memory readout")
        if q.shape != r.shape:
            raise BridgeStateError("Query and readout shapes must match")
        rms = torch.as_tensor(embedding_rms, device=r.device, dtype=torch.float32)
        if rms.numel() != 1 or not torch.isfinite(rms).all() or rms.item() <= 0:
            raise BridgeStateError("Reference embedding RMS must be a finite positive scalar")
        norm = r.norm(p=2, dim=-1, keepdim=True)
        gate = self.gate(torch.cat((q, r, norm), dim=-1)).sigmoid().unsqueeze(-1)
        output = self.Wo(r).tanh().reshape(*r.shape[:-1], self.num_slots, self.hidden_size)
        # Bias-free Wo ensures exact zero slots for empty memory even with a gate bias.
        return (2.0 * rms * gate * output).contiguous()

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        if self.value_encoder != self.LEGACY_VALUE_ENCODER:
            digest.update(("value_encoder=" + self.value_encoder + "\n").encode())
        for name, parameter in sorted(self.named_parameters()):
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def freeze(self) -> ParametricBridge:
        self.eval().requires_grad_(False)
        self._frozen_digest = self.parameter_digest()
        return self

    def assert_frozen(self) -> None:
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise BridgeStateError("Acquisition requires an eval/frozen Bridge")
        if self._frozen_digest is None or self.parameter_digest() != self._frozen_digest:
            raise BridgeStateError("Frozen Bridge parameters changed")

    def save(self, directory: str | Path) -> None:
        """Publish a standalone immutable tensor file through one atomic manifest."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        name = f"bridge-{uuid4().hex}.safetensors"
        tensor_path, temporary = root / name, root / f".bridge-{uuid4().hex}.tmp"
        try:
            state = {k: v.detach().cpu().contiguous().clone() for k, v in self.state_dict().items()}
            if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
                raise BridgeStateError("Cannot save a nonfinite/non-FP32 Bridge")
            save_file(state, str(tensor_path))
            with tensor_path.open("rb") as handle:
                os.fsync(handle.fileno())
            manifest = {
                "schema": self.SCHEMA, "latent_basis_id": self.basis_id,
                "hidden_size": self.hidden_size, "latent_dim": self.latent_dim,
                "slots": self.num_slots, "bindings": self.bindings,
                "value_encoder": self.value_encoder,
                "tensor_file": name,
                "sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
            }
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "bridge.json")
        except Exception:
            tensor_path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, *, latent_basis_id: str | None = None,
             device: str | torch.device = "cpu", expected_bindings: dict[str, str] | None = None,
             freeze: bool = True) -> ParametricBridge:
        root = Path(directory).resolve()
        try:
            manifest = json.loads((root / "bridge.json").read_text())
            required = {"schema", "latent_basis_id", "hidden_size", "latent_dim", "slots",
                        "bindings", "tensor_file", "sha256"}
            if (not isinstance(manifest, dict)
                    or set(manifest) not in (required, required | {"value_encoder"})
                    or manifest["schema"] != cls.SCHEMA):
                raise BridgeStateError("Invalid Bridge checkpoint manifest")
            if latent_basis_id is not None and manifest["latent_basis_id"] != latent_basis_id:
                raise BridgeStateError("Incompatible latent basis")
            if expected_bindings is not None and manifest["bindings"] != expected_bindings:
                raise BridgeStateError("Backbone/tokenizer binding mismatch")
            filename = manifest["tensor_file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise BridgeStateError("Invalid Bridge tensor filename")
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise BridgeStateError("Missing or outside-checkpoint Bridge tensors")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
                raise BridgeStateError("Bridge tensor digest mismatch")
            model = cls(manifest["hidden_size"], manifest["latent_dim"], manifest["slots"],
                        manifest["latent_basis_id"], manifest["bindings"],
                        manifest.get("value_encoder", cls.LEGACY_VALUE_ENCODER))
            state = load_file(str(path), device="cpu")
            if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
                raise BridgeStateError("Bridge tensors must be finite FP32")
            model.load_state_dict(state, strict=True)
            model.to(device=device)
            return model.freeze() if freeze else model
        except BridgeStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
            raise BridgeStateError(f"Cannot restore Bridge: {exc}") from exc
