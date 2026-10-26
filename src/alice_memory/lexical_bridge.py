"""Continuous lexical embedding interface with fixed, independent addresses.

The decoder never chooses a token or constructs text. Only Wv/Wo train; the
Fourier key map and acceptance rule are fixed before experience acquisition.
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
from torch.nn import functional as F

from .addressing import FixedFourierKeys
from .bridge import BridgeStateError
from .query_normalization import VERSION as NORMALIZER_VERSION


class LexicalEmbeddingBridge(nn.Module):
    SCHEMA = "alice.lexical_embedding_bridge/v1"
    value_input = "token_embedding_mean_v1"
    value_encoder = "layernorm_unit_v1"
    key_input = "framed_last_hidden_v1"
    input_layout = "known_answer_dense_v1"

    def __init__(self, hidden_size: int = 2048, latent_dim: int = 128,
                 slots: int = 1, basis_id: str | None = None,
                 bindings: dict[str, str] | None = None, *, bandwidth: float = 0.003,
                 acceptance_threshold: float = 0.6,
                 input_layout: str = "known_answer_dense_v1",
                 key_input: str = "framed_last_hidden_v1",
                 value_encoder: str = "layernorm_unit_v1") -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise BridgeStateError("hidden_size must be a positive integer")
        if type(latent_dim) is not int or latent_dim != 128 or type(slots) is not int or slots != 1:
            raise BridgeStateError("Lexical Bridge requires latent_dim=128 and slots=1")
        self.hidden_size, self.latent_dim, self.num_slots = hidden_size, latent_dim, slots
        self.basis_id = "alice.memory-v0/128/lexical-v1" if basis_id is None else basis_id
        self.bindings = dict(bindings) if isinstance(bindings, dict) else bindings
        if self.bindings is None:
            self.bindings = {}
        self.acceptance_threshold = acceptance_threshold
        self.input_layout = input_layout
        self.key_input = key_input
        self.value_encoder = value_encoder
        self.keys = FixedFourierKeys(hidden_size, bandwidth=bandwidth)
        self.Wv = nn.Linear(hidden_size, latent_dim, bias=False, dtype=torch.float32)
        self.Wo = nn.Linear(latent_dim, hidden_size, bias=False, dtype=torch.float32)
        # Legacy checkpoints retain their exact state/config/digest. Only the
        # affine codec owns a centering buffer and a constant confidence lane.
        if self.value_encoder == "affine_confidence_v1":
            self.register_buffer("value_mean", torch.zeros(hidden_size, dtype=torch.float32))
        self._frozen_digest: str | None = None
        self._frozen_basis: str | None = None
        self._validate_state()

    def _config(self) -> dict:
        # A coordinate ID can be derived from this digest, so it is not itself
        # a digest input. assert_frozen separately guards its identity.
        return {
            "schema": self.SCHEMA, "hidden_size": self.hidden_size,
            "latent_dim": self.latent_dim, "slots": self.num_slots,
            "bindings": self.bindings, "value_input": self.value_input,
            "value_encoder": self.value_encoder, "key_input": self.key_input,
            "input_layout": self.input_layout,
            "layer_norm_eps": 1e-5 if self.value_encoder == "layernorm_unit_v1" else None,
            "acceptance_threshold": self.acceptance_threshold,
            "bandwidth": float(self.keys.bandwidth.detach().cpu()),
        }

    def _validate_state(self) -> None:
        if not isinstance(self.basis_id, str) or not 1 <= len(self.basis_id) <= 128:
            raise BridgeStateError("A stable latent basis ID is required")
        if not isinstance(self.bindings, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in self.bindings.items()):
            raise BridgeStateError("Backbone/tokenizer bindings must be string pairs")
        threshold = self.acceptance_threshold
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold <= 0):
            raise BridgeStateError("Acceptance threshold must be finite and positive")
        if (self.value_input != "token_embedding_mean_v1"
                or self.value_encoder not in ("layernorm_unit_v1", "affine_confidence_v1")
                or self.key_input not in {"framed_last_hidden_v1", NORMALIZER_VERSION}
                or self.input_layout not in {"known_answer_dense_v1", "repeat_qa_dense_v1", "repeat_qa_turn_stop_v1"}):
            raise BridgeStateError("Unsupported lexical Bridge semantics")
        if self.value_encoder == "affine_confidence_v1" and (
                "value_mean" not in self._buffers
                or not isinstance(self.value_mean, torch.Tensor)
                or self.value_mean.shape != (self.hidden_size,)):
            raise BridgeStateError("Affine value mean must have the hidden dimension")
        try:
            self.keys._validate_buffers()
        except ValueError as exc:
            raise BridgeStateError(str(exc)) from exc
        device = self.Wv.weight.device
        if any(t.dtype != torch.float32 or t.device != device or not torch.isfinite(t).all()
               for t in self.state_dict().values()):
            raise BridgeStateError("Lexical Bridge state must be finite FP32 on one device")

    def _input(self, value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        self._validate_state()
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width:
            raise BridgeStateError(f"{name} must have last dimension {width}")
        if not value.is_floating_point() or value.device != self.Wv.weight.device:
            raise BridgeStateError(f"{name} must be floating point on the Bridge device")
        value = value.float()
        if not torch.isfinite(value).all():
            raise BridgeStateError(f"{name} must be finite in FP32")
        return value

    def key(self, features: torch.Tensor) -> torch.Tensor:
        try:
            return self.keys(self._input(features, self.hidden_size, "key/query features"))
        except ValueError as exc:
            raise BridgeStateError(str(exc)) from exc

    def value(self, features: torch.Tensor) -> torch.Tensor:
        features = self._input(features, self.hidden_size, "value features")
        if self.value_encoder == "affine_confidence_v1":
            projected = self.Wv(features - self.value_mean)
            # No per-value normalization: content amplitudes survive acquisition.
            # The fixed first coordinate carries read confidence, not content.
            return torch.cat((torch.ones_like(projected[..., :1]), projected[..., 1:]), dim=-1)
        return F.normalize(self.Wv(F.layer_norm(features, (self.hidden_size,), eps=1e-5)),
                           p=2, dim=-1, eps=1e-6)

    def slots(self, query: torch.Tensor, readout: torch.Tensor,
              embedding_rms: float | torch.Tensor) -> torch.Tensor:
        q = self._input(query, self.latent_dim, "query")
        r = self._input(readout, self.latent_dim, "memory readout")
        if q.shape != r.shape:
            raise BridgeStateError("Query and readout shapes must match")
        rms = torch.as_tensor(embedding_rms, device=r.device, dtype=torch.float32)
        if rms.numel() != 1 or not torch.isfinite(rms).all() or rms.item() <= 0:
            raise BridgeStateError("Reference embedding RMS must be finite and positive")
        # Intentionally preserve the decoded embedding's magnitude: no tanh,
        # readout normalization, vocabulary lookup or per-fact routing here.
        return self.Wo(r).unsqueeze(-2).contiguous()

    def accepts(self, readout: torch.Tensor) -> bool:
        """Scalar inference confidence, fixed before acquisition; not a learner."""
        self._validate_state()
        if (not isinstance(readout, torch.Tensor) or readout.shape != (self.latent_dim,)
                or not readout.is_floating_point() or readout.device != self.Wv.weight.device):
            raise BridgeStateError("Acceptance expects one latent vector on the Bridge device")
        value = readout.float()
        if self.value_encoder == "affine_confidence_v1":
            return bool(torch.isfinite(value).all()
                        and value[0] >= self.acceptance_threshold)
        norm = torch.linalg.vector_norm(value)
        return bool(torch.isfinite(value).all() and torch.isfinite(norm)
                    and norm >= self.acceptance_threshold)

    def parameter_digest(self) -> str:
        self._validate_state()
        digest = hashlib.sha256(json.dumps(self._config(), sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode())
        for name, tensor in sorted(self.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def freeze(self) -> LexicalEmbeddingBridge:
        self.eval().requires_grad_(False)
        self._frozen_digest = self.parameter_digest()
        self._frozen_basis = self.basis_id
        return self

    def assert_frozen(self) -> None:
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise BridgeStateError("Acquisition requires an eval/frozen lexical Bridge")
        if (self._frozen_digest is None or self.parameter_digest() != self._frozen_digest
                or self.basis_id != self._frozen_basis):
            raise BridgeStateError("Frozen lexical Bridge state or semantics changed")

    def save(self, directory: str | Path) -> None:
        self._validate_state()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        filename = f"lexical-bridge-{uuid4().hex}.safetensors"
        tensor_path, temporary = root / filename, root / f".lexical-{uuid4().hex}.tmp"
        published = False
        try:
            state = {k: v.detach().cpu().contiguous().clone() for k, v in self.state_dict().items()}
            save_file(state, str(tensor_path))
            with tensor_path.open("rb") as handle:
                os.fsync(handle.fileno())
            manifest = {**self._config(), "latent_basis_id": self.basis_id,
                        "tensor_file": filename, "sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
                        "state_digest": self.parameter_digest()}
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / "bridge.json")
            published = True
        except Exception:
            if not published:
                tensor_path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, *, latent_basis_id: str | None = None,
             device: str | torch.device = "cpu", expected_bindings: dict[str, str] | None = None,
             freeze: bool = True) -> LexicalEmbeddingBridge:
        root = Path(directory).resolve()
        try:
            manifest = json.loads((root / "bridge.json").read_text())
            required = {"schema", "hidden_size", "latent_dim", "slots", "bindings", "value_input",
                        "value_encoder", "key_input", "input_layout", "layer_norm_eps", "bandwidth",
                        "acceptance_threshold", "latent_basis_id", "tensor_file", "sha256", "state_digest"}
            if not isinstance(manifest, dict) or set(manifest) != required or manifest["schema"] != cls.SCHEMA:
                raise BridgeStateError("Invalid lexical Bridge checkpoint manifest")
            if latent_basis_id is not None and manifest["latent_basis_id"] != latent_basis_id:
                raise BridgeStateError("Incompatible latent basis")
            if expected_bindings is not None and manifest["bindings"] != expected_bindings:
                raise BridgeStateError("Backbone/tokenizer binding mismatch")
            filename = manifest["tensor_file"]
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or Path(filename).suffix != ".safetensors"):
                raise BridgeStateError("Invalid lexical Bridge tensor filename")
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise BridgeStateError("Missing or outside-checkpoint lexical Bridge tensors")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
                raise BridgeStateError("Lexical Bridge tensor digest mismatch")
            model = cls(manifest["hidden_size"], manifest["latent_dim"], manifest["slots"],
                        manifest["latent_basis_id"], manifest["bindings"],
                        bandwidth=manifest["bandwidth"], acceptance_threshold=manifest["acceptance_threshold"],
                        input_layout=manifest["input_layout"], key_input=manifest["key_input"],
                        value_encoder=manifest["value_encoder"])
            state = load_file(str(path), device="cpu")
            if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
                raise BridgeStateError("Lexical Bridge tensors must be finite FP32")
            model.load_state_dict(state, strict=True)
            expected_config = model._config()
            if any(manifest[k] != v for k, v in expected_config.items()):
                raise BridgeStateError("Lexical Bridge semantics/configuration mismatch")
            if model.parameter_digest() != manifest["state_digest"]:
                raise BridgeStateError("Lexical Bridge state digest mismatch")
            model.to(device=device)
            return model.freeze() if freeze else model
        except BridgeStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
            raise BridgeStateError(f"Cannot restore lexical Bridge: {exc}") from exc
