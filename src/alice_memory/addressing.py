"""Fixed neural addresses from answer-blind frozen-backbone representations.

The random Fourier map supplies a distance-sensitive key instead of learning a
shared direction for every question. Its frequencies and bandwidth belong to
the Bridge state and must be fixed before acquiring facts. It stores no cues,
values, per-fact vectors, or lookup table. Exact-cue addressing is the prototype
scope; semantic or paraphrase equivalence is not promised by this map.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FixedFourierKeys(nn.Module):
    """Normalize h, then map 64 fixed Gaussian phases to a unit 128-vector.

    ``bandwidth`` is a distance in the normalized input space. It is an explicit
    configuration choice, not a statistic updated from acquired experience.
    A private CPU generator makes initialization independent of global RNG state,
    the language model, and subsequent Bridge training.
    """

    NUM_FREQUENCIES = 64
    SEED = 1729

    def __init__(self, hidden_size: int = 2048, *, bandwidth: float) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if (isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float))
                or not math.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError("bandwidth must be finite and positive")
        self.hidden_size = hidden_size
        generator = torch.Generator(device="cpu").manual_seed(self.SEED)
        self.register_buffer("frequencies", torch.randn(
            self.NUM_FREQUENCIES, hidden_size, generator=generator, dtype=torch.float32))
        self.register_buffer("bandwidth", torch.tensor(bandwidth, dtype=torch.float32))
        self._validate_buffers()

    def _validate_buffers(self) -> None:
        if (self.frequencies.shape != (self.NUM_FREQUENCIES, self.hidden_size)
                or self.bandwidth.shape != torch.Size([])):
            raise ValueError("Invalid Fourier frequency/bandwidth shape")
        if (self.frequencies.dtype != torch.float32 or self.bandwidth.dtype != torch.float32
                or self.frequencies.device != self.bandwidth.device):
            raise ValueError("Fourier buffers must be FP32 on the same device")
        if self.frequencies.requires_grad or self.bandwidth.requires_grad:
            raise ValueError("Fourier buffers must remain fixed")
        if (not torch.isfinite(self.frequencies).all() or not torch.isfinite(self.bandwidth)
                or self.bandwidth.item() <= 0):
            raise ValueError("Fourier buffers must be finite with positive bandwidth")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._validate_buffers()
        if (not isinstance(features, torch.Tensor) or features.ndim < 1
                or features.shape[-1] != self.hidden_size):
            raise ValueError(f"Key features must have last dimension {self.hidden_size}")
        if not features.is_floating_point() or features.device != self.frequencies.device:
            raise ValueError("Key features must be floating point on the Fourier device")
        value = features.float()
        if not torch.isfinite(value).all():
            raise ValueError("Key features must remain finite in FP32")
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all() or (norm <= 1e-12).any():
            raise ValueError("Key features must have finite nonzero norm")
        phases = (value / norm) @ self.frequencies.T / self.bandwidth
        if not torch.isfinite(phases).all():
            raise ValueError("Fourier phases are nonfinite; check bandwidth")
        return torch.cat((phases.cos(), phases.sin()), dim=-1) / math.sqrt(self.NUM_FREQUENCIES)
