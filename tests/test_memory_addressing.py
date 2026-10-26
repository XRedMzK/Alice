"""Reproducible addresses, numerical boundaries, and independent persistence."""

import pytest
import torch

from alice_memory.addressing import FixedFourierKeys


def test_exact_repeat_unit_norm_and_close_input_distinction():
    generator = torch.Generator().manual_seed(31)
    common = torch.randn(2048, generator=generator)
    common = common / common.norm()
    perturbations = torch.randn(3, 2048, generator=generator)
    perturbations = perturbations / perturbations.norm(dim=-1, keepdim=True)
    features = common + 0.03 * perturbations
    encoder = FixedFourierKeys(bandwidth=0.012)
    keys = encoder(features)
    assert keys.shape == (3, 128) and keys.dtype == torch.float32
    assert torch.equal(keys, encoder(features.clone()))
    assert torch.allclose(keys.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=0)
    assert torch.allclose(keys, encoder(features * 7), atol=2e-5, rtol=0)
    # The source features have a common direction; the configured map must not
    # reproduce that directional collapse for these nearby, distinct inputs.
    normalized = features / features.norm(dim=-1, keepdim=True)
    assert (normalized @ normalized.T)[0, 1] > 0.99
    assert torch.pdist(keys).min() > 0.5
    assert not list(encoder.parameters())


def test_fixed_state_is_global_rng_independent_and_serializes(tmp_path):
    torch.manual_seed(3)
    encoder = FixedFourierKeys(hidden_size=8, bandwidth=0.08)
    torch.manual_seed(400)
    same = FixedFourierKeys(hidden_size=8, bandwidth=0.08)
    assert torch.equal(encoder.frequencies, same.frequencies)
    features = torch.arange(1, 9, dtype=torch.float32)
    expected = encoder(features)
    path = tmp_path / "fourier.pt"
    torch.save(encoder.state_dict(), path)
    restored = FixedFourierKeys(hidden_size=8, bandwidth=0.4)
    state = torch.load(path, weights_only=True)
    assert set(state) == {"frequencies", "bandwidth"}
    restored.load_state_dict(state)
    assert torch.equal(expected, restored(features))
    assert torch.equal(encoder.bandwidth, restored.bandwidth)


@pytest.mark.parametrize("bandwidth", [0, -0.2, float("nan"), float("inf"), True, 1e-60])
def test_invalid_bandwidth_rejected(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        FixedFourierKeys(hidden_size=8, bandwidth=bandwidth)


@pytest.mark.parametrize("features", [
    torch.ones(7), torch.ones(8, dtype=torch.long), torch.zeros(8),
    torch.full((8,), float("nan")), torch.full((8,), float("inf")),
    torch.ones(8, device="meta"),
])
def test_invalid_input_rejected(features):
    encoder = FixedFourierKeys(hidden_size=8, bandwidth=0.1)
    with pytest.raises(ValueError):
        encoder(features)


def test_corrupt_restored_buffers_are_rejected():
    encoder = FixedFourierKeys(hidden_size=8, bandwidth=0.1)
    state = encoder.state_dict()
    state["bandwidth"] = torch.tensor(-0.2)
    encoder.load_state_dict(state)
    with pytest.raises(ValueError, match="positive bandwidth"):
        encoder(torch.ones(8))
    encoder = FixedFourierKeys(hidden_size=8, bandwidth=0.1).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="FP32"):
        encoder(torch.ones(8))
