"""Affine value amplitudes, confidence and historical checkpoint compatibility."""

import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.lexical_bridge import LexicalEmbeddingBridge


def affine_bridge():
    bridge = LexicalEmbeddingBridge(hidden_size=8, value_encoder="affine_confidence_v1")
    with torch.no_grad():
        bridge.value_mean.copy_(torch.arange(8, dtype=torch.float32))
        bridge.Wv.weight.zero_()
        bridge.Wv.weight[1:9].copy_(torch.eye(8))
        bridge.Wo.weight.zero_()
        bridge.Wo.weight[:, 0].copy_(bridge.value_mean)
        bridge.Wo.weight[:, 1:9].copy_(torch.eye(8))
    return bridge


def test_affine_centering_preserves_content_amplitude_and_decoder_mean():
    bridge = affine_bridge()
    delta = torch.arange(1., 9.)
    a = bridge.value(bridge.value_mean + delta)
    b = bridge.value(bridge.value_mean + 3 * delta)
    assert a[0] == b[0] == 1
    assert torch.equal(b[1:], a[1:] * 3)
    assert not torch.allclose(a.norm(), b.norm())
    q = torch.ones(128)
    assert torch.equal(bridge.slots(q, a, .2)[0], bridge.value_mean + delta)
    assert torch.equal(bridge.slots(q, b, .2)[0], bridge.value_mean + 3 * delta)
    mean_code = bridge.value(bridge.value_mean)
    assert torch.equal(bridge.slots(q, mean_code, .2)[0], bridge.value_mean)
    assert torch.count_nonzero(bridge.slots(q, torch.zeros(128), .2)) == 0
    batch = bridge.value(torch.stack((bridge.value_mean + delta, bridge.value_mean + 3 * delta)))
    assert torch.equal(batch, torch.stack((a, b)))


def test_affine_confidence_is_independent_of_content_magnitude():
    bridge = affine_bridge()
    r = torch.full((128,), 1e20)
    r[0] = .59
    assert not bridge.accepts(r)
    r[0] = -1
    assert not bridge.accepts(r)
    r.zero_()
    r[0] = .6
    assert bridge.accepts(r)
    assert not bridge.accepts(torch.zeros(128))
    r[1] = float("nan")
    assert not bridge.accepts(r)
    r[1] = float("inf")
    assert not bridge.accepts(r)
    with pytest.raises(BridgeStateError, match="one latent"):
        bridge.accepts(torch.ones(2, 128))


def test_constant_confidence_has_no_gradient_while_content_and_decoder_train():
    bridge = affine_bridge()
    features = (bridge.value_mean + torch.arange(1., 9.)).requires_grad_()
    value = bridge.value(features)
    slots = bridge.slots(torch.ones(128), value, .2)
    slots.square().sum().backward()
    assert features.grad is not None and torch.count_nonzero(features.grad) == 8
    assert bridge.Wv.weight.grad is not None
    assert torch.count_nonzero(bridge.Wv.weight.grad[0]) == 0
    assert torch.count_nonzero(bridge.Wv.weight.grad[1:9])
    assert bridge.Wo.weight.grad is not None and torch.count_nonzero(bridge.Wo.weight.grad[:, 0])
    assert not bridge.value_mean.requires_grad and bridge.value_mean.grad is None
    assert set(dict(bridge.named_parameters())) == {"Wv.weight", "Wo.weight"}


def test_affine_roundtrip_hashes_mean_and_encoder_configuration(tmp_path):
    bridge = affine_bridge()
    bridge.basis_id = "affine-test-basis"
    bridge.bindings = {"backbone": "frozen"}
    bridge.freeze().save(tmp_path)
    manifest = json.loads((tmp_path / "bridge.json").read_text())
    assert manifest["value_encoder"] == "affine_confidence_v1"
    assert manifest["layer_norm_eps"] is None
    assert "value_mean" in bridge.state_dict()
    restored = LexicalEmbeddingBridge.load(tmp_path, latent_basis_id=bridge.basis_id,
                                           expected_bindings=bridge.bindings)
    restored.assert_frozen()
    assert restored.parameter_digest() == bridge.parameter_digest()
    features = bridge.value_mean + torch.arange(1., 9.)
    assert torch.equal(restored.value(features), bridge.value(features))
    assert torch.equal(restored.Wo(restored.value(features)), bridge.Wo(bridge.value(features)))
    before = restored.parameter_digest()
    with torch.no_grad():
        restored.value_mean[0] += 1
    assert restored.parameter_digest() != before
    with pytest.raises(BridgeStateError, match="changed"):
        restored.assert_frozen()
    manifest["layer_norm_eps"] = 1e-5
    (tmp_path / "bridge.json").write_text(json.dumps(manifest))
    with pytest.raises(BridgeStateError, match="semantics/configuration"):
        LexicalEmbeddingBridge.load(tmp_path)


def test_affine_state_is_strict_about_mean_shape_finiteness_and_semantics():
    bridge = affine_bridge()
    bridge.value_mean = torch.zeros(7)
    with pytest.raises(BridgeStateError, match="mean"):
        bridge.parameter_digest()
    bridge.value_mean = torch.full((8,), float("nan"))
    with pytest.raises(BridgeStateError, match="finite FP32"):
        bridge.parameter_digest()
    with pytest.raises(BridgeStateError, match="semantics"):
        LexicalEmbeddingBridge(hidden_size=8, value_encoder="unknown")
    legacy = LexicalEmbeddingBridge(hidden_size=8)
    legacy.value_encoder = "affine_confidence_v1"
    with pytest.raises(BridgeStateError, match="mean"):
        legacy.parameter_digest()


def test_default_encoder_retains_captured_pre_affine_digest_and_legacy_roundtrip(tmp_path):
    # Captured from f4db9c4 before adding the affine encoder, with these exact
    # deterministic weights. The old checkpoint has no added buffers/config keys.
    bridge = LexicalEmbeddingBridge(hidden_size=8)
    with torch.no_grad():
        bridge.Wv.weight.copy_(torch.arange(bridge.Wv.weight.numel(), dtype=torch.float32)
                               .reshape_as(bridge.Wv.weight) / 1024)
        bridge.Wo.weight.copy_(torch.arange(bridge.Wo.weight.numel(), dtype=torch.float32)
                               .reshape_as(bridge.Wo.weight) / 2048)
    assert bridge.parameter_digest() == "ac1f497728e768b9b6d9833dec1b138df6b942eebdce367f921c222138739cfc"
    assert "value_mean" not in bridge.state_dict()
    assert bridge._config()["layer_norm_eps"] == 1e-5
    features = torch.arange(1., 9.)
    value = bridge.value(features)
    assert torch.allclose(value.norm(), torch.tensor(1.))
    bridge.freeze().save(tmp_path)
    restored = LexicalEmbeddingBridge.load(tmp_path)
    assert restored.parameter_digest() == bridge.parameter_digest()
    assert torch.equal(restored.value(features), value)
    assert "value_mean" not in restored.state_dict()
