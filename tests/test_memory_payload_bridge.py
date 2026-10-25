"""Full-rank transport, confidence and independent payload Bridge state."""

import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_memory.query_normalization import VERSION


def test_deterministic_model_independent_rotation_and_lossless_continuous_transport():
    torch.manual_seed(44)
    before = torch.random.get_rng_state()
    bridge = PayloadEmbeddingBridge(value_scale=2.3)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(991)
    another = PayloadEmbeddingBridge(value_scale=2.3)
    assert torch.equal(bridge.R, another.R)
    assert bridge.R.requires_grad
    assert bridge.key_input == VERSION
    assert bridge.value_input == "token_embedding_mean_v1"
    assert bridge.input_layout == "repeat_qa_turn_stop_v1"
    assert set(dict(bridge.named_parameters())) == {"R"}
    query = bridge.key(torch.linspace(-1., 1., 2048))
    for features in (torch.randn(2048), torch.randn(3, 2048) * 12):
        payload = bridge.value(features)
        assert payload.shape == (*features.shape[:-1], 17, 128)
        assert torch.equal(payload[..., 0, 0], torch.ones_like(payload[..., 0, 0]))
        assert torch.count_nonzero(payload[..., 0, 1:]) == 0
        q = query.expand(*features.shape[:-1], 128)
        decoded = bridge.slots(q, payload, .2).squeeze(-2)
        assert torch.allclose(decoded, features, atol=2e-5, rtol=2e-6)
        assert torch.allclose(bridge.value(features * 3)[..., 1:, :],
                              payload[..., 1:, :] * 3, atol=2e-5, rtol=1e-6)
    assert torch.count_nonzero(bridge.slots(query, torch.zeros(17, 128), .2)) == 0


def test_control_head_confidence_ignores_value_magnitude_and_shapes_are_explicit():
    bridge = PayloadEmbeddingBridge()
    payload = torch.full((17, 128), 1e20)
    payload[0, 0] = .59
    assert not bridge.accepts(payload)
    payload[0, 0] = .6
    assert bridge.accepts(payload)
    payload.zero_()
    payload[0, 0] = 1
    assert bridge.accepts(payload)
    with pytest.raises(BridgeStateError, match="one"):
        bridge.accepts(torch.ones(2, 17, 128))
    with pytest.raises(BridgeStateError, match="2048"):
        PayloadEmbeddingBridge(hidden_size=1024)
    with pytest.raises(BridgeStateError, match="17,128"):
        bridge.slots(torch.zeros(128), torch.zeros(16, 128), .2)
    with pytest.raises(BridgeStateError, match="finite"):
        bridge.value(torch.full((2048,), float("nan")))


def test_freeze_covers_rotation_scale_fourier_buffers_and_semantics():
    bridge = PayloadEmbeddingBridge().freeze()
    original = bridge.parameter_digest()
    with torch.no_grad():
        bridge.R[:, 0].neg_()  # Still orthogonal; mutation must nevertheless fail.
    assert bridge.parameter_digest() != original
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge = PayloadEmbeddingBridge().freeze()
    with torch.no_grad():
        bridge.value_scale.fill_(2)
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge = PayloadEmbeddingBridge().freeze()
    with torch.no_grad():
        bridge.keys.frequencies[0, 0] += .1
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge = PayloadEmbeddingBridge().freeze()
    bridge.input_layout = "other"
    with pytest.raises(BridgeStateError, match="semantics"):
        bridge.assert_frozen()


def test_separate_strict_checkpoint_roundtrip_and_corruption(tmp_path):
    bridge = PayloadEmbeddingBridge(value_scale=1.7, bindings={"backbone": "fixed"}).freeze()
    bridge.save(tmp_path)
    manifest = json.loads((tmp_path / "bridge.json").read_text())
    assert manifest["schema"] == "alice.payload_embedding_bridge/v1"
    assert manifest["payload_shape"] == [17, 128]
    assert manifest["key_input"] == VERSION
    assert manifest["value_scale"] == pytest.approx(1.7)
    assert (tmp_path / manifest["tensor_file"]).stat().st_size < 2**20
    restored = PayloadEmbeddingBridge.load(tmp_path, latent_basis_id=bridge.basis_id,
                                           expected_bindings={"backbone": "fixed"})
    restored.assert_frozen()
    assert restored.parameter_digest() == bridge.parameter_digest()
    features = torch.randn(2048)
    assert torch.equal(restored.value(features), bridge.value(features))
    with pytest.raises(BridgeStateError, match="basis"):
        PayloadEmbeddingBridge.load(tmp_path, latent_basis_id="wrong")
    with pytest.raises(BridgeStateError, match="binding"):
        PayloadEmbeddingBridge.load(tmp_path, expected_bindings={"backbone": "wrong"})
    manifest["rotation_mode"] = "other"
    (tmp_path / "bridge.json").write_text(json.dumps(manifest))
    with pytest.raises(BridgeStateError, match="semantics/configuration"):
        PayloadEmbeddingBridge.load(tmp_path)
    manifest["rotation_mode"] = "shared_row_orthogonal_v1"
    (tmp_path / "bridge.json").write_text(json.dumps(manifest))
    (tmp_path / manifest["tensor_file"]).write_bytes(b"corrupt")
    with pytest.raises(BridgeStateError, match="digest"):
        PayloadEmbeddingBridge.load(tmp_path)


def test_orthogonality_required_and_retraction_keeps_an_authorized_trainable_rotation():
    bridge = PayloadEmbeddingBridge()
    with torch.no_grad():
        bridge.R[0, 0] += .1
    with pytest.raises(BridgeStateError, match="orthogonal"):
        bridge.value(torch.zeros(2048))
    bridge.project_orthogonal_()
    bridge.value(torch.zeros(2048))
    features = torch.randn(2048)
    bridge.slots(torch.ones(128), bridge.value(features), .2).square().sum().backward()
    assert bridge.R.grad is not None and torch.isfinite(bridge.R.grad).all()
    bridge.freeze()
    with pytest.raises(BridgeStateError, match="frozen"):
        bridge.project_orthogonal_()


def test_new_payload_schema_cannot_overwrite_a_legacy_checkpoint(tmp_path):
    manifest = tmp_path / "bridge.json"
    content = '{"schema":"alice.lexical_embedding_bridge/v1","latent_basis_id":"old"}'
    manifest.write_text(content)
    with pytest.raises(BridgeStateError, match="another Bridge"):
        PayloadEmbeddingBridge().save(tmp_path)
    assert manifest.read_text() == content
    assert not list(tmp_path.glob("*.safetensors"))
