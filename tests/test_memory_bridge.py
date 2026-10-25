"""Neural Bridge boundaries, full gradient path and independent checkpoints."""

import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError, ParametricBridge


def test_gradient_paths_and_zero_memory():
    torch.manual_seed(3)
    bridge = ParametricBridge(hidden_size=8, latent_dim=4, slots=2, basis_id="test-basis")
    cue = torch.randn(8)
    value = torch.randn(8)
    query = torch.randn(8)
    key = bridge.key(cue)
    latent = bridge.value(value)
    memory = torch.outer(latent, key)  # Functional episode write must retain gradients.
    q = bridge.key(query)
    output = bridge.slots(q, memory @ q, 0.2)
    assert output.shape == (2, 8)
    assert torch.count_nonzero(bridge.slots(q, torch.zeros(4), 0.2)) == 0
    assert latent.norm() <= 1
    output.square().sum().backward()
    for name, parameter in bridge.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_fixed_bridge_state_swap_changes_neural_payload():
    torch.manual_seed(4)
    bridge = ParametricBridge(hidden_size=8, latent_dim=4, slots=2).freeze()
    digest = bridge.parameter_digest()
    q = bridge.key(torch.randn(8))
    value_a, value_b = bridge.value(torch.randn(8)), bridge.value(torch.randn(8))
    memory_a, memory_b = torch.outer(value_a, q), torch.outer(value_b, q)
    payload_a = bridge.slots(q, memory_a @ q, 0.2)
    payload_b = bridge.slots(q, memory_b @ q, 0.2)
    assert not torch.equal(payload_a, payload_b)
    bridge.assert_frozen()
    assert bridge.parameter_digest() == digest
    with torch.no_grad():
        bridge.Wo.weight.add_(0.01)
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()


def test_separate_checkpoint_roundtrip_and_compatibility(tmp_path):
    bridge = ParametricBridge(hidden_size=8, latent_dim=4, slots=2,
                              basis_id="stable", bindings={"backbone": "fixed"}).freeze()
    bridge.save(tmp_path)
    restored = ParametricBridge.load(tmp_path, latent_basis_id="stable",
                                     expected_bindings={"backbone": "fixed"})
    restored.assert_frozen()
    assert restored.parameter_digest() == bridge.parameter_digest()
    manifest = json.loads((tmp_path / "bridge.json").read_text())
    assert set(manifest) == {"schema", "latent_basis_id", "hidden_size", "latent_dim", "slots",
                             "bindings", "value_encoder", "tensor_file", "sha256"}
    with pytest.raises(BridgeStateError, match="basis"):
        ParametricBridge.load(tmp_path, latent_basis_id="different")
    with pytest.raises(BridgeStateError, match="binding"):
        ParametricBridge.load(tmp_path, expected_bindings={"backbone": "different"})
    (tmp_path / manifest["tensor_file"]).write_bytes(b"corrupted")
    with pytest.raises(BridgeStateError, match="digest"):
        ParametricBridge.load(tmp_path)


def test_input_contract_and_acquisition_freeze():
    bridge = ParametricBridge(hidden_size=8, latent_dim=4, slots=2)
    with pytest.raises(BridgeStateError, match="frozen"):
        bridge.assert_frozen()
    with pytest.raises(BridgeStateError, match="dimension"):
        bridge.key(torch.zeros(7))
    with pytest.raises(BridgeStateError, match="nonfinite"):
        bridge.value(torch.full((8,), float("nan")))
    with pytest.raises(BridgeStateError, match="positive"):
        bridge.slots(torch.zeros(4), torch.zeros(4), 0)
    with pytest.raises(BridgeStateError, match="shapes"):
        bridge.slots(torch.zeros(2, 4), torch.zeros(4), 0.2)


def test_legacy_mode_is_preserved_and_encoder_mode_is_guarded(tmp_path):
    bridge = ParametricBridge(hidden_size=8, latent_dim=4, slots=2,
                              value_encoder=ParametricBridge.LEGACY_VALUE_ENCODER).freeze()
    x = torch.arange(8, dtype=torch.float32)
    old_output, old_digest = bridge.value(x), bridge.parameter_digest()
    bridge.save(tmp_path)
    path = tmp_path / "bridge.json"
    manifest = json.loads(path.read_text())
    manifest.pop("value_encoder")  # Exact original checkpoint envelope.
    path.write_text(json.dumps(manifest))
    restored = ParametricBridge.load(tmp_path)
    assert restored.value_encoder == ParametricBridge.LEGACY_VALUE_ENCODER
    assert torch.equal(old_output, restored.value(x))
    assert restored.parameter_digest() == old_digest
    restored.value_encoder = ParametricBridge.NORMALIZED_VALUE_ENCODER
    with pytest.raises(BridgeStateError, match="changed"):
        restored.assert_frozen()


def test_radial_bound_keeps_distinct_values_and_gradients_under_large_scale():
    bridge = ParametricBridge(hidden_size=4, latent_dim=4, slots=2)
    with torch.no_grad():
        bridge.Wv.weight.copy_(torch.eye(4))
    x = torch.tensor([[1000., 0., 0., 0.], [0., 1000., 0., 0.], [0., 0., 1000., 0.]],
                     requires_grad=True)
    values = bridge.value(x)
    assert torch.isfinite(values).all() and (values.norm(dim=-1) < 1).all()
    assert torch.pdist(values).min() > 0
    values.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(bridge.Wv.weight.grad).all()
