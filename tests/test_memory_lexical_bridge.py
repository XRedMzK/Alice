"""Continuous value transmission, fixed keys, and strict independent persistence."""

import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError, ParametricBridge
from alice_memory.lexical_bridge import LexicalEmbeddingBridge


def test_only_value_decoder_trains_and_fixed_keys_remain_unchanged():
    bridge = LexicalEmbeddingBridge(hidden_size=8)
    assert set(dict(bridge.named_parameters())) == {"Wv.weight", "Wo.weight"}
    before = {k: v.clone() for k, v in bridge.keys.state_dict().items()}
    key = bridge.key(torch.arange(1., 9.))
    value = bridge.value(torch.tensor([0., 2., 1., 4., 3., 6., 5., 7.]))
    assert value.shape == (128,) and torch.allclose(value.norm(), torch.tensor(1.))
    slots = bridge.slots(key, value, .2)
    assert slots.shape == (1, 8)
    slots.square().sum().backward()
    for parameter in bridge.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)
    assert all(torch.equal(v, before[k]) for k, v in bridge.keys.state_dict().items())


def test_continuous_output_and_scalar_acceptance_do_not_normalize_readout():
    with pytest.raises(BridgeStateError, match="basis"):
        LexicalEmbeddingBridge(hidden_size=8, basis_id="")
    bridge = LexicalEmbeddingBridge(hidden_size=8)
    q, r = torch.ones(128), torch.zeros(128)
    r[0] = 1
    assert not bridge.accepts(torch.zeros(128))
    assert not bridge.accepts(r * .59)
    assert bridge.accepts(r * .6) and bridge.accepts(r)
    assert not bridge.accepts(torch.full((128,), float("nan")))
    assert torch.count_nonzero(bridge.slots(q, torch.zeros_like(r), .2)) == 0
    assert torch.allclose(bridge.slots(q, r * .25, .2), bridge.slots(q, r, .9) * .25)
    with pytest.raises(BridgeStateError, match="one latent"):
        bridge.accepts(torch.ones(2, 128))
    with pytest.raises(BridgeStateError, match="device"):
        bridge.accepts(torch.ones(128, device="meta"))
    with pytest.raises(BridgeStateError, match="finite"):
        bridge.slots(q, torch.full((128,), float("nan")), .2)


def test_digest_and_freeze_cover_buffers_threshold_modes_and_coordinate_id():
    bridge = LexicalEmbeddingBridge(hidden_size=8).freeze()
    digest = bridge.parameter_digest()
    with torch.no_grad():
        bridge.keys.frequencies[0, 0] += 1
    assert bridge.parameter_digest() != digest
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge.freeze()
    bridge.acceptance_threshold = .7
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge.freeze()
    bridge.basis_id = "another-basis"
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge.freeze()
    bridge.input_layout = "unexpected"
    with pytest.raises(BridgeStateError, match="semantics"):
        bridge.assert_frozen()


def test_roundtrip_strict_schema_and_independent_old_bridge(tmp_path):
    root = tmp_path / "lexical"
    bridge = LexicalEmbeddingBridge(hidden_size=8, basis_id="stable",
                                     bindings={"backbone": "fixed"}).freeze()
    value = bridge.value(torch.arange(1., 9.))
    q = bridge.key(torch.arange(1., 9.))
    bridge.save(root)
    restored = LexicalEmbeddingBridge.load(root, latent_basis_id="stable",
                                           expected_bindings={"backbone": "fixed"})
    restored.assert_frozen()
    assert restored.parameter_digest() == bridge.parameter_digest()
    assert torch.equal(restored.slots(q, value, .2), bridge.slots(q, value, .2))
    assert torch.equal(restored.key(torch.arange(1., 9.)), q)
    with pytest.raises(BridgeStateError, match="basis"):
        LexicalEmbeddingBridge.load(root, latent_basis_id="other")
    with pytest.raises(BridgeStateError, match="binding"):
        LexicalEmbeddingBridge.load(root, expected_bindings={"backbone": "other"})
    manifest_path = root / "bridge.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == LexicalEmbeddingBridge.SCHEMA
    manifest["acceptance_threshold"] = .7
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BridgeStateError, match="digest"):
        LexicalEmbeddingBridge.load(root)
    old_root = tmp_path / "old"
    old = ParametricBridge(hidden_size=8, latent_dim=4, slots=2).freeze()
    old.save(old_root)
    assert ParametricBridge.load(old_root).parameter_digest() == old.parameter_digest()
    with pytest.raises(BridgeStateError, match="manifest"):
        LexicalEmbeddingBridge.load(old_root)


def test_atomic_save_keeps_previous_checkpoint_on_manifest_failure(tmp_path, monkeypatch):
    bridge = LexicalEmbeddingBridge(hidden_size=8).freeze()
    bridge.save(tmp_path)
    before = (tmp_path / "bridge.json").read_bytes()
    files = set(tmp_path.iterdir())
    def fail(*_args, **_kwargs):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr("alice_memory.lexical_bridge.os.replace", fail)
    with pytest.raises(OSError, match="simulated"):
        bridge.save(tmp_path)
    assert (tmp_path / "bridge.json").read_bytes() == before
    assert set(tmp_path.iterdir()) == files
    assert LexicalEmbeddingBridge.load(tmp_path).parameter_digest() == bridge.parameter_digest()


def test_layout_is_persisted_and_versioned_without_changing_neural_weights(tmp_path):
    bridge = LexicalEmbeddingBridge(hidden_size=8).freeze()
    state = {k: v.clone() for k, v in bridge.state_dict().items()}
    old_digest = bridge.parameter_digest()
    bridge.input_layout = "repeat_qa_dense_v1"
    assert bridge.parameter_digest() != old_digest
    with pytest.raises(BridgeStateError, match="changed"):
        bridge.assert_frozen()
    bridge.basis_id = "alice.memory-v0/128/" + bridge.parameter_digest()
    bridge.freeze().save(tmp_path)
    restored = LexicalEmbeddingBridge.load(tmp_path)
    assert restored.input_layout == "repeat_qa_dense_v1"
    assert restored.parameter_digest() == bridge.parameter_digest()
    assert all(torch.equal(state[k], v) for k, v in restored.state_dict().items())
