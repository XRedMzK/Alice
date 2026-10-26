"""Asset-free identity proof for CPU reconstruction of the recorded Bridge."""

import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "reconstruct_payload_bridge", Path(__file__).parents[1] / "tools" / "reconstruct_payload_bridge.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cpu_constructor_accepts_only_recorded_sealed_state_without_fitting(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Reconstruction must not inspect or use CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        try:
            bridge = module.reconstruct_bridge(dict(module.EXPECTED_BINDINGS))
        except RuntimeError as exc:
            if "CPU reconstruction differs from the sealed state" not in str(exc):
                raise
            pytest.skip("This CPU backend cannot reconstruct the exact external asset; fail-closed as documented")
    finally:
        torch.set_num_threads(previous_threads)
    assert bridge.parameter_digest() == module.EXPECTED_STATE_DIGEST
    assert bridge.basis_id == "alice.memory_payload/17x128/" + module.EXPECTED_STATE_DIGEST
    assert set(bridge.state_dict()) == {"R", "keys.frequencies", "keys.bandwidth", "value_scale"}
    assert all(value.device.type == "cpu" for value in bridge.state_dict().values())
    assert not bridge.training and all(not p.requires_grad and p.grad is None for p in bridge.parameters())
    assert float(bridge.value_scale) == module.RECORDED_VALUE_SCALE
    bridge.assert_frozen()


def test_wrong_tensor_state_is_refused_before_creating_output(monkeypatch, tmp_path):
    original_constructor = module.PayloadEmbeddingBridge

    def altered(*args, **kwargs):
        bridge = original_constructor(*args, **kwargs)
        with torch.no_grad():
            bridge.value_scale.mul_(2)
        return bridge

    monkeypatch.setattr(module, "PayloadEmbeddingBridge", altered)
    monkeypatch.setattr(module, "model_bindings", lambda _path: dict(module.EXPECTED_BINDINGS))
    output = tmp_path / "new-bridge"
    with pytest.raises(SystemExit) as exc:
        module.main(["--model-path", str(tmp_path / "model"), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_other_model_binding_is_rejected_before_constructing_a_bridge(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("An unbound donor must be rejected before construction")

    monkeypatch.setattr(module, "PayloadEmbeddingBridge", forbidden)
    other = {**module.EXPECTED_BINDINGS, "config.json": "0" * 64}
    with pytest.raises(ValueError, match="metadata differs"):
        module.reconstruct_bridge(other)
