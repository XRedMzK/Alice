"""Interactive state ownership and failed-write safety, using a tiny frozen LM."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.matrix import MemoryStateError
from alice_memory.payload_bridge import PayloadEmbeddingBridge
from alice_memory.payload_matrix import ParametricPayloadMemory
from alice_memory.runtime import Question
from alice_runtime.loading import LoadedQwen
from alice_runtime.neural_memory import NeuralMemorySession
from test_memory_payload_runtime import FrozenLM, Tokenizer


def setup(tmp_path, bindings=None):
    torch.manual_seed(107)
    loaded = LoadedQwen(FrozenLM(), Tokenizer(), {})
    bridge = PayloadEmbeddingBridge(bindings=bindings or {}).freeze()
    directory = tmp_path / "bridge"
    bridge.save(directory)
    return loaded, directory, tmp_path / "memory"


def tree_bytes(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in Path(directory).rglob("*") if p.is_file()}


def test_automatic_save_restart_source_correction_and_bounded_snapshots(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    bridge_bytes = tree_bytes(bridge_path)
    qwen = {n: p.detach().clone() for n, p in loaded.model.named_parameters()}
    cue = "What is the assigned hue of item C-42?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        assert app.engine.model is loaded.model
        assert not app.status()["persisted"]
        receipt = app.remember("user", cue, "quartz")
        assert receipt["persistent_m_changed"] and receipt["saved"]
        app.remember("person:1", cue, "ivory")
        app.remember("user", cue, "copper")
        assert len(list(state_path.glob("memory-payload-*.safetensors"))) == 1
        status = app.status()
        memory = app.memory.matrix.detach().clone()
        assert status["persisted"]
        assert status["source_ids"] == ["agent", "user", "person:1", "person:2"]
        assert app.engine.model is loaded.model
    assert tree_bytes(bridge_path) == bridge_bytes
    assert all(torch.equal(qwen[n], p) and p.grad is None
               for n, p in loaded.model.named_parameters())
    with NeuralMemorySession(loaded, bridge_path, state_path) as restored:
        assert restored.status()["memory_digest"] == status["memory_digest"]
        assert restored.status()["bridge_digest"] == status["bridge_digest"]
        assert torch.equal(restored.memory.matrix, memory)
        original = restored.memory.content_digest()
        for mode in ("off", "empty", "on"):
            result = restored.respond(Question(cue), mode, max_new_tokens=1)
            assert result["memory_applied"] == (mode == "on")
            assert restored.memory.content_digest() == original
        source_query = restored.engine.key_for(cue)
        user_value = restored.memory.read("user", source_query)
        other_value = restored.memory.read("person:1", source_query)
        assert not torch.equal(user_value, other_value)
        with pytest.raises(ValueError, match="one-token"):
            restored.remember("user", cue, "two words")
        assert restored.memory.content_digest() == original
    # Only neural metadata/tensor state and a process lock are persisted.
    assert sorted(p.name for p in state_path.iterdir()) == [
        ".alice-memory.lock", "memory-payload-" + json.loads((state_path / "memory.json").read_text())["snapshot_id"] + ".safetensors",
        "memory.json"]
    metadata = json.loads((state_path / "memory.json").read_text())
    assert not {"cue", "value", "facts", "messages", "token_ids"} & metadata.keys()


def test_failed_publication_restores_memory_and_old_checkpoint(tmp_path, monkeypatch):
    loaded, bridge_path, state_path = setup(tmp_path)
    cue = "What is the assigned hue of item B-35?"
    with NeuralMemorySession(loaded, bridge_path, state_path) as app:
        app.remember("user", cue, "quartz")
        before_memory = app.memory.matrix.detach().clone()
        before_state = tree_bytes(state_path)
        bridge_digest = app.bridge.parameter_digest()
        def failure(_path):
            raise OSError("disk full")
        monkeypatch.setattr(app.memory, "save", failure)
        with pytest.raises(OSError, match="disk full"):
            app.remember("user", cue, "copper")
        assert torch.equal(app.memory.matrix, before_memory)
        assert tree_bytes(state_path) == before_state
        assert app.bridge.parameter_digest() == bridge_digest
        app.engine.assert_qwen_frozen()


def test_exclusive_process_ownership_close_and_corruption_no_reset(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    first = NeuralMemorySession(loaded, bridge_path, state_path)
    try:
        with pytest.raises(MemoryStateError, match="already open"):
            NeuralMemorySession(loaded, bridge_path, state_path)
        first.remember("user", "What is the assigned hue of item M-24?", "ivory")
    finally:
        first.close()
    with pytest.raises(MemoryStateError, match="closed"):
        first.save()
    manifest_path = state_path / "memory.json"
    metadata = json.loads(manifest_path.read_text())
    metadata["source_ids"][1], metadata["source_ids"][2] = metadata["source_ids"][2], metadata["source_ids"][1]
    manifest_path.write_text(json.dumps(metadata))
    before = tree_bytes(state_path)
    with pytest.raises(MemoryStateError, match="integrity"):
        NeuralMemorySession(loaded, bridge_path, state_path)
    assert tree_bytes(state_path) == before


def test_paths_and_missing_manifests_fail_closed(tmp_path):
    loaded, bridge_path, state_path = setup(tmp_path)
    bridge_before = tree_bytes(bridge_path)
    for invalid in (bridge_path, bridge_path / "nested", tmp_path):
        with pytest.raises(MemoryStateError, match="separate"):
            NeuralMemorySession(loaded, bridge_path, invalid)
    assert tree_bytes(bridge_path) == bridge_before
    state_path.mkdir()
    (state_path / "unrelated.txt").write_text("Keep this user file")
    before = tree_bytes(state_path)
    with pytest.raises(MemoryStateError, match="refusing to reset"):
        NeuralMemorySession(loaded, bridge_path, state_path)
    assert tree_bytes(state_path) == before
    linked = tmp_path / "linked"
    linked.symlink_to(state_path, target_is_directory=True)
    with pytest.raises(MemoryStateError, match="symlinks"):
        NeuralMemorySession(loaded, bridge_path, linked)


def test_model_binding_and_source_contract_are_checked(tmp_path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model_root / name).write_text('{}\n')
    bindings = {name: hashlib.sha256((model_root / name).read_bytes()).hexdigest()
                for name in ("config.json", "tokenizer_config.json")}
    loaded, bridge_path, state_path = setup(tmp_path, bindings)
    with NeuralMemorySession(loaded, bridge_path, state_path, model_path=model_root) as app:
        app.save()
    (model_root / "config.json").write_text('{"changed":true}')
    with pytest.raises(BridgeStateError, match="binding mismatch"):
        NeuralMemorySession(loaded, bridge_path, state_path, model_path=model_root)
    other = tmp_path / "no-user"
    basis = json.loads((bridge_path / "bridge.json").read_text())["latent_basis_id"]
    ParametricPayloadMemory(("agent", "person:1"), basis).save(other)
    with pytest.raises(MemoryStateError, match="agent and user"):
        NeuralMemorySession(loaded, bridge_path, other)
