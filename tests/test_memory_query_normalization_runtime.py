"""Normalized addressing integration; fake-model tests make no language claims."""

import json

import pytest
import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.lexical_bridge import LexicalEmbeddingBridge
from alice_memory.matrix import AssociativeMemory
from alice_memory.query_normalization import VERSION, normalize_memory_query
from alice_memory.runtime import Fact, MemoryRuntime, Question
from alice_runtime.loading import LoadedQwen
from test_memory_runtime import FrozenLM, Tokenizer


OLD_KEY_INPUT = "framed_last_hidden_v1"
VARIANTS = (
    "What is the assigned material of object Z-19?",
    "Which material was assigned to object Z-19?",
    "Tell me the assigned material of object Z-19.",
)


def make_runtime(key_input=VERSION, input_layout="known_answer_dense_v1"):
    torch.manual_seed(123)
    model, tokenizer = FrozenLM(), Tokenizer()
    bridge = LexicalEmbeddingBridge(hidden_size=12, key_input=key_input,
                                     input_layout=input_layout)
    bridge.basis_id = "test-normalized/" + bridge.parameter_digest()
    bridge.freeze()
    memory = AssociativeMemory(latent_basis_id=bridge.basis_id)
    # The fake tokenizer emits one token per character, unlike the real BPE.
    return MemoryRuntime(LoadedQwen(model, tokenizer, {}), bridge, memory, max_tokens=512)


def test_actual_acquisition_and_three_read_paths_share_key_and_source(monkeypatch):
    app = make_runtime()
    written = []
    original_acquire = app.memory.acquire

    def record_write(source_id, key, value, **kwargs):
        written.append((source_id, key.detach().clone(), value.detach().clone()))
        return original_acquire(source_id, key, value, **kwargs)

    monkeypatch.setattr(app.memory, "acquire", record_write)
    app.acquire(Fact("person:1", VARIANTS[0], "private-value"))
    assert len(written) == 1 and written[0][0] == "person:1"
    source, written_key, written_value = written[0]
    for variant in VARIANTS:
        key = app.key_for(variant)
        assert torch.equal(written_key, key)
        assert torch.allclose(app.memory.read(source, key), written_value, atol=2e-6, rtol=0)
    assert torch.count_nonzero(app.memory.matrix[app.memory.source_index("user")]) == 0

    def forbidden_value_path(_text):
        raise AssertionError("Answer must not reopen a teaching value")

    monkeypatch.setattr(app, "_value_feature", forbidden_value_path)
    observed_readouts = []
    original_slots = app.bridge.slots

    def record_read(query, readout, rms):
        assert torch.equal(query, written_key)
        observed_readouts.append(readout.detach().clone())
        return original_slots(query, readout, rms)

    monkeypatch.setattr(app.bridge, "slots", record_read)
    bridge_before, memory_before = app.bridge.parameter_digest(), app.memory.content_digest()
    app.tokenizer.seen.clear()
    for variant in VARIANTS:
        result = app.answer(Question(variant, "user", source), "on", max_new_tokens=1)
        assert result["memory_applied"]
        assert result["knowledge_source_id"] == source
    assert len(observed_readouts) == 3
    assert all(torch.equal(observed_readouts[0], readout) for readout in observed_readouts)
    assert all("private-value" not in text for text in app.tokenizer.seen)
    assert app.bridge.parameter_digest() == bridge_before
    assert app.memory.content_digest() == memory_before
    app.assert_qwen_frozen()


def test_distinct_entities_and_relations_reach_distinct_feature_input_views():
    app = make_runtime()
    questions = (
        VARIANTS[0],
        "What is the assigned material of object Z-20?",
        "What is the assigned storage place of object Z-19?",
    )
    views = []
    for question in questions:
        app.tokenizer.seen.clear()
        app.key_for(question)
        assert app.tokenizer.seen == [f"Question: {normalize_memory_query(question)}\nAnswer:"]
        views.append(app.tokenizer.seen[0])
    assert len(set(views)) == 3
    # This assertion concerns distinct addressing inputs, not semantic ability
    # of the intentionally simple character-tokenizing fake language model.


@pytest.mark.parametrize("layout", ["known_answer_dense_v1", "repeat_qa_dense_v1"])
def test_generation_keeps_original_paraphrase_instead_of_canonical_key_text(layout):
    app = make_runtime(input_layout=layout)
    app.acquire(Fact("user", VARIANTS[0], "private-value"))
    original = VARIANTS[1]
    repeat = layout == "repeat_qa_dense_v1"
    prefix_text = f"Question: {original}\n" + ("Answer:" if repeat else "Known answer:")
    tail_text = f"\nQuestion: {original}\nAnswer:" if repeat else "\nAnswer:"
    expected_prefix = app._prefix(app._ids(prefix_text))
    expected_tail = app._prefix(app._ids(tail_text))
    captured = []

    def capture(_module, _args, kwargs):
        assert kwargs["use_cache"] is False and kwargs.get("labels") is None
        captured.append(kwargs["inputs_embeds"].detach().clone())

    handle = app.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = app.answer(Question(original), "on", max_new_tokens=1)
    finally:
        handle.remove()
    assert result["memory_applied"] and len(captured) == 1
    embeddings = captured[0]
    assert embeddings.shape[1] == expected_prefix.shape[1] + 1 + expected_tail.shape[1]
    assert torch.equal(embeddings[:, :expected_prefix.shape[1]], expected_prefix)
    assert torch.equal(embeddings[:, -expected_tail.shape[1]:], expected_tail)


def test_old_mode_uses_original_feature_view_without_normalizing():
    app = make_runtime(key_input=OLD_KEY_INPUT)
    for question in VARIANTS:
        app.tokenizer.seen.clear()
        actual = app.key_for(question)
        assert app.tokenizer.seen == [f"Question: {question}\nAnswer:"]
        expected = app.bridge.key(app._feature(app._question_ids(question)))
        assert torch.equal(actual, expected)
    default = LexicalEmbeddingBridge(hidden_size=12)
    assert default.key_input == OLD_KEY_INPUT


def test_normalizer_semantics_survive_independent_bridge_memory_restore(tmp_path):
    app = make_runtime()
    app.acquire(Fact("user", VARIANTS[0], "private-value"))
    app.bridge.save(tmp_path / "bridge")
    app.memory.save(tmp_path / "memory")
    restored_bridge = LexicalEmbeddingBridge.load(tmp_path / "bridge", latent_basis_id=app.bridge.basis_id)
    restored_memory = AssociativeMemory.load(tmp_path / "memory", latent_basis_id=restored_bridge.basis_id)
    restored = MemoryRuntime(LoadedQwen(app.model, Tokenizer(), {}), restored_bridge,
                             restored_memory, max_tokens=512)
    assert restored_bridge.key_input == VERSION
    assert restored_bridge.parameter_digest() == app.bridge.parameter_digest()
    for question in VARIANTS:
        assert torch.equal(restored.key_for(question), app.key_for(VARIANTS[0]))
    before = restored_bridge.parameter_digest()
    weights = {name: value.clone() for name, value in restored_bridge.state_dict().items()}
    restored_bridge.key_input = OLD_KEY_INPUT
    assert restored_bridge.parameter_digest() != before
    assert all(torch.equal(weights[name], value) for name, value in restored_bridge.state_dict().items())
    with pytest.raises(BridgeStateError, match="changed"):
        restored_bridge.assert_frozen()

    manifest_path = tmp_path / "bridge" / "bridge.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["key_input"] == VERSION
    manifest["key_input"] = OLD_KEY_INPUT
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BridgeStateError, match="digest"):
        LexicalEmbeddingBridge.load(tmp_path / "bridge")
