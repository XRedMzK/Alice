"""Versioned turn stopping with scripted frozen logits, not language claims."""

from types import SimpleNamespace

import pytest
import torch

from alice_memory.bridge import BridgeStateError
from alice_memory.lexical_bridge import LexicalEmbeddingBridge
from alice_memory.matrix import AssociativeMemory
from alice_memory.query_normalization import VERSION
from alice_memory.runtime import Fact, MemoryRuntime, Question
from alice_runtime.loading import LoadedQwen
from test_memory_runtime import FrozenLM, Tokenizer


TURN_LAYOUT = "repeat_qa_turn_stop_v1"
TURN = "\nQuestion:"
QUESTION = "What is the assigned label of object A-17?"


class ChunkTokenizer(Tokenizer):
    """Input encoding stays fake; generated IDs decode to controlled chunks."""

    def __init__(self, chunks):
        super().__init__()
        self.chunks = {i + 1: chunk for i, chunk in enumerate(chunks)}
        self.decoded_ids = []

    def decode(self, ids, skip_special_tokens=True):
        ids = tuple(int(token) for token in ids)
        self.decoded_ids.append(ids)
        return "".join(self.chunks[token] for token in ids)


class ScriptedFrozenLM(FrozenLM):
    """The real generation loop still selects from full fake-vocabulary logits."""

    def __init__(self, tokens):
        super().__init__()
        self.script = tuple(tokens)
        self.calls = 0

    def forward(self, inputs_embeds, labels=None, logits_to_keep=0, **kwargs):
        assert labels is None and kwargs["use_cache"] is False
        assert self.calls < len(self.script), "Generation exceeded the scripted boundary"
        token = self.script[self.calls]
        self.calls += 1
        logits = inputs_embeds.new_full((inputs_embeds.shape[0], 1, 32), -100)
        logits[:, :, token] = 100
        return SimpleNamespace(loss=None, logits=logits)


def make_runtime(chunks, *, layout=TURN_LAYOUT, append_eos=False, acquire=False):
    assert len(chunks) < 32
    torch.manual_seed(900)
    script = list(range(1, len(chunks) + 1)) + ([0] if append_eos else [])
    model, tokenizer = ScriptedFrozenLM(script), ChunkTokenizer(chunks)
    bridge = LexicalEmbeddingBridge(hidden_size=12, input_layout=layout, key_input=VERSION)
    bridge.basis_id = "test-turn-stop/" + bridge.parameter_digest()
    bridge.freeze()
    memory = AssociativeMemory(latent_basis_id=bridge.basis_id)
    app = MemoryRuntime(LoadedQwen(model, tokenizer, {}), bridge, memory, max_tokens=512)
    if acquire:
        app.acquire(Fact("user", QUESTION, "private-calibration-value"))
    return app


@pytest.mark.parametrize("mode", ["on", "off", "empty", "base", "format_only"])
def test_split_token_turn_boundary_stops_identically_in_every_arm(mode):
    chunks = [" amber-137", "\nQue", "st", "ion", ":", "ignored continuation"]
    app = make_runtime(chunks, acquire=True)
    bridge_before, memory_before = app.bridge.parameter_digest(), app.memory.content_digest()
    result = app.answer(Question(QUESTION), mode, max_new_tokens=16)
    assert result["text"] == "amber-137"
    assert result["raw_generated_text"] == " amber-137\nQuestion:"
    assert result["finish_reason"] == "turn_boundary"
    assert result["stop_sequence"] == TURN
    # Boundary tokens remain represented in the raw output/count. No following
    # token is requested, and no expected answer or palette reaches the runtime.
    assert result["tokens"] == app.model.calls == 5
    assert app.tokenizer.decoded_ids[-1] == (1, 2, 3, 4, 5)
    assert app.bridge.parameter_digest() == bridge_before
    assert app.memory.content_digest() == memory_before
    app.assert_qwen_frozen()


@pytest.mark.parametrize("value", ["copper", "медь", "two words", "123.45", "Question: literal"])
def test_turn_stop_preserves_arbitrary_answer_content(value):
    app = make_runtime([" " + value, "\nQuestion: next turn in same token"])
    result = app.answer(Question(QUESTION), "off", max_new_tokens=16)
    assert result["text"] == value
    assert result["raw_generated_text"] == " " + value + "\nQuestion: next turn in same token"
    assert result["tokens"] == app.model.calls == 2
    assert result["finish_reason"] == "turn_boundary"


def test_similar_text_does_not_stop_and_eos_is_reported():
    chunks = ["literal", " Question:", " is inline", "\nQuestionable:", " retained"]
    app = make_runtime(chunks, append_eos=True)
    result = app.answer(Question(QUESTION), "base", max_new_tokens=16)
    assert result["raw_generated_text"] == "".join(chunks)
    assert result["text"] == "".join(chunks)
    assert result["finish_reason"] == "eos"
    assert result["tokens"] == len(chunks)
    assert app.model.calls == len(chunks) + 1
    assert app.tokenizer.decoded_ids[-1] == tuple(range(1, len(chunks) + 1))


def test_incomplete_delimiter_remains_raw_at_token_limit():
    app = make_runtime([" non-color-answer", "\nQue"])
    result = app.answer(Question(QUESTION), "empty", max_new_tokens=2)
    assert result["text"] == "non-color-answer\nQue"
    assert result["raw_generated_text"] == " non-color-answer\nQue"
    assert result["finish_reason"] == "token_limit"
    assert result["tokens"] == 2


@pytest.mark.parametrize("layout", ["repeat_qa_dense_v1", "known_answer_dense_v1"])
def test_legacy_layout_does_not_apply_turn_stopping(layout):
    chunks = [" arbitrary-answer", "\nQue", "stion:", " still generated"]
    app = make_runtime(chunks, layout=layout)
    result = app.answer(Question(QUESTION), "off", max_new_tokens=len(chunks))
    assert result["text"] == "".join(chunks).strip()
    assert result["tokens"] == app.model.calls == len(chunks)
    assert not ({"raw_generated_text", "finish_reason", "stop_sequence"} & result.keys())


def test_turn_stop_layout_restores_and_is_covered_by_semantic_freeze(tmp_path):
    app = make_runtime(["unused"])
    original = app.bridge
    original.save(tmp_path)
    restored = LexicalEmbeddingBridge.load(tmp_path, latent_basis_id=original.basis_id)
    assert restored.input_layout == TURN_LAYOUT and restored.key_input == VERSION
    assert restored.parameter_digest() == original.parameter_digest()
    assert all(torch.equal(tensor, restored.state_dict()[name])
               for name, tensor in original.state_dict().items())
    restored.assert_frozen()
    restored.input_layout = "repeat_qa_dense_v1"
    with pytest.raises(BridgeStateError, match="changed"):
        restored.assert_frozen()
