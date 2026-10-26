"""Gradient separation, question-only inputs and causal state substitution."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from alice_memory.bridge import ParametricBridge
from alice_memory.matrix import AssociativeMemory
from alice_memory.runtime import Fact, MemoryRuntime, Question
from alice_runtime.loading import LoadedQwen


class Tokenizer:
    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=False):
        self.seen.append(text)
        return [1 + ord(c) % 30 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class Encoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=x.cumsum(1))


class FrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.model = Encoder(self.embedding)
        self.head = nn.Linear(12, 32)
        self.config = SimpleNamespace(eos_token_id=0)
        self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, labels=None, logits_to_keep=0, **kwargs):
        logits = self.head(self.model(inputs_embeds=inputs_embeds).last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(loss=loss, logits=logits)


def make_runtime():
    torch.manual_seed(5)
    model, tokenizer = FrozenLM(), Tokenizer()
    bridge = ParametricBridge(hidden_size=12)
    memory = AssociativeMemory(latent_basis_id=bridge.basis_id)
    return MemoryRuntime(LoadedQwen(model, tokenizer, {}), bridge, memory)


def test_distinct_gradient_path_keeps_qwen_and_persistent_memory_fixed():
    app = make_runtime()
    before = {k: p.clone() for k, p in app.model.named_parameters()}
    memory = app.memory.content_digest()
    loss, _ = app.training_loss(
        [Fact("user", "Box X color?", "blue"), Fact("user", "Box Y color?", "green")],
        Question("Box X color?"), "blue", Question("Unrelated question?"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in app.model.parameters())
    assert all(torch.equal(before[k], p) for k, p in app.model.named_parameters())
    assert app.memory.content_digest() == memory
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in app.bridge.parameters())
    assert app.bridge.Wv.weight.grad.norm() > 0
    assert app.bridge.Wk.weight.grad.norm() > 0


def test_acquisition_and_state_swap_with_fixed_bridge_and_no_answer_input():
    app = make_runtime()
    app.bridge.freeze()
    theta = app.bridge.parameter_digest()
    app.acquire(Fact("user", "Where is V?", "north"))
    app.acquire(Fact("person:1", "Where is V?", "south"))
    ids, ctx = app.context(Question("Where is V?", "user", "person:1"))
    assert ctx.current_source_id == "user" and ctx.knowledge_source_id == "person:1"
    assert ctx.message_spans[0].end_token == ids.shape[1]
    q = app.bridge.key(app._feature(ids))
    original = app.memory.matrix.detach().clone()
    before = original[app.memory.source_index("user")] @ q
    with torch.no_grad():
        swapped = original.clone()
        swapped[[1, 2]] = original[[2, 1]]
        app.memory.matrix.copy_(swapped)
    after = app.memory.matrix[app.memory.source_index("user")] @ q
    assert not torch.equal(before, after)
    assert app.bridge.parameter_digest() == theta
    app.tokenizer.seen.clear()
    app.answer(Question("Where is V?"), max_new_tokens=1)
    assert all("north" not in text and "south" not in text for text in app.tokenizer.seen)
