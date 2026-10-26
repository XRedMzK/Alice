"""Separate gradient and frozen acquisition paths over a local frozen backbone."""

from dataclasses import dataclass
import time

import torch
from torch.nn import functional as F

from alice_runtime.bridge import BridgeContext, MessageSpan
from alice_runtime.loading import LoadedQwen
from .bridge import ParametricBridge
from .matrix import AssociativeMemory, functional_delta


@dataclass(frozen=True)
class Fact:
    source_id: str
    cue: str
    value: str


@dataclass(frozen=True)
class Question:
    text: str
    speaker_source_id: str = "user"
    knowledge_source_id: str = "user"


class MemoryRuntime:
    """No session transcript, fact table or answer decoder is retained here."""

    def __init__(self, loaded: LoadedQwen, bridge: ParametricBridge,
                 memory: AssociativeMemory, max_tokens: int = 128):
        self.model, self.tokenizer, self.metadata = loaded.model, loaded.tokenizer, loaded.metadata
        self.bridge, self.memory, self.max_tokens = bridge, memory, max_tokens
        embedding = self.model.get_input_embeddings().weight
        self.device, self.dtype = embedding.device, embedding.dtype
        if (bridge.hidden_size != embedding.shape[-1] or bridge.latent_dim != 128
                or bridge.basis_id != memory.latent_basis_id
                or next(bridge.parameters()).device != self.device or memory.matrix.device != self.device):
            raise ValueError("Backbone, Bridge and memory tensor/basis boundary mismatch")
        self._versions = {n: p._version for n, p in self.model.named_parameters()}
        self.assert_qwen_frozen()

    def assert_qwen_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Qwen must be eval/frozen with no parameter gradients")
        if {n: p._version for n, p in self.model.named_parameters()} != self._versions:
            raise RuntimeError("Qwen parameters changed")

    def _ids(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Nonempty text required")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > self.max_tokens:
            raise ValueError("Input exceeds the prototype token budget")
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def _question_ids(self, question):
        # This view is identical for support keys and question-only queries.
        return self._ids(f"Question: {question}\nAnswer:")

    def context(self, request: Question, session_id="memory-request"):
        self.memory.source_index(request.speaker_source_id)
        self.memory.source_index(request.knowledge_source_id)
        ids = self._question_ids(request.text)
        context = BridgeContext(
            session_id, (request.speaker_source_id,),
            current_source_id=request.speaker_source_id,
            knowledge_source_id=request.knowledge_source_id,
            message_spans=(MessageSpan(request.speaker_source_id, 0, ids.shape[1]),),
        )
        return ids, context

    def _feature(self, ids):
        self.assert_qwen_frozen()
        # no_grad, never inference_mode: downstream projections may need autograd.
        with torch.no_grad():
            output = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      use_cache=False)
            return output.last_hidden_state[0, -1].detach().float().clone()

    def _value_feature(self, text):
        if getattr(self.bridge, "value_input", None) == "token_embedding_mean_v1":
            # Dense frozen input embeddings, never a value decoder or inference lookup.
            # These tokens are available only during generic fitting/acquisition.
            return self._prefix(self._ids(" " + text)).float().mean(dim=1)[0]
        return self._feature(self._ids(f"Value: {text}"))

    def key_for(self, question):
        """Answer-blind addressing; original question is kept for generation."""
        from .query_normalization import VERSION, normalize_memory_query
        view = (normalize_memory_query(question)
                if getattr(self.bridge, "key_input", None) == VERSION else question)
        return self.bridge.key(self._feature(self._question_ids(view)))

    def _lexical_input(self, question, readout):
        """Continuous neural value between fixed, answer-independent delimiters."""
        repeated = self.bridge.input_layout in {"repeat_qa_dense_v1", "repeat_qa_turn_stop_v1"}
        ids = self._ids(f"Question: {question}\n" + ("Answer:" if repeated else "Known answer:"))
        p = self._prefix(ids)
        query = self.key_for(question)
        slots = self.bridge.slots(query, readout, p.float().square().mean().sqrt()).to(self.dtype)
        tail = self._prefix(self._ids(f"\nQuestion: {question}\nAnswer:" if repeated else "\nAnswer:"))
        return torch.cat((p, slots.unsqueeze(0), tail), dim=1)

    def _prefix(self, ids):
        with torch.no_grad():
            return self.model.get_input_embeddings()(ids).detach()

    def _slots(self, query, state, target, prefix):
        readout = state[self.memory.source_index(target)] @ query
        rms = prefix.float().square().mean().sqrt().detach()
        slots = self.bridge.slots(query, readout, rms).to(self.dtype).unsqueeze(0)
        if not torch.isfinite(slots).all():
            raise RuntimeError("Nonfinite neural memory slots")
        return slots

    def _forward(self, embeddings, labels=None, last_only=False):
        if embeddings.shape[1] > self.max_tokens:
            raise ValueError("Combined question/slots/answer exceeds token budget")
        return self.model(inputs_embeds=embeddings.contiguous(),
                          attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                          labels=labels, use_cache=False,
                          logits_to_keep=1 if last_only else 0)

    def training_loss(self, supports: list[Fact], request: Question, target: str,
                      guard: Question | None = None):
        """Disposable differentiable memory; persistent M and Qwen never optimize.

        Only this method may teacher-force a target. Query/readout is complete
        before target tokenization; the inference answer method has no target arg.
        """
        if not self.bridge.training or not all(p.requires_grad for p in self.bridge.parameters()):
            raise RuntimeError("Training path requires trainable Bridge parameters")
        if getattr(self.bridge, "input_layout", None) in {"known_answer_dense_v1", "repeat_qa_dense_v1", "repeat_qa_turn_stop_v1"}:
            raise RuntimeError("Lexical Bridge uses its separate bounded fitting path")
        ids, _ = self.context(request)
        state = torch.zeros_like(self.memory.matrix)
        for fact in supports:
            key = self.bridge.key(self._feature(self._question_ids(fact.cue)))
            value = self.bridge.value(self._value_feature(fact.value))
            state = functional_delta(state, key, value, self.memory.source_index(fact.source_id))
        query = self.bridge.key(self._feature(ids))
        prefix = self._prefix(ids)
        slots = self._slots(query, state, request.knowledge_source_id, prefix)
        # No answer may reach state, query or slots above this line.
        target_ids = self._ids(" " + target)
        suffix = self._prefix(target_ids)
        embeddings = torch.cat((prefix, slots, suffix), dim=1)
        labels = torch.full(embeddings.shape[:2], -100, device=self.device, dtype=torch.long)
        labels[:, -target_ids.shape[1]:] = target_ids  # CausalLM shifts once internally.
        ce = self._forward(embeddings, labels=labels).loss
        loss = ce
        guard_loss = ce.detach().new_zeros(())
        if guard is not None:
            guard_ids, _ = self.context(guard)
            gp = self._prefix(guard_ids)
            gq = self.bridge.key(self._feature(guard_ids))
            gs = self._slots(gq, state, guard.knowledge_source_id, gp)
            with torch.no_grad():
                off = self._forward(torch.cat((gp, torch.zeros_like(gs)), 1), last_only=True).logits[0, -1].float()
            on = self._forward(torch.cat((gp, gs), 1), last_only=True).logits[0, -1].float()
            guard_loss = F.kl_div(on.log_softmax(-1), off.softmax(-1).detach(), reduction="sum")
            loss = ce + 0.1 * guard_loss
        self.assert_qwen_frozen()
        return loss, {"ce": float(ce.detach()), "guard_kl": float(guard_loss.detach())}

    def acquire(self, fact: Fact):
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        with torch.no_grad():
            key = self.key_for(fact.cue).detach()
            value = self.bridge.value(self._value_feature(fact.value)).detach()
            receipt = self.memory.acquire(fact.source_id, key, value)
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        return receipt

    def answer(self, request: Question, mode="on", max_new_tokens=8):
        if mode not in {"on", "off", "empty", "base", "format_only"} or not 1 <= max_new_tokens <= 16:
            raise ValueError("Invalid memory arm or generation budget")
        if mode == "format_only" and getattr(self.bridge, "input_layout", None) not in {"known_answer_dense_v1", "repeat_qa_dense_v1", "repeat_qa_turn_stop_v1"}:
            raise ValueError("format_only control requires the lexical embedding layout")
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        started = time.perf_counter()
        ids, context = self.context(request)
        before = self.memory.content_digest()
        with torch.no_grad():
            prefix = self._prefix(ids)
            query = self.key_for(request.text)
            state = torch.zeros_like(self.memory.matrix) if mode in {"empty", "format_only"} else self.memory.matrix
            readout = state[self.memory.source_index(request.knowledge_source_id)] @ query
            lexical = getattr(self.bridge, "input_layout", None) in {"known_answer_dense_v1", "repeat_qa_dense_v1", "repeat_qa_turn_stop_v1"}
            if lexical:
                # Relevance is derived from neural read strength, never cue/value text.
                applied = mode == "on" and self.bridge.accepts(readout)
                embeddings = self._lexical_input(request.text, readout) if applied or mode == "format_only" else prefix
            else:
                slots = self._slots(query, state, request.knowledge_source_id, prefix)
                if mode == "off":
                    slots = torch.zeros_like(slots)
                embeddings = prefix if mode == "base" else torch.cat((prefix, slots), dim=1)
                applied = mode == "on"
            generated = []
            stopped_text = None
            finish_reason = "token_limit"
            turn_stop = "\nQuestion:" if getattr(self.bridge, "input_layout", None) == "repeat_qa_turn_stop_v1" else None
            for _ in range(max_new_tokens):
                logits = self._forward(embeddings, last_only=True).logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite Qwen output")
                token = int(logits.argmax())
                if token == self.model.config.eos_token_id:
                    finish_reason = "eos"
                    break
                generated.append(token)
                # Ordinary turn delimiter, independent of any acquired value.
                if turn_stop:
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if turn_stop in decoded:
                        stopped_text = decoded.split(turn_stop, 1)[0]
                        finish_reason = "turn_boundary"
                        break
                next_id = torch.tensor([[token]], device=self.device)
                embeddings = torch.cat((embeddings, self._prefix(next_id)), dim=1)
        self.bridge.assert_frozen()
        self.assert_qwen_frozen()
        if before != self.memory.content_digest():
            raise RuntimeError("Query modified acquired memory")
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        result = {"text": (stopped_text if stopped_text is not None else raw).strip(),
                "mode": mode, "tokens": len(generated), "seconds": time.perf_counter() - started,
                "speaker_source_id": context.current_source_id,
                "knowledge_source_id": context.knowledge_source_id,
                "memory_applied": applied, "readout_norm": float(readout.norm()),
                "memory_digest": before, "bridge_digest": self.bridge.parameter_digest()}
        if turn_stop:
            result.update(raw_generated_text=raw, finish_reason=finish_reason, stop_sequence=turn_stop)
        if getattr(self.bridge, "payload_shape", None) == (17, 128):
            result["readout_confidence"] = float(readout[0, 0])
        return result
