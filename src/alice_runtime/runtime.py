"""Bounded local generation with explicit sessions and an optional Bridge port."""

from dataclasses import dataclass
import resource
import threading
import time

import torch

from .bridge import BridgeContext, EmbeddingBridgePort, MessageSpan
from .loading import LoadedQwen
from .sessions import Message, SessionStore


class RuntimeInputError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    session_id: str
    source_ids: tuple[str, ...]
    input_tokens: int
    generated_tokens: int
    wall_seconds: float
    bridge_calls: int
    context_messages_dropped: int
    memory_info: dict | None = None


class AliceRuntime:
    """Frozen generation with disposable sessions and optional persistent memory.

    Memory mode reuses the proven question-only neural input layout. Transcripts
    remain UI history and are never replayed into that path or learned implicitly.
    Explicit remember() changes only separately persisted memory parameters.
    """

    PROMPT_ASSEMBLY_VERSION = "message-segments-v1"

    def __init__(self, loaded: LoadedQwen, store: SessionStore | None = None,
                 max_context_tokens: int = 256) -> None:
        if type(max_context_tokens) is not int or not 8 <= max_context_tokens <= 1024:
            raise RuntimeInputError("Stage 1 context budget must be 8..1024 tokens")
        self.model, self.tokenizer = loaded.model, loaded.tokenizer
        self._loaded = loaded
        self.metadata = loaded.metadata
        self.store = store if store is not None else SessionStore()
        self.bridge = EmbeddingBridgePort()
        self.max_context_tokens = max_context_tokens
        self._lock = threading.RLock()
        self._neural_memory = None
        self._memory_mode = "on"
        self._closed = False
        self._versions = {name: p._version for name, p in self.model.named_parameters()}
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Qwen must remain in frozen evaluation mode")
        if {name: p._version for name, p in self.model.named_parameters()} != self._versions:
            raise RuntimeError("Qwen parameter versions changed during inference")

    @property
    def neural_memory(self):
        return self._neural_memory

    def _require_open(self):
        if self._closed:
            raise RuntimeInputError("Alice runtime is closed")

    def attach_memory(self, bridge_path, state_path, model_path=None) -> dict:
        """Attach the accepted payload implementation to this same Qwen instance."""
        from .neural_memory import NeuralMemorySession

        with self._lock:
            self._require_open()
            if self._neural_memory is not None or self.bridge.attached:
                raise RuntimeInputError("Detach external wiring or start a new runtime before attaching memory")
            self.assert_frozen()
            candidate = NeuralMemorySession(self._loaded, bridge_path, state_path,
                                           model_path=model_path,
                                           max_tokens=min(self.max_context_tokens, 128))
            try:
                for source in candidate.memory.source_ids:
                    if source.startswith("person:"):
                        self.store.add_person(int(source.split(":", 1)[1]))
                self._neural_memory = candidate
                return self.memory_status()
            except BaseException:
                candidate.close()
                self._neural_memory = None
                raise

    def memory_status(self) -> dict:
        with self._lock:
            if self._neural_memory is None:
                return {"attached": False, "mode": "off", "source_ids": []}
            return {**self._neural_memory.status(), "attached": True,
                    "mode": self._memory_mode,
                    "query_context": "current_question_only_v1"}

    def set_memory_mode(self, mode: str) -> None:
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("No Neural Memory is attached")
            if mode not in {"on", "off", "empty"}:
                raise RuntimeInputError("Memory mode must be on, off or empty")
            self._memory_mode = mode

    def remember(self, source_id: str, cue: str, value: str) -> dict:
        """Explicit structured acquisition; success means the write is saved.

        OFF/EMPTY affect reads. This explicit command still writes, and never
        appends a teaching record to any session transcript.
        """
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("Attach Neural Memory before using remember")
            self.store.validate_message(source_id, cue)
            self.store.validate_message(source_id, value)
            self.assert_frozen()
            receipt = self._neural_memory.remember(source_id, cue, value)
            self.assert_frozen()
            return receipt

    def save_memory(self) -> dict:
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("No Neural Memory is attached")
            return self._neural_memory.save()

    def close(self) -> None:
        with self._lock:
            if self._neural_memory is not None:
                self._neural_memory.close()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _prepare(self, session_id: str, messages: tuple[Message, ...]):
        # Each complete message owns its token segment, including its trailing
        # newline. Rendered prompt bytes match Stage 1, but preventing BPE merges
        # across source boundaries can change token IDs at those boundaries.
        # This versioned assembly supplies exact, non-overlapping source spans;
        # it does not claim logit equivalence with Stage 1's whole-string encode.
        selected = list(messages)
        participants = self.store.participants
        while True:
            ids, spans = [], []
            for message in selected:
                rendered = (
                    f"{participants[message.source_id].label} "
                    f"[source_id={message.source_id}]: {message.text}\n"
                )
                segment = self.tokenizer.encode(rendered, add_special_tokens=False)
                if not segment:
                    raise RuntimeInputError("Tokenizer returned an empty message segment")
                start = len(ids)
                ids.extend(segment)
                spans.append(MessageSpan(message.source_id, start, len(ids)))
            ids.extend(self.tokenizer.encode("Agent [source_id=agent]:", add_special_tokens=False))
            if len(ids) <= self.max_context_tokens:
                break
            if len(selected) <= 1:
                raise RuntimeInputError("Current message exceeds the context token budget")
            selected.pop(0)  # Drop whole old messages; never cut a source header.
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        context = BridgeContext(
            session_id,
            tuple(message.source_id for message in selected),
            current_source_id=selected[-1].source_id if selected else None,
            knowledge_source_id=None,  # The current speaker is not the queried knowledge source.
            message_spans=tuple(spans),
        )
        return inputs, context, len(messages) - len(selected)

    def _last_logits(self, inputs, context):
        self.assert_frozen()
        with torch.inference_mode(), self.bridge.activation(self.model.get_input_embeddings(), context):
            output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1, :].detach().float().cpu().clone()
        self.assert_frozen()
        return logits

    def probe_logits(self, session_id: str, source_id: str, text: str):
        """Technical Bridge probe without committing a session message."""
        with self._lock:
            self._require_open()
            if self._neural_memory is not None:
                raise RuntimeInputError("Stage 1 probe_logits is defined without Neural Memory; use respond")
            incoming = self.store.validate_message(source_id, text)
            inputs, context, _ = self._prepare(
                session_id, self.store.messages(session_id) + (incoming,)
            )
            return self._last_logits(inputs, context)

    def respond(self, session_id: str, source_id: str, text: str,
                max_new_tokens: int = 16, *,
                knowledge_source_id: str | None = None) -> GenerationResult:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise RuntimeInputError("Stage 1 generation budget must be 1..64 tokens")
        with self._lock:
            self._require_open()
            incoming = self.store.validate_message(source_id, text)
            history = self.store.messages(session_id)
            if self._neural_memory is not None:
                if self.bridge.attached:
                    raise RuntimeInputError("External embedding hook and Neural Memory cannot be combined")
                if max_new_tokens > 16:
                    raise RuntimeInputError("Neural Memory generation budget must be 1..16 tokens")
                target = source_id if knowledge_source_id is None else knowledge_source_id
                self.store.validate_message(target, text)
                self.assert_frozen()
                from alice_memory.runtime import Question
                answer = self._neural_memory.respond(
                    Question(text, source_id, target), mode=self._memory_mode,
                    max_new_tokens=max_new_tokens)
                self.assert_frozen()
                self.store.append_result(session_id, incoming, answer["text"])
                info = {k: v for k, v in answer.items() if k != "text"}
                info["query_context"] = "current_question_only_v1"
                info["prompt_assembly_version"] = "repeat_qa_turn_stop_v1"
                input_count = len(self.tokenizer.encode(
                    f"Question: {text}\nAnswer:", add_special_tokens=False))
                if answer["memory_applied"]:
                    input_count += 1 + len(self.tokenizer.encode(
                        f"\nQuestion: {text}\nAnswer:", add_special_tokens=False))
                return GenerationResult(
                    answer["text"], session_id, (source_id,),
                    input_count,
                    answer["tokens"], answer["seconds"], int(answer["memory_applied"]),
                    len(history), info,
                )
            if knowledge_source_id is not None:
                raise RuntimeInputError("knowledge_source_id requires attached Neural Memory")
            inputs, context, dropped = self._prepare(
                session_id, history + (incoming,)
            )
            self.assert_frozen()
            started, calls = time.perf_counter(), self.bridge.calls
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.model.config.eos_token_id
            try:
                with torch.inference_mode(), self.bridge.activation(self.model.get_input_embeddings(), context):
                    output = self.model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
                        eos_token_id=self.model.config.eos_token_id,
                        pad_token_id=pad,
                    )
                if inputs.device.type == "cuda":
                    torch.cuda.synchronize(inputs.device)
                generated = output[0, inputs.shape[1]:].tolist()
                response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                self.assert_frozen()
                self.store.append_result(session_id, incoming, response)
                return GenerationResult(
                    response, session_id, context.source_ids, inputs.shape[1], len(generated),
                    time.perf_counter() - started, self.bridge.calls - calls, dropped,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("Insufficient VRAM; reduce explicit context/output budget") from exc

    def resources(self) -> dict:
        device = self.model.get_input_embeddings().weight.device
        result = {"process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        if device.type == "cuda":
            result.update(
                cuda_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                cuda_reserved_mib=torch.cuda.memory_reserved(device) / 2**20,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            )
        return result
