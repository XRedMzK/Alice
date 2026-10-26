"""Bounded local generation with explicit sessions and an optional Bridge port."""

from dataclasses import dataclass
import hashlib
import json
import resource
import threading
import time
from uuid import uuid4

import torch

from .bridge import BridgeContext, EmbeddingBridgePort, MessageSpan
from .loading import LoadedQwen
from .sessions import Message, SessionStore
from .fact_intent import confirm_intent, normalize_fact_question, parse_memory_intent
from .generation import SingleTurnStoppingCriteria, cut_turn


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
    finish_reason: str | None = None
    stop_sequence: str | None = None


@dataclass(frozen=True)
class PendingMemoryFact:
    session_id: str
    source_id: str
    cue: str
    value: str
    object_id: str
    language: str
    session_incarnation: str
    confirmation_id: str

    @classmethod
    def create(cls, session_id, source_id, cue, value, object_id, language, *, session_incarnation):
        content = json.dumps(
            {"session_id": session_id, "source_id": source_id, "cue": cue,
             "value": value, "object_id": object_id, "language": language,
             "session_incarnation": session_incarnation},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        # The nonce distinguishes repeated proposals of even the identical fact.
        confirmation_id = f"{uuid4().hex}:{hashlib.sha256(content).hexdigest()}"
        return cls(session_id, source_id, cue, value, object_id, language,
                   session_incarnation, confirmation_id)


class AliceRuntime:
    """Frozen generation with disposable sessions and optional persistent memory.

    The legacy API profile keeps question-only input. An explicit conversation
    profile adds bounded active-session context without a persisted transcript
    channel or implicit learning. Neural slots retain their existing layout.
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
        self._memory_context = "question"
        self._closed = False
        self._pending_memory: PendingMemoryFact | None = None
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
                    "query_context": ("current_question_only_v1" if self._memory_context == "question"
                                      else "current_session_context_v1")}

    def set_memory_context(self, profile: str) -> None:
        """Select an explicit input profile without changing acquired state."""
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("Attach Neural Memory before choosing its context profile")
            self._neural_memory._assert_open()
            if profile not in {"question", "conversation"}:
                raise RuntimeInputError("Memory context must be question or conversation")
            self._memory_context = profile

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
            self._pending_memory = None
            return receipt

    def clear_pending_memory(self, session_id: str | None = None) -> None:
        """Forget only the transient confirmation proposal, never acquired M."""
        with self._lock:
            if self._pending_memory is not None and (
                    session_id is None or self._pending_memory.session_id == session_id):
                self._pending_memory = None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._require_open()
            self.store.clear_session(session_id)
            self.clear_pending_memory(session_id)

    def _dialogue_result(self, session_id, source_id, text, action, started,
                         *, receipt=None, confirmation_id=None):
        """Templated control turn, explicitly distinct from Qwen generation.

        Acquisition/confirmation turns never enter a transcript or model prompt.
        Only the pending proposal lives briefly in RAM; SessionStore cannot save it.
        """
        status = self.memory_status()
        info = {"dialogue_action": action, "control_turn": True,
                "mode": self._memory_mode, "memory_applied": False,
                "speaker_source_id": source_id, "knowledge_source_id": source_id,
                "memory_digest": status["memory_digest"],
                "bridge_digest": status["bridge_digest"],
                "query_context": "transient_memory_control_v1"}
        if receipt is not None:
            info["acquisition"] = receipt
        if confirmation_id is not None:
            info["confirmation_id"] = confirmation_id
        return GenerationResult(text, session_id, (source_id,), 0, 0,
                                time.perf_counter() - started, 0, 0, info)

    def _memory_dialogue(self, session_id, source_id, text, *, pending_at_entry,
                         confirmation_id=None):
        started = time.perf_counter()
        confirmation = confirm_intent(text)
        pending = self._pending_memory
        if confirmation is not None:
            if (pending is not None and pending.session_id == session_id
                    and pending.session_incarnation != self.store.session_incarnation(session_id)):
                # Retention may have evicted the former session with this ID.
                # A fresh session never inherits its proposal or consent token.
                self._pending_memory = None
                pending = None
            if pending is None or (pending.session_id, pending.source_id) != (session_id, source_id):
                language_ru = any("а" <= c.lower() <= "я" for c in text)
                reply = ("Нет ожидающего подтверждения для этого источника в этой сессии. "
                         "Повторите полное сообщение с просьбой запомнить."
                         if language_ru else
                         "There is no pending fact for this speaker in this session. Please repeat the full remember request.")
                return self._dialogue_result(session_id, source_id, reply, "no_pending", started)
            if ((confirmation_id is not None and confirmation_id != pending.confirmation_id)
                    or (confirmation_id is None and pending_at_entry is not pending)):
                reply = (f"Предыдущее подтверждение устарело. Запомнить: метка объекта "
                         f"{pending.object_id} — {pending.value}, источник {pending.source_id}? "
                         "Ответьте «да» или «нет». Память не менялась."
                         if pending.language == "ru" else
                         f"That confirmation is stale. Remember that object {pending.object_id} "
                         f"has label {pending.value}, from source {pending.source_id}? "
                         "Reply yes or no. Memory was not changed.")
                return self._dialogue_result(session_id, source_id, reply, "stale_confirmation", started,
                                             confirmation_id=pending.confirmation_id)
            if confirmation == "no":
                self._pending_memory = None
                reply = "Отменено. Память не менялась." if pending.language == "ru" else "Cancelled. Memory was not changed."
                return self._dialogue_result(session_id, source_id, reply, "cancelled", started)
            # Keep pending on a failed save so the same speaker may retry. The
            # existing remember path rolls M back and only acknowledges saved M.
            receipt = self.remember(pending.source_id, pending.cue, pending.value)
            reply = (f"Сохранено: метка объекта {pending.object_id} — {pending.value}. Источник: {pending.source_id}."
                     if pending.language == "ru" else
                     f"Saved: object {pending.object_id} has label {pending.value}. Source: {pending.source_id}.")
            return self._dialogue_result(session_id, source_id, reply, "remembered", started, receipt=receipt)

        # A different message supersedes any unanswered proposal. There is only
        # one small pending object, not a conversation/medium-term fact store.
        self._pending_memory = None
        intent = parse_memory_intent(text)
        if intent.kind == "no_intent":
            return None
        if intent.kind != "candidate":
            action = "clarification_required" if intent.kind == "clarify" else "unsupported"
            return self._dialogue_result(session_id, source_id, intent.message, action, started)
        if source_id not in self._neural_memory.memory.source_ids:
            reply = (f"Источник {source_id} пока не поддерживается этой памятью. Память не менялась."
                     if intent.language == "ru" else
                     f"Source {source_id} is not supported by this memory registry. Memory was not changed.")
            return self._dialogue_result(session_id, source_id, reply, "unsupported", started)
        from alice_memory.bridge import BridgeStateError
        from .neural_memory import MemoryRestartRequiredError
        try:
            self._neural_memory.validate_fact(source_id, intent.cue, intent.value)
        except (BridgeStateError, MemoryRestartRequiredError):
            raise  # A broken frozen Bridge is a system error, not a bad value.
        except ValueError:
            reply = ("Это знание пока не поддерживается: требуется однотокенное значение и короткий явный ID объекта. Память не менялась."
                     if intent.language == "ru" else
                     "This fact is not supported: the value must be one model token and the object ID short and explicit. Memory was not changed.")
            return self._dialogue_result(session_id, source_id, reply, "unsupported", started)
        self._pending_memory = PendingMemoryFact.create(session_id, source_id, intent.cue,
            intent.value, intent.object_id, intent.language,
            session_incarnation=self.store.session_incarnation(session_id))
        reply = (f"Запомнить: метка объекта {intent.object_id} — {intent.value}, источник {source_id}? Ответьте «да» или «нет»."
                 if intent.language == "ru" else
                 f"Remember that object {intent.object_id} has label {intent.value}, from source {source_id}? Reply yes or no.")
        return self._dialogue_result(session_id, source_id, reply, "confirmation_required", started,
                                     confirmation_id=self._pending_memory.confirmation_id)

    def save_memory(self) -> dict:
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("No Neural Memory is attached")
            return self._neural_memory.save()

    def backup_memory(self, destination) -> dict:
        """Back up the committed M and matching Bridge to a new bundle only."""
        with self._lock:
            self._require_open()
            if self._neural_memory is None:
                raise RuntimeInputError("Attach Neural Memory before creating its backup")
            self.assert_frozen()
            result = self._neural_memory.backup(destination)
            self.assert_frozen()
            return result

    def close(self) -> None:
        with self._lock:
            self._pending_memory = None
            if self._neural_memory is not None:
                self._neural_memory.close()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _prepare(self, session_id: str, messages: tuple[Message, ...], *, reserve_tokens: int = 0):
        # Encode each segment at most once, from the current bounded suffix.
        # Old messages beyond the first rejected segment are never tokenized.
        if type(reserve_tokens) is not int or not 0 <= reserve_tokens <= 64:
            raise RuntimeInputError("Invalid output reservation")
        limit = self.max_context_tokens - reserve_tokens
        participants = self.store.participants
        tail = self.tokenizer.encode("Agent [source_id=agent]:", add_special_tokens=False)
        used, selected = len(tail), []
        for message in reversed(messages):
            rendered = (
                f"{participants[message.source_id].label} "
                f"[source_id={message.source_id}]: {message.text}\n"
            )
            segment = self.tokenizer.encode(rendered, add_special_tokens=False)
            if not segment:
                raise RuntimeInputError("Tokenizer returned an empty message segment")
            if used + len(segment) > limit:
                if not selected:
                    raise RuntimeInputError("Current message plus reserved output exceeds the context token budget")
                break
            selected.append((message, segment))
            used += len(segment)
        if not selected:
            raise RuntimeInputError("A current message is required")
        selected.reverse()
        # A suffix must not begin with an assistant reply whose incoming turn
        # was dropped. Never remove the final/current message, including an
        # explicit Agent-authored request. Do not refill from older history.
        if len(selected) < len(messages):
            while len(selected) > 1 and selected[0][0].source_id == "agent":
                selected.pop(0)
        ids, spans = [], []
        for message, segment in selected:
            start = len(ids)
            ids.extend(segment)
            spans.append(MessageSpan(message.source_id, start, len(ids)))
        ids.extend(tail)
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        context = BridgeContext(
            session_id,
            tuple(message.source_id for message, _segment in selected),
            current_source_id=selected[-1][0].source_id,
            knowledge_source_id=None,
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
                knowledge_source_id: str | None = None,
                confirmation_id: str | None = None) -> GenerationResult:
        # Capture before waiting for another response: a plain legacy yes must
        # never approve a different proposal created while it waited for the lock.
        # Concurrent/UI clients can bind the exact displayed proposal explicitly.
        pending_at_entry = self._pending_memory
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise RuntimeInputError("Stage 1 generation budget must be 1..64 tokens")
        with self._lock:
            self._require_open()
            incoming = self.store.validate_message(source_id, text)
            history = self.store.messages(session_id)
            if self._neural_memory is not None:
                self._neural_memory._assert_open()
                if self.bridge.attached:
                    raise RuntimeInputError("External embedding hook and Neural Memory cannot be combined")
                if self._memory_context == "question" and max_new_tokens > 16:
                    raise RuntimeInputError("Neural Memory generation budget must be 1..16 tokens")
                control = self._memory_dialogue(session_id, source_id, text,
                                                pending_at_entry=pending_at_entry,
                                                confirmation_id=confirmation_id)
                if control is not None:
                    return control
                target = source_id if knowledge_source_id is None else knowledge_source_id
                self.store.validate_message(target, text)
                self.assert_frozen()
                from alice_memory.runtime import Question
                adapted_question = normalize_fact_question(text)
                request = Question(adapted_question or text, source_id, target)
                if self._memory_context == "conversation":
                    participants = self.store.participants

                    def rendered(message):
                        return (f"{participants[message.source_id].label} "
                                f"[source_id={message.source_id}]: {message.text}\n")

                    def history_segments():
                        for message in reversed(history):
                            yield message.source_id, tuple(self.tokenizer.encode(
                                rendered(message), add_special_tokens=False))

                    current_ids = tuple(self.tokenizer.encode(
                        rendered(incoming), add_special_tokens=False)) + tuple(self.tokenizer.encode(
                        "Agent [source_id=agent]:", add_special_tokens=False))
                    answer = self._neural_memory.respond_with_context(
                        request, history_segments(), current_ids,
                        history_message_count=len(history), mode=self._memory_mode,
                        max_new_tokens=max_new_tokens)
                else:
                    answer = self._neural_memory.respond(
                        request, mode=self._memory_mode, max_new_tokens=max_new_tokens)
                self.assert_frozen()
                self.store.append_result(session_id, incoming, answer["text"])
                info = {k: v for k, v in answer.items() if k != "text"}
                info["query_context"] = ("current_question_only_v1" if self._memory_context == "question"
                                         else "current_session_context_v1")
                info["prompt_assembly_version"] = answer.get(
                    "prompt_assembly_version", "repeat_qa_turn_stop_v1")
                info["dialogue_query_adapter"] = "bounded_object_label_ru_v1" if adapted_question else None
                input_count = answer.get("input_tokens")
                if input_count is None:
                    input_count = len(self.tokenizer.encode(
                        f"Question: {adapted_question or text}\nAnswer:", add_special_tokens=False))
                if self._memory_context == "question" and answer["memory_applied"]:
                    input_count += 1 + len(self.tokenizer.encode(
                        f"\nQuestion: {adapted_question or text}\nAnswer:", add_special_tokens=False))
                return GenerationResult(
                    answer["text"], session_id, tuple(answer.get("context_source_ids", (source_id,))),
                    input_count,
                    answer["tokens"], answer["seconds"], int(answer["memory_applied"]),
                    answer.get("context_messages_dropped", len(history)), info,
                    finish_reason=answer.get("finish_reason"), stop_sequence=answer.get("stop_sequence"),
                )
            if knowledge_source_id is not None:
                raise RuntimeInputError("knowledge_source_id requires attached Neural Memory")
            inputs, context, dropped = self._prepare(
                session_id, history + (incoming,), reserve_tokens=max_new_tokens
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
                        stopping_criteria=[SingleTurnStoppingCriteria(self.tokenizer, inputs.shape[1])],
                    )
                if inputs.device.type == "cuda":
                    torch.cuda.synchronize(inputs.device)
                generated = output[0, inputs.shape[1]:].tolist()
                suffix = self.tokenizer.decode(generated, skip_special_tokens=True)
                boundary = cut_turn(suffix, allow_partial=len(generated) >= max_new_tokens)
                response = boundary.text.strip()
                self.assert_frozen()
                self.store.append_result(session_id, incoming, response)
                return GenerationResult(
                    response, session_id, context.source_ids, inputs.shape[1], len(generated),
                    time.perf_counter() - started, self.bridge.calls - calls, dropped,
                    finish_reason=boundary.finish_reason or (
                        "token_limit" if len(generated) >= max_new_tokens else "eos"),
                    stop_sequence=boundary.stop_sequence,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("Insufficient VRAM; reduce explicit context/output budget") from exc

    def resources(self) -> dict:
        device = self.model.get_input_embeddings().weight.device
        result = {"process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        result["session_retention"] = self.store.retention_status()
        if device.type == "cuda":
            result.update(
                cuda_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                cuda_reserved_mib=torch.cuda.memory_reserved(device) / 2**20,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            )
        return result
