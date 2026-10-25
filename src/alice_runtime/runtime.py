"""Bounded local generation with explicit sessions and an optional Bridge port."""

from dataclasses import dataclass
import resource
import threading
import time

import torch

from .bridge import BridgeContext, EmbeddingBridgePort
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


class AliceRuntime:
    """Synchronous inference only; session transcripts are not learned memory."""

    def __init__(self, loaded: LoadedQwen, store: SessionStore | None = None,
                 max_context_tokens: int = 256) -> None:
        if type(max_context_tokens) is not int or not 8 <= max_context_tokens <= 1024:
            raise RuntimeInputError("Stage 1 context budget must be 8..1024 tokens")
        self.model, self.tokenizer = loaded.model, loaded.tokenizer
        self.metadata = loaded.metadata
        self.store = store if store is not None else SessionStore()
        self.bridge = EmbeddingBridgePort()
        self.max_context_tokens = max_context_tokens
        self._lock = threading.RLock()
        self._versions = {name: p._version for name, p in self.model.named_parameters()}
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Qwen must remain in frozen evaluation mode")
        if {name: p._version for name, p in self.model.named_parameters()} != self._versions:
            raise RuntimeError("Qwen parameter versions changed during inference")

    def _prepare(self, session_id: str, messages: tuple[Message, ...]):
        selected = list(messages)
        while True:
            prompt = "\n".join(
                f"{self.store.participants[m.source_id].label} [source_id={m.source_id}]: {m.text}"
                for m in selected
            ) + "\nAgent [source_id=agent]:"
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) <= self.max_context_tokens:
                break
            if len(selected) <= 1:
                raise RuntimeInputError("Current message exceeds the context token budget")
            selected.pop(0)  # Drop whole old messages; never cut a source header.
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        context = BridgeContext(session_id, tuple(m.source_id for m in selected))
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
            incoming = self.store.validate_message(source_id, text)
            inputs, context, _ = self._prepare(
                session_id, self.store.messages(session_id) + (incoming,)
            )
            return self._last_logits(inputs, context)

    def respond(self, session_id: str, source_id: str, text: str,
                max_new_tokens: int = 16) -> GenerationResult:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise RuntimeInputError("Stage 1 generation budget must be 1..64 tokens")
        with self._lock:
            incoming = self.store.validate_message(source_id, text)
            inputs, context, dropped = self._prepare(
                session_id, self.store.messages(session_id) + (incoming,)
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
