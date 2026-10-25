"""Replaceable inference connection at Qwen's input-embedding boundary."""

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Callable, Iterator

import torch


class BridgeError(ValueError):
    """A candidate Bridge does not satisfy the active tensor boundary."""


@dataclass(frozen=True)
class MessageSpan:
    """Half-open token interval for one complete rendered source message."""

    source_id: str
    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if (not isinstance(self.source_id, str) or not self.source_id
                or type(self.start_token) is not int or type(self.end_token) is not int
                or not 0 <= self.start_token < self.end_token):
            raise BridgeError("Message span requires a source and a nonempty token interval")


@dataclass(frozen=True)
class BridgeContext:
    session_id: str
    source_ids: tuple[str, ...]
    current_source_id: str | None = None
    knowledge_source_id: str | None = None
    message_spans: tuple[MessageSpan, ...] = ()


class EmbeddingBridgePort:
    """A wiring boundary, not a selected or trained Memory Bridge.

    A callable receives an independent embedding copy and explicit provenance.
    It returns the same shape/device/dtype. No conversion or silent fallback is
    allowed. The hook exists only during the protected inference invocation.
    """

    def __init__(self) -> None:
        self._bridge: Callable | None = None
        self._enabled = False
        self._lock = threading.RLock()
        self.calls = 0

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._bridge is not None

    def attach(self, bridge: Callable) -> None:
        if not callable(bridge):
            raise BridgeError("Bridge must be callable(embeddings, context)")
        with self._lock:
            self._bridge = bridge
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    def enable(self) -> None:
        with self._lock:
            if self._bridge is None:
                raise BridgeError("No Bridge is attached")
            self._enabled = True

    def detach(self) -> None:
        with self._lock:
            self._bridge = None
            self._enabled = False

    @contextmanager
    def activation(self, embedding_module, context: BridgeContext) -> Iterator[None]:
        with self._lock:
            handle = None
            if self._enabled and self._bridge is not None:
                def condition(_module, _args, original):
                    candidate = self._bridge(original.detach().clone(), context)
                    if not isinstance(candidate, torch.Tensor):
                        raise BridgeError("Bridge must return a tensor")
                    if candidate.shape != original.shape:
                        raise BridgeError(f"Bridge shape {candidate.shape} != {original.shape}")
                    if candidate.dtype != original.dtype:
                        raise BridgeError(f"Bridge dtype {candidate.dtype} != {original.dtype}")
                    if candidate.device != original.device:
                        raise BridgeError(f"Bridge device {candidate.device} != {original.device}")
                    if not candidate.is_contiguous() or not torch.isfinite(candidate).all():
                        raise BridgeError("Bridge tensor must be contiguous and finite")
                    self.calls += 1
                    return candidate

                handle = embedding_module.register_forward_hook(condition)
            try:
                yield
            finally:
                if handle is not None:
                    handle.remove()
