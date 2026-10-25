"""Strict single-token acquisition boundary for the full-rank payload version."""

from .runtime import MemoryRuntime


class PayloadMemoryRuntime(MemoryRuntime):
    def __init__(self, loaded, bridge, memory, max_tokens=128):
        super().__init__(loaded, bridge, memory, max_tokens)
        if (getattr(bridge, "payload_shape", None) != (17, 128)
                or memory.matrix.shape != (len(memory.source_ids), 17, 128, 128)):
            raise ValueError("Payload runtime requires the versioned 17-head memory boundary")

    def _value_feature(self, text):
        ids = self._ids(" " + text)
        if ids.shape[1] != 1:
            raise ValueError("Payload v1 supports one-token values; multi-token values are not silently averaged")
        return self._prefix(ids).float()[0, 0]
