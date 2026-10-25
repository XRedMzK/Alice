"""Explicit schema dispatch for independently replaceable embedding Bridges."""

import json
from pathlib import Path

from .bridge import BridgeStateError, ParametricBridge


def restore_bridge(directory, **kwargs):
    try:
        manifest = json.loads((Path(directory) / 'bridge.json').read_text())
    except (OSError, ValueError) as exc:
        raise BridgeStateError(f'Cannot read Bridge manifest: {exc}') from exc
    if not isinstance(manifest, dict):
        raise BridgeStateError('Bridge manifest must be an object')
    schema = manifest.get('schema')
    if schema == ParametricBridge.SCHEMA:
        return ParametricBridge.load(directory, **kwargs)
    from .lexical_bridge import LexicalEmbeddingBridge
    if schema == LexicalEmbeddingBridge.SCHEMA:
        return LexicalEmbeddingBridge.load(directory, **kwargs)
    from .payload_bridge import PayloadEmbeddingBridge
    if schema == PayloadEmbeddingBridge.SCHEMA:
        return PayloadEmbeddingBridge.load(directory, **kwargs)
    raise BridgeStateError(f'Unknown embedding Bridge schema: {schema}')


def memory_class_for(bridge):
    from .payload_bridge import PayloadEmbeddingBridge
    if bridge.SCHEMA == PayloadEmbeddingBridge.SCHEMA:
        from .payload_matrix import ParametricPayloadMemory
        return ParametricPayloadMemory
    from .matrix import AssociativeMemory
    return AssociativeMemory


def runtime_class_for(bridge):
    from .payload_bridge import PayloadEmbeddingBridge
    if bridge.SCHEMA == PayloadEmbeddingBridge.SCHEMA:
        from .payload_runtime import PayloadMemoryRuntime
        return PayloadMemoryRuntime
    from .runtime import MemoryRuntime
    return MemoryRuntime
