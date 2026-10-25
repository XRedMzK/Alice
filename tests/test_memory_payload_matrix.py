"""Parallel value payload correctness and independent persistence; no Qwen."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alice_memory.matrix import AssociativeMemory, DIM, MemoryStateError
from alice_memory.payload_matrix import (
    PAYLOAD_BASIS_ID, PAYLOAD_HEADS, ParametricPayloadMemory, functional_payload_delta,
)


def key(index):
    result = torch.zeros(DIM)
    result[index] = 1
    return result


def values(offset=0.0):
    return torch.arange(PAYLOAD_HEADS * DIM, dtype=torch.float32).reshape(PAYLOAD_HEADS, DIM) / 3000 + offset


def test_parallel_heads_correction_source_isolation_and_pure_read():
    memory = ParametricPayloadMemory()
    a, corrected, other = values(), values(-0.7), values(0.4)
    receipt = memory.acquire("user", key(0).requires_grad_(), a.requires_grad_())
    assert receipt["before"] != receipt["after"]
    assert isinstance(memory.matrix, torch.nn.Parameter) and not memory.matrix.requires_grad
    assert memory.matrix.shape == (4, PAYLOAD_HEADS, DIM, DIM)
    torch.testing.assert_close(memory.read("user", key(0)), a)
    torch.testing.assert_close(memory.read("person:1", key(0)), torch.zeros_like(a))
    memory.acquire("person:1", key(0), other)
    memory.acquire("user", key(1), other)
    memory.acquire("user", key(0), corrected)
    before_read = memory.content_digest()
    torch.testing.assert_close(memory.read("user", key(0)), corrected)
    torch.testing.assert_close(memory.read("person:1", key(0)), other)
    torch.testing.assert_close(memory.read("user", key(1)), other)
    assert memory.content_digest() == before_read
    assert memory.matrix.grad is None and a.grad is None


def test_invalid_write_rolls_back_all_heads():
    memory = ParametricPayloadMemory()
    memory.acquire("user", key(0), values())
    before = memory.content_digest()
    bad_value = values()
    bad_value[-1, -1] = float("nan")
    for source, query, value in [
        ("missing", key(0), values()),
        ("user", torch.zeros(DIM), values()),
        ("user", key(0).double(), values()),
        ("user", key(0), values().double()),
        ("user", key(0), values()[:-1]),
        ("user", key(0), bad_value),
    ]:
        with pytest.raises(MemoryStateError):
            memory.acquire(source, query, value)
        assert memory.content_digest() == before
    memory.matrix.requires_grad_(True)
    with pytest.raises(MemoryStateError):
        memory.acquire("user", key(0), values())
    with pytest.raises(MemoryStateError):
        memory.read("user", key(0))
    assert memory.content_digest() == before


def test_functional_delta_preserves_gradients_without_mutating_source():
    matrix = torch.zeros(1, PAYLOAD_HEADS, DIM, DIM, requires_grad=True)
    query = torch.linspace(-1, 1, DIM, requires_grad=True)
    payload = values().requires_grad_()
    updated = functional_payload_delta(matrix, query, payload, 0)
    assert torch.count_nonzero(matrix) == 0
    (updated[0] @ torch.linspace(0.2, -0.5, DIM)).square().sum().backward()
    for tensor in (matrix, query, payload):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_independent_roundtrip_fresh_process_and_basis_integrity(tmp_path):
    memory = ParametricPayloadMemory()
    memory.acquire("person:2", key(3), values())
    metadata = memory.save(tmp_path)
    assert metadata["schema"] == "alice.memory_payload/v1"
    restored = ParametricPayloadMemory.load(tmp_path)
    assert restored.content_digest() == memory.content_digest()
    assert restored.source_ids == memory.source_ids
    with pytest.raises(MemoryStateError, match="basis"):
        ParametricPayloadMemory.load(tmp_path, latent_basis_id="incompatible")
    with pytest.raises(MemoryStateError):
        AssociativeMemory.load(tmp_path)
    script = """
import json,sys,torch
from alice_memory.payload_matrix import ParametricPayloadMemory
m=ParametricPayloadMemory.load(sys.argv[1]);q=torch.zeros(128);q[3]=1
print(json.dumps({'digest':m.content_digest(),'shape':list(m.read('person:2',q).shape)}))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path)],
                            check=True, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    assert json.loads(result.stdout) == {"digest": memory.content_digest(), "shape": [17, 128]}
    before = memory.content_digest()
    with (tmp_path / metadata["tensor_file"]).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(MemoryStateError, match="integrity"):
        ParametricPayloadMemory.load(tmp_path)
    assert memory.content_digest() == before


def test_payload_source_and_basis_guards_reuse_original_contract():
    with pytest.raises(MemoryStateError):
        ParametricPayloadMemory(source_ids=("user", "user"))
    with pytest.raises(MemoryStateError):
        ParametricPayloadMemory(source_ids=("unnamed",))
    with pytest.raises(MemoryStateError):
        ParametricPayloadMemory(latent_basis_id="")
    memory = ParametricPayloadMemory()
    assert memory.latent_basis_id == PAYLOAD_BASIS_ID


def test_metadata_integrity_binds_source_order_and_save_rejects_invalid_state(tmp_path):
    memory = ParametricPayloadMemory()
    memory.acquire("user", key(0), values())
    metadata = memory.save(tmp_path)
    manifest_path = tmp_path / "memory.json"
    original = manifest_path.read_bytes()
    assert len(metadata["metadata_sha256"]) == 64
    metadata["source_ids"][0], metadata["source_ids"][1] = metadata["source_ids"][1], metadata["source_ids"][0]
    manifest_path.write_text(json.dumps(metadata))
    tampered = manifest_path.read_bytes()
    files_before = set(tmp_path.iterdir())
    with pytest.raises(MemoryStateError, match="metadata integrity"):
        ParametricPayloadMemory.load(tmp_path)
    with pytest.raises(MemoryStateError, match="metadata integrity"):
        memory.save(tmp_path)
    assert manifest_path.read_bytes() == tampered and set(tmp_path.iterdir()) == files_before
    manifest_path.write_bytes(original)
    memory.save(tmp_path)  # A compatible working checkpoint can be advanced.
    assert ParametricPayloadMemory.load(tmp_path).content_digest() == memory.content_digest()


def test_save_does_not_overwrite_other_schema_basis_or_valid_registry(tmp_path):
    old = AssociativeMemory()
    old.acquire("user", key(0), key(1))
    old.save(tmp_path / "old")
    payload = ParametricPayloadMemory()
    old_manifest = (tmp_path / "old" / "memory.json").read_bytes()
    with pytest.raises(MemoryStateError, match="schema/basis"):
        payload.save(tmp_path / "old")
    assert (tmp_path / "old" / "memory.json").read_bytes() == old_manifest
    assert AssociativeMemory.load(tmp_path / "old").content_digest() == old.content_digest()
    different = ParametricPayloadMemory(latent_basis_id="another-compatible-shape")
    different.save(tmp_path / "basis")
    basis_manifest = (tmp_path / "basis" / "memory.json").read_bytes()
    with pytest.raises(MemoryStateError, match="basis"):
        payload.save(tmp_path / "basis")
    assert (tmp_path / "basis" / "memory.json").read_bytes() == basis_manifest
    registry = ParametricPayloadMemory(source_ids=("user", "agent", "person:1", "person:2"))
    registry.save(tmp_path / "registry")
    registry_manifest = (tmp_path / "registry" / "memory.json").read_bytes()
    with pytest.raises(MemoryStateError, match="source registry"):
        payload.save(tmp_path / "registry")
    assert (tmp_path / "registry" / "memory.json").read_bytes() == registry_manifest
