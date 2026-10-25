"""Small state/gradient checks; no language model or factual benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alice_memory.matrix import (
    AssociativeMemory, DEFAULT_SOURCES, DIM, LATENT_BASIS_ID,
    MemoryStateError, functional_delta,
)


def basis(index):
    value = torch.zeros(DIM)
    value[index] = 1
    return value


def test_acquire_correction_source_isolation_and_rollback():
    memory = AssociativeMemory()
    key, second_key = basis(0), basis(1)
    first, corrected, other = basis(2), basis(3), basis(4)
    assert tuple(memory.matrix.shape) == (len(DEFAULT_SOURCES), DIM, DIM)
    assert not memory.matrix.requires_grad
    empty = memory.content_digest()
    receipt = memory.acquire("user", key.requires_grad_(), first.requires_grad_())
    assert receipt["before"] == empty and receipt["after"] != empty
    torch.testing.assert_close(memory.read("user", key), first)
    torch.testing.assert_close(memory.read("person:1", key), torch.zeros(DIM))
    memory.acquire("person:1", key, other)
    memory.acquire("user", second_key, other)
    memory.acquire("user", key, corrected)
    torch.testing.assert_close(memory.read("user", key), corrected)
    torch.testing.assert_close(memory.read("person:1", key), other)
    torch.testing.assert_close(memory.read("user", second_key), other)
    assert memory.matrix.grad is None and key.grad is None and first.grad is None
    snapshot = memory.matrix.detach().clone()
    for source, candidate_key, value in [
        ("missing", key, first),
        ("user", torch.zeros(DIM), first),
        ("user", key, torch.full((DIM,), float("nan"))),
        ("user", key.double(), first),
    ]:
        with pytest.raises(MemoryStateError):
            memory.acquire(source, candidate_key, value)
        assert torch.equal(memory.matrix, snapshot)


def test_functional_write_retains_key_value_and_prior_memory_gradients():
    matrix = torch.zeros(2, DIM, DIM, requires_grad=True)
    key = torch.linspace(-1, 1, DIM, requires_grad=True)
    value = torch.linspace(0.1, 0.9, DIM, requires_grad=True)
    query = torch.linspace(0.4, -0.8, DIM)
    updated = functional_delta(matrix, key, value, 1)
    assert not torch.equal(updated, matrix)
    assert torch.count_nonzero(matrix) == 0  # No persistent mutation.
    loss = (updated[1] @ query).square().sum()
    loss.backward()
    for tensor in (matrix, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_independent_restore_fresh_process_and_state_swap(tmp_path):
    a, b = AssociativeMemory(), AssociativeMemory()
    a.acquire("person:1", basis(0), basis(2))
    b.acquire("person:1", basis(0), basis(5))
    a.save(tmp_path / "a")
    b.save(tmp_path / "b")
    restored_a = AssociativeMemory.load(tmp_path / "a")
    restored_b = AssociativeMemory.load(tmp_path / "b")
    assert a.content_digest() == restored_a.content_digest()
    assert b.content_digest() == restored_b.content_digest()
    assert not torch.equal(restored_a.read("person:1", basis(0)),
                           restored_b.read("person:1", basis(0)))
    metadata = json.loads((tmp_path / "a" / "memory.json").read_text())
    assert set(metadata) == {"schema", "update_rule", "latent_basis_id", "source_ids",
                             "shape", "dtype", "snapshot_id", "tensor_file", "sha256"}
    script = """
import json, sys, torch
from alice_memory.matrix import AssociativeMemory, DIM
m = AssociativeMemory.load(sys.argv[1]); k = torch.zeros(DIM); k[0] = 1
r = m.read('person:1', k)
print(json.dumps({'hash': m.content_digest(), 'argmax': int(r.argmax())}))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path / "a")],
                            check=True, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    assert json.loads(result.stdout) == {"hash": a.content_digest(), "argmax": 2}


def test_restore_rejects_incompatible_or_corrupt_state_without_reset(tmp_path):
    memory = AssociativeMemory()
    memory.acquire("user", basis(0), basis(1))
    before = memory.content_digest()
    metadata = memory.save(tmp_path)
    with pytest.raises(MemoryStateError, match="basis"):
        AssociativeMemory.load(tmp_path, latent_basis_id="another-basis")
    payload = tmp_path / metadata["tensor_file"]
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(MemoryStateError, match="integrity"):
        AssociativeMemory.load(tmp_path)
    assert memory.content_digest() == before
    with pytest.raises(MemoryStateError):
        AssociativeMemory(source_ids=("user", "user"))
