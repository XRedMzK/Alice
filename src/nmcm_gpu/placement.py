"""Explicit HOT (CUDA) / WARM (host RAM) placement primitives.

The accepted CPU implementation intentionally has no placement policy.  This
module layers GPU placement on top of :class:`nmcm.neural.NeuralModule` without
changing that implementation or hiding its explicit recurrent state.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from nmcm.neural import ModuleOutput, NeuralModule, TensorMap


CANONICAL_DIGEST_ALGORITHM = "sha256/canonical-cpu-contiguous-v1"


def _normalise_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA placement requested but CUDA is unavailable")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {index}")
        return torch.device("cuda", index)
    if resolved.type != "cpu":
        raise ValueError("FOUNDATION_0 placement supports only HOT CUDA and WARM CPU")
    return torch.device("cpu")


def _module_owned_tensors(module: nn.Module) -> tuple[Tensor, ...]:
    return tuple(module.parameters(recurse=True)) + tuple(module.buffers(recurse=True))


def _all_owned_tensors(
    module: nn.Module, state: Mapping[str, Tensor]
) -> Iterable[Tensor]:
    yield from _module_owned_tensors(module)
    yield from state.values()


def placement_device(module: nn.Module, state: Mapping[str, Tensor]) -> torch.device:
    """Return the single device owning module tensors and explicit state.

    Mixed placement is rejected.  It is too easy for it to turn an intended
    module-level transfer into an implicit per-operation copy.
    """

    devices = {tensor.device for tensor in _all_owned_tensors(module, state)}
    if not devices:
        raise ValueError("cannot infer placement for a module with no tensors or state")
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise ValueError(f"module and explicit state have mixed placement: {rendered}")
    return _normalise_device(devices.pop())


def _update_tensor_digest(digest: Any, name: str, tensor: Tensor) -> None:
    if tensor.layout != torch.strided:
        raise TypeError("canonical FOUNDATION_0 digests require strided tensors")
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes(canonical.reshape(-1).view(torch.uint8).tolist()))
    digest.update(b"\0")


def tensor_digest(tensor: Tensor, *, name: str = "tensor") -> str:
    digest = hashlib.sha256()
    _update_tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def tensor_map_digest(tensors: Mapping[str, Tensor]) -> str:
    """Hash values independent of their current CPU/CUDA placement."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        _update_tensor_digest(digest, name, tensors[name])
    return digest.hexdigest()


def module_digest(module: nn.Module) -> str:
    """Hash module identity and every parameter/buffer by canonical value."""

    digest = hashlib.sha256()
    identity = (
        f"{type(module).__module__}.{type(module).__qualname__}|"
        f"{getattr(module, 'module_id', '')}|"
        f"{getattr(module, 'module_version', '')}"
    )
    digest.update(identity.encode("utf-8"))
    digest.update(b"\0")
    for name, tensor in sorted(module.state_dict().items()):
        _update_tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def tensor_exact_equal(left: Tensor, right: Tensor) -> bool:
    """Byte/value exact equality after a canonical host copy."""

    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return bool(
        torch.equal(
            left.detach().to(device="cpu"),
            right.detach().to(device="cpu"),
        )
    )


def tensor_map_exact_equal(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> bool:
    return set(left) == set(right) and all(
        tensor_exact_equal(left[name], right[name]) for name in left
    )


def module_exact_equal(left: nn.Module, right: nn.Module) -> bool:
    if (
        type(left) is not type(right)
        or getattr(left, "module_id", None) != getattr(right, "module_id", None)
        or getattr(left, "module_version", None)
        != getattr(right, "module_version", None)
    ):
        return False
    left_state = left.state_dict()
    right_state = right.state_dict()
    return tensor_map_exact_equal(left_state, right_state)


def canonical_cpu_tensor_map(tensors: Mapping[str, Tensor]) -> TensorMap:
    """Detach and copy a tensor map into independent contiguous host storage."""

    return {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in tensors.items()
    }


def _unique_storage_bytes(tensors: Iterable[Tensor], *, cuda_only: bool) -> int:
    storages: dict[tuple[str, int | None, int], int] = {}
    for tensor in tensors:
        if cuda_only and not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        size = int(storage.nbytes())
        if size == 0:
            continue
        key = (tensor.device.type, tensor.device.index, int(storage.data_ptr()))
        storages[key] = max(storages.get(key, 0), size)
    return sum(storages.values())


@dataclass(frozen=True)
class OwnedCudaMemory:
    module_bytes: int
    state_bytes: int
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def owned_cuda_memory(
    module: nn.Module, state: Mapping[str, Tensor]
) -> OwnedCudaMemory:
    """Count unique CUDA storages owned by a module and its explicit state."""

    module_bytes = _unique_storage_bytes(_module_owned_tensors(module), cuda_only=True)
    state_bytes = _unique_storage_bytes(state.values(), cuda_only=True)
    # Count the union as well so a pathological shared storage is never double-counted.
    total_bytes = _unique_storage_bytes(
        _all_owned_tensors(module, state), cuda_only=True
    )
    return OwnedCudaMemory(module_bytes, state_bytes, total_bytes)


def owned_cuda_bytes(module: nn.Module, state: Mapping[str, Tensor]) -> int:
    return owned_cuda_memory(module, state).total_bytes


@dataclass(frozen=True)
class TransferRecord:
    source_device: str
    target_device: str
    wall_time_ms: float
    cuda_time_ms: float | None
    module_value_bytes: int
    state_value_bytes: int
    owned_cuda_before: OwnedCudaMemory
    owned_cuda_after: OwnedCudaMemory
    module_digest_before: str
    module_digest_after: str
    state_digest_before: str
    state_digest_after: str
    module_exact: bool
    state_exact: bool
    non_blocking: bool

    @property
    def values_preserved(self) -> bool:
        return self.module_exact and self.state_exact

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["values_preserved"] = self.values_preserved
        result["digest_algorithm"] = CANONICAL_DIGEST_ALGORITHM
        return result


@dataclass(frozen=True)
class PlacementResult:
    module: NeuralModule
    state: TensorMap
    transfer: TransferRecord


def _value_bytes(tensors: Iterable[Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def move_module_and_state(
    module: NeuralModule,
    state: TensorMap,
    target: torch.device | str,
    *,
    non_blocking: bool = False,
) -> PlacementResult:
    """Move one whole neural module and its explicit state, with timing evidence.

    ``module`` follows the normal ``nn.Module.to`` in-place contract.  The state
    mapping is likewise updated in-place so source-device state storages are not
    accidentally kept alive alongside the returned destination state.  Its new
    tensors own independent storage.  CUDA events are used whenever either
    endpoint is CUDA; wall time is always recorded.  Digests and exactness
    checks are outside the timed interval.
    """

    source = placement_device(module, state)
    destination = _normalise_device(target)
    before_module_digest = module_digest(module)
    before_state_digest = tensor_map_digest(state)
    before_cuda = owned_cuda_memory(module, state)
    module_value_bytes = _value_bytes(_module_owned_tensors(module))
    state_value_bytes = _value_bytes(state.values())

    timing_device: torch.device | None = None
    if destination.type == "cuda":
        timing_device = destination
    elif source.type == "cuda":
        timing_device = source

    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    timing_stream: torch.cuda.Stream | None = None
    if timing_device is not None:
        torch.cuda.synchronize(timing_device)
        timing_stream = torch.cuda.current_stream(timing_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(timing_stream)

    wall_started_ns = time.perf_counter_ns()
    module.to(device=destination, non_blocking=non_blocking)
    moved_state = {
        name: tensor.detach()
        .to(device=destination, non_blocking=non_blocking, copy=True)
        .contiguous()
        for name, tensor in state.items()
    }
    if end_event is not None and timing_stream is not None:
        end_event.record(timing_stream)
        end_event.synchronize()
    wall_time_ms = (time.perf_counter_ns() - wall_started_ns) / 1_000_000.0
    cuda_time_ms = (
        float(start_event.elapsed_time(end_event))
        if start_event is not None and end_event is not None
        else None
    )
    state.clear()
    state.update(moved_state)

    after_module_digest = module_digest(module)
    after_state_digest = tensor_map_digest(moved_state)
    after_cuda = owned_cuda_memory(module, moved_state)
    record = TransferRecord(
        source_device=str(source),
        target_device=str(destination),
        wall_time_ms=wall_time_ms,
        cuda_time_ms=cuda_time_ms,
        module_value_bytes=module_value_bytes,
        state_value_bytes=state_value_bytes,
        owned_cuda_before=before_cuda,
        owned_cuda_after=after_cuda,
        module_digest_before=before_module_digest,
        module_digest_after=after_module_digest,
        state_digest_before=before_state_digest,
        state_digest_after=after_state_digest,
        module_exact=before_module_digest == after_module_digest,
        state_exact=before_state_digest == after_state_digest,
        non_blocking=non_blocking,
    )
    return PlacementResult(module=module, state=state, transfer=record)


@dataclass(frozen=True)
class HotWarmHotEvidence:
    source_device: str
    warm_device: str
    output_exact: bool
    final_state_exact: bool
    module_preserved: bool
    continuation_state_preserved: bool
    warm_owned_cuda_bytes: int
    cuda_owned_bytes_after_restore: int
    offload: TransferRecord
    restore: TransferRecord
    reference_output_digest: str
    resumed_output_digest: str
    reference_state_digest: str
    resumed_state_digest: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.output_exact,
                self.final_state_exact,
                self.module_preserved,
                self.continuation_state_preserved,
                self.warm_owned_cuda_bytes == 0,
                self.offload.values_preserved,
                self.restore.values_preserved,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        result["offload"] = self.offload.to_dict()
        result["restore"] = self.restore.to_dict()
        result["digest_algorithm"] = CANONICAL_DIGEST_ALGORITHM
        return result


def verify_hot_warm_hot_continuation(
    module: NeuralModule,
    continuation_state: TensorMap,
    continuation_inputs: Mapping[str, Tensor],
    *,
    hot_device: torch.device | str = "cuda:0",
) -> HotWarmHotEvidence:
    """Offload a live continuation to RAM, restore it, and compare replay.

    The caller supplies state produced by an already executed prefix and the
    input suffix.  A reference suffix is evaluated uninterrupted on the HOT
    placement.  Then the same module/state takes the HOT→WARM→HOT path and the
    suffix is replayed.  Pure transfer is required to preserve every byte.
    """

    hot = _normalise_device(hot_device)
    if hot.type != "cuda":
        raise ValueError("HOT placement must be a CUDA device")
    if placement_device(module, continuation_state) != hot:
        raise ValueError("module and continuation state are not on the requested HOT device")
    input_devices = {tensor.device for tensor in continuation_inputs.values()}
    if input_devices != {hot}:
        raise ValueError("continuation inputs must be wholly resident on the HOT device")

    module_before = module_digest(module)
    state_before = tensor_map_digest(continuation_state)
    reference_state_input = {
        name: tensor.detach().clone() for name, tensor in continuation_state.items()
    }
    with torch.no_grad():
        reference = module.forward_sequence(continuation_inputs, reference_state_input)
    torch.cuda.synchronize(hot)
    reference_outputs = canonical_cpu_tensor_map(reference.tensors)
    reference_final_state = canonical_cpu_tensor_map(reference.state)
    del reference_state_input, reference

    offloaded = move_module_and_state(module, continuation_state, "cpu")
    warm_bytes = owned_cuda_bytes(offloaded.module, offloaded.state)
    restored = move_module_and_state(offloaded.module, offloaded.state, hot)
    with torch.no_grad():
        resumed = restored.module.forward_sequence(
            continuation_inputs, restored.state
        )
    torch.cuda.synchronize(hot)

    output_exact = tensor_map_exact_equal(reference_outputs, resumed.tensors)
    final_state_exact = tensor_map_exact_equal(reference_final_state, resumed.state)
    resumed_output_digest = tensor_map_digest(resumed.tensors)
    resumed_state_digest = tensor_map_digest(resumed.state)
    return HotWarmHotEvidence(
        source_device=str(hot),
        warm_device="cpu",
        output_exact=output_exact,
        final_state_exact=final_state_exact,
        module_preserved=module_before == module_digest(restored.module),
        continuation_state_preserved=state_before == tensor_map_digest(restored.state),
        warm_owned_cuda_bytes=warm_bytes,
        cuda_owned_bytes_after_restore=owned_cuda_bytes(
            restored.module, restored.state
        ),
        offload=offloaded.transfer,
        restore=restored.transfer,
        reference_output_digest=tensor_map_digest(reference_outputs),
        resumed_output_digest=resumed_output_digest,
        reference_state_digest=tensor_map_digest(reference_final_state),
        resumed_state_digest=resumed_state_digest,
    )


def output_to_cpu(output: ModuleOutput) -> ModuleOutput:
    """Return an independent canonical host snapshot of a module result."""

    return ModuleOutput(
        canonical_cpu_tensor_map(output.tensors),
        canonical_cpu_tensor_map(output.state),
    )
