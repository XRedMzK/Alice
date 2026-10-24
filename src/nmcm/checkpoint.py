"""Independent safetensors module shards and atomic whole-checkpoint manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from safetensors.torch import load_file, save_file
from torch import Tensor

from .neural import NeuralModule, TensorMap


class CheckpointError(RuntimeError):
    pass


SHARD_RECORD_FIELDS = {
    "instance_id",
    "module_id",
    "module_version",
    "compatibility_version",
    "state_schema_version",
    "metadata_file",
    "metadata_sha256",
    "tensor_file",
    "tensor_sha256",
    "tensor_bytes",
}
SHARD_METADATA_FIELDS = {
    "schema_version",
    "instance_id",
    "module_id",
    "module_version",
    "compatibility_version",
    "state_schema_version",
    "snapshot_id",
    "logical_step",
    "tensor_file",
    "tensor_sha256",
    "tensor_bytes",
    "parameter_keys",
    "private_state_keys",
    "private_state_spec",
}
WHOLE_MANIFEST_FIELDS = {
    "schema_version",
    "graph_id",
    "graph_version",
    "snapshot_id",
    "logical_step",
    "modules",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _safe_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in value
        )
    ):
        raise CheckpointError(f"unsafe checkpoint identifier: {value!r}")
    return value


def _safe_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise CheckpointError(f"checkpoint path must be a plain file name: {name!r}")
    return root / name


def _validate_json_schema(payload: Mapping[str, Any], schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "configs" / schema_name
    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        Draft202012Validator(schema).validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise CheckpointError(
            f"checkpoint JSON violates Draft 2020-12 schema {schema_name}"
        ) from error


def _validate_module_state(
    module: NeuralModule, state: Mapping[str, Tensor]
) -> None:
    if set(state) != set(module.state_features):
        raise CheckpointError("private-state keys differ from module state schema")
    expected_state = module.initial_state(1)
    batch_sizes: set[int] = set()
    for name, tensor in state.items():
        expected = expected_state[name]
        if (
            tensor.dtype != expected.dtype
            or tensor.ndim != expected.ndim
            or tensor.shape[0] <= 0
            or tensor.shape[1:] != expected.shape[1:]
            or tensor.shape[-1] != module.state_features[name]
        ):
            raise CheckpointError(f"private-state schema mismatch for {name}")
        if not tensor.isfinite().all().item():
            raise CheckpointError(f"private-state contains non-finite values for {name}")
        batch_sizes.add(tensor.shape[0])
    if len(batch_sizes) != 1:
        raise CheckpointError("private-state tensors disagree on batch size")
    for name, tensor in module.state_dict().items():
        if tensor.is_floating_point() and not tensor.isfinite().all().item():
            raise CheckpointError(f"module tensor contains non-finite values for {name}")


@dataclass(frozen=True)
class _PreparedShard:
    model_tensors: dict[str, Tensor]
    private_state: TensorMap


def save_module_shard(
    root: str | Path,
    instance_id: str,
    module: NeuralModule,
    state: Mapping[str, Tensor],
    *,
    snapshot_id: str,
    logical_step: int,
    file_stem: str | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    safe_instance = _safe_name(instance_id)
    safe_snapshot = _safe_name(snapshot_id)
    if not isinstance(logical_step, int) or isinstance(logical_step, bool) or logical_step < 0:
        raise CheckpointError("logical step must be non-negative")
    _validate_module_state(module, state)
    safe_stem = (
        _safe_name(file_stem)
        if file_stem is not None
        else f"{safe_snapshot}.{safe_instance}"
    )
    tensor_path = root_path / f"{safe_stem}.safetensors"
    metadata_path = root_path / f"{safe_stem}.json"
    if tensor_path.exists() or metadata_path.exists():
        raise CheckpointError(f"immutable snapshot shard already exists: {safe_stem}")
    temporary_tensor = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
    tensors: dict[str, Tensor] = {
        f"model.{name}": value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }
    tensors.update(
        {
            f"private_state.{name}": value.detach().cpu().contiguous()
            for name, value in state.items()
        }
    )
    save_file(tensors, str(temporary_tensor))
    with temporary_tensor.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary_tensor, tensor_path)
    tensor_digest = sha256_file(tensor_path)
    metadata = {
        "schema_version": "nmcm.shard/v2",
        "instance_id": instance_id,
        "module_id": module.module_id,
        "module_version": module.module_version,
        "compatibility_version": module.compatibility_version,
        "state_schema_version": module.state_schema_version,
        "snapshot_id": snapshot_id,
        "logical_step": logical_step,
        "tensor_file": tensor_path.name,
        "tensor_sha256": tensor_digest,
        "tensor_bytes": tensor_path.stat().st_size,
        "parameter_keys": sorted(module.state_dict()),
        "private_state_keys": sorted(state),
        "private_state_spec": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(state.items())
        },
    }
    _validate_json_schema(metadata, "module_shard.schema.json")
    _atomic_json(metadata_path, metadata)
    return {
        "instance_id": instance_id,
        "module_id": module.module_id,
        "module_version": module.module_version,
        "compatibility_version": module.compatibility_version,
        "state_schema_version": module.state_schema_version,
        "metadata_file": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "tensor_file": tensor_path.name,
        "tensor_sha256": tensor_digest,
        "tensor_bytes": tensor_path.stat().st_size,
    }


def _prepare_module_shard(
    root: str | Path,
    record: Mapping[str, Any],
    module: NeuralModule,
    *,
    expected_snapshot_id: str | None = None,
    expected_logical_step: int | None = None,
) -> _PreparedShard:
    root_path = Path(root)
    metadata_path = _safe_child(root_path, record["metadata_file"])
    if sha256_file(metadata_path) != record["metadata_sha256"]:
        raise CheckpointError("module metadata digest mismatch")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    _validate_json_schema(metadata, "module_shard.schema.json")
    if set(record) != SHARD_RECORD_FIELDS:
        raise CheckpointError("module record does not match its closed schema")
    if set(metadata) != SHARD_METADATA_FIELDS:
        raise CheckpointError("module metadata does not match its closed schema")
    if metadata["schema_version"] != "nmcm.shard/v2":
        raise CheckpointError("unsupported module-shard schema")
    expected_identity = (
        module.module_id,
        module.module_version,
        module.compatibility_version,
        module.state_schema_version,
    )
    actual_identity = (
        metadata["module_id"],
        metadata["module_version"],
        metadata["compatibility_version"],
        metadata["state_schema_version"],
    )
    if actual_identity != expected_identity:
        raise CheckpointError("module shard identity or schema mismatch")
    if metadata["instance_id"] != record["instance_id"]:
        raise CheckpointError("module shard instance identity mismatch")
    for key in (
        "module_id",
        "module_version",
        "compatibility_version",
        "state_schema_version",
        "tensor_file",
        "tensor_sha256",
        "tensor_bytes",
    ):
        if metadata[key] != record[key]:
            raise CheckpointError(f"module record/metadata mismatch for {key}")
    if expected_snapshot_id is not None and metadata["snapshot_id"] != expected_snapshot_id:
        raise CheckpointError("mixed snapshot ID in module shard")
    if expected_logical_step is not None and metadata["logical_step"] != expected_logical_step:
        raise CheckpointError("mixed logical step in module shard")
    tensor_path = _safe_child(root_path, metadata["tensor_file"])
    if sha256_file(tensor_path) != metadata["tensor_sha256"]:
        raise CheckpointError("module tensor digest mismatch")
    if tensor_path.stat().st_size != metadata["tensor_bytes"]:
        raise CheckpointError("module tensor size mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    if any(tensor.is_floating_point() and not tensor.isfinite().all().item() for tensor in tensors.values()):
        raise CheckpointError("module shard contains non-finite tensor values")
    if any(
        not (name.startswith("model.") or name.startswith("private_state."))
        for name in tensors
    ):
        raise CheckpointError("module shard contains an undeclared tensor namespace")
    model_tensors = {
        name.removeprefix("model."): value
        for name, value in tensors.items()
        if name.startswith("model.")
    }
    state = {
        name.removeprefix("private_state."): value
        for name, value in tensors.items()
        if name.startswith("private_state.")
    }
    expected_model = module.state_dict()
    if set(model_tensors) != set(expected_model):
        raise CheckpointError("module parameter/buffer keys differ from implementation")
    if sorted(model_tensors) != metadata["parameter_keys"]:
        raise CheckpointError("module parameter keys differ from metadata")
    for name, tensor in model_tensors.items():
        expected = expected_model[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise CheckpointError(f"module tensor schema mismatch for {name}")
    if sorted(state) != metadata["private_state_keys"]:
        raise CheckpointError("private-state keys differ from metadata")
    if set(metadata["private_state_spec"]) != set(state):
        raise CheckpointError("private-state specification keys differ from tensors")
    if set(state) != set(module.state_features):
        raise CheckpointError("private-state keys differ from module state schema")
    expected_state = module.initial_state(1)
    batch_sizes: set[int] = set()
    for name, tensor in state.items():
        specification = metadata.get("private_state_spec", {}).get(name)
        if specification is None:
            raise CheckpointError(f"missing private-state metadata for {name}")
        if list(tensor.shape) != specification["shape"] or str(tensor.dtype) != specification["dtype"]:
            raise CheckpointError(f"private-state metadata mismatch for {name}")
        expected = expected_state[name]
        if (
            tensor.dtype != expected.dtype
            or tensor.ndim != expected.ndim
            or tensor.shape[0] <= 0
            or tensor.shape[1:] != expected.shape[1:]
            or tensor.shape[-1] != module.state_features[name]
        ):
            raise CheckpointError(f"private-state schema mismatch for {name}")
        batch_sizes.add(tensor.shape[0])
    if len(batch_sizes) != 1:
        raise CheckpointError("private-state tensors disagree on batch size")
    return _PreparedShard(model_tensors=model_tensors, private_state=state)


def load_module_shard(
    root: str | Path,
    record: Mapping[str, Any],
    module: NeuralModule,
) -> TensorMap:
    prepared = _prepare_module_shard(root, record, module)
    original = {name: value.detach().clone() for name, value in module.state_dict().items()}
    try:
        module.load_state_dict(prepared.model_tensors, strict=True)
    except Exception as error:
        module.load_state_dict(original, strict=True)
        raise CheckpointError("module commit failed and was rolled back") from error
    return prepared.private_state


def save_whole_checkpoint(
    root: str | Path,
    *,
    graph_id: str,
    graph_version: str,
    modules: Mapping[str, NeuralModule],
    states: Mapping[str, Mapping[str, Tensor]],
    snapshot_id: str,
    logical_step: int,
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    if set(modules) != set(states):
        raise CheckpointError("whole checkpoint module/state sets differ")
    safe_snapshot = _safe_name(snapshot_id)
    if not isinstance(graph_id, str) or not graph_id:
        raise CheckpointError("graph ID must be a non-empty string")
    if not isinstance(graph_version, str) or not graph_version:
        raise CheckpointError("graph version must be a non-empty string")
    if not isinstance(logical_step, int) or isinstance(logical_step, bool) or logical_step < 0:
        raise CheckpointError("logical step must be non-negative")
    batch_sizes: set[int] = set()
    for instance_id in sorted(modules):
        _safe_name(instance_id)
        _validate_module_state(modules[instance_id], states[instance_id])
        batch_sizes.add(next(iter(states[instance_id].values())).shape[0])
    if len(batch_sizes) != 1:
        raise CheckpointError("whole checkpoint module states disagree on batch size")
    records = [
        save_module_shard(
            root_path,
            instance_id,
            modules[instance_id],
            states[instance_id],
            snapshot_id=snapshot_id,
            logical_step=logical_step,
            file_stem=f"{safe_snapshot}.{_safe_name(instance_id)}",
        )
        for instance_id in sorted(modules)
    ]
    manifest = {
        "schema_version": "nmcm.whole_checkpoint/v2",
        "graph_id": graph_id,
        "graph_version": graph_version,
        "snapshot_id": snapshot_id,
        "logical_step": logical_step,
        "modules": records,
    }
    _validate_json_schema(manifest, "whole_checkpoint.schema.json")
    target = root_path / "checkpoint.json"
    _atomic_json(target, manifest)
    return target


def load_whole_checkpoint(
    checkpoint_path: str | Path,
    *,
    graph_id: str,
    graph_version: str,
    modules: Mapping[str, NeuralModule],
) -> tuple[dict[str, TensorMap], dict[str, Any]]:
    path = Path(checkpoint_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_json_schema(manifest, "whole_checkpoint.schema.json")
    if set(manifest) != WHOLE_MANIFEST_FIELDS:
        raise CheckpointError("whole checkpoint does not match its closed schema")
    if manifest["schema_version"] != "nmcm.whole_checkpoint/v2":
        raise CheckpointError("unsupported whole-checkpoint schema")
    if (manifest["graph_id"], manifest["graph_version"]) != (graph_id, graph_version):
        raise CheckpointError("whole-checkpoint graph identity mismatch")
    if not isinstance(manifest["modules"], list) or not manifest["modules"]:
        raise CheckpointError("whole checkpoint must contain module records")
    if any(set(record) != SHARD_RECORD_FIELDS for record in manifest["modules"]):
        raise CheckpointError("whole checkpoint module record violates its closed schema")
    records = {record["instance_id"]: record for record in manifest["modules"]}
    if len(records) != len(manifest["modules"]):
        raise CheckpointError("whole-checkpoint contains duplicate module instances")
    if set(records) != set(modules):
        raise CheckpointError("whole-checkpoint module set mismatch")
    prepared = {
        instance_id: _prepare_module_shard(
            path.parent,
            records[instance_id],
            module,
            expected_snapshot_id=manifest["snapshot_id"],
            expected_logical_step=manifest["logical_step"],
        )
        for instance_id, module in modules.items()
    }
    batch_sizes = {
        next(iter(item.private_state.values())).shape[0] for item in prepared.values()
    }
    if len(batch_sizes) != 1:
        raise CheckpointError("whole checkpoint module states disagree on batch size")
    originals = {
        instance_id: {
            name: value.detach().clone() for name, value in module.state_dict().items()
        }
        for instance_id, module in modules.items()
    }
    try:
        for instance_id, module in modules.items():
            module.load_state_dict(prepared[instance_id].model_tensors, strict=True)
    except Exception as error:  # all tensors were prevalidated; retain rollback defensively
        for instance_id, module in modules.items():
            module.load_state_dict(originals[instance_id], strict=True)
        raise CheckpointError("whole-checkpoint commit failed and was rolled back") from error
    states = {
        instance_id: item.private_state for instance_id, item in prepared.items()
    }
    return states, manifest
