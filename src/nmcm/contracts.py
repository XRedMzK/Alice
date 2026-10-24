"""Machine-readable module manifests and semantic conformance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class ManifestError(ValueError):
    """Raised when a manifest violates structural or semantic conformance."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    return {value for value in values if value in seen or seen.add(value)}


def _validate_interface_side(
    manifest: dict[str, Any], interfaces_key: str, schemas_key: str
) -> None:
    interfaces = manifest[interfaces_key]
    schemas = manifest[schemas_key]
    interface_ids = [item["interface_id"] for item in interfaces]
    duplicate_ids = _duplicates(interface_ids)
    if duplicate_ids:
        raise ManifestError(
            f"{interfaces_key} contains duplicate interface IDs: {sorted(duplicate_ids)}"
        )
    if set(interface_ids) != set(schemas):
        raise ManifestError(
            f"{interfaces_key} IDs must exactly equal {schemas_key} keys"
        )
    for interface in interfaces:
        interface_id = interface["interface_id"]
        if interface["latent_schema_id"] != schemas[interface_id]["schema_version"]:
            raise ManifestError(
                f"{interface_id} latent_schema_id does not match its schema_version"
            )
    for interface_id, latent_schema in schemas.items():
        tensor_names = [tensor["name"] for tensor in latent_schema["tensors"]]
        duplicate_names = _duplicates(tensor_names)
        if duplicate_names:
            raise ManifestError(
                f"{interface_id} contains duplicate tensor names: {sorted(duplicate_names)}"
            )
        for tensor in latent_schema["tensors"]:
            if tensor["rank"] != len(tensor["shape"]):
                raise ManifestError(
                    f"{interface_id}.{tensor['name']} rank differs from shape length"
                )


def _validate_state(manifest: dict[str, Any], key: str) -> None:
    state_schema = manifest[key]
    names = [tensor["name"] for tensor in state_schema["tensors"]]
    duplicate_names = _duplicates(names)
    if duplicate_names:
        raise ManifestError(f"{key} contains duplicate state names: {sorted(duplicate_names)}")
    if state_schema["enabled"] != bool(names):
        raise ManifestError(f"{key}.enabled must agree with whether tensors are declared")


def validate_manifest(
    manifest: dict[str, Any], schema: dict[str, Any], *, foundation_profile: bool = True
) -> None:
    """Validate JSON Schema plus invariants JSON Schema cannot express."""

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(manifest), key=lambda error: list(error.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ManifestError(rendered)

    _validate_interface_side(manifest, "input_interfaces", "input_latent_schema")
    _validate_interface_side(manifest, "output_interfaces", "output_latent_schema")
    _validate_state(manifest, "private_state_schema")
    _validate_state(manifest, "persistent_state_schema")

    if manifest["persistent_state_schema"]["enabled"]:
        if "persistent_state" not in manifest["checkpoint_schema"]["includes"]:
            raise ManifestError("enabled persistent state must be included in checkpoints")

    if foundation_profile:
        if not manifest["private_state_schema"]["enabled"]:
            raise ManifestError("foundation neural modules must own recurrent private state")
        if manifest["trainable_parameters"]["estimated_count"] <= 0:
            raise ManifestError("foundation neural modules must own trainable parameters")
        if manifest["plasticity_policy"]["default_mode"] != "frozen":
            raise ManifestError("plasticity must default to frozen")
        checkpoint = manifest["checkpoint_schema"]
        if checkpoint["shard_format"] != "safetensors":
            raise ManifestError("foundation tensor shards must use safetensors")
        if checkpoint["integrity"] != "sha256" or not checkpoint["atomic_write"]:
            raise ManifestError("foundation shards require atomic SHA-256 integrity")


def load_manifest(
    manifest_path: str | Path,
    schema_path: str | Path,
    *,
    foundation_profile: bool = True,
) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path))
    schema = _read_json(Path(schema_path))
    validate_manifest(manifest, schema, foundation_profile=foundation_profile)
    return manifest

