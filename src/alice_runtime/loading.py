"""Local-only, frozen text loading for the first technical Alice runtime.

Uses Transformers primitives and the previously established outer-to-text key
mapping. It does not import the archived M1 provider or load vision/MTP modules.
"""

from __future__ import annotations

import gc
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

_LOAD_LOCK = threading.RLock()
_FP32_PATTERNS = [r"\.A_log$", r"\.linear_attn\.norm\.weight$"]
_FP32_SUFFIXES = (".A_log", ".linear_attn.norm.weight")
_EXCLUDED_PREFIXES = ("model.visual.", "mtp.")


class LoadingError(RuntimeError):
    """The local snapshot or requested execution environment cannot be loaded."""


@dataclass(frozen=True)
class LoadedQwen:
    model: Any
    tokenizer: Any
    metadata: dict[str, Any]


@contextmanager
def _preserve_fp32(model_class: Any) -> Iterator[None]:
    # Transformers 5.15 creates the dtype plan from this exact class attribute.
    # Preserve official DeltaNet FP32 values before a BF16 load can cast them.
    absent = object()
    with _LOAD_LOCK:
        previous = model_class.__dict__.get("_keep_in_fp32_modules_strict", absent)
        model_class._keep_in_fp32_modules_strict = list(_FP32_PATTERNS)
        try:
            yield
        finally:
            if previous is absent:
                delattr(model_class, "_keep_in_fp32_modules_strict")
            else:
                model_class._keep_in_fp32_modules_strict = previous


def load_qwen(
    model_path: str | Path, device: str = "cpu", dtype: str = "float32"
) -> LoadedQwen:
    """Load an existing local Qwen3.5 snapshot; never download or update weights.

    CPU uses FP32. CUDA accepts BF16 or FP32 with no automatic device fallback.
    The tokenizer is returned raw; applying a chat template is a caller decision.
    """
    started = time.perf_counter()
    root = Path(model_path).expanduser().resolve()
    model = None
    torch = None
    target = None
    try:
        if not root.is_dir():
            raise LoadingError(f"Local model directory does not exist: {root}")
        raw = json.loads((root / "config.json").read_text())
        index = json.loads((root / "model.safetensors.index.json").read_text())
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise LoadingError("Local safetensors index has no weight_map")
        shards = sorted(set(weights.values()))
        for shard in shards:
            file = (root / shard).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                raise LoadingError(f"Missing or out-of-directory local shard: {shard}")

        import torch as torch_module
        import transformers

        torch = torch_module
        target = torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise LoadingError("Stage 1 supports explicit cpu or cuda devices")
        if dtype not in {"float32", "bfloat16"}:
            raise LoadingError("dtype must be float32 or bfloat16")
        if target.type == "cpu" and dtype != "float32":
            raise LoadingError("Stage 1 CPU loading requires dtype=float32")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise LoadingError("CUDA is unavailable; explicitly choose cpu/float32")
            if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise LoadingError("Requested CUDA device does not support bfloat16")

        outer = raw.get("model_type") == "qwen3_5"
        text = raw.get("text_config") if outer else raw
        if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_text":
            raise LoadingError("config.json does not describe Qwen3.5 text weights")
        config = transformers.Qwen3_5TextConfig.from_dict(text)
        model_class = transformers.Qwen3_5ForCausalLM
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        with _preserve_fp32(model_class):
            model, info = model_class.from_pretrained(
                str(root), config=config, dtype=getattr(torch, dtype),
                device_map={"": str(target)}, low_cpu_mem_usage=True,
                local_files_only=True, trust_remote_code=False, use_kernels=False,
                key_mapping={r"^model\.language_model\.": "model."} if outer else None,
                attn_implementation="sdpa", output_loading_info=True,
            )
        invalid = {
            key: info.get(key, [])
            for key in ("error_msgs", "mismatched_keys", "missing_keys")
            if info.get(key)
        }
        unexpected = info.get("unexpected_keys", [])
        bad_unexpected = [k for k in unexpected if not k.startswith(_EXCLUDED_PREFIXES)]
        if bad_unexpected:
            invalid["unexpected_keys"] = bad_unexpected
        if invalid:
            raise LoadingError(f"Incomplete local text checkpoint load: {invalid}")
        model.eval()
        model.requires_grad_(False)
        counts: dict[str, int] = {}
        parameter_count = parameter_bytes = official_fp32_parameters = 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + count
            parameter_count += count
            parameter_bytes += count * parameter.element_size()
            if name.endswith(_FP32_SUFFIXES):
                if parameter.dtype != torch.float32:
                    raise LoadingError(f"Official FP32 parameter was cast: {name}")
                official_fp32_parameters += count
            if parameter.requires_grad:
                raise LoadingError(f"Qwen parameter is not frozen: {name}")
        metadata = {
            "model_path": str(root), "model_class": type(model).__name__,
            "device": str(target), "requested_dtype": dtype,
            "embedding_dtype": str(model.get_input_embeddings().weight.dtype),
            "parameter_count": parameter_count, "parameter_bytes": parameter_bytes,
            "parameter_counts_by_dtype": counts,
            "official_fp32_parameter_count": official_fp32_parameters,
            "parameters_frozen": True, "training": model.training,
            "text_config": {k: getattr(config, k) for k in (
                "hidden_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"
            )},
            "excluded_checkpoint_tensor_count": sum(
                name.startswith(_EXCLUDED_PREFIXES) for name in weights
            ),
            "accepted_unexpected_keys": sorted(unexpected), "weight_shards": shards,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "safetensors_version": version("safetensors"),
            "local_files_only": True, "use_kernels": False,
            "load_wall_seconds": time.perf_counter() - started,
        }
        return LoadedQwen(model, tokenizer, metadata)
    except Exception as exc:
        model = None
        gc.collect()
        if torch is not None and target is not None and target.type == "cuda":
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        if isinstance(exc, LoadingError):
            raise
        raise LoadingError(f"Cannot load local Qwen snapshot {root}: {exc}") from exc
