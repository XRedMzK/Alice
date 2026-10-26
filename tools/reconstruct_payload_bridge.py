"""Reconstruct the existing sealed payload Bridge on CPU, without fitting.

The recorded scalar is part of this one tested profile. This utility does not
calibrate a different model, load Qwen weights, acquire facts or use a GPU.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from alice_memory.payload_bridge import PayloadEmbeddingBridge


EXPECTED_BINDINGS = {
    "config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    "tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}
EXPECTED_STATE_DIGEST = "d2b7a0572fef22b1cb8a7d91d32990ee529b9ad0338905793467bd894d750cde"
RECORDED_VALUE_SCALE = 0.6056226491928101


def model_bindings(model_path):
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Supply the existing local Qwen3.5-2B-Base directory")
    result = {}
    for name in EXPECTED_BINDINGS:
        file = root / name
        if not file.is_file() or file.stat().st_size > 2**20:
            raise ValueError(f"Missing or oversized local metadata: {name}")
        result[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if result != EXPECTED_BINDINGS:
        raise ValueError("Model/tokenizer metadata differs from the recorded Bridge profile")
    return result


def reconstruct_bridge(bindings):
    if bindings != EXPECTED_BINDINGS:
        raise ValueError("Model/tokenizer metadata differs from the recorded Bridge profile")
    bridge = PayloadEmbeddingBridge(bindings=bindings, value_scale=RECORDED_VALUE_SCALE)
    digest = bridge.parameter_digest()
    if digest != EXPECTED_STATE_DIGEST:
        raise RuntimeError(
            "CPU reconstruction differs from the sealed state; use the tested Torch/CPU "
            "environment or obtain the verified original artifact. Do not recalibrate or "
            "change the expected digest."
        )
    bridge.basis_id = "alice.memory_payload/17x128/" + digest
    return bridge.freeze()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="A new Bridge directory, separate from model assets")
    args = parser.parse_args(argv)
    try:
        # The original constructor used these CPU QR/RNG settings too.
        torch.set_num_threads(2)
        model = args.model_path.expanduser().resolve()
        output = args.output.expanduser().resolve()
        if output == model or output.is_relative_to(model) or model.is_relative_to(output):
            raise ValueError("Bridge output must be separate from model assets")
        bindings = model_bindings(model)
        bridge = reconstruct_bridge(bindings)
        output.mkdir(parents=True, exist_ok=False)
        bridge.save(output)
        restored = PayloadEmbeddingBridge.load(output, expected_bindings=bindings)
        if restored.parameter_digest() != EXPECTED_STATE_DIGEST:
            raise RuntimeError("Saved reconstruction failed sealed-state verification")
        print(json.dumps({
            "kind": "cpu_reconstruction_existing_sealed_bridge",
            "state_digest": restored.parameter_digest(),
            "latent_basis_id": restored.basis_id,
            "torch_version": torch.__version__,
            "model_weights_loaded": False, "optimizer_steps": 0, "backward_calls": 0,
        }, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
