"""Short opt-in local inference check. No training, downloads or quality scoring."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import torch

from alice_runtime.bridge import BridgeError
from alice_runtime.loading import load_qwen
from alice_runtime.runtime import AliceRuntime
from alice_runtime.sessions import SessionError, SessionStore


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            pid, memory = [value.strip() for value in line.split(",")]
            if int(pid) == os.getpid():
                return {"pid": os.getpid(), "process_vram_mib": int(memory)}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"pid": os.getpid(), "process_vram_mib": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    model_root = Path(args.model_path).resolve()
    weights = sorted(model_root.glob("*.safetensors"))
    bindings = lambda: {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in weights}
    before = bindings()
    loaded = load_qwen(model_root, args.device, args.dtype)
    gpu_after_load = gpu_snapshot()
    app = AliceRuntime(loaded)
    session = app.store.new_session("technical")
    person = app.store.add_person(1)
    person2 = app.store.add_person(2)
    events = []

    def identity(x, context):
        events.append({"shape": list(x.shape), "dtype": str(x.dtype),
                       "device": str(x.device), "source_ids": list(context.source_ids)})
        return x

    query = "Complete the sequence: one, two,"
    baseline = app.probe_logits(session, "user", query)
    app.bridge.attach(identity)
    same = app.probe_logits(session, "user", query)
    assert torch.equal(baseline, same), "Identity Bridge changed logits"
    app.bridge.disable()
    disabled = app.probe_logits(session, "user", query)
    assert torch.equal(baseline, disabled), "Disabled Bridge changed logits"
    app.bridge.enable()

    def test_signal(x, context):
        x[:, -1, :] = 0
        return x

    app.bridge.attach(test_signal)
    signaled = app.probe_logits(session, "user", query)
    difference = (baseline - signaled).abs().max().item()
    assert difference > 0, "Test signal did not reach model logits"
    app.bridge.detach()
    restored = app.probe_logits(session, "user", query)
    assert torch.equal(baseline, restored), "Detach failed to restore baseline"

    rejected = []
    snapshot = app.store.messages(session)
    for name, fn in [
        ("shape", lambda x, _: x[..., :-1]),
        ("dtype", lambda x, _: x.double()),
        ("device", lambda x, _: x.to("meta")),
        ("nonfinite", lambda x, _: x.fill_(float("nan"))),
    ]:
        app.bridge.attach(fn)
        try:
            app.respond(session, "user", query, max_new_tokens=1)
        except BridgeError:
            rejected.append(name)
        else:
            raise AssertionError(f"Invalid Bridge {name} was accepted")
        assert app.store.messages(session) == snapshot
        assert not app.model.get_input_embeddings()._forward_hooks
    app.bridge.attach(identity)
    outputs = [
        asdict(app.respond(session, "user", "Hello. My name is Mira.", max_new_tokens=12)),
        asdict(app.respond(session, person.source_id, "My suitcase is blue.", max_new_tokens=8)),
    ]
    separate = app.store.new_session("separate")
    outputs.append(asdict(app.respond(separate, "user", "Продолжи: один, два,", max_new_tokens=8)))
    assert outputs[-1]["source_ids"] == ("user",)
    assert app.store.participants[person2.source_id].source_id != person.source_id
    with tempfile.TemporaryDirectory(prefix="alice-session-smoke-") as tmp:
        state = Path(tmp) / "sessions.json"
        app.store.save(state)
        restored_store = SessionStore.load(state)
        assert restored_store.participants == app.store.participants
        assert restored_store.messages(session) == app.store.messages(session)
        app.store.clear_session(separate)
        app.store.close_session(separate)
        try:
            app.store.messages(separate)
        except SessionError:
            pass
        else:
            raise AssertionError("Closed session remained usable")
    app.bridge.detach()
    app.assert_frozen()
    assert bindings() == before, "Local checkpoint metadata changed"
    result = {
        "status": "PASS", "scope": "Stage 1 technical inference; no learned memory",
        "model": loaded.metadata, "resources": app.resources(),
        "wall_seconds": time.perf_counter() - started,
        "gpu_snapshots": {"after_load": gpu_after_load, "after_checks": gpu_snapshot()},
        "checks": {
            "frozen_eval_parameter_versions": True, "checkpoint_size_mtime_unchanged": True,
            "identity_bridge_logits_exact": True, "disabled_bridge_logits_exact": True,
            "test_signal_max_abs_logit_change": difference, "detached_logits_exact": True,
            "invalid_bridges_rejected": rejected, "error_preserves_session": True,
            "session_isolation_clear_close": True, "source_id_snapshot_roundtrip": True,
        },
        "bridge_events": events, "generations": outputs,
        "limitations": ["Base completions are not dialog-quality qualification",
                        "Test signal is not a trained Memory Bridge",
                        "Session snapshot is transcript/identity metadata, not Neural Memory",
                        "No memory acquisition or cross-restart neural recall is implemented"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"],
                      "resources": result["resources"], "generations": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
