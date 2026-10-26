"""Opt-in real-model check of an installed Alice CLI; no training or test helpers.

Requires separately supplied local Qwen and sealed Bridge assets. The small
demonstration facts and expected answers below belong only to this external
checker. Subprocesses receive ordinary CLI input, never an oracle answer file.
Run --help without Torch/model assets. Evidence is written outside the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time


class SmokeFailure(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise SmokeFailure(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkpoint(directory):
    """Inspect the public memory format using only the standard library."""
    root = Path(directory)
    metadata = json.loads((root / "memory.json").read_text(encoding="utf-8"))
    expected_fields = {"schema", "update_rule", "latent_basis_id", "source_ids", "shape",
                       "dtype", "snapshot_id", "tensor_file", "sha256", "metadata_sha256"}
    check(set(metadata) == expected_fields, "Unexpected memory metadata fields")
    canonical = json.dumps({k: v for k, v in metadata.items() if k != "metadata_sha256"},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False).encode()
    check(hashlib.sha256(canonical).hexdigest() == metadata["metadata_sha256"],
          "Memory metadata checksum mismatch")
    name = metadata["tensor_file"]
    check(isinstance(name, str) and re.fullmatch(r"memory-payload-[0-9a-f]{32}\.safetensors", name),
          "Unexpected memory payload filename")
    payload = root / name
    check(not payload.is_symlink() and payload.is_file(), "Memory payload must be a regular file")
    check(sha256(payload) == metadata["sha256"], "Memory payload checksum mismatch")
    with payload.open("rb") as handle:
        length_bytes = handle.read(8)
        check(len(length_bytes) == 8, "Truncated safetensors header")
        header_length = struct.unpack("<Q", length_bytes)[0]
        check(0 < header_length <= 65536, "Unexpected safetensors header size")
        header = json.loads(handle.read(header_length))
        data = handle.read()
    check(set(header) == {"matrix"}, "Memory checkpoint contains unexpected tensors")
    tensor = header["matrix"]
    sources = metadata["source_ids"]
    check(metadata["schema"] == "alice.memory_payload/v1" and metadata["dtype"] == "float32",
          "Unexpected memory schema/dtype")
    check(isinstance(sources, list) and len(set(sources)) == len(sources)
          and tensor["shape"] == metadata["shape"] == [len(sources), 17, 128, 128]
          and tensor["dtype"] == "F32", "Unexpected memory tensor shape/dtype")
    plane_bytes = 17 * 128 * 128 * 4
    check(tensor["data_offsets"] == [0, len(data)] and len(data) == len(sources) * plane_bytes,
          "Unexpected memory tensor byte layout")
    planes = {source: hashlib.sha256(data[i * plane_bytes:(i + 1) * plane_bytes]).hexdigest()
              for i, source in enumerate(sources)}
    return {"payload_sha256": metadata["sha256"], "plane_sha256": planes,
            "latent_basis_id": metadata["latent_basis_id"], "source_ids": sources}


def bridge_files(directory):
    root = Path(directory)
    manifest = root / "bridge.json"
    check(not manifest.is_symlink() and manifest.is_file(), "Bridge manifest must be a regular file")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    check(metadata.get("schema") == "alice.payload_embedding_bridge/v1", "Wrong Bridge schema")
    name = metadata.get("tensor_file")
    check(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors"),
          "Unexpected Bridge payload filename")
    tensor = root / name
    check(not tensor.is_symlink() and tensor.is_file(), "Bridge tensor must be a regular file")
    check(sha256(tensor) == metadata.get("sha256"), "Bridge payload checksum mismatch")
    return metadata, {"bridge.json": sha256(manifest), name: sha256(tensor)}


def action(row):
    return row.get("memory_info", {}).get("dialogue_action")


def recalled(row, expected):
    check(row.get("text") == expected, f"Expected recalled value {expected!r}, got {row.get('text')!r}")
    check(row.get("generated_tokens", 0) > 0 and action(row) is None,
          "Recall must come from model generation, not a dialogue acknowledgement")
    info = row.get("memory_info", {})
    check(info.get("mode") == "on" and info.get("memory_applied") is True,
          "Expected an applied ON memory read")
    check(info.get("query_context") == "current_question_only_v1", "Unexpected query context profile")


def bypassed(row, acquired):
    info = row.get("memory_info", {})
    check(not info.get("memory_applied") and info.get("mode") in {"off", "empty"},
          "OFF/EMPTY unexpectedly applied memory")
    check(row.get("text") != acquired, "OFF/EMPTY also returned the acquired value; causal control inconclusive")


def unrelated(row, pattern):
    check(re.fullmatch(pattern, row.get("text", "").strip(), flags=re.I) is not None,
          f"Unrelated answer failed: {row.get('text')!r}")
    check(not row.get("memory_info", {}).get("memory_applied"), "Memory interfered with an unrelated question")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory outside this checkout")
    parser.add_argument("--python", default=sys.executable, help="Python with Alice installed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--timeout", type=float, default=90, help="Maximum seconds for each fresh process")
    parser.add_argument("--expected-bridge-digest", help="Trusted sealed state_digest; compare before launch")
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 300:
        parser.error("--timeout must be in (0, 300] seconds")
    root, model, original_bridge = (p.expanduser().resolve() for p in
                                    (args.output, args.model_path, args.bridge))
    checkout = Path(__file__).resolve().parents[1]
    if root == checkout or root.is_relative_to(checkout):
        parser.error("Evidence must be outside the source checkout")
    for asset in (model, original_bridge):
        if root == asset or root.is_relative_to(asset) or asset.is_relative_to(root):
            parser.error("Evidence directory must not overlap model or Bridge assets")
    root.mkdir(parents=True, exist_ok=False)
    started, launches = time.perf_counter(), []
    result = {"status": "FAIL", "training_runs": 0, "launches": launches}
    try:
        metadata, before_bridge = bridge_files(original_bridge)
        if args.expected_bridge_digest:
            check(metadata.get("state_digest") == args.expected_bridge_digest, "Unexpected sealed Bridge identity")
        bindings = metadata.get("bindings", {})
        check(set(bindings) == {"config.json", "tokenizer_config.json"}, "Unexpected Bridge model bindings")
        for name, expected_hash in bindings.items():
            check(sha256(model / name) == expected_hash, f"Model binding mismatch: {name}")
        bridge = root / "bridge-copy"
        bridge.mkdir()
        for name in before_bridge:
            shutil.copyfile(original_bridge / name, bridge / name)
        state = root / "live-memory"
        runtime_cwd = root / "runtime-cwd"
        runtime_cwd.mkdir()
        python = str(Path(args.python).expanduser().absolute())
        base_command = [python, "-B", "-m", "alice_runtime", "--model-path", str(model),
                        "--device", args.device, "--dtype", args.dtype, "--max-new-tokens", "8",
                        "--json", "--memory-bridge", str(bridge)]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")

        def run(name, lines, memory=state):
            command = [*base_command, "--memory-state", str(memory)]
            body = "\n".join([*lines, "/quit", ""])
            check("/remember " not in body, "Smoke must exercise ordinary message acquisition")
            (root / (name + ".stdin")).write_text(body, encoding="utf-8")
            phase_started = time.perf_counter()
            process = subprocess.Popen(command, cwd=runtime_cwd, env=env, text=True, encoding="utf-8",
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timed_out = False
            try:
                stdout, stderr = process.communicate(body, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
            (root / (name + ".stdout.jsonl")).write_text(stdout, encoding="utf-8")
            (root / (name + ".stderr")).write_text(stderr, encoding="utf-8")
            record = {"phase": name, "pid": process.pid, "returncode": process.returncode,
                      "seconds": time.perf_counter() - phase_started, "command": command}
            launches.append(record)
            check(not timed_out, f"{name} exceeded {args.timeout:g} seconds")
            check(process.returncode == 0, f"{name} exited {process.returncode}; inspect stderr")
            events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            check(events and events[0].get("event") == "ready" and events[-1].get("event") == "resources",
                  f"{name} emitted incomplete CLI events")
            check(not any(row.get("event") == "error" for row in events), f"{name} emitted a runtime error")
            record["session_id"] = events[0]["session_id"]
            record["resources"] = events[-1]["resources"]
            record["bridge_digest"] = events[0]["memory"]["bridge_digest"]
            rows = [row for row in events if row.get("event") == "response"]
            record["responses"] = [row["text"] for row in rows]
            print(json.dumps({"phase": name, "responses": record["responses"]}, ensure_ascii=False), flush=True)
            return events, rows

        object_id, retained_id = "release-check-0", "release-check-1"
        question = f"Какая метка у объекта {object_id}?"
        transcript = root / "acquisition-transcript.json"
        events, rows = run("01-acquire", [f"Запомни: метка объекта {object_id} — window.",
            "/memory status", "да", f"Remember that the label of object {retained_id} is museum.",
            "yes", "/source person:1", f"Remember that object {object_id} has label helmet.",
            "yes", f"/save {json.dumps(str(transcript), ensure_ascii=False)}"])
        check([action(row) for row in rows] == ["confirmation_required", "remembered"] * 3,
              "Acquisition did not require and receive three confirmations")
        statuses = [row["memory"] for row in events if row.get("event") == "memory_status"]
        check(len(statuses) == 1 and statuses[0]["memory_digest"] == events[0]["memory"]["memory_digest"],
              "An unconfirmed proposal changed memory")
        receipts = [row["memory_info"]["acquisition"] for row in rows if action(row) == "remembered"]
        check([receipt["source_id"] for receipt in receipts] == ["user", "user", "person:1"],
              "Acquisition source attribution failed")
        for receipt in receipts:
            check(all(receipt.get(flag) is True for flag in
                      ("saved", "persistent_m_changed", "bridge_unchanged", "qwen_frozen")),
                  "Acquisition did not publish M with fixed Qwen/Bridge")
        sessions = json.loads(transcript.read_text(encoding="utf-8"))["sessions"]
        check(sessions and all(not session["messages"] for session in sessions),
              "Teaching/confirmation content leaked into saved session history")
        original = checkpoint(state)
        shutil.copytree(state, root / "original-memory")

        _, rows = run("02-fresh-recall", [question, "/memory off", question, "What is two plus two?",
            "/memory empty", question, "/memory on", f"Which label was assigned to object {object_id}?",
            f"Какая метка у объекта {retained_id}?", "/knowledge-source person:1", question,
            "/knowledge-source user", "What is two plus two?", "What is the capital of France?",
            "What is three plus four?"])
        check(len(rows) == 10, "Unexpected fresh recall response count")
        recalled(rows[0], "window")
        bypassed(rows[1], "window")
        unrelated(rows[2], r"(?:4|four)\.?")
        bypassed(rows[3], "window")
        recalled(rows[4], "window")
        recalled(rows[5], "museum")
        recalled(rows[6], "helmet")
        unrelated(rows[7], r"(?:4|four)\.?")
        unrelated(rows[8], r"(?:The capital of France is )?Paris\.?")
        unrelated(rows[9], r"(?:7|seven)\.?")
        check(checkpoint(state) == original, "Read-only recall changed persisted memory")

        _, rows = run("03-correct", [f"Исправь: метка объекта {object_id} — dragon.", "да"])
        check([action(row) for row in rows] == ["confirmation_required", "remembered"],
              "Correction confirmation failed")
        corrected = checkpoint(state)
        check(corrected["payload_sha256"] != original["payload_sha256"], "Correction did not change memory")
        check(all(corrected["plane_sha256"][source] == digest for source, digest in
                  original["plane_sha256"].items() if source != "user"),
              "Correction changed another source's parameters")
        shutil.copytree(state, root / "corrected-memory")

        _, rows = run("04-corrected-restore", [question, f"Скажи метку объекта {retained_id}.",
            "/knowledge-source person:1", question, "/knowledge-source user", "/memory off", question,
            "/memory empty", question, "/memory on", f"Скажи метку объекта {object_id}.",
            "What is two plus two?"])
        check(len(rows) == 7, "Unexpected corrected restore response count")
        for index, value in ((0, "dragon"), (1, "museum"), (2, "helmet"), (5, "dragon")):
            recalled(rows[index], value)
        bypassed(rows[3], "dragon")
        bypassed(rows[4], "dragon")
        unrelated(rows[6], r"(?:4|four)\.?")
        check(checkpoint(state) == corrected, "Corrected read-only restore changed persisted memory")

        _, rows = run("05-memory-swap", [question, f"Какая метка у объекта {retained_id}?",
            "/knowledge-source person:1", question], memory=root / "original-memory")
        check(len(rows) == 3, "Unexpected swapped memory response count")
        for row, value in zip(rows, ("window", "museum", "helmet"), strict=True):
            recalled(row, value)
        check(checkpoint(root / "original-memory") == original, "Swapped state was unexpectedly modified")
        check(checkpoint(state) == corrected, "Swapping the copy modified live corrected state")
        check(bridge_files(original_bridge)[1] == before_bridge and bridge_files(bridge)[1] == before_bridge,
              "Bridge asset files changed")
        check(len({launch["pid"] for launch in launches}) == 5
              and len({launch["session_id"] for launch in launches}) == 5,
              "Checks did not use five independent processes/sessions")
        check({launch["bridge_digest"] for launch in launches} == {metadata["state_digest"]},
              "Bridge changed between fresh processes")
        result.update(status="PASS", scope="Bounded confirmed RU/EN object-label acquisition and neural recall",
                      bridge_files_unchanged=True, bridge_state_digest=metadata["state_digest"],
                      original=original, corrected=corrected, manual_remember_commands=0,
                      teaching_transcript_restored=False, pythonpath_removed=True,
                      unrelated_controls="4, Paris, 7; OFF/ON paired arithmetic",
                      acquisition_confirmed=True, correction_retains_other_fact_and_source=True,
                      memory_swap_with_same_bridge=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"], file=sys.stderr, flush=True)
    result["seconds"] = time.perf_counter() - started
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
