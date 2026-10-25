"""Run local Alice, with optional independently persisted Neural Memory."""

import argparse
from dataclasses import asdict
import json
import shlex
import sys

from .loading import load_qwen
from .runtime import AliceRuntime
from .sessions import SessionError, SessionStore


_COMMANDS = (
    "Commands: /new, /clear, /person N, /source ID, /knowledge-source ID, "
    '/remember SOURCE_ID "question cue" "single-token value", '
    "/memory on|off|empty|status|save, /save PATH (transcript only), /quit"
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Existing local snapshot directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--person", type=int, action="append", default=[])
    parser.add_argument("--source-id", default="user", help="Current speaker")
    parser.add_argument("--knowledge-source-id", help="Memory source to ask; defaults to the current speaker")
    parser.add_argument("--session", help="Existing session from --restore-sessions; omit to create a new one")
    parser.add_argument("--restore-sessions", help="Explicit transcript/identity restore, not neural memory")
    parser.add_argument("--message", action="append", help="Generate once per message; otherwise interactive")
    parser.add_argument("--memory-bridge", help="Existing sealed payload Bridge checkpoint directory")
    parser.add_argument("--memory-state", help="Independent writable memory directory; new means empty memory")
    parser.add_argument("--memory-mode", choices=["on", "off", "empty"],
                        help="Initial memory mode; empty is temporary and never erases acquired state")
    parser.add_argument("--json", action="store_true", help="Interactive JSONL events without prompts")
    return parser


def _emit(event, data, json_mode, human=None):
    if json_mode:
        print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)
    elif human is not None:
        print(human, flush=True)


def _response_record(result):
    record = asdict(result)
    if record.get("memory_info") is None:
        record.pop("memory_info", None)
    return record


def _interactive(runtime, store, session, args):
    memory_enabled = args.memory_bridge is not None
    if args.json:
        ready = {"session_id": session, "source_id": args.source_id,
                 "knowledge_source_id": args.knowledge_source_id or args.source_id}
        if memory_enabled:
            ready["memory"] = runtime.memory_status()
        _emit("ready", ready, True)
    else:
        print(_COMMANDS, flush=True)
        if memory_enabled:
            print("/remember writes and saves one structured fact; ordinary messages are not automatically learned.", flush=True)
            print("The memory profile answers the current question without conversation history.", flush=True)
    while True:
        try:
            text = input("" if args.json else f"{session}/{args.source_id}> ")
            if not text.strip():
                continue
            if text.startswith("/"):
                parts = shlex.split(text)
                command = parts[0]
                if command == "/quit" and len(parts) == 1:
                    break
                if command == "/new" and len(parts) == 1:
                    session = store.new_session()
                    _emit("session", {"session_id": session}, args.json, f"Session: {session}")
                elif command == "/clear" and len(parts) == 1:
                    store.clear_session(session)
                    _emit("session_cleared", {"session_id": session}, args.json, "Session cleared.")
                elif command == "/person" and len(parts) == 2:
                    participant = store.add_person(int(parts[1]))
                    _emit("participant", asdict(participant), args.json, str(participant))
                elif command == "/source" and len(parts) == 2:
                    if parts[1] not in store.participants:
                        raise SessionError("Unknown source_id")
                    args.source_id = parts[1]
                    _emit("source", {"source_id": args.source_id}, args.json, f"Speaker: {args.source_id}")
                elif command == "/knowledge-source" and len(parts) == 2:
                    if not memory_enabled:
                        raise ValueError("Neural Memory is not attached")
                    if parts[1] not in runtime.memory_status()["source_ids"]:
                        raise SessionError("Unknown memory source_id")
                    args.knowledge_source_id = parts[1]
                    _emit("knowledge_source", {"knowledge_source_id": parts[1]}, args.json,
                          f"Knowledge source: {parts[1]}")
                elif command == "/remember" and len(parts) == 4:
                    receipt = runtime.remember(parts[1], parts[2], parts[3])
                    _emit("remembered", {"receipt": receipt}, args.json,
                          f"Remembered and saved for {parts[1]}.")
                elif command == "/memory" and len(parts) == 2:
                    action = parts[1]
                    if action in {"on", "off", "empty"}:
                        runtime.set_memory_mode(action)
                        status = runtime.memory_status()
                    elif action == "status":
                        status = runtime.memory_status()
                    elif action == "save":
                        status = runtime.save_memory()
                    else:
                        raise ValueError("Usage: /memory on|off|empty|status|save")
                    _emit("memory_status", {"memory": status}, args.json,
                          json.dumps(status, ensure_ascii=False))
                elif command == "/save" and len(parts) >= 2:
                    # Preserve Stage 1's unquoted paths with spaces, while also
                    # accepting the ordinary single quoted-path spelling.
                    path = parts[1] if len(parts) == 2 else text[len("/save "):].strip()
                    store.save(path)
                    _emit("transcript_saved", {"path": path}, args.json, f"Transcript saved: {path}")
                else:
                    raise ValueError("Unknown or malformed command. " + _COMMANDS)
            else:
                options = ({"knowledge_source_id": args.knowledge_source_id} if memory_enabled else {})
                result = runtime.respond(session, args.source_id, text, args.max_new_tokens, **options)
                _emit("response", _response_record(result), args.json, result.text)
        except (EOFError, KeyboardInterrupt):
            break
        except (ValueError, RuntimeError, OSError) as exc:
            if args.json:
                _emit("error", {"error": str(exc)}, True)
            else:
                print(f"Error: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    memory_enabled = args.memory_bridge is not None
    if memory_enabled != (args.memory_state is not None):
        parser.error("--memory-bridge and --memory-state must be supplied together")
    if not memory_enabled and (args.memory_mode is not None or args.knowledge_source_id is not None):
        parser.error("--memory-mode and --knowledge-source-id require attached Neural Memory")
    if memory_enabled and not 1 <= args.max_new_tokens <= 16:
        parser.error("The supported memory generation budget is 1..16 tokens")
    runtime = None
    try:
        store = SessionStore.load(args.restore_sessions) if args.restore_sessions else SessionStore()
        for number in args.person:
            store.add_person(number)
        session = args.session or store.new_session()
        store.messages(session)
        runtime = AliceRuntime(load_qwen(args.model_path, args.device, args.dtype), store,
                               args.max_context_tokens)
        if memory_enabled:
            runtime.attach_memory(args.memory_bridge, args.memory_state, model_path=args.model_path)
            runtime.set_memory_mode(args.memory_mode or "on")
        if args.source_id not in store.participants:
            raise SessionError("Unknown source_id")
        if args.knowledge_source_id is not None and args.knowledge_source_id not in runtime.memory_status()["source_ids"]:
            raise SessionError("Unknown memory source_id")
        if args.message:
            for text in args.message:
                options = ({"knowledge_source_id": args.knowledge_source_id} if memory_enabled else {})
                result = runtime.respond(session, args.source_id, text, args.max_new_tokens, **options)
                print(json.dumps(_response_record(result), ensure_ascii=False), flush=True)
        else:
            _interactive(runtime, store, session, args)
        result = {"resources": runtime.resources()}
        if args.json and not args.message:
            result["event"] = "resources"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        if args.json:
            _emit("error", {"error": str(exc)}, True)
        else:
            print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
