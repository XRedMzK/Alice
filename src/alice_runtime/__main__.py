"""Run a short local completion, or a small synchronous terminal session."""

import argparse
from dataclasses import asdict
import json
import sys

from .bridge import BridgeError
from .loading import LoadingError, load_qwen
from .runtime import AliceRuntime, RuntimeInputError
from .sessions import SessionError, SessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Existing local snapshot directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--person", type=int, action="append", default=[])
    parser.add_argument("--source-id", default="user")
    parser.add_argument("--session", help="Existing session from --restore-sessions; omit to create a new one")
    parser.add_argument("--restore-sessions", help="Explicit transcript/identity restore, not neural memory")
    parser.add_argument("--message", action="append", help="Generate once per message; otherwise interactive")
    args = parser.parse_args()
    try:
        store = SessionStore.load(args.restore_sessions) if args.restore_sessions else SessionStore()
        for number in args.person:
            store.add_person(number)
        session = args.session or store.new_session()
        store.messages(session)  # Validate an explicitly requested existing session.
        runtime = AliceRuntime(load_qwen(args.model_path, args.device, args.dtype), store,
                               args.max_context_tokens)
        if args.message:
            for text in args.message:
                result = runtime.respond(session, args.source_id, text, args.max_new_tokens)
                print(json.dumps(asdict(result), ensure_ascii=False))
        else:
            print("Commands: /new, /clear, /person N, /source ID, /save PATH, /quit")
            while True:
                try:
                    text = input(f"{session}/{args.source_id}> ")
                    if text == "/quit":
                        break
                    if text == "/new":
                        session = store.new_session()
                    elif text == "/clear":
                        store.clear_session(session)
                    elif text.startswith("/person "):
                        print(store.add_person(int(text[8:])))
                    elif text.startswith("/source "):
                        source = text[8:]
                        if source not in store.participants:
                            raise SessionError("Unknown source_id")
                        args.source_id = source
                    elif text.startswith("/save "):
                        store.save(text[6:])
                    elif text.strip():
                        print(runtime.respond(session, args.source_id, text, args.max_new_tokens).text)
                except (EOFError, KeyboardInterrupt):
                    break
                except (ValueError, RuntimeError) as exc:
                    print(f"Error: {exc}", file=sys.stderr)
        print(json.dumps({"resources": runtime.resources()}, ensure_ascii=False))
        return 0
    except (LoadingError, SessionError, BridgeError, RuntimeInputError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
