"""Explicit participant IDs and disposable session transcripts, not neural memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4


class SessionError(ValueError):
    """Invalid participant, session operation, or persisted session state."""


@dataclass(frozen=True, slots=True)
class Participant:
    source_id: str
    label: str


@dataclass(frozen=True, slots=True)
class Message:
    source_id: str
    text: str


class SessionStore:
    """Single-owner session support with stable, explicit source identities.

    Participant roles are supplied by the caller, never inferred from language.
    Save/restore persists transcripts and IDs only; it does not acquire knowledge
    into parameters or implement the planned independent Neural Memory.
    """

    MAX_MESSAGE_CHARS = 65536
    _SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")

    def __init__(self) -> None:
        self._participants = {
            "agent": Participant("agent", "Agent"),
            "user": Participant("user", "User"),
        }
        self._sessions: dict[str, list[Message]] = {}
        self._closed: set[str] = set()

    @property
    def participants(self) -> Mapping[str, Participant]:
        return MappingProxyType(dict(self._participants))

    def add_person(self, number: int) -> Participant:
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise SessionError("Person number must be a positive integer")
        source_id = f"person:{number}"
        participant = Participant(source_id, f"Another Person #{number}")
        self._participants[source_id] = participant
        return participant

    def new_session(self, session_id: str | None = None) -> str:
        if session_id is None:
            session_id = uuid4().hex
        if not isinstance(session_id, str) or not self._SESSION_ID.fullmatch(session_id):
            raise SessionError("Invalid session ID")
        if session_id in self._sessions:
            raise SessionError("Session ID already exists")
        self._sessions[session_id] = []
        return session_id

    def _messages(self, session_id: str) -> list[Message]:
        if not isinstance(session_id, str) or session_id not in self._sessions:
            raise SessionError("Unknown session")
        if session_id in self._closed:
            raise SessionError("Session is closed")
        return self._sessions[session_id]

    def validate_message(self, source_id: str, text: str) -> Message:
        """Validate a prospective message without changing any session."""
        if not isinstance(source_id, str) or source_id not in self._participants:
            raise SessionError("Unknown source_id")
        if (not isinstance(text, str) or not text.strip()
                or len(text) > self.MAX_MESSAGE_CHARS):
            raise SessionError("Message must be nonempty and within the character limit")
        return Message(source_id, text)

    def add_message(self, session_id: str, source_id: str, text: str) -> Message:
        messages = self._messages(session_id)
        message = self.validate_message(source_id, text)
        messages.append(message)
        return message

    def append_result(
        self, session_id: str, incoming: Message, response_text: str,
    ) -> tuple[Message, ...]:
        """Commit a successful exchange after validating all data, or change nothing.

        An empty response (for immediate model EOS) stores only the incoming
        message. The caller runs generation before invoking this method.
        """
        messages = self._messages(session_id)
        if not isinstance(incoming, Message):
            raise SessionError("Incoming message must be a Message")
        validated = self.validate_message(incoming.source_id, incoming.text)
        if not isinstance(response_text, str):
            raise SessionError("Response text must be a string")
        exchange = (validated,)
        if response_text:
            exchange += (self.validate_message("agent", response_text),)
        messages.extend(exchange)
        return exchange

    def messages(self, session_id: str) -> tuple[Message, ...]:
        return tuple(self._messages(session_id))

    def clear_session(self, session_id: str) -> None:
        self._messages(session_id).clear()

    def close_session(self, session_id: str) -> None:
        self._messages(session_id)
        self._closed.add(session_id)

    def save(self, path: str | Path) -> None:
        """Atomically replace a session snapshot, preserving the old file on error."""
        target = Path(path)
        payload = {
            "version": 1,
            "participants": [
                {"source_id": p.source_id, "label": p.label}
                for p in self._participants.values()
            ],
            "sessions": [
                {"session_id": session_id, "closed": session_id in self._closed,
                 "messages": [{"source_id": m.source_id, "text": m.text}
                              for m in messages]}
                for session_id, messages in self._sessions.items()
            ],
        }
        temporary: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except (OSError, ValueError, TypeError) as exc:
            raise SessionError(f"Cannot save session state: {exc}") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> SessionStore:
        """Validate into a separate store; malformed state never partially loads."""
        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise SessionError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"),
                                 object_pairs_hook=unique_object)
            if (not isinstance(payload, dict)
                    or set(payload) != {"version", "participants", "sessions"}
                    or type(payload["version"]) is not int
                    or payload["version"] != 1
                    or not isinstance(payload["participants"], list)
                    or not isinstance(payload["sessions"], list)):
                raise SessionError("Invalid session snapshot envelope")
            store = cls()
            seen: set[str] = set()
            for participant in payload["participants"]:
                if (not isinstance(participant, dict)
                        or set(participant) != {"source_id", "label"}
                        or not isinstance(participant["source_id"], str)
                        or participant["source_id"] in seen):
                    raise SessionError("Invalid or duplicate participant")
                source_id = participant["source_id"]
                seen.add(source_id)
                if source_id not in {"agent", "user"}:
                    match = re.fullmatch(r"person:([1-9][0-9]*)", source_id)
                    if match is None:
                        raise SessionError("Invalid participant source_id")
                    store.add_person(int(match[1]))
                if participant["label"] != store._participants[source_id].label:
                    raise SessionError("Participant label does not match its source_id")
            if not {"agent", "user"} <= seen:
                raise SessionError("Agent and User identities are required")
            for session in payload["sessions"]:
                if (not isinstance(session, dict)
                        or set(session) != {"session_id", "closed", "messages"}
                        or not isinstance(session["session_id"], str)
                        or type(session["closed"]) is not bool
                        or not isinstance(session["messages"], list)):
                    raise SessionError("Invalid session record")
                session_id = store.new_session(session["session_id"])
                for message in session["messages"]:
                    if not isinstance(message, dict) or set(message) != {"source_id", "text"}:
                        raise SessionError("Invalid message record")
                    store.add_message(session_id, message["source_id"], message["text"])
                if session["closed"]:
                    store.close_session(session_id)
            return store
        except SessionError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise SessionError(f"Cannot load session state: {exc}") from exc
