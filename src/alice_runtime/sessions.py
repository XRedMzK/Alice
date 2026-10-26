"""Explicit participant IDs and disposable session transcripts, not neural memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import stat
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from .export import TranscriptExportDirectory, TranscriptExportError


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
    Limits affect only disposable UI history. At the session cap, evict the
    oldest retained closed session in creation order; never evict an open one.
    Participant IDs are not evicted or reassigned. Whole-exchange trimming and
    saturating process-local counters do not save or reset acquired experience.
    """

    MAX_MESSAGE_CHARS = 65536
    MAX_SESSIONS = 16
    MAX_MESSAGES_PER_SESSION = 64
    MAX_SESSION_CHARS = 262144
    MAX_PARTICIPANTS = 32
    MAX_PERSON_NUMBER = 999999999
    MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024
    MAX_DROP_COUNTER = (1 << 63) - 1
    _SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")

    def __init__(self) -> None:
        self._participants = {
            "agent": Participant("agent", "Agent"),
            "user": Participant("user", "User"),
        }
        self._sessions: dict[str, list[Message]] = {}
        self._closed: set[str] = set()
        # A retained ID may be reused after eviction. Its new RAM-only identity
        # must not revive a confirmation created by the former session object.
        self._incarnations: dict[str, str] = {}
        # Unit lengths retain whole exchanges, not an unbounded second history.
        self._units: dict[str, list[int]] = {}
        self._chars: dict[str, int] = {}
        self._drops = {"sessions_evicted": 0, "messages_dropped": 0, "chars_dropped": 0}

    def _record_drop(self, messages, *, session=False):
        increments = {"sessions_evicted": int(session), "messages_dropped": len(messages),
                      "chars_dropped": sum(len(message.text) for message in messages)}
        for name, amount in increments.items():
            self._drops[name] = min(self.MAX_DROP_COUNTER, self._drops[name] + amount)

    def retention_status(self) -> dict:
        """Bounded process-local retention counters; never acquired memory state."""
        return {
            "sessions": len(self._sessions), "open_sessions": len(self._sessions) - len(self._closed),
            "participants": len(self._participants),
            "retained_messages": sum(len(messages) for messages in self._sessions.values()),
            "retained_chars": sum(self._chars.values()),
            "limits": {"sessions": self.MAX_SESSIONS, "messages_per_session": self.MAX_MESSAGES_PER_SESSION,
                       "chars_per_session": self.MAX_SESSION_CHARS, "participants": self.MAX_PARTICIPANTS},
            "drops": dict(self._drops),
        }

    @property
    def participants(self) -> Mapping[str, Participant]:
        return MappingProxyType(dict(self._participants))

    def add_person(self, number: int) -> Participant:
        if (isinstance(number, bool) or not isinstance(number, int)
                or not 1 <= number <= self.MAX_PERSON_NUMBER):
            raise SessionError("Person number must be an integer from 1 through 999999999")
        source_id = f"person:{number}"
        if source_id in self._participants:
            return self._participants[source_id]
        if len(self._participants) >= self.MAX_PARTICIPANTS:
            raise SessionError("Participant limit reached; existing source IDs are retained")
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
        if len(self._sessions) >= self.MAX_SESSIONS:
            oldest_closed = next((sid for sid in self._sessions if sid in self._closed), None)
            if oldest_closed is None:
                raise SessionError("Session limit reached; close an old session before creating another")
            removed = self._sessions.pop(oldest_closed)
            self._closed.remove(oldest_closed)
            self._units.pop(oldest_closed)
            self._chars.pop(oldest_closed)
            self._incarnations.pop(oldest_closed)
            self._record_drop(removed, session=True)
        self._sessions[session_id] = []
        self._incarnations[session_id] = uuid4().hex
        self._units[session_id] = []
        self._chars[session_id] = 0
        return session_id

    def _messages(self, session_id: str) -> list[Message]:
        if not isinstance(session_id, str) or session_id not in self._sessions:
            raise SessionError("Unknown session")
        if session_id in self._closed:
            raise SessionError("Session is closed")
        return self._sessions[session_id]

    def session_incarnation(self, session_id: str) -> str:
        """Opaque RAM identity of this open session; absent from v1 snapshots."""
        self._messages(session_id)
        return self._incarnations[session_id]

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
        # Low-level sequential assistant additions belong to the preceding turn.
        self._append_exchange(session_id, (message,), merge_last=source_id == "agent" and bool(messages))
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
        self._append_exchange(session_id, exchange)
        return exchange

    def _append_exchange(self, session_id, exchange, *, merge_last=False):
        messages = self._messages(session_id)
        units = list(self._units[session_id])
        last = units[-1] if merge_last else 0
        new_unit = messages[-last:] + list(exchange) if last else list(exchange)
        if (len(new_unit) > self.MAX_MESSAGES_PER_SESSION
                or sum(len(message.text) for message in new_unit) > self.MAX_SESSION_CHARS):
            raise SessionError("One exchange exceeds the retained session limit; history was not changed")
        candidate = messages + list(exchange)
        if merge_last:
            units[-1] += len(exchange)
        else:
            units.append(len(exchange))
        total_chars = self._chars[session_id] + sum(len(message.text) for message in exchange)
        removed = []
        while len(candidate) > self.MAX_MESSAGES_PER_SESSION or total_chars > self.MAX_SESSION_CHARS:
            length = units.pop(0)
            dropped, candidate = candidate[:length], candidate[length:]
            total_chars -= sum(len(message.text) for message in dropped)
            removed.extend(dropped)
        self._sessions[session_id] = candidate
        self._units[session_id] = units
        self._chars[session_id] = total_chars
        self._record_drop(removed)

    def messages(self, session_id: str) -> tuple[Message, ...]:
        return tuple(self._messages(session_id))

    def clear_session(self, session_id: str) -> None:
        self._messages(session_id).clear()
        self._units[session_id].clear()
        self._chars[session_id] = 0

    def close_session(self, session_id: str) -> None:
        self._messages(session_id)
        self._closed.add(session_id)

    def save(self, name: str | Path, *, exporter: TranscriptExportDirectory | None = None) -> Path:
        """Create a new transcript in an explicitly configured export directory.

        Existing files are never replaced. Passing an arbitrary path without an
        exporter is deliberately unsupported, including through this public API.
        This support snapshot does not save or modify acquired Neural Memory.
        """
        if not isinstance(exporter, TranscriptExportDirectory):
            raise SessionError("Transcript export is disabled; configure a separate --transcript-dir and use a new leaf filename")
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
        try:
            content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            if len(content) > self.MAX_SNAPSHOT_BYTES:
                raise SessionError("Session snapshot exceeds the 20 MiB restore limit; clear old transcript history")
            return exporter.write(name, content)
        except (TranscriptExportError, OSError, ValueError, TypeError) as exc:
            raise SessionError(f"Cannot save session state: {exc}") from exc

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
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise SessionError("Session snapshot must be a regular local file")
                if info.st_size > cls.MAX_SNAPSHOT_BYTES:
                    raise SessionError("Session snapshot exceeds the 20 MiB byte limit")
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = None
                    raw = handle.read(cls.MAX_SNAPSHOT_BYTES + 1)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            if len(raw) > cls.MAX_SNAPSHOT_BYTES:
                raise SessionError("Session snapshot exceeds the 20 MiB byte limit")
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
            if (not isinstance(payload, dict)
                    or set(payload) != {"version", "participants", "sessions"}
                    or type(payload["version"]) is not int
                    or payload["version"] != 1
                    or not isinstance(payload["participants"], list)
                    or not isinstance(payload["sessions"], list)):
                raise SessionError("Invalid session snapshot envelope")
            # Check collection/character limits before populating any store.
            # Restore rejects oversized history; it never silently trims it.
            if len(payload["participants"]) > cls.MAX_PARTICIPANTS or len(payload["sessions"]) > cls.MAX_SESSIONS:
                raise SessionError("Session snapshot exceeds participant/session retention limits")
            for record in payload["sessions"]:
                if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
                    raise SessionError("Invalid session record")
                if len(record["messages"]) > cls.MAX_MESSAGES_PER_SESSION:
                    raise SessionError("Session snapshot exceeds message retention limit")
                characters = 0
                for item in record["messages"]:
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                        raise SessionError("Invalid message record")
                    characters += len(item["text"])
                    if characters > cls.MAX_SESSION_CHARS:
                        raise SessionError("Session snapshot exceeds character retention limit")
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
                    match = re.fullmatch(r"person:([1-9][0-9]{0,8})", source_id)
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
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise SessionError(f"Cannot load session state: {exc}") from exc
