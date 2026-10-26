"""Small explicit-intent grammar for object labels, not general language NLU.

This pure parser proposes one temporary fact. It never chooses a source, reads
a model, writes memory or confirms on behalf of the caller. The runtime must
validate the value with its tokenizer and ask the actual speaker to confirm.
"""

from dataclasses import dataclass
import re


VERSION = "explicit_object_label_v1"


@dataclass(frozen=True)
class MemoryIntent:
    kind: str  # no_intent, candidate, clarify, unsupported
    message: str = ""
    language: str = "en"
    object_id: str | None = None
    value: str | None = None
    cue: str | None = None


_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
_PREFIXES = (
    ("en", re.compile(r"(?:please\s+)?(?:remember|correct)(?=$|[\s:,.!?])", re.I)),
    ("ru", re.compile(r"(?:пожалуйста\s*,?\s+)?(?:запомни|исправь)(?=$|[\s:,.!?])", re.I)),
)
_FACTS = {
    "en": tuple(re.compile(pattern, re.I) for pattern in (
        rf"(?:that\s+)?(?:the\s+)?(?:assigned\s+)?label\s+of\s+object\s+(?P<object_id>{_ID})\s+is\s+(?P<value>.+)",
        rf"(?:that\s+)?object\s+(?P<object_id>{_ID})\s+has\s+(?:the\s+)?(?:assigned\s+)?label\s+(?P<value>.+)",
    )),
    "ru": tuple(re.compile(pattern, re.I) for pattern in (
        rf"(?:что\s+)?метка\s+объекта\s+(?P<object_id>{_ID})(?:\s*(?:—|–|:)\s*|\s+(?:-|это|равна)\s+)(?P<value>.+)",
        rf"(?:что\s+)?у\s+объекта\s+(?P<object_id>{_ID})\s+метка\s+(?P<value>.+)",
    )),
}
_OBJECT_AMBIGUITY = frozenset({
    "it", "its", "they", "their", "them", "this", "that", "these", "those",
    "not", "no", "none", "never", "without", "except", "excluding", "unassigned",
    "previous", "previously", "former", "formerly", "current", "currently",
    "latest", "earlier", "before", "after", "old", "new", "instead", "other",
    "different", "another", "either", "neither", "both", "all", "and", "or",
    "nor", "versus", "vs", "than", "who", "whom", "whose", "which", "what",
    "where", "when", "why", "how", "person", "people", "participant",
    "participants", "user", "users", "agent", "agents", "speaker", "speakers",
    "source", "sources",
})
_VALUE_AMBIGUITY = frozenset({
    "or", "and", "nor", "either", "neither", "not", "maybe", "perhaps",
    "или", "либо", "и", "не", "возможно", "наверное", "может",
})
_QUOTES = {"\"": "\"", "'": "'", "«": "»"}
_RUSSIAN_QUESTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    rf"Какая\s+метка\s+у\s+объекта\s+(?P<object_id>{_ID})[?.!]?",
    rf"Скажи\s+метку\s+объекта\s+(?P<object_id>{_ID})[?.!]?",
))


def _words(text):
    return {word.casefold() for word in re.findall(r"[^\W_]+", text, re.UNICODE)}


def _message(kind, language):
    if language == "ru":
        return ("Уточните один объект с явным ID и одно значение метки, без вариантов. "
                "Например: «Запомни: метка объекта atlas-0 — значение»."
                if kind == "clarify" else
                "Пока поддерживаю только одну метку объекта с явным ID. "
                "Например: «Запомни: метка объекта atlas-0 — значение».")
    return ("Please specify one explicit object ID and one label value, without alternatives. "
            "For example: 'Remember that the label of object atlas-0 is VALUE.'"
            if kind == "clarify" else
            "I currently support only one object label with an explicit ID. "
            "For example: 'Remember that the label of object atlas-0 is VALUE.'")


def parse_memory_intent(text: str) -> MemoryIntent:
    """Extract only the documented affirmative object-label sentence forms.

    Bare facts, reported/quoted requests, negated requests and mention of another
    speaker do not grant an automatic write. Candidate values retain their case
    and content; even quoted multiword values require later one-token admission.
    Alternatives/multiple assertions return clarification without any guessed fact.
    """
    if not isinstance(text, str):
        raise TypeError("Memory intent input must be a string")
    stripped = text.strip()
    language = "ru" if re.search(r"[А-Яа-яЁё]", stripped) else "en"
    prefix = None
    for language_code, pattern in _PREFIXES:
        prefix = pattern.match(stripped)
        if prefix is not None:
            language = language_code
            break
    if prefix is None:
        return MemoryIntent("no_intent", language=language)
    if len(stripped) > 1024 or any(c in stripped for c in "\r\n\t"):
        return MemoryIntent("unsupported", _message("unsupported", language), language)
    body = stripped[prefix.end():].lstrip(" ,:")
    if body.endswith("."):
        body = body[:-1].rstrip()  # One sentence terminator, never arbitrary trimming.
    matched = next((m for pattern in _FACTS[language] if (m := pattern.fullmatch(body))), None)
    if matched is None:
        kind = "clarify" if re.search(r"\blabel\b|метк", body, re.I) else "unsupported"
        return MemoryIntent(kind, _message(kind, language), language)
    object_id, value = matched["object_id"], matched["value"].strip()
    quoted = bool(value and value[0] in _QUOTES and value.endswith(_QUOTES[value[0]]))
    if quoted:
        value = value[1:-1]
    if (_words(object_id) & _OBJECT_AMBIGUITY or not value or value != value.strip()
            or len(value) > 128 or any(c in value for c in ";:,\"'«».!?=")
            or (not quoted and _words(value) & _VALUE_AMBIGUITY)):
        return MemoryIntent("clarify", _message("clarify", language), language)
    if not value[0].isalpha() or any(not (c.isalnum() or c in " _-") for c in value):
        explanation = ("Пока поддерживаю словесное значение метки; при записи оно также "
                       "должно быть одним токеном модели." if language == "ru" else
                       "I currently support a word label value; it must also be one model token when written.")
        return MemoryIntent("unsupported", explanation, language)
    return MemoryIntent("candidate", language=language, object_id=object_id, value=value,
                        cue=f"What is the assigned label of object {object_id}?")


def confirm_intent(text: str) -> str | None:
    """Exact affirmative/negative replies; the runtime owns pending/source checks."""
    if not isinstance(text, str):
        return None
    reply = text.strip().casefold().replace("’", "'")
    if reply.endswith((".", "!")):
        reply = reply[:-1]
    if reply in {"yes", "yes, remember it", "confirm", "save it", "да", "да, запомни",
                 "да, сохрани", "подтверждаю", "верно"}:
        return "yes"
    if reply in {"no", "cancel", "do not save", "don't save", "no, don't remember",
                 "нет", "отмена", "не запоминай", "нет, не запоминай"}:
        return "no"
    return None


def normalize_fact_question(text: str) -> str | None:
    """Translate only two literal Russian label queries to the existing cue.

    This answer-blind UI adaptation stores no aliases, facts or sources and does
    not change the Bridge normalizer. Other questions, including the existing
    English forms, return None and remain the runtime's original question.
    """
    if not isinstance(text, str):
        raise TypeError("Fact question must be a string")
    if len(text) > 1024 or any(c in text for c in "\r\n\t"):
        return None
    for pattern in _RUSSIAN_QUESTIONS:
        matched = pattern.fullmatch(text.strip())
        if matched is not None:
            object_id = matched["object_id"]
            if not _words(object_id) & _OBJECT_AMBIGUITY:
                return f"What is the assigned label of object {object_id}?"
    return None
