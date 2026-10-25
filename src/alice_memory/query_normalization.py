"""Bounded, answer-blind normalization for three relation/entity query forms.

This is interface normalization, not an open-ended semantic parser. It recognizes
only the documented English forms below. Unsupported or potentially ambiguous
questions are returned byte-for-byte unchanged for the caller's exact-query
fallback. No acquired entities, relations, values, sources, or aliases are kept.

The caller must apply the same version before both write and read key features,
and include VERSION in the Bridge semantics/basis. The original question should
still be used for language-model generation.
"""

from __future__ import annotations

import json
import re


VERSION = "bounded_relation_entity_v1"

_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"What\s+is\s+the\s+assigned\s+(?P<relation>.+?)\s+of\s+(?P<entity>.+?)[?.!]?",
    r"Which\s+(?P<relation>.+?)\s+was\s+assigned\s+to\s+(?P<entity>.+?)[?.!]?",
    r"Tell\s+me\s+the\s+assigned\s+(?P<relation>.+?)\s+of\s+(?P<entity>.+?)[?.!]?",
))

# Conservative rejection, not an attempt to interpret negation, time, identity,
# comparisons, or groups. Unknown literal identifiers containing these words
# retain the exact-query fallback too, rather than gaining an ambiguous alias.
_AMBIGUOUS_WORDS = frozenset({
    "not", "no", "none", "never", "without", "except", "excluding", "unassigned",
    "previous", "previously", "former", "formerly", "current", "currently",
    "latest", "earlier", "before", "after", "old", "new", "instead",
    "other", "different", "another", "either", "neither", "both", "all",
    "and", "or", "nor", "versus", "vs", "than",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "person", "people", "participant", "participants", "user", "users",
    "agent", "agents", "speaker", "speakers", "source", "sources",
    "it", "its", "they", "their", "them", "this", "that", "these", "those",
})


def _words(text: str) -> set[str]:
    return {word.casefold() for word in re.findall(r"[^\W_]+", text, re.UNICODE)}


def normalize_memory_query(text: str) -> str:
    """Return a stable relation/entity rendering, or the untouched input.

    Recognized forms are ``What is the assigned R of E?``, ``Which R was
    assigned to E?``, and ``Tell me the assigned R of E.`` (case-insensitive
    framing; optional final punctuation). Relation words are casefolded, with
    the generic spelling alias ``colour`` → ``color``. Entity spelling, case,
    spacing, and internal identifier punctuation are retained verbatim.

    Only alphanumeric relation words and conservative single-entity literals
    are admitted. The helper cannot resolve pronouns, participant selection,
    temporal qualifiers, multiple entities, or arbitrary paraphrases.
    """
    if not isinstance(text, str):
        raise TypeError("Memory query must be a string")
    if not text.strip() or len(text) > 1024 or any(c in text for c in "\r\n\t"):
        return text
    candidate = text.strip()
    for pattern in _PATTERNS:
        match = pattern.fullmatch(candidate)
        if match is None:
            continue
        relation = " ".join(match["relation"].split()).casefold()
        entity = match["entity"].strip()
        words = relation.split()
        if (not words or len(words) > 12 or len(relation) > 96
                or any(not word.isalnum() for word in words)):
            return text
        if (not entity or len(entity) > 256 or not any(c.isalnum() for c in entity)
                or any(not (c.isalnum() or c in " _-.'\"#") for c in entity)):
            return text
        if (_words(relation) | _words(entity)) & _AMBIGUOUS_WORDS:
            return text
        relation = " ".join("color" if word == "colour" else word for word in words)
        # Structured rendering avoids relation/entity delimiter ambiguities;
        # neither field contains an answer or comes from acquired fact storage.
        return f"Memory query {VERSION}: " + json.dumps(
            {"relation": relation, "entity": entity}, ensure_ascii=False,
            separators=(",", ":"))
    return text
