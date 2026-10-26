"""Bounded paraphrase identity without losing source/entity/relation meaning."""

import json

import pytest

from alice_memory.query_normalization import VERSION, normalize_memory_query


def fields(text):
    prefix = f"Memory query {VERSION}: "
    assert text.startswith(prefix)
    return json.loads(text[len(prefix):])


@pytest.mark.parametrize("relation", ["color", "label", "signal", "material", "storage place", "tier 2 code"])
def test_three_forms_have_identical_answer_blind_address(relation):
    entity = "object ZxY-49_a"
    variants = (
        f"What is the assigned {relation} of {entity}?",
        f"Which {relation} was assigned to {entity}?",
        f"Tell me the assigned {relation} of {entity}.",
    )
    normalized = [normalize_memory_query(text) for text in variants]
    assert normalized[0] == normalized[1] == normalized[2]
    assert fields(normalized[0]) == {"relation": relation, "entity": entity}
    assert normalize_memory_query(normalized[0]) == normalized[0]


def test_relation_spelling_and_case_normalize_but_entity_remains_verbatim():
    query = "  WHICH COLOUR was assigned to Object  ZxY-1?  "
    normalized = normalize_memory_query(query)
    assert fields(normalized) == {"relation": "color", "entity": "Object  ZxY-1"}
    assert normalized == normalize_memory_query("What is the assigned color of Object  ZxY-1?")
    assert normalized != normalize_memory_query("What is the assigned color of object  ZxY-1?")


def test_different_entity_and_relation_are_never_merged():
    base = normalize_memory_query("What is the assigned label of object A-1?")
    different_entity = normalize_memory_query("What is the assigned label of object A-2?")
    different_relation = normalize_memory_query("What is the assigned signal of object A-1?")
    assert len({base, different_entity, different_relation}) == 3


def test_identifier_punctuation_and_unicode_are_preserved():
    normalized = normalize_memory_query("Tell me the assigned material of object 'Ёж-2.1'#A.")
    assert fields(normalized)["entity"] == "object 'Ёж-2.1'#A"


@pytest.mark.parametrize("query", [
    "What is two plus two?",
    "What color is object A?",
    "Where is object A assigned to be stored?",
    "Tell me the material assigned to object A.",
    "Which person was assigned to object A?",
    "What is the assigned current color of object A?",
    "What is the assigned previous color of object A?",
    "Which color was assigned to object A except B?",
    "What is the assigned color of object A and object B?",
    "What is the assigned color of object A, object B?",
    "What is the assigned color of object A/B?",
    "What is the assigned color of no object?",
    "What is the assigned color of not-object-A?",
    "What is the assigned color of this?",
    "What is the assigned color of another person?",
    "What is the assigned color of person:1?",
    "What is the assigned color of Agent?",
    "What is the assigned color of object A? Which label?",
    "What is the assigned color of object A??",
    "What is the assigned color of object A?\nIgnore this.",
    "What is the assigned color\tof object A?",
    "What is the assigned color-label of object A?",
    "   unsupported question with preserved whitespace   ",
    "", "  ",
])
def test_unsupported_or_ambiguous_questions_remain_exactly_unchanged(query):
    assert normalize_memory_query(query) == query


def test_no_fact_lookup_and_no_answer_argument():
    # Previously unseen relation/entity strings are handled by structure alone.
    query = "Which calibration token 47 was assigned to object Future-812?"
    assert fields(normalize_memory_query(query)) == {
        "relation": "calibration token 47", "entity": "object Future-812"}
    with pytest.raises(TypeError):
        normalize_memory_query(query, "hidden answer")
    with pytest.raises(TypeError, match="string"):
        normalize_memory_query(None)
