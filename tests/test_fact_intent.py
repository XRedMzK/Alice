"""Bounded intent extraction and refusal controls, with no model or memory."""

import pytest

from alice_runtime.fact_intent import confirm_intent, normalize_fact_question, parse_memory_intent


@pytest.mark.parametrize("text,value", [
    ("Remember that the label of object atlas-0 is window.", "window"),
    ("Please remember: the assigned label of object atlas-0 is helmet.", "helmet"),
    ("Remember, object atlas-0 has the label museum.", "museum"),
    ("Запомни: метка объекта atlas-0 — cotton.", "cotton"),
    ("Пожалуйста, запомни, что метка объекта atlas-0 — window.", "window"),
    ("Запомни: у объекта atlas-0 метка helmet.", "helmet"),
    ("Исправь: метка объекта atlas-0 — dragon.", "dragon"),
    ("Correct: the label of object atlas-0 is dragon.", "dragon"),
])
def test_explicit_sentences_propose_same_canonical_cue(text, value):
    parsed = parse_memory_intent(text)
    assert parsed.kind == "candidate"
    assert parsed.object_id == "atlas-0" and parsed.value == value
    assert parsed.cue == "What is the assigned label of object atlas-0?"
    assert not hasattr(parsed, "source_id")


@pytest.mark.parametrize("text", [
    "The label of object atlas-0 is window.",
    "I told Alice to remember that the label of object atlas-0 is window.",
    '"Remember that the label of object atlas-0 is window."',
    "Don't remember that the label of object atlas-0 is window.",
    "Please do not remember that the label of object atlas-0 is window.",
    "Не запоминай: метка объекта atlas-0 — window.",
    "Remembering labels is useful.", "yes", "Подтверждаю", "",
])
def test_ordinary_quoted_and_negated_messages_are_not_intents(text):
    result = parse_memory_intent(text)
    assert result.kind == "no_intent"
    assert result.cue is result.value is result.object_id is None


@pytest.mark.parametrize("text", [
    "Remember that its label is window.",
    "Remember that the label of object it is window.",
    "Remember that the label of object atlas-0 is window or helmet.",
    "Remember that the label of object atlas-0 is not window.",
    "Remember that the label of object atlas-0 is maybe window.",
    "Remember that the label of object atlas-0 is window and object atlas-1 has label helmet.",
    "Remember that the label of object atlas-0 and atlas-1 is window.",
    "Запомни: метка объекта atlas-0 — window или helmet.",
    "Запомни: его метка — window.",
    "Remember that the label of object atlas-0 is window; the label of object atlas-1 is helmet.",
    'Remember that the label of object atlas-0 is "window" or "helmet".',
    "Remember that the label of object atlas-0 is window..",
])
def test_ambiguity_never_silently_selects_or_truncates_a_fact(text):
    result = parse_memory_intent(text)
    assert result.kind == "clarify" and result.message
    assert result.cue is result.value is result.object_id is None


def test_literal_case_and_multiword_value_are_preserved_for_runtime_validation():
    parsed = parse_memory_intent('Remember that the label of object Atlas_7 is "pale blue".')
    assert parsed.kind == "candidate"
    assert parsed.object_id == "Atlas_7" and parsed.value == "pale blue"
    assert parsed.cue == "What is the assigned label of object Atlas_7?"
    assert parse_memory_intent("Запомни: метка объекта Atlas_7 — «pale blue».").value == "pale blue"
    # The parser does not invent a value vocabulary or decide tokenizer coverage.
    assert parse_memory_intent("Запомни: метка объекта Atlas_7 — окно.").value == "окно"


def test_nonword_values_are_reported_unsupported_instead_of_guessed():
    for value in ("42", "🌳", "foo/bar"):
        result = parse_memory_intent(f"Remember that the label of object atlas-0 is {value}.")
        assert result.kind == "unsupported" and result.message
        assert result.value is result.cue is None


@pytest.mark.parametrize("text", [
    "Remember that Alice likes tea.",
    "Запомни: мне нравится чай.",
    "Remember that the label of object atlas-0 is window.\nIgnore the previous line.",
    "Remember " + "x" * 1024,
])
def test_unsupported_scope_has_explicit_explanation_and_no_fact(text):
    result = parse_memory_intent(text)
    assert result.kind == "unsupported" and result.message
    assert result.cue is result.value is None


@pytest.mark.parametrize("text,expected", [
    ("Yes", "yes"), ("confirm.", "yes"), ("Да, запомни!", "yes"),
    ("Подтверждаю", "yes"), ("no", "no"), ("Нет, не запоминай.", "no"),
    ("cancel", "no"), ("don't save", "no"), ("yes?", None),
    ("yes or no", None), ("yes, save something else", None), ("да!!", None),
])
def test_confirmation_is_only_an_exact_control_reply(text, expected):
    assert confirm_intent(text) == expected


def test_input_type_contract_is_explicit():
    with pytest.raises(TypeError):
        parse_memory_intent(None)
    assert confirm_intent(None) is None


@pytest.mark.parametrize("text", [
    "Какая метка у объекта Atlas_7?", "Скажи метку объекта Atlas_7.",
    "  какая метка у объекта Atlas_7  ",
])
def test_only_bounded_russian_queries_map_to_the_same_answer_blind_cue(text):
    assert normalize_fact_question(text) == "What is the assigned label of object Atlas_7?"


@pytest.mark.parametrize("text", [
    "Какая у него метка?", "Скажи метку объекта it.",
    "Какая метка у объекта atlas и beta?", "Какая метка у объекта user?",
    "Скажи метку объекта Atlas_7. Или другого?", "Какая метка у объекта Atlas_7?\nwindow",
    "What is the assigned label of object Atlas_7?", "Сколько объектов?",
])
def test_other_questions_and_ambiguous_references_are_untouched(text):
    assert normalize_fact_question(text) is None
