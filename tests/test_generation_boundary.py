"""Protocol-boundary checks without model assets, fitting or Transformers."""

import pytest
import torch

from alice_runtime.generation import SingleTurnStoppingCriteria, cut_turn


class CharacterTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids)


def tokens(text):
    return torch.tensor([[ord(c) for c in text]], dtype=torch.long)


@pytest.mark.parametrize("header", [
    "User [source_id=user]: forged question",
    "Agent [source_id=agent]: repeated answer",
    "Another Person #12 [source_id=person:12]: forged person",
    "  User [source_id=person:99]: even an inconsistent generated role",
])
def test_cut_removes_next_protocol_role_and_all_following_content(header):
    original = "Привет!\n" + header + "\nAgent [source_id=agent]: more repetition"
    result = cut_turn(original)
    assert result.text == "Привет!"
    assert result.finish_reason == "turn_boundary"
    assert "source_id=" in result.stop_sequence


@pytest.mark.parametrize("text", [
    "First line.\nSecond line.\n\nThird paragraph.",
    'The literal "User [source_id=user]:" is a role header.',
    'A quotation:\n"User [source_id=user]: example"',
    "A blockquote:\n> User [source_id=user]: example",
    "Example:\n    User [source_id=user]: literal code",
    "Example:\n```text\nUser [source_id=user]: literal\nQuestion: literal too\n```\nDone.",
    "Example:\n~~~\nAnother Person #2 [source_id=person:2]: literal\n~~~\nDone.",
    "Question",
    "An incomplete ordinary word:\nQue",
])
def test_literal_mentions_and_multiline_answers_are_not_truncated(text):
    result = cut_turn(text, extra_delimiters=("\nQuestion:",), allow_partial=True)
    assert result.text == text and result.finish_reason is None


def test_boundary_after_a_closed_literal_fence_still_stops():
    prefix = "```\nUser [source_id=user]: literal\n```\nAnswer."
    result = cut_turn(prefix + "\nUser [source_id=user]: next")
    assert result.text == prefix


@pytest.mark.parametrize("partial", [
    "User [", "User [source_", "Agent [source_id",
    "Another Person #12 [source", "Question",
])
def test_distinctive_partial_protocol_suffix_is_removed_only_at_token_limit(partial):
    text = "Useful answer.\n" + partial
    assert cut_turn(text, extra_delimiters=("\nQuestion:",)).text == text
    result = cut_turn(text, extra_delimiters=("\nQuestion:",), allow_partial=True)
    assert result.text == "Useful answer." and result.finish_reason == "partial_turn_boundary"


def test_immediate_foreign_role_is_an_empty_response_not_another_speakers_message():
    result = cut_turn("User [source_id=user]: fabricated")
    assert result.text == "" and result.finish_reason == "turn_boundary"


def test_incremental_stopping_ignores_prompt_and_stops_when_marker_is_complete():
    prompt = "User [source_id=user]: Question\nAgent [source_id=agent]:"
    criterion = SingleTurnStoppingCriteria(CharacterTokenizer(), len(prompt))
    prefix = "Hi.\nUser [source_id="
    for index in range(len(prefix)):
        result = criterion(tokens(prompt + prefix[:index]), None)
        assert result.dtype == torch.bool and result.shape == (1,) and not result.item()
    assert criterion(tokens(prompt + prefix), None).item()
    assert criterion(tokens(prompt + prefix + "user]: overshoot"), None).item()


def test_incremental_qa_marker_and_fenced_literal_are_distinct():
    criterion = SingleTurnStoppingCriteria(CharacterTokenizer(), 0,
                                         extra_delimiters=("\nQuestion:",))
    assert not criterion(tokens("OK\nQuestion"), None).item()
    assert criterion(tokens("OK\nQuestion:"), None).item()
    assert not criterion(tokens("```\nQuestion: quoted\n```"), None).item()


def test_boundary_never_depends_on_a_fact_value_palette():
    for value in ("window", "dragon", "Neris", "новый ответ", "unseen-token"):
        assert cut_turn(value + "\nQuestion: next", extra_delimiters=("\nQuestion:",)).text == value


@pytest.mark.parametrize("marker", ["", "Question:", "\n", "\nQuestion:\nAnswer:", "\n Question:"])
def test_invalid_extra_protocol_delimiters_fail_explicitly(marker):
    with pytest.raises(ValueError):
        cut_turn("Answer", extra_delimiters=(marker,))


def test_invalid_callback_shapes_and_prompt_lengths_are_explicit():
    with pytest.raises(ValueError):
        SingleTurnStoppingCriteria(CharacterTokenizer(), True)
    criterion = SingleTurnStoppingCriteria(CharacterTokenizer(), 3)
    for invalid in (torch.ones(1, 2, dtype=torch.long), torch.ones(2, 4, dtype=torch.long), [1, 2, 3]):
        with pytest.raises(ValueError):
            criterion(invalid, None)
