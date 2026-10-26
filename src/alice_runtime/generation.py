"""One-response boundaries for the existing local text-generation protocol.

Unquoted protocol headers on their own lines are reserved control text. Ordinary
multiline prose, inline mentions and quoted/code examples remain answer content.
This module changes neither model inputs nor generation probabilities.
"""

from dataclasses import dataclass
import re

import torch


_ROLE = re.compile(r"(?:User|Agent|Another Person #[1-9][0-9]*) \[source_id=")
_ROLE_PARTIAL = re.compile(r"(?:User|Agent|Another Person #[1-9][0-9]*) (\[.*)")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")


@dataclass(frozen=True)
class TurnText:
    text: str
    finish_reason: str | None = None
    stop_sequence: str | None = None


def _delimiters(extra_delimiters):
    result = []
    for marker in extra_delimiters:
        if (not isinstance(marker, str) or not marker.startswith("\n")
                or marker.count("\n") != 1 or not marker[1:]
                or marker[1:] != marker[1:].strip()):
            raise ValueError("Extra turn delimiters must be nonempty newline-prefixed lines")
        result.append(marker[1:])
    return tuple(result)


def cut_turn(text: str, *, extra_delimiters: tuple[str, ...] = (),
             allow_partial: bool = False) -> TurnText:
    """Remove a generated next-turn header and its tail before session append.

    Match standalone source headers, plus caller-supplied literal protocol lines.
    At token exhaustion, remove only distinctive incomplete headers: ``User [``
    through ``User [source_id`` and a full extra-delimiter word such as
    ``\nQuestion``. Short ambiguous prose prefixes and one-word answers remain.
    Headers inside fenced/indented code or quoted/inline text are not boundaries.
    """
    if not isinstance(text, str):
        raise TypeError("Generated text must be a string")
    markers = _delimiters(extra_delimiters)
    offset, fence = 0, None
    lines = text.splitlines(keepends=True)
    for number, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        fenced = _FENCE.fullmatch(line)
        if fence is not None:
            if (fenced is not None and fenced[1][0] == fence[0]
                    and len(fenced[1]) >= fence[1] and not fenced[2].strip()):
                fence = None
            offset += len(raw)
            continue
        if fenced is not None:
            fence = (fenced[1][0], len(fenced[1]))
            offset += len(raw)
            continue
        # Up to three spaces can indent a protocol line. Four spaces denote
        # an ordinary Markdown code example rather than a speaker transition.
        content = line.lstrip(" ") if len(line) - len(line.lstrip(" ")) <= 3 else line
        role = _ROLE.match(content)
        marker = role[0] if role else next((m for m in markers if content.startswith(m)), None)
        if marker is not None:
            return TurnText(text[:offset].rstrip(), "turn_boundary", "\n" + marker)
        if allow_partial and number == len(lines) - 1 and not raw.endswith(("\n", "\r")):
            role_prefix = _ROLE_PARTIAL.fullmatch(content)
            if role_prefix is not None and "[source_id=".startswith(role_prefix[1]):
                return TurnText(text[:offset].rstrip(), "partial_turn_boundary", "\n" + content)
            # Do not erase an answer consisting solely of the word Question.
            if offset and content and any(
                    m.startswith(content) and content == m.rstrip(":") for m in markers):
                return TurnText(text[:offset].rstrip(), "partial_turn_boundary", "\n" + content)
        offset += len(raw)
    return TurnText(text)


class SingleTurnStoppingCriteria:
    """Transformers-compatible callable for the runtime's one-item batch.

    Decode only newly generated tokens, never the prompt/history. The caller
    still applies cut_turn to the final suffix because the token completing a
    boundary can contain several characters, and fake/test generators may not
    execute stopping callbacks. No completed answer is committed by this class.
    """

    def __init__(self, tokenizer, prompt_length: int, *,
                 extra_delimiters: tuple[str, ...] = ()):
        if type(prompt_length) is not int or prompt_length < 0:
            raise ValueError("Prompt length must be a nonnegative integer")
        _delimiters(extra_delimiters)
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        self.extra_delimiters = tuple(extra_delimiters)

    def __call__(self, input_ids, scores=None, **kwargs):
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] < self.prompt_length):
            raise ValueError("Single-turn stopping expects one complete token sequence")
        suffix = input_ids[0, self.prompt_length:].tolist()
        text = self.tokenizer.decode(suffix, skip_special_tokens=True)
        stopped = cut_turn(text, extra_delimiters=self.extra_delimiters).finish_reason is not None
        return torch.tensor([stopped], dtype=torch.bool, device=input_ids.device)
