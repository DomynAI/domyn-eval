# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Find doom loops: an output that ends stuck repeating the same text, typically until it ran out of tokens."""

from dataclasses import dataclass

_PROBE_LEN = 16


@dataclass(frozen=True)
class Loop:
    """The loop an output ends in.

    Attributes:
        period: The length of the repeating unit.
        repeats: How many whole units the loop holds.
        unit: The repeating unit.
    """

    period: int
    repeats: int
    unit: str


def find_loop(
    text: str, *, min_repeats: int = 4, min_chars: int = 60, min_period: int = 2, max_period: int = 1024
) -> Loop | None:
    """The loop ``text`` ends in, or None; a repetition the text moves on from is no loop.

    Args:
        text: The output to scan.
        min_repeats: The fewest units a loop repeats.
        min_chars: The shortest loop, in characters.
        min_period: The shortest unit; a single repeated character is mostly a separator line.
        max_period: The longest unit.
    """
    if len(text) < min_chars:
        return None
    end = len(text) - _PROBE_LEN
    tail = text[end:]
    # Each earlier occurrence of the tail, nearest first, is a candidate period: the first the text keeps
    # repeating with is the loop's shortest unit.
    start = end
    while (start := text.rfind(tail, max(0, end - max_period), start + _PROBE_LEN - 1)) != -1:
        period = end - start
        i = len(text) - period - 1
        while i >= 0 and text[i] == text[i + period]:
            i -= 1
        length = len(text) - 1 - i
        if length // period >= min_repeats and length >= min_chars:
            # The shortest unit decides, as its multiples are the same loop; a run of whitespace is formatting.
            unit = text[i + 1 : i + 1 + period]
            if period < min_period or not unit.strip():
                return None
            return Loop(period=period, repeats=length // period, unit=unit)
    return None
