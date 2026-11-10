import pytest

from domyn_eval.drivers.stat.loop import Loop, find_loop

# Real prose rather than "abcabc", which would hide period bugs behind its own periodicity.
PROSE = (
    "The capital of France is Paris, a city known for its museums and its river. "
    "Historians note the settlement grew around an island in the Seine before its "
    "later expansion followed old Roman roads north and east toward the coast."
)
LOOP = "I cannot answer that. "
SENTENCE = "Let me reconsider the problem again. "


def test_an_output_ending_in_a_repeated_sentence_loops() -> None:
    assert find_loop(PROSE + LOOP * 30) == Loop(period=22, repeats=30, unit=LOOP)


def test_a_loop_cut_mid_unit_by_the_token_budget_still_loops() -> None:
    assert find_loop(LOOP * 30 + LOOP[:9]) == Loop(period=22, repeats=30, unit=LOOP)


def test_a_repetition_the_output_moves_on_from_is_no_loop() -> None:
    assert find_loop(LOOP * 30 + PROSE) is None


def test_a_loop_is_its_shortest_unit() -> None:
    assert find_loop(PROSE + "ha" * 100) == Loop(period=2, repeats=100, unit="ha")


def test_a_long_unit_loops_up_to_the_longest_period() -> None:
    block = "".join(chr(0x4E00 + i) for i in range(1100))  # 1100 distinct characters

    assert find_loop(block * 5) is None
    assert find_loop(block * 5, max_period=2048) == Loop(period=1100, repeats=5, unit=block)


@pytest.mark.parametrize("text", [PROSE, "", "ab" * 5])
def test_prose_and_short_text_do_not_loop(text: str) -> None:
    assert find_loop(text) is None


def test_a_loop_repeats_at_least_min_repeats_times() -> None:
    assert find_loop(SENTENCE * 3) is None
    assert find_loop(SENTENCE * 4) == Loop(period=37, repeats=4, unit=SENTENCE)


def test_a_loop_spans_at_least_min_chars() -> None:
    unit = "abcdefghijklmn "  # 15 characters

    assert find_loop(unit * 3) is None
    assert find_loop(unit * 4) is not None
    assert find_loop(unit * 4, min_chars=61) is None


@pytest.mark.parametrize("text", [PROSE + " " * 200, PROSE + "\n" * 300])
def test_trailing_whitespace_is_not_a_loop(text: str) -> None:
    assert find_loop(text) is None


def test_a_repeated_character_loops_only_below_min_period() -> None:
    assert find_loop(PROSE + "-" * 300) is None
    assert find_loop(PROSE + "-" * 300, min_period=1) == Loop(period=1, repeats=300, unit="-")
