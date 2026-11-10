from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from domyn_eval.config import Config
from domyn_eval.drivers.stat.base import BaseStatDriver, Sample, StatRow, StatsProfile
from tests.drivers.stat.helpers import normal_sample


class ListStats(BaseStatDriver):
    def __init__(self, run: tuple[str, str], samples: list[Sample]):
        super().__init__(*run)
        self._samples = samples

    @property
    def suite(self) -> str:
        return "list"

    def samples(self) -> Iterator[Sample]:
        yield from self._samples


def test_stats_count_each_benchmarks_samples_in_order(saved_run: tuple[str, str]) -> None:
    samples = [
        normal_sample(benchmark="b", answer="42", reasoning="because", completion_tokens=30),
        normal_sample(benchmark="a", answer="", truncated=True, completion_tokens=1024, scored=False),
        normal_sample(benchmark="b", answer="7", completion_tokens=50),
        normal_sample(benchmark="b", answer="  ", reasoning="Let me check again. " * 10, looping=True),
    ]

    rows = ListStats(saved_run, samples).stats()

    assert rows == [
        StatRow(
            benchmark="b",
            samples=3,
            truncated=0,
            reasoning=2,
            looping=1,
            empty=1,
            unscored=0,
            max_completion_tokens=50,
        ),
        StatRow(
            benchmark="a",
            samples=1,
            truncated=1,
            reasoning=0,
            looping=0,
            empty=1,
            unscored=1,
            max_completion_tokens=1024,
        ),
    ]


def test_a_sample_must_say_whether_it_is_truncated_looping_and_scored() -> None:
    with pytest.raises(ValidationError, match="3 validation errors"):
        Sample(benchmark="b")


def test_stats_of_a_run_without_samples_are_empty(saved_run: tuple[str, str]) -> None:
    assert ListStats(saved_run, []).stats() == []


def test_stats_read_the_runs_results_with_its_saved_config(
    config: Config, saved_run: tuple[str, str], tmp_path: Path
) -> None:
    config.generation.reasoning = False
    config.save(tmp_path / "test-model" / "r1" / "config.toml")

    driver = ListStats(saved_run, [])

    assert driver.results_dir == tmp_path / "test-model" / "r1" / "results" / "list"
    assert driver.config.generation.reasoning is False
    assert driver.profile == StatsProfile()


def test_stats_need_the_runs_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match=r"config\.toml"):
        ListStats(("test-model", "r1"), [])


HEALTHY = {
    "samples": 100,
    "truncated": 0,
    "reasoning": 100,
    "looping": 0,
    "empty": 0,
    "unscored": 0,
}


def _row(**counts: int) -> StatRow:
    return StatRow(benchmark="b", **{**HEALTHY, **counts})


@pytest.mark.parametrize(
    ("row", "reasoning", "warnings"),
    [
        (_row(), True, []),
        (_row(reasoning=0), False, []),
        (_row(truncated=5, empty=10), True, []),
        (_row(truncated=6), True, ["6% truncated"]),
        (_row(reasoning=40), True, ["only 40% reason, with reasoning on"]),
        (_row(reasoning=11), False, ["11% reason, with reasoning off"]),
        (_row(looping=5), True, []),
        (_row(looping=6), True, ["6% stuck in a loop"]),
        (_row(empty=11, unscored=2), True, ["11% empty", "2 unscored"]),
    ],
)
def test_warnings_flag_untrustworthy_scores(row: StatRow, reasoning: bool, warnings: list[str]) -> None:
    assert row.warnings(StatsProfile(), reasoning=reasoning) == warnings


def test_warnings_follow_the_configured_thresholds() -> None:
    assert _row(truncated=6).warnings(StatsProfile(max_truncated=0.1), reasoning=True) == []


def test_profile_loads_from_yaml_keeping_the_defaults_it_leaves_out(tmp_path: Path) -> None:
    path = tmp_path / "stats.yaml"
    path.write_text("max_truncated: 0.2\n")

    assert StatsProfile.load(path) == StatsProfile(max_truncated=0.2)


@pytest.mark.parametrize(("text", "error"), [("max_empty: 10", "less than or equal to 1"), ("colour: red", "Extra")])
def test_profile_rejects_invalid_thresholds(tmp_path: Path, text: str, error: str) -> None:
    path = tmp_path / "stats.yaml"
    path.write_text(text)

    with pytest.raises(ValidationError, match=error):
        StatsProfile.load(path)


def test_shipped_profile_holds_the_defaults() -> None:
    assert StatsProfile.load(Path(__file__).parents[3] / "configs" / "stats_profile.yaml") == StatsProfile()
