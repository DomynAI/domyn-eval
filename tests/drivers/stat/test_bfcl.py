"""Tests for the BFCL stat driver, on result files laid out as BFCL writes them."""

import json

import pytest

from domyn_eval.config import Config
from domyn_eval.drivers.stat.base import StatsProfile
from domyn_eval.drivers.stat.bfcl import BFCLStatDriver
from tests.drivers.stat.helpers import normal_sample


@pytest.fixture
def config(config: Config) -> Config:
    config.model.generation.context.max_tokens = 1024
    config.model.generation.context.max_model_len = 32768
    return config


@pytest.fixture
def driver(saved_run: tuple[str, str]) -> BFCLStatDriver:
    return BFCLStatDriver(*saved_run)


def _results(driver: BFCLStatDriver, category: str, entries: list[dict], *, scored: bool = True) -> None:
    folder = driver.results_dir / "single_turn" / "Org_Model-FC" / "live"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"BFCL_v4_{category}_result.json").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    if scored:
        (folder / f"BFCL_v4_{category}_score.json").write_text('{"accuracy": 1.0, "total_count": 1}\n')


def _entry(result: object = "[f()]", prompt: object = 100, completion: object = 50, **fields: object) -> dict:
    return {"id": "x", "result": result, "input_token_count": prompt, "output_token_count": completion, **fields}


def test_entries_are_read_per_test_category_set(driver: BFCLStatDriver) -> None:
    _results(driver, "live_simple", [_entry(reasoning_content="let me see"), _entry("")])

    assert list(driver.samples()) == [
        normal_sample(benchmark="single_turn", answer="[f()]", reasoning="let me see", completion_tokens=50),
        normal_sample(benchmark="single_turn", answer="", completion_tokens=50),
    ]


def test_a_multi_turn_entry_joins_its_steps(driver: BFCLStatDriver) -> None:
    entry = _entry(
        result=[["[cd()]", "[ls()]"], ["[pwd()]", None]],
        prompt=[[100, 200], [300, None]],  # the last step failed
        completion=[[10, 1024], [20, None]],
        reasoning_content=[["a", ""], ["b"]],
    )
    _results(driver, "multi_turn_base", [entry])

    (sample,) = driver.samples()

    assert sample == normal_sample(
        benchmark="single_turn",
        answer="[cd()]\n[ls()]\n[pwd()]",
        reasoning="a\nb",
        truncated=True,
        completion_tokens=1024,
    )


def test_only_scored_leaderboard_categories_count(driver: BFCLStatDriver) -> None:
    _results(driver, "live_multiple", [_entry()], scored=False)
    _results(driver, "format_sensitivity", [_entry()])

    (row,) = driver.stats()

    assert (row.samples, row.unscored) == (1, 1)


@pytest.mark.parametrize(
    ("max_tokens", "prompt", "completion", "truncated"),
    [
        (1024, 100, 1003, False),
        (1024, 100, 1004, True),
        (0, 32000, 503, True),  # the window leaves 512 tokens past the safety margin
        (0, 100, 30000, False),
    ],
)
def test_truncation_is_a_step_near_its_budget(
    driver: BFCLStatDriver, max_tokens: int, prompt: int, completion: int, truncated: bool
) -> None:
    driver.config.model.generation.context.max_tokens = max_tokens
    _results(driver, "live_simple", [_entry(prompt=prompt, completion=completion)])

    (sample,) = driver.samples()

    assert sample.truncated is truncated


def test_the_profile_sets_how_near_the_budget_counts(saved_run: tuple[str, str]) -> None:
    driver = BFCLStatDriver(*saved_run, StatsProfile(near_budget=0.5))
    _results(driver, "live_simple", [_entry(completion=600)])

    (sample,) = driver.samples()

    assert sample.truncated


@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [(["stop"], False), (["stop", "length"], True), ([], False)],
)
def test_a_recorded_finish_reason_tells_truncation(
    driver: BFCLStatDriver, finish_reason: list[str], truncated: bool
) -> None:
    # 1024 of 1024 tokens would be near the budget: the finish reason wins
    _results(driver, "live_simple", [_entry(completion=1024, finish_reason=finish_reason)])

    (sample,) = driver.samples()

    assert sample.truncated is truncated


def test_an_entry_loops_when_any_step_ends_in_a_loop(driver: BFCLStatDriver) -> None:
    loop = "I will call the tool. " * 30
    entry = _entry(result=[["[cd()]", "[ls()]"]], reasoning_content=[[loop, "done"]])
    _results(driver, "multi_turn_base", [entry, _entry(reasoning_content="I will call the tool.")])

    assert [sample.looping for sample in driver.samples()] == [True, False]
