"""Tests for the lm_eval report driver, on results files shaped like lm_eval's."""

import json
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from domyn_eval.drivers.report.base import DuplicateBenchmarkError
from domyn_eval.drivers.report.lmeval import LMEvalReportDriver

RESULTS = {
    "results": {
        "niah_single_1": {
            "name": "niah_single_1",
            "alias": "niah_single_1",
            "sample_len": 500,
            "4096,none": 1.0,
            "4096_stderr,none": "N/A",
        },
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.7,
            "exact_match_stderr,strict-match": 0.01,
            "exact_match,flexible-extract": 0.75,
            "exact_match_stderr,flexible-extract": 0.012,
        },
        "ruler": {
            "alias": "ruler",
            "4096,none": 0.9,
            "4096_stderr,none": "N/A",
            "sample_count": {"4096,none": 6500},
        },
    },
    "n-samples": {
        "niah_single_1": {"original": 3000, "effective": 500},
        "gsm8k": {"original": 1319, "effective": 1319},
    },
    "higher_is_better": {
        "niah_single_1": {"4096": True},
        "gsm8k": {"exact_match": True},
        "ruler": {"4096": None},  # a group whose tasks disagree
    },
}


def _write_run(run_dir: Path, results: dict | None = None) -> None:
    """Write a ``results_<date>.json`` into ``run_dir``: ``results``, default ``RESULTS``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results_2026-11-04T10-00-00.json").write_text(json.dumps(results or RESULTS))


def _rows(df: pd.DataFrame) -> dict[tuple[str, str], dict[Hashable, Any]]:
    """The report's rows keyed by ``(benchmark, metric)``."""
    return {(row["benchmark"], row["metric"]): row for row in df.to_dict("records")}


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir, where the tests write the model's runs."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


def _task_dir(runs_root: Path, run_id: str, task: str = "ruler") -> Path:
    """Model ``m``'s folder for ``task`` in run ``run_id``: ``<runs>/m/<run_id>/results/lm_eval/<task>``."""
    return runs_root / "m" / run_id / "results" / "lm_eval" / task


def test_report_reads_tasks_groups_and_filters(runs_root: Path) -> None:
    _write_run(_task_dir(runs_root, "run1"))
    _task_dir(runs_root, "no_results").mkdir(parents=True)

    df = LMEvalReportDriver("m").report()

    assert set(df["run"]) == {"run1"}
    rows = _rows(df)
    assert rows["niah_single_1", "4096"]["score"] == pytest.approx(1.0)
    assert pd.isna(rows["niah_single_1", "4096"]["stderr"])  # "N/A"
    assert rows["niah_single_1", "4096"]["samples"] == 500
    assert rows["gsm8k", "strict-match/exact_match"]["stderr"] == pytest.approx(0.01)
    assert rows["gsm8k", "flexible-extract/exact_match"]["score"] == pytest.approx(0.75)
    assert rows["ruler", "4096"]["samples"] == 6500  # a group's own sample_count
    assert all(df["higher_is_better"])
    # metadata (name, alias, sample_len) and stderr keys are not rows
    assert len(df) == 4
    assert tuple(df.columns) == LMEvalReportDriver.REPORT_COLUMNS


def test_report_rejects_a_task_in_two_runs(runs_root: Path) -> None:
    _write_run(_task_dir(runs_root, "run1"))
    _write_run(_task_dir(runs_root, "run2"))

    with pytest.raises(DuplicateBenchmarkError, match="run1, run2"):
        LMEvalReportDriver("m").report()


def _ruler_run(run_dir: Path, length: int) -> None:
    """A RULER run at one sequence length: lm_eval reports the length as the metric."""
    _write_run(
        run_dir,
        {
            "results": {"ruler": {"alias": "ruler", f"{length},none": 0.9}},
            "n-samples": {"ruler": {"original": 500, "effective": 500}},
        },
    )


def test_report_allows_one_task_split_over_runs_by_length(runs_root: Path) -> None:
    _ruler_run(_task_dir(runs_root, "run64k"), 65536)
    _ruler_run(_task_dir(runs_root, "run128k"), 131072)

    df = LMEvalReportDriver("m").report()

    assert dict(zip(df["metric"], df["run"], strict=True)) == {"65536": "run64k", "131072": "run128k"}


def test_report_rejects_the_same_length_in_two_runs(runs_root: Path) -> None:
    _ruler_run(_task_dir(runs_root, "a"), 65536)
    _ruler_run(_task_dir(runs_root, "b"), 65536)
    _ruler_run(_task_dir(runs_root, "c"), 131072)

    # only the repeated length clashes; the benchmark is split, so the length is named
    with pytest.raises(DuplicateBenchmarkError, match=r"ruler \(65536\) in a, b"):
        LMEvalReportDriver("m").report()


def test_report_keeps_lower_is_better_metrics(runs_root: Path) -> None:
    _write_run(
        _task_dir(runs_root, "run1", "wikitext"),
        {
            "results": {"wikitext": {"alias": "wikitext", "word_perplexity,none": 12.5, "bits_per_byte,none": 0.6}},
            "higher_is_better": {"wikitext": {"word_perplexity": False, "bits_per_byte": False}},
            "n-samples": {"wikitext": {"original": 62, "effective": 62}},
        },
    )

    df = LMEvalReportDriver("m").report()

    assert not df["higher_is_better"].any()
