"""Tests for the BFCL report driver, on score files shaped like bfcl_eval's."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from domyn_eval.drivers.report.base import DuplicateBenchmarkError, ReportRow
from domyn_eval.drivers.report.bfcl import AGGREGATES, BFCLReportDriver

LIVE = {"live_simple": (0.8, 100), "live_multiple": (0.6, 300), "live_parallel": (0.5, 10)}
LIVE_ALL = {**LIVE, "live_parallel_multiple": (0.4, 20)}


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir, where the tests write the model's runs."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


def _write_scores(
    runs_root: Path,
    scores: dict[str, tuple[float, int]],
    run: str = "r1",
    category_set: str = "single_turn",
    model_folder: str = "org_model",
) -> Path:
    """Write one ``BFCL_v4_{category}_score.json`` per score, as ``bfcl evaluate`` lays them out."""
    run_dir = runs_root / "m" / run / "results" / "bfcl" / category_set
    for category, (accuracy, total) in scores.items():
        path = run_dir / model_folder / "live" / f"BFCL_v4_{category}_score.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"accuracy": accuracy, "correct_count": round(accuracy * total), "total_count": total}
        path.write_text(json.dumps(header) + "\n" + json.dumps({"id": "failed_entry"}) + "\n")
    return run_dir


def _rows(df: pd.DataFrame) -> dict[str, dict]:
    """The report's rows keyed by metric."""
    return {row["metric"]: row for row in df.to_dict("records")}


def test_a_category_set_reports_its_scored_aggregates(runs_root: Path) -> None:
    _write_scores(runs_root, {**LIVE_ALL, "live_relevance": (0.9, 16)})
    rows = _rows(BFCLReportDriver("m").report())
    assert set(rows) == {"live_accuracy", "relevance_accuracy"}
    relevance = rows["relevance_accuracy"]
    assert relevance["benchmark"] == "single_turn"
    assert relevance["run"] == "r1"
    assert relevance["score"] == pytest.approx(0.9)
    assert relevance["stderr"] == pytest.approx(math.sqrt(0.9 * 0.1 / 15))
    assert relevance["scorer"] == "relevance"


def test_live_weights_its_categories_by_their_samples(runs_root: Path) -> None:
    _write_scores(runs_root, LIVE_ALL)
    live = _rows(BFCLReportDriver("m").report())["live_accuracy"]
    total = sum(n for _, n in LIVE_ALL.values())
    assert live["score"] == pytest.approx(sum(acc * n for acc, n in LIVE_ALL.values()) / total)
    assert live["samples"] == total
    assert live["scorer"] == "ast"


def test_an_aggregate_needs_every_member(runs_root: Path) -> None:
    _write_scores(runs_root, {**LIVE, "live_relevance": (0.9, 16)})  # no live_parallel_multiple
    assert set(_rows(BFCLReportDriver("m").report())) == {"relevance_accuracy"}


def test_a_full_run_reports_every_aggregate(runs_root: Path) -> None:
    leaves = {m for a in AGGREGATES.values() for m in a.members} - set(AGGREGATES)
    _write_scores(runs_root, dict.fromkeys(leaves, (0.5, 10)), category_set="all")
    rows = _rows(BFCLReportDriver("m").report())
    assert rows["overall_accuracy"]["score"] == pytest.approx(0.5)
    assert pd.isna(rows["overall_accuracy"]["scorer"])  # its members use different checkers
    expected = {"overall", "non_live", "live", "multi_turn", "web_search", "memory", "irrelevance"}
    assert set(rows) == {f"{name}_accuracy" for name in expected}


def _row(score: float) -> ReportRow:
    return ReportRow(benchmark="b", metric="accuracy", score=score, stderr=0.1, samples=10, scorer="ast")


def test_overall_weights_its_members_by_bfcls_weights() -> None:
    # members: non_live, live, irrelevance_detection, multi_turn, agentic, weighted 10/10/10/30/40
    overall = BFCLReportDriver._aggregate("overall", [_row(1.0), _row(0), _row(0), _row(0), _row(0)])
    assert overall is not None
    assert overall.score == pytest.approx(0.1)
    assert overall.stderr == pytest.approx(0.1 * math.sqrt(0.1**2 + 0.1**2 + 0.1**2 + 0.3**2 + 0.4**2))


def test_non_leaderboard_categories_are_skipped(runs_root: Path) -> None:
    _write_scores(runs_root, {"format_sensitivity": (0.3, 10), "live_relevance": (0.9, 16)})
    assert set(_rows(BFCLReportDriver("m").report())) == {"relevance_accuracy"}


def test_a_category_scored_twice_in_a_run_raises(runs_root: Path) -> None:
    _write_scores(runs_root, {"live_relevance": (0.9, 16)}, model_folder="a")
    _write_scores(runs_root, {"live_relevance": (0.8, 16)}, model_folder="b")
    with pytest.raises(DuplicateBenchmarkError, match="live_relevance"):
        BFCLReportDriver("m").report()


def test_the_same_category_set_in_two_runs_raises(runs_root: Path) -> None:
    _write_scores(runs_root, {"live_relevance": (0.9, 16)}, run="r1")
    _write_scores(runs_root, {"live_relevance": (0.8, 16)}, run="r2")
    with pytest.raises(DuplicateBenchmarkError, match="single_turn"):
        BFCLReportDriver("m").report()


def test_category_sets_of_different_runs_are_reported_together(runs_root: Path) -> None:
    _write_scores(runs_root, {"live_relevance": (0.9, 16)}, run="r1", category_set="single_turn")
    _write_scores(runs_root, {"live_relevance": (0.8, 16)}, run="r2", category_set="live")
    df = BFCLReportDriver("m").report()
    assert sorted(zip(df["run"], df["benchmark"], strict=True)) == [("r1", "single_turn"), ("r2", "live")]


def test_folders_without_scores_are_not_runs(runs_root: Path) -> None:
    (runs_root / "m" / "r1" / "results" / "bfcl" / "single_turn" / "org_model").mkdir(parents=True)
    assert BFCLReportDriver("m").run_dirs() == []
    assert BFCLReportDriver("m").report().empty
