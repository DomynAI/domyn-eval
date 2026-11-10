"""Tests for the inspect report driver, on real ``.eval`` logs."""

from pathlib import Path

import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalScore,
    EvalSpec,
    EvalStats,
    EvalStatus,
    write_eval_log,
)

from domyn_eval.drivers.report.base import DuplicateBenchmarkError
from domyn_eval.drivers.report.inspect import InspectReportDriver


def _group(name: str, scorer: str, reducer: str | None = None, **metrics: float) -> EvalScore:
    return EvalScore(
        name=name,
        scorer=scorer,
        reducer=reducer,
        metrics={k: EvalMetric(name=k, value=v) for k, v in metrics.items()},
    )


def _dual(
    scorer: str = "dual_choice", reducer: str | None = None, strict: float = 0.5, loose: float = 0.6
) -> list[EvalScore]:
    return [
        _group("strict", scorer, reducer, accuracy=strict, stderr=0.01),
        _group("loose", scorer, reducer, accuracy=loose, stderr=0.02),
    ]


def _write_log(
    run_dir: Path,
    task: str,
    scores: list[EvalScore] | None,
    status: EvalStatus = "success",
    display_name: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = EvalLog(
        status=status,
        eval=EvalSpec(
            created="2026-01-01T00:00:00+00:00",
            task=f"inspect_evals/{task}",
            task_display_name=display_name,
            dataset=EvalDataset(),
            model="m",
            config=EvalConfig(),
        ),
        results=EvalResults(total_samples=10, completed_samples=10, scores=scores) if scores else None,
        stats=EvalStats(
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:40+00:00",
        ),
    )
    write_eval_log(log, str(run_dir / f"{task}.eval"))


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir, where the tests write the model's runs."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


def _run(runs_root: Path, run_id: str) -> Path:
    """Model ``m``'s inspect folder in run ``run_id``: ``<runs>/m/<run_id>/results/inspect``."""
    return runs_root / "m" / run_id / "results" / "inspect"


def test_report_reads_dual_scored_logs(runs_root: Path) -> None:
    _write_log(_run(runs_root, "run1"), "gpqa_diamond", _dual(), display_name="gpqa_diamond")
    _run(runs_root, "no_logs").mkdir(parents=True)

    df = InspectReportDriver("m").report()

    assert set(df["run"]) == {"run1"}
    rows = df.set_index("metric")
    assert rows.loc["strict/accuracy", "score"] == pytest.approx(0.5)
    assert rows.loc["strict/accuracy", "stderr"] == pytest.approx(0.01)
    assert rows.loc["loose/accuracy", "scorer"] == "dual_choice"
    assert rows.loc["loose/accuracy", "samples"] == 10
    assert rows.loc["loose/accuracy", "duration"] == pytest.approx(100)
    assert tuple(df.columns) == InspectReportDriver.REPORT_COLUMNS


def test_report_labels_each_reducer(runs_root: Path) -> None:
    scores = [
        *_dual("dual_aime_scorer", "mean", strict=0.8, loose=0.8),
        *_dual("dual_aime_scorer", "max", strict=0.9, loose=0.9),
        *_dual("dual_aime_scorer", "pass_at_2", strict=0.85, loose=0.85),
    ]
    _write_log(_run(runs_root, "run1"), "aime2025_pass_k", scores)

    df = InspectReportDriver("m").report()

    by_metric = dict(zip(df["metric"], df["score"], strict=True))
    assert set(by_metric) == {
        "strict/mean/accuracy",
        "loose/mean/accuracy",
        "strict/max/accuracy",
        "loose/max/accuracy",
        "strict/pass_at_2/accuracy",
        "loose/pass_at_2/accuracy",
    }
    assert by_metric["strict/mean/accuracy"] == pytest.approx(0.8)
    assert by_metric["strict/max/accuracy"] == pytest.approx(0.9)
    assert by_metric["strict/pass_at_2/accuracy"] == pytest.approx(0.85)


def test_report_pairs_prefixed_stderr_with_its_metric(runs_root: Path) -> None:
    metrics = {"prompt_strict_acc": 0.7, "prompt_strict_stderr": 0.02, "final_acc": 0.8}
    scores = [
        _group("dual_strict_if", "dual_strict_if", None, **metrics),
        _group("dual_loose_if", "dual_loose_if", "mean", **metrics),
    ]
    _write_log(_run(runs_root, "run1"), "ifeval", scores)

    df = InspectReportDriver("m").report().set_index("metric")

    assert set(df.index) == {
        "dual_strict_if/prompt_strict_acc",
        "dual_strict_if/final_acc",
        "dual_loose_if/mean/prompt_strict_acc",
        "dual_loose_if/mean/final_acc",
    }
    assert df.loc["dual_strict_if/prompt_strict_acc", "stderr"] == pytest.approx(0.02)
    assert df.loc["dual_strict_if/prompt_strict_acc", "scorer"] == "dual_strict_if"


@pytest.mark.parametrize("metric", ["mean", "f1"])
def test_report_pairs_stderr_with_the_scorers_only_other_metric(runs_root: Path, metric: str) -> None:
    _write_log(_run(runs_root, "run1"), "drop", [_group("", "f1_scorer", None, **{metric: 0.4, "stderr": 0.03})])

    df = InspectReportDriver("m").report()

    assert list(df["metric"]) == [metric]
    assert df["stderr"].iloc[0] == pytest.approx(0.03)


def test_report_keeps_stderr_as_a_row_when_its_metric_is_ambiguous(runs_root: Path) -> None:
    _write_log(_run(runs_root, "run1"), "drop", [_group("", "f1_scorer", accuracy=0.5, f1=0.4, stderr=0.03)])

    df = InspectReportDriver("m").report()

    assert set(df["metric"]) == {"accuracy", "f1", "stderr"}
    assert df["stderr"].isna().all()


def test_report_drops_unset_label_parts(runs_root: Path) -> None:
    _write_log(_run(runs_root, "run1"), "mmlu", [_group("", "choice", accuracy=0.6, stderr=0.01)])

    df = InspectReportDriver("m").report()

    assert list(df["metric"]) == ["accuracy"]
    assert df["stderr"].iloc[0] == pytest.approx(0.01)


def test_report_skips_unfinished_logs(runs_root: Path) -> None:
    _write_log(_run(runs_root, "run1"), "gpqa_diamond", _dual())
    _write_log(_run(runs_root, "run1"), "scicode", None, status="error")

    df = InspectReportDriver("m").report()

    assert set(df["benchmark"]) == {"inspect_evals/gpqa_diamond"}


def test_report_rejects_a_task_in_two_runs(runs_root: Path) -> None:
    _write_log(_run(runs_root, "run1"), "gpqa_diamond", _dual())
    _write_log(_run(runs_root, "run2"), "gpqa_diamond", _dual())

    with pytest.raises(DuplicateBenchmarkError, match="run1, run2"):
        InspectReportDriver("m").report()


def test_report_rejects_a_task_twice_in_one_run(runs_root: Path) -> None:
    run_dir = _run(runs_root, "run1")
    _write_log(run_dir, "gpqa_diamond", _dual(), display_name="gpqa")
    _write_log(run_dir, "gpqa_retry", _dual(), display_name="gpqa")

    with pytest.raises(DuplicateBenchmarkError, match="more than once") as info:
        InspectReportDriver("m").report()
    assert list(info.value.clashes) == ["gpqa"]
    assert len(info.value.clashes["gpqa"]) == 2
