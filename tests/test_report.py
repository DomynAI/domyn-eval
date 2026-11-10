import dataclasses
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

from domyn_eval import config
from domyn_eval.cli import cli
from domyn_eval.config import SuiteDrivers, register_suite
from domyn_eval.drivers.report.base import BaseReportDriver, ReportRow
from domyn_eval.report import Leaderboard, ModelReport, ReportProfile

REPO = Path(__file__).parents[1]


class ScoresReport(BaseReportDriver):
    """Reads the rows a test wrote to each run's ``scores.json``."""

    @property
    def suite(self) -> str:
        return "scores"

    def parse_run(self, run_dir: Path) -> list[ReportRow]:
        return [ReportRow(**row) for row in json.loads((run_dir / "scores.json").read_text())]


class OtherReport(ScoresReport):
    @property
    def suite(self) -> str:
        return "other"


@dataclasses.dataclass
class NoConfig:
    pass


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "_SUITES", {})
    monkeypatch.setattr(config, "_DRIVERS", {})
    for suite, report in [
        ("scores", f"{__name__}:ScoresReport"),
        ("other", f"{__name__}:OtherReport"),
        ("absent", "not_installed:Report"),
    ]:
        register_suite(suite, NoConfig, drivers=SuiteDrivers(run="unused:Run", report=report))
    return tmp_path


def _scores(runs: Path, model: str, run: str, suite: str, rows: list[dict]) -> None:
    run_dir = runs / model / run / "results" / suite
    run_dir.mkdir(parents=True)
    (run_dir / "scores.json").write_text(json.dumps(rows))


def _row(benchmark: str, score: float, metric: str = "accuracy", **fields: object) -> dict:
    return {"benchmark": benchmark, "metric": metric, "score": score, **fields}


PROFILE = {
    "sections": [
        {"name": "Math", "benchmarks": [{"suite": "scores", "benchmark": "gsm8k", "metrics": ["accuracy"]}]},
        {
            "name": "Other",
            "benchmarks": [
                {"suite": "other", "benchmark": "ruler", "metrics": ["4096", "8192"], "label": "RULER"},
                {"suite": "scores", "benchmark": "aime", "metrics": ["accuracy"]},
            ],
        },
    ]
}


def test_build_reports_every_suite_the_model_ran(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.5, samples=10)])
    _scores(runs, "m", "r2", "other", [_row("ruler", 0.7, metric="4096", stderr=0.01)])

    df = ModelReport.build("m").df

    assert list(df.columns) == list(ModelReport.COLUMNS)
    assert df[["model", "suite", "run", "benchmark", "score"]].values.tolist() == [
        ["m", "scores", "r1", "gsm8k", 0.5],
        ["m", "other", "r2", "ruler", 0.7],
    ]
    assert df["samples"].dtype == "Int64"
    assert pd.isna(df.loc[0, "stderr"])


def test_build_reports_only_the_selected_suites(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.5)])
    _scores(runs, "m", "r1", "other", [_row("ruler", 0.7)])

    assert ModelReport.build("m", ["other"]).df["suite"].tolist() == ["other"]


def test_build_of_a_model_without_results_is_empty(runs: Path) -> None:
    (runs / "m" / "r1" / "results" / "scores").mkdir(parents=True)
    (runs / "m" / "r1" / "results" / "scores" / "scores.json").write_text("[]")

    report = ModelReport.build("m")

    assert report.df.empty
    assert list(report.df.columns) == list(ModelReport.COLUMNS)


@pytest.mark.usefixtures("runs")
def test_build_of_an_unknown_model_fails() -> None:
    with pytest.raises(FileNotFoundError, match="no runs of m under"):
        ModelReport.build("m")


def test_profile_keeps_its_rows_in_order_and_reports_missing_ones(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("aime", 0.3), _row("gsm8k", 0.5), _row("mmlu", 0.9)])
    _scores(runs, "m", "r1", "other", [_row("ruler", 0.7, metric="4096")])

    df = ModelReport.build("m").with_profile(ReportProfile.model_validate(PROFILE)).df

    assert list(df.columns[:2]) == ["model", "section"]
    assert df[["section", "suite", "benchmark", "metric"]].values.tolist() == [
        ["Math", "scores", "gsm8k", "accuracy"],
        ["Other", "other", "RULER", "4096"],
        ["Other", "other", "RULER", "8192"],
        ["Other", "scores", "aime", "accuracy"],
    ]
    assert df["score"].tolist()[:2] == [0.5, 0.7]
    assert pd.isna(df.loc[2, "score"])
    assert df.loc[2, "model"] == "m"


def test_profile_leaves_out_unselected_suites(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.5)])

    df = ModelReport.build("m", ["scores"]).with_profile(ReportProfile.model_validate(PROFILE), ["scores"]).df

    assert df["benchmark"].tolist() == ["gsm8k", "aime"]


@pytest.mark.parametrize(
    ("benchmarks", "error"),
    [
        ([{"suite": "s", "benchmark": "b", "metrics": []}], "at least one metric"),
        ([{"suite": "s", "benchmark": "b", "metrics": ["a", "a"]}], "listed twice"),
        ([{"suite": "s", "benchmark": "b", "metrics": ["a"], "colour": "red"}], "Extra inputs"),
        ([{"suite": "s", "benchmark": "b", "metrics": ["a"]}] * 2, "benchmarks listed more than once: s/b"),
        (
            [
                {"suite": "s", "benchmark": "aime2024", "metrics": ["a"], "label": "AIME"},
                {"suite": "s", "benchmark": "aime2025", "metrics": ["a"], "label": "AIME"},
            ],
            "labels listed more than once: s/AIME",
        ),
        (
            [
                {"suite": "s", "benchmark": "aime2024", "metrics": ["a"], "label": "aime2025"},
                {"suite": "s", "benchmark": "aime2025", "metrics": ["a"]},
            ],
            "labels listed more than once: s/aime2025",
        ),
    ],
)
def test_profile_rejects_invalid_benchmarks(benchmarks: list[dict], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        ReportProfile.model_validate({"sections": [{"name": "S", "benchmarks": benchmarks}]})


def test_shipped_profile_loads() -> None:
    profile = ReportProfile.load(REPO / "configs" / "report_profile.yaml")

    assert {b.suite for s in profile.sections for b in s.benchmarks} <= {"inspect", "bfcl", "lm_eval"}


def test_save_keeps_full_precision_in_the_models_folder(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.123456, stderr=0.00012, duration=12.345678)])

    path = ModelReport.build("m").save()

    assert path == runs / "m" / "report.csv"
    saved = pd.read_csv(path)
    assert saved.loc[0, ["score", "stderr", "duration"]].tolist() == [0.123456, 0.00012, 12.345678]


def _report(model: str, rows: list[tuple]) -> ModelReport:
    """A report of ``(section, suite, benchmark, metric, score[, higher_is_better])`` rows."""
    columns = ["model", "section", "suite", "benchmark", "metric", "score", "higher_is_better"]
    return ModelReport(model, pd.DataFrame([(model, *row, True)[:7] for row in rows], columns=columns))


def test_leaderboard_lines_models_up_in_first_seen_order() -> None:
    a = _report("a", [("Math", "s", "gsm8k", "acc", 0.5), ("Math", "s", "aime", "acc", 0.2)])
    b = _report("b", [("Math", "s", "aime", "acc", 0.4), ("IF", "s", "ifeval", "acc", 0.8)])

    df = Leaderboard.from_reports([b, a]).df

    assert list(df.columns) == ["section", "suite", "benchmark", "metric", "higher_is_better", "b", "a"]
    assert df["benchmark"].tolist() == ["aime", "ifeval", "gsm8k"]
    assert df.loc[0, ["b", "a"]].tolist() == [0.4, 0.2]
    assert pd.isna(df.loc[1, "a"])


def test_leaderboard_keys_on_section_only_when_every_report_has_one() -> None:
    a = _report("a", [("Math", "s", "gsm8k", "acc", 0.5)])
    b = ModelReport("b", a.df.drop(columns="section").assign(model="b"))

    df = Leaderboard.from_reports([a, b]).df

    assert list(df.columns) == ["suite", "benchmark", "metric", "higher_is_better", "a", "b"]


def test_leaderboard_takes_a_metrics_direction_from_any_report_that_scored_it() -> None:
    a = _report("a", [("LM", "s", "wiki", "ppl", 12.0, False), ("LM", "s", "gsm8k", "acc", float("nan"), float("nan"))])
    b = _report("b", [("LM", "s", "wiki", "ppl", float("nan"), float("nan"))])

    df = Leaderboard.from_reports([b, a]).df

    assert df["higher_is_better"].tolist() == [False, True]  # gsm8k: scored by none, the default


def test_leaderboard_highlights_the_best_score_in_the_metrics_direction() -> None:
    a = _report("a", [("LM", "s", "wiki", "ppl", 12.0, False), ("LM", "s", "gsm8k", "acc", 0.25)])
    b = _report("b", [("LM", "s", "wiki", "ppl", 15.0, False), ("LM", "s", "gsm8k", "acc", 0.75)])
    console = Console(record=True, width=120, color_system="standard")

    Leaderboard.from_reports([a, b]).render(console)

    highlighted = set(re.findall(r"\x1b\[1;32m([\d.]+)", console.export_text(styles=True)))
    assert highlighted == {"12.0000", "0.7500"}


def test_leaderboard_rejects_a_repeated_model() -> None:
    a = _report("a", [("Math", "s", "gsm8k", "acc", 0.5)])

    with pytest.raises(ValueError, match="more than once: a"):
        Leaderboard.from_reports([a, a])


def test_cli_writes_each_models_report(runs: Path) -> None:
    _scores(runs, "a", "r1", "scores", [_row("gsm8k", 0.5)])
    _scores(runs, "b", "r1", "scores", [_row("gsm8k", 0.6)])

    result = CliRunner().invoke(cli, ["report", "a", "b"])

    assert result.exit_code == 0, result.output
    assert (runs / "a" / "report.csv").is_file()
    assert (runs / "b" / "report.csv").is_file()


def test_cli_writes_the_leaderboard_with_a_profile(runs: Path, tmp_path: Path) -> None:
    _scores(runs, "a", "r1", "scores", [_row("gsm8k", 0.5)])
    _scores(runs, "b", "r1", "scores", [_row("gsm8k", 0.6), _row("aime", 0.1)])
    profile = tmp_path / "profile.yaml"
    profile.write_text(json.dumps(PROFILE))
    output = tmp_path / "board.csv"

    args = ["report", "a", "b", "--leaderboard", "--profile", str(profile), "--output", str(output)]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    board = pd.read_csv(output, dtype={"metric": str})
    assert board[["benchmark", "metric"]].values.tolist() == [
        ["gsm8k", "accuracy"],
        ["RULER", "4096"],
        ["RULER", "8192"],
        ["aime", "accuracy"],
    ]
    assert board.loc[0, ["a", "b"]].tolist() == [0.5, 0.6]
    assert not (runs / "a" / "report.csv").exists()


def test_cli_explains_a_benchmark_reported_twice(runs: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.5)])
    _scores(runs, "m", "r2", "scores", [_row("gsm8k", 0.6)])

    result = CliRunner().invoke(cli, ["report", "m"])

    assert result.exit_code == 1
    assert "scores: benchmarks" in result.output  # the title wraps to the table's width
    assert "r1, r2" in result.output


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["report", "a", "b", "--output", "x.csv"], "--output takes a single model"),
        (["report", "a", "--suite", "nope"], "unknown suites ['nope']"),
        (["report", "a"], "no runs of a under"),
    ],
)
@pytest.mark.usefixtures("runs")
def test_cli_rejects_bad_arguments(args: list[str], error: str) -> None:
    result = CliRunner().invoke(cli, args)

    assert result.exit_code != 0
    assert error in result.output


def test_cli_rejects_an_invalid_profile(runs: Path, tmp_path: Path) -> None:
    _scores(runs, "m", "r1", "scores", [_row("gsm8k", 0.5)])
    profile = tmp_path / "profile.yaml"
    profile.write_text("sections: [{name: S, benchmarks: [{suite: s}]}]")

    result = CliRunner().invoke(cli, ["report", "m", "--profile", str(profile)])

    assert result.exit_code != 0
    assert "Field required" in result.output
