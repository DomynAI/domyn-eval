"""Tests for the logic BaseReportDriver shares across all suite report drivers."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from domyn_eval.drivers.report.base import BaseReportDriver, DuplicateBenchmarkError, ReportRow


def test_report_rejects_rows_that_are_not_report_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DictReport(BaseReportDriver):
        @property
        def suite(self) -> str:
            return "dicts"

        def parse_run(self, run_dir: Path) -> list:  # noqa: ARG002 - every run reports the same row
            return [{"benchmark": "b", "metric": "accuracy", "score": 1.0}]

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "m" / "r1" / "results" / "dicts").mkdir(parents=True)

    with pytest.raises(TypeError, match="must return ReportRow"):
        DictReport("m").report()


def test_report_rejects_one_run_reporting_a_score_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class TwiceReport(BaseReportDriver):
        @property
        def suite(self) -> str:
            return "twice"

        def parse_run(self, run_dir: Path) -> list[ReportRow]:  # noqa: ARG002 - every run reports the same rows
            return [ReportRow(benchmark="b", metric="accuracy", score=score) for score in (0.5, 0.7)]

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "m" / "r1" / "results" / "twice").mkdir(parents=True)

    with pytest.raises(DuplicateBenchmarkError, match=r"b in r1$"):
        TwiceReport("m").report()


def test_report_row_validates_fields() -> None:
    with pytest.raises(ValidationError):
        ReportRow.model_validate({"benchmark": "b", "metric": "accuracy", "score": "not a number"})
    with pytest.raises(ValidationError):
        ReportRow.model_validate({"benchmark": "b", "metric": "accuracy", "score": 1.0, "run": "r1"})
