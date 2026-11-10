# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from domyn_eval.log import display_path
from domyn_eval.paths import runs_root

logger = logging.getLogger(__name__)


class ReportRow(BaseModel):
    """One benchmark score parsed from a run; the fields are the report CSV's columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str
    metric: str
    score: float
    higher_is_better: bool = True
    stderr: float | None = None
    samples: int | None = None
    scorer: str | None = None
    duration: float | None = None


class DuplicateBenchmarkError(ValueError):
    """The same benchmark appears more than once among one model's runs of a suite.

    ``clashes`` maps each repeated benchmark to the runs that report it: two or more, or one run reporting it twice.
    """

    def __init__(self, suite: str, location: str | os.PathLike, clashes: dict[str, list[str]]):
        """Build the error message listing every clash.

        Args:
            suite: Suite whose results repeat the benchmarks.
            location: The model folder whose runs were searched.
            clashes: Each repeated benchmark mapped to where it was found.
        """
        self.suite = suite
        self.location = str(location)
        self.clashes = clashes
        listing = "; ".join(f"{b} in {', '.join(where)}" for b, where in clashes.items())
        super().__init__(f"{suite}: benchmarks reported more than once under {self.location}: {listing}")


class BaseReportDriver(ABC):
    """Turns one suite's results across a model's runs into a flat table.

    Each suite subclasses it and implements ``suite`` and ``parse_run``.
    """

    REPORT_COLUMNS = ("suite", "run", *ReportRow.model_fields)
    REPORT_FILE = "report_{suite}.csv"  # formatted with the instance's suite

    @property
    @abstractmethod
    def suite(self) -> str:
        """Name of the suite whose results this reports."""

    def __init__(self, model: str):
        """Point the report at one model's runs: ``<runs>/<model>/<run_id>/results/<suite>``.

        Args:
            model: The model's id, which names its folder under the runs root.
        """
        self.model_dir = runs_root() / model

    def run_dirs(self) -> list[Path]:
        """This suite's folder in each of the model's runs: ``<runs>/<model>/<run_id>/results/<suite>``."""
        return sorted(p for p in self.model_dir.glob(f"*/results/{self.suite}") if p.is_dir())

    @abstractmethod
    def parse_run(self, run_dir: Path) -> list[ReportRow]:
        """The benchmark scores of one run.

        Each suite's report driver implements this.
        """

    def report(self) -> pd.DataFrame:
        """All runs' rows, one per benchmark score.

        Raises:
            TypeError: If ``parse_run`` returns something other than ``ReportRow``s.
            DuplicateBenchmarkError: If a benchmark's metric is reported more than once.
        """
        rows = []
        for run_dir in self.run_dirs():
            run = run_dir.relative_to(self.model_dir).parts[0]
            for row in self.parse_run(run_dir):
                if not isinstance(row, ReportRow):
                    msg = f"{type(self).__name__}.parse_run must return ReportRow, got {type(row).__name__}"
                    raise TypeError(msg)
                rows.append({"suite": self.suite, "run": run, **row.model_dump()})
        df = pd.DataFrame(rows, columns=list(self.REPORT_COLUMNS))
        self.check_unique_benchmarks(df)
        return df

    def check_unique_benchmarks(self, df: pd.DataFrame) -> None:
        """A model reports each score once, so the same one reported twice, by two runs or by one, is ambiguous.

        A benchmark may be spread over several runs as long as each of its metrics comes from only one: e.g., RULER at
        64k in one run and at 128k in another is fine, two runs both reporting RULER at 64k are not.
        """
        runs = df.groupby(["benchmark", "metric"])["run"].agg(list)  # a run per row, so one run reporting twice counts
        repeated = runs[runs.map(len) > 1]
        assert isinstance(repeated, pd.Series)
        metric_counts = df.groupby("benchmark")["metric"].nunique()
        clashes = {}
        for benchmark, by_metric in repeated.groupby(level="benchmark"):
            metrics = list(by_metric.index.get_level_values("metric"))
            where = sorted({r for rs in by_metric for r in rs})
            # name the metrics only when the benchmark is split, i.e. some do not clash
            key = benchmark if len(metrics) == metric_counts[benchmark] else f"{benchmark} ({', '.join(metrics)})"
            clashes[key] = where
        if clashes:
            raise DuplicateBenchmarkError(self.suite, self.model_dir, clashes)

    def save(self, path: str | os.PathLike | None = None) -> Path | None:
        """Write the report CSV (default: ``<runs>/<model>/report_<suite>.csv``); None if empty."""
        df = self.report()
        if df.empty:
            logger.warning("%s: no results under %s", self.suite, display_path(self.model_dir))
            return None
        path = Path(path or self.model_dir / self.REPORT_FILE.format(suite=self.suite))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("%s: report saved to %s", self.suite, display_path(path))
        return path
