# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Model reports and leaderboards, built from the suites' report drivers."""

import logging
from pathlib import Path
from typing import Self

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from domyn_eval.config import registered_suites
from domyn_eval.drivers import report_driver
from domyn_eval.drivers.report.base import BaseReportDriver, DuplicateBenchmarkError
from domyn_eval.log import display_path
from domyn_eval.paths import runs_root

logger = logging.getLogger(__name__)

_DECIMALS = 4  # of the scores printed; the CSVs keep full precision


class BenchmarkSpec(BaseModel):
    """A benchmark a report profile keeps, with the metrics to show and an optional label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite: str
    benchmark: str
    metrics: list[str]
    label: str | None = None

    @field_validator("metrics")
    @classmethod
    def _distinct_metrics(cls, metrics: list[str]) -> list[str]:
        if not metrics:
            msg = "list at least one metric"
            raise ValueError(msg)
        repeated = sorted({m for m in metrics if metrics.count(m) > 1})
        if repeated:
            msg = f"metrics listed twice: {repeated}"
            raise ValueError(msg)
        return metrics


class Section(BaseModel):
    """A named group of benchmarks in a report profile, e.g. ``Math``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    benchmarks: list[BenchmarkSpec]


class ReportProfile(BaseModel):
    """Which benchmarks and metrics a report shows, grouped into sections, in order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sections: list[Section] = []

    @model_validator(mode="after")
    def _each_benchmark_once(self) -> Self:
        """A benchmark, and the name it is shown under, appear once per suite; the report's rows are keyed on both."""
        specs = [b for s in self.sections for b in s.benchmarks]
        for what, keys in (
            ("benchmarks", [(b.suite, b.benchmark) for b in specs]),
            ("labels", [(b.suite, b.label or b.benchmark) for b in specs]),
        ):
            repeated = sorted({k for k in keys if keys.count(k) > 1})
            if repeated:
                msg = f"{what} listed more than once: {', '.join(f'{s}/{b}' for s, b in repeated)}"
                raise ValueError(msg)
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a profile from a YAML file."""
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


class ModelReport:
    """One model's report: a row per (suite, run, benchmark, metric), in ``df``."""

    FILE = "report.csv"
    COLUMNS = ("model", *BaseReportDriver.REPORT_COLUMNS)

    def __init__(self, model: str, df: pd.DataFrame):
        """Wrap a model's report rows.

        Args:
            model: The model's id.
            df: The rows, with ``COLUMNS``.
        """
        self.model = model
        self.df = df

    @classmethod
    def build(cls, model: str, suites: list[str] | None = None) -> Self:
        """Report every suite the model has runs of, through its report driver.

        Args:
            model: The model's id, which names its folder under the runs root.
            suites: Report only these suites.

        Raises:
            FileNotFoundError: If the model has no runs folder.
            DuplicateBenchmarkError: If a suite reports a score more than once.
        """
        model_dir = runs_root() / model
        if not model_dir.is_dir():
            msg = f"no runs of {model} under {model_dir.parent}"
            raise FileNotFoundError(msg)
        frames = []
        for suite in registered_suites():
            if (suites and suite not in suites) or not any(model_dir.glob(f"*/results/{suite}")):
                continue
            df = report_driver(suite)(model).report()
            if df.empty:
                logger.warning("%s: no results under %s", suite, display_path(model_dir))
                continue
            frames.append(df.dropna(axis=1, how="all"))  # all-NA columns would decide the dtypes of the concat
        if not frames:
            return cls(model, pd.DataFrame(columns=list(cls.COLUMNS)))
        df = pd.concat(frames, ignore_index=True).reindex(columns=list(BaseReportDriver.REPORT_COLUMNS))
        df["samples"] = df["samples"].astype("Int64")
        df.insert(0, "model", model)
        return cls(model, df)

    def with_profile(self, profile: ReportProfile, suites: list[str] | None = None) -> "ModelReport":
        """This report cut down to the profile's rows, in its order, with a ``section`` column after ``model``.

        A listed metric with no result becomes a row without a score. With ``suites``, benchmarks of other suites are
        left out rather than reported missing.
        """
        found = {(r["suite"], r["benchmark"], r["metric"]): r for r in self.df.to_dict("records")}
        rows = []
        for section in profile.sections:
            for spec in section.benchmarks:
                if suites and spec.suite not in suites:
                    continue
                for metric in spec.metrics:
                    row = found.get((spec.suite, spec.benchmark, metric)) or {
                        "model": self.model,
                        "suite": spec.suite,
                        "metric": metric,
                    }
                    rows.append({**row, "section": section.name, "benchmark": spec.label or spec.benchmark})
        df = pd.DataFrame(rows, columns=["model", "section", *self.COLUMNS[1:]])
        df["samples"] = df["samples"].astype("Int64")
        return ModelReport(self.model, df)

    def save(self, path: Path | None = None) -> Path:
        """Write the report CSV (default: ``<runs>/<model>/report.csv``)."""
        path = path or runs_root() / self.model / self.FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return path

    def render(self, console: Console | None = None) -> None:
        """Print the report as a table."""
        console = console or Console()
        if self.df.empty:
            console.print(f"[yellow]No results to report for {escape(self.model)}.[/yellow]")
            return
        sections = "section" in self.df.columns
        table = Table(title=f"Report: {escape(self.model)}", header_style="bold")
        if sections:
            table.add_column("section", style="bold", no_wrap=True)
        table.add_column("suite", style="cyan", no_wrap=True)
        table.add_column("run", style="dim")
        table.add_column("benchmark")
        table.add_column("metric")
        table.add_column("score", justify="right")
        previous = None
        records = self.df.to_dict("records")
        for row in records:
            score = "[red]missing[/red]" if pd.isna(row["score"]) else f"{row['score']:.{_DECIMALS}f}"
            run = "" if pd.isna(row["run"]) else escape(str(row["run"]))
            cells = [escape(str(row["suite"])), run, escape(str(row["benchmark"])), escape(str(row["metric"])), score]
            if sections:
                cells.insert(0, escape(str(row["section"])) if row["section"] != previous else "")
                previous = row["section"]
            table.add_row(*cells)
        console.print(table)
        missing = sum(pd.isna(row["score"]) for row in records)
        if missing:
            console.print(f"[yellow]{missing} metrics of the profile have no result.[/yellow]")


class Leaderboard:
    """Model reports side by side: a row per metric, its direction, then a score column per model, in ``df``."""

    FILE = "leaderboard.csv"
    KEYS = ("section", "suite", "benchmark", "metric")

    def __init__(self, df: pd.DataFrame, models: list[str]):
        """Wrap a leaderboard table.

        Args:
            df: The key columns, ``higher_is_better``, then a score column per model.
            models: The models' columns, in order.
        """
        self.df = df
        self.models = models

    @classmethod
    def from_reports(cls, reports: list[ModelReport]) -> Self:
        """Line the reports' scores up by metric; rows keep the order they first appear in.

        ``section`` is a key only when every report has one, i.e. was cut down to the same profile.

        Raises:
            ValueError: If a model is given more than once.
        """
        models = [r.model for r in reports]
        repeated = sorted({m for m in models if models.count(m) > 1})
        if repeated:
            msg = f"models given more than once: {', '.join(repeated)}"
            raise ValueError(msg)
        keys = [k for k in cls.KEYS if all(k in r.df.columns for r in reports)]
        long = pd.concat([r.df[["model", *keys, "higher_is_better", "score"]] for r in reports], ignore_index=True)
        keyed = long[keys]
        assert isinstance(keyed, pd.DataFrame)
        order = pd.MultiIndex.from_frame(keyed.drop_duplicates())
        wide = long.pivot(index=keys, columns="model", values="score").reindex(index=order, columns=models)
        # from any report that scored the metric; a profile's row that none did defaults to higher is better
        direction = long.groupby(keys, sort=False)["higher_is_better"].first().reindex(order)
        wide.insert(0, "higher_is_better", direction.astype("boolean").fillna(value=True).astype(bool))
        df = wide.reset_index().rename_axis(columns=None)
        return cls(df, models)

    def save(self, path: Path | None = None) -> Path:
        """Write the leaderboard CSV (default: ``<runs>/leaderboard.csv``)."""
        path = path or runs_root() / self.FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return path

    def render(self, console: Console | None = None) -> None:
        """Print the leaderboard; the best score of each row, the lowest where lower is better, is highlighted."""
        console = console or Console()
        sections = "section" in self.df.columns
        table = Table(title="Leaderboard", header_style="bold")
        if sections:
            table.add_column("section", style="bold", no_wrap=True)
        else:
            table.add_column("suite", style="cyan", no_wrap=True)
        table.add_column("benchmark")
        table.add_column("metric", style="dim")
        for model in self.models:
            table.add_column(escape(model), justify="right")
        previous = None
        for row in self.df.to_dict("records"):
            scores = [row[m] for m in self.models if not pd.isna(row[m])]
            best = (max if row["higher_is_better"] else min)(scores, default=None)
            cells = []
            for model in self.models:
                score = row[model]
                if pd.isna(score):
                    cells.append("[dim]-[/dim]")
                elif score == best and len(self.models) > 1:
                    cells.append(f"[bold green]{score:.{_DECIMALS}f}[/bold green]")
                else:
                    cells.append(f"{score:.{_DECIMALS}f}")
            if sections:
                first = escape(str(row["section"])) if row["section"] != previous else ""
                previous = row["section"]
            else:
                first = escape(str(row["suite"]))
            table.add_row(first, escape(str(row["benchmark"])), escape(str(row["metric"])), *cells)
        console.print(table)


def render_duplicates(error: DuplicateBenchmarkError, console: Console | None = None) -> None:
    """Explain which benchmarks a suite reports more than once, and where."""
    console = console or Console()
    table = Table(
        title=f"[red]{escape(error.suite)}: benchmarks reported more than once[/red]",
        caption=escape(error.location),
        header_style="bold",
    )
    table.add_column("benchmark", style="cyan", no_wrap=True)
    table.add_column("found in")
    for benchmark, where in error.clashes.items():
        table.add_row(escape(benchmark), escape(", ".join(where)))
    console.print(table)
    console.print(
        "[yellow]A model reports each score once: move the extra runs out of the model's folder, "
        "or leave the suite out with --suite.[/yellow]"
    )
