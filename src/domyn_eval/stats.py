# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""A model's sample stats across its runs, built from the suites' stat drivers."""

import logging
from typing import NamedTuple, Self

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from domyn_eval.config import registered_suites
from domyn_eval.drivers import stat_driver
from domyn_eval.drivers.stat.base import StatRow, StatsProfile
from domyn_eval.paths import runs_root

logger = logging.getLogger(__name__)


class RunStats(NamedTuple):
    """A benchmark's stats in one run of a suite, and what they flag."""

    suite: str
    run: str
    row: StatRow
    warnings: list[str]


class ModelStats:
    """One model's stats: a row per (suite, run, benchmark), in ``rows``."""

    def __init__(self, model: str, rows: list[RunStats]):
        """Wrap a model's stat rows.

        Args:
            model: The model's id.
            rows: The rows, by suite then run.
        """
        self.model = model
        self.rows = rows

    @classmethod
    def build(cls, model: str, suites: list[str] | None = None, profile: StatsProfile | None = None) -> Self:
        """Count every run's samples, for the suites that have a stat driver.

        A run without its ``config.toml`` is skipped, as it does not tell whether it reasoned.

        Args:
            model: The model's id, which names its folder under the runs root.
            suites: Count only these suites.
            profile: The thresholds; None uses the defaults.
        """
        model_dir = runs_root() / model
        rows = []
        for suite in registered_suites():
            if suites and suite not in suites:
                continue
            runs = sorted(p.parents[1].name for p in model_dir.glob(f"*/results/{suite}") if p.is_dir())
            driver = stat_driver(suite) if runs else None
            if driver is None:
                continue
            for run in runs:
                try:
                    stats = driver(model, run, profile)
                except FileNotFoundError:
                    logger.warning("%s: run %s has no config.toml; no stats", suite, run)
                    continue
                reasoning = stats.config.generation.reasoning
                rows += [RunStats(suite, run, r, r.warnings(stats.profile, reasoning=reasoning)) for r in stats.stats()]
        return cls(model, rows)

    def render(self, console: Console | None = None) -> None:
        """Print the stats as a table, if any suite has some; flagged benchmarks are in red."""
        if not self.rows:
            return
        console = console or Console()
        table = Table(title=f"Stats: {escape(self.model)}", header_style="bold")
        table.add_column("suite", style="cyan", no_wrap=True)
        table.add_column("run", style="dim")
        table.add_column("benchmark")
        for name in ("samples", "truncated", "reasoning", "looping", "empty", "unscored", "max tokens"):
            table.add_column(name, justify="right")
        table.add_column("warnings", style="red")
        for suite, run, row, warnings in self.rows:
            counts = (row.samples, row.truncated, row.reasoning, row.looping, row.empty, row.unscored)
            longest = "" if row.max_completion_tokens is None else str(row.max_completion_tokens)
            cells = [escape(suite), escape(run), escape(row.benchmark), *map(str, counts), longest]
            table.add_row(*cells, escape("; ".join(warnings)))
        console.print(table)
        flagged = sum(bool(r.warnings) for r in self.rows)
        if flagged:
            console.print(f"[yellow]{flagged} benchmark runs have scores to distrust; see the warnings.[/yellow]")
