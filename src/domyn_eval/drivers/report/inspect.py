# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalScore, read_eval_log
from pydantic import BaseModel, ConfigDict

from domyn_eval.drivers.report.base import BaseReportDriver, DuplicateBenchmarkError, ReportRow

logger = logging.getLogger(__name__)


class _ParsedScore(BaseModel):
    """One metric read from an inspect log's ``results.scores``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer: str
    metric: str  # ``{group}/{reducer}/{metric}``, see InspectReportDriver._parse_scores
    score: float
    stderr: float | None = None


class InspectReportDriver(BaseReportDriver):
    """Report of a model's inspect runs: the ``.eval`` logs in each ``<runs>/<model>/<run_id>/results/inspect``.

    Scores are read from the logs' headers with inspect_ai's log API, so a run whose ``results_{run_id}.csv`` was never
    written still reports.
    """

    @property
    def suite(self) -> str:
        """Name of the suite whose results this reports."""
        return "inspect"

    def run_dirs(self) -> list[Path]:
        """The run folders that hold at least one ``.eval`` log."""
        return [p for p in super().run_dirs() if any(p.glob("*.eval"))]

    def parse_run(self, run_dir: Path) -> list[ReportRow]:
        """One row per metric of every successful ``.eval`` log in ``run_dir``.

        Logs that did not finish (``error``, ``cancelled``, ``started``) carry no scores and are skipped.

        Raises:
            DuplicateBenchmarkError: If two successful logs in ``run_dir`` are of the same task.
        """
        rows, tasks = [], {}
        for path in sorted(run_dir.glob("*.eval")):
            log = read_eval_log(str(path), header_only=True)
            benchmark = log.eval.task_display_name or log.eval.task
            if log.status != "success" or not log.results:
                logger.warning("inspect: %s is '%s' in %s; skipped", benchmark, log.status, path)
                continue
            if benchmark in tasks:
                raise DuplicateBenchmarkError(self.suite, run_dir, {benchmark: [tasks[benchmark].name, path.name]})
            tasks[benchmark] = path
            duration = None
            if log.stats.started_at and log.stats.completed_at:
                finish_time = datetime.fromisoformat(log.stats.completed_at)
                start_time = datetime.fromisoformat(log.stats.started_at)
                duration = (finish_time - start_time).total_seconds()
            rows.extend(
                ReportRow(
                    benchmark=benchmark,
                    metric=parsed.metric,
                    score=parsed.score,
                    stderr=parsed.stderr,
                    samples=log.results.completed_samples,
                    scorer=parsed.scorer,
                    duration=duration,
                )
                for parsed in self._parse_scores(log.results.scores)
            )
        return rows

    @staticmethod
    def _parse_scores(scores: list[EvalScore]) -> list[_ParsedScore]:
        """One ``_ParsedScore`` per metric of a log's ``results.scores``.

        The label is ``{group}/{reducer}/{metric}``, leaving out whichever part the log does not set. The group is the
        scorer's name, or the named group of a scorer that reports several (``strict`` / ``loose``). A task scored over
        several epochs repeats each group once per epoch reducer, so the reducer is what tells those rows apart:

            match/accuracy                 <- gsm8k: one group, no epochs
            choice/accuracy                <- gpqa_diamond
            strict/accuracy                <- mmlu_pro: named groups, no epochs
            loose/accuracy
            strict/mean/accuracy           <- aime2025_pass_k: one row per reducer
            strict/max/accuracy
            strict/pass_at_1/accuracy

        A ``<x>stderr`` metric fills the stderr of the scorer's one other metric named ``<x>...`` (``accuracy``,
        ``mean``, ``prompt_strict_acc`` for ``prompt_strict_stderr``, ...); with several such metrics it is a row of its
        own.
        """
        rows = []
        for sc in scores:
            if not sc.metrics:
                continue
            prefix = "".join(f"{part}/" for part in (sc.name, sc.reducer) if part)

            values = {
                m.name or "accuracy": float(m.value) for m in sc.metrics.values() if isinstance(m.value, (int, float))
            }
            stderrs = {}
            for name in [n for n in values if n.endswith("stderr")]:
                base = name.removesuffix("stderr")
                siblings = [n for n in values if n.startswith(base) and not n.endswith("stderr")]
                if len(siblings) == 1:
                    stderrs[siblings[0]] = values.pop(name)
            rows.extend(
                _ParsedScore(scorer=sc.scorer, metric=prefix + name, score=value, stderr=stderrs.get(name))
                for name, value in values.items()
            )
        return rows
