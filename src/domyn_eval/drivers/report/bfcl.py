# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from domyn_eval.drivers.report.base import BaseReportDriver, DuplicateBenchmarkError, ReportRow

logger = logging.getLogger(__name__)


class _Aggregate(BaseModel):
    """A BFCL leaderboard aggregate: how its members are weighted, and its members.

    A member is a test category (one ``BFCL_v4_{category}_score.json``) or another aggregate. ``mean`` is BFCL's
    unweighted accuracy, ``count`` weights each member by its samples, and a list weights the members by its numbers.
    """

    model_config = ConfigDict(frozen=True)

    weighting: Literal["mean", "count"] | list[int]
    members: list[str]

    @model_validator(mode="after")
    def _one_weight_per_member(self) -> Self:
        if isinstance(self.weighting, list) and len(self.weighting) != len(self.members):
            msg = f"{len(self.weighting)} weights for {len(self.members)} members"
            raise ValueError(msg)
        return self


# BFCL v4's leaderboard aggregates, as bfcl_eval's generate_leaderboard_csv builds them (eval_checker/
# eval_runner_helper.py).
AGGREGATES = {
    "overall": _Aggregate(
        weighting=[10, 10, 10, 30, 40],
        members=["non_live", "live", "irrelevance_detection", "multi_turn", "agentic"],
    ),
    "non_live": _Aggregate(weighting="mean", members=["simple", "multiple", "parallel", "parallel_multiple"]),
    "simple": _Aggregate(weighting="mean", members=["simple_python", "simple_java", "simple_javascript"]),
    "live": _Aggregate(
        weighting="count",
        members=["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"],
    ),
    "irrelevance_detection": _Aggregate(weighting="mean", members=["irrelevance", "live_irrelevance"]),
    "multi_turn": _Aggregate(
        weighting="mean",
        members=["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
    ),
    "agentic": _Aggregate(weighting="mean", members=["web_search", "memory"]),
    "web_search": _Aggregate(weighting="mean", members=["web_search_base", "web_search_no_snippet"]),
    "memory": _Aggregate(weighting="mean", members=["memory_kv", "memory_vector", "memory_rec_sum"]),
}

# What the report shows, in this order: report name -> aggregate or test category.
REPORTED = {
    "overall": "overall",
    "non_live": "non_live",
    "live": "live",
    "multi_turn": "multi_turn",
    "web_search": "web_search",
    "memory": "memory",
    "relevance": "live_relevance",
    "irrelevance": "irrelevance_detection",
}

# The test categories scored on their own: the aggregates' leaves, plus live_relevance, which is in no aggregate.
LEAVES = {m for a in AGGREGATES.values() for m in a.members} - set(AGGREGATES) | {"live_relevance"}


class BFCLReportDriver(BaseReportDriver):
    """Report of a model's BFCL runs: each test category set's folder in ``<runs>/<model>/<run_id>/results/bfcl``.

    Scores are read from the ``BFCL_v4_{category}_score.json`` files domyn-bfcl's ``evaluate`` writes under
    ``<category set>/<model>/``, whose first line holds a test category's accuracy and sample count.
    """

    _SCORE_FILE = re.compile(r"BFCL_v\d+_(?P<category>.+)_score\.json")

    @property
    def suite(self) -> str:
        """Name of the suite whose results this reports."""
        return "bfcl"

    def run_dirs(self) -> list[Path]:
        """Each test category set's folder in each of the model's runs that holds a score file."""
        return sorted(p for p in self.model_dir.glob(f"*/results/{self.suite}/*") if any(self._score_files(p)))

    @staticmethod
    def _score_files(run_dir: Path) -> Iterator[Path]:
        return run_dir.glob("*/**/BFCL_v*_score.json")

    @staticmethod
    def _checker(category: str) -> str:
        """The bfcl_eval checker that scores a test category."""
        if category in ("irrelevance", "live_irrelevance", "live_relevance"):
            return "relevance"
        if category.startswith("multi_turn"):
            return "multi_turn"
        if category.startswith(("web_search", "memory")):
            return "agentic"
        return "ast"

    def parse_run(self, run_dir: Path) -> list[ReportRow]:
        """One ``{name}_accuracy`` row per ``REPORTED`` name scored in ``run_dir``.

        The benchmark is the run's test category set (``single_turn``, ``all_scoring``, ...). Each aggregate is rebuilt
        from its test categories' unrounded scores, the way BFCL builds its leaderboard, and reported only when every
        member was scored: BFCL counts a missing category as 0% (``single_turn`` runs get a ~24% overall), which would
        read as a genuine failure.

        Raises:
            DuplicateBenchmarkError: If a test category is scored twice in ``run_dir`` (two model folders).
        """
        rows: dict[str, ReportRow] = {}
        paths: dict[str, Path] = {}
        for path in sorted(self._score_files(run_dir)):
            match = self._SCORE_FILE.fullmatch(path.name)
            category = match["category"] if match else ""
            if category not in LEAVES:
                logger.debug("bfcl: %s is not a leaderboard category; skipped", path.name)
                continue
            if category in paths:
                clash = [str(p.relative_to(run_dir)) for p in (paths[category], path)]
                raise DuplicateBenchmarkError(self.suite, run_dir, {category: clash})
            paths[category] = path
            with path.open() as f:
                header = json.loads(f.readline())  # the rest are the failed entries
            accuracy, samples = float(header["accuracy"]), int(header["total_count"])
            stderr = math.sqrt(accuracy * (1 - accuracy) / (samples - 1)) if samples > 1 else None
            rows[category] = ReportRow(
                benchmark=category,
                metric="accuracy",
                score=accuracy,
                stderr=stderr,
                samples=samples,
                scorer=self._checker(category),
            )

        aggregates: dict[str, ReportRow | None] = {}

        def build(name: str) -> ReportRow | None:
            if name not in AGGREGATES:
                return rows.get(name)
            if name not in aggregates:
                aggregates[name] = self._aggregate(name, [build(m) for m in AGGREGATES[name].members])
            return aggregates[name]

        return [
            row.model_copy(update={"benchmark": run_dir.name, "metric": f"{name}_accuracy"})
            for name, source in REPORTED.items()
            if (row := build(source)) is not None
        ]

    @staticmethod
    def _aggregate(name: str, parts: list[ReportRow | None]) -> ReportRow | None:
        """``name``'s row from its members' rows; None unless every member is scored.

        The stderr combines the members' as independent samples: categories never share an entry, so
        ``sqrt(sum(w_i^2 * stderr_i^2))`` over the normalised weights.
        """
        weighting = AGGREGATES[name].weighting
        missing = [m for m, p in zip(AGGREGATES[name].members, parts, strict=True) if p is None]
        if missing:
            logger.debug("bfcl: %s not reported, %s not scored", name, ", ".join(missing))
            return None
        scored = [p for p in parts if p is not None]
        if weighting == "mean":
            weights = [1.0] * len(scored)
        elif weighting == "count":
            weights = [float(p.samples or 0) for p in scored]
        else:
            weights = [float(w) for w in weighting]
        total = sum(weights)
        weights = [w / total for w in weights]

        stderr = None
        if all(p.stderr is not None for p in scored):
            stderr = math.sqrt(sum((w * (p.stderr or 0)) ** 2 for w, p in zip(weights, scored, strict=True)))
        scorers = {p.scorer for p in scored}
        return ReportRow(
            benchmark=name,
            metric="accuracy",
            score=sum(w * p.score for w, p in zip(weights, scored, strict=True)),
            stderr=stderr,
            samples=sum(p.samples or 0 for p in scored),
            scorer=scorers.pop() if len(scorers) == 1 else None,
        )
