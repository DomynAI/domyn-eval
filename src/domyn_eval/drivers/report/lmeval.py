# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domyn_eval.drivers.report.base import BaseReportDriver, ReportRow

logger = logging.getLogger(__name__)


class LMEvalTaskResult(BaseModel):
    """The scores of one task or group, as lm_eval writes them under ``results``.

    Each metric is a key named ``"<metric>,<filter>"``, with its stderr under ``"<metric>_stderr,<filter>"``:

        "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.75, "exact_match_stderr,strict-match": 0.01}

    The metric names differ per task, so they are not fields of this model: they are kept as extra fields and read
    through ``metrics``.
    """

    model_config = ConfigDict(extra="allow")

    alias: str | None = None
    # A group's own count per metric key; a task's is in LMEvalResults.n_samples_effective.
    sample_count: dict[str, int] = Field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        """The ``"<metric>,<filter>"`` keys (stderr included); metadata keys have no comma."""
        return {k: v for k, v in (self.model_extra or {}).items() if "," in k}


class LMEvalResults(BaseModel):
    """The top-level keys of an lm_eval ``results_<date>.json`` that the report reads."""

    results: dict[str, LMEvalTaskResult]
    # Per task, each metric's direction; None for a group whose tasks disagree.
    higher_is_better: dict[str, dict[str, bool | None]] = Field(default_factory=dict)
    # Per task, from n-samples: the dataset size, and how many samples were run.
    n_samples_original: dict[str, int]
    n_samples_effective: dict[str, int]

    @model_validator(mode="before")
    @classmethod
    def _split_n_samples(cls, data: Any) -> Any:  # noqa: ANN401 - pydantic hands over the raw input
        """lm_eval nests both counts per task, ``{task: {original, effective}}``."""
        if isinstance(data, dict) and "n-samples" in data:
            data = dict(data)
            n_samples = data.pop("n-samples")
            data["n_samples_original"] = {t: c.get("original") for t, c in n_samples.items()}
            data["n_samples_effective"] = {t: c.get("effective") for t, c in n_samples.items()}
        return data


class LMEvalReportDriver(BaseReportDriver):
    """Report of a model's lm_eval runs: each task folder in ``<runs>/<model>/<run_id>/results/lm_eval``."""

    RESULTS_FILES = "results_*.json"  # lm_eval's results_<date>.json

    @property
    def suite(self) -> str:
        """Name of the suite whose results this reports."""
        return "lm_eval"

    def run_dirs(self) -> list[Path]:
        """Each task's folder in each of the model's runs that holds a results file."""
        return sorted({p.parent for p in self.model_dir.glob(f"*/results/{self.suite}/*/{self.RESULTS_FILES}")})

    def parse_run(self, run_dir: Path) -> list[ReportRow]:
        """One row per metric of every task (and group) in the latest results file of ``run_dir``.

        lm_eval keys a metric ``"<metric>,<filter>"`` and its stderr ``"<metric>_stderr,<filter>"``; keys without a
        comma (name, alias, ...) are not metrics. The label is ``{filter}/{metric}``, leaving out lm_eval's ``none``
        filter.
        """
        paths = sorted(run_dir.glob(self.RESULTS_FILES))  # the date in the name sorts them in time
        if not paths:
            return []
        header = LMEvalResults.model_validate(json.loads(paths[-1].read_text()))

        rows = []
        for task, entry in header.results.items():
            effective = header.n_samples_effective.get(task)
            directions = header.higher_is_better.get(task, {})
            metrics = entry.metrics
            for key, value in metrics.items():
                metric, filt = key.split(",", 1)
                score = self._number(value)
                if metric.endswith("_stderr") or score is None:
                    continue
                rows.append(
                    ReportRow(
                        benchmark=task,
                        metric=metric if filt == "none" else f"{filt}/{metric}",
                        score=score,
                        higher_is_better=directions.get(metric) is not False,
                        stderr=self._number(metrics.get(f"{metric}_stderr,{filt}")),
                        samples=entry.sample_count.get(key, effective),
                    )
                )
        return rows

    @staticmethod
    def _number(value: object) -> float | None:
        """A metric value as float; None for lm_eval's ``"N/A"`` (stderr not computed)."""
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
