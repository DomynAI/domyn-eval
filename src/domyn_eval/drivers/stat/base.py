# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from domyn_eval.config import load_config
from domyn_eval.paths import results_dir, run_dir

_Share = Annotated[float, Field(ge=0, le=1)]


class StatsProfile(BaseModel):
    """When the stats flag a benchmark's scores as untrustworthy, as shares of its samples.

    Attributes:
        max_truncated: The share cut off by the token budget above which scores are flagged.
        max_empty: The share of empty answers above which scores are flagged.
        min_reasoning: With reasoning on, the share with a reasoning trace below which scores are flagged.
        max_reasoning: With reasoning off, the share with a reasoning trace above which scores are flagged.
        max_looping: The share that ends stuck in a loop above which scores are flagged.
        near_budget: The share of its token budget a request must use to count as truncated, for results that
            record no stop reason (bfcl, before domyn-bfcl recorded it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_truncated: _Share = 0.05
    max_empty: _Share = 0.10
    min_reasoning: _Share = 0.50
    max_reasoning: _Share = 0.10
    max_looping: _Share = 0.05
    near_budget: _Share = 0.98

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a profile from a YAML file; the thresholds it leaves out keep their defaults."""
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


class Sample(BaseModel):
    """One answer of a run, reduced to what the stats count.

    Reasoning is only seen when the server splits it out of the answer, i.e. serves the model with a reasoning parser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str
    truncated: bool
    looping: bool
    scored: bool
    answer: str = ""
    reasoning: str | None = None
    completion_tokens: int | None = None


class StatRow(BaseModel):
    """How many of a benchmark's samples in a run show each problem."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str
    samples: int
    truncated: int
    reasoning: int
    looping: int
    empty: int
    unscored: int
    max_completion_tokens: int | None = None

    def warnings(self, profile: StatsProfile, *, reasoning: bool) -> list[str]:
        """What makes the benchmark's scores untrustworthy.

        Args:
            profile: The thresholds.
            reasoning: Whether the run had reasoning on.
        """
        share = {k: getattr(self, k) / self.samples for k in ("truncated", "reasoning", "looping", "empty")}
        warnings = []
        if share["truncated"] > profile.max_truncated:
            warnings.append(f"{share['truncated']:.0%} truncated")
        if reasoning and share["reasoning"] < profile.min_reasoning:
            warnings.append(f"only {share['reasoning']:.0%} reason, with reasoning on")
        if not reasoning and share["reasoning"] > profile.max_reasoning:
            warnings.append(f"{share['reasoning']:.0%} reason, with reasoning off")
        if share["looping"] > profile.max_looping:
            warnings.append(f"{share['looping']:.0%} stuck in a loop")
        if share["empty"] > profile.max_empty:
            warnings.append(f"{share['empty']:.0%} empty")
        if self.unscored:
            warnings.append(f"{self.unscored} unscored")
        return warnings


class BaseStatDriver(ABC):
    """Reads one run's samples of a suite and counts what can make its scores untrustworthy.

    Each suite subclasses it and implements ``suite`` and ``samples``.
    """

    @property
    @abstractmethod
    def suite(self) -> str:
        """Name of the suite whose samples this reads."""

    def __init__(self, model: str, run_id: str, profile: StatsProfile | None = None):
        """Point the stats at the suite's results in a run, ``<runs>/<model>/<run_id>/results/<suite>``.

        The run's own ``config.toml`` tells whether it reasoned, and its token budget.

        Args:
            model: The model's id, which names its folder under the runs root.
            run_id: The run.
            profile: The thresholds; None uses the defaults.

        Raises:
            FileNotFoundError: If the run has no ``config.toml``.
        """
        self.config = load_config(run_dir(model, run_id) / "config.toml")
        self.profile = profile or StatsProfile()
        self.results_dir = results_dir(model, run_id, self.suite)

    @abstractmethod
    def samples(self) -> Iterator[Sample]:
        """Every sample of the run, read from the suite's results."""

    def stats(self) -> list[StatRow]:
        """A row per benchmark, in the order the samples come."""
        tallies: dict[str, Counter[str]] = {}
        longest: dict[str, int] = {}
        for sample in self.samples():
            tallies.setdefault(sample.benchmark, Counter()).update(
                samples=1,
                truncated=sample.truncated,
                reasoning=bool(sample.reasoning),
                looping=sample.looping,
                empty=not sample.answer.strip(),
                unscored=not sample.scored,
            )
            if sample.completion_tokens is not None:
                longest[sample.benchmark] = max(longest.get(sample.benchmark, 0), sample.completion_tokens)
        counts = ("samples", "truncated", "reasoning", "looping", "empty", "unscored")
        return [
            StatRow(benchmark=b, max_completion_tokens=longest.get(b), **{k: tally[k] for k in counts})
            for b, tally in tallies.items()
        ]
