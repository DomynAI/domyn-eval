# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from domyn_eval.drivers.report.bfcl import LEAVES
from domyn_eval.drivers.stat.base import BaseStatDriver, Sample
from domyn_eval.drivers.stat.loop import find_loop

logger = logging.getLogger(__name__)


def _flatten(value: Any) -> Iterator[Any]:  # noqa: ANN401 - BFCL nests values per turn and per step
    if isinstance(value, list | tuple):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


class BFCLStatDriver(BaseStatDriver):
    """Stats of a run's BFCL entries, read from the ``*_result.json`` files of each test category set.

    Each test category set (``single_turn``, ``all_scoring``, ...) is a benchmark, as in the report.

    A multi-turn entry makes one request per step of each turn. Its answer joins the requests' outputs, and it is
    truncated when any request was.

    A request is truncated when its ``finish_reason`` is ``length``. Results written before domyn-bfcl recorded the
    finish reason fall back to the token counts: a request is truncated when it used at least the profile's
    ``near_budget`` share of its token budget.
    """

    _RESULT_FILE = re.compile(r"BFCL_v\d+_(?P<category>.+)_result\.json")

    @property
    def suite(self) -> str:
        """Name of the suite whose samples this reads."""
        return "bfcl"

    def samples(self) -> Iterator[Sample]:
        """The entries of every leaderboard test category, in each test category set's folder."""
        for category_dir in sorted(p for p in self.results_dir.glob("*") if p.is_dir()):
            for path in sorted(category_dir.glob("*/**/BFCL_v*_result.json")):
                match = self._RESULT_FILE.fullmatch(path.name)
                if not match or match["category"] not in LEAVES:
                    logger.debug("bfcl: %s is not a leaderboard category; skipped", path.name)
                    continue
                scored = path.with_name(path.name.replace("_result.json", "_score.json")).is_file()
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            yield self._sample(category_dir.name, json.loads(line), scored=scored)

    def _sample(self, benchmark: str, entry: dict[str, Any], *, scored: bool) -> Sample:
        # A step that failed has no token counts.
        steps = zip(_flatten(entry.get("input_token_count")), _flatten(entry.get("output_token_count")), strict=False)
        counted = [(int(p), int(c)) for p, c in steps if p is not None and c is not None]
        answers = [str(r) for r in _flatten(entry.get("result")) if r is not None]
        reasoning = [r for r in _flatten(entry.get("reasoning_content")) if r]
        return Sample(
            benchmark=benchmark,
            answer="\n".join(answers),
            reasoning="\n".join(reasoning) or None,
            truncated=(
                "length" in entry["finish_reason"]
                if "finish_reason" in entry
                else any(self._near_budget(p, c) for p, c in counted)
            ),
            looping=any(find_loop(text) for text in answers + reasoning),
            completion_tokens=max((c for _, c in counted), default=None),
            scored=scored,
        )

    def _near_budget(self, prompt: int, completion: int) -> bool:
        """Whether a step reached its budget: ``max_tokens``, cut to what the prompt leaves of the window."""
        context = self.config.model.generation.context
        budget = context.max_tokens
        window = context.max_model_len  # the run saves the endpoint's when unset
        if window:
            room = window - prompt - context.token_safety_margin
            budget = min(budget or room, room)
        return budget > 0 and completion >= self.profile.near_budget * budget
