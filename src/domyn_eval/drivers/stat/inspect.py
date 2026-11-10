# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import zipfile
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log, read_eval_log_samples

from domyn_eval.drivers.stat.base import BaseStatDriver, Sample
from domyn_eval.drivers.stat.loop import find_loop

logger = logging.getLogger(__name__)

_TRUNCATED = ("max_tokens", "model_length")

_INSPECT_VERSION = version("inspect_ai")


class InspectStatDriver(BaseStatDriver):
    """Stats of a run's inspect samples, read from the ``.eval`` logs the report reads.

    A log written by the installed inspect is read straight from its zip, many times faster than through inspect,
    which validates every event of every sample. Any other log goes through inspect, which knows its format.

    Only a sample's final generation is seen, so an agent truncated in an earlier turn is not counted.
    """

    @property
    def suite(self) -> str:
        """Name of the suite whose samples this reads."""
        return "inspect"

    def samples(self) -> Iterator[Sample]:
        """The samples of every successful ``.eval`` log, under the task's report name."""
        for path in sorted(self.results_dir.glob("*.eval")):
            header = read_eval_log(str(path), header_only=True)
            if header.status != "success":
                logger.debug("inspect: %s is '%s'; skipped", path, header.status)
                continue
            benchmark = header.eval.task_display_name or header.eval.task
            written_by = header.eval.packages.get("inspect_ai")
            read = _read_zip if written_by == _INSPECT_VERSION else _read_inspect
            for sample in read(path):
                yield _sample(benchmark, sample)


def _read_zip(path: Path) -> Iterator[dict[str, Any]]:
    # Logs are zstd-compressed: importing inspect_ai adds zstd to zipfile before Python 3.14.
    with zipfile.ZipFile(path) as log:
        # A retried sample is written again; its last entry is the final attempt.
        entries = {e.filename: e for e in log.infolist() if e.filename.startswith("samples/")}
        for entry in entries.values():
            yield json.loads(log.read(entry))


def _read_inspect(path: Path) -> Iterator[dict[str, Any]]:
    for sample in read_eval_log_samples(str(path), all_samples_required=False):
        yield {"output": sample.output.model_dump(), "scores": sample.scores}


def _sample(benchmark: str, sample: dict[str, Any]) -> Sample:
    output = sample["output"]
    reasoning, truncated = [], False
    if output["choices"]:  # none when the sample errored
        choice = output["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            reasoning = [c["reasoning"] for c in content if c["type"] == "reasoning"]
        truncated = choice["stop_reason"] in _TRUNCATED
    usage = output.get("usage")
    return Sample(
        benchmark=benchmark,
        answer=output["completion"],
        reasoning="\n".join(reasoning) or None,
        truncated=truncated,
        looping=any(find_loop(text) for text in (output["completion"], *reasoning)),
        completion_tokens=usage["output_tokens"] if usage else None,
        scored=bool(sample.get("scores")),
    )
