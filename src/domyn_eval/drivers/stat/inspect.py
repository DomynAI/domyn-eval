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

_ATTACHMENT = "attachment://"

_INSPECT_VERSION = version("inspect_ai")


class InspectStatDriver(BaseStatDriver):
    """Stats of a run's inspect samples, read from the ``.eval`` logs the report reads.

    A log written by the installed inspect is read straight from its zip, many times faster than through inspect,
    which validates every event of every sample. Any other log goes through inspect, which knows its format.

    In an agentic eval, answer, reasoning and loops are checked on the evaluated model's last turn only; truncation and
    completion tokens look at all its turns.
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
        events = [
            {"event": "model", "role": e.role, "output": e.output.model_dump()}
            if e.event == "model"
            else {"event": "score"}
            for e in sample.events
            if e.event in ("model", "score")
        ]
        yield {
            "output": sample.output.model_dump(),
            "scores": sample.scores,
            "events": events,
            "attachments": sample.attachments,
        }


def _generations(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """The evaluated model's generations before scoring; the sample's output when the log records none.

    The sample's output is whichever model spoke last, e.g. the user simulator of an agentic task. The evaluated
    model is the one without a role: a user simulator or grader has one even when it is the same model (self-play).
    """
    generations = []
    for event in sample.get("events") or []:
        if event["event"] == "score":
            break
        if event["event"] == "model" and event.get("role") is None:
            generations.append(event["output"])
    return generations or [sample["output"]]


def _attached(text: str, attachments: dict[str, str]) -> str:
    """``text``, or the attachment it refers to: a log moves its events' long texts to the sample's attachments."""
    if text.startswith(_ATTACHMENT):
        return attachments.get(text.removeprefix(_ATTACHMENT), text)
    return text


def _sample(benchmark: str, sample: dict[str, Any]) -> Sample:
    generations = _generations(sample)
    output = generations[-1]
    attachments = sample.get("attachments") or {}
    reasoning = []
    if output["choices"]:  # none when the sample errored
        content = output["choices"][0]["message"]["content"]
        if isinstance(content, list):
            reasoning = [_attached(c["reasoning"], attachments) for c in content if c["type"] == "reasoning"]
    tokens = [g["usage"]["output_tokens"] for g in generations if g.get("usage")]
    return Sample(
        benchmark=benchmark,
        answer=output["completion"],
        reasoning="\n".join(reasoning) or None,
        truncated=any(g["choices"] and g["choices"][0]["stop_reason"] in _TRUNCATED for g in generations),
        looping=any(find_loop(text) for text in (output["completion"], *reasoning)),
        completion_tokens=max(tokens, default=None),
        scored=bool(sample.get("scores")),
    )
