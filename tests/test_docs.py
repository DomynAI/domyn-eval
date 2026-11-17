"""docs/config.md documents every config key, and nothing else."""

import dataclasses
import re
from pathlib import Path

import pytest

from domyn_eval import config
from domyn_eval.drivers.run.inspect import InspectTask

CONFIG_MD = Path(__file__).parents[1] / "docs" / "config.md"


def _fields(cls: type, *, without: tuple[str, ...] = ()) -> set[str]:
    names = set(cls.model_fields) if hasattr(cls, "model_fields") else {f.name for f in dataclasses.fields(cls)}
    return names - set(without)


# Each section heading's keys; set at run time, or subsections of their own, are left out.
SECTIONS = {
    "[general]": _fields(config.GeneralConfig, without=("file",)),
    "[model]": _fields(config.ModelConfig, without=("endpoint_max_model_len", "generation")),
    "[model.generation]": _fields(config.GenerationConfig, without=("think", "nothink", "user", "grader", "context")),
    "[model.generation.think]": _fields(config.SamplingConfig),
    "[model.generation.context]": _fields(config.ContextConfig),
    "[slurm]": _fields(config.SlurmConfig),
    "[inspect]": _fields(config.InspectConfig) | _fields(InspectTask),  # with its custom_tasks table
    "[bfcl]": _fields(config.BFCLConfig),
    "[lm_eval]": _fields(config.LMEvalConfig),
}


def _documented() -> dict[str, set[str]]:
    """The keys of the tables under each ``## `[section]` ...`` heading."""
    keys: dict[str, set[str]] = {}
    section = None
    for line in CONFIG_MD.read_text().splitlines():
        if line.startswith("#"):
            match = re.search(r"`(\[[^\]]+\])`", line)
            section = match.group(1) if match else None
        elif section and (row := re.match(r"\| `(\w+)` \|", line)):
            keys.setdefault(section, set()).add(row.group(1))
    return keys


@pytest.mark.parametrize("section", SECTIONS)
def test_every_key_of_a_section_is_documented(section: str) -> None:
    assert _documented().get(section) == SECTIONS[section]


def test_every_suite_has_a_section() -> None:
    assert {f"[{suite}]" for suite in config.registered_suites()} <= set(_documented())
