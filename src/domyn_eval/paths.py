# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Locations of suites and run outputs.

A run writes everything under ``<runs_root>/<model>/<run_id>/``: its config
snapshot, ``logs/<suite>/`` and ``results/<suite>/``.
"""

import os
import re
from pathlib import Path

UNRESOLVED_REF = re.compile(r"\$\{[^}]*\}")


def resolve_env_path(name: str, default: str = "") -> str:
    """Return env var ``name`` with its ``${...}`` references expanded.

    Args:
        name: The environment variable.
        default: The value used when ``name`` is not set.

    Raises:
        RuntimeError: If a reference names an unset variable.
    """
    value = os.path.expandvars(os.getenv(name, default))
    leftover = UNRESOLVED_REF.search(value)
    if leftover:
        msg = f"{name}={value!r} references unset {leftover.group(0)}"
        raise RuntimeError(msg)
    return value


def suites_base_dir() -> Path:
    """Return the directory relative suite paths are resolved from: ``SUITES_BASE_DIR``, else the working directory."""
    return Path(resolve_env_path("SUITES_BASE_DIR", ".")).resolve()


def suite_dir(path: str) -> Path:
    """Return the checkout of a suite.

    Args:
        path: The suite's ``path`` setting, relative to ``suites_base_dir()``.
    """
    return suites_base_dir() / path


def suite_python(path: str) -> Path:
    """Return the interpreter of the ``.venv`` in a suite's checkout.

    Args:
        path: The suite's ``path`` setting, relative to ``suites_base_dir()``.
    """
    return suite_dir(path) / ".venv" / "bin" / "python"


def runs_root() -> Path:
    """Return the base directory of runs: ``RUNS_DIR``, else ``runs``."""
    return Path(resolve_env_path("RUNS_DIR", "runs"))


def run_dir(model: str, run_id: str) -> Path:
    """Return the directory of a run.

    Args:
        model: The model name.
        run_id: The run id.
    """
    return runs_root() / (model or "unknown") / run_id


def logs_dir(model: str, run_id: str, suite: str | None = None, *, create: bool = False) -> Path:
    """Return the log directory of a run, or of one of its suites.

    Args:
        model: The model name.
        run_id: The run id.
        suite: The suite name, if any.
        create: Whether to create the directory.
    """
    return _run_subdir("logs", model, run_id, suite, create=create)


def results_dir(model: str, run_id: str, suite: str | None = None, *, create: bool = False) -> Path:
    """Return the results directory of a run, or of one of its suites.

    Args:
        model: The model name.
        run_id: The run id.
        suite: The suite name, if any.
        create: Whether to create the directory.
    """
    return _run_subdir("results", model, run_id, suite, create=create)


def _run_subdir(kind: str, model: str, run_id: str, suite: str | None, *, create: bool) -> Path:
    path = run_dir(model, run_id) / kind
    if suite:
        path /= suite
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
