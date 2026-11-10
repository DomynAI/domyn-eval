# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Import a suite's drivers, registered with ``config.register_suite``."""

import importlib
from typing import TYPE_CHECKING, Any

from domyn_eval.config import suite_drivers

if TYPE_CHECKING:
    from domyn_eval.drivers.report.base import BaseReportDriver
    from domyn_eval.drivers.run.base import BaseRunDriver
    from domyn_eval.drivers.stat.base import BaseStatDriver


def run_driver(suite: str) -> type["BaseRunDriver"]:
    """Return the run driver class of ``suite``."""
    return _import(suite_drivers(suite).run)


def report_driver(suite: str) -> type["BaseReportDriver"]:
    """Return the report driver class of ``suite``."""
    return _import(suite_drivers(suite).report)


def stat_driver(suite: str) -> type["BaseStatDriver"] | None:
    """Return the stat driver class of ``suite``; None if its results keep no samples."""
    path = suite_drivers(suite).stat
    return _import(path) if path else None


def _import(path: str) -> Any:  # noqa: ANN401 - the caller knows the class it loads
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)
