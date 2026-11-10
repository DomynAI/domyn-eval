# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""domyn-eval's console logging: one Rich handler; domyn-eval at its level, other libraries at WARNING."""

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Send every record through one Rich handler; domyn-eval logs at ``level``, other libraries only warnings."""
    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(show_path=False, rich_tracebacks=True, markup=False)
        handler.addFilter(_traceback_only_when_debugging)
        root.addHandler(handler)
    root.setLevel(logging.WARNING)
    set_level(level)


def _traceback_only_when_debugging(record: logging.LogRecord) -> bool:
    """Below debug level, replace a record's traceback with its error, appended to the message."""
    if record.exc_info and not logging.getLogger("domyn_eval").isEnabledFor(logging.DEBUG):
        record.msg, record.args = f"{record.getMessage()}; {record.exc_info[1]}", None
        record.exc_info = record.exc_text = None
    return True


def set_level(level: str) -> None:
    """Set the level of domyn-eval's own logs."""
    logging.getLogger("domyn_eval").setLevel(level.upper())


def display_path(path: Path | str) -> str:
    """A path for a console message: relative to the working directory when under it, else absolute."""
    path = Path(path).absolute()
    return str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path)
