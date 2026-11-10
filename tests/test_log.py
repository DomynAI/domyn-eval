"""Tests for domyn-eval's console logging."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from domyn_eval.log import display_path, set_level, setup_logging


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Put back the root handlers and levels setup_logging replaces, for the tests after these."""
    root = logging.getLogger()
    own = logging.getLogger("domyn_eval")
    handlers, levels = root.handlers[:], (root.level, own.level)
    yield
    root.handlers[:] = handlers
    root.setLevel(levels[0])
    own.setLevel(levels[1])


def test_domyn_eval_logs_at_its_level_and_other_libraries_only_warnings() -> None:
    setup_logging("INFO")
    assert logging.getLogger("domyn_eval.run").isEnabledFor(logging.INFO)
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    assert logging.getLogger("httpx").isEnabledFor(logging.WARNING)


def test_the_configs_level_only_changes_domyn_evals() -> None:
    setup_logging()
    set_level("debug")
    assert logging.getLogger("domyn_eval.run").isEnabledFor(logging.DEBUG)
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)


def test_display_paths_are_relative_to_the_working_directory_when_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert display_path(tmp_path / "runs" / "a.log") == "runs/a.log"
    assert display_path("/elsewhere/a.log") == "/elsewhere/a.log"


@pytest.mark.parametrize(("level", "traceback"), [("INFO", False), ("DEBUG", True)])
def test_tracebacks_only_at_debug_level(level: str, traceback: bool, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level)
    msg = "boom"
    try:
        raise ValueError(msg)  # noqa: TRY301
    except ValueError:
        logging.getLogger("domyn_eval.run").exception("%s: failed", "inspect")
    out = capsys.readouterr().out
    assert ("Traceback" in out) is traceback
    assert ("inspect: failed; boom" in out) is not traceback  # the traceback names the error itself
