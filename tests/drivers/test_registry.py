import dataclasses

import pytest

from domyn_eval import config
from domyn_eval.config import SuiteDrivers, register_suite, registered_suites
from domyn_eval.drivers import report_driver, run_driver, stat_driver


@pytest.mark.parametrize("suite", registered_suites())
def test_builtin_drivers_name_their_suite(suite: str) -> None:
    assert report_driver(suite)("m").suite == suite
    assert run_driver(suite).__name__.endswith("RunDriver")


@pytest.mark.parametrize("suite", ["bfcl", "inspect"])
def test_builtin_stat_drivers_name_their_suite(suite: str, saved_run: tuple[str, str]) -> None:
    driver = stat_driver(suite)
    assert driver is not None
    assert driver(*saved_run).suite == suite


def test_a_suite_whose_results_keep_no_samples_has_no_stat_driver() -> None:
    assert stat_driver("lm_eval") is None


@dataclasses.dataclass
class NoConfig:
    pass


def test_drivers_are_imported_only_on_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_SUITES", {})
    monkeypatch.setattr(config, "_DRIVERS", {})
    register_suite("missing", NoConfig, drivers=SuiteDrivers(run="not_installed:Run", report="not_installed:Report"))

    with pytest.raises(ModuleNotFoundError, match="not_installed"):
        report_driver("missing")


def test_unknown_suite_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match=r"unknown suite 'nope'; .*'inspect'"):
        report_driver("nope")
