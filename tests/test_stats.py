import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from domyn_eval import config as config_module
from domyn_eval.cli import cli
from domyn_eval.config import Config, SuiteDrivers, register_suite
from domyn_eval.drivers.report.base import BaseReportDriver, ReportRow
from domyn_eval.drivers.stat.base import BaseStatDriver, Sample
from domyn_eval.stats import ModelStats
from tests.drivers.stat.helpers import normal_sample


class ScoresReport(BaseReportDriver):
    """Reports one score per run, so that the model has a report."""

    @property
    def suite(self) -> str:
        return "counted"

    def parse_run(self, run_dir: Path) -> list[ReportRow]:  # noqa: ARG002
        return [ReportRow(benchmark="gsm8k", metric="accuracy", score=0.5)]


class SamplesStats(BaseStatDriver):
    """Reads the samples a test wrote to each run's ``samples.json``."""

    @property
    def suite(self) -> str:
        return "counted"

    def samples(self) -> Iterator[Sample]:
        for sample in json.loads((self.results_dir / "samples.json").read_text()):
            yield Sample.model_validate(sample)


@dataclasses.dataclass
class NoConfig:
    pass


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    # Next to the real suites, which the saved configs have sections of.
    monkeypatch.setattr(config_module, "_SUITES", dict(config_module._SUITES))
    monkeypatch.setattr(config_module, "_DRIVERS", dict(config_module._DRIVERS))
    report = f"{__name__}:ScoresReport"
    register_suite(
        "counted", NoConfig, drivers=SuiteDrivers(run="unused:Run", report=report, stat=f"{__name__}:SamplesStats")
    )
    register_suite("uncounted", NoConfig, drivers=SuiteDrivers(run="unused:Run", report=report))
    return tmp_path


def _run(
    runs: Path, config: Config, run: str, samples: list[Sample], *, suite: str = "counted", reasoning: bool = True
) -> None:
    run_dir = runs / "m" / run
    (run_dir / "results" / suite).mkdir(parents=True)
    (run_dir / "results" / suite / "samples.json").write_text(json.dumps([s.model_dump() for s in samples]))
    config.generation.reasoning = reasoning
    config.save(run_dir / "config.toml")


def _fine(benchmark: str = "gsm8k", **fields: object) -> Sample:
    return normal_sample(**{"benchmark": benchmark, "answer": "42", "reasoning": "because", **fields})


def test_build_counts_each_run_judged_by_its_own_mode(runs: Path, config: Config) -> None:
    _run(runs, config, "r1", [_fine(), _fine("aime", truncated=True)])
    _run(runs, config, "r2", [_fine()], reasoning=False)
    _run(runs, config, "r3", [_fine()], suite="uncounted")  # no stat driver

    stats = ModelStats.build("m")

    assert [(s.suite, s.run, s.row.benchmark, s.row.samples, s.warnings) for s in stats.rows] == [
        ("counted", "r1", "gsm8k", 1, []),
        ("counted", "r1", "aime", 1, ["100% truncated"]),
        ("counted", "r2", "gsm8k", 1, ["100% reason, with reasoning off"]),
    ]


def test_build_counts_only_the_selected_suites(runs: Path, config: Config) -> None:
    _run(runs, config, "r1", [_fine()])

    assert ModelStats.build("m", ["uncounted"]).rows == []


def test_build_skips_a_run_without_its_config(runs: Path, config: Config, caplog: pytest.LogCaptureFixture) -> None:
    _run(runs, config, "r1", [_fine()])
    _run(runs, config, "r2", [_fine()])
    (runs / "m" / "r1" / "config.toml").unlink()

    assert [s.run for s in ModelStats.build("m").rows] == ["r2"]
    assert "counted: run r1 has no config.toml" in caplog.text


def test_render_counts_the_flagged_benchmarks(runs: Path, config: Config) -> None:
    _run(runs, config, "r1", [_fine(), _fine("aime", answer="")])
    console = Console(record=True, width=200)

    ModelStats.build("m").render(console)

    text = console.export_text()
    assert "Stats: m" in text
    assert "100% empty" in text
    assert "1 benchmark runs have scores to distrust" in text


def test_render_of_no_stats_prints_nothing() -> None:
    console = Console(record=True)

    ModelStats("m", []).render(console)

    assert console.export_text() == ""


def test_cli_reports_the_stats_after_the_scores(runs: Path, config: Config) -> None:
    _run(runs, config, "r1", [_fine(looping=True)])

    result = CliRunner().invoke(cli, ["report", "m"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert result.output.index("Report: m") < result.output.index("Stats: m")
    assert "100% stuck in a loop" in result.output


def test_cli_judges_the_stats_with_the_given_profile(runs: Path, config: Config, tmp_path: Path) -> None:
    _run(runs, config, "r1", [_fine(looping=True)])
    profile = tmp_path / "stats.yaml"
    profile.write_text("max_looping: 1\n")

    result = CliRunner().invoke(cli, ["report", "m", "--stats-profile", str(profile)], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert "Stats: m" in result.output
    assert "stuck in a loop" not in result.output


@pytest.mark.parametrize("flag", ["--no-stats", "--leaderboard"])
def test_cli_can_leave_the_stats_out(runs: Path, config: Config, flag: str) -> None:
    _run(runs, config, "r1", [_fine()])

    result = CliRunner().invoke(cli, ["report", "m", flag])

    assert result.exit_code == 0, result.output
    assert "Stats" not in result.output


@pytest.mark.parametrize("flag", ["--no-stats", "--leaderboard"])
def test_cli_rejects_a_stats_profile_without_stats(runs: Path, flag: str) -> None:
    profile = runs / "stats.yaml"
    profile.write_text("{}")

    result = CliRunner().invoke(cli, ["report", "m", flag, "--stats-profile", str(profile)])

    assert result.exit_code == 2
    assert "--stats-profile goes with the stats" in result.output


def test_cli_rejects_a_bad_stats_profile(runs: Path, config: Config) -> None:
    _run(runs, config, "r1", [_fine()])
    profile = runs / "stats.yaml"
    profile.write_text("max_looping: 2\n")

    result = CliRunner().invoke(cli, ["report", "m", "--stats-profile", str(profile)])

    assert result.exit_code == 1
    assert "max_looping" in result.output
