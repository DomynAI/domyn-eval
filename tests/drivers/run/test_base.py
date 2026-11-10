"""Tests for the logic BaseRunDriver shares across all suite drivers."""

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from domyn_eval import config as config_module
from domyn_eval.config import Config
from domyn_eval.drivers.run.base import BaseRunDriver


@dataclass
class DemoConfig:
    path: str = "suites/demo"


class DemoDriver(BaseRunDriver):
    @property
    def suite(self) -> str:
        return "demo"

    def build_command(self) -> list[str]:
        return ["true"]

    def setup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def demo_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the demo suite for one test; monkeypatch removes it again afterwards, so other tests never see it."""
    monkeypatch.setitem(config_module._SUITES, "demo", DemoConfig)


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir so the drivers' run folders land there, not in runs/."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


def test_output_dir_is_the_suites_results_dir_of_the_run(config: Config, runs_root: Path) -> None:
    config.general.run_id = "r1"
    driver = DemoDriver(config)
    assert driver.output_dir == str(runs_root / "test-model" / "r1" / "results" / "demo")
    assert Path(driver.output_dir).is_dir()


def test_model_id_is_required(config: Config) -> None:
    config.model.id = None
    with pytest.raises(ValueError, match=r"\[model\] id"):
        DemoDriver(config)


def test_run_id_is_generated_and_shared_through_the_config(config: Config) -> None:
    driver = DemoDriver(config)
    assert driver.run_id
    assert config.general.run_id == driver.run_id  # the next suite of the run reads the same id


def test_write_config_saves_the_resolved_window(config: Config) -> None:
    config.generation.max_model_len = 0
    config.model.endpoint_max_model_len = 65536
    path = DemoDriver(config).write_config()
    assert tomllib.loads(path.read_text())["generation"]["max_model_len"] == 65536
    assert config.generation.max_model_len == 0  # the shared config keeps its own value


def test_write_config_keeps_every_written_suites_section(config: Config) -> None:
    class LMEvalDemoDriver(DemoDriver):
        @property
        def suite(self) -> str:
            return "lm_eval"

    path = DemoDriver(config).write_config()
    LMEvalDemoDriver(config).write_config()  # the same run, so the same file
    raw = tomllib.loads(path.read_text())
    assert raw["demo"] == {"path": "suites/demo"}  # not in the loaded TOML: saved because it was used
    assert "lm_eval" in raw


def test_sbatch_command_runs_in_command_cwd_with_command_env(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "suite dir"  # a space, which must survive without quoting
    workdir.mkdir()
    monkeypatch.setattr(DemoDriver, "command_cwd", property(lambda self: workdir))  # noqa: ARG005
    monkeypatch.setattr(DemoDriver, "command_env", property(lambda self: {"DEMO_VAR": "a b"}))  # noqa: ARG005
    monkeypatch.setattr(DemoDriver, "build_command", lambda self: ["sh", "-c", 'pwd; echo "$DEMO_VAR"'])  # noqa: ARG005
    args = DemoDriver(config).sbatch_command()
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [str(workdir.resolve()), "a b"]


def test_a_commands_output_goes_only_to_its_own_log(config: Config, capfd: pytest.CaptureFixture[str]) -> None:
    driver = DemoDriver(config)
    assert driver._run_command(["sh", "-c", "echo out; echo err >&2"], log="run") == 0
    assert driver._run_command(["echo", "synced"], log="setup") == 0
    assert capfd.readouterr() == ("", "")
    header, *output = (driver.log_dir / "run.log").read_text().splitlines()
    assert header.startswith("# ")
    assert header.endswith("$ sh -c 'echo out; echo err >&2'")
    assert output == ["out", "err"]
    assert (driver.log_dir / "setup.log").read_text().splitlines()[1:] == ["synced"]


def test_a_failed_command_logs_its_last_lines(config: Config, caplog: pytest.LogCaptureFixture) -> None:
    driver = DemoDriver(config)
    with caplog.at_level("ERROR", logger="domyn_eval"):
        code = driver._run_command(["sh", "-c", "seq 1 20; exit 3"], log="run")
    assert code == 3
    (record,) = caplog.records
    message = record.getMessage()
    assert message.startswith("demo: run failed (exit 3); see ")
    assert message.splitlines()[1:] == [f"  │ {n}" for n in range(11, 21)]
