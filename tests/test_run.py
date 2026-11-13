import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from domyn_eval import run as run_module
from domyn_eval.cli import cli
from domyn_eval.config import Config
from domyn_eval.drivers.run.base import BaseRunDriver
from domyn_eval.preflight import FAIL, OK, WARN, Check
from domyn_eval.run import Entry, RunOptions, run
from domyn_eval.swarm import SwarmHandler


class Calls(list):
    """What the fake drivers and swarm handler were asked to do, in order."""


@pytest.fixture
def calls() -> Calls:
    return Calls()


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs" / "test-model" / "r1"


@pytest.fixture
def config(config: Config) -> Config:
    config.general.run_id = "r1"
    config.general.auto_setup = True
    return config


@pytest.fixture(autouse=True)
def drivers(calls: Calls, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Fake run drivers: ``behaviour[suite]`` sets a suite's exit code, setup error and roles."""
    behaviour: dict[str, dict] = {}

    def driver_class(suite: str) -> type[BaseRunDriver]:
        class FakeDriver(BaseRunDriver):
            @property
            def suite(self) -> str:
                return suite

            @property
            def roles(self) -> frozenset[str]:
                return frozenset(behaviour.get(suite, {}).get("roles", ()))

            def setup(self) -> None:
                calls.append(("setup", suite))
                if error := behaviour.get(suite, {}).get("setup_error"):
                    raise RuntimeError(error)

            def build_command(self) -> list[str]:
                section = getattr(self.config.suite, suite)
                calls.append(("command", suite, getattr(section, "test_category", None)))
                return ["sh", "-c", f"exit {behaviour.get(suite, {}).get('exit', 0)}"]

        return FakeDriver

    monkeypatch.setattr(run_module, "run_driver", driver_class)
    return behaviour


class FakeHandler:
    """A swarm handler that deploys a swarm for whatever has no url, unless dry."""

    ModelType = SwarmHandler.ModelType

    def __init__(self, config: Config, calls: Calls):
        self.config, self.calls = config, calls
        self.fresh: dict[SwarmHandler.ModelType, SimpleNamespace] = {}

    def ensure_endpoint(self, model_type: SwarmHandler.ModelType, dry: bool) -> None:
        self.calls.append(("ensure", model_type.value, dry))
        section = getattr(self.config, model_type.value)
        if section.url or dry:
            return
        section.url = f"http://{model_type.value}-node:8000"
        self.fresh[model_type] = SimpleNamespace(name=f"{model_type.value}-swarm")

    def down(self, model_type: SwarmHandler.ModelType) -> bool:
        self.calls.append(("down", model_type.value))
        return model_type in self.fresh

    def deployed_swarms(self) -> dict[SwarmHandler.ModelType, str]:
        return {t: swarm.name for t, swarm in self.fresh.items()}


@pytest.fixture
def swarm(calls: Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "_swarm_handler", lambda config: FakeHandler(config, calls))


@pytest.fixture
def preflight(calls: Calls, monkeypatch: pytest.MonkeyPatch) -> list[Check]:
    """The checks a fake preflight returns; it records whether a swarm is to be deployed."""
    checks = [Check(OK, "fine")]

    class FakePreflight:
        def __init__(self, _config: Config, suites: list[str], *, deploying: bool = False):
            calls.append(("preflight", tuple(suites), deploying))

        def run(self) -> list[Check]:
            return list(checks)

    monkeypatch.setattr(run_module, "Preflight", FakePreflight)
    return checks


@pytest.fixture(autouse=True)
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """The model cards the endpoint serves."""
    cards = [{"id": "", "max_model_len": 65536}]
    monkeypatch.setattr(run_module, "list_models", lambda *_: cards)
    return cards


NO_CHECK = RunOptions(check=False)


def _ran(calls: Calls) -> list[tuple]:
    return [c for c in calls if c[0] in ("setup", "command")]


@pytest.mark.usefixtures("runs")
def test_entries_run_in_order_each_with_its_overrides(config: Config, calls: Calls) -> None:
    entries = [Entry("bfcl", {"test_category": "simple"}), Entry("lm_eval"), Entry("bfcl", {"test_category": "live"})]

    assert run(config, entries, NO_CHECK) == 0

    assert _ran(calls) == [
        ("setup", "bfcl"),
        ("command", "bfcl", "simple"),
        ("setup", "lm_eval"),
        ("command", "lm_eval", None),
        ("setup", "bfcl"),
        ("command", "bfcl", "live"),
    ]


def test_the_runs_config_holds_every_suite_of_the_run(config: Config, runs: Path) -> None:
    run(config, [Entry("bfcl", {"test_category": "simple"}), Entry("lm_eval")], NO_CHECK)

    saved = tomllib.loads((runs / "config.toml").read_text())

    assert saved["bfcl"]["test_category"] == "simple"
    assert "lm_eval" in saved
    assert saved["model"]["generation"]["context"]["max_model_len"] == 32768


def test_a_suites_entries_each_log_to_their_own_file(config: Config, runs: Path) -> None:
    entries = [Entry("bfcl", {"test_category": "simple"}), Entry("lm_eval"), Entry("bfcl", {"test_category": "live"})]
    run(config, entries, NO_CHECK)
    assert sorted(p.relative_to(runs / "logs").as_posix() for p in (runs / "logs").rglob("*.log")) == [
        "bfcl/run-1.log",
        "bfcl/run-2.log",
        "lm_eval/run.log",
    ]


@pytest.mark.usefixtures("runs")
def test_a_failed_entry_fails_the_run_but_not_the_others(
    config: Config, calls: Calls, drivers: dict[str, dict], capsys: pytest.CaptureFixture[str]
) -> None:
    drivers["bfcl"] = {"exit": 3}
    drivers["inspect"] = {"setup_error": "uv sync failed"}

    assert run(config, [Entry("bfcl"), Entry("inspect"), Entry("lm_eval")], NO_CHECK) == 1

    assert ("command", "lm_eval", None) in calls
    summary = capsys.readouterr().out
    rows = [line.split("│") for line in summary.splitlines() if line.startswith("│")]
    assert [(row[1].strip(), row[2].strip()) for row in rows] == [
        ("bfcl", "failed"),
        ("inspect", "failed"),
        ("lm_eval", "ok"),
    ]
    assert "domyn-eval report test-model" in summary


@pytest.mark.usefixtures("runs")
def test_no_setup_skips_the_setup(config: Config, calls: Calls) -> None:
    config.general.auto_setup = False

    run(config, [Entry("bfcl")], NO_CHECK)

    assert _ran(calls) == [("command", "bfcl", "single_turn")]


@pytest.mark.usefixtures("runs")
def test_the_context_window_is_read_from_the_endpoint(config: Config, served: list[dict]) -> None:
    config.model.path = served[0]["id"] = "org/model"

    run(config, [Entry("bfcl")], NO_CHECK)

    assert config.model.endpoint_max_model_len == 65536


@pytest.mark.usefixtures("runs")
def test_a_failed_preflight_runs_nothing(config: Config, calls: Calls, preflight: list[Check]) -> None:
    preflight.append(Check(FAIL, "endpoint unreachable"))

    assert run(config, [Entry("bfcl")], RunOptions()) == 1

    assert _ran(calls) == []


@pytest.mark.usefixtures("runs", "preflight")
def test_an_entry_needing_a_judge_fails_the_preflight_without_one(
    config: Config, calls: Calls, drivers: dict[str, dict]
) -> None:
    drivers["inspect"] = {"roles": ["user"]}
    config.judge.url = None

    assert run(config, [Entry("inspect")], RunOptions()) == 1

    assert _ran(calls) == []


@pytest.mark.usefixtures("runs", "swarm")
def test_a_swarm_the_run_deploys_comes_down_when_it_ends(config: Config, calls: Calls) -> None:
    config.model.url = None

    run(config, [Entry("bfcl")], RunOptions(swarm=True, check=False))

    assert calls[0] == ("ensure", "model", False)
    assert config.model.url == "http://model-node:8000/v1"
    assert calls[-3:] == [("down", "model"), ("down", "judge"), ("down", "user")]


@pytest.mark.usefixtures("runs", "swarm")
def test_keep_swarm_leaves_the_swarm_up(config: Config, calls: Calls) -> None:
    config.model.url = None

    run(config, [Entry("bfcl")], RunOptions(swarm=True, keep_swarm=True, check=False))

    assert not [c for c in calls if c[0] == "down"]


@pytest.mark.usefixtures("runs", "swarm")
def test_a_swarm_comes_down_when_the_preflight_fails(config: Config, calls: Calls, preflight: list[Check]) -> None:
    config.model.url = None
    preflight.append(Check(FAIL, "endpoint unreachable"))

    assert run(config, [Entry("bfcl")], RunOptions(swarm=True)) == 1

    assert calls[-3:] == [("down", "model"), ("down", "judge"), ("down", "user")]


@pytest.mark.usefixtures("runs", "swarm")
def test_the_judge_is_served_only_for_entries_that_need_one(
    config: Config, calls: Calls, drivers: dict[str, dict]
) -> None:
    run(config, [Entry("bfcl")], RunOptions(swarm=True, check=False))
    assert ("ensure", "judge", False) not in calls

    drivers["inspect"] = {"roles": ["grader"]}
    run(config, [Entry("inspect")], RunOptions(swarm=True, check=False))
    assert ("ensure", "judge", False) in calls


@pytest.mark.usefixtures("runs", "swarm")
@pytest.mark.parametrize(
    ("roles", "served"),
    [(["user"], ["user"]), (["grader"], ["judge"]), (["user", "grader"], ["judge", "user"])],
)
def test_a_user_model_is_served_for_the_user_and_the_judge_for_the_grader(
    config: Config, calls: Calls, drivers: dict[str, dict], roles: list[str], served: list[str]
) -> None:
    config.user.swarm_config = "user.yaml"
    drivers["inspect"] = {"roles": roles}

    run(config, [Entry("inspect")], RunOptions(swarm=True, check=False))

    assert [c[1] for c in calls if c[0] == "ensure"] == ["model", *served]


@pytest.mark.usefixtures("runs", "swarm")
def test_without_a_user_model_the_judge_is_served_for_the_user(
    config: Config, calls: Calls, drivers: dict[str, dict]
) -> None:
    drivers["inspect"] = {"roles": ["user"]}

    run(config, [Entry("inspect")], RunOptions(swarm=True, check=False))

    assert [c[1] for c in calls if c[0] == "ensure"] == ["model", "judge"]


@pytest.mark.parametrize(
    ("role", "message"),
    [
        ("user", "need a user model, but no [user] or [judge] is set"),
        ("grader", "need a grader model, but no [judge] is set"),
    ],
)
def test_a_role_no_section_plays_fails(config: Config, role: str, message: str) -> None:
    config.judge.url = None

    checks = run_module._check_roles(config, {role: ["inspect"]}, deploying=False)

    assert [(c.level, c.message) for c in checks] == [(FAIL, f"inspect: {message}")]


def test_a_role_model_about_to_be_deployed_only_warns(config: Config) -> None:
    config.judge.url = None
    config.user.swarm_config = "user.yaml"

    checks = run_module._check_roles(config, {"user": ["inspect"], "grader": ["inspect"]}, deploying=True)

    assert [c.level for c in checks] == [WARN, FAIL]  # the judge has nothing to deploy


@pytest.mark.usefixtures("runs", "preflight")
def test_a_user_model_without_an_endpoint_fails_the_preflight(
    config: Config, calls: Calls, drivers: dict[str, dict]
) -> None:
    config.user.path = "user-model"  # set, so it plays the user, but not served
    drivers["inspect"] = {"roles": ["user"]}

    assert run(config, [Entry("inspect")], RunOptions()) == 1

    assert _ran(calls) == []


def _sbatch_calls(sbatch: Path) -> list[list[str]]:
    return [line.split() for line in sbatch.read_text().splitlines()]


@pytest.mark.usefixtures("swarm")
def test_sbatch_chains_the_jobs_and_tears_the_swarm_down_after_them(
    config: Config, calls: Calls, runs: Path, sbatch: Path
) -> None:
    config.model.url = None
    entries = [Entry("bfcl", {"test_category": "simple"}), Entry("bfcl", {"test_category": "live"}), Entry("lm_eval")]

    assert run(config, entries, RunOptions(swarm=True, check=False, sbatch=True, after="17")) == 0

    logs = runs / "logs"
    assert _sbatch_calls(sbatch) == [
        ["--parsable", "--dependency=afterany:17", str(logs / "bfcl" / "run-1.sbatch")],
        ["--parsable", "--dependency=afterany:4242", str(logs / "bfcl" / "run-2.sbatch")],
        ["--parsable", "--dependency=afterany:4243", str(logs / "lm_eval" / "run.sbatch")],
        ["--parsable", "--dependency=afterany:4242:4243:4244", str(logs / "swarm" / "down-model.sbatch")],
    ]
    assert "down model-swarm -y" in (logs / "swarm" / "down-model.sbatch").read_text()
    assert ("down", "model") not in calls
    assert [c[0] for c in _ran(calls)][::2] == ["setup", "setup", "setup"]  # set up here, before submitting


@pytest.mark.usefixtures("runs", "swarm")
def test_sbatch_with_keep_swarm_submits_no_teardown(config: Config, sbatch: Path) -> None:
    config.model.url = None

    run(config, [Entry("bfcl")], RunOptions(swarm=True, keep_swarm=True, check=False, sbatch=True))

    assert len(_sbatch_calls(sbatch)) == 1


@pytest.mark.usefixtures("runs")
def test_sbatch_without_deps_lets_the_jobs_run_together(config: Config, sbatch: Path) -> None:
    run(config, [Entry("bfcl"), Entry("lm_eval")], RunOptions(check=False, sbatch=True, chain=False, after="17"))

    assert [call[1] for call in _sbatch_calls(sbatch)] == ["--dependency=afterany:17"] * 2


@pytest.mark.usefixtures("runs")
def test_a_dry_run_runs_and_submits_nothing(config: Config, calls: Calls, sbatch: Path) -> None:
    assert run(config, [Entry("bfcl")], RunOptions(check=False, dry=True)) == 0
    assert run(config, [Entry("bfcl")], RunOptions(check=False, dry=True, sbatch=True)) == 0

    assert [c[0] for c in _ran(calls)] == ["command", "command"]
    assert not sbatch.exists()


@pytest.mark.usefixtures("runs", "swarm", "preflight")
def test_a_dry_run_checks_and_finds_the_swarm_it_would_deploy(config: Config, calls: Calls) -> None:
    config.model.url = None

    assert run(config, [Entry("bfcl")], RunOptions(swarm=True, dry=True)) == 0

    assert calls[:2] == [("ensure", "model", True), ("preflight", ("bfcl",), True)]
    assert ("down", "model") not in calls


@pytest.mark.parametrize(("flag", "error"), [("--no-deps", "go with --sbatch"), ("--keep-swarm", "goes with --swarm")])
def test_cli_options_need_the_option_they_modify(tmp_path: Path, flag: str, error: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[model]\nid = "m"\n')

    result = CliRunner().invoke(cli, ["run", str(path), "--suite", "bfcl", flag])

    assert result.exit_code == 2
    assert error in result.output


@pytest.mark.parametrize(("args", "error"), [([], "select suites"), (["--suite", "nope"], "unknown suites")])
def test_cli_needs_known_suites(tmp_path: Path, args: list[str], error: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[model]\nid = "m"\n')

    result = CliRunner().invoke(cli, ["run", str(path), *args])

    assert result.exit_code == 2
    assert error in result.output
