"""Tests for the inspect run driver."""

import copy
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from domyn_eval.config import Config
from domyn_eval.drivers.run.inspect import InspectRunDriver


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir so the drivers' run folders land there, not in runs/."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


def _task_config(config: Config) -> dict:
    """The task TOML the driver writes for ``config``, parsed."""
    return tomllib.loads(InspectRunDriver(config)._write_inline_task_config().read_text())


def test_needs_judge_when_a_task_sets_judge(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/math"}, {"task": "inspect_evals/hle", "judge": True}]
    assert InspectRunDriver(config).needs_judge


def test_needs_no_judge_when_no_task_sets_judge(config: Config) -> None:
    # the benchmark name alone does not decide it
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/hle"}]
    assert not InspectRunDriver(config).needs_judge


def test_custom_tasks_reject_unknown_keys(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/math", "mesage_limit": 10}]
    with pytest.raises(ValidationError, match="mesage_limit"):
        _ = InspectRunDriver(config).tasks


def test_custom_tasks_need_a_task_name(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": ""}]
    with pytest.raises(ValidationError):
        _ = InspectRunDriver(config).tasks


def test_task_config_forwards_fail_on_error_http_timeout_and_max_tokens(config: Config) -> None:
    inspect = config.suite.inspect
    inspect.fail_on_error = 0.1
    inspect.request_timeout = 2400
    inspect.max_tokens = 16384
    inspect.custom_tasks = [{"task": "inspect_evals/mgsm"}]
    data = _task_config(config)
    assert data["general"]["fail_on_error"] == 0.1
    assert data["inspect"] == {"request_timeout": 2400, "max_tokens": 16384}


def test_task_config_leaves_unset_retry_settings_to_domyn_inspect(config: Config) -> None:
    inspect = config.suite.inspect
    inspect.custom_tasks = [{"task": "inspect_evals/math"}]
    assert not {"max_tasks", "retry_attempts", "retry_wait"} & set(_task_config(config)["general"])

    inspect.max_tasks, inspect.retry_attempts, inspect.retry_wait = 20, 0, 30
    general = _task_config(config)["general"]
    assert general | {"max_tasks": 20, "retry_attempts": 0, "retry_wait": 30} == general


def test_task_config_passes_fail_on_error_through(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/math"}]
    assert "fail_on_error" not in _task_config(config)["general"]  # unset: domyn-inspect's own default applies
    # set to false it is kept: false means "never abort", not "unset"
    config.suite.inspect.fail_on_error = False
    assert _task_config(config)["general"]["fail_on_error"] is False


def test_task_config_forwards_pass_k(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/aime2025", "pass_k": 16}]
    assert _task_config(config)["tasks"][0]["pass_k"] == 16


def test_task_config_keeps_task_args_types(config: Config) -> None:
    task_args = {"fewshot": 5, "challenges": ["a", "b"], "shuffle": True}
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/mmlu_5_shot", "task_args": task_args}]
    assert _task_config(config)["tasks"][0]["task_args"] == task_args


def test_task_config_drops_the_judge_flag(config: Config) -> None:
    # judge is ours; domyn-inspect does not know the key
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/hle", "judge": True}]
    assert _task_config(config)["tasks"] == [{"task": "inspect_evals/hle"}]


def test_build_command_requires_custom_tasks(config: Config) -> None:
    with pytest.raises(ValueError, match="custom_tasks"):
        InspectRunDriver(config).build_command()


@pytest.mark.parametrize(
    ("suite_max_samples", "max_samples", "expected"),
    [(None, 0, None), (None, 50, "50"), (1000, 50, "1000"), (0, 50, None), (-1, 50, None)],
)
def test_build_command_limit_prefers_the_suite_max_samples(
    config: Config, suite_max_samples: int | None, max_samples: int, expected: str | None
) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/math"}]
    config.suite.inspect.max_samples = suite_max_samples
    config.generation.max_samples = max_samples
    args = InspectRunDriver(config).build_command()
    assert (args[args.index("--limit") + 1] if "--limit" in args else None) == expected


def test_build_command_passes_the_entry_config_and_the_task_config_it_writes(config: Config) -> None:
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/math"}]
    config.suite.inspect.task = "math"
    driver = InspectRunDriver(config)
    args = driver.build_command()
    model = Path(args[args.index("--model") + 1])
    assert model == Path(driver.output_dir, "config_math.toml")
    assert tomllib.loads(model.read_text())["inspect"]["task"] == "math"
    assert Path(args[args.index("--task") + 1]).is_file()


def test_an_entrys_config_survives_the_later_entries_of_the_run(config: Config) -> None:
    # A submitted job reads its config when it starts, after every entry has been written.
    entries = [("math", 20), ("long_context", 1)]
    commands = []
    for task, max_tasks in entries:
        entry = copy.deepcopy(config)
        entry.suite.inspect.custom_tasks = [{"task": f"inspect_evals/{task}"}]
        entry.suite.inspect.task = task
        entry.suite.inspect.max_tasks = max_tasks
        commands.append(InspectRunDriver(entry).build_command())
    for (task, max_tasks), args in zip(entries, commands, strict=True):
        saved = tomllib.loads(Path(args[args.index("--model") + 1]).read_text())["inspect"]
        assert (saved["task"], saved["max_tasks"]) == (task, max_tasks)


def _module_file(cwd: Path, module: str) -> Path:
    """The file ``python -m module`` runs from ``cwd``."""
    path = cwd.joinpath(*module.split("."))
    return path / "__main__.py" if path.is_dir() else path.with_suffix(".py")


def test_commands_run_modules_of_the_suite_checkout(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_commands(monkeypatch)
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/gsm8k"}]
    driver = InspectRunDriver(config)
    if not (driver.command_cwd / "pyproject.toml").is_file():
        pytest.skip("suites/domyn-inspect is not checked out")
    driver.setup()
    run = driver.build_command()
    assert run[1:3] == ["-m", "domyn_inspect"]
    for args in (calls[1], run):  # the prefetch, then the run
        assert args[1] == "-m"
        assert _module_file(driver.command_cwd, args[2]).is_file(), args[2]


def _stub_commands(monkeypatch: pytest.MonkeyPatch, *returncodes: int) -> list[list[str]]:
    """Record the commands the driver runs; the n-th returns ``returncodes[n]``, or 0 past their end."""
    calls = []

    def run_command(self: InspectRunDriver, args: list[str], **kwargs: object) -> int:  # noqa: ARG001
        calls.append(args)
        return returncodes[len(calls) - 1] if len(calls) <= len(returncodes) else 0

    monkeypatch.setattr(InspectRunDriver, "_run_command", run_command)
    return calls


def test_setup_installs_then_prefetches_the_task_config_build_command_passes(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_commands(monkeypatch)
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/gsm8k"}]
    driver = InspectRunDriver(config)
    driver.setup()
    install, prefetch = calls  # no prefetch of every dataset besides the run's tasks
    assert install == ["uv", "sync"]
    args = driver.build_command()
    assert prefetch[prefetch.index("--tasks") + 1] == args[args.index("--task") + 1]


def test_setup_aborts_when_the_install_fails(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_commands(monkeypatch, 1)
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/gsm8k"}]
    with pytest.raises(RuntimeError, match="uv sync failed"):
        InspectRunDriver(config).setup()
    assert calls == [["uv", "sync"]]  # no prefetch after a failed install


def test_setup_aborts_when_task_prefetch_fails(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_commands(monkeypatch, 0, 1)
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/gsm8k"}]
    with pytest.raises(RuntimeError, match="auto_setup"):
        InspectRunDriver(config).setup()


def test_build_command_runs_no_setup(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_commands(monkeypatch)
    config.general.auto_setup = True  # acting on it is the caller's job
    config.suite.inspect.custom_tasks = [{"task": "inspect_evals/gsm8k"}]
    InspectRunDriver(config).build_command()
    assert calls == []
