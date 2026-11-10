"""Tests for the BFCL run driver."""

import json
import shlex
from pathlib import Path

import pytest

from domyn_eval.config import Config
from domyn_eval.drivers.run.bfcl import BFCLRunDriver


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point RUNS_DIR at a tmp dir so the drivers' run folders land there, not in runs/."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def config(config: Config) -> Config:
    config.model.path = "org/served-model"
    return config


def _commands(config: Config) -> tuple[list[str], list[str]]:
    """The ``generate`` and ``evaluate`` commands the driver chains."""
    program, flag, script = BFCLRunDriver(config).build_command()
    assert (program, flag) == ("sh", "-c")
    generate, evaluate = script.split(" && ")
    return shlex.split(generate), shlex.split(evaluate)


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def test_output_dir_is_the_test_categorys_folder_in_the_run(config: Config, runs_root: Path) -> None:
    config.general.run_id = "r1"
    driver = BFCLRunDriver(config)
    assert driver.output_dir == str(runs_root / "test-model" / "r1" / "results" / "bfcl" / "single_turn")
    assert Path(driver.output_dir).is_dir()


def test_test_category_defaults_to_all(config: Config) -> None:
    config.suite.bfcl.test_category = None
    generate, _ = _commands(config)
    assert _option(generate, "--test-category") == "all"


def test_generate_then_evaluate_the_same_results(config: Config) -> None:
    generate, evaluate = _commands(config)
    assert generate[0].endswith(".venv/bin/domyn-bfcl")
    assert generate[1:2] == ["generate"]
    assert evaluate[1:2] == ["evaluate"]
    assert "--skip-server-setup" in generate
    out = BFCLRunDriver(config).output_dir
    assert _option(generate, "--result-dir") == _option(evaluate, "--result-dir") == out
    assert _option(evaluate, "--score-dir") == out


@pytest.mark.parametrize(
    ("model_class", "fc_model", "expected"),
    [
        (None, False, "org/served-model"),
        (None, True, "org/served-model-FC"),
        ("Qwen/Qwen3-8B-FC", True, "Qwen/Qwen3-8B-FC"),
        ("Qwen/Qwen3-8B", False, "Qwen/Qwen3-8B"),
    ],
)
def test_model_name(config: Config, model_class: str | None, fc_model: bool, expected: str) -> None:
    config.suite.bfcl.model_class = model_class
    config.suite.bfcl.fc_model = fc_model
    generate, evaluate = _commands(config)
    assert _option(generate, "--model") == _option(evaluate, "--model") == expected


def test_local_model_path_only_for_local_weights(config: Config, tmp_path: Path) -> None:
    generate, _ = _commands(config)
    assert "--local-model-path" not in generate
    config.model.path = ""
    generate, _ = _commands(config)
    assert "--local-model-path" not in generate
    config.model.path = str(tmp_path)
    generate, _ = _commands(config)
    assert _option(generate, "--local-model-path") == str(tmp_path)


def test_a_relative_local_model_path_is_passed_absolute(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "weights").mkdir()
    monkeypatch.chdir(tmp_path)
    config.model.path = "weights"
    generate, _ = _commands(config)
    assert _option(generate, "--local-model-path") == str(tmp_path / "weights")  # the command runs in the suite folder


def test_partial_eval_only_when_samples_are_capped(config: Config) -> None:
    _, evaluate = _commands(config)
    assert "--partial-eval" not in evaluate
    config.generation.max_samples = 5
    _, evaluate = _commands(config)
    assert "--partial-eval" in evaluate


def test_num_threads_prefers_the_suites_own(config: Config) -> None:
    config.generation.num_threads = 16
    generate, _ = _commands(config)
    assert _option(generate, "--num-threads") == "16"
    config.suite.bfcl.num_threads = 4
    generate, _ = _commands(config)
    assert _option(generate, "--num-threads") == "4"


def test_generation_config_carries_the_modes_params_and_the_budget(config: Config) -> None:
    generate, evaluate = _commands(config)
    sent = json.loads(_option(generate, "--generation-config"))
    assert sent == json.loads(_option(evaluate, "--generation-config"))
    assert sent["temperature"] == 0.6  # think mode
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["min_p"] is None  # unset params are sent, as null
    assert sent["reasoning"] is True
    assert sent["max_model_len"] == 32768
    assert sent["max_tokens"] == 1024
    assert sent["token_safety_margin"] == config.generation.token_safety_margin
    assert sent["request_timeout"] == config.generation.request_timeout


def test_generation_config_follows_nothink(config: Config) -> None:
    config.generation.reasoning = False
    sent = json.loads(_option(_commands(config)[0], "--generation-config"))
    assert sent["temperature"] == 0.7
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_an_unset_max_tokens_is_sent_as_null(config: Config) -> None:
    config.generation.max_tokens = 0
    sent = json.loads(_option(_commands(config)[0], "--generation-config"))
    assert sent["max_tokens"] is None  # domyn-bfcl then budgets the whole window


def test_command_env_points_bfcl_at_the_endpoint(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "secret")
    env = BFCLRunDriver(config).command_env
    assert env["REMOTE_OPENAI_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert env["REMOTE_OPENAI_MODEL"] == "org/served-model"
    assert "secret" not in env.values()  # domyn-bfcl reads $API_KEY itself


def test_bfcl_project_root_is_the_runs_folder(config: Config) -> None:
    driver = BFCLRunDriver(config)
    assert driver.command_env["BFCL_PROJECT_ROOT"] == driver.output_dir


def test_command_env_passes_the_tokenizer(config: Config) -> None:
    config.generation.tokenizer_path = "/models/tok"
    assert BFCLRunDriver(config).command_env["REMOTE_OPENAI_TOKENIZER_PATH"] == "/models/tok"


def test_the_tokenizer_defaults_to_the_model_path(config: Config) -> None:
    assert BFCLRunDriver(config).command_env["REMOTE_OPENAI_TOKENIZER_PATH"] == "org/served-model"


def test_no_tokenizer_without_a_model_path(config: Config) -> None:
    config.model.path = ""
    assert "REMOTE_OPENAI_TOKENIZER_PATH" not in BFCLRunDriver(config).command_env


def _stub_commands(monkeypatch: pytest.MonkeyPatch, *returncodes: int) -> list[list[str]]:
    """Record the commands the driver runs; the n-th returns ``returncodes[n]``, or 0 past their end."""
    calls = []

    def run_command(self: BFCLRunDriver, args: list[str], **kwargs: object) -> int:  # noqa: ARG001
        calls.append(args)
        return returncodes[len(calls) - 1] if len(calls) <= len(returncodes) else 0

    monkeypatch.setattr(BFCLRunDriver, "_run_command", run_command)
    return calls


def test_setup_aborts_when_the_install_fails(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_commands(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="uv sync failed"):
        BFCLRunDriver(config).setup()
    assert calls == [["uv", "sync"]]


def test_build_command_runs_no_setup(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_commands(monkeypatch)
    config.general.auto_setup = True  # acting on it is the caller's job
    BFCLRunDriver(config).build_command()
    assert calls == []
