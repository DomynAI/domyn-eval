"""Tests for the lm_eval run driver.

``_load_tasks_with_selective_override`` must not clobber a task's own sampling params (a task yaml may set its own
generation_kwargs) unless override_all says to always win. Uses fake TaskManager/Task stand-ins rather than
real lm_eval task registries, so this stays fast and independent of which tasks happen to be registered.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from lm_eval.tasks import TaskManager

from domyn_eval.config import Config
from domyn_eval.drivers.report.lmeval import LMEvalReportDriver
from domyn_eval.drivers.run import lmeval
from domyn_eval.drivers.run.lmeval import LMEvalRunDriver

LOGGER = lmeval.__name__
CONTINUE = {"continue_final_message": True, "add_generation_prompt": False}


class _FakeTask:
    def __init__(self, name: str, gen_prefix: str | None = None):
        self.name = name
        self.config = SimpleNamespace(gen_prefix=gen_prefix)
        self.set_config_calls = []

    def set_config(self, key: str, value: dict, update: bool = False) -> None:  # noqa: FBT002 - lm_eval's API
        self.set_config_calls.append({"key": key, "value": value, "update": update})


class _FakeEntry:
    def __init__(self, cfg: dict | None, name: str = "default_name"):
        self.cfg = cfg
        self.name = name


class _FakeTaskManager:
    def __init__(self, tasks: dict[str, _FakeTask], task_index: dict[str, _FakeEntry]):
        self._tasks = tasks
        self.task_index = task_index

    def load(self, task_or_group: str) -> dict:  # noqa: ARG002 - every name loads the same tasks
        return {"tasks": dict(self._tasks)}


def _task_manager(task: _FakeTask, cfg: dict | None) -> TaskManager:
    """A task manager holding only ``task``, indexed with ``cfg`` as its YAML config."""
    return cast("TaskManager", _FakeTaskManager(tasks={task.name: task}, task_index={task.name: _FakeEntry(cfg)}))


@pytest.mark.parametrize("cfg", [{}, None], ids=["empty_cfg", "no_cfg"])
def test_applies_gen_kwargs_when_task_has_none_of_its_own(cfg: dict | None) -> None:
    task = _FakeTask("t")
    result = LMEvalRunDriver._load_tasks_with_selective_override(
        _task_manager(task, cfg), "group", {"temperature": 0.7}, override_all=False
    )
    assert task.set_config_calls == [{"key": "generation_kwargs", "value": {"temperature": 0.7}, "update": True}]
    assert result == [task]


def test_warns_when_task_keeps_its_own_gen_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    task = _FakeTask("default_name")
    with caplog.at_level("WARNING", logger=LOGGER):
        LMEvalRunDriver._load_tasks_with_selective_override(
            _task_manager(task, {"generation_kwargs": {"temperature": 0.0}}),
            "group",
            {"temperature": 0.7},
            override_all=False,
        )
    assert len(caplog.records) == 1
    assert "default_name" in caplog.records[0].getMessage()
    assert "override_generation_kwargs is off" in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    ("cfg", "override_all"),
    [({"generation_kwargs": {"temperature": 0.0}}, True), ({}, False)],
    ids=["override_all", "no_own_gen_kwargs"],
)
def test_no_warning_when_gen_kwargs_are_applied(
    caplog: pytest.LogCaptureFixture, cfg: dict, override_all: bool
) -> None:
    with caplog.at_level("WARNING", logger=LOGGER):
        LMEvalRunDriver._load_tasks_with_selective_override(
            _task_manager(_FakeTask("default_name"), cfg), "group", {"temperature": 0.7}, override_all=override_all
        )
    assert caplog.records == []


def test_skips_task_with_own_gen_kwargs_when_not_overriding() -> None:
    """The motivating case: a task yaml that sets generation_kwargs itself."""
    task = _FakeTask("default_name")
    result = LMEvalRunDriver._load_tasks_with_selective_override(
        _task_manager(task, {"generation_kwargs": {"temperature": 0.0}}),
        "group",
        {"temperature": 0.7},
        override_all=False,
    )
    assert task.set_config_calls == []
    assert result == [task]  # still runs, just with its own settings untouched


def test_chat_template_kwargs_still_applied_when_rest_of_gen_kwargs_is_skipped() -> None:
    """Chat template keyword arguments should be always applied."""
    task = _FakeTask("default_name")
    gen_kwargs = {"temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    LMEvalRunDriver._load_tasks_with_selective_override(
        _task_manager(task, {"generation_kwargs": {"temperature": 0.0}}), "group", gen_kwargs, override_all=False
    )
    assert task.set_config_calls == [
        {"key": "generation_kwargs", "value": {"chat_template_kwargs": {"enable_thinking": False}}, "update": True}
    ]


def test_override_all_wins_even_when_task_has_own_gen_kwargs() -> None:
    task = _FakeTask("default_name")
    LMEvalRunDriver._load_tasks_with_selective_override(
        _task_manager(task, {"generation_kwargs": {"temperature": 0.0}}),
        "group",
        {"temperature": 0.7},
        override_all=True,
    )
    assert task.set_config_calls == [{"key": "generation_kwargs", "value": {"temperature": 0.7}, "update": True}]


@pytest.mark.parametrize(
    ("gen_prefix", "continue_gen_prefix", "expected"),
    [("Answer:", True, [CONTINUE]), (None, True, []), ("Answer:", False, [])],
    ids=["prefix_chat_api", "no_prefix", "not_chat_api"],
)
def test_gen_prefix_is_continued_only_on_chat_apis(
    gen_prefix: str | None, continue_gen_prefix: bool, expected: list[dict]
) -> None:
    """A task's answer prefix must be continued, not answered in a new reply (which wastes its few answer tokens)."""
    task = _FakeTask("default_name", gen_prefix=gen_prefix)
    LMEvalRunDriver._load_tasks_with_selective_override(
        _task_manager(task, {"generation_kwargs": {"max_gen_toks": 30}}),
        "group",
        {"temperature": 0.7},
        override_all=False,
        continue_gen_prefix=continue_gen_prefix,
    )
    assert [c["value"] for c in task.set_config_calls] == expected


@pytest.mark.parametrize("chat_completions", [True, False])
@pytest.mark.parametrize(("max_tokens", "expected"), [(0, None), (4096, 4096)])
def test_max_tokens_reaches_the_request_and_zero_sends_no_cap(
    config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chat_completions: bool,
    max_tokens: int,
    expected: int | None,
) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    config.suite.lm_eval.chat_completions = chat_completions
    config.model.generation.context.max_tokens = max_tokens
    args_file = Path(LMEvalRunDriver(config).build_command()[-1])
    gen_kwargs = lmeval._LMEvalArgs.model_validate_json(args_file.read_text()).gen_kwargs.model_dump(exclude_none=True)

    # lm_eval's own payload builder, on a stand-in for the model it would construct; null lets the server budget
    model = SimpleNamespace(model="m", _max_gen_toks=256)
    prompt = [{"role": "user", "content": "hi"}] if chat_completions else "hi"
    api = LocalChatCompletion if chat_completions else LocalCompletionsAPI
    create_payload = api._create_payload  # pyright: ignore[reportAttributeAccessIssue] - register_model types it as LM
    payload = create_payload(model, prompt, generate=True, gen_kwargs=gen_kwargs)
    assert payload["max_tokens"] == expected


def test_lm_eval_runs_in_domyn_eval_python(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    driver = LMEvalRunDriver(config)
    assert driver.python == Path(sys.executable)
    assert driver.build_command()[0] == sys.executable


def test_include_path_only_for_extra_tasks(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SUITES_BASE_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    assert LMEvalRunDriver(config)._lmeval_args().include_path is None
    config.suite.lm_eval.path = "tasks"
    assert LMEvalRunDriver(config)._lmeval_args().include_path == str(tmp_path / "tasks")


def test_args_file_reads_back_as_the_args_written(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    config.suite.lm_eval.max_seq_lengths = [4096]
    config.suite.lm_eval.allow_unsafe_code = True
    driver = LMEvalRunDriver(config)
    args_file = Path(driver.build_command()[-1])
    assert lmeval._LMEvalArgs.model_validate_json(args_file.read_text()) == driver._lmeval_args()


@pytest.mark.parametrize(("allow_unsafe_code", "expected"), [(None, False), (True, True)])
def test_unsafe_code_reaches_lm_eval_and_is_off_by_default(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_unsafe_code: bool | None, expected: bool
) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    if allow_unsafe_code is not None:
        config.suite.lm_eval.allow_unsafe_code = allow_unsafe_code
    calls = []
    monkeypatch.setattr(lmeval, "TaskManager", lambda **_: None)
    monkeypatch.setattr(LMEvalRunDriver, "_load_tasks_with_selective_override", staticmethod(lambda *_, **__: []))
    monkeypatch.setattr(lmeval, "simple_evaluate", lambda **kwargs: calls.append(kwargs))  # None: nothing to write

    LMEvalRunDriver.run_lmeval(LMEvalRunDriver(config)._lmeval_args())

    [kwargs] = calls
    assert kwargs["confirm_run_unsafe_code"] is expected


def test_args_require_a_known_context_window(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    config.model.generation.context.max_model_len = 0
    config.model.endpoint_max_model_len = None
    with pytest.raises(ValueError, match="context window is unknown"):
        LMEvalRunDriver(config)._lmeval_args()


def test_samples_go_to_their_own_files_and_the_report_reads_the_results(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    config.model.url = "http://localhost:8000/v1"
    config.suite.lm_eval.task = "task"
    config.suite.lm_eval.log_samples = True
    sample = {
        "doc_id": 0,
        "arguments": [["prompt", {}]],
        "resps": [["a"]],
        "filtered_resps": ["a"],
        "target": "a",
        "doc_hash": "d",
        "prompt_hash": "p",
        "target_hash": "t",
    }
    results = {
        "results": {"task": {"acc,none": 1.0}},
        "n-samples": {"task": {"original": 1, "effective": 1}},
        "samples": {"task": [sample]},
    }
    monkeypatch.setattr(lmeval, "TaskManager", lambda **_: None)
    monkeypatch.setattr(LMEvalRunDriver, "_load_tasks_with_selective_override", staticmethod(lambda *_, **__: []))
    monkeypatch.setattr(lmeval, "simple_evaluate", lambda **_: results)
    args = LMEvalRunDriver(config)._lmeval_args()

    LMEvalRunDriver.run_lmeval(args)

    # searched across the whole runs root, so a file written anywhere else would show up
    task_dir = tmp_path / str(config.model.id) / str(config.general.run_id) / "results" / "lm_eval" / "task"
    [results_file] = tmp_path.rglob("results_*.json")
    [samples_file] = tmp_path.rglob("samples_*.jsonl")
    assert results_file.parent == samples_file.parent == task_dir
    assert "samples" not in json.loads(results_file.read_text())
    assert samples_file.name.startswith("samples_task_")
    assert [json.loads(line)["doc_id"] for line in samples_file.read_text().splitlines()] == [0]
    report = LMEvalReportDriver(config.model.id or "").report()
    assert list(zip(report["benchmark"], report["metric"], report["score"], strict=True)) == [("task", "acc", 1.0)]
