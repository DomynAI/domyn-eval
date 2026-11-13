"""Tests for the inspect stat driver, on real ``.eval`` logs."""

from importlib.metadata import version
from pathlib import Path

import pytest
from inspect_ai.event import ModelEvent, ScoreEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, EvalStatus, write_eval_log
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.scorer import Score

from domyn_eval.drivers.stat import inspect
from domyn_eval.drivers.stat.inspect import InspectStatDriver
from tests.drivers.stat.helpers import normal_sample

INSTALLED = version("inspect_ai")


@pytest.fixture
def driver(saved_run: tuple[str, str]) -> InspectStatDriver:
    return InspectStatDriver(*saved_run)


@pytest.fixture(params=[INSTALLED, "0.3.0"], ids=["zip", "inspect"])
def written_by(request: pytest.FixtureRequest) -> str:
    """The inspect version that wrote the logs: the installed one's are read from the zip, others through inspect."""
    return request.param


def _output(
    answer: str = "42", reasoning: str | None = None, stop: StopReason = "stop", tokens: int = 20
) -> ModelOutput:
    content = [ContentText(text=answer)]
    if reasoning:
        content.insert(0, ContentReasoning(reasoning=reasoning))
    return ModelOutput(
        model="m",
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content), stop_reason=stop)],
        usage=ModelUsage(input_tokens=10, output_tokens=tokens, total_tokens=10 + tokens),
    )


def _model_event(model: str, output: ModelOutput) -> ModelEvent:
    return ModelEvent(model=model, input=[], tools=[], tool_choice="auto", config=GenerateConfig(), output=output)


def _sample(
    answer: str = "42",
    reasoning: str | None = None,
    stop: StopReason = "stop",
    tokens: int = 20,
    *,
    scored: bool = True,
) -> EvalSample:
    output = _output(answer, reasoning, stop, tokens)
    scores = {"match": Score(value="C")} if scored else None
    return EvalSample(id=answer, epoch=1, input="q", target="42", output=output, scores=scores)


def _write_log(
    results: Path,
    task: str,
    samples: list[EvalSample],
    status: EvalStatus = "success",
    display_name: str | None = None,
    written_by: str = INSTALLED,
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    log = EvalLog(
        status=status,
        eval=EvalSpec(
            created="2026-01-01T00:00:00+00:00",
            task=f"inspect_evals/{task}",
            task_display_name=display_name,
            dataset=EvalDataset(samples=len(samples), sample_ids=[s.id for s in samples]),
            model="m",
            config=EvalConfig(),
            packages={"inspect_ai": written_by},
        ),
        samples=samples,
    )
    write_eval_log(log, str(results / f"{task}.eval"))


def test_samples_are_read_from_the_runs_successful_logs(driver: InspectStatDriver, written_by: str) -> None:
    _write_log(
        driver.results_dir,
        "gsm8k",
        [_sample(reasoning="think hard"), _sample("", stop="max_tokens", tokens=1024, scored=False)],
        display_name="GSM8K",
        written_by=written_by,
    )
    _write_log(driver.results_dir, "aime", [_sample()], status="error", written_by=written_by)

    assert list(driver.samples()) == [
        normal_sample(benchmark="GSM8K", answer="42", reasoning="think hard", completion_tokens=20),
        normal_sample(benchmark="GSM8K", answer="", truncated=True, completion_tokens=1024, scored=False),
    ]


def test_a_sample_is_the_evaluated_models_final_generation_before_scoring(
    driver: InspectStatDriver, written_by: str
) -> None:
    """In an agentic task a user simulator speaks last, and a grader may be the evaluated model itself."""
    agent = _output("Your flight is booked.", reasoning="Book it.", tokens=30)
    user = _output("###STOP###", tokens=3)
    sample = _sample()
    sample.output = user
    sample.events = [
        _model_event("m", _output("Which flight?", reasoning="Ask.")),
        _model_event("user", _output("The 9am one.")),
        _model_event("m", agent),
        _model_event("user", user),
        ScoreEvent(score=Score(value="C")),
        _model_event("m", _output("GRADE: C")),
    ]
    _write_log(driver.results_dir, "tau2_airline", [sample], written_by=written_by)

    assert list(driver.samples()) == [
        normal_sample(
            benchmark="inspect_evals/tau2_airline",
            answer="Your flight is booked.",
            reasoning="Book it.",
            completion_tokens=30,
        )
    ]


def test_a_sample_is_truncated_when_any_of_its_generations_hit_the_budget(
    driver: InspectStatDriver, written_by: str
) -> None:
    sample = _sample()
    sample.events = [
        _model_event("m", _output("", reasoning="Let me think", stop="max_tokens", tokens=1024)),
        _model_event("user", _output("Hello?")),
        _model_event("m", _output("42", tokens=20)),
    ]
    _write_log(driver.results_dir, "tau2_airline", [sample], written_by=written_by)

    (row,) = driver.stats()

    assert (row.truncated, row.empty, row.max_completion_tokens) == (1, 0, 1024)


def test_a_generations_reasoning_is_read_from_the_samples_attachments(
    driver: InspectStatDriver, written_by: str
) -> None:
    """A log moves its events' long texts to the sample's attachments, leaving ``attachment://<hash>`` behind."""
    reasoning = "Wait, let me reconsider. " * 30
    sample = _sample()
    sample.events = [_model_event("m", _output("42", reasoning="attachment://loop"))]
    sample.attachments = {"loop": reasoning}
    _write_log(driver.results_dir, "aime2025", [sample], written_by=written_by)

    (row,) = driver.samples()

    assert (row.reasoning, row.looping) == (reasoning, True)


def test_an_errored_sample_is_empty_and_unscored(driver: InspectStatDriver, written_by: str) -> None:
    errored = EvalSample(id=1, epoch=1, input="q", target="42")
    _write_log(driver.results_dir, "ifeval", [errored], written_by=written_by)

    (row,) = driver.stats()

    assert (row.samples, row.empty, row.unscored, row.truncated) == (1, 1, 1, 0)


def test_logs_of_the_installed_inspect_are_read_from_the_zip(
    driver: InspectStatDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_log(driver.results_dir, "ifeval", [_sample()])
    monkeypatch.setattr(inspect, "read_eval_log_samples", lambda *_, **__: pytest.fail("read through inspect"))

    assert len(list(driver.samples())) == 1


@pytest.mark.parametrize(
    ("answer", "reasoning", "looping"),
    [
        ("I cannot answer that. " * 30, None, True),
        ("", "Wait, let me reconsider. " * 30, True),
        ("42", "First, 6 times 7 is 42.", False),
    ],
)
def test_a_sample_loops_when_its_answer_or_reasoning_ends_in_a_loop(
    driver: InspectStatDriver, answer: str, reasoning: str | None, looping: bool
) -> None:
    _write_log(driver.results_dir, "ifeval", [_sample(answer, reasoning)])

    (sample,) = driver.samples()

    assert sample.looping is looping
