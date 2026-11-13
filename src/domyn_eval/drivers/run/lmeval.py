# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""lm_eval run driver, plus the code that runs lm_eval through its Python API in a separate process.

The driver writes the run's arguments to a JSON file and runs this module with domyn-eval's own Python:

    python -m domyn_eval.drivers.run.lmeval <args.json>
"""

import argparse
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Literal, cast

from lm_eval.api.group import Group
from lm_eval.api.task import Task
from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from domyn_eval.config import Config
from domyn_eval.drivers.run.base import BaseRunDriver
from domyn_eval.paths import suite_dir

logger = logging.getLogger(__name__)


class _GenKwargs(BaseModel):
    """lm_eval's ``gen_kwargs``: the sampling params of ``model.generation.mode``; None leaves a param to the task."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    do_sample: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    max_gen_toks: int | None  # None: no cap sent, the server budgets each request

    @model_serializer(mode="wrap")
    def _keep_max_gen_toks(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {**handler(self), "max_gen_toks": self.max_gen_toks}  # sent even when None


class _ModelArgs(BaseModel):
    """``model_args`` of lm_eval's ``local-completions`` / ``local-chat-completions`` models."""

    model: str
    base_url: str
    num_concurrent: int
    max_retries: int
    tokenized_requests: bool = False
    max_length: int
    tokenizer: str
    timeout: int


class _TaskMetadata(BaseModel):
    """``metadata`` of lm_eval's ``TaskManager``, read by long-context tasks such as RULER."""

    max_seq_lengths: list[int]
    tokenizer: str


class _LMEvalArgs(BaseModel):
    """What the driver hands to the lm_eval process, through ``lmeval_run_args.json``."""

    task: str
    include_path: str | None = None
    model_type: Literal["local-completions", "local-chat-completions"]
    model_args: _ModelArgs
    batch_size: int | str | None = None
    output_path: str
    gen_kwargs: _GenKwargs
    override_generation_kwargs: bool
    apply_chat_template: bool
    log_samples: bool
    allow_unsafe_code: bool
    limit: int | None = None
    metadata: _TaskMetadata | None = None


class LMEvalRunDriver(BaseRunDriver):
    """Runs an lm_eval task or group against the model endpoint, in a separate process of domyn-eval's Python."""

    @property
    def suite(self) -> str:
        """Name of the suite; selects its section of the config."""
        return "lm_eval"

    @property
    def tokenizer_path(self) -> str:
        """Tokenizer lm_eval counts tokens with: the model's ``context.tokenizer_path``, else the served model's."""
        return (self.config.model.generation.context.tokenizer_path or self.config.model.path).rstrip("/")

    def __init__(self, config: Config):
        """Give the task its own folder in the run: ``<runs>/<model>/<run_id>/results/lm_eval/<task>``.

        Args:
            config: The run's config; ``config.suite.lm_eval`` holds the suite's settings.

        Raises:
            ValueError: If ``[lm_eval] task`` is not set.
        """
        super().__init__(config)
        task = self.config.suite.lm_eval.task
        if not task:
            msg = "[lm_eval] task is not set; it names the lm_eval task or group to run"
            raise ValueError(msg)
        task_dir = Path(self.output_dir, task)
        task_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(task_dir)

    def setup(self) -> None:
        """Nothing to prepare: lm_eval runs from domyn-eval's own environment."""

    def build_command(self) -> list[str]:
        """This module on the run's lm_eval arguments, which it writes to a JSON file first."""
        args_path = Path(self.output_dir, "lmeval_run_args.json").absolute()
        args_path.write_text(self._lmeval_args().model_dump_json(exclude_none=True, indent=2))
        return [str(self.python), "-m", __name__, str(args_path)]

    def _lmeval_args(self) -> _LMEvalArgs:
        """The arguments of ``run_lmeval``, resolved from the config.

        Raises:
            ValueError: If ``[model] url`` is not set, or the context window is unknown.
        """
        model_url = self.config.model.url
        if not model_url:
            msg = "[model] url is not set; it is the endpoint lm_eval sends its requests to"
            raise ValueError(msg)
        if not self.max_model_len:
            msg = (
                "the context window is unknown: set model.generation.context.max_model_len. lm_eval needs it as "
                "max_length, and with 0 its loglikelihood requests lose their first token"
            )
            raise ValueError(msg)
        lm_eval, gen, general = self.config.suite.lm_eval, self.config.model.generation, self.config.general
        chat_completions = lm_eval.chat_completions

        gen_kwargs = _GenKwargs.model_validate(
            {
                **dataclasses.asdict(gen.mode),
                "chat_template_kwargs": gen.mode.chat_template_kwargs if chat_completions else None,
                "max_gen_toks": gen.context.max_tokens or None,
            }
        )
        metadata = None
        if lm_eval.max_seq_lengths:
            metadata = _TaskMetadata(max_seq_lengths=lm_eval.max_seq_lengths, tokenizer=self.tokenizer_path)
        model_type, endpoint = (
            ("local-chat-completions", "chat/completions") if chat_completions else ("local-completions", "completions")
        )
        return _LMEvalArgs(
            task=lm_eval.task,
            include_path=str(suite_dir(self.path)) if self.path else None,
            model_type=model_type,
            model_args=_ModelArgs(
                model=self.config.model.path,
                base_url=f"{model_url}/{endpoint}",
                num_concurrent=lm_eval.num_threads or general.num_threads,
                max_retries=general.max_retries,
                max_length=self.max_model_len,
                tokenizer=self.tokenizer_path,
                timeout=general.request_timeout,
            ),
            batch_size=lm_eval.batch_size,
            output_path=self.output_dir,
            gen_kwargs=gen_kwargs,
            override_generation_kwargs=lm_eval.override_generation_kwargs,
            apply_chat_template=chat_completions,
            log_samples=lm_eval.log_samples,
            allow_unsafe_code=lm_eval.allow_unsafe_code,
            limit=general.max_samples if general.max_samples > 0 else None,
            metadata=metadata,
        )

    @staticmethod
    def _load_tasks_with_selective_override(
        task_manager: TaskManager,
        task_or_group: str,
        gen_kwargs: dict[str, Any],
        override_all: bool,
        *,
        continue_gen_prefix: bool = False,
    ) -> list[str | dict[str, Any] | Task | Group]:
        """Load a task or group, applying ``gen_kwargs`` only to tasks that set no sampling params of their own.

        With ``override_all``, ``gen_kwargs`` is applied to every task. ``chat_template_kwargs`` (e.g.
        ``enable_thinking``) is always applied, even to a task that otherwise keeps its own params.

        With ``continue_gen_prefix`` (a chat-completions API), a task with a ``gen_prefix`` (e.g. RULER's "Answer: ...
        they are:") asks the server to continue it. lm_eval sends the prefix as a closed assistant message, so the
        server would otherwise open a new reply, and a model that repeats the prefix spends the task's few answer tokens
        on it.

        Raises:
            ValueError: If a loaded task is missing from the task manager's index.
        """
        loaded = task_manager.load(task_or_group)
        chat_template_kwargs = gen_kwargs.get("chat_template_kwargs")
        for name, task in loaded["tasks"].items():
            entry = task_manager.task_index.get(name)
            if entry is None:
                msg = f"Task {name} not available in TaskManager index."
                raise ValueError(msg)
            has_own_gen_kwargs = bool(entry.cfg and "generation_kwargs" in entry.cfg)
            if override_all or not has_own_gen_kwargs:
                task.set_config(key="generation_kwargs", value=gen_kwargs, update=True)
            elif chat_template_kwargs is not None:
                task.set_config(
                    key="generation_kwargs", value={"chat_template_kwargs": chat_template_kwargs}, update=True
                )

            if continue_gen_prefix and task.config.gen_prefix:
                task.set_config(
                    key="generation_kwargs",
                    value={"continue_final_message": True, "add_generation_prompt": False},
                    update=True,
                )

            if entry.cfg and has_own_gen_kwargs and not override_all:
                logger.warning(
                    "lm_eval: %s keeps its own generation_kwargs %s, as override_generation_kwargs is off",
                    entry.name,
                    entry.cfg["generation_kwargs"],
                )

        groups = loaded.get("groups")
        if groups:
            return list(groups.values())
        return list(loaded["tasks"].values())

    @staticmethod
    def run_lmeval(args: _LMEvalArgs) -> None:
        """Run lm_eval and save its output in ``output_path`` as lm_eval's CLI does.

        The scores go to ``results_<date>.json``; with ``log_samples``, each task's samples go to their own
        ``samples_<task>_<date>.jsonl``.
        """
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("API_KEY", "EMPTY"))
        # A folder path would look more natural, but the tracker would add a per-model subfolder, redundant with
        # <model id> in the run layout; a .json path keeps its files in output_path (as results_<date>.json).
        tracker = EvaluationTracker(output_path=str(Path(args.output_path, "results.json")))

        metadata = args.metadata.model_dump() if args.metadata else None
        task_manager = TaskManager(include_path=args.include_path, metadata=metadata)
        tasks = LMEvalRunDriver._load_tasks_with_selective_override(
            task_manager,
            args.task,
            args.gen_kwargs.model_dump(exclude_none=True),
            args.override_generation_kwargs,
            continue_gen_prefix=args.apply_chat_template and args.model_type == "local-chat-completions",
        )

        results = simple_evaluate(
            model=args.model_type,
            model_args=args.model_args.model_dump(),
            # simple_evaluate hands these to TaskManager.load, which takes Groups as they are; its annotation omits them
            tasks=cast("list[str | dict[str, Any] | Task]", tasks),
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
            apply_chat_template=args.apply_chat_template,
            gen_kwargs=None,  # already applied selectively above
            confirm_run_unsafe_code=args.allow_unsafe_code,
            task_manager=task_manager,
            evaluation_tracker=tracker,
        )
        if results is None:
            return

        samples = results.pop("samples", {})
        tracker.save_results_aggregated(results=dict(results), samples=samples)
        for task, task_samples in samples.items():
            # it iterates the task's sample records, as lm_eval's CLI passes them; its dict annotation is wrong
            tracker.save_results_samples(task_name=task, samples=task_samples)  # pyright: ignore[reportArgumentType]


def main() -> None:
    """Command line of the lm_eval process: run lm_eval with the arguments in a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("args_file", type=Path, help="JSON file with the arguments of run_lmeval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    LMEvalRunDriver.run_lmeval(_LMEvalArgs.model_validate_json(args.args_file.read_text()))


if __name__ == "__main__":
    main()
