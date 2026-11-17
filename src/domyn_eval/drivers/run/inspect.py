# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from pathlib import Path
from typing import ClassVar

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from domyn_eval.drivers.run.base import BaseRunDriver
from domyn_eval.log import display_path
from domyn_eval.paths import suite_dir

logger = logging.getLogger(__name__)


class InspectTask(BaseModel):
    """One ``custom_tasks`` entry: domyn-inspect's ``TaskEntry`` (``configs/config.py``) plus ``judge``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str = Field(min_length=1)  # registry name, e.g. ``inspect_evals/mmlu_0_shot``
    task_args: dict[str, object] = {}
    display_name: str = ""
    generate: dict[str, object] = {}
    message_limit: int | None = None
    pass_k: int | None = None
    dual_score: bool = False
    judge: bool = Field(default=False, exclude=True)  # ours, not domyn-inspect's: the task needs a judge swarm


class _TasksGeneral(BaseModel):
    """The tasks TOML's ``[general]``: domyn-inspect's ``TasksGeneralConfig``; None leaves a field to its default."""

    name: str
    log_shared: bool  # required, so the file always states it
    max_tasks: int | None = None
    retry_attempts: int | None = None
    retry_wait: int | None = None
    fail_on_error: bool | float | None = None


class _TasksInspect(BaseModel):
    """The tasks TOML's ``[inspect]``: domyn-inspect's ``TasksInspectConfig``; None leaves a field to its default."""

    request_timeout: int | None = None
    max_tokens: int | None = None


class _TasksConfig(BaseModel):
    """The tasks TOML domyn-inspect reads as ``--task``: domyn-inspect's ``TasksConfig``."""

    general: _TasksGeneral
    inspect: _TasksInspect
    tasks: list[InspectTask]


class InspectRunDriver(BaseRunDriver):
    """Runs the inline ``custom_tasks`` of ``[inspect]`` with domyn-inspect, one log dir per run id."""

    # Folders of the external task packages (``task_packages``) this run already installed; see ``setup``.
    _installed: ClassVar[set[str]] = set()

    @property
    def suite(self) -> str:
        """Name of the suite; selects its section of the config."""
        return "inspect"

    @property
    def tasks(self) -> list[InspectTask]:
        """The ``custom_tasks`` of ``[inspect]``, validated; raises ``ValidationError`` on a malformed entry."""
        return [InspectTask.model_validate(task) for task in self.config.suite.inspect.custom_tasks or []]

    @property
    def _name(self) -> str:
        """The entry's task group, which names its config files: ``task`` of ``[inspect]``."""
        return self.config.suite.inspect.task or "inline"

    @property
    def needs_judge(self) -> bool:
        """Whether any task sets ``judge = true``."""
        return any(task.judge for task in self.tasks)

    @property
    def command_cwd(self) -> Path:
        """The suite's folder, whose ``domyn_inspect`` package the commands run."""
        return suite_dir(self.path)

    def setup(self) -> None:
        """Install the suite's environment and the entry's ``task_packages``, then prefetch the entry's tasks.

        All the inspect entries of a run use the same environment so the setup of a later entry must not remove or
        reinstall the ``task_packages`` (the external packages of custom tasks) that an earlier job uses.

        The prefetch also checks that the tasks load offline.

        Raises:
            ValueError: If ``[inspect]`` has no ``custom_tasks``, or a task package is not a folder.
            RuntimeError: If an install or the prefetch fails.
        """
        folders = [Path(p).expanduser().resolve() for p in self.config.suite.inspect.task_packages]
        if missing := [str(f) for f in folders if not f.is_dir()]:
            msg = f"task_packages: no such folder {missing}"
            raise ValueError(msg)
        # uv sync uninstalls the external packages, which its lock file does not list: none must be installed yet
        if not self._installed and self._run_command(["uv", "sync"], log="setup", cwd=self.command_cwd) != 0:
            msg = f"uv sync failed in {self.command_cwd}"
            raise RuntimeError(msg)
        packages = [p for p in dict.fromkeys(map(str, folders)) if p not in self._installed]
        if packages:
            # editable and without dependencies
            editable = [arg for p in packages for arg in ("--editable", p)]
            install = ["uv", "pip", "install", "--python", str(self.python), "--no-deps", *editable]
            if self._run_command(install, log="setup", cwd=self.command_cwd) != 0:
                msg = f"installing task_packages {packages} failed"
                raise RuntimeError(msg)
            self._installed.update(packages)
        self._prefetch_tasks(self._write_inline_task_config())

    def build_command(self) -> list[str]:
        """``python -m domyn_inspect`` on the entry's own config and task config, both written first.

        The entry's config is its own copy, not the run's ``config.toml``: that holds only the last entry of each
        suite, and a submitted job reads its config when it starts, after the later entries have been written.

        Raises:
            ValueError: If ``[inspect]`` has no ``custom_tasks``.
        """
        task_config = self._write_inline_task_config()
        args = [
            str(self.python),
            "-m",
            "domyn_inspect",
            "--model",
            str(self.write_config(Path(self.output_dir, f"config_{self._name}.toml").absolute())),
            "--task",
            str(task_config),
            "--log-dir",
            self.output_dir,
            "--run",
        ]
        # The suite's max_samples (e.g. set by a recipe entry) overrides generation.max_samples.
        max_samples = self.config.suite.inspect.max_samples
        if max_samples is None:
            max_samples = self.config.generation.max_samples
        if max_samples > 0:
            args += ["--limit", str(max_samples)]
        return args

    def _prefetch_tasks(self, task_config_path: Path) -> None:
        """Cache what the tasks in ``task_config_path`` load, and check they build offline.

        Runs on the submitting (online) node; the job runs offline, so a task that fails here would only fail later on
        the compute node.

        Raises:
            RuntimeError: If a task cannot be prefetched or loaded offline, to abort before submitting.
        """
        args = [str(self.python), "-m", "domyn_inspect.prefetch.prefetch_datasets", "--tasks", str(task_config_path)]
        if self._run_command(args, log="prefetch", cwd=self.command_cwd) != 0:
            msg = (
                f"task prefetch failed for {task_config_path}: the job could not load these tasks offline; fix the "
                "cache, or set `general.auto_setup = false` in the config to submit anyway"
            )
            raise RuntimeError(msg)

    def _write_inline_task_config(self) -> Path:
        """Write the tasks and run settings of ``[inspect]`` to the TOML domyn-inspect reads as ``--task``.

        Settings left unset are omitted, so domyn-inspect's own defaults apply. The file goes next to the run's logs,
        where an ``--sbatch`` job finds it on the compute node and where it records which tasks the run executed.

        Returns:
            The absolute path of the written file.

        Raises:
            ValueError: If ``[inspect]`` has no ``custom_tasks``.
        """
        if not self.tasks:
            msg = "No inspect tasks to run: list them in [inspect] custom_tasks, or run a recipe that defines them."
            raise ValueError(msg)
        inspect = self.config.suite.inspect
        # each table keeps its own fields
        values = {**dataclasses.asdict(inspect), "name": self._name, "log_shared": True}
        task_config = _TasksConfig(
            general=_TasksGeneral.model_validate(values),
            inspect=_TasksInspect.model_validate(values),
            tasks=self.tasks,
        )
        path = Path(self.output_dir, f"tasks_{self._name}.toml").absolute()
        with path.open("wb") as f:
            tomli_w.dump(task_config.model_dump(exclude_defaults=True), f)
        logger.debug("inspect: task config in %s", display_path(path))
        return path
