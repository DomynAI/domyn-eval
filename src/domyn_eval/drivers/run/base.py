# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
import shlex
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path

from domyn_eval.config import Config, has_own_venv
from domyn_eval.log import display_path
from domyn_eval.paths import logs_dir, results_dir, run_dir, suite_python

logger = logging.getLogger(__name__)

_TAIL_LINES = 10  # of a failed command's output, logged with its error


class BaseRunDriver(ABC):
    """Sets up and runs one evaluation suite against a model endpoint.

    Each suite subclasses it and implements ``suite``, ``setup`` and ``build_command``.
    """

    @property
    @abstractmethod
    def suite(self) -> str:
        """Name of the suite; selects its section of the config."""

    @property
    def max_model_len(self) -> int:
        """Context window for this run; 0 means "unknown"."""
        return self.config.model.generation.context.max_model_len or self.config.model.endpoint_max_model_len or 0

    @property
    def config_file(self) -> Path:
        """The run's resolved config, shared by its suites: ``<runs>/<model id>/<run_id>/config.toml``."""
        return (run_dir(self.model_id, self.run_id) / "config.toml").absolute()

    @property
    def roles(self) -> frozenset[str]:
        """The model roles (``config.ROLES``) a run of this suite needs served."""
        return frozenset()

    @property
    def log_dir(self) -> Path:
        """The suite's logs: ``<runs>/<model id>/<run_id>/logs/<suite>/``, one ``<name>.log`` per command."""
        return logs_dir(self.model_id, self.run_id, self.suite, create=True)

    @property
    def command_cwd(self) -> Path | None:
        """Working directory ``build_command`` runs in; None keeps the current one."""
        return None

    @property
    def command_env(self) -> dict[str, str]:
        """Environment variables ``build_command`` runs with, on top of the current environment."""
        return {}

    def __init__(self, config: Config):
        """Resolve the suite's environment and output folder.

        Args:
            config: The run's config; ``config.suite.<suite>`` holds this suite's settings.

        Raises:
            ValueError: If ``[model] id`` is not set.
        """
        self.config = config

        self.path = getattr(self.config.suite, self.suite).path
        # A suite without a .venv of its own runs in domyn-eval's Python.
        self.python = suite_python(self.path) if has_own_venv(self.suite) else Path(sys.executable)

        if not self.config.model.id:
            msg = "[model] id is not set; it names the folder the run's results are written to"
            raise ValueError(msg)
        self.model_id = self.config.model.id
        if not self.config.general.run_id:
            self.config.general.run_id = str(int(time.time()))
        self.run_id = self.config.general.run_id
        self.output_dir = str(results_dir(self.model_id, self.run_id, self.suite, create=True).absolute())

    @abstractmethod
    def build_command(self) -> list[str]:
        """The program and arguments that run the evaluation; run directly, without a shell."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare everything the suite needs before it can run, e.g. install its environment."""

    def write_config(self, path: Path | None = None) -> Path:
        """Save the run's resolved config, with this suite's section and the context window.

        The suite is marked used on the shared config, so the file keeps the sections of every suite the run has
        written so far. ``max_model_len`` is added to ``model.generation.context.max_model_len``.

        Args:
            path: Where to save it; ``config_file`` by default.

        Returns:
            The path of the written file.
        """
        path = path or self.config_file
        self.config.suite.use(self.suite)
        config = copy.deepcopy(self.config)
        config.model.generation.context.max_model_len = self.max_model_len
        config.save(path)
        return path

    def run(self, log: str = "run") -> int:
        """Run the suite's command, its output going to ``<log_dir>/<log>.log``.

        Args:
            log: Names the log file; ``run-2`` for a suite's second entry in the run.

        Returns:
            The command's exit code.
        """
        self.write_config()
        return self._run_command(self.build_command(), log=log, cwd=self.command_cwd, env=self.command_env)

    def _run_command(
        self, args: list[str], *, log: str, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> int:
        """Run a command, appending its output to ``<log_dir>/<log>.log``; on failure, log its last lines.

        The output never reaches the console: the file starts each command with a ``# <time> $ <command>`` line.

        Args:
            args: The program and its arguments; run directly, without a shell.
            log: Names the log file, and the step in the console's messages (e.g. ``setup``, ``run``).
            cwd: Directory to run in; None keeps the current one.
            env: Variables set on top of the current environment.

        Returns:
            The command's exit code.
        """
        path = self.log_dir / f"{log}.log"
        logger.info("%s: %s; log in %s", self.suite, log, display_path(path))
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"# {datetime.now().astimezone().isoformat(timespec='seconds')} $ {shlex.join(args)}\n")
            file.flush()
            # stderr goes into stdout, so the log keeps both in the order they were printed. errors="replace" because
            # tools like curl/tqdm can emit split multibyte chars.
            with subprocess.Popen(
                args,
                cwd=cwd,
                env={**os.environ, **env} if env else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    tail.append(line)
                    file.write(line)
                    file.flush()

        if proc.returncode:
            lines = "".join(f"  │ {line}" for line in tail).rstrip("\n")
            logger.error(
                "%s: %s failed (exit %s); see %s\n%s", self.suite, log, proc.returncode, display_path(path), lines
            )
        return proc.returncode

    def sbatch_command(self) -> list[str]:
        """The sbatch script's command: ``build_command`` run in ``command_cwd`` with ``command_env``.

        A driver may override this to return a *re-invocation* of the ``domyn-eval`` CLI (so the real command is rebuilt
        at run time inside each array task), e.g. for SLURM-array sharding where the shard index must resolve per task.
        """
        self.write_config()
        command = ["env", *(f"{key}={value}" for key, value in self.command_env.items()), *self.build_command()]
        if self.command_cwd is None:
            return command
        return ["sh", "-c", f"cd {shlex.quote(str(self.command_cwd))} && exec {shlex.join(command)}"]
