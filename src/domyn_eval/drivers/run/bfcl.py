# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from domyn_eval.config import Config
from domyn_eval.drivers.run.base import BaseRunDriver
from domyn_eval.paths import suites_base_dir


class _GenerationConfig(BaseModel):
    """What domyn-bfcl hands every handler through ``--generation-config``: sampling, reasoning, budget and timeout.

    Unset sampling params are sent as null, which domyn-bfcl does not send on. domyn-bfcl
    validates the keys strictly (``domyn_bfcl.config.GenerationConfig``), so a field added
    here must be added there too.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning: bool
    max_model_len: int
    max_tokens: int | None
    token_safety_margin: int
    request_timeout: int
    max_samples: int


class BFCLRunDriver(BaseRunDriver):
    """Runs BFCL's ``generate`` then ``evaluate`` with domyn-bfcl, one folder per test category."""

    @property
    def suite(self) -> str:
        """Name of the suite; selects its section of the config."""
        return "bfcl"

    @property
    def test_category(self) -> str:
        """The test categories to run: ``[bfcl] test_category``, else BFCL's default ``all``."""
        return self.config.suite.bfcl.test_category or "all"

    @property
    def model_name(self) -> str:
        """The BFCL registry name that selects the handler: ``model_class``, else ``model.path``, ``-FC`` if native."""
        bfcl = self.config.suite.bfcl
        name = bfcl.model_class or self.config.model.path
        if bfcl.fc_model and not name.endswith("-FC"):
            name += "-FC"
        return name

    @property
    def command_cwd(self) -> Path:
        """The suite's folder."""
        return suites_base_dir() / self.path

    @property
    def command_env(self) -> dict[str, str]:
        """The endpoint bfcl's handlers query in place of a local server; bfcl reads it only from the environment.

        The key is not among them: domyn-bfcl reads ``$API_KEY`` itself, so it never lands in an sbatch script.
        """
        env = {
            # bfcl creates its scratch folders and loads a .env from here; the results dirs are passed explicitly
            "BFCL_PROJECT_ROOT": self.output_dir,
            "REMOTE_OPENAI_BASE_URL": self.config.model.url or "",
            # model_class only selects the handler; requests name the model the endpoint serves
            "REMOTE_OPENAI_MODEL": self.config.model.path,
        }
        # bfcl otherwise loads the tokenizer from the registry name, which need not exist
        if tokenizer := self.config.generation.tokenizer_path or self.config.model.path:
            env["REMOTE_OPENAI_TOKENIZER_PATH"] = tokenizer
        return env

    def __init__(self, config: Config):
        """Give the test categories their own folder in the run: ``<runs>/<model>/<run_id>/results/bfcl/<category>``.

        Args:
            config: The run's config; ``config.suite.bfcl`` holds the suite's settings.
        """
        super().__init__(config)
        category_dir = Path(self.output_dir, self.test_category)
        category_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(category_dir)

    def setup(self) -> None:
        """Install the suite's uv environment.

        Raises:
            RuntimeError: If the install fails.
        """
        if self._run_command(["uv", "sync"], log="setup", cwd=self.command_cwd) != 0:
            msg = f"uv sync failed in {self.command_cwd}"
            raise RuntimeError(msg)

    def build_command(self) -> list[str]:
        """domyn-bfcl's ``generate``, then its ``evaluate`` on what was generated, chained by ``sh``."""
        binary = str(self.python.with_name("domyn-bfcl"))
        common = ["--model", self.model_name, "--test-category", self.test_category, "--result-dir", self.output_dir]
        generation_config = ["--generation-config", self._generation_config().model_dump_json()]

        generate = [binary, "generate", *common, "--skip-server-setup", *generation_config]
        generate += ["--num-threads", str(self.config.suite.bfcl.num_threads or self.config.generation.num_threads)]
        if self.config.model.path and Path(self.config.model.path).exists():  # Path("") is the current folder
            # absolute, as the command runs in the suite's folder
            generate += ["--local-model-path", str(Path(self.config.model.path).resolve())]

        evaluate = [binary, "evaluate", *common, "--score-dir", self.output_dir, *generation_config]
        if self.config.generation.max_samples > 0:
            # only the first max_samples of each category were generated
            evaluate.append("--partial-eval")

        return ["sh", "-c", f"{shlex.join(generate)} && {shlex.join(evaluate)}"]

    def _generation_config(self) -> _GenerationConfig:
        """The settings domyn-bfcl hands every handler, from ``generation`` and its selected mode."""
        gen = self.config.generation
        return _GenerationConfig.model_validate(
            {
                **dataclasses.asdict(gen.mode),
                "reasoning": gen.reasoning,
                "max_model_len": self.max_model_len,
                "max_tokens": gen.max_tokens or None,
                "token_safety_margin": gen.token_safety_margin,
                "request_timeout": gen.request_timeout,
                "max_samples": gen.max_samples,
            }
        )
