# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Checks run before an evaluation, behind ``domyn-eval check``.

Each check returns ``Check`` results; only a ``FAIL`` blocks a run.
"""

import enum
import functools
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from rich.console import Console
from rich.markup import escape

from domyn_eval.config import Config, has_own_venv, registered_suites
from domyn_eval.paths import suite_dir, suite_python


class Level(enum.Enum):
    """The outcome of a check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class Check(NamedTuple):
    """The result of one check."""

    level: Level
    message: str


OK, WARN, FAIL = Level.OK, Level.WARN, Level.FAIL


def check_env() -> list[Check]:
    """Check the environment variables of the endpoint."""
    if os.getenv("API_KEY"):
        return [Check(OK, "API_KEY is set")]
    return [Check(WARN, "API_KEY is not set")]


def _has_transformers(python: Path) -> bool | None:
    """Return whether ``python`` has transformers, None if it does not exist.

    A venv's site-packages is looked into, rather than running its Python.
    """
    if not python.exists():
        return None
    if python == Path(sys.executable):
        return importlib.util.find_spec("transformers") is not None
    return any(python.parent.parent.glob("lib/python*/site-packages/transformers"))


def _check_tokenizer_dir(path: Path, estimate: str) -> Check:
    """Load the tokenizer in ``path`` as domyn-inspect does.

    It tries AutoTokenizer, then tokenizer.json alone, in this Python.
    """
    try:
        import transformers
    except ImportError:
        return Check(WARN, f"found {path}, but transformers is not installed here to load it")
    try:
        transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        try:
            transformers.PreTrainedTokenizerFast.from_pretrained(path)
        except Exception:  # noqa: BLE001
            # Report AutoTokenizer's error, the more telling one, on one line.
            error = " ".join(str(e).split())
            return Check(FAIL, f"cannot load the tokenizer in {path} ({type(e).__name__}: {error}); {estimate}")
    return Check(OK, f"tokenizer loads from {path}")


def endpoint_client() -> httpx.Client:
    """Return an HTTP client for the endpoints, authenticated with ``API_KEY`` if it is set."""
    api_key = os.getenv("API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(headers=headers, timeout=10)


def list_models(client: httpx.Client, url: str) -> list[dict[str, Any]]:
    """Return the model cards an OpenAI-compatible endpoint serves.

    Args:
        client: The HTTP client.
        url: The base URL, ending in ``/v1``.

    Raises:
        httpx.HTTPError: If the request fails.
        ValueError: If the response is not a model list.
    """
    response = client.get(f"{url}/models")
    response.raise_for_status()
    data = response.json().get("data", [])
    return [card for card in data if isinstance(card, dict)]


def served_max_model_len(card: dict[str, Any]) -> int | None:
    """Return the context window a model card advertises, None if it has none.

    Args:
        card: The model card.
    """
    max_model_len = card.get("max_model_len")
    return max_model_len if isinstance(max_model_len, int) and max_model_len > 0 else None


def check_context_budget(config: Config, served: int | None) -> Check:
    """Check the configured context window against the endpoint's.

    Args:
        config: The run config.
        served: The endpoint's context window, if it advertises one.
    """
    context = config.model.generation.context
    configured = context.max_model_len
    if served and configured > served:
        return Check(
            FAIL,
            f"model.generation.context.max_model_len={configured} exceeds the endpoint's "
            f"{served}; lower it, or set 0 to read it from the endpoint",
        )
    # Not on vLLM, which advertises max_model_len on every model card; other
    # OpenAI-compatible servers and hosted APIs may not, and then only the config knows it.
    if not (served or configured):
        return Check(
            FAIL,
            "context window unknown: the endpoint does not advertise it; set model.generation.context.max_model_len",
        )
    max_model_len = configured or served or 0
    source = "config" if configured else "endpoint"
    max_tokens = context.max_tokens or max_model_len
    if max_tokens > max_model_len:
        return Check(
            WARN,
            f"model.generation.context.max_tokens={max_tokens} exceeds max_model_len={max_model_len} ({source}) "
            "and will be clamped",
        )
    return Check(
        OK,
        f"token budget: max_tokens={max_tokens} of max_model_len={max_model_len} ({source})",
    )


def _check_suite_dir(suite: str, path: Path | None, python: Path | None) -> list[Check]:
    if path is None:
        return []
    if not path.is_dir():
        return [Check(FAIL, f"{suite}: {path} does not exist")]
    if not any(path.iterdir()):
        return [
            Check(
                FAIL,
                f"{suite}: {path} is empty; run `git submodule update --init {path}`",
            )
        ]
    checks = [Check(OK, f"{suite}: found {path}")]
    if python is None or not python.exists():
        checks.append(Check(WARN, f"{suite}: {path} has no .venv; set it up"))
    return checks


class _Suite(NamedTuple):
    """A selected suite's checkout and interpreter, None if it has none."""

    path: Path | None
    python: Path | None


class Preflight:
    """The checks run before an evaluation of one config on its selected suites.

    They run once the swarm handler, if any, has set ``model.url``: the
    suites send their requests there, not to ``swarm_config`` or ``swarm_name``.
    """

    def __init__(
        self, config: Config, suites: list[str], client: httpx.Client | None = None, *, deploying: bool = False
    ):
        """Resolve the selected suites' checkouts and interpreters.

        Args:
            config: The run config.
            suites: The selected suites.
            client: The HTTP client of the endpoint checks; by default one
                authenticated with ``API_KEY``.
            deploying: Whether a swarm will be deployed for the model, which then
                has no ``url`` yet: a dry run.
        """
        self.config = config
        self._client = client
        self._deploying = deploying
        # In selection order; unknown suites map to None.
        self._resolved = {suite: self._resolve(suite) for suite in suites}

    def _resolve(self, suite: str) -> _Suite | None:
        if suite not in registered_suites():
            return None
        if not has_own_venv(suite):
            # Its path, if any, holds only optional extras: there is no checkout to check.
            return _Suite(None, Path(sys.executable))
        path = getattr(getattr(self.config.suite, suite), "path", None)
        return _Suite(suite_dir(path), suite_python(path)) if path else _Suite(None, None)

    @functools.cached_property
    def client(self) -> httpx.Client:
        """The HTTP client of the endpoint checks."""
        return self._client if self._client is not None else endpoint_client()

    def run(self, *, ping: bool = True) -> list[Check]:
        """Run every check, closing the HTTP client afterwards.

        Args:
            ping: Whether to query the endpoints.
        """
        checks = self.check_config() + check_env() + self.check_tokenizer()
        if ping:
            with self.client:
                checks += self.check_endpoints()
        return checks

    def check_config(self) -> list[Check]:
        """Check the model, its endpoint and the selected suites' checkouts."""
        model = self.config.model
        checks = [Check(OK, f"model.id = {model.id}") if model.id else Check(FAIL, "model.id is not set")]
        if not model.url and self._deploying:
            checks.append(Check(WARN, "model.url is set once the swarm is deployed; the endpoint is not checked"))
        elif not model.url:
            checks.append(Check(FAIL, "model.url is not set; it is the endpoint the suites send their requests to"))
        for suite, resolved in self._resolved.items():
            if resolved is None:
                checks.append(Check(FAIL, f"unknown suite {suite!r}"))
            elif has_own_venv(suite):
                checks += _check_suite_dir(suite, *resolved)
            else:
                checks.append(Check(OK, f"{suite}: runs in domyn-eval's Python ({resolved.python})"))
        return checks

    def check_tokenizer(self) -> list[Check]:
        """Check that the token budget will count tokens rather than estimate.

        A local directory is loaded once, in this Python; a hub id is not
        downloaded. The selected suites' environments must have transformers.
        """
        estimate = "prompt lengths will be estimated"
        path = self.config.model.generation.context.tokenizer_path or self.config.model.path
        if not path:
            return [
                Check(
                    WARN,
                    f"model.generation.context.tokenizer_path and model.path are unset; {estimate}",
                )
            ]
        local = Path(path).expanduser()
        if local.is_dir():
            checks = [_check_tokenizer_dir(local, estimate)]
        elif local.exists() or local.is_absolute() or path.startswith((".", "~")):
            # A hub id is never absolute nor dot-relative: this is a bad path.
            checks = [Check(FAIL, f"{path} is not a directory; {estimate}")]
        else:
            checks = [Check(WARN, f"tokenizer {path!r} is taken as a hub id")]
        for suite, resolved in self._resolved.items():
            python = resolved.python if resolved else None
            if python is not None and _has_transformers(python) is False:
                checks.append(Check(WARN, f"{suite}: no transformers in {python}; {estimate}"))
        return checks

    def check_endpoints(self) -> list[Check]:
        """Check the model endpoint and, if they are different ones, the judge's and the user model's.

        The endpoint must serve ``model.path``, the model name the suites send.
        An unreachable judge or user model is only a warning: not every suite uses one.
        """
        url = self.config.model.url
        if not url:
            return []
        try:
            cards = list_models(self.client, url)
        except (httpx.HTTPError, ValueError) as e:
            return [Check(FAIL, f"endpoint unreachable: {url} ({e!r})")]
        served = ", ".join(str(c.get("id")) for c in cards) or "no models"
        checks = [Check(OK, f"endpoint reachable: {url} serves {served}")]
        model = self.config.model.path
        card = next((card for card in cards if card.get("id") == model), None)
        if card is None:
            checks.append(Check(FAIL, f"model.path={model!r} is not served there; the suites send it as model name"))
        else:
            checks.append(check_context_budget(self.config, served_max_model_len(card)))
            checks += self.check_reasoning_parser(model)
        pinged = {url}
        for section in ("judge", "user"):
            other = getattr(self.config, section).url
            if not other or other in pinged:
                continue
            pinged.add(other)
            try:
                list_models(self.client, other)
            except (httpx.HTTPError, ValueError) as e:
                checks.append(Check(WARN, f"{section} unreachable: {other} ({e!r})"))
            else:
                checks.append(Check(OK, f"{section} reachable: {other}"))
        return checks

    def check_reasoning_parser(self, model_id: str, max_tokens: int = 1024) -> list[Check]:
        """Check that the endpoint returns the thinking apart from the answer.

        Without a reasoning parser, vLLM returns the thinking inside the content,
        and scorers grade it. One request is sent, with the run's
        ``chat_template_kwargs``.

        Args:
            model_id: The served model to query.
            max_tokens: The completion budget of the probe.
        """
        generation = self.config.model.generation
        if not generation.reasoning:
            return []
        body: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if generation.mode.chat_template_kwargs:
            body["chat_template_kwargs"] = generation.mode.chat_template_kwargs
        try:
            response = self.client.post(
                f"{self.config.model.url}/chat/completions",
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            return [Check(WARN, f"reasoning parser: probe failed ({e!r})")]
        if message.get("reasoning_content") or message.get("reasoning"):
            return [Check(OK, "reasoning parser: thinking returned separately")]
        return [
            Check(
                WARN,
                f"reasoning parser: no separate thinking within {max_tokens} tokens; "
                "serve the model with --reasoning-parser",
            )
        ]


def failed(checks: list[Check]) -> bool:
    """Return whether any check failed.

    Args:
        checks: The check results.
    """
    return any(check.level is FAIL for check in checks)


def render_checks(checks: list[Check], console: Console | None = None) -> None:
    """Print the check results.

    Args:
        checks: The check results.
        console: The console to print to.
    """
    icons = {OK: "[green]✓[/]", WARN: "[yellow]⚠[/]", FAIL: "[red]✗[/]"}
    console = console or Console()
    for check in checks:
        console.print(f"{icons[check.level]} {escape(check.message)}", highlight=False)
