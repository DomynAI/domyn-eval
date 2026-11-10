# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Evaluation config: sections, TOML loading, suite registry and recipes."""

import dataclasses
import logging
import os
import tomllib
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

import tomli_w
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass

from domyn_eval.paths import UNRESOLVED_REF

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(validate_assignment=True, extra="forbid")

# A value whose type a function hands back unchanged, e.g. a parsed TOML document and its expanded copy
_T = TypeVar("_T")


def expand_env(value: _T, context: dict[str, Any]) -> _T:
    """Expand ``${ENV}``, then ``${key}`` references to ``context``.

    Args:
        value: The value to expand; non-strings are returned unchanged.
        context: Values of the keys a ``${key}`` reference may name.
    """
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    for key, sub in context.items():
        if isinstance(sub, str):
            expanded = expanded.replace(f"${{{key}}}", sub)
    return cast("_T", expanded)  # a string in, a string out


def expand_config(value: _T, context: dict[str, Any] | None = None, path: str = "") -> _T:
    """Expand ``${...}`` in every string of a parsed TOML document.

    A string sees the environment and the string keys of its own table and of
    every enclosing table. A reference left unresolved is kept and logged.

    Args:
        value: The document, or a part of it.
        context: Keys visible from the enclosing tables.
        path: Dotted location of ``value``, used in the warning.
    """
    if isinstance(value, dict):
        ctx = dict(context or {})
        for key, sub in value.items():
            if isinstance(sub, str):
                ctx[key] = expand_env(sub, ctx)
        expanded = {key: expand_config(sub, ctx, f"{path}.{key}" if path else key) for key, sub in value.items()}
        return cast("_T", expanded)  # a table in, a table out
    if isinstance(value, list):
        return cast("_T", [expand_config(sub, context, f"{path}[{i}]") for i, sub in enumerate(value)])
    if isinstance(value, str):
        expanded = expand_env(value, context or {})
        leftover = UNRESOLVED_REF.search(expanded)
        if leftover:
            logger.warning(
                "%s: unresolved reference %s in %r",
                path or "config",
                leftover.group(0),
                value,
            )
        return expanded
    return value


@dataclass(config=_STRICT)
class GeneralConfig:
    """``[general]``: run-wide settings.

    Attributes:
        run_id: The run, named at run time; it keys the run directory.
        auto_setup: Whether drivers set up their suite before running.
        log_level: The logging level.
        file: Absolute path of the loaded config, set by ``load_config``.
        recipes_file: The recipes TOML.
    """

    run_id: str | None = None
    auto_setup: bool = True
    log_level: str = "INFO"
    file: str | None = None
    recipes_file: str = "configs/recipes.toml"


@dataclass(config=_STRICT)
class ModelConfig:
    """``[model]`` and ``[judge]``: a model and its endpoint.

    Attributes:
        swarm_config: The domyn-swarm YAML serving the model.
        id: The model's id, which keys its results directory.
        swarm_name: The name of a running swarm serving the model.
        name: The display name of the model.
        path: The weights, or a HF hub id; the model name sent to the
            endpoint.
        url: The OpenAI-compatible base URL; ``/v1`` is appended if
            missing. Set by the swarm handler for a swarm-served model.
        endpoint_max_model_len: The context window read from the endpoint at
            run time.
    """

    swarm_config: str | None = None
    id: str | None = None
    swarm_name: str | None = None
    name: str | None = None
    path: str = ""
    url: str | None = None
    endpoint_max_model_len: int | None = None

    @field_validator("url")
    @classmethod
    def _ends_in_v1(cls, url: str | None) -> str | None:
        if not url:
            return None
        url = url.rstrip("/")
        return url if url.endswith("/v1") else f"{url}/v1"

    @property
    def endpoint_port(self) -> int | None:
        """Return the port of the model's endpoint, or None if unknown."""
        if self.url is None:
            return None
        parsed = urlparse(self.url)
        return parsed.port

    @property
    def endpoint_node(self) -> str | None:
        """Return the hostname of the model's endpoint, or None if unknown."""
        if self.url is None:
            return None
        parsed = urlparse(self.url)
        return parsed.hostname


@dataclass(config=_STRICT)
class GenerationModeConfig:
    """``[generation.think]`` and ``[generation.nothink]``: per-mode params.

    None leaves a sampling param to the endpoint's default.

    Attributes:
        temperature: The sampling temperature.
        top_p: The nucleus sampling mass.
        min_p: The minimum token probability, relative to the top token.
        top_k: The number of top tokens sampled from.
        presence_penalty: The penalty on tokens already present.
        repetition_penalty: The multiplicative penalty on repeated tokens.
        frequency_penalty: The penalty scaled by a token's count so far.
        do_sample: Whether to sample rather than decode greedily.
        chat_template_kwargs: Extra arguments of the chat template, e.g.
            ``enable_thinking``.
        system_prompt_template: The system prompt; ``{instruction}`` is the
            task's own.
        min_completion_tokens: The completion budget kept when a
            long-context prompt is truncated; None keeps none.
    """

    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    do_sample: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    system_prompt_template: str = "{instruction}"
    min_completion_tokens: int | None = None


@dataclass(config=_STRICT)
class GenerationConfig:
    """``[generation]``: how requests are built.

    Attributes:
        reasoning: Whether the model reasons before answering, which selects
            ``think`` over ``nothink`` as ``mode``.
        think: The params used with reasoning.
        nothink: The params used without reasoning.
        max_tokens: The completion budget; 0 uses what the context allows.
        max_model_len: The context window; 0 reads it from the endpoint.
        token_safety_margin: Number of tokens to keep free when estimating the available token budget.
        request_timeout: The per-request client timeout, in seconds.
        max_retries: The retries of a failed request.
        max_samples: The samples per task; -1 runs them all.
        num_threads: The concurrent requests of every suite that does
            not set its own ``num_threads``.
        tokenizer_path: The tokenizer used to count prompt tokens.
    """

    reasoning: bool = True
    think: GenerationModeConfig = dataclasses.field(default_factory=GenerationModeConfig)
    nothink: GenerationModeConfig = dataclasses.field(default_factory=GenerationModeConfig)
    max_tokens: int = 0
    max_model_len: int = 0
    token_safety_margin: int = 256
    request_timeout: int = 3600
    max_retries: int = 0
    max_samples: int = -1
    num_threads: int = 64
    tokenizer_path: str | None = None

    @property
    def mode(self) -> GenerationModeConfig:
        """Return the params of the selected mode."""
        return self.think if self.reasoning else self.nothink


def _sbatch(option: str) -> Any:  # noqa: ANN401 - a dataclass field
    """Return a ``[slurm]`` field emitted as ``#SBATCH --<option>``; None leaves it to the cluster."""
    return dataclasses.field(default=None, metadata={"sbatch": option})


@dataclass(config=_STRICT)
class SlurmConfig:
    """``[slurm]``: the resources of submitted jobs.

    None leaves a directive out of the job script, to the cluster's default.

    Attributes:
        template: The job script template; None uses the packaged one.
        account: The account charged.
        partition: The partition.
        qos: The quality of service.
        walltime: The time limit, as ``HH:MM:SS``.
        nodes: The nodes per job.
        ntasks_per_node: The tasks per node.
        cpus_per_task: The CPUs per task.
        memory: The memory per node, e.g. ``16G``.
        gres: The generic resources, e.g. ``gpu:1``.
        exclude: The nodes to avoid.
    """

    template: str | None = None
    account: str | None = _sbatch("account")
    partition: str | None = _sbatch("partition")
    qos: str | None = _sbatch("qos")
    walltime: str | None = _sbatch("time")
    nodes: int | None = _sbatch("nodes")
    ntasks_per_node: int | None = _sbatch("ntasks-per-node")
    cpus_per_task: int | None = _sbatch("cpus-per-task")
    memory: str | None = _sbatch("mem")
    gres: str | None = _sbatch("gres")
    exclude: str | None = _sbatch("exclude")

    def directives(self) -> dict[str, str]:
        """Return the ``#SBATCH`` options set here; unset ones are left to the cluster's defaults."""
        options = {
            f.metadata["sbatch"]: getattr(self, f.name) for f in dataclasses.fields(self) if "sbatch" in f.metadata
        }
        return {option: str(value) for option, value in options.items() if value is not None}


@dataclass(config=_STRICT)
class InspectConfig:
    """``[inspect]``: the domyn-inspect suite.

    Attributes:
        path: The suite checkout, relative to ``SUITES_BASE_DIR`` (default: the working directory).
        task: The name of the task group, set by recipe entries.
        custom_tasks: The tasks to run, as recipe entries define them.
        max_samples: The samples per task of the task group; overrides
            ``generation.max_samples``.
        num_threads: The concurrent requests, inspect's
            ``max_connections``; overrides ``generation.num_threads``.
        max_tasks: The tasks run in parallel.
        fail_on_error: Whether a failed sample fails its task; a float is
            the tolerated error rate.
        tiktoken_model: The tiktoken encoding inspect counts tokens with;
            None uses inspect's default.
        retry_attempts: The retries of a failed task.
        retry_wait: The seconds between task retries.
        request_timeout: The read timeout of inspect's client, in seconds;
            overrides ``generation.request_timeout``.
        max_tokens: The completion budget of the tasks; overrides
            ``generation.max_tokens``.
    """

    path: str = "suites/domyn-inspect"
    task: str | None = None
    custom_tasks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    max_samples: int | None = None
    num_threads: int | None = None
    max_tasks: int | None = None
    fail_on_error: bool | float | None = None
    tiktoken_model: str | None = None
    retry_attempts: int | None = None
    retry_wait: int | None = None
    request_timeout: int | None = None
    max_tokens: int | None = None


def _check_inspect_entries(entries: list[InspectConfig]) -> None:
    """Reject inspect entries sharing a task name, which keys their results."""
    tasks = [entry.task for entry in entries if entry.task is not None]
    clash = sorted({task for task in tasks if tasks.count(task) > 1})
    if clash:
        msg = f"inspect entries share task name(s) {clash}; each entry needs its own"
        raise ValueError(msg)


@dataclass(config=_STRICT)
class BFCLConfig:
    """``[bfcl]``: the domyn-bfcl suite.

    Attributes:
        path: The suite checkout, relative to ``SUITES_BASE_DIR`` (default: the working directory).
        test_category: The BFCL test categories to run, comma-separated;
            None uses BFCL's default.
        fc_model: Whether the model uses native function calling, rather
            than tools described in the prompt.
        model_class: The BFCL handler class; None picks the default one.
        num_threads: The concurrent requests; overrides
            ``generation.num_threads``.
    """

    path: str = "suites/domyn-bfcl"
    test_category: str | None = None
    fc_model: bool = False
    model_class: str | None = None
    num_threads: int | None = None


@dataclass(config=_STRICT)
class LMEvalConfig:
    """``[lm_eval]``: lm-eval's tasks, run in domyn-eval's Python (the ``lm-eval`` extra).

    Attributes:
        path: A folder of extra task configs, lm-eval's ``include_path``, relative to ``SUITES_BASE_DIR``;
            None runs only lm-eval's own tasks.
        task: The lm-eval task or group.
        batch_size: The requests per batch, or ``auto``; None uses
            lm-eval's default.
        num_threads: The concurrent requests, lm-eval's
            ``num_concurrent``; overrides ``generation.num_threads``.
        log_samples: Whether to write every sample to the results.
        max_seq_lengths: The context lengths of the RULER tasks.
        chat_completions: Whether to use the chat endpoint, which applies the
            chat template but returns no logprobs.
        override_generation_kwargs: Whether our generation params win over a
            task's own.
        allow_unsafe_code: Whether tasks may run model-generated code on the
            host (humaneval, mbpp, ...); lm-eval refuses them otherwise.
    """

    path: str | None = None
    task: str | None = None
    batch_size: int | str | None = None
    num_threads: int | None = None
    log_samples: bool = False
    max_seq_lengths: list[int] = dataclasses.field(default_factory=list)
    chat_completions: bool = False
    override_generation_kwargs: bool = True
    allow_unsafe_code: bool = False


@dataclasses.dataclass(frozen=True)
class SuiteDrivers:
    """A suite's run, report and stat drivers, as ``module:Class`` paths.

    They are imported on first use, so a suite's optional dependencies are needed only when that suite runs or reports.
    ``stat`` is None for a suite whose results keep no samples.
    """

    run: str
    report: str
    stat: str | None = None


_RESERVED_SECTIONS = frozenset({"general", "model", "judge", "generation", "slurm"})
_SUITES: dict[str, type] = {}
_DRIVERS: dict[str, SuiteDrivers] = {}
_ENTRY_CHECKS: dict[str, Callable[[list[Any]], None]] = {}
_SHARED_ENV: set[str] = set()


def register_suite(
    name: str,
    config_cls: type,
    check_entries: Callable[[list[Any]], None] | None = None,
    *,
    drivers: SuiteDrivers,
    own_venv: bool = True,
) -> None:
    """Register a suite, the dataclass of its ``[name]`` config section and its drivers.

    Args:
        name: The suite name, also the name of its config section.
        config_cls: A dataclass whose fields all have defaults.
        check_entries: Called with a recipe's entries of the suite, as
            ``config_cls`` instances; raises ValueError if they cannot run
            together.
        drivers: The suite's drivers.
        own_venv: Whether the suite runs from a ``.venv`` in its checkout; if
            not, it runs in domyn-eval's own Python.

    Raises:
        ValueError: If ``name`` is reserved or already registered.
        TypeError: If ``config_cls`` is not a dataclass with defaults.
    """
    if name in _RESERVED_SECTIONS or name in _SUITES:
        msg = f"suite name {name!r} is reserved or already registered"
        raise ValueError(msg)
    if not is_dataclass(config_cls):
        msg = f"{config_cls.__name__} is not a dataclass"
        raise TypeError(msg)
    required = [
        f.name
        for f in dataclasses.fields(config_cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if required:
        msg = f"{config_cls.__name__} fields without a default: {required}"
        raise TypeError(msg)
    _SUITES[name] = config_cls
    _DRIVERS[name] = drivers
    if check_entries is not None:
        _ENTRY_CHECKS[name] = check_entries
    if not own_venv:
        _SHARED_ENV.add(name)


def registered_suites() -> tuple[str, ...]:
    """Return the registered suite names, in registration order."""
    return tuple(_SUITES)


def suite_drivers(name: str) -> SuiteDrivers:
    """Return the drivers of suite ``name``.

    Raises:
        ValueError: If the suite is not registered.
    """
    if name not in _DRIVERS:
        msg = f"unknown suite {name!r}; registered suites are {list(_DRIVERS)}"
        raise ValueError(msg)
    return _DRIVERS[name]


def has_own_venv(name: str) -> bool:
    """Return whether suite ``name`` runs from a ``.venv`` in its checkout.

    Args:
        name: A suite name.
    """
    return name not in _SHARED_ENV


class SuiteConfig:
    """The config sections of the registered suites, as attributes.

    A suite is used when its section is in the loaded TOML, or once
    ``use`` marks it; snapshots only write the sections of used suites.
    """

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        """Build each registered suite's section from its raw table.

        Args:
            raw: Raw tables by suite name; missing suites get defaults.

        Raises:
            ValueError: If ``raw`` names an unregistered suite.
        """
        raw = raw or {}
        unknown = sorted(set(raw) - set(_SUITES))
        if unknown:
            msg = f"unknown config sections {unknown}; registered suites are {list(_SUITES)}"
            raise ValueError(msg)
        self._sections = {name: cls(**raw.get(name, {})) for name, cls in _SUITES.items()}
        self._used = set(raw)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - the section's class depends on the suite named
        """Return the section of suite ``name``."""
        sections = self.__dict__.get("_sections", {})
        if name not in sections:
            raise AttributeError(name)
        return sections[name]

    def sections(self) -> dict[str, Any]:
        """Return the sections by suite name."""
        return dict(self._sections)

    def use(self, name: str) -> None:
        """Mark suite ``name`` as used.

        Args:
            name: A registered suite.

        Raises:
            ValueError: If ``name`` is not a registered suite.
        """
        if name not in self._sections:
            msg = f"unknown suite {name!r}; registered suites are {list(_SUITES)}"
            raise ValueError(msg)
        self._used.add(name)

    def used(self) -> dict[str, Any]:
        """Return the sections of the used suites, by suite name."""
        return {name: section for name, section in self._sections.items() if name in self._used}

    def override(self, name: str, overrides: dict[str, Any]) -> None:
        """Replace fields of suite ``name``'s section, as a recipe entry does, and mark it used.

        Args:
            name: A registered suite.
            overrides: The fields to replace.

        Raises:
            ValueError: If ``name`` is not a registered suite, or an override is invalid.
        """
        self.use(name)
        self._sections[name] = dataclasses.replace(self._sections[name], **overrides)


# Fields set at run time or derived on load, left out of snapshots.
_RUNTIME_ONLY = (
    "general.file",
    "general.run_id",
    "model.endpoint_max_model_len",
    "judge.endpoint_max_model_len",
    "generation.max_samples",
)


@dataclass(config=ConfigDict(validate_assignment=True, extra="forbid", arbitrary_types_allowed=True))
class Config:
    """A whole evaluation config.

    Attributes:
        general: The ``[general]`` section.
        model: The model under evaluation.
        judge: The model grading answers, for suites that use one.
        generation: The ``[generation]`` section.
        slurm: The ``[slurm]`` section.
        suite: The sections of the registered suites.
    """

    general: GeneralConfig = dataclasses.field(default_factory=GeneralConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    judge: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    generation: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    slurm: SlurmConfig = dataclasses.field(default_factory=SlurmConfig)
    suite: SuiteConfig = dataclasses.field(default_factory=SuiteConfig)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the config at ``path``, merged into the file if it exists.

        Values are written resolved, as the run used them. The section of a
        used suite is written in full; other suites' sections are dropped
        from the file. Tables that are not config sections are kept.

        Args:
            path: The TOML file to write.
        """
        path = Path(path)
        existing = tomllib.loads(path.read_text()) if path.exists() else {}
        new = dataclass_to_clean_dict(self)
        new.pop("suite")
        for entry in _RUNTIME_ONLY:
            *parents, key = entry.split(".")
            table = new
            for parent in parents:
                table = table.get(parent, {})
            table.pop(key, None)
        for name in registered_suites():
            existing.pop(name, None)
        for name, section in self.suite.used().items():
            new[name] = dataclass_to_clean_dict(section)
        existing.update(new)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            tomli_w.dump(existing, f)
        tmp.replace(path)


def _load_generation(raw: dict[str, Any]) -> GenerationConfig:
    raw = dict(raw)
    return GenerationConfig(
        think=GenerationModeConfig(**raw.pop("think", {})),
        nothink=GenerationModeConfig(**raw.pop("nothink", {})),
        **raw,
    )


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load and validate a config TOML.

    Every top-level table other than the fixed sections must be the section
    of a registered suite.

    Args:
        path: The TOML file.

    Raises:
        ValueError: If a table or a field is unknown.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = expand_config(tomllib.load(f))
    general = raw.pop("general", {})
    general["file"] = str(path.absolute())
    return Config(
        general=GeneralConfig(**general),
        model=ModelConfig(**raw.pop("model", {})),
        judge=ModelConfig(**raw.pop("judge", {})),
        generation=_load_generation(raw.pop("generation", {})),
        slurm=SlurmConfig(**raw.pop("slurm", {})),
        suite=SuiteConfig(raw),
    )


def dataclass_to_clean_dict(obj: object) -> Any:  # noqa: ANN401 - a dict, list or value, shaped like ``obj``
    """Convert dataclasses to dicts, dropping None values (TOML has no null).

    Args:
        obj: A dataclass instance, or a list, dict or value inside one.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        cleaned = {k: dataclass_to_clean_dict(v) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v is not None}
    if isinstance(obj, list):
        cleaned = [dataclass_to_clean_dict(item) for item in obj]
        return [item for item in cleaned if item is not None]
    return obj


register_suite(
    "bfcl",
    BFCLConfig,
    drivers=SuiteDrivers(
        run="domyn_eval.drivers.run.bfcl:BFCLRunDriver",
        report="domyn_eval.drivers.report.bfcl:BFCLReportDriver",
        stat="domyn_eval.drivers.stat.bfcl:BFCLStatDriver",
    ),
)
register_suite(
    "inspect",
    InspectConfig,
    check_entries=_check_inspect_entries,
    drivers=SuiteDrivers(
        run="domyn_eval.drivers.run.inspect:InspectRunDriver",
        report="domyn_eval.drivers.report.inspect:InspectReportDriver",
        stat="domyn_eval.drivers.stat.inspect:InspectStatDriver",
    ),
)
register_suite(
    "lm_eval",
    LMEvalConfig,
    own_venv=False,
    drivers=SuiteDrivers(
        run="domyn_eval.drivers.run.lmeval:LMEvalRunDriver",
        report="domyn_eval.drivers.report.lmeval:LMEvalReportDriver",
    ),
)


# A suite entry of a recipe: a "key=value ..." override string, a dict, or a
# list of either, one suite invocation per item.
SuiteOverride = str | dict[str, Any] | list[str | dict[str, Any]]


@dataclass(config=_STRICT)
class RecipeConfig:
    """A named bundle of suite invocations.

    Attributes:
        name: The recipe name, by default its table key.
        suites: The invocations of each suite.
        description: What the recipe evaluates.
    """

    name: str
    suites: dict[str, SuiteOverride] = dataclasses.field(default_factory=dict)
    description: str = ""

    def entries(self) -> list[tuple[str, dict[str, Any]]]:
        """Return the suite invocations in order, as ``(suite, overrides of its section)``."""
        return [
            (suite, _override_args(entry))
            for suite, value in self.suites.items()
            for entry in (value if isinstance(value, list) else [value])
        ]


def _ancestors(raw: dict[str, Any], key: str, chain: tuple[str, ...] = ()) -> list[str]:
    """Return the recipes ``key`` extends, directly or not, each once.

    A recipe comes after the recipes it extends and parents keep their
    ``extend`` order, so a recipe shared by two parents is included once.
    """
    if key in chain:
        msg = f"recipe extend cycle: {' -> '.join((*chain, key))}"
        raise ValueError(msg)
    parents = raw[key].get("extend", [])
    parents = [parents] if isinstance(parents, str) else parents
    repeated = sorted({parent for parent in parents if parents.count(parent) > 1})
    if repeated:
        msg = f"recipe {key!r} extends {repeated} more than once"
        raise ValueError(msg)
    order: list[str] = []
    for parent in parents:
        if parent not in raw:
            msg = f"recipe {key!r} extends unknown recipe {parent!r}"
            raise ValueError(msg)
        for name in (*_ancestors(raw, parent, (*chain, key)), parent):
            if name not in order:
                order.append(name)
    return order


def _resolve_extend(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return recipe ``key`` with the suites of the recipes it extends.

    Every contribution to a suite concatenates: the extended recipes in
    ``_ancestors`` order, then the recipe's own entry.
    """
    merged: dict[str, list[Any]] = {}
    for name in (*_ancestors(raw, key), key):
        for suite, value in raw[name].get("suites", {}).items():
            merged.setdefault(suite, []).extend(value if isinstance(value, list) else [value])
    entry = {k: v for k, v in raw[key].items() if k != "extend"}
    entry["suites"] = {suite: values[0] if len(values) == 1 else values for suite, values in merged.items()}
    entry.setdefault("name", key)
    return entry


def _override_value(text: str) -> Any:  # noqa: ANN401 - whatever TOML value the override text parses to
    """Read an override value as TOML, as a dict entry's; other text is a string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _override_args(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Return the arguments of a suite entry: a dict, or ``key=value`` tokens."""
    if isinstance(entry, dict):
        return entry
    args = {}
    for token in entry.split():
        name, sep, value = token.partition("=")
        if not sep:
            msg = f"override {token!r} is not key=value"
            raise ValueError(msg)
        args[name] = _override_value(value)
    return args


def _check_entries(suite: str, values: list[Any]) -> None:
    """Check that a recipe's entries of ``suite`` can run together.

    Each entry is built as the suite's config section, so a string and a
    dict, or the same keys in another order, compare equal.
    """
    entries = [_SUITES[suite](**_override_args(v)) for v in values]
    if any(entry in entries[:i] for i, entry in enumerate(entries)):
        msg = "would run the same arguments twice"
        raise ValueError(msg)
    if suite in _ENTRY_CHECKS:
        _ENTRY_CHECKS[suite](entries)


def _validate_resolved(key: str, suites: dict[str, Any]) -> None:
    """Check the suites of recipe ``key`` and their entries."""
    unknown = sorted(set(suites) - set(_SUITES))
    if unknown:
        msg = f"recipe {key!r} names unknown suites {unknown}; registered suites are {list(_SUITES)}"
        raise ValueError(msg)
    for suite, value in suites.items():
        try:
            _check_entries(suite, value if isinstance(value, list) else [value])
        except ValueError as e:
            msg = f"recipe {key!r}: suite {suite!r}: {e}"
            raise ValueError(msg) from e


def load_recipes(path: str | os.PathLike[str]) -> dict[str, RecipeConfig]:
    """Load a recipes TOML, resolving ``extend``.

    Args:
        path: The TOML file.

    Raises:
        ValueError: If ``extend`` names an unknown recipe, repeats one or
            forms a cycle, or if a recipe names an unknown suite or has
            entries of a suite that are invalid or cannot run together.
    """
    with Path(path).open("rb") as f:
        raw = expand_config(tomllib.load(f))
    recipes = {}
    for key in raw:
        entry = _resolve_extend(raw, key)
        _validate_resolved(key, entry["suites"])
        recipes[key] = RecipeConfig(**entry)
    return recipes
