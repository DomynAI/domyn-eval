# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Run a model's evaluation, behind ``domyn-eval run``: suite entries run locally or as SLURM jobs."""

import copy
import dataclasses
import logging
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from rich.console import Console
from rich.table import Table

from domyn_eval.config import Config
from domyn_eval.drivers import run_driver
from domyn_eval.drivers.run.base import BaseRunDriver
from domyn_eval.log import display_path
from domyn_eval.paths import logs_dir
from domyn_eval.preflight import (
    FAIL,
    WARN,
    Check,
    Preflight,
    endpoint_client,
    failed,
    list_models,
    render_checks,
    served_max_model_len,
)
from domyn_eval.slurm import compiler

if TYPE_CHECKING:
    from domyn_eval.swarm import SwarmHandler

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    """One invocation of a suite: the suite, and the overrides of its section.

    Attributes:
        suite: A registered suite.
        overrides: Fields of the suite's section to replace, as a recipe entry sets them.
    """

    suite: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """The suite, with its overrides."""
        args = " ".join(f"{key}={value}" for key, value in self.overrides.items())
        return f"{self.suite}({args})" if args else self.suite


@dataclass(frozen=True)
class RunOptions:
    """How to run the entries.

    Attributes:
        swarm: Serve the model, and the judge or user model if an entry needs its role, with domyn-swarm.
        keep_swarm: With ``swarm``, leave up the swarms this run deploys.
        check: Run the preflight checks first.
        dry: Print the commands, or write the job scripts, without running or submitting them.
        sbatch: Submit each entry as a SLURM job.
        chain: With ``sbatch``, start each job once the previous one has ended.
        after: With ``sbatch``, a job id the first job waits for.
    """

    swarm: bool = False
    keep_swarm: bool = False
    check: bool = True
    dry: bool = False
    sbatch: bool = False
    chain: bool = True
    after: str | None = None


@dataclass(frozen=True)
class Outcome:
    """How an entry went, for the summary.

    Attributes:
        label: The entry.
        status: ``ok``, ``failed``, ``submitted`` or ``dry run``.
        detail: The elapsed time, the job id, or what a dry run would do.
        results: The entry's results folder.
    """

    label: str
    status: str
    detail: str
    results: str


def run(config: Config, entries: list[Entry], options: RunOptions, console: Console | None = None) -> int:
    """Run the entries against the model's endpoint and print a summary.

    A swarm this run deploys is torn down when the run ends, or with ``sbatch`` by a job that waits for the others,
    unless ``keep_swarm`` is set.

    Args:
        config: The run config; its endpoints and ``general.run_id`` are filled in.
        entries: The suite invocations, in order.
        options: How to run them.
        console: Where to print the summary.

    Returns:
        The exit code: 1 if the preflight or an entry failed.

    Raises:
        ValueError: If the swarm handler cannot serve the model, or a model a role needs.
        RuntimeError: If a swarm fails to come up.
    """
    config.general.run_id = config.general.run_id or str(int(time.time()))
    handler = _swarm_handler(config) if options.swarm else None
    submitted = False
    try:
        if handler:
            handler.ensure_endpoint(handler.ModelType.MODEL, options.dry)
        drivers = [run_driver(e.suite)(_entry_config(config, entries, e)) for e in entries]
        needed = _needed_roles(entries, drivers)
        if handler:
            for section in sorted({s for role in needed if (s := config.role_section(role)) is not None}):
                handler.ensure_endpoint(handler.ModelType(section), options.dry)
        if options.check:
            deploying = options.dry and handler is not None
            checks = Preflight(config, [e.suite for e in entries], deploying=deploying).run()
            checks += _check_roles(config, needed, deploying=deploying)
            render_checks(checks, console)
            if failed(checks):
                logger.error("preflight: failed; fix the above, or pass --no-check")
                return 1
        if not options.dry and config.model.url:
            _read_context_window(config)
        if options.sbatch:
            outcomes, submitted = _submit(config, entries, drivers, handler, options)
        else:
            outcomes = [
                _run_local(e, d, log, config, dry=options.dry)
                for e, d, log in zip(entries, drivers, _log_names(entries), strict=True)
            ]
    finally:
        if handler and not options.dry:
            _release(handler, keep=options.keep_swarm, submitted=submitted)
    _render_summary(config, outcomes, console or Console())
    return int(any(o.status == "failed" for o in outcomes))


def _swarm_handler(config: Config) -> "SwarmHandler":
    from domyn_eval.swarm import SwarmHandler  # here, as domyn_swarm takes seconds to import

    return SwarmHandler(config)


def _release(handler: "SwarmHandler", *, keep: bool, submitted: bool) -> None:
    """Tear down the swarms the run deployed, unless kept or still needed by submitted jobs."""
    if keep:
        for name in handler.deployed_swarms().values():
            logger.info("swarm: %s stays up; stop it with `domyn-swarm down %s -y`", name, name)
    elif not submitted:  # submitted jobs still need the swarm: a teardown job follows them instead
        for model_type in handler.ModelType:
            handler.down(model_type)


def _entry_config(config: Config, entries: list[Entry], entry: Entry) -> Config:
    """The config of an entry: the run's, with each suite's section overridden by its entry.

    Endpoints and their generation settings are shared with the run's config, the suite sections are copied. The other
    suites take the overrides of their last entry, so the run's ``config.toml`` holds every suite of the run.
    """
    entry_config = dataclasses.replace(config, suite=copy.deepcopy(config.suite))
    last = {e.suite: e for e in entries} | {entry.suite: entry}
    for e in last.values():
        entry_config.suite.override(e.suite, e.overrides)
    return entry_config


def _needed_roles(entries: list[Entry], drivers: list[BaseRunDriver]) -> dict[str, list[str]]:
    """The entries needing each model role, by role."""
    needed: dict[str, list[str]] = {}
    for entry, driver in zip(entries, drivers, strict=True):
        for role in sorted(driver.roles):
            needed.setdefault(role, []).append(entry.label)
    return needed


def _check_roles(config: Config, needed: dict[str, list[str]], *, deploying: bool) -> list[Check]:
    """Check that the model playing each needed role has an endpoint, or a swarm about to serve it."""
    checks = []
    for role, labels in needed.items():
        who = f"{', '.join(labels)}: need a {role} model"
        section = config.role_section(role)
        if section is None:
            where = "[user] or [judge]" if role == "user" else "[judge]"
            checks.append(Check(FAIL, f"{who}, but no {where} is set"))
            continue
        model = getattr(config, section)
        if model.url:
            continue
        if deploying and (model.swarm_config or model.swarm_name):
            checks.append(Check(WARN, f"{section}.url is set once its swarm is deployed; the {role} is not checked"))
        else:
            checks.append(Check(FAIL, f"{who}, but {section}.url is not set; set it, or serve one with --swarm"))
    return checks


def _read_context_window(config: Config) -> None:
    """Set ``model.endpoint_max_model_len`` from the model card the endpoint serves, if it advertises one."""
    try:
        with endpoint_client() as client:
            cards = list_models(client, config.model.url or "")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("model: could not read the endpoint's context window; %s", e)
        return
    card = next((card for card in cards if card.get("id") == config.model.path), None)
    config.model.endpoint_max_model_len = served_max_model_len(card) if card else None


def _run_local(entry: Entry, driver: BaseRunDriver, log: str, config: Config, *, dry: bool) -> Outcome:
    if dry:
        command = shlex.join(driver.build_command())
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in driver.command_env.items())
        cwd = f"cd {shlex.quote(str(driver.command_cwd))} && " if driver.command_cwd else ""
        logger.info("%s: %s%s%s", entry.label, cwd, f"{env} " if env else "", command)
        return Outcome(entry.label, "dry run", "command printed", driver.output_dir)
    start = time.monotonic()
    try:
        if config.general.auto_setup:
            driver.setup()
        code = driver.run(log)
    except Exception:
        logger.exception("%s: failed", entry.label)
        code = 1
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - start))
    if not code:
        logger.info("%s: done in %s", entry.label, elapsed)
    return Outcome(entry.label, "failed" if code else "ok", elapsed, driver.output_dir)


def _submit(
    config: Config,
    entries: list[Entry],
    drivers: list[BaseRunDriver],
    handler: "SwarmHandler | None",
    options: RunOptions,
) -> tuple[list[Outcome], bool]:
    """Write and submit each entry's job, then the jobs tearing down the swarms this run deployed.

    Returns:
        The outcomes, and whether any job was submitted.
    """
    outcomes: list[Outcome] = []
    jobs: list[str] = []
    previous = [options.after] if options.after else []
    try:
        for log, entry, driver in zip(_log_names(entries), entries, drivers, strict=True):
            try:
                if config.general.auto_setup and not options.dry:
                    driver.setup()  # here, where the network is: compute nodes may have none
                job_name = f"{config.model.id}-{entry.suite}{log.removeprefix('run')}"
                script = compiler.write(config.slurm, job_name, driver.sbatch_command(), driver.log_dir, log)
                if options.dry:
                    outcomes.append(Outcome(entry.label, "dry run", f"wrote {script}", driver.output_dir))
                    continue
                job = compiler.submit(script, after=previous)
            except Exception:
                logger.exception("%s: failed", entry.label)
                outcomes.append(Outcome(entry.label, "failed", "not submitted", driver.output_dir))
                continue
            logger.info(
                "%s: submitted job %s; log in %s", entry.label, job, display_path(compiler.job_log(script, job))
            )
            jobs.append(job)
            previous = [job] if options.chain else previous
            outcomes.append(Outcome(entry.label, "submitted", f"job {job}", driver.output_dir))
    finally:
        if jobs and handler and not options.keep_swarm:
            _submit_teardown(config, handler, jobs)
    return outcomes, bool(jobs)


def _log_names(entries: list[Entry]) -> list[str]:
    """Each entry's run log: ``run``, numbered ``run-<n>`` when a suite has several entries, as they share a folder."""
    counts = {e.suite: sum(o.suite == e.suite for o in entries) for e in entries}
    seen: dict[str, int] = {}
    names = []
    for e in entries:
        seen[e.suite] = seen.get(e.suite, 0) + 1
        names.append(f"run-{seen[e.suite]}" if counts[e.suite] > 1 else "run")
    return names


def _submit_teardown(config: Config, handler: "SwarmHandler", jobs: list[str]) -> None:
    """Submit a job per swarm this run deployed, stopping it once every job of the run has ended."""
    domyn_swarm = str(Path(sys.executable).with_name("domyn-swarm"))
    log_dir = logs_dir(config.model.id or "", config.general.run_id or "", "swarm", create=True)
    for model_type, name in handler.deployed_swarms().items():
        command = [domyn_swarm, "down", name, "-y"]
        try:
            job_name = f"{config.model.id}-down-{model_type.value}"
            script = compiler.write(config.slurm, job_name, command, log_dir, f"down-{model_type.value}")
            job = compiler.submit(script, after=jobs)
            logger.info("swarm: %s goes down after the run; job %s", name, job)
        except (OSError, RuntimeError):
            logger.exception(
                "swarm: could not submit the teardown of %s; stop it once the jobs end with `%s`",
                name,
                shlex.join(command),
            )


def _render_summary(config: Config, outcomes: list[Outcome], console: Console) -> None:
    table = Table(title=f"Run {config.general.run_id} of {config.model.id}")
    for column in ("entry", "status", "", "results"):
        table.add_column(column)
    colors = {"ok": "green", "failed": "red", "submitted": "cyan", "dry run": "yellow"}
    for o in outcomes:
        table.add_row(o.label, f"[{colors[o.status]}]{o.status}[/]", o.detail, o.results)
    console.print(table)
    if any(o.status in ("ok", "submitted") for o in outcomes):
        console.print(f"Once the results are in: domyn-eval report {config.model.id}", highlight=False)
