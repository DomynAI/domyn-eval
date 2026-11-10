# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Command line interface of domyn-eval."""

from pathlib import Path
from typing import TypeVar

import click
import yaml

from domyn_eval.config import Config, load_config, load_recipes, registered_suites
from domyn_eval.drivers.report.base import DuplicateBenchmarkError
from domyn_eval.drivers.stat.base import StatsProfile
from domyn_eval.log import set_level, setup_logging
from domyn_eval.preflight import Preflight, failed, render_checks
from domyn_eval.report import Leaderboard, ModelReport, ReportProfile, render_duplicates
from domyn_eval.run import Entry, RunOptions
from domyn_eval.run import run as run_entries
from domyn_eval.stats import ModelStats

_CONFIG_PATH = click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)

_Profile = TypeVar("_Profile", ReportProfile, StatsProfile)


def _load(path: Path) -> Config:
    try:
        config = load_config(path)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    set_level(config.general.log_level)
    return config


def _selected_entries(config: Config, suites: tuple[str, ...], recipe: str | None) -> list[Entry]:
    """The --suite entries the recipe does not run, then the recipe's."""
    unknown = sorted(set(suites) - set(registered_suites()))
    if unknown:
        msg = f"unknown suites {unknown}; registered suites are {list(registered_suites())}"
        raise click.UsageError(msg)
    recipe_entries = []
    if recipe:
        try:
            recipes = load_recipes(Path(config.general.recipes_file))
        except FileNotFoundError as e:
            msg = f"no recipes file {e.filename}; set general.recipes_file"
            raise click.ClickException(msg) from e
        except (OSError, ValueError) as e:
            raise click.ClickException(str(e)) from e
        if recipe not in recipes:
            msg = f"unknown recipe {recipe!r}; known are {sorted(recipes)}"
            raise click.ClickException(msg)
        recipe_entries = [Entry(suite, overrides) for suite, overrides in recipes[recipe].entries()]
    in_recipe = {e.suite for e in recipe_entries}
    return [Entry(s) for s in dict.fromkeys(suites) if s not in in_recipe] + recipe_entries


@click.group()
@click.version_option(package_name="domyn-eval", prog_name="domyn-eval")
def cli() -> None:
    """Evaluate LLMs on benchmark suites."""
    setup_logging()


@cli.command()
@_CONFIG_PATH
@click.option("--suite", "suites", multiple=True, help="A suite to run.")
@click.option("--recipe", help="Run the suites of this recipe.")
@click.option("--run-id", help="The run's id; by default the current time.")
@click.option("--max-samples", type=click.IntRange(min=1), help="Evaluate at most this many samples per task.")
@click.option("--no-setup", is_flag=True, help="Skip setting up the suites' environments.")
@click.option(
    "--swarm",
    is_flag=True,
    help="Serve the model (and a judge, if needed) with domyn-swarm when its url is down or unset.",
)
@click.option("--keep-swarm", is_flag=True, help="With --swarm, leave up the swarms the run deploys.")
@click.option("--no-check", is_flag=True, help="Skip the preflight checks.")
@click.option("--dry", is_flag=True, help="Print the commands, or write the job scripts, without running them.")
@click.option("--sbatch", is_flag=True, help="Submit each suite as a SLURM job, one after another.")
@click.option("--no-deps", is_flag=True, help="With --sbatch, let the jobs run at the same time.")
@click.option("--after", metavar="JOB", help="With --sbatch, start once this SLURM job has ended.")
def run(
    config_path: Path,
    suites: tuple[str, ...],
    recipe: str | None,
    run_id: str | None,
    max_samples: int | None,
    no_setup: bool,
    swarm: bool,
    keep_swarm: bool,
    no_check: bool,
    dry: bool,
    sbatch: bool,
    no_deps: bool,
    after: str | None,
) -> None:
    """Evaluate the model on the selected suites, locally or as SLURM jobs.

    The suites send their requests to model.url. With --swarm, a model.url that is down or unset is replaced by a
    running swarm of the model, or by one deployed from model.swarm_config and torn down when the run ends; the same
    goes for the judge, if a suite needs one.
    """
    if (no_deps or after) and not sbatch:
        msg = "--no-deps and --after go with --sbatch"
        raise click.UsageError(msg)
    if keep_swarm and not swarm:
        msg = "--keep-swarm goes with --swarm"
        raise click.UsageError(msg)
    config = _load(config_path)
    entries = _selected_entries(config, suites, recipe)
    if not entries:
        msg = "select suites with --suite or --recipe"
        raise click.UsageError(msg)
    if run_id:
        config.general.run_id = run_id
    if max_samples:
        config.generation.max_samples = max_samples
    if no_setup:
        config.general.auto_setup = False
    options = RunOptions(
        swarm=swarm,
        keep_swarm=keep_swarm,
        check=not no_check,
        dry=dry,
        sbatch=sbatch,
        chain=not no_deps,
        after=after,
    )
    try:
        code = run_entries(config, entries, options)
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    if code:
        raise SystemExit(code)


@cli.command()
@_CONFIG_PATH
@click.option("--suite", "suites", multiple=True, help="A suite to check.")
@click.option("--recipe", help="Check the suites of this recipe.")
@click.option("--no-ping", is_flag=True, help="Skip the endpoint checks.")
def check(
    config_path: Path,
    suites: tuple[str, ...],
    recipe: str | None,
    no_ping: bool,
) -> None:
    """Check the config, environment and endpoint before a run.

    Suites are checked only when selected with --suite or --recipe.
    """
    config = _load(config_path)
    selected = list(dict.fromkeys(e.suite for e in _selected_entries(config, suites, recipe)))
    checks = Preflight(config, selected).run(ping=not no_ping)
    render_checks(checks)
    if failed(checks):
        raise SystemExit(1)


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--suite", "suites", multiple=True, help="Only report this suite.")
@click.option(
    "--profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A YAML profile of the benchmarks and metrics to keep, in order.",
)
@click.option("--leaderboard", is_flag=True, help="Line the models up side by side instead.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Where to write the CSV.")
@click.option(
    "--stats-profile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A YAML profile of when the stats flag scores; by default configs/stats_profile.yaml's values.",
)
@click.option("--no-stats", is_flag=True, help="Skip the stats, which read every sample.")
def report(
    models: tuple[str, ...],
    suites: tuple[str, ...],
    profile: Path | None,
    leaderboard: bool,
    output: Path | None,
    stats_profile: Path | None,
    no_stats: bool,
) -> None:
    """Build the results report of each model, or their leaderboard.

    MODELS are model ids, the folders under the runs directory. Reports are written to <runs>/<model>/report.csv and
    the leaderboard to <runs>/leaderboard.csv, unless --output is given.

    Each report is followed by the stats of its runs' samples, which flag the scores to distrust: truncated, empty,
    looping or unscored answers, and reasoning that does not match the run's mode.
    """
    models = tuple(dict.fromkeys(models))
    if output and len(models) > 1 and not leaderboard:
        msg = "--output takes a single model, unless with --leaderboard"
        raise click.UsageError(msg)
    if stats_profile and (leaderboard or no_stats):
        msg = "--stats-profile goes with the stats, which --leaderboard and --no-stats leave out"
        raise click.UsageError(msg)
    unknown = sorted(set(suites) - set(registered_suites()))
    if unknown:
        msg = f"unknown suites {unknown}; registered suites are {list(registered_suites())}"
        raise click.UsageError(msg)
    report_profile = _load_profile(ReportProfile, profile)
    thresholds = _load_profile(StatsProfile, stats_profile)
    reports = [_model_report(model, list(suites), report_profile) for model in models]
    if leaderboard:
        board = Leaderboard.from_reports(reports)
        board.render()
        click.echo(f"Wrote {board.save(output)}")
        return
    for model_report in reports:
        model_report.render()
        if not model_report.df.empty:
            click.echo(f"Wrote {model_report.save(output)}")
        if not no_stats:
            _model_stats(model_report.model, list(suites), thresholds).render()


def _load_profile(cls: type[_Profile], path: Path | None) -> _Profile | None:
    try:
        return cls.load(path) if path else None
    except (yaml.YAMLError, ValueError) as e:
        msg = f"{path}: {e}"
        raise click.ClickException(msg) from e


def _model_report(model: str, suites: list[str], profile: ReportProfile | None) -> ModelReport:
    try:
        model_report = ModelReport.build(model, suites)
    except DuplicateBenchmarkError as e:
        render_duplicates(e)
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except ImportError as e:
        raise _missing_extra(e) from e
    return model_report.with_profile(profile, suites) if profile else model_report


def _model_stats(model: str, suites: list[str], profile: StatsProfile | None) -> ModelStats:
    try:
        return ModelStats.build(model, suites, profile)
    except ImportError as e:
        raise _missing_extra(e) from e


def _missing_extra(error: ImportError) -> click.ClickException:
    return click.ClickException(f"{error}; install the extra of the suite that needs it, e.g. domyn-eval[inspect-ai]")
