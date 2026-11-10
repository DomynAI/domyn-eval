# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""SLURM job scripts for a run's suites, and their submission with ``sbatch``.

The job inherits the submitting shell's environment (sbatch's default ``--export=ALL``), so the script holds no
secrets: only its ``#SBATCH`` directives and the suite's command.
"""

import logging
import shlex
import subprocess
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from domyn_eval.config import SlurmConfig
from domyn_eval.log import display_path

logger = logging.getLogger(__name__)


def render(slurm: SlurmConfig, job_name: str, command: list[str], log_file: Path) -> str:
    """Return the job script that runs ``command``.

    Args:
        slurm: The ``[slurm]`` section; its ``template`` replaces the packaged one.
        job_name: The job's name.
        command: The program and its arguments.
        log_file: Where the job's stdout and stderr go.
    """
    if slurm.template:
        source = Path(slurm.template).expanduser().read_text()
    else:
        source = files("domyn_eval.slurm").joinpath("template.sh.j2").read_text()
    # A shell script, not HTML: nothing to escape. StrictUndefined fails on a variable the template misspells.
    template = Environment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False).from_string(source)
    return template.render(
        job_name=job_name,
        log_file=log_file,
        directives=slurm.directives(),
        command=shlex.join(command),
    )


def write(slurm: SlurmConfig, job_name: str, command: list[str], log_dir: Path, name: str) -> Path:
    """Write the job script to ``<log_dir>/<name>.sbatch``, next to the ``<name>-<job id>.log`` the job writes.

    Args:
        slurm: The ``[slurm]`` section.
        job_name: The job's name, as ``squeue`` shows it.
        command: The program and its arguments.
        log_dir: The folder of the script and the log.
        name: Names the script and the log, e.g. ``run``.

    Returns:
        The script's path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}.sbatch"
    path.write_text(render(slurm, job_name, command, (log_dir / f"{name}-%j.log").absolute()))
    logger.debug("slurm: script %s", display_path(path))
    return path


def job_log(script: Path, job_id: str) -> Path:
    """The log a submitted job of ``script`` writes."""
    return script.with_name(f"{script.stem}-{job_id}.log")


def submit(script: Path, after: Sequence[str] = ()) -> str:
    """Submit a job script.

    Args:
        script: The job script.
        after: Job ids; the job starts once they have all ended, whatever their outcome.

    Returns:
        The id of the submitted job.

    Raises:
        RuntimeError: If ``sbatch`` fails.
    """
    args = ["sbatch", "--parsable"]
    if after:
        args.append(f"--dependency=afterany:{':'.join(after)}")
    result = subprocess.run([*args, str(script)], capture_output=True, text=True, check=False)
    if result.returncode:
        msg = f"sbatch failed on {script}: {result.stderr.strip()}"
        raise RuntimeError(msg)
    # --parsable prints "<job id>" or "<job id>;<cluster>"
    return result.stdout.strip().split(";")[0]
