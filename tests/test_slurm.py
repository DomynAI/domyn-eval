import pytest
from jinja2 import UndefinedError

from domyn_eval.config import SlurmConfig
from domyn_eval.slurm.compiler import job_log, render, submit, write

COMMAND = ["env", "--chdir", "/suites/x", "KEY=a b", "python", "main.py", "--run"]


def _sbatch_lines(script: str) -> list[str]:
    return [line for line in script.splitlines() if line.startswith("#SBATCH")]


def test_only_set_resources_become_directives():
    slurm = SlurmConfig(account="acct", walltime="02:00:00", nodes=1, memory="16G")
    assert slurm.directives() == {"account": "acct", "time": "02:00:00", "nodes": "1", "mem": "16G"}
    assert SlurmConfig().directives() == {}


def test_every_resource_has_an_sbatch_option():
    slurm = SlurmConfig(
        account="a",
        partition="p",
        qos="q",
        walltime="1:00:00",
        nodes=2,
        ntasks_per_node=1,
        cpus_per_task=8,
        memory="1G",
        gres="gpu:1",
        exclude="n1",
    )
    assert set(slurm.directives()) == {
        "account",
        "partition",
        "qos",
        "time",
        "nodes",
        "ntasks-per-node",
        "cpus-per-task",
        "mem",
        "gres",
        "exclude",
    }


def test_the_script_runs_the_command_quoted(tmp_path):
    script = render(SlurmConfig(partition="gpu"), "inspect", COMMAND, tmp_path / "inspect-%j.log")
    assert script.startswith("#!/bin/bash\n")
    assert _sbatch_lines(script) == [
        "#SBATCH --job-name=inspect",
        f"#SBATCH --output={tmp_path}/inspect-%j.log",
        "#SBATCH --partition=gpu",
    ]
    assert script.rstrip().endswith("env --chdir /suites/x 'KEY=a b' python main.py --run")


def test_a_custom_template_replaces_the_packaged_one(tmp_path):
    template = tmp_path / "job.j2"
    template.write_text("#!/bin/sh\n#SBATCH --exclusive\n{{ command }}\n")
    script = render(SlurmConfig(template=str(template)), "bfcl", ["true"], tmp_path / "log")
    assert script == "#!/bin/sh\n#SBATCH --exclusive\ntrue\n"


def test_a_template_with_an_unknown_variable_fails(tmp_path):
    template = tmp_path / "job.j2"
    template.write_text("{{ comand }}\n")
    with pytest.raises(UndefinedError, match="comand"):
        render(SlurmConfig(template=str(template)), "bfcl", ["true"], tmp_path / "log")


def test_write_puts_the_script_next_to_its_log(tmp_path):
    log_dir = tmp_path / "logs" / "inspect"
    path = write(SlurmConfig(), "model-inspect", ["true"], log_dir, "run")
    assert path == log_dir / "run.sbatch"
    assert job_log(path, "42") == log_dir / "run-42.log"
    text = path.read_text()
    assert "#SBATCH --job-name=model-inspect" in text
    assert f"#SBATCH --output={log_dir.absolute()}/run-%j.log" in text


def test_submit_returns_the_job_id(sbatch, tmp_path):
    assert submit(tmp_path / "job.sbatch") == "4242"
    assert sbatch.read_text().split() == ["--parsable", str(tmp_path / "job.sbatch")]


def test_submit_chains_after_jobs(sbatch, tmp_path):
    submit(tmp_path / "job.sbatch", after=["17", "18"])
    assert "--dependency=afterany:17:18" in sbatch.read_text().split()


@pytest.mark.usefixtures("sbatch")
def test_a_failed_submission_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SBATCH_FAIL", "1")
    with pytest.raises(RuntimeError, match="invalid account"):
        submit(tmp_path / "job.sbatch")
