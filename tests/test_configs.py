"""The configs shipped in configs/ load and point at files that exist."""

from pathlib import Path

import domyn_swarm
import pytest
import yaml
from domyn_swarm.config import defaults
from domyn_swarm.config.settings import get_settings

from domyn_eval.config import load_config, load_recipes
from domyn_eval.drivers.run.inspect import InspectTask
from domyn_eval.report import ReportProfile

REPO = Path(__file__).parents[1]
CONFIGS = REPO / "configs"
MODEL_CONFIGS = sorted(CONFIGS.glob("*/*.toml"))
SWARM_CONFIGS = sorted(CONFIGS.glob("*/swarms/*.yaml"))


def test_every_config_is_found() -> None:
    assert [p.relative_to(CONFIGS).as_posix() for p in MODEL_CONFIGS] == [
        "domyn/domyn-edge.toml",
        "qwen/qwen3.5-4b.toml",
    ]
    assert len(SWARM_CONFIGS) == 3


def test_the_recipes_load() -> None:
    recipes = load_recipes(CONFIGS / "recipes.toml")

    assert set(recipes["domyn_edge"].suites) == {"inspect", "bfcl"}


def _recipe_inspect_tasks() -> list[InspectTask]:
    entries = [o for r in load_recipes(CONFIGS / "recipes.toml").values() for s, o in r.entries() if s == "inspect"]
    return [InspectTask.model_validate(task) for entry in entries for task in entry["custom_tasks"]]


def test_the_recipes_inspect_tasks_are_valid_and_only_tau_needs_a_user() -> None:
    tasks = _recipe_inspect_tasks()

    assert tasks
    for task in tasks:
        tau = task.task.startswith("inspect_evals/tau2_")
        assert task.roles == (["user"] if tau else []), task.task


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.name)
def test_a_model_config_loads_and_its_swarms_exist(path: Path) -> None:
    config = load_config(path)

    assert config.model.id == path.stem
    for swarm_config in (config.model.swarm_config, config.judge.swarm_config):
        assert swarm_config
        assert (REPO / swarm_config).is_file()


@pytest.mark.parametrize("path", SWARM_CONFIGS, ids=lambda p: p.name)
def test_a_swarm_config_is_valid_with_the_sites_images(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "defaults.yaml"
    slurm = {"partition": "gpu", "account": "acct", "qos": "normal", "endpoint": {"nginx_image": "/images/nginx.sif"}}
    site.write_text(yaml.safe_dump({"image": "/images/vllm.sif", "slurm": slurm}))
    monkeypatch.setenv("DOMYN_SWARM_DEFAULTS", str(site))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    get_settings.cache_clear()  # both cache what they read, here the site's defaults.yaml
    defaults.reload_defaults_cache()

    try:
        config = domyn_swarm.DomynLLMSwarmConfig.read(str(path))
    finally:
        get_settings.cache_clear()
        defaults.reload_defaults_cache()

    assert config.name == path.stem
    assert config.image == "/images/vllm.sif"


def test_a_recipe_runs_every_inspect_benchmark_of_the_report_profile() -> None:
    # domyn-inspect names a task's results after its display_name, else its task name, "_pass_k" added with pass_k
    produced = {
        t.display_name or t.task.rsplit("/", 1)[-1] + ("_pass_k" if t.pass_k else "") for t in _recipe_inspect_tasks()
    }
    profile = ReportProfile.load(CONFIGS / "report_profile.yaml")
    wanted = {b.benchmark for section in profile.sections for b in section.benchmarks if b.suite == "inspect"}

    assert wanted <= produced, sorted(wanted - produced)
