from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import domyn_eval.swarm as swarm_driver
from domyn_eval.config import Config, ModelConfig, load_config
from domyn_eval.swarm import SwarmHandler
from tests.helpers import RUNNING, ModelConfigStub

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "sample_config.toml")


@pytest.fixture
def sbatch(tmp_path, monkeypatch):
    """A fake ``sbatch`` on PATH that records its arguments and prints job ids from 4242 up, or fails if told to."""
    calls = tmp_path / "calls"
    fake = tmp_path / "bin" / "sbatch"
    fake.parent.mkdir()
    fake.write_text(
        f'#!/bin/sh\necho "$@" >> {calls}\n'
        '[ -n "$SBATCH_FAIL" ] && { echo "invalid account" >&2; exit 1; }\n'
        f'echo "$((4241 + $(wc -l < {calls})));cluster"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{Path('/usr/bin')}:{Path('/bin')}")
    return calls


@pytest.fixture
def swarms(monkeypatch):
    """Install records (newest first) and ``{name: [phase, ...]}`` probe scripts.

    Each probe pops the next phase; the last one repeats. A name with no script
    fails to load, like a record whose swarm can no longer be rehydrated.
    """
    probes = []

    def install(records, phases):
        monkeypatch.setattr(swarm_driver.SwarmStateManager, "list_all", staticmethod(lambda: records))
        urls = {r["deployment_name"]: r["endpoint"] for r in records}

        def from_state(name):
            seq = phases[name]  # KeyError, probe failure
            probes.append(name)

            def serving_status():
                phase = seq.pop(0) if len(seq) > 1 else seq[0]
                url = urls.get(name) if phase is RUNNING else None
                return SimpleNamespace(phase=phase, url=url)

            return SimpleNamespace(serving_status=serving_status)

        monkeypatch.setattr(
            swarm_driver.domyn_swarm,
            "DomynLLMSwarm",
            SimpleNamespace(from_state=from_state),
        )
        return probes

    return install


@pytest.fixture
def swarm_handler(tmp_path):
    def build(
        main_model_cfg: ModelConfigStub,
        judge_model_cfg: ModelConfigStub | None = None,
    ):
        def build_model_config(cfg: ModelConfigStub, label: str) -> ModelConfig:
            swarm_cfg = tmp_path / f"{label}_swarm.yaml"
            swarm_cfg.write_text(
                yaml.safe_dump(
                    {
                        "name": cfg.model_name,
                        "model": f"/models/{cfg.model_name}",
                        "image": "fake_image",
                        "backend": {"type": "lepton"},
                    }
                )
            )
            return ModelConfig(swarm_config=str(swarm_cfg), swarm_name=cfg.swarm_name, url=cfg.model_url)

        model_config = build_model_config(main_model_cfg, "model")
        if judge_model_cfg is not None:
            judge_config = build_model_config(judge_model_cfg, "judge")
            config = Config(model=model_config, judge=judge_config)
        else:
            config = Config(model=model_config)

        return SwarmHandler(config)

    return build
