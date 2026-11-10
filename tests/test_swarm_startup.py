"""Tests for failing fast when a swarm dies during startup.

``deploy_swarm`` runs ``DomynLLMSwarm.__enter__`` on a worker thread and watches
the Slurm job ids it records, so every test here stubs ``DomynLLMSwarm`` with a
fake whose ``__enter__`` blocks (as domyn-swarm's readiness loop does) and
``SlurmDriver.get_job_state`` with a scripted sequence of states.
"""

import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest
from domyn_swarm import DomynLLMSwarmConfig

import domyn_eval.swarm as swarm_driver
from domyn_eval.swarm import SwarmHandler, SwarmStartupError, deploy_swarm
from tests.helpers import RUNNING, ModelConfigStub, record

NEW_DEPLOYMENT_ENDPOINT = "http://node1:9000"


class FakeSwarm:
    """Records its job ids, then blocks until released or returns/raises."""

    instances: ClassVar[list] = []

    def __init__(self, cfg, meta=None, block=True, exc=None):  # noqa: FBT002
        self.cfg = cfg
        self.name = "sw-test"
        self.swarm_dir = "/tmp/sw-test"
        self.serving_handle = None
        self.meta = {"jobid": 1, "lb_jobid": 2} if meta is None else meta
        self.block = block
        self.exc = exc
        self.release = threading.Event()
        self.cleanups = 0
        FakeSwarm.instances.append(self)
        self.endpoint = NEW_DEPLOYMENT_ENDPOINT
        if isinstance(cfg, DomynLLMSwarmConfig):
            self.model = cfg.model

    def __enter__(self):
        self.serving_handle = SimpleNamespace(meta=self.meta)
        if self.block:
            self.release.wait(5)
        if self.exc:
            raise self.exc
        return self

    def _cleanup_failed_startup(self):
        self.cleanups += 1
        self.release.set()

    def cleanup(self):
        self.cleanups += 1


@pytest.fixture
def fake(monkeypatch):
    """Install FakeSwarm (built with ``kwargs``) and a scripted job-state list."""
    FakeSwarm.instances = []

    def install(states=("RUNNING",), **kwargs):
        seq = list(states)

        class ConfiguredFakeSwarm(FakeSwarm):
            # A class rather than a lambda: swarm.py uses DomynLLMSwarm in
            # runtime-evaluated annotations (``DomynLLMSwarm | None``).
            def __init__(self, cfg):
                super().__init__(cfg, **kwargs)

        monkeypatch.setattr(swarm_driver, "DomynLLMSwarm", ConfiguredFakeSwarm)
        monkeypatch.setattr(
            swarm_driver,
            "SlurmDriver",
            lambda _cfg: SimpleNamespace(get_job_state=lambda _jobid: seq.pop(0) if len(seq) > 1 else seq[0]),
        )
        monkeypatch.setattr(swarm_driver, "_poll_interval", lambda _cfg: 0.01)

    return install


def test_dead_job_raises_and_releases_the_allocation(fake):
    fake(states=["RUNNING", "RUNNING", "OUT_OF_MEMORY"])
    start = time.monotonic()
    with pytest.raises(SwarmStartupError, match="OUT_OF_MEMORY"):
        deploy_swarm(object(), "model")
    assert time.monotonic() - start < 2
    assert FakeSwarm.instances[0].cleanups == 1


def test_pending_jobs_keep_waiting_until_ready(fake):
    fake(states=["PENDING"])
    swarm = FakeSwarm.instances  # filled on deploy
    t = threading.Timer(0.1, lambda: swarm[0].release.set())
    t.start()
    assert deploy_swarm(object()) is swarm[0]
    assert swarm[0].cleanups == 0


def test_successful_startup_returns_the_swarm(fake):
    fake(block=False)
    swarm = deploy_swarm(object())
    assert swarm is FakeSwarm.instances[0]
    assert isinstance(swarm, FakeSwarm)
    assert swarm.cleanups == 0


def test_library_errors_propagate(fake):
    fake(block=False, exc=RuntimeError("Replica array ended in FAILED"))
    with pytest.raises(RuntimeError, match="ended in FAILED"):
        deploy_swarm(object())


def test_handle_without_job_ids_just_waits(fake):
    # e.g. Lepton: nothing to watch, so a terminal state is never consulted
    fake(states=["COMPLETED"], meta={})
    t = threading.Timer(0.1, lambda: FakeSwarm.instances[0].release.set())
    t.start()
    assert deploy_swarm(object()) is FakeSwarm.instances[0]


def test_try_reusing_swarm_and_deploy_new_model(fake, swarms, swarm_handler):
    # Regression: a fresh deploy sets model = the swarm YAML's name, but a reuse kept
    # the config's model, so the rerun wrote to a different results dir and resumed nothing.
    swarms([record("edge-1")], {"edge-1": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("other-model"))
    fake(states=["RUNNING"], block=False)
    assert not handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    # try start the swarm
    handler.up(model_type=SwarmHandler.ModelType.MODEL)
    assert handler.config.model.swarm_name == "sw-test"
    assert handler.config.model.url == f"{NEW_DEPLOYMENT_ENDPOINT}/v1"
    assert (handler.config.model.id, handler.config.model.name) == ("other-model", "other-model")


def test_fresh_deploy_drops_a_stale_model_url(swarm_handler, fake):
    handler = swarm_handler(ModelConfigStub("sw", model_url="https://old-proxy/v1"))
    fake(states=["RUNNING"], block=False)
    handler.up(model_type=SwarmHandler.ModelType.MODEL)
    assert handler.config.model.url == f"{NEW_DEPLOYMENT_ENDPOINT}/v1"


def test_swarm_can_be_shutdown_only_by_same_handler(swarm_handler, fake):
    # Regression: a swarm started by one handler was shut down by another, so the
    # first handler's up() returned a handle whose job ids were gone.
    fake(states=["RUNNING"], block=False)
    handler1 = swarm_handler(ModelConfigStub("sw"))
    handler1.up(model_type=SwarmHandler.ModelType.MODEL)
    # simulate a new run with a new handler
    handler2 = swarm_handler(ModelConfigStub("sw"))
    # the second handler can't shut it down, but the first can
    assert handler2.down(model_type=SwarmHandler.ModelType.MODEL) is False
    # check that we can shut down the swarm with the same handler that started it
    assert handler1.down(model_type=SwarmHandler.ModelType.MODEL) is True
    # check there is no judge handler to shut down
    assert handler1.down(model_type=SwarmHandler.ModelType.JUDGE) is False


def test_both_swarms_can_be_shutdown_by_same_handler(swarm_handler, fake):
    # Regression: a swarm started by one handler was shut down by another, so the
    # first handler's up() returned a handle whose job ids were gone.
    fake(states=["RUNNING"], block=False)
    handler1 = swarm_handler(ModelConfigStub("sw"), judge_model_cfg=ModelConfigStub("judge"))
    handler1.up(model_type=SwarmHandler.ModelType.MODEL)
    handler1.up(model_type=SwarmHandler.ModelType.JUDGE)
    # simulate a new run with a new handler
    handler2 = swarm_handler(ModelConfigStub("sw"), judge_model_cfg=ModelConfigStub("judge"))
    # the second handler can't shut it down, but the first can
    assert handler2.down(model_type=SwarmHandler.ModelType.MODEL) is False
    assert handler2.down(model_type=SwarmHandler.ModelType.JUDGE) is False
    # check that we can shut down the swarm with the same handler that started it
    assert handler1.down(model_type=SwarmHandler.ModelType.MODEL) is True
    assert handler1.down(model_type=SwarmHandler.ModelType.JUDGE) is True
