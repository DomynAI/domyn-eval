"""Tests for ``--swarm`` reusing a live deployment instead of deploying a twin.

``find_live_swarm`` matches state-DB records on their swarm YAML ``cfg.name`` and
probes each with ``DomynLLMSwarm.from_state(name).serving_status()``, so every test
here stubs ``SwarmStateManager.list_all`` with hand-built records and
``DomynLLMSwarm`` with a scripted sequence of phases per deployment.
"""

from types import SimpleNamespace

import httpx
import pytest

import domyn_eval.swarm as swarm_driver
from domyn_eval.config import Config, ModelConfig
from domyn_eval.swarm import SwarmHandler, endpoint_is_live, find_live_swarm
from tests.helpers import FAILED, INITIALIZING, PENDING, RUNNING, STOPPED, ModelConfigStub, record


def found_name(found):
    return found[0] if found else None


# ── find_live_swarm ───────────────────────────────────────────────────────────


def test_running_match_is_reused_and_other_configs_ignored(swarms, swarm_handler):
    swarms(
        [record("other-1", cfg_name="other"), record("edge-1")],
        {"other-1": [RUNNING], "edge-1": [RUNNING]},
    )
    handler = swarm_handler(ModelConfigStub("edge"))
    name, status = handler.find(SwarmHandler.ModelType.MODEL)
    assert name == "edge-1"
    assert status.url == "http://node1:9000"


def test_newest_running_match_wins(swarms, swarm_handler):
    probes = swarms(
        [record("edge-new"), record("edge-old")],
        {"edge-new": [RUNNING], "edge-old": [RUNNING]},
    )
    handler = swarm_handler(ModelConfigStub("edge"))
    assert found_name(handler.find(SwarmHandler.ModelType.MODEL)) == "edge-new"
    assert probes == ["edge-new"]


def test_dead_and_unloadable_records_are_skipped(swarms, swarm_handler):
    swarms(
        [record("edge-3"), record("edge-2"), record("edge-1")],
        {"edge-3": [FAILED], "edge-2": [STOPPED]},  # edge-1 cannot be loaded
    )
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.find(SwarmHandler.ModelType.MODEL) is None


def test_starting_swarm_is_waited_on_until_it_serves(swarms, swarm_handler):
    probes = swarms([record("edge-1")], {"edge-1": [PENDING, INITIALIZING, RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert found_name(handler.find(SwarmHandler.ModelType.MODEL, poll_interval=0)) == "edge-1"
    assert probes == ["edge-1"] * 3


def test_starting_swarm_that_fails_falls_through(swarms, swarm_handler):
    swarms(
        [record("edge-2"), record("edge-1")],
        {"edge-2": [PENDING, FAILED], "edge-1": [STOPPED]},
    )
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.find(SwarmHandler.ModelType.MODEL, poll_interval=0) is None


def test_starting_swarm_is_skipped_after_the_timeout(swarms):
    probes = swarms([record("edge-1")], {"edge-1": [PENDING]})  # never comes up
    assert find_live_swarm("edge", poll=0.01, timeout=1) is None
    assert len(probes) > 1  # it did wait before giving up


def test_timeout_falls_through_to_a_running_swarm(swarms):
    swarms(
        [record("edge-2"), record("edge-1")],
        {"edge-2": [INITIALIZING], "edge-1": [RUNNING]},
    )
    assert found_name(find_live_swarm("edge", poll=0, timeout=0)) == "edge-1"


def test_no_wait_returns_a_starting_swarm_without_polling(swarms):
    probes = swarms([record("edge-1")], {"edge-1": [PENDING, RUNNING]})
    found = find_live_swarm("edge", wait=False)
    assert found is not None
    name, status = found
    assert (name, status.phase) == ("edge-1", PENDING)
    assert probes == ["edge-1"]


def test_live_pinned_name_wins_over_the_name_match(swarms, swarm_handler):
    swarms(
        [record("edge-1"), record("pinned-1", cfg_name="renamed")],
        {"edge-1": [RUNNING], "pinned-1": [RUNNING]},
    )
    handler = swarm_handler(ModelConfigStub("edge", swarm_name="pinned-1"))
    assert found_name(handler.find(SwarmHandler.ModelType.MODEL)) == "pinned-1"


def test_dead_pinned_name_falls_back_to_the_name_match(swarms, swarm_handler):
    swarms(
        [record("edge-1"), record("pinned-1")],
        {"edge-1": [RUNNING], "pinned-1": [FAILED]},
    )
    handler = swarm_handler(ModelConfigStub("edge", swarm_name="pinned-1"))
    assert found_name(handler.find(SwarmHandler.ModelType.MODEL)) == "edge-1"


# ── port resolution ───────────────────────────────────────────────────────────


def test_port_in_the_url_wins_over_meta(swarms, swarm_handler):
    swarms([record("sw", endpoint="http://node1:8000", meta={"port": 9000})], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.endpoint_port == 8000


# ── URL normalization ─────────────────────────────────────────────────────────


def test_endpoint_already_ending_in_v1_is_not_double_appended(swarms, swarm_handler):
    swarms([record("sw", endpoint="http://node1:8000/v1")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.url == "http://node1:8000/v1"


def test_endpoint_without_v1_gets_one(swarms, swarm_handler):
    swarms([record("sw", endpoint="http://node1:8000")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.url == "http://node1:8000/v1"


def test_compute_node_endpoint_resolves_to_node_and_port(swarms, swarm_handler):
    """A bare compute-node endpoint is kept as recorded, with ``/v1`` appended."""
    swarms([record("sw", endpoint="http://node1234:8000")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.endpoint_node == "node1234"
    assert handler.config.model.url == "http://node1234:8000/v1"


def test_fills_identity_fields_and_returns_the_name(swarms, swarm_handler):
    swarms([record("sw", model="/models/edge-sft")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.swarm_name == "sw"
    assert handler.config.model.path == "/models/edge-sft"
    assert handler.config.model.id == "edge"  # the swarm cfg name
    assert handler.config.model.name == "edge-sft"  # basename of the served path


def test_blank_name_is_a_no_op(swarms, swarm_handler):
    swarms([], {})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert not handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)


def test_swarm_is_not_found(swarms, swarm_handler):
    swarms([record("edge", cfg_name="other")], {"edge": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    assert not handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)


def test_swarm_is_found_by_swarm_name(swarms, swarm_handler):
    swarms([record("edge", cfg_name="other")], {"edge": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("sw", swarm_name="edge"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)


def test_duplicate_names_raise_rather_than_guess(swarms, swarm_handler):
    swarms([record("sw"), record("sw")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge", swarm_name="sw"))
    with pytest.raises(RuntimeError, match="2 deployments named"):
        handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)


def test_endpoint_without_a_hostname_raises(swarms, swarm_handler):
    """A swarm that never finished coming up has no usable endpoint."""
    swarms([record("sw", endpoint="")], {"sw": [RUNNING]})
    handler = swarm_handler(ModelConfigStub("edge"))
    with pytest.raises(RuntimeError, match="no usable recorded endpoint"):
        handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)


def test_pinned_name_with_static_node_reuses_the_live_swarm(swarms, swarm_handler):
    swarms(
        [record("edge-1", endpoint="http://node0954:9000")],
        {"edge-1": [RUNNING]},
    )
    handler = swarm_handler(ModelConfigStub("edge", swarm_name="edge-1"))
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.swarm_name == "edge-1"
    assert (handler.config.model.endpoint_node, handler.config.model.endpoint_port) == ("node0954", 9000)


def test_nothing_live_deploys_and_drops_the_stale_name(swarms, swarm_handler):
    swarms([record("edge-1")], {"edge-1": [FAILED]})
    handler = swarm_handler(ModelConfigStub("edge", swarm_name="edge-1"))
    assert not handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.swarm_name is None


# ── model config without a swarm_config ───────────────────────────────────────


def handler_without_swarm_config(**model_fields):
    return SwarmHandler(Config(model=ModelConfig(**model_fields)))


def test_no_swarm_config_url_or_swarm_raises(swarms):
    swarms([], {})
    with pytest.raises(ValueError, match="missing data"):
        handler_without_swarm_config()


def test_url_is_enough_and_skips_the_swarm_lookup(swarms, endpoint):
    probes = swarms([record("edge-1")], {"edge-1": [RUNNING]})
    requested = endpoint(health=200)
    handler = handler_without_swarm_config(url="http://node1:9000/v1")
    assert handler.model_swarm_config is None
    assert requested == []  # liveness is checked by ensure_endpoint, not on construction
    assert probes == []


def test_url_is_not_health_checked_when_a_swarm_config_is_set(swarm_handler, endpoint):
    requested = endpoint(health=503)
    swarm_handler(ModelConfigStub("edge", model_url="http://stale:9000/v1"))
    assert requested == []


def test_swarm_name_of_a_live_swarm_is_enough(swarms):
    swarms([record("edge-1")], {"edge-1": [RUNNING]})
    handler = handler_without_swarm_config(swarm_name="edge-1")
    assert handler.model_swarm_config is None
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.id == "edge"
    assert handler.config.model.swarm_name == "edge-1"


def test_id_matching_a_live_swarm_config_is_enough(swarms):
    swarms([record("edge-1", cfg_name="edge")], {"edge-1": [RUNNING]})
    handler = handler_without_swarm_config(id="edge")
    assert handler.try_reuse_swarm(model_type=SwarmHandler.ModelType.MODEL, dry=False)
    assert handler.config.model.id == "edge"


def test_swarm_name_of_a_dead_swarm_raises(swarms):
    swarms([record("edge-1")], {"edge-1": [FAILED]})
    with pytest.raises(ValueError, match="missing data"):
        handler_without_swarm_config(swarm_name="edge-1")


def test_id_without_a_matching_swarm_raises(swarms):
    swarms([record("other-1", cfg_name="other")], {"other-1": [RUNNING]})
    with pytest.raises(ValueError, match="missing data"):
        handler_without_swarm_config(id="edge")


def test_starting_swarm_is_accepted_without_waiting(swarms):
    probes = swarms([record("edge-1")], {"edge-1": [PENDING, RUNNING]})
    handler_without_swarm_config(swarm_name="edge-1")
    assert probes == ["edge-1"]  # probed once, not polled until RUNNING


# ── ensure_endpoint ───────────────────────────────────────────────────────────


@pytest.fixture
def endpoint(monkeypatch):
    """Stub ``httpx.get`` with the answers of ``/health`` and ``/v1/models``.

    Each answer is a status code or an exception to raise. Returns the requested URLs.
    """
    requested = []

    def install(health, models=404):
        def get(url, **_kwargs):
            requested.append(url)
            answer = health if url.endswith("/health") else models
            if isinstance(answer, Exception):
                raise answer
            return httpx.Response(answer, request=httpx.Request("GET", url))

        monkeypatch.setattr(swarm_driver.httpx, "get", get)
        return requested

    return install


@pytest.fixture
def deployed(monkeypatch):
    """Stub ``SwarmHandler.up`` on a handler; returns the model types it deployed."""
    calls = []

    def install(handler):
        monkeypatch.setattr(handler, "up", calls.append)
        return calls

    return install


MODEL = SwarmHandler.ModelType.MODEL


def test_live_url_is_kept_without_looking_for_a_swarm(swarms, endpoint, swarm_handler, deployed):
    probes = swarms([record("edge-1")], {"edge-1": [RUNNING]})
    requested = endpoint(health=200)
    handler = swarm_handler(ModelConfigStub("edge", model_url="http://node1:9000/v1"))
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=False)
    assert handler.config.model.url == "http://node1:9000/v1"
    assert requested == ["http://node1:9000/health"]
    assert probes == []
    assert ups == []


def test_dead_url_falls_back_to_a_live_swarm(swarms, endpoint, swarm_handler, deployed):
    swarms([record("edge-1", endpoint="http://node2:9000")], {"edge-1": [RUNNING]})
    endpoint(health=httpx.ConnectError("connection refused"))
    handler = swarm_handler(ModelConfigStub("edge", model_url="http://node1:9000/v1"))
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=False)
    assert handler.config.model.url == "http://node2:9000/v1"
    assert handler.config.model.swarm_name == "edge-1"
    assert ups == []


@pytest.mark.parametrize("answer", [503, httpx.ConnectError("connection refused")])
def test_url_failing_the_health_check_deploys_without_a_live_swarm(swarms, endpoint, swarm_handler, deployed, answer):
    swarms([record("edge-1")], {"edge-1": [FAILED]})
    endpoint(health=answer)
    handler = swarm_handler(ModelConfigStub("edge", model_url="http://node1:9000/v1"))
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=False)
    assert ups == [MODEL]


def test_dead_url_without_a_swarm_config_raises(swarms, endpoint):
    swarms([], {})
    endpoint(health=503)
    handler = handler_without_swarm_config(url="http://node1:9000/v1")
    with pytest.raises(ValueError, match="No swarm config found for model model"):
        handler.ensure_endpoint(MODEL, dry=False)


def test_dry_run_never_deploys(swarms, endpoint, swarm_handler, deployed):
    swarms([record("edge-1")], {"edge-1": [FAILED]})
    endpoint(health=503)
    handler = swarm_handler(ModelConfigStub("edge", model_url="http://node1:9000/v1"))
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=True)
    assert ups == []


def test_dry_run_reuses_a_starting_swarm_without_waiting(swarms, endpoint, swarm_handler, deployed):
    probes = swarms([record("edge-1")], {"edge-1": [PENDING, RUNNING]})
    endpoint(health=503)
    handler = swarm_handler(ModelConfigStub("edge", model_url="http://node1:9000/v1"))
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=True)
    assert probes == ["edge-1"]
    assert handler.config.model.swarm_name == "edge-1"
    assert handler.config.model.url is None  # still starting: no endpoint yet
    assert ups == []


@pytest.mark.parametrize("url", [None, ""])
def test_missing_url_goes_straight_to_the_swarm_lookup(swarms, endpoint, swarm_handler, deployed, url):
    swarms([record("edge-1")], {"edge-1": [RUNNING]})
    requested = endpoint(health=200)
    handler = swarm_handler(ModelConfigStub("edge", model_url=url))
    assert handler.config.model.url is None  # an empty url is normalized to None
    ups = deployed(handler)
    handler.ensure_endpoint(MODEL, dry=False)
    assert requested == []
    assert handler.config.model.swarm_name == "edge-1"
    assert ups == []


JUDGE = SwarmHandler.ModelType.JUDGE


def test_judge_without_a_swarm_config_url_or_swarm_raises(swarms, swarm_handler):
    swarms([], {})
    handler = swarm_handler(ModelConfigStub("edge"))
    handler.config.judge = ModelConfig()
    with pytest.raises(ValueError, match=r"\[judge\] configuration is missing data"):
        handler.ensure_endpoint(JUDGE, dry=False)


def test_judge_url_is_enough(swarms, endpoint, swarm_handler, deployed):
    swarms([], {})
    endpoint(health=200)
    handler = swarm_handler(ModelConfigStub("edge"))
    handler.config.judge = ModelConfig(url="http://judge:9000/v1")
    ups = deployed(handler)
    handler.ensure_endpoint(JUDGE, dry=False)
    assert handler.config.judge.url == "http://judge:9000/v1"
    assert ups == []


def test_deployed_swarms_names_only_the_swarms_the_handler_deployed(swarm_handler, monkeypatch):
    deployed = SimpleNamespace(
        name="judge-1", cfg=SimpleNamespace(name="judge"), endpoint="http://judge-node:8000", model="/models/judge"
    )
    monkeypatch.setattr(swarm_driver, "deploy_swarm", lambda *_: deployed)
    handler = swarm_handler(ModelConfigStub("edge"), ModelConfigStub("judge"))
    assert handler.deployed_swarms() == {}

    handler.up(JUDGE)

    assert handler.deployed_swarms() == {JUDGE: "judge-1"}
    assert handler.config.judge.url == "http://judge-node:8000/v1"


# ── endpoint_is_live ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "health_url"),
    [
        ("http://node1:9000/v1", "http://node1:9000/health"),
        ("http://node1:9000/v1/", "http://node1:9000/health"),
        ("http://node1:9000", "http://node1:9000/health"),
        ("https://proxy/vllm/v1", "https://proxy/vllm/health"),
    ],
)
def test_health_check_targets_the_server_root(endpoint, url, health_url):
    requested = endpoint(health=200)
    assert endpoint_is_live(url)
    assert requested == [health_url]


@pytest.mark.parametrize(
    ("health", "models", "live", "requests"),
    [
        (200, 500, True, 1),  # vLLM-like: /health is enough
        (503, 200, False, 1),  # vLLM with a dead engine: no fallback
        (httpx.ConnectError("refused"), 200, False, 1),  # nothing listening
        (404, 200, True, 2),  # no /health (Ollama, OpenAI): /v1/models answers
        (404, 401, True, 2),  # auth-gated endpoint: up, just needs a key
        (401, 403, True, 2),  # proxy gating every path behind auth
        (404, 404, False, 2),  # not an OpenAI-compatible server
        (404, 502, False, 2),  # proxy whose upstream is down
        (404, httpx.ReadTimeout("timeout"), False, 2),
    ],
)
def test_liveness_falls_back_to_the_models_endpoint(endpoint, health, models, live, requests):
    requested = endpoint(health=health, models=models)
    assert endpoint_is_live("https://proxy/vllm/v1") is live
    assert requested == ["https://proxy/vllm/health", "https://proxy/vllm/v1/models"][:requests]
