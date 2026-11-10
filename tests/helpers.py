"""Plain test helpers shared across test modules (fixtures live in conftest.py)."""

from domyn_swarm.platform.protocols import ServingPhase

RUNNING, PENDING, INITIALIZING, FAILED, STOPPED = (
    ServingPhase.RUNNING,
    ServingPhase.PENDING,
    ServingPhase.INITIALIZING,
    ServingPhase.FAILED,
    ServingPhase.STOPPED,
)


def record(name, cfg_name="edge", endpoint="http://node1:9000", meta=None, model="/models/edge-sft"):
    return {
        "deployment_name": name,
        "endpoint": endpoint,
        "serving_handle": {"url": endpoint, "meta": meta or {}},
        "cfg": {"name": cfg_name, "model": model},
    }


class ModelConfigStub:
    """
    A subset of a model config that can be used in tests to instantiate a SwarmHandler with the corresponding fixture.
    """

    def __init__(self, model_name="edge", swarm_name=None, model_url=""):
        self.model_name = model_name
        self.swarm_name = swarm_name
        self.model_url = model_url
