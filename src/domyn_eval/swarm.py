# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Model endpoints served by domyn-swarm."""

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import domyn_swarm
import httpx
import yaml
from domyn_swarm import DomynLLMSwarm
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus

from domyn_eval.config import Config, ModelConfig

logger = logging.getLogger(__name__)


ENDPOINT_CHECK_TIMEOUT = 5.0  # seconds


def endpoint_is_live(url: str, timeout: float = ENDPOINT_CHECK_TIMEOUT) -> bool:
    """Whether the OpenAI-compatible server behind ``url`` (a base URL, ``/v1`` optional) is up.

    Tries ``GET /health`` at the server root first: vLLM (and SGLang, TGI, llama.cpp) answer
    it, and vLLM returns 5xx there once its engine has died. A 4xx means the server has no such
    endpoint (or gates it behind auth), so fall back to ``GET /v1/models``, which every
    OpenAI-compatible server implements; a 401/403 there still proves the server is answering.
    """
    root = url.rstrip("/").removesuffix("/v1")

    def get(path: str) -> httpx.Response | None:
        try:
            return httpx.get(root + path, timeout=timeout)
        except httpx.HTTPError as e:
            logger.debug("swarm: liveness check %s%s failed; %s", root, path, e)
            return None

    response = get("/health")
    if response is None or response.is_server_error:
        return False
    if response.is_success:
        return True
    response = get("/v1/models")
    return response is not None and (response.is_success or response.status_code in {401, 403})


def _swarm_status(name: str) -> ServingStatus | None:
    """Live ``ServingStatus`` of a recorded deployment, or None if it can't be probed."""
    try:
        return domyn_swarm.DomynLLMSwarm.from_state(name).serving_status()
    except Exception as e:  # noqa: BLE001 — a stale record must not break the lookup
        logger.debug("swarm: status probe of %s failed; %s", name, e)
        return None


# A swarm in one of these phases is on its way up: wait for it rather than deploy a twin.
STARTING_PHASES = {ServingPhase.PENDING, ServingPhase.INITIALIZING}


def find_live_swarm(
    cfg_name: str,
    *,
    pinned: str | None = None,
    wait: bool = True,
    poll: float = 10,
    timeout: int | None = None,
) -> tuple[str, ServingStatus] | None:
    """``(name, status)`` of a live deployment to reuse, or None to deploy a new swarm.

    Candidates are ``pinned`` (the config's ``swarm_name``) first, then every recorded
    deployment of the swarm config named ``cfg_name``, newest first. A RUNNING one is
    returned; a PENDING/INITIALIZING one is waited on until it serves (or returned
    as-is when ``wait`` is False, e.g. a dry run); dead ones are skipped.

    ``timeout`` (seconds, None = no limit) bounds the time spent waiting on each
    starting swarm; once it has passed, a swarm that is still starting is skipped.
    """
    names = [pinned] if pinned else []
    names += [
        r["deployment_name"] for r in SwarmStateManager.list_all() if (r.get("cfg") or {}).get("name") == cfg_name
    ]
    for name in dict.fromkeys(names):
        status = _swarm_status(name)
        deadline = None if timeout is None else time.monotonic() + timeout
        while wait and status and status.phase in STARTING_PHASES:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                logger.warning("swarm: %s is still %s after %ss; giving up on it", name, status.phase.value, timeout)
                break
            logger.info("swarm: %s is %s; waiting for it to serve", name, status.phase.value)
            time.sleep(poll if remaining is None else min(poll, remaining))
            status = _swarm_status(name)
        # When waiting, a swarm still starting here has timed out: only a dry run reuses one
        if status and (status.phase == ServingPhase.RUNNING or (not wait and status.phase in STARTING_PHASES)):
            logger.info("swarm: reusing %s (%s)", name, status.phase.value)
            return name, status
        logger.info("swarm: %s is %s; skipping it", name, status.phase.value if status else "unreachable")
    return None


# Slurm states a swarm job can be in while still on its way up. Anything else
# (COMPLETED, OUT_OF_MEMORY, PREEMPTED, ...) means it died before serving.
LIVE_SLURM_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
    "UNKNOWN",
}


class SwarmStartupError(RuntimeError):
    """A swarm's Slurm jobs ended before its endpoint came up."""


def _poll_interval(cfg: domyn_swarm.DomynLLMSwarmConfig | None) -> float:
    endpoint = getattr(getattr(cfg, "backend", None), "endpoint", None)
    return float(getattr(endpoint, "poll_interval", None) or 10)


def deploy_swarm(  # noqa: C901 — long but clear
    cfg: domyn_swarm.DomynLLMSwarmConfig, label: str = "model"
) -> DomynLLMSwarm:
    """Bring up a swarm, failing fast if its Slurm jobs die during startup."""
    swarm = DomynLLMSwarm(cfg=cfg)
    result = {}

    def _enter() -> None:
        try:
            swarm.__enter__()
        except BaseException as e:  # noqa: BLE001 — re-raised on the caller's thread
            result["exc"] = e

    worker = threading.Thread(target=_enter, name=f"swarm-up-{label}", daemon=True)
    worker.start()
    driver = SlurmDriver(cfg)
    poll = _poll_interval(cfg)
    try:
        while True:
            worker.join(poll)
            if not worker.is_alive():
                break
            handle = swarm.serving_handle
            meta = (getattr(handle, "meta", None) or {}) if handle else {}
            for kind, jobid in (("replica array", meta.get("jobid")), ("LB job", meta.get("lb_jobid"))):
                if jobid is None:
                    continue
                state = driver.get_job_state(jobid)
                if state not in LIVE_SLURM_STATES:
                    swarm._cleanup_failed_startup()  # noqa: SLF001
                    error_msg = (
                        f"{label} swarm '{swarm.name}': {kind} {jobid} ended in {state} before the endpoint "
                        f"came up. Check the logs under {swarm.swarm_dir}."
                    )
                    raise SwarmStartupError(error_msg)
    except KeyboardInterrupt:
        if worker.is_alive():
            swarm._cleanup_failed_startup()  # noqa: SLF001
        raise
    if "exc" in result:
        raise result["exc"]
    return swarm


def _load_swarm_config(path: str | None) -> domyn_swarm.DomynLLMSwarmConfig | None:
    if not path:
        return None
    return domyn_swarm.DomynLLMSwarmConfig.model_validate(yaml.safe_load(Path(path).read_text()))


class SwarmHandler:
    """Deploys, stops and finds domyn-swarm deployments."""

    fresh_model_swarm: DomynLLMSwarm | None = None  # swarm deployed by this handler for the main model
    fresh_judge_swarm: DomynLLMSwarm | None = None  # swarm deployed by this handler for the judge model

    def __init__(self, config: Config):
        """Build a handler for a config with a [model] and an optional [judge] section.

        Raises:
            ValueError: If the model has no swarm config, url or live swarm.
        """
        self.config = config
        self.model_swarm_config = _load_swarm_config(config.model.swarm_config)
        self.judge_swarm_config = _load_swarm_config(config.judge.swarm_config)
        self._check_section(SwarmHandler.ModelType.MODEL)

    class ModelType(Enum):
        """Enum for model types."""

        MODEL = "model"
        JUDGE = "judge"

    def _get_configs(self, model_type: ModelType) -> tuple[domyn_swarm.DomynLLMSwarmConfig | None, ModelConfig]:
        """Get the swarm configs for the model type."""
        match model_type:
            case SwarmHandler.ModelType.MODEL:
                return self.model_swarm_config, self.config.model
            case SwarmHandler.ModelType.JUDGE:
                return self.judge_swarm_config, self.config.judge

    def _check_section(self, model_type: ModelType) -> None:
        """Raise ValueError unless the section has a swarm config to deploy, a url, or a live swarm."""
        swarm_cfg, section = self._get_configs(model_type)
        if swarm_cfg is not None or section.url:
            return
        if (section.swarm_name or section.id) and find_live_swarm(
            section.id or "", pinned=section.swarm_name, wait=False
        ) is not None:
            return
        msg = (
            f"[{model_type.value}] configuration is missing data: set `swarm_config` to deploy a swarm, "
            "`url` to use an existing endpoint, or `swarm_name`/`id` of a live swarm"
        )
        raise ValueError(msg)

    def up(self, model_type: ModelType) -> None:
        """Deploy a swarm for the given model type (either MODEL or JUDGE).

        Fill up the model configuration with the informaton about the deployed swarm.
        """
        swarm_cfg, config = self._get_configs(model_type)
        if swarm_cfg is None:
            # no swarm to tear up, return
            error_msg = f"No swarm config found for model {model_type.value}"
            raise ValueError(error_msg)
        swarm = deploy_swarm(swarm_cfg, model_type.value)
        swarm_name = swarm.name
        model = swarm.cfg.name
        endpoint = swarm.endpoint
        assert endpoint is not None, f"Swarm '{swarm_name}' has no endpoint"
        model_path = swarm.model
        model_name = model_path.rstrip("/").split("/")[-1]

        logger.info("swarm: %s is %s, serving %s at %s", model_type.value, swarm_name, model_path, endpoint)

        config.swarm_name = swarm_name
        config.id = model
        config.path = model_path
        config.name = model_name
        config.url = endpoint  # if /v1 is missing, it's already added by field validator method in ModelConfig
        match model_type:
            case SwarmHandler.ModelType.MODEL:
                self.fresh_model_swarm = swarm
            case SwarmHandler.ModelType.JUDGE:
                self.fresh_judge_swarm = swarm

    def down(self, model_type: ModelType) -> bool:
        """Stop a swarm.

        Note that this method only tears down a swarm that was deployed by
        this handler, to prevent stopping swarms being potentially used by other clients.
        """

        def shutdown_swarm(swarm: DomynLLMSwarm | None) -> bool:
            if swarm:
                swarm.cleanup()
                return True
            return False

        match model_type:
            case SwarmHandler.ModelType.MODEL:
                return shutdown_swarm(self.fresh_model_swarm)
            case SwarmHandler.ModelType.JUDGE:
                return shutdown_swarm(self.fresh_judge_swarm)

    def deployed_swarms(self) -> dict[ModelType, str]:
        """Names of the swarms this handler deployed, which are the ones to tear down, by model type."""
        swarms = {
            SwarmHandler.ModelType.MODEL: self.fresh_model_swarm,
            SwarmHandler.ModelType.JUDGE: self.fresh_judge_swarm,
        }
        return {model_type: swarm.name for model_type, swarm in swarms.items() if swarm is not None}

    def find(self, model_type: ModelType, poll_interval: float | None = None) -> tuple[str, ServingStatus] | None:
        """Find a running swarm, waiting at most ``timeout`` seconds for starting ones."""
        swarm_cfg, config = self._get_configs(model_type)
        cfg_name = swarm_cfg.name if swarm_cfg else ""
        poll = poll_interval if poll_interval is not None else _poll_interval(swarm_cfg)
        timeout = swarm_cfg.wait_endpoint_s if swarm_cfg else None
        return find_live_swarm(cfg_name, pinned=config.swarm_name, poll=poll, timeout=timeout)

    def ensure_endpoint(self, model_type: ModelType, dry: bool) -> None:
        """Ensure an endpoint for the model `model_type` is up and running, if possible.

        1. check if the model config has a live endpoint, and leave that endpoint
        2. Otherwise, try to reuse and existing running swarm for the model
        3. If there is no swarm available, spawn a new swarm
        """
        if model_type is SwarmHandler.ModelType.JUDGE:
            self._check_section(model_type)  # the model's is checked on construction
        _, config = self._get_configs(model_type)
        if config.url and endpoint_is_live(config.url):
            # there is a live endpoint, so we don't need to use a swarm
            return
        running_swarm = self.try_reuse_swarm(model_type, dry)
        if not running_swarm and not dry:
            self.up(model_type)

    def try_reuse_swarm(self, model_type: ModelType, dry: bool) -> bool:
        """Try to reuse a running swarm, returning True if successful."""
        swarm_cfg, config = self._get_configs(model_type)
        cfg_name = swarm_cfg.name if swarm_cfg else config.id or ""
        timeout = swarm_cfg.wait_endpoint_s if swarm_cfg else None
        found = find_live_swarm(
            cfg_name, pinned=config.swarm_name, wait=not dry, poll=_poll_interval(swarm_cfg), timeout=timeout
        )
        if found is None:
            config.swarm_name = None  # replaced by the new deployment's name
            return False
        name, status = found
        if status.url is not None:
            self._fill_from_swarm(name, model_type=model_type, endpoint=status.url)
        else:  # dry run on a swarm still starting: no endpoint yet, nothing will be sent
            config.swarm_name = name
            config.url = None
        # A fresh deploy names the run after the swarm YAML; do the
        # same here, or a rerun writes to another results dir
        if swarm_cfg:
            config.id = cfg_name
        return True

    def _fill_from_swarm(self, name: str, *, model_type: ModelType, endpoint: str | None = None) -> str | None:
        """Fill a ``[model]``/``[judge]`` section in place from a deployed swarm.

        ``name`` is one exact deployment name, as ``domyn-swarm swarm list`` prints it.
        ``endpoint`` (a live status URL) wins over the recorded one, which is missing when
        the process that deployed the swarm died before it came up.

        Returns the resolved deployment name, or None if the config was left alone.
        """
        name = (name or "").strip()
        if not name:
            return None
        _, section = self._get_configs(model_type)
        # we keep the existing config only if:
        # - there is already a configured model endpoint
        # - there is no input endpoint that overrides the configured endpoint
        # - the configured endpoint refers to a swarm with name `name`
        if not endpoint and urlparse(section.url or "").hostname and section.swarm_name == name:
            logger.debug("swarm: %s.swarm_name = %s, but its url is set; keeping the url", model_type.value, name)
            return None

        records = SwarmStateManager.list_all()
        matches = [r for r in records if r.get("deployment_name") == name]
        if len(matches) == 0:
            error_msg = f"no swarm deployment named '{name}' in the domyn-swarm state DB"
            raise RuntimeError(error_msg)
        if len(matches) > 1:
            # The name is supposed to identify one deployment, so this is a
            # corrupt state DB — guessing would pick an endpoint at random.
            error_msg = f"{len(matches)} deployments named '{name}' in the domyn-swarm state DB"
            raise RuntimeError(error_msg)
        record = matches[0]
        handle = record.get("serving_handle") or {}
        endpoint = endpoint or record.get("endpoint") or handle.get("url") or ""
        parsed = urlparse(endpoint)
        if not parsed.hostname:
            error_msg = (
                f"swarm '{name}' has no usable recorded endpoint ({endpoint!r})— it may never have finished coming up"
            )
            raise RuntimeError(error_msg)

        cfg = record.get("cfg") or {}
        base_name = cfg.get("name") or name  # swarm cfg name, e.g. "domyn-edge-sft"
        model_path = cfg.get("model") or ""  # what the replicas serve
        model_name = model_path.rstrip("/").split("/")[-1]

        section.swarm_name = name
        section.path = model_path
        section.url = endpoint
        section.id = base_name
        section.name = model_name

        logger.info(
            "swarm: %s is %s, serving %s at %s",
            model_type.value,
            name,
            model_path or "an unknown model",
            section.url,
        )
        return name
