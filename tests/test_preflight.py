import json
import sys

import httpx
import pytest

from domyn_eval import preflight
from domyn_eval.config import Config, ContextConfig, GenerationConfig, ModelConfig
from domyn_eval.preflight import (
    FAIL,
    OK,
    WARN,
    Preflight,
    check_context_budget,
    check_env,
    failed,
    served_max_model_len,
)

URL = "http://model:8000/v1"
JUDGE_URL = "http://judge:8000/v1"


def _config(*, reasoning=True, **context):
    generation = GenerationConfig(reasoning=reasoning, context=ContextConfig(**context))
    return Config(model=ModelConfig(id="m", path="m", url=URL, generation=generation))


def _levels(checks):
    return [check.level for check in checks]


# ── config ───────────────────────────────────────────────────────────────────


def test_a_good_config_passes():
    assert _levels(Preflight(_config(), []).check_config()) == [OK]


def test_a_missing_model_and_endpoint_fail():
    checks = Preflight(Config(), []).check_config()
    assert _levels(checks) == [FAIL, FAIL]
    assert "model.url is not set" in checks[1].message


def test_a_missing_endpoint_only_warns_when_a_swarm_will_be_deployed():
    config = _config()
    config.model.url = None
    checks = Preflight(config, [], deploying=True).check_config()
    assert _levels(checks) == [OK, WARN]
    assert "once the swarm is deployed" in checks[1].message


def test_an_unknown_suite_fails():
    checks = Preflight(_config(), ["nope"]).check_config()
    assert checks[-1] == (FAIL, "unknown suite 'nope'")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SUITES_BASE_DIR", str(tmp_path))
    return tmp_path


def _suite_checks(root, make):
    config = _config()
    config.suite.bfcl.path = "bfcl"
    make(root / "bfcl")
    return Preflight(config, ["bfcl"]).check_config()[1:]


def test_a_missing_suite_dir_fails(root):
    checks = _suite_checks(root, lambda _path: None)
    assert _levels(checks) == [FAIL]
    assert "does not exist" in checks[0].message


def test_an_empty_suite_dir_hints_at_the_submodule(root):
    checks = _suite_checks(root, lambda path: path.mkdir())
    assert _levels(checks) == [FAIL]
    assert "git submodule update --init" in checks[0].message


def test_a_suite_without_a_venv_warns(root):
    def make(path):
        path.mkdir()
        (path / "pyproject.toml").write_text("")

    assert _levels(_suite_checks(root, make)) == [OK, WARN]


def test_a_set_up_suite_passes(root):
    def make(path):
        (path / ".venv" / "bin").mkdir(parents=True)
        (path / ".venv" / "bin" / "python").write_text("")

    assert _levels(_suite_checks(root, make)) == [OK]


def test_a_suite_without_own_venv_needs_no_checkout():
    checks = Preflight(_config(), ["lm_eval"]).check_config()
    assert _levels(checks) == [OK, OK]
    assert checks[1].message == f"lm_eval: runs in domyn-eval's Python ({sys.executable})"


def test_a_missing_api_key_warns(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert _levels(check_env()) == [WARN]
    monkeypatch.setenv("API_KEY", "k")
    assert _levels(check_env()) == [OK]


# ── tokenizer ────────────────────────────────────────────────────────────────


def _tokenizer_checks(path, suites=()):
    return Preflight(_config(tokenizer_path=path), list(suites)).check_tokenizer()


def test_no_tokenizer_warns():
    config = _config()
    config.model.path = ""
    checks = Preflight(config, []).check_tokenizer()
    assert _levels(checks) == [WARN]
    assert "unset" in checks[0].message


def _save_tokenizer(path):
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "hi": 1}, unk_token="[UNK]"))
    tokenizer.save(str(path / "tokenizer.json"))


def test_a_lone_tokenizer_json_loads(tmp_path):
    _save_tokenizer(tmp_path)
    checks = _tokenizer_checks(str(tmp_path))
    assert _levels(checks) == [OK]
    assert "loads" in checks[0].message


@pytest.mark.parametrize("files", [["tokenizer_config.json"], ["vocab.json"], []])
def test_an_incomplete_tokenizer_fails(tmp_path, files):
    pytest.importorskip("transformers")
    for name in files:
        (tmp_path / name).write_text("{}")
    checks = _tokenizer_checks(str(tmp_path))
    assert _levels(checks) == [FAIL]
    assert "cannot load the tokenizer" in checks[0].message


@pytest.mark.parametrize("path", ["/does/not/exist", "./gone"])
def test_a_stale_local_path_fails(path):
    assert _levels(_tokenizer_checks(path)) == [FAIL]


def test_a_hub_id_warns():
    checks = _tokenizer_checks("org/name")
    assert _levels(checks) == [WARN]
    assert "hub id" in checks[0].message


def _suite_venv(root, *, transformers):
    venv = root / "bfcl" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    (site_packages / ("transformers" if transformers else "httpx")).mkdir(parents=True)


@pytest.mark.parametrize(("transformers", "levels"), [(True, [WARN]), (False, [WARN, WARN])])
def test_a_suite_venv_without_transformers_warns(root, transformers, levels):
    _suite_venv(root, transformers=transformers)
    config = _config(tokenizer_path="org/name")
    config.suite.bfcl.path = "bfcl"
    checks = Preflight(config, ["bfcl"]).check_tokenizer()
    assert _levels(checks) == levels  # the first warns of the hub id
    if not transformers:
        assert "bfcl: no transformers" in checks[1].message


def test_a_suite_without_own_venv_checks_this_python(monkeypatch):
    probed = []
    monkeypatch.setattr(preflight, "_has_transformers", lambda python: probed.append(python) or False)
    checks = _tokenizer_checks("org/name", ["lm_eval"])
    assert probed == [preflight.Path(sys.executable)]
    assert _levels(checks) == [WARN, WARN]


# ── context budget ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("configured", "max_tokens", "served", "level"),
    [
        (0, 0, 32768, OK),
        (16384, 0, 32768, OK),
        (65536, 0, 32768, FAIL),
        (0, 0, None, FAIL),
        (16384, 0, None, OK),
        (16384, 32768, None, WARN),
    ],
)
def test_context_budget(configured, max_tokens, served, level):
    config = _config(max_model_len=configured, max_tokens=max_tokens)
    assert check_context_budget(config, served).level == level


def test_served_max_model_len_reads_the_card():
    assert served_max_model_len({"id": "m", "max_model_len": 4096}) == 4096
    assert served_max_model_len({"id": "m"}) is None
    assert served_max_model_len({"id": "m", "max_model_len": 0}) is None


# ── endpoints ────────────────────────────────────────────────────────────────


_CARDS = ({"id": "m", "max_model_len": 32768},)


def _client(message=None, *, judge_up=True, model_up=True, cards=_CARDS):
    """Fake an endpoint serving ``cards``, answering chats to ``m`` with ``message``."""

    def handler(request):
        if request.url.host == "judge" and not judge_up:
            msg = "refused"
            raise httpx.ConnectError(msg, request=request)
        if request.url.host == "model" and not model_up:
            return httpx.Response(503)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": list(cards)})
        sent = json.loads(request.content)
        assert sent["model"] == "m"
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_endpoint_checks_pass():
    checks = Preflight(_config(), [], _client({"content": "4", "reasoning_content": "2+2"})).check_endpoints()
    assert _levels(checks) == [OK, OK, OK]
    assert "serves m" in checks[0].message
    assert "max_model_len=32768 (endpoint)" in checks[1].message


def test_the_budget_and_probe_use_the_card_of_model_path():
    cards = [{"id": "base", "max_model_len": 4096}, {"id": "m", "max_model_len": 32768}]
    checks = Preflight(_config(), [], _client({"content": "4", "reasoning_content": "2+2"}, cards=cards)).run()
    assert not failed(checks)
    assert any("max_model_len=32768 (endpoint)" in check.message for check in checks)


def test_a_model_path_the_endpoint_does_not_serve_fails():
    checks = Preflight(_config(), [], _client(cards=[{"id": "other"}])).check_endpoints()
    assert _levels(checks) == [OK, FAIL]
    assert "model.path='m' is not served" in checks[1].message


def test_an_unreachable_endpoint_fails():
    checks = Preflight(_config(), [], _client(model_up=False)).check_endpoints()
    assert _levels(checks) == [FAIL]


def test_no_separate_thinking_warns():
    checks = Preflight(_config(), [], _client({"content": "4"})).check_endpoints()
    assert checks[-1].level == WARN
    assert "--reasoning-parser" in checks[-1].message


def test_the_probe_is_skipped_without_reasoning():
    checks = Preflight(_config(reasoning=False), [], _client()).check_endpoints()
    assert _levels(checks) == [OK, OK]


def test_the_probe_sends_the_modes_chat_template_kwargs():
    sent = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        sent.append(json.loads(request.content))
        return httpx.Response(500)

    config = _config(max_model_len=1024)
    config.model.generation.think.chat_template_kwargs = {"enable_thinking": True}
    client = httpx.Client(transport=httpx.MockTransport(handler))
    checks = Preflight(config, [], client).check_endpoints()
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert checks[-1].level == WARN  # the probe failed


@pytest.mark.parametrize(("judge_up", "level"), [(True, OK), (False, WARN)])
def test_a_separate_judge_is_pinged(judge_up, level):
    config = _config(reasoning=False)
    config.judge.url = JUDGE_URL
    checks = Preflight(config, [], _client(judge_up=judge_up)).check_endpoints()
    assert checks[-1].level == level
    assert "judge" in checks[-1].message


def test_no_ping_sends_no_request():
    sent = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(500)))
    checks = Preflight(_config(), [], client).run(ping=False)
    assert sent == []
    assert not any("endpoint" in check.message for check in checks)


def test_run_uses_the_given_client():
    checks = Preflight(_config(reasoning=False), [], _client()).run()
    assert any("endpoint reachable" in check.message for check in checks)
