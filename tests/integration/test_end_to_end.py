"""check, run, and report on the real suites, against a fake endpoint: the scores mean nothing, the wiring is tested."""

import tomllib
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from domyn_eval.cli import cli
from tests.integration.conftest import MODEL, REPO, FakeEndpoint

pytestmark = pytest.mark.integration

SUITES = ("inspect", "bfcl", "lm_eval")

CONFIG = """
[general]
auto_setup = true

[model]
id = "it-model"
path = "{model}"
url = "{url}"

[generation]
reasoning = true
max_tokens = 64

[generation.think]
chat_template_kwargs = {{ enable_thinking = true }}

[inspect]
task = "it"
custom_tasks = [{{ task = "inspect_evals/hello_world" }}]

[bfcl]
fc_model = true
model_class = "Domyn/Domyn-Edge-FC"
test_category = "live_relevance"  # the one category the report shows on its own

[lm_eval]
task = "gsm8k"
"""


def _invoke(*args: str) -> str:
    result = CliRunner().invoke(cli, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_a_run_of_every_suite_is_reported(
    fake_endpoint: FakeEndpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SUITES_BASE_DIR", str(REPO))
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(model=MODEL, url=fake_endpoint.url))
    suites = [arg for suite in SUITES for arg in ("--suite", suite)]

    assert "endpoint reachable" in _invoke("check", str(config), *suites)

    _invoke("run", str(config), *suites, "--run-id", "it", "--max-samples", "2")
    run = tmp_path / "runs" / "it-model" / "it"
    assert set(SUITES) <= set(tomllib.loads((run / "config.toml").read_text()))
    for suite in SUITES:
        assert (run / "logs" / suite / "run.log").is_file()
    assert any((run / "results" / "inspect").glob("*.eval"))
    assert any((run / "results" / "bfcl" / "live_relevance").glob("**/BFCL_v4_live_relevance_score.json"))
    assert any((run / "results" / "lm_eval" / "gsm8k").glob("**/results_*.json"))
    assert {("/v1/chat/completions", MODEL), ("/v1/completions", MODEL)} <= set(fake_endpoint.requests)

    _invoke("report", "it-model")
    report = pd.read_csv(tmp_path / "runs" / "it-model" / "report.csv")
    assert set(report["suite"]) == set(SUITES)
