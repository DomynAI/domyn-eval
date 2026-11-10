from pathlib import Path

import pytest

from domyn_eval.paths import (
    logs_dir,
    resolve_env_path,
    results_dir,
    run_dir,
    runs_root,
    suite_dir,
    suite_python,
)


def test_resolve_env_path_expands_refs(monkeypatch):
    monkeypatch.setenv("_DOMYN_BASE", "/base")
    monkeypatch.setenv("_DOMYN_DIR", "${_DOMYN_BASE}/out")
    assert resolve_env_path("_DOMYN_DIR") == "/base/out"


def test_resolve_env_path_rejects_unset_refs(monkeypatch):
    monkeypatch.delenv("_DOMYN_NOT_SET", raising=False)
    monkeypatch.setenv("_DOMYN_DIR", "${_DOMYN_NOT_SET}/out")
    with pytest.raises(RuntimeError, match=r"unset \$\{_DOMYN_NOT_SET\}"):
        resolve_env_path("_DOMYN_DIR")


def test_resolve_env_path_default(monkeypatch):
    monkeypatch.delenv("_DOMYN_DIR", raising=False)
    assert resolve_env_path("_DOMYN_DIR", "fallback") == "fallback"


def test_suites_resolve_from_suites_base_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SUITES_BASE_DIR", str(tmp_path))
    assert suite_dir("suites/bfcl") == tmp_path / "suites" / "bfcl"
    assert suite_python("suites/bfcl") == tmp_path / "suites" / "bfcl" / ".venv" / "bin" / "python"


def test_runs_root_prefers_runs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    assert runs_root() == tmp_path


def test_runs_root_defaults_to_runs(monkeypatch):
    monkeypatch.delenv("RUNS_DIR", raising=False)
    assert runs_root() == Path("runs")


def test_logs_and_results_share_the_run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    run = tmp_path / "m" / "r1"
    assert run_dir("m", "r1") == run
    assert run_dir("", "r1") == tmp_path / "unknown" / "r1"
    assert logs_dir("m", "r1") == run / "logs"
    assert results_dir("m", "r1", "bfcl") == run / "results" / "bfcl"


def test_create_makes_the_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    path = logs_dir("m", "r1", "bfcl", create=True)
    assert path.is_dir()
    assert not results_dir("m", "r1", "bfcl").exists()
