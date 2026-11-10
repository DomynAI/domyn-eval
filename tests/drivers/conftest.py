from pathlib import Path

import pytest

from domyn_eval.config import Config


@pytest.fixture
def saved_run(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Run ``r1`` of the sample config's model under a tmp ``RUNS_DIR``, with the config saved as the run's own.

    Returns:
        The model and run id, the arguments of a stat driver.
    """
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    path = tmp_path / "test-model" / "r1"
    path.mkdir(parents=True)
    config.save(path / "config.toml")
    return "test-model", "r1"
