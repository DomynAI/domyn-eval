"""Tests that domyn-eval reads each suite's outputs with the library version the suite writes them with."""

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (suite checkout under suites/, package both lockfiles must pin to the same version)
SHARED_PACKAGES = [("domyn-inspect", "inspect-ai")]


def _locked_version(lockfile: Path, package: str) -> str:
    """The version of ``package`` in a uv lockfile."""
    packages = tomllib.loads(lockfile.read_text())["package"]
    return next(p["version"] for p in packages if p["name"] == package)


@pytest.mark.parametrize(("suite", "package"), SHARED_PACKAGES)
def test_package_matches_the_suites_version(suite: str, package: str) -> None:
    suite_lock = REPO_ROOT / "suites" / suite / "uv.lock"
    if not suite_lock.is_file():
        pytest.skip(f"suites/{suite} is not checked out")
    assert _locked_version(REPO_ROOT / "uv.lock", package) == _locked_version(suite_lock, package)
