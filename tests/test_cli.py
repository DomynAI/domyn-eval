from click.testing import CliRunner

from domyn_eval.cli import cli


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert sorted(cli.commands) == ["check", "report", "run"]
    for command in cli.commands:
        assert command in result.output


def _config(tmp_path, body):
    path = tmp_path / "c.toml"
    path.write_text(body)
    return str(path)


GOOD = '[model]\nid = "m"\nurl = "http://h:1"\n'


def test_check_passes_on_a_good_config(tmp_path):
    result = CliRunner().invoke(cli, ["check", _config(tmp_path, GOOD), "--no-ping"])
    assert result.exit_code == 0, result.output
    assert "model.id = m" in result.output


def test_check_fails_without_a_model(tmp_path):
    result = CliRunner().invoke(cli, ["check", _config(tmp_path, ""), "--no-ping"])
    assert result.exit_code == 1
    assert "model.id is not set" in result.output


def test_check_reports_an_invalid_config(tmp_path):
    result = CliRunner().invoke(cli, ["check", _config(tmp_path, "[nope]\n")])
    assert result.exit_code == 1
    assert "unknown config sections" in result.output


def test_check_selects_the_suites_of_a_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "recipes.toml").write_text('[core]\n[core.suites]\nbfcl = ""\n')
    body = GOOD + '[general]\nrecipes_file = "recipes.toml"\n'
    args = ["check", _config(tmp_path, body), "--no-ping", "--recipe", "core"]
    result = CliRunner().invoke(cli, args)
    assert "bfcl:" in result.output
    assert "inspect:" not in result.output


def test_check_rejects_an_unknown_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "recipes.toml").write_text("")
    args = ["check", _config(tmp_path, GOOD), "--no-ping", "--recipe", "x"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "unknown recipe 'x'" in result.output


def test_check_names_the_setting_when_the_recipes_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["check", _config(tmp_path, GOOD), "--no-ping", "--recipe", "core"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "no recipes file configs/recipes.toml; set general.recipes_file" in result.output
