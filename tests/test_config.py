import copy
import dataclasses
import tomllib

import pydantic
import pytest
from pydantic.dataclasses import dataclass

from domyn_eval import config as config_module
from domyn_eval.config import (
    RecipeConfig,
    SuiteDrivers,
    expand_config,
    expand_env,
    load_config,
    load_recipes,
    register_suite,
    registered_suites,
)

# ── loading ──────────────────────────────────────────────────────────────────


def test_load_sample_config(config):
    assert config.general.log_level == "DEBUG"
    assert config.general.file.endswith("tests/fixtures/sample_config.toml")
    assert config.model.id == "test-model"
    assert config.slurm.walltime == "02:00:00"
    assert config.suite.bfcl.test_category == "single_turn"
    assert config.suite.inspect.max_samples == 10
    assert config.suite.lm_eval.task is None  # not in the file: defaults


def test_builtin_suites_are_registered():
    assert registered_suites() == ("bfcl", "inspect", "lm_eval")


def test_slurm_defaults_leave_every_directive_out():
    slurm = config_module.SlurmConfig()
    assert all(value is None for value in dataclasses.asdict(slurm).values())


def _load(tmp_path, body):
    path = tmp_path / "c.toml"
    path.write_text(body)
    return load_config(path)


def test_url_gets_v1_appended(config):
    assert config.judge.url == "http://localhost:8080/v1"


def test_url_is_normalized_on_assignment(config):
    config.judge.url = "http://h:2/"
    assert config.judge.url == "http://h:2/v1"


def test_url_stays_unset_without_an_endpoint(tmp_path):
    assert _load(tmp_path, "").judge.url is None


@pytest.mark.parametrize("url", ["http://h:1/v1", "http://h:1/v1/", "http://h:1", "http://h:1/"])
def test_url_is_normalized(tmp_path, url):
    config = _load(tmp_path, f'[model]\nurl = "{url}"\n')
    assert config.model.url == "http://h:1/v1"


def test_unknown_section_raises(tmp_path):
    with pytest.raises(ValueError, match=r"unknown config sections \['nope'\]"):
        _load(tmp_path, "[nope]\nx = 1\n")


# ── generation: think / nothink ──────────────────────────────────────────────


def test_reasoning_selects_think_params(config):
    assert config.generation.mode is config.generation.think
    assert config.generation.mode.temperature == pytest.approx(0.6)
    assert config.generation.mode.min_completion_tokens == 16384


def test_reasoning_selects_nothink_params(tmp_path):
    config = _load(
        tmp_path,
        """
[generation]
reasoning = false
[generation.think]
temperature = 0.6
[generation.nothink]
temperature = 0.7
""",
    )
    assert config.generation.mode.temperature == pytest.approx(0.7)


def test_missing_mode_blocks_keep_the_generation_section(tmp_path):
    """A [generation] without think/nothink used to reset to defaults."""
    config = _load(
        tmp_path,
        """
[generation]
max_model_len = 1234
request_timeout = 60
[generation.think]
temperature = 0.6
""",
    )
    assert config.generation.max_model_len == 1234
    assert config.generation.request_timeout == 60
    assert config.generation.mode.temperature == pytest.approx(0.6)


@pytest.mark.parametrize(
    "body",
    ["", "[generation.think]\ntemperature = 0.6\n"],
    ids=["no-file-value", "think-block-without-it"],
)
def test_min_completion_tokens_defaults_to_none(tmp_path, body):
    generation = _load(tmp_path, body).generation
    assert generation.think.min_completion_tokens is None
    assert generation.nothink.min_completion_tokens is None


def test_generation_defaults_match_loading_an_empty_file(tmp_path):
    assert _load(tmp_path, "").generation == config_module.GenerationConfig()


@pytest.mark.parametrize("key", ["temperature", "min_completion_tokens"])
def test_mode_params_outside_a_mode_block_raise(tmp_path, key):
    with pytest.raises(pydantic.ValidationError, match=key):
        _load(tmp_path, f"[generation]\n{key} = 1\n")


# ── strict sections ──────────────────────────────────────────────────────────


def test_unknown_field_assignment_raises(config):
    with pytest.raises(pydantic.ValidationError):
        config.suite.bfcl.test_categoryyy = "core"


def test_unknown_field_in_a_section_raises(tmp_path):
    with pytest.raises(pydantic.ValidationError):
        _load(tmp_path, "[inspect]\nnot_a_field = 1\n")


def test_wrong_type_assignment_raises(config):
    with pytest.raises(pydantic.ValidationError):
        config.suite.bfcl.fc_model = ["not", "a", "bool"]


def test_strings_are_coerced_on_assignment(config):
    config.suite.bfcl.fc_model = "true"
    config.suite.inspect.max_samples = "1000"
    assert config.suite.bfcl.fc_model is True
    assert config.suite.inspect.max_samples == 1000


def test_deepcopy_is_independent(config):
    new = copy.deepcopy(config)
    new.suite.bfcl.test_category = "core"
    assert config.suite.bfcl.test_category == "single_turn"


def test_unknown_suite_attribute_raises(config):
    with pytest.raises(AttributeError):
        _ = config.suite.nope


# ── suite registry ───────────────────────────────────────────────────────────


@dataclass(config=config_module._STRICT)
class ExtraConfig:
    path: str = "suites/extra"
    n_threads: int = 4


@dataclass(config=config_module._STRICT)
class RequiredFieldConfig:
    path: str


UNUSED_DRIVERS = SuiteDrivers(run="unused:Run", report="unused:Report")


@pytest.fixture
def extra_suite(monkeypatch):
    for registry in ("_SUITES", "_DRIVERS", "_ENTRY_CHECKS"):
        monkeypatch.setattr(config_module, registry, dict(getattr(config_module, registry)))
    register_suite("extra", ExtraConfig, drivers=UNUSED_DRIVERS)


@pytest.mark.usefixtures("extra_suite")
def test_registered_suite_section_loads(tmp_path):
    config = _load(tmp_path, "[extra]\nn_threads = 8\n")
    assert registered_suites()[-1] == "extra"
    assert config.suite.extra.n_threads == 8


@pytest.mark.parametrize("name", ["bfcl", "generation"])
def test_register_suite_rejects_taken_names(name):
    with pytest.raises(ValueError, match="reserved or already registered"):
        register_suite(name, ExtraConfig, drivers=UNUSED_DRIVERS)


def test_register_suite_rejects_non_dataclasses():
    with pytest.raises(TypeError, match="not a dataclass"):
        register_suite("plain", dict, drivers=UNUSED_DRIVERS)


def test_register_suite_rejects_fields_without_defaults():
    with pytest.raises(TypeError, match=r"without a default: \['path'\]"):
        register_suite("required", RequiredFieldConfig, drivers=UNUSED_DRIVERS)


# ── ${...} expansion ─────────────────────────────────────────────────────────


def test_expand_env_os_var(monkeypatch):
    monkeypatch.setenv("_DOMYN_TEST_VAR", "hello")
    assert expand_env("${_DOMYN_TEST_VAR}/world", {}) == "hello/world"


def test_expand_env_context_var():
    result = expand_env("${_TOML_ONLY_KEY}/data", {"_TOML_ONLY_KEY": "/base"})
    assert result == "/base/data"


def test_expand_env_passes_non_strings_through():
    assert expand_env(42, {}) == 42


def test_expand_config_env_and_siblings(monkeypatch):
    monkeypatch.setenv("_DOMYN_EXPAND_ROOT", "/base")
    doc = {
        "model": {
            "root": "${_DOMYN_EXPAND_ROOT}",
            "path": "${root}/w",
            "n": 3,
        },
        "generation": {
            "dir": "${_DOMYN_EXPAND_ROOT}/g",
            "think": {"sub": "${dir}/t"},
        },
        "tasks": [{"t": "${_DOMYN_EXPAND_ROOT}/a"}],
    }
    out = expand_config(doc)
    assert out["model"]["path"] == "/base/w"
    assert out["generation"]["think"]["sub"] == "/base/g/t"
    assert out["tasks"][0]["t"] == "/base/a"
    assert out["model"]["n"] == 3


def test_expand_config_keeps_unresolved_refs_and_templates(monkeypatch):
    monkeypatch.delenv("_DOMYN_NOT_SET", raising=False)
    out = expand_config({"s": {"a": "${_DOMYN_NOT_SET}/y", "b": "{x}"}})
    assert out["s"] == {"a": "${_DOMYN_NOT_SET}/y", "b": "{x}"}


def test_load_config_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("_DOMYN_EXPAND_ROOT", "/base")
    config = _load(
        tmp_path,
        """
[model]
path = "${_DOMYN_EXPAND_ROOT}/model"
[generation]
tokenizer_path = "${path}/tok"
""",
    )
    assert config.model.path == "/base/model"
    assert config.generation.tokenizer_path == "${path}/tok"


# ── snapshots ────────────────────────────────────────────────────────────────


def _snapshot(config, tmp_path):
    path = tmp_path / "run.config.toml"
    config.save(path)
    return path


def test_snapshot_round_trips(config, tmp_path):
    reloaded = load_config(_snapshot(config, tmp_path))
    assert reloaded.model.url == config.model.url
    assert reloaded.generation == config.generation
    assert reloaded.suite.inspect == config.suite.inspect
    assert reloaded.suite.bfcl == config.suite.bfcl


def test_snapshot_persists_runtime_changes(config, tmp_path):
    config.suite.inspect.task = "full"
    config.generation.tokenizer_path = "/changed"
    reloaded = load_config(_snapshot(config, tmp_path))
    assert reloaded.suite.inspect.task == "full"
    assert reloaded.generation.tokenizer_path == "/changed"


def test_snapshot_leaves_out_runtime_only_fields(config, tmp_path):
    config.general.run_id = "run-abc123"
    config.model.endpoint_max_model_len = 4096
    raw = tomllib.loads(_snapshot(config, tmp_path).read_text())
    assert "run_id" not in raw["general"]
    assert "file" not in raw["general"]
    assert "endpoint_max_model_len" not in raw["model"]
    assert raw["generation"]["think"]["temperature"] == pytest.approx(0.6)


def test_snapshot_writes_used_suites_in_full_at_the_top_level(config, tmp_path):
    raw = tomllib.loads(_snapshot(config, tmp_path).read_text())
    assert raw["inspect"]["task"] == "core"
    assert raw["inspect"]["path"] == "suites/domyn-inspect"  # a default
    assert raw["bfcl"]["test_category"] == "single_turn"
    assert "lm_eval" not in raw  # not in the loaded file
    assert "suite" not in raw


def test_snapshot_writes_a_suite_marked_used(config, tmp_path):
    config.suite.use("lm_eval")
    raw = tomllib.loads(_snapshot(config, tmp_path).read_text())
    assert raw["lm_eval"]["allow_unsafe_code"] is False  # a default


def test_using_an_unknown_suite_raises(config):
    with pytest.raises(ValueError, match="unknown suite 'nope'"):
        config.suite.use("nope")


def test_an_override_replaces_fields_of_a_suite_section_and_uses_it(config):
    config.suite.override("lm_eval", {"num_threads": 8})
    assert config.suite.lm_eval.num_threads == 8
    assert "lm_eval" in config.suite.used()


def test_an_invalid_override_raises(config):
    with pytest.raises(ValueError, match="num_threads"):
        config.suite.override("lm_eval", {"num_threads": "many"})


def test_snapshot_drops_a_value_reset_to_its_default(tmp_path):
    """Suite sections used to be merged, so a value reset to its default kept the file's."""
    path = tmp_path / "c.toml"
    path.write_text("[bfcl]\nnum_threads = 8\n")
    config = load_config(path)
    config.suite.bfcl.num_threads = None
    config.save(path)
    assert load_config(path).suite.bfcl.num_threads is None


def test_snapshot_drops_unused_suites_from_an_existing_file(config, tmp_path):
    path = tmp_path / "run.config.toml"
    path.write_text('[lm_eval]\ntask = "stale"\n')
    config.save(path)
    assert "lm_eval" not in tomllib.loads(path.read_text())


def test_snapshot_merges_into_an_existing_file(config, tmp_path):
    path = tmp_path / "run.config.toml"
    path.write_text('[note]\ntext = "kept"\n')
    config.save(path)
    assert tomllib.loads(path.read_text())["note"] == {"text": "kept"}


# ── recipes ──────────────────────────────────────────────────────────────────


def _recipes(tmp_path, body):
    path = tmp_path / "r.toml"
    path.write_text(body)
    return load_recipes(path)


def test_recipe_name_defaults_to_the_table_key(tmp_path):
    recipes = _recipes(tmp_path, '[solo]\n[solo.suites]\nbfcl = ""\n')
    assert recipes["solo"].name == "solo"


def test_recipe_entries_list_each_invocation_with_its_overrides(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[mix]
[mix.suites]
bfcl = ["test_category=simple", { test_category = "live" }]
lm_eval = ""
""",
    )
    assert recipes["mix"].entries() == [
        ("bfcl", {"test_category": "simple"}),
        ("bfcl", {"test_category": "live"}),
        ("lm_eval", {}),
    ]


def test_recipe_accepts_dict_shaped_suite_entries(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[core]
[core.suites]
inspect = { task = "core", custom_tasks = [{ task = "t/a" }] }
""",
    )
    assert recipes["core"].suites["inspect"] == {
        "task": "core",
        "custom_tasks": [{"task": "t/a"}],
    }


def test_extend_inherits_parent_suites(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[parent]
[parent.suites]
inspect = { task = "p", custom_tasks = [{ task = "t/a" }] }

[child]
extend = "parent"
""",
    )
    assert recipes["child"].suites == recipes["parent"].suites


def test_extend_concatenates_parents_in_extend_order(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[a]
[a.suites]
inspect = { task = "ta" }

[b]
[b.suites]
inspect = { task = "tb" }

[child]
extend = ["b", "a"]
""",
    )
    assert recipes["child"].suites["inspect"] == [
        {"task": "tb"},
        {"task": "ta"},
    ]


def test_extend_appends_the_childs_own_entry_last(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[parent]
[parent.suites]
inspect = { task = "tp" }

[child]
extend = "parent"
[child.suites]
inspect = { task = "tc" }
bfcl = "test_category=core"
""",
    )
    assert recipes["child"].suites["inspect"] == [
        {"task": "tp"},
        {"task": "tc"},
    ]
    assert list(recipes["child"].suites) == ["inspect", "bfcl"]


def test_extend_is_recursive_and_allows_forward_references(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[child]
extend = "mid"

[mid]
extend = "base"
[mid.suites]
bfcl = "test_category=core"

[base]
[base.suites]
inspect = { task = "tb" }
""",
    )
    assert recipes["child"].suites["inspect"] == {"task": "tb"}
    assert recipes["child"].suites["bfcl"] == "test_category=core"


def test_extend_unknown_parent_raises(tmp_path):
    with pytest.raises(ValueError, match="extends unknown recipe 'nope'"):
        _recipes(tmp_path, '[child]\nextend = "nope"\n')


@pytest.mark.parametrize(
    "body",
    ['[a]\nextend = "a"\n', '[a]\nextend = "b"\n\n[b]\nextend = "a"\n'],
    ids=["self", "mutual"],
)
def test_extend_cycle_raises(tmp_path, body):
    with pytest.raises(ValueError, match="extend cycle"):
        _recipes(tmp_path, body)


def test_extend_rejects_shared_inspect_task_names(tmp_path):
    with pytest.raises(ValueError, match="share task name"):
        _recipes(
            tmp_path,
            """
[a]
[a.suites]
inspect = { task = "same", custom_tasks = [{ task = "t/a" }] }

[b]
[b.suites]
inspect = { task = "same", custom_tasks = [{ task = "t/b" }] }

[child]
extend = ["a", "b"]
""",
        )


def test_extend_rejects_a_duplicated_parent(tmp_path):
    with pytest.raises(ValueError, match=r"extends \['a'\] more than once"):
        _recipes(
            tmp_path,
            """
[a]
[a.suites]
bfcl = "test_category=core"

[child]
extend = ["a", "a"]
""",
        )


def test_extend_includes_a_shared_ancestor_once(tmp_path):
    recipes = _recipes(
        tmp_path,
        """
[base]
suites.inspect = { task = "mmlu", max_samples = 100 }

[left]
extend = "base"
suites.inspect = { task = "gsm8k" }

[right]
extend = "base"
suites.inspect = { task = "arc" }

[both]
extend = ["left", "right"]
""",
    )
    assert recipes["both"].suites["inspect"] == [
        {"task": "mmlu", "max_samples": 100},
        {"task": "gsm8k"},
        {"task": "arc"},
    ]


def test_recipe_rejects_an_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match=r"names unknown suites \['inpsect'\]"):
        _recipes(tmp_path, '[r]\nsuites.inpsect = "task=x"\n')


@pytest.mark.parametrize("entry", ['"tsk=x"', '{ tsk = "x" }', '"task"'], ids=["string", "dict", "no-equals"])
def test_recipe_rejects_an_invalid_entry(tmp_path, entry):
    with pytest.raises(ValueError, match="recipe 'r': suite 'inspect'"):
        _recipes(tmp_path, f"[r]\nsuites.inspect = {entry}\n")


def test_recipe_rejects_shared_task_names_in_override_strings(tmp_path):
    with pytest.raises(ValueError, match="share task name"):
        _recipes(tmp_path, '[r]\nsuites.inspect = ["task=x", "task=x max_samples=1"]\n')


@pytest.mark.parametrize(
    "entries",
    [
        '["task=a batch_size=1", "batch_size=1 task=a"]',
        '[{ task = "a", batch_size = 1 }, { batch_size = 1, task = "a" }]',
        '["task=a batch_size=1", { task = "a", batch_size = 1 }]',
        '["task=a log_samples=true", { task = "a", log_samples = true }]',
    ],
    ids=["strings", "dicts", "string-and-dict", "string-and-dict-bool"],
)
def test_recipe_rejects_the_same_arguments_in_any_form(tmp_path, entries):
    with pytest.raises(ValueError, match="same arguments twice"):
        _recipes(tmp_path, f"[r]\nsuites.lm_eval = {entries}\n")


def test_override_values_read_as_toml():
    args = config_module._override_args("a=1 b=0.5 c=true d=auto e=[1,2] f=")
    assert args == {"a": 1, "b": 0.5, "c": True, "d": "auto", "e": [1, 2], "f": ""}


@pytest.mark.usefixtures("extra_suite")
def test_register_suite_check_entries_runs_on_recipes(tmp_path):
    def single(entries):
        if len(entries) > 1:
            msg = "one entry only"
            raise ValueError(msg)

    config_module._ENTRY_CHECKS["extra"] = single
    with pytest.raises(ValueError, match="one entry only"):
        _recipes(tmp_path, '[r]\nsuites.extra = ["n_threads=1", "n_threads=2"]\n')


@pytest.mark.parametrize("key", ["extend", "collapse"])
def test_unknown_recipe_keys_raise(key):
    with pytest.raises(pydantic.ValidationError):
        RecipeConfig(name="x", suites={}, **{key: "y"})
