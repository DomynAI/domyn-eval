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


# ── generation: modes, roles, context ────────────────────────────────────────


def test_reasoning_selects_think_params(config):
    generation = config.model.generation
    assert generation.mode is generation.think
    assert generation.mode.temperature == pytest.approx(0.6)
    assert generation.context.completion_reserve == 16384


def test_reasoning_selects_nothink_params(tmp_path):
    config = _load(
        tmp_path,
        """
[model.generation]
reasoning = false
[model.generation.think]
temperature = 0.6
[model.generation.nothink]
temperature = 0.7
""",
    )
    assert config.model.generation.mode.temperature == pytest.approx(0.7)


def test_missing_tables_keep_the_others(tmp_path):
    config = _load(
        tmp_path,
        """
[model.generation.context]
max_model_len = 1234
[model.generation.think]
temperature = 0.6
""",
    )
    assert config.model.generation.context.max_model_len == 1234
    assert config.model.generation.mode.temperature == pytest.approx(0.6)


def test_completion_reserve_defaults_to_none(tmp_path):
    assert _load(tmp_path, "").model.generation.context.completion_reserve is None


def test_generation_defaults_match_loading_an_empty_file(tmp_path):
    assert _load(tmp_path, "").model.generation == config_module.GenerationConfig()


def test_the_run_settings_are_general(tmp_path):
    general = _load(tmp_path, "[general]\nnum_threads = 8\nrequest_timeout = 60\nmax_retries = 2\n").general
    assert (general.num_threads, general.request_timeout, general.max_retries, general.max_samples) == (8, 60, 2, -1)


@pytest.mark.parametrize("key", ["num_threads", "request_timeout", "max_retries", "max_samples"])
def test_the_run_settings_are_not_a_models(tmp_path, key):
    with pytest.raises(pydantic.ValidationError, match=key):
        _load(tmp_path, f"[model.generation]\n{key} = 1\n")


def test_the_old_generation_section_is_unknown(tmp_path):
    with pytest.raises(ValueError, match=r"unknown config sections \['generation'\]"):
        _load(tmp_path, "[generation]\nreasoning = true\n")


def test_the_judge_is_capped_by_default(tmp_path):
    assert _load(tmp_path, "").judge.generation.context.max_tokens == 4096
    assert config_module.Config().judge.generation.context.max_tokens == 4096
    assert _load(tmp_path, '[judge]\npath = "j"\n').judge.generation.context.max_tokens == 4096
    context = _load(tmp_path, "[judge.generation.context]\nmax_model_len = 8\n").judge.generation.context
    assert context.max_tokens == 4096


def test_the_judges_cap_can_be_lifted(tmp_path):
    assert _load(tmp_path, "[judge.generation.context]\nmax_tokens = 0\n").judge.generation.context.max_tokens == 0


def test_the_model_under_test_is_not_capped_by_default(tmp_path):
    assert _load(tmp_path, "").model.generation.context.max_tokens == 0


def test_the_roles_default_to_deterministic(tmp_path):
    judge = _load(tmp_path, "").judge.generation
    assert (judge.user.temperature, judge.grader.temperature) == (0.0, 0.0)  # as tau2 asks of its user


def test_a_role_table_keeps_the_defaults_it_does_not_set(tmp_path):
    body = "[judge.generation.user]\nchat_template_kwargs = { enable_thinking = false }\n"
    user = _load(tmp_path, body).judge.generation.user
    assert (user.temperature, user.chat_template_kwargs) == (0.0, {"enable_thinking": False})


def test_a_role_table_overrides_its_defaults(tmp_path):
    assert _load(tmp_path, "[judge.generation.grader]\ntemperature = 0.2\n").judge.generation.grader.temperature == 0.2


def test_a_role_samples_with_its_own_params_over_the_mode(tmp_path):
    body = """
[model.generation.think]
temperature = 1.0
top_k = 20
chat_template_kwargs = { enable_thinking = true }
[model.generation.user]
top_k = 40
"""
    user = _load(tmp_path, body).model.generation.role("user")
    assert (user.temperature, user.top_k, user.chat_template_kwargs) == (0.0, 40, {"enable_thinking": True})


def test_the_judge_inherits_nothing_from_the_model(config):
    assert config.model.generation.think.temperature is not None
    assert config.judge.generation.role("user") == config_module.SamplingConfig(temperature=0.0)


def test_snapshot_writes_the_judges_generation_for_the_suites(config, tmp_path):
    config.judge.generation.context.max_tokens = 1024
    config.judge.generation.user.top_k = 20
    saved = tomllib.loads(_snapshot(config, tmp_path).read_text())
    assert saved["judge"]["generation"]["context"]["max_tokens"] == 1024
    assert saved["judge"]["generation"]["user"] == {"temperature": 0.0, "top_k": 20}
    assert load_config(tmp_path / "run.config.toml").judge.generation == config.judge.generation


def test_the_user_model_is_capped_by_default(tmp_path):
    assert _load(tmp_path, '[user]\npath = "u"\n').user.generation.context.max_tokens == 4096
    assert config_module.Config().user.generation.context.max_tokens == 4096


@pytest.mark.parametrize(
    ("body", "user", "grader"),
    [
        ("", None, None),
        ('[judge]\nurl = "http://j:1"\n', "judge", "judge"),
        ('[judge]\nurl = "http://j:1"\n[user]\nswarm_config = "u.yaml"\n', "user", "judge"),
        ('[user]\npath = "u"\n', "user", None),
    ],
)
def test_the_user_is_played_by_the_user_model_else_the_judge(tmp_path, body, user, grader):
    config = _load(tmp_path, body)
    assert (config.role_section("user"), config.role_section("grader")) == (user, grader)


def test_an_unknown_role_raises(config):
    with pytest.raises(ValueError, match="unknown role 'judge'"):
        config.role_section("judge")


def test_snapshot_writes_the_user_model(config, tmp_path):
    config.user.path = "u"
    config.user.generation.user.temperature = 0.5
    reloaded = load_config(_snapshot(config, tmp_path))
    assert reloaded.user == config.user


def test_sampling_params_outside_a_sampling_table_raise(tmp_path):
    with pytest.raises(pydantic.ValidationError, match="temperature"):
        _load(tmp_path, "[model.generation]\ntemperature = 1\n")


@pytest.mark.parametrize("key", ["max_tokens", "completion_reserve"])
def test_a_sampling_table_takes_only_sampling_params(tmp_path, key):
    with pytest.raises(pydantic.ValidationError, match=key):
        _load(tmp_path, f"[judge.generation.grader]\n{key} = 1\n")


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


@pytest.mark.parametrize("name", ["bfcl", "judge", "user"])
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
[model.generation.context]
tokenizer_path = "${path}/tok"
""",
    )
    assert config.model.path == "/base/model"
    # [model] encloses the table, so its keys resolve
    assert config.model.generation.context.tokenizer_path == "/base/model/tok"


# ── snapshots ────────────────────────────────────────────────────────────────


def _snapshot(config, tmp_path):
    path = tmp_path / "run.config.toml"
    config.save(path)
    return path


def test_snapshot_round_trips(config, tmp_path):
    reloaded = load_config(_snapshot(config, tmp_path))
    assert reloaded.model.url == config.model.url
    assert reloaded.model.generation == config.model.generation
    assert reloaded.judge == config.judge
    assert reloaded.suite.inspect == config.suite.inspect
    assert reloaded.suite.bfcl == config.suite.bfcl


def test_snapshot_persists_runtime_changes(config, tmp_path):
    config.suite.inspect.task = "full"
    config.model.generation.context.tokenizer_path = "/changed"
    reloaded = load_config(_snapshot(config, tmp_path))
    assert reloaded.suite.inspect.task == "full"
    assert reloaded.model.generation.context.tokenizer_path == "/changed"


def test_snapshot_leaves_out_runtime_only_fields(config, tmp_path):
    config.general.run_id = "run-abc123"
    config.general.max_samples = 5
    config.model.endpoint_max_model_len = 4096
    raw = tomllib.loads(_snapshot(config, tmp_path).read_text())
    assert "run_id" not in raw["general"]
    assert "file" not in raw["general"]
    assert "max_samples" not in raw["general"]
    assert "endpoint_max_model_len" not in raw["model"]
    assert raw["model"]["generation"]["think"]["temperature"] == pytest.approx(0.6)


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
