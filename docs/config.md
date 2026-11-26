# Configuration

A run is configured by one TOML file: the model and its endpoint, how requests
are sampled, and the settings of each suite. Unknown sections and keys are
errors. [configs/domyn/](../configs/domyn/) and [configs/qwen/](../configs/qwen/)
have complete examples.

Strings may reference environment variables, `${HF_HOME}`, and keys of their
own table or an enclosing one, `${path}`. A reference left unresolved is kept
and logged as a warning.

The run saves the resolved config to `runs/<model id>/<run id>/config.toml`,
with the sections of the suites it ran.

## `[general]`

| Key | Default | Description |
| --- | --- | --- |
| `auto_setup` | `true` | Install each suite's environment before running it; `run --no-setup` turns it off. |
| `log_level` | `"INFO"` | Level of domyn-eval's own logs. |
| `recipes_file` | `"configs/recipes.toml"` | The recipes `--recipe` picks from. |
| `run_id` | the current time | Names the run's folder; set at run time with `run --run-id`. |
| `max_samples` | `-1` | Samples per task, `-1` for all; set at run time with `run --max-samples`. |
| `num_threads` | `1` | Concurrent requests of each suite that sets no `num_threads` of its own. The default sends one request at a time, the same in every suite (theirs differ: lm-eval 1, inspect 10, BFCL 100); raise it to what the endpoint can serve. |
| `request_timeout` | `3600` | Seconds a request may take. |
| `max_retries` | `0` | Retries of a failed request, for every model of the run, the user and grader models included. |

## `[model]`, `[judge]` and `[user]`

`[model]` is the model under evaluation. The others play the inspect model
roles some tasks need, named by their `roles`: a task that calls
`get_model(role="user")` or `get_model(role="grader")` gets the model playing it.

- `user` is played by `[user]`, else by `[judge]`, else by the model under test;
- `grader` is played by `[judge]`, else by the model under test.

A run whose tasks need a role fails its checks if no section plays it, or if
that section's `url` is unset and `run --swarm` does not serve one.

| Key | Default | Description |
| --- | --- | --- |
| `id` | | Names the model's folder under `runs/`; required by `run`. |
| `path` | `""` | The model name sent to the endpoint: its weights path or HF hub id. |
| `url` | | The OpenAI-compatible base URL; `/v1` is appended if missing. Filled in by `run --swarm`. |
| `name` | | The model's display name. |
| `swarm_config` | | The domyn-swarm YAML `run --swarm` deploys the model from. |
| `swarm_name` | | A running domyn-swarm deployment `run --swarm` reuses. Filled in by `run --swarm`. |

A model with a `swarm_name` gets a session ID on each request, so the swarm's
router keeps the turns of a conversation on the same replica. The ID goes in the
`X-Session-ID` header and the `session_params.session_id` body field. A multi-turn conversation cannot tell its last
turn in advance, so Dynamo drops its session when the session timer expires.

### `[model.generation]`, `[judge.generation]` and `[user.generation]`

How each model's requests are built. A model sets the tables it uses: the
model under test its modes, the others their roles.

| Key | Default | Description |
| --- | --- | --- |
| `reasoning` | `true` | Whether the model reasons before answering; picks `think` over `nothink` as its mode. |

### `[model.generation.think]` and `[model.generation.nothink]`

The sampling parameters of each mode. Unset parameters are left to the
endpoint's defaults.

| Key | Default | Description |
| --- | --- | --- |
| `temperature` | | Sampling temperature. |
| `top_p` | | Nucleus sampling mass. |
| `top_k` | | Number of top tokens sampled from. |
| `min_p` | | Minimum token probability, relative to the top token. |
| `presence_penalty` | | Penalty on tokens already present. |
| `frequency_penalty` | | Penalty scaled by a token's count so far. |
| `repetition_penalty` | | Multiplicative penalty on repeated tokens. |
| `do_sample` | | Sample rather than decode greedily; only lm_eval uses it. |
| `chat_template_kwargs` | | Extra chat template arguments, e.g. `{ enable_thinking = true }`. |

### `[judge.generation.user]` and `[judge.generation.grader]`

The sampling parameters of each role, as played by this model, e.g.
`[user.generation.user]`. They take the keys of the modes and apply over the
selected mode. Both default to `temperature = 0`; a table keeps it unless it sets its own.

### `[model.generation.context]` and `[judge.generation.context]`

The context window and how it is budgeted. Only `max_tokens` applies to
`[judge]` and `[user]`, whose roles it caps.

| Key | Default | Description |
| --- | --- | --- |
| `max_tokens` | `0`, and `4096` for `[judge]` and `[user]` | The cap on each generation; `0` uses what the context window leaves, which lets a looping judge write its whole window into the conversation. |
| `max_model_len` | `0` | The context window; `0` reads it from the endpoint. |
| `tokenizer_path` | `model.path` | The tokenizer prompt tokens are counted with; a run that counts them (a `max_tokens` cap, LongBench v2's cut) fails at startup without one. |

## `[slurm]`

The resources of the jobs `run --sbatch` submits. Each key set here becomes an
`#SBATCH` directive; unset ones are left to the cluster's defaults.

| Key | Default | Description |
| --- | --- | --- |
| `template` | | A Jinja job script replacing the packaged one; it sees `job_name`, `log_file`, `directives` and `command`. |
| `account` | | `--account` |
| `partition` | | `--partition` |
| `qos` | | `--qos` |
| `walltime` | | `--time`, as `HH:MM:SS` |
| `nodes` | | `--nodes` |
| `ntasks_per_node` | | `--ntasks-per-node` |
| `cpus_per_task` | | `--cpus-per-task` |
| `memory` | | `--mem`, e.g. `16G` |
| `gres` | | `--gres`, e.g. `gpu:1` |
| `exclude` | | `--exclude` |

## `[inspect]`

The domyn-inspect suite. Recipes usually set `task` and `custom_tasks`.

| Key | Default | Description |
| --- | --- | --- |
| `path` | `"suites/domyn-inspect"` | The suite's checkout, relative to `SUITES_BASE_DIR`. |
| `task` | | Names the task group; each recipe entry needs its own. |
| `custom_tasks` | `[]` | The tasks to run; see below. |
| `max_samples` | | Samples per task; overrides `general.max_samples`. |
| `num_threads` | | Concurrent requests (inspect's `max_connections`); overrides `general.num_threads`. |
| `max_tasks` | | Tasks run in parallel. |
| `request_timeout` | | Seconds a request may take; overrides `general.request_timeout`. |
| `fail_on_error` | | Whether a failed sample fails its task; a number is the tolerated share of failed samples. |
| `retry_attempts` | | Retries of a failed task. |
| `retry_wait` | | Seconds between task retries. |
| `tiktoken_model` | | The tiktoken encoding inspect counts tokens with. |
| `task_packages` | `[]` | Packages of extra tasks to install into the suite's environment; see below. |

Each `custom_tasks` entry is a table:

| Key | Default | Description |
| --- | --- | --- |
| `task` | | The task's registry name, e.g. `inspect_evals/gpqa_diamond`. |
| `task_args` | `{}` | Arguments of the task function. |
| `display_name` | | The task's name in logs and results; tells apart two runs of a task with different arguments. |
| `generate` | `{}` | This task's generation settings (inspect's `GenerateConfig`), over the config's. |
| `message_limit` | | Messages an agentic sample may exchange. |
| `pass_k` | | Samples per prompt; reports the mean and pass@1, 2, 4, ... up to k. |
| `g_pass` | `false` | With `pass_k`, also reports [G-Pass@v](https://arxiv.org/abs/2412.13147) at τ = 0.5, 0.75 and 1 (`g_pass_t50_at_<v>`, ...) and mG-Pass@v (`mg_pass_at_<v>`) for v = 2, 4, ... up to k: how often the model is right in at least τ of v samples, not just once. |
| `dual_score` | `false` | Report both the upstream scorer's strict accuracy and domyn-inspect's looser answer extraction. |
| `roles` | `[]` | The model roles the task needs, `"user"` and `"grader"`; `run --swarm` serves the models playing them. |

### Tasks of your own

`task_packages` runs tasks kept outside domyn-inspect. Each entry is the folder of a small package of task definitions,
relative to the working directory. It is installed editable into the suite's environment **without its dependencies**.

A package registers its tasks through an `inspect_ai` entry point named after the package, and inspect prefixes the
tasks it installs with that name:

```toml
# my-tasks/pyproject.toml
[project.entry-points.inspect_ai]
my_tasks = "my_tasks.registry"    # a module that imports every @task
```

```python
@task(name="my_task")    # run as { task = "my_tasks/my_task" }
def my_task() -> Task: ...
```

```toml
# a recipe entry
inspect = { task = "mine", task_packages = ["../my-tasks"], custom_tasks = [{ task = "my_tasks/my_task" }] }
```

## `[bfcl]`

The domyn-bfcl suite.

| Key | Default | Description |
| --- | --- | --- |
| `path` | `"suites/domyn-bfcl"` | The suite's checkout, relative to `SUITES_BASE_DIR`. |
| `test_category` | | The BFCL test categories, comma-separated, e.g. `all_scoring`, `all_scoring_no_websearch`. |
| `fc_model` | `false` | The model calls tools natively, rather than through tools described in the prompt. |
| `model_class` | | The BFCL handler class; unset picks the default one. |
| `num_threads` | | Concurrent requests; overrides `general.num_threads`. |

## `[lm_eval]`

lm-evaluation-harness tasks, run in domyn-eval's own environment (the `lm-eval`
extra).

| Key | Default | Description |
| --- | --- | --- |
| `task` | | The lm-eval task or group. |
| `path` | | A folder of extra task configs (lm-eval's `include_path`), relative to `SUITES_BASE_DIR`. |
| `batch_size` | | Requests per batch, or `"auto"`. |
| `num_threads` | | Concurrent requests (lm-eval's `num_concurrent`); overrides `general.num_threads`. |
| `chat_completions` | `false` | Use the chat endpoint, which applies the chat template but returns no logprobs. |
| `log_samples` | `false` | Write every sample to the results. |
| `max_seq_lengths` | `[]` | The context lengths of the RULER tasks. |
| `override_generation_kwargs` | `true` | Our generation parameters win over a task's own. |
| `allow_unsafe_code` | `false` | Let tasks run model-generated code on the host (humaneval, mbpp, ...). |

## Recipes

A recipe is a named list of suite runs, picked with `run --recipe` from
`general.recipes_file`:

```toml
[fc_offline]
description = "BFCL v4 without web search, so no internet access is needed."
[fc_offline.suites]
bfcl = "test_category=all_scoring_no_websearch"    # "key=value ..." overrides of [bfcl]

[domyn_edge]
description = "The Domyn Edge results: math, knowledge, if, fc_offline, tau3, code, swe."
extend = ["math", "knowledge", "if", "fc_offline", "tau3", "code", "swe"]    # the entries of these recipes come first
```

A suite's entry overrides the keys of its section, as a `"key=value ..."`
string, a table, or a list of either to run the suite several times. `--suite`
adds suites with the config's own settings.

An entry's `generation` table overrides `[model.generation]` for that entry
only, in any suite. It takes the keys of `[model.generation]`; tables such as
`think`, `context` and `chat_template_kwargs` merge key by key with the
model's:

```toml
[swe.suites]
inspect = { task = "swe", generation = { context = { max_tokens = 32768 } }, custom_tasks = [...] }

[ruler.suites]
lm_eval = { task = "ruler", max_seq_lengths = [65536, 131072], generation = { reasoning = false } }
```

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | | The key every suite sends to the model, judge and user endpoints. |
| `RUNS_DIR` | `runs` | Where runs are written and `report` reads them. |
| `SUITES_BASE_DIR` | the working directory | The base of the suites' `path`. |
| `HF_HOME` | | The Hugging Face cache, referenced by the swarm configs. |
| `DOMYN_SWARM_HOME` | `~/.domyn_swarm/` | domyn-swarm's home directory, where defaults and logs are stored. |
| `BRAVE_API_KEY` | | bfcl's web_search categories query Brave with it. |
| `BFCL_ENCODER_MODEL` | | A local all-MiniLM-L6-v2 for bfcl's memory categories on offline nodes. |
| `INSPECT_SINGULARITY_SIF_DIR` | `.cache/singularity_sif` in the domyn-inspect checkout | The Singularity image cache of inspect's code and swe tasks; see [benchmarks.md](benchmarks.md#inspect). |
| `INSPECT_SINGULARITY_MEMORY_LIMIT_MB` | no cap | The virtual memory cap, in MB, of every command run in inspect's Singularity sandboxes, so one runaway command cannot get the whole job OOM-killed. See [domyn-inspect's sandbox README](../suites/domyn-inspect/domyn_inspect/sandboxes/README.md#memory). |
| `SINGULARITY_TMPDIR` | | Where `singularity pull` builds its temporary images. |

[.env.example](../.env.example) lists them: copy it to `.env` and run with
`uv run --env-file .env domyn-eval ...`, or `export UV_ENV_FILE=$PWD/.env`
(absolute, as uv reads a relative path from the folder it runs in). Variables
set in the shell take precedence, and `--sbatch` jobs inherit them.
