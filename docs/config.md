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

## `[model]` and `[judge]`

`[model]` is the model under evaluation. `[judge]` is the model that grades
answers or plays the user, for the tasks that need one.

| Key | Default | Description |
| --- | --- | --- |
| `id` | | Names the model's folder under `runs/`; required by `run`. |
| `path` | `""` | The model name sent to the endpoint: its weights path or HF hub id. |
| `url` | | The OpenAI-compatible base URL; `/v1` is appended if missing. Filled in by `run --swarm`. |
| `name` | | The model's display name. |
| `swarm_config` | | The domyn-swarm YAML `run --swarm` deploys the model from. |
| `swarm_name` | | A running domyn-swarm deployment `run --swarm` reuses. |

## `[generation]`

| Key | Default | Description |
| --- | --- | --- |
| `reasoning` | `true` | Whether the model reasons before answering; picks `[generation.think]` over `[generation.nothink]`. |
| `max_tokens` | `0` | The completion budget; `0` uses what the context window leaves. |
| `max_model_len` | `0` | The context window; `0` reads it from the endpoint. |
| `token_safety_margin` | `256` | Tokens kept free when the completion budget is computed. |
| `tokenizer_path` | `model.path` | The tokenizer prompt tokens are counted with; without one, prompt lengths are estimated. |
| `request_timeout` | `3600` | Seconds a request may take. |
| `max_retries` | `0` | Retries of a failed request. |
| `num_threads` | `64` | Concurrent requests of each suite that sets no `num_threads` of its own. |

### `[generation.think]` and `[generation.nothink]`

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
| `do_sample` | | Sample rather than decode greedily. |
| `chat_template_kwargs` | | Extra chat template arguments, e.g. `{ enable_thinking = true }`. |
| `system_prompt_template` | `"{instruction}"` | The system prompt; `{instruction}` is the task's own. |
| `min_completion_tokens` | | Completion tokens kept when a long prompt is truncated to fit the context. |

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
| `max_samples` | | Samples per task; overrides `run --max-samples`. |
| `max_tokens` | | The tasks' completion budget; overrides `generation.max_tokens`. |
| `num_threads` | | Concurrent requests (inspect's `max_connections`); overrides `generation.num_threads`. |
| `max_tasks` | | Tasks run in parallel. |
| `request_timeout` | | Seconds a request may take; overrides `generation.request_timeout`. |
| `fail_on_error` | | Whether a failed sample fails its task; a number is the tolerated share of failed samples. |
| `retry_attempts` | | Retries of a failed task. |
| `retry_wait` | | Seconds between task retries. |
| `tiktoken_model` | | The tiktoken encoding inspect counts tokens with. |

Each `custom_tasks` entry is a table:

| Key | Default | Description |
| --- | --- | --- |
| `task` | | The task's registry name, e.g. `inspect_evals/gpqa_diamond`. |
| `task_args` | `{}` | Arguments of the task function. |
| `display_name` | | The task's name in logs and results; tells apart two runs of a task with different arguments. |
| `generate` | `{}` | This task's generation settings (inspect's `GenerateConfig`), over the config's. |
| `message_limit` | | Messages an agentic sample may exchange. |
| `pass_k` | | Samples per prompt; reports the mean and pass@1, 2, 4, ... up to k. |
| `dual_score` | `false` | Report both the upstream scorer's strict accuracy and domyn-inspect's looser answer extraction. |
| `judge` | `false` | The task needs `[judge]`; `run --swarm` serves one. |

## `[bfcl]`

The domyn-bfcl suite.

| Key | Default | Description |
| --- | --- | --- |
| `path` | `"suites/domyn-bfcl"` | The suite's checkout, relative to `SUITES_BASE_DIR`. |
| `test_category` | | The BFCL test categories, comma-separated, e.g. `all_scoring`, `all_scoring_no_websearch`. |
| `fc_model` | `false` | The model calls tools natively, rather than through tools described in the prompt. |
| `model_class` | | The BFCL handler class; unset picks the default one. |
| `num_threads` | | Concurrent requests; overrides `generation.num_threads`. |

## `[lm_eval]`

lm-evaluation-harness tasks, run in domyn-eval's own environment (the `lm-eval`
extra).

| Key | Default | Description |
| --- | --- | --- |
| `task` | | The lm-eval task or group. |
| `path` | | A folder of extra task configs (lm-eval's `include_path`), relative to `SUITES_BASE_DIR`. |
| `batch_size` | | Requests per batch, or `"auto"`. |
| `num_threads` | | Concurrent requests (lm-eval's `num_concurrent`); overrides `generation.num_threads`. |
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

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `RUNS_DIR` | `runs` | Where runs are written and `report` reads them. |
| `SUITES_BASE_DIR` | the working directory | The base of the suites' `path`. |
| `API_KEY` | | The key every suite sends to the model and judge endpoints. |
| `DOMYN_SWARM_DEFAULTS` | `~/.domyn_swarm/defaults.yaml` | domyn-swarm's site defaults: container image, SLURM account and partition. |
| `HF_HOME` | | The Hugging Face cache, referenced by the swarm configs. |
| `BRAVE_API_KEY` | | bfcl's web_search categories query Brave with it. |
| `BFCL_ENCODER_MODEL` | | A local all-MiniLM-L6-v2 for bfcl's memory categories on offline nodes. |
| `INSPECT_SINGULARITY_SIF_DIR` | `.cache/singularity_sif` in the domyn-inspect checkout | The Singularity image cache of inspect's code and swe tasks; see [benchmarks.md](benchmarks.md#inspect). |

[.env.example](../.env.example) lists them: copy it to `.env` and run with
`uv run --env-file .env domyn-eval ...`, or `export UV_ENV_FILE=$PWD/.env`
(absolute, as uv reads a relative path from the folder it runs in). Variables
set in the shell take precedence, and `--sbatch` jobs inherit them.
