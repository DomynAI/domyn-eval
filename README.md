<div align="center">

<!-- TODO: add the logos to static/ and drop the title below.
<img src="https://raw.githubusercontent.com/DomynAI/domyn-eval/main/static/domyn-eval-logo-primary.svg#gh-light-mode-only" alt="Domyn Eval" height="100">
<img src="https://raw.githubusercontent.com/DomynAI/domyn-eval/main/static/domyn-eval-logo-white.svg#gh-dark-mode-only" alt="Domyn Eval" height="100">
-->

# domyn-eval

**Evaluate LLMs on benchmark suites, against any OpenAI-compatible endpoint, locally or on SLURM.**

domyn-eval is a CLI that runs the recipes behind Domyn's published results: inspect-ai tasks, BFCL v4
and lm-evaluation-harness, each in its own environment. It writes each run's config, logs and results
to one folder, and reports them side by side. Some of these recipes are not published yet;
we will add them over time.

[![CI](https://github.com/DomynAI/domyn-eval/actions/workflows/ci.yaml/badge.svg)](https://github.com/DomynAI/domyn-eval/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-brightgreen)](https://github.com/DomynAI/domyn-eval/blob/main/pyproject.toml)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/DomynAI/domyn-eval/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)

<a href="#quick-start">Quick start</a> •
<a href="#suites">Suites</a> •
<a href="#runs">Runs</a> •
<a href="https://github.com/DomynAI/domyn-eval/blob/main/docs/config.md">Configuration</a> •
<a href="https://github.com/DomynAI/domyn-eval/blob/main/docs/benchmarks.md">Benchmarks</a> •
<a href="https://github.com/DomynAI/domyn-eval/blob/main/CONTRIBUTING.md">Contributing</a>

</div>

## Suites

| Suite | Runs | Source |
| --- | --- | --- |
| `inspect` | inspect-ai tasks: math, knowledge, instruction following, agents | [domyn-inspect](https://github.com/DomynAI/domyn-inspect) (submodule) |
| `bfcl` | the Berkeley Function Calling Leaderboard v4 | [domyn-bfcl](https://github.com/DomynAI/domyn-bfcl) (submodule) |
| `lm_eval` | lm-evaluation-harness tasks | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with the `lm-eval` extra |

`inspect` and `bfcl` run in their own uv environment, which `run` installs in
their checkout unless `--no-setup` is passed. [docs/benchmarks.md](https://github.com/DomynAI/domyn-eval/blob/main/docs/benchmarks.md)
lists the benchmarks of each suite and the recipes that run them.

## Install

```sh
git clone --recurse-submodules https://github.com/DomynAI/domyn-eval.git
cd domyn-eval
uv sync --extra inspect-ai   # add --extra lm-eval for the lm_eval suite
cp .env.example .env         # API_KEY, HF_HOME, ...; see docs/config.md#environment
uv run --env-file .env domyn-eval --help
```

The `inspect-ai` extra is needed to report inspect results.

## Quick start

A config names the model, its endpoint and how to sample from it; see
[configs/domyn/domyn-edge.toml](https://github.com/DomynAI/domyn-eval/blob/main/configs/domyn/domyn-edge.toml) and
[docs/config.md](https://github.com/DomynAI/domyn-eval/blob/main/docs/config.md). A recipe from [configs/recipes.toml](https://github.com/DomynAI/domyn-eval/blob/main/configs/recipes.toml)
picks the suites and benchmarks.

```sh
domyn-eval check configs/domyn/domyn-edge.toml --recipe math   # config, suites and endpoint
domyn-eval run configs/domyn/domyn-edge.toml --recipe math
domyn-eval report domyn-edge
```

Requests go to `model.url`, authenticated with `API_KEY` when it is set.

- **`--swarm`**: when `model.url` is unset or down, reuse a running
  [domyn-swarm](https://github.com/DomynAI/domyn-swarm) of the model, or deploy
  one from `model.swarm_config` and tear it down when the run ends. The same
  goes for the judge and user models the selected tasks need. domyn-swarm takes the
  container image and SLURM account from the site's `defaults.yaml`
  (`DOMYN_SWARM_DEFAULTS`, else `~/.domyn_swarm/defaults.yaml`).
- **`--sbatch`**: submit each suite as a SLURM job, with the `[slurm]` resources;
  jobs run one after another unless `--no-deps` is passed.
- **`--dry`**: print the commands, or write the job scripts, without running them.
- **`--suite`, `--max-samples`**: run single suites, or a few samples per task.

`domyn-eval <command> --help` lists every option.

## Runs

```text
runs/<model id>/
├── <run id>/
│   ├── config.toml       the resolved config the run used
│   ├── logs/<suite>/     one log per command (setup.log, run.log, …) and the job scripts
│   └── results/<suite>/  the suites' results
└── report.csv            written by `domyn-eval report`
```

The console shows only domyn-eval's own messages: each suite command's output goes
to its log, whose path is printed when it starts, and a failure prints the log's
last lines. `RUNS_DIR` moves `runs/`.

`report MODEL...` reads every run of the models; `--leaderboard` lines them up side
by side, and `--profile` keeps the benchmarks of a profile such as
[configs/report_profile.yaml](https://github.com/DomynAI/domyn-eval/blob/main/configs/report_profile.yaml). Each model's scores are
followed by its samples' stats, which flag the scores to distrust: truncated, empty,
looping or unscored answers, and reasoning that does not match the run's mode. The
thresholds are in [configs/stats_profile.yaml](https://github.com/DomynAI/domyn-eval/blob/main/configs/stats_profile.yaml) (`--stats-profile`
takes another file; `--no-stats` skips them, as they read every sample).
[reproduction/](https://github.com/DomynAI/domyn-eval/tree/main/reproduction/) holds the runs behind published results.

## Using it from another repo

Suite paths in a config are relative to `SUITES_BASE_DIR` (default: the working
directory), and `general.recipes_file` points at any recipes TOML, so a project
can depend on domyn-eval and keep its own configs and suite checkouts. A suite
of its own is added with `domyn_eval.config.register_suite`.

## License

[Apache-2.0](https://github.com/DomynAI/domyn-eval/blob/main/LICENSE)
