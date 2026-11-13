# Benchmarks

The benchmarks of the recipes in [configs/recipes.toml](../configs/recipes.toml),
by suite. `domyn-eval run <config> --recipe <name>` runs a recipe;
`domyn-eval report <model> --profile configs/report_profile.yaml` reports the
metrics of Domyn's leaderboard.

## inspect

Tasks come from [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals),
or from domyn-inspect's own `domyn_inspect/tasks/` folder. Any of them can be added to a
recipe as a `custom_tasks` entry; see [config.md](config.md#inspect).

| Benchmark | Task | Source | Recipes |
| --- | --- | --- | --- |
| AIME 2025 (pass@16) | `aime2025` | inspect_evals | math, domyn_edge |
| AIME 2026 (pass@16) | `aime2026` | inspect_evals | math, domyn_edge |
| MGSM (de, en, es, fr, ru) | `mgsm` | inspect_evals | math, domyn_edge |
| GPQA Diamond | `gpqa_diamond` | inspect_evals | knowledge, domyn_edge |
| MMLU-Pro | `mmlu_pro` | inspect_evals | knowledge, domyn_edge |
| IFEval | `ifeval` | inspect_evals | if, domyn_edge |
| IFBench | `ifbench` | domyn-inspect | if, domyn_edge |
| IFStruct | `ifstruct` | domyn-inspect | if, domyn_edge |
| tau2 airline | `tau2_airline` | inspect_evals | tau2, tau3, domyn_edge |
| tau2 retail | `tau2_retail` | inspect_evals | tau2, tau3, domyn_edge |
| tau2 telecom | `tau2_telecom` | inspect_evals | tau2, tau3, domyn_edge |
| tau3 banking | `tau2_banking` | inspect_evals | tau3, domyn_edge |
| SciCode (with background) | `scicode_singularity` | domyn-inspect | code, domyn_edge |
| BigCodeBench | `bigcodebench_singularity` | domyn-inspect | code, domyn_edge |
| MBPP | `mbpp_singularity` | domyn-inspect | code, domyn_edge |
| LiveBench coding (2024-11-25 release, offline) | `livebench_offline` | domyn-inspect | code, domyn_edge |
| SWE-bench Verified | `swe_bench_singularity` | domyn-inspect | swe, domyn_edge |
| LongBench v2 | `longbench_v2` | domyn-inspect | long_context |

Task names are registered under `inspect_evals/`, e.g. `inspect_evals/gpqa_diamond`.
The tau tasks need a model playing the user: `[user]`, else `[judge]`. Set its
`url`, or pass `run --swarm` to serve one from its `swarm_config`.

The `code` and `swe` tasks run the model's code in [Singularity](https://sylabs.io/singularity/)
containers, so `singularity` must be on the path of the node that runs them.
Their images are cached in `$INSPECT_SINGULARITY_SIF_DIR` (default:
`.cache/singularity_sif` in the domyn-inspect checkout), and a task pulls a missing
one when it starts. On compute nodes without internet, pull them first from a node
that has it, with the same `INSPECT_SINGULARITY_SIF_DIR`:

```sh
cd suites/domyn-inspect
uv run python -m domyn_inspect.tasks.bigcodebench_singularity.bigcodebench_singularity    # BigCodeBench, MBPP, LiveBench coding
uv run python -m domyn_inspect.tasks.swe_bench_singularity.swe_bench_singularity          # one image per SWE-bench instance
```

SciCode's sandbox is built by `run`'s setup, which prefetches the tasks on the
submitting node.

## bfcl

[BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html), through domyn-bfcl.

| Test category | Covers | Recipes |
| --- | --- | --- |
| `all_scoring` | every scored category: single-turn, multi-turn, memory, web search | fc |
| `all_scoring_no_websearch` | `all_scoring` without web search, so no internet access is needed | fc_offline, domyn_edge |

`[bfcl] test_category` takes any category BFCL defines, or a comma-separated list.

## lm_eval

Any [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
task, set with `[lm_eval] task`, and the tasks of an `[lm_eval] path` folder.
RULER runs at the context lengths of `max_seq_lengths`. No shipped recipe uses
lm_eval yet.
