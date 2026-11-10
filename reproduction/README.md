# Reproduction

Eval logs behind published results, from us and from the community.

## Layout

```text
reproduction/<model>/<run_id>/
├── config.toml       the config that produced the run
├── logs/<suite>/     the suite's logs
└── results/<suite>/  the suite's results
```

## Contributing logs

Open a pull request that adds one `<model>/<run_id>/` folder per run, copied
from domyn-eval's `runs/` directory. State the domyn-eval version and the
model endpoint (serving engine and version) in the pull request description.
