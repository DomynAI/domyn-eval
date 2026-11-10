# How to contribute to the project

Things to keep in mind when contributing to domyn-eval.

## Editor setup

The following instructions assume you are using VSCode; any editor with ruff and editorconfig support works too.

### Linting and Formatting

Download the [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension and configure it for the project (the extension overview shows how to do it).

### editorconfig

Install the [editorconfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig) extension.

### Pre-commit

Install [pre-commit](https://pre-commit.com/) as a UV tool:

```sh
uv tool install pre-commit
```

and use it to install all hooks:

```sh
pre-commit install
```

### Type checking

It's done with [pyright](https://microsoft.github.io/pyright/#/), the hook will run automatically after every commit if the hooks are installed.

## Tests

`uv run pytest` runs the unit tests. The integration test runs `check`, `run` and `report` on the real suites against
a fake endpoint, so it needs the submodules and uv; `pytest` leaves it out unless asked:

```sh
uv run pytest -m integration
```

It sets up the suites' environments and loads the `openai/gsm8k` dataset and the `Domyn/Domyn-Edge-SFT` tokenizer, so it
needs internet access, or an `HF_HOME` that already holds them. Run it after bumping a submodule.

## Development workflow

1. Create an issue and add the relevant tags
2. Create a branch linked to the issue to work on it
3. Open a PR to main when the code is ready
