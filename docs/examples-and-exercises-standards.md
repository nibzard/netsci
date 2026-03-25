# Examples and Exercises Standards

This document defines how we create, run, review, and maintain executable teaching material in this repository.

## Scope

These rules apply to:

- runnable examples
- exercise starter code
- exercise solutions kept in the repository
- interactive teaching material that accompanies lectures or tutorials

## Default Tooling

We use:

- Astral `uv` for environment management and command execution
- `marimo` for new runnable examples and exercises

Do not introduce a second parallel workflow unless there is a clear repo-level reason to do so.

## Environment Rules

Use the project environment, not the global Python installation.
The environment should be a local `.venv/` created with `uv`, not a checked-in `env/` directory.

Preferred setup in the current repository:

```bash
uv sync
```

Run commands through `uv run` so execution happens inside the project environment:

```bash
uv run python check_versions.py
uv run path/to/example.py
```

If `marimo` is not yet part of the project dependencies, use:

```bash
uv run --with marimo marimo edit path/to/notebook.py
uv run --with marimo marimo run path/to/notebook.py
```

If `marimo` is installed in the project environment, use:

```bash
uv run marimo edit path/to/notebook.py
uv run marimo run path/to/notebook.py
```

## Authoring Standards

For new executable teaching material:

- prefer `marimo` notebooks saved as plain `.py` files
- keep files small, readable, and git-friendly
- make execution deterministic and free from hidden state
- write code that can run from the repository root

For existing materials:

- do not convert legacy Jupyter notebooks just because they exist
- migrate to `marimo` only when the task benefits from it and the migration is in scope

## Modern Practices

We expect modern Python workflow and teaching materials:

- Python 3.12+ compatible code
- reproducible execution
- explicit imports and dependencies
- no reliance on undeclared local machine state
- no side effects at import time unless the file is intentionally a notebook/app entrypoint
- clear variable names and short, composable functions
- `pathlib.Path` over hard-coded string paths where practical
- examples that fail loudly and clearly when required data is missing

## Exercise and Example Design

Runnable examples and exercises should:

- demonstrate one concept clearly instead of mixing many ideas
- run end-to-end with a short command
- use small local datasets or clearly documented data-loading steps
- include brief context at the top stating the goal, required inputs, and expected output
- avoid unnecessary framework complexity

When an exercise needs interactivity, `marimo` is the default choice.

## File and Output Rules

Treat source files as the canonical artifact.

- prefer plain-text source files over generated binary artifacts
- do not commit bulky generated outputs unless they are intentionally part of course material
- keep generated files isolated and easy to delete or rebuild
- avoid storing secrets, tokens, or private data in examples

## Review Checklist

Before considering an example or exercise done, verify:

- it runs with `uv run ...`
- it works from a clean environment after dependency install
- instructions are short and accurate
- filenames and locations are consistent with nearby course material
- the code matches the pedagogical goal and is not overengineered

## Commands We Standardize On

Common commands:

```bash
uv sync
uv run python some_script.py
uv run marimo edit exercises/example.py
uv run marimo run exercises/example.py
```

## Source of Truth

If another document suggests a conflicting workflow for examples or exercises, this file takes precedence for executable teaching material.

Official references used to align these standards:

- `uv`: https://docs.astral.sh/uv/
- `uv run`: https://docs.astral.sh/uv/concepts/projects/run/
- `uv` with `marimo`: https://docs.astral.sh/uv/guides/integration/marimo/
- `marimo`: https://docs.marimo.io/
