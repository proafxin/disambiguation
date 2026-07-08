# Coding rules

Mandatory. These persist across sessions and devices because this file is committed to the repo and auto-loaded every session.

## Hard rules

- NO comments, docstrings, or any commentary in code. Code only.
- NEVER write anything in an `__init__.py` — every `__init__` module stays empty. Register/expose things from their defining module, not via the package init.
- Top-level absolute imports only, from the DEFINING module (`from abc.models.base import Base`). NEVER package/`__init__` aggregation (`from abc.models import Base`), NEVER relative, NEVER nested/lazy/conditional.
- Prefer `pydantic.BaseModel` for structured data. Use `dataclass` only for genuinely transient internal data.
- No `global` / `nonlocal` — use `@functools.lru_cache` on a factory for cached singletons.
- No broad or defensive `except`. Let exceptions propagate unless there is a specific, named recovery.
- NEVER disable or ignore a lint/type rule (`# noqa`, `# type: ignore`, broad except). Write code that satisfies ruff (`select = ALL`, preview) and mypy. If a rule genuinely needs an exception, ASK first.
- Functions over classes. Typed data containers (pydantic / dataclass) are fine; behavior classes are not.
- Every DB table inherits `abc.models.base.Base` (id, created_at, updated_at).
- `uv` for dependencies (`uv add`); run everything (scripts, migrations, tests, the app) via `uv run` (e.g. `uv run python ...`, `uv run alembic ...`, `uv run pytest`) so it uses the project env. No `print` — use logging. Python >= 3.13, line length 120.

## Build exactly what's asked

Do NOT invent design decisions, abstractions, terms, or out-of-scope concepts. Use the user's exact vocabulary and stated decisions. When unsure, ASK.
