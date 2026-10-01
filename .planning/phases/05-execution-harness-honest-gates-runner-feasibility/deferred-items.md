# Deferred Items

- README 🧪 Testing section's pre-existing `uv run pytest` command form fails under uv 0.12.20 universal resolution ("No solution found ... dnallm[mamba] ... torch>=2.6.0,<2.12 ... for split python_full_version >= '3.14' and sys_platform == 'darwin'").
  status: open
  **What:** `uv run` (without `--no-sync`) resolves the whole project across the `requires-python >= 3.10` span; the `conflicts`-matrix cuda forks combined with the `mamba` extra are unsatisfiable on a hypothetical py3.14-darwin fork, so every `uv run pytest` invocation fails at resolve time. Pre-existing at a8f8460 (05-02 changed nothing in pyproject.toml); it fails identically with or without any install line.
  **Evidence (2026-10-01):** `uv run pytest tests/configuration/test_yaml_load.py -v` → resolver error; `uv run --no-sync pytest tests/configuration/test_yaml_load.py -q` → 21 passed; `.venv/bin/python -m pytest ...` → 21 passed; `git diff a8f8460..HEAD -- pyproject.toml` → empty.
  **Fix shape (out of 05-02 scope):** restructure the `[tool.uv] conflicts` matrix / bound `requires-python` / add `uv.lock` — an architectural pyproject change (Rule 4). Until then, the working invocation forms are `uv run --no-sync pytest` (after the documented `uv pip install -e '.[test,dev,mcp]'`) or `pytest` inside the activated venv.
