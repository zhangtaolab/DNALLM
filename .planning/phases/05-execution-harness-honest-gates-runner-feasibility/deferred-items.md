# Deferred Items

- README 🧪 Testing section's pre-existing `uv run pytest` command form fails under uv 0.12.20 universal resolution ("No solution found ... dnallm[mamba] ... torch>=2.6.0,<2.12 ... for split python_full_version >= '3.14' and sys_platform == 'darwin'").
  status: open
  **What:** `uv run` (without `--no-sync`) resolves the whole project across the `requires-python >= 3.10` span; the `conflicts`-matrix cuda forks combined with the `mamba` extra are unsatisfiable on a hypothetical py3.14-darwin fork, so every `uv run pytest` invocation fails at resolve time. Pre-existing at a8f8460 (05-02 changed nothing in pyproject.toml); it fails identically with or without any install line.
  **Evidence (2026-10-01):** `uv run pytest tests/configuration/test_yaml_load.py -v` → resolver error; `uv run --no-sync pytest tests/configuration/test_yaml_load.py -q` → 21 passed; `.venv/bin/python -m pytest ...` → 21 passed; `git diff a8f8460..HEAD -- pyproject.toml` → empty.
  **Fix shape (out of 05-02 scope):** restructure the `[tool.uv] conflicts` matrix / bound `requires-python` / add `uv.lock` — an architectural pyproject change (Rule 4). Until then, the working invocation forms are `uv run --no-sync pytest` (after the documented `uv pip install -e '.[test,dev,mcp]'`) or `pytest` inside the activated venv.

- dnallm-mcp-server CLI `--host`/`--port` are silently overridden by the yaml config (`start_server`, dnallm/mcp/server.py:1696-1700 reads server.host/server.port unconditionally; the "avoid overriding CLI args" guard in `_start_http_server` runs too late to matter).
  status: open
  **What:** discovered 261003-csd when the executor's `--host 127.0.0.1` launch bound 0.0.0.0 (T-mcp1-04 mitigation ineffective as coded). Same override applies to `--port`, which also invalidates the plan's `--port 8005` fallback runbook — a port switch requires editing both yaml port keys (exactly what the owner's coordinated-commit directive prescribes).
  **Evidence (2026-10-03):** launched with `--host 127.0.0.1`, log line `_start_http_server:1819 - Streamable HTTP endpoint: http://0.0.0.0:8000/mcp`; `ss -tlnp` shows `0.0.0.0:8000`; main() did pass `args.host` through (`Host: 127.0.0.1` logged at server.py:1995).
  **Fix shape (product decision, not this task's scope):** make CLI args win over config when explicitly provided (argparse sentinel defaults), or document that config always wins and drop the misleading flags. Note the server is short-lived behind loopback consumers either way in this workflow.
- MCP server `dna_interpret` runs captum work inline on the event loop (no executor), so a long interpretation blocks all concurrent tool traffic and the 30s `_with_timeout_wrapper` cannot fire while blocked (observed 172s lig call completing "past" the timeout).
  status: RESOLVED 2026-10-03 — quick task 261003-ij4 (commit 3fe80bf) moved the interpret body into the default executor behind a dedicated `_interpret_thread_lock` (CR-01 pattern); 3 red-then-green regression tests in tests/mcp/test_interpret_tool.py
  **What:** observed 261003-csd; mamba models are now guarded (fa19675) but non-mamba interpretations still block the loop for their full duration.
  **Evidence (2026-10-03):** `.scratch/mcp-server-bringup.log` — `Tool dna_interpret completed [duration=172...]` while the wrapper cap is 30s.
  **Fix shape:** run `_dna_interpret` body via `run_in_executor` (pattern already used by ModelManager) — owner-scope server change.
