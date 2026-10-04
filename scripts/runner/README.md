# Runner host setup — ollama loopback service (MCP-01, D-12)

This directory holds the in-repo definition of the `dnallm-nightly` self-hosted
runner's ollama service, so it is auditable and rebuildable by diff instead of
living only as host state. The service backs the two MCP client example
notebooks (`example/mcp_example/`), whose LLM endpoint is
`http://127.0.0.1:11434`.

## One-time owner setup on the runner

1. Install ollama (official Linux install — places the binary at
   `/usr/local/bin/ollama` and creates the `ollama` user/group).
2. Install this unit:
   ```bash
   sudo cp scripts/runner/ollama.service /etc/systemd/system/ollama.service
   # -- or, if the stock unit is already installed, add the two Environment
   # lines to it: systemctl edit ollama.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now ollama
   ```
3. Pull the notebooks' literal model reference (17.74GB):
   ```bash
   ollama pull qwen3.8:latest
   ```
4. Verify:
   ```bash
   curl -s http://127.0.0.1:11434/api/tags   # must list qwen3.8:latest
   ```

## Why loopback-only (D-12)

`OLLAMA_HOST=127.0.0.1:11434` is pinned in the unit on purpose: loopback
binding **is the access control** for this service. The runner is a shared,
single-tenant CI box; binding any non-loopback address would expose an
unauthenticated model server (and a 17GB loaded model) to the network. Do not
substitute the `0.0.0.0` example from the ollama FAQ.

## How the D-13 readiness probe treats a down service

The test-layer probe (`_gate_ollama_stack` in
`tests/examples/test_notebook_execution.py`) retries
`http://127.0.0.1:11434/api/tags` for ~60s before deciding. If the service is
down or the model is not pulled, the two MCP client notebook tests emit a typed
`network-unavailable:` skip whose message carries the probe output and retry
log as evidence. This fallback is for **missing infrastructure only** — if the
model answers but its output is wrong, that is an assertion failure to be
repaired, never a skip.

## Ops footnote

When restarting the runner's self-hosted GitHub Actions service, use a
sanitized environment (`env -i ...`), otherwise `uv` resolves the wrong
interpreter and installs into the wrong venv — see the dnallm-nightly runner
ops note. Keep ollama and the actions-runner service on separate systemd
units (they are) so runner restarts never touch the model server.
