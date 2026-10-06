# Runner host setup — ollama loopback service (MCP-01, D-12, D-06)

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
   # -- or, if the stock unit is already installed, add the Environment
   # lines to it: systemctl edit ollama.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now ollama
   ```
3. Pull the notebooks' literal model reference (~3.3GB):
   ```bash
   ollama pull qwen3.5:4b
   ```
4. Verify:
   ```bash
   curl -s http://127.0.0.1:11434/api/tags   # must list qwen3.5:4b
   ```
5. Re-apply after any unit edit in this repo (the Environment values are
   read at server START, so copying the file alone changes nothing):
   ```bash
   sudo cp scripts/runner/ollama.service /etc/systemd/system/ollama.service
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   # read-only check (no sudo):
   systemctl show ollama -p Environment   # must show OLLAMA_CONTEXT_LENGTH=8192
                                         # and OLLAMA_HOST=127.0.0.1:11434
   ```

**Live-drift notice (2026-10-05):** the LIVE unit on the runner was probed at
`OLLAMA_HOST=0.0.0.0:11434` — a real LAN exposure of the unauthenticated
model server, not the in-repo loopback pin. The re-apply step above restores
`127.0.0.1:11434` and closes that exposure; verify with
`systemctl show ollama -p Environment` after restarting. Never configure or
document a `0.0.0.0` bind for this service.

## Why loopback-only (D-12)

`OLLAMA_HOST=127.0.0.1:11434` is pinned in the unit on purpose: loopback
binding **is the access control** for this service. The runner is a shared,
single-tenant CI box; binding any non-loopback address would expose an
unauthenticated model server (and a ~3.3GB loaded model) to the network. Do
not substitute the `0.0.0.0` example from the ollama FAQ.

## Why num_ctx 8192 (D-06)

`OLLAMA_CONTEXT_LENGTH=8192` pins the server-default context window. The
qwen3.5:4b model (4.2B Q4_K_M, ~3.3GB pull) carries a default context length
of 262144 — still 256k-class, so a request at native context still allocates
a giant kv-cache and dominates mcp-notebook latency and the nightly VRAM
budget (the retired 17.74GB predecessor measured ~36GB per request at 256k;
owner A/A runtime cut, 2026-10-05). A server default is the only seam that
covers BOTH mcp client stacks uniformly: the pydantic_ai sibling talks the
OpenAI-compatible `/v1` endpoint, which has no per-request context parameter.
Precedence is per-request `options.num_ctx` > Modelfile `PARAMETER num_ctx`
> this env > built-in default; re-probe with
`ollama show qwen3.5:4b --modelfile` if the model is ever re-pulled. The env
is read at server start: a unit edit only takes effect after the re-apply
step above (daemon-reload + `systemctl restart ollama`).

**Deferral status (owner 2026-10-06 00:52 CST, standing decision):** the
num_ctx cut is DEFERRED entirely. The live ollama service was NOT
reconfigured — the live unit does not carry this env pin — so the in-repo
`OLLAMA_CONTEXT_LENGTH=8192` above stays committed but INERT: it records
the intended end state, not the running service. Until the deferral is
lifted and the re-apply step above runs, the mcp_example pair is served at
the model's default context (256k-class) and the latency/VRAM cost is
accepted. The harness budgets tell this same story
(`tests/examples/_execution.py` cell_timeout comments).

Swapped 2026-10-06 (owner decision 15:27 CST): the notebooks' literal model
reference became qwen3.5:4b (capability probe PASSED 15:24 CST — 3-turn
tool-calling, 34.8s cold / 6.1s / 5.1s warm). The Modelfile num_ctx probe
has NOT been re-run for qwen3.5:4b: the 2026-10-05 live probe (no Modelfile
num_ctx) covered only the replaced qwen3.8:latest, so if qwen3.5:4b's
Modelfile sets `PARAMETER num_ctx` the env pin would be silently overridden
once the unit is re-applied — re-probe with
`ollama show qwen3.5:4b --modelfile` before relying on the env governing.

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
