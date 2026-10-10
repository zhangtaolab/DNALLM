---
quick_task: 261003-csd-execute-the-two-owner-deferred-mcp-clien
phase: quick-261003-csd
plan: 01
status: complete
started: 2026-10-03T01:23:03Z
completed: "2026-10-03T03:13:00Z"
duration_min: 110
branch: phs
push: manual-only
estimate:
  tokens: 60000
actuals:
  tokens: 51130   # chars/4 over git diff e1d54f7..HEAD (realized changes)
  tasks: 3
  commits: 5      # git log --oneline e1d54f7..HEAD | wc -l
plan_head_before: e1d54f7
plan_head_after: 9453d23
tags: [notebook-execution, mcp, ollama, gated-lane, isolated-kernel]
key-files:
  modified:
    - tests/examples/test_notebook_execution.py
    - tests/examples/_execution.py
    - example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
    - example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
    - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
    - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
    - dnallm/mcp/model_manager.py
    - dnallm/mcp/server.py
    - tests/mcp/test_model_manager.py
    - tests/mcp/test_interpret_tool.py
commits:
  - bd5496a "test(quick-261003-csd): execute-state ollama/MCP gate + 4xx-honest probe + isolated langchain kernel"
  - 114390f "fix(quick-261003-csd): langchain mcp notebook server-start cell -> probe-then-ensure guard"
  - ae76a3e "fix(quick-261003-csd): single-flight MCP inference — concurrent DataLoader forks deadlock under threaded serving"
  - fa19675 "fix(quick-261003-csd): dna_interpret refuses mamba models — captum backward SIGKILLs the serving process"
  - 9453d23 "test(quick-261003-csd): both mcp client notebooks green in gated lane; refreshed pairs + agent-loop budgets"
---

# Quick Task 261003-csd: Execute the two owner-deferred MCP client notebooks to green

**One-liner:** Both never-executed mcp client notebooks (langchain + pydantic_ai) now run green end-to-end in the gated pytest lane against the live ollama+MCP stack, after two real dnallm serving bugs were fixed with same-change tests (single-flight inference; mamba interpret guard).

## Outcome

| Notebook | Status | Where proven | Total wall | Heaviest cell |
|---|---|---|---|---|
| mcp_client_ollama_langchain_agents.ipynb | GREEN, isolated kernel `dnallm-mcp-langchain` | gated pytest lane (official run) + driver runs | 414.0s (r1) | cell 7 agent analysis 401.8s |
| mcp_client_ollama_pydantic_ai.ipynb | GREEN, project `python3` kernel | gated pytest lane (official run) + driver runs | 294.0s (r6) | cell 6 agent analysis 263.8s |

Official record run: `.venv/bin/python -m pytest "tests/examples/test_notebook_execution.py::TestGatedNotebookExecution" -m slow -k "mcp_client" -q --timeout 7500` → **2 passed in 365.34s** with the live stack up. Post-teardown (server killed, ollama still up) the same two tests **typed-skip** with the `network-unavailable:` prefix carrying both live probe results in-message — the new gate is honest in both directions. With this, every one of the 21 example notebooks has now been executed at least once (D-08's 2-of-21 gap closed).

## Port decision (owner directive executed)

**Port 8000 used; the 8005 fallback NEVER fired** (8000 was free at launch and bound cleanly) — so the conditional coordinated commit (yaml ports, notebook endpoint cells, MCP_ENDPOINT constant, docs .md mentions) was **not needed and not landed**. Ollama references untouched throughout. Server launched with `--host 127.0.0.1 --port 8000`, killed by recorded PID at teardown (exit 143, port freed, verified).

## D-08 closure note

Owner decision D-08/T-05-16 (2026-10-03, "先处理 mcp 的两个") is closed: the gate's fail-loudly never-auto-execute sentinel is retired, both-up is the owner-approved execute state, and the coexistence (ollama 17GB qwen3.8 + 3 DNA models on GB10) ran the full campaign without VRAM issues.

**Next round (out of scope here, per plan contract): the 5 remaining gated notebooks** — `notebooks/generation_evo_models/inference.ipynb` (evo gate), `notebooks/generation_megaDNA/inference.ipynb` + `notebooks/finetune_custom_head/finetune.ipynb` + `notebooks/finetune_generation/finetune_generation.ipynb` (megaDNA gate), `notebooks/lora_finetune_inference/lora_finetune.ipynb` + `lora_inference.ipynb` (mamba gate).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - serving bug] Single-flight MCP inference (commit ae76a3e)**
- **Found during:** Task 3 first campaign run
- **Issue:** `predict_multi_model` gathers concurrent `predict_sequence` executor submits; each `infer_seqs` builds a `DataLoader(num_workers=4)` whose worker spawn forks. Concurrent forks from several threads while a hub-cache `filelock` changes descriptor ownership raise `os.fork is unsafe while filelock is changing descriptor ownership` in one thread and hang another (stuck `pt_data_worker` children holding the uvicorn socket). Every `dna_*_predict` then hits the 30s tool timeout — the pydantic notebook's first "green" was hollow (agent narrating around tool timeouts).
- **Fix:** `ModelManager._infer_lock` (asyncio.Lock) around the executor submit in `predict_sequence` and `predict_batch` — at most one `infer_seqs` in the process at any instant. Multi-model predict went from always-timeout to 1.6-2.1s warm.
- **Files:** dnallm/mcp/model_manager.py, tests/mcp/test_model_manager.py (TestSingleFlightInference, RED→GREEN: max_active==3 observed pre-fix)
- **Commit:** ae76a3e

**2. [Rule 2 - process-killing tool] dna_interpret refuses mamba models (commit fa19675)**
- **Found during:** Task 3 campaign — two mid-campaign server deaths, root-caused via exit-code instrumentation (`SERVER_EXIT_CODE=137`, SIGKILL)
- **Issue:** captum gradient backward on Mamba-family models (pure-PyTorch fallback backend; CUDA selective-scan kernels not installed) exhausts memory and the whole serving process is SIGKILLed — reproduced deterministically in-process for BOTH `lig` and `layer_conductance` on the open_chromatin DNAMamba model (exit 137 each time). qwen3.8 calls `_dna_interpret` during "analyze thoroughly" runs, so any notebook run could kill the server mid-flight.
- **Fix:** guard in `_dna_interpret` consults the model config architecture; mamba-family → typed `isError` error dict per the protocol-boundary convention, server lives.
- **Files:** dnallm/mcp/server.py, tests/mcp/test_interpret_tool.py (mamba guard + DNABERT-unaffected tests, RED→GREEN)
- **Commit:** fa19675

**3. [Rule 1 - API drift] pydantic notebook minimal 1.102.0 alignment (commit 9453d23, notebook pair)**
- **Issue (execute-proven):** `pydantic_ai.mcp.FastMCPClient` is the raw fastmcp `Client` in 1.102.0 — a client, not a Toolset; passing it to `Agent(toolsets=[...])` raises `TypeError: 'Client' object is not callable` at run time. `result.usage()` call style works but is a `_DeprecatedCallableProperty` (warns).
- **Fix:** `MCPToolset(FastMCPClient(...))` + `result.usage` property style. No speculative edits beyond these (execute-first honored).
- **Commits:** notebook changes landed inside 9453d23

**4. [Rule 3 - budget] mcp cell_timeout 600→1800 (commit 9453d23)**
- **Evidence:** with tools actually working, the pydantic analysis cell exceeded the 600s starter (r3 RED: CellTimeoutError at 601.6s); green re-run measured 263.8s — 1800s stays strictly under the gated class's 3600s mark, per the plan's budget ladder.

### Out-of-scope discoveries (documented, not fixed)

- **CLI `--host`/`--port` are silently overridden by the yaml** (`start_server`, dnallm/mcp/server.py:1696-1700): my `--host 127.0.0.1` launch bound 0.0.0.0 (T-mcp1-04's stated mitigation is ineffective as coded; server was short-lived and torn down). Also invalidates a `--port 8005` relaunch — the owner's yaml-edit directive was the correct mechanism all along. Logged to phase deferred-items.md (product decision: make CLI win or drop the flags).
- **`dna_interpret` runs inline on the event loop** (no executor): a long interpretation blocks all concurrent traffic and the 30s wrapper cannot fire while blocked (observed 172s lig completing "past" the cap). Logged to deferred-items.md.
- **Full-tree docs-sync gate is red from pre-existing out-of-scope residue** (owner's benchmark/inference notebook churn + gitignored run artifacts under example/notebooks/benchmark/). The `mcp_example` subtree scoped-sync (same comparison semantics) is **OK** — no mcp entries in the error list. Not fixed: orchestrator forbids staging those files.

## Auth gates

None.

## Evidence trail

- `.scratch/mcp-server-bringup.log` + `.scratch/mcp-server-bringup.pid` — bring-up, fork-error forensics, exit-code instrumentation (137 root-cause, 143 clean teardown)
- `.scratch/mcp-round/` — `*-times.json` per-cell wall times for every campaign round (pydantic r1-r6, langchain r1, finals), executed notebooks, `campaign.py`/`refresh.py` drivers, `prewarm*.py`, `gentle_warm.py`, `verify_fixes.py`, `direct_*.py` root-cause repros
- Isolation proof: `langchain_ollama` absent from the project venv post-campaign (`find_spec` None); isolated venv `.scratch/mcp-example-venvs/langchain/` carries langchain 1.4.3 + langchain-ollama 1.1.0 + ollama 0.6.3 (installed by the notebook's own cells)

## Package provenance (T-mcp1-02)

- langchain-ollama: https://pypi.org/project/langchain-ollama/ — langchain org first-party (project home: docs.langchain.com/oss/python/integrations/providers/ollama); installed only into the throwaway venv
- ipykernel: https://pypi.org/project/ipykernel/ — Project Jupyter (IPython Development Team); installed only into the throwaway venv
- No project-venv or pyproject installs occurred at any point (T-mcp1-SC honored)

## Verification summary

- Gated lane (up): 2 passed in 365.34s; both-direction honesty: 2 typed-skips with live evidence post-teardown
- Fast examples lane: 109 passed / 1 skipped (baseline parity); mcp suite: 217 passed (includes the 4 new single-flight + 2 new guard tests)
- Full-suite fast lane: **1716 passed / 1 skipped** (baseline 1703/1 + exactly the 13 new fast-lane tests: 9 gate/probe/kernel + 2 single-flight + 2 mamba-guard; zero failures)
- `cmp` byte-identity: both example/ ↔ docs/ pairs identical; scoped mcp_example docs-sync OK; ruff format --check + ruff check clean on all changed test/source files
- Only sanctioned files staged; the unrelated working-tree residue (05-UAT.md, benchmark/inference notebooks, .planning/ churn) untouched

## Self-Check: PASSED

- Files: tests/examples/test_notebook_execution.py, tests/examples/_execution.py, dnallm/mcp/model_manager.py, dnallm/mcp/server.py, tests/mcp/test_model_manager.py, tests/mcp/test_interpret_tool.py, both notebook pairs — all present in commits below
- Commits bd5496a, 114390f, ae76a3e, fa19675, 9453d23 all present on phs (verified via git log)
- Full-suite fast lane: 1716 passed / 1 skipped — parity + 13 new, zero failures
