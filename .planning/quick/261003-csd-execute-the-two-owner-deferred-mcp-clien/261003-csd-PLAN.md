---
quick_task: 261003-csd-execute-the-two-owner-deferred-mcp-clien
phase: quick-261003-csd
plan: 01
type: execute
wave: 1
depends_on: []
branch: phs
push: manual-only   # commits only — never push (owner instruction, matches sibling quick tasks)
autonomous: true
files_modified:
  - tests/examples/test_notebook_execution.py
  - tests/examples/_execution.py
  - example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
# Conditional (ONLY if the 8005 port fallback fires, owner-coordinated single commit):
#   - dnallm/mcp/configs/mcp_server_config.yaml
#   - docs/example/mcp_langchain.md
#   - docs/example/mcp_pydantic_ai.md
estimate:
  tokens: 60000
  raw_tokens: 36000
  tasks: 3
  confidence: med
must_haves:
  truths:
    - "tests/examples/test_notebook_execution.py `_gate_ollama_stack` executes when BOTH probes are green (owner decision D-08/T-05-16 given 2026-10-03 '先处理 mcp 的两个'): the pytest.fail fail-loudly branch is RETIRED and the skip evidence text no longer says 'deferred pending the Phase-8 ollama/VRAM coexistence plan' — the coexistence is owner-sanctioned"
    - "Either endpoint down still produces an honest typed `network-unavailable:` skip carrying both live probe results (prefix already registered in tests/expected_skips.yaml — no new skip type)"
    - "`_probe_http` counts ANY HTTP answer below 500 as reachable: the MCP streamable-http endpoint /mcp answers a bare GET with a 4xx status (session/method semantics), and urllib raises HTTPError on 4xx — without this fix a genuinely-up server probes 'down' and the gate would skip forever; proven by a fast-lane unit test against a local 405-returning http.server"
    - "The langchain notebook's `!uv pip install -U ...` cells NEVER touch the project venv: the notebook executes under a dedicated kernelspec `dnallm-mcp-langchain` whose kernel.json env pins VIRTUAL_ENV to the throwaway venv .scratch/mcp-example-venvs/langchain/ (.gitignore line 60 covers .scratch/); if that kernelspec is unresolvable, nbclient raises NoSuchKernel BEFORE any cell runs — fail-safe, the project venv is unreachable by construction; the pydantic_ai notebook (no install cells, verified) keeps the default project-venv python3 kernel"
    - "Both notebooks execute green end-to-end against the live stack: ollama qwen3.8:latest at localhost:11434 (UP at preflight; both notebooks already reference exactly this model name — verified, no model-name fix needed) and the dnallm MCP server started BY THE EXECUTOR at localhost:8000/mcp (owner directive 09:16: agent-driven bring-up, not user setup)"
    - "Port fallback is a documented conditional, not a pre-emptive switch: port 8000 measured FREE at 09:16 so it is the default; if bind conflicts at execution time, the switch to 8005 lands as ONE coordinated commit touching mcp_server_config.yaml (server.port AND streamable_http.port), the endpoint cells of BOTH notebook copies of BOTH mcp_example notebooks, the MCP_ENDPOINT constant at tests/examples/test_notebook_execution.py:324, and the localhost:8000 mentions in docs/example/mcp_langchain.md + docs/example/mcp_pydantic_ai.md (owner: 'mcp 服务器和 notebook 都要改'); ollama references at localhost:11434 never change"
    - "example/ and docs/ copies are byte-identical for both refreshed notebooks (committed with their post-green execution outputs, owner-sanctioned refresh); docs-validation sync gate green"
    - "Fast lane unaffected (baseline 1703 passed / 1 skipped) and the campaign stages ONLY the sanctioned files — never the unrelated residue (05-UAT.md, example/notebooks/benchmark+inference notebooks, .planning/ churn)"
  artifacts:
    - "tests/examples/test_notebook_execution.py: rewritten _gate_ollama_stack (execute-when-both-green), 4xx-tolerant _probe_http, retired T-05-16 sentinel in module/layer docstrings, new fast-lane gate/probe unit-test class"
    - "tests/examples/_execution.py: run_notebook(kernel_name=...) param (default 'python3' keeps the ACTIVE lane unchanged), NOTEBOOK_EXEC_SPECS langchain entry carries kernel_name 'dnallm-mcp-langchain', idempotent ensure_isolated_kernel() provisioning helper (venv + kernelspec + VIRTUAL_ENV pin)"
    - "example/mcp_example/mcp_client_ollama_langchain_agents.ipynb + docs mirror: cell 3 converted from the raw blocking `!dnallm mcp-server` invocation to a probe-then-ensure guard (skip when up; detached Popen + readiness wait when down; original command kept as a commented terminal alternative)"
    - "example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb + docs mirror: only such minimal API-alignment fixes as execution on pydantic-ai 1.102.0 actually demands (e.g. result.usage property-call style) — execute-first, no speculative edits"
    - ".planning/quick/261003-csd-.../261003-csd-SUMMARY.md recording: actual port used, per-cell wall times, evidence paths under .scratch/mcp-round/, whether the 8005 fallback fired, D-08 closure note"
  key_links:
    - "MCP_ENDPOINT constant (test_notebook_execution.py:324) <-> _gate_ollama_stack probes <-> live dnallm-mcp-server bind at :8000 (or :8005 after the coordinated switch) <-> both notebooks' localhost:8000/mcp endpoint cells"
    - "NOTEBOOK_EXEC_SPECS['...langchain_agents.ipynb'].kernel_name -> gated test passes it to run_notebook -> jupyter resolves kernelspec dnallm-mcp-langchain -> kernel.json env VIRTUAL_ENV=.scratch/mcp-example-venvs/langchain -> the notebook's `!uv pip install -U` cells target the throwaway venv (uv honors VIRTUAL_ENV when no --python is given)"
    - "ensure_isolated_kernel() idempotency contract: kernelspec resolvable AND its argv[0] interpreter exists -> no-op; either missing -> (re)create venv + `uv pip install --python <venv>/bin/python ipykernel nest-asyncio` + `ipykernel install --user --name dnallm-mcp-langchain` + post-edit kernel.json env block; provisioning failure with the gate green RAISES (environment claims readiness — that is an owner-visible failure, never a skip)"
    - "server bring-up runbook: from repo root (the argparse default config 'dnallm/mcp/configs/mcp_server_config.yaml' at dnallm/mcp/server.py:1944 is cwd-relative, and config_manager.py:69-72 resolves model config_path against the CONFIG FILE's dir) -> nohup .venv/bin/dnallm-mcp-server --transport streamable-http --host 127.0.0.1 --port 8000 > .scratch/mcp-server-bringup.log 2>&1 & -> readiness poll on http://localhost:8000/mcp treating 2xx/4xx as up -> record PID -> pre-warm lazy ModelManager (one list-tools/small call) so first-tool model loads do not eat notebook cell budgets -> kill by PID after the campaign"
---

# Quick Task: Execute the two owner-deferred MCP client notebooks (langchain + pydantic_ai) to green

Owner decision NOW GIVEN (D-08/T-05-16, 2026-10-03): these are the only 2 of 21
notebooks never executed. This round: bring up the ollama+MCP-server stack
(agent-driven per owner directive 09:16), move the gate to the owner-approved
execute state, isolate the langchain install cells from the project venv, and run
both notebooks to green in the gated pytest lane. The 5 remaining gated notebooks
(evo/megaDNA/mamba families) are OUT OF SCOPE (next round).

## Verified facts (live, this session — trust over re-derivation)

1. **Model names already match.** Both notebooks call exactly `qwen3.8:latest`
   (langchain cell 6 `"ollama:qwen3.8:latest"`, pydantic cell 3
   `model_name='qwen3.8:latest'`); ollama has `qwen3.8:latest` (17GB) loaded and
   answers at localhost:11434/api/tags. No model-name fix or ollama pull needed.
2. **Project venv readiness split.** pydantic_ai 1.102.0, nest-asyncio 1.6.0,
   langchain 1.4.3, langchain-mcp-adapters 0.3.2 all import in the project venv
   (pydantic_ai notebook needs nothing more — it has NO install cells, verified);
   `langchain_ollama` is NOT installed in the project venv (find_spec None) — the
   langchain notebook's cell-2 `!uv pip install -U langchain langchain-mcp-adapters
   langchain-ollama` must land in an isolated venv (T-05-16 hazard), which also
   supplies langchain-ollama.
3. **Langchain cell 3 is fatal as-authored, three ways.** `!dnallm mcp-server
   --transport streamable-http` (a) runs a BLOCKING uvicorn server in-cell (hangs to
   cell timeout), (b) if an external server already holds :8000 it dies fast on
   bind-conflict, (c) `dnallm` is not on the isolated venv's PATH by right. It MUST
   become a probe-then-ensure guard cell (see artifacts). The `dnallm mcp-server`
   subcommand itself EXISTS (`dnallm mcp-server --help` verified; click dash-converts
   `def mcp_server` at dnallm/cli/cli.py:305).
4. **The GET-probe blind spot.** MCP streamable-http endpoints answer bare GET with
   4xx; `_probe_http` (test_notebook_execution.py:327-334) folds every exception —
   including HTTPError 4xx — into "down". Without the 4xx-tolerant fix, a live server
   still probes down and the new execute-state gate would skip forever.
5. **Kernelspec landscape.** Only `python3` (.venv/share/jupyter/kernels/python3)
   exists; the isolated kernelspec is new and registers user-level
   (~/.local/share/jupyter/kernels/dnallm-mcp-langchain). `uv` resolves at
   .venv/bin/uv and kernels inherit the pytest process env, so `!uv` works inside the
   isolated kernel with VIRTUAL_ENV pinning the install target.
6. **Server config mechanics.** Config at dnallm/mcp/configs/mcp_server_config.yaml:
   server.port 8000 AND streamable_http.port 8000 (both must move together under the
   8005 fallback); three enabled models (promoter/conservation/open-chromatin =
   Plant DNAMamba BPE) loaded LAZILY by ModelManager; console-script argparse defaults
   at dnallm/mcp/server.py:1940-1981 (--config cwd-relative, --host 0.0.0.0, --port
   8000, --transport stdio). Launch from repo root; pass --host 127.0.0.1 explicitly
   (operational hardening only — the yaml host value is NOT part of the owner's
   port-switch commit list and stays untouched).
7. **Port ground truth (orchestrator, 09:16).** 8000 free, 8005 free -> default 8000,
   fallback conditional. Docs pages docs/example/mcp_langchain.md:34,58 and
   docs/example/mcp_pydantic_ai.md:55 reference localhost:8000/mcp and ride the
   fallback commit if it fires.
8. **Known benign mismatch (watch, do not pre-fix).** The pydantic notebook's system
   prompt names `_list_loaded_models`, but the server registers `list_loaded_models`
   (dnallm/mcp/server.py:260) — prompt text only; the agent sees true tool names via
   MCP. Fix only if execution actually derails on it.
9. **Residue policy.** Both target notebooks carry pre-existing ' M' residue (owner's
   earlier runs) — this task IS owner-sanctioned to commit refreshed copies of exactly
   these two example/docs pairs; everything else dirty stays unstaged.

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@tests/examples/test_notebook_execution.py
@tests/examples/_execution.py
@.planning/quick/261002-sl7-run-and-fix-the-5-non-gated-census-faili/261002-sl7-PLAN.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Execute-state gate + 4xx-honest probe + isolated langchain kernel machinery (fast lane, no endpoints needed)</name>
  <files>tests/examples/test_notebook_execution.py, tests/examples/_execution.py</files>
  <behavior>
    - Test (probe honesty): `_probe_http` against a local `http.server` bound to 127.0.0.1:0 whose handler returns 405 yields (True, evidence-containing "405"); against an unbound local port yields (False, ...). No external network.
    - Test (gate matrix): monkeypatched `_probe_http` outcomes — (ollama up, server up) returns None (execute state); (up, down), (down, up), (down, down) each raise the typed skip whose message starts with `network-unavailable:` and carries BOTH probe evidence strings; no code path reaches pytest.fail. Assert via pytest.raises(pytest.skip.Exception).
    - Test (kernel plumbing contract): the NOTEBOOK_EXEC_SPECS entry for mcp_client_ollama_langchain_agents.ipynb carries kernel_name "dnallm-mcp-langchain", the pydantic_ai entry carries none (defaults to python3), and run_notebook's signature exposes kernel_name defaulting to "python3".
  </behavior>
  <action>
    (a) `_probe_http` (tests/examples/test_notebook_execution.py:327): catch `urllib.error.HTTPError` BEFORE the generic except; any status below 500 returns (True, f"HTTP {status} ...") — an HTTP answer at any 4xx proves a server is bound there (verified fact 4); URLError/timeout/generic exceptions remain (False, verbatim evidence).
    (b) Rewrite `_gate_ollama_stack` (line 343): both probes green -> plain return (execute); otherwise one network_unavailable_skip naming the missing endpoint(s) with combined live evidence and phrasing per the D-08 owner decision of 2026-10-03 (owner-approved execute state requires both endpoints) — DELETE the pytest.fail both-up branch and the "additionally deferred pending the Phase-8 ollama/VRAM coexistence plan" clause. Update the function docstring, the GATED_NOTEBOOKS layer comment (line 312-324 area), and the module docstring: the T-05-16 never-auto-execute sentinel is retired by the owner decision; both-up now executes with owner-sanctioned ollama/VRAM coexistence.
    (c) tests/examples/_execution.py: add `kernel_name: str = "python3"` to run_notebook and pass it to NotebookClient (all existing callers unchanged via the default). Add `LANGCHAIN_KERNEL_NAME = "dnallm-mcp-langchain"` and `LANGCHAIN_VENV_DIR = REPO_ROOT / ".scratch" / "mcp-example-venvs" / "langchain"`; set `"kernel_name": LANGCHAIN_KERNEL_NAME` ONLY on the langchain spec entry (lines 157-160). Add idempotent `ensure_isolated_kernel()` per the must_haves key_links contract: venv create (uv venv or venv module), `uv pip install --python <venv>/bin/python ipykernel nest-asyncio` (nest-asyncio pre-seeded because notebook cell 4 imports it and cell 2 never installs it), `ipykernel install --user --name dnallm-mcp-langchain --display-name "Python (dnallm mcp langchain isolated)"`, then post-edit the generated kernel.json to add env VIRTUAL_ENV pointing at the venv (ipykernel's installer has no env flag); no-op when the kernelspec resolves AND its argv[0] interpreter exists; RAISE on provisioning failure (gate-green means the environment claims readiness — never skip). In the gated test body, resolve `kernel = spec.get("kernel_name", "python3")`, call ensure_isolated_kernel() first when kernel is LANGCHAIN_KERNEL_NAME, and pass kernel_name to run_notebook. RED-verify the three behavior tests first, then implement to GREEN.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/examples/test_notebook_execution.py -q -m "not slow" -k "Gate or Probe or Kernel or SeedSandbox" && .venv/bin/python -m pytest tests/examples/ -q -m "not slow" --timeout 600 | tail -3</automated>
  </verify>
  <done>Gate executes on both-green, typed-skips with live evidence otherwise, 4xx probes honest, langchain spec pinned to the isolated kernel with fail-safe NoSuchKernel semantics; new unit tests RED->GREEN in the same change; fast examples lane fully green (baseline parity).</done>
</task>

<task type="auto">
  <name>Task 2: Executor-driven MCP server bring-up (8000, 8005 conditional) + pre-known notebook fixes</name>
  <files>example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb, docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb</files>
  <precondition>Task 1 committed (gate + kernel machinery active) — this task starts live services.</precondition>
  <action>
    (a) Bring the server up (owner directive: agent-driven). From repo root: launch `nohup .venv/bin/dnallm-mcp-server --transport streamable-http --host 127.0.0.1 --port 8000 > .scratch/mcp-server-bringup.log 2>&1 &`, record the PID to .scratch/mcp-server-bringup.pid, then poll http://localhost:8000/mcp until it answers (2xx or 4xx per the new probe semantics), deadline 300s; on timeout diagnose from the log (model-registry/VRAM issues are owner-visible blockers, never silent skips). Pre-warm the lazy ModelManager with one cheap MCP call (list_loaded_models via a short python client using dnallm/mcp/client.py or a raw streamable-http POST) so first-tool model loads stay out of notebook cell budgets.
    (b) Port fallback (CONDITIONAL, owner pre-authorized 09:16): only if :8000 bind-conflicts at launch — relaunch with `--port 8005`, then land ONE coordinated commit switching everything to 8005: dnallm/mcp/configs/mcp_server_config.yaml (server.port AND streamable_http.port), every `localhost:8000/mcp` occurrence in BOTH copies of BOTH mcp_example notebooks (langchain cell 5 url + markdown mentions incl. the escaped variants; pydantic cell 4 url), MCP_ENDPOINT at tests/examples/test_notebook_execution.py:324, and the localhost:8000 mentions in docs/example/mcp_langchain.md + docs/example/mcp_pydantic_ai.md. Ollama references untouched. Fallback fired or not goes in the SUMMARY.
    (c) Langchain notebook cell 3 guard (pre-known fatal, verified fact 3): replace the raw `!dnallm mcp-server --transport streamable-http` body with an ensure-server cell — urllib-probe the endpoint constant (short timeout, 2xx/4xx = up) and print an already-running note when up; otherwise subprocess.Popen the console script detached (same host/port args as probed) and poll readiness with a deadline; keep the original `!` command as a commented one-liner for terminal users. One endpoint constant (URL/port) used for probe, message and spawn args so the 8005 switch (if it fires) stays a one-line change per copy. Stage example/ + docs/ byte-identical pair (edit via nbformat; all other cells untouched).
    (d) pydantic_ai notebook: NO speculative edits (execute-first). Anticipated likely fix spots if execution fails there: cell 7 `result.usage()` call-style vs property on pydantic-ai 1.102.0, and `result.output` attribute naming; any such fix is minimal and lands as the byte-identical pair. Before the campaign, verify package legitimacy for the two packages new to this repo (pypi.org/project/langchain-ollama — langchain org first-party; ipykernel — Project Jupyter): record both URLs in the SUMMARY (installs go ONLY to the throwaway venv; the project venv gains nothing).
  </action>
  <verify>
    <automated>curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8000/mcp; echo; for f in example/mcp_example/mcp_client_ollama_langchain_agents.ipynb example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb; do cmp "$f" "docs/$f" || exit 1; done && .venv/bin/python scripts/check_docs_sync.py && .venv/bin/python -m pytest tests/examples/test_notebook_execution.py -m "not slow" -q -k "Gate or Probe or Kernel" | tail -2</automated>
  </verify>
  <done>Server up (8000, or 8005 with the coordinated single commit landed), readiness + pre-warm evidenced in .scratch/mcp-server-bringup.log; langchain cell-3 guard committed as a byte-identical pair; docs sync gate green; port decision recorded.</done>
</task>

<task type="auto">
  <name>Task 3: Campaign — run both notebooks to green in the gated lane, refresh committed copies, teardown + both-direction gate evidence</name>
  <files>example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb, example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb, docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb</files>
  <precondition>Task 2 done: server answering at the chosen port, notebook fixes staged, isolated kernel machinery committed.</precondition>
  <action>
    (a) First execution: `.venv/bin/python -m pytest "tests/examples/test_notebook_execution.py::TestGatedNotebookExecution" -m slow -k "mcp_client" -q --timeout 7500`. First run also provisions the isolated venv via ensure_isolated_kernel() and populates it through the notebook's own install cells (uv install of the three langchain packages, well inside the 600s cell budget). Capture per-cell wall times to .scratch/mcp-round/.
    (b) Iterate failures to green — candidate rungs in likelihood order: langchain-mcp-adapters latest requiring an async-with context around client.get_tools() (fix: wrap cells 5-6 usage in `async with` — minimal, both copies); create_agent string-model resolution needing explicit ChatOllama; pydantic-ai 1.102 API deltas (Task 2d spots); agent-loop cell timeouts on the 17GB qwen3.8 (fix: bump that spec entry's cell_timeout with RECORDED wall-time evidence, keeping the outer pytest mark strictly above — 1800s cells stay under the 3600s class mark; only a 3600s cell joins _TIMEOUT_7200_GATED); MCP-server-side bugs surfaced by real traffic (any dnallm/ fix ships its pytest coverage in the SAME change — owner rule). Every notebook-side fix lands as the byte-identical example/docs pair.
    (c) Refresh the committed copies (owner-sanctioned): after green, re-run each notebook once through run_notebook (sandbox cwd) and write the RETURNED executed node byte-identically to both the example/ and docs/ paths, then stage exactly those two pairs plus the test-file changes — never the unrelated working-tree residue.
    (d) Teardown + both-direction evidence: kill the server by recorded PID, confirm the port freed; re-run the same two tests with the server down and ollama up — both must SKIP with the `network-unavailable:` prefix citing the live server-down probe (proves the new gate is honest in both directions, not just execute-happy). Final sweep: `.venv/bin/python -m pytest tests/examples/ -m "not slow" -q` (fast-lane parity), `.venv/bin/python scripts/check_docs_sync.py`, ruff format --check + ruff check on the two changed test files, tree-clean under example/ + docs/example/.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest "tests/examples/test_notebook_execution.py::TestGatedNotebookExecution" -m slow -k "mcp_client" -q --timeout 7500 && for f in example/mcp_example/mcp_client_ollama_langchain_agents.ipynb example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb; do cmp "$f" "docs/$f" || exit 1; done && .venv/bin/python scripts/check_docs_sync.py</automated>
  </verify>
  <done>Both mcp client notebooks PASS real end-to-end execution in the gated pytest lane with the live stack up; post-teardown the same tests typed-skip with live server-down evidence; refreshed output-carrying copies committed byte-identically on both sides; fast lane, docs sync, ruff all green; server stopped, port freed; only sanctioned files staged; commits on phs, no push.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| notebook install cells → filesystem/env | `!uv pip install` inside the langchain notebook could mutate the project venv that hosts the pytest kernel |
| localhost loopback | MCP server (bind 127.0.0.1 at launch) + ollama at :11434 exchange DNA sequences and LLM traffic on the box |
| internet → throwaway venv | unpinned `uv pip install -U langchain langchain-mcp-adapters langchain-ollama` fetches latest releases at notebook runtime |
| kernel subprocesses → repo tree | notebook kernels run with sandbox cwd; assert_tree_clean watches example/ + docs/example |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-mcp1-01 | Tampering | langchain notebook install cells vs project venv | high | mitigate | Dedicated kernelspec dnallm-mcp-langchain with kernel.json env VIRTUAL_ENV pinned to .scratch/mcp-example-venvs/langchain (uv honors VIRTUAL_ENV); spec-level kernel_name routing; NoSuchKernel raises before any cell executes if the kernelspec is missing — project venv unreachable by construction |
| T-mcp1-02 | Supply-chain | unpinned latest langchain/langchain-ollama installs | medium | mitigate | Installs land ONLY in the throwaway gitignored venv (docs fidelity: these are the notebook's own documented commands); project venv and pyproject untouched; legitimacy of the two packages new to this repo (langchain-ollama, ipykernel) verified via pypi.org provenance (langchain org / Project Jupyter) and recorded in the SUMMARY |
| T-mcp1-03 | DoS (resource exhaustion) | ollama 17GB + 3 DNA models coexisting on GB10 | medium | mitigate | Coexistence owner-sanctioned (D-08); DNA models are DNABERT-scale and lazy-loaded; pre-warm before measuring; server killed by PID immediately after the campaign; VRAM/OOM during bring-up is surfaced as an owner-visible blocker, never silently skipped |
| T-mcp1-04 | Information disclosure | MCP server default bind 0.0.0.0 exposes DNA-prediction tools on all interfaces | low | mitigate | Executor launches with explicit `--host 127.0.0.1` (notebooks probe localhost); short-lived process, torn down post-campaign; yaml host untouched (outside the owner's port-switch list) |
| T-mcp1-05 | Tampering | port-8005 fallback commit skew (config vs notebooks vs gate constant drifting apart) | medium | mitigate | Owner-specified single coordinated commit covers yaml (both port keys), both notebook copies' endpoint cells, MCP_ENDPOINT constant, and the docs .md mentions — verified by a post-switch grep for stray localhost:8000 references before committing |
| T-mcp1-SC | Tampering | package installs | high | mitigate | No package-manager installs into the project environment in this task; the only installs are the notebook's own documented cells into the throwaway venv plus ipykernel/nest-asyncio pre-seed into that same venv; if any project-env install becomes necessary mid-execution, stop and insert a blocking checkpoint:human-verify before installing (auto_advance ignored) |
</threat_model>

<verification>
- Task-level automated commands above (fast-lane gate/probe/kernel tests; live endpoint curl status; byte-identical cmp + docs sync; gated slow lane green with endpoints up; typed skips with endpoints down; fast-lane parity sweep).
- Evidence trail: .scratch/mcp-server-bringup.log + .pid, .scratch/mcp-round/ wall-times, SUMMARY records port decision, fallback-fired flag, pypi provenance URLs, per-notebook outcomes.
</verification>

<success_criteria>
- Both owner-deferred MCP client notebooks execute green end-to-end against the live ollama+MCP stack in the gated pytest lane (the 2-of-21 gap closed; 19 of 21 now executed).
- Gate semantics match the owner-approved execute state: both-up executes, any-down typed-skips with live evidence, fail-loudly sentinel retired; proven in both directions (up-run green, post-teardown skip).
- Project venv provably untouched by the install cells (isolated kernel + VIRTUAL_ENV pin; langchain_ollama still absent from the project venv afterwards — re-check with find_spec and record).
- Port decision executed per owner directive: 8000 default (measured free), 8005 fallback only as ONE coordinated commit if it fired.
- example/ ↔ docs/ byte-identical for both refreshed notebooks; docs sync gate green; fast lane at baseline; only sanctioned files staged; commits on phs, no push; D-08 closed in SUMMARY with the 5 remaining gated notebooks flagged as next round.
</success_criteria>

<output>
Create `.planning/quick/261003-csd-execute-the-two-owner-deferred-mcp-clien/261003-csd-SUMMARY.md` when done; surface the D-08 closure, the port decision (8000 vs 8005 and whether the coordinated commit landed), budget actuals vs the 600s starter cell budgets, and any dnallm/ fixes with their test coverage.
</output>
