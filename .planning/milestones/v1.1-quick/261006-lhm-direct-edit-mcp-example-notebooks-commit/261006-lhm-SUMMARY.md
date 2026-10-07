---
phase: 261006-lhm
plan: 01
subsystem: mcp-example-notebooks
tags: [mcp, ollama, model-swap, docs-sync, runner-ops, contract-tests]
requires:
  - "D-11 original same-model decision (superseded by this re-decision)"
  - "00:52 CST num_ctx deferral (stands; unit pin inert)"
provides:
  - "Committed MCP client example pair + all mirrors referencing qwen3.5:4b"
  - "TestMcpExampleModelSwap revert pin (3 fast contract tests)"
  - "Runner README ops path pulling/verifying qwen3.5:4b"
affects:
  - nightly example stage-3 gated mcp pair (executes against qwen3.5:4b)
  - runner box rebuild instructions (scripts/runner/README.md)
tech-stack:
  added: []
  patterns:
    - "JSON-aware structure-preserving notebook rewrite (stdlib json.dumps indent=1 roundtrip guard + per-file replacement-count asserts)"
key-files:
  created: []
  modified:
    - tests/test_runner_infra_contracts.py
    - example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
    - example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
    - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
    - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
    - docs/example/mcp_pydantic_ai.md
    - docs/example/mcp_langchain.md
    - scripts/runner/README.md
    - .github/workflows/ci.yml
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py
decisions:
  - "D-11 re-decision 2026-10-06 15:27 CST recorded in STATE.md: same-model constraint lifted; direct committed-content edit (supersedes the discarded sandbox-seam plan)"
  - "Notebook rewrite counts replacement LINES (pydantic 2, langchain 1) per the plan's per-file counts; token occurrences were 3/1 (markdown line carries prose + pull command)"
metrics:
  duration: 9 min
  completed: 2026-10-06
status: complete
actuals:
  tokens: 11000
  tasks: 3
  commits: 3
  plan_head_before: aa4a6e7bcfccd6b6e60bc7b6bae9819ca108f717
  plan_head_after: 0a5ca0de58bd6abc4a34e2273d679036a4d0d19c
---

# Quick Task 261006-lhm: MCP example notebooks model swap (qwen3.8 -> qwen3.5:4b) Summary

Direct committed-content swap of the MCP client example notebooks' agent-brain
model to qwen3.5:4b across all repo-facing surfaces, pinned by a RED->GREEN
contract test class, with sync gates green and phs pushed (owner decision
2026-10-06 15:27 CST; capability probe PASSED 15:24 CST — 3-turn tool-calling,
34.8s cold / 6.1s / 5.1s warm, cited not re-run).

## What Was Done

| Task | Commit | Description |
| ---- | ------ | ----------- |
| 1 (RED) | dc44856 | TestMcpExampleModelSwap pin class (3 kernel-free tests) added to tests/test_runner_infra_contracts.py; proven failing at test level against the pre-swap tree |
| 2 (GREEN) | b91f2a6 | Notebooks swapped via JSON-aware rewrite; docs mirrors byte-copied; md pages hand-edited; runner README ops + narrative restated; one ci.yml comment line |
| 3 | 0a5ca0d | Harness narrative alignment (comment/docstring-only) in the two example-harness files + contracts module docstring; STATE.md D-11 bullet appended (uncommitted, orchestrator owns docs commit); push |

## RED -> GREEN Evidence

- RED (Task 1, test-level failures per the 06-02 RED_EVIDENCE_OK convention):
  `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py::TestMcpExampleModelSwap -q`
  → **3 failed, exit=1** (assertion failures: notebook literals, md/mirror
  literals, README pull step — no collection/import error). Pre-existing 5
  contract tests: **5 passed**.
- GREEN (Task 2): `tests/test_runner_infra_contracts.py` → **8 passed**
  (5 pre-existing + 3 new).

## Gate Outputs

- `check_notebook_md_sync.py`: Checked 24 markdown/notebook pair(s); all in
  sync (24/24) — run after Task 2 and again in the final battery.
- `check_docs_sync.py`: OK, docs/example/ in sync with example/ (mirrors
  byte-identical, proven by `cmp`).
- Scoped sweep: `grep -R "qwen3.8" example/mcp_example docs/example/mcp_example
  docs/example/mcp_pydantic_ai.md docs/example/mcp_langchain.md
  scripts/runner/README.md` → empty (**SWAP-SWEEP-CLEAN**).
- Final kernel-free battery (Task 3): `pytest tests/mcp
  tests/test_runner_infra_contracts.py -q` → **231 passed, 1 warning, 73.65s**.

## Notebook Edit Minimality

The rewrite script asserted a byte-roundtrip (`json.dumps(nb, indent=1,
ensure_ascii=False)` + original trailing newline) BEFORE editing, asserted
per-file changed source lines (pydantic 2, langchain 1; token occurrences 3/1
— the markdown line carries both the availability prose and the pull command),
and `git diff` showed only the three intended source lines across both
notebooks. Outputs, execution_count, metadata, and cell ids are byte-preserved.

## Output Provenance

The committed notebook OUTPUTS were NOT fabricated, refreshed, or cleared.
They remain as-executed evidence from the previous model's run; the next full
nightly re-execution refreshes them against qwen3.5:4b.

## ci.yml Precondition Outcome

The porcelain precondition HELD (`git status --porcelain
.github/workflows/ci.yml` empty immediately before editing), so the edit
proceeded: exactly one comment line changed (stage-2.5 settle note now cites
qwen3.5:4b's ~3.3GB instead of the 17GB figure); `git diff --stat` confirmed
1 insertion / 1 deletion, no step logic or keys touched. No other .github/
file was modified.

## Push Evidence

- Push: `git push origin phs` → `aa4a6e7..0a5ca0d phs -> phs` (first attempt,
  no retry needed).
- Verification: `git rev-parse HEAD` = `git rev-parse origin/phs` =
  `0a5ca0de58bd6abc4a34e2273d679036a4d0d19c` (**PUSHED-AND-GREEN**).
- Note: origin's pre-task state was aa4a6e7 (one commit behind local start
  HEAD febb627 — the planner's plan-file commit, also pushed by this push).
- Diff-scope audit (origin/pre-task..HEAD): exactly the 11 files_modified plus
  the planner's `.planning/quick/261006-lhm-.../261006-lhm-PLAN.md` record;
  ci.yml present as the single comment line only.

## Deviations from Plan

### Observations (no action required)

**1. [Inventory discrepancy] scripts/runner/ollama.service also carries the old token (comments only)**
- **Found during:** Task 2 pre-sweep
- **Issue:** The plan's verified-at-planning inventory claimed the ONLY files
  carrying the old model string were the 11 in files_modified; grep found a
  12th — `scripts/runner/ollama.service` (3 comment lines describing the D-06
  num_ctx rationale).
- **Disposition:** Left untouched deliberately. ollama.service is not in
  files_modified, is excluded from every sweep scope and must-have truth, and
  the plan explicitly keeps the unit "unchanged and inert pending the owner
  re-apply" (00:52 CST deferral). Editing it was not instructed and would
  churn the auditable unit source for no behavioral gain.

**2. [Interpretation] replacement-count asserts counted changed source lines**
- The plan's "assert per-file replacement counts (pydantic 2, langchain 1)"
  matches changed source LINES (the plan's own occurrence map describes the
  markdown line as carrying two occurrences); token occurrences are 3/1. The
  script asserted lines (2/1) and reported tokens (3/1) — both recorded.

## Standing Fences Honored

- No systemd/ollama live configuration touched; no model cache deleted; no
  GitHub runs dispatched or cancelled; the 15:24 CST capability probe cited,
  not re-run. D-12 loopback / never-0.0.0.0 wording byte-preserved in
  scripts/runner/README.md. Pre-existing unit pins
  (OLLAMA_CONTEXT_LENGTH=8192, loopback host) untouched and still passing.

## Self-Check: PASSED

All 12 key files (11 files_modified + SUMMARY.md) exist on disk; all three
task commits (dc44856, b91f2a6, 0a5ca0d) are ancestors of HEAD; the STATE.md
D-11 re-decision bullet present; origin/phs == HEAD verified above.
