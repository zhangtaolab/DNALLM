---
phase: 01-harness-integrity-measured-baseline
reviewed: 2026-10-01T12:23:09Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - .github/workflows/ci.yml
  - .github/workflows/README.md
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 01: Code Review Report (Incremental Re-Review — Fix Round 4)

**Reviewed:** 2026-10-01T12:23:09Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found
**Scope:** Incremental re-review scoped to `eac58e8..HEAD`, whose only source delta is commit de4b5cc (CR-03 fix): the nightly `test-mamba` leg's install switched from `.[test,dev]` to `.[base]`, plus the matching README step-5 line and a timeout-comment update. Phase 01 harness-integrity lens: does anything here weaken single-config pytest semantics, honest exit codes, or the canary?

Finding IDs continue from the phase ledger (previous rounds reached WR-08 / IN-11) so the disposition record's existing rows are not clobbered by ID reuse.

## Summary

The CR-03 fix is **correct and complete for the CI leg**; every factual claim in the new ci.yml comment block and the README step-5 line was independently verified against source and installed metadata. No Critical issues. The harness-integrity lens is clear: the pytest invocation (`pytest tests/ -v -m "not slow" --tb=short`), `set -o pipefail`, the absence of `continue-on-error`, the artifact-upload condition (`always() && steps.mamba-tests.outcome == 'failure'`), and the exit-code canary steps are all untouched by this delta — and the change eliminates the last extras-set divergence among the legs that run the not-slow census, which *strengthens* single-config semantics. One Warning and one Info remain, both documentation-accuracy items in the changed files' orbit.

**de4b5cc (CR-03, ci.yml:308-320) — verified correct, every claim traced to source:**

- **Strict-superset claim holds.** `pyproject.toml:120-123`: `base = ["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]`, with `dev` (line 81) itself pulling `dnallm[test,notebook]`. So `.[base]` = `.[test,dev]` + the `mcp` extra + two small tools; no dependency is lost. Critically, all plugins required by the shared `[tool.pytest.ini_options]` addopts (`--asyncio-mode=auto`, `--timeout=300`) arrive via the `test` extra (pyproject.toml:92-99: pytest, pytest-asyncio, pytest-cov, pytest-progress, pytest-timeout), so the ci.yml:322-325 claim that hung runs are bounded by the per-test 300s timeout remains true.
- **The exceptiongroup mechanism is real and the chain is described exactly right.** `tests/mcp/test_client_sdk.py:585` does a function-local `from exceptiongroup import ExceptionGroup` inside `test_connection_failure_surfaces_through_exception_group` — a hard failure (not a skip) because py3.11+ has no importable module of that name. Installed metadata confirms the ci.yml:310-314 chain verbatim: `pydantic-ai 1.102.0` → `pydantic-ai-slim[...,mcp,...]` → `fastmcp-slim[client]>=3.3.0` → `exceptiongroup>=1.2.2; extra == 'client'` — and that fastmcp-slim extras are the **only unconditional** requirers of exceptiongroup on py3.11+ (every other requirer is `python_version < '3.11'`-gated), so the backport genuinely arrives only via the `mcp` extra.
- **The mcp_example notebook claim is real.** `tests/examples/test_examples.py:63` collects `example/mcp_example/*.ipynb`; `test_notebook_imports` (lines 227-272) `exec`s each extracted import statement and `pytest.fail`s on `ModuleNotFoundError`. Both notebooks (`mcp_client_ollama_langchain_agents.ipynb`, `mcp_client_ollama_pydantic_ai.ipynb`) import exactly the modules the comment names: `langchain`, `langchain_mcp_adapters`, `pydantic_ai`, `nest_asyncio`. Under the old `.[test,dev]` these fail deterministically — matching the CR-03 live log ("3 failed, 1580 passed" with notebook-import failures).
- **README:78 step-5 line is accurate.** The `(dev,test,notebook,mcp)` enumeration matches the pyproject `base` definition exactly; "the same extras set the other legs use" matches ci.yml:68/157/252-254/386/470 (all census legs install `.[base]`, some with cuda/docs added).
- **No collateral damage.** ci.yml parses as valid YAML; the timeout-comment edit (ci.yml:274) is consistent — `base` adds only the 5 pure-Python mcp packages plus isort/types-transformers over the old set (jupyter/marimo were already present via `dev → dnallm[test,notebook]`), negligible against the 180-min kernel-build budget.

**Residual gap:** the fix corrected the CI leg and its job-section description, but the README's own *Local Testing* section still prescribes the exact `.[test,dev]` setup that CR-03 just proved deterministically fails the census — see WR-09. One comment sentence in ci.yml overstates its evidence — see IN-12.

## Critical Issues

None.

## Warnings

### WR-09: README "Local Testing" still prescribes the `.[test,dev]` install that CR-03 just proved fails the documented census commands

**File:** `.github/workflows/README.md:205`
**Issue:** The Local Testing section opens with `uv pip install -e ".[test,dev]"` directly above commands framed as CI replication ("Run quality checks (what CI runs)", line 210; "Census of record (what coverage-nightly runs)", line 215; "Fast census (what the coverage gate runs)", line 218). A contributor who follows this verbatim deterministically reproduces the exact CR-03 failure class locally: (a) `tests/examples/test_examples.py` `test_notebook_imports` params for both `example/mcp_example` notebooks `pytest.fail` on missing `langchain`/`langchain_mcp_adapters`/`pydantic_ai`/`nest_asyncio`; (b) `tests/mcp/test_client_sdk.py::test_connection_failure_surfaces_through_exception_group` raises `ModuleNotFoundError: exceptiongroup` on py3.11+ (the backport only arrives transitively via the `mcp` extra, verified against installed dist metadata). No CI leg installs `.[test,dev]` anymore — every census-running leg installs `.[base]` — so the README's local path and its own "what CI runs" framing have diverged from every CI leg. This is the sole stale instance: root `README.md:109,135` already says `.[base]`, and `CONTRIBUTING.md` / `tests/TESTING.md` carry no extras-install line (verified by grep).
**Fix:**
```bash
# Install development dependencies (the extras set every CI census leg uses)
uv pip install -e ".[base]"
```
(Optionally add a one-line note that the nightly mamba leg uses `.[base,mamba]` on the self-hosted GPU box.)

## Info

### IN-12: New ci.yml comment overstates cross-version evidence — "coverage-nightly proves .[base] resolves green on this exact box"

**File:** `.github/workflows/ci.yml:315-316`
**Issue:** The sentence "coverage-nightly proves .[base] resolves green on this exact box" is interpreter-blind: `coverage-nightly` runs Python 3.12 (ci.yml:439) while `test-mamba` runs Python 3.11 (ci.yml:282), and dependency resolution is per-interpreter. What is proven *on that exact box* is a py3.12 resolve; the py3.11 `.[base]` resolve is proven by the hosted `test` matrix leg (ci.yml:28 python 3.11, ci.yml:68 `.[base]`) — a different box and arch. Practical risk is nil (every package in the delta is a pure-Python wheel, so py3.11/py3.12/aarch64 resolution cannot meaningfully diverge), but the comment hands a future maintainer a stronger guarantee than the evidence supports, in the same accuracy class the ledger has tracked as IN-tier.
**Fix:** Reword to split the evidence, e.g.: "Same extras set the other legs install; the py3.12 coverage-nightly leg proves .[base] resolves green on this exact box, and the py3.11 hosted test-matrix leg proves the same resolve on 3.11."

---

_Reviewed: 2026-10-01T12:23:09Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
