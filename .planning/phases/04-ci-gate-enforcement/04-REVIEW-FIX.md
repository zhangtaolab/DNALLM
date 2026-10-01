---
phase: 04-ci-gate-enforcement
fixed_at: 2026-10-01T12:07:14Z
review_path: .planning/phases/04-ci-gate-enforcement/04-REVIEW.md
iteration: 1
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 04: Code Review Fix Report

**Fixed at:** 2026-10-01T12:07:14Z
**Source review:** .planning/phases/04-ci-gate-enforcement/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 1 (1 Critical, 0 Warning; fix_scope = critical_warning — the review's 3 Info findings IN-08/IN-09/IN-10 are out of scope)
- Fixed: 1
- Skipped: 0

## Fixed Issues

### CR-03: Nightly test-mamba leg ships deterministically red — env lacks the `mcp` extra (and `exceptiongroup`) for 3 fast tests it runs

**Files modified:** `.github/workflows/ci.yml`, `.github/workflows/README.md`
**Commit:** de4b5cc
**Applied fix:** Exactly the fix the review prescribes. The `test-mamba` job's install
step (`ci.yml:308-320`) now installs the base extras set the other legs use before the
kernel build: `uv pip install -e ".[base]"` (was `.[test,dev]`), followed unchanged by
`uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation`. A comment on the
step records why `.[base]` is required (the not-slow census imports mcp-extra deps),
and the job's 180-min timeout comment's stale `.[test,dev]` mention was updated to
`.[base]`. The README's test-mamba step 5 (`.github/workflows/README.md:78`) now says
`.[base]` with the same rationale instead of "Installs `.[test,dev]` plus `.[mamba]`".
No `continue-on-error` was re-added; the fail-fast gate semantics are untouched.

**Verification (all ran in the main checkout at `/home/forrest/Github/DNALLM`, `.venv`
— per `workflow.use_worktrees=false` this fixer ran sequentially with no worktree;
numbers are reproducible from that tree):**

- **Exact extras resolution against `pyproject.toml`** (lines 80-127):
  `base = ["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]`
  and `dev = ["dnallm[test,notebook]", ...]` — so `.[base]` is a strict superset of
  `.[test,dev]` (test + dev + notebook) plus the `mcp` extra
  (`mcp`, `langchain>=1.3.6`, `langchain_mcp_adapters>=0.2.1`, `nest-asyncio>=1.5.9`,
  `pydantic-ai<3`) plus isort/types-transformers. Covers failures 1 and 2 directly.
- **Failure 3 (`exceptiongroup`) chain verified, not assumed:** the review's
  "transitively via the mcp-extra dependency chain" claim was re-derived from installed
  metadata: `pydantic-ai` 1.102.0 unconditionally requires
  `pydantic-ai-slim[ag-ui,...,fastmcp,mcp,...]==1.102.0`;
  `pydantic-ai-slim` with its `mcp` extra requires `fastmcp-slim[client]>=3.3.0`;
  `fastmcp-slim` 3.4.7 requires `exceptiongroup>=1.2.2` under its `client`/`mcp`/`server`
  extras **with no python_version marker** — so a fresh py3.11 `.[base]` venv provides
  the backport that `tests/mcp/test_client_sdk.py:585` hard-imports (anyio/pytest only
  pull it on py<3.11, so the mcp-extra chain is indeed the carrier on 3.11).
- **Local import check** (main-checkout `.venv`, a `.[base]`-shaped env):
  `import langchain, langchain_mcp_adapters, nest_asyncio, pydantic_ai` and
  `from exceptiongroup import ExceptionGroup` all succeed — every dependency named by
  the three failing tests is importable in the target extras set.
- **YAML syntax:** `ci.yml` parses with PyYAML post-edit (7 jobs); the install step's
  run block reads exactly `uv venv` / `uv pip install -e ".[base]"` /
  `uv pip install -e ".[mamba]" --no-cache-dir --no-build-isolation`. The only
  remaining `test,dev` text in the workflow is the explanatory comment on the step.
- **Not verifiable here:** the `workflow_dispatch` rehearsal run on the `dnallm-nightly`
  box itself (kernel build + full not-slow census). `coverage-nightly` on that same box
  already proves `.[base]` resolves green there (dispatch run 36821471332). Per the
  review's own fix text, a dispatch run should still be triggered before the next
  03:00 UTC schedule fires to confirm the leg is green end-to-end — that is the one
  outstanding follow-up the fixer cannot perform.

## Skipped Issues

None — the single in-scope finding was fixed.

---

_Fixed: 2026-10-01T12:07:14Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
