---
phase: 02-suite-hygiene-known-bug-fixes
reviewed: 2026-10-01T12:23:59Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - .github/workflows/ci.yml
  - .github/workflows/README.md
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 2: Code Review Report (Iteration 4 — final incremental re-review, CR-03 fix)

**Reviewed:** 2026-10-01T12:23:59Z
**Depth:** standard
**Files Reviewed:** 2 (increment since d07a716: commit de4b5cc only)
**Status:** issues_found (0 Critical, 2 Warning, 1 Info)

## Summary

Final incremental re-review scoped to the only source change since d07a716:
commit de4b5cc (CR-03), which switches the nightly `test-mamba` install from
`.[test,dev]` to `.[base]`, updates the 180-min timeout comment, and rewrites
the README step-5 line. Cumulative phase scope and prior findings live in the
iteration-3 report (superseded by this file per orchestrator instruction); IDs
continue that ledger.

**The CR-03 increment is verified correct at the source — every claim in the
new comment checks out:**

- **Strict superset:** `pyproject.toml` defines `base =
  ["dnallm[dev,test,notebook,mcp]", "isort>=6.0.1", "types-transformers>=0.1.0"]`
  and `dev = ["dnallm[test,notebook]", ...]`, so `.[base]` ⊇ `.[test,dev]`
  plus the `mcp` extra. "Same extras set the other legs install" holds: test,
  test-windows, coverage-gate, coverage-nightly all install plain `.[base]`;
  test-cuda adds only a torch-index extra; deploy adds only `docs`.
- **The 3 deterministically failing tests are real and now fixed:**
  (1) `test_notebook_imports[mcp_example/mcp_client_ollama_langchain_agents.ipynb]`
  and (2) `test_notebook_imports[mcp_example/mcp_client_ollama_pydantic_ai.ipynb]`
  import `langchain`/`langchain_mcp_adapters`/`nest_asyncio` and
  `pydantic_ai`/`nest_asyncio` respectively; `tests/examples/test_examples.py`
  `pytest.fail`s on any missing import outside `OPTIONAL_IMPORT_MODULES`
  (only `pybedtools`), so these are hard failures — not skips — without the
  mcp extra. (3) `test_connection_failure_surfaces_through_exception_group`
  (`tests/mcp/test_client_sdk.py:585`) does
  `from exceptiongroup import ExceptionGroup` inside the test body.
- **The exceptiongroup provenance chain is verified against installed
  metadata:** on the py3.11 matrix every other requirer of `exceptiongroup`
  (pytest, anyio, pydantic-ai-slim) is gated `python_version < '3.11'` and
  thus inert; the unconditional route is exactly the comment's chain
  `pydantic-ai` → `pydantic-ai-slim[mcp]` → `fastmcp-slim[client]` →
  `exceptiongroup>=1.2.2; extra == 'client'` (no python_version marker).

**Phase-02 hygiene lens (skip typing / allowlist / junit audit wiring): the
increment is neutral-to-positive; nothing regresses.**

- The diff touches only the `test-mamba` install step, two comments, and one
  README line. The four audited jobs and their blocking audit steps are
  byte-identical to before: `test` (ci.yml:93-96), `test-windows`
  (ci.yml:172-175), `coverage-gate` (ci.yml:398-401), `coverage-nightly`
  (ci.yml:488-491) — each runs `python scripts/audit_skips.py <junit>
  tests/expected_skips.yaml` with no `continue-on-error`; `audit_skips.py`
  remains fail-closed (ValueError on malformed/empty allowlist entries,
  exit 1 on unmatched skip messages).
- All four audited jobs already installed `.[base]`, so their skip sets
  cannot change due to this commit. `tests/expected_skips.yaml` needs no
  edit: no allowlisted entry depends on the mcp extra's absence. The
  `MCP client modules not available:` optional-dep entry concerns the mcp
  SDK itself, which is a *core* dependency (`pyproject.toml:18`,
  `mcp>=1.3.0,<2`) and was present even under the old `.[test,dev]` install.
- The change converts 3 hard failures into passes on `test-mamba` and
  introduces zero new skips anywhere: all three tests fail (rather than skip)
  when the deps are missing, so the fix removes failures without touching
  skip semantics. `test-mamba` itself has no junit/audit step (pre-existing,
  accepted design — its skip surface now converges with the audited `test`
  job's, since the extras sets are now identical).
- The `continue-on-error` removal on `test-mamba` remains intact: the only
  occurrences in ci.yml are the explanatory comment (line 322) and the
  explicit `continue-on-error: false` on coverage-nightly (line 421).

**Two Warnings, both about the *residue* of the retired `.[test,dev]` extras
set rather than the increment itself:** the README's Local Testing section
still instructs contributors to install the exact extras set CR-03 just
removed (WR-07), and — discovered while verifying the README claim — the
`docs-validation.yml` workflow still installs `.[test,dev]` and runs the
example-import tests that deterministically fail without the mcp extra, with
the failure masked by `continue-on-error: true` (WR-08, file outside this
round's declared scope, disclosed as such). IN-10 (`actions/cache@v3` in the
deploy job) is carried forward unchanged: its file is in this round's scope
and the finding still holds at what is now line 517.

**Ledger continuity:** IDs continue iteration 3 (WR-05/WR-06, IN-02..IN-12
used). Still open but with files outside this round's 2-file scope, so not
re-issued as findings here (per iteration-3 precedent): WR-05
(metrics.py nested r2 dict), WR-06 (task_type alias normalization),
IN-02 (.gitignore duplicates), IN-06 (test_sse_client return True), IN-07
(vacuous fp16/bf16 validator tests), IN-08 (models.lock stale entry), IN-09
(untested multilabel AUROC/AUPRC guards), IN-11 (two untyped defensive
skipTests), IN-12 (mkdtemp leaks in MCP config tests).

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

## Warnings

### WR-07: README "Local Testing" still tells contributors to install the retired `.[test,dev]` extras set — the exact configuration CR-03 just removed from CI for deterministically failing the fast census

**File:** `.github/workflows/README.md:205` (Local Testing block, lines
199-229); contradicts `.github/workflows/README.md:78` and
`.github/workflows/ci.yml:308-319` from this same commit
**Issue:** The step-5 line added by de4b5cc correctly documents that
`.[base]` — specifically its `mcp` extra — "is required by the not-slow
census: the `mcp_example` notebook-import tests and the `exceptiongroup`
backport." But the Local Testing section 200 lines below still opens with
`uv pip install -e ".[test,dev]"` (line 205) and then instructs the reader
to run the very census that needs those deps: "Fast census (what the
coverage gate runs): `pytest -m "not slow" ...`" (line 219). A contributor
who follows the README verbatim reproduces the CR-03 failure mode locally —
2 `test_notebook_imports[mcp_example/...]` hard failures plus
`test_connection_failure_surfaces_through_exception_group` (ModuleNotFoundError
on `exceptiongroup`) — and will reasonably conclude the repo is broken. The
same document now asserts two mutually incompatible installation
requirements. (The same stale string also survives in
`.github/workflows/docs-validation.yml:42` — see WR-08.)
**Fix:**
```bash
# .github/workflows/README.md, Local Testing section
# Install development dependencies (same extras set CI installs; the mcp
# extra is required by the not-slow census — see job `test-mamba`, step 5)
uv pip install -e ".[base]"
```

### WR-08: `docs-validation.yml` still installs `.[test,dev]` and runs `tests/examples/test_examples.py` — the "Run example tests" step deterministically fails on every push/PR, masked by `continue-on-error: true` (discovered out-of-increment while verifying WR-07's README claim; file is outside this round's declared scope, disclosed as such)

**File:** `.github/workflows/docs-validation.yml:42` (install) and `:62-66`
(Run example tests, `continue-on-error: true`)
**Issue:** This workflow triggers on every push/PR to main/master/dev, creates
a py3.11 venv with `uv pip install -e ".[test,dev]"` — no mcp extra — and
then runs `pytest tests/examples/test_examples.py -v`. That module's
`test_notebook_imports` collects the two `example/mcp_example/*.ipynb`
notebooks, whose imports (`langchain`, `langchain_mcp_adapters`,
`pydantic_ai`, `nest_asyncio`) are not in `OPTIONAL_IMPORT_MODULES`
(only `pybedtools`), so both parametrizations `pytest.fail` — the identical
defect CR-03 fixed in ci.yml's `test-mamba` leg. Because the step carries
`continue-on-error: true`, the workflow stays green while its example-test
validation has been permanently red (validating nothing) since the
mcp-extra-importing notebooks entered the example-import census. This is
precisely the exit-code-masking class phase 02 polices (ci.yml carries a
dedicated exit-code canary for it). No skip-audit interaction: the step
produces no junit, and the failures are failures, not skips — but the
phase's "a failing test must fail the job" guarantee is violated in spirit
here.
**Fix:** Mirror the CR-03 fix and drop the masking in one change:
```yaml
# docs-validation.yml
- name: Create virtual environment and install dependencies
  ...
  run: |
    uv venv
    uv pip install -e ".[base]"     # mcp extra needed by test_examples.py
...
- name: Run example tests
  run: |                             # drop continue-on-error once green
    source .venv/bin/activate
    pytest tests/examples/test_examples.py -v
```
(If the advisory posture of the whole workflow is intentional, at minimum
fix the install so the step fails only on real regressions.)

## Info

### IN-10 (carried from iteration 3): `deploy` job pins `actions/cache@v3` while every other job uses `@v4`

**File:** `.github/workflows/ci.yml:517` (was 510 before this increment's
comment additions shifted the file)
**Issue:** Version skew in the action set (test/gate/nightly jobs all use
`actions/cache@v4`, `actions/upload-artifact@v4`). v3 of the actions cache
toolkit family is the deprecated major; staying on it risks future brownouts
and misses v4's cache-size accounting fixes. Re-verified unchanged this
round — ci.yml is in this round's scope and the `uses: actions/cache@v3`
pin is still present under the deploy job's "Cache mkdocs dependencies"
step.
**Fix:** Bump the deploy job's mkdocs cache step to `actions/cache@v4`.

---

_Reviewed: 2026-10-01T12:23:59Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (iteration 4 — final incremental, diff_base d07a716)_
