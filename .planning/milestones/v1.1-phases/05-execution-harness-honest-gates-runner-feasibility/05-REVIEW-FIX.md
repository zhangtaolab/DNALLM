---
phase: 05-execution-harness-honest-gates-runner-feasibility
fixed_at: 2026-10-06T15:05:38Z
review_path: .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 5: Code Review Fix Report

**Fixed at:** 2026-10-06T15:05:38Z
**Source review:** `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md` (incremental re-review at HEAD, 2026-10-06T14:51:10Z)
**Iteration:** 1
**Mode:** `workflow.use_worktrees = false` — all edits and commits made directly in the main checkout on `phs`; branch pushed (`3215fd6..9d44cbf`).

**Summary:**
- Findings in scope: 4 (fix_scope: all — 0 critical, 2 warnings, 2 info)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: Model-id swap incomplete — Prerequisites still pulls `qwen3.6:latest`

**Files modified:** `docs/example/mcp_pydantic_ai.md`, `docs/example/mcp_langchain.md`
**Commit:** `0ff9ee4`
**Applied fix:** Changed the Prerequisites bash block in both tutorial wrappers
(`mcp_pydantic_ai.md:21`, `mcp_langchain.md:23`) from `ollama pull qwen3.6:latest` to
`ollama pull qwen3.5:4b`, matching the model ids already used by the code blocks and the
mirrored notebooks. Verified `qwen3.6` no longer appears anywhere under `docs/` or
`example/`. Markdown prose lines only — the md-sync checkers parse Python code blocks
exclusively, so no mirror churn (as the review noted).

### WR-02: ACTIVE-lane sandbox fixture ignores the spec `yaml_patch` key the gated lane forwards

**Files modified:** `tests/examples/test_notebook_execution.py`
**Commit:** `d03ab4d`
**Applied fix:** The `notebook_sandbox` fixture now mirrors `gated_sandbox`: it looks up
`NOTEBOOK_EXEC_SPECS[str(nb_path)]` and forwards `yaml_overrides=spec.get("yaml_patch")`
to `seed_sandbox`, so a future `yaml_patch` on an ACTIVE notebook can no longer be
silently ignored. D-05 fail-closed semantics intact (a bad patch still raises
`ValueError` from `seed_sandbox`). Verified as a runtime no-op today: all 13
`ACTIVE_NOTEBOOKS` entries resolve in `NOTEBOOK_EXEC_SPECS` (no KeyError risk at
fixture time) and none currently carries a `yaml_patch`. No new fast-lane pin was added:
the fixture is only exercised on the slow execution lane and there is no in-genre
source-inspection precedent in this file; the forwarding is covered by every future
ACTIVE-lane execution.

### IN-01: `yaml_overrides` on a non-mapping section raises `AttributeError`, not the documented `ValueError`

**Files modified:** `tests/examples/_execution.py`, `tests/examples/test_notebook_execution.py`
**Commit:** `374e8e6`
**Applied fix:** Widened the fail-closed guard in `seed_sandbox` to
`if not isinstance(data, dict) or not isinstance(data.get(section), dict): raise
ValueError(...)` (message: "section ... absent or non-mapping ... refusing to patch"),
so a null/scalar section — or a whole-file null/scalar — raises the documented
`ValueError` instead of `AttributeError` at `.update()`. Updated the block comment and
the docstring `Raises:` clause to match. Added the pinning test
`TestSeedSandboxYamlOverrides::test_non_mapping_override_section_raises` (a YAML
`null_section:` parsing to `None` must raise `ValueError` naming the section); the two
pre-existing fail-closed pins still pass unchanged.

### IN-02: Prerequisite probes let `subprocess.TimeoutExpired` escape instead of reporting `(False, evidence)`

**Files modified:** `tests/examples/_execution.py`, `tests/examples/test_notebook_execution.py`
**Commit:** `9d44cbf`
**Applied fix:** Both `megadna_prerequisites_installed()` and
`evo_prerequisites_installed()` now wrap their `subprocess.run(..., timeout=120)` in
`try/except subprocess.TimeoutExpired`, returning
`(False, f"venv probe timed out after 120s ({venv_python}): {exc}")` — the honest typed
skip with probe evidence, never a test ERROR. The `# ruff:
ignore[subprocess-without-shell-equals-true]` directive moved inside the `try` so it
still binds to the `subprocess.run` statement (ruff flagged the orphaned directive
otherwise). Updated the evo docstring `Returns:` clause to mention the hung-interpreter
case. Added the hermetic pin `TestVenvProbeTimeoutContract` (fake venv dir +
monkeypatched hanging `subprocess.run` on the defining module; asserts both probes
return `False` with the timeout evidence naming the interpreter path).

## Verification

All verification ran in the MAIN checkout (no worktree was created,
`workflow.use_worktrees = false`), so every number below is reproducible from `phs`
at `9d44cbf`.

Per-fix:
- Tier 1 (all four): re-read / diff-inspected every edited region; diffs scoped to
  exactly the finding.
- Tier 2 (python files): `ast.parse` clean; `ruff check` and `ruff format --check`
  clean on both touched files (ruff 0.16.10). WR-01 is Markdown prose — Tier 1 only,
  per fallback rules.
- WR-02 extra: asserted all 13 `ACTIVE_NOTEBOOKS` paths resolve in
  `NOTEBOOK_EXEC_SPECS` and none carries `yaml_patch` today.

Fast-lane test evidence (project venv, slow/giants lanes NOT run, per instruction):

```
.venv/bin/python -m pytest tests/examples/test_notebook_execution.py \
  tests/examples/test_script_execution.py tests/examples/test_marimo_execution.py \
  -m "not slow and not giants" -q --tb=short
→ 46 passed, 27 deselected in 5.44s
```

(Baseline in 05-REVIEW.md was 44 passed / 27 deselected; the +2 are the new IN-01 and
IN-02 pins.) Targeted runs during fixing: `TestSeedSandboxYamlOverrides` 6 passed;
`TestVenvProbeTimeoutContract` + `TestMegadnaIsolatedLane` + `TestEvoIsolatedLane`
8 passed.

## Skipped Issues

None — all four in-scope findings were fixed.

---

_Fixed: 2026-10-06T15:05:38Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
