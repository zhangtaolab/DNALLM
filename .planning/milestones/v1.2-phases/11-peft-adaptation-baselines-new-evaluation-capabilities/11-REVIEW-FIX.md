---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
fixed_at: 2026-10-09T15:34:26Z
review_path: .planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-REVIEW.md
iteration: 2
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 11: Code Review Fix Report

**Fixed at:** 2026-10-09T15:34:26Z
**Source review:** `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-REVIEW.md`
**Iteration:** 2

**Summary:**
- Findings in scope: 1 (1 Critical, 0 Warning, 0 Info — fix_scope=all)
- Fixed: 1
- Skipped: 0

## Fixed Issues

### CR-01: Stale test assertion makes the CI fast-lane gate red at HEAD

**Files modified:** `tests/inference/test_inference.py`
**Commit:** `7c06b5c` (`fix(11): CR-01 align adapter-reload test regex with PEFT-agnostic message`, pathspec commit, no trailers)
**Applied fix:** Updated the stale regex in
`TestConstructionBranches.test_lora_adapter_failure_raises`
(`tests/inference/test_inference.py:1711`) from
`match=r"Failed to load LoRA adapter"` to
`match=r"Failed to load PEFT adapter"`, matching the adapter-agnostic
ValueError message that phase-11 commit `3b644bd` introduced at
`dnallm/inference/inference.py:122`
(`raise ValueError(f"Failed to load PEFT adapter from {lora_adapter}: {e}") from e`).
The test body is otherwise unchanged: it still patches
`dnallm.inference.inference._get_model_path_and_imports` with
`side_effect=OSError("cannot resolve")` and asserts the `ValueError` via
`pytest.raises`, so the failure path and error-chaining behavior remain covered.

Confirmed the reviewer's uniqueness claim before and after the fix:
`grep -rn "Failed to load LoRA adapter" tests/ dnallm/` returned exactly one
hit (line 1711) pre-fix and **zero hits** post-fix.

**Verification** (ran in the main checkout — `workflow.use_worktrees=false`, so
no isolated worktree was used; every number below is reproducible from this
tree):

- `grep -rn "Failed to load LoRA adapter" tests/ dnallm/` → no matches
- `python3 -c "import ast; ast.parse(...)"` on the modified file → parses
- `uv run --no-sync pytest tests/inference/test_inference.py -q` → **120
  passed** in 9.64s (full file, fast lane — includes the fixed test)
- `uv run --no-sync ruff format --check tests/inference/test_inference.py` →
  1 file already formatted
- `uv run --no-sync ruff check tests/inference/test_inference.py` → All
  checks passed

This resolves the sole red gate from the iteration-2 review: the full CI
fast-lane invocation (`pytest tests/ dnallm/mcp/tests/ -m "not slow"`) was
failing only on this matcher, and its containing module is now fully green.

## Skipped Issues

None — all 1 in-scope finding was fixed.

---

_Fixed: 2026-10-09T15:34:26Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
