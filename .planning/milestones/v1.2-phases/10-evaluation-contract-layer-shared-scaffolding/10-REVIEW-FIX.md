---
phase: 10-evaluation-contract-layer-shared-scaffolding
fixed_at: 2026-10-09T12:19:27Z
review_path: .planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-REVIEW.md
iteration: 2
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-10-09T12:19:27Z
**Source review:** .planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-REVIEW.md
**Iteration:** 2 (post-fix re-review findings)

**Summary:**
- Findings in scope: 2 (both Info; fix scope = all)
- Fixed: 2
- Skipped: 0

**Mode:** `workflow.use_worktrees = false` — all edits and commits were made
directly in the main checkout on branch `revision` (no isolated worktree), so
all verification results below are reproducible from the main checkout.

## Fixed Issues

### IN-01: `evaluate(split=...)` guard misses `output_dir=""` — empty string still writes the result JSON to the CWD

**Files modified:** `dnallm/finetune/trainer.py`, `tests/finetune/test_trainer.py`
**Commit:** 39c5c12
**Applied fix:** Widened the guard at `dnallm/finetune/trainer.py:620` from
`if self.train_config.output_dir is None:` to the falsy check
`if not self.train_config.output_dir:`, so `output_dir: ""` now raises the same
matchable `ValueError` ("finetune.output_dir is not set...") before any
`trainer.predict` call. Per the same-change pytest rule, the IN-03 test
`test_missing_output_dir_raises_instead_of_cwd_fallback` was parametrized over
`[None, ""]` (ids `none` / `empty-string`) so the empty-string case asserts the
raise plus `predict.assert_not_called()` in the same commit.

**Verification:**
- `pytest tests/finetune/test_trainer.py -q` — 65 passed (was 64; +1 from the
  new `empty-string` parametrized case). Targeted `-k missing_output_dir -v`
  run shows both `[none]` and `[empty-string]` PASSED.
- `.venv/bin/ruff check` and `.venv/bin/ruff format --check` clean on both
  modified files.

### IN-02: WR-05 fix restored the pre-phase old terminology in `gpu_optimization.md` — the only swept docs page with zero occurrences of the standard term

**Files modified:** `docs/user_guide/performance/gpu_optimization.md`
**Commit:** 20f6b72
**Applied fix:** Line 3 now reads "Training and running DNA large language
models can be computationally intensive." — dropped the original "large"
adjective (the cause of the earlier doubling) and kept the phase-standard
swept term "DNA large language models".

**Verification:**
- `grep -c "DNA large language" docs/user_guide/performance/gpu_optimization.md`
  — 1 (standard term present).
- `grep -c "DNA language"` — 0 (old short-form term fully absent from the page).
- No doubled "large" in the sentence (line 3 re-read confirms grammatical
  wording).

## Skipped Issues

None — both in-scope findings were fixed.

---

_Fixed: 2026-10-09T12:19:27Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
