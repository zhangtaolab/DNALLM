---
phase: 261003-jpr
plan: 01
subsystem: utils
tags: [transformers_compat, absence-contract, import-time-safety, in-01, code-review, phase-05]
requires:
  - IN-01 open finding from the Phase 05 incremental review (05-REVIEW.md:234-247)
provides:
  - IN-01 closed — every _patch_* installer in transformers_compat no-ops when transformers is unimportable
  - structurally self-pinning absence contract (dynamic vars() collection + pinned roster) covering future installers
affects:
  - dnallm/utils/transformers_compat.py
  - tests/utils/test_transformers_compat.py
  - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW-DISPOSITION.md
tech-stack:
  added: []  # stdlib try/except only — zero new dependencies
  patterns:
    - try-import-guard (module's six pre-existing installer guards, extended to the three sl7 installers)
key-files:
  created: []
  modified:
    - dnallm/utils/transformers_compat.py
    - tests/utils/test_transformers_compat.py
decisions:
  - Guard comment variants follow the plan exactly — "not installed / module renamed" on the
    configuration_utils/cache_utils guards (a future transformers submodule rename is the IN-01 failure
    mode), plain "not installed" on the modeling_utils guard matching the two pre-existing modeling_utils
    guards verbatim
  - `# pragma: no cover` markers kept on the new except branches despite the new tests exercising them —
    matches the file's established style (bitsandbytes guard at line 274 is likewise pragma-marked while
    covered by test_passthrough_when_bitsandbytes_unavailable)
  - Absence test collects installers dynamically from vars(transformers_compat) so any future _patch_*
    installer enters the absence contract automatically; the pinned EXPECTED_PATCH_INSTALLERS roster
    forces conscious extension when the installer set changes
metrics:
  duration: 4 min
  completed: 2026-10-03T06:24:19Z
status: complete
actuals:
  tokens: 1967       # chars/4 over the realized diff (7868 diff chars)
  tasks: 2
  commits: 2         # measured: git rev-list --count 651ea7f..HEAD
  files: 3
plan_head_before: 651ea7f
plan_head_after: 56a72c9
---

# Quick Task 261003-jpr: Fix IN-01 — absence guards for the three new transformers_compat patch installers Summary

One-liner: the three installers that landed with quick task 261002-sl7
(`_patch_pretrained_config_legacy_defaults`, `_patch_mamba_cache`,
`_patch_legacy_init_weights_bookkeeping`) now wrap their bare transformers imports in the module's
standard try/except-Exception no-op guard, and a new `TestTransformersAbsenceContract` (9 dynamically
collected installer items + pinned roster + apply_patches survival) pins the module-docstring contract
that `import dnallm` never crashes in a stripped or renamed-transformers environment — red
4 failed / 83 passed before the guards, 87 green after.

## What Was Built

**dnallm/utils/transformers_compat.py** (fix, commit cad7370):

1. `_patch_pretrained_config_legacy_defaults` (was line 553): `import transformers.configuration_utils`
   wrapped in try / `except Exception:  # pragma: no cover - transformers not installed / module renamed` /
   `return` — the DeBERTa-guard variant; a future transformers renaming this submodule is exactly the
   IN-01 failure mode.
2. `_patch_mamba_cache` (was line 764): `import transformers.cache_utils` wrapped, same comment variant.
3. `_patch_legacy_init_weights_bookkeeping` (was line 990): `import transformers.modeling_utils` wrapped
   with `# pragma: no cover - transformers not installed` — verbatim match with the two pre-existing
   modeling_utils guards.

Nothing else in the module changed (verified via `git diff`: exactly the three hunks, +9/-3 lines); the
six already-guarded installers, `_post_init_computes_tied_weights_keys`, every gate/sentinel/setattr
body, `apply_patches`, and the module docstring are byte-for-byte untouched.

**tests/utils/test_transformers_compat.py** (contract tests, same commit cad7370 — owner rule: dnallm/
change ships with its pytest coverage):

1. Module-level `_collect_patch_installers()`: sorted names from `vars(transformers_compat)` where the
   name starts with `_patch_` and the value is callable — dynamic so a future unguarded installer is
   caught by this test automatically, not by the next reviewer.
2. Module-level `EXPECTED_PATCH_INSTALLERS` frozenset pinning the nine current names.
3. `class TestTransformersAbsenceContract` (11 items) with a Google-style docstring citing IN-01 /
   05-REVIEW.md:234 and the mechanism (None sys.modules entry → every import form raises
   ModuleNotFoundError via parent-first resolution; guards' broad `except Exception` is what the
   transformers_compat.py:7-11 contract promises):
   - `test_installer_noops_when_transformers_unimportable` × 9 parametrized items —
     `monkeypatch.setitem(sys.modules, "transformers", None)` then installer call `is None`.
   - `test_installer_roster_is_pinned` — roster equality with the frozenset.
   - `test_apply_patches_survives_unimportable_transformers` — `apply_patches() is None` under the same
     monkeypatch (the eager-at-import chain through dnallm/utils/__init__.py:8 and the module body).

**05-REVIEW-DISPOSITION.md** (ledger, commit 56a72c9): IN-01 flipped open → fixed in all three places —
front-matter finding, markdown table row (`cad7370 fix(quick-261003-jpr)`), and the open count 3 → 2.
Phase 05 incremental review now has zero open findings of any severity.

## TDD Evidence

- RED (before the guards): `4 failed, 83 passed in 3.81s` — exactly the plan's predicted tally. The
  three IN-01 installer items failed with `ModuleNotFoundError: import of transformers halted; None in
  sys.modules`; `test_apply_patches_survives_unimportable_transformers` failed at the
  `_patch_pretrained_config_legacy_defaults` call inside apply_patches; the 6 guarded-installer items,
  the roster test, and all 76 pre-existing tests passed.
- GREEN (after the guards): `87 passed in 3.37s` — all 76 pre-existing tests byte-identical and
  unmodified (the guards only add a path imports never take when transformers is importable; live
  attach/sentinel/idempotence state on transformers 5.17.0 untouched).
- Broader sanity: full `tests/utils/` directory — `129 passed in 4.17s`.
- Lint: `ruff check` and `ruff format --check` clean on both code files (baseline-clean too).

## Deviations from Plan

**1. [Docstring wording] apply_patches raise point described as "fifth call", not the plan's "fourth installer"**
- **Found during:** Task 1
- **Issue:** the plan's Test 3 behavior text says apply_patches raises "at the fourth installer,
  `_patch_pretrained_config_legacy_defaults`" — that installer is the fifth call in apply_patches' body
  (and the first unguarded one). A planning-time ordinal slip; the mechanics, the failing installer, and
  the RED tally (4 failed / 83 passed) matched the plan exactly.
- **Fix:** the test docstring states the accurate description ("the first unguarded one,
  `_patch_pretrained_config_legacy_defaults`, is the fifth call in its body").
- **Files modified:** tests/utils/test_transformers_compat.py
- **Commit:** cad7370

**2. [Formatting] ruff format reflowed the new EXPECTED_PATCH_INSTALLERS literal**
- **Found during:** Task 2 verification
- **Issue:** the frozenset as first written used the non-hugged brace style; `ruff format --check`
  flagged it (the plan requires format-clean).
- **Fix:** ran `ruff format` on the test file — the file was format-clean at baseline, so the reformat
  touched only the new literal (verified: diff is 90 pure insertions).
- **Files modified:** tests/utils/test_transformers_compat.py
- **Commit:** cad7370

## Verification

- `.venv/bin/python -m pytest tests/utils/test_transformers_compat.py -q` → 87 passed (baseline 76;
  RED midpoint 4 failed / 83 passed, exactly as the plan predicted).
- `.venv/bin/python -m ruff check dnallm/utils/transformers_compat.py
  tests/utils/test_transformers_compat.py` → All checks passed; same for `ruff format --check` → 2 files
  already formatted.
- `git log --oneline -3`: `56a72c9 docs(quick-261003-jpr): IN-01 marked fixed in 05 disposition ledger`
  on top of `cad7370 fix(quick-261003-jpr): guard the three unguarded transformers_compat patch
  installers (IN-01)` on top of the plan commit 651ea7f — the established CR-01/WR-01 two-commit
  close-out pattern, no attribution trailers.

## Self-Check: PASSED

- Files exist: dnallm/utils/transformers_compat.py (guards verified in diff), tests/utils/
  test_transformers_compat.py (class verified in diff), 05-REVIEW-DISPOSITION.md (open: 2, IN-01 fixed).
- Commits exist: cad7370 and 56a72c9 on phs, confirmed via git log.
