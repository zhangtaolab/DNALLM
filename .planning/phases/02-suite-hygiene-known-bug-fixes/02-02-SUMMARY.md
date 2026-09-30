---
phase: 02-suite-hygiene-known-bug-fixes
plan: "02"
subsystem: testing
tags: [pytest, tmp-path, monkeypatch, autouse-fixture, gitignore, pdf-markers, test-hygiene]

requires:
  - phase: 01-harness-integrity-measured-baseline
    provides: full-suite census identifying the PDF-writing test module and its repo-tree writes; honest single-config pytest harness
provides:
  - PDF test artifacts written exclusively under pytest tmp_path (autouse module-global rebind; zero call-site churn to create_pdf_file/cleanup_pdf_file/assert_pdf_created)
  - @pytest.mark.pdf applied at class level to all 9 PDF-writing classes — `-m pdf` now selects 53 tests (previously 0 of 65)
  - .gitignore consolidated to the single correct directory entry `tests/inference/pdf/` (kept, per locked belt-and-suspenders decision)
  - 9 untracked stray PDF artifacts deleted; the path is clean in git status
  - Twice-run tree-clean proof green: two consecutive `-m pdf` runs, repo pdf dir never recreated
affects: [02-03 (skip-allowlist census unaffected by this plan — no skip changes), 03-coverage-tests (plot tests now tmp_path-isolated), 04-ci-gate]

actuals:
  tokens: 955
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Module-global rebind fixture: function-scoped autouse `pdf_output_dir(tmp_path, monkeypatch)` patching `sys.modules[__name__]` — the only import-mode-proof target when the test package has no __init__.py"
    - "Collection-time-write elimination: never mkdir at module level in test files; create the directory at call time inside the helper so the rebind redirects it"

key-files:
  created: []
  modified:
    - .gitignore
    - tests/inference/test_plot.py

key-decisions:
  - "Rebind target is `sys.modules[__name__]` (object form), NOT the plan's dotted string target: tests/ has no __init__.py, so pytest (prepend import mode) imports the module as top-level 'test_plot' — `monkeypatch.setattr(\"tests.inference.test_plot.PDF_OUTPUT_DIR\", ...)` would import a SECOND module object via the namespace-package path and rebind the wrong copy, silently failing the tree-clean goal while passing the AST shape gate. The end-to-end twice-run gate is the tripwire that catches this class of mistake"
  - "Markers applied to exactly the 9 census classes (53 tests selected / 12 deselected); TestPrepareData and TestEdgeCases left unmarked per the exact AST census (zero create_pdf_file callers)"
  - "Pre-commit proof run via before/after git-status write-equivalence (the verbatim per-run emptiness check cannot pass while the source edit itself is uncommitted); the plan's verbatim verify script re-run post-commit — both green"
  - "The `if __name__ == \"__main__\"` direct-run guard's mkdir left untouched — never executes under pytest (guarded by `'pytest' not in sys.modules`); outside FIX-04's pytest-run criterion"

patterns-established:
  - "Import-mode-proof monkeypatching of test-module globals: `monkeypatch.setattr(sys.modules[__name__], \"GLOBAL\", value)` instead of dotted-string targets when the tests tree lacks __init__.py"
  - "Proof-order discipline for tree-clean gates: run the twice-run proof pre-commit with status-equivalence, then re-run the verbatim gate post-commit so both the letter and the spirit of the criterion are demonstrated"

requirements-completed: [FIX-04]

coverage:
  - id: D1
    description: "FIX-04 (tests) — every PDF write lands under tmp_path: import-time mkdir removed, autouse rebind fixture redirects all 25 create_pdf_file call sites, @pytest.mark.pdf applied to the 9 writing classes, twice-run tree-clean proof green"
    requirement: FIX-04
    verification:
      - kind: other
        ref: "AST gate: no module-level mkdir in tests/inference/test_plot.py; autouse rebind fixture present"
        status: pass
      - kind: other
        ref: "pytest tests/inference/test_plot.py -m pdf --collect-only -> 53 selected / 12 deselected (>= 19 gate)"
        status: pass
      - kind: other
        ref: "pytest -m pdf x2 consecutive runs -> 53 passed each, tests/inference/pdf absent after both, git status --porcelain tests/inference/ empty after both"
        status: pass
      - kind: other
        ref: "full-file run: 65 passed, pdf dir absent"
        status: pass
    human_judgment: false
  - id: D2
    description: "FIX-04 (repo state) — .gitignore carries exactly one correct directory entry tests/inference/pdf/ (entry KEPT per locked decision), misspelled + six enumerated lines gone, notebook rule intact, 9 stray artifacts deleted, path clean in git status"
    requirement: FIX-04
    verification:
      - kind: other
        ref: "verify block: grep -qx 'tests/inference/pdf/' .gitignore; misspelled/enumerated entries absent; example/notebooks/*/*.pdf intact; ls -A tests/inference/pdf empty; git status --porcelain tests/inference/pdf/ empty"
        status: pass
    human_judgment: false

duration: 11 min
completed: 2026-09-30
status: complete
commits: 2
plan_head_before: aaf8228ac29e233007368dd88c883484107ef730
plan_head_after: b20f63ecb19af639284da0b554e95ce05d0037c2
---

# Phase 2 Plan 2: PDF Artifact Isolation & .gitignore Fix (FIX-04) Summary

**Autouse tmp_path rebind of the PDF_OUTPUT_DIR module global redirects all 25 PDF-writing call sites with zero churn; class-level pdf markers make 53 writing tests selectable (was 0 of 65); .gitignore consolidated to one correct directory entry; 9 strays deleted — twice-run tree-clean proof green**

## Performance

- **Duration:** 11 min
- **Started:** 2026-09-30T00:23:06Z
- **Completed:** 2026-09-30T00:34:29Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **Task 1 (.gitignore + strays):** the seven lines 132-138 (misspelled `test/inference/pdf/` plus six enumerated demo PDF filenames) replaced by the single directory entry `tests/inference/pdf/`; the entry is KEPT per the locked FIX-04 belt-and-suspenders decision (the tmp_path rebind removes the write, the ignore covers residue). The unrelated `example/notebooks/*/*.pdf` rule untouched. The 9 untracked runtime-generated PDF artifacts under `tests/inference/pdf/` deleted by plain filesystem removal (nothing tracked there — no history rewrite); the directory itself removed; `git status --porcelain tests/inference/pdf/` empty.
- **Task 2 (rebind + marker):** function-scoped autouse fixture `pdf_output_dir(tmp_path, monkeypatch)` added after the helpers, rebinding the module global `PDF_OUTPUT_DIR` to `tmp_path`; `create_pdf_file` resolves the global at call time so all 25 direct call sites redirect with zero signature churn (`cleanup_pdf_file` / `assert_pdf_created` operate on absolute paths — unchanged). The import-time `PDF_OUTPUT_DIR.mkdir(exist_ok=True)` deleted (a repo write at collection time, before any fixture runs — RESEARCH Pitfall 7); the in-function `mkdir` stays and guarantees creation under tmp_path. `@pytest.mark.pdf` applied at class level to the 9 census classes (TestPlotBars, TestPlotCurve, TestPlotScatter, TestPlotAttentionMap, TestPlotEmbeddings, TestPlotMuts, TestPerformance, TestPDFOutputQuality, TestIntegration); TestPrepareData and TestEdgeCases (zero callers) deliberately unmarked. Module docstring updated to reflect tmp_path output (it claimed the repo directory).
- **Proofs:** `-m pdf --collect-only` selects 53 / deselects 12; two consecutive `-m pdf` runs each 53 passed with `tests/inference/pdf` never recreated and `git status --porcelain tests/inference/` clean (run pre-commit as before/after write-equivalence, then the plan's verbatim script re-run post-commit — both green); full-file run 65 passed, dir absent.

## Task Commits

Each task was committed atomically:

1. **Task 1: .gitignore consolidation + stray artifact deletion** - `219ddcf` (chore)
2. **Task 2: tmp_path rebind fixture + class-level pdf marker + end-to-end proof** - `b20f63e` (test)

**Plan metadata:** committed after this SUMMARY (docs)

## Files Created/Modified

- `.gitignore` - seven lines (misspelled directory entry + six enumerated demo filenames) replaced by the single directory entry `tests/inference/pdf/`
- `tests/inference/test_plot.py` - import-time mkdir removed; autouse `pdf_output_dir(tmp_path, monkeypatch)` fixture added; `@pytest.mark.pdf` on 9 classes (+`import sys`); module docstring corrected

## Decisions Made

- Rebind implemented as `monkeypatch.setattr(sys.modules[__name__], "PDF_OUTPUT_DIR", tmp_path)` rather than the plan's string target — see key-decisions and Deviations entry 1.
- Markers at class level on exactly the 9 census classes (assumption A4's safe superset bounded by the >= 19 selection gate; actual selection 53).
- The `if __name__ == "__main__"` guard's `PDF_OUTPUT_DIR.mkdir` left as-is (never runs under pytest; direct-execution ergonomics unchanged).
- No skip changes in this file (FIX-03 scope); no new markers registered (pdf already declared, `--strict-markers` satisfied).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rebind target changed from dotted string to `sys.modules[__name__]`**
- **Found during:** Task 2 (implementation, before first run)
- **Issue:** The plan specified `monkeypatch.setattr("tests.inference.test_plot.PDF_OUTPUT_DIR", tmp_path)` ("string-target form works because the module is importable"). The dotted path IS importable — but `tests/` has no `__init__.py`, so pytest's default prepend import mode imports the test module as top-level `test_plot`. The string target would import a SECOND, separate module object through the namespace-package path and rebind that copy, leaving the module under test pointing at the repo path: every test green, AST shape gate green, but PDFs still written to the repo (the tree-clean criterion silently unmet).
- **Fix:** Object-form setattr on the module object that is actually executing: `monkeypatch.setattr(sys.modules[__name__], "PDF_OUTPUT_DIR", tmp_path)` — correct under any import mode. PATTERNS.md explicitly allows the object form ("object form via `from tests.inference import test_plot` also fine" — that variant carries the same dual-module hazard, which is why the `sys.modules[__name__]` form was chosen).
- **Files modified:** tests/inference/test_plot.py
- **Verification:** plan's end-to-end twice-run gate green (53 passed twice, repo pdf dir absent, tree clean); AST gate green (its normalized-quote substring check matches the object form).
- **Committed in:** b20f63e

**2. [Rule 3 - Blocking] Pre-commit proof executed as before/after write-equivalence**
- **Found during:** Task 2 (running the verify block before commit, as the plan's action instructs)
- **Issue:** The per-run gate `git status --porcelain tests/inference/` must be EMPTY — but the uncommitted Task 2 source edit itself shows as ` M tests/inference/test_plot.py`, so the verbatim gate cannot pass pre-commit (false positive on the executor's own edit, not on test-written artifacts).
- **Fix:** Pre-commit runs captured status before/after each pytest run and asserted equality (proving the run wrote nothing); after committing, the plan's verbatim verify script was re-run and passes as written.
- **Files modified:** none (procedural)
- **Verification:** both forms green; verbatim script output `pdf-clean twice (selected=65)`.
- **Committed in:** n/a (procedural; evidence in run logs /tmp/fix4_run*.log)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking/procedural)
**Impact on plan:** Both fixes preserve the plan's contract exactly — the criterion (zero repo writes, twice-proven) is met and strengthened; no scope creep (zero call-site changes held).

## Issues Encountered

- Diagnostic detour: the first `-m pdf` run appeared to recreate `tests/inference/pdf` (empty). Root cause was not the test run — an executor-run importability probe (`python -c "import tests.inference.test_plot"`) executed the ORIGINAL module code (pre-edit, import-time mkdir still present) and recreated the empty dir before the proof ran. Re-run from verified-clean state: both runs clean. No code change needed; noted so future executors avoid importing test modules mid-edit.

## Known Stubs

None — no stubs, placeholders, or unwired data paths introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-02 complete: FIX-04 landed; ready for 02-03 (typed skips + allowlist). This plan introduced no skip changes, so the 02-03 census inputs are unaffected.
- For Phase 3: plot tests are tmp_path-isolated and `-m pdf`-selectable (53 tests) — coverage runs can deselect or select them deterministically.
- Observation for the verifier: the plan's selection-gate script extracts `65` (the total) from pytest's `53/65 tests collected` summary line; the actual selected count is 53 (12 deselected = TestPrepareData 4 + TestEdgeCases 8). Both numbers clear the >= 19 gate; the deselect boundaries were additionally verified by class-name grep of the collect log (0 TestPrepareData/TestEdgeCases lines).

## Self-Check: PASSED

02-02-SUMMARY.md exists; both modified files present on disk; both task commits (219ddcf, b20f63e) present in git log; measured commits from the plan ledger (aaf8228..HEAD) = 2, matching frontmatter; `git status --porcelain tests/inference/` empty; all four task verify blocks re-run verbatim post-commit and green (gitignore-ok / rebind-ok / collect 53 selected / pdf-clean twice).

---
*Phase: 02-suite-hygiene-known-bug-fixes*
*Completed: 2026-09-30*
