---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 01
subsystem: docs
tags: [mkdocs, strict-build, documentation, link-fix, slugify, ledger]

requires:
  - phase: 18-docs-honesty
    provides: the docs-extra floors + isolated-venv strict recipe (18-02) and the getting_started pages whose em-dash anchors this plan fixes
provides:
  - mkdocs build --strict at exit 0 with zero WARNING lines (the DOCR-03 surface cleared — remaining phase work only has to keep it green)
  - The phase's ledger discipline: 19-REVIEW-strict.md with the 15-warning RED byte-record + per-file verdict rows, rows-same-commit-as-fixes
  - The proven strict-verification recipe invocation on /tmp/mkdocs-venv19 (reused from research; toolchain re-confirmed mkdocs 1.6.1 / material 9.7.7 / marimo 0.25.1)
affects: [19-02 cuda130 re-point, 19-03..19-06 review batches, 19-07 closeout vehicle]

actuals:
  tokens: 3688   # chars/4 over the realized diff (git diff e50de44..09f9d91 = 14,750 chars)
  tasks: 2
  commits: 2     # MEASURED: git rev-list --count e50de44..HEAD
plan_head_before: e50de44197705dcaf10f6c529e334638a777a556
plan_head_after: 09f9d911baa2814bc9dc6c1cead9807b3747940f

tech-stack:
  added: []      # nothing new — /tmp/mkdocs-venv19 reuses the vetted docs-extra floors
  patterns:
    - "Strict-build verification lane: venv-ensure + PATH-pinned mkdocs build --strict, log-redirected, exit code read directly"
    - "Ledger rows land in the SAME commit as the fixes they record (16-EVIDENCE table shape)"
    - "Anchors are never hand-computed — copy id= from a built site/**/index.html"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-strict.md
  modified:
    - docs/example/marimo/benchmark/benchmark_demo.md
    - docs/example/marimo/finetune/finetune_demo.md
    - docs/example/marimo/inference/inference_demo.md
    - docs/user_guide/benchmark/advanced_techniques.md
    - docs/getting_started/gpu_support.md
    - docs/getting_started/install_honesty.md

key-decisions:
  - "All 15 warnings fixed by hand-edit only — the three marimo See-Also tails survive as hand-maintained content (generate_md_from_marimo.py emits none; regeneration would destroy them)"
  - "Warning 15's docs-dir escape uses the in-repo GitHub-URL precedent, not root-absolute /CONTRIBUTING.md (clears the warning too, but logs an absolute-link INFO and breaks under the gh-pages subpath — site_url unset)"
  - "Anchor slugs confirmed against the built site's id= before editing (mkdocs slugify DROPS the em-dash — the single-dash slug contradicts hand computation)"

patterns-established:
  - "19-REVIEW-*.md per-file ledger shape (bold verdict + one-line evidence, 16-EVIDENCE precedent) — every later review batch appends to its own 19-REVIEW file, consolidated by 19-07"
  - "RED-first discipline: reproduce the warning record byte-exactly into the ledger BEFORE fixing, so the fix is measured against evidence"

requirements-completed: [DOCR-03, DOCR-01]

coverage:
  - id: D1
    description: "mkdocs build --strict driven from 15 warnings to zero — 13 See-Also links rewritten page-relative, warning 14's rotten target re-pointed, warning 15's docs escape substituted with the GitHub-URL precedent"
    requirement: DOCR-03
    verification:
      - kind: other
        ref: "PATH=/tmp/mkdocs-venv19/bin:$PATH mkdocs build --strict → exit 0, grep -c WARNING = 0 (STRICT-CLEAN; run 3x: GREEN, tracer gate re-verify, Task-2 rebuild)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The two Class-4 em-dash broken anchors (gpu_support.md:24, install_honesty.md:7) now link the real single-dash slug, confirmed against the built site's id="
    requirement: DOCR-01
    verification:
      - kind: other
        ref: "grep -rn 'pip--the-tooluvsources' docs/getting_started/ → 0; single-dash count 1 per file; slug from site/getting_started/installation/index.html id=\"two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry\"; strict rebuild shows 0 anchor hint lines"
        status: pass
    human_judgment: false
  - id: D3
    description: "Ledger discipline established: 19-REVIEW-strict.md carries the verbatim 15-warning RED byte-record, provenance (recipe + toolchain + duration), and 6 per-file verdict rows committed same-commit as the fixes"
    verification:
      - kind: other
        ref: "19-REVIEW-strict.md exists with 15 verbatim WARNING lines + 6 rows; both task commits (47b9bc2, 09f9d91) carry ledger + fixes together"
        status: pass
    human_judgment: false
  - id: D4
    description: "Zero collateral: census stable at 216/217 (1 deselected), check_docs_sync + check_notebook_md_sync green, zero example/ content edits, mirrors untouched"
    verification:
      - kind: other
        ref: "check_docs_sync OK; check_notebook_md_sync 24 pairs in sync; pytest tests/examples --collect-only -q -m 'not giants' → 216/217 tests collected (1 deselected)"
        status: pass
    human_judgment: false

duration: 9 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 01: Strict-Clean Tracer Summary

**mkdocs --strict driven 15 warnings → zero (page-relative See-Also rewrite + rotten-target re-point + GitHub-URL precedent) plus the two em-dash anchor fixes, with the phase's RED-first ledger discipline established**

## Performance

- **Duration:** 9 min
- **Started:** 2026-10-15T00:37:39Z
- **Completed:** 2026-10-15T00:46:41Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- All 15 ledgered mkdocs warnings cleared in one pass: warnings 1-13 rewritten page-relative in the three hand-maintained marimo See-Also tails (hand-edit only), warning 14's rotten `api/inference/visualization.md` re-pointed to `../../../user_guide/inference/visualization.md`, warning 15's docs-dir escape substituted with `https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md` per the continuous_integration.md precedent
- The two 18-02-introduced Class-4 broken anchors fixed (1 char each), slug confirmed against the built site's `id="two-install-lanes-uv-and-pip-the-tooluvsources-asymmetry"` — both anchor INFO hints now gone from the build
- 19-REVIEW-strict.md created: verbatim 15-warning RED byte-record (reproduced this run — exit 1, "Aborted with 15 warnings in strict mode!"), provenance header, 6 per-file rows — rows same-commit as fixes
- Regression surface untouched: census 216/217 (1 deselected), check_docs_sync OK, check_notebook_md_sync 24 pairs in sync, zero example/ edits

## Task Commits

Each task was committed atomically:

1. **Task 1: 15 warnings to zero — the four-file strict fix, proven RED first, ledger same-commit** - `47b9bc2` (docs)
2. **Task 2: Class-4 broken anchors — the two em-dash slug fixes, slug confirmed against the built site** - `09f9d91` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `docs/example/marimo/benchmark/benchmark_demo.md` - See-Also tail: 4 links → page-relative (warnings 1-4)
- `docs/example/marimo/finetune/finetune_demo.md` - See-Also tail: 5 links → page-relative (warnings 5-9)
- `docs/example/marimo/inference/inference_demo.md` - See-Also tail: 4 links page-relative + warning 14 rotten target re-point (warnings 10-14)
- `docs/user_guide/benchmark/advanced_techniques.md` - CONTRIBUTING.md docs-dir escape → GitHub absolute URL (warning 15)
- `docs/getting_started/gpu_support.md` - 1-char anchor fix, otherwise byte-untouched (18-02 verified record)
- `docs/getting_started/install_honesty.md` - 1-char anchor fix
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-strict.md` - the ledger: RED byte-record + provenance + 6 per-file verdict rows

## Decisions Made

- Hand-edit only for the three marimo wrappers — regeneration via `scripts/generate_md_from_marimo.py` emits no See-Also section and would destroy the hand-maintained tails
- GitHub absolute URL (not root-absolute `/CONTRIBUTING.md`) for warning 15: both clear the warning, but root-absolute logs an absolute-link INFO and breaks under the gh-pages subpath deployment (`site_url` commented out in mkdocs.yml)
- RED recorded before any edit: the strict build was run and its 15 WARNING lines captured verbatim into the ledger before the fixes, so GREEN (exit 0, 0 warnings, built in 4.49s) is measured against evidence, not memory

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The /tmp/mkdocs-venv19 venv from the research session was reused as the precondition anticipated (no recreation needed); RED reproduced byte-exactly on the first run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Strict-clean baseline established: every later 19-xx commit must keep `mkdocs build --strict` at exit 0 / zero warnings (the recipe and venv are recorded in 19-REVIEW-strict.md provenance)
- Ledger discipline established for the review batches (19-03..19-06): per-file verdict rows, same-commit as fixes, 19-REVIEW-*.md shape
- Ready for 19-02 (cuda130 re-point) — this plan's six files interact with 19-02 only via gpu_support.md, which 19-02 verifies but does not rewrite

## Self-Check: PASSED

- All 8 key-files exist on disk (SUMMARY, ledger, 6 edited docs) — `[ -f ]` FOUND for each
- Both task commits (47b9bc2, 09f9d91) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- `gsd-tools check evaluation-scope --plan 19-01 --commits-only` resolves with 2 commits on this branch (strict build re-verified exit 0 / 0 warnings as part of Task 2's rebuild)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
