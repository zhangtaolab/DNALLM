---
phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
plan: 01
subsystem: testing
tags: [windows-parity, dispatch, sys-platform, pytest, notebook, jaccard, bedtools, numpy]

requires:
  - phase: 15-windows-parity-core-numpy-interval-evaluators
    provides: dnallm.utils.jaccard (locked import-pure numpy evaluator surface)
provides:
  - CRE notebook cell 11 platform dispatch seam (win32 -> dnallm.utils.jaccard in-process; POSIX -> bedtools CLI byte-equivalent)
  - Platform-aware showcase gate (test_plant_helixseek_showcase.py) — POSIX fail-loud PATH assert retained, Windows routes to numpy execution
  - tests/test_dispatch_direction.py born at tests/ root (CRE structure + win32/linux runtime contracts) — extended by 16-02 for the NER seams
  - Proven gate-edit / structure-test-reader / mirror mechanics for 16-02/16-03 to reuse
affects: [16-02, 16-03, 16-04, WPAR-04, WPAR-06, WPAR-08]

actuals:
  tokens: 4432      # chars/4 over the realized diff (17729 chars across 4 files)
  tasks: 2
  commits: 2        # MEASURED: git rev-list --count ddaa546..HEAD (task commits; docs commit follows)

plan_head_before: ddaa54631a76abbbdb77583e30d84abf83513ddf
plan_head_after: 8367bf5cb0831e34d9439f4bef4cf325e5e22d60

tech-stack:
  added: []          # no new packages — consumes the phase-15 locked surface only
  patterns:
    - "Acquisition-seam platform dispatch: `if sys.platform == \"win32\":` at the seam ONLY, POSIX branch byte-preserved, no try/except-ImportError ladder (WPAR-06 shape, now structure-tested)"
    - "Root-level platform-neutral contract tests (phase-15 posture: no skips, no markers) that exec committed notebook cell source under a monkeypatched platform with a poisoned subprocess.run"

key-files:
  created:
    - tests/test_dispatch_direction.py
  modified:
    - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
    - tests/examples/test_plant_helixseek_showcase.py

key-decisions:
  - "Platform predicate pinned as the `sys.platform == \"win32\"` equality check (AS-1); the inverse `!= \"win32\"` guard wraps the showcase gate assert so POSIX behavior is byte-equivalent"
  - "Cell 11 keeps its linear shape — the guard is inline in the cell, POSIX branch preserved byte-for-byte modulo else-indentation (AS-2); the factored-helper pattern stays a 16-02 decision"
  - "Runtime contracts live in tests/test_dispatch_direction.py at tests/ root, not in tests/examples — zero census movement (ci.yml triple and live collect-only both stayed 216/217 (1 deselected), Pitfall 4)"
  - "Two structure assertions strengthened beyond the plan's minimum list, both straight from the truth list: the branch-shared 4-decimal print format pinned textually (TRUTH-1/Pitfall 8) and a skip-machinery ban inside the gate region (TRUTH-2/WPAR-04 no-ever-green-skip)"

requirements-completed: [WPAR-04, WPAR-06, WPAR-08]

coverage:
  - id: D1
    description: "CRE notebook cell 11 platform dispatch — win32 computes jaccard via dnallm.utils in-process into the same variable; POSIX branch keeps the subprocess bedtools CLI call (argv, check=True, stdout parse) byte-equivalent; shared print keeps the exact 4-decimal jaccard= format"
    requirement: WPAR-04
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_win32_contract_cell11_runs_evaluator_zero_subprocess"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_cell11_posix_branch_keeps_cli_call"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_cell11_shared_print_keeps_the_stream_format"
        status: pass
      - kind: other
        ref: "plan verify probe: exec cell 11 under monkeypatched win32 with subprocess.run poisoned -> WIN32 CELL11 END-TO-END OK: jaccard=0.1923"
        status: pass
    human_judgment: false
  - id: D2
    description: "Showcase gate platform-aware — shutil.which hard assert wrapped in the inverse win32 guard (exact message retained, POSIX fail-loud unchanged); Windows proceeds into numpy execution with no skip call of any kind; band assertions/harness contract below the gate untouched"
    requirement: WPAR-04
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_showcase_gate_guard_precedes_path_probe"
        status: pass
      - kind: unit
        ref: "tests/examples/test_plant_helixseek_showcase.py fast lane (-m 'not slow'): 23 passed, 3 deselected"
        status: pass
    human_judgment: false
  - id: D3
    description: "tests/test_dispatch_direction.py born at tests/ root — CRE structure assertions (platform predicate, CLI branch, no try-wrapped import ladder via AST, md cells 0/10/22 bedtools-worded) plus win32/linux runtime contracts; phase-15 posture (no skips, no markers, no bedtools binary); zero census movement"
    requirement: WPAR-06
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py (8 passed)"
        status: pass
      - kind: other
        ref: "census guards: ci.yml triple grep = '216/217 tests collected (1 deselected)' AND live `pytest tests/examples --collect-only -q -m 'not giants'` tail = 216/217"
        status: pass
    human_judgment: false
  - id: D4
    description: "Docs mirror discipline — the CRE notebook edit landed with its docs/example twin byte-identical in the same commit"
    requirement: WPAR-08
    verification:
      - kind: other
        ref: "scripts/check_docs_sync.py -> OK exit 0 (cmp byte-identical)"
        status: pass
    human_judgment: false

duration: 13 min
completed: 2026-10-13
status: complete
---

# Phase 16 Plan 01: CRE Dispatch Tracer Summary

**First numpy-evaluator consumer wired end-to-end: CRE cell 11 platform dispatch (win32 in-process jaccard / POSIX bedtools CLI byte-equivalent), platform-aware showcase gate, and the born WPAR-06 dispatch-direction structure+runtime test file — mirrored, fast-lane green, zero census movement.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-10-13T16:11:57Z
- **Completed:** 2026-10-13T16:25:00Z
- **Tasks:** 2 (1 tracer + 1 auto)
- **Files modified:** 4

## Accomplishments
- CRE notebook cell 11 dispatches on `sys.platform == "win32"`: the Windows branch imports `jaccard` from `dnallm.utils` under a short alias and computes in-process into the same `jaccard_value` variable; the POSIX branch keeps the subprocess bedtools CLI call byte-for-byte (same argv, capture settings, `check=True`, stdout parsing); the trailing print stays branch-shared with the exact 4-decimal format (the nightly stream regex + CRE cell 21 parse normalization point)
- The showcase gate (`test_cre_notebook_executes_within_selection_bands`) is platform-aware: the `shutil.which("bedtools")` hard assert is wrapped in `if sys.platform != "win32":` (message unchanged, POSIX fail-loud, one-line numpy-evaluator comment); zero skip machinery on either route
- `tests/test_dispatch_direction.py` born at tests/ root (8 tests): structure group (predicate equality, POSIX CLI branch AST-verified incl. `check=True`, no try-wrapped import anywhere in cell 11, shared print format, md cells 0/10/22 bedtools-worded guard) + runtime contract A (win32 exec, `subprocess.run` poisoned — never called, output equals the evaluator value at 4 decimals over a union-non-trivial BED pair) + runtime contract B (POSIX exec propagates the SENTINEL-CLI-UNUSABLE error — the numpy path can never silently substitute on Linux) + gate structure assertion (guard textually precedes the retained probe; skip-machinery ban in the region)
- Docs mirror copied byte-identical in the same commit; full fast lane green (2534 passed, 2 pre-existing allowlisted skips, 75 deselected)

## Task Commits

Each task was committed atomically:

1. **Task 1: CRE cell 11 platform dispatch + showcase gate platform-aware + docs mirror** - `d05a3b9` (feat)
2. **Task 2: tests/test_dispatch_direction.py born — CRE structure assertions + win32/linux runtime contracts** - `8367bf5` (test)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified
- `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` - cell 11 only: platform dispatch seam (md cells 0/10/22 byte-identical to HEAD, outputs/ids preserved, single-hunk diff)
- `docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` - byte-identical mirror copy, same commit
- `tests/examples/test_plant_helixseek_showcase.py` - gate wrapped in inverse win32 guard + `import sys`; nothing below the gate touched
- `tests/test_dispatch_direction.py` - NEW root-level structure + runtime-contract test file (phase-15 posture; extended by 16-02 for the NER seams)

## Decisions Made
- Predicate pinned as the `sys.platform == "win32"` equality check per AS-1; gate uses the inverse form so the POSIX branch is textually byte-equivalent
- Cell 11 kept linear (AS-2): guard inline, no helper refactor — that decision belongs to 16-02's loj seams
- Two structure assertions added beyond the plan's minimum enumeration, both implementing the plan's own truths: `test_cell11_shared_print_keeps_the_stream_format` (TRUTH-1/Pitfall 8) and the skip-machinery ban in the gate region (TRUTH-2). No plan text contradicted; they harden the seam against exactly the drift the truths describe
- ruff posture: repo's `# ruff: ignore[...]` comment form reused for the deliberate `subprocess` import (monkeypatch target only — the win32 contract never spawns) and the two `exec` calls (committed repo cell source, tmp_path inputs only)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. Tracer feedback gate (row 3: interactive, end-of-phase, automated-only verify) re-ran the full Task 1 verify chain end-to-end after commit — all green — before expansion into Task 2.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The dispatch shape, gate-edit mechanics, structure-test reader (nb JSON + ast.parse), runtime-contract pattern (monkeypatched platform + poisoned runner), and mirror discipline are proven on the smallest seam — 16-02 rolls the larger NER loj surface (script :7/:249 + notebook cells 0/1/2/8) against these exact patterns
- tests/test_dispatch_direction.py exists for 16-02's NER extensions (WPAR-06 SC4 "same change" clause satisfied for the CRE seam; 16-02 completes it for the NER seams)
- WPAR-04/WPAR-06/WPAR-08 advanced but not complete (the other two gates, NER seams, and frozen-pair work land in 16-02/16-03/16-04); REQUIREMENTS.md marking is shared-ID-gated until the last declaring plan finishes
- Linux nightly re-execution of the bands is territory for 16-04's evidence plan (the POSIX branch is behaviorally byte-equivalent, so the nightlies hold by construction; full fast lane + showcase suite green here)

---
*Phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity*
*Completed: 2026-10-13*

## Self-Check: PASSED

- All 5 key-files exist on disk (FOUND each)
- Both task commits (d05a3b9, 8367bf5) are ancestors of HEAD on this branch
- `check evaluation-scope --plan 16-01 --commits-only`: resolved with both commits on this branch
- Re-ran all acceptance criteria live: win32 probe OK (jaccard=0.1923), showcase fast lane 23 passed, dispatch-direction file 8 passed, check_docs_sync exit 0, census triple + live collect-only both 216/217 (1 deselected), full fast lane 2534 passed / 2 pre-existing skips
