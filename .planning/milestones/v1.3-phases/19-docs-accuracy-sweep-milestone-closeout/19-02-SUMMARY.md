---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 02
subsystem: docs
tags: [cuda130, install-defaults, docs-accuracy, packaging, mirror-sync, re-point, gpu-matrix]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plan 01)
    provides: the strict zero-warning baseline this plan's edits had to keep green, and the 19-REVIEW-*.md ledger discipline this plan extends
  - phase: 18-packaging-install-honesty-atomic-extras-surgery (plan 02)
    provides: the lane-label vocabulary (29 labels), gpu_support.md GPU matrix (the re-point's honesty authority), and the 2556-passed fast-lane baseline the README test counts now state
provides:
  - cuda130 as the recommended default install command across every generic-default surface: the mirrored overview.md pair, 22 wrapper install lines, installation.md lead/scenarios/table, quick_start, user_guide/getting_started, index, tutorial, README headline/table/examples
  - Honest README dated numbers: the three "1,900+" claims replaced by the 2556-passed fast-lane baseline
  - 19-REVIEW-repoint.md — the per-surface re-point ledger with before/after counts and the sanctioned-survivor lists as landed
affects: [19-03..19-06 review batches (verify prose against the cuda130-default state), 19-07 closeout (ci.yml test-cuda leg + matrix evidence; consolidates this ledger into 19-EVIDENCE.md)]

actuals:
  tokens: 8126     # chars/4 over the realized diff (git diff e564845..313579e = 32,504 chars; plan estimate 65000 at confidence:low carried the full sweep-adjacent surface)
  tasks: 3
  commits: 3       # MEASURED: git rev-list --count e564845..HEAD
plan_head_before: e5648458400af712ee259a00976bd7a7b886babf
plan_head_after: 313579e1b40362aa06ef891af7bbc951d101e14f

tech-stack:
  added: []
  patterns:
    - "Re-point honesty rule (T-19-02 mitigate): the DEFAULT moves, never blanket substitution — sanctioned cuda124 survivors pinned per file (installation.md 5+table-row, README 3, gpu_support.md 4 untouched) and enforced by exact/band count gates; every re-pointed GPU-choice block keeps or links the GPU-matrix pointer"
    - "Prose-lane demotion: where an install-command count gate forbids a literal cuda124 command (README RC=0, installation.md IC≤1), the older-GPU lane survives as spaced 'CUDA 12.4–12.8' prose + matrix pointer, not a deleted fact"

key-files:
  created:
    - .planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-repoint.md
  modified:
    - example/notebooks/overview.md
    - docs/example/notebooks/overview.md
    - docs/example/notebooks/*.md (19 wrappers)
    - docs/example/marimo/benchmark/benchmark_demo.md
    - docs/example/marimo/finetune/finetune_demo.md
    - docs/example/marimo/inference/inference_demo.md
    - docs/getting_started/installation.md
    - docs/getting_started/quick_start.md
    - docs/user_guide/getting_started.md
    - docs/user_guide/tutorial/end_to_end_workflow.md
    - docs/index.md
    - README.md

key-decisions:
  - ":457 (Scenario 6 mamba combination) flipped to .[cuda130,mamba] despite the plan's 'keep' parenthetical — the plan's own count arithmetic (5 listed survivors + the kept label-stripped table row = the band's exact ceiling of 6) makes keeping :457 land I=7 outside the verify band; three 'exactly 5 sanctioned survivors' statements and the main action clause agree"
  - "The older-GPU (Turing) lane is demoted to prose everywhere a literal cuda124 install command would violate the plan's install-cmd gates — 'CUDA 12.4–12.8 lanes' (spaced) + GPU-matrix pointer, mirroring the plan's own README treatment (cuda124 block 'may remain … WITHOUT the recommended annotation' under RC=0)"
  - "README test counts use '2,500+ test cases' for the two suite-size claims and the exact 'fast lane (2,556 tests)' for the CI sentence — the plan's own example phrasing, consistent across the three sites"
  - "Lane labels untouched at 29; pip lanes never promised a cuda index pin (verified in the 18-02 vocabulary, no pip-lane block edited beyond default flips)"

patterns-established:
  - "Sanctioned-survivor reconciliation: when a re-point plan's survivor list and its count arithmetic conflict, the count gates (the executable truth) win; document the reconciliation in the ledger"

requirements-completed: [PKGH-05, DOCR-02]  # docs-surface portion; PKGH-05's ci.yml leg lands in 19-07, DOCR-02's per-file review in 19-03..19-06 — REQUIREMENTS.md flips blocked by the shared-ID gate until the last declaring plan finishes

coverage:
  - id: D1
    description: "Mirrored headline + wrappers: overview.md both sides byte-identical with cuda130 at :9 prose and :25 bash (C1=C2=2 per side); 19 notebook wrappers + 3 marimo wrappers (install line only, See-Also tails untouched) on .[full,cuda130]; example/ cuda124-free"
    requirement: PKGH-05
    verification:
      - kind: other
        ref: "Task 1 automated verify: example-side cuda124=0, wrapper [full,cuda124] files=0, C1=C2=2, check_docs_sync OK -> MIRROR-INTACT"
        status: pass
    human_judgment: false
  - id: D2
    description: "installation.md default flip (cuda130 leads the GPU section and the extras table annotation; 6 scenario/troubleshooting sites flipped; 5 sanctioned survivors + label-stripped table row = I:6) plus quick_start/user_guide-getting_started/index/tutorial generic defaults, gpu_support.md untouched at G:4 (Turing row intact), lane labels 29"
    requirement: PKGH-05
    verification:
      - kind: other
        ref: "Task 2 automated verify: I=6 (band 1..6), T=14 (>=13), IC=0 (<=1), R=0, G=4, Q=1, U=0, W=0, N=1, L=29, census 216/217 -> CENSUS-STABLE"
        status: pass
    human_judgment: false
  - id: D3
    description: "README: cuda130 recommended default in headline/examples, '(recommended)' moved to the cuda130 table row, stale 5090 torch==2.7 line replaced by the 18-02 matrix guidance, three stale '1,900+' claims replaced by 2556-based counts, exactly 3 sanctioned survivors (:176/:180/:201)"
    requirement: PKGH-05
    verification:
      - kind: other
        ref: "Task 3 automated verify: A=0, B=0, S=0, RC=0, R=3 (<=3), T=6 (>=6) -> README-GATES-GREEN"
        status: pass
    human_judgment: false
  - id: D4
    description: "No regression anywhere: extras guard 11 passed, both sync gates green, fast lane exactly 2556 passed / 1 skipped / 75 deselected, census 216/217 stable, mkdocs --strict exit 0 with zero warnings (19-01's state survived)"
    verification:
      - kind: other
        ref: "pytest tests/test_extras_guard.py -q --no-cov -> 11 passed; check_docs_sync OK; check_notebook_md_sync 24 pairs; pytest -m 'not slow' --no-cov -q -> 2556 passed, 1 skipped, 75 deselected; mkdocs build --strict -> exit 0, 0 warnings -> BATTERY-GREEN"
        status: pass
    human_judgment: false

duration: 20 min
completed: 2026-10-15
status: complete
---

# Phase 19 Plan 02: cuda130 Default Re-Point Summary

**cuda130 re-pointed to the recommended default across ~30 generic install sites — mirrored overview.md pair (byte-identical same-commit), 22 wrapper install lines, installation.md lead/scenarios/table flip, four getting-started surfaces, README headline/table/examples + honest 2556-based test counts — with the Turing exception intact (gpu_support.md untouched at 4) and the full battery green**

## Performance

- **Duration:** 20 min (1,188 s)
- **Started:** 2026-10-15T00:50:20Z
- **Completed:** 2026-10-15T01:10:08Z
- **Tasks:** 3/3
- **Files modified:** 31 (24 mirror/wrapper files + 6 getting-started/README docs + ledger)

## Accomplishments

- **PKGH-05 (docs surface):** every generic-default install command now recommends cuda130 — the mirrored `overview.md` pair (both `.[full,notebook,cuda130]` sites per side, byte-identical in one commit, `check_docs_sync` OK), the 19 docs-side notebook wrappers + 3 marimo wrappers (`.[full,cuda130]`, install line only), installation.md's GPU-section lead + extras-table annotation + 6 scenario/troubleshooting sites, quick_start / user_guide-getting_started / index / tutorial, and README's headline block, table row, and example. `example/` carries zero cuda124 strings.
- **DOCR-02 (re-verified against the final extras surface):** cuda130 exists as re-pointed (`torch>=2.9.0,<2.12`, uv index `torch-cuda130`, verified basis torch 2.11.0+cu130); the flipped Scenario-6 combination `.[cuda130,mamba]` resolves (mamba `torch>=2.6.0,<2.12` ∩ cuda130 `torch>=2.9.0,<2.12`).
- **Honesty, both directions:** the Turing exception survives untouched — `gpu_support.md` verify-only at cuda124=4 ("20xx → cuda124–cuda128, NOT cu130"), README:180's matrix pointer and :176's hardware-group warning kept, and every re-pointed GPU-choice block keeps/links the GPU-matrix pointer. The README's stale "Nvidia 5090 → cuda128 & torch==2.7" line is replaced by the 18-02 matrix guidance (driver < 580: cuda128; driver >= 580: cuda130 works).
- **Dated numbers honest:** the three README "1,900+" / "~1,900" claims now state the 2556-passed fast-lane baseline ("2,500+ test cases" ×2, "fast lane (2,556 tests)" ×1).
- **Ledger:** 19-REVIEW-repoint.md created with per-surface before/after counts, the sanctioned-survivor lists as landed, both reconciliation notes, and the battery results — rows same-commit as the fixes throughout.

## Task Commits

Each task was committed atomically:

1. **Task 1: Mirrored headline + wrappers — overview.md both sides byte-identical, 19+3 wrapper install lines** - `b7f1985` (docs)
2. **Task 2: getting-started surfaces — installation.md default flip + quick_start/getting_started/index/tutorial** - `3a13419` (docs)
3. **Task 3: README + full gate battery — headline/table/examples, stale 5090 line, test counts, strict regression** - `313579e` (docs)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified

- `example/notebooks/overview.md` + `docs/example/notebooks/overview.md` - the PKGH-05 headline pair, `.[full,notebook,cuda130]` at :9/:25 per side, byte-identical
- `docs/example/notebooks/*.md` (19 files, ~:17) + `docs/example/marimo/{benchmark,finetune,inference}/*_demo.md` (:18) - wrapper install lines → `.[full,cuda130]` (marimo See-Also tails untouched)
- `docs/getting_started/installation.md` - GPU lead: cuda130 "(recommended default)", cuda124 demoted to prose older-GPU lane with matrix pointer; extras table "(recommended)" moved cuda124→cuda130 row; Scenarios 2/6/7/8 + troubleshooting flipped; 5 sanctioned survivors + label-stripped table row
- `docs/getting_started/quick_start.md` - same lead-flip shape; :328 choose-ONE enumeration kept
- `docs/user_guide/getting_started.md` - `.[cuda130]` with Turing prose note
- `docs/index.md` - :79 example comment → `.[all,cuda130]`; :78 enumeration kept
- `docs/user_guide/tutorial/end_to_end_workflow.md` - Dockerfile RUN → `.[full,cuda130] --system`
- `README.md` - install block restructured (cuda130 default leads, 5090 line replaced per matrix), table annotation moved, :211 example flipped, test counts → 2556-based
- `.planning/phases/19-docs-accuracy-sweep-milestone-closeout/19-REVIEW-repoint.md` - the ledger (created Task 1, appended Tasks 2-3)

## Decisions Made

- **:457 reconciliation (see Deviations):** flipped to `.[cuda130,mamba]` — the plan's count arithmetic overrode its own "keep" parenthetical
- **Prose-lane demotion:** the older-GPU lane survives as spaced "CUDA 12.4–12.8 lanes" prose + GPU-matrix pointer wherever a literal cuda124 install command would break the plan's install-cmd count gates (installation.md lead, README block) — no honest fact deleted, only the default moved
- **Test-count phrasing:** "2,500+ test cases" for suite-size claims, exact "2,556 tests" in the CI fast-lane sentence (the plan's example phrasing, consistent across all three sites)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Plan-literal conflict] :457 (Scenario 6) flipped despite the action-(1) "keep" parenthetical**
- **Found during:** Task 2 (edit-map construction)
- **Issue:** the plan's action (1) parenthetical marks :457's `.[cuda124,mamba]` "sanctioned survivor, keep", but its own arithmetic contradicts it: the 5 listed survivors (:213/:223/:255/:268/:272 post-shift) + the label-stripped extras-table row (kept per action (1)'s annotation-move) already total I=6 — the verify band's exact ceiling — so keeping :457 lands I=7 outside `[ "$I" -le 6 ]` and RC/IC coherence breaks
- **Fix:** flipped :457 to `.[cuda130,mamba]` per the main action clause (":309, :457, :474, :497, :573 switch their generic default") and the three "exactly 5 sanctioned survivors" statements; the mamba-combination example class survives at :255 (warning) and README :176
- **Files modified:** docs/getting_started/installation.md
- **Verification:** I=6 (band 1..6), IC=0 (≤1), combination resolves against the final extras surface
- **Committed in:** 3a13419 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 plan-literal conflict resolved toward the plan's own executable gates)
**Impact on plan:** No scope change — the flip was already the majority reading of the plan text; documenting the reconciliation keeps the ledger honest for 19-07's consolidation.

## Issues Encountered

- Task 3's first block rewrite dropped `cuda130` from README's "Other supported versions" enumeration, landing T=4 against the ≥6 gate — the plan's own gate caught it pre-commit; restored the enumeration verbatim (the plan's "the 'Other supported versions' enumeration stays") and carried the driver>=580/cuda130 fact on the 5090 line → T=6, gates green. Fixed before the Task 3 commit.

## Authentication Gates

None — docs-only plan, no authenticated surfaces touched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 19-03..19-06 (review batches) verify prose against the cuda130-default state; the sanctioned-survivor lists as landed are recorded in 19-REVIEW-repoint.md
- 19-07 (closeout) carries PKGH-05's remaining surface (ci.yml test-cuda leg → 13.0 + the temporary v1.3 push-filter entry) and consolidates this ledger into 19-EVIDENCE.md
- REQUIREMENTS.md flips for PKGH-05/DOCR-02 remain blocked by the shared-ID gate until the last declaring plan (19-07 / 19-03..19-06) finishes — expected and correct
- All gates green at HEAD 313579e: extras guard 11, sync gates, fast lane 2556/1/75, census 216/217, strict 0 warnings

## Self-Check: PASSED

- All key-files exist on disk (SUMMARY, ledger, mirrored pair, 6 getting-started/README docs, wrapper samples) — `[ -f ]` FOUND for each
- All 3 task commits (b7f1985, 3a13419, 313579e) are ancestors of HEAD — `git merge-base --is-ancestor` FOUND
- `gsd-tools check evaluation-scope --plan 19-02 --commits-only` resolves with 3 commits on this branch
- Frontmatter `commits: 3` MEASURED from the persisted ledger (`git rev-list --count e564845..HEAD` = 3)

---
*Phase: 19-docs-accuracy-sweep-milestone-closeout*
*Completed: 2026-10-15*
