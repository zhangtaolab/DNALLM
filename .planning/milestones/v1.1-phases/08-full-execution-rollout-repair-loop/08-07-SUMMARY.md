---
phase: 08-full-execution-rollout-repair-loop
plan: 07
subsystem: testing
tags: [evo-family-closure, full-census-reconciliation, A4-load-time-proof, OPTIONAL_IMPORT_MODULES, giants-tier]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: empty OPEN-RUNG ledger + giants tier + isolated dnallm-evo lane + repaired notebook (08-06); allow_patterns/spec-env/np.fromstring shims (08-03)
provides:
  - evo family CLOSED: census row PASS by real execution with zero open rungs (EXEC-02 family closure)
  - A4/CI-05 empirically verified end-to-end at load time (offline load, evo-1 .pt-free, no 16.81GB re-fetch)
  - Full examples census reconciled post-evo: 188 P / 5 honest typed skips / 0 F; fast lane 1803 P / 1 pre-existing skip
  - OPTIONAL_IMPORT_MODULES extended with the evo venv-only stack (flash_attn/stripedhyena/evo2)
affects: [08-08, 08-09]

actuals:
  tokens: 1634   # chars/4 over the realized diff (6534 chars); estimate 44000 — the zero-rung collapse held, diff size never drives this plan class
  tasks: 2
  commits: 2     # measured: git rev-list --count a15fdaf..HEAD (cd6debd, 648b91b)
plan_head_before: a15fdafa87e03e753803601995ba902a3dcede36
plan_head_after: 648b91b

tech-stack:
  added: []
  patterns: [OPTIONAL_IMPORT_MODULES as the documented-optional-dep seam for venv-locked family stacks (pybedtools precedent extended), chunked census re-runs (three disjoint -k passes) to fit runner time caps]

key-files:
  created: []
  modified:
    - tests/examples/test_examples.py
    - .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md

key-decisions:
  - "Task 1 collapsed to a confirmation run exactly per the plan's flagged assumption: 08-06's OPEN-RUNG ledger is legitimately empty (first execution exited 0 under transformers 5.18), so no shim work was invented — the evo lane re-ran green (5 P / 0 SKIPPED, 49.45s) and tests/utils stayed 168 P"
  - "Census run chunked into three disjoint -k passes (51:28 + 48:21 + 50:34) instead of one 2.5h+ invocation — every pass under the 2h background cap, full 193-item coverage preserved, combined log at /tmp/08_07_examples.log"
  - "08-06's D-21 stamp cell (literal 'import flash_attn') made the static import check fail in the project venv — fixed at the check's documented-optional-dep seam (OPTIONAL_IMPORT_MODULES), NOT by touching the notebook or masking the check"
  - "The plan's unscoped 'find ~/models-giants -name *.pt == 0' verify collides with evo2's OWN native .pt checkpoint (present since 08-06's recorded 2.7GB fetch); the CI-05 truth it encodes — no 16.81GB evo-1 pytorch_model.pt re-fetch — is asserted scoped to the evo-1 snapshot (0 .pt) plus an unchanged 14,889 MiB blob store"

patterns-established:
  - "Zero-rung collapse discipline: an empty OPEN-RUNG ledger means the expansion plan confirms, never invents, work"
  - "Full-census lane chunking: disjoint -k partitions with per-chunk rc capture, concatenated for the skip greps"

requirements-completed: [EXEC-02, CI-05, REPAIR-01]

coverage:
  - id: D1
    description: "evo family fully green by real execution with zero open rungs — confirmation run of the isolated lane plus tests/utils regression sweep"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "pytest -k evo: 5 passed / 0 SKIPPED in 49.45s, rc=0; combined census log greps NO evo SKIPPED; tests/utils 168 passed rc=0"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-03 census reconciliation: evo row PASS, family CLOSED, full examples lane 188 P / 5 honest typed skips / 0 F post-repair"
    requirement: REPAIR-01
    verification:
      - kind: integration
        ref: "three census passes 14P+25P+149P (pre-fix import failure re-run green 106P/1S); rollup grep 'generation_evo_models.*PASS' OK; fast lane 1803 P / 1 pre-existing skip exit 0 91.46s"
        status: pass
    human_judgment: false
  - id: D3
    description: "A4/CI-05 load-time proof: safetensors-only evo-1 snapshot loads with no .pt re-fetch; giants dir outside cached paths"
    requirement: CI-05
    verification:
      - kind: automated_ui
        ref: "exit-code chain: evo-1 snapshot find *.pt == 0, blobs 14,889 MiB unchanged (12.3+2.7GB), HF_HUB_OFFLINE execution log zero fetch lines, refs/main present both models"
        status: pass
    human_judgment: false

status: complete
duration: 2h 37min
completed: 2026-10-04
---

# Phase 8 Plan 7: evo Giants Expansion — Full Green + Census Reconciliation Summary

**evo family CLOSED at depth: zero-rung confirmation run green (5 P / 0 SKIPPED, 49s), full examples census reconciled to 188 P / 0 F with the import-check seam repaired, and A4/CI-05 proven at load time — offline load, evo-1 snapshot .pt-free, no 16.81GB re-fetch**

## Performance

- **Duration:** ~2h 37min (19:58–22:35 UTC; ~2h 32m of it the three sequential census passes + fast lane, run in background while idle)
- **Started:** 2026-10-04T19:58:08Z
- **Completed:** 2026-10-04T22:35:26Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- **Task 1 — residual sweep collapsed to confirmation (as flagged).** 08-06's rung ledger is legitimately empty (first real execution exited 0; transformers 5.18 carried both model legs). The evo lane re-ran green by real execution — **5 passed / 0 SKIPPED in 49.45s** — and `tests/utils` (the shim contract suites incl. the absence-guard roster) passed **168/168**. No rung was invented; `transformers_compat.py` untouched.
- **Task 2 — full D-03 census reconciled.** `tests/examples` re-run completely post-evo-repairs in three disjoint chunked passes (NER family + multi_labels 14 P in 51:28; megaDNA family 25 P in 48:21; everything else 149 P in 50:34): **188 passed / 5 honest typed skips / 0 failed**. The 5 skips are the documented out-of-scope pairs (2 mcp network-unavailable with probe evidence, 2 lora optional-dep for 08-08) plus 1 benign no-imports. Fast lane: **1803 passed / 1 pre-existing skip, exit 0, 91.46s** — +3 vs the 08-05 baseline are exactly 08-06's TestEvoIsolatedLane; zero new skips.
- **A4 VERIFIED (CI-05 empirical proof).** The census execution loaded the evo-1 giants snapshot under `HF_HUB_OFFLINE=1` — zero fetch lines in the log, re-fetch impossible by construction. The evo-1 snapshot dir holds **zero `.pt` files** (safetensors + index + configs, 12.3GB window, `refs/main` present) and the giants blob store is unchanged at **14,889 MiB** (evo-1 12.3GB + evo2 2.7GB) — no 16.81GB `pytorch_model.pt` ever landed. The single `.pt` in `~/models-giants` is evo2's own native checkpoint from 08-06's recorded fetch (scoped note added to the rollup). Giants dir remains outside every cached path.
- **Census rollup updated.** evo row → PASS (name-then-PASS row format, verify grep green), family view → CLOSED (08-07), A4 + reconciliation block appended for 08-09.

## Task Commits

1. **Task 1: residual sweep (zero-rung confirmation)** — no code changes, no commit (per the plan's flagged-assumption collapse; green evidence above)
2. **Task 2 fix: import-check seam** — `cd6debd` (fix)
3. **Task 2: census rollup reconciliation** — `648b91b` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `tests/examples/test_examples.py` — OPTIONAL_IMPORT_MODULES += flash_attn/stripedhyena/evo2 with the 08-06 FEASIBILITY-lock comment (pybedtools precedent)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` — evo row PASS, family CLOSED (08-07), A4 verification + reconciliation block

## Decisions Made

- Census chunking into three disjoint `-k` passes rather than one 2.5h+ run (single-run would exceed the 2h background cap); coverage is exact (14+25+154 selected = 193 items) and the skip greps run on the concatenated log.
- The import-check fix landed at the check's own optional-dep seam, not in the notebook — the stamp cell is deliberate 08-06 content; the project venv must stay free of the evo stack.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 08-06's D-21 stamp cell broke the static import check in the project venv**
- **Found during:** Task 2 (census chunk 3: `test_notebook_imports[notebooks/generation_evo_models/inference.ipynb]` FAILED — `import flash_attn: No module named 'flash_attn'`)
- **Issue:** The stamp cell's literal `import flash_attn` can never resolve in the project venv, where the evo stack is FEASIBILITY-locked to the throwaway dnallm-evo venv; 08-06 only ran the `-k evo` execution slice, so the latent fast-check regression surfaced only in this full census — exactly the reconciliation class Task 2 exists to catch
- **Fix:** Extended `OPTIONAL_IMPORT_MODULES` (the pybedtools documented-optional-dep seam) with `flash_attn`, `stripedhyena`, `evo2` + a comment naming the 08-06 lock; the fast module re-ran **106 passed / 1 benign skip** post-fix
- **Files modified:** tests/examples/test_examples.py (outside the plan's declared file list — conditional files were scoped to the never-materialized rung shims)
- **Verification:** ruff check + format green; fast module green; combined census log greps green
- **Committed in:** cd6debd

**2. [Rule 1 - Plan-defect] Task 2's unscoped `.pt`-count verify can never pass as written**
- **Found during:** Task 2 (A4 evidence collection: `find ~/models-giants -name '*.pt' | wc -l` returned 1)
- **Issue:** The verify sweeps the whole giants dir, but `evo2_1b_base.pt` is evo2's OWN native checkpoint format, present since 08-06's recorded 2.7GB full fetch (recorded in 08-06-SUMMARY); the truth the check encodes is the CI-05 evo-1 guarantee
- **Fix:** Asserted the guarantee scoped the way 08-06 itself measured it: evo-1 snapshot `.pt` count == 0 (exit-code asserted) AND giants blob store unchanged at 14,889 MiB (no 16.81GB growth); scoped note added to the census rollup so 08-09 inherits the correct assertion
- **Files modified:** .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md (note)
- **Verification:** EVO1_PT_FREE echoed; blobs du re-measured post-execution
- **Committed in:** 648b91b

---

**Total deviations:** 2 auto-fixed (2 bugs incl. 1 plan-expression defect). **Impact:** Both necessary for the census gate to be honest and passable; no scope creep — the evo stack stays out of the project venv.

## Issues Encountered

- Chunk 3's single pre-fix failure is Deviation 1; everything else in the census ran green on the first pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **08-08 (mamba/lora family)** inherits a fully reconciled census: its two `optional-dep` lora skips are the only gated notebooks left besides the endpoint-gated mcp pair.
- **08-09 (final census/runner)** must reuse the scoped `.pt` assertion (evo-1 dir, not the whole giants tree) and the chunked-census pattern if it re-runs the full lane; A4 is closed — record it as verified in the final rollup.
- Giants tier + dnallm-evo venv + kernelspec remain dev-box runtime artifacts (uncommitted); 08-06's runner-wiring facts still apply verbatim.

## Self-Check: PASSED

- Both key files committed (git diff a15fdaf..648b91b lists exactly them); commits cd6debd / 648b91b present on phs (2 measured from the a15fdaf ledger)
- Task 1 verify re-run: evo 5 P / 0 SKIPPED rc=0, tests/utils 168 P rc=0; Task 2 verify greps: NO_EVO_SKIP, ROW_PASS_OK (name-then-PASS), EVO1_PT_FREE, fast lane 1803 P rc=0
- Census rollup row + family table + A4 block present; ruff green on the touched test file

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
