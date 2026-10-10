---
phase: 08-full-execution-rollout-repair-loop
plan: 05
subsystem: testing
tags: [megadna, finetune-custom-head, pinned-install, provenance-stamp, docs-mirror, census-reconciliation, reversible-venv-install]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: D-03 census baseline + family view (08-02); DNATokenizer generate repair + isolated dnallm-megadna lane + finetune_generation first green (08-04)
provides:
  - Repaired megaDNA sibling notebooks (finetune_custom_head demo cell + generation_megaDNA pinned install) with D-21 stamps and D-15-verified source= routes
  - First real green executions of both siblings on the default project kernel (reversible pinned prereqs in the project venv)
  - Fast JSON content contracts pinning the sibling repair invariants (3 tests)
  - D-03 family reconciliation: all three megaDNA-family census rows PASS, family view CLOSED
affects: [08-08, 08-09]

actuals:
  tokens: 5789   # chars/4 over the realized diff (23154 chars); estimate 52000 overshot the same way 08-04's did — execution time dominates, not diff size
  tasks: 2
  commits: 2     # measured: git rev-list --count a9be98c..HEAD (081a886, 5d88028)
plan_head_before: a9be98c2a3e7604f942e62d950605b35a98d1a25
plan_head_after: 5d8802850a9d47b8ed97570b1e64546b40d7392

tech-stack:
  added: []   # nothing new; megadna 1.0 (pinned clone) + MEGABYTE-pytorch 0.2.1 live in the project .venv as reversible exact-version installs, never pyproject extras
  patterns: [active-line source= contract checking (notebooks document alternative source= routes as comments — strip comment lines before asserting the D-15 route), per-family census closure rows (name-then-PASS rollup rows reconciled against the 08-02 baseline)]

key-files:
  created: []
  modified:
    - example/notebooks/finetune_custom_head/finetune.ipynb
    - example/notebooks/generation_megaDNA/inference.ipynb
    - docs/example/notebooks/finetune_custom_head.md
    - docs/example/notebooks/finetune_custom_head/finetune.ipynb
    - docs/example/notebooks/generation_megaDNA/inference.ipynb
    - docs/example/notebooks/inference_megaDNA.md
    - tests/examples/test_notebook_execution.py
    - .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md

key-decisions:
  - "finetune_custom_head demo-cell repair = executable pinned install cell placed immediately before the megaDNA model load (the census ImportError fired at megadna.py:146 inside that load); structural freedom per D-02 was used for exactly this insertion, nothing else moved"
  - "generation_megaDNA's commented floating clone cell was replaced in place by the same executable pinned form — the notebook is now self-sufficient instead of documenting an unpinned install"
  - "source= routes were verified already D-15-aligned and left unchanged (plant-dnagpt-BPE -> modelscope active, lingxusb/megaDNA_updated -> huggingface active); the new contract test checks ACTIVE lines only because both notebooks document the alternative route as a comment"
  - "Sibling un-gating rides a reversible exact-version install into the project .venv (uv pip install megadna @ pinned clone + MEGABYTE_pytorch==0.2.1; reverse: uv pip uninstall megadna megabyte-pytorch) — the default-kernel pin (test_megadna_siblings_keep_default_project_kernel) is preserved; the installing finetune_generation stays on the isolated kernelspec (T-08-19)"
  - "Wrapper mirrors stay hand-curated: the generator skeleton provided the excerpts, the committed wrappers keep their prose and add the stamp/install sections — check_notebook_md_sync (AST) + check_docs_sync (byte) are the commit gates, the established 08-04 pattern"

patterns-established:
  - "Active-line contract checking: when a notebook documents alternative API routes as comments, content contracts must strip comment lines before asserting which route is active"

requirements-completed: [EXEC-02, REPAIR-01]

coverage:
  - id: D1
    description: "Repaired sibling notebooks with D-21 stamps, pinned install cells, and D-15-aligned source= routes; mirrors byte-synced in the same commits (D-20)"
    requirement: REPAIR-01
    verification:
      - kind: unit
        ref: "python3 scripts/check_notebook_md_sync.py (24/24) + python3 scripts/check_docs_sync.py both exit 0 at commit 081a886; tests/examples/test_notebook_execution.py TestMegadnaSiblingContentContracts 3/3 green"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both siblings executed end-to-end for real on the dev box (default project kernel, reversible pinned prereqs) — zero skips"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "pytest -k 'megadna or custom_head or finetune_generation': 12 passed / 0 SKIPPED in 2885.10s (/tmp/08_05_family.log); finetune_custom_head ran BOTH trainings (DNAGPT + megaDNA halves)"
        status: pass
    human_judgment: false
  - id: D3
    description: "D-03 family reconciliation: all three megaDNA census rows PASS and the family view is CLOSED; fast lane green with zero new skips"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "grep name-then-PASS green for generation_megaDNA / finetune_custom_head / finetune_generation in 08-CENSUS-ROLLUP.md; full fast lane 1800 passed / 1 pre-existing skip, exit 0 in 91.79s"
        status: pass
    human_judgment: false
  - id: D4
    description: "Project .venv carries exactly the reversible pinned prereqs (megadna 1.0 @ cb2f5ab4 clone, MEGABYTE-pytorch 0.2.1) with no other residue"
    requirement: EXEC-02
    verification:
      - kind: manual_procedural
        ref: "pip list | grep -iE 'megadna|megabyte' -> exactly MEGABYTE-pytorch 0.2.1 + megaDNA 1.0; reversible uninstall recorded in SUMMARY"
        status: pass
    human_judgment: false

status: complete
duration: 57min
completed: 2026-10-04
---

# Phase 8 Plan 5: megaDNA Family Siblings — Repair, Execution, D-03 Reconciliation Summary

**finetune_custom_head's install-gated demo cell and generation_megaDNA's floating clone repaired to pinned self-sufficient form (D-21 stamps, synced mirrors), both siblings executed green on the project venv via a reversible pinned install, and all three megaDNA census rows flipped to PASS — the family is closed**

## Performance

- **Duration:** 57 min (18:32–19:29 UTC; 48 of them the family execution lane — two full trainings in finetune_custom_head dominate)
- **Started:** 2026-10-04T18:32:23Z
- **Completed:** 2026-10-04T19:29:36Z
- **Tasks:** 2/2
- **Files modified:** 8 (2 notebooks, 4 docs mirrors, 1 test module, 1 census rollup)

## Accomplishments

- **Demo-cell repair landed.** The census failure (`ImportError: megaDNA package is required ...` at dnallm/models/special/megadna.py:146, fired inside the demo model-load cell because the checkpoint unpickles classes from the `megaDNA` package) is fixed by an executable pinned install cell placed immediately before that load — clone @ cb2f5ab4cc88dc0effe05c5f23358862c837014a + `MEGABYTE_pytorch==0.2.1` via `uv pip install` targeting the running kernel's `VIRTUAL_ENV`. Never a floating clone (FEASIBILITY lock).
- **generation_megaDNA is self-sufficient.** Its only prerequisite documentation was a commented floating `!git clone` — now the same executable pinned form, plus a D-21 stamp (the generate path rides 08-04's DNATokenizer unknown→id-1 fix, precondition-verified committed).
- **Both siblings green by real execution.** With the pinned prereqs installed into the project .venv (reversible), `_gate_megadna` probed green and both notebooks executed end-to-end on the default project kernel — finetune_custom_head ran BOTH trainings (DNAGPT 417 steps + the megaDNA half). Family lane: **12 passed / 0 SKIPPED in 2885s**.
- **D-03 family reconciliation complete.** 08-CENSUS-ROLLUP.md's three megaDNA rows (both siblings + finetune_generation from 08-04) are PASS with evidence; the family-view table marks megaDNA CLOSED. Full fast lane: **1800 passed / 1 pre-existing skip, exit 0** (08-04 baseline 1797 + the 3 new contract tests — zero new skips).
- **Fast contracts pin the repairs.** TestMegadnaSiblingContentContracts (3 tests) pins the D-21 stamps, the pinned-install-before-load ordering, and the D-15 source= routes — checking ACTIVE lines only, since both notebooks document the alternative route as a comment.

## Task Commits

1. **Task 1: sibling notebook repairs + mirrors** — `081a886` (fix)
2. **Task 2: family execution + D-03 census reconciliation** — `5d88028` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `example/notebooks/finetune_custom_head/finetune.ipynb` — D-21 stamp cell + environment note; pinned install cell before the megaDNA demo load
- `example/notebooks/generation_megaDNA/inference.ipynb` — floating clone comment → executable pinned install; D-21 stamp
- `docs/example/notebooks/finetune_custom_head.md` + `docs/example/notebooks/inference_megaDNA.md` — wrapper excerpts + stamp/install sections (D-20 same-commit)
- `docs/example/notebooks/finetune_custom_head/finetune.ipynb`, `docs/example/notebooks/generation_megaDNA/inference.ipynb` — raw byte-identical copies
- `tests/examples/test_notebook_execution.py` — TestMegadnaSiblingContentContracts (3 fast JSON contracts)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` — megaDNA rows PASS + family view CLOSED

## Decisions Made

- See key-decisions (frontmatter). The load-bearing one: the sibling un-gating rides a reversible exact-version install into the project .venv rather than a kernel override — the default-kernel pin test (08-04) is preserved by design, and the installing notebook (finetune_generation) keeps its isolated kernelspec, matching the T-08-19 mitigation.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Both sync gates were green at baseline and stayed green through both commits; the family lane passed on the first run with zero surfaced failures (no REPAIR-01 triage needed beyond the planned repairs themselves).

## Reversible Install Record (T-08-19 / T-08-SC)

- **Installed into the project .venv** (exact pins, never pyproject extras):
  - `megadna==1.0` from the local pinned clone `.scratch/megadna-venvs/megadna-clone-src` @ `cb2f5ab4cc88dc0effe05c5f23358862c837014a`
  - `MEGABYTE_pytorch==0.2.1`
- **Reverse:** `uv pip uninstall --python .venv/bin/python megadna megabyte-pytorch`
- Verified post-lane: `pip list` shows exactly those two entries and nothing else megaDNA-related; the repo tree stayed clean through all executions (`assert_tree_clean`).

## User Setup Required

None - no external service configuration required. (Boxes without the pinned prerequisites keep getting the honest `optional-dep:` typed skip from `_gate_megadna` — the family gate remains probe-then-execute.)

## Next Phase Readiness

- The megaDNA family is fully closed for D-03: all three notebooks PASS by real execution, isolation respected (finetune_generation on `dnllm-megadna`, siblings on the project venv), census reconciled against the 08-02 baseline.
- 08-08/08-09 (runner wiring + final census) inherit: the runner job steps must reproduce the reversible project-venv install for the two siblings (or pin a dedicated lane) — never an implicit test-time install; the D-15 lock entry for lingxusb/megaDNA_updated (HF-only id) lands in 08-09 per the plan's own note.
- Remaining gated families for later plans: evo (08-06/08-07 area), lora/mamba pair, mcp/ollama pair.

## Self-Check: PASSED

- All 8 key-files committed (git diff a9be98c..5d88028 lists exactly them)
- Commits 081a886 / 5d88028 present on phs (2 measured from the a9be98c ledger)
- Task 1 verify re-run: both sync scripts exit 0; Task 2 verify components re-run: contracts 11 passed, census greps green, family log 12 passed / NO-SKIPS, fast lane 1800/1 exit 0

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
