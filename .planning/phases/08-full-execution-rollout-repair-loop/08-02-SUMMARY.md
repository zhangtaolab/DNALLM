---
phase: 08-full-execution-rollout-repair-loop
plan: 02
subsystem: testing
tags: [modelscope-cache, transformers-compat-shims, nbclient, pytest, pyproject, ipython-pin, rice-outage, sandbox-seeding, census]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: ACTIVE/GATED lanes, nbclient harness, typed-skip allowlist, 05-CENSUS inventory, census input cache
provides:
  - D-17 fully dispositioned per item with pristine-cache proof (08-D17-DISPOSITION.md)
  - EXEC-04 closed — script lane heals to real green (marker conversion now dead fallback)
  - EXEC-05 re-verified — YAML 21/21, fast lane 1761/1 with zero new skips
  - D-03 reconciliation baseline census (08-CENSUS-ROLLUP.md) for every later plan
  - ipython<9 kernel-plot compat pin in the notebook extra with static+dynamic guards
  - rice.uga.edu outage resilience (census-cache seeding for script + data_generation lanes)
  - combined showcase sibling-seeding repair (the 08-01 runner FileNotFoundError class, healed + contract-tested)
affects: [08-03, 08-04, 08-05, 08-06, 08-07, 08-08, 08-09]

actuals:
  tokens: 10874   # chars/4 over the plan's realized diff (43498 chars); estimate 88000 overshot because the plan is execution-dominated
  tasks: 3
  commits: 7      # measured from plan_head_before; 4 are this plan's (439370f, 328c125, e90ff21, c5f0916) — 3 concurrent owner chores landed in-range (c1b6e03, 1354d75, 62369fe)
plan_head_before: 876e576f74efad645db9093030becc86bfe9130e
plan_head_after: c5f0916cf57fbcd08bdce8d86f067d5c0fa2f891

tech-stack:
  added: []   # no new libraries; ipython>=8.31,<9 is a constraint on an existing transitive dep
  patterns: [census-cache-first input seeding exploiting wget -c skip-complete, COMBINED_EXTRA_INPUTS sibling-seeding with JSON-parsed coverage contract test, installed-pair compat probe (IPython x matplotlib) beside static extras guards]

key-files:
  created:
    - .planning/phases/08-full-execution-rollout-repair-loop/08-D17-DISPOSITION.md
    - .planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md
  modified:
    - pyproject.toml
    - tests/test_extras_guard.py
    - tests/examples/test_script_execution.py
    - tests/examples/test_notebook_execution.py
    - tests/examples/test_plant_helixseek_showcase.py

key-decisions:
  - "D-17 verdict executed as shim-covered everywhere: the 9-shim layer closes all NT rungs on a pristine snapshot — the 05-04 typed-skip termination stays superseded; marker conversion survives as dead fallback code"
  - "ipython>=8.31,<9 pinned in the notebook extra (not a matplotlib move): pgt's hard <3.9 cap makes IPython 8 the only lever; floor 8.31 = first IPython 8 with Python 3.13 support"
  - "rice.uga.edu outage handled by census-cache seeding, not URL changes: cache is a self-refreshing mirror of the documented URLs; cold cache keeps full download provenance and the honest typed skip; 4xx still re-raises (WR-04)"
  - "combined-notebook sibling gap fixed at the harness seeding layer (benchmark _NOTEBOOK_EXTRA_INPUTS precedent), with a fast contract test that parses the notebook JSON for ../ refs so the next missing sibling fails in seconds, not on the nightly"

patterns-established:
  - "Installed-pair guard: static extras assertions plus a live-environment probe that red-flags broken resolved pairs (the mpl/ipython class) without spawning kernels"
  - "Outage-resilient input seeding: cache-first for dev-box lanes, download-first for CI/fresh checkouts, typed skip only when both are cold"

requirements-completed: [EXEC-04, EXEC-05, REPAIR-01, REPAIR-03]

coverage:
  - id: D1
    description: "D-17 per-item disposition on a pristine NT snapshot (restore + delete-and-refetch reproducibility + 3 NT-family notebook tests green on shims alone)"
    requirement: REPAIR-03
    verification:
      - kind: integration
        ref: "pytest -k 'benchmark or ner or embedding' one-pass: 5 passed / 3 gated-skips / exit 0; NER 33:20 full training; benchmark + embedding green on re-fetched pristine cache"
        status: pass
    human_judgment: false
  - id: D2
    description: "EXEC-04 — generate_bpe_dataset.py heals past its typed-skip marker to real green with a fresh in-sandbox dataset artifact"
    requirement: EXEC-04
    verification:
      - kind: integration
        ref: "tests/examples/test_script_execution.py — 4 passed, no SKIPPED naming generate_bpe_dataset, verify gate exit 0; artifact 11,299,268 B fresh"
        status: pass
    human_judgment: false
  - id: D3
    description: "EXEC-05 — YAML leg 21/21 via real load_config and fast lane green with zero new skips (1761/1)"
    requirement: EXEC-05
    verification:
      - kind: unit
        ref: "scripts/validate_yaml.py 'All YAML files passed validation.' (21/21) + tests/configuration/test_yaml_load.py 21 passed + fast lane 1761 passed/1 pre-existing skip exit 0"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-03 reconciliation baseline census recorded per item (21 notebooks + 3 marimo + 1 script + YAML) with family view for 08-04..08-08"
    requirement: REPAIR-01
    verification:
      - kind: integration
        ref: "08-CENSUS-ROLLUP.md verify gate: all-passed grep + file exists + 42 table pipe-lines >= 26"
        status: pass
    human_judgment: false
  - id: D5
    description: "En-route failures repaired same-plan with tests: ipython<9 pin (439370f), rice cache seeding (e90ff21, c5f0916), combined sibling seeding (c5f0916)"
    requirement: REPAIR-01
    verification:
      - kind: unit
        ref: "tests/test_extras_guard.py 5 passed; TestRiceInputSeeding 3; TestRiceCacheExtras 3; TestCombinedSiblingSeeding 2; combined notebook re-run 1 passed in 197.54s"
        status: pass
    human_judgment: false

status: complete
duration: 7h 27min
completed: 2026-10-04
---

# Phase 8 Plan 2: NT Pristine Restore, Script Heal, D-03 Census Baseline Summary

**D-17 closed with pristine-cache proof (all four NT items shim-covered), EXEC-04 healed to real green, EXEC-05 re-verified with zero new skips, and the D-03 per-item census baseline recorded after repairing three en-route failure classes (IPython×matplotlib kernel-plot conflict, rice.uga.edu outage, combined-notebook sibling seeding)**

## Performance

- **Duration:** 7h 27min (08:29–15:57 UTC; dominated by six real-model notebook executions and a 48-minute input-host outage window)
- **Started:** 2026-10-04T08:29:09Z
- **Completed:** 2026-10-04T15:56:57Z
- **Tasks:** 3/3
- **Files modified:** 7 (2 planning docs + pyproject + 4 test files)

## Accomplishments

- Pristine NT snapshot restored AND reproducibility-proven: hand patch characterized (config legacy defaults + 3× init_weights→post_init), .dnallm-bak eliminated, snapshot deleted and re-fetched byte-identical through dnallm's own download path
- All three NT-family notebook tests green on shims alone (benchmark, NER 33-min full training, embedding_attention) — the 05-04 evidence debt cleared per item in 08-D17-DISPOSITION.md
- EXEC-04: generate_bpe_dataset.py executed past its marker to a real 11.3MB in-sandbox artifact; the marker conversion never fired and stays as the documented dead fallback
- EXEC-05: YAML 21/21 (both validate_yaml.py and test_yaml_load.py); full fast lane 1761 passed / 1 pre-existing skip — zero new skips vs baseline
- 08-CENSUS-ROLLUP.md: the D-03 baseline every later repair plan reconciles against, with the gated-family map for 08-04..08-08
- Three failure classes repaired en route with same-commit tests (REPAIR-01): ipython<9 pin, rice census-cache seeding (script + notebook lanes), combined-notebook sibling seeding (the 08-01 runner failure class, reproduced and healed)

## Task Commits

1. **Task 1: Pristine NT restore + shim-only re-execution + D-17 ledger** — `328c125` (docs) + repair `439370f` (fix)
2. **Task 2: EXEC-04 script lane heal + cache-first seeding** — `e90ff21` (test)
3. **Task 3: EXEC-05 + D-03 census baseline + lane repairs** — `c5f0916` (test)

**Plan metadata:** (this commit)

## Files Created/Modified

- `.planning/.../08-D17-DISPOSITION.md` — per-item D-17 ledger (verdicts, signatures, shims, evidence, pristine proof)
- `.planning/.../08-CENSUS-ROLLUP.md` — D-03 baseline census + family view
- `pyproject.toml` — ipython>=8.31,<9 in the notebook extra
- `tests/test_extras_guard.py` — notebook-extra bracket guards + installed kernel-plot pair probe
- `tests/examples/test_script_execution.py` — healed lane + _seed_rice_input cache-first + 3 unit tests
- `tests/examples/test_notebook_execution.py` — _rice_cache_extras() pre-seed for data_generation + 3 unit tests
- `tests/examples/test_plant_helixseek_showcase.py` — COMBINED_EXTRA_INPUTS sibling seeding + 2 contract tests

## Decisions Made

- See key-decisions (frontmatter); all four were REPAIR-01/infrastructure-driven, none architectural.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] IPython 9 × matplotlib 3.8.4 breaks every plt figure in notebook kernels**
- **Found during:** Task 1 (embedding_attention cell 9, ImportError backend2gui)
- **Issue:** 261004-dyw's pygenometracks install downgraded matplotlib 3.11.2→3.8.4 under IPython 9.17.1; matplotlib <3.9 imports the IPython-9-removed `backend2gui` unguarded in `install_repl_displayhook`
- **Fix:** `ipython>=8.31,<9` declared in the notebook extra; dev venv → 8.39.0; static + dynamic guards (RED proven pre-fix)
- **Files:** pyproject.toml, tests/test_extras_guard.py — **Committed in:** 439370f

**2. [Rule 3 - Blocking] rice.uga.edu hard-down 48 min, then half-up at ~8KB/s**
- **Found during:** Task 2 (script inputs) and Task 3 (data_generation cell timeout)
- **Issue:** documented input URLs unfetchable; the notebook's wget cell hit its 3600s timeout
- **Fix:** census-cache-first seeding for the script lane (`_seed_rice_input`) and the notebook lane (`_rice_cache_extras` pre-seed; `wget -c` skips complete files); cold cache keeps URL download + honest typed skip; 4xx re-raises; 6 fast unit tests
- **Files:** tests/examples/test_script_execution.py, tests/examples/test_notebook_execution.py — **Committed in:** e90ff21, c5f0916

**3. [Rule 1 - Bug] Combined showcase notebook FileNotFoundError on fresh sandboxes**
- **Found during:** Task 3 (D-03 lane; the 08-01 runner failure reproduced on dev box)
- **Issue:** notebook reads `../plant_helixseek_cre/data/chr1_5100001_5300000.fas`; sandbox lacked the sibling dir
- **Fix:** COMBINED_EXTRA_INPUTS seeds the sibling; TestCombinedSiblingSeeding parses the notebook JSON and pins every `../` ref covered + every source committed; healed re-run 1 passed in 197.54s
- **Files:** tests/examples/test_plant_helixseek_showcase.py — **Committed in:** c5f0916

**4. [Rule 1 - Bug] Executor-side: first full-lane pass killed by the 2h background cap mid embedding_attention**
- **Found during:** Task 3
- **Issue:** embedded run at the moment of the process-group kill → one spurious F (no artifacts)
- **Fix:** remainder re-run with explicit deselects (not a product defect); embedding green in pass B and standalone
- **Committed in:** — (execution strategy only)

**Total deviations:** 4 auto-fixed (2 bug, 1 blocking, 1 execution strategy). **Impact:** all necessary for the plan's gates; no scope creep — every fix is the documented repair pattern for its failure class.

## Issues Encountered

- NER notebook failed once mid-training without artifacts (kernel-death shape, non-reproducing) while the owner's 7 live JupyterLab kernels contended for the box; isolated re-run and the pass-B lane run both green — recorded honestly in 08-D17-DISPOSITION.md
- rice.uga.edu outage consumed ~1.5h of wall time (probe window + crawl) before the cache-seeding design landed
- Box contention (owner's interactive kernels + a concurrent CI dispatch on the shared runner) roughly doubled NER training time vs census (33 min vs 15.4)

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- D-17 fully closed; D-03 baseline on record — 08-03 (np.fromstring shim) can proceed; no transformers_compat.py edit was needed by this plan (the conditional Task 1 branch never fired), so no serialization against 08-03 is required
- Family plans 08-04..08-08 own the gated-family un-gating per the rollup's family view
- Rice-outage resilience is dev-box-local; the nightly runner keeps fresh downloads (cold cache) until 08-09 considers a runner-side cache

## Self-Check: PASSED

- Both planning docs exist on disk; all 5 modified files committed; commits 439370f/328c125/e90ff21/c5f0916 present on phs
- All three task verify commands re-run exit-0 (NT selection, script gate, census rollup gate)

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
