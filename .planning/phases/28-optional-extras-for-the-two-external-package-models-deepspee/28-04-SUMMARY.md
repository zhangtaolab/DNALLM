---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
plan: "04"
subsystem: ci
tags: [extras-compat-lane, workflow_call, self-hosted-runner, deepspeed, flash-attn, pinned-installer, census-drift-gate, ruling-a, length-64-probe]

requires:
  - phase: 28-optional-extras-for-the-two-external-package-models-deepspee
    provides: curated extras + sha256-pinned installer (28-01), CrossDNA extra-required-tf5 flip + ruling-A distribution (28-02), base-provided fla docs story (28-03/28-05)
provides:
  - ".github/workflows/extras_compat.yml — the dedicated extras lane (D-05): 04:00 UTC cron + standalone dispatch + internal workflow_call, contents:read, [self-hosted, dnallm-nightly], 360 min, two independent pass gates"
  - "ci.yml run_extras_compat fourth dispatch selector + dispatch-gated reusable extras-compat call, no caller schedule arm (D-06)"
  - "TestExtrasLaneContract (15 tests) + TestExtrasDispatchSelection (5 tests) — offline security/install-order/two-gate/dispatch contracts (D-05..D-08/D-12)"
  - "Workflows README: four-selector docs, extras-lane section, both full,fla occurrences retired, deferred post-merge first-live-dispatch operator step"
affects: [post-v1.4-to-main merge operator runbook, future census/extras lane changes, optional_dependencies guide cross-links]

actuals:
  tokens: 14047
  tasks: 2
  commits: 4

tech-stack:
  added: []
  patterns:
    - diff-proven one-line tool copy generated at lane runtime (occurrence-asserted sed swap) so the committed census tool keeps its default-lane constant while the extras lane probes at the length the model demands
    - seed-strip forcing: null the seeded load evidence of the probed entries so apply_loads re-measures every run instead of carrying last cycle's green
    - dispatch-gated workflow_call as the manual route to a cron-owning lane (exactly-once nightly + exactly-once dispatch)

key-files:
  created: [".github/workflows/extras_compat.yml"]
  modified: [".github/workflows/ci.yml", ".github/workflows/README.md", "tests/test_runner_infra_contracts.py"]

key-decisions:
  - "Owner ruling A re-scope (binding prior-wave fact): the plan's pre-ruling gate-1 wording ('both fresh records load+forward green') is impossible for BigBird — gate 1 asserts the ruling-A split instead (CrossDNA fully green; BigBird load green + forward red AT the known superblock cause), and gate 2 pins the committed 19/21/5/1-extra/3-broken/0 distribution through --verify"
  - "D-07 probe route: the lane generates the length-64 census copy at runtime (grep occurrence assertion + sed swap + diff-proven one changed line); the committed census_compat.py stays at LOAD_SEQUENCE_LENGTH=32 and is never invoked with --run by this lane (pinned offline)"
  - "Seed-strip discipline: the two entries' seeded load evidence is nulled before the probe because apply_loads carries a prior passed load instead of re-fetching — without the strip the lane would assert the committed/Plan-01 green, never tonight's"
  - "Lane pins uv venv to --python 3.13 (run 38678158479 lesson + 28-01 measured cp313 extras env; bare uv venv now picks a 3.16 alpha on this box) and exports MAX_JOBS=8 for the ~64-min flash-attn build"
  - "uv posture is stricter than the census lane: fail-closed with NO curl installer fallback (a wiped box is an operator incident, not something this lane repairs ahead of a 64-minute build)"
  - "README job sections renumbered (extras lane becomes 8; feas-spike 9, deploy 10) rather than appending out of order"

patterns-established:
  - "Runtime-generated diff-proven tool copy: keep a shared committed tool byte-stable for its original lane while a second lane needs one changed constant"
  - "Exactly-once nightly via cron ownership: the callee owns the schedule; callers reach it only through dispatch-gated reusable calls with no schedule arm"

requirements-completed: [D-02, D-05, D-06, D-07, D-08, D-12]

coverage:
  - id: D1
    description: "Dedicated extras lane workflow with secure self-hosted posture (D-05, T-28-09): schedule/dispatch/call-only triggers, contents:read, no secrets, cron-string event gate, GPU-absent fail-closed marker"
    requirement: D-05
    verification:
      - kind: unit
        ref: "tests/test_runner_infra_contracts.py::TestExtrasLaneContract (15 tests: triggers, permissions/runner/timeout, event gate, uv fail-closed, no masking, gpu marker, always-upload)"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-12 install order: .[full] -> pip bootstrap -> .[deepspeed] -> sha256-pinned install_flash_attn.sh only; no bare package-index flash-attn install anywhere in the lane"
    requirement: D-12
    verification:
      - kind: unit
        ref: "TestExtrasLaneContract::test_install_order_full_pip_deepspeed_installer_probe + test_flash_attn_installs_only_through_the_pinned_script (forbidden-shape tuple)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two independent pass gates in green-then-verify order (D-07/D-08, ruling-A shape): gate 1 asserts raw load booleans per model (CrossDNA green; BigBird load-green/forward-structurally-red at superblock; never dispositions), gate 2 runs the committed census tool --verify"
    requirement: D-07
    verification:
      - kind: unit
        ref: "TestExtrasLaneContract::test_two_distinct_gates_green_then_verify + test_gate1_reads_raw_load_evidence_not_dispositions + test_gate1_asserts_the_ruling_a_expected_shape + test_probe_runs_bigbird_at_a_64_multiple + test_fresh_probes_are_forced_not_carried"
        status: pass
      - kind: integration
        ref: "Runner-replica execution of the exact lane sequence in .scratch/p28-lane (extras venv: torch 2.14.1, transformers 5.17.0, deepspeed 0.19.6, flash-attn 2.8.3.post2): probe re-measured both entries at length 64 (CrossDNA ok=true forward_ok=true shape (1,64,5) cuda 3.1s; BigBird ok=true forward_ok=false cuda 7.9s with superblock in the forward error, no divisibility text); GATE 1 exit 0; gate 2 --verify '49 entry/entries compared' + 'OK: census matches committed registry fields' (CrossDNA matched extra-required-tf5, BigBird matched remote-code-broken); re-run green end-to-end as the tracer feedback gate"
        status: pass
    human_judgment: false
  - id: D4
    description: "Fourth ci.yml dispatch selector + reusable call that cannot duplicate the nightly run (D-06, T-28-09): run_extras_compat boolean default true, inputs-context gate, no secrets, no caller schedule arm, extras workflow sole owner of 0 4 * * *"
    requirement: D-06
    verification:
      - kind: unit
        ref: "tests/test_runner_infra_contracts.py::TestExtrasDispatchSelection (5 tests) + parsed-YAML assertions from the plan's verify (exit 0)"
        status: pass
      - kind: automated
        ref: "git diff --exit-code -- .github/workflows/census_compat.yml (D-02: default census byte-untouched) + uvx ruff@0.17.1 check . clean + format --check on the touched test file"
        status: pass
    human_judgment: false
  - id: D5
    description: "Operator documentation: four selectors, run_extras_compat=false in the single-lane example, extras-lane section (both gates, extras-fresh-evidence, deferred first live dispatch), both full,fla occurrences replaced with base-provided fla wording"
    requirement: D-06
    verification:
      - kind: automated
        ref: "grep receipts: full,fla count 0 in workflows/README.md; run_extras_compat x3; extras-fresh-evidence x1; post-v1.4-to-main-merge operator step recorded; section headings renumbered 1-10"
        status: pass
    human_judgment: false
  - id: D6
    description: "Live registration + first dispatch of extras_compat.yml — deliberately deferred (GitHub registers dispatchable workflows from the default branch only; main frozen during v1.4)"
    verification: []
    human_judgment: true
    rationale: "Not executable on branch v1.4: dispatchable workflows register from the default branch. The structural lane, its offline contracts, and the runner-replica execution of the exact two-gate probe sequence are the interim evidence; the operator step 'gh workflow run extras_compat.yml --ref v1.4' (post-v1.4-to-main merge) is recorded in the workflows README and STATE operator next steps."

duration: 18min
completed: 2026-10-30
status: complete
commits: 4
plan_head_before: 269ef05eff2ed81853337ef801186175f46fb8f7
plan_head_after: 60d5682
---

# Phase 28 Plan 04: Optional extras CI lane + fourth dispatch selector Summary

**The dedicated extras_compat.yml lane landed with exactly-once nightly ownership (its own 04:00 UTC cron), a dispatch-gated reusable call behind ci.yml's fourth boolean, the pinned-installer-only flash-attn route, a length-64 BigBird probe, and two independent gates — all 20 offline contracts green and the exact two-gate probe sequence proven live on the runner replica (CrossDNA fully green; BigBird load-green/forward-structurally-red at superblock; 49-matched registry verify), with census_compat.yml byte-untouched.**

## Performance

- **Duration:** ~18 min (2026-10-29T16:43Z → 2026-10-30T01:01Z box-local; replica probes included)
- **Tasks:** 2 (both TDD: RED + GREEN commits each)
- **Files modified:** 4 (1 new workflow, 2 workflow/docs files, 1 test file)
- **Estimate vs actual:** 50,000 tokens estimated / 14,047 actual (chars/4 over the realized diff) — the ruling-A re-scope shrank gate 1's surface and no code-package files were touched

## Accomplishments

- D-02/D-05..D-08/D-12 closed in one plan: the default census stays the default-install measurer (byte-untouched, diff-verified), while the new lane answers the mirror question in an extras-equipped environment.
- The two-gate design is tamper-honest by construction: gate 1 reads only raw `load.ok`/`forward_ok` booleans and the forward error text (never `compat_verdict`, never the seeded judgment — pinned offline), gate 2 is the committed tool's fail-closed `--verify` over the full 49-record artifact.
- The D-07 length fact is now lane-resident machinery, not scratch knowledge: the workflow generates the length-64 census copy at runtime with an occurrence assertion and a one-line diff proof, and an offline contract forbids the committed tool's `--run` in this lane entirely.
- Fresh-evidence forcing: the seed-prep nulls the two entries' load evidence so `apply_loads` re-measures every run (a prior passed load is carried, never re-fetched — the trap that would have made the lane assert Plan-01's green forever).
- Tracer discipline held: RED_EVIDENCE_OK on both tasks (records at `.scratch/tdd-red-28-04-task1.json` / `-task2.json`), and the tracer's automated-only verify re-ran green end-to-end before expansion into Task 2.

## Task Commits

1. **Task 1 RED: failing TestExtrasLaneContract (15 tests)** — `e9fcef1` (test)
2. **Task 1 GREEN: dedicated extras-compat lane + both gates** — `67bff09` (feat)
3. **Task 2 RED: failing TestExtrasDispatchSelection (5 tests)** — `c4b97b0` (test)
4. **Task 2 GREEN: fourth selector + reusable call + README docs** — `60d5682` (feat)

TDD gate compliance: every `test(28-04)` commit precedes its `feat(28-04)` commit; both RED runs validated RED_EVIDENCE_OK before GREEN. Tracer feedback gate: verify re-run end-to-end green (29 tests + replica probe chain) under `HUMAN_VERIFY_MODE=end-of-phase`, no checkpoint.

## Files Created/Modified

- `.github/workflows/extras_compat.yml` (new) — the extras lane: triggers, permissions, event gate, fail-closed GPU/uv steps, D-12 install order, seed-strip + length-64 probe, gate 1 (ruling-A), gate 2 (--verify), always-upload evidence
- `.github/workflows/ci.yml` — `run_extras_compat` boolean (default true) after `run_test_mamba`; `extras-compat` reusable-call job (`if: github.event_name == 'workflow_dispatch' && inputs.run_extras_compat`, contents:read, no secrets, no schedule arm); dispatch-selection comment updated to four inputs
- `.github/workflows/README.md` — four-selector trigger discussion, `run_extras_compat=false` in the single-lane example, new "Extras Compat Job" section, both `full,fla` occurrences retired (base-provided fla wording), job sections renumbered
- `tests/test_runner_infra_contracts.py` — `TestExtrasLaneContract` (15) + `TestExtrasDispatchSelection` (5) with lane-specific loaders

## Replica Evidence (runner-box, extras venv: torch 2.14.1 / transformers 5.17.0 / deepspeed 0.19.6 / flash-attn 2.8.3.post2)

| Step | Result |
|---|---|
| Seed + strip | `cp` committed 49-row census → extras-fresh.json; 2 seeded loads nulled |
| Length-64 copy | occurrence count = 1; diff-proven exactly 1 changed line |
| Probe (BigBird) | `ok=True forward_ok=False` cuda 7.9s — error contains `superblock`, no divisibility text |
| Probe (CrossDNA) | `ok=True forward_ok=True` shape `(1, 64, 5)` cuda 3.1s |
| Gate 1 | exit 0 — ruling-A per-model assertions |
| Gate 2 | `Registry verification: 49 entry/entries compared` … `OK: census matches committed registry fields`; `matched AIRI-Institute/gena-lm-bigbird-base-sparse remote-code-broken`; `matched chengCCC/CrossDNA_pretrain extra-required-tf5` |
| Tracer re-run | whole chain re-executed green from a clean `.scratch/p28-lane` |

## Decisions Made

Recorded in key-decisions frontmatter. The load-bearing one: **owner ruling A governs gate 1** — the plan's pre-ruling text ("require both load.ok and forward_ok to be true", "both fresh records are load+forward green", "--verify matches the committed extra-required registry rows" plural) describes a world the measurement refuted; the binding ruling + prior-wave facts re-scope it to the split shape (CrossDNA green; BigBird load-green/forward-structurally-red at the named cause) and to the committed one-extra-row distribution. Every re-scope is pinned offline so the lane cannot silently regress to the pre-ruling wording.

## Deviations from Plan

### Ruling-Driven Re-scopes (owner directives, not auto-fixes)

**1. [Owner Ruling A + prior-wave fact D-07] Gate 1 and the replica verify assert the ruling-A split, not "both records green"**
- **Found during:** Task 1 design (precondition + plan action text)
- **Issue:** the plan (authored pre-ruling) requires `load.ok and forward_ok true` for BOTH records and probes via the committed tool — BigBird's forward is measured structurally impossible (deepspeed 0.19.6 sparse LUT → `triton libtriton.superblock`, absent in triton 3.6/3.8), and the committed tool's length-32 probe cannot even reach that verdict (divisibility red)
- **Fix:** gate 1 asserts per-model expected outcomes (CrossDNA fully green; BigBird `ok=True`, `forward_ok=False`, `superblock` in the forward error); the probe runs through a runtime-generated, diff-proven length-64 copy of the committed tool (occurrence-asserted sed swap; committed tool pinned never `--run` in this lane); gate 2 unchanged (committed tool `--verify`, 49 matched)
- **Files modified:** .github/workflows/extras_compat.yml, tests/test_runner_infra_contracts.py
- **Verification:** 20 offline contracts green; replica chain green end-to-end (twice)
- **Committed in:** e9fcef1, 67bff09

**2. [Owner Ruling A] "Committed extra-required registry rows" is one row, not two**
- **Found during:** Task 1 gate 2 design
- **Issue:** plan wording implies both probed rows are `extra-required-tf5`; the committed ruling-A registry keeps BigBird at `remote-code-broken` with split evidence
- **Fix:** gate 2 is the unchanged fail-closed `--verify` over the full artifact — the replica run shows exactly the committed distribution (CrossDNA `extra-required-tf5`, BigBird `remote-code-broken`), which IS the "matches the committed registry" requirement
- **Verification:** `49 entry/entries compared — OK`, both probe rows matched
- **Committed in:** 67bff09

### Auto-fixed Issues

**3. [Rule 3 - Blocking] YAML step names containing ": " must be quoted**
- **Found during:** Task 1 GREEN (first pytest run)
- **Issue:** unquoted `- name: Gate 1: …` broke YAML parsing (ScannerError), failing all lane-contract tests
- **Fix:** quoted both gate step names (ci.yml's own `"Stage 0.5: …"` precedent)
- **Files modified:** .github/workflows/extras_compat.yml
- **Verification:** 29 → 34 tests green across the two tasks
- **Committed in:** 67bff09

**4. [Rule 1 - Style] Pinned ruff 0.17.1 flagged 5 PT018 composite assertions + formatting in the new tests**
- **Found during:** Task 2 verify-3
- **Issue:** `and`-joined asserts and two formatting shapes in the new test code (same class 28-02 hit)
- **Fix:** split into single-purpose asserts; `ruff format` applied to the touched file
- **Files modified:** tests/test_runner_infra_contracts.py
- **Verification:** repo-wide `ruff@0.17.1 check` clean; `format --check` clean; 34 tests green
- **Committed in:** 60d5682

---

**Total deviations:** 4 (2 ruling-driven re-scopes applied as directed, 1 Rule 3, 1 Rule 1)
**Impact on plan:** all four must-have truths hold in their ruling-A form; no gate disagrees with the committed registry.

## Issues Encountered

- First RED-evidence record for Task 1 was rejected (`INVALID_RED: invalid_record` — the classifier needs a parseable report in `record.output`, not a summary `report` object; and the junit adapter expects the XML text itself). Re-recorded with the junit XML as `output` and the class name as `targetTest` (the adapter's supported class-target shape) → `RED_EVIDENCE_OK` on both tasks.
- The Task-2 RED class had one pre-green test (`test_caller_keeps_no_extras_schedule_arm`): its invariants (two committed crons in ci.yml; the 04:00 cron only in extras_compat.yml) became satisfiable the moment Task 1's workflow landed. The other four failed on the planned missing-input/missing-job assertions; recorded and validated as RED.

## Authentication Gates

None.

## User Setup Required

None — no external service configuration required. (The deferred live dispatch is an operator runbook item, not user setup.)

## Known Stubs

None — every shipped surface is wired to measured evidence.

## Deferred Post-Merge Operator Step

The first live dispatch — `gh workflow run extras_compat.yml --ref v1.4` (after the v1.4→main merge registers it) — is recorded in `.github/workflows/README.md` §Extras Compat Job and in STATE operator next steps, beside the census-lane dispatch already owed. Expected receipts: green gates 1–2, `extras-fresh-evidence` artifact, 49-matched verify line.

## Next Phase Readiness

- Phase 28 is now fully executed (all 5 plans summarized). D-02 closes with this plan (the shared-ID gate held it for 28-04).
- The lane is structurally ready for registration; nothing in `dnallm/` changed (owner pytest rule not triggered — workflow/docs/test-file changes only, matching the plan's file scope).

## Self-Check: PASSED

- All 4 task commits (e9fcef1, 67bff09, c4b97b0, 60d5682) are ancestors of HEAD 60d5682.
- `.github/workflows/extras_compat.yml` exists; both RED evidence records exist on disk; `.scratch/p28-lane/extras-fresh.json` holds the replica evidence.
- Final battery green on the committed tree: 34 runner-infra tests, YAML assertions, `census_compat.yml` diff-free, repo-wide ruff 0.17.1 clean, format-clean touched file; replica two-gate chain green twice.

---
*Phase: 28-optional-extras-for-the-two-external-package-models-deepspee*
*Completed: 2026-10-30*
