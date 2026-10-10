---
phase: 04-ci-gate-enforcement
plan: "02"
subsystem: infra
tags: [github-actions, coverage-gate, coverage-nightly, fail-under, workflow-dispatch, cron, actions-cache, models-lock, codecov-removal]

requires:
  - phase: 04-ci-gate-enforcement plan 01
    provides: fail_under=90 ratchet (unpushed), models.lock manifest, 7 per-test timeout marks
provides:
  - Live two-job CI gate on dev — coverage-gate (push/PR fast leg, py3.12, fail_under via pytest rc) green at 96.27%
  - coverage-nightly job (cron 03:00 UTC + workflow_dispatch) running the full slow census under the same fail_under, with models.lock-keyed whole-hub-dir caching
  - Event guards on test/test-cuda/test-mamba/deploy — schedule/dispatch runs execute ONLY the nightly job (proven live both directions)
  - GATE-03 removal: codecov-action@v3 uploader + orphaned coverage.xml export gone; permissions stay contents: read
  - Calibration dispatch run in flight (healthy past install into census) with owner runtime-decision framing
affects: [04-03-PLAN.md, phase-04 close, owner runtime decision]

actuals:
  tokens: 1616   # 6,466 diff chars / 4 over the realized ci.yml diff (plan estimate 40,000 — infra plans overestimate on token scale; the real cost is wall-clock CI observation)
  tasks: 3
  commits: 1     # measured: git rev-list --count 2231530..HEAD (production commit; Tasks 2-3 are evidence-only, no file changes)

plan_head_before: 223153003500e47ff7fe242be8fcfe079057f467
plan_head_after: a4720efd2dfd2e61468d53e6f895b6282b4216cf

tech-stack:
  added: []       # zero new packages/actions — jobs reuse actions/checkout@v4, setup-python@v7, cache@v4 already in the workflow
  patterns:
    - "Event-name guards (push||pull_request vs schedule||workflow_dispatch) as the single isolation barrier between a shared workflow's PR loop and its nightly lane"
    - "Whole-hub-dir model caching (HF + ModelScope) keyed on hashFiles('models.lock') — content-addressed rotation, save-only-on-success"
    - "Two-lane coverage enforcement: PR-side fast leg for feedback speed, nightly slow census for the full denominator, one fail_under for both"

key-files:
  created: []
  modified:
    - .github/workflows/ci.yml

key-decisions:
  - "Two-job split landed exactly per the locked owner amendment: coverage-gate runs -m 'not slow' on every push/PR; the slow suite moved to coverage-nightly (schedule+dispatch only) under the identical fail_under"
  - "GATE-03 resolved by REMOVAL (owner disposition): the v3 uploader step and its dead coverage.xml export line are deleted with no replacement — the native fail_under gate is the enforcement; coverage trends remain v2 PATCH-02 territory"
  - "deploy.needs left unchanged (owner disposition) and deploy's if tightened to push+main/master — identical outcome on real triggers while making nightly/manual runs structurally unable to deploy docs"
  - "OWNER NOTE: GitHub runs scheduled workflows only from the default branch — the nightly CRON goes live when this ci.yml reaches main; manual dispatch on dev already works (the dispatch API accepts it because the CI workflow exists on main; the run uses the dev ref's file)"

patterns-established:
  - "Live-log evidence is unavailable mid-run on GitHub's API (gh run view --job --log returns 404 until completion) — step-state machines (step status/conclusion) are the runtime health proof for long jobs; harvest logs at completion"
  - "Watch-run verify commands must disambiguate --workflow when a repo has multiple push-triggered workflows, or a later-finishing sibling workflow steals the 'latest push run' slot"

requirements-completed: [GATE-02, GATE-03]
# GATE-05 deliberately NOT claimed: shared with 04-03 (no SUMMARY yet) — the PR-side dev proof is 04-03's probe PR; requirements.ready-ids correctly blocked it

coverage:
  - id: D1
    description: "ci.yml structurally complete: two gated jobs with exact timeouts (90/480), event guards, py3.12 legs, free-disk steps, verbatim census commands, own junit + skip audits, models.lock-keyed dual-hub cache"
    requirement: GATE-02
    verification:
      - kind: other
        ref: "python3 yaml assertion block (triggers, guards, timeouts, verbatim gate/nightly census lines, audit steps, cache key/paths, canary exactly once) — CI-STRUCTURE-OK"
        status: pass
    human_judgment: false
  - id: D2
    description: "GATE-03 removal clean: zero codecov references and zero 'coverage xml -o' lines anywhere in ci.yml; permissions block unchanged at contents: read"
    requirement: GATE-03
    verification:
      - kind: other
        ref: "! grep -qi codecov .github/workflows/ci.yml (empty) + grep -c 'coverage xml -o' == 0 + permissions grep — PASS"
        status: pass
    human_judgment: false
  - id: D3
    description: "Gate live and green on dev: push run 36745734429 with coverage-gate success at 96.27% total (1635 passed / 1 allowlisted skip / 27 deselected), all six matrix legs success at 96.27%, nightly and deploy skipped (guards proven at runtime)"
    requirement: GATE-02
    verification:
      - kind: other
        ref: "gh run view 36745734429 --json jobs: zero conclusions outside success/skipped; coverage-gate=success; nightly=skipped; deploy=skipped — GATE-LIVE-GREEN"
        status: pass
    human_judgment: false
  - id: D4
    description: "Nightly calibration dispatch healthy: run 36747594207 with exactly one non-skipped job (coverage-nightly), install steps success, census step in progress at observation time; run left running, not cancelled"
    requirement: GATE-02
    verification:
      - kind: other
        ref: "gh run view 36747594207: non_skipped==1; install step completed/success; 'Run gated full census (slow included)' in_progress (sustained 4+ min, no termination)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Owner runtime decision record: install-to-census timings, cold-cache seeding note, and the accept / larger-runner / LANE-01-pull-forward menu the nightly's final wall time feeds"
    verification: []
    human_judgment: true
    rationale: "The decision itself is the owner's (hours-long nightly acceptance vs paid larger runner vs pulling LANE-01 forward from v2); executor records the framing, the owner decides at phase close with the completed run's numbers"

duration: 27 min
completed: 2026-09-30
status: complete
---

# Phase 4 Plan 2: CI Gate Live — Two-Job Workflow, Event Isolation, Uploader Removal Summary

**The amended two-job CI shape is live on dev: coverage-gate green at 96.27% on push run 36745734429 with all six matrix legs green under the same fail_under ratchet, nightly+deploy guards proven at runtime, codecov uploader gone (GATE-03), and calibration dispatch run 36747594207 healthy into its slow census with the models.lock cache seeding**

## Performance

- **Duration:** 27 min (started 2026-09-30T16:35:27Z, completed 2026-09-30T17:03Z) — CI run observation dominates (push run 16:38-16:52Z; dispatch monitored to 17:00Z)
- **Started:** 2026-09-30T16:35:27Z
- **Completed:** 2026-09-30T17:03:00Z
- **Tasks:** 3/3
- **Files modified:** 1 (.github/workflows/ci.yml)

## Accomplishments

- **The gate is live and green (GATE-01/02 live moment):** push run [36745734429](https://github.com/zhangtaolab/DNALLM/actions/runs/36745734429) (created 16:38:56Z, completed 16:52:02Z, ~13 min) shows `coverage-gate (py3.12, fast leg)` at **success** with the observed total **96.27%** — `Required test coverage of 90.0% reached. Total coverage: 96.27%`, 1635 passed / 1 allowlisted skip / 27 deselected (slow), census step 137.18s. All six matrix legs green at the same 96.27% — `fail_under` rode their existing `--cov` with zero special-casing (within 0.03 points of the local 96.29-96.30% measurement: no environment drift, Pitfall 10 did not bite).
- **Event isolation proven live, both directions:** on the push — `coverage-nightly` concluded **skipped** (guard working) and `deploy` **skipped** (dev ref); on the dispatch — every job except `coverage-nightly` concluded **skipped** (test, test-cuda, test-mamba, coverage-gate, deploy), i.e. exactly one non-skipped job.
- **Nightly lane calibrated in flight (GATE-02):** manual dispatch `gh workflow run CI --ref dev` at 2026-09-30T16:54:12Z created run [36747594207](https://github.com/zhangtaolab/DNALLM/actions/runs/36747594207), nightly job id **109997628011**. Install-to-census was ~2m20s (see timings below), the census step entered `in_progress` at 16:56:32Z and was still healthily running at 17:00:34Z observation. Left running per plan — projected 4-7.5h; outcome handed to phase close.
- **GATE-03 owner disposition executed (removal):** the `codecov/codecov-action@v3` step and the `coverage xml -o coverage.xml` export line (its only consumer) are deleted; no replacement reporting step; workflow permissions stay `contents: read`. The last unpinned third-party action in the test job is gone.
- **Structural verification idempotent:** the full YAML assertion block re-run green from the working tree identical to the committed file (`CI-STRUCTURE-OK` + both negative greps).

## Task Commits

Each task was committed atomically:

1. **Task 1: ci.yml rework — two gated jobs, event guards, uploader removal** - `a4720ef` (feat)
2. **Task 2: Push to dev — gate goes live green across all legs** - no commit (evidence-only task: Task 1's commit pushed; run 36745734429 observed green)
3. **Task 3: Nightly calibration dispatch — healthy start, run recorded** - no commit (evidence-only task: dispatch run 36747594207 observed healthy; recorded here)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified

- `.github/workflows/ci.yml` - the five coordinated edits: workflow_dispatch + nightly cron triggers; new `coverage-gate` job (fast leg, timeout 90) and `coverage-nightly` job (full census, timeout 480, continue-on-error false, dual-hub models.lock-keyed cache); push/PR guards on test/test-cuda/test-mamba; deploy tightened to push+main/master with `needs` unchanged; codecov uploader + XML export removed

## Decisions Made

- **Two-job split exactly per the locked owner amendment** — no deviation from the amendment shape; fast leg gates PRs, slow suite moved to the nightly lane under the same `fail_under`.
- **`requirements-completed` claims only GATE-02/GATE-03** (not the plan frontmatter's full `[GATE-02, GATE-03, GATE-05]`): GATE-05 is shared with 04-03 and its live PR-side proof (the probe PR to dev) is 04-03's deliverable — `requirements.ready-ids` enforced the same shared-ID gate.
- **Cron activation note surfaced as an owner decision input** (see Deviations/Notes): the schedule trigger only fires from the default branch, so the nightly cron arms when dev reaches main; manual dispatch on dev already works and is the calibration path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 2 verify command resolved the wrong run once Docs Validation finished**
- **Found during:** Task 2 (verify step)
- **Issue:** The plan's verbatim `gh run list --branch dev --event push --limit 1` has no workflow filter; the repo's Docs Validation workflow is also push-triggered and finished after CI, becoming the "latest push run" — the verify initially inspected a 1-job docs run and produced empty output (asserting nothing).
- **Fix:** Re-ran the identical assertion chain with `--workflow CI` disambiguation → `GATE-LIVE-GREEN run=36745734429`, all facts asserted as planned.
- **Files modified:** none (verification-command composition only)
- **Verification:** `GATE-LIVE-GREEN run=36745734429`; job listing: 10 success + 2 skipped, zero failures
- **Committed in:** n/a (evidence task; pattern recorded in WINDOWS ledger)

**2. [Rule 1 - Bug] Task 3's mid-run "pytest header via gh run view --job --log" is mechanically impossible**
- **Found during:** Task 3 (health confirmation)
- **Issue:** GitHub's API returns `job ... is still in progress; logs will be available when it is complete` (HTTP 404 on the raw logs endpoint) for in-progress jobs — the plan assumed live log access that the platform does not provide.
- **Fix:** Health established the way the plan's own formal `<verify>` blocks encode it: install step `completed/success`, census step present in the step list and `in_progress`, sustained 4+ minutes with no job termination (the Pitfall-2 OOM/exit-137 warning sign did not occur). The pytest header is harvestable from the completed run's log at phase close.
- **Files modified:** none
- **Verification:** Task 3 verify block 2 verbatim: install=success, census step in step list; job status in_progress at 17:00:34Z
- **Committed in:** n/a (evidence task; pattern recorded in WINDOWS ledger)

---

**Total deviations:** 2 auto-fixed (2x Rule 1 — both defects in plan-verify mechanics, not in the shipped workflow)
**Impact on plan:** None on deliverables — all plan success criteria met; both fixes are verify-command corrections fully within the plan's intent.

## Issues Encountered

None beyond the two verify-mechanics deviations above. Notable non-issue: `gh workflow run CI --ref dev` succeeded despite the `workflow_dispatch` trigger not yet being on the default branch — the dispatch API requires the workflow to exist on the default branch (it does) and runs the ref's file version. The related hard fact IS recorded for the owner: **scheduled (cron) workflows run only from the default branch**, so the nightly cron arms when this ci.yml reaches main.

## Calibration Data for Phase Close (consume without re-derivation)

**Push run (gate live proof):** run 36745734429, a4720ef, created 16:38:56Z / completed 16:52:02Z (~13 min wall). coverage-gate: **96.27%**, 1635 passed / 1 allowlisted skip / 27 deselected, census 137.18s. Matrix legs (6): all success, 96.27%. test-cuda x2, test-mamba: success. nightly + deploy: skipped.

**Nightly dispatch run (calibration in flight):** run 36747594207, job 109997628011, dispatched 2026-09-30T16:54:12Z (created 16:54:19Z). Step timings:

| Step | Window | Duration |
|---|---|---|
| Free disk space | 16:54:26 → 16:55:17 | 51s |
| Set up Python 3.12 | 16:55:17 | <1s (toolchain cached) |
| Install uv | 16:55:17 → 16:55:18 | 1s |
| Cache uv dependencies | 16:55:18 → 16:56:27 | 69s (**warm** — restored from the green push run's save) |
| Restore model caches | 16:56:27 | **0s — COLD MISS** (first nightly ever; seeds only if the job succeeds) |
| venv + install | 16:56:27 → 16:56:32 | 5s (uv warm) |
| numpy==2.2.0 | 16:56:32 | <1s |
| Run gated full census | started 16:56:32Z | in_progress, healthy at 17:00:34Z observation |

Cold-cache note: this run downloads ~5G of models inside the census step; a red first run poisons nothing (actions/cache saves only on success). Dispatch-to-census was 2m20s, far under the plan's 15-25 min estimate, because the push run had just seeded the uv cache — a genuinely cold runner will be slower on install but identical on census.

**Owner runtime-decision menu (feeds phase close):** (a) accept a 4-7.5h nightly on standard 4-core/16GB runners (480-min backstop in place; per-test timeout marks 7200/3600 from Wave 1); (b) larger runner (8-core, paid) to cut wall time; (c) pull LANE-01 (two-lane split + nightly drift detection) forward from v2. The completed run's wall time and `--durations=0` output decide. Separate documented follow-ups: make `coverage-gate` a required check once branches are protected (none are today); the cron arms on main when dev merges.

## Known Stubs

None — no stubs, placeholders, or unwired paths were introduced; both new jobs execute the census command of record.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 04-03 has everything it needs: the gated PR lane is live on dev (its probe PR to dev will trigger `coverage-gate` — the GATE-04/GATE-05 live proofs), and the 04-01 drop-probe correction (use `--ignore=tests/models` directory-level, predicted 78.92% red) tells it how to force the red check.
- The nightly calibration run 36747594207 should be allowed to finish; its conclusion (or its failure log — watch for exit 137 / "job has been terminated") is the phase-close evidence for the owner's runtime decision.
- Local dev note already flagged in 04-01: every `--cov` invocation is now gated — scoped runs drop `--cov` or pass `--no-cov`.

## Self-Check: PASSED

- Files exist: .github/workflows/ci.yml modified with coverage-gate + coverage-nightly jobs (committed a4720ef) — FOUND
- Commits exist on dev: a4720ef — FOUND (pushed: origin/dev at a4720ef)
- Plan-level verification re-run green: CI-STRUCTURE-OK / no-codecov / no-xml-export (idempotent)
- Live evidence: push run 36745734429 (gate green, 96.27%) and dispatch run 36747594207 (isolation + healthy census) both gh-observable

---
*Phase: 04-ci-gate-enforcement*
*Completed: 2026-09-30*
