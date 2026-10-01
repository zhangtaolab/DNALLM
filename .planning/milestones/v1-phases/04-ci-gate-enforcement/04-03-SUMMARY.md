---
phase: 04-ci-gate-enforcement
plan: "03"
subsystem: infra
tags: [coverage-gate, gate-regression-probe, fail-under, github-actions, wr-05-readme, branch-protection-handoff, gate-04]

requires:
  - phase: 04-ci-gate-enforcement plan 02
    provides: live two-job CI gate on dev (coverage-gate green on push run 36745734429) + nightly calibration run 36747594207
provides:
  - GATE-04 proof of record — a real coverage-dropping PR produced a red coverage-gate check with the verbatim fail-under line in the CI job log (Phase-1 exit-code fix exercised end to end)
  - GATE-05 live dev-side proof — PR #39 (base dev) provably triggered the gated coverage job; PRs to main inherit the same pull_request block (structural, asserted in 04-02)
  - Zero-residue guarantee — probe PR closed unmerged, branch deleted locally and on origin, dev history clean, no models-cache write from the red run
  - WR-05 minimal-touch workflows README — both new jobs documented by name, flipped slow-test claims corrected, enforced 90 floor + --no-cov scoped-run guidance
  - Owner hand-off record — exact branch-protection commands for dev and main, nightly runtime decision menu, WR-02/WR-06 leave-untouched rationale
affects: [phase-04 close, milestone verification, owner decisions (required checks, nightly runtime)]

actuals:
  tokens: 1457   # 5,831 diff chars / 4 over the realized README diff (plan estimate 35,000 — the real cost was wall-clock CI observation + census, not tokens)
  tasks: 2
  commits: 1     # measured: git rev-list --count 2d02fff..HEAD (README production commit; Task 1 is evidence-only on dev by design)

plan_head_before: 2d02fff9cb8a250d6710f714db9304cbdd3bbfe3
plan_head_after: 29ed1644185bd9fcd104058ad21215c07560e045

tech-stack:
  added: []
  patterns:
    - "Ephemeral probe PR pattern: rehearse the drop locally (identical command family) before pushing; declare never-merge in the PR body; close unmerged + delete branch; residue asserted via ls-remote and PR state"
    - "Job-level log harvest: gh api repos/.../actions/jobs/<id>/logs works for a completed job even while its run is still in progress (gh run view --job --log-failed gates on whole-run completion)"

key-files:
  created:
    - .planning/phases/04-ci-gate-enforcement/04-03-SUMMARY.md
  modified:
    - .github/workflows/README.md

key-decisions:
  - "Probe target re-planned to the whole tests/models directory (7 files, 4,936 lines) per the plan's own rehearsal gate: the as-written single-file rehearsal measured GREEN at 91.64% on today's tree (04-01 calibration confirmed), so pushing it would have proven nothing; directory deletion rehearsed RED at 78.92% and CI reproduced 78.91%"
  - "Evidence harvested via the job-level logs API instead of run-level --log-failed (the run's test-cuda 12.4 leg was still running; gh gates run-level logs on run completion)"
  - "GATE-05 claimed here together with 04-02's structural proof: the probe PR is the live dev-side trigger evidence; requirements.ready-ids unblocked the shared ID once this SUMMARY existed"

patterns-established:
  - "A gate that has never been observed failing is a hope, not a control: the red-proof belongs in the phase that enables the gate, executed as a local-first rehearsal + ephemeral PR + verbatim log capture + zero-residue assertion"

requirements-completed: [GATE-04, GATE-05]

coverage:
  - id: D1
    description: "GATE-04: the coverage-gate check on a real coverage-dropping PR (base dev) concluded FAILURE with the verbatim fail-under line in the CI job log"
    requirement: GATE-04
    verification:
      - kind: other
        ref: "gh pr view 39 --json statusCheckRollup -q '[.statusCheckRollup[] | select(.name|startswith(\"coverage-gate\"))][0] | .conclusion' → FAILURE; job 110005226448 log: 'ERROR: Coverage failure: total of 79 is less than fail-under=90' + 'FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%' (TOTAL 7405/1562, 1261 passed — all tests green, only the floor red)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Zero residue: PR closed unmerged (state CLOSED / mergedAt null), probe branch absent from origin and locally, repo on dev at origin/dev, tests/models restored, dev history clean"
    requirement: GATE-04
    verification:
      - kind: other
        ref: "PROBE-CLEAN-EVIDENCED pr=39 (gh pr view state=CLOSED/false; git ls-remote --heads origin gate-regression-probe empty; git rev-parse dev == origin/dev == 2d02fff)"
        status: pass
    human_judgment: false
  - id: D3
    description: "WR-05 README minimal-touch: both new jobs documented by exact name, matrix line corrected to 3.11/3.12/3.13 x numpy 1.26.4/2.2.0, both flipped slow-test claims corrected, codecov references gone, census + --no-cov guidance in Local Testing; Black/isort/Flake8 claims and unrelated sections untouched"
    verification:
      - kind: other
        ref: "README-GATE-DOC-OK + ! grep 'excluded from CI' (empty) + ! grep -qi codecov (empty) — all PASS"
        status: pass
    human_judgment: false
  - id: D4
    description: "Owner hand-off recorded: exact branch-protection gh api commands for dev and main naming the check context verbatim ('coverage-gate (py3.12, fast leg)'), nightly runtime decision menu fed by the calibration run, WR-02/WR-06 leave-untouched rationale, deploy-needs-unchanged note"
    verification:
      - kind: other
        ref: "grep 'branches/dev/protection' + grep 'coverage-gate (py3.12, fast leg)' in 04-03-SUMMARY.md — both PASS"
        status: pass
    human_judgment: false
  - id: D5
    description: "The owner's two remaining decisions — making coverage-gate a required check, and accepting/paying for the nightly runtime — are recorded as runnable artifacts but belong to the owner"
    verification: []
    human_judgment: true
    rationale: "Branch protection changes repo merge policy and the runtime choice spends money/latency; the executor records the exact commands and the decision menu, the owner decides at phase close"

duration: 52 min
completed: 2026-09-30
status: complete
---

# Phase 4 Plan 3: GATE-04 Probe + WR-05 README Summary

**The gate's teeth are proven: probe PR #39 deleted tests/models, the real coverage-gate check on dev concluded FAILURE with the verbatim line `ERROR: Coverage failure: total of 79 is less than fail-under=90` in the CI log (all 1261 collected tests green — only the floor red), and the probe left zero residue — plus the WR-05 README now documents the two-job gate and the owner holds the branch-protection and nightly-runtime decisions as runnable artifacts**

## Performance

- **Duration:** 52 min (started 2026-09-30T17:06:12Z, completed 2026-09-30T17:58Z — CI observation ~40 min of it; the post-task census re-run adds 15 min through 18:14Z)
- **Started:** 2026-09-30T17:06:12Z
- **Completed:** 2026-09-30T17:58:36Z (verification census to 18:14:09Z)
- **Tasks:** 2/2
- **Files modified:** 1 (.github/workflows/README.md)

## GATE-04 evidence

**The gate failed a real coverage-dropping PR end to end.** The Phase-1 exit-code fix carried a coverage failure all the way from pytest rc=1 to a red PR check.

| Fact | Value |
|---|---|
| Probe PR | https://github.com/zhangtaolab/DNALLM/pull/39 (closed, unmerged) |
| CI run | https://github.com/zhangtaolab/DNALLM/actions/runs/36749810723 (pull_request, created 2026-09-30T17:13:06Z) |
| Gate job | `coverage-gate (py3.12, fast leg)` — job id 110005226448 |
| Final conclusion | **FAILURE** (`gh pr view 39 --json statusCheckRollup` → `FAILURE`) |
| Verbatim failure line (CI job log, 17:19:29Z) | `ERROR: Coverage failure: total of 79 is less than fail-under=90` |
| Verbatim pytest-cov line (CI job log) | `FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%` |
| CI coverage table | `TOTAL 7405 1562 79%` (local rehearsal: 1561 missing, 78.92% — 0.01-point environment drift) |
| CI pytest summary | `1261 passed, 1 skipped, 25 deselected, 7 warnings in 157.51s` — every collected test GREEN; only the floor made the job red |
| Probe commit | `a1727bf` on branch `gate-regression-probe` (deleted; never merged) |
| Local rehearsal (pre-push, /tmp/p4-03-rehearsal.log) | rc=1, `ERROR: Coverage failure: total of 79 is less than fail-under=90`, 78.92%, 1261 passed |

Verbatim, as grepped from `gh api repos/zhangtaolab/DNALLM/actions/jobs/110005226448/logs`:

```text
2026-09-30T17:19:29.7455716Z ERROR: Coverage failure: total of 79 is less than fail-under=90
2026-09-30T17:19:29.7499058Z TOTAL                                   7405   1562    79%
2026-09-30T17:19:29.7499360Z FAIL Required test coverage of 90.0% not reached. Total coverage: 78.91%
```

Other checks on the probe PR (all expected): the six matrix legs enforce the same floor, so `test (py3.11, numpy2.2.0)` failed with the identical coverage failure and the remaining legs were cancelled by matrix fail-fast; `test-cuda (3.11, 12.1)` succeeded (12.4 was still running when evidence was captured — its outcome is irrelevant to the gate); `test-mamba` no-op success (no GPU on the runner); `coverage-nightly` correctly skipped (pull_request event — the event guard held); `docs-validation` and `GitGuardian Security Checks` succeeded.

## Zero-residue assertion

- PR #39: `state=CLOSED`, `mergedAt=null` (closed unmerged)
- Remote branch: `git ls-remote --heads origin gate-regression-probe` → empty
- Local branch: deleted; repo back on `dev` at `2d02fff` == `origin/dev`
- `tests/models/` fully restored on dev; dev history carries no trace of the deletion
- No models-cache write from the red run: the coverage-gate job declares no models cache at all, and actions/cache saves only on job success

## Plan-level verification (re-run at plan end)

- Full local census of record: **rc=0**, `Required test coverage of 90.0% reached. Total coverage: 96.30%` (TOTAL 7405 / 274 missing), 1656 passed / 7 allowlisted skips in 875.09s — identical to the Phase-3 landing state (source tree unchanged since 04-01 apart from CI/docs files)
- `scripts/audit_skips.py /tmp/p4-03-census.xml tests/expected_skips.yaml` → exit 0 (all 7 skips allowed)
- `pragma: no cover` count in `dnallm/` → exactly 3 (transformers_compat.py:87,156,181 — Phase-3 baseline held)

## Task Commits

Each task was committed atomically:

1. **Task 1: GATE-04 synthetic-regression probe** - no commit on dev (evidence-only by design: the probe commit `a1727bf` lived on the throwaway `gate-regression-probe` branch and was deleted with it; the evidence lives in this SUMMARY)
2. **Task 2: WR-05 README minimal-touch + owner hand-off** - `29ed164` (docs)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified

- `.github/workflows/README.md` - triggers gain the nightly cron + manual dispatch; new `### 5. Coverage Gate Job (coverage-gate)` and `### 6. Nightly Coverage Job (coverage-nightly)` sections; test-job Coverage-Upload step removed and fast-test step states the enforced floor; matrix corrected to Python 3.11/3.12/3.13 x NumPy 1.26.4/2.2.0; both "slow tests excluded from CI" claims corrected to the nightly reality; Coverage Requirements states the enforced `fail_under = 90` floor instead of the removed codecov upload; Local Testing gains the census command of record and the `--no-cov` scoped-run rule. Black/isort/Flake8 claims and all unrelated sections untouched (WR-05 breadth boundary).
- `.planning/phases/04-ci-gate-enforcement/04-03-SUMMARY.md` - this file: probe evidence + owner hand-off

## Decisions Made

- **Probe target = the whole `tests/models` directory, not the single file** — the plan's own rehearsal gate fired: the as-written single-file rehearsal exited 0 at 91.64% on today's tree (fresh confirmation of 04-01's calibration), so per action step 1 + assumption FA-GATE-04 the target was re-planned to the directory deletion (rehearsed red at 78.92%; CI reproduced at 78.91%). The prohibition is respected: the deletion exists only on the never-merged probe branch.
- **Evidence via the job-level logs API** — run-level `--log-failed` was unavailable (test-cuda 12.4 still running; GitHub gates run logs on run completion), but `gh api .../actions/jobs/<id>/logs` serves a completed job mid-run. The captured lines are byte-identical to what run-level harvest would return.
- **GATE-05 claimed now, with 04-02**: the probe PR is the live dev-side proof (a PR targeting dev provably triggered `coverage-gate` to FAILURE); the main-side path is structural (same inherited `pull_request` block, asserted in 04-02's structural verification).

## Owner hand-off (the decisions that remain yours)

### (a) Branch protection — make the gate REQUIRED (runnable, needs your admin)

Neither branch is protected today (re-verified at plan end: `gh api repos/zhangtaolab/DNALLM/branches/{dev,main}/protection` → 404 "Branch not protected"), so a red check does not block merges until you run:

```bash
gh api -X PUT repos/zhangtaolab/DNALLM/branches/dev/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["coverage-gate (py3.12, fast leg)"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

Run the identical command with `branches/main/protection` for main (same payload). The context string must match the job's `name:` field exactly — `coverage-gate (py3.12, fast leg)` — or protection silently waits on a check that never reports. Optionally add more contexts from the rollup (e.g. the six `test (pyX.Y, numpyZ)` legs); the gate check alone is the coverage enforcement. Note: with `strict: false`, coverage is enforced without forcing branch-up-to-date rebase churn.

### (b) Nightly runtime decision menu (fed by calibration run 36747594207)

Known now: dispatch-to-census was 2m20s (uv cache warm from the green push run); the models cache COLD-MISSED at 0s and seeds only on job success (~5G of HF+ModelScope models download inside the first census); census started 16:56:32Z; projection 4-7.5h on the standard 4-core/16GB runner with the 480-minute backstop and per-test timeout marks (7200/3600) from Wave 1. The run was still healthy and in progress at this plan's close (62+ min into its census) — **read its final conclusion, wall time, and `--durations=0` output from https://github.com/zhangtaolab/DNALLM/actions/runs/36747594207 at phase close** (watch specifically for exit 137 / "job has been terminated" — the OOM warning sign). Then choose:

1. **Accept** a hours-long nightly on standard runners (zero cost, slow feedback; 480-min backstop already in place)
2. **Larger runner** (8-core, paid) to cut wall time roughly in half
3. **Pull LANE-01 forward** from the v2 backlog (two-lane split + nightly drift detection) — the structural fix

### (c) WR-02 / WR-06 leave-untouched rationale + deploy note

- **WR-02 (test-mamba no-op GPU leg)** — left untouched: on CI runners without a GPU the job's GPU check short-circuits and succeeds in ~1 min without running tests; `deploy.needs: [test, test-cuda, test-mamba]` already tolerates it. Touching it is behavior change beyond gate scope.
- **WR-06 (unpinned `curl | sh` uv installer)** — left untouched: pinning is supply-chain hygiene orthogonal to the gate and would touch six install steps across jobs (scope creep). If you want it later, swap each `curl -LsSf https://astral.sh/uv/install.sh | sh` for `astral-sh/setup-uv@<pinned-version>`.
- **deploy needs unchanged** — the multi-hour nightly must not delay docs deploys: deploy keeps `needs: [test, test-cuda, test-mamba]` (not the coverage jobs) and its `if` was tightened in 04-02 to push+main/master, so nightly/manual runs can never deploy.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's single-file probe target cannot clear the floor on the fast leg — re-planned to the directory deletion**
- **Found during:** Task 1 (local rehearsal, before anything was pushed)
- **Issue:** The plan's rehearsal (`--ignore=tests/models/test_model.py`) exited **0 at 91.64%** ("Required test coverage of 90.0% reached", 1474 passed — /tmp/p4-03-rehearsal-singlefile.log), re-confirming 04-01's finding that under `-m "not slow"` the file's fast tests cover only ~344 statements; the plan's A5 arithmetic forgot the marker filter already removes the slow tests' coverage from the denominator. Pushing it would have produced a green probe proving nothing.
- **Fix:** Executed the plan's own re-plan path (action step 1 + assumption FA-GATE-04): target re-planned to the whole `tests/models` directory (7 tracked files, 4,936 lines). Rehearsal with `--ignore=tests/models` → **rc=1, 78.92%**, fail-under line present (/tmp/p4-03-rehearsal.log — the artifact at the verify block's named paths); the pushed probe deleted exactly that directory; CI reproduced at 78.91%.
- **Files modified:** none on dev (the deletion lived only on the ephemeral probe branch)
- **Verification:** REHEARSAL-RED rc=1 + CI gate conclusion FAILURE with verbatim lines captured
- **Committed in:** n/a (probe branch, deleted)

**2. [Rule 1 - Bug] Task 1 verify block 2 uses a gh JSON field that does not exist (`merged`)**
- **Found during:** Task 1 (verify step)
- **Issue:** `gh pr view --json state,merged` rejects `merged` in the installed gh version ("Unknown JSON field"); the field is `mergedAt` (null when unmerged).
- **Fix:** Recomposed the identical assertion as `.state + "/" + (.mergedAt != null | tostring)` → `CLOSED/false`; semantics unchanged.
- **Files modified:** none (verification-command composition only)
- **Verification:** `PROBE-CLEAN-EVIDENCED pr=39`
- **Committed in:** n/a

**3. [Rule 1 - Bug] Run-level `gh run view --job --log-failed` unavailable while the run's last leg is still executing**
- **Found during:** Task 1 (evidence capture)
- **Issue:** The coverage-gate job had concluded FAILURE, but `gh run view --job ... --log-failed` returned "run is still in progress; logs will be available when it is complete" (the test-cuda 12.4 leg was still running) — the plan's capture command could not execute verbatim at that moment.
- **Fix:** Harvested the log via the job-level endpoint `gh api repos/zhangtaolab/DNALLM/actions/jobs/110005226448/logs`, which serves a completed job mid-run; grepped the verbatim lines from it. Same log content, byte-identical.
- **Files modified:** none
- **Verification:** the verbatim lines above, grepped from the job log
- **Committed in:** n/a

---

**Total deviations:** 3 auto-fixed (3x Rule 1 — one plan-arithmetic defect anticipated by the plan's own rehearsal gate, two verify-mechanics compositions; none affect the shipped artifacts)
**Impact on plan:** No scope creep. GATE-04's substance (red check + verbatim evidence + zero residue) is exactly what the plan required, with the calibrated probe target.

## Issues Encountered

- `gh pr close --delete-branch` closed PR #39 and deleted the remote branch, but its automatic local checkout/branch-delete step failed (local `.planning/config.json` modifications blocked gh's internal checkout). Cleaned up explicitly: `git push origin --delete gate-regression-probe` + `git branch -D gate-regression-probe`; zero-residue assertion then passed. (The `.planning/config.json` working-tree modification is GSD orchestrator bookkeeping, predating this plan and left untouched.)
- The nightly calibration run 36747594207 (04-02, still in flight) ran concurrently with the probe; both runs were healthy and independent (event isolation held on both: probe PR ran only push/PR jobs; the nightly dispatch ran only coverage-nightly).

## Known Stubs

None — no stubs, placeholders, or unwired paths were introduced.

## User Setup Required

None - no external service configuration required. (The branch-protection command in the Owner hand-off is an owner decision, not environment setup.)

## Next Phase Readiness

- Phase 4 is complete pending verification: GATE-01 (04-01), GATE-02/GATE-03 (04-02), GATE-04/GATE-05 (this plan) all evidenced. Remaining owner actions are recorded above as runnable artifacts: branch protection on dev+main, and the nightly-runtime choice once run 36747594207 finishes.
- Reminder carried from 04-02: the nightly CRON arms only when this ci.yml reaches main (scheduled workflows run from the default branch only); manual dispatch works on dev today.
- Local dev note: every `--cov` invocation is gated — scoped runs drop `--cov` or pass `--no-cov` (now documented in the workflows README).

## Self-Check: PASSED

- Files exist: .github/workflows/README.md modified (committed 29ed164, both job names + matrix + no-cov guidance verified by grep) — FOUND
- Commits exist on dev: 29ed164 — FOUND (git log); probe commit a1727bf existed on the deleted probe branch (recorded here as evidence)
- Plan-level verification re-run green: census rc=0 at 96.30%, skip audit exit 0, pragma count exactly 3
- Live evidence: PR #39 CLOSED/false with coverage-gate FAILURE; run 36749810723; zero residue asserted

---
*Phase: 04-ci-gate-enforcement*
*Completed: 2026-09-30*
