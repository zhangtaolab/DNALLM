---
phase: quick-261001-ith
plan: 01
subsystem: infra
tags: [github-actions, self-hosted-runner, ci, mamba, gpu]

provides:
  - "test-mamba CI job executing on the self-hosted dnallm-nightly GPU runner (schedule/dispatch cadence) instead of no-op skipping on GPU-less hosted runners"
affects: [ci-gate-enforcement, ship-triage]

actuals:
  tokens: 4000
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Nightly-cadence event guard (schedule || workflow_dispatch) as the pattern for heavy recurring-build jobs on the single self-hosted runner"

key-files:
  created: []
  modified:
    - .github/workflows/ci.yml

key-decisions:
  - "Nightly cadence (schedule || workflow_dispatch), not push/PR: preserves the repo invariant that PR-authored code (incl. forks) never executes on the self-hosted box, and the per-run CUDA kernel source build is far too heavy for per-push cadence (GATE-02 amended)"
  - "deploy needs trimmed to [test, test-cuda]: GitHub skips dependents of skipped needs, so leaving the event-gated test-mamba in needs would silently stop gh-pages deploys on push"

requirements-completed: []

coverage:
  - id: D1
    description: "test-mamba job re-targeted to [self-hosted, dnallm-nightly] with schedule/workflow_dispatch guard, 180-min timeout backstop, and deploy needs fixed"
    verification:
      - kind: integration
        ref: "ci.yml@0d5a831 deep-compare (yaml.safe_load vs pre-change HEAD): test-mamba header changed, all other jobs + top-level keys identical"
        status: pass
    human_judgment: false
  - id: D2
    description: "Live dispatch proof that test-mamba executes (not skips) on the GPU runner"
    verification:
      - kind: e2e
        ref: "workflow_dispatch run 36821471332 on dev: test-mamba (3.11) on runner dnallm-nightly, labels [self-hosted, dnallm-nightly], every step success incl. GPU-check + mamba install + pytest, job conclusion success"
        status: pass
    human_judgment: false

duration: 40min
completed: 2026-10-01
status: complete
---

# Quick Task 261001-ith: Move test-mamba to Self-Hosted GPU Runner

**test-mamba CI job re-targeted from GPU-less ubuntu-latest (where every step no-op skipped) to the self-hosted dnallm-nightly box on nightly cadence — proven executing green end-to-end via dispatch run 36821471332**

## Performance

- **Duration:** ~40 min (plan 13:42 → ci.yml commit 13:46 → dispatch run green 14:22 CST)
- **Tasks:** 2
- **Files modified:** 1 (`.github/workflows/ci.yml`)

## Accomplishments

- `test-mamba` now runs on `[self-hosted, dnallm-nightly]` under `if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'` with a 180-minute `timeout-minutes` backstop against hung CUDA kernel builds monopolizing the single runner
- `deploy` job `needs` trimmed to `[test, test-cuda]` — test-mamba being skipped on every push would otherwise silently stop gh-pages doc deploys (GitHub skips dependents of skipped needs)
- Live proof: dispatch run **36821471332** (2026-10-01T05:47Z, dev) — `test-mamba (3.11)` assigned to runner `dnallm-nightly` (labels `self-hosted`, `dnallm-nightly`), queued ~23 min behind coverage-nightly as predicted, then ran all steps green (GPU-check, venv + mamba kernel install, mamba pytest, log upload) in ~12 min; whole-run conclusion **success**. On the same run, `test`/`test-cuda`/`coverage-gate`/`deploy` were correctly skipped by their event guards — push/PR behavior structurally unchanged.

## Task Commits

1. **Task 1: Re-target test-mamba + deploy needs fix** — `0d5a831` (ci)
2. **Task 2: Live dispatch validation** — no commit (evidence recorded here; run 36821471332)

**Plan metadata:** `94169ea` (docs)

## Files Created/Modified

- `.github/workflows/ci.yml` — test-mamba job header (runs-on, if-guard, timeout-minutes, rationale comments) + deploy needs

## Decisions Made

- Nightly cadence rather than push/PR (see key-decisions; matches GATE-02 amended two-lane architecture and the coverage-nightly security posture comment in the same file)
- GPU-check step kept as fail-safe no-op per task brief — if the box ever loses its GPU the job green-no-ops instead of failing the nightly run

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. (Task 2's bounded-wait contingency for queueing behind coverage-nightly was exercised — the job sat queued ~23 min behind the nightly census — but the run completed green within the observation window, so no pending-live-proof fallback was needed.)

## Subsequent Related Changes (not part of this task)

- `8151c09` (WR-01): nightly test-mamba failures now fail the job (continue-on-error removed)
- `de4b5cc` (CR-03): install extras set corrected to `.[base]`; its first live nightly rehearsal was pending at milestone close (see v1-MILESTONE-AUDIT.md Phase 04 tech debt)

## User Setup Required

None.

## Next Phase Readiness

- Mamba CI signal is real for the first time; the de4b5cc install rehearsal lands on the next 03:00 UTC nightly schedule

---
*Phase: quick-261001-ith*
*Completed: 2026-10-01*
