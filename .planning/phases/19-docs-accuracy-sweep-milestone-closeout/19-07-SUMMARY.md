---
phase: 19-docs-accuracy-sweep-milestone-closeout
plan: 07
subsystem: ci
tags: [milestone-closeout, pkgh-05, docr-01, docr-02, docr-03, cuda130, ci-matrix, publish-smoke, nightly-trio, evidence-ledger, interpreter-pin]

requires:
  - phase: 19-docs-accuracy-sweep-milestone-closeout (plans 01-06)
    provides: the strict-clean docs baseline, the cuda130-default install surface, the model-name lock alignment, and the 149-file disposition set this plan consolidates
  - phase: 18-packaging-install-honesty-atomic-extras-surgery (plan 01)
    provides: the final extras surface (full/cuda130) the test-cuda re-point and install surfaces describe
  - phase: 17-publish-chain-robustness (plan 01)
    provides: the event-gated publish chain whose smoke leg this battery re-proves
provides:
  - The milestone-close CI vehicle: test-cuda leg re-pointed to 13.0 (three synchronized lines) + temp `- v1.3` push-filter entry (removal decision recorded in 19-EVIDENCE.md; stays until actual merge or explicit removal)
  - The closeout battery, all green on the final tree: push-fired matrix run 38677302395 (test ×6, test-windows, **test-cuda 3.11@13.0 — the first-ever cuda130 CI leg, PKGH-05's executable proof**, coverage-gate) + publish smoke run 38678155010 (smoke success / publish skipped — upload guard holds) + nightly trio run 38695301631 (test-mamba, example-nightly, coverage-nightly ALL success under interpreter-pinned labels)
  - 19-EVIDENCE.md — the consolidated disposition ledger (149 per-file rows + class rows + the closeout battery record) replacing the five per-batch 19-REVIEW ledgers (retired same-commit; git history keeps provenance)
  - Two repo-caused reds en route fixed in-pass and re-proven green: ruff-format on two sweep-touched docs blocks (3075b09), extras-guard path tuples posix-normalized for Windows (15bf646)
  - The nightly interpreter-honesty fix (bdf6ba3): self-hosted nightly venvs pinned to setup-python's exact toolcache interpreter + mamba wheelhouse cache key py-tagged — closes the unpinned-uv-venv drift class (uv-managed 3.11 since 10-07, 3.13 since 10-14, silently replacing the labeled 3.12/3.11), with the disclosure recorded in 19-EVIDENCE.md
  - PKGH-05, DOCR-01, DOCR-02, DOCR-03 marked complete (each declared only by this plan)
affects: [ci.yml self-hosted nightly legs (interpreter pin), wheelhouse cache keying, milestone close-out records, WINDOWS ledger #25/#27 owner-deferred summaries at audit]

actuals:
  tokens: 60389    # chars/4 over the realized diff (git diff 49b8b14..13950bf = 241,557 chars; the diff is dominated by the consolidated evidence ledger's docs churn)
  tasks: 3
  commits: 10      # 49b8b14..summary-commit: 165800d, 3075b09, 15bf646, a68c1a0, 060a583, 71c0e8c, bdf6ba3, 9404bdf, 13950bf, + this SUMMARY commit
plan_head_before: 49b8b1409e081d0a5f8ad0acfe24a37138292cd3
plan_head_after: 13950bf869d6a5b3e04a5e579d4e0e5f8a35c369   # last work commit; the SUMMARY commit rides on top

tech-stack:
  added: []
  patterns:
    - "Interpreter-label honesty on persistent self-hosted runners: an unpinned `uv venv` resolves uv-managed interpreters first, so the newest one installed on the box silently replaces setup-python's version — pin with `uv venv --python \"$pythonLocation/bin/python\"` and never key a wheelhouse cache without the venv's cpXY tag"
    - "Async-job closeout discipline (16-04 precedent) scales to a fix cycle: manifest keeps the prior_runs chain with per-job conclusions + classification + fix commit, so a repo-caused red and its superseding green stay one discoverable record"

---

## Performance

- 260 min active / ~34 h wall (the wall is dominated by two serialized nightly trios: 38678158479 03:44→06:33, then the post-fix 38695301631 06:54→15:09 — the single-runner serial cost is the price of the closeout battery's honesty)
- 10 commits, 18 files, +771/−492 code-lines class plus the evidence ledger churn

## Accomplishments

1. **CI vehicle** (165800d): test-cuda re-pointed to `cuda-version: ['13.0']` + install `.[full,cuda130]` (three synchronized lines), temp `- v1.3` push-filter entry with the 2026-10-15 dated lesson comment, `.github/workflows/README.md:59` same-change. Removal decision (stay until actual merge or explicit removal) recorded in 19-EVIDENCE.md.
2. **Closeout battery, all green**: matrix 38677302395 (test ×6 + test-windows + **test-cuda 3.11@13.0, the first cuda130 leg ever run — PKGH-05's executable proof** + coverage-gate); publish smoke 38678155010 (fresh wheel installs + import probe; publish skipped — upload guard holds); nightly trio 38695301631 (mamba 09:59→10:08, example 06:54→09:59 incl. the Stage 0 wheelhouse rebuild under the pinned 3.12 saved under the py-tagged key 8986514177, coverage 10:49→15:09 — 4h20m, the full slow suite green; pytest liveness verified at 3h39m before letting it run out).
3. **Two repo-caused first-exposure reds fixed in-pass and re-proven**: ruff-format on two sweep-touched docs code blocks (3075b09); extras-guard path tuples posix-normalized (Windows path-separator vs the frozenset — `.as_posix()`, 15bf646). Matrix green on the second push (38677302395).
4. **Nightly interpreter-honesty fix** (bdf6ba3): the first trio dispatch (38678158479) died in example-nightly Stage 0 in 15 ms — root-caused per A.5 as repo-caused (NOT rice, NOT infra): unpinned `uv venv` + Python-untagged wheelhouse cache key let a cp311 wheelhouse (saved by scheduled run 38400225941's drifted 3.11 venv) restore into a cp313 venv (uv-managed 3.13.16 appeared on the box 10-14 10:12Z). Fixed by pinning all three self-hosted nightly legs to setup-python's exact interpreter + py-tagging the cache key; poisoned entry 8913155507 deleted (3.57 GiB); re-dispatched trio 38695301631 **all green**. The drift disclosure (nightly venvs ran 3.11 since 10-07 / 3.13 since 10-14 under 3.12-labeled jobs — every green real, labels now true) recorded in 19-EVIDENCE.md for the milestone audit.
5. **Consolidation** (a68c1a0 → 13950bf): 19-EVIDENCE.md assembled — 149 per-file disposition rows + class rows + the closeout battery with both nightly runs' terminal records; the five per-batch 19-REVIEW ledgers retired same-commit.

## Task Commits

- Task 1 (vehicle): 165800d
- Task 2 (battery): 3075b09, 15bf646 (repo-caused reds), 060a583, 71c0e8c (PENDING records), bdf6ba3 (interpreter-pin fix), 9404bdf (38678158479 terminal record + re-dispatch), 13950bf (38695301631 terminal, all green)
- Task 3 (consolidation): a68c1a0 + the ledger rows carried in 13950bf

## Files Created/Modified

.ci.yml (vehicle + interpreter pin + cache key), .github/workflows/README.md, docs/ (two ruff-format blocks), tests/test_extras_guard.py, .planning/: 19-EVIDENCE.md, the five retired 19-REVIEW-*.md, async-jobs/19-07-nightly.json, STATE.md, ROADMAP.md, REQUIREMENTS.md, this SUMMARY.

## Decisions Made

- Rice-class rule NOT triggered this battery: the one red was repo-caused (interpreter/cache-key defect), classified before any retry per A.5; no rice manifestation occurred in 38695301631.
- Hosted legs left unpinned deliberately: ephemeral runners cannot accumulate uv-managed interpreters, PATH resolution is stable there — the pin is scoped to the persistent box legs.
- No new WINDOWS row for the interpreter drift: fixed-and-disclosed, no going-forward defect; surfaced for the milestone audit instead.

## Deviations from Plan

- The plan's expected wall (~8 h one trio) doubled: the first dispatch hit the repo-caused Stage 0 red, requiring the fix cycle + full re-dispatch. The async-jobs manifest carried the external_job_waiting state across both cycles per the 16-04 precedent (prior_runs chain).

## Issues Encountered

- None outstanding. The `gh api` annotations endpoint 404'd for the failed job (check-runs annotations unavailable); full job logs via `gh run view --log-failed` + per-step timings supplied the root-cause evidence instead.

## Authentication Gates

- None this plan. (Owner-deferred gates #25 PyPI trusted-publisher registration and #27 Windows KDA smoke remain open WINDOWS rows for the milestone audit to surface.)

## User Setup Required

- None.

## Next Phase Readiness

- Phase 19 is the last phase of v1.3: all 7 plans complete, battery honest-green. Next: phase 19 verification (19-VERIFICATION.md) → transition (milestone 7/7) → milestone lifecycle (audit-milestone → complete-milestone v1.3 → cleanup).

## Self-Check: PASSED

- All 3 tasks executed and committed; battery records every run ID with per-job conclusions; requirements PKGH-05/DOCR-01/DOCR-02/DOCR-03 marked complete only after the green battery + this SUMMARY existed; no Co-Authored-By trailers on any commit (verified per commit).
