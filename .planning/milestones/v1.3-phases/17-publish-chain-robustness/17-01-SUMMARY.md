---
phase: 17-publish-chain-robustness
plan: 01
subsystem: infra
tags: [github-actions, pypi, oidc, trusted-publishing, publish-chain, ci-cd, uv, twine]

requires: []
provides:
  - release-only OIDC upload path in .github/workflows/publish.yml (job-level event gate + pypa/gh-action-pypi-publish@release/v1)
  - never-uploading smoke chain (dispatch + weekly 0 8 * * 0 cron): uv build --no-sources -> twine check -> clean-venv install-from-dist -> import probe
  - live dispatch proof that a non-release event cannot instantiate any upload-capable step (run 38554913144)
affects: [17-03 (environment pypi + trusted-publisher registration bind this workflow filename), 18-packaging-extras-surgery (smoke dispatches validate its output), 19-docs-sweep]

actuals:
  tokens: 890      # chars/4 over the realized diff (publish.yml: 76 add / 9 del = 3563 chars)
  tasks: 2
  commits: 1       # measured: git rev-list --count 5e78ba3..HEAD (docs commit excluded by construction)

plan_head_before: 5e78ba31a2f368dd942b4e354691dd1af6a7ec3f
plan_head_after: 2c649fa0c48bc0fc465e404ab38094c3b538040d

tech-stack:
  added: [pypa/gh-action-pypi-publish@release/v1]
  patterns:
    - "job-level event gating of an upload job (if: github.event_name == 'release') — stronger than step-level: the whole job is skipped, so a later-added step cannot re-open the dispatch path"
    - "D-19 schedule-aware smoke gate extended to publish.yml (dispatch disjunct + own cron string match)"
    - "atomicity invariant: upload guard and schedule trigger must share one introducing commit (pickaxe-provable)"

key-files:
  created:
    - .planning/phases/17-publish-chain-robustness/17-01-SUMMARY.md
  modified:
    - .github/workflows/publish.yml

key-decisions:
  - "Upload gate at JOB level, not step level — dispatch/schedule structurally cannot instantiate any upload-capable step"
  - "OIDC upload with print-hash true and attestations true set explicitly (OQ4); skip-existing left off — fail loud on immutable-filename collisions"
  - "Registry pip-install verify step deleted entirely (OQ2 drop branch) — the smoke's install-from-dist + import probe is the uniform artifact verification"
  - "Smoke python 3.12 (OQ3) and env pair UV_HTTP_TIMEOUT/UV_CONCURRENT_DOWNLOADS carried on both install-bearing smoke steps"

patterns-established:
  - "Never-uploading CI smoke: reads dist/ only; no publisher action, no upload command, no registry install, no token references"
  - "Guard+trigger atomicity proof by pickaxe: git log -S on both strings must yield the identical introducing hash"

requirements-completed: [PUBS-01, PUBS-02, PUBS-03, PUBS-04]

coverage:
  - id: D1
    description: "Atomic publish.yml transformation: job-level release-only upload gate, OIDC migration (pypa action, environment pypi, job-level id-token), never-uploading smoke job, weekly schedule trigger — one commit"
    requirement: PUBS-01
    verification:
      - kind: other
        ref: "structural python proof (yaml.safe_load assertions on triggers/jobs/gates/permissions/action inputs/forbidden strings) -> PUBLISH-GATES-OK"
        status: pass
      - kind: other
        ref: "pickaxe atomicity: git log -S '0 8 * * 0' and git log -S \"if: github.event_name == 'release'\" both resolve to 2c649fa -> ATOMIC-COMMIT-OK"
        status: pass
    human_judgment: false
  - id: D2
    description: "Live dispatch proof on the remote ref: publish job skipped by the event gate, smoke chain green through the import probe"
    requirement: PUBS-02
    verification:
      - kind: other
        ref: "gh api runs/38554913144/jobs jq: publish conclusion skipped AND smoke conclusion success -> true (JOBS-ASSERT-OK)"
        status: pass
      - kind: other
        ref: "gh run view 38554913144 --log | grep -F 'installed from dist' -> 'DNALLM 1.2.1 installed from dist'"
        status: pass
    human_judgment: false

duration: 10 min
completed: 2026-10-14
status: complete
---

# Phase 17 Plan 01: Publish Guard + Never-Uploading Smoke Tracer Summary

**publish.yml transformed in ONE atomic commit: job-level release-only upload gate + OIDC trusted publishing + dispatch/weekly-cron smoke chain (uv build --no-sources → twine check → clean-venv install-from-dist → import probe) — proven live with the publish job skipped and smoke green on run 38554913144**

## Performance

- **Duration:** 10 min (595 s)
- **Started:** 2026-10-14T11:05:01Z
- **Completed:** 2026-10-14T11:14:56Z
- **Tasks:** 2/2
- **Files modified:** 2 (.github/workflows/publish.yml + this SUMMARY)

## Accomplishments

- The live upload hole is closed structurally: the `publish` job now carries a job-level `if: github.event_name == 'release'` gate — a workflow_dispatch or schedule event cannot instantiate ANY upload-capable step (the whole job is skipped, proven live)
- The release path migrated to OIDC in the same commit: `pypa/gh-action-pypi-publish@release/v1` (environment `pypi`, job-level `id-token: write` + `contents: read`, `print-hash: true`, `attestations: true`, no token inputs); the token-auth upload step and the registry pip-install verify step are deleted from the file entirely
- The never-uploading smoke is wired end-to-end and proven live on a dispatched run: `uv build --no-sources` → `twine check dist/*` (twine pinned `>=7,<8`, redundant `build` dep dropped) → clean-venv install of `dist/*.whl` → import probe printing the version
- Atomicity invariant held: the guard and the schedule trigger (`0 8 * * 0`, Sunday 08:00 UTC) landed in the same commit — pickaxe-proven (both strings' introducing hash is 2c649fa)
- Byte-compat preserved on every kept release-job step (d104e9e `-e '.[base]'`, 98f7aa5 `--system`, a8102da check_code.py, 0281c66 tests-as-default, UV env pair); census untouched (no tests/ or ci.yml movement — 216/217 triple unchanged)

## Task Commits

Each task was committed atomically:

1. **Task 1: ONE atomic commit — release-only OIDC upload job + never-uploading smoke job** - `2c649fa` (ci)
2. **Task 2: Live end-to-end dispatch proof** - evidence recorded in this SUMMARY; committed with the plan metadata commit below (Task 2's only file is this SUMMARY)

**Plan metadata:** (docs commit — see below)

## Live Dispatch Proof (Task 2 evidence)

**Pre-flight assertions (all on record, per the never-dispatch-a-pre-guard-ref discipline):**
- Remote ref pre-flight: `gh api repos/zhangtaolab/DNALLM/contents/.github/workflows/publish.yml?ref=v1.3` (base64-decoded) contains BOTH the job-level gate line `if: github.event_name == 'release'` AND the cron string `0 8 * * 0` → REMOTE-GUARD-PRESENT-OK
- In-flight check: `gh run list --workflow=publish.yml` in_progress=0 queued=0 → NO-INFLIGHT-OK (AS-1 snapshot-semantics mitigation)
- Local HEAD == origin/v1.3 == 2c649fa → REF-SYNC-OK (the atomic commit was pushed BEFORE dispatch)

**Dispatch:** `gh workflow run publish.yml --ref v1.3` at **2026-10-14T11:12:09Z** on ref **v1.3** (sha 2c649fa)

**Run:** run id 38554913144 — https://github.com/zhangtaolab/DNALLM/actions/runs/38554913144 (event workflow_dispatch, workflow "Publish to PyPI", completed success at ~2026-10-14T11:13:5xZ)

**Job conclusions (from the run's jobs API):**

| Job | Conclusion | Meaning |
|-----|------------|---------|
| publish | **skipped** | the job-level gate evaluated false on a dispatch event — the entire upload-capable job was never instantiated (SC1 live clause) |
| smoke | **success** | the full never-uploading chain ran green reading dist/ only (SC2 dispatch clause) |

**Probe line (verbatim from the run log, smoke / Import probe step):**

```
DNALLM 1.2.1 installed from dist
```

Smoke step-level results: Set up job ✓ / checkout ✓ / Set up Python ✓ / Install uv ✓ / Build package (publishable metadata) `uv build --no-sources` ✓ / Check distributions (`twine check dist/*`) ✓ / Install from dist in a clean venv ✓ / Import probe ✓

**Scheduled-leg note (Pitfall 3):** the weekly cron leg is unobservable pre-merge — cron fires only from the default branch, and v1.3 is not it. Its post-merge verification obligation carries to 17-03 Task 3 (carried ledger item, WINDOWS #22/#23 precedent). No re-dispatch ever targeted a pre-guard ref (pre-flight above).

## Files Created/Modified

- `.github/workflows/publish.yml` — three triggers (release [published], workflow_dispatch, schedule 0 8 * * 0) and two jobs: `publish` (release-only OIDC) and `smoke` (dispatch + weekly cron, never uploading)
- `.planning/phases/17-publish-chain-robustness/17-01-SUMMARY.md` — this record

## Decisions Made

- Upload gate at JOB level (not step-level): the whole upload-capable job is skipped on non-release events; a later-added step can never re-open the dispatch path
- OQ4 decided explicitly: `print-hash: true` AND `attestations: true` set on the pypa step; `skip-existing` deliberately NOT set — fail loud on immutable-filename collisions (the 403-retry history argues for surfacing)
- OQ2 drop branch taken: the registry pip-install verify step is deleted entirely; the smoke's install-from-dist + import probe is the uniform artifact verification, removing the last non-publish registry-install path (TRUTH-2)
- Smoke python 3.12 (OQ3): a packaging smoke, not interpreter coverage — ci.yml owns 3.11–3.13

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Caught pre-commit (no landed defect): the initial working-tree draft of publish.yml lost the original file's 4-space trailing-whitespace separator lines between kept steps (byte-compat, Pitfall 6). Detected via the pre-commit git diff review, restored the exact separator bytes before committing — the committed diff shows kept steps as pure context; only the schedule trigger, job headers, the second install spec, the build command, and the upload/verify step replacement changed.

## Authentication Gates

None — gh CLI was already authenticated with admin (verified in Task 1 precondition).

## User Setup Required

None for this plan. (17-03 carries the owner human-gates: PyPI trusted-publisher registration and, later, PYPI_API_TOKEN deletion.)

## Known Stubs

None — no stubs, no placeholders; both deliverables verified end-to-end.

## Next Phase Readiness

- 17-02 (tooling caps) and 17-03 (environment pypi + owner registration + docs/ledger) can proceed; the workflow-side OIDC wiring this plan delivered is what 17-03's registration binds to (exact filename `.github/workflows/publish.yml` + environment `pypi` — never rename either)
- The smoke dispatch is now the on-demand packaging verification Phase 18's extras surgery will rely on (roadmap position note)
- Carried obligations: scheduled-cron leg verification (post-merge, 17-03 Task 3) and PYPI_API_TOKEN deletion (owner gate, first green OIDC release)

## Self-Check: PASSED

- Files exist: `.github/workflows/publish.yml` (transformed), `17-01-SUMMARY.md` (this file) — FOUND
- Task commit `2c649fa` is an ancestor of HEAD — FOUND
- Structural proof re-run post-commit: PUBLISH-GATES-OK
- Pickaxe atomicity re-run post-commit: ATOMIC-COMMIT-OK (A == B == 2c649fa0c48bc0fc465e404ab38094c3b538040d)
- Live run re-check: run id 38554913144 resolves — publish skipped / smoke success / probe line present

---
*Phase: 17-publish-chain-robustness*
*Completed: 2026-10-14*
