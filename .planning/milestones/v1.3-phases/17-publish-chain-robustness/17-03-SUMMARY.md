---
phase: 17-publish-chain-robustness
plan: "03"
subsystem: infra
tags: [github-environments, pypi, oidc, trusted-publishing, owner-gate, halted-checkpoint]

requires: []
provides:
  - GitHub environment `pypi` (id 24275955008) — the OIDC claim's environment anchor, created and API-verified BEFORE any registration ask (Pitfall 4 ordering honored)
  - halted-state record of the deferred Task 2 owner gate — the five exact registration strings are on record here and in 17-03-PLAN Task 2, with the resume path carried as a WINDOWS ledger item
affects: [19-docs-sweep (Task 3 README trigger docs still pending), 18-packaging-extras-surgery (smoke dispatches unchanged — upload path still release-gated), v1.3 milestone close (PUBS-04 registration clause open)]

actuals:
  tokens: 4758    # chars/4 over files actually changed (SUMMARY 13858 + WINDOWS/STATE diff chars); plan estimate 30000 — halted early at the owner-deferred Task 2 checkpoint
  tasks: 1        # Task 1 complete; Task 2 checkpoint presented -> owner DEFERRED; Task 3 not executed
  commits: 0      # measured from plan ledger at finalization: zero task commits (Task 1 is external GitHub state only); the halted-state docs record commit follows this write and is excluded by construction (17-01/17-02 convention)

plan_head_before: 26e23a574c5507872812e14e8a470bcd8c0723ea
plan_head_after: 26e23a574c5507872812e14e8a470bcd8c0723ea

tech-stack:
  added: []       # no packages — external GitHub configuration state only
  patterns:
    - "owner-deferral handling at a blocking-human gate: partial SUMMARY with status: halted (#2830 vocabulary) + WINDOWS carried item + STATE session note — never mark the requirement complete, never improvise past the gate"

key-files:
  created:
    - .planning/phases/17-publish-chain-robustness/17-03-SUMMARY.md
  modified:
    - .planning/WINDOWS.md
    - .planning/STATE.md

key-decisions:
  - "Owner explicitly DEFERRED the Task 2 registration checkpoint (稍后再做, relayed by the coordinator 2026-10-14 ~11:5xZ) — recorded as a carried WINDOWS unrun-verify item; plan left halted with PUBS-04's registration clause open"
  - "Environment pypi created with ZERO protection rules (deliberate) — required-reviewer protections would add a manual-approval dependency to the release path; deployment_branch_policy null"
  - "Partial SUMMARY written as status: halted rather than complete — Task 1's environment record and the five registration strings live here so the resume is self-contained; requirements-completed deliberately empty"

patterns-established:
  - "Halt-at-owner-deferral bookkeeping: SUMMARY(status: halted) + WINDOWS carried item + STATE session note; no requirement completion, no downstream content written past the gate"

requirements-completed: []   # deliberately EMPTY — PUBS-04's registration clause is open (owner deferred); do not mark complete until the registration lands and Task 3 closes the plan

coverage:
  - id: D1
    description: "GitHub environment `pypi` created BEFORE any registration ask (Pitfall 4 ordering), with zero protection rules, API-verified twice (post-create and at halt)"
    requirement: PUBS-04
    verification:
      - kind: other
        ref: "gh api repos/zhangtaolab/DNALLM/environments read-back: pypi id 24275955008 created_at 2026-10-14T11:48:46Z protection_rules count 0; plan Task-1 verify command echoed ENVIRONMENT-OK; re-verified at halt (ENVIRONMENT-OK-AT-HALT, 2026-10-14 ~11:55Z)"
        status: pass
    human_judgment: false
  - id: D2
    description: "PyPI trusted-publisher registration for the exact five-string quadruple (dnallm / zhangtaolab / DNALLM / .github/workflows/publish.yml / environment pypi)"
    verification: []
    human_judgment: true
    rationale: "Requires the owner's 2FA PyPI session (no CLI/API from this side); owner explicitly DEFERRED at the blocking-human checkpoint — NOT delivered; resumes via the WINDOWS carried item"
  - id: D3
    description: "README trigger docs + the two carried ledger items (scheduled-leg post-merge verification; PYPI_API_TOKEN deletion gated on the first green OIDC release run id)"
    verification: []
    human_judgment: true
    rationale: "Task 3 is blocked behind the deferred Task 2 checkpoint — deliberately not written at halt (no content past the gate)"

duration: 10 min
completed: 2026-10-14
status: halted
---

# Phase 17 Plan 03: Publish-Chain Owner Readiness Summary (HALTED at owner-deferred checkpoint)

**GitHub environment `pypi` created and API-verified (id 24275955008, zero protection rules) — then the plan HALTED at the Task 2 blocking-human checkpoint: the owner explicitly DEFERRED the PyPI trusted-publisher registration (稍后再做, 2026-10-14); Task 3 (README trigger docs + carried ledger items) not executed; PUBS-04's registration clause remains open**

## Performance

- **Duration:** ~10 min (halt span; started 2026-10-14T11:47:44Z, halted-state recorded ~2026-10-14T11:55Z)
- **Started:** 2026-10-14T11:47:44Z
- **Halted:** 2026-10-14 (owner answer relayed by the coordinator before 11:54:14Z)
- **Tasks:** 1/3 complete (Task 2 checkpoint presented → owner deferred; Task 3 not started)
- **Files modified:** 3 (this SUMMARY, .planning/WINDOWS.md, .planning/STATE.md) + external GitHub state (environment)

## Status: HALTED (not complete)

| Item | State at halt |
|------|---------------|
| Task 1 — GitHub environment `pypi` | **DONE** — created 2026-10-14T11:48:46Z, id 24275955008, verified ENVIRONMENT-OK |
| Task 2 — OWNER PyPI trusted-publisher registration | **CHECKPOINT PRESENTED (~11:49Z) → owner answer 稍后再做 (DEFERRED)** — not registered, no obstacle reported |
| Task 3 — README trigger docs + carried ledger items | **NOT EXECUTED** (blocked behind Task 2) |
| PUBS-04 | **OPEN** — workflow side (17-01) + environment anchor done; registration clause open |

## Task 1 Record (environment)

| Field | Value |
|-------|-------|
| Name | `pypi` |
| id | 24275955008 |
| node_id | EN_kwDORyMrTs8AAAAFpvWtQA |
| created_at | 2026-10-14T11:48:46Z |
| protection_rules | **0 — deliberate**: required-reviewer protections are NOT wanted (they would add a manual-approval dependency to the release path); the environment's role is to anchor the OIDC claim |
| deployment_branch_policy | null (no branch restriction) |

- Created via `gh api -X PUT repos/zhangtaolab/DNALLM/environments/pypi` (account forrestzhang, admin:true verified pre-flight).
- Plan verify command: **ENVIRONMENT-OK** (environments list contains pypi); re-verified at halt: **ENVIRONMENT-OK-AT-HALT** with `[{pypi, id 24275955008, created 2026-10-14T11:48:46Z, rules 0}]`.
- Environments now: `github-pages` (id 13609591726, 1 protection rule) + `pypi` (id 24275955008, 0 rules).
- Structural cross-check (registration must byte-match live workflow state): tracked workflow file is exactly `.github/workflows/publish.yml` (git ls-files); `name: pypi` at line 24; release-only job gate at line 18; job-level `id-token: write` at line 29; `pypa/gh-action-pypi-publish@release/v1` at line 76; smoke cron gate (`0 8 * * 0`) at line 89.
- Ordering honored (Pitfall 4): the environment existed BEFORE the Task 2 checkpoint asked the owner to register.

## Task 2 Record (owner checkpoint — presented, then deferred)

Presented to the owner via the orchestrator at ~2026-10-14T11:49Z with this exact-string checklist:

Register at **https://pypi.org/manage/project/dnallm/publishing/** (sign in with the PyPI account that owns dnallm, 2FA as prompted) — add a GitHub-based publisher with these EXACT values (byte-match matters — a mismatch surfaces later as 403-style upload failures):

| Field | Exact value |
|-------|-------------|
| PyPI project | `dnallm` |
| GitHub owner (organization or user) | `zhangtaolab` |
| GitHub repository | `DNALLM` |
| Workflow filename | `.github/workflows/publish.yml` |
| Environment name | `pypi` |

Presented alongside: the pending-vs-active note (either state satisfies; record which — AS-3) and the warning **do NOT delete the legacy `PYPI_API_TOKEN` secret now** — deletion is gated on the first observed green OIDC release run id (deleting earlier would leave no working publisher if the OIDC path had a defect).

**Owner answer (relay):** 稍后再做 — "do it later". An explicit deferral, 2026-10-14 (~11:5xZ, coordinator-relayed). Neither "registered" nor "obstacle". Token status at halt: secret still present, referenced by no workflow (grep of publish.yml: no PYPI_API_TOKEN reference — unused since 17-01, deletion still gated).

## Remaining Work at Resume (Task 2 → Task 3) — NOT executed at halt

1. Owner registers the publisher with the five strings above; the completing agent records: registration timestamp, the strings verbatim, active-or-pending state, owner confirmation statement (per 17-03-PLAN Task 2 acceptance criteria).
2. Task 3 per 17-03-PLAN: additive trigger-scoped publish-workflow section in `.github/workflows/README.md` (release `published` = the ONLY upload path; dispatch + weekly `0 8 * * 0` cron = smoke-only, citing dispatch run 38554913144; the retired dispatch-to-publish habit, citing run 38112794446; post-merge scheduled-leg note; `PYPI_API_TOKEN` deletion gate; coverage backup note — bare `--cov` and `coverage run` forms, WINDOWS row 20), then this SUMMARY's carried-ledger heading with both items, commit scope `docs(17)`, and plan close-out (PUBS-04 then completable).
3. Both carried obligations noted by 17-01 remain live independently of this halt: (a) scheduled smoke leg — one green scheduled run post-merge to the default branch; (b) `PYPI_API_TOKEN` deletion — gated on the first observed green OIDC release run id (expected at v1.3 close).

## Task Commits

1. **Task 1: Create the GitHub environment `pypi`** — no commit by design: the deliverable is external GitHub state; its record lives in this SUMMARY (17-01/17-02 precedent for SUMMARY-only task records).
2. **Halted-state record** — the single `docs(17-03)` commit carrying this SUMMARY + `.planning/WINDOWS.md` + `.planning/STATE.md` (this commit; hash in the execution report).

**Plan metadata:** this halted-state record commit IS the plan's metadata commit; `plan_head_before` == `plan_head_after` == 26e23a5 — zero task commits by construction (the resume's close-out re-measures from the same ledger).

## Files Created/Modified

- `.planning/phases/17-publish-chain-robustness/17-03-SUMMARY.md` — this halted-state record
- `.planning/WINDOWS.md` — carried unrun-verify item appended (owner-deferred registration; #22/#23 precedent class)
- `.planning/STATE.md` — session note (phase 17 halted at owner-deferred checkpoint, 2/3 plans complete)
- External (not a repo file): GitHub environment `pypi` on zhangtaolab/DNALLM

## Decisions Made

- The owner's deferral is recorded, not worked around: no registration was attempted from this side (impossible without the owner's 2FA session), no Task 3 content was written past the gate, and PUBS-04 stays open.
- Environment `pypi` left with zero protection rules — the release path gains no manual-approval dependency; the environment anchors the OIDC claim only.
- This SUMMARY uses `status: halted` (the machine-read halt vocabulary) so downstream tooling sees an intentional non-completion, not a finished plan.

## Deviations from Plan

None in the deviation-rule sense (no Rule 1-4 fixes were needed). The plan designed Task 2 for two outcomes (confirmation or obstacle); the owner produced a third — explicit deferral — handled on the honest halt path per the coordinator's instruction: partial `status: halted` SUMMARY, WINDOWS carried item, STATE session note. Also disclosed: Task 1's SUMMARY record was held until this halt-state write rather than written immediately after Task 1 (a `17-03-SUMMARY.md` on disk while the checkpoint was pending would have made plan-selection treat 17-03 as complete).

## Issues Encountered

None — the deferral is an owner decision recorded as carried state, not an execution problem.

## User Setup Required

Owner action pending (the deferral itself): register the trusted publisher using the checklist in the Task 2 Record above, then resume per the WINDOWS carried item. No other external configuration is outstanding from this plan.

## Known Stubs

None — no code stubs or placeholders; the halt is recorded state, and the pending owner action is tracked in the WINDOWS ledger (not a silent pending).

## Next Phase Readiness

- Phase 17 stands at 2/3 plans complete (17-01, 17-02); 17-03 is halted at the owner-deferred registration gate.
- Everything scriptable in 17-03 is done (environment); the remaining work is the owner's registration + Task 3's docs/ledger close-out, estimated at minutes once the owner acts.
- Resume: register at https://pypi.org/manage/project/dnallm/publishing (five exact strings above), then re-run `/gsd-execute-phase 17` (see the WINDOWS carried item).
- Downstream phases are NOT blocked: 18 (extras surgery) uses dispatch smoke only; 19 (docs sweep) will simply fold in Task 3's README section if it has not landed by then.

## Self-Check (halt-state, pre-commit facts)

- Environment re-verified live at halt: `pypi` present, id 24275955008, created_at 2026-10-14T11:48:46Z, 0 protection rules — ENVIRONMENT-OK-AT-HALT
- This SUMMARY carries: Task 1's full environment record, the five registration strings, the deferral record, and the resume path
- `.github/workflows/README.md` untouched (Task 3 NOT executed — `git status` shows no `.github/` change)
- WINDOWS entry appended via `gsd_run windows append` (kind unrun-verify) and STATE session note recorded — both verified in the halted-state commit

---
*Phase: 17-publish-chain-robustness*
*Halted: 2026-10-14 (owner-deferred Task 2 checkpoint)*
