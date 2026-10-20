---
phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
plan: 04
subsystem: testing
tags: [windows-parity, nightly-evidence, ci-dispatch, sc-matrix, honesty-ledger, rice-upstream, census-guard]

requires:
  - phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
    provides: "16-01/02/03 — the complete dispatch/gates/frozen-pairs change this plan produces evidence over"
provides:
  - 16-EVIDENCE.md — the phase's evidence file: SC4 nightly terminal record (run 38441258701 over headSha 4863883) + the 5/5 SC proof matrix + the phase-wide honesty ledger (census/skips/deps/lock clean)
  - The SC4 conclusion: every phase-16 surface green on the dispatched ref (live-pybedtools parity EXECUTED green on CI, CRE band re-executed green, NER notebook green end-to-end in example-nightly, both root suites 19+4 green in the full census, D-03 gate held byte-identical) with the owner-accepted rice.uga.edu caveat on two input-download lanes
  - Phase 16 close-out: all five ROADMAP success criteria mapped to green re-runnable commands; WPAR-06/WPAR-08 completed (shared-ID gate satisfied — this was the last declaring plan)
affects: [phase-17, WPAR-06, WPAR-08, rice-upstream-dependency]

actuals:
  tokens: 5200       # chars/4 over the realized diff (16-EVIDENCE.md + manifest + WINDOWS + STATE, ~21K chars)
  tasks: 2
  commits: 4         # MEASURED: git rev-list --count 4863883..09296c5 (3 x 16-04 docs commits + 1 interleaved phase-14 H.6 record 30671e7 by another session)

plan_head_before: 4863883890fe62c845e3c81da06c7e07d4e8e282
plan_head_after: 09296c5e787517f8c9d9c39d9a4ab24b5c4a4c28

tech-stack:
  added: []
  patterns:
    - "Evidence-plan execution across an external-job boundary: dispatch -> async-jobs manifest (external_job_waiting, no SUMMARY) -> coordinator-triggered resume reconciles the terminal run -> record + close; failures recorded with resolutions, never silently retried"
    - "Held-where-executable evidence semantics: an external-host failure upstream of all phase code cannot falsify phase surfaces — the proof burden shifts to per-surface green rows (junit PASS rows + live-parity execution) with the caveat recorded"

key-files:
  created:
    - .planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-EVIDENCE.md
    - .planning/async-jobs/16-04-sc4-nightly.json
  modified: []

key-decisions:
  - "SC4 evidence concluded green-where-executable with an owner-accepted rice.uga.edu caveat: the 4th/5th manifestations of the phase-14-accepted external-host class (H.4/H.4.final) hit the two genome-download lanes; both failures sit upstream of all phase-16 code (the BPE script dispatch never executed); zero phase-16 regression signal — positively proven by the live-pybedtools-parity test EXECUTING green on coverage-nightly, the CRE band re-executing green, and both root suites going 19+4 green in the full census"
  - "The plan's single-sanctioned-re-dispatch rice routing was superseded by the owner-side accept-with-caveat directive (the routing's escalation endpoint; a re-dispatch would re-roll the same external host, as phase-14's own re-dispatch demonstrated)"
  - "Queue-hygiene boundary honored: cancelling the two foreign scheduled runs ahead of the dispatch was denied by the permission classifier (Interfere With Workloads) — respected, recorded, and the PENDING/manifest close-out used instead of a workaround"

requirements-completed: [WPAR-06, WPAR-08]

coverage:
  - id: D1
    description: "SC4 nightly evidence over the complete phase change — dispatched run 38441258701 (ref v1.3 @ 4863883), both nightlies executed, terminal record with live-recheckable ids/rows in 16-EVIDENCE.md section A"
    requirement: WPAR-08
    verification:
      - kind: other
        ref: "16-EVIDENCE.md A.4: junit artifact rows — census gate 216/217 held; CRE band PASS 254.961s; NER notebook PASS 4023.932s; coverage 2607 passed incl. all 19 dispatch-direction + 4 frozen-pair PASS rows; live gh re-check run 38441258701 = failure (rice-class reds, section A.5)"
        status: pass
    human_judgment: true
    rationale: "The nightly jobs concluded failure on the owner-accepted rice.uga.edu external-host class — green-where-executable is an owner sign-off (coordinator directive recorded in A.5), not an all-green automated proof; the verify's live conclusion check returns failure by the facts and must not be auto-passed"
  - id: D2
    description: "5/5 SC proof matrix + phase-wide honesty ledger — every criterion mapped to a green re-runnable command; census triple byte-identical to phase entry, expected_skips/pyproject/uv.lock untouched, docs sync exit 0, fast lane green"
    requirement: WPAR-08
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py + tests/test_frozen_pairs.py -> 23 passed (re-run this session)"
        status: pass
      - kind: unit
        ref: "full fast lane: 2550 passed, 1 pre-existing allowlisted skip, 75 deselected, exit 0 (102.93s)"
        status: pass
      - kind: other
        ref: "scripts/check_docs_sync.py -> OK exit 0; census grep '216/217 tests collected (1 deselected)'; git diff ff95ef7..HEAD over expected_skips.yaml/pyproject.toml/ci.yml/uv.lock all EMPTY"
        status: pass
    human_judgment: false
  - id: D3
    description: "WPAR-06 same-change structure-test proof — the dispatch-direction suite green locally AND all 19 tests PASSED in coverage-nightly's full census on the dispatched ref, including the live-pybedtools parity branch executing green on CI"
    requirement: WPAR-06
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py -> 19 passed locally; /tmp/p16-sc4-cov-log.txt: all 19 per-test PASSED rows incl. test_ner_script_win32_rows_match_live_pybedtools_loj"
        status: pass
    human_judgment: false

duration: 70 min active (16h wall incl. overnight CI queue + run 38441258701)
completed: 2026-10-14
status: complete
---

# Phase 16 Plan 04: SC4 Nightly Evidence + Phase Close-Out Summary

**SC4 concluded over the complete phase change (run 38441258701 on headSha 4863883): every phase-16 surface green on the Linux truth chain — live-pybedtools parity executed green on CI, CRE band re-executed green, NER notebook green end-to-end, both root suites 19+4 green in the full census, D-03 gate held byte-identical — with the owner-accepted rice.uga.edu caveat on two input-download lanes; 5/5 SC matrix + clean phase-wide honesty ledger close the phase.**

## Performance

- **Duration:** 70 min active executor time (2026-10-13T17:40Z window: dispatch + local proofs + PENDING close-out; 2026-10-14T09:30Z resume: terminal record + close-out) — 16h wall dominated by the external CI queue (~6h foreign work; runner cancel denied) + the nightly run itself
- **Started:** 2026-10-13T17:40:34Z
- **Completed:** 2026-10-14T09:27:42Z (run terminal) / ~09:45Z (close-out)
- **Tasks:** 2
- **Files created:** 2 (16-EVIDENCE.md, async-jobs manifest)

## Accomplishments
- SC4 dispatched per the session's established pattern (gh workflow run on ref v1.3) with pre-flight proof the ref carried the COMPLETE change (headSha equality vs local HEAD); both nightlies + test-mamba executed overnight on the single self-hosted runner
- Terminal record (16-EVIDENCE.md A.4): example-nightly — census hard gate HELD `216/217 tests collected (1 deselected)` byte-identical, CRE band re-executed GREEN (254.961s), NER notebook GREEN end-to-end on POSIX (4023.932s), 5 skips exactly the historical allowlisted set, sole red = rice.uga.edu fixture-seeding hang BEFORE dispatch code ran; coverage-nightly — 2607 passed incl. ALL 19 dispatch-direction tests (live-pybedtools parity EXECUTED green on CI) and ALL 4 frozen-pair band tests; 2 reds = the same rice class (CellTimeout 3600s + pytest-timeout); test-mamba success
- 5/5 SC proof matrix: every ROADMAP criterion mapped to a green re-runnable command (SC4 green-where-executable with the recorded caveat)
- Phase-wide honesty ledger clean: census triple byte-identical to phase entry (ff95ef7..HEAD diff over ci.yml EMPTY), expected_skips.yaml/pyproject.toml/uv.lock untouched, docs sync exit 0, fast lane 2550 passed, footprint = exactly the plan's enumeration (23 non-planning files)
- Assumption dispositions: 16-03 AS-1..AS-5 all confirmed; 16-04 AS-1 confirmed-active, AS-2 confirmed-with-adjustment

## Task Commits

1. **Task 1 (dispatch + PENDING close-out, 2026-10-13): SC4 dispatch record + queue forensics; matrix 4/5 green** - `1cc203b` (docs; includes async-jobs manifest + WINDOWS #24 arm) with `25f90e3` (STATE session/blocker)
2. **Task 1+2 (terminal reconciliation, 2026-10-14): SC4 terminal record + matrix 5/5** - `09296c5` (docs; WINDOWS #24 resolved)

**Plan metadata:** (docs commit follows this SUMMARY)

Note: this plan is evidence-only (no production code); both tasks' deliverable is the evidence file, committed in docs commits. The ledger range 4863883..09296c5 also contains `30671e7` (phase-14 H.6 record by another session — foreign to this plan, noted for ledger honesty).

## Files Created/Modified
- `.planning/phases/16-windows-parity-rollout-dispatch-gates-frozen-value-parity/16-EVIDENCE.md` - NEW: the phase evidence file (14-EVIDENCE precedent) — section A dispatch/queue/terminal record, B re-check commands, C the 5/5 matrix, D honesty ledger, E assumption dispositions
- `.planning/async-jobs/16-04-sc4-nightly.json` - NEW: the external_job_waiting manifest for the overnight run; terminal-resolved

## Decisions Made
- SC4 disposition = green-where-executable + owner-accepted rice.uga.edu caveat (4th/5th manifestations of the phase-14 H.4/H.4.final class; both failures upstream of phase code; re-dispatch superseded by owner directive — the routing's escalation endpoint)
- Foreign-run cancellation not attempted past the classifier denial; PENDING/manifest close-out used instead (the legal external_job_waiting state)
- The rice.uga.edu fragility remains an upstream external dependency (no cache/timeout resilience in the NER input path) — carried in STATE as resolved-context, not a repo blocker

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Queue blockage -> PENDING close-out via async-jobs manifest**
- **Found during:** Task 1 (pre-flight)
- **Issue:** the single runner held an in-flight foreign job + two queued scheduled main-branch runs (~6h) ahead of the dispatch; cancelling foreign runs was denied by the permission classifier
- **Fix:** dispatched anyway, committed the external_job_waiting manifest + partial evidence, returned PENDING (14-02 precedent); the coordinator-triggered resume reconciled the terminal run
- **Files modified:** .planning/async-jobs/16-04-sc4-nightly.json
- **Verification:** manifest schema per phase-13 r5 precedent; resume contract executed verbatim on resume
- **Committed in:** 1cc203b

**2. [Rule 1 - Verify-defect] Task 1 fails_when disjunction incomplete for the rice class**
- **Found during:** Task 1 verify (2026-10-14)
- **Issue:** the verify's fails_when names only "phase regression OR recorded inaccurately" for a non-success live conclusion — the actual case was the third branch the plan's own action text routes separately (known external transient -> escalation/acceptance), leaving the verify with no honest green path
- **Fix:** ran the verify commands and recorded their true outcomes (census greps pass; conclusion-success grep and live check fail BY THE FACTS — not gamed); the triage disposition (A.5) governs, exactly as the action text's "records the failure and the resolution" contract specifies
- **Files modified:** none (record-keeping correction in 16-EVIDENCE.md A.5)
- **Committed in:** 09296c5

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 verify-snippet defect)
**Impact on plan:** Both were forced by external state (queue ownership; upstream host). No scope creep; no fabricated evidence anywhere.

## Issues Encountered
- The coverage-nightly runner-workspace junit was cleaned by the next queued job before capture — the GitHub job log was extracted instead (`/tmp/p16-sc4-cov-log.txt`, re-fetchable via `gh run view --job 115448126935 --log`) and its lines are the durable record of the 2607-passed census and both failure signatures.
- The example-nightly stage-1 log carries no per-test band-assertion internals (`-q` posture); the junit PASS rows (band tests green at full runtime) are the machine-readable band evidence, with the frozen-pair band tests supplying the parsed-band assertions in the coverage census.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 16 complete: all five SCs evidenced (SC4 with the recorded caveat); WPAR-06/WPAR-08 marked complete — REQUIREMENTS.md traceability closed
- The honesty ledger is clean (census/skips/deps/lock untouched across the whole phase; zero Co-Authored-By trailers) — Phase 17 (Publish Chain Robustness) starts from an honest-green baseline with the census triple still 216/217
- Standing external dependency for future planning: rice.uga.edu genome downloads (NER input path) remain fragile with no cache/timeout resilience — any phase touching the NER example's CI lanes inherits this caveat

---
*Phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity*
*Completed: 2026-10-14*

## Self-Check: PASSED

- All 3 key-files exist on disk (16-EVIDENCE.md, 16-04-SUMMARY.md, async-jobs manifest) — FOUND each
- All 3 plan commits (1cc203b, 25f90e3, 09296c5) are ancestors of HEAD on this branch; `check evaluation-scope --plan 16-04 --commits-only` resolved
- Task 2 verify chain re-run at close-out: 23 passed (both root suites), check_docs_sync exit 0, census triple grep = `216/217 tests collected (1 deselected)`, GUARD diff ff95ef7..HEAD over expected_skips.yaml/pyproject.toml/ci.yml EMPTY, SC-section grep 21 (>= 5); full fast lane green earlier this session (2550 passed, exit 0)
- Task 1 verify run with true outcomes recorded: 216/217 greps pass; the conclusion-success grep and the live `gh run view 38441258701` check return the failure facts honestly — the rice-class triage disposition (16-EVIDENCE.md A.5, owner-directed) governs per the plan's own action-text contract
