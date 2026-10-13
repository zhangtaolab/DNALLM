---
phase: 14-gap-1-transformers-5-x-disposition
plan: 02
subsystem: testing
tags: [transformers-5x, gap-1, comment-only-edits, pruned-ast-gate, windows-ledger, census-gate, ci-nightly, pending-ci]

requires:
  - phase: 14-gap-1-transformers-5-x-disposition
    provides: the section A-G evidence record and the owner shim-zero-rung decision (section G) this plan executed
  - phase: 13-ollama-surface-disposition
    provides: the comment-only-edit discipline with mechanical diff gates, the 216/217 census triple, and the #22 dispatch-discipline precedent
provides:
  - The shim-zero-rung disposition executed end-to-end — resolved-state GAP-1 prose in both test files with the tripwire machinery byte-identical, WINDOWS #11/#12/#15 closed fixed on the recorded evidence, GAPC-01/GAPC-02 complete
  - The D-06 local regression proof set on record (fast lane 2487 passed, 35-family gate 35 passed, skip audit exit 0) plus the pruned-AST comment-only gate pattern
  - The CI-side re-verification first-pass record: models-lane smoke PASSING for real on CI's freshly-resolved transformers (drift refuted) with the sole red forensically classified as the rice.uga.edu fasta-truncation transient, re-dispatched once per the #22 discipline (PENDING, WINDOWS #23)
affects: [phase 14 close (blocked on run 38354875692 coverage-nightly green), WINDOWS #23 resolution, benchmark third-model lane, BPE pkl script lane]

actuals:
  tokens: 6486        # chars/4 over the realized diff 27831a2..c081a52 (5 files, +138/-40); plan estimate was 16000 (confidence low — overestimated; the comment rewrites were smaller than the planner's allowance and the ladder stayed at zero by design)
  tasks: 3
  commits: 2          # MEASURED: git rev-list --count 27831a2..c081a52

plan_head_before: 27831a204a316ea54a67e26ff8a714eb395789f5
plan_head_after: c081a52f13363140021b98e50114e1a9d2aedb87

tech-stack:
  added: []            # zero new dependencies, zero rung code, zero functional edits
  patterns:
    - "Pruned-AST equality gate (/tmp/gapc14-ast-gate.py): HEAD vs working tree with all docstrings pruned — the mechanical proof that a prose-only edit changed no statement or expression (T-14-03)"
    - "Transient-infra forensics for CI triage: junit row extraction + runner-sandbox input inspection (truncated fasta sizes, chromosome census) before classifying a nightly red — never mask, never assume code"
    - "PENDING escape with ledger tracking: an in-flight CI gate becomes WINDOWS unrun-verify #23 so the phase-close block survives the executor's window (#22 discipline)"

key-files:
  created:
    - .planning/phases/14-gap-1-transformers-5-x-disposition/14-02-SUMMARY.md
  modified:
    - tests/models/test_model_remote_code.py
    - tests/examples/test_script_execution.py
    - .planning/WINDOWS.md
    - .planning/REQUIREMENTS.md
    - .planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md

key-decisions:
  - "Comment-only discipline proven mechanically, not narratively: pruned-AST equality + marker-literal byte-identity + census-pin equality + matcher-set identity all inside the same atomic commit as the prose resolution (D-05/D-07 vacuous)"
  - "First CI red triaged on evidence, not assumption: the smoke row PASSED on CI's own transformers (Pitfall-4 drift refuted) and the sole failure was forensically proven a rice.uga.edu truncated-download transient (22.1MB of ~115MB fasta, Chr1-Chr2 only) — recorded, re-dispatched once, phase close blocked (never papered over)"
  - "The in-flight re-dispatch (run 38354875692) is tracked as WINDOWS #23 unrun-verify so the phase-close gate outlives the executor window — the plan's PENDING escape executed exactly as designed"

patterns-established:
  - "Pruned-AST gate as the canonical proof for comment-only test edits (stronger than diff inspection: it names the first differing statement on failure)"
  - "CI-nightly junit capture from the runner workspace between job boundaries when the job uploads no artifact (the next serialized job's checkout cleans it — capture on the write event)"

requirements-completed: [GAPC-02]

coverage:
  - id: D1
    description: "Resolved-state GAP-1 prose in the two test files with zero functional edits: module docstring + marker comment in test_model_remote_code.py, GAP-1 comment block in test_script_execution.py — all citing the recorded evidence and naming the closed-map rung"
    requirement: GAPC-02
    verification:
      - kind: other
        ref: ".venv/bin/python /tmp/gapc14-ast-gate.py -> AST-IDENTICAL both files, AST-GATE: PASSED"
        status: pass
      - kind: unit
        ref: "grep marker literals -> MARKERS-INTACT (each exactly once, byte-identical); census collect -> CENSUS-MATCH 216/217 (1 deselected); matcher diff -> MATCHERS-UNCHANGED"
        status: pass
    human_judgment: false
  - id: D2
    description: "D-06 local regression proofs green and recorded in 14-EVIDENCE.md section H (fast lane, 35-family gate, skip audit) with WINDOWS #11/#12/#15 closed fixed on the per-item evidence and GAPC-01/GAPC-02 complete"
    requirement: GAPC-02
    verification:
      - kind: integration
        ref: "pytest -m 'not slow' -> 2487 passed, 2 skipped, exit 0 (FASTLANE-GREEN double-run); pytest tests/benchmark/test_benchmark.py -> 35 passed (BENCHMARK-35)"
        status: pass
      - kind: other
        ref: "scripts/audit_skips.py over fast-lane junit -> exit 0, OK: every skip matches; windows status -> fixed_count 15 / open_count 4 (LEDGER-CLOSED); REQUIREMENTS greps -> REQS-COMPLETE"
        status: pass
    human_judgment: false
  - id: D3
    description: "CI-side re-verification on the pushed ref: models-lane smoke passing for real on CI's freshly-resolved transformers; first run's sole red forensically classified transient-infra (rice fasta truncation) and re-dispatched once; final green coverage-nightly pending"
    requirement: GAPC-02
    verification:
      - kind: other
        ref: "run 38337783698 junit: smoke testcase test_nt_v2_promoter_loads_and_forwards time=3.400 PASS, no failure child, on headSha 2c737516; suite 2587 tests / 1 failure (script lane, KeyError Chr6 — truncated 22.1MB of 115MB fasta)"
        status: pass
      - kind: other
        ref: "re-dispatch run 38354875692 coverage-nightly conclusion — PENDING (WINDOWS #23); phase close blocked until green per plan Task 3 step 6 / verification item 6"
        status: unknown
    human_judgment: true
    rationale: "The final CI gate (re-dispatched coverage-nightly green) cannot land inside the executor's window — queued behind the first run's example-nightly plus its own ~3h census. The owner confirms the run reads success (or triages a further red) before the phase closes; the smoke's own PASS on CI transformers is already on record."

duration: 228 min
completed: 2026-10-13
status: complete
---

# Phase 14 Plan 02: Shim-Zero-Rung Disposition Execution Summary

**GAP-1 prose resolved comment-only in both test files (pruned-AST identical to HEAD, tripwires byte-identical, census/matchers unmoved), WINDOWS #11/#12/#15 closed fixed on the recorded probe evidence, GAPC-01/02 complete — and the CI re-verification's models-lane smoke passed for real on CI's own transformers while the sole nightly red was forensically proven a rice.uga.edu truncated-download transient, re-dispatched once with the phase close gated on its green (PENDING).**

## Performance

- **Duration:** 228 min (2026-10-13T02:22:06Z → 06:10:00Z; ~200 min of that is waiting on the self-hosted CI census)
- **Started:** 2026-10-13T02:22:06Z
- **Completed:** 2026-10-13T06:10:00Z
- **Tasks:** 3/3
- **Files modified:** 5 (2 test files, WINDOWS.md, REQUIREMENTS.md, 14-EVIDENCE.md)

## Accomplishments

- The stale GAP-1 prose now narrates the resolved state in both edit targets: `tests/models/test_model_remote_code.py` (module docstring rewritten — closed-map `_patch_pretrained_config_legacy_defaults` rung answers both names, checkpoint loads and forwards `(1, 2)` on transformers 5.17, WINDOWS #11 closed fixed; the marker comment restated as the dormant tripwire) and `tests/examples/test_script_execution.py` (GAP-1 comment block — same rung, the script's real PASS in example-nightly run 38308972028, the phase-14 shim-path-zero-rung disposition, the branch as dormant tripwire)
- Comment-only discipline proven mechanically in the same change: pruned-AST equality on both files (AST-IDENTICAL), marker literals byte-identical and exactly once (MARKERS-INTACT), measured census equal to the ci.yml pin `216/217 tests collected (1 deselected)` (CENSUS-MATCH), and the expected_skips.yaml matcher set identical to HEAD (MATCHERS-UNCHANGED, file untouched)
- D-06 local regression proofs green and recorded verbatim in 14-EVIDENCE.md section H: fast lane `2487 passed, 2 skipped, 75 deselected in 101.77s`, 35-family gate `35 passed in 7.86s`, skip audit exit 0 with both fast-lane skips allowlisted; the D-07 vacuous-satisfaction statement (no un-skip, no removal, census unchanged, matchers identical) rides the same section
- WINDOWS #11/#12/#15 closed fixed via gsd-tools (resolved_at 02:29:17Z; counters recomputed to fixed 15 / open 4) with 14-EVIDENCE.md sections A-D as the reason of record — #15 on the per-item nightly classification, never blanket (D-08, Pitfall 5); REQUIREMENTS GAPC-01/GAPC-02 checkboxes and traceability rows Complete
- CI re-verification executed on the pushed ref (run 38337783698, headSha 2c737516 = the atomic commit): the models-lane smoke `test_nt_v2_promoter_loads_and_forwards` PASSED for real (`time="3.400"`, no failure child, junit captured from the runner workspace) on CI's freshly-resolved transformers — the Pitfall-4 drift surface refuted; the census's sole failure (BPE script lane, `KeyError: 'Chr6 does not exist in fasta file'`) was forensically classified as the documented rice.uga.edu truncated-download transient (sandbox fasta 22,134,784 bytes of ~115MB carrying only Chr1-Chr2 against a complete GFF3; no cache poisoning), recorded in section H.4 and re-dispatched once per the #22 discipline (run 38354875692); test-mamba came back green (success)

## Task Commits

1. **Task 1 + Task 2 (atomic disposition change set)** — `2c73751` (docs): prose resolution in both test files + 14-EVIDENCE.md section H.1-H.3 (local gates, D-07 statement, closure records) + WINDOWS #11/#12/#15 fixed + REQUIREMENTS GAPC complete — one commit so every same-change proof refers to one diff
2. **Task 3 (CI re-verification record, first pass)** — `c081a52` (docs): section H.4 — dispatch provenance, suite header, smoke PASS row, the truncation forensics, triage, re-dispatch + PENDING marker

**Plan metadata:** this commit (docs: complete plan; carries the H.4 first-run-outcomes amendment and WINDOWS #23).

## Files Created/Modified

- `tests/models/test_model_remote_code.py` — module docstring (lines 1-18) and the marker comment block (29-32) rewritten to the resolved state; line 33 marker and the 68-76 except-branch byte-identical
- `tests/examples/test_script_execution.py` — GAP-1 comment block (35-45) rewritten to the resolved state; line 46 marker and the ~186 branch byte-identical
- `.planning/WINDOWS.md` — entries 11/12/15 status fixed (gsd-tools; resolved_at 2026-10-13T02:29:17Z) + new entry #23 (unrun-verify, the pending CI gate)
- `.planning/REQUIREMENTS.md` — GAPC-01/GAPC-02 `[x]` + traceability rows Complete
- `.planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md` — section H appended: H.1 local gates (verbatim outputs), H.2 D-07 vacuous-satisfaction statement, H.3 ledger closures, H.4 CI re-verification (first run + triage + forensics + re-dispatch PENDING + first-run co-enqueued outcomes)
- Ephemeral (not committed, /tmp only, as planned): `/tmp/gapc14-ast-gate.py`, `/tmp/gapc14-head-skips.yaml`, `/tmp/gapc14-fastlane-junit.xml`, `/tmp/gapc14-cov-nightly-junit.xml` (the captured nightly junit, 377,942 bytes)

## Notebook-file zero-diff inspection (verbatim, plan Task 1 step 3)

```
$ grep -n "GAP-1" tests/examples/test_notebook_execution.py
(1 exit status 1 — no matches)
$ grep -n "is_decoder" tests/examples/test_notebook_execution.py
(1 exit status 1 — no matches)
```

Zero pending-state narration hits — the tree's only GAP-1 prose lived in the two edited files (also verified tree-wide: `grep -rln "GAP-1" tests/` → exactly the two edit targets). Inspected-zero-diff, no edit required.

## Decisions Made

- Executed exactly the recorded section G disposition (shim-zero-rung); `dnallm/utils/transformers_compat.py` untouched (D-05 satisfied by the already-landed rungs — verified by the untouched diff)
- The ledger closures went through gsd-tools with the evidence file as the reason of record (the CLI takes no reason flag); #15 closed only after re-confirming section D's per-item classification with its explicit zero-#15-class line
- First CI red handled by the plan's recorded triage, not improvisation: smoke-first triage (PASS → transformers drift refuted), forensic classification of the script-lane failure (input truncation, not code, not GAP-1), one re-dispatch, phase close blocked (PENDING) rather than masked

## Deviations from Plan

None - plan executed exactly as written, including Task 3's designed triage and PENDING paths.

## Issues Encountered

- **CI coverage-nightly first-pass red on the BPE script lane (transient infra, resolved by re-dispatch):** run 38337783698's census failed exactly one test — `test_generate_bpe_dataset_produces_artifact` — at `generate_bpe_dataset.py:159` with `KeyError: 'Chr6 does not exist in fasta file'`. Forensics on the runner sandbox proved the rice fasta download truncated at 22.1MB of ~115MB (only Chr1-Chr2 present) against a complete GFF3 referencing Chr6 — the documented rice.uga.edu flaky-host class; a truncated 200-body ends cleanly at EOF so the download-path typed skip cannot absorb this variant. The GAP-1 surface itself succeeded in the same run (model load precedes tokenization, and the models-lane smoke passed 3.400s). Recorded in 14-EVIDENCE.md H.4 with the full traceback, classified, re-dispatched once (run 38354875692). **Not falsifying the disposition evidence** (sections B/D stand; this run's own model load succeeded).
- **The re-dispatch's coverage-nightly cannot conclude inside this executor's window** (it queues behind the first run's example-nightly and then runs a ~3h census): handled by the plan's PENDING escape — recorded in section H.4 and tracked as WINDOWS #23 (unrun-verify). The phase must not close before it reads success; the plan's verification item 6 (owner human-check) is the closing step.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None — no stubs were written. The single deferred item is the pending CI gate, which is not a stub but a recorded PENDING (WINDOWS #23) with its resolution path owned by the verification human-check.

## Next Phase Readiness

- **Phase 14 close is BLOCKED on run 38354875692's coverage-nightly reading success** (WINDOWS #23): when it lands green, resolve #23 (`node ~/.claude/gsd-core/bin/gsd-tools.cjs windows fixed 23`), append the run's job conclusions + smoke junit row to section H.4, and the phase's verification human-check (plan item 6) is satisfied. If it lands red on the rice class again, triage per section H.4's forensics — the one sanctioned re-dispatch is spent, so a further red escalates to the owner rather than another dispatch. The first run's example-nightly outcome (in flight at window close) also records there.
- Everything else in the phase is done and green: ladder untouched at zero new rungs, prose resolved, ledger and requirements closed, local proofs on record; the next milestone phases (15+) do not depend on the pending CI gate

---

*Phase: 14-gap-1-transformers-5-x-disposition*
*Completed: 2026-10-13*
