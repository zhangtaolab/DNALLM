---
phase: 14-gap-1-transformers-5-x-disposition
plan: 01
subsystem: testing
tags: [transformers-5x, trust_remote_code, gap-1, evidence-recording, modelscope, pytest-smoke, census-gate, owner-checkpoint]

requires:
  - phase: 05-remote-code-compat-ladder
    provides: the GAP-1 smoke test (test_nt_v2_promoter_loads_and_forwards) and the marker/tripwire machinery this plan executed as its evidence run
  - phase: 13-ollama-surface-disposition
    provides: the 216/217 census triple under the -m "not giants"-only selector, the nightly junit evidence shape, and run 38308972028 as the green example-nightly of record
provides:
  - The committed GAPC-01 record of decision — 14-EVIDENCE.md sections A-G (recorded promoter load run, BPE-leg probe, invert_attention_mask reachability proof, nightly junit forensics, census baseline, evidence table, owner decision)
  - The measured ladder sizing `new rungs required: 0` from real execution (D-01 evidence-sizing satisfied; no defensive rung written)
  - The owner's shim-zero-rung disposition decision on record (D-04 blocking-human checkpoint satisfied; plan 14-02 unblocked with WINDOWS #11/#12/#15 closure semantics locked per D-08)
affects: [14-02 shim-path disposition execution, WINDOWS #11/#12/#15 closure, benchmark third-model lane, BPE pkl script lane]

actuals:
  tokens: 4944        # chars/4 over the realized 14-EVIDENCE.md diff (+254 lines, 19779 chars); plan estimate was 15000 (estimate covered probe harness + evidence; probes live in /tmp by design)
  tasks: 3
  commits: 3          # MEASURED: git rev-list --count c6710a0..216fbad

plan_head_before: c6710a014f99e4994d02962e4323c2ae5ad28a82
plan_head_after: 216fbade609d52d24357c27bd37d3ef071ac9cc2

tech-stack:
  added: []            # evidence-recording plan; zero new dependencies, zero rung code, zero test edits
  patterns:
    - "Evidence-on-record for disposition gates: the smoke test IS the load run — execute it with -m slow re-included, capture verbatim tee + junitxml, size the ladder only from the recorded execution (D-01)"
    - "Run-status vs job-status reading for cancelled CI runs: co-enqueued coverage-nightly cancellation does not falsify the example-nightly JOB's green junit (provenance re-downloaded byte-identical)"
    - "Census idempotency proven with the stage-0.5 gate's own sed normalization, never raw-line comparison"

key-files:
  created:
    - .planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md
    - .planning/phases/14-gap-1-transformers-5-x-disposition/14-01-SUMMARY.md
  modified: []

key-decisions:
  - "Ladder measured at ZERO new rungs on the recorded run — both NT legs load and forward green on transformers 5.17.0 under the existing ladder; invert_attention_mask is a dead branch (attribute inventory, not execution trace), so no rung is written (D-01/D-02)"
  - "Owner disposition decision shim-zero-rung taken exactly once on the section F evidence table via the D-04 blocking-human checkpoint; consequence: proceed to 14-02 with WINDOWS #11/#12/#15 closing fixed-with-probe-evidence (D-08)"
  - "Nightly evidence read at the JOB level: run 38308972028 is run-level cancelled (power-outage recovery co-enqueued cancellation) but its example-nightly job concluded success; artifact re-downloaded fresh and byte-identical to the research copy before citation"

patterns-established:
  - "Phase-evidence artifact as the decision-gating record: verbatim captured outputs (tee + junit + quoted guard block + sed-normalized census) under a fixed section order A-G, with machine-checkable literals asserted by verify blocks (T-14-02 mitigation)"

requirements-completed: [GAPC-01]

coverage:
  - id: D1
    description: "The GAPC-01 empirical promoter load run on record: real load + real (1, 2)-logits forward through load_model_and_tokenizer source=modelscope, captured verbatim with junit, sizing the ladder at new rungs required: 0"
    requirement: GAPC-01
    verification:
      - kind: integration
        ref: "pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards -m slow -q -> 1 passed in 6.92s (recorded run) + verify re-run SMOKE-GREEN"
        status: pass
      - kind: other
        ref: "/tmp/gapc14-smoke-junit.xml: tests=1 failures=0 errors=0 skipped=0, single passing testcase for the smoke id (recorded in section A)"
        status: pass
    human_judgment: false
  - id: D2
    description: "BPE-leg load probe on record: zhangtaolab/plant-nucleotide-transformer-BPE with task_type=token (num_labels 7 per ner_task_config.yaml) resolving remote EsmForTokenClassification and forwarding (1, 17, 7)"
    requirement: GAPC-01
    verification:
      - kind: integration
        ref: ".venv/bin/python /tmp/gapc14-bpe-probe.py -> EsmForTokenClassification, forward logits (1, 17, 7), seq_len 17 (verify grep count 1)"
        status: pass
    human_judgment: false
  - id: D3
    description: "invert_attention_mask reachability proof on record: the only call site verbatim-quoted from the cached remote modeling_esm.py, config.json KEYS-ABSENT check, three-step unreachable-branch conclusion"
    requirement: GAPC-01
    verification:
      - kind: other
        ref: "config.json key check -> CONFIG-CLEAN (KEYS-ABSENT); guard conditional 'if self.config.is_decoder and encoder_hidden_states is not None:' present verbatim in section C (EVIDENCE-BF-COMPLETE)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Nightly junit forensics on record: example-nightly JOB success with run-level cancellation context, artifact re-downloaded byte-identical, both PASS rows (BPE script 102.244s, benchmark notebook 48.714s), all 5 skips classified with the explicit zero-#15-class statement"
    requirement: GAPC-01
    verification:
      - kind: other
        ref: "gh api run 38308972028 jobs -> example-nightly success / coverage-nightly cancelled; gh run download -n example-nightly-junit -> 12/12 files cmp-identical to research copy; junit tests=216 skipped=5 failures=0 errors=0"
        status: pass
    human_judgment: false
  - id: D5
    description: "Census baseline on record and idempotent: two identical sed-normalized collect lines reading 216/217 tests collected (1 deselected) — the pre-edit baseline 14-02 proves against"
    requirement: GAPC-01
    verification:
      - kind: other
        ref: "collect-only -m 'not giants' twice with the stage-0.5 sed normalization -> IDEMPOTENT: 216/217 tests collected (1 deselected)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The owner disposition decision recorded: shim-zero-rung selected via the D-04 blocking-human checkpoint (one question, two options, evidence table rendered); section G carries the option id verbatim, timestamp, question, and consequence line"
    verification:
      - kind: other
        ref: "grep gates: exactly one '## G. Decision record' heading, option id shim-zero-rung present -> DECISION-RECORDED"
        status: pass
    human_judgment: false

duration: 14 min
completed: 2026-10-13
status: complete
---

# Phase 14 Plan 01: GAPC-01 Evidence + Owner Disposition Checkpoint Summary

**Both NT legs load and forward green on transformers 5.17.0 under the existing rung ladder (promoter smoke 1 passed with (1,2) logits; BPE-leg EsmForTokenClassification (1,17,7)); invert_attention_mask proven a dead branch; ladder measured at zero new rungs and the owner selected shim-zero-rung on the recorded section F evidence table.**

## Performance

- **Duration:** 14 min (2026-10-13T02:02:00Z → 02:16:26Z, including the blocking-human checkpoint round-trip)
- **Tasks:** 3/3
- **Files created:** 2 (14-EVIDENCE.md, this SUMMARY); 0 production files modified

## Accomplishments

- The GAPC-01 empirical load run is on record: the smoke `test_nt_v2_promoter_loads_and_forwards` executed with `-m slow` re-included (the default fast-lane markers deselect it) — real load plus real forward through `load_model_and_tokenizer` with `source="modelscope"`, `1 passed in 6.92s`, junit `tests="1" failures="0" errors="0" skipped="0"`, sizing line `new rungs required: 0` measured from the execution (no traceback; D-01)
- The BPE-leg probe ran green with the exact script config (task_type `token`, num_labels 7): remote `EsmForTokenClassification`, tokenizer roundtrip, forward logits `(1, 17, 7)` — Pitfall 3 (wrong task_type → native EsmModel → rope_theta) avoided by construction
- The reachability proof closes the milestone probe's one known gap as an attribute inventory over a dead branch: only call site verbatim-quoted from the cached remote `modeling_esm.py:1029-1044`, config.json KEYS-ABSENT, three-step conclusion — no `invert_attention_mask` rung is justified (Anti-Pattern 1 avoided)
- Nightly forensics recorded with provenance re-verified live: run 38308972028 re-queried via gh (example-nightly JOB success; run-level cancelled is the co-enqueued coverage-nightly cancellation from the 2026-10-12 power-outage recovery) and the junit artifact re-downloaded fresh — byte-identical, 12/12 files, to the research copy; both PASS rows and all 5 skips classified with the explicit zero-#15-class statement (D-08 per-item evidence)
- The census baseline reproduced idempotently at `216/217 tests collected (1 deselected)` under the exact stage-0.5 flags with the gate's own sed normalization — the pre-edit baseline plan 14-02 proves against
- The owner decision landed exactly once via the D-04 blocking-human checkpoint: **shim-zero-rung** — plan 14-02 unblocked, WINDOWS #11/#12/#15 closure semantics locked as fixed-with-probe-evidence

## Task Commits

1. **Task 1 (tracer): Record the GAPC-01 empirical load run end-to-end** — `726489b` (docs)
2. **Task 2: Complete the evidence table — BPE-leg probe, reachability proof, nightly junit forensics, census baseline** — `ef7a5ab` (docs)
3. **Task 3: Owner disposition checkpoint — section G decision record** — `216fbad` (docs)

Tracer feedback gate (Task 1, row 3 — interactive/end-of-phase, automated-only verify): verify re-run end-to-end green (SMOKE-GREEN, EVIDENCE-A-RECORDED, junit suite count 1) — expanded to Tasks 2-3 without a checkpoint.

**Plan metadata:** this commit (docs: complete plan).

## Files Created/Modified

- `.planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md` — the GAPC-01 record of decision, sections A-G: recorded promoter load run (command, verbatim 15-line tail, junit header, versions 5.17.0/2.11.0+cu130, measured sizing line), BPE-leg probe (verbatim output incl. benign warnings), reachability proof (verbatim guard block + key-absence check + conclusion), nightly junit forensics (job-vs-run status, census line, both PASS rows, 5-skip table, zero-#15 statement), census baseline (idempotent), evidence table + both options + D-02 threshold + models.lock pins, decision record (shim-zero-rung)
- Ephemeral (not committed, /tmp only, as planned): `/tmp/gapc14-smoke-junit.xml`, `/tmp/gapc14-smoke-run.log`, `/tmp/gapc14-bpe-probe.py`, `/tmp/gapc14-bpe-probe.log`, `/tmp/gapc14-nightly-junit/`

## Decisions Made

- All decisions were locked in 14-CONTEXT.md (D-01..D-08) and followed exactly; the single execution-time decision was the owner's own: shim-zero-rung on the recorded table (D-04), recorded in section G with consequence "proceed to 14-02"

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The green path was taken on both recorded runs (no LADDER-UNCLOSED / BPE-LEG-UNCLOSED else-branch fired); auto mode was inactive so the blocking-human decision checkpoint correctly stopped for the owner rather than auto-selecting.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None — the evidence file carries no placeholders (all seven sections A-G filled; verified by grep). The only deferred human check is the CI-side re-verification, which is plan 14-02 Task 3's own scope (flagged assumption A1/A2 carried by design, not a stub).

## Next Phase Readiness

- Plan 14-02 (shim-path disposition execution) is unblocked exactly as planned: comment-only prose resolution in the two test files, local gates (fast lane, benchmark 35-family, census triple hold), WINDOWS #11/#12/#15 fixed with the recorded probe evidence per D-08, expected_skips.yaml untouched, CI coverage-gate leg re-verification
- The evidence base 14-02 cites is committed at `14-EVIDENCE.md` (sections A-G); `dnallm/utils/transformers_compat.py` stays read-only (D-05), and no test file was touched by this plan (prohibition held)

---
*Phase: 14-gap-1-transformers-5-x-disposition*
*Completed: 2026-10-13*

## Self-Check: PASSED

- Both created files exist on disk (`14-EVIDENCE.md`, this SUMMARY)
- All three task commits (726489b, ef7a5ab, 216fbad) are ancestors of HEAD, measured 3 via `git rev-list --count c6710a0..HEAD` from the persisted ledger
- Evidence file is placeholder-free (grep pending = 0) and carries the artifact must-have literal `new rungs required: 0`
- No test file or compat-ladder file was modified by this plan (prohibitions verified against git status)
