---
phase: 14-gap-1-transformers-5-x-disposition
verified: 2026-10-13T13:38:30Z
status: passed
score: 12/12 must-haves verified
covered_files:
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-01-PLAN.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-01-SUMMARY.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-02-PLAN.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-02-SUMMARY.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md
  - tests/examples/test_script_execution.py
  - tests/models/test_model_remote_code.py
covered_digest: "v3:sha256:ef848b555685bc3195d19c7250456df86f1776dcbeebd9c1162f16a8d1eaa48c"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "The CI-side re-verification lane (D-06) runs: the dispatched CI workflow on ref v1.3 reaches coverage-nightly job success — executing the models-lane smoke on CI's freshly-resolved transformers (the Pitfall-4 drift surface) under the full census with coverage and the skip audit — and the run id, job conclusions, and the smoke's junit row are recorded in 14-EVIDENCE.md section H"
    reason: "Coverage-nightly job success was never reached: both dispatched runs (38337783698, 38354875692, both headSha 2c737516 on ref v1.3) concluded failure solely on rice.uga.edu external-host transients (truncated FASTA -> KeyError Chr6/Chr12; genome-download wget hang -> CellTimeoutError 3600s), never on the GAP-1/transformers surface — the models-lane smoke PASSED for real on CI's freshly-resolved transformers in both runs (3.400s first run; also green in the re-dispatch census per H.4.final), refuting the drift risk the gate existed to catch. No code change can make the gate green (external upstream). The owner resolved the recorded PENDING by explicit accept-and-close at the escalation point of the one-re-dispatch contract (14-EVIDENCE.md H.4.final; WINDOWS #23 closed fixed-with-caveat via gsd-tools)."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-13T13:18:38Z"
re_verification:
  previous_status: gaps_found
  previous_score: 11/12
  gaps_closed:
    - "The PENDING-gate resolution record is complete and accurate — run 38337783698's example-nightly terminal outcome now recorded verbatim and the premature H.4.final test-mamba claim corrected (14-EVIDENCE.md section H.5, commit 1f50391, 2026-10-13T13:36:38Z)"
  gaps_remaining: []
  regressions: []
---

# Phase 14: GAP-1 Transformers 5.x Disposition Verification Report

**Phase Goal:** WINDOWS #11/#12 leave the owner-pending state — the NT promoter checkpoint either loads for real on transformers 5.17 through an evidence-sized absence-gated shim ladder, or the third benchmark model is honestly removed by recorded decision
**Verified:** 2026-10-13T13:38:30Z
**Status:** passed
**Re-verification:** Yes — after gap closure (commit 1f50391, section H.5)

## Goal Achievement

The phase goal is achieved on the codebase evidence, independently re-verified by this verifier (not taken from SUMMARY): the NT promoter checkpoint loads and forwards for real on transformers 5.17.0 under the existing (pre-landed, absence-gated) rung ladder — re-proven by this verifier's own execution of the single named smoke test (`1 passed in 7.60s`, transformers 5.17.0 / torch 2.11.0+cu130) and by CI's own freshly-resolved transformers (smoke row PASS `time="3.400"`, benchmark notebook PASS `50.213s` in the dispatched run) — the ladder is evidence-sized at zero new rungs, WINDOWS #11/#12/#15 left the owner-pending state as fixed (ledger resolved_at 2026-10-13T02:29:17Z), the owner disposition decision is on record (shim-zero-rung, section G), and GAPC-01/GAPC-02 are complete. The single CI-gate literal (coverage-nightly job success) was not met due to a documented rice.uga.edu external-host transient and carries a recorded owner accept-and-close (override below). The one record-completeness gap found on the first pass was closed by commit 1f50391 (section H.5) and re-verified against the live GitHub record.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | An empirical load run of the NT promoter checkpoint on transformers 5.17 under the current shim set is on record, sizing the remaining rung ladder (SC1, GAPC-01) | ✓ VERIFIED | 14-EVIDENCE.md section A: verbatim command (`-m slow`), 15-line output tail (`1 passed in 6.92s`), versions 5.17.0/2.11.0+cu130, sizing line `new rungs required: 0`; /tmp/gapc14-smoke-junit.xml inspected by verifier: `tests="1" failures="0" errors="0" skipped="0"`, single passing testcase for the smoke id; **verifier re-ran the single named test: `1 passed in 7.60s`** |
| 2 | Ladder sized from the RECORDED run, never speculatively; rungs extend only on real tracebacks (D-01/D-02) | ✓ VERIFIED | Zero commits touch `dnallm/utils/transformers_compat.py` in the phase range c6710a0..HEAD (git log empty); no LADDER-UNCLOSED/BPE-LEG-UNCLOSED heading in 14-EVIDENCE.md; both recorded runs green |
| 3 | invert_attention_mask reachability analysis on record (dead-branch proof justifying zero rungs) | ✓ VERIFIED | Section C quotes the guard verbatim; verifier located it in the cached remote file (modeling_esm.py:1031 conditional, :1040 the ONLY `invert_attention_mask` call site); config.json key-absence re-run by verifier: `KEYS-ABSENT` |
| 4 | BPE-leg probe on record with task_type exactly `token` (Pitfall 3) | ✓ VERIFIED | Section B verbatim output: `EsmForTokenClassification`, forward logits `(1, 17, 7)`; verifier inspected /tmp/gapc14-bpe-probe.py: `task_type="token"`, num_labels 7 |
| 5 | Example-nightly junit forensics on record with per-item skip classification and the explicit zero-#15-class statement (D-08, Pitfall 5) | ✓ VERIFIED | Section D: run 38308972028 job conclusions, stage-1 census line 211 passed/5 skipped/1 deselected, both PASS rows, 5-skip table, explicit "none of the 5 is a #15-class skip" |
| 6 | Owner decision taken exactly once on the evidence table before any execution (D-04), recorded in section G | ✓ VERIFIED | Section G: `shim-zero-rung` verbatim, timestamp 2026-10-13T02:15Z, question-as-presented, consequence line; commit 216fbad precedes all 14-02 execution commits; 14-02 executed exactly the selected option |
| 7 | Census baseline 216/217 (1 deselected) idempotent under exact stage-0.5 flags | ✓ VERIFIED | Section E (two sed-normalized identical lines); verifier re-ran collect twice: `IDEMPOTENT: 216/217 tests collected (1 deselected)`; equals ci.yml pin (lines 867-868) |
| 8 | Disposition executed comment-only: resolved prose, marker constants and skip branches byte-identical (SC2, D-05) | ✓ VERIFIED | Verifier's independent pruned-AST gate (27831a2 vs HEAD, docstrings pruned): `AST-IDENTICAL` both files; markers appear exactly once each, literals unchanged (grep -cF = 1/1); diff hunks are docstring/comment-only; notebook file inspected-zero-diff (`grep GAP-1/is_decoder` exit 1); tree-wide GAP-1 prose lives only in the two edited files. Note: SC2's literal "rung extension lands and NT tests un-skip" is satisfied vacuously — the rung pre-landed 2026-10-02 and the self-healing tests turned green on it; the owner-selected shim-zero-rung option explicitly contemplated zero new rungs |
| 9 | Non-NT families provably unaffected — ladder untouched and absence-gated, 35-family gate green (SC3) | ✓ VERIFIED | transformers_compat.py untouched in range; rung gate predicate verified absence-based (`_legacy_config_defaults_missing` probes fresh construction; never overwrites an existing `__getattr__`); verifier ran the 35-family gate: `35 passed in 7.18s` |
| 10 | Fast lane green with census triple and expected_skips.yaml reconciled in the SAME change (SC4, D-07) | ✓ VERIFIED | Atomic commit 2c73751 (02:35:08Z) carries prose edits + WINDOWS closures + REQUIREMENTS + evidence H.1-H.3; /tmp/gapc14-fastlane-junit.xml verified internally consistent: 2489 testcase elements, 0 failures/errors, 2 skipped = the recorded `2487 passed, 2 skipped`; verifier re-ran skip audit: exit 0 `OK: every skip ... matches the allowlist`; matcher set identical 27831a2→HEAD (MATCHERS-UNCHANGED); census triple unchanged (no un-skip, no removal — D-07 vacuous); tests/dnallm/scripts/example identical from 2c73751 to HEAD |
| 11 | CI-side re-verification lane (D-06): dispatched run reaches coverage-nightly job success, smoke green on CI's freshly-resolved transformers, recorded in section H | ✓ PASSED (override) | Dispatch verified via gh: runs 38337783698 and 38354875692 both on ref v1.3, headSha 2c737516 (the pushed atomic commit); smoke junit row PASS `time="3.400"` no failure child on CI's freshly-resolved transformers (drift refuted); benchmark.ipynb PASS 50.213s in the first run's example-nightly; BUT coverage-nightly concluded failure in both runs — solely rice.uga.edu external-host transients (truncated FASTA KeyError Chr6/Chr12; wget hang CellTimeoutError 3600s); owner accept-and-close on record (H.4.final; WINDOWS #23 resolved_at 2026-10-13T13:18:38Z = commit 818fc44) — see overrides |
| 12 | The PENDING-gate resolution record is complete and accurate (H.4's own commitment: first run's example-nightly outcome + re-dispatch's true job conclusions recorded by the resolver) | ✓ VERIFIED (gap closed) | **First pass FAILED (partial); closed by commit 1f50391 (13:36:38Z), section H.5.** H.5 records run 38337783698's example-nightly terminal outcome — matching this verifier's independent gh/junit forensics exactly: FAILURE at 09:41:57Z, exactly 2 rice.uga.edu-class reds (NER notebook CellTimeoutError 3600s on the wget genome-download cell; BPE script `KeyError: 'Chr12'` truncated FASTA at generate_bpe_dataset.py:159), GAP-1 surfaces green in that run (benchmark.ipynb PASS 50.213s; census held) — and corrects H.4.final's premature test-mamba claim with the true timestamped state (in_progress as of 13:36:14Z, started 12:53:53Z; example-nightly queued; watch armed to append terminal outcomes; disposition conclusion documented as independent of either job). Verifier re-checked the live GitHub record at 13:37:45Z: test-mamba in_progress, example-nightly queued — consistent with H.5's timestamped, non-terminal formulation |

**Score:** 12/12 truths verified (11 VERIFIED + 1 PASSED via owner-accepted override; 0 present-but-behavior-unverified)

### Re-Verification Detail (Gap Closure)

- **Previous status:** gaps_found (11/12) — one partial truth: the H.4 PENDING-resolution record omitted run 38337783698's example-nightly terminal outcome and asserted a re-dispatch test-mamba conclusion ("success") contradicted by the GitHub record at verification time.
- **Closure:** commit `1f50391` (2026-10-13T13:36:38Z) appends section H.5 "Record corrections" to 14-EVIDENCE.md. Both recorded facts were cross-checked by this verifier against the live gh API and the downloaded example-nightly junit artifact of run 38337783698 — they match exactly. The test-mamba entry is now honestly timestamped and non-terminal ("in_progress as of 13:36:14Z"), with terminal outcomes delegated to an armed watch — the phase's verbatim-evidence standard (T-14-02) is restored.
- **Regression check on passed items:** commit 1f50391 touches only 14-EVIDENCE.md (+5 lines); tests/dnallm/scripts/example remain identical from 2c73751 through HEAD; WINDOWS counters unchanged (16 fixed / 4 open); REQUIREMENTS GAPC rows unchanged (Complete). No regressions.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md` | GAPC-01/GAPC-02 record of decision, sections A-G + H (incl. H.5 corrections) | ✓ VERIFIED | 352 lines, all sections present in fixed order, `new rungs required: 0` present; verify.artifacts: passed (issues none) |
| `tests/models/test_model_remote_code.py` | resolved-state docstring + dormant-tripwire comment; marker byte-identical | ✓ VERIFIED | contains `_STRUCTURAL_MARKER` exactly once, literal unchanged; AST-identical to pre-edit |
| `tests/examples/test_script_execution.py` | resolved-state GAP-1 comment block; marker byte-identical | ✓ VERIFIED | contains `_NT_STRUCTURAL_MARKER` exactly once, literal unchanged; AST-identical to pre-edit |
| `.planning/WINDOWS.md` | entries 11, 12, 15 status fixed, counters recomputed | ✓ VERIFIED | #11/#12/#15 fixed resolved_at 02:29:17Z (x3, via gsd-tools — no hand-edit pattern); #23 added then fixed 13:18:38Z; counters 16 fixed / 4 open |
| `.planning/REQUIREMENTS.md` | GAPC-01/GAPC-02 checkboxes complete + traceability rows | ✓ VERIFIED | `[x] **GAPC-01**`, `[x] **GAPC-02**`; rows `\| GAPC-01 \| Phase 14 \| Complete \|`, `\| GAPC-02 \| Phase 14 \| Complete \|` |

All artifacts exist, are substantive, and are wired (Level 4 trace: every recorded evidence value traces to a machine-readable artifact — junit files inspected byte-level, gh API queried live).

### Key Link Verification

The gsd-tools key-link verb reports all 6 links "Source file not found" because the planner wrote prose `from:` fields instead of file paths — a plan-formatting artifact. Each link was verified manually:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| smoke test | 14-EVIDENCE.md section A | recorded GAPC-01 run = smoke with -m slow | ✓ WIRED | Section A carries the exact test id and command; junit testcase matches |
| section F evidence table | owner decision checkpoint (Task 3) | D-04 blocking checkpoint on the recorded table | ✓ WIRED | Section G records the selection taken on the F table, timestamped, before execution |
| cached remote modeling_esm.py guard | section C | verbatim quotation of the only call site + key check | ✓ WIRED | Verifier located the quoted block in the cached file; KEYS-ABSENT re-run green |
| test marker constant | transformers_compat `_patch_pretrained_config_legacy_defaults` | tripwire exists because the rung answers the detected names | ✓ WIRED | Marker literal present unchanged; rung present (transformers_compat.py:541, map :507) and applied in `apply_patches()` (:1241) |
| 14-EVIDENCE sections A-D | WINDOWS fixed entries 11/12/15 | closure evidence trail | ✓ WIRED | Ledger rows fixed with resolved_at; H.3 maps each id to its evidence sections |
| pushed v1.3 ref | dispatched coverage-nightly job | gh workflow run ci.yml --ref v1.3 | ✓ WIRED | gh confirms both dispatched runs on v1.3 at headSha 2c737516 |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| NT promoter checkpoint loads + forwards on transformers 5.17 (the phase goal's core claim) | single named test: `pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards -m slow -q --no-cov` | `1 passed in 7.60s` | ✓ PASS |
| 35-family benchmark gate (SC3) | `pytest tests/benchmark/test_benchmark.py -q --no-cov` | `35 passed in 7.18s` | ✓ PASS |
| Census idempotent + equals ci.yml pin (SC4) | collect-only x2 with stage-0.5 sed | `IDEMPOTENT: 216/217 tests collected (1 deselected)`; ci.yml:867 pins the same | ✓ PASS |
| config.json carries no is_decoder/add_cross_attention (reachability premise) | json key check | `KEYS-ABSENT` | ✓ PASS |
| Comment-only discipline (prohibition 1) | verifier's own pruned-AST gate, 27831a2 vs HEAD | `AST-IDENTICAL` both files | ✓ PASS |
| Skip audit over fast-lane junit | `scripts/audit_skips.py /tmp/gapc14-fastlane-junit.xml tests/expected_skips.yaml` | exit 0, `OK: every skip ... matches the allowlist` | ✓ PASS |
| Full fast lane green (SC4) — not re-run by verifier | recorded junit artifact inspected | 2489 cases / 0 failures / 2 skipped, internally consistent with the recorded tail `2487 passed, 2 skipped`; tests/dnallm identical from 2c73751 to HEAD (post-atomic commits are .planning-only) | ✓ PASS (artifact + no-drift) |
| Dispatched CI runs exist on the pushed ref | `gh run view 38337783698 / 38354875692` | both on v1.3, headSha 2c737516 | ✓ PASS |
| First run's example-nightly GAP-1 surfaces | gh junit artifact download | benchmark.ipynb PASS 50.213s; 2 failures both rice-class (NER CellTimeout on wget rice.uga.edu; BPE KeyError Chr12 truncated FASTA) — now recorded in H.5 | ✓ PASS |
| H.5 record accuracy (gap closure) | gh live state at 13:37:45Z vs H.5 text | test-mamba in_progress (started 12:53:53Z), example-nightly queued — consistent with H.5's timestamped non-terminal claims; 38337783698 example-nightly facts match verifier forensics exactly | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes are declared by this phase's plans. The phase's probe discipline was evidence-recording (junit capture + gh API), re-executed by this verifier where runnable — see Behavioral Spot-Checks.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GAPC-01 | 14-01 | Empirical load run on transformers 5.17 sizes the remaining rung ladder | ✓ SATISFIED | Truths 1-7; REQUIREMENTS row Complete |
| GAPC-02 | 14-02 | Shim-vs-removal executed on that evidence with ledger closure | ✓ SATISFIED (with the recorded owner caveat on the CI gate) | Truths 8-12; REQUIREMENTS row Complete |

Orphaned requirements: none — REQUIREMENTS maps exactly GAPC-01/GAPC-02 to Phase 14; both are claimed by plans. Decision coverage gate: skipped ("no trackable decisions" — the CONTEXT register uses phase-local D-01..D-08 IDs, not the D-XX trackable format; non-blocking by design).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| .planning/REQUIREMENTS.md | 78 | `cudaXXX` matches the XXX regex | ℹ️ Info | False positive — glob-style reference to cuda extras in the pre-existing Out-of-Scope table (defined 2026-10-11, not a phase-14 debt marker) |
| 14-EVIDENCE.md | 337-339 | `PENDING` mentions | ℹ️ Info | Recorded CI-gate state, resolved by H.4.final/H.5 + WINDOWS #23 — not a stub |

No TBD/FIXME/XXX debt markers, no TODO/HACK/PLACEHOLDER, no empty implementations in any phase-modified file. Test Quality Audit: no test files were functionally edited (AST-identical) so no disabled-test, circular-test, or assertion-strength exposure was introduced; the reachability proof's oracle is external to the system under test (cached remote file + checkpoint config.json) — VALID provenance.

### Human Verification Required

Infrastructure/foundation phase (evidence record + test-prose disposition; no user-facing elements) — auto-pass UAT applies. The one harvested deferred human-check (plan 14-02 verification item 6: owner confirms coverage-nightly success or resolves the recorded PENDING) is RESOLVED on record: the owner resolved the PENDING by accept-and-close (H.4.final; WINDOWS #23 closed 13:18:38Z). No behavior-unverified truths remain. Nothing routes to human verification.

### Gaps Summary

None. The single first-pass gap (PENDING-gate record completeness/accuracy) was closed by commit 1f50391 (section H.5) and re-verified against the live GitHub record; both recorded facts match this verifier's independent forensics exactly. The owner-accepted override on the coverage-nightly green clause (rice.uga.edu external upstream, two independent transient classes, GAP-1 chain green in both runs) stands as transcribed in the frontmatter.

Context, not a gap: the underlying rice.uga.edu upstream fragility (truncated body + hang-to-timeout, neither absorbable by the download-path typed skip) is recorded for phase-16 planning in H.4.final, matching WPAR-03/WPAR-04's scope over generate_bpe_dataset.py and the NER notebook. The re-dispatch's example-nightly was still queued at 13:37:45Z; its terminal outcome is delegated to the armed watch (H.5) and is documented as independent of the disposition conclusion.

---

_Verified: 2026-10-13T13:38:30Z (re-verification after gap closure 1f50391)_
_Verifier: Claude (gsd-verifier)_
