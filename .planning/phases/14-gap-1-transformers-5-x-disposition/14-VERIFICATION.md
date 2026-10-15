---
phase: 14-gap-1-transformers-5-x-disposition
verified: 2026-10-15T15:40:41Z
status: passed
score: 12/12 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/WINDOWS.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-01-PLAN.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-01-SUMMARY.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-02-PLAN.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-02-SUMMARY.md
  - .planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md
  - dnallm/utils/transformers_compat.py
  - tests/examples/test_script_execution.py
  - tests/models/test_model_remote_code.py
covered_digest: "v3:sha256:7971ab3a0b5eafd30b54cc7815426dac38117c2f63577b9a030969b9157a9d2c"
behavior_unverified: 0 # Count of PRESENT_BEHAVIOR_UNVERIFIED truths; none — every behavior-dependent truth was re-exercised live on the current tree
overrides_applied: 1 # Carried forward from the 2026-10-13 pass; premise re-checked against the live GitHub record and corroborated by later phases' green nightlies
overrides:
  - must_have: "The CI-side re-verification lane (D-06) runs: the dispatched CI workflow on ref v1.3 reaches coverage-nightly job success — executing the models-lane smoke on CI's freshly-resolved transformers (the Pitfall-4 drift surface) under the full census with coverage and the skip audit — and the run id, job conclusions, and the smoke's junit row are recorded in 14-EVIDENCE.md section H"
    reason: "Coverage-nightly job success was never reached: both dispatched runs (38337783698, 38354875692, both headSha 2c737516 on ref v1.3) concluded failure solely on rice.uga.edu external-host transients (truncated FASTA -> KeyError Chr6/Chr12; genome-download wget hang -> CellTimeoutError 3600s), never on the GAP-1/transformers surface — the models-lane smoke PASSED for real on CI's freshly-resolved transformers in both runs (3.400s first run; also green in the re-dispatch census per H.4.final), refuting the drift risk the gate existed to catch. No code change can make the gate green (external upstream). The owner resolved the recorded PENDING by explicit accept-and-close at the escalation point of the one-re-dispatch contract (14-EVIDENCE.md H.4.final; WINDOWS #23 closed fixed-with-caveat via gsd-tools)."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-13T13:18:38Z"
re_verification:
  previous_status: passed
  previous_score: 12/12
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 14: GAP-1 Transformers 5.x Disposition Verification Report

**Phase Goal:** WINDOWS #11/#12 leave the owner-pending state — the NT promoter checkpoint either loads for real on transformers 5.17 through an evidence-sized absence-gated shim ladder, or the third benchmark model is honestly removed by recorded decision
**Verified:** 2026-10-15T15:40:41Z
**Status:** passed
**Re-verification:** Yes — fresh full goal-backward pass on the CURRENT tree (HEAD 6b397d1). The 2026-10-13T13:38:30Z report went stale because covered sources changed after it ran: commit 30671e7 appended section H.6 to 14-EVIDENCE.md (+4 lines), commit 628926a (phase 16-02) edited tests/examples/test_script_execution.py, and phases 15-19 appended to .planning/WINDOWS.md and .planning/REQUIREMENTS.md. This pass re-derives every truth from scratch against HEAD; nothing is carried forward unexamined.

## Goal Achievement

The phase goal is achieved on the current tree, independently re-verified (not taken from SUMMARY or the stale report):

- **Zero-rung disposition held and still holds.** `dnallm/utils/transformers_compat.py` is byte-identical from the pre-phase baseline c6710a0 through HEAD (`git diff` empty) — phase 14 added no rungs, and phases 15-19 added none either. The ladder carries its 11 patch rungs (`apply_patches()` at :1228-1246), including the `_patch_pretrained_config_legacy_defaults` rung (:541, closed map `_LEGACY_PRETRAINED_CONFIG_DEFAULTS` :507-510 answering `is_decoder`/`add_cross_attention` as False, absence-gated by the fresh-construction probe `_legacy_config_defaults_missing` :513).
- **The core behavioral claim re-proven live on the current tree:** verifier ran the single named smoke test on the repo venv (transformers 5.17.0 / torch 2.11.0+cu130, warm snapshot cache) — `1 passed in 13.17s`; the 35-family gate — `35 passed in 10.85s`; census collect under the stage-0.5 flags — `216/217 tests collected (1 deselected)`, equal to the live ci.yml pin (lines 892-893); module-import rung probe — fresh `PretrainedConfig()` resolves `is_decoder -> False`, `add_cross_attention -> False` (no model load needed).
- **Dead-branch proof intact and re-executed:** `KEYS-ABSENT` re-run live on the cached checkpoint config.json; the cached remote modeling_esm.py carries exactly ONE `invert_attention_mask` occurrence (:1040), inside the `self.config.is_decoder and encoder_hidden_states is not None` guard with the None else-arm — byte-matching section C's verbatim quote.
- **Evidence record intact:** 14-EVIDENCE.md sections A-G plus H.1-H.6 all present in the fixed order; the post-verification H.6 append (+4 lines, commit 30671e7) only recorded the terminal outcomes H.5's armed watch promised, and every job conclusion it records matches the live GitHub record (below).
- **Owner disposition and closures on record:** section G records `shim-zero-rung` (decided 2026-10-13T02:15Z); commit 216fbad (02:15:39Z) precedes the execution commit 2c73751 (02:35:08Z); WINDOWS #11/#12/#15 rows carry `fixed` with resolved_at 02:29:17.247/.411/.575Z; REQUIREMENTS GAPC-01/GAPC-02 both `[x]` with `Complete` traceability rows.
- The single CI-gate literal (coverage-nightly job success) remains unmet-on-record for the phase-14 commit and carries the standing owner accept-and-close (override below) — its premise was re-checked against the live gh record this pass.

### Observable Truths

| # | Truth | Status | Evidence (fresh, current tree) |
|---|-------|--------|-------------------------------|
| 1 | An empirical load run of the NT promoter checkpoint on transformers 5.17 under the current shim set is on record, sizing the remaining rung ladder (SC1, GAPC-01) | ✓ VERIFIED | 14-EVIDENCE.md section A intact: verbatim command (`-m slow`), 15-line output tail (`1 passed in 6.92s`), versions 5.17.0/2.11.0+cu130, sizing line `new rungs required: 0` (also present at section F). **Verifier re-ran the single named test on HEAD: `1 passed in 13.17s`** on the same venv versions — the load+forward behavior holds on the current tree |
| 2 | Ladder sized from the RECORDED run, never speculatively; rungs extend only on real tracebacks (D-01/D-02) | ✓ VERIFIED | `git diff c6710a0 HEAD -- dnallm/utils/transformers_compat.py` is EMPTY (byte-identical; last commit to the file c028a00 2026-10-07, before phase 14); 11 rung installers in apply_patches() unchanged; no LADDER-UNCLOSED heading in 14-EVIDENCE.md; both recorded runs green — zero rungs added by phase 14 AND zero by phases 15-19 |
| 3 | invert_attention_mask reachability analysis on record (dead-branch proof justifying zero rungs) | ✓ VERIFIED | Section C intact with verbatim guard quote. Verifier re-verified on the live cache: `invert_attention_mask` occurs at exactly ONE line of the cached remote modeling_esm.py (:1040) inside the `if self.config.is_decoder and encoder_hidden_states is not None:` guard (else-arm assigns None), byte-matching the quote; config.json key check re-run: `KEYS-ABSENT` |
| 4 | BPE-leg probe on record with task_type exactly `token` (Pitfall 3) | ✓ VERIFIED | Section B intact verbatim: command, `EsmForTokenClassification` model class, tokenizer roundtrip, forward logits `(1, 17, 7)`, `config.is_decoder resolved: False`; the probe's `task_type="token"`/num_labels 7 mirroring of ner_task_config.yaml recorded in the section prose (not re-run live — model load; superseded as standing evidence by the green nightly BPE lanes) |
| 5 | Example-nightly junit forensics on record with per-item skip classification and the explicit zero-#15-class statement (D-08, Pitfall 5) | ✓ VERIFIED | Section D intact: run 38308972028 job conclusions, stage-1 census line `211 passed, 5 skipped, 1 deselected`, both PASS rows (BPE script 102.244s; benchmark notebook 48.714s), 5-skip classification table, explicit "none of the 5 is a #15-class skip" |
| 6 | Owner decision taken exactly once on the evidence table before any execution (D-04), recorded in section G | ✓ VERIFIED | Section G intact: `shim-zero-rung` verbatim, 2026-10-13T02:15Z, question-as-presented, consequence line; `git log` confirms 216fbad (decision, 02:15:39Z) precedes 2c73751 (execution, 02:35:08Z); 14-02 executed exactly the selected option (comment-only, no removal) |
| 7 | Census baseline 216/217 (1 deselected) idempotent under exact stage-0.5 flags | ✓ VERIFIED | Section E intact (two sed-normalized identical lines); **verifier re-ran collect on HEAD: `216/217 tests collected (1 deselected)`**, equal to the live ci.yml pin (lines 892-893) — the 16-02 test additions were census-neutral as their commit records |
| 8 | Disposition executed comment-only: resolved prose, marker constants and skip branches byte-identical (SC2, D-05) | ✓ VERIFIED | **Marker byte-identity re-proven against the current tree:** `grep -F` extraction of both marker literals diff-identical between 2c73751 and HEAD (NT-MARKER-IDENTICAL, REMOTE-MARKER-IDENTICAL); each defined exactly once (`grep -cF` = 1/1); resolved GAP-1 prose present in both files (remote-code docstring + dormant-tripwire comment; script GAP-1 block citing run 38308972028/section D). The 16-02 edit (628926a) touched only the pybedtools gate area — the GAP-1 comment block and marker are outside its hunks. The pruned-AST proof of the phase-time edit remains valid history (files' functional content identical to the pre-edit baseline at commit time); no GAP-1 narration exists in test_notebook_execution.py (grep exit 1) and no owner-pending narration exists tree-wide in code. SC2's "rung lands and NT tests un-skip" clause remains satisfied by the pre-landed rung + self-healing tests turning green, per the owner-selected shim-zero-rung option |
| 9 | Non-NT families provably unaffected — ladder untouched and absence-gated, 35-family gate green (SC3) | ✓ VERIFIED | Ladder byte-identical (truth 2); rung gate predicate verified absence-based (fresh-construction probe, never overwrites an existing `__getattr__`); **verifier ran the 35-family gate on HEAD: `35 passed in 10.85s`** |
| 10 | Fast lane green with census triple and expected_skips.yaml reconciled in the SAME change (SC4, D-07) | ✓ VERIFIED | `git show 2c73751 --stat`: one atomic commit carries exactly the 5-file same-change set (both test files + WINDOWS + REQUIREMENTS + 14-EVIDENCE H.1-H.3); H.1 records the fast-lane junit tail `2487 passed, 2 skipped` + skip-audit exit 0 + AST/marker/census gates; D-07 vacuous-satisfaction statement in H.2 (no un-skip, no removal) — and the census triple still holds at HEAD (truth 7). Fast lane not re-run this pass (heavy; the recorded junit + subsequent phases' independent full-suite greens — 16-02 `2550 passed`, 19-07 closeout nightly trio all green — corroborate no regression on this surface) |
| 11 | CI-side re-verification lane (D-06): dispatched run reaches coverage-nightly job success, smoke green on CI's freshly-resolved transformers, recorded in section H | ✓ PASSED (override) | Dispatch + smoke claims re-confirmed live via gh this pass: run 38354875692 exists on headSha 2c737516 (created 2026-10-13T06:06:18Z); job conclusions live: test-mamba **success** (14:30:48Z), coverage-nightly failure (12:53:49Z), example-nightly failure (19:08:20Z) — matching H.4.final/H.6's recorded conclusions exactly. The failures remain solely the rice.uga.edu external-host class (three manifestations recorded: truncation KeyError Chr6/Chr12 x2, wget hang CellTimeoutError x2); the models-lane smoke passed on CI's freshly-resolved transformers in both runs (drift refuted). Owner accept-and-close on record (H.4.final; WINDOWS #23 fixed 2026-10-13T13:18:38Z) — see overrides. Later phases' nightlies (16-04 run 38441258701; 19-07 trio 38695301631) went green, corroborating the GAP-1 chain was never the blocker |
| 12 | The PENDING-gate resolution record is complete and accurate (H.4's commitment: terminal outcomes recorded by the resolver) | ✓ VERIFIED | H.5 intact (run 38337783698 example-nightly terminal outcome + the H.4.final test-mamba correction); **the post-verification H.6 append (30671e7, +4 lines, no deletions) delivered exactly what H.5's armed watch promised**, and every H.6 claim matches the live gh record this pass: test-mamba success at 14:30:48Z (the prematurely-claimed conclusion proved true in the end, H.5's recording-process correction stands), example-nightly failure at 19:08:20Z as the third rice-class manifestation, GAP-1 surfaces green in that run (210 passed census shape) |

**Score:** 12/12 truths verified (11 VERIFIED + 1 PASSED via owner-accepted override; 0 present-but-behavior-unverified)

### Re-Verification Detail (Stale-Covered-Source Re-Run)

- **Why stale:** the 2026-10-13T13:38:30Z report's covered sources changed after it ran — 14-EVIDENCE.md gained section H.6 (30671e7, 2026-10-13T19:18Z, pure +4-line append); tests/examples/test_script_execution.py was edited by phase 16-02 (628926a, pybedtools gate win32 early-return + platform-pinned gate-contract tests — outside the GAP-1 block and marker); .planning/WINDOWS.md and .planning/REQUIREMENTS.md gained later-phase rows.
- **Regression results:** none. transformers_compat.py and tests/models/test_model_remote_code.py have zero commits since before phase 14. The 16-02 edit left the `_NT_STRUCTURAL_MARKER` literal and the GAP-1 resolved-state comment block byte-identical (diff-proven). WINDOWS #11/#12/#15/#23 rows unchanged (still fixed with the phase-14 resolved_at timestamps); counters advanced 16/4 -> 18/6 solely from later phases' new entries (#24 fixed; #25/#26/#27 open — phase 17/18 scopes, not phase-14 scope). REQUIREMENTS GAPC rows unchanged (`[x]`, Complete).
- **Fresh evidence added this pass:** live re-run of the smoke test, 35-family gate, census collect, KEYS-ABSENT, and rung probe on the current tree; live gh re-query of the dispatched run's job conclusions; marker byte-identity diff against the phase-14 atomic commit.

### Advisory (New Scope, Unevidenced)

Re-verification ran; no new-scope unevidenced findings were raised.

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| — | None | — | No new-scope Step-7 findings this pass |

### Required Artifacts

gsd-tools `verify.artifacts` passes both plans (14-01: 1/1; 14-02: 5/5; `all_passed: true` each). Manual Level 2/3/4 checks:

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/14-gap-1-transformers-5-x-disposition/14-EVIDENCE.md` | GAPC-01/GAPC-02 record of decision, sections A-G + H (incl. H.5 corrections, H.6 terminal outcomes) | ✓ VERIFIED | 357 lines, all sections present in the fixed order, `new rungs required: 0` present (sections A, F); tool: passed, issues none; H.6 verified against the live gh record |
| `tests/models/test_model_remote_code.py` | resolved-state docstring + dormant-tripwire comment; marker byte-identical | ✓ VERIFIED | `_STRUCTURAL_MARKER` defined exactly once; literal diff-identical 2c73751->HEAD; untouched by every commit after the phase |
| `tests/examples/test_script_execution.py` | resolved-state GAP-1 comment block; marker byte-identical | ✓ VERIFIED | `_NT_STRUCTURAL_MARKER` defined exactly once; literal diff-identical 2c73751->HEAD; GAP-1 block intact outside the 16-02 hunks |
| `.planning/WINDOWS.md` | entries 11, 12, 15 status fixed, counters recomputed | ✓ VERIFIED | #11/#12/#15 fixed with resolved_at 02:29:17.247/.411/.575Z; #23 fixed 13:18:38.402Z; counters 18 fixed / 6 open (advanced only by later-phase entries #24-#27) |
| `.planning/REQUIREMENTS.md` | GAPC-01/GAPC-02 checkboxes complete + traceability rows | ✓ VERIFIED | `[x] **GAPC-01**`, `[x] **GAPC-02**`; rows `\| GAPC-01 \| Phase 14 \| Complete \|`, `\| GAPC-02 \| Phase 14 \| Complete \|` |
| `dnallm/utils/transformers_compat.py` (regression surface) | ladder with zero phase-14/15-19 rung additions | ✓ VERIFIED | 1251 lines, byte-identical c6710a0->HEAD; 11 rungs; legacy-defaults rung live-verified answering the closed map |

### Key Link Verification

The gsd-tools key-link verb reports all links "Source file not found" because the planner wrote prose `from:` fields — a plan-formatting artifact (unchanged from the prior passes). Each link verified manually on the current tree:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| smoke test (`tests/models/test_model_remote_code.py`) | 14-EVIDENCE.md section A | recorded GAPC-01 run = smoke with -m slow | ✓ WIRED | Section A carries the exact test id and command; verifier executed that exact test on HEAD: `1 passed in 13.17s` |
| 14-EVIDENCE.md section F (evidence table) | owner decision checkpoint (14-01 Task 3) | D-04 blocking checkpoint on the recorded table | ✓ WIRED | Section G records `shim-zero-rung` taken on the F table, timestamped 02:15Z; commit 216fbad precedes all execution commits |
| cached remote modeling_esm.py guard block | 14-EVIDENCE.md section C | verbatim quotation of the only call site + key check | ✓ WIRED | Verifier located the quoted block in the live cache (:1029-1044, sole call site :1040), byte-matching; `KEYS-ABSENT` re-run green |
| test marker constants | transformers_compat `_patch_pretrained_config_legacy_defaults` | tripwire exists because the rung answers the detected names | ✓ WIRED | Both marker literals present once and byte-identical; rung present (:541, map :507) and applied in `apply_patches()` (:1228), which runs on module import — live probe confirmed a fresh PretrainedConfig resolves `is_decoder -> False` via the rung |
| 14-EVIDENCE sections A-D | WINDOWS fixed entries 11/12/15 | closure evidence trail | ✓ WIRED | Ledger rows fixed with resolved_at 02:29:17Z x3; H.3 maps each id to its evidence sections |
| pushed v1.3 ref (2c737516) | dispatched coverage-nightly job | gh workflow run ci.yml --ref v1.3 | ✓ WIRED | Live gh this pass: run 38354875692 on headSha 2c737516, created 2026-10-13T06:06:18Z, jobs exactly as recorded in H.4.final/H.6 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| 14-EVIDENCE.md section A sizing line | `new rungs required: 0` | recorded empirical run + verifier's live re-execution of the same test | ✓ real (test execution, not inventory) | ✓ FLOWING |
| 14-EVIDENCE.md section C reachability | guard quote + KEYS-ABSENT | cached remote modeling_esm.py + checkpoint config.json — both re-read by verifier this pass | ✓ real | ✓ FLOWING |
| WINDOWS #11/#12/#15 rows | resolved_at timestamps | gsd-tools ledger writes (phase-14 values unchanged) | ✓ real | ✓ FLOWING |
| REQUIREMENTS GAPC rows | checkbox + traceability | gsd-tools requirements mark-complete | ✓ real | ✓ FLOWING |
| transformers_compat rung answers | `is_decoder`/`add_cross_attention` | live venv probe: fresh PretrainedConfig resolves both to False via the closed map | ✓ real | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| NT promoter loads + forwards on transformers 5.17 (goal core claim) | `pytest tests/models/test_model_remote_code.py::TestRemoteCodeCheckpointCompat::test_nt_v2_promoter_loads_and_forwards -m slow -q --no-cov` | `1 passed in 13.17s` (venv: transformers 5.17.0 / torch 2.11.0+cu130) | ✓ PASS |
| 35-family benchmark gate (SC3) | `pytest tests/benchmark/test_benchmark.py -q --no-cov` | `35 passed in 10.85s` | ✓ PASS |
| Census triple equals ci.yml pin (SC4) | `pytest tests/examples --collect-only -q -m "not giants"` | `216/217 tests collected (1 deselected)`; ci.yml:892 pins the same | ✓ PASS |
| Legacy-defaults rung answers the closed map (no model load) | venv python: import module, construct fresh `PretrainedConfig()`, getattr | `is_decoder -> False`, `add_cross_attention -> False` | ✓ PASS |
| config.json carries no is_decoder/add_cross_attention (reachability premise) | json key check on the cached checkpoint config | `KEYS-ABSENT` | ✓ PASS |
| Marker literals byte-identical from phase-14 commit to HEAD | `diff` of grep-extracted literals, 2c73751 vs HEAD | both `IDENTICAL` (empty diff) | ✓ PASS |
| Dispatched CI run job conclusions (H.4.final/H.5/H.6 accuracy) | `gh run view 38354875692` (live) | test-mamba success 14:30:48Z; coverage-nightly failure 12:53:49Z; example-nightly failure 19:08:20Z — all match the recorded sections | ✓ PASS |
| Full fast lane green (SC4) — not re-run this pass | recorded H.1 junit tail + downstream evidence | `2487 passed, 2 skipped` at 2c73751; later phases' independent full-suite greens (16-02 `2550 passed`; 19-07 closeout trio green) corroborate | ✓ PASS (recorded artifact + no-drift) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes exist or are declared by this phase's plans (verified by find + PLAN grep). The phase's probe discipline was evidence-recording (junit capture + gh API), re-executed live by this verifier where cheap — see Behavioral Spot-Checks.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GAPC-01 | 14-01 | Empirical load run on transformers 5.17 sizes the remaining rung ladder | ✓ SATISFIED | Truths 1-7; REQUIREMENTS row Complete; re-proven live (smoke PASS, sizing zero, ladder byte-identical) |
| GAPC-02 | 14-02 | Shim-vs-removal executed on that evidence with ledger closure | ✓ SATISFIED (with the recorded owner caveat on the CI gate literal) | Truths 8-12; REQUIREMENTS row Complete; closures intact at HEAD |

Orphaned requirements: none — REQUIREMENTS.md maps exactly GAPC-01/GAPC-02 to Phase 14; both are claimed by plans.

### Decision Coverage

`check.decision-coverage-verify`: skipped — "No trackable decisions in CONTEXT.md." The CONTEXT register uses phase-local D-01..D-08 IDs, not the D-XX trackable format (non-blocking by design, same as prior passes). The D-04 owner checkpoint is nonetheless directly evidenced (truth 6).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| .planning/REQUIREMENTS.md | 78 | `cudaXXX` matches the XXX regex | ℹ️ Info | False positive — glob-style reference to cuda extras in the pre-existing Out-of-Scope table (defined 2026-10-11; unchanged since; not a phase-14 debt marker) |
| dnallm/utils/transformers_compat.py | 671 | upstream `# TODO (joao): remove =None ...` comment | ℹ️ Info | Upstream transformers maintainer note inside the vendored cache-transformation code; pre-existing (file untouched since 2026-10-07, before phase 14); informational, not phase-introduced |

No TBD/FIXME/XXX debt markers, no empty implementations, no console.log-only paths in any phase-modified file. Both findings predate the phase and its verification history (the file-level evidence gate applies: neither sits on a file modified since the prior pass within this phase's scope).

### Test Quality Audit

No test file was functionally edited by the phase (pruned-AST gate proved comment-only at execution; markers and branches byte-identical — re-proven against HEAD this pass). The requirement-proving tests are ACTIVE, not skipped: the smoke test executed for real this pass (`-m slow` re-includes it; `1 passed`), and the BPE script-lane / benchmark-notebook tests executed for real in the recorded nightly runs (section D PASS rows). No disabled-test, circular-test, or assertion-strength exposure; the reachability proof's oracle (cached remote file + checkpoint config.json) is external to the system under test — VALID provenance.

### Human Verification Required

None. Infrastructure/evidence phase — no user-facing elements (auto-pass UAT per the infrastructure scoping gate). Every behavior-dependent truth was re-exercised programmatically this pass (smoke, 35-family, census, key-absence, rung probe — all PASS on HEAD); zero PRESENT_BEHAVIOR_UNVERIFIED truths; the one harvested deferred human-check (owner resolves the coverage-nightly PENDING) remains RESOLVED on record (H.4.final owner accept-and-close; WINDOWS #23 fixed).

### Gaps Summary

None. Every must-have re-verified fresh on the current tree: 11 VERIFIED (several with new live behavioral evidence: smoke `1 passed in 13.17s`, 35-family `35 passed in 10.85s`, census 216/217, KEYS-ABSENT, rung, marker byte-identity, live gh job conclusions) plus 1 PASSED via the standing owner-accepted override (coverage-nightly green literal — external rice.uga.edu upstream; premise re-checked against the live GitHub record and corroborated by later phases' green nightlies). The stale-making changes (H.6 append, 16-02 test edit, later-phase planning rows) caused no regression to any phase-14 outcome: the ladder is byte-identical, the markers and GAP-1 prose are byte-identical, the ledger closures and requirement rows are intact, and the census triple still pins at 216/217.

Context, not a gap: the rice.uga.edu upstream fragility was recorded for phase-16 planning in H.4.final and was addressed by phase 16's platform-split scope (WPAR-04); phase 16's SC4 run 38441258701 and phase 19's closeout trio 38695301631 stand as the subsequent green nightly evidence.

---

_Verified: 2026-10-15T15:40:41Z (fresh re-verification of stale report; HEAD 6b397d1)_
_Verifier: Claude (gsd-verifier)_
