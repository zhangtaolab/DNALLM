---
phase: 13-ollama-surface-disposition
verified: 2026-10-15T15:40:49Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - .planning/phases/13-ollama-surface-disposition/13-01-PLAN.md
  - .planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md
  - scripts/runner/README.md
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
covered_digest: "v3:sha256:74e0ee7860f923b78de741e437d6972cb9273f4bd7f2223e070c1928023fb733"
behavior_unverified: 0 # Every behavior-dependent claim has behavioral evidence this pass: live nightly run 38695301631 (census gate step green, stage-1 typed skips with 30-attempt probe evidence, 4 audits =0) plus local runs on the current tree (collect-only triple, gated-pair collection, neighbor contracts, gate matrix)
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 4/5
  gaps_closed:
    - "Runner-side first-post-merge nightly green (the sole human-verification item, WINDOWS #22) — closed by owner-dispatched run 38308972028 (UAT pass, 2026-10-13) and re-proven on the final tree by run 38695301631 (2026-10-15)"
  gaps_remaining: []
  regressions: []
---

# Phase 13: Ollama Surface Disposition Verification Report

**Phase Goal:** The ollama pytest surface is deliberately gone or honestly skip-marked — no blanket skip, no deleted neighbor contracts — with the census and nightly gates reconciled in the same change
**Verified:** 2026-10-15T15:40:49Z
**Status:** passed
**Re-verification:** Yes — fresh goal-backward re-verification of the current tree (HEAD 6b397d1). The prior report (2026-10-11T09:45:34Z) went stale: phases 14-19 later touched `tests/` and `ci.yml` (commits 628926a, 399f9f1, 15bf646, bdf6ba3 among others). Nothing below is taken from SUMMARY claims; every item was re-measured against the current tree or re-proven from the live GitHub Actions run.

## Goal Achievement

Must-haves merged from ROADMAP Success Criteria (Step 2a) and 13-01-PLAN frontmatter (Step 2b); the plan's 13 truths are refinements of the 4 roadmap SCs and fold into the rows below (mapping noted).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (SC1 + plan truths 7/8/10/12) Every ollama ref across the 5 swept test files carries its per-assertion disposition, realized by CI shape: no ollama pytest execution surface remains, no blanket/unconditional skip marks, no deleted neighbor contracts | ✓ VERIFIED | Live case-sensitive `ollama` residue on the current tree: exactly 56 lines in exactly the 5 swept files (per-file 5/30/5/1/15 — `_execution.py` 5, `test_notebook_execution.py` 30, `test_extras_guard.py` 5, `test_models_lock_contracts.py` 1, `test_runner_infra_contracts.py` 15). Every residue line inspected: comments, committed-file content pins (unit file, README re-apply, notebook-cell string contracts), the monkeypatched TestGateMatrix, or the `_gate_ollama_stack`/`GATED_NOTEBOOKS` probe mechanism itself — zero lines that start, serve, or live-call ollama. Zero `@pytest.mark.skip`/`skipif` in the swept files; the only skip paths are the 3 pre-existing typed helpers (`environment_unavailable_skip`/`optional_dep_skip`/`network_unavailable_skip`, tests/examples/_execution.py:1197/1215/1230). ci.yml: zero occurrences of `not mcp_example`; stage 3 deleted with the D-04 (2026-10-11, Phase 13) marker at ci.yml:1049-1058 naming the restore path |
| 2 | (SC2 + plan truth 4) Non-ollama contracts sharing the swept surface still exist and pass on the current tree | ✓ VERIFIED | Live run this pass: `pytest tests/test_runner_infra_contracts.py tests/test_extras_guard.py tests/test_models_lock_contracts.py -q` → **31 passed** in 3.07s (grew from 25 at phase time via later-phase additions — includes TestOllamaUnitPins, TestRunnerReadmeReapply, TestExampleNightlySseProbe, TestMcpExampleModelSwap, test_langchain_ollama_declared_in_mcp_extra). `pytest tests/examples/test_notebook_execution.py -k "TestGateMatrix or KernelPlotCompat" -q` → **4 passed** (gate semantics both-up/down ×3; kernel-plot compat pins) |
| 3 | (SC3 + plan truths 5/6/9/11/13) Post-sweep residue equals the sanctioned 56-line list; the two mcp_example notebooks and docs mirrors remain in-repo untouched; `langchain-ollama>=1.1.0` stays in the mcp extra | ✓ VERIFIED | Residue content-diff (line-number column stripped): **56/56 lines byte-identical** to the SUMMARY's verbatim ledger record, same files, same ledger order. Line-number drift only (e.g. `_gate_ollama_stack` 1293→1297, extras test 77→103), fully attributable to later-phase commits inserting lines above (628926a 16-02 gates, 399f9f1 18-01 extras surgery, 15bf646 19-07 posix paths) — no phase-13 content regression. Notebooks: `example/mcp_example/mcp_client_ollama_{pydantic_ai,langchain_agents}.ipynb` + `docs/example/mcp_example/` mirrors all in-repo; `git log --since=2026-10-11 -- example/mcp_example/ docs/example/mcp_example/` is EMPTY (untouched by every later phase). pyproject.toml:137 declares `langchain-ollama>=1.1.0` in the mcp extra (pinned green by test_langchain_ollama_declared_in_mcp_extra in the 31-pass run) |
| 4 | (SC4 local + plan truths 1/2/3) The D-03 census triple reads 216/217 (1 deselected) on the current ci.yml, re-pinned from 208/217 (9 deselected) with the pattern un-widened; the -k mcp deselect stays dropped from stages 0.5 and 1; expected_skips.yaml carries the sanctioned `network-unavailable:` prefix entry; the audit chain is wired; the pair stays collected | ✓ VERIFIED | Current ci.yml:892-893 pins `216/217 tests collected \(1 deselected\)` on both gate lines; stage-0.5 (ci.yml:885) and stage-1 (ci.yml:909-910) commands carry exactly `-m "not giants"`, no `-k` anywhere in either; `grep -c 'not mcp_example' ci.yml` = 0; stage-3 junit absent from the 4-member audit list (ci.yml:1065-1067 — example-stage1, yaml-stage1, mcp-stage2-streamable, mcp-stage2-sse, each produced by a live stage-1/stage-2 invocation at ci.yml:910/918/979/1014); `Stage 2.5` literal present (ci.yml:1023; its `_load_sse_probe_block` consumer test passed). expected_skips.yaml:32-33 carries `- prefix: "network-unavailable:"` with the Phase-13 D-04 comment (lines 29-31). Local behavior this pass: `pytest tests/examples --collect-only -q -m "not giants"` (the exact stage-0.5 flags) → `216/217 tests collected (1 deselected)`; `-k mcp_example` collect → 2/72, the gated pair still collected. D-19 cron gate intact (03:00/05:30 UTC entries, per-job cron-string gate, ci.yml:28-36) |
| 5 | (SC4 runner-side clause) The nightly run is green on the final tree: census gate passes with the exact stage-0.5 flags, the pair typed-skips `network-unavailable:` with live probe evidence, the fail-closed audit exits 0, no pytest exit 5 | ✓ VERIFIED | Live evidence, run **38695301631** (workflow "CI", branch v1.3, sha **bdf6ba3** = HEAD~1 — HEAD 6b397d1 is docs-only; the working tree modifies only `.planning/` files): example-nightly job 116290360119 **success 2026-10-15T06:54:50Z → 09:59:00Z**; "Stage 0.5: census collection assertion (D-03)" step conclusion success, log line `census collection measured: 216/217 tests collected (1 deselected)`; stage 1 → `211 passed, 5 skipped, 1 deselected in 8969.64s`; stage-4 audit log: both `test_gated_notebook_probes_then_executes[mcp_example/...]` ids recorded `allowed [network]` with `network-unavailable:` messages carrying the full 30-attempt connection-refused probe evidence for both :11434 and :8000; all four `stage4-audit-*=0`; coverage-nightly job also success. Prior UAT (13-UAT.md, status complete): owner-dispatched run 38308972028 job 115009783996 green 2026-10-13 with the same census/skip/audit shape — the phase's declared first-post-merge check (WINDOWS #22) is closed |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified)

### Advisory (New Scope, Unevidenced)

None. No new-scope findings arose during the Step 7 re-scan; every discrepancy observed (residue line-number drift, ci.yml venv-python pinning at bdf6ba3, extras renames at 399f9f1) traces to a documented later-phase commit with the phase-13 gate lines surviving intact.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci.yml` | Stage 3 deleted with D-04 marker, triple re-pinned, deselect dropped from 0.5/1, 4-member audit list, `Stage 2.5` literal kept, contains `216/217 tests collected (1 deselected)` | ✓ VERIFIED | verify.artifacts: passed (exists, substantive). All shape checks re-confirmed live this pass (truths 1/4): triple at 892-893, marker at 1049-1058 with restore path, audit list at 1065-1067, Stage 2.5 at 1023. Survived later-phase ci.yml edits (399f9f1, 165800d, bdf6ba3) unchanged in substance |
| `tests/examples/test_notebook_execution.py` | Rewritten D-07 stage-contract comment; probe gate + gate matrix intact | ✓ VERIFIED | verify.artifacts: passed. `_gate_ollama_stack` (1297), TestGateMatrix (688-743), GATED_NOTEBOOKS pairs (1438-1439), `_probe_http_with_retry` defaults `attempts: int = 30, interval_s: float = 2.0` (1263-1264) all present; 4 gate-matrix/compat tests pass |
| `tests/expected_skips.yaml` | `network-unavailable:` prefix entry present; phase-13 diff comment-only; matchers byte-identical | ✓ VERIFIED | verify.artifacts: passed. Entry at 32-33 with Phase-13 D-04 comment (29-31). `git diff b67e772..3fcfef7` over the file = exactly 3 added `#` lines, zero matcher changes; `git log --since=2026-10-11T10:00Z -- tests/expected_skips.yaml` is EMPTY — no later phase touched it either |
| `.github/workflows/README.md` | Post-surgery stage table (no row 3) + D-04 disposition naming `network-unavailable:` | ✓ VERIFIED | verify.artifacts: passed |
| `scripts/runner/README.md` | Nightly-VRAM phrase accuracy fix; pin literals intact | ✓ VERIFIED | verify.artifacts: passed; TestRunnerReadmeReapply + TestMcpExampleModelSwap green in the 31-pass run |
| `.planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md` | Verbatim 56-line residue record + measured census evidence | ✓ VERIFIED | verify.artifacts: passed; residue record content-verified 56/56 against live grep this pass (truth 3) |

### Key Link Verification

gsd-tools `verify.key-links` reports 1/5 because 4 `from:` fields carry descriptive suffixes the tool cannot resolve as file paths ("Source file not found") — a plan-authoring format limitation, not wiring failures. All 4 were verified manually this pass:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml stage 0.5 | ci.yml stage 1 | Identical selector set (`-m "not giants"` only, no -k) | ✓ WIRED | Both invocations read exactly `-m "not giants"` (ci.yml:885, 909-910); `grep -c 'not mcp_example' ci.yml` = 0 |
| ci.yml stage-4 audit list | junit files produced by stages 1-2 | No listed junit orphaned by the stage-3 deletion | ✓ WIRED | 4-member list (1065-1067) ↔ producers at 910 (example-stage1), 918 (yaml-stage1), 979 (mcp-stage2-streamable), 1014 (mcp-stage2-sse); `mcp-stage3` absent from ci.yml; live proof: nightly 38695301631 ran all four audits over real junits, all `=0` |
| ci.yml stage-2.5 step name | `_load_sse_probe_block` (tests/test_runner_infra_contracts.py) | `text.index("Stage 2.5", start)` literal | ✓ WIRED | Step named `"Stage 2.5: hygiene — kernel cleanup + memory floor assertion (D-13, pre-audit)"` (ci.yml:1023); TestExampleNightlySseProbe passed in the 31-test run |
| `_gate_ollama_stack` | tests/expected_skips.yaml → scripts/audit_skips.py | Typed prefix `network-unavailable:` match through the fail-closed audit | ✓ WIRED | Live-proven on the final tree: nightly 38695301631 stage-4 log records both gated ids `allowed` with `network-unavailable:` + 30-attempt probe evidence; `stage4-audit-pytest-junit-example-stage1.xml=0` |
| scripts/runner/README.md | TestRunnerReadmeReapply + TestMcpExampleModelSwap | Committed-content pins | ✓ WIRED | Tool-verified ("Pattern found in source"); both classes green in the 31-pass run |

### Data-Flow Trace (Level 4)

Deletion + re-pin phase — no new rendered values. The one dynamic chain (probe result → typed skip message → junit → allowlist match → audit verdict) traced live end-to-end on the final tree via run 38695301631: real 30-attempt connection-refused probe logs for both endpoints flow verbatim into the skip messages, the junit, and the audit's `allowed [network]` verdicts. ✓ FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Census triple reproduces under the exact stage-0.5 flags on the current tree | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` | `216/217 tests collected (1 deselected) in 1.57s` — matches both pinned ci.yml gate lines | ✓ PASS |
| Gated mcp pair stays collected (no silent deselection) | `.venv/bin/python -m pytest tests/examples/test_notebook_execution.py --collect-only -q -m "not giants" -k "mcp_example"` | `2/72 tests collected` — both `test_gated_notebook_probes_then_executes[mcp_example/...]` ids present | ✓ PASS |
| Neighbor pin contracts green on the current tree | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py tests/test_extras_guard.py tests/test_models_lock_contracts.py -q` | 31 passed in 3.07s | ✓ PASS |
| Gate semantics deterministic proof | `.venv/bin/python -m pytest tests/examples/test_notebook_execution.py -k "TestGateMatrix or KernelPlotCompat" -q` | 4 passed in 1.75s | ✓ PASS |
| Runner-side nightly: census gate + typed skips + audit on the final tree (remote) | `gh run view 38695301631` / job 116290360119 | Job success 06:54:50→09:59:00Z 2026-10-15; Stage 0.5 step success, measured triple matches; stage 1: 211P/5S/1D; both gated ids `allowed` with `network-unavailable:` + full probe evidence; 4×`stage4-audit-*=0` | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` probes exist or were declared for this phase; the phase's runnable checks are the pytest/audit commands above, all executed directly (locally and via the live nightly run).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OLLM-01 | 13-01 | All ~59 ollama refs dispositioned per-assertion remove-or-skip with rationale; non-ollama contracts preserved | ✓ SATISFIED | Realized shape re-verified on the current tree (truths 1-2): residue 56, zero execution surface, zero unconditional marks, all neighbor contracts green; ledger partition 56 keep + 3 comment-rewritten + 0 deleted unchanged in content |
| OLLM-02 | 13-01 | Census triple re-pinned, expected_skips.yaml updated, ci.yml stages reconciled same-change — nightly stays green | ✓ SATISFIED | Truths 4-5: gate lines hold on the current ci.yml, local triple reproduces, and the nightly is live-green on the final tree (38695301631) plus the owner-dispatched first post-merge run (38308972028, UAT pass) — the runner-side clause that was pending at first verification is now evidenced |
| OLLM-03 | 13-01 | Post-sweep residue equals the sanctioned list; notebooks remain in-repo per WINDOWS #13 | ✓ SATISFIED | Truth 3: 56/56 content-identical (line-number drift from later phases only, cause-attributed per commit); notebooks + mirrors untouched since 2026-10-11 (empty git log); langchain-ollama>=1.1.0 in the mcp extra |

Orphaned requirements: none — REQUIREMENTS.md maps exactly OLLM-01/02/03 to Phase 13 (all marked Complete) and the single plan claims all three.

### Prohibitions (must-NOT checks — all test-tier, all with deterministic evidence)

| # | Prohibition | Status | Evidence |
|---|-------------|--------|----------|
| P1 | MUST NOT widen or loosen the stage-0.5 census grep pattern | ✓ HELD | `git diff b67e772..3fcfef7 -- .github/workflows/ci.yml`: the only gate-line change is `208/217 tests collected \(9 deselected\)` → `216/217 tests collected \(1 deselected\)` on both lines — pattern text, anchors, escaping identical, numbers only. Current tree (892-893) still carries the un-widened pattern, and the nightly gate matched the measured line exactly |
| P2 | MUST NOT add/edit/remove any matcher entry in tests/expected_skips.yaml | ✓ HELD | Phase-window diff = exactly 3 added `#`-comment lines above the pre-existing `network-unavailable:` entry; zero matcher-line changes. No commit has touched the file since 3fcfef7 (git log empty) — the file at HEAD is byte-stable through all later phases |
| P3 | MUST NOT shrink the _probe_http_with_retry window (30 attempts × 2s) | ✓ HELD | Current defaults `attempts: int = 30, interval_s: float = 2.0` intact (test_notebook_execution.py:1263-1264); live proof on the final tree: the nightly 38695301631 skip messages carry the full `attempt 1` … `attempt 30` connection-refused sequence — the window operates end-to-end |

### Anti-Patterns Found

None. Zero `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers in any of the 5 implementation files (ci.yml, test_notebook_execution.py, expected_skips.yaml, both READMEs). The phase adds no code (deletes a CI stage, rewords comments), so no stub/hollow/empty-implementation patterns apply.

### Test Quality Audit

No disabled tests linked to the requirements: the gated pair are runtime probe-then-execute tests (no skip marks), their skip path proven deterministically (TestGateMatrix, monkeypatched probes, 4/4 green) and live (nightly junit: both ids `allowed` with typed prefix + probe evidence). No circular patterns; assertion strength is value-level (typed-prefix + evidence-in-message pins; audit exact-match on the allowlist).

### Decision Coverage

All 5 trackable CONTEXT.md decisions (D-01..D-05) honored by shipped artifacts: 5/5 honored, 0 not honored (check.decision-coverage-verify). Non-blocking gate; status impact none.

### Human Verification Required

None. The single item from the previous verification (first post-merge example-nightly green, WINDOWS #22) is closed with recorded evidence: 13-UAT.md status `complete`, test 1 `result: pass` citing owner-dispatched run 38308972028 (job 115009783996, success 2026-10-13, log-verified census triple + typed skips + 4 audits =0), superseded on the final tree by scheduled run 38695301631 (success 2026-10-15). No new human-verification items arose this pass.

### Gaps Summary

No gaps. Every automated and behavioral check passed against the CURRENT tree (HEAD 6b397d1) rather than the phase-time snapshot:

- The D-04 disposition survives six later phases untouched in substance: stage 3 still deleted with its dated marker, deselect still absent from stages 0.5/1, census triple still pinned at 216/217 (1 deselected) and re-measured identical locally, audit list still 4 non-orphaned junits, expected_skips.yaml byte-stable since 3fcfef7.
- The residue oracle holds in content (56/56 byte-identical modulo line numbers; drift cause-attributed to named later-phase commits).
- The typed-skip chain is live-proven on the final tree by run 38695301631 (bdf6ba3): census gate green at the exact pinned triple, both gated ids typed-skipped `network-unavailable:` with full 30-attempt probe evidence, all four audits exit 0, job green 06:54:50→09:59:00Z 2026-10-15.
- The previously pending runner-side clause (OLLM-02 nightly-green, WINDOWS #22) is closed by UAT run 38308972028 and re-proven by 38695301631.
- All neighbor contracts and the gate matrix are green on the current tree (31 + 4 passed).

The prior report's staleness was confined to line-number drift in its recorded evidence; no phase-13 contract regressed.

---

_Verified: 2026-10-15T15:40:49Z_
_Verifier: Claude (gsd-verifier)_
