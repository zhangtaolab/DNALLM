---
phase: 13-ollama-surface-disposition
verified: 2026-10-11T09:45:34Z
status: passed
score: 4/5 must-haves verified
covered_files:
  - .github/workflows/README.md
  - .github/workflows/ci.yml
  - .planning/phases/13-ollama-surface-disposition/13-01-PLAN.md
  - .planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md
  - scripts/runner/README.md
  - tests/examples/test_notebook_execution.py
  - tests/expected_skips.yaml
covered_digest: "v3:sha256:c61787b59894f44bab5f09407a0d702ff4b53d0b6fadade71ec76c18760e6ead"
behavior_unverified: 0 # No PRESENT_BEHAVIOR_UNVERIFIED truths — every behavior-dependent claim had a behavioral test run this pass (live pair run, TestGateMatrix, census x2, full fast lane)
overrides_applied: 0
human_verification:
  - test: "Confirm the first post-merge example-nightly run comes green (next 05:30 UTC scheduled run after merge to the nightly-running branch, or an owner workflow_dispatch). Do NOT dispatch casually: a ci.yml workflow_dispatch also enqueues the dispatch-only feas-spike job on the self-hosted box."
    expected: "Stage 0.5 census gate passes at 216/217 tests collected (1 deselected); stage 1 records 2 typed skips with 'network-unavailable:' messages carrying live probe evidence (not passes, not failures); the stage-4 audit over pytest-junit-example-stage1.xml reports OK; stage-results.txt ledger has zero non-zero lines; job green. A red nightly falsifies OLLM-02 and must be triaged from the uploaded complete failure scene (D-14) before phase close."
    why_human: "The example-nightly runs only on the self-hosted dnallm-nightly runner under the D-19 cron-string gate (05:30 UTC schedule or dispatch) — its outcome is not observable from this verifier. Local deterministic proxies for both named failure modes were verified green (no pytest exit 5: collect-only collects 217, 1 deselected, zero errors; no unexpected-skip audit failure: audit_skips.py over a fresh stage-1-shaped pair junit exits 0), but the runner-side confirmation is the phase's own declared human-check (PLAN verification item 6, SUMMARY coverage D5 human_judgment: true, WINDOWS.md entry 22 open unrun-verify)."
---

# Phase 13: Ollama Surface Disposition Verification Report

**Phase Goal:** The ollama pytest surface is deliberately gone or honestly skip-marked — no blanket skip, no deleted neighbor contracts — with the census and nightly gates reconciled in the same change
**Verified:** 2026-10-11T09:45:34Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

Must-haves merged from ROADMAP Success Criteria (Step 2a) and 13-01-PLAN frontmatter (Step 2b). The plan's 13 truths are refinements of the 4 roadmap SCs plus edge-resolved variants; they are folded into the rows below (mapping noted). All checks below were performed by this verifier against the live codebase and live test runs — not taken from SUMMARY claims.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | (SC1 + plan truths 7/8) Every one of the 59 ollama refs across the 5 inventoried test files carries a per-assertion remove-or-skip disposition; skip realized by CI shape — no blanket skip, no unconditional marks, zero functional edits to test files, no deleted neighbor contracts | ✓ VERIFIED | git diff b67e772..HEAD over tests/ touches exactly one file (test_notebook_execution.py) and every changed line is a comment (the D-07 block at 1218-1232); zero `+`-side mark.skip/skipif/pytest.skip lines in the tests/ diff; the only pytest.skip calls in the swept files are the 3 pre-existing typed helpers (environment-unavailable:/optional-dep:/network-unavailable:, tests/examples/_execution.py:1212/1227/1246); residue partition 59 = 56 keep + 3 comment-rewritten + 0 deleted-from-test-files reproduces (per-file 5/30/5/1/15, tree-wide 56) |
| 2 | (SC2 + plan truth 4) Non-ollama contracts sharing the swept surface still exist and pass: TestOllamaUnitPins, TestRunnerReadmeReapply, TestExampleNightlySseProbe, TestMcpExampleModelSwap, test_langchain_ollama_declared_in_mcp_extra, TestNotebookKernelPlotCompat, models-lock docstring | ✓ VERIFIED | All named classes located and exercised: `pytest tests/test_runner_infra_contracts.py tests/test_extras_guard.py tests/test_models_lock_contracts.py -q` → 25 passed in 0.65s; TestGateMatrix 4 passed; full fast lane `pytest -m "not slow" --no-cov -q` → 2487 passed, 2 skipped, 0 failed in 98.66s (matches SUMMARY's 2487/2/0) |
| 3 | (SC3 + plan truths 5/6/9/10/13) Post-sweep `grep ollama tests/` residue equals exactly the sanctioned 56-line list; the two mcp_example notebooks and docs mirrors remain in-repo byte-identical; `langchain-ollama>=1.1.0` stays in the pyproject mcp extra | ✓ VERIFIED | Live grep: tree-wide 56, exactly the 5 swept files, per-file 5/30/5/1/15; the SUMMARY's fenced residue record diffs byte-identical against a live per-file `grep -Hn ollama` in ledger order; `git diff b67e772..HEAD -- '*mcp_client_ollama*' 'docs/**'` is empty (notebooks + mirrors untouched); pyproject.toml line 137 declares `langchain-ollama>=1.1.0` in the mcp extra (pinned green by test_langchain_ollama_declared_in_mcp_extra) |
| 4 | (SC4 local + plan truths 1/2/3/11/12) Census triple re-pinned under the `-m "not giants"`-only selector and reproducing idempotently; expected_skips.yaml updated comment-only; ci.yml example-nightly stages reconciled in the same change; the typed-skip chain fires end-to-end with the fail-closed audit exiting 0 | ✓ VERIFIED | Live: collect-only run twice → `216/217 tests collected (1 deselected)` identical, matching both pinned ci.yml lines (867 grep pattern, 868 FAIL message); stage-0.5 and stage-1 invocations both carry exactly `-m "not giants"`, no -k (zero occurrences of `not mcp_example` or `mcp-stage3` in ci.yml); stage 3 deleted with a D-04/2026-10-11 deletion-marker naming the restore path; stage-4 audit list has exactly 4 junit members; `Stage 2.5` literal prefix present (its `_load_sse_probe_block` consumer at tests/test_runner_infra_contracts.py:63 passes); D-19 cron gate untouched by the diff; expected_skips.yaml diff vs baseline is 3 comment lines only, matchers byte-identical. Live end-to-end: the gated pair under this box's down-stack (both :11434 and :8000 refused) → 2 skipped, messages prefixed `network-unavailable:` carrying the full 30-attempt probe log, pytest exit 0 in 116.90s; `audit_skips.py` over the fresh junit → "OK: every skip ... matches the allowlist", exit 0. TestGateMatrix 4/4 green as the deterministic both-up/down-path proof |
| 5 | (SC4 runner-side clause + PLAN verification item 6) The example-nightly run stays green on the self-hosted runner (no pytest exit 5, no unexpected-skip audit failure) | ⚠ HUMAN-CHECK PENDING | Not observable from this verifier — the job runs only on the self-hosted dnallm-nightly box under the D-19 cron-string gate. Local proxies for both named failure modes are green (collect-only: 217 collected, 1 deselected, no errors → no exit-5 shape; audit exit 0 over a stage-1-shaped junit → no unexpected-skip failure shape). Deferred to the first post-merge run per the phase's own declaration: WINDOWS.md entry 22 (open, unrun-verify), SUMMARY coverage D5 (human_judgment: true). See Human Verification Required |

**Score:** 4/5 truths verified (0 present-but-behavior-unverified; 1 routed to human — external runner observation, not a code invariant)

### Prohibitions (must-NOT checks — all test-tier, all with deterministic evidence)

| # | Prohibition | Status | Evidence |
|---|-------------|--------|----------|
| P1 | MUST NOT widen or loosen the stage-0.5 census grep pattern | ✓ HELD | git diff b67e772..HEAD: the only change to the gate line is `208/217 tests collected \(9 deselected\)` → `216/217 tests collected \(1 deselected\)` — pattern text, anchors, and escaping identical; only the measured numbers re-pinned (c99fa9d shape) |
| P2 | MUST NOT add/edit/remove any matcher entry in tests/expected_skips.yaml (comment-lines-only diff) | ✓ HELD | git diff shows exactly 3 added lines, all `#`-prefixed comments above the pre-existing `- prefix: "network-unavailable:"` entry; zero matcher-line changes |
| P3 | MUST NOT shrink the _probe_http_with_retry window (30 attempts × 2s, ~60s) | ✓ HELD | tests/examples/test_notebook_execution.py:1255-1259: defaults `attempts: int = 30, interval_s: float = 2.0` intact; the tests/ diff is comment-only (zero functional edits); the live pair run burned the full 30-attempt window (116.90s for the pair), confirming the window operates |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci.yml` | Post-surgery example-nightly: stage 3 deleted with D-04 marker, triple re-pinned, deselect dropped from 0.5/1, 4-member audit list, `Stage 2.5` literal kept; contains `216/217 tests collected (1 deselected)` | ✓ VERIFIED | All shape checks pass; triple present on lines 867-868; deletion marker at the former stage-3 site carries D-04 + 2026-10-11 + rationale + restore path and names no deleted junit/log filename |
| `tests/examples/test_notebook_execution.py` | Rewritten D-07 stage-contract comment; zero functional edits elsewhere | ✓ VERIFIED | Diff touches only comment lines in the 1218-1232 block; rewritten block contains 0 occurrences of the token `ollama`; `_gate_ollama_stack`, `TestGateMatrix`, `GATED_NOTEBOOKS`, timeout rows byte-identical |
| `tests/expected_skips.yaml` | Comment-only refresh on the network-unavailable: entry; matchers byte-identical | ✓ VERIFIED | 3 comment lines added (naming Phase 13 D-04, 2026-10-11); matcher set unchanged vs baseline |
| `.github/workflows/README.md` | Post-surgery stage table (no row 3, no deselect claims) + D-04 disposition naming network-unavailable: | ✓ VERIFIED | 0 rows numbered 3; `network-unavailable:` appears 3 times; rows 0.5/1/2.5 reworked; numbering keeps its gap; deletion prose names the restore path |
| `scripts/runner/README.md` | Nightly-VRAM phrase no longer claims the nightly loads the model; every pin literal intact | ✓ VERIFIED | Diff rewords the kv-cache/nightly sentence only; `daemon-reload`, `ollama pull qwen3.5:4b`, `must list qwen3.5:4b`, `restart ollama` all present; `qwen3.8` absent (0 hits) |
| `.planning/phases/13-ollama-surface-disposition/13-01-SUMMARY.md` | Verbatim 56-line residue record + measured census evidence + nightly-evidence note | ✓ VERIFIED | Residue record reproduces byte-for-byte against live grep in ledger order; two identical census lines recorded; ledger reconciliation line and nightly-evidence note (with the feas-spike co-trigger reason and D-05 owner-ops note) present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ci.yml stage 0.5 | ci.yml stage 1 | Identical selector set (`-m "not giants"` only, no -k) | ✓ WIRED | Both invocations read `-m "not giants"`; `grep -c 'not mcp_example' ci.yml` = 0 |
| ci.yml stage-4 audit list | junit files produced by stages 1-2 | No listed junit orphaned by the stage-3 deletion | ✓ WIRED | List has exactly 4 members, each produced by a live stage-1/stage-2 invocation; `mcp-stage3` absent from ci.yml entirely |
| ci.yml stage-2.5 step name | `_load_sse_probe_block` (tests/test_runner_infra_contracts.py:63) | `text.index("Stage 2.5", start)` literal | ✓ WIRED | Step named `"Stage 2.5: hygiene — ... (D-13, pre-audit)"`; TestExampleNightlySseProbe passed in the 25-test pin run |
| `_gate_ollama_stack` | tests/expected_skips.yaml → scripts/audit_skips.py | Typed prefix `network-unavailable:` match | ✓ WIRED | Proven live this pass: gate emitted `network-unavailable:` with probe evidence → audit reported both skips `allowed` → exit 0 |
| scripts/runner/README.md | TestRunnerReadmeReapply + TestMcpExampleModelSwap | Committed-content pins | ✓ WIRED | All pin literals present; both test classes passed |

### Data-Flow Trace (Level 4)

Deletion + re-pin + documentation phase — no new rendered values. The one dynamic chain (skip message → allowlist → audit verdict) was traced live: real probe output (30-attempt connection-refused log from this box's down-stack) flows verbatim into the skip message, the junit, and the audit's allowlist match. ✓ FLOWING.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Census triple reproduces under the new selector, idempotent | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants"` ×2 | `216/217 tests collected (1 deselected)` identical both runs; matches pinned ci.yml lines | ✓ PASS |
| Gated pair typed-skips with live probe evidence under down-stack | `.venv/bin/python -m pytest tests/examples/test_notebook_execution.py -k "mcp_example" -q -rs --junitxml=...` | 2 skipped; both messages prefixed `network-unavailable:` carrying the full 30-attempt probe log; exit 0; 116.90s | ✓ PASS |
| Fail-closed audit over a stage-1-shaped junit | `.venv/bin/python scripts/audit_skips.py <fresh pair junit> tests/expected_skips.yaml` | "OK: every skip ... matches the allowlist"; exit 0 | ✓ PASS |
| Deterministic gate-semantics proof | `.venv/bin/python -m pytest tests/examples/test_notebook_execution.py -k TestGateMatrix -q` | 4 passed (both-up plain return; server-down/ollama-down/both-down typed skips) | ✓ PASS |
| Neighbor pin contracts | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py tests/test_extras_guard.py tests/test_models_lock_contracts.py -q` | 25 passed | ✓ PASS |
| Full fast lane regression (run once) | `.venv/bin/python -m pytest -m "not slow" --no-cov -q` | 2487 passed, 2 skipped, 0 failed in 98.66s | ✓ PASS |
| OLLM-03 residue equality | tree-wide + per-file greps | 56 tree-wide; exactly 5 files; 5/30/5/1/15 per file; SUMMARY record byte-identical in ledger order | ✓ PASS |

Note: on this dev box `grep` is ugrep 7.8.4, which reorders output across multiple file arguments; the byte-for-byte record comparison was done with per-file invocation in the fixed ledger order (GNU grep on the CI runner preserves argument order). Count-based gates are order-insensitive either way.

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` probes exist or were declared for this phase; the phase's runnable checks are the pytest/audit commands above, all executed directly.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OLLM-01 | 13-01 | All ~59 ollama refs dispositioned per-assertion remove-or-skip with rationale; non-ollama contracts preserved | ✓ SATISFIED | Disposition ledger in 13-01-PLAN (59 rows); realized shape verified: residue partition 56/3/0, zero functional test-file edits, no blanket/unconditional marks, all neighbor contracts green (truths 1-2) |
| OLLM-02 | 13-01 | Census triple re-pinned, expected_skips.yaml updated, ci.yml stages reconciled same-change — nightly stays green | ✓ SATISFIED locally / runner-side green pending human check | Census verified live and idempotent; yaml comment-only; stages reconciled in commits ac4115f + 3fcfef7 within the phase window (measured triple recorded in the ac4115f commit message per the c99fa9d precedent); runner-side nightly = Human Verification item 1 (WINDOWS #22) |
| OLLM-03 | 13-01 | Post-sweep residue equals the sanctioned list; notebooks remain in-repo per WINDOWS #13 | ✓ SATISFIED | Live residue 56 = the plan's ledger list, recorded verbatim in the SUMMARY and reproduced byte-for-byte this pass; both notebooks + docs mirrors byte-identical vs baseline |

Orphaned requirements: none — REQUIREMENTS.md maps exactly OLLM-01/02/03 to Phase 13 and the single plan claims all three.

### Anti-Patterns Found

None. Zero `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers in any of the 5 modified files; no stub patterns (the phase adds no code — it deletes a CI stage and rewords comments); no console-log-only implementations; no hardcoded-empty returns. The git window b67e772..3fcfef7 touches exactly the 5 planned files (+74/-83, matching the SUMMARY's measured diffstat).

### Test Quality Audit

No disabled tests linked to the requirements: the gated pair are runtime probe-then-execute tests (not skip-marked), and their skip path is proven both deterministically (TestGateMatrix, monkeypatched params) and live (this pass's pair run). No circular patterns; assertion strength is value-level (typed-prefix + evidence-in-message assertions; audit exact-match on allowlist).

### Decision Coverage

All trackable CONTEXT.md decisions are honored by shipped artifacts: 5/5 honored, 0 not honored (check.decision-coverage-verify; non-blocking gate).

### Human Verification Required

### 1. First post-merge example-nightly run comes green

**Test:** Let the next 05:30 UTC scheduled example-nightly run execute after merge (or have the owner dispatch it — noting a ci.yml workflow_dispatch also enqueues the dispatch-only feas-spike job on the self-hosted box).
**Expected:** Census gate passes at 216/217 (1 deselected); stage 1 records 2 typed skips (`network-unavailable:` with live probe evidence); stage-4 audit over pytest-junit-example-stage1.xml reports OK; stage-results.txt has zero non-zero lines; job green. A red run falsifies OLLM-02 and must be triaged from the uploaded complete failure scene (D-14) before phase close.
**Why human:** The job runs only on the self-hosted dnallm-nightly runner under the D-19 cron-string gate — not observable from this verifier. Both named failure modes (pytest exit 5; unexpected-skip audit failure) are locally disproven, but the runner-side confirmation is the phase's own declared human-check (PLAN verification item 6, SUMMARY D5, WINDOWS #22 open).

### Gaps Summary

No gaps. Every automated check passed, several with live behavioral evidence produced by this verifier rather than trusted from the SUMMARY:

- The D-04 disposition exists in the codebase exactly as planned: stage 3 deleted with a dated marker, deselect gone from both stages, census re-pinned to the measured triple (re-measured twice this pass), audit list at 4 members, stage-2.5 literal preserved for its test consumer.
- The typed-skip chain was re-proven end-to-end on this box's down-stack: 2 typed skips carrying the full 30-attempt probe log, allowlist match, fail-closed audit exit 0.
- The residue oracle holds exactly: 56 lines, 5 files, 5/30/5/1/15, byte-identical to the SUMMARY record.
- All three test-tier prohibitions hold with git-diff/read evidence.
- All neighbor contracts and the full fast lane are green (2487/2/0).

The only outstanding item is the runner-side nightly observation, which the phase itself deferred to the first post-merge run (WINDOWS #22) — it requires the 05:30 UTC schedule or an owner dispatch and is recorded above as the single human-verification item. Status is therefore `human_needed`, not `gaps_found`.

---

_Verified: 2026-10-11T09:45:34Z_
_Verifier: Claude (gsd-verifier)_
