---
phase: 02-suite-hygiene-known-bug-fixes
verified: 2026-10-01T10:53:14Z
status: passed
score: 17/17 must-haves verified
covered_files:
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-01-PLAN.md
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-01-SUMMARY.md
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-02-PLAN.md
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-02-SUMMARY.md
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-03-PLAN.md
  - .planning/phases/02-suite-hygiene-known-bug-fixes/02-03-SUMMARY.md
  - dnallm/tasks/metrics.py
  - tests/tasks/test_metrics.py
  - dnallm/models/model.py
  - tests/models/test_model.py
  - tests/inference/test_plot.py
  - .gitignore
  - dnallm/mcp/tests/_network_skip.py
  - dnallm/mcp/tests/test_network_skip.py
  - dnallm/mcp/tests/test_sse_client.py
  - dnallm/mcp/tests/test_streamable_http_client.py
  - tests/expected_skips.yaml
  - scripts/audit_skips.py
  - .github/workflows/ci.yml
  - tests/scripts/test_audit_skips.py
covered_digest: "v2:sha256:04b14bb6b461bb211f56404ab38c6071d68f6a3adc4b2cf19d0e9efaf1d50d9e"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 17/17
  previous_verified: 2026-09-30T01:51:43Z
  gaps_closed: []
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "WR-05: metrics_for_dnabert2(\"regression\") returns a nested {\"r2\": {\"r2\": float}} dict and a phase-03 test pins that malformed shape"
    category: other
    reason: "Open in 02-REVIEW-DISPOSITION.md; later-phase addition (vendored-loading fix 882d211), no failing test, phase-02 truths unaffected"
    evidence_status: "none provided"
  - finding: "WR-06: verbose task_type aliases validate but store the verbose spelling downstream dispatchers reject; new tests pin the verbatim storage"
    category: other
    reason: "Open in 02-REVIEW-DISPOSITION.md; phase-03 test additions, no deterministic evidence"
    evidence_status: "none provided"
  - finding: "IN-11: two defensive skipTest calls in real-model tests (test_trainer_real_model.py:52, test_inference_real_model.py:206) are untyped relative to the FIX-03 taxonomy"
    category: other
    reason: "Condition-protected by committed fixtures — never fire in CI (fast leg census shows 1 skip only); if one ever fired the nightly audit fails closed, so the SC3 enforcement invariant holds; taxonomy classification is a triage nicety, owner-dispositioned open"
    evidence_status: "none provided"
  - finding: "IN-07..IN-10, IN-12 (vacuous fp16/bf16 validator tests, models.lock stale cache key, uncovered plot.py multilabel guards, actions/cache@v3 in deploy, mkdtemp leaks in MCP config tests)"
    category: other
    reason: "All later-phase scope, info severity, owner-dispositioned open in 02-REVIEW-DISPOSITION.md; no deterministic evidence of a phase-02 contract breach"
    evidence_status: "none provided"
---

# Phase 2: Suite Hygiene & Known-Bug Fixes Verification Report

**Phase Goal:** The suite reports true code behavior — no test is skipped because the code crashes, and every remaining skip is a typed, intentional network skip
**Verified:** 2026-10-01T10:53:14Z (stale-digest re-verification at HEAD c190252)
**Status:** passed
**Re-verification:** Yes — covered source changed after the 2026-09-30 verification (phases 03–04 plus later fix rounds); no prior gaps existed, so all 17 truths were fully re-executed

## Goal Achievement

Stale-digest re-verification: the 2026-09-30 pass (17/17, no `gaps:`) predates phases 03–04 and today's fix rounds. Git diff of the covered set since 5fd8dd1 shows drift confined to 5 files — `ci.yml` (phase-04 two-job gate, windows leg, nightly runner moves, continue-on-error removal), `dnallm/tasks/metrics.py` (24 lines, all inside `metrics_for_dnabert2` — vendored network-free loading, 882d211; the phase-02 guard in `multi_classification_metrics` untouched), and 3 test files that grew via phase 03 (`test_plot.py` +719, `test_model.py` +1486, `test_metrics.py` +130). `model.py`, `.gitignore`, all MCP test files, `expected_skips.yaml`, `audit_skips.py`, `test_audit_skips.py` are byte-identical to the verified state. Every truth below was re-executed in the verifier's own process at HEAD.

### Observable Truths

Merged set: 4 ROADMAP Success Criteria (contract wording kept) + 13 plan truths that add detail beyond an SC.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | [SC1] The multiclass AUROC test runs unskipped and passes; `compute_metrics` handles multiclass targets without crashing | ✓ VERIFIED | Verifier run: `pytest tests/tasks/test_metrics.py` → **43 passed / 0 skipped** (junit 43/0/0; file grew 38→43 via phase 03, still zero skips). Parametrize tuple `("multiclass", 3, ["A","B","C"])` at :776-780 with all-classes-present batch `[[0.1,0.7,0.2],[0.8,0.1,0.1],[0.2,0.3,0.5]]` / labels `[1,0,2]` (:805-808); `pytest.skip` count in file = 0. |
| 2 | [SC2] CrossDNA handler results are returned instead of overwritten — a regression test asserts the handler's result survives the dispatch chain | ✓ VERIFIED | `test_load_model_crossdna_result_not_overwritten` (test_model.py:739) run singly → **1 passed**. AST at HEAD: 1 `_handle_crossdna_models` call site inside an If testing crossdna membership; 1 `_handle_dnabert2_models` call site behind `model is None or tokenizer is None`; `model.py` byte-identical since 5fd8dd1. |
| 3 | [SC3] Every skip is typed and matches an expected-skip allowlist; an unexpected skip fails the run instead of passing silently | ✓ VERIFIED | Full CI-shaped fast leg at HEAD (plain pytest + junit): **1636 passed / 1 skipped / 27 deselected, exit 0** — matching the caller's reported regression-suite numbers. `audit_skips.py` on the junit → **exit 0**, single skip allowlisted `[content] "No import statements found"`. Negative direction re-proven with synthetic junit (unexpected skip → exit 1 naming `demo::test_bad`, allowed row in trail); fail-closed re-proven (absent junit → exit 1; unparseable junit → exit 1). |
| 4 | [SC4] PDF-marked tests leave the git tree clean (artifacts under `tmp_path`); `.gitignore` ignores `tests/inference/pdf/` | ✓ VERIFIED | Two consecutive `pytest -m pdf` runs at HEAD → **53 passed each** (81 deselected of 134 collected), `tests/inference/pdf` never created, `git status --porcelain tests/inference/` empty after both. `.gitignore:133` carries exactly one `tests/inference/pdf/` entry. |
| 5 | [02-01 T2] Multiclass path raises ValueError with 'missing class id' on absent-class batches; single-sample and empty shapes fail deterministically, never nan | ✓ VERIFIED | Direct probe re-executed: single-sample (1 of 3 classes) → `ValueError: Multiclass metrics require every class id in the eval predictions; missing class id(s) [1, 2], unexpected id(s)…` (WR-04 bidirectional message); empty batch → deterministic sklearn `ValueError: Found empty input array…`. No nan on any shape. |
| 6 | [02-01 T3] `roc_auc_score` carries `labels=expected_classes`; no try/except wraps either metric call | ✓ VERIFIED | Verifier AST walk at HEAD: roc_auc_score kwargs `[average, labels, multi_class]`; 0 metric calls inside any Try node; 1 Raise in the function. |
| 7 | [02-01 T5] Dispatch order fixed, first-resolved-wins (crossdna → dnabert2 → generic); partial (model, None) falls through like (None, None) | ✓ VERIFIED | AST: disjunctive None-guards on both later stages (shape above); ordering/survival behaviorally proven by the sentinel test (truth #2). Handler contracts unchanged (file untouched since verified state). |
| 8 | [02-01 T6] `tests/models/test_model.py` green with `-m "not slow"` (0 failures, 0 skips) | ✓ VERIFIED | Run at HEAD: **161 passed / 2 deselected / 0 skipped** (grew 49→161 via phase 03; junit 161/0/0). |
| 9 | [02-01 T7] 02-01-SUMMARY records the 12-handler audit table with fix count exactly one | ✓ VERIFIED | Table present (12 rows, verdict each); "Confirmed overwrite instances fixed: exactly one (CrossDNA, row 10)"; GPN/OmniDNA recorded as str|None gates. |
| 10 | [02-02 T2 — `verification: backstop`] Interrupted/hard-killed PDF-marked run cannot write into the repo tree | ✓ VERIFIED (explicit directly-observed evidence) | Killed run re-executed at HEAD: `timeout -s KILL 9s pytest -m pdf -v` → exit **137** with **51 tests already PASSED** pre-kill (each PDF-writing body ran and asserted its file existed); `tests/inference/pdf` absent; `git status --porcelain tests/inference/` empty. Writes demonstrably landed outside the repo. (Static half also re-proven: 0 module-level mkdir in test_plot.py.) |
| 11 | [02-02 T3] `@pytest.mark.pdf` selects every PDF-writing test (≥19) | ✓ VERIFIED | `--collect-only -m pdf` → **53/134 collected, 81 deselected** (file grew 65→134 via phase 03; the 9 writing classes still select 53 — no writer lost, gate ≥19 met). |
| 12 | [02-02 T4] `.gitignore` has exactly one PDF entry line; 9 strays deleted | ✓ VERIFIED | `grep -c "^tests/inference/pdf/$"` = 1 (line 133); misspelled entry and enumerated demo filenames absent; `ls -A tests/inference/pdf` → 0 entries (dir absent); path clean in git status. Entry KEPT per locked decision (prohibition held). |
| 13 | [02-03 T1] No broad-except skip remains in either root; `test_model.py` has zero `pytest.skip` | ✓ VERIFIED | Verifier AST walk at HEAD: 0 runtime `pytest.skip` calls inside any except handler in both MCP client files; exactly 1 `allow_module_level=True` guard each; `skip_if_unreachable` wired in both. `grep -c pytest.skip tests/models/test_model.py` = 0; dead message literal = 0. |
| 14 | [02-03 T2] With no server on :8000, all 6 live-server MCP tests skip with `network-unavailable:` prefix; non-network leaf re-raises and FAILS | ✓ VERIFIED | Port 8000 probed refused, then live run at HEAD: **6/6 skips, every message byte-prefixed** (e.g. `network-unavailable: SSE connection test (no server reachable: ConnectError)`), 0 failures, exit 0. Re-raise branches proven offline: `test_network_skip.py` 3/3 pass. |
| 15 | [02-03 T3] `audit_skips.py` exits 0 on all-allowed, 1 naming unmatched skips, fails closed on absent/unparseable junit | ✓ VERIFIED | Re-executed with verifier-built synthetic fixtures: negative junit → exit 1 with `test_bad` named and `test_ok` in trail; unparseable → exit 1; absent → exit 1. Pinned by 19 tests in `tests/scripts/test_audit_skips.py` (all pass; run together with helper tests: 22 passed). |
| 16 | [02-03 T5] CI test job emits pytest-junit.xml and runs the skip-audit step; canary untouched | ✓ VERIFIED | ci.yml parses (yaml.safe_load). `test` job: line 91 `pytest -m "not slow" --cov --junitxml=pytest-junit.xml`; line 93 "Skip audit (unexpected skips fail the job)" — **no continue-on-error, no `if:`** (verified via parsed YAML across all audit steps). "Exit-code canary" present (line 98). Post-phase drift audited: audit wiring now ALSO on `test-windows` (:172, added today), `coverage-gate` (:391), `coverage-nightly` (:481) — all blocking; all are phase-04/today extensions of the same gate, none neuter it. test-cuda/test-mamba remain unwired (prohibition held). |
| 17 | [02-03 T6] `expected_skips.yaml` frozen from verbatim messages, every entry categorized, network prefix present | ✓ VERIFIED | 11 entries (file unchanged since verified state); verifier validation: every entry has `category` + exactly one non-empty matcher (no wildcard/empty); `prefix: network-unavailable:` and `exact: No import statements found` present; categories {content, environment, network, optional-dep}. Freeze accuracy re-proven: the real HEAD fast-leg skip matched the exact entry verbatim. |

**Score:** 17/17 truths verified (0 present, behavior-unverified)

### Prohibition Checks (all hold — verifier-executed evidence at HEAD)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| 02-01: no try/except-to-nan or skip-as-error-handler around the multiclass path | ✓ HELD | AST: 0 metric calls inside `Try`; 0 `pytest.skip` in tests/tasks/test_metrics.py |
| 02-01: GPN/OmniDNA not converted to model-tuple early returns | ✓ HELD | model.py unchanged since 5fd8dd1; `_ = _handle_gpn_models(model_name)` / `_ = _handle_omnidna_models(model_name)` verified present by AST/source |
| 02-02: .gitignore PDF entry not removed | ✓ HELD | `tests/inference/pdf/` present at .gitignore:133 (exactly one entry) |
| 02-03: no match-all/empty allowlist entry | ✓ HELD | YAML validation: 11 entries, single non-empty matcher each, no wildcards; pinned by `TestLoadAllowlist` tests |
| 02-03: no audit wiring on test-cuda/test-mamba legs | ✓ HELD | Parsed-YAML check: `audit_skips` absent from both jobs (no `junitxml` either); slow-leg coverage was instead delivered by phase 04 as the separate `coverage-nightly` job — a different, later-phase mechanism, not wiring on these legs |

### Decision Coverage

`check.decision-coverage-verify` → skipped ("No trackable decisions in CONTEXT.md"); non-blocking by design.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/tasks/metrics.py` | presence guard + `labels=` on roc_auc_score | ✓ VERIFIED | Guard intact (bidirectional missing/unexpected message); labels kwarg present; post-phase drift confined to `metrics_for_dnabert2` (vendored loading) |
| `tests/tasks/test_metrics.py` | unskipped multiclass + plot + edge tests | ✓ VERIFIED | 3-class parametrize, working plot test, edge tests; 43/0 census |
| `dnallm/models/model.py` | guarded chain at crossdna/dnabert2 segment | ✓ VERIFIED | AST shape + sentinel test; byte-identical since verified state |
| `tests/models/test_model.py` | sentinel test in TestLoadModelAndTokenizer | ✓ VERIFIED | :739; passes singly; file 161/0 not-slow |
| `tests/inference/test_plot.py` | autouse rebind fixture, no import-time mkdir, 9 class markers | ✓ VERIFIED | 0 module-level mkdir; 53/134 pdf-selected; twice-run + killed-run proofs green |
| `.gitignore` | single `tests/inference/pdf/` entry (+ `pytest-junit.xml` at :56) | ✓ VERIFIED | grep-verified; path clean |
| `dnallm/mcp/tests/_network_skip.py` | typed tuple + flattener + prefixed helper | ✓ VERIFIED | TransportError tuple + `network-unavailable:` prefix present |
| `dnallm/mcp/tests/test_network_skip.py` | 3 offline branch tests | ✓ VERIFIED | 3/3 pass (part of 22-test run) |
| `dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py` | 6 rewrites, guards intact | ✓ VERIFIED | AST + live slow-leg run (6/6 prefixed) |
| `tests/expected_skips.yaml` | 11 categorized entries | ✓ VERIFIED | Validated; live-matched at HEAD |
| `scripts/audit_skips.py` | fail-closed junit-vs-allowlist gate | ✓ VERIFIED | Both directions + fail-closed ×2 re-executed |
| `.github/workflows/ci.yml` | junit flag + Skip audit step | ✓ VERIFIED | Shape gates pass at HEAD (all 4 audit steps blocking) |
| `tests/scripts/test_audit_skips.py` | 19 audit-gate tests | ✓ VERIFIED | All pass |
| `02-01-SUMMARY.md` audit table | 12 handlers, fix count 1 | ✓ VERIFIED | Present and accurate |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| metrics.py message | edge-test regex | `missing class id(s)` in both files | ✓ WIRED (test passes — drift would fail it) |
| sentinel `.to(return_value=sentinel)` | model.py `.to(_get_device())` rebind | identity survives device rebind | ✓ WIRED (test passes) |
| AUROC skip removal | 02-03 allowlist census | fast leg = exactly 1 content skip | ✓ WIRED (observed at HEAD) |
| junit `<skipped message>` | allowlist entries | freeze protocol | ✓ FLOWING (real HEAD skip matched verbatim) |
| ci.yml fast-test step | audit step → job verdict | `--junitxml=pytest-junit.xml` → `audit_skips.py` (blocking) | ✓ WIRED |
| `PDF_OUTPUT_DIR` global | `create_pdf_file` call-time resolution | autouse rebind before test bodies | ✓ FLOWING (killed-run proof at HEAD) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| scripts/audit_skips.py | skip messages | real junit from full fast-leg run at HEAD | Yes — 1 skip matched allowlist verbatim | ✓ FLOWING |
| tests/expected_skips.yaml | allowed matchers | verbatim captured messages (freeze protocol) | Yes — live message matched `exact` entry | ✓ FLOWING |
| _network_skip.py | leaf exception types | real httpx.ConnectError via live refused-port run | Yes — 6/6 messages carried `ConnectError` | ✓ FLOWING |

No static/hollow data paths found in any phase artifact.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Metrics file census | `pytest tests/tasks/test_metrics.py` | 43 passed / 0 skipped in 3.72s | ✓ PASS |
| Sentinel identity | `pytest …::test_load_model_crossdna_result_not_overwritten` | 1 passed | ✓ PASS |
| Models file not-slow | `pytest tests/models/test_model.py -m "not slow"` | 161 passed / 2 deselected / 0 skipped | ✓ PASS |
| Helper + audit gate units | `pytest test_network_skip.py test_audit_skips.py` | 22 passed | ✓ PASS |
| Typed skips, no server | `pytest <2 MCP files> -m slow` | 6/6 prefixed skips, exit 0 | ✓ PASS |
| pdf marker selection | `--collect-only -m pdf` | 53/134 (81 deselected) | ✓ PASS |
| Tree-clean twice | `-m pdf` × 2 | 53 passed ×2; no pdf dir; tree clean ×2 | ✓ PASS |
| Hard-killed run (backstop) | `timeout -s KILL 9s pytest -m pdf -v` | exit 137 with 51 tests PASSED pre-kill; tree clean; no pdf dir | ✓ PASS |
| Audit negative / fail-closed ×2 | synthetic neg/bad/absent junit | exit 1 (names test_bad) / exit 1 / exit 1 | ✓ PASS |
| Full fast leg at HEAD + audit | `pytest -m "not slow" --junitxml` + audit | 1636 passed / 1 skipped / 27 deselected, exit 0; audit exit 0 | ✓ PASS |
| Guard shapes (direct) | invoke on 1-class and empty batches | ValueError (guard) / ValueError (sklearn empty-input) — no nan | ✓ PASS |

Environment note: the fast leg was run with plain pytest (`--junitxml`, no `--cov`) per the reported pandas 3.0.6/numpy 2.5.3 ABI conflict under `--cov` in this venv. Skip-audit semantics are coverage-flag-independent (the audit reads the junit skip population), so this substitution loses no evidentiary power; recorded as an environment anomaly, not a phase failure.

### Probe Execution

Not applicable — the phase declares no `scripts/*/tests/probe-*.sh` probes; its machine gates are the pytest/AST/audit checks executed above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIX-01 | 02-01 | Multiclass AUROC crash fix + unskip | ✓ SATISFIED | Truths 1, 5, 6 |
| FIX-02 | 02-01 | CrossDNA overwrite fix + regression test | ✓ SATISFIED | Truths 2, 7 |
| FIX-03 | 02-03 | Typed network skips + enforced allowlist | ✓ SATISFIED | Truths 3, 13-17 |
| FIX-04 | 02-02 | PDF tmp_path isolation + .gitignore typo | ✓ SATISFIED | Truths 4, 10-12 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly FIX-01..FIX-04 to Phase 2; all four are claimed by plans (02-01: FIX-01/FIX-02; 02-02: FIX-04; 02-03: FIX-03) and verified. REQUIREMENTS.md checkboxes mark all four `[x] Complete`, consistent with code state at HEAD.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/tasks/test_metrics.py | FIX-01 | 43 | 0 | 0 | Value (metric values, pytest.raises match) | OK |
| tests/models/test_model.py | FIX-02 | 161 (not-slow) | 0 | 0 | Behavioral (identity `is`, fault-injection AssertionError) | OK |
| dnallm/mcp/tests/test_network_skip.py | FIX-03 | 3 | 0 | 0 | Behavioral (skip raised / original re-raised `is`) | OK |
| dnallm/mcp/tests/test_{sse,streamable_http}_client.py | FIX-03 | 6 (slow) | 6 typed network | 0 | Status + typed message | OK (skips are the specified behavior; fail on non-network leaf) |
| tests/scripts/test_audit_skips.py | FIX-03 | 19 | 0 | 0 | Value (exit codes, named findings) | OK |
| tests/inference/test_plot.py | FIX-04 | 53 (pdf) | 0 | 0 | Value + filesystem (file exists under tmp_path) | OK |

Disabled tests on requirements: 0 (only the 2 sanctioned `allow_module_level` ImportError guards). Circular patterns: 0. The sentinel test's mocks are plan-specified fault-injection inputs, not circular value generation.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/models/model.py | 831 | `TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (e0b3494, 2026-03-27 — before this phase); unchanged by this phase; warning-level marker |
| Open review items | — | WR-05/WR-06 + IN-07..IN-12 (see Advisory) | ℹ️ Info | Later-phase additions; owner-dispositioned open in 02-REVIEW-DISPOSITION.md |

Zero `TBD`/`FIXME`/`XXX` markers in any covered file (grep at HEAD). No placeholder/stub patterns in the phase-02 core artifacts.

### Advisory (New Scope, Unevidenced)

New-scope findings from Step 7 with no deterministic evidence — reported, not blocking, do not revert a completed must-have. These are the open items from today's phase-02 code review (02-REVIEW.md), tracked with disposition `open` in 02-REVIEW-DISPOSITION.md:

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | WR-05: nested `{"r2": {"r2": float}}` shape from `metrics_for_dnabert2("regression")` pinned by a phase-03 test | other | new-scope (later-phase addition), no failing test |
| 2 | WR-06: verbose `task_type` aliases stored verbatim, rejected downstream; pinned by phase-03 tests | other | new-scope, no deterministic evidence |
| 3 | IN-11: two defensive `skipTest` calls in real-model tests untyped vs FIX-03 taxonomy | other | never fire (condition-protected by committed fixtures; fast-leg census = 1 skip); if they ever fired the nightly audit fails closed — the SC3 invariant holds |
| 4 | IN-07..IN-10, IN-12 (vacuous validator tests; stale models.lock key; uncovered plot.py multilabel guards; actions/cache@v3 in deploy; mkdtemp leaks) | other | info-severity, later-phase scope, dispositioned open |

### Human Verification Required

None. Every must-have — including the one `verification: backstop` truth — was confirmed with explicit, verifier-executed evidence at HEAD (the killed-run observation for the backstop item, re-executed this pass). No visual, UX, or external-service judgment remains open. (Informational only: the wired CI steps execute in their native environment on push; the identical command shape was reproduced locally with matching results.)

### Gaps Summary

No gaps and no regressions. All 17 merged must-have truths re-verified at HEAD c190252 after the phase-03/04 and fix-round drift: the fast leg carries exactly one intentional content skip (1636 passed / 1 skipped / audit exit 0), the slow-leg population is 6 typed network skips, an unexpected skip still fails the audit gate, and both former crash-skip fixes remain pinned by passing regression tests. The covered-source drift was audited file-by-file and none of it touches a phase-02 seam (metrics.py drift is confined to `metrics_for_dnabert2`; ci.yml drift only extends the audit gate to more legs, all blocking). Digest regenerated at HEAD.

---

_Verified: 2026-10-01T10:53:14Z_
_Verifier: Claude (gsd-verifier)_
