---
phase: 02-suite-hygiene-known-bug-fixes
verified: 2026-10-01T12:31:30Z
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
covered_digest: "v2:sha256:ecafd58cffdd2d5605afd043312a15231e9699b65e9da501a9eb0afcc6df2706"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 17/17
  previous_verified: 2026-10-01T10:53:14Z
  gaps_closed: []
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "WR-05: metrics_for_dnabert2(\"regression\") returns a nested {\"r2\": {\"r2\": float}} dict and a phase-03 test pins that malformed shape"
    category: other
    reason: "Still dispositioned open in 02-REVIEW-DISPOSITION.md; later-phase addition (vendored-loading fix 882d211), no failing test, phase-02 truths unaffected"
    evidence_status: "none provided"
  - finding: "WR-06: verbose task_type aliases validate but store the verbose spelling downstream dispatchers reject; new tests pin the verbatim storage"
    category: other
    reason: "Still dispositioned open in 02-REVIEW-DISPOSITION.md; phase-03 test additions, no deterministic evidence"
    evidence_status: "none provided"
  - finding: "IN-11: two defensive skipTest calls in real-model tests (test_trainer_real_model.py, test_inference_real_model.py) are untyped relative to the FIX-03 taxonomy"
    category: other
    reason: "Condition-protected by committed fixtures — never fire in CI (fast-leg census shows 1 skip only); if one ever fired the nightly audit fails closed, so the SC3 enforcement invariant holds; taxonomy classification is a triage nicety, owner-dispositioned open"
    evidence_status: "none provided"
  - finding: "IN-07..IN-10, IN-12 (vacuous fp16/bf16 validator tests, models.lock stale cache key, uncovered plot.py multilabel guards, actions/cache@v3 in deploy, mkdtemp leaks in MCP config tests)"
    category: other
    reason: "All later-phase scope, info severity, owner-dispositioned open in 02-REVIEW-DISPOSITION.md; no deterministic evidence of a phase-02 contract breach"
    evidence_status: "none provided"
---

# Phase 2: Suite Hygiene & Known-Bug Fixes Verification Report

**Phase Goal:** The suite reports true code behavior — no test is skipped because the code crashes, and every remaining skip is a typed, intentional network skip
**Verified:** 2026-10-01T12:31:30Z (stale-digest re-verification at HEAD 89194f1)
**Status:** passed
**Re-verification:** Yes — covered source changed after the 2026-10-01T10:53:14Z pass at c190252; no prior gaps existed, so all 17 truths were fully re-executed

## Goal Achievement

Stale-digest re-verification: the previous pass (17/17, no `gaps:`) was taken at c190252. `git diff c190252..HEAD --stat` shows the covered-source drift since then is exactly ONE commit — de4b5cc (CR-03), touching `.github/workflows/ci.yml` (11 lines: the nightly test-mamba leg install step `.[test,dev]` → `.[base]` plus its timeout comment) and `.github/workflows/README.md` (the matching step-5 doc line); every other changed path is `.planning/` docs. All 13 non-CI covered files are byte-identical to the verified state. The de4b5cc increment was re-audited (see Truth 16) and every truth below was re-executed in the verifier's own process at HEAD.

### Observable Truths

Merged set: 4 ROADMAP Success Criteria (contract wording kept) + 13 plan truths that add detail beyond an SC.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | [SC1] The multiclass AUROC test runs unskipped and passes; `compute_metrics` handles multiclass targets without crashing | ✓ VERIFIED | Verifier run: `pytest tests/tasks/test_metrics.py` → **43 passed / 0 skipped** (junit 43/0/0). Parametrize tuple `("multiclass", 3, ["A","B","C"])` at :776-780 with all-classes-present batch `[[0.1,0.7,0.2],[0.8,0.1,0.1],[0.2,0.3,0.5]]` / labels `[1,0,2]` (:805-808); `pytest.skip` count in file = 0; working multiclass plot test at :295 (`plot=True`, curve assertions). |
| 2 | [SC2] CrossDNA handler results are returned instead of overwritten — a regression test asserts the handler's result survives the dispatch chain | ✓ VERIFIED | `test_load_model_crossdna_result_not_overwritten` run singly → **1 passed**. Source at HEAD (:861-873): `model, tokenizer = None, None` init, 1 `_handle_crossdna_models` call site, `_handle_dnabert2_models` and `_load_model_by_task_type` each behind `if model is None or tokenizer is None`. `model.py` byte-identical since c190252. |
| 3 | [SC3] Every skip is typed and matches an expected-skip allowlist; an unexpected skip fails the run instead of passing silently | ✓ VERIFIED | Full CI-shaped fast leg at HEAD (plain pytest + junit): **1636 passed / 1 skipped / 27 deselected, exit 0** — matching the expected census exactly. `audit_skips.py` on the junit → **exit 0**, single skip allowlisted `[content] "No import statements found"` (test_examples predict_data.ipynb). Negative direction re-proven with synthetic junit (unexpected skip → exit 1 naming `demo::test_bad`, allowed row in trail); fail-closed re-proven (absent junit → exit 1; unparseable junit → exit 1). |
| 4 | [SC4] PDF-marked tests leave the git tree clean (artifacts under `tmp_path`); `.gitignore` ignores `tests/inference/pdf/` | ✓ VERIFIED | Two consecutive `pytest -m pdf` runs at HEAD → **53 passed each** (81 deselected of 134 collected), `tests/inference/pdf` never created, `git status --porcelain tests/inference/` empty after both. `.gitignore:133` carries exactly one `tests/inference/pdf/` entry. |
| 5 | [02-01 T2] Multiclass path raises ValueError with 'missing class id' on absent-class batches; single-sample and empty shapes fail deterministically, never nan | ✓ VERIFIED | Direct probe re-executed (consistent-length batches): absent-class (labels [0,1,0] of 3 classes) → `ValueError: Multiclass metrics require every class id in the eval predictions; missing class id(s) [2], unexpected id(s) [] (2/3 distinct ids present).` (WR-04 bidirectional message); empty batch → deterministic sklearn `ValueError: Found empty input array…`; working batch → finite metrics (AUROC 1.0), zero nan values. |
| 6 | [02-01 T3] `roc_auc_score` carries `labels=expected_classes`; no try/except wraps either metric call | ✓ VERIFIED | Verifier AST walk at HEAD: roc_auc_score kwargs `[average, labels, multi_class]`; 0 roc_auc_score/average_precision_score calls inside any Try node; 1 Raise in the function (the guard). |
| 7 | [02-01 T5] Dispatch order fixed, first-resolved-wins (crossdna → dnabert2 → generic); partial (model, None) falls through like (None, None) | ✓ VERIFIED | Source :855-873: disjunctive None-guards on both later stages (shape above); ordering/survival behaviorally proven by the sentinel test (truth #2). File byte-identical since c190252. |
| 8 | [02-01 T6] `tests/models/test_model.py` green with `-m "not slow"` (0 failures, 0 skips) | ✓ VERIFIED | Run at HEAD: **161 passed / 2 deselected / 0 skipped** (junit 161/0/0). |
| 9 | [02-01 T7] 02-01-SUMMARY records the 12-handler audit table with fix count exactly one | ✓ VERIFIED | Table present (rows 1-12, verdict each); "Confirmed overwrite instances fixed: exactly one (CrossDNA, row 10)"; GPN/OmniDNA recorded as str\|None gates (rows 3/6). |
| 10 | [02-02 T2 — `verification: backstop`] Interrupted/hard-killed PDF-marked run cannot write into the repo tree | ✓ VERIFIED (explicit directly-observed evidence) | Killed run re-executed at HEAD: `timeout -s KILL 9s pytest -m pdf -v` → exit **137** with **51 tests already PASSED** pre-kill (each PDF-writing body ran and asserted its file existed); `tests/inference/pdf` absent; `git status --porcelain tests/inference/` empty. Writes demonstrably landed outside the repo. |
| 11 | [02-02 T3] `@pytest.mark.pdf` selects every PDF-writing test (≥19) | ✓ VERIFIED | `--collect-only -m pdf` → **53/134 collected, 81 deselected** (gate ≥19 met; file is 134 tests after phase-03 growth, all 9 writing classes still selected). |
| 12 | [02-02 T4] `.gitignore` has exactly one PDF entry line; 9 strays deleted | ✓ VERIFIED | Exactly one `tests/inference/pdf/` line (.gitignore:133); no other inference/pdf line; misspelled entry and demo filenames absent; `tests/inference/pdf` does not exist; path clean in git status. Entry KEPT per locked decision (prohibition held). |
| 13 | [02-03 T1] No broad-except skip remains in either root; `test_model.py` has zero `pytest.skip` | ✓ VERIFIED | Verifier AST walk at HEAD: 0 runtime `pytest.skip` calls inside any except handler in both MCP client files; exactly 1 `allow_module_level=True` guard each; `skip_if_unreachable` wired in both. `pytest.skip` count in tests/models/test_model.py = 0; dead message literal = 0. |
| 14 | [02-03 T2] With no server on :8000, all 6 live-server MCP tests skip with `network-unavailable:` prefix; non-network leaf re-raises and FAILS | ✓ VERIFIED | Port 8000 probed refused, then live run at HEAD: **6/6 skips, every message byte-prefixed** (e.g. `network-unavailable: SSE connection test (no server reachable: ConnectError)`), 0 failures, exit 0. Re-raise branches proven offline: `test_network_skip.py` passes (part of the 22-test run). |
| 15 | [02-03 T3] `audit_skips.py` exits 0 on all-allowed, 1 naming unmatched skips, fails closed on absent/unparseable junit | ✓ VERIFIED | Re-executed in verifier's own process: real HEAD fast-leg junit → exit 0; negative synthetic junit → exit 1 with `UNEXPECTED demo::test_bad` named and the allowed row in the trail; unparseable junit → exit 1 (`cannot parse junit artifact`); absent junit → exit 1 (`No such file or directory`). Pinned by 19 tests in `tests/scripts/test_audit_skips.py` (run together with helper tests: 22 passed). |
| 16 | [02-03 T5] CI test job emits pytest-junit.xml and runs the skip-audit step; canary untouched | ✓ VERIFIED | ci.yml parses (yaml.safe_load). All four audited jobs carry blocking audit steps — `test` ("Skip audit (unexpected skips fail the job)"), `test-windows`, `coverage-gate` ("Skip audit (gated junit)"), `coverage-nightly` ("Skip audit (nightly junit)") — **no continue-on-error, no step `if:`** on any of them; each has a junit-emitting pytest step; "Exit-code canary" step present in `test`. de4b5cc increment audited: the change touches only the test-mamba install extras (`.[test,dev]` → `.[base]`, line 319, plus comment/README lines) — it alters dependency resolution on a leg that carries no audit and no junit flag, so no audited leg's skip set can change. test-cuda/test-mamba remain unwired (prohibition held). |
| 17 | [02-03 T6] `expected_skips.yaml` frozen from verbatim messages, every entry categorized, network prefix present | ✓ VERIFIED | 11 entries; verifier validation: every entry has `category` + exactly one non-empty matcher (no wildcard/empty); `prefix: network-unavailable:` and `exact: "No import statements found"` present; categories {content, environment, network, optional-dep}. Freeze accuracy re-proven: the real HEAD fast-leg skip matched the exact entry verbatim. |

**Score:** 17/17 truths verified (0 present, behavior-unverified)

### Prohibition Checks (all hold — verifier-executed evidence at HEAD)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| 02-01: no try/except-to-nan or skip-as-error-handler around the multiclass path | ✓ HELD | AST: 0 metric calls inside `Try`; 0 `pytest.skip` in tests/tasks/test_metrics.py |
| 02-01: GPN/OmniDNA not converted to model-tuple early returns | ✓ HELD | `_ = _handle_gpn_models(model_name)` (model.py:783) / `_ = _handle_omnidna_models(model_name)` (:796) — discard-form import-gate calls; no `model, tokenizer =` assignment from either handler; file byte-identical since c190252 |
| 02-02: .gitignore PDF entry not removed | ✓ HELD | `tests/inference/pdf/` present at .gitignore:133 (exactly one entry) |
| 02-03: no match-all/empty allowlist entry | ✓ HELD | YAML validation: 11 entries, single non-empty matcher each, no wildcards; pinned by `TestLoadAllowlist` tests (in the 22-test pass) |
| 02-03: no audit wiring on test-cuda/test-mamba legs | ✓ HELD | Parsed-YAML check at HEAD: `audit_skips` absent from both jobs (no `junitxml` flag either); slow-leg coverage delivered by phase 04 as the separate `coverage-nightly` job |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/tasks/metrics.py` | presence guard + `labels=` on roc_auc_score | ✓ VERIFIED | Guard intact at :280-294 (bidirectional missing/unexpected message); labels kwarg present (AST) |
| `tests/tasks/test_metrics.py` | unskipped multiclass + plot + edge tests | ✓ VERIFIED | 3-class parametrize :776-780, working plot test :295, edge test :308; 43/0 census |
| `dnallm/models/model.py` | guarded chain at crossdna/dnabert2 segment | ✓ VERIFIED | Source shape :855-873 + sentinel test; byte-identical since c190252 |
| `tests/models/test_model.py` | sentinel test in TestLoadModelAndTokenizer | ✓ VERIFIED | Passes singly; file 161/0 not-slow |
| `tests/inference/test_plot.py` | autouse rebind fixture, no import-time mkdir, 9 class markers | ✓ VERIFIED | 53/134 pdf-selected; twice-run + killed-run proofs green at HEAD |
| `.gitignore` | single `tests/inference/pdf/` entry | ✓ VERIFIED | grep-verified line 133; path clean |
| `dnallm/mcp/tests/_network_skip.py` | typed tuple + flattener + prefixed helper | ✓ VERIFIED | TransportError tuple + `network-unavailable:` prefix present |
| `dnallm/mcp/tests/test_network_skip.py` | 3 offline branch tests | ✓ VERIFIED | Pass (part of 22-test run) |
| `dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py` | 6 rewrites, guards intact | ✓ VERIFIED | AST + live slow-leg run (6/6 prefixed) |
| `tests/expected_skips.yaml` | 11 categorized entries | ✓ VERIFIED | Validated; live-matched at HEAD |
| `scripts/audit_skips.py` | fail-closed junit-vs-allowlist gate | ✓ VERIFIED | Positive + negative + fail-closed ×2 re-executed |
| `.github/workflows/ci.yml` | junit flag + Skip audit step | ✓ VERIFIED | All 4 audit steps blocking at HEAD; de4b5cc drift audited (install-extras only) |
| `tests/scripts/test_audit_skips.py` | 19 audit-gate tests | ✓ VERIFIED | All pass |
| `02-01-SUMMARY.md` audit table | 12 handlers, fix count 1 | ✓ VERIFIED | Present and accurate |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| metrics.py message | edge-test regex | `missing class id(s)` in both files (:289 / :315) | ✓ WIRED (test passes — drift would fail it) |
| sentinel `.to(return_value=sentinel)` | model.py `.to(_get_device())` rebind | identity survives device rebind | ✓ WIRED (test passes) |
| AUROC skip removal | 02-03 allowlist census | fast leg = exactly 1 content skip | ✓ WIRED (observed at HEAD) |
| junit `<skipped message>` | allowlist entries | freeze protocol | ✓ FLOWING (real HEAD skip matched verbatim) |
| ci.yml fast-test step | audit step → job verdict | `--junitxml=pytest-junit.xml` → `audit_skips.py` (blocking, ×4 jobs) | ✓ WIRED |
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
|---------|---------|--------|--------|
| Metrics file census | `pytest tests/tasks/test_metrics.py` | 43 passed / 0 skipped in 3.77s | ✓ PASS |
| Sentinel identity | `pytest …::test_load_model_crossdna_result_not_overwritten` | 1 passed | ✓ PASS |
| Models file not-slow | `pytest tests/models/test_model.py -m "not slow"` | 161 passed / 2 deselected / 0 skipped | ✓ PASS |
| Helper + audit gate units | `pytest test_network_skip.py test_audit_skips.py` | 22 passed | ✓ PASS |
| Typed skips, no server | `pytest <2 MCP files> -m slow` | 6/6 prefixed skips, exit 0 | ✓ PASS |
| pdf marker selection | `--collect-only -m pdf` | 53/134 (81 deselected) | ✓ PASS |
| Tree-clean twice | `-m pdf` × 2 | 53 passed ×2; no pdf dir; tree clean ×2 | ✓ PASS |
| Hard-killed run (backstop) | `timeout -s KILL 9s pytest -m pdf -v` | exit 137 with 51 tests PASSED pre-kill; tree clean; no pdf dir | ✓ PASS |
| Audit positive | `audit_skips.py <real HEAD junit> allowlist` | exit 0, 1 skip allowed `[content]` | ✓ PASS |
| Audit negative / fail-closed ×2 | synthetic neg/bad/absent junit | exit 1 (names test_bad) / exit 1 / exit 1 | ✓ PASS |
| Full fast leg at HEAD | `pytest -m "not slow" --junitxml` | 1636 passed / 1 skipped / 27 deselected, exit 0 (89.2s) | ✓ PASS |
| Guard shapes (direct) | invoke on absent-class and empty batches | ValueError (guard, missing class id(s) [2]) / ValueError (sklearn empty-input) — no nan | ✓ PASS |

Environment note: the fast leg was run with plain pytest (`--junitxml`, no `--cov`) per the known pandas 3.0.6/numpy 2.5.3 ABI conflict under `--cov` in this venv. Skip-audit semantics are coverage-flag-independent (the audit reads the junit skip population), so this substitution loses no evidentiary power; recorded as an environment anomaly, not a phase failure.

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
| dnallm/models/model.py | 831 | `TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (dated via `git log -S` to e0b3494, 2026-03-27 — months before this phase); unchanged by this phase; warning-level marker |
| tests/inference/test_plot.py | 2582/2612 | test names containing "placeholder" | ℹ️ Info | Phase-03 tests asserting the plotting code's no-data placeholder branch — behavior assertions, not stubs |
| Open review items | — | WR-05/WR-06 + IN-07..IN-12 (see Advisory) | ℹ️ Info | Later-phase additions; owner-dispositioned open in 02-REVIEW-DISPOSITION.md |

Zero `TBD`/`FIXME`/`XXX` markers in any covered file (grep at HEAD). No placeholder/stub patterns in the phase-02 core artifacts.

### Advisory (New Scope, Unevidenced)

New-scope findings from Step 7 with no deterministic evidence — reported, not blocking, do not revert a completed must-have. These are the open items from the phase-02 code review (02-REVIEW.md), tracked with disposition `open` in 02-REVIEW-DISPOSITION.md (re-checked this pass; statuses unchanged):

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | WR-05: nested `{"r2": {"r2": float}}` shape from `metrics_for_dnabert2("regression")` pinned by a phase-03 test | other | new-scope (later-phase addition), no failing test |
| 2 | WR-06: verbose `task_type` aliases stored verbatim, rejected downstream; pinned by phase-03 tests | other | new-scope, no deterministic evidence |
| 3 | IN-11: two defensive `skipTest` calls in real-model tests untyped vs FIX-03 taxonomy | other | never fire (condition-protected by committed fixtures; fast-leg census = 1 skip); if they ever fired the nightly audit fails closed — the SC3 invariant holds |
| 4 | IN-07..IN-10, IN-12 (vacuous validator tests; stale models.lock key; uncovered plot.py multilabel guards; actions/cache@v3 in deploy; mkdtemp leaks) | other | info-severity, later-phase scope, dispositioned open |

### Human Verification Required

None. Every must-have — including the one `verification: backstop` truth — was confirmed with explicit, verifier-executed evidence at HEAD (the killed-run observation for the backstop item, re-executed this pass). No visual, UX, or external-service judgment remains open. (Informational only: the wired CI steps execute in their native environment on push; the identical command shape was reproduced locally with matching results.)

### Gaps Summary

No gaps and no regressions. All 17 merged must-have truths re-verified at HEAD 89194f1 after the single-commit covered-source drift (de4b5cc): the fast leg carries exactly one intentional content skip (1636 passed / 1 skipped / audit exit 0), the slow-leg population is 6 typed network skips, an unexpected skip still fails the audit gate, and both former crash-skip fixes remain pinned by passing regression tests. The de4b5cc increment was audited directly: it changes only the nightly test-mamba leg's install extras (`.[base]`, the same set every other leg installs) — a leg with no audit wiring and no junit flag — so no audited leg's skip population can change from it; all four audit steps remain blocking. Digest regenerated at HEAD.

---

_Verified: 2026-10-01T12:31:30Z_
_Verifier: Claude (gsd-verifier)_
