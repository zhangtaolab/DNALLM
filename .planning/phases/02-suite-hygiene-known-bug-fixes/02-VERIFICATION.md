---
phase: 02-suite-hygiene-known-bug-fixes
verified: 2026-09-30T01:51:43Z
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
covered_digest: "v2:sha256:e41364e7fa5fed1d0e05fb88b477b4b4fffa75ee6663b7635e99deae6a73b045"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 2: Suite Hygiene & Known-Bug Fixes Verification Report

**Phase Goal:** The suite reports true code behavior — no test is skipped because the code crashes, and every remaining skip is a typed, intentional network skip
**Verified:** 2026-09-30T01:51:43Z (at HEAD 5fd8dd1, including all post-SUMMARY review fixes c36e981/a0cd13b/8e2e071/fdbed6e)
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

Every claim below was re-executed against the current tree by the verifier (own process, own fixtures); no SUMMARY assertion was taken on trust.

### Observable Truths

Merged set: 4 ROADMAP Success Criteria (contract wording kept) + 13 plan truths that add detail beyond an SC.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | [SC1] The multiclass AUROC test runs unskipped and passes; `compute_metrics` handles multiclass targets without crashing | ✓ VERIFIED | `pytest tests/tasks/test_metrics.py` → **38 passed / 0 skipped** (plan gate ≥37/0; 38th is the WR-04 stray-id test). Parametrize tuple is `("multiclass", 3, ["A","B","C"])` (test_metrics.py:775-778); zero `pytest.skip` in the file (grep=0). Both former crash-skips are gone. |
| 2 | [SC2] CrossDNA handler results are returned instead of overwritten — a regression test asserts the handler's result survives the dispatch chain | ✓ VERIFIED | `test_load_model_crossdna_result_not_overwritten` run singly → **1 passed**. Exact `is`-identity on both sentinel objects after `.to()` rebind; `_handle_dnabert2_models` mocked with a distinct would-be-overwriter tuple; generic loader fault-injected `side_effect=AssertionError` (proves no fall-through). |
| 3 | [SC3] Every skip is typed and matches an expected-skip allowlist; an unexpected skip fails the run instead of passing silently | ✓ VERIFIED | Full CI-shaped fast leg at HEAD: **622 passed / 1 skipped / 0 failed, exit 0**; `audit_skips.py` on the junit → **exit 0**, single skip allowlisted `[content] "No import statements found"`. Negative direction proven with synthetic junit (unexpected skip → exit 1 naming `demo::test_bad`); fail-closed on absent and unparseable junit (both exit 1). Slow leg (2 MCP files, `-m slow`, port 8000 refused): **6/6 skips, every message `network-unavailable: …`**. |
| 4 | [SC4] PDF-marked tests leave the git tree clean (artifacts under `tmp_path`); `.gitignore` ignores `tests/inference/pdf/` | ✓ VERIFIED | Two consecutive `pytest -m pdf` runs → **53 passed each**, `tests/inference/pdf` never created, `git status --porcelain tests/inference/` empty after both. `.gitignore:133` carries exactly one `tests/inference/pdf/` entry; notebook rule (line 115) intact; 0 stray files; path clean. |
| 5 | [02-01 T2] Multiclass path raises ValueError with 'missing class id' on absent-class batches; single-sample and empty shapes fail deterministically, never nan | ✓ VERIFIED | Guard at metrics.py:288-298 raises `ValueError("…missing class id(s) {missing}, unexpected id(s) {unexpected}…")`; `test_multi_classification_metrics_missing_class_raises` + `…unexpected_class_id_raises` pass. Direct probe: single-sample (1 of 3 classes) → guard ValueError with the fragment; empty batch → deterministic `ValueError("Found empty input array…")` from the earlier sklearn accuracy call. Nuance: the empty shape fails loudly via sklearn before the guard is reached — outcome (deterministic ValueError, no nan) holds for all three shapes. |
| 6 | [02-01 T3] `roc_auc_score` carries `labels=expected_classes`; no try/except wraps either metric call | ✓ VERIFIED | Verifier-executed AST walk of `multi_classification_metrics`: roc_auc_score kwargs `[average, multi_class, labels]`; 0 metric calls inside any `Try` node; 1 `Raise` (line 294); `average_precision_score` takes no labels kwarg (guard precedes it). |
| 7 | [02-01 T5] Dispatch order fixed, first-resolved-wins (crossdna → dnabert2 → generic); partial (model, None) falls through like (None, None) | ✓ VERIFIED | model.py:861-878: `model, tokenizer = None, None` before the membership test; both later stages behind `if model is None or tokenizer is None` (disjunctive → partial pairs fall through by construction); post-processing (mutbert/basenji2, `_model_path`/`.source`, padding, `.to(device)`, bnb fix) runs after the chain. Ordering/survival behaviorally proven by the sentinel test (truth #2). Partial-pair sub-clause verified by handler contract: `_handle_crossdna_models` returns only `None, None` or full `(model, tokenizer)` (crossdna.py); `_handle_dnabert2_models` same (dnabert2.py:22,62) — partial pairs are unreachable through real handlers. |
| 8 | [02-01 T6] `tests/models/test_model.py` green with `-m "not slow"` (≥49, 0 failures) | ✓ VERIFIED | Run at HEAD: **49 passed / 2 deselected / 0 skipped**, junit census 49/0/0. |
| 9 | [02-01 T7] 02-01-SUMMARY records the 12-handler audit table with fix count exactly one | ✓ VERIFIED | Table present (12 rows, verdict each); "Confirmed overwrite instances fixed: exactly one (CrossDNA, row 10)"; GPN/OmniDNA recorded as str|None gates. |
| 10 | [02-02 T2 — `verification: backstop`] Interrupted/hard-killed PDF-marked run cannot write into the repo tree | ✓ VERIFIED (explicit evidence) | Three independent pieces, all verifier-executed: (a) AST — zero module-level `mkdir` calls in test_plot.py (the import-time repo write is gone; remaining `mkdir` at L106 is inside `create_pdf_file`, L1970 inside the `__main__` guard which cannot run under pytest); (b) code path — every PDF write resolves the module global `PDF_OUTPUT_DIR` at call time, rebound by the autouse fixture at setup before any test body (L152-157); (c) **directly observed killed run** — SIGKILL at 7s mid-execution (exit 137, 32 tests had already PASSED, i.e. 32 PDF-writing bodies ran and each asserted its file existed): no `tests/inference/pdf` created, `git status --porcelain tests/inference/` empty. Writes demonstrably landed outside the repo. |
| 11 | [02-02 T3] `@pytest.mark.pdf` selects every PDF-writing test (≥19) | ✓ VERIFIED | `--collect-only -m pdf` → **53/65 collected, 12 deselected**; markers on exactly the 9 writing classes (TestPlotBars/Curve/Scatter/AttentionMap/Embeddings/Muts/Performance/PDFOutputQuality/Integration); TestPrepareData/TestEdgeCases unmarked. |
| 12 | [02-02 T4] `.gitignore` has exactly one PDF entry line; 9 strays deleted | ✓ VERIFIED | `grep -c "^tests/inference/pdf/$"` = 1 (line 133); misspelled entry and 6 enumerated demo filenames absent; `ls -A tests/inference/pdf` → 0 entries (dir absent); `git status --porcelain tests/inference/` empty. Entry KEPT per locked decision (prohibition held). |
| 13 | [02-03 T1] No broad-except skip remains in either root; `test_model.py` has zero `pytest.skip` | ✓ VERIFIED | Verifier AST walk: 0 runtime `pytest.skip` calls inside any except handler in both MCP client files; exactly 1 `allow_module_level=True` guard each; `skip_if_unreachable` used at 3+3 call sites. `grep -c pytest.skip tests/models/test_model.py` = 0; dead message literal "Skipping due to network" = 0. |
| 14 | [02-03 T2] With no server on :8000, all 6 live-server MCP tests skip with `network-unavailable:` prefix; non-network leaf re-raises and FAILS | ✓ VERIFIED | Port 8000 probed refused, then live run: 6/6 skips with byte-stable prefixed messages (e.g. `network-unavailable: SSE connection test (no server reachable: ConnectError)`). Re-raise branches proven offline: `test_network_skip.py` 3/3 pass (network-leaf group skips; bare ValueError re-raises; mixed group re-raises the ORIGINAL object). Helper is typed `NETWORK_ERRORS = (httpx.TransportError,)` with all-leaves flattening. |
| 15 | [02-03 T3] `audit_skips.py` exits 0 on all-allowed, 1 naming unmatched skips, fails closed on absent/unparseable junit | ✓ VERIFIED | Verifier-run synthetic fixtures: negative junit → exit 1 with `UNEXPECTED demo::test_bad` named and allowed row shown in trail; unparseable → exit 1; absent → exit 1; all-allowed → exit 0. xfail-typed `<skipped>` correctly excluded (narrow `type.startswith("pytest.xfail")`, audit_skips.py:100). Pinned additionally by 19 tests in `tests/scripts/test_audit_skips.py` (all pass). |
| 16 | [02-03 T5] CI test job emits pytest-junit.xml and runs the skip-audit step; canary/codecov untouched; no cuda/mamba wiring | ✓ VERIFIED | ci.yml parses (yaml.safe_load); line 84 `pytest -m "not slow" --cov --junitxml=pytest-junit.xml`; line 87 "Skip audit (unexpected skips fail the job)" step with no `continue-on-error`/`if: always`; "Exit-code canary" present (line 98); `codecov-action@v3` count 1 (untouched); 0 `audit_skips` occurrences in the test-cuda→EOF region. First live GitHub run will confirm end-to-end on push (wiring is deterministic; identical command shape reproduced locally). |
| 17 | [02-03 T6] `expected_skips.yaml` frozen from verbatim messages, every entry categorized, network prefix present | ✓ VERIFIED | 11 entries; verifier validation: every entry has `category` + exactly one non-empty matcher (no wildcard/empty); `prefix: network-unavailable:` and `exact: No import statements found` present. Freeze accuracy proven behaviorally: the real HEAD fast-leg skip matched the exact entry verbatim. |

**Score:** 17/17 truths verified (0 present, behavior-unverified)

### Prohibition Checks (all hold — verifier-executed evidence)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| 02-01: no try/except-to-nan or skip-as-error-handler around the multiclass path | ✓ HELD | AST: 0 metric calls inside `Try`; 0 `pytest.skip` in tests/tasks/test_metrics.py |
| 02-01: GPN/OmniDNA not converted to model-tuple early returns | ✓ HELD | model.py:783/796 still `_ = _handle_gpn_models(model_name)` / `_ = _handle_omnidna_models(model_name)`; phase diff (6eb5a30..5fd8dd1) shows zero changes to those lines |
| 02-02: .gitignore PDF entry not removed | ✓ HELD | `tests/inference/pdf/` present at .gitignore:133 |
| 02-03: no match-all/empty allowlist entry | ✓ HELD | YAML validation: 11 entries, single non-empty matcher each, no wildcards; pinned by 8 `TestLoadAllowlist` tests |
| 02-03: no audit wiring on test-cuda/test-mamba legs | ✓ HELD | `sed -n '/test-cuda/,$p'` region contains 0 `audit_skips` references |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/tasks/metrics.py` | presence guard + `labels=` on roc_auc_score | ✓ VERIFIED | Guard L288-298 (bidirectional missing/unexpected since WR-04); labels kwarg present; file is 24 lines changed over pre-phase |
| `tests/tasks/test_metrics.py` | unskipped multiclass + plot + edge tests | ✓ VERIFIED | 3-class parametrize, working plot test (curve keys), missing-class + unexpected-id edge tests; 38/0 census |
| `dnallm/models/model.py` | guarded chain at crossdna/dnabert2 segment | ✓ VERIFIED | None-init + two disjunctive None-checks; post-processing preserved |
| `tests/models/test_model.py` | sentinel test in TestLoadModelAndTokenizer | ✓ VERIFIED | L487-532; passes; file 49/0 not-slow |
| `tests/inference/test_plot.py` | autouse rebind fixture, no import-time mkdir, 9 class markers | ✓ VERIFIED | Fixture L152-157 (`sys.modules[__name__]` object form — import-mode-proof); 0 module-level mkdir; 9 marked classes |
| `.gitignore` | single `tests/inference/pdf/` entry (+ post-review `pytest-junit.xml` at :56) | ✓ VERIFIED | grep-verified; `git check-ignore` behavior confirmed by review fix report |
| `dnallm/mcp/tests/_network_skip.py` | typed tuple + flattener + prefixed helper | ✓ VERIFIED | Read in full; matches spec exactly |
| `dnallm/mcp/tests/test_network_skip.py` | 3 offline branch tests | ✓ VERIFIED | 3/3 pass |
| `dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py` | 6 rewrites, guards intact | ✓ VERIFIED | AST + live slow-leg run |
| `tests/expected_skips.yaml` | 11 categorized entries | ✓ VERIFIED | Validated; live-matched |
| `scripts/audit_skips.py` | fail-closed junit-vs-allowlist gate | ✓ VERIFIED | Both directions + fail-closed x2 executed |
| `.github/workflows/ci.yml` | junit flag + Skip audit step | ✓ VERIFIED | Shape gates all pass |
| `tests/scripts/test_audit_skips.py` | 19 audit-gate tests (post-review WR-03) | ✓ VERIFIED | All pass (part of the 22-test run) |
| `02-01-SUMMARY.md` audit table | 12 handlers, fix count 1 | ✓ VERIFIED | Present and accurate against source |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| metrics.py message | edge-test regex | `missing class id\(s\)` in both files | ✓ WIRED (test passes — drift would fail it) |
| sentinel `.to(return_value=sentinel)` | model.py `.to(_get_device())` rebind | identity survives device rebind | ✓ WIRED (test passes) |
| AUROC skip removal | 02-03 allowlist census | fast leg = exactly 1 content skip | ✓ WIRED (observed) |
| junit `<skipped message>` | allowlist entries | freeze protocol | ✓ FLOWING (real HEAD skip matched verbatim) |
| ci.yml fast-test step | audit step → job verdict | `--junitxml=pytest-junit.xml` → `audit_skips.py` (blocking) | ✓ WIRED |
| `PDF_OUTPUT_DIR` global | `create_pdf_file` call-time resolution | autouse rebind before test bodies | ✓ FLOWING (killed-run proof) |

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
| Metrics file census | `pytest tests/tasks/test_metrics.py` | 38 passed / 0 skipped in 3.82s | ✓ PASS |
| Sentinel identity | `pytest …::test_load_model_crossdna_result_not_overwritten` | 1 passed | ✓ PASS |
| Models file not-slow | `pytest tests/models/test_model.py -m "not slow"` | 49 passed / 2 deselected / 0 skipped | ✓ PASS |
| Helper + audit gate units | `pytest test_network_skip.py test_audit_skips.py` | 22 passed | ✓ PASS |
| Typed skips, no server | `pytest <2 MCP files> -m slow` | 6/6 prefixed skips, exit 0 | ✓ PASS |
| pdf marker selection | `--collect-only -m pdf` | 53/65 (12 deselected) | ✓ PASS |
| Tree-clean twice | `-m pdf` × 2 | 53 passed ×2; no pdf dir; tree clean ×2 | ✓ PASS |
| Hard-killed run (backstop) | `timeout -s KILL 7s pytest -m pdf` | exit 137 with 32 tests PASSED pre-kill; tree clean; no pdf dir | ✓ PASS |
| Audit fail-closed ×2 | synthetic bad/absent junit | exit 1, exit 1 | ✓ PASS |
| Audit negative/positive | synthetic neg/ok junit | exit 1 (names test_bad) / exit 0 | ✓ PASS |
| Full fast leg at HEAD | `pytest -m "not slow" --junitxml` + audit | 622/1/0, audit exit 0, tree clean | ✓ PASS |
| Guard shapes (direct) | invoke `multi_classification_metrics` on 1-class and empty batches | ValueError (guard) / ValueError (sklearn empty-input) — no nan | ✓ PASS |

### Probe Execution

Not applicable — the phase declares no `scripts/*/tests/probe-*.sh` probes; its machine gates are the pytest/AST/audit checks executed above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIX-01 | 02-01 | Multiclass AUROC crash fix + unskip | ✓ SATISFIED | Truths 1, 5, 6 |
| FIX-02 | 02-01 | CrossDNA overwrite fix + regression test | ✓ SATISFIED | Truths 2, 7 |
| FIX-03 | 02-03 | Typed network skips + enforced allowlist | ✓ SATISFIED | Truths 3, 13-17 |
| FIX-04 | 02-02 | PDF tmp_path isolation + .gitignore typo | ✓ SATISFIED | Truths 4, 10-12 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly FIX-01..FIX-04 to Phase 2 and all four are claimed by plans and verified. REQUIREMENTS.md checkboxes for FIX-01..04 are marked `[x] Complete`, consistent with the code state.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/models/model.py | ~824 | `TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (introduced e0b3494, 2026-03-27 — before this phase); not phase-scope |
| dnallm/tasks/metrics.py | 72 | bare `print` in library code (IN-05) | ℹ️ Info | Pre-existing (e0b3494); tests patch it; outside phase scope |
| tests/inference/test_plot.py | 1964-1982 | `__main__` harness passes unregistered `--pdf-output-dir` (IN-03) | ℹ️ Info | Dead code outside pytest; never executes in the verified path |
| tests/tasks/test_metrics.py | 154-186 et al. | inert `evaluate.load` mocks in 3 regression tests (IN-04) | ℹ️ Info | Pre-existing pattern; tests pass against real vendored metrics |
| .gitignore | various | duplicate unrelated entries, no EOF newline (IN-02) | ℹ️ Info | Cosmetic; PDF-entry scope of FIX-04 is correct |
| dnallm/mcp/tests/test_sse_client.py | 60-68 | `return True` from async test + noisy prints before skip decision (IN-06) | ℹ️ Info | Cosmetic; skip/re-raise decision verified correct |

Zero `TBD`/`FIXME`/`XXX` markers in any phase-modified file (grep-verified). All Info items are pre-existing hygiene debt recorded with disposition in 02-REVIEW.md / 02-REVIEW-DISPOSITION.md; none were introduced by this phase (provenance checked via `git log -S`).

### Human Verification Required

None. Every must-have — including the one `verification: backstop` truth — was confirmed with explicit, verifier-executed evidence (the killed-run observation for the backstop item). No visual, UX, or external-service judgment remains open. (Informational only: the first real GitHub Actions run after push will exercise the wired CI steps in their native environment; the identical command shape was reproduced locally with matching results.)

### Gaps Summary

No gaps. All 17 merged must-have truths verified at HEAD 5fd8dd1; all artifacts present, substantive, wired, and data-flowing; all five prohibitions hold with executed evidence; requirements FIX-01..FIX-04 all satisfied; no orphaned requirements; no blocker anti-patterns. The phase goal is observably true: the fast leg carries exactly one intentional content skip (622 passed / 1 skip / audit exit 0), the full population is 6 typed network skips + 1 content skip, and an unexpected skip now fails the audit gate.

---

_Verified: 2026-09-30T01:51:43Z_
_Verifier: Claude (gsd-verifier)_
