---
phase: 02-suite-hygiene-known-bug-fixes
plan: "01"
subsystem: testing
tags: [pytest, sklearn, roc-auc, metrics, model-loading, dispatch-chain, regression-tests, fault-injection]

requires:
  - phase: 01-harness-integrity-measured-baseline
    provides: full-suite pass/fail/skip census (the two AUROC crash-skips located), honest single-config pytest harness
provides:
  - Honest multiclass metric path — presence guard raising a matchable ValueError plus labels=-anchored macro-ovr roc_auc_score in dnallm/tasks/metrics.py
  - Guarded first-resolved-wins dispatch chain (crossdna -> dnabert2 -> generic) in load_model_and_tokenizer; CrossDNA handler results survive verbatim
  - Sentinel exact-identity fault-injection regression test pinning the dispatch contract permanently
  - Both crash-skips removed from the suite census (metrics file now 37 tests / 0 skips; models file not-slow 49 / 0)
affects: [02-02 (dead-skip deletion sites untouched here), 02-03 (frozen skip allowlist — the two AUROC skip messages vanish from the census), 03-coverage-tests (crossdna.py dead-path now live), 04-ci-gate]

actuals:
  tokens: 2328
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Presence guard before threshold metrics: validate np.unique(labels) == np.arange(len(label_list)) and raise a matchable ValueError BEFORE any roc_auc_score/average_precision_score call — never try/except-to-nan"
    - "Guarded dispatch chain (return-from-the-chain, not from-the-function): init model/tokenizer to None, each later stage behind a disjunctive None-check so post-processing still runs for resolved results"
    - "Sentinel identity fault-injection: Mock with .to(return_value=sentinel) (loader rebinds via .to(device)) + side_effect=AssertionError on the must-not-run fallback"

key-files:
  created: []
  modified:
    - dnallm/tasks/metrics.py
    - tests/tasks/test_metrics.py
    - dnallm/models/model.py
    - tests/models/test_model.py

key-decisions:
  - "FIX-01 realized as presence guard + labels=expected_classes (not labels= alone): sklearn 1.9.1 silently returns nan on absent-class batches with labels= only, which is the hide-failures pattern this milestone removes; the guard makes behavior version-independent across the >=1.4.0 span"
  - "Guard placed immediately before the AUROC call (below accuracy/precision/recall/F1/MCC, which tolerate absent classes) — minimal diff, and it also protects average_precision_score (takes no labels kwarg) and the plot=True per-class curve branch"
  - "Parametrize tuple changed to 3 classes: sklearn routes a 2-unique-class target to its binary path demanding 1-D scores — 2-class multiclass is a probe-proven dead end regardless of labels="
  - "FIX-02 realized as guarded chain, NOT a literal function-level return at the crossdna site: post-processing (mutbert/basenji2 tokenizers, _model_path/.source, _configure_model_padding, .to(device), _fix_bnb_quantized_layers) must still run for CrossDNA results"
  - "12-handler audit verdict: fix count exactly one (CrossDNA); _handle_gpn_models/_handle_omnidna_models are str|None import-availability gates (their ImportError side effect is the point) and must NOT be converted to model-tuple early returns"

patterns-established:
  - "Presence-guard-then-call for class-threshold metrics (ValueError with 'missing class id(s)' fragment kept in sync with the test regex)"
  - "First-resolved-wins guarded dispatch with disjunctive None-checks; partial (model, None) results fall through exactly like (None, None)"
  - "Sentinel .to() self-return Mock for identity assertions across device rebinds"

requirements-completed: [FIX-01, FIX-02]

coverage:
  - id: D1
    description: "FIX-01 — honest multiclass AUROC/AUPRC: presence guard raises ValueError('missing class id(s)...') on absent-class batches, roc_auc_score anchored with labels=expected_classes, both crash-skips removed and their tests pass"
    requirement: FIX-01
    verification:
      - kind: unit
        ref: "tests/tasks/test_metrics.py#test_compute_metrics_task_types[multiclass-3-*]"
        status: pass
      - kind: unit
        ref: "tests/tasks/test_metrics.py#TestMultiClassificationMetrics::test_multi_classification_metrics_with_plot"
        status: pass
      - kind: unit
        ref: "tests/tasks/test_metrics.py#TestMultiClassificationMetrics::test_multi_classification_metrics_missing_class_raises"
        status: pass
      - kind: other
        ref: "AST gate: labels kwarg on roc_auc_score; no Try-wrapped metric calls; raise present; 'missing class id' in both source and test"
        status: pass
    human_judgment: false
  - id: D2
    description: "FIX-02 — CrossDNA handler result survives the dispatch chain (guarded first-resolved-wins), pinned by a sentinel exact-identity regression test; GPN/OmniDNA gates left untouched"
    requirement: FIX-02
    verification:
      - kind: unit
        ref: "tests/models/test_model.py#TestLoadModelAndTokenizer::test_load_model_crossdna_result_not_overwritten"
        status: pass
      - kind: other
        ref: "AST gate: crossdna call conditional on path membership; dnabert2 call behind 'model is None or tokenizer is None'; each handler call site count == 1"
        status: pass
      - kind: other
        ref: "pytest tests/models/test_model.py -m 'not slow' -> 49 tests, 0 skips"
        status: pass
    human_judgment: false

duration: 6 min
completed: 2026-09-30
status: complete
commits: 2
plan_head_before: 6eb5a300ff78969e82bc7055772291041b0597c5
plan_head_after: e7af353951765492ef1cf35893fb3f56397258a5
---

# Phase 2 Plan 1: Suite Hygiene Tracer (FIX-01 + FIX-02) Summary

**Presence-guarded multiclass AUROC (matchable ValueError on absent-class batches, labels=-anchored macro-ovr) plus a guarded first-resolved-wins dispatch chain so the CrossDNA handler's result survives load_model_and_tokenizer — both formerly-skipped tests now run green**

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-30T00:12:20Z
- **Completed:** 2026-09-30T00:18:59Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- **FIX-01 (dnallm/tasks/metrics.py):** class-presence guard inserted immediately before the AUROC/AUPRC pair — computes `expected_classes = np.arange(len(label_list))` and `present_classes = np.unique(labels)`, raises `ValueError` containing the fragment `missing class id(s)` with the sorted missing ids and present/total counts when they are not array-equal; `roc_auc_score` now carries `labels=expected_classes` (average/multi_class unchanged). The guard converts sklearn's version-dependent degradation (silent nan on 1.9.1, raise on older >=1.4.0) into a deterministic, matchable failure.
- **FIX-01 (tests/tasks/test_metrics.py):** both AUROC skips deleted — the parametrized multiclass test now uses 3 classes with an all-classes-present batch (`[[0.1,0.7,0.2],[0.8,0.1,0.1],[0.2,0.3,0.5]]` / labels `[1,0,2]`), the plot test now asserts `metrics["curve"]` keys `{"fpr","tpr","precision","recall"}`, and a new edge regression test `test_multi_classification_metrics_missing_class_raises` asserts the ValueError on a class-2-absent batch. File census: 37 tests / 0 skipped (grounded before: 36 / 2).
- **FIX-02 (dnallm/models/model.py):** the dispatch segment restructured into a guarded chain — `model, tokenizer = None, None` before the crossdna membership test; `_handle_dnabert2_models` and `_load_model_by_task_type` each behind `if model is None or tokenizer is None`. Previously line 868 unconditionally overwrote the CrossDNA result. **Intended behavior activation:** real CrossDNA models now load through their special handler (previously dead path — this also explains part of crossdna.py's 249 missing coverage lines in the Phase-1 audit); not a regression.
- **FIX-02 (tests/models/test_model.py):** sentinel regression test `test_load_model_crossdna_result_not_overwritten` — exact-identity assertions (`is`) that the objects returned by `_handle_crossdna_models` survive `load_model_and_tokenizer`, with `.to()` self-return Mock, a distinct would-be-overwriter tuple on `_handle_dnabert2_models`, and `_load_model_by_task_type` fault-injected with `AssertionError("generic loader must not run")`. Verified RED against the pre-fix code before applying the fix, GREEN after.
- **Tracer gate:** Task 1 (type=tracer) verified end-to-end post-commit (both automated blocks re-run: 37/0 census + AST shape) before expanding to Task 2.

## Task Commits

Each task was committed atomically:

1. **Task 1: FIX-01 — multiclass AUROC honest end-to-end** - `9f12815` (fix)
2. **Task 2: FIX-02 — CrossDNA dispatch fix + sentinel regression test** - `e7af353` (fix)

**Plan metadata:** committed after this SUMMARY (docs)

## Files Created/Modified

- `dnallm/tasks/metrics.py` - presence guard (ValueError with 'missing class id(s)') immediately before the AUROC call; `labels=expected_classes` on roc_auc_score
- `tests/tasks/test_metrics.py` - unskipped parametrized multiclass test (3 classes), rewritten working plot test, new absent-class edge test
- `dnallm/models/model.py` - guarded dispatch chain at the crossdna/dnabert2 segment (None-init + disjunctive None-checks)
- `tests/models/test_model.py` - new `TestLoadModelAndTokenizer::test_load_model_crossdna_result_not_overwritten`

## 12-Handler Dispatch Audit (locked-decision deliverable)

Walk of every `special/*` handler call site in `load_model_and_tokenizer` (dnallm/models/model.py, post-fix line numbers):

| # | Handler | Call site (post-fix) | Current shape | Verdict |
|---|---------|----------------------|---------------|---------|
| 1 | `_handle_evo2_models` | model.py:773-775 | `if ... is not None: return` | Correct (documented early-return chain, runs before path resolution) |
| 2 | `_handle_evo1_models` | model.py:778-780 | same | Correct |
| 3 | `_handle_gpn_models` | model.py:783 | `_ = _handle_gpn_models(model_name)` | **Not a bug** — returns `str \| None` (import-availability gate; its `ImportError` side effect is the point). Converting to an early return would return a string where a model tuple is expected. Left untouched (verified unchanged post-fix). |
| 4 | `_handle_megadna_models` | model.py:786-788 | `if ... is not None: return` | Correct |
| 5 | `_handle_lucaone_models` | model.py:791-793 | commented out | Dead code, not a bug; deliberately left alone |
| 6 | `_handle_omnidna_models` | model.py:796 | `_ = _handle_omnidna_models(model_name)` | **Not a bug** — same `str \| None` gate pattern as GPN. Left untouched (verified unchanged post-fix). |
| 7 | `_handle_enformer_models` | model.py:799-807 | `if ... is not None: return` | Correct |
| 8 | `_handle_space_models` | model.py:810-818 | same | Correct |
| 9 | `_handle_borzoi_models` | model.py:821-829 | same | Correct |
| 10 | `_handle_crossdna_models` | model.py:861-872 | guarded chain (fixed this plan) | **THE bug — FIXED**: result was unconditionally overwritten by the dnabert2 call; now first-resolved-wins with post-processing preserved |
| 11 | `_handle_dnabert2_models` | model.py:873-874 | None-check fallback (now guarded) | Correct pattern; no longer runs when crossdna already resolved (a CrossDNA snapshot never matches its dnabert-2/dnabert-s basename check, so skipping it loses nothing) |
| 12 | `_handle_mutbert_tokenizer` / `_handle_basenji2_tokenizer` | model.py:877-880 | tokenizer post-processors | Not result handlers; correct |

**Confirmed overwrite instances fixed: exactly one (CrossDNA, row 10).**

## Decisions Made

- Guard message fragment locked to `missing class id(s)` (superset of the plan-required `missing class id`), with sorted missing ids and `(present/total classes present)` counts; the test regex `r"missing class id\(s\)"` is kept in sync and asserted by the AST gate in both directions.
- Sentinel test inserted in `TestLoadModelAndTokenizer` directly after `test_load_model_regular_huggingface` (dispatch-chain tests kept adjacent), mirroring the existing patch-stack style at the import site.
- Dead network-skip scaffolding in `tests/models/test_model.py:97-130` deliberately NOT touched — that is 02-02/FIX-03 scope; this plan's scope gate (prohibition: no edits beyond the two fixes) applies.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-01 (this plan) complete: FIX-01 and FIX-02 landed with regression coverage; ready for 02-02.
- For 02-03's frozen skip allowlist: the two AUROC skip messages (`Multiclass AUROC implementation has issues`, `Multi-class plotting with AUROC requires complex implementation`) are gone from the census — full-run skip population drops 9 -> 7; the fast leg keeps exactly the content skip.
- For Phase 3: `dnallm/models/special/crossdna.py` is now on the live load path (249 previously-dead lines are reachable), which changes its coverage worklist economics.

## Self-Check: PASSED

All 5 key files exist on disk; both task commits (9f12815, e7af353) present in git log; 12-handler audit table with fix count exactly 1 confirmed in this SUMMARY; all task `<verify>` blocks and the plan-level cross-check re-run green (metrics 37/0, models not-slow 49/0, combined 86 passed).

---
*Phase: 02-suite-hygiene-known-bug-fixes*
*Completed: 2026-09-30*
