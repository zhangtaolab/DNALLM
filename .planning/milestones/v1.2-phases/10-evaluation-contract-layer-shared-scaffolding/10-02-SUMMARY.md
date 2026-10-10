---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: "02"
subsystem: testing
tags: [metric-registry, evaluation-contract, sklearn, pytest, coverage, rev-02]

requires:
  - phase: 10-evaluation-contract-layer-shared-scaffolding
    provides: "none — wave-1 plan, depends_on: [] (parallel with 10-01/10-03/10-04 on disjoint files)"
provides:
  - "dnallm.tasks.metric_registry: single metric name authority — METRIC_REGISTRY {canonical: (fn, aliases)}, resolve(), canonical_name(), registered_names(), validate_emission()"
  - "28 canonical metric names anchoring current metrics.py emitted spellings; eval_-prefixed + 4 historical aliases recognized (never emitted)"
  - "metrics.py emission gate: every compute path validates emitted keys through validate_emission before returning"
  - "Read-only resolution surface for Phase 11 probing/VEP/sweep lanes and the dnallmmark F3 exporter"
affects: [11-adaptation-evaluation, dnallmmark-repo, phase-12-closeout]

actuals:
  tokens: 12484        # chars/4 over this plan's three commit diffs (estimate was 26000)
  tasks: 3
  commits: 3           # 58bbf41, fad6a33, aabba58 (ledger range d3097d6..HEAD counts 14 — shared-tree wave includes sibling lanes' commits)

tech-stack:
  added: []            # zero new dependencies (milestone rule)
  patterns:
    - "frozen registry surface: invariant-checking private builder + MappingProxyType read-only mapping, no mutation API"
    - "import-light module contract: heavy libs (sklearn/scipy/numpy) imported inside per-metric callables, AST-verified by test"
    - "emission gate pattern: _emit(metrics) local helper calling validate_emission at every compute-path return"

key-files:
  created:
    - dnallm/tasks/metric_registry.py
    - tests/tasks/test_metric_registry.py
  modified:
    - dnallm/tasks/metrics.py
    - tests/tasks/test_metrics.py
    - CHANGELOG.md
    - .planning/phases/10-evaluation-contract-layer-shared-scaffolding/deferred-items.md

key-decisions:
  - "METRIC_REGISTRY exposed as types.MappingProxyType (read-only mapping) rather than a plain dict — structurally implements the immutability must-have and T-10-03 tampering mitigation; item assignment now raises TypeError (test-asserted)"
  - "Score-metric callables (AUROC/AUPRC/AUROC_ovr/AUROC_ovo) dispatch on input shape: 1D binary scores, 2D multilabel indicator, 2D two-column binary (positive column), 2D multiclass matrix (macro ovr/ovo) — one canonical primitive per emitted spelling, matching how metrics.py computes each today"
  - "coverage CLI (coverage run -m pytest + coverage report) used for the same-change coverage-row proof: pytest-cov 7.1 dotted-target --cov=dnallm.* crashes at conftest load (pre-existing torch double-execution, reproducible with any dnallm cov target) — logged to deferred-items.md, not fixed in-phase (out of scope)"
  - "Line 314 of metric_registry.py (duplicate-canonical guard in _build_registry) left as the single uncovered line (99% >= 96% standard): structurally unreachable via dict construction since dict keys are unique by definition; kept as defense-in-depth for exotic mapping inputs"

patterns-established:
  - "Registry contract: recognition-only aliases (one-directional), exact case-sensitive matching, matchable ValueError on unknown names"
  - "Emission contract test shape: parameterized case builders per compute path + alias-disjointness + injected-bogus-key gate probe"

requirements-completed: [METR-01]

coverage:
  - id: D1
    description: "Metric registry module with full canonical set, lazy callables, invariants, frozen surface"
    requirement: METR-01
    verification:
      - kind: unit
        ref: "tests/tasks/test_metric_registry.py (67 tests: TestResolve, TestCanonicalName, TestInvariants, TestImmutability, TestRegisteredNames, TestImportLight, TestMetricCallables, TestValidateEmission)"
        status: pass
  - id: D2
    description: "metrics.py emits exclusively through the registry; emitted key spellings unchanged; contract tests over every task type"
    requirement: METR-01
    verification:
      - kind: unit
        ref: "tests/tasks/test_metrics.py::TestRegistryEmissionContract (13 parameterized emission cases x 2 assertions + gate probe); 137 passed across both files; pre-existing metrics tests unchanged and green"
        status: pass
  - id: D3
    description: "Coverage-row visibility proof (outside vendored omit glob) at >=96% + CHANGELOG REV-02 entry same-commit"
    requirement: METR-01
    verification:
      - kind: unit
        ref: "coverage report: dnallm/tasks/metric_registry.py 146 stmts 99% (row present), dnallm/tasks/metrics.py 100%; grep -c '(REV-02, R1-2d)' CHANGELOG.md == 1; uv run --no-sync pytest tests/tasks/ -q -> 192 passed"
        status: pass

status: complete
---

# Phase 10 Plan 02: Metric Registry Contract Summary

**One-liner:** Single-name-authority metric registry at `dnallm.tasks.metric_registry` (28 canonical names + eval_-prefixed/historical aliases, frozen surface, import-light) with every metrics.py compute path gated through `validate_emission` — proven coverage-visible at 99% with same-commit CHANGELOG evidence.

## What Was Built

### Task 1 — metric_registry.py (tracer) [commit 58bbf41]

- `dnallm/tasks/metric_registry.py`: sibling of metrics.py, outside the vendored `dnallm/tasks/metrics/` omit glob (coverage row proof below).
- `METRIC_REGISTRY: MappingProxyType[str, tuple[Callable, tuple[str, ...]]]` — exactly the 28 canonical names emitted anywhere in metrics.py: accuracy, precision, recall, f1 (+_micro/_weighted/_samples variants across precision/recall/f1), mcc, matthews_correlation, AUROC, AUPRC, AUROC_ovr, AUROC_ovo, TPR, TNR, FPR, FNR, mse, mae, r2, pearsonr, spearmanr.
- Aliases: `eval_`-prefixed form of every canonical (eval_accuracy, eval_AUROC, eval_spearmanr, ...) PLUS historical eval_auroc, eval_auprc, eval_spearman_r, eval_pearson_r. Exact case-sensitive matching: eval_auroc and eval_AUROC both resolve to AUROC; "Eval_Auroc" raises.
- Each metric callable is `(y_true, y_pred) -> float` with the sklearn/scipy import inside the function body — module imports only stdlib (types/typing/collections.abc); AST test enforces no top-level torch/sklearn.
- `_build_registry` validates construction invariants (no alias equals another entry's canonical, no alias duplicated across entries, canonicals unique) and raises matchable ValueError; a test feeds deliberately colliding tables to prove it.
- Public API (`__all__`): resolve, canonical_name, registered_names (sorted), validate_emission (payload whitelist {"curve", "scatter"}), METRIC_REGISTRY. No mutation API; item assignment raises TypeError (MappingProxyType).
- 67 tests in `tests/tasks/test_metric_registry.py` across 8 classes.

### Task 2 — metrics.py emission rewiring [commit fad6a33]

- `from .metric_registry import validate_emission`; one-line `_emit(metrics)` helper (validates keys, returns dict) wired at every compute-path return point — calculate_metric_with_sklearn, classification_metrics, regression_metrics, multi_classification_metrics, multi_labels_metrics, token_classification_metrics, and all three metrics_for_dnabert2 arms (9 call sites, 7 compute paths).
- Emitted key spellings byte-identical to pre-change (all pre-existing metrics tests pass unchanged) — cross-lane contract with 10-01's evaluate() prefix-stripping preserved.
- D-08 owner-side docstring sweep of metrics.py: "DNA Language Model Evaluation Metrics Module" -> "DNA Large Language Model Evaluation Metrics Module" etc.; grep for old terminology returns nothing; module docstring documents the registry emission contract.
- `TestRegistryEmissionContract` in tests/tasks/test_metrics.py: 13 parameterized emission cases (binary/multiclass/multilabel +/- plot payloads, regression single- and multi-output +/- scatter, token, calculate_metric_with_sklearn, all three dnabert2 arms) asserting (a) every emitted key is canonical or payload-whitelisted, (b) no alias is ever emitted (disjointness against the union of all alias tuples), (c) negative probe — an injector at the real call site proves validate_emission fires on an unregistered key.
- tests/tasks/test_metrics.py module docstring swept to new terminology (the file 10-03's deferred table attributed to this lane).

### Task 3 — same-change proofs [commit aabba58]

- Coverage-row proof (ROADMAP SC 3, PITFALLS #2): `dnallm/tasks/metric_registry.py  146 stmts  1 miss  99%` — the row IS produced, proving placement outside the omit glob; `dnallm/tasks/metrics.py` at 100%.
- Full `uv run --no-sync pytest tests/tasks/ -q`: 192 passed.
- `uv run --no-sync python scripts/check_code.py`: SUCCESS (ruff green; mypy advisory per CI convention — module itself has zero mypy issues).
- CHANGELOG `(REV-02, R1-2d)` entry under `## [Unreleased]` / `### Added`, landed in the same commit as the registry (58bbf41, D-09 evidence chain).
- Cross-lane composite (run after both lanes landed): `pytest tests/tasks/test_metrics.py tests/finetune/test_trainer.py -k "TestRegistryEmissionContract or TestEvaluateSplit"` -> 31 passed; registry/trainer import assertion -> `CROSS-LANE-CONTRACT-OK`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FPR rate metric referenced an underscore-shadowed variable**
- **Found during:** Task 1 (ruff undefined-name)
- **Issue:** Underscore-prefixing unused unpacked confusion-matrix variables (ruff dummy-variable requirement) accidentally shadowed `tn`, which _fpr needs.
- **Fix:** Restored `tn, fp, _fn, _tp` unpacking in _fpr; ruff + hand-computed rate tests confirm correctness.
- **Files modified:** dnallm/tasks/metric_registry.py
- **Commit:** 58bbf41

**2. [Rule 3 - Blocking] pytest-cov dotted-target invocation crashes (pre-existing env issue)**
- **Found during:** Task 3 (coverage proof)
- **Issue:** `pytest --cov=dnallm.tasks.metric_registry` fails at conftest load: torch.overrides double-execution -> `RuntimeError: function '_has_torch_function' already has a docstring` (pandas/numpy ImportError is a masking cascade). Reproduces with ANY `--cov=dnallm.*` target (e.g. `--cov=dnallm.utils.sequence`), i.e. pre-existing and unrelated to this plan; venv unchanged since 2026-09-29.
- **Fix:** Equivalent proof via the coverage CLI (`python -m coverage run -m pytest ...` + `coverage report -m`) which produces the same term-report row evidence (99% / 100%). Environment issue NOT fixed in-phase (whole-repo scope); logged to phase deferred-items.md for owner disposition.
- **Files modified:** .planning/phases/10-evaluation-contract-layer-shared-scaffolding/deferred-items.md
- **Commit:** aabba58

### Plan-strengthening notes (not deviations)

- METRIC_REGISTRY typed as `MappingProxyType` instead of the plan's `dict[...]` annotation — implements the "registry surface is immutable" truth structurally (see key-decisions).
- TestValidateEmission class added beyond the plan's named class list — the plan's Task 3 action names validate_emission's branches as expected coverage gaps to close; covered here at 99% with the single unreachable-by-construction guard line documented.
- CHANGELOG.md committed with Task 1 (registry) rather than Task 3, satisfying the "same commit as the registry" requirement (D-09) under concurrent shared-tree constraints (no rebase/amend possible mid-wave).

## Auth Gates

None.

## Known Stubs

None — no stubs, no skipped tests, no unrun verify steps. All 67 + 70 new/updated tests run on the fast lane; no new skips introduced.

## Self-Check: PASSED

- Files exist: dnallm/tasks/metric_registry.py, tests/tasks/test_metric_registry.py, dnallm/tasks/metrics.py, tests/tasks/test_metrics.py (verified via ls/coverage report)
- Commits are ancestors of HEAD: 58bbf41, fad6a33, aabba58 (verified via git merge-base --is-ancestor)
- CHANGELOG entry grep = 1; tests/tasks/ suite 192 passed; check_code.py SUCCESS
