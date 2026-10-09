---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/tasks/metric_registry.py
  - dnallm/tasks/metrics.py
  - tests/tasks/test_metric_registry.py
  - tests/tasks/test_metrics.py
  - CHANGELOG.md
autonomous: true
requirements: [METR-01]
user_setup: []
estimate:
  tokens: 26000
  raw_tokens: 26000
  tasks: 3
  confidence: low   # 0 calibration samples (first phase of v1.2); factor 1 per estimate-calibration

must_haves:
  truths:
    # METR-01 core (goal-backward from ROADMAP SC 3)
    - "dnallm/tasks/metric_registry.py exists as a sibling of metrics.py, provably OUTSIDE the vendored dnallm/tasks/metrics/ coverage-omit glob: a coverage report row for metric_registry.py is produced (same-change proof, PITFALLS #2)"
    - "METRIC_REGISTRY is a single {canonical_name: (fn, aliases)} mapping; canonical names anchor CURRENT emitted spellings — accuracy, precision, recall, f1 (+_micro/_weighted/_samples variants), mcc, matthews_correlation, AUROC, AUPRC, AUROC_ovr, AUROC_ovo, TPR, TNR, FPR, FNR, mse, mae, r2, pearsonr, spearmanr — i.e. every key emitted by any metrics.py path"
    - "resolve(name) returns the canonical metric callable; unknown names raise a matchable ValueError listing the valid names (empty-input edge: resolve('') and any unregistered spelling raise)"
    - "Alias matching is exact and case-sensitive (encoding edge): 'eval_auroc' and 'eval_AUROC' are both registered aliases of AUROC and both resolve; any unregistered casing (e.g. 'Eval_Auroc') raises ValueError"
    - "Aliases are recognition-only and never emitted: no emitted key from any metrics.py path is an alias; the historical aliases eval_auroc, eval_auprc, eval_spearman_r, eval_pearson_r resolve to AUROC/AUPRC/spearmanr/pearsonr respectively"
    - "Registration invariants hold at construction (adjacency edge): no alias equals a canonical name of a different entry, no alias is duplicated across entries — violated tables raise ValueError when the module builds the registry, and a test feeds a deliberately colliding table to the builder to prove it"
    - "The registry surface is immutable (concurrency edge): no register/append/mutation API is exported, resolve() is side-effect-free (repeated calls leave METRIC_REGISTRY unchanged), and registered_names() returns the sorted canonical names — a stable documented order"
    - "The module is import-light: no module-level torch/sklearn imports (AST-level test); sklearn imports live inside the per-metric callables so resolution costs nothing until a metric is computed"
    - "metrics.py emits exclusively through the registry: every compute path (classification_metrics, regression_metrics, multi_classification_metrics, multi_labels_metrics, token_classification_metrics, calculate_metric_with_sklearn, metrics_for_dnabert2 arms) validates its emitted keys against the registry before returning; an unregistered key raises ValueError at emission time"
    - "Emitted keys are unchanged from today — the rewiring alters resolution, not output spellings (cross-lane contract with plan 10-01's evaluate() prefix-stripping)"
    - "Contract tests cover every metric key used across the benchmark task set: parameterized over all factories, emitted keys are a subset of registered_names() minus the documented non-metric payload keys ('curve', 'scatter')"
    - "metric_registry.py reaches >=96% line coverage via mocked fast-lane tests (per-module standard)"
    - "metrics.py docstrings carry no pre-sweep terminology (owner-side sweep per D-08 — A2 owns metrics.py)"
  artifacts:
    - dnallm/tasks/metric_registry.py  # new: METRIC_REGISTRY, resolve, canonical_name, registered_names, validate_emission
    - tests/tasks/test_metric_registry.py  # new: resolution, alias, invariant, immutability, import-light, per-metric invocation tests
    - dnallm/tasks/metrics.py  # emission rewired through the registry; docstring sweep
    - tests/tasks/test_metrics.py  # contract tests over all task types + alias-never-emitted
    - CHANGELOG.md  # REV-02 Unreleased entry in the same commit as the registry
  key_links:
    - "dnallm.tasks.metric_registry.resolve(name) -> callable usable as fn(y_true, y_pred_or_score) -> float (Phase 11 probing/VEP/sweep consume this surface read-only)"
    - "metrics.py factory returns -> metric_registry.validate_emission(keys) before every return (single name authority)"
    - "canonical spelling contract: registry canonical names == unprefixed current emitted keys == plan 10-01 evaluate() result-JSON keys"
  prohibitions:
    - "The registry must NEVER emit an alias — alias handling is one-directional (recognize historical, emit canonical only; PITFALLS #2 drift recurrence guard)"
    - "NO runtime registration/mutation API on the registry (frozen module-level contract surface)"
    - "metrics.py must not introduce any metric key absent from the registry (validate_emission enforces at emission time)"
    - "metric_registry.py must NOT import torch or sklearn at module level (import-light dnallmmark CI requirement)"
    - "The registry must NOT be placed inside dnallm/tasks/metrics/ (vendored coverage/ruff/mypy-omitted glob — PITFALLS #2)"
    - "No dnallm/__init__.py or dnallm/tasks/__init__.py re-export of the registry (import via full path dnallm.tasks.metric_registry; research ARCHITECTURE decision)"
    - "No new skips (registry tests are pure fast-lane; if a skip becomes necessary it is typed + allowlisted same-change)"
---

<objective>
Agent lane A2 (owner-fixed wave structure): land the METR-01 metric registry contract at dnallm/tasks/metric_registry.py — the single name authority every metric-emitting path resolves through — and rewire metrics.py to emit exclusively through it, with same-change proof the module is coverage-visible (outside the vendored omit glob) and import-light.

Purpose: METR-01 (REV-02, reviewer R1-2d) gates the entire dnallmmark re-run (F3 exporter) and is consumed read-only by Phase 11 probing/VEP/sweep lanes; landing it first means never re-touching outputs (research SUMMARY ordering rationale).
Output: new registry module + tests, rewired metrics.py + contract tests, CHANGELOG REV-02 entry.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-CONTEXT.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-PATTERNS.md
@.planning/research/SUMMARY.md
</context>

<tasks>

<task type="tracer">
  <name>Task 1: metric_registry.py — METRIC_REGISTRY with the full canonical set, resolve()/canonical_name(), lazy per-metric callables, construction invariants</name>
  <files>dnallm/tasks/metric_registry.py, tests/tasks/test_metric_registry.py</files>
  <read_first>
  - dnallm/tasks/metrics.py (whole file — enumerate every emitted key: 73-81, 128-149, 206-231, 271-318, 438-495, 549-554, 607-646; module docstring style 1-18)
  - 10-PATTERNS.md (metric_registry section — dispatcher analog at metrics.py:653-682, test analogs, module conventions)
  - .planning/research/PITFALLS.md Pitfall 2 (registry placement + one-directional alias contract)
  </read_first>
  <action>
  Create dnallm/tasks/metric_registry.py (sibling of metrics.py — NEVER inside dnallm/tasks/metrics/, the vendored omit glob). Module docstring in house style (summary + numbered features + Example block) declaring it the single metric name authority for dnallm and dnallmmark.

  Structure:
  1. Zero module-level heavy imports: only stdlib (typing). Each canonical metric callable is a small module-level function with signature (y_true, y_pred) -> float whose body lazily imports the sklearn metric it wraps (sklearn imports INSIDE the function — import-light contract). Score metrics (AUROC, AUPRC, AUROC_ovr, AUROC_ovo) take probabilities in y_pred; count metrics take hard predictions; continuous metrics (mse, mae, r2, pearsonr, spearmanr) take floats — document per-entry in the docstring.
  2. Canonical set = exactly the keys emitted anywhere in metrics.py today: accuracy, precision, recall, f1, f1_micro, f1_weighted, f1_samples, precision_micro, precision_weighted, precision_samples, recall_micro, recall_weighted, recall_samples, mcc, matthews_correlation, AUROC, AUPRC, AUROC_ovr, AUROC_ovo, TPR, TNR, FPR, FNR, mse, mae, r2, pearsonr, spearmanr.
  3. METRIC_REGISTRY: dict[str, tuple[Callable, tuple[str, ...]]] — {canonical_name: (fn, aliases)}. Alias table: the eval_-prefixed form of EVERY canonical name (current HF Trainer output spelling, e.g. eval_accuracy, eval_AUROC, eval_spearmanr) PLUS the historical spellings eval_auroc, eval_auprc, eval_spearman_r, eval_pearson_r mapping to AUROC/AUPRC/spearmanr/pearsonr. Matching is exact string, case-sensitive (encoding contract: eval_auroc and eval_AUROC are distinct registered aliases of the same target; anything else raises).
  4. Public API (module __all__): resolve(name) -> Callable (raises ValueError(f"Unknown metric name: '{name}'. Valid canonical names: {sorted(...)}") — matchable substring "Unknown metric name"); canonical_name(name) -> str (alias or canonical -> canonical; same ValueError on unknown); registered_names() -> tuple[str, ...] sorted (stable documented order); validate_emission(keys) -> None (raises ValueError naming the offending keys when any key is not a canonical name and not one of the documented non-metric payload keys {"curve", "scatter"}).
  5. Construction invariants: a private builder/validation helper checks that (a) no alias equals any canonical name of a DIFFERENT entry, (b) no alias appears under two entries, (c) canonical names are unique — raising ValueError on violation; METRIC_REGISTRY is built through it at import.

  tests/tasks/test_metric_registry.py (absolute imports, class-per-function grouping): TestResolve — resolve("AUROC") returns a callable; resolve("eval_auroc") and resolve("eval_AUROC") return the SAME object as resolve("AUROC"); resolve("") / resolve(None-typed-str like "bogus") / resolve("Eval_Auroc") raise ValueError match="Unknown metric name". TestCanonicalName — canonical_name("eval_spearman_r") == "spearmanr", canonical_name("eval_AUROC") == "AUROC", canonical_name("accuracy") == "accuracy". TestInvariants — the real registry passes all three invariants; feeding a deliberately colliding table (alias == another entry's canonical) to the private builder raises ValueError (adjacency edge). TestImmutability — resolve() called repeatedly leaves METRIC_REGISTRY unchanged (deep-compare keys and alias tuples before/after); the module exports no register/append function (concurrency edge). TestRegisteredNames — result is sorted and equals the canonical key set. TestImportLight — AST scan of the module source: no top-level Import/ImportFrom node names torch or sklearn. TestMetricCallables — parametrized over EVERY registered name with a minimal valid (y_true, y_pred) pair per metric family (hard labels for count metrics, probabilities for AUROC/AUPRC, multiclass labels+probs for *_micro/_weighted and AUROC_ovr/ovo, 2D multilabel arrays for *_samples, floats for regression metrics), asserting a finite float returns — this parameterization is the >=96% coverage driver.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/tasks/test_metric_registry.py -q</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or any parametrized metric-callable case fails</fails_when>
  </verify>
  <acceptance_criteria>
  - File exists at dnallm/tasks/metric_registry.py (verify with ls — it must NOT be under dnallm/tasks/metrics/)
  - grep -c "canonical" returns the mapping; __all__ exports exactly resolve, canonical_name, registered_names, validate_emission, METRIC_REGISTRY
  - A test asserts resolve("eval_auroc") is resolve("AUROC") (same object)
  - A test asserts the builder rejects a colliding alias table with ValueError
  - The AST test fails if any top-level torch/sklearn import is added to the module
  </acceptance_criteria>
  <done>The registry resolves every canonical name and all required aliases to lazy callables, rejects unknowns with a matchable ValueError, and is proven import-light and immutable.</done>
</task>

<task type="auto">
  <name>Task 2: metrics.py emits exclusively through the registry + contract tests over every task type</name>
  <files>dnallm/tasks/metrics.py, tests/tasks/test_metrics.py</files>
  <read_first>
  - dnallm/tasks/metrics.py (all return points: 73-81, 128-152, 196-233, 261-387, 414-497, 519-556, 607-650)
  - tests/tasks/test_metrics.py (existing class structure, mocked eval_pred tuples, `with patch("builtins.print")` convention)
  - dnallm/tasks/metric_registry.py (from Task 1 — the validate_emission surface)
  </read_first>
  <action>
  1. Wire emission validation: metrics.py imports `from .metric_registry import validate_emission` (relative, house convention; import-light so no cycle — metric_registry imports nothing from metrics.py). Every compute path validates its metric dict keys before returning: classification_metrics, regression_metrics, multi_classification_metrics, multi_labels_metrics, token_classification_metrics, calculate_metric_with_sklearn, and both metrics_for_dnabert2 arms call validate_emission(metrics.keys()) (or a one-line local helper `_emit(metrics)` that validates then returns metrics) immediately before each return. The plot branches may additionally carry the payload keys "curve"/"scatter" — already whitelisted in validate_emission. Emitted key spellings do NOT change (cross-lane contract with 10-01).

  2. Docstring sweep of metrics.py (D-08 owner-side, A2 owns this file): replace the four "DNA language model" variants per the D-08 mapping in the module and function docstrings ("DNA Language Model Evaluation Metrics Module" -> "DNA Large Language Model Evaluation Metrics Module", etc.). Add one line to the module docstring stating that all emitted keys are canonical registry names resolved through dnallm.tasks.metric_registry.

  3. tests/tasks/test_metrics.py additions — class TestRegistryEmissionContract: parameterized over the five factories (binary, multiclass, multilabel, regression single- and multi-output, token) plus calculate_metric_with_sklearn and the metrics_for_dnabert2 arms, using the file's existing mocked eval_pred tuples (with patch("builtins.print") where legacy prints fire): (a) every emitted key is in metric_registry.registered_names() or the payload whitelist; (b) NO emitted key is an alias — assert disjointness against the union of all alias tuples (alias-never-emitted, PITFALLS #2); (c) a negative probe: monkeypatching a factory closure to inject an unregistered key ("bogus_metric") makes validate_emission raise ValueError (proves the emission gate actually fires).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/tasks/test_metrics.py tests/tasks/test_metric_registry.py -q</automated>
    <fails_when>non-zero exit, or any TestRegistryEmissionContract case fails, or any pre-existing metrics test regresses</fails_when>
  </verify>
  <acceptance_criteria>
  - grep -n "validate_emission" dnallm/tasks/metrics.py shows the call wired at every compute-path return point (7 paths)
  - grep -rn "DNA language model\|DNA Language Model" dnallm/tasks/metrics.py returns nothing
  - A contract test proves every task-type factory's emitted keys are canonical registry names
  - A contract test proves no alias string ever appears in emitted keys
  - Emitted key sets are byte-identical to pre-change behavior for all five task types (tests unchanged except additions)
  </acceptance_criteria>
  <done>metrics.py resolves names exclusively through the registry with the emission gate proven at every path; contract tests cover the full benchmark metric key set.</done>
</task>

<task type="auto">
  <name>Task 3: Same-change proofs — coverage-row visibility (outside the omit glob), >=96% module coverage, CHANGELOG REV-02 entry</name>
  <files>tests/tasks/test_metric_registry.py, CHANGELOG.md</files>
  <read_first>
  - pyproject.toml ([tool.coverage.run] omit at 539-549 — the vendored glob the registry must sit outside; [tool.pytest.ini_options])
  - CHANGELOG.md (the ## [Unreleased] anchor created by plan 10-01 may or may not be present yet — the lanes run in parallel; re-read immediately before editing)
  </read_first>
  <action>
  1. Coverage-row proof (ROADMAP SC 3, PITFALLS #2): run the registry+metrics tests under coverage scoped to the two modules and confirm the term report SHOWS a row for dnallm/tasks/metric_registry.py (an omitted file produces no row — the row's presence is the same-change proof of placement). If any branch gaps keep metric_registry.py below 96%, add fast-lane tests to close them (typical gaps: validate_emission's payload-whitelist branch, canonical_name unknown-name branch, builder invariant branches — all coverable with pure unit tests, no mocks).

  2. CHANGELOG.md (deliberate shared append surface per D-09; re-read immediately before editing, insert with a unique anchor): ensure `## [Unreleased]` exists (create it above `## [0.7.1] - 2026-10-08` if absent, with `### Added`/`### Changed` subheadings). Under `### Added`: "- Metric registry contract at `dnallm.tasks.metric_registry`: a single {canonical: (fn, aliases)} registry with resolve()/canonical_name(); `dnallm.tasks.metrics` now emits exclusively canonical registry names, historical aliases (eval_auroc, eval_spearman_r, ...) are recognized but never emitted (REV-02, R1-2d)". Commit this entry IN THE SAME COMMIT as the registry work.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/tasks/test_metric_registry.py tests/tasks/test_metrics.py --cov=dnallm.tasks.metric_registry --cov=dnallm.tasks.metrics --cov-report=term -q</automated>
    <fails_when>the term table shows no row for dnallm/tasks/metric_registry.py (the module was omitted — placement failure), or its coverage percentage is below 96, or any test fails</fails_when>
    <automated>grep -c "(REV-02, R1-2d)" CHANGELOG.md</automated>
    <fails_when>count is 0 — the entry must exist under ## [Unreleased] in the same commit as the registry</fails_when>
  </verify>
  <acceptance_criteria>
  - The coverage term report contains a `dnallm/tasks/metric_registry.py` row at >=96%
  - CHANGELOG.md carries the (REV-02, R1-2d) entry under ## [Unreleased]
  - The full tests/tasks/ directory passes: uv run --no-sync pytest tests/tasks/ -q exits 0
  </acceptance_criteria>
  <done>The registry is proven coverage-visible (outside the vendored omit glob), at the per-module 96% standard, with the CHANGELOG evidence entry landed same-commit.</done>
</task>

</tasks>

## Artifacts this phase produces (plan 10-02)

- `dnallm/tasks/metric_registry.py` — new module
- `METRIC_REGISTRY: dict[str, tuple[Callable, tuple[str, ...]]]` (28 canonical names; eval_-prefixed + 4 historical aliases)
- `resolve(name) -> Callable` (matchable ValueError on unknown)
- `canonical_name(name) -> str`
- `registered_names() -> tuple[str, ...]` (sorted)
- `validate_emission(keys) -> None` (payload whitelist {"curve", "scatter"})
- metrics.py: `validate_emission` wiring at every compute-path return (emitted keys unchanged)
- Test classes: TestResolve, TestCanonicalName, TestInvariants, TestImmutability, TestRegisteredNames, TestImportLight, TestMetricCallables, TestRegistryEmissionContract

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| dnallmmark (companion repo) -> metric_registry | External consumer imports the registry as the cross-repo metric-name contract (read-only, no runtime input crosses back) |

## STRIDE Threat Register

Threat IDs continue after plan 10-01 (T-10-01..02, T-10-SC reserved).

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-03 | Tampering | METRIC_REGISTRY runtime mutation | medium | mitigate | No mutation API exported; immutability asserted by test; validate_emission raises on any producer introducing unregistered keys (drift becomes loud, not silent) |
| T-10-04 | Denial of Service | resolve() on hostile name strings | low | accept | Pure dict lookups on a 28-entry table; unknown names fail fast with ValueError; no dynamic imports of caller-supplied strings (sklearn imports are static, inside fixed wrappers) |
| T-10-SC | Tampering | package installs | high | accept | Phase 10 adds no dependencies (the milestone's single approved addition, scikit-allel, lands in Phase 11 B5); no package-manager install tasks in this plan |
</threat_model>

<verification>
- `uv run --no-sync pytest tests/tasks/ -q` green
- Coverage row proof above; module >=96%
- `uv run --no-sync python scripts/check_code.py` green (ruff line-length 100, mypy)
- Cross-lane contract: canonical names == current unprefixed emitted keys == plan 10-01's evaluate() JSON keys (phase verifier cross-checks after both lanes land)
</verification>

<success_criteria>
- METR-01 holds: single registry, resolve() with matchable ValueError, canonical-anchored current spellings, aliases recognized never emitted, metrics.py emitting exclusively through it, import-light, coverage-row visible, contract tests over the full benchmark key set
- CHANGELOG carries the (REV-02, R1-2d) entry in the same commit as the registry
</success_criteria>

<output>
Create `.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-02-SUMMARY.md` when done
</output>
