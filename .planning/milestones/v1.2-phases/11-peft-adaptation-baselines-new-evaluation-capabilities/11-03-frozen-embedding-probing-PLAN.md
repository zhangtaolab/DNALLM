---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/inference/probing.py
  - tests/inference/test_probing.py
  - CHANGELOG.md
autonomous: true
requirements: [PROB-01]
coupling_justified: >
  CHANGELOG.md is the single sanctioned cross-lane append surface (one REV-07
  unique-anchor bullet, D-09 discipline: re-read-before-edit, reviewer-comment id
  inline, never touching sibling lanes' entries). All other files are new and
  exclusively this lane's — probing.py + its test twin touch no other lane's file;
  the metric registry is consumed strictly read-only.

estimate:
  tokens: 24000
  raw_tokens: 24000
  tasks: 3
  confidence: low   # calibration sample_count=0, factor=1

must_haves:
  truths:
    - "Any model × any binary classification task runs frozen-embedding probing end-to-end: extract_embeddings → fit_probe → metrics emitted through the Phase-10 metric registry (PROB-01)"
    - "extract_embeddings reuses the existing hidden-states mechanics read-only (forward with output_hidden_states) with selectable layer (int, default last) and pooling (mean + cls) — no re-implementation of model-facing code (PROB-01)"
    - "fit_probe(kind='logistic'|'mlp') uses sklearn with fixed hyperparameters held as module-level constants with docstrings (D-13: LOGISTIC_MAX_ITER=1000, LOGISTIC_SOLVER='lbfgs', MLP_HIDDEN=(256,), MLP_EARLY_STOP=True) and NO ProbeConfig YAML surface"
    - "The StandardScaler is fit on the train split only — probe metrics never see test-split statistics (leakage test asserts disjoint-split discipline; PROB-01/Pitfall 8)"
    - "Embeddings cache to npz under output_dir/probe_cache/ keyed by (model, dataset, layer, pooling) — never CWD writes (D-12); second-run cache hits asserted AND a pooling/layer change MISSES the cache (same-change test, Pitfall 8)"
    - "Cache filenames are built from a sanitized hash of the 4-tuple key, never raw model/dataset ids (path-safety); writes are temp-file + os.replace (atomic) so concurrent same-key writes yield one valid winner (PROB-01 concurrency edge)"
    - "Embeddings are stored as float32 with the dtype recorded in the npz metadata; the reload roundtrip preserves dtype and values within allclose (PROB-01 precision edge)"
    - "extract_embeddings on a dataset yielding 0 rows after filtering raises a matchable ValueError — no empty cache file, no silent empty array (PROB-01 empty edge)"
    - "Every probe output row records layer and pooling alongside the metrics (comparability requirement, Pitfall 8)"
    - "The output schema is documented in the module docstring for the dnallmmark F4 lane"
  artifacts:
    - path: dnallm/inference/probing.py
      provides: "extract_embeddings (layer/pooling selectable, npz cache), fit_probe (logistic|mlp, fixed constants, train-only scaler), ProbeResult shape, F4 output-schema doc"
      contains: "fit_probe"
    - path: tests/inference/test_probing.py
      provides: "fast-lane probing tests on tiny_model_factory synthetic embeddings + slow-lane end-to-end"
      contains: "extract_embeddings"
  key_links:
    - from: dnallm/inference/probing.py
      to: dnallm/tasks/metric_registry.py
      via: "probe metrics emitted exclusively through metric_registry.resolve() (canonical AUROC/AUPRC spellings) — read-only consumption of the Phase-10 contract"
      pattern: "metric_registry"
    - from: dnallm/inference/probing.py
      to: dnallm/inference/inference.py
      via: "hidden-states extraction reuses the scoring/embedding mechanics (output_hidden_states path) — consumer, not reimplementer"
      pattern: "output_hidden_states"
    - from: dnallm/inference/probing.py
      to: output_dir/probe_cache/*.npz
      via: "cache keyed by (model, dataset, layer, pooling); mkdir parents=True exist_ok=True idiom; atomic os.replace writes"
      pattern: "probe_cache"
  prohibitions:
    - "No pyproject.toml changes (B5 solely owns pyproject this wave)"
    - "No new dnallm/__init__.py re-exports (facade stays byte-stable)"
    - "No writes to metric_registry.py or metrics.py — the registry is read-only for this lane"
    - "No CWD writes — cache lives under caller-supplied output_dir only (Phase-10 lesson IN-03)"
    - "No torch probe heads — sklearn LogisticRegression/MLPClassifier only (fixed-hyperparameter comparability)"
    - "No ProbeConfig YAML surface — constants live in probing.py (D-13)"
    - "No edits outside this lane's files (probing.py, its test twin, CHANGELOG append)"
    - "No Co-Authored-By trailers; commits are pathspec-limited (git commit -- <paths>) in the shared tree"
---

<objective>
B3 (REV-07): frozen-embedding probing — `extract_embeddings` with layer/pooling
selection and a (model, dataset, layer, pooling)-keyed npz cache, plus `fit_probe`
with fixed sklearn hyperparameters and train-only scaling, metrics through the shared
registry.

Purpose: PROB-01 — reviewers probe frozen DNA large language model embeddings at any
layer without fine-tuning, with leakage-free splits and a documented output schema the
dnallmmark F4 lane consumes.
Output: new dnallm/inference/probing.py, its test twin at the >= 96% standard,
slow-lane end-to-end acceptance, CHANGELOG entry.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-CONTEXT.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-RESEARCH.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-PATTERNS.md
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Probing end-to-end — tiny model → extract_embeddings (layer/pooling) → npz cache → fit_probe(logistic) → registry metrics</name>
  <reversibility rating="costly">The npz cache key contract (model, dataset, layer, pooling) and the F4-facing output schema are published for the dnallmmark lane to consume; changing the key later invalidates existing caches and downstream readers.</reversibility>
  <files>dnallm/inference/probing.py, tests/inference/test_probing.py</files>
  <read_first>
  - dnallm/inference/inference.py (lines 611-713: _setup_hidden_states_config + the embedding extraction path probing reuses read-only; lines 1746+: scoring entry)
  - dnallm/tasks/metric_registry.py (resolve/registered_names API at lines 380/420 — read-only consumption)
  - dnallm/inference/vep.py lines 1-44 (module docstring skeleton: numbered features + Example block — the style to mirror)
  - 11-RESEARCH.md sections: Code Examples (fit_probe fixed constants), Pitfall 8 (leakage + stale-cache), Architectural Responsibility Map (probing row)
  - 11-PATTERNS.md: probing.py analog (hidden-states mechanics + vep.py docstring skeleton + output-path discipline)
  - tests/conftest.py (tiny_model_factory, simple_dna_tokenizer fixtures)
  </read_first>
  <action>
  New module `dnallm/inference/probing.py` — one end-to-end path first: a tiny model's
  hidden states flow through extract_embeddings (default layer/pooling) into the npz
  cache, then fit_probe('logistic') emits binary metrics through the registry.

  Module skeleton (mirror vep.py's docstring style):
  - `from __future__ import annotations`; Google-style module docstring with numbered
    Features list, an `Example:` block, and the F4 output-schema documentation:
    every probe result row carries {model, dataset, layer, pooling, kind, metrics
    (canonical registry spellings: auroc, auprc, accuracy, ...), n_train, n_test,
    cache_hit} (schema documentation is a PROB-01 deliverable).
  - Relative imports inside dnallm/ (metric_registry via
    `from ..tasks.metric_registry import resolve`).

  extract_embeddings:
  - Signature shape: `extract_embeddings(model, tokenizer, sequences, labels, *,
    layer=-1, pooling="mean", model_name=None, dataset_name=None, output_dir=None)`
    returning a result object (dataclass) with the embeddings ndarray (float32),
    labels, and metadata carrying the full 4-tuple key.
  - Forward with output_hidden_states=True (reuse the model's existing forward-arg
    handling — introspect accepted args the same way inference.py does, do not
    re-implement); select `hidden_states[layer]`; pooling: "mean" (masked mean over
    sequence positions) and "cls" (first token). Unknown pooling string → matchable
    ValueError.
  - Cache (D-12): when output_dir is provided, write
    `{output_dir}/probe_cache/{sanitized_hash}.npz` — filename from a stable hash of
    (model_name, dataset_name, layer, pooling) (never raw ids — path safety); npz
    holds embeddings (float32), labels, and a metadata entry with the 4-tuple + dtype;
    write via unique temp name + os.replace (atomic, concurrency edge); read-back on
    next call with the same key = cache hit (recorded in metadata/result).
  - 0 rows after any filtering → matchable ValueError (empty edge).

  fit_probe:
  - `fit_probe(train_embeddings, train_labels, test_embeddings, test_labels, *,
    kind="logistic")` → ProbeResult dataclass.
  - Module-level constants with docstrings (D-13): LOGISTIC_MAX_ITER = 1000,
    LOGISTIC_SOLVER = "lbfgs", MLP_HIDDEN = (256,), MLP_EARLY_STOP = True.
  - StandardScaler fit on train only, applied to both; LogisticRegression or
    MLPClassifier from sklearn with exactly the constants; kind validated against
    {"logistic", "mlp"} with matchable ValueError.
  - Metrics exclusively via `metric_registry.resolve(...)` on the held-out
    predictions/scores (binary task: auroc, auprc, accuracy); pooling + layer +
    kind recorded in the result.

  Tests (tests/inference/test_probing.py, fast lane, tiny_model_factory +
  simple_dna_tokenizer, synthetic binary labels):
  - End-to-end: extract → fit('logistic') → metrics dict has registry-canonical keys;
    layer and pooling present in the result.
  - Cache hit: second extract_embeddings with identical 4-tuple hits the npz (no
    recompute — assert via cache_hit flag or patched forward counter).
  - Cache miss on change: altering pooling (or layer) MUST miss the cache (Pitfall 8
    same-change test).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - `dnallm/inference/probing.py` exists with the F4 schema documented in its module docstring; `grep -c "metric_registry" dnallm/inference/probing.py` >= 1 and `grep -c "probe_cache" dnallm/inference/probing.py` >= 1
  - `grep -c "LOGISTIC_MAX_ITER" dnallm/inference/probing.py` >= 2 (definition + use); constants carry docstrings
  - `uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"` passes with >= 6 tests including cache-hit and cache-miss-on-pooling-change
  - No CWD writes: cache paths always join a caller-supplied output_dir (grep shows no bare `Path("probe_cache")` / cwd-relative construction)
  </acceptance_criteria>
  <done>The full probing slice works on the fast lane: layer/pooling-selectable extraction, keyed atomic npz cache with hit/miss semantics, fixed-hyperparameter logistic probe, registry-emitted metrics.</done>
</task>

<task type="auto">
  <name>Task 2: Expansion — mlp kind, leakage discipline, edge battery, schema hardening</name>
  <files>dnallm/inference/probing.py, tests/inference/test_probing.py</files>
  <read_first>
  - dnallm/inference/probing.py (Task 1's module — extend in place)
  - 11-RESEARCH.md Pitfall 8 (leakage, stale cache, pooling comparability) + Validation Architecture B3 lane strategy
  </read_first>
  <action>
  Complete the module's behavior surface and drive it to the >= 96% standard:
  - kind='mlp': MLPClassifier path with exactly MLP_HIDDEN/MLP_EARLY_STOP; test that
    mlp and logistic both run end-to-end on synthetic embeddings and differ in their
    estimators (constants actually consumed).
  - Leakage test: construct embeddings where test-split statistics would change the
    scaling (e.g., shifted test distribution); assert the scaler is fit on train only
    — patch/wrap StandardScaler fit calls or assert via transformed-value checks on a
    disjoint-split construction (the test must fail if someone fits on all data).
  - Edge battery: 0-rows ValueError; unknown-pooling ValueError; unknown-kind
    ValueError; empty cache dir auto-created (parents=True, exist_ok=True); float32
    roundtrip (dtype preserved + allclose); atomic-write discipline (a temp file
    exists only transiently — assert os.replace usage or that the final npz is
    load-valid immediately after a simulated concurrent double-write with the same
    key).
  - Output rows: every result/row records layer + pooling + kind (comparability
    requirement).
  - Close any per-module coverage gaps found by
    `coverage report --include="dnallm/inference/probing.py"` until >= 96%.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary</fails_when>
    <automated>uv run --no-sync coverage run -m pytest tests/inference/test_probing.py -q && uv run --no-sync coverage report --include="dnallm/inference/probing.py"</automated>
    <fails_when>non-zero exit, or the probing.py coverage row shows a total below 96</fails_when>
  </verify>
  <acceptance_criteria>
  - All three ValueError edges (0-rows, unknown pooling, unknown kind) test-proven with matchable messages
  - Leakage test present and passing; scaler discipline enforced by construction
  - float32 roundtrip + atomic-write/concurrency behavior test-proven
  - probing.py per-module coverage >= 96% (coverage report row)
  </acceptance_criteria>
  <done>probing.py behavior-complete at the 96% standard with the full edge battery: both probe kinds, leakage-free scaling, dtype- and race-safe caching, documented schema.</done>
</task>

<task type="auto">
  <name>Task 3: Slow-lane end-to-end acceptance + CHANGELOG</name>
  <precondition>models.lock-pinned small model zhangtaolab/plant-dnabert-BPE (ms) is cached or the slow-lane network route is available; the binary dataset zhangtaolab/plant-multi-species-core-promoters (models.lock dataset row) is reachable.</precondition>
  <files>tests/inference/test_probing.py, CHANGELOG.md</files>
  <read_first>
  - tests/inference/test_inference_real_model.py (slow-lane real-model conventions in this directory)
  - models.lock (pinned model + dataset rows)
  - .planning/research/261009-paper-revision-suite-plan.md (reviewer-comment id to cite inline for REV-07)
  - CHANGELOG.md (## [Unreleased] anchor + entry format)
  </read_first>
  <action>
  Slow-lane acceptance (slow-marked, typed network skip when uncached — mirror this
  directory's existing real-model conventions):
  - Any model × any binary classification task end-to-end: load
    `zhangtaolab/plant-dnabert-BPE` (modelscope route, pinned), load the
    plant-multi-species-core-promoters binary task dataset, run extract_embeddings at
    TWO layers (last + one intermediate — the NT-paper layer finding makes the
    intermediate layer a real case, not decoration) with mean pooling into a
    tmp output_dir cache, fit_probe both kinds, and assert: registry-canonical metric
    keys present with float values, second extract call hits the cache, output rows
    carry layer/pooling/kind.
  - CHANGELOG (D-09 discipline): re-read, then append one bullet tagged
    `(REV-07, <reviewer-comment-id>)` under the idempotent `## [Unreleased]` anchor
    (### Added). Unique anchor; never touch sibling lanes' entries.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/inference/test_probing.py -q -m slow</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary</fails_when>
    <automated>test "$(grep -c "(REV-07," CHANGELOG.md)" -ge 1 && echo CHANGELOG-OK</automated>
    <fails_when>CHANGELOG-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - Slow test proves the any-model × binary-task contract on a pinned real model at two layers with cache-hit assertion
  - `grep -c "(REV-07," CHANGELOG.md` >= 1 under ## [Unreleased]
  - Fast lane still green after the additions: `uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"`
  </acceptance_criteria>
  <done>PROB-01 accepted end-to-end on a real model × binary task with the F4 schema, cache semantics, and registry metrics; CHANGELOG evidence chain extended.</done>
</task>

</tasks>

<artifacts_produced>
Symbols this plan creates (B3 lane):
- `dnallm/inference/probing.py` (new module): `extract_embeddings(...)`, `fit_probe(...)`, `ProbeResult` dataclass (or equivalent result object), module constants LOGISTIC_MAX_ITER / LOGISTIC_SOLVER / MLP_HIDDEN / MLP_EARLY_STOP (D-13)
- F4 output-schema documentation (module docstring): {model, dataset, layer, pooling, kind, metrics, n_train, n_test, cache_hit}
- Cache contract: output_dir/probe_cache/*.npz keyed by (model, dataset, layer, pooling), float32, atomic writes
- Test twin `tests/inference/test_probing.py` (fast-lane edge battery + slow-lane real-model acceptance)
- CHANGELOG.md: REV-07 entry under ## [Unreleased]
- No new public exports outside dnallm/inference/probing.py; registry consumed read-only
</artifacts_produced>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller output_dir → filesystem | probe cache writes under a caller-supplied path (V12) |
| dataset/model ids → cache filenames | ids are external strings folded into filesystem names |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-06 | Tampering | probe_cache path handling in probing.py | medium | mitigate | Cache filenames built from a sanitized stable hash of the key tuple — never raw model/dataset ids (no path traversal via ids containing separators); writes confined to caller output_dir (D-12, never CWD); parents=True exist_ok mkdir |
| T-11-07 | Information Disclosure | fit_probe scaler discipline | medium | mitigate | Scaler fit on train split only — probe metrics that silently encode test-split statistics are leaked results masquerading as held-out performance: leakage test enforced same-change (Pitfall 8) |
| T-11-SC | Tampering | package installs | high | mitigate | This lane installs nothing (scikit-learn/scipy existing dependencies; zero pyproject changes). The phase's only sanctioned install is plan 11-05's scikit-allel under owner decision D-08 |
</threat_model>

<verification>
- Fast lane: `uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"` — network-free
- Slow lane: `uv run --no-sync pytest tests/inference/test_probing.py -q -m slow` — real-model end-to-end at two layers
- Per-module coverage (cov-crash workaround): `uv run --no-sync coverage run -m pytest tests/inference/test_probing.py -q` then `uv run --no-sync coverage report --include="dnallm/inference/probing.py"` — >= 96%
- Invariants: `git diff HEAD -- pyproject.toml` and `git diff HEAD -- dnallm/__init__.py` and `git diff HEAD -- dnallm/tasks/metric_registry.py` all empty over this lane's commits; `grep -c "(REV-07," CHANGELOG.md` >= 1
- Owner directive 2026-10-09: run ONLY the targeted verifiers above and this lane's test files — no repo-wide lanes, no check_code.py full sweeps
</verification>

<success_criteria>
- PROB-01 fully test-proven: layer/pooling-selectable extraction over frozen embeddings, fixed-hyperparameter logistic+mlp probes with train-only scaling, registry metrics, keyed npz cache with hit/miss semantics, F4 schema documented, real-model end-to-end
- Every dnallm/ change shipped with its tests in the same commit; probing.py >= 96% per-module coverage
- No new dependencies, no facade re-exports, registry untouched, no out-of-lane edits
</success_criteria>

<output>
Create `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-03-SUMMARY.md` when done
</output>
