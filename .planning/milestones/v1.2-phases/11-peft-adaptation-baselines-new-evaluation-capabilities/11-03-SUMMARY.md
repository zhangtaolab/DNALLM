---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 11-03
subsystem: inference/probing
tags: [probing, embeddings, sklearn, cache, metrics-registry, prob-01]
requires:
  - dnallm/tasks/metric_registry.py (resolve/validate_emission — read-only Phase-10 contract)
  - output_hidden_states forward mechanics (inference.py idiom — consumed read-only)
provides:
  - dnallm.inference.probing.extract_embeddings (layer/pooling-selectable, float32 npz probe_cache keyed by (model, dataset, layer, pooling), atomic writes)
  - dnallm.inference.probing.fit_probe (logistic|mlp, fixed D-13 constants, train-only StandardScaler, registry-canonical metrics)
  - ProbeResult.to_row() — F4 output schema {model, dataset, layer, pooling, kind, metrics, n_train, n_test, cache_hit}
  - tests/inference/test_probing.py (fast-lane edge battery + slow-lane real-model acceptance)
affects: []
tech-stack:
  added: []
  patterns:
    - sanitized-hash cache filenames + temp-file/os.replace atomic writes
    - forward-signature introspection incl. **kwargs absorption (transformers 5.x)
key-files:
  created:
    - dnallm/inference/probing.py
    - tests/inference/test_probing.py
  modified:
    - CHANGELOG.md
decisions:
  - forward kwarg passes output_hidden_states=True when the signature has the named param OR var-keywords (transformers 5.x task heads absorb it via **kwargs); config flag set best-effort with warning degradation
  - cache filenames are sha256 hex of the key-tuple repr (path safety, T-11-06); cache key mismatch or corrupt bytes degrade to a miss, never a failure
  - fit_probe accepts optional layer/pooling/model/dataset/cache_hit metadata so every F4 row is comparable; scalers/estimators exposed on ProbeResult for introspection
metrics:
  duration: 977s
  completed: 2026-10-09
  tests: 35 (34 fast + 1 slow)
  coverage: 100% (dnallm/inference/probing.py, 199/199 statements)
status: complete
actuals:
  tokens: 14800   # chars/4 over probing.py + test_probing.py + CHANGELOG bullet
  tasks: 3
  commits: 3      # lane-scoped; shared-tree rev-list 24d90cb..HEAD = 11 includes sibling lanes
plan_head_before: 24d90cb
plan_head_after: 2b6f55e
---

# Phase 11 Plan 03: Frozen-Embedding Probing Summary

**One-liner:** layer/pooling-selectable frozen-embedding extraction with a (model, dataset, layer, pooling)-keyed atomic npz cache, plus fixed-hyperparameter logistic/MLP probes with train-only scaling emitting registry-canonical metrics — 100% module coverage and a pinned real-model × binary-task acceptance at two layers.

## What Was Built

### dnallm/inference/probing.py (new, 199 statements, 100% coverage)

- `extract_embeddings(model, tokenizer, sequences, labels, *, layer=-1, pooling="mean", model_name=None, dataset_name=None, output_dir=None, batch_size=32)` → `EmbeddingResult` (float32 `(n, d)` embeddings, aligned labels, full 4-tuple key metadata).
  - Hidden-states mechanics reused read-only from the `DNAInference` idiom: forward-signature introspection + config flag; layer selection over the returned hidden-state stack with bounds-checked matchable `ValueError`; masked-mean and cls pooling; attention-mask fallback chain (tokenizer mask → pad-id derivation → all-ones).
  - Cache (D-12): `output_dir/probe_cache/probe_<sha256[:24]>.npz` — filenames are hashes of the key tuple, never raw ids (T-11-06); npz carries embeddings + labels + JSON metadata (4-tuple + dtype); writes go through a unique temp file + `os.replace` (concurrent same-key writes leave exactly one valid winner); unreadable or key-mismatched entries degrade to misses with a warning.
  - 0 rows after empty-sequence filtering → matchable `ValueError`; no cache file written for that case.
- `fit_probe(train_X, train_y, test_X, test_y, *, kind="logistic", layer=None, pooling=None, model_name=None, dataset_name=None, cache_hit=None)` → `ProbeResult`.
  - D-13 constants with docstrings: `LOGISTIC_MAX_ITER=1000`, `LOGISTIC_SOLVER="lbfgs"`, `MLP_HIDDEN=(256,)`, `MLP_EARLY_STOP=True` — no YAML surface; tests assert the estimators consume exactly these.
  - `StandardScaler` fit on the train split only (leakage discipline, T-11-07), applied to both splits; metrics exclusively via `metric_registry.resolve` (`AUROC`, `AUPRC`, `accuracy`) with `validate_emission` guarding canonical spellings.
- F4 output schema documented in the module docstring and emitted by `ProbeResult.to_row()`: `{model, dataset, layer, pooling, kind, metrics, n_train, n_test, cache_hit}`.

### tests/inference/test_probing.py (new, 35 tests)

- Fast lane (34, network-free, `tiny_model_factory` + `simple_dna_tokenizer`): end-to-end logistic probe with registry-canonical keys; cache-hit on identical key; cache-miss on pooling change and on layer change (Pitfall 8 same-change tests); cls-vs-mean divergence; F4 row schema; locked constants; mlp-vs-logistic estimator divergence + consumed-constants; leakage (spy on `StandardScaler.fit` records exactly the train row count, `scaler.mean_` equals train-only statistics); edge battery (0-rows / unknown-pooling / unknown-kind / layer-out-of-range / bool-layer / length-mismatch / no-hidden-states / read-only-config / strict-signature / pad-id and ones mask fallbacks / parents=True cache creation / float32 roundtrip / concurrent atomic double-write / failed-replace cleanup / corrupt + key-mismatched cache / multiclass matrix branch / hidden-state precedence chain).
- Slow lane (1, `@pytest.mark.slow`, 1800s timeout): `zhangtaolab/plant-dnabert-BPE` (modelscope, pinned) × `zhangtaolab/plant-multi-species-core-promoters` binary task — balanced 60-row subsample, extraction at last + one intermediate layer, both probe kinds, registry-canonical float metrics, two cache entries, second extract hits the cache; typed `network-unavailable:` skip when uncached (prefix already allowlisted in `tests/expected_skips.yaml`).

### CHANGELOG.md

One `(REV-07, R2-2)` bullet appended under the idempotent `## [Unreleased]` / `### Added` anchor (D-09 discipline; sibling lanes' entries untouched).

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| Fast lane | `uv run --no-sync pytest tests/inference/test_probing.py -q -m "not slow"` | 34 passed |
| Slow lane | `uv run --no-sync pytest tests/inference/test_probing.py -q -m slow` | 1 passed |
| Per-module coverage | `coverage run -m pytest tests/inference/test_probing.py -q` + `coverage report --include="dnallm/inference/probing.py"` | 100% (199/199) |
| CHANGELOG evidence | `grep -c "(REV-07," CHANGELOG.md` | 1 |
| Lane invariants | per-commit `--name-only` over 0cb0780, 691809d, 2b6f55e | only probing.py / test_probing.py / CHANGELOG.md — pyproject.toml, dnallm/__init__.py, dnallm/tasks/metric_registry.py untouched |
| Style | `ruff format --check` + `ruff check` on lane files | clean |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] transformers 5.x heads absorb `output_hidden_states` via `**kwargs`**
- **Found during:** Task 3 (slow-lane acceptance run)
- **Issue:** The inference.py-style named-parameter signature check (`"output_hidden_states" in params`) is false for transformers 5.17 task heads (e.g. `BertForSequenceClassification`), so the kwarg was never passed and the real-model forward returned `hidden_states=None`, tripping the module's own matchable "no hidden states" ValueError.
- **Fix:** The kwarg is now also passed when the forward signature carries var-keywords, and the config flag is set best-effort (warning degradation) whenever the model has a config. A strict-signature fast-lane regression test (no network) covers the neither-param-nor-kwargs branch.
- **Files modified:** dnallm/inference/probing.py, tests/inference/test_probing.py
- **Commit:** 2b6f55e

**2. [Rule 1 - Bug] explicit `scaler.fit()` instead of `fit_transform()`**
- **Found during:** Task 2 (leakage-test design)
- **Issue:** `fit_transform` obscures which array the scaler's `fit` sees; the leakage contract is greppable and spy-testable only with an explicit train-split fit call.
- **Fix:** `scaler.fit(x_train)` followed by `transform` on both splits — behaviorally identical, contract-explicit.
- **Files modified:** dnallm/inference/probing.py
- **Commit:** 691809d

Otherwise the plan executed as written; no architectural changes, no new dependencies, no registry writes, no facade re-exports.

## Commits

- 0cb0780 — feat(11-03): probing end-to-end slice — extract_embeddings with layer/pooling + npz cache + fit_probe(logistic)
- 691809d — test(11-03): probing expansion — mlp kind, leakage discipline, edge battery, 100% module coverage
- 2b6f55e — feat(11-03): slow-lane probing acceptance on plant-dnabert-BPE + CHANGELOG REV-07 entry

## Self-Check: PASSED

- dnallm/inference/probing.py — FOUND (committed)
- tests/inference/test_probing.py — FOUND (committed)
- CHANGELOG.md REV-07 bullet — FOUND (grep count 1)
- Commits 0cb0780, 691809d, 2b6f55e — all ancestors of HEAD

## Notes for the Verifier

- The slow test ran against the locally cached snapshot (353M safetensors present); its typed skip fires only on a cold, network-less box and matches the existing `network-unavailable:` allowlist prefix — no `expected_skips.yaml` edit was required.
- `models.lock` rows for both the model (line 14) and the dataset (line 36) pre-existed; no new rows needed.
- The shared-tree `rev-list` count (11) includes sibling lanes' interleaved commits; this lane's scoped commit count is 3, listed above.
