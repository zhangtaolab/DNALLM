---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: "04"
subsystem: testing
tags: [multi-seed, statistics, reproducibility, scipy, bootstrap, sweep]

# Dependency graph
requires:
  - phase: 10-evaluation-hardening
    provides: SweepConfig scaffold fields (seeds/out_root/n_bootstrap/bootstrap_seed/small_n_ci), DNATrainer.evaluate(split=) result-JSON writer whose shape the per-seed JSON mirrors
provides:
  - dnallm.finetune.sweep.aggregate_seeds — pure, torch-free, n-guarded mean/sd/ci95 statistics block
  - dnallm.finetune.sweep.run_seeds — {out_root}/{model}/{task}/seed_{s}/ directory protocol + aggregate statistics.json with D-16 same-split seed semantics
  - dnallm.finetune.sweep.run_sweep_from_config — SweepConfig adapter (fields consumed verbatim, zero configs.py edits)
  - Result-JSON statistics block spec {n_seeds, mean, sd, ci95, method} with method in none|t|omitted|bootstrap-percentile
affects: [phase-12 closeout, paper-revision reproducibility answer R1-2a]

# Actuals (#2632) — pairs with the plan's estimate to calibrate future estimates.
actuals:
  tokens: 9641     # chars/4 over the realized lane diff (sweep.py + test twin + CHANGELOG bullet)
  tasks: 3
  commits: 3       # lane commits 9e3a427, 99866ed, 3a719a7 (rev-list from ledger reports 7 — includes sibling lanes' interleaved commits on the shared branch)
plan_head_before: 0cb07808c8793522908121bace90924727bd273c
plan_head_after: 3a719a79493c5c8831231b5ecd98bea51991246f

# Tech tracking
tech-stack:
  added: []         # zero new dependencies (scipy.stats.t + numpy default_rng already core deps)
  patterns:
    - "Pure statistics function kept import-light (lazy get_logger import) so the aggregation layer is importable without torch"
    - "Per-seed paths stored relative to {model}/{task}/ inside statistics.json so equal runs are byte-identical across out_roots"
    - "Directory-segment sanitization with matchable ValueError at the config-strings-to-filesystem trust boundary (V12)"

key-files:
  created:
    - dnallm/finetune/sweep.py
    - tests/finetune/test_sweep.py
  modified:
    - CHANGELOG.md

key-decisions:
  - "Path separators in model_name/task_name are REJECTED (not auto-replaced) with a matchable ValueError; the slow trial passes the registry-style label 'plant-dnabert-BPE' rather than the slashed repo id"
  - "statistics.json carries per-seed paths relative to the {model}/{task}/ directory — the only way the byte-determinism test can compare two different tmp out_roots"
  - "Data preparation (sampling + split) uses a dedicated SWEEP_DATA_SEED=0, deliberately not one of the sweep seeds, so D-16 same-split semantics hold by construction"

patterns-established:
  - "n-guard boundary matrix: n=2/3/9/10 each test-proven to land in exactly its branch, plus n=1 sd=None and the small_n_ci=omit variant at n=9"
  - "Seeded percentile bootstrap reproducibility pair: identical (values, bootstrap_seed) -> identical ci95; different seed -> different ci95 on a chosen seed-sensitive array"

requirements-completed: [SEED-01]

coverage:
  - id: D1
    description: "aggregate_seeds pure statistics with the D-14 n-guard and documented statistics block spec"
    requirement: SEED-01
    verification:
      - kind: unit
        ref: "tests/finetune/test_sweep.py#TestAggregateSeeds (15 tests: exact moments, t-interval hand-computation, boundary matrix, bootstrap determinism)"
        status: pass
  - id: D2
    description: "run_seeds orchestration: directory protocol, D-16 same-split seed semantics, byte-deterministic statistics.json, V12 sanitization"
    requirement: SEED-01
    verification:
      - kind: unit
        ref: "tests/finetune/test_sweep.py#TestRunSeeds + TestRunSweepFromConfig (21 tests, stubbed fn, network-free)"
        status: pass
  - id: D3
    description: ">= 3-seed end-to-end acceptance trial on the models.lock-pinned small model with method=t statistics block"
    requirement: SEED-01
    verification:
      - kind: integration
        ref: "tests/finetune/test_sweep.py#test_sweep_three_seed_end_to_end (slow-marked, typed network skip)"
        status: pass
  - id: D4
    description: "sweep.py per-module coverage at the >= 96% standard"
    requirement: SEED-01
    verification:
      - kind: unit
        ref: "coverage run -m pytest tests/finetune/test_sweep.py -m 'not slow' -> coverage report --include=dnallm/finetune/sweep.py = 100% (112/112 stmts)"
        status: pass

metrics:
  duration: 14 min
  completed: 2026-10-09
  tests_added: 37
  coverage: "sweep.py 100% (112/112 statements)"

status: complete
---

# Phase 11 Plan 04: Multi-Seed Sweep Protocol Summary

Multi-seed sweep protocol (`run_seeds` + pure `aggregate_seeds`) with n-guarded t-interval/seeded-percentile-bootstrap uncertainty — the R1-2a reproducibility answer that never fakes precision at n=3.

## Accomplishments

- **`dnallm/finetune/sweep.py` (new, 416 lines):** `aggregate_seeds(values, *, n_bootstrap, bootstrap_seed, small_n_ci)` — pure numpy/scipy, torch-free (lazy `get_logger` import keeps module import light), implementing the RESEARCH code-example branch order exactly: n<3 -> ci95 None/method "none" (sd None at n=1); 3<=n<10 -> `scipy.stats.t.ppf(0.975, n-1)` Student-t interval (method "t"), or "omitted" under `small_n_ci="omit"`; n>=10 -> `np.random.default_rng(bootstrap_seed)` percentile bootstrap (method "bootstrap-percentile"). The statistics block spec `{n_seeds, mean, sd, ci95, method}` is documented in the module docstring as the SEED-01 deliverable.
- **`run_seeds(fn, seeds, out_root, *, model_name, task_name, ...)`:** directory protocol `{out_root}/{model}/{task}/seed_{s}/` (mkdir parents=True exist_ok=True), per-seed `seed_result.json` mirroring the Phase-10 `{split, timestamp, metrics}` result-JSON shape, and the aggregate `statistics.json` with per-seed paths (relative to `{model}/{task}/`, for byte-determinism across out_roots) plus the per-metric statistics blocks. D-16 same-split seed semantics documented in the docstring and structurally pinned: `run_seeds` takes no dataset argument and cannot re-split; the sweep seed threads init/shuffle only. Determinism contract documented as CPU-scoped (GPU kernel nondeterminism out of contract).
- **`run_sweep_from_config(sweep_config, fn, ...)`:** adapts `SweepConfig` verbatim (seeds/out_root positionally; n_bootstrap/bootstrap_seed/small_n_ci as aggregation knobs) — zero configs.py edits, honoring lane disjointness from B1.
- **V12 sanitization (T-11-08 mitigation):** empty/None out_root, path separators (`/`, `\`, `\0`), `.`/`..` segments, and empty seed lists all rejected with matchable `ValueError`s before any recursive mkdir; no CWD fallback anywhere.
- **`tests/finetune/test_sweep.py` (new, 37 tests):** 15 pure-function aggregation tests (exact moments [1,2,3]->mean 2.0/sd 1.0, t-interval bounds hand-computed via the same t.ppf, n=1/2/3/9/10 boundary matrix, omit-policy variants at n=2/9/10, bootstrap determinism + seed sensitivity on a chosen seed-sensitive array, invalid-policy/empty/non-flat rejections); 21 orchestration tests (directory protocol, byte-identical statistics.json across two out_roots, fn called exactly once per seed with (seed, seed_dir), path-separator/dot/empty rejections, metric_keys restrict/missing, non-numeric metric skip, adapter threading + omit policy + out_root/type guards); 1 slow-lane end-to-end trial.
- **Slow-lane acceptance (D-15):** seeds [42, 43, 44] of `zhangtaolab/plant-dnabert-BPE` (modelscope route, models.lock-pinned) fine-tuned via `run_seeds` over a core-promoters sample prepared ONCE with the dedicated data seed 0 (D-16); asserts seed_{s}/ dirs, both per-seed result JSONs (sweep's `seed_result.json` + trainer's `eval_test_result.json`), and statistics.json with n_seeds=3, method "t", finite ordered ci95 bounds over the canonical `AUROC` metric. Typed `network-unavailable:` skip when the pinned artifacts are uncached and the modelscope route is unreachable (socket reachability probe; prefix already allowlisted in expected_skips.yaml).
- **CHANGELOG.md:** one `(REV-09, R1-2a)` bullet appended under the `## [Unreleased]` / `### Added` anchor (D-09 discipline: re-read-before-edit, sibling lanes' entries untouched).

## Verification Results

| Verifier | Command | Result |
|----------|---------|--------|
| Task 1 aggregate | `uv run --no-sync pytest tests/finetune/test_sweep.py -q -k "aggregate"` | 15 passed |
| Task 2 fast lane | `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m "not slow"` | 36 passed, 1 deselected |
| Task 3 slow lane | `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m slow` | 1 passed, 36 deselected (3 real training runs, ~36s) |
| Coverage | `coverage run -m pytest ... -m "not slow" && coverage report --include="dnallm/finetune/sweep.py"` | 100% (112/112 statements) — >= 96% standard met |
| CHANGELOG anchor | `grep -c "(REV-09," CHANGELOG.md` | 1 (under `## [Unreleased]`) |
| Acceptance greps | `grep -c "t.ppf"`, `grep -c "default_rng"`, `grep -c "import torch"` on sweep.py | 1, 1, 0 |
| Lane invariants | my commits touch only sweep.py / test_sweep.py / CHANGELOG.md | confirmed via `git show --stat` per commit |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Bootstrap seed-sensitivity test initially used a degenerate array**
- **Found during:** Task 1 verify
- **Issue:** `[0.0..9.0]` and `[1.0]*9+[10.0]` both gave seed-invariant percentile bounds (binomial-granular / symmetric resample-mean distributions), so `different bootstrap_seed -> different ci95` failed.
- **Fix:** Used a distinct, spread array `[0.13, 0.87, 0.41, ...]` verified to be seed-sensitive across RNG streams (42 vs 7 differ), with a comment explaining why.
- **Files modified:** tests/finetune/test_sweep.py
- **Commit:** 9e3a427

**2. [Rule 1 - Bug] Slow trial asserted lowercase 'auroc' but the registry emits canonical 'AUROC'**
- **Found during:** Task 3 verify
- **Issue:** The metric-registry canonical names are Title-case; the auroc-block lookup returned None and failed the trial's final assertion (the sweep itself ran green end-to-end).
- **Fix:** Case-insensitive exact match `name.lower() == "auroc"`; trial then passed with all 12 canonical metrics aggregated.
- **Files modified:** tests/finetune/test_sweep.py
- **Commit:** 3a719a7

**3. [Rule 2 - Correctness] Extra input validation at the sweep boundaries**
- **Found during:** Task 2 implementation
- **Issue:** The plan signature passes caller strings straight into a recursive mkdir; RESEARCH's example lacked guards for invalid `small_n_ci` on direct (non-Pydantic) calls, empty seeds, non-flat/empty value arrays, and non-numeric metric values.
- **Fix:** Matchable `ValueError`s for each (all test-proven); non-numeric per-seed metrics are skipped from aggregation with a logged reason instead of crashing the JSON writer.
- **Files modified:** dnallm/finetune/sweep.py
- **Commit:** 99866ed

### Deferred Issues

None. (mypy's numpy-stub `type` statement error is pre-existing repo-wide — reproduced identically on `dnallm/finetune/trainer.py` before this lane — and advisory per CI `|| true`; out of scope.)

## Known Stubs

None — no placeholder logic shipped.

## Threat Flags

None — no security-relevant surface beyond the plan's threat model. T-11-08 (out_root tampering) mitigated as planned; T-11-09 (vacuous CI) mitigated by the test-proven n-guard boundary matrix; T-11-SC honored (zero installs, zero pyproject changes).

## Commits

| Task | Commit | Subject |
|------|--------|---------|
| 1 | 9e3a427 | feat(11-04): aggregate_seeds pure statistics with the D-14 n-guard |
| 2 | 99866ed | feat(11-04): run_seeds orchestration with D-16 same-split seed semantics |
| 3 | 3a719a7 | test(11-04): slow-lane 3-seed end-to-end trial + CHANGELOG entry |

## Self-Check: PASSED

All four lane artifacts exist on disk (sweep.py, test_sweep.py, CHANGELOG.md, this SUMMARY); all three task commits (9e3a427, 99866ed, 3a719a7) are ancestors of HEAD; the lane's worktree paths are clean (all changes committed).
