---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/finetune/sweep.py
  - tests/finetune/test_sweep.py
  - CHANGELOG.md
autonomous: true
requirements: [SEED-01]
coupling_justified: >
  CHANGELOG.md is the single sanctioned cross-lane append surface (one REV-09
  unique-anchor bullet, D-09 discipline: re-read-before-edit, reviewer-comment id
  inline — R1-2a is the known thread for the reproducibility answer — never touching
  sibling lanes' entries). All other files are new and exclusively this lane's.
  trainer.py is NOT modified: run_seeds orchestrates DNATrainer from outside (the
  Phase-10 evaluate(split=)/result-JSON contract is consumed, not changed), keeping
  this lane file-disjoint from B1's hot files.

estimate:
  tokens: 20000
  raw_tokens: 20000
  tasks: 3
  confidence: low   # calibration sample_count=0, factor=1

must_haves:
  truths:
    - "aggregate_seeds is a pure function returning mean/sd/ci95 with the n-guard: n<3 → ci95=None (method 'none'), 3<=n<10 → scipy t-interval (method 't', or 'omitted' when small_n_ci='omit'), n>=10 → seeded percentile bootstrap (method 'bootstrap-percentile') — never a vacuous bootstrap at n=3 (D-14, SEED-01)"
    - "The n-guard boundary matrix is test-proven at n=2, n=3, n=9, n=10 — each n lands in exactly its branch (SEED-01 n-guard boundary edge)"
    - "The bootstrap is seeded via np.random.default_rng(bootstrap_seed): identical seeds+data → identical ci95 (reproducible CI); constructed known arrays reproduce exact mean/sd/t-interval values (SEED-01 acceptance)"
    - "run_seeds writes the {model}/{task}/seed_{s}/ directory protocol under out_root (mkdir parents=True exist_ok=True) and produces a result-JSON statistics block per the documented spec (SEED-01)"
    - "run_seeds threads ONE seed into every stochastic stage with same-data-split-across-seeds semantics: the split seed is derived from the dataset, not the sweep seed, so seed-to-seed variance measures init/shuffle only (D-16) — documented in the docstring and asserted by a same-change determinism test"
    - "Same seeds → identical CPU outputs (determinism test with a stubbed fn); the determinism guarantee is CPU-scoped, with GPU kernel nondeterminism explicitly out of contract (documented)"
    - "A >= 3-seed trial of one small task completes end-to-end with the result-JSON statistics block present (D-15, SEED-01 acceptance)"
    - "The aggregation layer is import-light: aggregate_seeds is testable without torch (pure numpy/scipy)"
  flagged_assumptions:
    - "GPU same-seed determinism is NOT guaranteed (CUDA kernel nondeterminism): the determinism contract is CPU-canonical by design; GPU runs report this limitation in the docstring. Owner-facing: if a reviewer demands GPU determinism, that is a future requirement (cublas workspace pinning), not this milestone."
  artifacts:
    - path: dnallm/finetune/sweep.py
      provides: "aggregate_seeds (pure, n-guarded), run_seeds (directory protocol, D-16 seed semantics), statistics JSON block spec"
      contains: "aggregate_seeds"
    - path: tests/finetune/test_sweep.py
      provides: "pure-function aggregation tests on constructed arrays + determinism tests + slow >= 3-seed end-to-end"
      contains: "run_seeds"
  key_links:
    - from: dnallm/finetune/sweep.py
      to: dnallm/finetune/trainer.py
      via: "run_seeds orchestrates DNATrainer train/evaluate per seed (consumer, not modifier — B1 owns trainer.py this wave); per-seed result JSON mirrors the Phase-10 eval_{split}_result.json shape {split, timestamp, metrics}"
      pattern: "DNATrainer"
    - from: dnallm/finetune/sweep.py
      to: dnallm/configuration/configs.py
      via: "SweepConfig consumed verbatim (seeds, out_root, n_bootstrap, bootstrap_seed, small_n_ci — Phase-10 scaffold fields)"
      pattern: "SweepConfig"
    - from: dnallm/finetune/sweep.py
      to: scipy.stats.t / numpy.random.default_rng
      via: "t.ppf(0.975, n-1) for the small-n interval; seeded percentile bootstrap for n>=10 (Don't Hand-Roll table)"
      pattern: "t.ppf"
  prohibitions:
    - "No pyproject.toml changes (B5 solely owns pyproject this wave)"
    - "No new dnallm/__init__.py re-exports (facade stays byte-stable)"
    - "No edits to trainer.py or configs.py — SweepConfig is consumed as-is; orchestration stays outside the trainer (lane disjointness from B1)"
    - "No statsmodels/BCa bootstrap — scipy t-interval + numpy seeded percentile bootstrap only (D-14; STACK rejected list)"
    - "No manual t-tables or normal approximations for the CI"
    - "No CWD writes — out_root comes from the caller/config, validated before recursive mkdir"
    - "No ci95 emitted for n<3 in any branch (the vacuous-precision trap)"
    - "No edits outside this lane's files (sweep.py, its test twin, CHANGELOG append)"
    - "No Co-Authored-By trailers; commits are pathspec-limited (git commit -- <paths>) in the shared tree"
---

<objective>
B4 (REV-09): multi-seed sweep protocol — `run_seeds` with the {model}/{task}/seed_{s}/
directory protocol and D-16 same-split seed semantics, plus `aggregate_seeds` as a
pure function returning mean/sd/ci95 via the n-guarded t-interval / seeded percentile
bootstrap, and the result-JSON statistics block.

Purpose: SEED-01 — the reviewer reproducibility answer (R1-2a): multi-seed runs with
honest uncertainty aggregates that never fake precision at n=3.
Output: new dnallm/finetune/sweep.py (import-light aggregation), its test twin at the
>= 96% standard, slow >= 3-seed end-to-end trial, CHANGELOG entry.
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
  <name>Task 1: aggregate_seeds end-to-end — pure function, constructed arrays, n-guard matrix, seeded bootstrap</name>
  <reversibility rating="costly">The statistics block schema ({n_seeds, mean, sd, ci95, method}) is the result-JSON contract dnallmmark F2 reads and the reviewer-facing reproducibility answer; changing it later breaks downstream consumers.</reversibility>
  <files>dnallm/finetune/sweep.py, tests/finetune/test_sweep.py</files>
  <read_first>
  - 11-RESEARCH.md Code Examples: the aggregate_seeds n-guard implementation (verbatim starting point — copy, do not redesign) and the SweepConfig field verification (configs.py:455-498)
  - dnallm/configuration/configs.py (lines 455-498: SweepConfig — seeds, out_root, n_bootstrap, bootstrap_seed, small_n_ci with pattern ^(t-interval|omit)$)
  - dnallm/inference/vep.py lines 1-44 (module docstring skeleton style to mirror)
  - 11-RESEARCH.md Pitfall 9 (seed illusion, vacuous bootstrap) + Don't Hand-Roll rows (t-interval, bootstrap RNG)
  - 11-PATTERNS.md: sweep.py analog (trainer result-JSON writer idiom + import-light structure)
  </read_first>
  <action>
  New module `dnallm/finetune/sweep.py` — aggregation first (pure, torch-free):
  - Module docstring: numbered Features + Example block; documents the result-JSON
    `statistics` block spec: {n_seeds, mean, sd, ci95, method} where method is one of
    "none" | "t" | "omitted" | "bootstrap-percentile" — this spec IS a SEED-01
    deliverable.
  - `aggregate_seeds(values, *, n_bootstrap, bootstrap_seed, small_n_ci)` implementing
    the RESEARCH Code Example shape exactly: values → float ndarray; n<3 →
    ci95=None, method="none" (sd None at n=1); 3<=n<10 with small_n_ci="t-interval" →
    sem = std(ddof=1)/sqrt(n), ci95 = mean ± scipy.stats.t.ppf(0.975, n-1)*sem,
    method="t"; small_n_ci="omit" in that range → ci95=None, method="omitted";
    n>=10 → rng = np.random.default_rng(bootstrap_seed), resample indices
    (n_bootstrap, n), percentile [2.5, 97.5] of resampled means, method
    "bootstrap-percentile". Import scipy/numpy lazily or module-level — but keep the
    function torch-free and import-light (testable without torch).
  - Tests (tests/finetune/test_sweep.py, pure-function, no models):
    - Constructed known arrays: e.g. [1.0, 2.0, 3.0] → mean 2.0, sd 1.0, t-interval
      bounds hand-computable via the same t.ppf — assert exact floats.
    - Boundary matrix: n=2 → method "none"/ci95 None; n=3 → "t"; n=9 → "t"; n=10 →
      "bootstrap-percentile"; n=9 with small_n_ci="omit" → "omitted".
    - Bootstrap determinism: identical (values, bootstrap_seed) → identical ci95
      twice; different bootstrap_seed → (almost surely) different ci95.
    - n=1: sd None, ci95 None, method "none".
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_sweep.py -q -k "aggregate"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - `dnallm/finetune/sweep.py` exists; `grep -c "t.ppf" dnallm/finetune/sweep.py` >= 1 and `grep -c "default_rng" dnallm/finetune/sweep.py` >= 1
  - `grep -c "import torch" dnallm/finetune/sweep.py` returns 0 (aggregation layer torch-free)
  - The n=2/3/9/10 matrix and the exact-moments tests pass; bootstrap determinism proven
  - `uv run --no-sync pytest tests/finetune/test_sweep.py -q -k "aggregate"` shows >= 8 passed
  </acceptance_criteria>
  <done>aggregate_seeds lands as a pure, exact, n-guarded function with the statistics block spec documented and every boundary test-proven.</done>
</task>

<task type="auto">
  <name>Task 2: run_seeds orchestration — directory protocol, D-16 seed semantics, determinism test</name>
  <files>dnallm/finetune/sweep.py, tests/finetune/test_sweep.py</files>
  <read_first>
  - dnallm/finetune/trainer.py (lines 559-660: evaluate(split=) entry + result-JSON writer — the per-seed contract consumed read-only; DNATrainer.__init__ seed plumbing)
  - dnallm/datahandling/data.py (lines ~817/823: train_test_split seed parameter — the caller-supplied split seed D-16 pins)
  - 11-RESEARCH.md Pitfall 9 (split seed is NOT auto-derived — the seed illusion) + D-16
  </read_first>
  <action>
  Extend sweep.py with the orchestration layer:
  - `run_seeds(fn, seeds, out_root, *, model_name, task_name)` where `fn(seed,
    seed_dir) -> dict` (or a JSON-path return) runs ONE fully-seeded training run —
    in the acceptance, fn constructs a DNATrainer with train seed=s over a PRE-SPLIT
    dataset. Directory protocol: `{out_root}/{model_name}/{task_name}/seed_{s}/`
    (mkdir parents=True, exist_ok=True — the trainer's own output_dir discipline);
    sanitize model/task name path segments (reject or replace path separators with a
    matchable ValueError — V12).
  - D-16 semantics, enforced by contract and documented in the run_seeds docstring:
    the dataset split is performed ONCE by the caller (outside fn's seed dependence)
    or with a dataset-derived fixed seed; the sweep seed threads only into
    init/shuffle stages. run_seeds must not itself re-split per seed; the docstring
    states this and the determinism test pins it.
  - Collector: after all seeds, write
    `{out_root}/{model_name}/{task_name}/statistics.json` containing the statistics
    block (Task 1 spec) over each seed's chosen metric, plus per-seed paths — JSON
    written with the trainer's indent=2 + parents=True idiom.
  - SweepConfig consumption: a small loader/adapter reading SweepConfig
    (seeds/out_root/n_bootstrap/bootstrap_seed/small_n_ci) into run_seeds +
    aggregate_seeds arguments (consumed verbatim — no configs.py edits).
  - Determinism tests (fast lane, stubbed fn): same seeds + stubbed deterministic fn
    → identical statistics.json bytes (run twice into two tmp out_roots); the
    directory protocol asserted ({model}/{task}/seed_42/ exists; a path-separator in
    model_name raises the matchable ValueError).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_sweep.py -q -m "not slow"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - run_seeds exists with the documented D-16 docstring; statistics.json written under {model}/{task}/
  - Determinism test passes: identical seeds + deterministic fn → byte-identical statistics.json across two runs
  - Path-segment sanitization raises matchable ValueError on separators (test-proven)
  - `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m "not slow"` fully green (>= 12 tests cumulative)
  </acceptance_criteria>
  <done>run_seeds orchestrates the locked directory protocol with same-split seed semantics, deterministic outputs, and the statistics block on disk — all fast-lane proven.</done>
</task>

<task type="auto">
  <name>Task 3: Slow-lane >= 3-seed end-to-end trial + coverage + CHANGELOG</name>
  <precondition>models.lock-pinned small model zhangtaolab/plant-dnabert-BPE (ms) and the dataset zhangtaolab/plant-multi-species-core-promoters are cached or the slow-lane network route is available (same pinned small models as the IA³ acceptance, D-15).</precondition>
  <files>tests/finetune/test_sweep.py, CHANGELOG.md</files>
  <read_first>
  - tests/finetune/test_trainer_real_model.py (slow-lane conventions: model + dataset loading, tiny epoch budgets)
  - models.lock (pinned model + dataset rows)
  - .planning/research/261009-paper-revision-suite-plan.md (reviewer-comment id for REV-09 — the R1-2a reproducibility thread)
  - CHANGELOG.md (## [Unreleased] anchor + entry format)
  </read_first>
  <action>
  Slow-lane acceptance (slow-marked, typed network skip when uncached):
  - >= 3-seed end-to-end trial of one small binary task (D-15): pre-split
    plant-multi-species-core-promoters ONCE (fixed dataset seed — D-16), then for
    each of seeds [42, 43, 44] run a tiny DNATrainer fine-tune of
    `zhangtaolab/plant-dnabert-BPE` (modelscope route, pinned) via run_seeds into a
    tmp out_root; assert the {model}/{task}/seed_{s}/ dirs exist, each holds the
    per-seed result JSON, and statistics.json carries the statistics block with
    method "t" (n=3) and finite ci95 bounds.
  - Per-module coverage: close gaps until sweep.py >= 96% via
    coverage run/report (cov-crash workaround — never pytest --cov).
  - CHANGELOG (D-09 discipline): re-read, then append one bullet tagged
    `(REV-09, R1-2a)` under the idempotent `## [Unreleased]` anchor (### Added).
    Unique anchor; never touch sibling lanes' entries.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_sweep.py -q -m slow</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary</fails_when>
    <automated>uv run --no-sync coverage run -m pytest tests/finetune/test_sweep.py -q -m "not slow" && uv run --no-sync coverage report --include="dnallm/finetune/sweep.py"</automated>
    <fails_when>non-zero exit, or the sweep.py coverage row shows a total below 96</fails_when>
    <automated>test "$(grep -c "(REV-09," CHANGELOG.md)" -ge 1 && echo CHANGELOG-OK</automated>
    <fails_when>CHANGELOG-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - The >= 3-seed trial completes end-to-end with statistics.json carrying n_seeds=3, method "t", finite ci95
  - sweep.py per-module coverage >= 96%
  - `grep -c "(REV-09," CHANGELOG.md` >= 1 under ## [Unreleased]
  - Fast lane still green: `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m "not slow"`
  </acceptance_criteria>
  <done>SEED-01 accepted: the honest multi-seed protocol runs end-to-end on a pinned small model with the t-interval statistics block, at the 96% per-module standard, CHANGELOG traceable.</done>
</task>

</tasks>

<artifacts_produced>
Symbols this plan creates (B4 lane):
- `dnallm/finetune/sweep.py` (new module): `aggregate_seeds(values, *, n_bootstrap, bootstrap_seed, small_n_ci)` (pure, torch-free), `run_seeds(fn, seeds, out_root, *, model_name, task_name)`, the SweepConfig adapter, the result-JSON statistics block spec
- Directory protocol artifact: {out_root}/{model}/{task}/seed_{s}/ per seed + {model}/{task}/statistics.json aggregate
- Test twin `tests/finetune/test_sweep.py` (pure-function boundary matrix + determinism + slow >= 3-seed trial)
- CHANGELOG.md: REV-09 entry under ## [Unreleased]
- No trainer.py/configs.py changes; no new public exports beyond sweep.py
</artifacts_produced>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| config/user strings → filesystem | model_name/task_name/out_root flow into paths under a recursive mkdir (V12) |
| per-seed outputs → statistics | aggregation consumes files this run wrote (trusted within the run) |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-08 | Tampering | run_seeds out_root handling in sweep.py | low | mitigate | out_root validated before recursive mkdir; model/task path segments sanitized with a matchable ValueError on separators; parents=True exist_ok idiom; no CWD fallback |
| T-11-09 | Repudiation | aggregate_seeds n-guard | medium | mitigate | A vacuous bootstrap CI at n=3 reports fake precision as the reviewer-facing reproducibility answer: D-14 guard (t at 3<=n<10, None below 3, bootstrap only at n>=10) with the n=2/3/9/10 boundary matrix test-proven same-change |
| T-11-SC | Tampering | package installs | high | mitigate | This lane installs nothing (scipy/numpy existing dependencies; zero pyproject changes). The phase's only sanctioned install is plan 11-05's scikit-allel under owner decision D-08 |
</threat_model>

<verification>
- Fast lane: `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m "not slow"` — torch-light, network-free
- Slow lane: `uv run --no-sync pytest tests/finetune/test_sweep.py -q -m slow` — >= 3-seed trial with statistics block
- Per-module coverage (cov-crash workaround): `uv run --no-sync coverage run -m pytest tests/finetune/test_sweep.py -q -m "not slow"` then `uv run --no-sync coverage report --include="dnallm/finetune/sweep.py"` — >= 96%
- Invariants: `git diff HEAD -- pyproject.toml`, `git diff HEAD -- dnallm/__init__.py`, `git diff HEAD -- dnallm/finetune/trainer.py`, `git diff HEAD -- dnallm/configuration/configs.py` all empty over this lane's commits; `grep -c "(REV-09," CHANGELOG.md` >= 1
- Owner directive 2026-10-09: run ONLY the targeted verifiers above and this lane's test files — no repo-wide lanes, no check_code.py full sweeps
</verification>

<success_criteria>
- SEED-01 fully test-proven: run_seeds directory protocol + D-16 same-split semantics + determinism, aggregate_seeds n-guarded exact statistics on constructed arrays, >= 3-seed end-to-end trial with the statistics block
- Every dnallm/ change shipped with its tests in the same commit; sweep.py >= 96% per-module coverage
- No new dependencies, no facade re-exports, no trainer/configs edits, no out-of-lane files
</success_criteria>

<output>
Create `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-04-SUMMARY.md` when done
</output>
