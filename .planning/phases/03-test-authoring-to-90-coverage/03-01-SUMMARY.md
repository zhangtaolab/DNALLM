---
phase: 03-test-authoring-to-90-coverage
plan: 01
subsystem: testing
tags: [pytest, coverage, torch, captum, altair, dna-inference]

requires:
  - phase: 01-harness-integrity-measured-baseline
    provides: measured 45.92% baseline, 43-row ranked worklist, coverage tooling of record
  - phase: 02-suite-hygiene-known-bug-fixes
    provides: green 622-test fast leg, typed-skip allowlist + audit gate, tmp_path pdf discipline
provides:
  - 176 inference-area missing lines (from 1,575 fast-leg pre-wave) — wave-1 area gate ≤250 met with 74 lines of slack
  - 279 new behavior tests (inference 99, interpret 43, mutagenesis 46, plot 68, benchmark 21, plus conftest fixtures)
  - Shared cross-file fixtures (SimpleDNATokenizer, TinyDNAModel, inference_config_factory) reusable by waves 2-5
  - Refreshed post-Phase-2 baseline recorded (first re-measure since Phase 1)
  - coverage-wave1-missing.txt — re-ranked worklist input for wave 2
  - Five Rule-1 source fixes in plot.py/benchmark.py (latent crashes found by test authoring)
affects: [03-test-authoring-to-90-coverage, 04-coverage-gate-ci]

actuals:
  tokens: 48784
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Real tiny torch module + real deterministic tokenizer through the full engine path (config → DNAInference → logits → semantic predictions), recomputed independently in tests"
    - "Chart-contract assertions on altair specs (mark types, encodings, titles) without file writes; pdf marker only for actual savers"
    - "sys.modules stubbing (monkeypatch.setitem) for absent optional deps (evo package)"
    - "Autouse fixture resetting the altair data transformer around vegafusion-enabling functions"

key-files:
  created:
    - tests/inference/test_interpret.py
    - tests/inference/test_mutagenesis.py
  modified:
    - tests/inference/test_inference.py
    - tests/inference/test_plot.py
    - tests/benchmark/test_benchmark.py
    - tests/conftest.py
    - dnallm/inference/plot.py
    - dnallm/inference/benchmark.py

key-decisions:
  - "End-to-end engine tests use real collaborators (SimpleDNATokenizer + deterministic TinyDNAModel) instead of Mocks wherever autograd or encode semantics are the behavior under test"
  - "Five latent crashes fixed under Rule 1 rather than avoided: multilabel curve assembly, dict-input annotations, entropy normalization, code-based Benchmark init, stratified k-fold — each blocked coverage of a real user path"
  - "Dead/quirky branches documented as accepted residuals (never pragma'd): benchmark no-datasets config branch, k_folds=1 branch, mutagenesis strategy 'max', generate-from-DataLoader prompt accumulation"

patterns-established:
  - "conftest SimpleDNATokenizer/TinyDNAModel/inference_config_factory: shared engine-test scaffolding for later waves"
  - "Transformer-reset autouse fixture for altair spec assertions after vegafusion consumers"

requirements-completed: []

coverage:
  - id: D1
    description: "tests/inference/test_inference.py extended to 120 collected tests with five-task-type semantic logits→predictions assertions and end-to-end engine paths"
    requirement: TEST-03
    verification:
      - kind: unit
        ref: "tests/inference/test_inference.py (120 collected, all passing)"
        status: pass
      - kind: command
        ref: ".venv/bin/python -m pytest tests/inference/test_inference.py -q --tb=short"
        status: pass
    human_judgment: false
  - id: D2
    description: "tests/inference/test_interpret.py and test_mutagenesis.py: real captum attributions and saturation scans on real tiny torch modules (≥12 tests each; 43 and 46 delivered)"
    requirement: TEST-03
    verification:
      - kind: unit
        ref: "tests/inference/test_interpret.py (43) + tests/inference/test_mutagenesis.py (46), all passing"
        status: pass
      - kind: command
        ref: ".venv/bin/python scripts/audit_skips.py /tmp/p3-01-junit.xml tests/expected_skips.yaml (exit 0; zero skips in new files)"
        status: pass
    human_judgment: false
  - id: D3
    description: "plot/benchmark chart-contract extensions green with pickling-safe fakes (133 plot tests, 27 benchmark tests)"
    requirement: TEST-03
    verification:
      - kind: unit
        ref: "tests/inference/test_plot.py (133) + tests/benchmark/test_benchmark.py (27), all passing"
        status: pass
    human_judgment: false
  - id: D4
    description: "Wave-1 gate: full census green (919 passed / 7 allowlisted skips), audit exit 0, inference-area missing sum 176 ≤ 250, tree clean"
    requirement: TEST-03
    verification:
      - kind: command
        ref: "full census: pytest -ra --junitxml --cov (exit 0) → coverage json → sum(missing_lines over dnallm/inference/) = 176"
        status: pass
      - kind: command
        ref: "git status stray-artifact gate (tree-clean)"
        status: pass
    human_judgment: false

duration: 80 min
completed: 2026-09-30
status: complete
commits: 3
plan_head_before: d7819a6be907fcb6389814c16fdb347d58fae9aa
plan_head_after: 972ec98282b0a9e17436566591a1bd21fe10fdfa
---

# Phase 3 Plan 1: Inference Wave — Test Authoring Summary

**279 new behavior tests closing the inference area from 1,575 to 176 missing lines (gate ≤250) via real-torch end-to-end engine paths, real captum attributions, and altair chart-contract assertions; five latent crashes fixed under Rule 1**

## Performance

- **Duration:** 80 min (incl. 15-min full census)
- **Started:** 2026-09-30T08:24:27Z
- **Completed:** 2026-09-30T09:45:04Z
- **Tasks:** 3/3
- **Files modified:** 9 (7 test/fixture files, 2 source fixes)

## Refreshed Baseline (first re-measure since Phase 1 — Task 1 opening move)

Measured on the fast leg (both roots, `-m "not slow"`, config-only `--cov`):

- **42.94% covered** (3,174 / 7,391 lines; 4,217 missing)
- Inference area pre-wave missing (fast leg): **1,575**
  - inference.py 593 · plot.py 332 · mutagenesis.py 269 · interpret.py 261 · benchmark.py 120
- Note vs Phase-1 full census (45.92%, 3,390/7,383): the fast-leg numbers above exclude slow-test coverage; the comparable full-census numbers are the post-wave ones below.

## Post-Wave Measurement (full census, both roots, slow included)

- **Census:** 919 passed / 7 skipped (all allowlisted) / 0 failed / exit 0 — 914s
- **Suite coverage:** **64.75%** (4,791 covered / 2,608 missing on 7,399 stmts)
- **Inference-area missing sum: 176 (gate ≤ 250 — passed with 74 lines of slack)**
  - inference.py 93 (was 523 at Phase-1 census) · plot.py 40 (was 332) · interpret.py 23 (was 261) · benchmark.py 11 (was 120) · mutagenesis.py 9 (was 269)
- `scripts/audit_skips.py` exit 0 on the census junit; zero new skips introduced
- Fast leg at every task boundary: 622 → 721 → 810 → 898 passed, exit 0 each time
- Pragma budget: still exactly 3 occurrences under `dnallm/` (no additions)
- `pyproject.toml` coverage/pytest config untouched; working tree clean of pdf/log artifacts

## Accomplishments

- Engine core paths verified end-to-end with real collaborators: config load → DNAInference → real tokenization/encode → real torch forward → logits → semantic predictions per task type (binary/multiclass/multilabel/regression/token), including file/evaluate/save flows, embeddings/attention extraction contracts, generate (causallm/evo2/evo1-stubbed/megadna), scoring (embedding/logits/probability MLM+causal), device aliases + mamba fp32 downgrade + CustomEvo + LoRA construction branches
- Real captum attributions (LIG, LayerDeepLift, GradientShap, Occlusion, FeatureAblation, LayerConductance, NoiseTunnel×2) on a deterministic real torch module with finite per-token attribution assertions
- Saturation mutagenesis verified structurally and numerically (PLL/CLM scores matching independent recomputation; strategy matrix; ISM/hotspots/tfmodisco assembly)
- Chart-contract test layer for the whole plot surface (polar/radar/token-scatter/line/annotations/attention-normalization/reducers/embeddings/attributions) asserting altair specs without file writes
- Benchmark orchestration branches (code-based init, run prepared/unprepared, evaluate_single_model, stratified k-fold CV, plot selection) with pickling-safe fakes

## Task Commits

1. **Task 1: engine core paths + refreshed baseline** — `a32a973` (test)
2. **Task 2: interpret + mutagenesis on real tiny modules** — `6265d9f` (test)
3. **Task 3: plot + benchmark + wave-1 census re-measure** — `972ec98` (test + fix)

**Plan metadata:** this commit (docs)

## Files Created/Modified

- `tests/inference/test_inference.py` — 21 → 120 collected tests
- `tests/inference/test_interpret.py` — NEW, 43 tests
- `tests/inference/test_mutagenesis.py` — NEW, 46 tests
- `tests/inference/test_plot.py` — 65 → 133 tests
- `tests/benchmark/test_benchmark.py` — 6 → 27 tests
- `tests/conftest.py` — SimpleDNATokenizer, TinyDNAModel, tiny_model_factory, inference_config_factory fixtures
- `dnallm/inference/plot.py` — 3 Rule-1 fixes
- `dnallm/inference/benchmark.py` — 2 Rule-1 fixes
- `.planning/phases/03-test-authoring-to-90-coverage/coverage-wave1-missing.txt` — re-ranked worklist for wave 2

## Decisions Made

- Real-collaborator strategy over Mocks wherever autograd or tokenization semantics are the behavior (research anti-pattern honored); Mocks retained only for pure call-contract branches
- Latent crashes fixed rather than routed around (each fix unblocks a real user path and its coverage); every fix mirrors an existing correct pattern in the same codebase (e.g. the field-filter loop `__load_from_config` already used)
- Accepted residuals documented below instead of pragma'd, per the locked verification discipline

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `_process_curve_data` crashed on scalar summary scores**
- **Found during:** Task 3 (plot multilabel tests)
- **Issue:** Multilabel curve dicts carry per-label AUROC/AUPRC scalars alongside point arrays; the loop called `.extend(scalar)` → TypeError — the whole multilabel plotting path was unusable
- **Fix:** Skip non-iterable (scalar/str) values; summaries are consumed separately by the caller
- **Files modified:** dnallm/inference/plot.py
- **Verification:** test_multilabel_curves_split_by_label passes with real-shaped data
- **Committed in:** 972ec98

**2. [Rule 1 - Bug] `_prepare_annotations` crashed on dict input**
- **Found during:** Task 3
- **Issue:** `data[models[0]]` subscripted `dict_keys` → TypeError for every dict caller
- **Fix:** `models = list(data.keys())`
- **Files modified:** dnallm/inference/plot.py
- **Verification:** test_dict_input_per_model passes
- **Committed in:** 972ec98

**3. [Rule 1 - Bug] Entropy attention normalization collapsed the heatmap**
- **Found during:** Task 3
- **Issue:** `1 - ent/log2(L)` produced shape (L,1) → IndexError at DataFrame assembly
- **Fix:** Row-entropy weight broadcast against the (L,L) heatmap: `attn_head * (1 - ent/log2(L))`
- **Files modified:** dnallm/inference/plot.py
- **Verification:** parametrized normalization test incl. entropy passes
- **Committed in:** 972ec98

**4. [Rule 1 - Bug] Code-based `Benchmark()` init crashed under pydantic v2**
- **Found during:** Task 3
- **Issue:** Raw setattr of every EvaluationConfig field onto InferenceConfig raised "object has no field" (e.g. mixed_precision) — the documented config-less path never constructed
- **Fix:** Copy only fields InferenceConfig declares (the pattern `__load_from_config` already used)
- **Files modified:** dnallm/inference/benchmark.py
- **Verification:** TestCodeBasedInit both tests pass
- **Committed in:** 972ec98

**5. [Rule 1 - Bug] `stratified=True` k-fold always raised TypeError**
- **Found during:** Task 3
- **Issue:** `StratifiedKFold.split(indices)` without the required `y` argument
- **Fix:** Pass the dataset's labels column when stratified
- **Files modified:** dnallm/inference/benchmark.py
- **Verification:** test_stratified_folds_preserve_count passes
- **Committed in:** 972ec98

### Verify-command adjustment (no code impact)

pytest 9.1.1's `--collect-only -q` prints only the summary count (no `nodeid::` lines), so the plan's `grep -c "::"` tripwire always returned 0. Equivalent criterion enforced via the summary-line count (and cross-checked with the full id listing): 120 ≥ 40 for test_inference.py; 36/43 ≥ 12 for the Task 2 files.

---

**Total deviations:** 5 auto-fixed (all Rule 1 bugs) + 1 verify-command adjustment
**Impact on plan:** All fixes required for coverage of real user paths; each mirrors an existing correct pattern. No scope creep; no new dependencies; pragma budget intact.

## Accepted-Uncovered Residual Ledger (documented, never pragma'd)

- `inference.py` (93): CustomEvo/scoring deep branches (2030-2103 corners), xla construction fallback (221-227), auto-device mps/xpu/npu probes (243-249), pre-encoded tensor rename (495-499), DatasetDict labels (506-508), generate-from-DataLoader prompt accumulation quirk (1643-1648 — prompt_seqs never appended, latent bug recorded not fixed per bug-fix scope), scoring dict-input quirk (1824), empty-token NaN score (1960)
- `plot.py` (40): pacmap presets (package not installed), radar/polar/annotation save corners
- `interpret.py` (23): constructor eos-None→0 chain (144-160), noise-tunnel gradshap base, batch token_indices/target_layers plumbing
- `benchmark.py` (11): no-datasets config branch (169-170 — unreachable, `datasets` is a required BenchmarkConfig field), k_folds=1 fold branch (dead — KFold(n_splits=1) raises first), Subset fallback lines
- `mutagenesis.py` (9): strategy "max" (calls `.index` on ndarray — latent bug, recorded), hotspot dedupe corners

## Known Stubs

None — every new test asserts observable behavior; no placeholder logic introduced.

## Issues Encountered

None beyond the deviations above. The full census ran clean on the first post-authoring attempt (919/0/7).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 1 (inference) complete at 176/250; wave 2 executor should start from `coverage-wave1-missing.txt` (re-ranked) — models area (1,210 at Phase 1 census) is next by rank
- Suite now 64.75% overall; gap to 90.5% target = 1,911 lines (2,608 missing − 697 allowed)
- Shared fixtures (SimpleDNATokenizer/TinyDNAModel/inference_config_factory) available in tests/conftest.py for models/mcp waves
- Windows ledger: two latent-bug residuals recorded (generate-from-DataLoader accumulation; mutagenesis strategy "max")

---
*Phase: 03-test-authoring-to-90-coverage*
*Completed: 2026-09-30*

## Self-Check: PASSED

All created/modified files exist on disk; all three task commits (a32a973, 6265d9f, 972ec98) present in history.
