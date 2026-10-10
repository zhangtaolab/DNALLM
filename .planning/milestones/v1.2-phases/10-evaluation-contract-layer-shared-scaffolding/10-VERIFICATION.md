---
phase: 10-evaluation-contract-layer-shared-scaffolding
verified: 2026-10-10T10:21:06Z
status: passed
score: 24/24 must-haves verified
covered_files: [".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-01-SUMMARY.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-01-trainer-eval-guard-scaffolding-PLAN.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-02-SUMMARY.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-02-metric-registry-contract-PLAN.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-03-SUMMARY.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-03-docs-terminology-changelog-PLAN.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-04-SUMMARY.md", ".planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-04-vep-core-kernels-PLAN.md", "CHANGELOG.md", "README.md", "dnallm/cli/cli.py", "dnallm/cli/inference.py", "dnallm/cli/train.py", "dnallm/configuration/configs.py", "dnallm/datahandling/data.py", "dnallm/finetune/trainer.py", "dnallm/inference/benchmark.py", "dnallm/inference/inference.py", "dnallm/inference/interpret.py", "dnallm/inference/plot.py", "dnallm/inference/vep.py", "dnallm/mcp/server.py", "dnallm/models/losses.py", "dnallm/models/model.py", "dnallm/models/modeling_auto.py", "dnallm/tasks/metric_registry.py", "dnallm/tasks/metrics.py", "dnallm/tasks/task.py", "dnallm/utils/sequence.py", "docs/concepts/architecture/tokenization.md", "docs/concepts/biology/biological_tasks.md", "docs/concepts/biology/dna_sequences.md", "docs/concepts/inference.md", "docs/concepts/mcp.md", "docs/concepts/technical/transfer_learning.md", "docs/concepts/training.md", "docs/example/marimo/benchmark/benchmark_demo.md", "docs/example/marimo/finetune/finetune_demo.md", "docs/example/marimo/inference/inference_demo.md", "docs/example/notebooks/benchmark.md", "docs/example/notebooks/data_prepare_finetune.md", "docs/example/notebooks/finetune_NER_task.md", "docs/example/notebooks/finetune_binary.md", "docs/example/notebooks/finetune_multi_labels.md", "docs/example/notebooks/inference.md", "docs/example/notebooks/inference_megaDNA.md", "docs/example/notebooks/overview.md", "docs/getting_started/installation.md", "docs/getting_started/quick_start.md", "docs/index.md", "docs/resources/model_selection.md", "docs/resources/model_zoo.md", "docs/resources/troubleshooting_models.md", "docs/user_guide/benchmark/configuration.md", "docs/user_guide/benchmark/getting_started.md", "docs/user_guide/benchmark/index.md", "docs/user_guide/cli/config_generator.md", "docs/user_guide/cli/index.md", "docs/user_guide/cli/mcp_server.md", "docs/user_guide/cli/usage.md", "docs/user_guide/data_processing/data_preparation.md", "docs/user_guide/fine_tuning/getting_started.md", "docs/user_guide/fine_tuning/index.md", "docs/user_guide/fine_tuning/peft_adapters.md", "docs/user_guide/fine_tuning/task_guides.md", "docs/user_guide/getting_started.md", "docs/user_guide/inference/getting_started.md", "docs/user_guide/models.md", "docs/user_guide/performance/gpu_optimization.md", "docs/user_guide/performance/inference_speed.md", "docs/user_guide/performance/model_quantization.md", "example/notebooks/overview.md", "mkdocs.yml", "pyproject.toml", "scripts/generate_md_from_marimo.py", "tests/configuration/test_configs.py", "tests/datahandling/test_dna_dataset.py", "tests/finetune/test_trainer.py", "tests/inference/test_vep.py", "tests/tasks/test_metric_registry.py", "tests/tasks/test_metrics.py"]
covered_digest: "v3:sha256:63250ee055f18f5803ee872427524b2ab9074d19693ed5f57bed7cd9fabec0db"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  # Second fixpoint regeneration — HEAD de5cd80; delta since the prior green report
  # (1e77a87) = release-bump docs only: pyproject.toml line 3 (0.7.1→0.8.0) and the
  # CHANGELOG.md 0.8.0 header/Overview insertion. dnallm/version.py version constant
  # also bumped but is NOT in covered_files. No other dnallm/ or tests/ file changed.
  previous_status: passed
  previous_score: 24/24
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 10: Evaluation Contract Layer & Shared Scaffolding Verification Report

**Phase Goal:** The evaluation-semantics contract that gates the benchmark re-run is in place — the trainer can never silently evaluate on the test split, every metric resolves through one shared registry, and the revision docs surface is honest — and all shared files are scaffolded in one pass so Phase 11's parallel agents only fill modules
**Verified:** 2026-10-10T10:21:06Z
**Status:** passed
**Re-verification:** Yes — second fixpoint regeneration at HEAD de5cd80. The prior green report (2026-10-10T07:05:28Z, 24/24 at HEAD 1e77a87) went stale again solely through the milestone close-out release commits: `git log --oneline 1e77a87..HEAD --stat` shows six commits, of which the only ones touching this phase's covered_files are 32d242a (pyproject.toml line 3, `version = "0.7.1"` → `"0.8.0"`) and 234bf3c (CHANGELOG.md: the `## [0.8.0] - 2026-10-10` header plus one Overview paragraph inserted above the existing entries — no entry content changed, verified by reading the full diff). The same commit also bumped the `dnallm/version.py` `__version__` constant — that file is NOT in covered_files and is a data-only version string, not contract surface; `git diff 1e77a87..HEAD --stat -- dnallm/ tests/` lists version.py as the single dnallm/tests change. Every contract source and test file is byte-identical to the state the 07:05:28Z report verified. No code changes accompany this regeneration.

## Goal Achievement

Re-verification of the 24 baseline truths at the new milestone fixpoint (HEAD de5cd80). Staleness provenance is docs-only (see above), so this pass: (a) re-ran the METR-01 lane first-hand at HEAD — 137 passed in 4.29s; (b) re-spot-checked six file:line anchors from the truth table across `trainer.py`, `metrics.py`, `metric_registry.py`, `configs.py`, and `docs/user_guide/fine_tuning/peft_adapters.md` — all resolve exactly as documented; (c) recomputed the covered_files digest over the unchanged 81-file list; and (d) carries forward the first-hand behavioral evidence from the 1e77a87 report (guard test classes, collision selections, tasks+vep lanes, cross-lane composite, coverage measurements, docs gates, terminology grep) — valid to carry because every file that evidence exercised is byte-identical since 1e77a87, as proven by the git stat above.

### Observable Truths

| # | Truth | Status | Evidence at HEAD de5cd80 |
|---|-------|--------|--------------------------|
| 1 | SC1: No-dev-split training never evaluates on test — `eval_dataset=None` AND `eval_strategy="no"` set atomically; `load_best_model_at_end` collision fails loud | ✓ VERIFIED | `trainer.py:581-590` (atomic else-branch, anchor re-verified at de5cd80); hoisted collision guard `:599-605`; `allow_test_as_eval` field at `configs.py:317` (anchor re-verified); 23 guard-class tests passed at 1e77a87, files unchanged since |
| 2 | SC1: Flip WARN fires once at guard time, all three D-04 facts; opt-in WARN names the leak risk | ✓ VERIFIED | `trainer.py:584-590` (excluded + disabled + differs-from-previous-versions + opt-in instruction in one print at setup time — text re-verified at de5cd80); `:575-580` (leak-risk WARN); flip-warn/opt-in tests passed at 1e77a87 (14-test selection), files unchanged since |
| 3 | SC1: Collision ValueErrors symmetric across test-only AND train-only AND unsplit; early stopping + no eval raises naming both remedies | ✓ VERIFIED | Hoisted guard `:599-605` sits AFTER the split-selection block so all three no-eval paths hit it; early-stopping neighbor `:658-664` names both remedies; `collision` selector: 14 passed at 1e77a87, files unchanged since |
| 4 | SC1: `allow_test_as_eval=True` + load_best/early-stopping does NOT raise; dev+test force-enable preserved | ✓ VERIFIED | Opt-in branch sets eval_dataset before the guard; `opt_in` selector passed at 1e77a87; force-enable at `:671-676` intact |
| 5 | SC1: train-only and unsplit keep behavior — no flip WARN | ✓ VERIFIED | `trainer.py:591-593` plain branch (no print); `train_only_emits`/`unsplit_dataset` tests passed at 1e77a87 |
| 6 | SC1: Unit-test matrix present and passing | ✓ VERIFIED | `TestDatasetSplitWiring`/`TestEvalSemanticsGuard`/`TestEarlyStoppingCollision`/`TestEvaluateSplit` → 23 selected, 23 passed at 1e77a87; `tests/finetune/test_trainer.py` unchanged since (git stat) |
| 7 | SC2: `evaluate(split=...)` accepts any split key; absent key raises matchable ValueError listing sorted splits; routes through predict — never evaluate | ✓ VERIFIED | `trainer.py:942-945` (sorted available splits in message — anchor re-verified at de5cd80), `:954` region (predict call); `TestEvaluateSplit` passed at 1e77a87 within the 34-test cross-lane composite |
| 8 | SC2: Result JSON `eval_{split}_result.json` under output_dir with split/UTC-ISO-timestamp/metrics; runtime keys separated; no checkpoint param; output_dir falsy guard incl. `""` | ✓ VERIFIED | Signature `:884-889` (no checkpoint); `PREDICT_RUNTIME_KEYS` at `:294` separated `:961-963`; JSON write `:965-975` with `datetime.now(timezone.utc).isoformat()`; falsy guard `:946-952` (message re-verified at de5cd80) |
| 9 | SC2: Legacy HF kwargs forward unchanged; no-args calls evaluate() with no kwargs; docstring carries D-03 model-selection wording | ✓ VERIFIED | `:931-941` (only non-default kwargs forwarded); class docstring `:307-315` verbatim "the best checkpoint when load_best_model_at_end or early stopping fired, otherwise final-epoch weights" |
| 10 | SC3: `metric_registry.py` sibling of metrics.py, OUTSIDE vendored glob, coverage row produced | ✓ VERIFIED | Path `dnallm/tasks/metric_registry.py` (unchanged since 1e77a87 per git stat); first-hand coverage run at 1e77a87 produced the row: 146 stmts / 1 miss / 99% |
| 11 | SC3: `resolve(name)` returns canonical callable; unknown raises matchable ValueError; exact case-sensitive; historical aliases map | ✓ VERIFIED | Live probe at 1e77a87: `resolve('eval_auroc') is resolve('AUROC')` → True; `canonical_name('eval_spearman_r')` → `spearmanr`; `Eval_Auroc` → `ValueError: Unknown metric name...`; 28 registered names; METR-01 lane re-run green at de5cd80 (137 passed) |
| 12 | SC3: Aliases recognition-only, never emitted; invariants raise on violated tables; immutable surface | ✓ VERIFIED | Live mutation probe at 1e77a87 → TypeError; `_build_registry` invariants; MappingProxyType at `metric_registry.py:367-369` (anchor re-verified at de5cd80); alias-disjointness contract tests in the METR-01 lane pass again at de5cd80 |
| 13 | SC3: `metrics.py` emits exclusively through the registry | ✓ VERIFIED | `from .metric_registry import validate_emission` (`metrics.py:48` — anchor re-verified at de5cd80); `_emit` at all 9 return sites (`:95,172,253,407,517,571,634,638,663`); zero `print(` in either module (WR-04 removal holds); METR-01 lane re-run at de5cd80: 137 passed |
| 14 | SC3: Import-light — no torch/sklearn at module level | ✓ VERIFIED | Module imports exactly `collections.abc`/`types`/`typing` (`metric_registry.py:34-36`); sklearn imports live inside each callable |
| 15 | SC3: >=96% line coverage | ✓ VERIFIED | First-hand at 1e77a87: `metric_registry.py` 99% (1 miss = line 314 duplicate-canonical guard, structurally unreachable), `metrics.py` 100%; both files byte-identical at de5cd80 |
| 16 | SC4: Terminology unified — 5 sanctioned paper-title exceptions only | ✓ VERIFIED | Whole-tree grep at 1e77a87: exactly 5 hits in `modeling_auto.py`, ALL verbatim `"title"` citation fields (PlantCaduceus PNAS `:240`, GENA-LM NAR `:318`, GENA-LM-BigBird NAR `:352`, GPN PNAS `:417`, GROVER Nat. MI `:429`); docs/, README, example/: zero; no docs file changed since except the CHANGELOG release header |
| 17 | SC4: `validate_sequences` docstring with comparability hazard, whole-sequence-drop, case-sensitivity | ✓ VERIFIED | `data.py:869-898` — Warning block names the 13 strict-charset families, different-subset hazard, common-subset requirement, D3 pipeline-side ruling, `set(seq.upper()) - set(valid_chars)` mechanics |
| 18 | SC4: Dropped-row count log fires only when rows drop, with counts + filters; silent at zero | ✓ VERIFIED | `data.py:900-910` — `if n_dropped > 0` guard, dropped/before counts and all four filter params in the line |
| 19 | SC4: LoRA/QLoRA/IA³ chapter grounded in real config fields; nav registered — IA³ section NOW COMPLETE (Phase-12 closure of the DOCS-01 split delivery) | ✓ VERIFIED | `docs/user_guide/fine_tuning/peft_adapters.md` — IA³ section `:174+` re-verified at de5cd80 (heading + symmetric-support paragraph on page); maps fields 1:1 to `Ia3Config`, real `use_ia3: true` example, preset log line; grep for "not yet / placeholder / future phase" → zero hits; `mkdocs.yml:79` nav entry re-verified at de5cd80 |
| 20 | SC4: Docs-validation gate green; mirror pair byte-identical | ✓ VERIFIED | First-hand at 1e77a87: check_docs_sync OK; validate_yaml 21/21; validate_docs_snippets 147 files / 352 blocks valid; mirror diff empty; no docs file changed since except the CHANGELOG release header |
| 21 | SC4: CHANGELOG one entry per revision fix, each traceable to its commit | ✓ VERIFIED | CHANGELOG.md at de5cd80: REV-02 line 18 → 58bbf41, REV-01 line 30 → dae194a, REV-03 line 31 → 36d0c74 (line numbers shifted +6 by the inserted 0.8.0 release header; entry content, commit URLs, and ordering unchanged — the release Overview paragraph itself cites 58bbf41 and the REV-01..REV-11 chain, reinforcing traceability); Phase-11/12 entries layered after without disturbing the Phase-10 chain |
| 22 | SC5: One-pass scaffolding — `use_ia3` field, Ia3/Vep/SweepConfig field-complete, registered in DNALLMConfig + load_config | ✓ VERIFIED | `configs.py:326` (use_ia3), `:426` Ia3Config, `:481` VepConfig, `:510` SweepConfig, `:692-694` DNALLMConfig, `:732-741` load_config. EVOLUTION: Phase-11 commit 3b644bd closed the documented scaffold boundary — Ia3Config refined to peft-0.21.1 field parity and the interim no-validator pin test replaced by real `reject_ia3_with_qlora` + tests; the Phase-10 one-pass scaffold was consumed exactly as the goal required ("Phase 11's agents only fill modules") |
| 23 | SC5: Phase-10 pyproject diff = package-data only; no new `__init__.py` re-exports; no new skips | ✓ VERIFIED | `git diff d3097d6 36d0c74 -- pyproject.toml` = package-data entries only (zero dependency lines); the only pyproject change since 1e77a87 is the line-3 version bump (release 32d242a), not contract surface; grep for ia3/vep/sweep/metric_registry/resolve re-exports in `dnallm/__init__.py`, `dnallm/tasks/__init__.py`, `dnallm/inference/__init__.py` → zero hits; skip-marker scan over the six phase test files → zero |
| 24 | SC5: REV-08 core — `align_variant` same-slot rule + CLM/MLM kernels, formula docstrings, no_grad + device helper, deterministic | ✓ VERIFIED | `vep.py` at 1e77a87: `align_variant:114`, `clm_log_likelihood:219`, `mlm_slot_log_prob:262`, `get_model_device:194`, formula docstrings `:224` (`log P(sequence) = sum_t ...`) and `:269`; 6 no_grad/device sites; core intact beneath Phase-11's documented extensions (score_variant/evaluate_vcf/CLI per VEP-01); file unchanged since 1e77a87 (git stat); vep lane passed within the 253-test run |

**Score:** 24/24 truths verified (0 present, behavior-unverified)

**Cross-lane contract composite (carried forward from 1e77a87):** `pytest -k "TestRegistryEmissionContract or TestEvaluateSplit"` → 34 passed. Every factory-emitted spelling remains a registered canonical name; `evaluate(split=...)` prefix-stripping lands on exactly those spellings. The registry half of that contract was re-exercised at de5cd80 via the METR-01 lane (137 passed).

### Prohibition Checks (must-NOT hold)

| Prohibition | Status | Evidence at HEAD |
|---|---|---|
| Guard must NOT carve a dev split from train/test | ✓ HELD | `trainer.py:568-593` only selects/excludes existing split keys; no split creation |
| `evaluate(split=...)` must NOT reload checkpoints or accept a checkpoint param | ✓ HELD | Signature `(split, eval_dataset, ignore_keys, metric_key_prefix)`; docstring reaffirms "no checkpoint is reloaded" |
| Guard must NOT silently downgrade user load_best/early-stopping | ✓ HELD | Collisions raise ValueError; guard placement after `TrainingArguments.__post_init__` comment intact |
| Test split must NOT route through `trainer.evaluate` on the split path | ✓ HELD | predict-only routing at `:954`; test-asserted |
| Phase 10 must NOT add use_ia3 cross-field rejection validators | ✓ HELD (phase-scoped) | `git log -S reject_ia3_with_qlora` → first appears in Phase-11 commit 3b644bd; the Phase-10 range (through 36d0c74) contains no use_ia3 validator — the scaffold stayed field-only, and Phase 11 filled it per plan |
| No new `dnallm/__init__.py` / `dnallm/tasks/__init__.py` / `dnallm/inference/__init__.py` re-exports | ✓ HELD | grep for phase-surface names in all three at HEAD → zero hits |
| Registry: no mutation API; NOT inside vendored `dnallm/tasks/metrics/`; no module-level torch/sklearn | ✓ HELD | Sibling path; MappingProxyType (live probe → TypeError); stdlib-only imports |
| Registry must never emit an alias | ✓ HELD | One-directional alias table; emission contract tests green (METR-01 lane re-run at de5cd80) |
| vep.py Phase-10 boundary: no evaluate_vcf/CLI at that stage | ✓ HELD (phase-scoped) | Phase-10 commits added only the core (`93bd52f`, `d782c7c`); evaluate_vcf/CLI landed in Phase-11 commits (9a0ddef, 39eac44) per the documented VEP-01 plan |
| Sweep must NOT touch A1/A2/A4-owned files | ✓ HELD | Owner-lane attribution verified in the original report; current state of those files matches their owner lanes' contracts |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/finetune/trainer.py` | Guard, evaluate(split=), docstrings — intact around Phase-11/12 additions | ✓ VERIFIED | All EVAL-01 behaviors present at anchored line numbers (re-verified at de5cd80); 37 targeted tests passed at 1e77a87, file unchanged since |
| `dnallm/configuration/configs.py` | allow_test_as_eval, use_ia3, Ia3/Vep/SweepConfig, registration | ✓ VERIFIED | All present (`:317` anchor re-verified at de5cd80); Ia3Config refined to peft parity by Phase 11 (documented evolution) |
| `dnallm/tasks/metric_registry.py` | Registry + 4-function API | ✓ VERIFIED | 449 lines, 28 canonicals, 99% coverage; lane re-run green at de5cd80 (137 passed) |
| `dnallm/tasks/metrics.py` | Exclusive registry emission | ✓ VERIFIED | 9 `_emit` sites (`:48` import anchor re-verified at de5cd80); 100% coverage; no debug prints |
| `dnallm/inference/vep.py` | align_variant + kernels core | ✓ VERIFIED | Core intact under Phase-11 extension; formulas in docstrings; file unchanged since 1e77a87 |
| `dnallm/datahandling/data.py` | validate_sequences docstring + count log | ✓ VERIFIED | Both present verbatim |
| `docs/user_guide/fine_tuning/peft_adapters.md` | Chapter, IA³ now complete | ✓ VERIFIED | Grounded in real Ia3Config fields (`:174+` re-verified at de5cd80); in nav (`mkdocs.yml:79` re-verified) |
| `pyproject.toml` | Package-data entry (Phase-10 range) | ✓ VERIFIED | Package-data-only diff in the Phase-10 commit range; sole delta since 1e77a87 is the line-3 version bump (release commit 32d242a) |
| `CHANGELOG.md` | REV-01/02/03 under [Unreleased] | ✓ VERIFIED | All three with commit links; line numbers shifted +6 by the 0.8.0 release header, entry content unchanged |
| Test files ×6 | Suites passing | ✓ VERIFIED | 192 tasks + 253 tasks/vep + 37 trainer selections green at 1e77a87; METR-01 lane re-run green at de5cd80 (137 passed); no test file changed between (git stat) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| YAML `finetune.allow_test_as_eval` | Guard decision | TrainingConfig field → set_up_trainer | ✓ WIRED | `configs.py:317` → pop at `trainer.py:532` region → guard reads it at `:573` |
| `DNATrainer.evaluate(split=...)` | Result JSON | predict → prefix strip → output_dir file | ✓ WIRED | Re-confirmed at `:942-975` (anchors re-verified at de5cd80); TestEvaluateSplit green at 1e77a87 |
| `TrainingConfig.model_dump()` | TrainingArguments | pop list | ✓ WIRED | `allow_test_as_eval`, `use_ia3`, `peft_dry_run` popped; trainer tests green |
| `load_config()` | Phase 11 consumers | ia3/vep/sweep keys | ✓ WIRED | `configs.py:732-741`; consumed by Phase-11 sweep/vep modules |
| metrics.py factories | Registry | `_emit` → validate_emission | ✓ WIRED | 9 sites (`:48` anchor re-verified at de5cd80); negative probe covered by contract tests; METR-01 lane re-run green at de5cd80 |
| evaluate(split) prefix-stripping | Registry canonical names | cross-lane contract | ✓ WIRED | 34-test composite passed at 1e77a87; registry half re-exercised at de5cd80 |
| validate_sequences docstring | Rendered API docs | mkdocstrings + docs gate | ✓ WIRED | Snippet gate green at 1e77a87 (147 files / 352 blocks); docs unchanged since except CHANGELOG release header |
| Mirror pair | check_docs_sync | byte-identity | ✓ WIRED | diff empty at 1e77a87; mirror files unchanged since |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| evaluate(split=) metrics dict | `predict_result.metrics` | `self.trainer.predict(dataset[split])` | Yes | ✓ FLOWING |
| Result JSON | split/timestamp/metrics/runtime | live values at call time | Yes | ✓ FLOWING |
| METRIC_REGISTRY | 28 entries | explicit `_RAW_REGISTRY` table | Yes (static by design) | ✓ FLOWING |
| Dropped-row warning | n_dropped/before | `_row_count` around real `.filter` | Yes | ✓ FLOWING |
| vep kernels | token_logps / logp[token_id] | real forward passes | Yes (exercised by real-model tests at 1e77a87) | ✓ FLOWING |

### Behavioral Spot-Checks

Evidence provenance: rows marked **de5cd80** were run first-hand during this regeneration; all other rows are carried forward from the 1e77a87 report and remain valid because every file they exercise is byte-identical since 1e77a87 (proven by `git diff 1e77a87..HEAD --stat -- dnallm/ tests/` → version.py only, plus the docs-only CHANGELOG/README-repo delta).

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| METR-01 lane (registry + metrics contract) | `uv run --no-sync pytest tests/tasks/test_metric_registry.py tests/tasks/test_metrics.py -q --no-cov --timeout=240` | **137 passed in 4.29s (de5cd80)** | ✓ PASS |
| Guard/evaluate test classes | `pytest tests/finetune/test_trainer.py -k "TestDatasetSplitWiring or TestEvalSemanticsGuard or TestEarlyStoppingCollision or TestEvaluateSplit"` | 23 passed (1e77a87, carried) | ✓ PASS |
| Collision/flip-warn/opt-in named tests | `pytest tests/finetune/test_trainer.py -k "collision or flip_warn or opt_in or train_only_emits or unsplit_dataset or output_dir"` | 14 passed (1e77a87, carried) | ✓ PASS |
| Tasks + VEP lanes | `pytest tests/tasks/ tests/inference/test_vep.py -q` | 253 passed (1e77a87, carried) | ✓ PASS |
| Cross-lane composite | `-k "TestRegistryEmissionContract or TestEvaluateSplit"` | 34 passed (1e77a87, carried) | ✓ PASS |
| Registry live behavior | python probe (alias identity, bad-casing raise, mutation block) | all OK, 28 names (1e77a87, carried) | ✓ PASS |
| Registry/metrics coverage | `coverage run --include=... -m pytest tests/tasks/` + report | 146/1 miss 99%; 280/0 100% (1e77a87, carried) | ✓ PASS |
| Docs gates ×4 (targeted) | check_docs_sync / validate_yaml / validate_docs_snippets / mirror diff | OK / 21/21 / 352 blocks / identical (1e77a87, carried) | ✓ PASS |
| Terminology grep | `grep -rniE "DNA[ -]language[ -]model" dnallm/ docs/ README.md example/` | 5 hits, all sanctioned paper titles (1e77a87, carried) | ✓ PASS |
| Truth-table anchor spot-checks ×6 | sed at `trainer.py:581-590`, `trainer.py:942-954`, `metrics.py:48`, `metric_registry.py:367-369`, `configs.py:317`, `peft_adapters.md:174-180` + `mkdocs.yml:79` | all anchors resolve exactly as documented (de5cd80) | ✓ PASS |

Owner directive honored: targeted/lane tests over this phase's files only, no repo-wide lanes; this mechanical regeneration re-ran a single lane (METR-01) per the docs-only staleness provenance, with the 1e77a87 first-hand runs carried forward. Coverage measurements retain the documented coverage-CLI workaround for the known `pytest --cov` env crash.

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared by the plans; the plans' verification commands (pytest selectors, live probes, docs gate scripts) were executed as behavioral spot-checks in the original and 1e77a87 reports, with the METR-01 lane re-executed first-hand at de5cd80.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVAL-01 | 10-01 | Trainer never silently uses test as eval; atomic guard; opt-in WARN; early-stopping neighbor; evaluate(split) predict path; test matrix; docstring | ✓ SATISFIED | Truths 1-9, re-anchored at de5cd80; test evidence carried from 1e77a87 (files unchanged) |
| METR-01 | 10-02 | Registry at metric_registry.py outside vendored glob; resolve() ValueError; canonical anchors; aliases recognition-only; metrics.py exclusive; contract tests; import-light | ✓ SATISFIED | Truths 10-15; lane re-run green at de5cd80 (137 passed) |
| DOCS-01 | 10-03 | Terminology unified; validate_sequences warning + count log; LoRA/QLoRA/IA³ chapter (IA³ part completes after PEFT-01); CHANGELOG evidence chain; docs gate green | ✓ SATISFIED | Truths 16-21; the IA³ split-delivery is CLOSED — Phase 12 completed the section, forward pointer removed |
| VEP-01 (head start only) | 10-04 | Phase 10 SC5 required only the core: align_variant + kernels | ✓ SATISFIED (Phase-10 scope) | Truth 24; full VEP-01 completed and verified in Phase 11 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly EVAL-01, METR-01, DOCS-01 to Phase 10 (all marked Complete), each claimed by plan frontmatter; VEP-01 maps to Phase 11 with the Phase-10 head start documented in the traceability notes.

### Decision Coverage

All 9 trackable decisions (D-01..D-09) remain honored by shipped artifacts at HEAD — verified directly in the 1e77a87 pass (D-02/D-03 guard semantics + docstring wording, D-04 one-directional aliases, D3 pipeline-side filtering ruling in the docstring, D-07 scaffold-consumed-by-Phase-11, D-09 same-commit CHANGELOG chain); the underlying files are byte-identical at de5cd80. Note: the `check.decision-coverage-verify` verb could not locate `10-CONTEXT.md` (tool expects bare `CONTEXT.md`) — reported as a tooling naming mismatch, not a coverage gap; the 9/9 result plus direct artifact evidence stands.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers in any phase code file at HEAD; no skip markers in the six phase test files; no debug prints in metrics/registry (WR-04 fix holds) | - | - |

Info-level, not defects: the `pyproject.toml` comment at the package-data entry ("until then this glob matches nothing") is stale since Phase 11 landed `presets/lora_targets.yaml` — a comment-accuracy nit in a Phase-11-owned region, outside Phase 10's file surface at its commit range.

### Human Verification Required

None. All behavior-dependent truths (guard transitions, collision raises, WARN content, JSON write, log boundary, kernel determinism, registry resolution/immutability) are exercised by named tests that passed at 1e77a87 and whose files are byte-identical at de5cd80; the METR-01 lane was additionally re-run green first-hand at de5cd80. The two items flagged for human attention in the original review chain (WR-03 guard hoist, D3 log boundary) remain pinned by their named tests.

### Gaps Summary

No gaps. All 24 baseline truths hold at HEAD de5cd80. This second staleness event was caused solely by the milestone close-out release commits (32d242a: pyproject.toml line-3 version bump, plus the `dnallm/version.py` `__version__` constant which is not in covered_files; 234bf3c: CHANGELOG.md 0.8.0 release header + Overview paragraph inserted above unchanged entries) — no contract source, test, or docs-contract file changed since the 1e77a87 green report, as proven by `git log --oneline 1e77a87..HEAD --stat` and `git diff 1e77a87..HEAD --stat -- dnallm/ tests/`. The METR-01 lane re-run (137 passed), six re-verified truth-table anchors, and the recomputed digest over the unchanged 81-file covered list confirm the phase contract remains intact at the fixpoint.

---

_Verified: 2026-10-10T10:21:06Z_
_Verifier: Claude (gsd-verifier)_
