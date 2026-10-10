# Feature Research

**Domain:** Paper-revision-driven capabilities for a DNA-language-model toolkit (eval hygiene, PEFT adapters, probing, zero-shot VEP, multi-seed protocol, motif scanning, MCP tools)
**Researched:** 2026-10-09
**Confidence:** HIGH for PEFT IA³ (official docs, fetched directly) and FIMO scanning semantics (official MEME docs); MEDIUM for everything else (cross-checked across ≥2 independent sources — papers, official library docs, GitHub issues); individually noted LOW where only single/indirect sources exist

"Features" here are the REV-01…REV-11 candidate requirements from `.planning/research/261009-paper-revision-suite-plan.md`. "Users" are (a) the dnallmmark benchmark pipeline re-running E1'–E8', (b) the paper revision itself (reply-letter evidence), and (c) ordinary dnallm library users. Every section answers: how do comparable bio-ML tools implement this behavior, what is table stakes vs differentiator vs anti-feature, how complex is it, and what existing dnallm machinery does it sit on — so REQUIREMENTS.md can scope each REV testable and atomic.

---

## REV-01 · Evaluation-semantics leak guard

### How established tools do it

The Hugging Face Trainer — which `DNATrainer` wraps — **never silently substitutes a dataset as the eval set**. Evaluation happens only on a dataset the caller explicitly passes as `eval_dataset`; requesting evaluation without one raises `Trainer: evaluation requires an eval_dataset` (raised from `get_eval_dataloader()`), and `load_best_model_at_end=True` combined with `eval_strategy="no"` raises `--load_best_model_at_end requires the save and eval strategy to match` (validated in `training_args.py`, with `metric_for_best_model` defaulting to `"loss"`). The HF philosophy is fail-loud on ambiguous eval semantics; there is no code path where a split the user did not designate becomes the eval set (verified: [Trainer docs](https://huggingface.co/docs/transformers/v4.38.2/ja/main_classes/trainer), [training_args.py source](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/training_args.py), [SO 76310533](https://stackoverflow.com/questions/76310533/how-to-fix-trainer-evaluation-requires-an-eval-dataset-in-huggingface-transfo)). A documented counter-example exists in the ecosystem: TRL's `SFTConfig` with `load_best_model_at_end=True` but no eval dataset **silently loads the last checkpoint instead of the best** — exactly the class of silent semantics drift REV-01 exists to prevent ([The Neural Base, SFT course](https://theneuralbase.com/sft/learn/beginner/evaluation-during-training/)). Held-out evaluation is conventionally an explicit, gradient-free, checkpoint-selection-free call: `Trainer.predict()` in HF idiom, `score(X_test)` in scikit-learn idiom — dnallm's existing `infer()` (trainer.py:502) is already the correct shape. Leakage hygiene extends to preprocessing in comparable probing suites (scalers fit on train statistics only — see REV-07), reinforcing that "no test-derived state touches training" is the invariant, not just "don't call evaluate on test".

**Expected behavior derived from ecosystem norms:** with no dev split, the default must be `eval_strategy="no"` **and** `load_best_model_at_end=False` (mirroring HF's incompatibility rule, since silently keeping best-model-loading on while disabling eval is the TRL failure mode). Any override must be explicit, logged at WARN, and the new `evaluate(split=...)` must go through the predict path with no influence on checkpoint selection.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Default `eval_strategy="no"` + `load_best_model_at_end=False` when dev absent but test present (no auto-promotion of test to eval) | HF never substitutes; dnallm's current trainer.py:234-241 does — the bug | LOW-MED | `DNATrainer.__init__` (trainer.py:133-241) |
| Table stakes | Explicit `evaluate(split="test"\|"dev"\|...)` entry point via predict path, no checkpoint interaction | `Trainer.predict` idiom; `infer()` (trainer.py:502) already correct | LOW | `infer()` exists |
| Table stakes | `allow_test_as_eval=True` explicit override emitting WARN | TRL silent-last-checkpoint lesson: overrides must be loud | LOW | logging util |
| Table stakes | Tests over 3 split combinations (dev+test / test-only / train-only) × default/override | intake acceptance | LOW | existing pytest suite |
| Differentiator | Docstring + docs section on held-out semantics ("which split is eval, when is best-model loaded") | rarely documented in wrappers; reviewer-facing | LOW | docs pipeline (docs-validation gate exists) |
| Anti-feature | Auto-creating a dev split from test/train when dev is missing | silent data reshuffling breaks benchmark comparability (F2 layouts) | — | — |
| Anti-feature | Changing split-selection heuristics without an escape hatch | would invalidate the dnallmmark re-run baseline | — | — |

**Sources:** HF Trainer docs + training_args.py (HIGH legitimacy, fetched via search snippets — MEDIUM overall); TRL caveat (MEDIUM); SO/Geneformer threads corroborate error behavior (MEDIUM).

---

## REV-02 · Metric registry contract

### How established tools do it

HF `evaluate` resolves names three ways: canonical Hub name via `evaluate.load("accuracy")`, a local script path, or a local directory whose script matches the directory name; canonical names are enumerable with `evaluate.list_evaluation_modules(module_type="metric")` ([loading methods docs](https://huggingface.co/docs/evaluate/package_reference/loading_methods)). **There is no centralized canonical→alias mapping table in `evaluate` itself** — aliasing exists only at the task level (`text-classification` ≡ `sentiment-analysis` in the Evaluator classes). Torchmetrics registers metrics as classes keyed by snake_case names; Lightning prefixes keys (`eval_/test_`) mechanically. The lesson for dnallm: canonical-name registries are standard, but the **alias layer is something benchmark authors add themselves** — which is precisely where the historical drift (`eval_auroc`/`eval_spearman_r` → `eval_AUROC`/`eval_spearmanr`) happened. A `{canonical_name: (fn, aliases)}` registry with `resolve()` that anchors current pipeline spellings and recognizes historical ones for identification-only is the ecosystem-consistent fix. The registry must import cleanly without torch/sklearn at module level so dnallmmark (F3) can import it in CI cheaply.

**Expected behavior:** `metrics.py` emits only canonical names (`AUROC`, `AUPRC`, `spearmanr`, `pearsonr` — current spellings per the intake fact-correction D2); `resolve(name)` returns the canonical entry and raises a matchable `ValueError` on unknown names (dnallm's 127-of-170 ValueError convention); aliases map old spellings for recognition; a round-trip contract test covers every metric key any task path can emit (47 tasks).

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Single `{canonical: (fn, aliases)}` registry module; `metrics.py` fully routed through it | evaluate/torchmetrics registry norms | LOW | `dnallm/tasks/metrics.py` emitters (AUROC :134/:135/:299/:306, spearmanr/pearsonr :181-219) |
| Table stakes | `resolve(name)` raising ValueError on unknown name | dnallm error convention; fail-loud | LOW | — |
| Table stakes | Contract tests: all emitted keys resolvable; alias→canonical correct for both current and historical spellings; cross-repo importable | the drift-prevention mechanism (D2: P0 before re-run) | LOW | pytest |
| Differentiator | Registry version stamp asserted by both repos' CI | makes contract drift visible in CI, not in a paper table | LOW | CI wiring |
| Anti-feature | Renaming canonical names (again) | the registry exists to stop exactly this | — | — |
| Anti-feature | Fuzzy/case-insensitive auto-coercion of arbitrary names | hides typos, reopens silent drift | — | — |
| Anti-feature | Torch/sklearn import at registry module import time | blocks lightweight cross-repo import (F3) | — | — |

**Sources:** HF evaluate docs (MEDIUM); no-alias-table finding is a negative claim verified against evaluate docs + searches (MEDIUM).

---

## REV-03 · Docs / terminology / comparability warnings

### How established tools do it

Terminology: "DNA language model(s)" is the standard term in this literature — e.g. "DNA language models are powerful zero-shot predictors of genome-wide variant effects" (GPN paper title), "The DNA dialect: a comprehensive guide to pretrained genomic language models" (Mol Syst Biol 2025) — so unifying on the editor's requested phrasing follows field convention. Comparability warnings: cross-model evaluation comparability is a recognized issue in the DNA-LM benchmarking literature (models differ in `valid_chars`; 13 models in the dnallm registry cannot accept N). The D3 ruling in the intake plan is consistent with how benchmark suites handle it: the **pipeline** applies one common eval subset; the **library** documents the constraint loudly at the API surface (`validate_sequences` docstring, data.py:842 / sequence.py:89) rather than adding a new API mode. CHANGELOG-per-fix with clickable commits is standard revision-evidence practice.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `validate_sequences` docstring + docs page on cross-model valid_chars comparability (N-containing sequences silently drop for 13 models) | D3 ruling; benchmarking-literature comparability concern | LOW | `check_sequence` (sequence.py:89), `validate_sequences` (data.py:842) |
| Table stakes | Terminology unification to "DNA large language models" per Ed-2 | field-standard phrasing (paper titles above) | LOW | docs tree |
| Table stakes | LoRA/QLoRA/IA³ usage chapter (after REV-04/05 land) | PEFT docs pattern: per-method quickstart | LOW | REV-04/05 |
| Table stakes | CHANGELOG entries mapping v0.7.1→v0.7.2 fixes to commits | reply-letter evidence chain | LOW | — |
| Differentiator | Published per-model-family minimal `valid_chars` table (generated from model_info.yaml) so users can pre-filter to the intersection themselves | turns a warning into an actionable artifact | LOW | model registry |
| Anti-feature | Changing `validate_sequences` default drop behavior, or adding a unified-subset API mode in the suite | ruled out by D3 (pipeline side implements it); silent behavior change breaks F7 audit | — | — |

**Sources:** paper titles / Mol Syst Biol guide (MEDIUM); D3 ruling is project-internal (HIGH, from intake doc).

---

## REV-04 · IA³ adapter support

### How established tools do it

Verified against the official PEFT docs (fetched directly — HIGH confidence): `IA3Config(task_type, target_modules, feedforward_modules, init_ia3_weights=True, modules_to_save, exclude_modules, fan_in_fan_out)` via `get_peft_model`. IA³ injects three learned vectors per block that rescale the **outputs of attention key/value projections** and the **input of the second FFN layer** — the critical semantic difference from LoRA-style configs is `feedforward_modules`, which must be a **subset of `target_modules`** and tells PEFT to apply the vector to the module *input* rather than *output*. The official recipe for autoregressive models is `target_modules=["k_proj","v_proj","down_proj"], feedforward_modules=["down_proj"]`; encoder-style (BERT-family) targets are `query/key/value` + the FFN intermediate dense layer. IA³ trains ~0.01% of parameters (LoRA >0.1%), merges into the base weights with zero inference latency, and uses the same `save_pretrained`/`PeftModel` lifecycle as LoRA — so dnallm's existing adapter save/reload path (inference.py:112-130) and the trainer's LoRA init branch (trainer.py:153-170) are the right seams. `init_ia3_weights=False` is officially discouraged. For Mamba-family models there is no canonical IA³ recipe; the SSM-PEFT literature (REV-05) warns that adapting SSM core parameters can break stability constraints — treat Mamba IA³ as experimental and validated only empirically (the intake's "1 transformer + 1 mamba model" acceptance).

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `Ia3Config` Pydantic model with `target_modules` (default from REV-05 presets) + per-family `feedforward_modules` | official IA3Config semantics | LOW-MED | configs.py BaseModel patterns |
| Table stakes | `TrainingConfig.use_ia3` branch in trainer init mirroring the LoRA branch; mutual exclusion `use_ia3`×`use_lora` → ValueError | both-on is undefined in PEFT practice | LOW-MED | trainer.py:153-170 |
| Table stakes | Adapter save/reload round-trip through the existing PeftModel path (byte-identical semantics to LoRA path) | PEFT unified adapter lifecycle | LOW | inference.py:112-130 |
| Table stakes | Runs end-to-end on 1 transformer + 1 Mamba model, one task each | intake acceptance | MED (GPU runs) | models registry |
| Differentiator | IA³ vs LoRA parity comparison exported to the same results table (F4 lane) | reviewer-facing comparability | LOW | REV-09 export shape |
| Anti-feature | Custom IA³ reimplementation instead of `peft.IA3Config/get_peft_model` | peft is already a dependency and battle-tested | — | — |
| Anti-feature | IA³ on Mamba SSM cores (A/B/C/Δ) | ICML25 ssm-peft: A must remain negative-definite; low-rank/vector perturbation of SSM cores underperforms and risks instability | — | — |

**Sources:** PEFT IA³ official docs (HIGH); ssm-peft stability finding (MEDIUM, cross-checked).

---

## REV-05 · Per-model PEFT target-module presets

### How established tools do it

PEFT itself auto-selects targets from `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` keyed on model type, and **raises an error when the architecture is unknown** (documented in `LoraConfig.target_modules` help text; confirmed in [peft #1289](https://github.com/huggingface/peft/issues/1289) where the Llama default is recorded as `["q_proj","v_proj"]`). Community registries exist for exactly this (easylora's model-support table: Llama/Qwen/Gemma → `q_proj,v_proj`; GPT-NeoX/Falcon/Bloom → fused `query_key_value`; GPT-2 → `c_attn`; BERT-family → `query,key,value`). Two hard lessons from the literature make verification-from-checkpoint non-negotiable: (1) **silent partial application** — PEFT #3554 (NemotronH hybrid): defaults covered only 7% of layers (4 attention of 56), DPO loss never moved; and PEFT #2556: custom Mamba kernels can silently skip LoRA layers, and PEFT keeps a forbidden list for Mamba modules (`in_proj` blocked in some versions). (2) **Architecture-appropriate targets** — the ICML-2025 systematic study of SSM PEFT ([arXiv 2410.09016](http://export.arxiv.org/pdf/2410.09016)) shows LoRA on Mamba *linear projections* (`in_proj`, `out_proj`, `x_proj`, `dt_proj`) matches full fine-tuning (GLUE 81.2 vs 80.5) while SSM-matrix targets underperform (76.9). Since dnallm loads DNA models via `trust_remote_code=True` with custom module names (DNABERT-2 BPE BertModel, Evo/StripedHyena, Caduceus/Mamba variants), PEFT's stock mapping cannot be trusted — presets must be generated/checked against each model's `config.json`-derived module names, with a dry-run validator that errors listing available candidates when a preset name matches nothing.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Family-keyed preset YAML (BERT/GPT/Mamba/Gemma/Llama/hybrid) with target_modules + recommended r, **verified from actual model module names, not guessed** | peft mapping + #3554/#2556 silent-failure lessons | MEDIUM (≈44-model verification workload, not algorithmic) | `PRETRAIN_MODEL_MAPS` (modeling_auto.py), model_info.yaml |
| Table stakes | `target_modules=None` → family auto-select with INFO log | peft auto-select norm + auditability | LOW | config plumbing |
| Table stakes | Dry-run validation: preset module name not found in model → hard error listing candidate module names | prevents the #3554/#2556 silent no-op class | LOW | — |
| Table stakes | Preset regression test (table locked against drift) | intake acceptance | LOW | pytest |
| Differentiator | Explicit "unsupported" entries for non-transformer families (GPN CNN etc.) rather than absence | turns silent failure into a loud contract | LOW | — |
| Anti-feature | Copying PEFT's stock mapping blindly | trust_remote_code DNA architectures use custom names outside the mapping | — | — |
| Anti-feature | Guessing module names from papers/READMEs | exactly the "臆测" the intake plan forbids | — | — |

**Sources:** PEFT docs/issues #1289/#2556/#3554 (MEDIUM-HIGH); easylora registry (MEDIUM); ssm-peft ICML25 + Memba (MEDIUM, cross-checked).

---

## REV-06 · `random_init=True` from-scratch loading

### How established tools do it

The canonical transformers pattern: `AutoModel.from_config(config)` / `ModelClass(config)` builds the architecture with random weights; `from_pretrained()` internally builds from config **then loads checkpoint weights** ([HF LLM course ch. 2](https://huggingface.co/learn/llm-course/zh-TW/chapter2/3), [transformers #26901 maintainer comment](https://github.com/huggingface/transformers/issues/26901), [#1283](https://github.com/huggingface/transformers/issues/1283)). The canonical horror story is Geneformer: a refactor swapped `BertModel.from_pretrained(path)` for `BertModel(self.config)` and the project **silently trained from scratch** until someone noticed — which is exactly why the intake plan requires a loud "randomly initialized" log plus a parameter hash as proof. Empirically, from-scratch baselines sit well below pretrained ones on downstream tasks (illustrative example: 6-layer DistilBERT on SST-2 ~80.7% scratch vs ~90.1% pretrained), which is the expected result shape for R2-5's learning-curve lane. dnallm's implementation should: run the normal config/tokenizer resolution through the dispatch chain, skip weight download/loading, re-init with `torch.nn.init` per model config, log the hash, and leave everything else (device, dtype, head attach) identical.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `load_model_and_tokenizer(..., random_init=True)`: config+tokenizer loaded, weights skipped, re-initialized | `from_config` idiom | LOW | dispatch chain (model.py:736); special handlers must tolerate weight-skip |
| Table stakes | Loud "randomly initialized" log + parameter hash; test asserts hash differs from pretrained path | Geneformer silent-scratch lesson | LOW | logging util |
| Table stakes | Two-architecture unit coverage + downstream loss-curve divergence check | intake acceptance | LOW-MED | pytest, tiny models |
| Differentiator | From-scratch rows exported in the same format as pretrained runs (F8 learning-curve lane compat) | makes the baseline drop into the paper table unchanged | LOW | REV-09 export shape |
| Anti-feature | random_init altering device/dtype/quantization behavior or bypassing special-family config resolution | baseline must differ from pretrained **only** in weights | — | — |
| Anti-feature | Seeding the init with the training seed silently | init seed and training seed are distinct provenance; record both | — | — |

**Sources:** HF course + transformers issues (MEDIUM-HIGH); Geneformer commit (HIGH legitimacy, public commit); gap example (LOW, illustrative single source — do not cite as benchmark).

---

## REV-07 · Frozen-embedding probing

### How established tools do it

The frozen-probing protocol is well standardized in the DNA-LM literature (cross-checked across three independent sources): backbone runs strictly in inference mode (no grads, no parameter updates); probes are **logistic regression** (tests linear accessibility) and/or a **shallow MLP** (nonlinear decodability); embeddings are **standardized with train-set statistics only** (no test leakage — same invariant as REV-01); results reported as mean ± sd over ≥5 seeds for stochastic probes; and a "Recovery" ratio (frozen-probe score ÷ full-fine-tune score × 100) contextualizes the probe against fine-tuning ([Frozen-but-Not-Always-Accessible, arXiv 2608.05329](https://arxiv.org/html/2608.05329v1); NT paper's own probing used 10 random 90/10 splits and found **intermediate layers often probe better than the last layer** — [bioRxiv 2023.01.11.523679](https://www.biorxiv.org/content/biorxiv/early/2024/10/07/2023.01.11.523679.full.pdf); reference implementation recipe: `StandardScaler` → `LogisticRegression(max_iter=1000, solver="lbfgs")` with acc/P/R/F1/MCC/AUC — [leannmlindsey/NTv2_generic_sequence_classification](https://github.com/leannmlindsey/NTv2_generic_sequence_classification/blob/main/embedding_analysis_nt.py)). DART-Eval's cross-cutting finding: probing underperforms fine-tuning but is far cheaper — which is the *point* of the R2-2 comparison lane. Layer selection must be exposed (not hardcoded to last layer) or the probe systematically misreports.

### Categories

| Kind | Feature | Grounding | Complexity | depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `extract_embeddings(...)` reusing the existing `scoring()` embedding path with selectable pooling + layer | mirrors NT/2608.05329 protocols; zero new model code | MED | scoring() (inference.py:1746), pooling helpers |
| Table stakes | `fit_probe(kind="logistic"\|"mlp")` with **fixed** hyperparameters, scaler fit on train only, dev-based early stop for MLP | standardization-leakage hygiene + comparability requires fixed probes | MED | sklearn (already a dep) |
| Table stakes | Metrics via REV-02 registry; export comparable with full fine-tune rows (F4 lane same-table) | the reviewer comparison is probe-vs-finetune | LOW | REV-02 |
| Table stakes | npz embedding cache keyed by (model, dataset, layer, pooling); second run hits cache | probing suites always cache embeddings (extraction dominates cost) | LOW | — |
| Table stakes | Layer selection parameter (not last-layer-only) | NT intermediate-layer finding | LOW | — |
| Differentiator | Layer-sweep probing curve report (per-layer probe metric) + Recovery% metric | NT-style analysis; strong paper figure | LOW | matplotlib/altair (altair already dep) |
| Differentiator | Composition with `DNADataset.sampling` for annotation-fraction learning curves (R2-5 synergy) | sampling already stratified-ratio+seed | LOW | datahandling |
| Anti-feature | Fine-tuning the backbone during "probing" | defeats the definition; it's just fine-tuning | — | — |
| Anti-feature | Probe hyperparameter search | unfixed probes destroy cross-model comparability (the lane's purpose) | — | — |
| Anti-feature | Deep GPU probes (many layers/epochs) | scope creep toward full fine-tuning; shallow MLP is the convention | — | — |

**Sources:** 2608.05329 + NT paper + reference repo (MEDIUM, mutually consistent); DART-Eval (MEDIUM).

---

## REV-08 · Zero-shot VEP module (largest scope)

### How established tools do it — three paradigms, all with real implementations

1. **CLM Δlog-likelihood** (Evo/Evo2, autoregressive): build ref/alt windows, compute per-token-average sequence log-likelihoods, score `delta_loglik = loglik(alt) − loglik(ref)`; negative ⇒ deleterious. Verified implementation reference: [ToolUniverse `evo2_variant_effect_tool`](https://zitniklab.hms.harvard.edu/ToolUniverse/zh-CN/_modules/tooluniverse/evo2_variant_effect_tool.html). Ecosystem practices that are effectively protocol requirements: **reverse-complement averaging** (DNA LMs are not RC-equivariant; (ΔLL_fwd+ΔLL_rev)/2 recommended), and **window size is a declared parameter** (an E. coli calibration study measured 5–7% AUROC loss from the 8kb default vs optimal 2–4kb windows; sanity controls: nonsense variants in essential genes ΔLL ≈ −46.5 vs benign synonymous ≈ −3.5). Magnitude calibration: Evo2-40B ClinVar AUROC ≈ 0.98.
2. **MLM log-odds / LLR at the variant position** (GPN: `log P(ALT)/P(REF)` with only the variant position masked — GPN's single-nucleotide tokenization was **chosen specifically** to make this well-defined; NT: overlapping 6-mer tokens, LLR at the variant token, ClinVar AUC ≈ 0.7–0.8 for the 2.5B model). DART-Eval notes masked objectives need iterative masking (cost) and — critically — that **BPE tokenizations are problematic because a single-base change can alter multiple tokens unpredictably**.
3. **Embedding distance** (BEND benchmark: cosine distance between the reference-nucleotide embedding and the variant-nucleotide embedding, 512 bp context, AUROC evaluated overall **and per variant-consequence type**; the Enformer-VEP family uses the same idea). This is the paradigm actually used for DNABERT-2-class BPE models in benchmarks — Feng et al. (Nat Commun 2025) scored DNABERT-2's zero-shot VEP this way and measured AUC 0.538 on pathogenic-vs-common (i.e., weak — an honest expectation), vs NT-v2 ≈ 0.73.

**The alignment pitfall — the core of R1-3e① — is real and documented:** with BPE, ref and alt alleles need not tokenize into the same token slot; the 2025 Mut-BPE paper shows standard BPE caps zero-shot VEP AUROC near 0.6 and recovers ~11% by splitting tokens to restore single-nucleotide resolution. dnallm's same-slot rule (ref/alt must fall in the same token slot in the same window, else **explicit skip with reason + count**) is the defensible protocol answer — GPN avoided the problem by architecture choice; BEND avoided it by not using likelihoods; dnallm faces it head-on because it scores 150+ heterogeneous tokenizers.

**Expected behavior:** `align_variant(seq,pos,ref,alt,tokenizer)` → alignable/skip verdict; `score_variant(paradigm="clm"|"mlm")` with formulas written into docstrings (protocol declaration); RC behavior and window size declared parameters with documented defaults; `evaluate_vcf` → per-variant scores + skip accounting + AUROC/AUPRC via REV-02; CLI entry. Skip accounting reported alongside metrics (e.g., "N evaluated / M skipped:slot-mismatch") — a skipped-variant fraction is itself a finding about the tokenizer.

### Categories

| Kind | Feature | Grounding | Complexity | Depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Same-slot alignment check + explicit skip with per-variant reason and aggregate skip counts | Mut-BPE/DART-Eval BPE findings; GPN's architecture choice | MED | tokenizer access via load_model_and_tokenizer |
| Table stakes | CLM Δlog-lik (per-token normalized, RC behavior declared, window parameter with documented default) | Evo/Evo2 protocol | MED | `clm_evaluate` kernel (mutagenesis.py:312) |
| Table stakes | MLM log-odds at masked variant position | GPN/NT protocol | MED | `mlm_evaluate` kernel (mutagenesis.py:258) |
| Table stakes | `evaluate_vcf(...)` → scores + skip stats + AUROC/AUPRC via registry | every VEP benchmark reports AUROC/AUPRC | MED | REV-02; VCF parsing (pysam/cyvcf2 — STACK lane) |
| Table stakes | CLI entry point + formulas in docstrings/README (protocol declaration) | intake: 评分公式写入 docstring | LOW | click CLI framework |
| Table stakes | ClinVar-sampled validation run (1k × ≥5 models) with literature-magnitude AUROCs | acceptance criterion; calibration numbers above | MED (GPU) | models.lock pattern from v1.1 |
| Differentiator | Embedding-distance paradigm (BEND-compatible) as third scoring mode | makes BPE-only models evaluable zero-shot; BEND/Enformer-VEP lineage | LOW-MED | scoring() embedding path |
| Differentiator | Per-consequence-type metric breakdown (splice/intronic/…) | BEND reports per-consequence AUROC | LOW | registry metrics |
| Differentiator | RC-averaging toggle (default on, per literature) | plant-genetics Evo paper; GPN averages strands | LOW | kernels operate on sequences already |
| Anti-feature | Imputing/heuristically rescoring non-alignable variants | the explicit skip IS the reviewer response; imputation would re-create the credibility problem | — | — |
| Anti-feature | Indels / multi-allelic / MNV support in v1 | even SNV alignment is the hard part (Mut-BPE); scope discipline | — | — |
| Anti-feature | Folding GPN-style explicit genomic models into the tokenizer-based protocol in v1 | owner open question #4: different scoring surface (per-position nucleotide probabilities, no tokenizer slots); design as extension point instead | — | — |
| Anti-feature | Any training/calibration on the eval set | "zero-shot" must stay zero-shot (same invariant as REV-01) | — | — |

**Sources:** GPN bioRxiv 2022.08.22.504706 (MEDIUM-HIGH); Evo2 ToolUniverse source + E. coli calibration bioRxiv (MEDIUM); Feng et al. Nat Commun 2025 (HIGH legitimacy journal, MEDIUM-HIGH overall); Mut-BPE bioRxiv (MEDIUM); BEND arXiv 2311.12570 + Polaris hub (MEDIUM-HIGH); NT paper (MEDIUM-HIGH).

---

## REV-09 · Multi-seed sweep protocol

### How established tools do it

The reporting convention across ML-methodology sources (cross-checked): run k seeds (3 is the common floor, ≥5 recommended for headline claims), report **mean ± standard deviation over seeds** — sd, not standard error, because SE shrinks arbitrarily with more runs and understates run-to-run variance (Varoquaux & Colliot, *Best Practices Guidelines* chapter) — and complement with a **bootstrap 95% CI** using the percentile method (2.5/97.5 percentiles over ~10³–10⁴ resamples) for metrics without closed-form standard errors (AUROC, AUPRC, macro-F1). Two documented caveats worth encoding as anti-features: overlapping mean±sd bars are **not** a significance test (paired tests like McNemar/paired-bootstrap are a separate concern, out of revision scope), and with very few seeds BCa correction has been recommended — overkill for k=3–5 here. Sweep runners (wandb sweeps, HF runs) conventionally materialize one directory per run with the seed in the path; dnallm's `{model}/{task}/seed_{s}/` layout + a JSON `statistics` block is the standard shape. The one dnallm-specific requirement: the bootstrap must be **seeded** (pure function with RNG seed recorded in the JSON) or the CI itself becomes irreproducible — the exact class of complaint (R1-2a) this REV answers.

### Categories

| Kind | Feature | Grounding | Complexity | depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | `run_seeds(fn, seeds, out_root)` → `{model}/{task}/seed_{s}/` layout | sweep-runner convention; must match dnallmmark F2 exactly | LOW-MED | `TrainingConfig.seed` (configs.py:291) |
| Table stakes | `aggregate_seeds(...)` pure function → mean/sd/ci95_bootstrap | mean±sd + percentile bootstrap convention | LOW | numpy only (no new dep) |
| Table stakes | Seeded bootstrap RNG; JSON `statistics` block records n_seeds, mean, sd, ci95, n_resamples, bootstrap seed | reproducibility of the CI itself | LOW | — |
| Table stakes | Aggregation unit tests on constructed arrays with known moments; ≥3-seed full-chain trial on a small task | intake acceptance | LOW | pytest |
| Differentiator | Per-seed provenance in each seed dir (config hash, git rev, timestamp) | strengthens the provenance-break complaint fix (R1-2d lineage) | LOW | — |
| Differentiator | Uniform hook with REV-01 `evaluate(split=...)` so every seed evaluates the same split the same way | eliminates seed-level eval-semantics drift | LOW | REV-01 |
| Anti-feature | Significance tests / p-value machinery in v1 | right tool but out of revision scope; report CIs only | — | — |
| Anti-feature | Reporting best-seed or median-seed as headline | defeats the protocol; always mean±spread over all seeds | — | — |
| Anti-feature | Mixing test-set bootstrap resampling with seed variance in one interval | conflates two variance sources; keep ci95 per seed-mean convention and document it | — | — |

**Sources:** Varoquaux & Colliot OAPEN chapter (MEDIUM-HIGH); CODECRUNCH evaluation notes; arXiv 2511.19794 / 2607.09816 (MEDIUM).

---

## REV-10 · JASPAR/CIS-BP PWM matching

### How established tools do it

FIMO (MEME Suite) is the reference semantics (official docs, HIGH legitimacy): scans each motif **independently** on **both strands**; scores are **log-odds in bits** against a (zero-order) background; a **dynamic-programming algorithm converts log-odds scores to p-values** under the background model; default reports matches with **p < 1e-4** (`--thresh`), and `--qv-thresh` switches selection to **Benjamini-Hochberg q-values** (FDR). MOODS (the standard Python-accessible scanner, used inside TOBIAS) reads JASPAR `.pfm` natively: `pfm_to_log_odds(matrix, bg, pseudocount)` → `threshold_from_p(matrix, bg, p)` → `Scanner(window).set_motifs(matrices, bg, thresholds).scan(seq, max_hits)`; the background is baked into the log-odds conversion (the scanner's bg argument doesn't affect results). A pure-Python alternative exists (motifmatchry — bit-identical to MOODS — but requires Python 3.12+/numba, conflicting with dnallm's 3.11–3.13 matrix). Practical implication for dnallm: the sliding log-odds scan is a trivial numpy windowed dot-product; **p-value calibration is the only hard part** — FIMO's exact DP is overkill for a v1 hotspot scanner; an empirical null (scores on GC/shuffle-matched null windows) with BH correction is a defensible, honest calibration, with the exact-DP threshold as a future refinement. JASPAR distributes PFMs directly (meme/pfm formats); CIS-BP likewise exports PFM/TRANSFAC — format loading is table stakes.

### Categories

| Kind | Feature | Grounding | Complexity | depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | PFM loading (JASPAR `.pfm` + MEME format) with pseudocount and configurable background (uniform default; sequence-derived option) | MOODS API semantics | LOW-MED | numpy |
| Table stakes | Both-strand scan of hotspot windows; output table: motif ID, coords, strand, log-odds score, p (or q), matched sequence | FIMO output schema | LOW-MED | mutagenesis hotspot windows (`prepare_tfmodescan_inputs` mutagenesis.py:583) |
| Table stakes | Threshold by p-value (default 1e-4, FIMO convention) + Benjamini-Hochberg FDR option | FIMO default + `--qv-thresh` | MED (calibration) | scipy (already dep) |
| Table stakes | HBG1 case: BCL11A motif hit coordinates match the paper's Fig 4a annotation | intake acceptance — the credibility anchor | LOW (given above) | committed test locus pattern from v1.1 showcase |
| Differentiator | E-value reporting (p × scan space size) | FIMO reports E-values; reviewers recognize them | LOW | — |
| Differentiator | Background estimated from scanned hotspot windows (composition-matched null) | MOODS `bg_from_sequence` analog; improves plant-genome scans where uniform bg is wrong | LOW | — |
| Anti-feature | De novo motif discovery (MEME/STREME's job) | dnallm's story is scoring found hotspots, not discovering motifs | — | — |
| Anti-feature | TF-MoDISco integration | chain already stops at `prepare_tfmodesc_inputs` by design; leave it | — | — |
| Anti-feature | Genome-wide scanning service / bigwig export | scope explosion; hotspot windows only in v1 | — | — |
| Anti-feature | Claiming FIMO-identical p-values with an empirical null | state the calibration method honestly in docs (reviewers will ask) | — | — |

**Sources:** FIMO official docs meme-suite.org (HIGH legitimacy, MEDIUM-HIGH overall); MOODS wiki + TOBIAS example (MEDIUM); motifmatchry PyPI (MEDIUM).

---

## REV-11 · MCP tools (ism_scan / hotspots / zero_shot_score)

### How established tools do it

FastMCP conventions (official docs, HIGH legitimacy): tools are `@mcp.tool`-decorated functions; **name** (snake_case, action-oriented) and **docstring-derived description** are what LLM clients select on; input schema is generated from type annotations (no `*args`/`**kwargs`); errors surface via `ToolError` → `isError=True` result, and unhandled exceptions are auto-converted so a tool never crashes the server ([FastMCP tools docs](https://gofastmcp.com/servers/tools), [python-sdk error-handling commit](https://github.com/modelcontextprotocol/python-sdk/commit/c68e254bad1dd39e6a10dad43d954c6d17f9f514)). dnallm's existing MCP layer already matches this contract (11 tools, timeout wrappers `_with_timeout_wrapper` server.py:282, error dicts rather than raising across the boundary, executor-bridged blocking loads in ModelManager). There is direct precedent for agent-facing VEP: Harvard Zitnik Lab's ToolUniverse exposes an `evo2_variant_effect_tool` with delta-log-lik semantics — i.e., a `zero_shot_score` MCP tool has a real-world analog, not a novelty. Tool naming should follow the established action-oriented snake_case (`ism_scan`, `hotspots`, `zero_shot_score` already conform). The carried v1.1 audit item (CLI `--host/--port` silently overridden by YAML) is folded in: flag-parse precedence (CLI > YAML > default) is the universal CLI convention.

### Categories

| Kind | Feature | Grounding | Complexity | depends on (dnallm) |
|------|---------|-----------|------------|---------------------|
| Table stakes | Three tools wrapping existing/new classes with existing timeout + error-dict + ModelManager patterns | FastMCP conventions; repo's own 11-tool contract | LOW-MED | Mutagenesis (ism_scan), hotspot extraction (hotspots), REV-08 scorer (zero_shot_score) |
| Table stakes | Handshake regression tests: server up → client calls 3 tools → JSON assertions | repo's existing MCP test pattern (v1.1 mcp_example probes) | LOW | dnallm/mcp/tests |
| Table stakes | `--host/--port` CLI-override fix with tests (CLI > YAML > default) | v1.1 audit W item; universal CLI precedence | LOW | server.py main/argparse |
| Table stakes | Docstrings written for LLM consumption (what it does, parameter semantics, skip behavior for zero_shot_score) | name+description drive selection | LOW | — |
| Differentiator | `zero_shot_score` exposing skip accounting (evaluated/skipped counts) in its result JSON | agents can report data-quality caveats to users; differentiates from ToolUniverse's bare delta | LOW | REV-08 |
| Anti-feature | Unbounded/whole-genome scans via MCP without result-size caps or async job semantics | blocks event loop / explodes context; cap inputs + return summaries | — | — |
| Anti-feature | Adding tools whose underlying feature doesn't exist yet (REV-11 before REV-08/10) | a tool surface is a contract; wrap working code | — | — |
| Anti-feature | Streaming large result tables | return chunked/capped JSON summaries consistent with existing tools | — | — |

**Sources:** FastMCP official docs (HIGH legitimacy, MEDIUM-HIGH); python-sdk (MEDIUM-HIGH); ToolUniverse (MEDIUM).

---

## Feature Dependencies

```
REV-02 metric registry ──required──> REV-07 (probe metrics)
                     └──required──> REV-08 (VEP AUROC/AUPRC)
                     └──required──> REV-09 (statistics keys) [soft: aggregation can emit raw]
REV-05 presets ──required──> REV-04 (IA3 target defaults)
REV-04/05 ──then──> REV-03 (usage docs chapter after adapters land)
REV-08 VEP module ──required──> REV-11 zero_shot_score tool
REV-10 motif module ──required──> REV-11 hotspots tool (hotspot windows feed scans)
[existing] Mutagenesis mlm/clm kernels ──reused──> REV-08
[existing] scoring() embedding path ──reused──> REV-07, REV-08(embedding paradigm)
[existing] infer() predict path ──reused──> REV-01 evaluate(split=...)
[existing] TrainingConfig.seed ──reused──> REV-09
REV-01 ──enhances──> REV-09 (uniform per-seed evaluation split)
REV-06 ──enhances──> REV-07 (probe-vs-scratch-vs-pretrained three-way comparison)
```

### Dependency Notes

- **REV-02 before everything metric-emitting:** probing (REV-07), VEP (REV-08) and sweep aggregation (REV-09) all emit metrics; landing them before the registry means re-touching their outputs later. This matches the intake's Phase A gating.
- **REV-05 before REV-04:** IA³'s target_modules default comes from the preset table; the intake dependency chain (REV-04 → REV-05 → F4) resolves to "build presets first, IA³ consumes them".
- **REV-11 conflicts with nothing but requires REV-08 (zero_shot_score) and benefits from REV-10 (hotspots)**: `ism_scan` wraps existing Mutagenesis and could land independently.
- **REV-01/02/03/06/09 are mutually independent** — parallelizable wave candidates; REV-08 is the long pole.

## MVP Definition

### Launch With (Phase A — P0, gate for the benchmark re-run)

- [ ] REV-01 eval-semantics leak guard + `evaluate(split=...)` — prevents reproducing the R1-2c leak in the re-run
- [ ] REV-02 metric registry contract — must exist before any new results are generated (D2)
- [ ] REV-03 comparability warnings + terminology + CHANGELOG — reviewer-facing evidence chain (docs chapter for IA³ lands with Phase B)

### Add After Validation (Phase B — P1, parallelizable with re-run)

- [ ] REV-05 presets then REV-04 IA³ — R2-2 adaptation lane
- [ ] REV-06 random_init — unblocks F8 learning curves (R2-5)
- [ ] REV-07 probing — R2-2 probe lane
- [ ] REV-08 zero-shot VEP — E5 lane (long pole; start alignment/scoring kernels early)
- [ ] REV-09 multi-seed protocol — F2 dependency; pure functions can land immediately

### Future Consideration (Phase C — P2, post-submission commitment)

- [ ] REV-10 JASPAR/CIS-BP matching — manuscript uses external-annotation wording meanwhile (B-plan)
- [ ] REV-11 MCP tools — narrative value (R2-7/Ed-6); strictly after REV-08/10 exist

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Complexity (from sections) |
|---------|------------|---------------------|----------|------------------------------|
| REV-01 leak guard | HIGH (re-run integrity) | LOW | P0 | LOW-MED |
| REV-02 metric registry | HIGH (re-run integrity) | LOW | P0 | LOW |
| REV-03 docs/warnings | MED (reply-letter evidence) | LOW | P0 | LOW |
| REV-05 PEFT presets | HIGH (unblocks REV-04, silent-failure prevention) | MED | P1 | MED |
| REV-04 IA³ | MED-HIGH (R2-2) | LOW-MED | P1 | LOW-MED |
| REV-06 random_init | MED-HIGH (R2-5) | LOW | P1 | LOW |
| REV-07 probing | MED-HIGH (R2-2) | MED | P1 | MED |
| REV-08 zero-shot VEP | HIGH (R2-3 + R1-3e①, new capability class) | HIGH | P1 | HIGH |
| REV-09 multi-seed | HIGH (R1-2a/R2-4) | LOW-MED | P0→P1 (aggregation pure functions can be P0-adjacent) | LOW-MED |
| REV-10 motif matching | MED (post-submission) | MED | P2 | MED |
| REV-11 MCP tools | MED (narrative) | LOW-MED | P2 | LOW-MED |

## Comparable-Tool Feature Analysis

| Capability | How comparables do it | dnallm plan |
|------------|----------------------|-------------|
| Zero-shot VEP scoring | GPN: MLM log-odds, single-nuc tokenizer; Evo/Evo2: CLM Δlog-lik + RC averaging; NT: 6-mer LLR at variant token; BEND: embedding cosine distance; DNABERT-2 in benchmarks: embedding distance (LLR ill-defined under BPE) | All three likelihood/embedding paradigms behind `score_variant(paradigm=...)`, same-slot alignment with explicit skips — the differentiator is doing this uniformly across 150+ heterogeneous tokenizers, which none of the comparables attempt |
| Variant-token alignment | Comparables avoid it by architecture (GPN single-nuc) or by paradigm choice (BEND embeddings); Mut-BPE rewrites tokenization | Same-slot rule + skip accounting — a protocol contribution, directly answering R1-3e① |
| Metric naming | evaluate: canonical Hub names, no alias table; torchmetrics: class registry | Own `{canonical: (fn, aliases)}` registry shared cross-repo with contract tests |
| PEFT target selection | peft: internal arch mapping, error on unknown; ssm-peft: Mamba linear projections; silent partial application documented in issues | Verified per-model preset YAML + dry-run validator + explicit-unsupported entries |
| IA³ | peft IA3Config with feedforward_modules subset semantics | Ia3Config pydantic mirror + presets; Mamba marked experimental |
| Probing | frozen backbone + LR/shallow-MLP, train-only scaler, layer sweep, Recovery% | probing.py reusing scoring() embedding path; fixed hyperparams |
| Motif scanning | FIMO: log-odds bits, DP p-values, p<1e-4, BH q-values; MOODS: JASPAR pfm, threshold_from_p | Hotspot-window scanner, empirical-null calibration + BH, FIMO-compatible output schema |
| Agent VEP tools | ToolUniverse evo2_variant_effect_tool (single model, bare ΔLL) | zero_shot_score over the whole registry with skip accounting |
| From-scratch baselines | transformers from_config idiom; Geneformer silent-scratch incident as cautionary tale | random_init=True + loud log + param hash |

## Sources

- HF Trainer eval semantics: [Trainer docs](https://huggingface.co/docs/transformers/v4.38.2/ja/main_classes/trainer), [training_args.py](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/training_args.py), [SO 76310533](https://stackoverflow.com/questions/76310533/how-to-fix-trainer-evaluation-requires-an-eval-dataset-in-huggingface-transfo), [TRL caveat](https://theneuralbase.com/sft/learn/beginner/evaluation-during-training/) — MEDIUM
- Metric registries: [evaluate loading methods](https://huggingface.co/docs/evaluate/package_reference/loading_methods), [evaluate README](https://raw.githubusercontent.com/huggingface/evaluate/main/README.md) — MEDIUM
- PEFT IA³: [official IA3 docs](https://huggingface.co/docs/peft/package_reference/ia3) — HIGH (official, directly fetched)
- PEFT target selection: [peft #1289](https://github.com/huggingface/peft/issues/1289), [easylora registry](https://alexsuw.github.io/easylora/model-support/), [peft #2556 / #3554] via Mamba-PEFT search — MEDIUM
- SSM/Mamba PEFT: [arXiv 2410.09016 (ICML25)](http://export.arxiv.org/pdf/2410.09016), [Memba arXiv 2506.18184](https://ar5iv.labs.arxiv.org/html/2506.18184) — MEDIUM
- From-scratch loading: [HF LLM course](https://huggingface.co/learn/llm-course/zh-TW/chapter2/3), [transformers #1283](https://github.com/huggingface/transformers/issues/1283), [#26901](https://github.com/huggingface/transformers/issues/26901), [Geneformer silent-scratch commit](https://huggingface.co/ctheodoris/Geneformer/commit/33693688b3eaa03e4ed385a115e3a74ae37af9ff) — MEDIUM-HIGH
- Probing: [arXiv 2608.05329](https://arxiv.org/html/2608.05329v1), [NT paper](https://www.biorxiv.org/content/biorxiv/early/2024/10/07/2023.01.11.523679.full.pdf), [NTv2 probe reference impl](https://github.com/leannmlindsey/NTv2_generic_sequence_classification/blob/main/embedding_analysis_nt.py), [DART-Eval](https://arxiv.org/html/2412.05430) — MEDIUM
- Zero-shot VEP: [GPN paper](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v2.full), [Evo2 delta-loglik tool](https://zitniklab.hms.harvard.edu/ToolUniverse/zh-CN/_modules/tooluniverse/evo2_variant_effect_tool.html), [Evo2 calibration study](https://www.biorxiv.org/content/10.64898/2026.07.02.736037v1), [Feng et al. Nat Commun 2025](https://www.nature.com/articles/s41467-025-65823-8), [Mut-BPE](https://www.biorxiv.org/content/biorxiv/early/2025/12/01/2025.12.01.691503.source.xml), [BEND](https://github.com/dna-llm/BEND) + [Polaris task](https://polarishub.io/benchmarks/mlls/bend-zeroshot-variant-effects-disease) — MEDIUM-HIGH (multi-source)
- Multi-seed reporting: [Varoquaux & Colliot chapter (OAPEN)](https://library.oapen.org/handle/20.500.12657/75361), [arXiv 2511.19794](https://browse-export.arxiv.org/pdf/2511.19794), [arXiv 2607.09816](https://arxiv.org/pdf/2607.09816v3) — MEDIUM
- Motif scanning: [FIMO official docs](https://meme-suite.org/meme/doc/fimo.html), [MOODS wiki](https://github.com/jhkorhonen/MOODS/wiki/Brief-theoretical-introduction), [TOBIAS MOODS example](https://github.molgen.mpg.de/loosolab/TOBIAS-nextflow/blob/ae165153649f375895f74e9f956b75bad7ab993a/TOBIAS_MAPOKS/dockerfile/old_TOBIAS/MOODS-python-1.9.2/scripts/ex-scanner.py), [motifmatchry](https://pypi.org/project/motifmatchpy/) — MEDIUM-HIGH (FIMO official)
- MCP conventions: [FastMCP tools](https://gofastmcp.com/servers/tools), [python-sdk error handling](https://github.com/modelcontextprotocol/python-sdk/commit/c68e254bad1dd39e6a10dad43d954c6d17f9f514), [MCP spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — MEDIUM-HIGH

---
*Feature research for: DNALLM v1.2 Paper Revision Suite Support (REV-01…REV-11)*
*Researched: 2026-10-09*
