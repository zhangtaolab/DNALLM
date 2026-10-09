# Requirements: DNALLM

**Defined:** 2026-10-09
**Core Value:** Ship the suite-side capabilities the paper revision requires so the dnallmmark full re-run and the reviewer-requested experiments (E1'–E8') can start — on a codebase whose test suite, coverage gate, and CI honesty remain fully green throughout.

## v1.2 Requirements

Requirements for milestone v1.2 "Paper Revision Suite Support". Each maps to roadmap phases (traceability filled at roadmap creation). REQ-IDs are 1:1 with intake REV-01..REV-11 (`.planning/research/261009-paper-revision-suite-plan.md`); acceptance criteria fold in the research convergences from `.planning/research/SUMMARY.md` (2026-10-09, four-lane research: STACK/FEATURES/ARCHITECTURE/PITFALLS).

### Evaluation Semantics (EVAL)

- [ ] **EVAL-01** (REV-01): The trainer never silently uses the test split as the eval set — when no dev split exists, `eval_strategy="no"` AND `eval_dataset=None` are set atomically and `load_best_model_at_end` defaults to False; an explicit opt-in override (`allow_test_as_eval=True`) WARNs loudly; the guard covers the early-stopping neighbor path (trainer.py:298-303, which force-enables `load_best_model_at_end`) and raises a descriptive `ValueError` on collision with user-set best-model loading; a new `evaluate(split="test"|"dev"|...)` explicit entry point evaluates via the predict path; unit tests cover dev+test / test-only / train-only × default/override plus the early-stopping collision case; trainer docstring documents eval-set selection and held-out semantics

### Metric Contract (METR)

- [ ] **METR-01** (REV-02): A metric registry lands at `dnallm/tasks/metric_registry.py` (sibling of `metrics.py`, OUTSIDE the vendored `dnallm/tasks/metrics/` coverage/ruff/mypy-excluded glob — same-change check that the registry row is coverage-visible): a single `{canonical_name: (fn, aliases)}` mapping with a `resolve(name)` API raising a matchable `ValueError`; canonical names anchor CURRENT spellings (`AUROC`/`AUPRC`/`spearmanr`/`pearsonr`); aliases (`eval_auroc`, `eval_spearman_r`, …) are recognition-only and never emitted; `metrics.py` emits exclusively through the registry; contract unit tests cover every metric key used across the benchmark task set; the module is import-light (no torch/sklearn at import) so dnallmmark CI can import it cheaply

### Documentation (DOCS)

- [ ] **DOCS-01** (REV-03): Terminology unified to "DNA large language models"; `validate_sequences` gains a docstring + docs cross-model `valid_chars` comparability warning including a dropped-row count log line (D3 adjudication: unified-subset filtering itself stays pipeline-side); a LoRA/QLoRA/IA³ usage chapter (IA³ part completes after PEFT-01); CHANGELOG records one entry per revision fix, each traceable to its commit (rebuttal-letter evidence chain); docs build stays green under the docs-validation gate

### PEFT Adapters (PEFT)

- [ ] **PEFT-01** (REV-04): IA³ adapter support — `Ia3Config` (Pydantic) + `TrainingConfig.use_ia3`; trainer init branch symmetric to LoRA (trainer.py:153-170 shape) sharing the adapter save/reload path (PeftModel reuse at inference.py:111-131); `use_ia3 × use_qlora` rejected at Pydantic config time with a matchable `ValueError`; acceptance: one transformer model AND one Mamba model each fine-tune one task; IA³ adapter save/reload roundtrip test (peft #2429 corruption class); `lora × ia3` combination likewise rejected
- [ ] **PEFT-02** (REV-05): Per-model PEFT target-module presets — `dnallm/configuration/presets/lora_targets.yaml` packaged in the wheel (`[tool.setuptools.package-data]` + importlib.resources; NOT repo-root `configs/` which is unpackaged), recording `target_modules` and recommended `r` per architecture family (BERT/GPT/Mamba/Gemma/Llama/hybrid), derived from real `config.json` module names (never guessed); `target_modules=None` auto-selects by family with a log line; a dry-run validator errors on wrong module names (peft silently skips non-matching modules on Mamba/hybrid — the validator plus a runtime trainable-parameter-count guard are the countermeasures); ~44 benchmark models covered; presets-table regression tests

### From-Scratch Baselines (BASE)

- [ ] **BASE-01** (REV-06): `load_model_and_tokenizer(..., random_init=True)` — `AutoConfig.from_pretrained` + `AutoModel*.from_config` (never `from_pretrained`), skipping pretrained weights and re-initializing; a loud "randomly initialized" log plus parameter-hash proof; acceptance: per-tensor hashes differ from the pretrained path (a single global hash passes with leftover pretrained tensors), CPU-canonical seeding, same-seed reproducibility, no-download assertion, tokenizer still loads normally; supported on generic Auto* families only — special-family handlers raise an explicit `ValueError`; two architectures covered by tests

### Probing (PROB)

- [ ] **PROB-01** (REV-07): `dnallm/inference/probing.py` — `extract_embeddings(...)` (reuses the scoring embedding path; layer and pooling selectable) + `fit_probe(kind='logistic'|'mlp')` (sklearn, fixed hyperparameters, scaler fit on train split only); probe metrics emitted through the METR-01 registry; embeddings cached to npz keyed by (model, dataset, layer, pooling) with second-run cache hits asserted; acceptance: any model × any binary classification task end-to-end; output schema documented for the dnallmmark F4 lane

### Zero-Shot VEP (VEP)

- [ ] **VEP-01** (REV-08): `dnallm/inference/vep.py` — `align_variant(seq,pos,ref,alt,tokenizer)` same-slot evaluability rule (ref/alt must tokenize into the identical token slot; otherwise the variant is explicitly skipped with reason + count — the skip fraction is itself reported as a finding; this is the protocol answer to reviewer R1-3e①); `score_variant(paradigm='clm'|'mlm')` reusing the mutagenesis.py:258/312 kernels with a paradigm↔architecture mismatch guard; `evaluate_vcf(...)` (stdlib VCF reader — ~100 lines, gzip+str-splitting; cyvcf2/pysam rejected: no Windows wheels) yielding per-variant scores + skip accounting + AUROC/AUPRC via the registry, with VCF coordinate-system fixtures; a CLI entry point; scoring formulas written into docstrings and README (protocol declaration); acceptance: ClinVar 1k-sample × ≥5 models (CLM/MLM mix) producing literature-magnitude AUROCs, same-slot-differs-by-exactly-one asserted in tests

### Multi-Seed Protocol (SEED)

- [ ] **SEED-01** (REV-09): `dnallm/finetune/sweep.py` — `run_seeds(fn, seeds, out_root)` with the directory protocol `{model}/{task}/seed_{s}/`; `aggregate_seeds(...)` as a pure function returning mean/sd/ci95 via a SEEDED percentile bootstrap (BCa degenerates at n=3) with an n<10 guard (omit CI or t-interval — never a vacuous bootstrap); a result-JSON `statistics` block spec; acceptance: aggregation unit tests against constructed known arrays; ≥3-seed trial run of one small task end-to-end; directory protocol consistent with dnallmmark F2

### Motif Matching (MOTIF)

- [ ] **MOTIF-01** (REV-10): `dnallm/interpret/motifs.py` — hotspot windows ↔ JASPAR/CIS-BP PWM similarity scan following FIMO conventions (GC-matched background, both strands, log-odds threshold p<1e-4, BH FDR via `scipy.stats.false_discovery_control`), emitting a motif-ID/coordinates/E-value table; a stdlib JASPAR REST client with the base URL parameterized (canonical host migrated to jaspar.elixir.no; prefer `format=meme`); acceptance: the HBG1/BCL11A motif hit coordinates match the paper's Fig 4a annotation; p-value calibration method documented honestly (empirical-null vs exact-DP choice recorded)

### MCP Expansion (MCPE)

- [ ] **MCPE-01** (REV-11): The MCP server gains `ism_scan`, `hotspots`, `zero_shot_score` tools wrapping existing classes (existing `_with_timeout_wrapper` + error-dict-not-raise conventions; `zero_shot_score` wraps the VEP-01 module); handshake regression tests: server up → client calls all 3 tools → JSON assertions; the known `--host/--port` silently-overridden-by-yaml bug (v1.1 audit W-item) is fixed on BOTH sse and streamable-http paths with CLI-precedence tests

## Future Requirements

Deferred (v1.3+, tracked for the rebuttal letter's "future versions" commitments):

- **VEP-GPN**: GPN-class explicit-genomic (tokenizer-less) models in the unified VEP scoring protocol — extension point, alignment rules need extension
- **VEP-INDEL**: indel / multi-allelic variant support
- **STAT-TEST**: significance-testing machinery over multi-seed results
- **TFMODISCO**: TF-MoDISco integration beyond `prepare_tfmodescan_inputs`
- **GENOMEWIDE-SCAN**: genome-wide motif scanning (hotspot windows only in v1.2)
- **MCP-2X**: mcp SDK 2.x upgrade (explicitly NOT this milestone)

## Out of Scope

| Item | Reason |
|------|--------|
| dnallmmark repository changes (F1–F10) | Companion repo, tracked in its own plan; cross-repo contract tests import THIS repo's registry |
| Manuscript text edits (E#) | Revision plan v1/v2 docs own these |
| Benchmark re-run compute | Owner-scheduled after the P0 contract layer lands |
| cyvcf2 / pysam | No Windows wheels — would break the Windows CI leg; stdlib reader instead |
| biopython / statsmodels / torchmetrics / any VEP framework | Wrong footprint; scipy+sklearn+stdlib suffice (STACK.md rejected list) |
| mcp SDK 2.x upgrade | Out of milestone; current pin `>=1.3.0,<2` covers all needs |
| New `dnallm/__init__.py` re-exports | Facade stays byte-stable; parallel-agent collision avoidance (research convergence) |

## Traceability

Filled at roadmap creation. Phase mapping follows the research-recommended three-phase structure (contract layer → adaptation+evaluation → narrative), phases starting at Phase 10.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-01 | TBD | Pending |
| METR-01 | TBD | Pending |
| DOCS-01 | TBD | Pending |
| PEFT-01 | TBD | Pending |
| PEFT-02 | TBD | Pending |
| BASE-01 | TBD | Pending |
| PROB-01 | TBD | Pending |
| VEP-01 | TBD | Pending |
| SEED-01 | TBD | Pending |
| MOTIF-01 | TBD | Pending |
| MCPE-01 | TBD | Pending |

**Coverage:**
- v1.2 requirements: 11 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 11 (until roadmap creation) ⚠️

---
*Requirements defined: 2026-10-09*
*Last updated: 2026-10-09 after research synthesis (SUMMARY.md) and owner confirmation*
