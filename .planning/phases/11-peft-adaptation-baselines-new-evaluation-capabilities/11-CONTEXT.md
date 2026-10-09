# Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities - Context

**Gathered:** 2026-10-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase makes every reviewer-experiment capability work end-to-end: IA³ fine-tuning exactly as easy as LoRA with per-family preset targets, genuinely-from-scratch baseline loading, frozen-embedding probing, zero-shot variant scoring from VCF (completing the Phase-10 vep.py core with `evaluate_vcf` + CLI via the owner-approved scikit-allel reader), and multi-seed sweeps with honest uncertainty aggregates. Executed as 5 file-disjoint agents (owner-fixed): B1 REV-04+REV-05 (configs.py/trainer.py/inference.py hot files), B2 REV-06 (model.py sole owner), B3 REV-07 probing (new file, registry read-only), B4 REV-09 sweep (new file, pure aggregation first), B5 REV-08 completion (vep.py completion + cli.py sole owner + scikit-allel dependency).

Everything motif/MCP/closeout belongs to Phase 12.

</domain>

<decisions>
## Implementation Decisions

### IA³ integration & presets (B1)
- **D-01:** IA³ presets use **explicit per-family target-module lists** (attention-value + FFN targets, `feedforward_only=False`), derived from real `config.json` module names the same way LoRA presets are — NOT peft's coarse `feedforward_only=True` default. Explicit lists are what make the ~44-model presets-table regression tests meaningful.
- **D-02:** IA³ acceptance ("one transformer-family AND one Mamba model each fine-tune one task") runs on **small fast models already pinned in models.lock** — acceptance must be repeatably runnable, not bound to large models.
- **D-03:** The dry-run validator lands as a **`TrainingConfig.peft_dry_run: bool` field** handled inside the trainer (B1's own files) — B5 solely owns cli.py in this wave, so B1 must not add CLI surface.
- **D-04:** The runtime trainable-parameter-count guard **fails hard**: `ValueError` naming the preset, the expected trainable-ratio band, and the actual count. A silent module-skip is exactly what the guard exists to catch (research Pitfall #4).

### random_init from-scratch loading (B2)
- **D-05:** The per-tensor parameter-hash proof emits through **`get_logger` INFO lines** (short-hash table alongside the loud "randomly initialized" banner) — greppable, no sidecar file, no output_dir dependency.
- **D-06:** The two proven architectures are **one generic AutoModel family (BERT-style small model, exercising the `from_config` path) + one special-family allowlist member (mamba path)**; the explicit `ValueError` for unsupported special families is tested with a mock/other family.
- **D-07:** The special-family allowlist is a **module-level `frozenset` in model.py** (`RANDOM_INIT_SUPPORTED_FAMILIES`), documented in README — not config-driven this milestone.

### VEP completion × scikit-allel (B5)
- **D-08:** The dependency lands as **`scikit-allel>=1.3.13,<2`** — bounded range per project convention; the maintenance-mode library's successor (sgkit) justifies the upper guard.
- **D-09:** ClinVar test data is **two-tier**: a committed small synthetic VCF fixture for the fast lane (alignment rules + AUROC-shape with mocked model scores) + the real ClinVar 1k download in the `slow` lane (typed network skip + models.lock rows). Real data stays out of the repo; the fast lane stays network-free.
- **D-10:** CLI entry mirrors the `dnallm-mutagenesis` precedent: **`dnallm/cli/vep.py` + `dnallm-vep` console script** (one facade, one entry point).
- **D-11:** The paradigm↔architecture mismatch guard **raises `ValueError`** (e.g., CLM scoring on a bidirectional MLM) — per research Pitfall #6, mismatched-paradigm AUROCs are *expected* near-random; a silent skip would let a misconfiguration masquerade as a finding. Skip-as-data remains reserved for the same-slot alignment rule (its own documented channel).

### Probing & sweep interfaces (B3/B4)
- **D-12:** Embeddings cache defaults to **`output_dir/probe_cache/`** npz files keyed by (model, dataset, layer, pooling) — never CWD writes (Phase-10 lesson, IN-03).
- **D-13:** `fit_probe` fixed hyperparameters live as **module-level constants in probing.py with docstrings** — SC3 fixes them this milestone; no `ProbeConfig` YAML surface.
- **D-14:** `aggregate_seeds` n-guard: **t-interval CI (with `method: "t"` field) for n≥3; `ci95=None` for n<3** — the bootstrap path is refused at small n (Pitfall #14), never vacuous.
- **D-15:** The ≥3-seed end-to-end acceptance targets the **same models.lock small models as IA³ acceptance**, `slow`-marked, on a small binary classification task.
- **D-16 (research-grounded default):** `run_seeds` threads ONE seed into every stochastic stage with **same-data-split-across-seeds semantics** (split seed derived from the dataset, not the sweep seed) so seed-to-seed variance measures init/shuffle only; documented in docstring + asserted by a same-change determinism test.

### Claude's Discretion
- Exact small-model choices from models.lock for acceptance runs (B1/B2/B4) — any already-pinned small model with a fast CPU path qualifies.
- `validate_peft_targets` report structure (fields/format) inside the trainer.
- The synthetic VCF fixture's exact variant set (must cover SNV + indel + boundary + multi-allelic cases).
- npz cache file naming under `probe_cache/` (beyond the (model, dataset, layer, pooling) key contract).
- `run_seeds` out_root layout details beyond the locked `{model}/{task}/seed_{s}/` protocol.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/inference/vep.py` (Phase 10) — `align_variant` same-slot rule + `VariantAlignment` skip-as-data + CLM/MLM kernels (`clm_log_likelihood`, `mlm_slot_log_prob`) with formula docstrings; B5 wraps these, must not re-implement
- `dnallm/tasks/metric_registry.py` (Phase 10) — `resolve()`/`registered_names()`; probing/VEP/sweep metrics emit through it (read-only for B3/B4)
- `dnallm/finetune/trainer.py` — Phase 10 landed `allow_test_as_eval`, `use_ia3` field (no validators yet — B1 adds the cross-field rejections), LoRA/QLoRA paths (`get_peft_model` at inference.py:111-131 region), result-JSON writer pattern (`eval_{split}_result.json`) B4's seed-JSON should mirror
- `dnallm/configuration/configs.py` — field-complete Ia3Config/VepConfig/SweepConfig stubs + `load_config()` registration (Phase 10); B1 refines Ia3, B4 consumes SweepConfig, B5 consumes VepConfig — neither touches the section registry
- `LoraConfig`/LoRA trainer branch — the exact pattern the IA³ branch mirrors (peft #2429 save/reload corruption class already covered by LoRA roundtrip tests)
- `dnallm/models/model.py` — `load_model_and_tokenizer` dispatch chain; `Auto*Class.from_config` is the no-download generic path for random_init; 12 special handlers need the allowlist ValueError
- `dnallm/inference/inference.py` — embedding extraction paths (`scoring()` at ~1746) probing reuses; `extract_embeddings` layer/pooling selection connects here
- `tests/conftest.py` fixtures — `tiny_real_model`, `simple_dna_tokenizer`, `mock_hf_boundary` (trainer fast-lane pattern)

### Established Patterns
- Pydantic `model_validator` cross-field rejection with matchable ValueError messages (B1: `use_ia3 × use_qlora`, `lora × ia3`)
- `[Warning] ...` house print style in trainer.py for loud WARNs; `get_logger` elsewhere
- Same-change pytest rule: every dnallm/ behavior change ships with its test in the same commit; ≥96% per-module coverage via mocked fast-lane tests
- typed network skips + `tests/expected_skips.yaml` allowlist same-change; models.lock rows for any newly downloaded model
- Zero new dependencies beyond scikit-allel (Phase 10 landed none; only B5's pyproject addition is sanctioned, package-data + deps untouched by B1–B4)
- CHANGELOG.md D-09-discipline append (REV-ID + reviewer-comment inline, unique anchors)

### Integration Points
- `dnallm/finetune/trainer.py` + `dnallm/configuration/configs.py` + `dnallm/inference/inference.py:111-131` — B1's exclusive hot files (freed by Phase 10)
- `dnallm/models/model.py` — B2 sole owner
- new `dnallm/inference/probing.py`, new `dnallm/finetune/sweep.py` (or tasks/ — planner decides per architecture) — B3/B4 new-file lanes
- `dnallm/inference/vep.py` completion + `dnallm/cli/vep.py` new + pyproject entry — B5 sole owner of cli.py and pyproject
- scikit-allel read path: `allel.read_vcf(fields=[...])` → variant tuples → Phase-10 kernels; INFO parsing for ClinVar CLNSIG filtering

</code_context>

<specifics>
## Specific Ideas

- REV-08 lane carries the milestone's highest research flag (ROADMAP): tokenizer-class alignment semantics (char/k-mer/BPE), ClinVar filtering conventions for literature-comparable AUROCs, split-token alignment — run plan-time research for this phase before the planner.
- The ClinVar ascertainment-bias trap (Pitfall #7): the acceptance "literature-magnitude AUROCs" must use the same filtering conventions as the reference literature or magnitudes are incomparable — the research pass should pin those conventions.
- IA³ docs chapter section stays a Phase-12 C3 completion (forward pointer already in peft_adapters.md).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
</content>
