# Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities - Research

**Researched:** 2026-10-09
**Domain:** PEFT (IA³/presets), from-scratch model init, frozen-embedding probing, zero-shot VEP from VCF (scikit-allel), multi-seed statistics — on the existing dnallm/pytest/coverage-gate codebase
**Confidence:** HIGH for repo-grounded and locally-verified-empirical facts; MEDIUM for literature conventions (web-verified against primary sources); LOW only where flagged in the Assumptions Log

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
Copied verbatim from `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-CONTEXT.md` § Implementation Decisions:

- **D-01:** IA³ presets use **explicit per-family target-module lists** (attention-value + FFN targets, `feedforward_only=False`), derived from real `config.json` module names the same way LoRA presets are — NOT peft's coarse `feedforward_only=True` default. Explicit lists are what make the ~44-model presets-table regression tests meaningful.
- **D-02:** IA³ acceptance ("one transformer-family AND one Mamba model each fine-tune one task") runs on **small fast models already pinned in models.lock** — acceptance must be repeatably runnable, not bound to large models.
- **D-03:** The dry-run validator lands as a **`TrainingConfig.peft_dry_run: bool` field** handled inside the trainer (B1's own files) — B5 solely owns cli.py in this wave, so B1 must not add CLI surface.
- **D-04:** The runtime trainable-parameter-count guard **fails hard**: `ValueError` naming the preset, the expected trainable-ratio band, and the actual count. A silent module-skip is exactly what the guard exists to catch (research Pitfall #4).
- **D-05:** The per-tensor parameter-hash proof emits through **`get_logger` INFO lines** (short-hash table alongside the loud "randomly initialized" banner) — greppable, no sidecar file, no output_dir dependency.
- **D-06:** The two proven architectures are **one generic AutoModel family (BERT-style small model, exercising the `from_config` path) + one special-family allowlist member (mamba path)**; the explicit `ValueError` for unsupported special families is tested with a mock/other family.
- **D-07:** The special-family allowlist is a **module-level `frozenset` in model.py** (`RANDOM_INIT_SUPPORTED_FAMILIES`), documented in README — not config-driven this milestone.
- **D-08:** The dependency lands as **`scikit-allel>=1.3.13,<2`** — bounded range per project convention; the maintenance-mode library's successor (sgkit) justifies the upper guard.
- **D-09:** ClinVar test data is **two-tier**: a committed small synthetic VCF fixture for the fast lane (alignment rules + AUROC-shape with mocked model scores) + the real ClinVar 1k download in the `slow` lane (typed network skip + models.lock rows). Real data stays out of the repo; the fast lane stays network-free.
- **D-10:** CLI entry mirrors the `dnallm-mutagenesis` precedent: **`dnallm/cli/vep.py` + `dnallm-vep` console script** (one facade, one entry point).
- **D-11:** The paradigm↔architecture mismatch guard **raises `ValueError`** (e.g., CLM scoring on a bidirectional MLM) — per research Pitfall #6, mismatched-paradigm AUROCs are *expected* near-random; a silent skip would let a misconfiguration masquerade as a finding. Skip-as-data remains reserved for the same-slot alignment rule (its own documented channel).
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (Phase 12 owns motif/MCP/closeout; the IA³ docs chapter section stays a Phase-12 C3 completion.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PEFT-01 | IA³ adapter support: Ia3Config validators + trainer branch symmetric to LoRA, shared save/reload, config-time rejections, transformer+mamba acceptance, IA³ roundtrip | peft 0.21.1 IA3Config surface verified verbatim from installed source; LoRA branch shape verified at trainer.py:181-197; reload path at inference.py:111-131; merge-in-4bit ValueError verified at peft ia3/model.py:226-229; #2429 corruption class documented |
| PEFT-02 | Per-model PEFT target-module presets (lora_targets.yaml packaged, dry-run validator, runtime trainable-count guard, ~44 models) | pyproject package-data glob for `dnallm.configuration.presets/*.yaml` already landed (Phase-10 scaffolding); PRETRAIN_MODEL_MAPS 35 families verified; SSM-target literature + silent-skip evidence compiled |
| BASE-01 | `random_init=True` via AutoConfig.from_pretrained + AutoModel*.from_config, loud banner + per-tensor hash proof, CPU-canonical seeding, no-download, family allowlist | transformers 5.17.0 `from_config` → `_from_config` → `cls(config)` random-init path verified from installed source; post_init tied-weights registration verified; Geneformer lesson + meta-device traps compiled |
| PROB-01 | probing.py: extract_embeddings (layer/pooling selectable) + fit_probe (fixed hyperparams, train-only scaler), registry metrics, npz cache | scoring()/hidden-states mechanics mapped in inference.py:611-760,1746+; NT paper layer finding cited; sklearn 1.9.1 verified installed; cache-key pitfall documented |
| VEP-01 | evaluate_vcf via scikit-allel + CLNSIG parsing + CLI + ClinVar slow tests; literature-magnitude AUROCs | read_vcf API documented from official readthedocs; ClinVar conventions pinned from NT paper / evo2-clinvar / songlab / GPN-Star primary sources; tokenizer-class alignment semantics EMPIRICALLY verified against cached real tokenizers; magnitude anchors per paradigm compiled |
| SEED-01 | sweep: run_seeds + aggregate_seeds (seeded percentile bootstrap, t-interval n-guard, JSON statistics block), directory protocol | SweepConfig fields verified verbatim from configs.py:455-498; Phase-10 result-JSON writer shape verified at trainer.py:639-660 as the mirror target; determinism semantics (D-16) grounded in data.py seed plumbing |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Tech stack:** pytest + pytest-cov only; no new test frameworks. Coverage via `[tool.coverage.run]` omit list in `pyproject.toml`.
- **Compatibility:** suite green on CI matrix Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0; tests must not pin to a single transformers minor version (and by extension peft — floor is `peft>=0.14.0`).
- **CI:** coverage-gated run includes `slow` tests with network; runtime cost accepted.
- **Scope:** bug fixes limited to what correctness/coverage requires; no refactors beyond that.
- **Conventions that bind this phase:** PEP 604 unions (`X | None`, `list[str]`), Google-style docstrings (`Args:`/`Returns:`/`Raises:`), `ValueError` with matchable regex-able messages, relative imports inside `dnallm/`, absolute in tests, `test_<module>.py` naming, `[Warning] ...`/`[Info] ...` print style in trainer.py vs `get_logger` elsewhere, new comments in English.
- **GSD workflow enforcement:** phase work runs through `/gsd-plan-phase` → `/gsd-execute-phase` (this research is that entry).
- **Owner memory rules:** every `dnallm/` change ships with same-change pytest coverage (≥96% per-module standard); commits carry no Co-Authored-By trailers; never clean local model caches.

## Summary

Phase 11 lands five reviewer-experiment capabilities as five file-disjoint lanes (B1 IA³+presets, B2 random_init, B3 probing, B4 sweep, B5 VEP completion). The codebase is unusually well-prepared for this phase: Phase 10 already landed the field-complete config stubs (`Ia3Config`/`VepConfig`/`SweepConfig` registered in `load_config()`), the VEP alignment rule and scoring kernels (`dnallm/inference/vep.py`), the metric registry all three evaluation lanes consume read-only, the pyproject package-data glob for `dnallm/configuration/presets/*.yaml`, and the result-JSON writer pattern B4 mirrors. The interim `use_ia3` warn block in the trainer (trainer.py:174-179) and its two Phase-10 tests are the designated demolition site for B1's real branch.

The heaviest research flag (REV-08/VEP-01) resolved decisively. **Tokenizer-class alignment semantics are now empirically verified against the real cached tokenizers** (offline probes of NT-v2-50m, plant-dnagpt-6mer, plant-dnabert-BPE): non-overlapping 6-mer tokenizers keep every interior/boundary SNV single-slot (the same-slot rule passes); this repo's plant-dnabert-BPE Unigram vocab tokenizes per-character in practice (no merges found in 3,000 random 40-mers or homopolymer/motif probes), so BPE skip rates will be model-specific data, not a protocol failure. Two NEW traps surfaced from the same probes: **lowercase input maps whole k-mers to `<unk>` producing silent "no change" skips** (evaluate_vcf must uppercase windows — soft-masked reference FASTA is common), and **1-bp indels are `length-changing allele` skips on every tokenizer class** (confirming VEP-INDEL deferral). **ClinVar conventions were pinned from primary sources**: NT paper (likely-pathogenic SNPs vs 1000G MAF>5% within-100kb negatives, 6 kb windows, AUC 0.80 for the 2.5B MLM), evo2-clinvar (≥2 gold stars, P/LP vs B/LB, precomputed Δlog-likelihood, Evo2-40B ≈ 0.98 CLM), songlab/GPN-Star (≥1-star floor, VUS/conflicting/0-star excluded). The named references "Chen et al. 2022 CAD-BERT" and "Silva et al. 2023 HyenaDNA Sec 3.7" could NOT be verified — the acceptance should anchor on the verified convention set instead.

For the standard-pattern lanes: peft 0.21.1's `IA3Config` has NO `feedforward_only` field — the real field is `feedforward_modules` (subset-of-target_modules, check skipped for regex strings), which D-01's intent maps onto cleanly; the installed IA³ model raises `ValueError("Cannot merge ia3 layers when the model is loaded in 4-bit mode")` exactly as the milestone research recorded. transformers 5.17.0 `from_config` is verified random-init-only (`_from_config` → `cls(config)` under init contexts; no checkpoint). Plant DNAMamba loads through the GENERIC AutoModelForCausalLM path (no special handler) — its remote-code config exercises the `trust_remote_code` `from_config` branch, which is the "mamba path" D-06's allowlist member covers.

**Primary recommendation:** Plan B5's `evaluate_vcf` around uppercase-window + same-left-context windows feeding the landed `align_variant`, pin the ClinVar convention as SNV-only + P/LP-vs-B/LB + ≥1-star floor (report the filter alongside results), and let B1 compute the trainable-parameter ratio directly from `requires_grad` tensors rather than parsing `print_trainable_parameters` output.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| IA³ injection / adapter save-reload | finetune/trainer.py + inference.py reload seam | peft library (owned externally) | Trainer owns the init branch; the reload path is the existing `lora_adapter` seam — adapter-type-agnostic `PeftModel.from_pretrained` |
| Target-module presets | configuration/presets/*.yaml (packaged data) | trainer dry-run validator | Presets are data derived from real config.json module names; the trainer validates at runtime |
| From-scratch init | models/model.py (generic loader + allowlist) | — | `load_model_and_tokenizer` owns the dispatch chain and the `random_init` flag |
| Embedding extraction | inference/inference.py `scoring()` hidden-states path | new probing.py consumes read-only | Model-facing mechanics already exist; probing adds layer/pooling selection + cache |
| Probe fitting | new inference/probing.py | sklearn (LogisticRegression/MLPClassifier/StandardScaler) | Frozen-backbone discipline lives in the probe module |
| VCF reading + label filtering | new evaluate_vcf driver in inference/vep.py | scikit-allel (parser), cli/vep.py (entry) | Parsing is delegated; alignment + scoring reuse the landed kernels |
| Variant alignment + scoring kernels | inference/vep.py (landed Phase 10) | mutagenesis.py provenance | Already shipped; B5 wraps, must not re-implement |
| Multi-seed orchestration | new finetune/sweep.py (planner may choose tasks/) | trainer evaluate(split=) per seed | Sweep is pure orchestration + aggregation over existing trainer runs |
| Uncertainty statistics | finetune/sweep.py `aggregate_seeds` | scipy.stats.t, numpy Generator | Pure function; no model dependency |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| peft (installed) | 0.21.1 (floor `>=0.14.0`) | IA³ injection via `get_peft_model(model, IA3Config)` | Existing LoRA/QLoRA path; same save/reload lifecycle [VERIFIED: installed venv source, `peft/tuners/ia3/config.py`] |
| transformers (installed) | 5.17.0 (span `>=4.49.0,<6`) | `AutoConfig.from_pretrained` + `Auto*Class.from_config` random init | Canonical no-weights path [VERIFIED: installed venv source, `auto_factory.py:206-233`, `modeling_utils.py:1398-1464`] |
| scikit-allel | 1.3.13 (D-08: `>=1.3.13,<2`) | `allel.read_vcf` parsing for evaluate_vcf | Owner-approved 2026-10-09; Windows cp310–313 wheels + numpy 1.26.4/2.2.0 verified empirically before approval [CITED: scikit-allel.readthedocs.io/en/stable/io.html] |
| scikit-learn (installed) | 1.9.1 | `LogisticRegression`, `MLPClassifier`, `StandardScaler` for probing | Already a core dep; fixed-hyperparameter probes are the literature convention |
| scipy (installed) | 1.18.1 | `scipy.stats.t.ppf` for the t-interval CI | Already a dep; no statsmodels (rejected in STACK research) |
| numpy (installed) | 2.5.3 local / 1.26.4 & 2.2.0 in CI | seeded percentile bootstrap via `np.random.default_rng` | Already a dep |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| click | (existing) | `dnallm-vep` CLI (B5, mirrors dnallm-mutagenesis) | CLI entry only |
| importlib.resources | stdlib | loading packaged `presets/*.yaml` from the wheel | PEFT-02 preset loading |
| pytest + pytest-cov | 9.1.1 | same-change coverage at ≥96% per module | every lane |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scikit-allel read_vcf | cyvcf2 / pysam | Rejected (REQUIREMENTS Out of Scope): no Windows wheels — would break the Windows CI leg |
| scipy t-interval | BCa bootstrap | Rejected at n=3 (D-14): BCa degenerates at small n; t-interval is the honest fallback |
| sklearn MLPClassifier | torch probe head | sklearn keeps probes CPU-cheap and fixed-hyperparameter comparable; torch head would blur probe-vs-finetune boundary |
| numpy percentile bootstrap | statsmodels | Rejected: wrong footprint (STACK.md rejected list); scipy+numpy suffice |

**Installation (B5 only, sole pyproject owner):**
```bash
uv pip install "scikit-allel>=1.3.13,<2"
```
Version verification run this session: `pip index versions scikit-allel` → latest 1.3.13 (published 2024-09-17). scikit-allel is currently NOT installed in the dev venv — B5's first task adds it. STATE.md records the owner's empirical check: only new required transitive dep is `dask[array]`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| scikit-allel | PyPI | latest 1.3.13 (2024-09-17); first release ~2015 | unknown-downloads (signal unavailable) | github.com/cggh/scikit-allel (canonical) | SUS (unknown-downloads only) | Approved — owner decision D-08 (2026-10-09) supersedes; registry existence + official docs verified |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** scikit-allel — the SUS verdict rests solely on the checker's `unknown-downloads` signal (PyPI download stats unavailable to it); the library is the maintstream population-genomics VCF reader from cggh/scikit-allel, in maintenance mode with sgkit as successor (which is exactly why D-08 bounds `<2`). No `postinstall` scripts (Python ecosystem). No action needed beyond the already-locked bounded range.

*No other new packages: B1–B4 install nothing (zero-new-deps convergence; only B5's pyproject addition is sanctioned).*

## Architecture Patterns

### System Architecture Diagram

```
                        ┌─────────────────────────── B5: dnallm-vep lane ───────────────────────────┐
 ClinVar VCF (download,│  allel.read_vcf(fields=[CHROM,POS,ID,REF,ALT,CLNSIG,CLNREVSTAT,CLNVC,...]) │
 untrusted input) ───► │        │ 1-based POS → pos0; SNV filter; CLNSIG→label; uppercase window   │
 Reference genome  ───► │        ▼                                                                    │
 (window builder)       │  for each variant: align_variant(seq,pos0,ref,alt,tokenizer)  [LANDED]      │
                        │      ├─ evaluatable ──► score_variant(paradigm=clm|mlm)                    │
                        │      │        ├─ clm: Δ = clm_log_likelihood(alt) − clm_log_likelihood(ref)│
                        │      │        └─ mlm: Δ = mlm_slot_log_prob(alt_id) − ...(ref_id) @ slot   │
                        │      │             paradigm↔architecture guard: ValueError (D-11)          │
                        │      └─ skip ──► skip record (reason: length-changing allele |             │
                        │                  multi-slot token difference | no change) + counts          │
                        │        ▼                                                                    │
                        │  per-variant scores + skip accounting ──► AUROC/AUPRC via metric_registry   │
                        └───────────────────────────────────────────────────────────────────────────┘

                        ┌────────────── B1: IA³ lane ──────────────┐   ┌──── B2: random_init lane ────┐
 YAML config ──► load_config() ──► TrainingConfig.use_ia3           │   │ load_model_and_tokenizer(   │
   (ia3:/lora: sections registered)│  ├─ Pydantic: use_ia3×use_qlora │   │   ..., random_init=True)    │
                        │          │  │  rejected (matchable)         │   │  ├─ special handler? no ──► │
                        │          │  └─ lora×ia3 rejected            │   │  │  allowlist check ──►    │
                        │          ▼                                 │   │  │  ValueError off-list     │
                        │  DNATrainer(use_ia3 path)                 │   │  ├─ AutoConfig.from_pretrained│
                        │  ├─ presets YAML (packaged) ──► IA3Config │   │  └─ Auto*.from_config ──►   │
                        │  ├─ dry-run validator (peft_dry_run)      │   │     random weights + banner │
                        │  ├─ get_peft_model ──► trainable-ratio    │   │     + per-tensor hash log    │
                        │  │   guard: ValueError (D-04)             │   └─────────────────────────────┘
                        │  └─ train() ──► save_pretrained (adapter)  │
                        │       └─► DNAInference(lora_adapter=...)   │   ┌──── B4: sweep lane ────────┐
                        │           PeftModel.from_pretrained        │   │ run_seeds(fn,seeds,out_root)│
                        └───────────────────────────────────────────┘   │  {model}/{task}/seed_{s}/    │
                                                                     │   └─► aggregate_seeds ──►     │
                        ┌────────────── B3: probing lane ──────────┐    │  mean/sd/ci95 (t for n≥3,  │
                        │ extract_embeddings(layer,pooling)       │    │  None n<3; seeded bootstrap │
                        │  └─ npz cache probe_cache/ (D-12)        │    │  for n≥10) + statistics JSON│
                        │ fit_probe(logistic|mlp) ──► metrics ──►  │    └─────────────────────────────┘
                        │   metric_registry.resolve() (read-only)  │
                        └──────────────────────────────────────────┘
```

### Recommended Project Structure
```
dnallm/
├── configuration/
│   ├── configs.py            # B1: Ia3Config validators + TrainingConfig.peft_dry_run (hot file)
│   └── presets/
│       └── lora_targets.yaml # B1: NEW packaged preset table (package-data glob already landed)
├── finetune/
│   ├── trainer.py            # B1: IA³ branch replaces interim warn (trainer.py:174-197 region)
│   └── sweep.py              # B4: NEW (planner may prefer tasks/ — decide by import-weight)
├── models/
│   └── model.py              # B2 sole owner: random_init flag + RANDOM_INIT_SUPPORTED_FAMILIES
├── inference/
│   ├── inference.py          # B1 hot file (111-131 adapter reload region)
│   ├── vep.py                # B5: evaluate_vcf + score_variant completion (wraps landed kernels)
│   └── probing.py            # B3: NEW
├── cli/
│   └── vep.py                # B5: NEW dnallm-vep entry (mirrors mutagenesis CLI)
└── tests (mirror): tests/configuration/, tests/finetune/, tests/models/, tests/inference/
```

### Pattern 1: IA³ branch mirrors the LoRA branch (B1)
**What:** `get_peft_model(model, ia3_config)` in the trainer init, same save/reload lifecycle.
**When to use:** `use_ia3=True` in TrainingConfig.
**Example (verified LoRA shape to mirror, `dnallm/finetune/trainer.py:181-197`):**
```python
# Existing LoRA branch (the template):
if use_lora:
    from ..models.model import peft_forward_compatiable
    print("[Info] Applying LoRA to the model...")
    if self.train_config.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**config["lora"].dict())
    model = peft_forward_compatiable(model)
    self.model = get_peft_model(model, lora_config)
    self.model.print_trainable_parameters()
```
The IA³ branch replaces the interim warn at trainer.py:174-179 (`use_ia3` print block) with the same shape minus the kbit prep (rejected combination). Both `use_ia3` AND the new `peft_dry_run` must be popped from `training_args` before `TrainingArguments(**...)` (line 230 already pops `use_ia3` — extend, don't remove).

### Pattern 2: peft IA3Config construction from the Pydantic stub (B1)
**What:** The Pydantic `Ia3Config` field set already mirrors peft's — construction is a field-for-field pass-through.
**Verified peft surface (installed peft 0.21.1, `peft/tuners/ia3/config.py:24-112`), quoted verbatim:**
```python
target_modules: Optional[Union[list[str], str]] = field(default=None, ...)
exclude_modules: Optional[Union[list[str], str]] = field(default=None, ...)
feedforward_modules: Optional[Union[list[str], str]] = field(default=None, ...)
fan_in_fan_out: bool = field(default=False, ...)
modules_to_save: Optional[list[str]] = field(default=None, ...)
init_ia3_weights: bool = field(default=True, ...)
```
and the subset check: `if not self.feedforward_modules.issubset(self.target_modules): raise ValueError("`feedforward_modules` should be a subset of `target_modules`")` — **the check runs only when both are sets** (skipped for regex strings). There is **no `feedforward_only` field** in peft 0.21.1's IA3Config; D-01's "not peft's coarse feedforward_only default" translates operationally to: explicit `target_modules` + explicit `feedforward_modules` lists per family.

### Pattern 3: random_init via from_config (B2)
**Verified transformers 5.17.0 path:** `AutoModel.from_config(config, **kwargs)` (`auto_factory.py:206-233`) resolves remote code via `config.auto_map` + `trust_remote_code` then calls `model_class._from_config(config, **kwargs)`; `_from_config` (`modeling_utils.py:1398-1464`) instantiates `model = cls(config, **kwargs)` under init contexts — random initialization, no checkpoint fetch. `post_init` (`modeling_utils.py:1294+`) registers `all_tied_weights_keys` at init. Note: `AutoConfig.from_pretrained` still downloads `config.json` (needed and fine — "no-download" means no weight files; the no-download proof patches/asserts the weight-fetch path).
```python
# B2 shape:
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.num_labels = safe_num_labels          # set head-shaping fields on the config
config.id2label, config.label2id = id2label, label2id
model = AutoModelClass.from_config(config, trust_remote_code=True)   # random weights
torch.manual_seed(seed)                      # CPU-canonical seeding BEFORE init
```

### Pattern 4: evaluate_vcf driver (B5)
```python
# Source: scikit-allel official docs (CITED) + landed vep.py contract
callset = allel.read_vcf(
    vcf_path,
    fields=["variants/CHROM", "variants/POS", "variants/ID", "variants/REF",
            "variants/ALT", "variants/CLNSIG", "variants/CLNREVSTAT", "variants/CLNVC"],
    alt_number=4,   # default 3 truncates 4+-allelic rows; size deliberately
)
# INFO fields land under the variants/ namespace; returns dict[str, ndarray] (None if empty)
pos0 = int(rec_pos) - 1                        # VCF POS is 1-based
window = fetch_window(ref_genome, chrom, pos0, context_window).upper()  # .upper() REQUIRED (see Pitfall 5)
alignment = align_variant(window, pos0 - window_start, ref, alt, tokenizer)
```

### Anti-Patterns to Avoid
- **Re-implementing VCF parsing or alignment:** `allel.read_vcf` + the landed `align_variant` own these; B5 wraps.
- **Parsing `print_trainable_parameters` stdout for the guard:** no public peft getter exists in 0.21.x — compute `sum(p.numel() for p in model.parameters() if p.requires_grad)` directly; keep the print for logs.
- **Testing foreign exception strings** (peft/transformers internals) — assert dnallm's own matchable `ValueError`s (version-span rule, Pitfall 14).
- **Touching files outside lane ownership** (the B1–B5 file map in CONTEXT is the collision contract).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| VCF parsing (gzip, header, INFO) | custom VCF reader | `allel.read_vcf` | Coordinate conventions, INFO typing, ALT dimensionality are trap-dense; owner already approved the dep |
| Variant↔token alignment | per-tokenizer alignment heuristics | landed `align_variant` + skip-as-data | It IS the R1-3e① protocol answer; re-implementation re-opens the reviewer challenge |
| AUROC/AUPRC computation | bespoke ROC code | `metric_registry.resolve()` | METR-01 contract; canonical spellings only |
| t-interval CI | manual t-tables/normal approx | `scipy.stats.t.ppf(0.975, df=n-1)` | Correct small-n statistics for free |
| Percentile bootstrap RNG | `random.sample` loops | `np.random.default_rng(bootstrap_seed)` + `np.percentile` | Seeded, reproducible, vectorized |
| Probe estimators | torch linear/MLP heads | `sklearn LogisticRegression`/`MLPClassifier` | Fixed-hyperparameter comparability; CPU-cheap; scaler discipline built in |
| Adapter injection | custom IA³ vector modules | `peft.get_peft_model(model, IA3Config(...))` | Merged-inference lifecycle, save/reload, task-type wiring already handled |

**Key insight:** every lane is composition over proven components (peft, transformers from_config, sklearn, scipy, scikit-allel, landed vep kernels, landed registry). The engineering risk is contract drift (wrong field names, silent skips, stale caches) — not algorithmic novelty.

## Common Pitfalls

### Pitfall 1: IA³ × 4-bit explodes late, in peft's words (B1)
**What goes wrong:** Installed peft 0.21.1 allows IA³ injection on a quantized model but raises on merge: `ValueError("Cannot merge ia3 layers when the model is loaded in 4-bit mode")` (and the 8-bit variant) — verified at `peft/tuners/ia3/model.py:226-229`. Older peft in the supported span raised `NotImplementedError` at injection instead (version-dependent foreign error).
**How to avoid:** Reject `use_ia3 × use_qlora` at Pydantic time with a matchable dnallm `ValueError` (PEFT-01 requirement; test with `pytest.raises(ValidationError, match="use_ia3")`).
**Warning signs:** any test matching peft's merge-error string; `use_ia3+use_qlora` test absent.

### Pitfall 2: peft #2429 — IA³ save/reload corruption class (B1)
**What goes wrong:** peft's target-module list minimization rewrote saved `adapter_config.json` with simplified `target_modules` but full-path `feedforward_modules`, failing on reload (issue #2429, fixed PR #2432; fixed in current peft but the *class* is what the test guards).
**How to avoid:** IA³-specific save→reload roundtrip through `DNAInference(lora_adapter=...)` asserting output identity on a fixed input — not just the LoRA roundtrip.
**Warning signs:** roundtrip test only exercising LoRA.

### Pitfall 3: Silent target-module skip on mamba/hybrid (B1)
**What goes wrong:** Wrong `target_modules` match nothing and peft skips silently (documented for Mamba/hybrid; PEFT #3554: defaults covered 7% of layers; PEFT #2556: Mamba kernels silently skip). A preset listing `query,key` against Mamba (`in_proj`/`out_proj`/`x_proj`/`dt_proj`) yields a frozen model.
**How to avoid:** D-04's hard guard: after `get_peft_model`, compute trainable/total ratio from `requires_grad` tensors directly; `ValueError` naming preset + expected band + actual count. Plus the dry-run validator (`peft_dry_run`) erroring on non-matching names. Plus per-family preset regression tests against real module names.
**Warning signs:** `trainable params: 0`-adjacent log lines with no assertion; IA³ results ≈ frozen-model probing results.

### Pitfall 4: `from_config` still needs config download; tied weights + meta device (B2)
**What goes wrong:** (a) asserting "no network at all" fails on `AutoConfig.from_pretrained` (legitimately fetches config.json); (b) transformers 5.x memory-efficient init can leave tied weights on meta device → `Cannot copy out of meta tensor` on `.to(device)` (transformers #41038/#30703, partial fix #43523); (c) a single global hash passes with leftover pretrained tensors.
**How to avoid:** no-download proof = patched weight-fetch path (assert `_get_model_path_and_imports`/snapshot_download never invoked); seed + init on CPU then `.to(device)`; per-tensor hash comparison (D-05 INFO lines) asserting every parameter tensor differs; `data_ptr()` tie assertion where applicable.
**Warning signs:** random-init test downloading weights; HF cache growth; one global-hash test only.

### Pitfall 5: Lowercase windows tokenize to `<unk>` and silently skip variants (B5 — NEW, empirically verified)
**What goes wrong:** Verified with the cached 6-mer tokenizer: `tok("acgttg")` → `['<cls>', '<unk>', '<eos>']`. Soft-masked (lowercase) reference FASTA — common in repeat regions — makes ref/alt lowercase bases map to the SAME `<unk>` token, so `align_variant` returns `skip_reason="no change"` and variants vanish as data. Verified: `align_variant(seq.lower(), 3, "t", "a", tok6)` → `no change`.
**How to avoid:** `evaluate_vcf` uppercases every window before alignment/scoring (`window.upper()`), with a docstring statement. Reference-fetch helper precedent: `dnallm/utils/genomic_coords.py:130` `fetch_sequence(..., uppercase=True)` already defaults to uppercase.
**Warning signs:** high "no change" skip fractions on real ClinVar runs; lowercase k-mers in tokenizer debug output.

### Pitfall 6: Tokenizer-class alignment expectations (B5 — empirically grounded)
**What goes wrong:** Assuming one skip-rate story for all tokenizers, or hand-re-implementing per-class alignment. Verified reality (offline probes of cached tokenizers, this session):
| Tokenizer class | Example (verified) | SNV effect | Notes |
|---|---|---|---|
| Non-overlapping 6-mer + single-base fallback (EsmTokenizer, greedy left-to-right) | NT-v2-50m (`['<cls>','ACGTTG','ACGTAC','GTTGAC','GTACGT','T','G','A','C']` for 28bp); plant-dnagpt-6mer (+`<eos>`) | **Single-slot at every tested position** (pos 0, 3, 11, 27 — interior, boundary, remainder region, N-adjacent) | Same-slot rule passes; N breaks a k-mer into fallback singles but neighbors stay scorable |
| Unigram BPE (DebertaV2Tokenizer) | plant-dnabert-BPE | **Per-character in practice** — no multi-char merges in 3,000 random 40-mers or homopolymer/motif probes | This vocab's skip rate ≈ 0 for SNVs; DNABERT-2 byte-BPE (not locally cached) remains the multi-slot risk → covered by skip-as-data |
| Character-level | DNAOneHotTokenizer (case-insensitive map, N→id 4) | Always single-slot | Case-insensitive at the tokenizer itself |
**How to avoid:** consume `align_variant`'s verdict as data; report skip fractions per reason per model — the fraction IS the finding about each tokenizer. Ref/alt windows must share identical left context (the landed kernel already substitutes within one window — preserve that in the driver).
**Warning signs:** any driver code calling `tokenizer` on ref and alt sequences built from different windows.

### Pitfall 7: ClinVar ascertainment-bias incomparability (B5)
**What goes wrong:** "Literature-magnitude AUROCs" compared across different filtering conventions are meaningless: label sets (P/LP vs B/LB vs P-only), star floors (0/1/2), variant types (SNV vs +indels), negatives (ClinVar-benign vs gnomAD-common) all shift AUROC by more than the model differences being measured.
**How to avoid:** pin one convention (see State of the Art table), report it next to every AUROC, and compare only within-paradigm/within-convention. Verified magnitude anchors: NT-2.5B MLM ClinVar AUC **0.80**; Evo2-40B CLM ≈ **0.98**; DNABERT-2 BPE embedding-distance ≈ **0.538** vs NT-v2 ≈ **0.73** (pathogenic-vs-common); Caduceus-Ps songlab-ClinVar Cos-AUROC 0.9333; Alfisi et al. normalized-Wilcoxon <0.6 for most DNA models under strict labeling.
**Warning signs:** acceptance written as bare "AUROC ≥ X" with no convention statement.

### Pitfall 8: Probe leakage and stale embedding cache (B3)
**What goes wrong:** scaler fit on all data (leakage); npz key missing layer/pooling (second run hits the WRONG embeddings); pooling default differing from the F4 comparison lane.
**How to avoid:** `StandardScaler` fit on train split only; cache key = (model, dataset, layer, pooling) — D-12; same-change test asserting a pooling change MISSES the cache; pooling recorded in every output row.
**Warning signs:** probe metrics ≥ fine-tuned metrics; cache-hit test passing with no cache-miss-on-param-change test.

### Pitfall 9: The seed illusion and vacuous bootstrap (B4)
**What goes wrong:** sweep seed not reaching dataset ops before the trainer (`train_test_split(seed=...)` at data.py:817/823 takes a caller seed — verified it is NOT auto-derived); bootstrap CI on n=3 (10 distinct resample multisets — vacuous precision); split varying with seed (conflates split variance with init variance).
**How to avoid:** D-16 semantics (split seed derived from dataset, not sweep seed); same-change determinism test (same seeds → identical CPU metrics); D-14 guard (t-interval n≥3, `ci95=None` n<3, bootstrap only n≥10 per `SweepConfig.small_n_ci` description); bootstrap seeded via `bootstrap_seed`.
**Warning signs:** identical metrics across "different" seeds; `ci95` present with n=2.

### Pitfall 10: Version-span test pinning (B1/B2)
**What goes wrong:** tests asserting peft/transformers exception types or messages pass on the dev env (peft 0.21.1 / transformers 5.17.0) and fail the matrix (peft 0.14–0.16, transformers 4.49).
**How to avoid:** assert only dnallm's own `ValueError` surfaces; feature-detect; no private peft/transformers symbol imports in tests.
**Warning signs:** any `match=` targeting a foreign library's error text.

### Pitfall 11: Slow-lane discipline (all lanes)
**What goes wrong:** real-model acceptance tests dropped into the fast PR leg pull network; new model ids without models.lock rows break the consistency guard; bulk heavy tests breach nightly budgets.
**How to avoid:** `slow` marker from birth (pytest `--strict-markers`, `markers = ["slow: ..."]` verified in pyproject); models.lock rows same-change (sha-pinned, prefix↔source aligned); typed network skips allowlisted same-change in `tests/expected_skips.yaml` (exact/prefix/reason_like matchers — verified format).
**Warning signs:** fast-leg runtime creep; `audit_skips.py` exit 1.

### Pitfall 12: The 90% gate won't catch under-tested new modules
**What goes wrong:** ~400-line vep/probing/sweep modules landing with thin tests stay green above `fail_under=90` while giving back the 96%+ working standard.
**How to avoid:** per-module scoped coverage runs at ≥96% in the same change (owner memory rule); verifier reproduces module-level numbers, not just the global gate.
**Warning signs:** coverage trending down across the phase; a new module without its `tests/<pkg>/test_<module>.py` twin.

## Code Examples

### evaluate_vcf variant loop consuming the landed alignment (B5)
```python
# Source: landed dnallm/inference/vep.py (Phase 10) + scikit-allel docs (CITED)
from dnallm.inference.vep import align_variant, clm_log_likelihood, mlm_slot_log_prob

scored, skips = [], {"length-changing allele": 0, "multi-slot token difference": 0, "no change": 0}
for pos, ref, alt, label in variants:            # SNVs only for v1.2 (VEP-INDEL deferred)
    window = window_upper(seq_ref, pos, context_window)   # identical left context, .upper()
    a = align_variant(window, local_pos(pos), ref, alt, tokenizer)
    if not a.evaluatable:
        skips[a.skip_reason] += 1
        continue
    if paradigm == "mlm":
        delta = (mlm_slot_log_prob(model, tokenizer, window, a.slot_index, a.alt_token_id)
                 - mlm_slot_log_prob(model, tokenizer, window, a.slot_index, a.ref_token_id))
    else:  # "clm"
        alt_seq = window[:local_pos] + alt + window[local_pos + len(ref):]
        delta = clm_log_likelihood(model, tokenizer, alt_seq) - clm_log_likelihood(model, tokenizer, window)
    scored.append((delta, label))
# AUROC via registry; report skip counts + evaluated/skipped totals alongside (D-09 fast lane asserts shape)
```

### Trainable-parameter ratio guard (B1, D-04)
```python
# No public peft getter in 0.21.x — compute directly:
trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in self.model.parameters())
ratio = trainable / max(total, 1)
lo, hi = preset.expected_ratio_band          # from the preset row
if not (lo <= ratio <= hi):
    raise ValueError(
        f"IA³ preset '{preset_name}' attached {trainable}/{total} trainable parameters "
        f"(ratio {ratio:.2e}); expected band [{lo:.2e}, {hi:.2e}]. A silent module-skip "
        f"is the likely cause — check target_modules against this backbone."
    )
```

### aggregate_seeds n-guard (B4, D-14 + SweepConfig fields)
```python
# SweepConfig (verified verbatim, configs.py:463-498): seeds=[42,43,44] (min_length=1),
# out_root, n_bootstrap=2000 (ge=1), bootstrap_seed=42,
# small_n_ci: pattern ^(t-interval|omit)$, default "t-interval"
import numpy as np
from scipy import stats

def aggregate_seeds(values, *, n_bootstrap, bootstrap_seed, small_n_ci):
    values = np.asarray(values, dtype=float)
    n = values.size
    out = {"n_seeds": n, "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if n > 1 else None}
    if n < 3:
        out["ci95"], out["method"] = None, "none"          # never vacuous
    elif n < 10 and small_n_ci == "t-interval":
        sem = values.std(ddof=1) / np.sqrt(n)
        out["ci95"] = [float(values.mean() - stats.t.ppf(0.975, n - 1) * sem),
                       float(values.mean() + stats.t.ppf(0.975, n - 1) * sem)]
        out["method"] = "t"
    elif n >= 10:
        rng = np.random.default_rng(bootstrap_seed)         # seeded: reproducible CI
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        means = values[idx].mean(axis=1)
        out["ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        out["method"] = "bootstrap-percentile"
    else:  # small_n_ci == "omit"
        out["ci95"], out["method"] = None, "omitted"
    return out
```

### fit_probe fixed constants (B3, D-13)
```python
# Module-level constants with docstrings — SC3 fixes them; literature recipe
# (StandardScaler -> LogisticRegression(max_iter=1000, solver="lbfgs")).
LOGISTIC_MAX_ITER = 1000
LOGISTIC_SOLVER = "lbfgs"
MLP_HIDDEN = (256,)          # shallow, per convention
MLP_EARLY_STOP = True        # sklearn early stopping uses an internal train-split fraction
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Stdlib/VCF-line parser plan for VEP | scikit-allel `read_vcf` | owner decision 2026-10-09 (supersedes stdlib plan) | B5 dep add only; INFO parsing native |
| LoRA-only PEFT in trainer | IA³ symmetric branch + per-family presets | this phase | ~0.01% trainable params; NT paper itself fine-tunes with IA³ [CITED: NT preprint] |
| `from_pretrained`-everywhere loading | `from_config` random-init lane with allowlist | this phase | Honest from-scratch baselines with hash proof |
| Last-layer-only embeddings | layer-selectable probing | NT finding: intermediate layers often probe better | PROB-01 layer parameter is not cosmetic |
| `mean ± sd` over seeds only | + t-interval (n<10) / seeded percentile bootstrap (n≥10) with n-guard | this phase | R1-2a reproducibility answer |
| BCa bootstrap | refused at n=3 (D-14) | this phase | honest small-n statistics |

**Verified ClinVar convention anchors (primary sources, this session):**
| Source | Positives | Negatives | Types | Stars | Assembly | Metric & magnitude |
|--------|-----------|-----------|-------|-------|----------|--------------------|
| NT paper (Dalla-Torre 2023, §A.5.2) | ClinVar SNPs "likely pathogenic" (n=14,626) | 1000 Genomes MAF>5%, within 100 kb, balanced | SNPs only | none stated in preprint | ref-genome windows 6,000 bp | ROC-AUC **0.80** (2.5B MLM multispecies) |
| evo2-clinvar (Arc) | P/LP | B/LB | SNV + indel | **≥2 gold stars** | RefSeq accessions, Feb 28 2024 release | precomputed Δlog-likelihood; Evo2-40B ≈ **0.98** [CITED: HF dataset card + FEATURES.md] |
| songlab/clinvar-missense | Pathogenic | Benign | missense | 5 review-status values present (filter applied by consumer) | — | paired with CADD/phyloP/ESM scores |
| GPN-Star (2026) | P/LP | B/LB (deconfounded variant per Lu et al.) | SNV + indel benchmarks separate | **≥1 star (non-zero)**, VUS/conflicting/0-star excluded | — | AUPRC |
| GPN-MSA (Benegas) | ClinVar pathogenic-like | **gnomAD common (MAF>5%) as controls** (CADD recommendation — ascertainment-bias reduction) | SNVs | — | — | LLR at variant position |
| Alfisi et al. (Genome Biology 2026) | pathogenic SNVs (65,865) | benign SNVs (156,060) + VUS (108,386, subsampled) | SNVs (GRCh38 VCF) | not stated | GRCh38 | normalized Wilcoxon (not AUROC); most DNA LMs <0.6 |

**Recommended dnallm convention (planner/owner to confirm floor):** ClinVar GRCh38 VCF → `CLNVC=single_nucleotide_variant` only (indels skip-as-data under the alignment rule anyway) → labels P/LP=1 vs B/LB=0, exclude Conflicting/VUS → `CLNREVSTAT` ≥1-star floor (report both the floor and per-star counts) → no AF filter on positives (gnomAD-negative variant is a documented alternative, defer). Report the convention block next to every AUROC.

**Deprecated/undated:** "Chen et al. 2022 CAD-BERT" and "Silva et al. 2023 HyenaDNA §3.7" — not verifiable in this session's searches (no such paper/section surfaced; HyenaDNA is Nguyen et al. 2023 and third-party benchmarks are the ClinVar comparables). Do NOT anchor acceptance magnitude claims on them.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "Chen et al. 2022 CAD-BERT" and "Silva et al. 2023 HyenaDNA §3.7" references from the objective are garbled/nonexistent | State of the Art | Low — acceptance anchored on verified sources instead; if the owner has a specific paper in mind, its conventions should be pinned before acceptance runs |
| A2 | DNABERT-2 byte-BPE produces multi-slot skips for some SNVs (predicted from Mut-BPE/DART-Eval; NOT tested locally — no cached DNABERT-2 tokenizer) | Pitfall 6 | Low — skip-as-data handles it either way; slow-lane ClinVar run will measure the actual fraction |
| A3 | IA³ Mamba targets ≈ `in_proj`/`out_proj`/`x_proj`/`dt_proj` (SSM-PEFT ICML-25: LoRA on linear projections matches full FT) | Pitfall 3 | Medium — presets must still be derived from real config.json module names (D-01); literature only informs where to look |
| A4 | `Auto*.from_config(config, ...)` accepts head-shaping via config attributes (num_labels/id2label set on config) rather than ctor kwargs | Pattern 3 | Low-Medium — B2 verifies in its first test; `_from_config` passes `**kwargs` to the model ctor but head config is conventionally on the config object |
| A5 | Expected trainable-ratio bands per family (~1e-4 IA³ / ~1e-3+ LoRA orders of magnitude) | Code Examples | Low — exact bands pinned at preset-derivation time from real module counts; the guard's value is the mechanism, not the default band |
| A6 | The mamba allowlist member (D-06) means a Plant DNAMamba-family model exercising the trust_remote_code `from_config` branch (mamba is NOT a `_handle_*` special family — verified absent from model.py dispatch) | Pattern 3 / B2 | Low — if the owner intended a literal `_handle_*` family instead, the allowlist member choice changes (e.g., a DNABERT-2-family model); planner should confirm which pinned small mamba model to use (models.lock has `plant-dnamamba-BPE-open_chromatin`, ms route) |
| A7 | scikit-allel lands as a REQUIRED dependency (STATE.md: "only new required transitive dep is dask[array]"), not behind an extra | Standard Stack | Low — if the owner prefers a `vep` extra, B5 adds an import guard instead; dependency placement is a one-line planner decision |

## Open Questions

1. **Where does the `lora × ia3` rejection live?**
   - What we know: `use_lora` is a `DNATrainer.__init__` kwarg, NOT a TrainingConfig field (verified: no `use_lora` in configs.py). `use_ia3` IS a TrainingConfig field. The `use_ia3 × use_qlora` rejection is Pydantic-time (both fields present).
   - What's unclear: PEFT-01 says "`lora × ia3` combination likewise rejected" — but Pydantic can't see the ctor kwarg.
   - Recommendation: trainer-init-time `ValueError` when `use_lora=True and self.train_config.use_ia3` (matchable message), tested via the mocked fast lane. Alternative (adding `use_lora` to TrainingConfig) changes the config surface B1 doesn't otherwise own.
2. **ClinVar star floor for the acceptance run** — 1-star (GPN-Star convention, larger n) vs 2-star (evo2-clinvar convention, higher-confidence, smaller n)?
   - Recommendation: ≥1-star floor for the 1k-sample slow-lane run (sampling headroom), report per-star counts; owner confirms.
3. **sweep module location** — `dnallm/finetune/sweep.py` (CONTEXT default) vs `dnallm/tasks/`?
   - Recommendation: `finetune/sweep.py` — it orchestrates trainer runs; import-light pure functions keep it testable without torch at the aggregation layer.
4. **Which pinned small models for B1/B2/B4 acceptances** (Claude's discretion per CONTEXT)?
   - Recommendation from models.lock: transformer = `zhangtaolab/plant-dnabert-BPE` (ms, already most-referenced) or `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (hf, 50m); mamba = `zhangtaolab/plant-dnamamba-BPE-open_chromatin` (ms); task dataset = `zhangtaolab/plant-multi-species-core-promoters` (existing trainer-test precedent).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| peft | B1 | ✓ | 0.21.1 (venv) | — |
| transformers | B1/B2/B5 | ✓ | 5.17.0 (venv; span <6) | — |
| torch | all model lanes | ✓ | 2.11.0+cu130 (venv) | CPU paths canonical for proofs |
| scikit-learn | B3 | ✓ | 1.9.1 | — |
| scipy | B4 | ✓ | 1.18.1 | — |
| numpy | all | ✓ | 2.5.3 local (CI: 1.26.4 & 2.2.0) | — |
| **scikit-allel** | B5 | **✗ (not installed)** | latest 1.3.13 on PyPI | none — B5 task 1 installs per D-08 |
| Cached tokenizers/models (offline dev probes) | B1/B2/B5 dev + slow-lane dev | ✓ | plant-dnagpt-6mer, plant-dnabert-BPE (+variants), NT-v2-50m, plant-dnagpt-BPE-promoter, PlantHelixSeek, evo2_1b_base (see models.lock) | network when cache misses |

**Missing dependencies with no fallback:** none blocking — scikit-allel install is B5's first task and is owner-sanctioned.
**Missing dependencies with fallback:** none.

## Validation Architecture

> `workflow.nyquist_validation` is `false` in `.planning/config.json`, but the phase objective explicitly requests this section — it feeds the phase's VALIDATION strategy. Test-infrastructure facts: pytest 9.1.1, `--strict-markers`, `--timeout=300`, `markers = ["slow: ..."]`, two collected roots (`tests/` + `dnallm/mcp/tests/`, no `__init__.py`), coverage gate `fail_under=90` (working standard ≥96% per module), `tests/expected_skips.yaml` audit gate, unique-basename convention.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 + pytest-cov, pytest-asyncio (auto), pytest-timeout (300s) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest --no-sync -m "not slow" -x -q` (note: `--no-sync` per v1.1 ledger — uv 0.12.20 resolver issue) |
| Full suite command | `uv run pytest --no-sync -q` (slow included; coverage-gated CI leg adds `--cov`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PEFT-01 | `use_ia3×use_qlora` + `lora×ia3` rejections | unit (Pydantic/trainer mocks) | `uv run pytest --no-sync tests/configuration/test_configs.py tests/finetune/test_trainer.py -q -k ia3` | Partial (stub tests exist: `test_use_ia3_defaults_false`, `test_use_ia3_has_no_cross_field_rejection_yet` — B1 REPLACES the latter) |
| PEFT-01 | IA³ wiring mirrors LoRA branch | unit (mock `get_peft_model`/`IA3Config`, `mock_hf_boundary`) | `... tests/finetune/test_trainer.py -q -k "TestLoraWiring or ia3"` | Partial (TestLoraWiring exists to extend) |
| PEFT-01 | IA³ save/reload roundtrip (transformer + mamba) | integration, slow, network | `... tests/finetune/test_trainer_real_model.py -q -k ia3 -m slow` | ❌ Wave 0 (B1) |
| PEFT-02 | Presets table regression + dry-run validator + ratio guard | unit (no network; fake module lists) | `... tests/configuration/test_peft_presets.py -q` | ❌ Wave 0 (B1) |
| BASE-01 | random_init: per-tensor hashes, same-seed repro, no-download, allowlist ValueError | unit (tiny config CPU, patched downloader) | `... tests/models/test_model.py -q -k random_init` | ❌ Wave 0 (B2) |
| BASE-01 | Two real architectures (generic + mamba) | integration, slow | `... -m slow -k random_init` | ❌ Wave 0 (B2, models.lock rows) |
| PROB-01 | extract/fit/cache/leakage | unit (tiny_model_factory synthetic embeddings) | `... tests/inference/test_pro.py -q` | ❌ Wave 0 (B3) |
| PROB-01 | any model × binary task end-to-end | integration, slow | `... -m slow -k probing` | ❌ Wave 0 (B3) |
| VEP-01 | Synthetic VCF fixture: alignment rules, skip counts, coordinate/case/multi-allelic cases, AUROC shape with mocked scores | unit (network-free, D-09) | `... tests/inference/test_vep.py -q -k evaluate_vcf` | Partial (Phase-10 align/kernel tests exist) |
| VEP-01 | ClinVar 1k × ≥5 models literature-magnitude AUROCs | integration, slow, network, typed skip + models.lock rows | `... -m slow -k clinvar` | ❌ Wave 0 (B5) |
| SEED-01 | aggregate_seeds on constructed arrays (known moments, n-guard paths) | unit (pure function, no models) | `... tests/finetune/test_sweep.py -q` | ❌ Wave 0 (B4) |
| SEED-01 | run_seeds determinism + ≥3-seed end-to-end | unit (mock/tiny fn, CPU) + slow acceptance | `... -m slow -k sweep` | ❌ Wave 0 (B4) |

### Sampling Rate
- **Per task commit:** quick run command (fast lane, `not slow`) — every dnallm/ change ships with its tests (owner rule).
- **Per wave merge:** full suite + per-module scoped coverage ≥96% (e.g., `uv run pytest --no-sync tests/inference/test_vep.py --cov=dnallm.inference.vep --cov-report=term-missing -q`).
- **Phase gate:** full suite green (incl. slow leg) before `/gsd-verify-work`; verifier reproduces per-module coverage numbers.

### Lane test strategy summary
- **B1:** mocked fast lane for wiring/validators/guard (patch `get_peft_model`, fake PeftModel with controllable `requires_grad` tensors); presets regression vs `model_info.yaml`/families offline; IA³ roundtrip + transformer/mamba acceptance in slow lane.
- **B2:** tiny `AutoConfig`+`from_config` CPU units with patched weight-fetch; per-tensor hash vs cached pretrained model (slow, one network use); mamba allowlist path slow.
- **B3:** synthetic-embedding units via `tiny_model_factory`; cache hit/miss/pooling-miss; disjoint-split asserts; end-to-end slow on a models.lock model.
- **B4:** pure-function aggregation units (constructed arrays, exact moments); determinism via stubbed `fn`; slow ≥3-seed trial.
- **B5:** committed synthetic VCF fixture (SNV + indel + boundary + multi-allelic + lowercase + N cases — Claude's discretion per CONTEXT, must cover these); mocked-score AUROC-shape tests; real ClinVar slow lane with typed network skip + expected_skips.yaml allowlist entry + models.lock rows.

### Wave 0 Gaps
- [ ] `tests/configuration/test_peft_presets.py` — PEFT-02 presets table + validator (B1)
- [ ] `tests/finetune/test_sweep.py` — SEED-01 (B4)
- [ ] `tests/inference/test_probing.py` — PROB-01 (B3)
- [ ] `tests/inference/test_vep.py` extensions — evaluate_vcf/CLI/fixture (B5; file exists with Phase-10 kernel tests)
- [ ] `tests/models/test_model.py` random_init class — BASE-01 (B2; file exists)
- [ ] `tests/finetune/test_trainer.py` IA³ branch tests replacing the interim-warn pair (B1)
- [ ] Fixture: `tests/inference/data/synthetic_variants.vcf` (committed, network-free) (B5)
- Framework install: none needed.

## Security Domain

> `security_enforcement: true`, ASVS level 1, block on high.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | no new auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | CLI/library local execution |
| V5 Input Validation | **yes** | ClinVar/VCF downloads are untrusted data: strict `allel.read_vcf` field validation; REF-must-match-reference assertion (landed `align_variant` ValueError); label whitelist for CLNSIG parsing; never construct filesystem paths from VCF record fields; uppercase/case-fold windows; bounded `alt_number` |
| V6 Cryptography | no | none added |
| V12 File/Resource | **yes (L1)** | probe_cache + sweep out_root under caller-supplied output_dir (D-12: never CWD); validate out_root before recursive mkdir; npz cache files keyed by sanitized hashes, not raw model ids |
| V14 Config | **yes (L1)** | new config flags (`peft_dry_run`, ia3 section) validated at Pydantic boundary with matchable errors; reject incompatible combinations early |

### Known Threat Patterns for Python ML CLI/library stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/corrupted ClinVar or reference FASTA downloads crashing parsers (zip bombs, huge ALT arrays) | Tampering/DoS | strict field requests, row-count sanity bounds, treat as untrusted input (download into isolated dirs per environment policy) |
| Path traversal via VCF record fields used as filenames | Tampering | never derive paths from record fields; deterministic output names |
| `trust_remote_code=True` widening via random_init invocation paths | Elevation | D-07 allowlist limits families; unlisted families raise `ValueError` (pre-existing surface, explicitly bounded) |
| Uncontrolled egress via model downloads in new paths | Information disclosure | random_init must not fetch weights (no-download proof is also a security property); slow-lane network confined to typed-skipped tests |

## Sources

### Primary (HIGH confidence)
- Repo source read this session: `dnallm/inference/vep.py` (full), `dnallm/configuration/configs.py:263-499` (TrainingConfig/Ia3Config/VepConfig/SweepConfig verbatim), `dnallm/finetune/trainer.py:100-240,440-505,560-660`, `dnallm/inference/inference.py:80-158,611-725,1700-1830`, `dnallm/models/model.py:753-917`, `dnallm/models/tokenizer.py:1-80`, `dnallm/models/modeling_auto.py:4-40`, `dnallm/tasks/metric_registry.py:1-80`, `dnallm/utils/genomic_coords.py` (signatures), `tests/conftest.py:160-235`, `tests/inference/test_vep.py` (test inventory), `tests/finetune/test_trainer.py` (TestLoraWiring + interim-warn tests), `tests/configuration/test_configs.py` (stub tests), `tests/expected_skips.yaml`, `models.lock`, `pyproject.toml` (scripts/package-data/markers)
- Installed-library source: peft 0.21.1 `tuners/ia3/config.py` (full class) + `tuners/ia3/model.py:226-229` (merge ValueError); transformers 5.17.0 `models/auto/auto_factory.py:206-233` + `modeling_utils.py:1294-1318,1398-1464` (from_config/post_init)
- **Empirical offline probes (this session, cached real tokenizers):** NT-v2-50m + plant-dnagpt-6mer greedy non-overlapping 6-mer + single fallback — SNV single-slot at all tested positions; plant-dnabert-BPE per-char on 3,000 random 40-mers + motif probes; lowercase→`<unk>` "no change" skip; 1bp indel→"length-changing allele" on all classes
- Milestone research (repo, HIGH grounding): `.planning/research/PITFALLS.md` (P3-P10, P13-P15), `.planning/research/FEATURES.md` (REV-04..09 sections), `.planning/REQUIREMENTS.md`, `.planning/STATE.md`

### Secondary (MEDIUM confidence)
- [Nucleotide Transformer preprint, bioRxiv 2023.01.11.523679v2](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full) — ClinVar §A.5.2 conventions (likely-pathogenic SNPs; 1000G MAF>5% within-100kb negatives; 6 kb windows; AUC 0.80), 6-mer tokenizer greedy+fallback description, IA³-as-its-own-finetuning, intermediate-layer probing finding (fetched via webReader)
- [evo2-clinvar HF dataset card](https://huggingface.co/datasets/goodarzilab/evo2-clinvar) — ≥2 gold stars, P/LP vs B/LB, Feb 28 2024 release (WebFetch)
- [scikit-allel readthedocs, io.html](https://scikit-allel.readthedocs.io/en/stable/io.html) — read_vcf fields/namespace/alt_number API (WebFetch)
- [songlab/clinvar-missense dataset](https://huggingface.co/datasets/songlab/clinvar-missense) + GPN-Star/GPN-MSA search results — star floors, gnomAD-as-controls convention
- [Alfisi et al., Genome Biology 2026 / bioRxiv 2025.06.15.659748](https://www.biorxiv.org/content/10.1101/2025.06.15.659748v1.full.pdf) — ClinVar GRCh38 SNV census + normalized-Wilcoxon framing
- [Mut-BPE bioRxiv 2025.12.01.691503](https://www.biorxiv.org/content/10.1101/2025.12.01.691503v1) — training-free split-token strategy abstract (full text rate-limited this session)
- peft issues #2429/#2432, #3554, #2556 — via milestone PITFALLS.md research (already grounded there)

### Tertiary (LOW confidence)
- None used for load-bearing claims.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every version verified against the project venv / PyPI this session; scikit-allel API from official docs
- Tokenizer alignment semantics: HIGH — empirical probes against the actual cached tokenizers the slow lane will use
- ClinVar conventions: MEDIUM — primary sources fetched, but two objective-named references unverifiable (A1); the recommended convention is a defensible synthesis, owner confirms the star floor
- Architecture: HIGH — all seams read from source; Phase-10 scaffolding verified present
- Pitfalls: HIGH for repo/empirical items; MEDIUM for literature-derived magnitude anchors

**Research date:** 2026-10-09
**Valid until:** 2026-11-08 (stable domain; re-check scikit-allel version and peft minor if the venv changes)
