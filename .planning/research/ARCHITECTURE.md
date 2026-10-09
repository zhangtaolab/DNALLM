# Architecture Research

**Domain:** Integration of 11 paper-revision features (REV-01…REV-11) onto the existing DNALLM layered architecture — milestone v1.2 "Paper Revision Suite Support"
**Researched:** 2026-10-09 (milestone v1.2)
**Confidence:** HIGH for all integration points — every claim below was verified against the working tree on branch `revision` (post-`c99fa9d`): trainer eval semantics, loader dispatch chain, config classes, MCP tool registration, coverage/lint exclusion lists, CI marker wiring. No web sources needed; the codebase is the source of truth. MEDIUM only where a design choice is still open (flagged inline as **decision needed**).

---

## Standard Architecture

### System Overview

The 11 REVs decompose into **four integration shapes**: (1) behavioral guards inside existing facades (REV-01), (2) a new contract module beside existing dispatch code (REV-02), (3) new sibling modules inside existing subpackages that reuse existing kernels via composition (REV-05/07/08/09/10), and (4) config + branch extensions to existing PEFT/loading seams (REV-04/06). REV-11 wraps whatever exists at the end. Nothing restructures a layer; every REV lands inside the current `cli/ → configuration → models → datahandling → finetune/inference → mcp → utils` stack.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ENTRY LAYER                                                                  │
│  cli/cli.py:  NEW `vep` command (REV-08)         [lazy imports, Click]       │
│  mcp/server.py: NEW ism_scan/hotspots/zero_shot_score tools (REV-11)         │
│                 FIX --host/--port flag-override (main, server.py:1928)       │
├──────────────────────────────────────────────────────────────────────────────┤
│ CONFIGURATION LAYER  configuration/configs.py                                │
│  TrainingConfig.allow_test_as_eval (REV-01)   Ia3Config + "ia3" key (REV-04) │
│  load_config(): fills new optional sections — DNALLMConfig stays total=False │
│  NEW packaged data: configuration/presets/lora_targets.yaml (REV-05)         │
├──────────────────────────────────────────────────────────────────────────────┤
│ MODELS LAYER  models/model.py                                                │
│  load_model_and_tokenizer(random_init=True) → generic Auto* path only        │
│  (REV-06). Special-family handlers (evo/enformer/...) untouched & documented ││
│  as out of random_init scope. peft_forward_compatiable (:1007) shared (R-04) │
├──────────────────────────────────────────────────────────────────────────────┤
│ FINETUNE LAYER  finetune/                                                    │
│  trainer.py: eval-semantics guard at :234-241 + evaluate(split=) (REV-01);   │
│              IA³ branch beside LoRA at :153-170 (REV-04)                     │
│  NEW sweep.py (REV-09): run_seeds/aggregate_seeds pure functions             │
│  NEW presets.py (REV-05): family → target_modules resolver                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ INFERENCE LAYER  inference/                                                  │
│  inference.py: adapter-reuse path :111-131 generalized beyond LoRA (REV-04)  │
│  NEW probing.py (REV-07) — composes DNAInference.get_embeddings (:1978)      │
│  NEW vep.py (REV-08) — reuses mutagenesis kernels mlm/clm_evaluate (:258/312)│
│  (interpret.py / mutagenesis.py / benchmark.py unchanged)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ TASKS LAYER  tasks/                                                          │
│  NEW metric_registry.py (REV-02) — BESIDE metrics.py, NOT inside the         │
│  vendored metrics/ dir (coverage-omit + ruff-exclude hazard, see Anti-Pats)  │
│  metrics.py rewired to emit keys through the registry                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ NEW SUBPACKAGE  interpret/  (REV-10)                                         │
│  motifs.py: PWM scan over Mutagenesis hotspots/ISM outputs (tfmodisco seam   │
│  mutagenesis.py:583). Coexists with inference/interpret.py — do NOT move it  │
├──────────────────────────────────────────────────────────────────────────────┤
│ CROSS-CUTTING: docs/terminology (REV-03) · CHANGELOG v0.7.2 · tests layered  │
│ fast (mock, PR-leg gate) / slow (real models, nightly census)                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (per REV: what is new vs modified)

| REV | Extends (existing) | New components | Modified components | Verified anchor |
|-----|--------------------|----------------|--------------------|-----------------|
| REV-01 eval-semantics guard | `DNATrainer.set_up_trainer` / `evaluate` / `infer` | `evaluate(split=...)` method; guard tests | `dnallm/finetune/trainer.py:234-241` (test-silently-becomes-eval), `:489` evaluate, `:502` infer (predict path already correct); `dnallm/configuration/configs.py` TrainingConfig (`allow_test_as_eval: bool = False`) | trainer.py read; split names are literally `"train"/"test"/"val"` from `DNADataset.split_data` (data.py:798-835) — eval_key at :234 picks the first non-train/test key (i.e. `"val"`) |
| REV-02 metric registry | `dnallm/tasks/metrics.py` dispatchers | `dnallm/tasks/metric_registry.py` — `{canonical: (fn, aliases)}` + `resolve(name)`; contract tests; alias table for `eval_AUROC`/`eval_spearmanr` (current pipeline) + `eval_auroc`/`eval_spearman_r` (historical, recognize-only) | `metrics.py` emission sites rewired: AUROC/AUPRC `:134-135`, `:299`, `:306`; spearmanr/pearsonr `:181-219` + single-output `**spm` merges | metrics.py read; pyproject omit `*/dnallm/tasks/metrics/*` (line 542) + ruff exclude `dnallm/tasks/metrics/` confirmed |
| REV-03 docs/terminology | docs site, README, API docs, CHANGELOG | comparability-warning docs; terminology sweep ("DNA large language models"); `validate_sequences` docstring warning | `dnallm/datahandling/data.py` docstring only; CHANGELOG v0.7.1→v0.7.2; docs-validation gate stays green | docs mirror sync (`check_docs_sync.py`) applies |
| REV-04 IA³ adapter | LoRA branch `trainer.py:153-170`; adapter-reuse `inference.py:111-131`; `peft_forward_compatiable` `model.py:1007` | `Ia3Config` Pydantic class; `DNALLMConfig["ia3"]` key; `DNATrainer(use_ia3=True)` branch; IA³ rows in presets YAML | `configs.py` (+1 class, +1 dict key, load_config branch); `trainer.py` (branch beside LoRA); `inference.py` `lora_adapter` param generalized to accept any PEFT adapter dir (PeftModel.from_pretrained is adapter-type-agnostic — verified :111-131) | `LoraConfig(**config["lora"].dict())` → `get_peft_model` chain read at :165-167; QLoRA kbit-prep at :159-163 precedes it |
| REV-05 PEFT presets | `LoraConfig.target_modules` (configs.py:350); model-family registry `modeling_auto.py` | `dnallm/configuration/presets/lora_targets.yaml` (packaged; families: BERT/GPT/Mamba/Gemma/Llama/hybrid × {lora: modules+r, ia3: modules}); `dnallm/finetune/presets.py` resolver; `[tool.setuptools.package-data]` entry | `trainer.py` LoRA/IA³ init when `target_modules=None`: resolve by family + log | Precedent verified: `dnallm/models/model_info.yaml` is already packaged data; `dnallm/configuration/evo/*.yml` is the in-package YAML precedent |
| REV-06 random_init | `load_model_and_tokenizer` (model.py:753) generic path | `random_init: bool = False` kwarg; config-only load branch (`AutoConfig.from_config` + weight re-init via `model.init_weights()`); param-hash logging ("randomly initialized" + hash) | `_load_model_by_task_type` (model.py:549) — from_pretrained sites at :613-659 get a config-load twin; scope: generic Auto* families ONLY | Early-return special handlers (evo2 :807, evo1 :812, megadna :820, enformer :833, space :844, borzoi :855) bypass the generic path — document as out of scope |
| REV-07 probing | `DNAInference.get_embeddings` (inference.py:1978) + `scoring(score_type="embedding")` (:1746 general branch) | `dnallm/inference/probing.py`: `extract_embeddings(...)` (pooling/layer options), `fit_probe(kind="logistic"\|"mlp")` (sklearn fixed hyperparams, val-split early stop), metrics via REV-02 registry, npz cache | None in existing files (pure composition) | get_embeddings verified (:1978-2077, layer/pooling machinery already there); sklearn is an existing dependency |
| REV-08 zero-shot VEP | `clm_evaluate` (mutagenesis.py:312) / `mlm_evaluate` (:258) kernels; `scoring()` (:1746) | `dnallm/inference/vep.py`: `align_variant(seq,pos,ref,alt,tokenizer)` same-slot rule (skip + count on mismatch — BPE/k-mer), `score_variant(paradigm="clm"\|"mlm")`, `evaluate_vcf(...)` (minimal stdlib VCF reader — no new dep), `vep` CLI command; formulas in docstrings | `dnallm/cli/cli.py` (+1 command, lazy import); `dnallm/inference/__init__.py` optional | kernels verified (:258-347: mask-per-token PLL and shifted-logprob CLM); tokenizer slot alignment is new pure logic — fully unit-testable with a fake tokenizer |
| REV-09 multi-seed sweep | `TrainingConfig.seed` (configs.py:291); `DNADataset.sampling(ratio, seed)` (data.py:1058); `DNATrainer.extra_args` override | `dnallm/finetune/sweep.py`: `run_seeds(fn, seeds, out_root)` — dir protocol `{model}/{task}/seed_{s}/`; `aggregate_seeds(...) -> mean/sd/ci95_bootstrap` (pure fn); results-JSON `statistics` block spec | None required — sweep drives DNATrainer from outside (seed injected via `extra_args={"seed": s}` which `set_up_trainer` already merges at :195-196) | extra_args merge verified; config-dict stays untouched per seed via shallow copies |
| REV-10 motif matching | `Mutagenesis.prepare_tfmodisco_inputs` (mutagenesis.py:583) + hotspots (:575 area); ISM outputs | NEW SUBPACKAGE `dnallm/interpret/` (`__init__.py` + `motifs.py`): PWM log-odds scan + threshold + FDR (numpy; no new deps), motif-ID/coords/E-value table | None in existing files | `dnallm/interpret/` does not exist today; auto-included by `[tool.setuptools] include=["dnallm*"]`; naming coexists with `dnallm/inference/interpret.py` (do NOT move/rename — no-refactor constraint) |
| REV-11 MCP tools | `_register_tools` (server.py:238) + `_with_timeout_wrapper` (:295); `ModelManager` engines; per-tool test precedents `tests/mcp/test_mutagenesis_tool.py`, `test_interpret_tool.py` | 3 tools: `ism_scan`, `hotspots` (wrap Mutagenesis), `zero_shot_score` (wrap vep.py — **soft dependency on REV-08**) | `dnallm/mcp/server.py` (`_register_tools` + 3 handler methods); `main()` (:1928) `--host/--port` override fix (v1.1 audit carryover) | tool registration pattern verified; ModelManager supplies model+tokenizer for the vep functions |

## Recommended Project Structure

```
dnallm/
├── cli/
│   └── cli.py                    # + vep command (REV-08; lazy import of vep module)
├── configuration/
│   ├── configs.py                # + Ia3Config, TrainingConfig.allow_test_as_eval (MODIFIED)
│   ├── evo/*.yml                 # existing precedent for packaged YAML
│   └── presets/
│       └── lora_targets.yaml     # NEW packaged data (REV-05) + package-data entry
├── finetune/
│   ├── trainer.py                # MODIFIED: REV-01 guard + evaluate(split=); REV-04 IA³ branch
│   ├── presets.py                # NEW (REV-05): family → target_modules resolver
│   └── sweep.py                  # NEW (REV-09): run_seeds + aggregate_seeds
├── inference/
│   ├── inference.py              # MODIFIED (REV-04): adapter param generalized at :111-131
│   ├── probing.py                # NEW (REV-07)
│   ├── vep.py                    # NEW (REV-08)
│   ├── mutagenesis.py            # UNCHANGED (kernels reused read-only)
│   └── interpret.py              # UNCHANGED
├── interpret/                    # NEW SUBPACKAGE (REV-10)
│   ├── __init__.py
│   └── motifs.py                 # JASPAR/CIS-BP PWM matching
├── models/
│   └── model.py                  # MODIFIED (REV-06): random_init kwarg + config-load twin
├── tasks/
│   ├── metric_registry.py        # NEW (REV-02) — sibling of metrics.py
│   ├── metrics.py                # MODIFIED (REV-02): emit via registry
│   └── metrics/                  # VENDORED — DO NOT place registry.py here
├── mcp/
│   └── server.py                 # MODIFIED (REV-11): 3 tools + host/port fix
└── datahandling/
    └── data.py                   # docstring-only edit (REV-03)

configs/presets/lora_targets.yaml # NOT here — repo-root configs/ is not in the wheel
tests/
├── finetune/test_trainer.py          # + REV-01 split-semantics matrix (fast, mocked)
├── finetune/test_sweep.py            # NEW (REV-09) — aggregation on constructed arrays (fast)
├── tasks/test_metric_registry.py     # NEW (REV-02) — contract over all task-type keys (fast)
├── inference/test_probing.py         # NEW (REV-07) — mocked embeddings (fast)
├── inference/test_vep.py             # NEW (REV-08) — fake-tokenizer alignment (fast)
├── interpret/test_motifs.py          # NEW (REV-10) — synthetic PWMs (fast)
├── models/...                        # + REV-06 fast tests (tiny config-only init)
├── mcp/test_ism_tool.py etc.         # NEW (REV-11) — mocked ModelManager (fast)
└── finetune/test_ia3_real_model.py   # + slow/network lane (see Test Layering)
```

### Structure Rationale

- **`metric_registry.py` beside `metrics.py`, not inside `metrics/` (deviation from intake path — decision needed but strongly recommended):** `pyproject.toml` omits `*/dnallm/tasks/metrics/*` from coverage (line 542) and ruff excludes `dnallm/tasks/metrics/` (line 302). Placing the contract module there would exempt the single most safety-critical new file from the >90% gate, lint, and mypy — inverting the milestone's core value. A sibling module keeps the denominator append-only. If the owner insists on the intake path, the omit/exclude globs must be narrowed — denominator churn that violates the byte-stable-denominator decision. dnallmmark F3 simply imports the new path (F3 is unwritten; open question #3 covers cross-repo locking anyway).
- **`configuration/presets/` for the YAML, not repo-root `configs/presets/`:** root `configs/` holds user-facing examples and is NOT packaged (`[tool.setuptools.packages.find] include=["dnallm*"]`). dnallmmark's F4 lane consumes presets through installed dnallm. In-package data has two precedents (`model_info.yaml`, `configuration/evo/*.yml`) and one required wiring step (`[tool.setuptools.package-data]`). Resolver loads via `importlib.resources`; explicit user `target_modules` still wins; resolution order: explicit → packaged preset → dry-run error.
- **`dnallm/interpret/` new subpackage (intake path honored):** costs one `__init__.py` and a naming note in REV-03 docs (`dnallm.inference.interpret.DNAInterpret` stays). Zero coverage/lint/packaging friction (auto-included by `dnallm*`). Alternative `dnallm/inference/motifs.py` is acceptable but inference/ is already the largest grab-bag; follow the intake.
- **`sweep.py`/`vep.py`/`probing.py`/`motifs.py` are new files only** — this is what makes the Wave plan conflict-free (see Build Order).
- **No new re-exports in `dnallm/__init__.py` for v1.2:** import new capabilities via full paths (`from dnallm.inference.vep import evaluate_vcf`). The facade's `__all__` stays byte-stable; avoids the one file every agent would otherwise touch. Revisit at v1.3 if these graduate to headline API. (Exception: none of the 11 REVs require facade changes.)

## Architectural Patterns

### Pattern 1: Config-dict flow for new features

**What:** New YAML-facing surface follows the established section pattern: Pydantic class in `configs.py` → key in `DNALLMConfig` (TypedDict, `total=False`) → filled by `load_config()` only when the YAML carries the section → consumed by a facade via `config["<key>"]`.

**Per-REV application:**

| REV | YAML surface | Pydantic placement | Notes |
|-----|-------------|--------------------|-------|
| REV-01 | `finetune.allow_test_as_eval` | field on existing `TrainingConfig` | popped before TrainingArguments construction alongside `use_qlora` etc. (set_up_trainer already pops non-TrainingArguments fields at :197-201) |
| REV-04 | new `ia3:` section | new `Ia3Config` class + `DNALLMConfig["ia3"]` + load_config branch | mirrors `lora:` exactly; `DNATrainer(use_ia3=True)` ctor flag mirrors `use_lora` (:133) |
| REV-05 | none (packaged YAML, not user config) | — | resolver is code + packaged data, not a config section |
| REV-06 | none | — | loader **kwarg**, not a config field; from-scratch baselines are experiment-time, driven by scripts (F8) |
| REV-07/08/09/10 | none for v1.2 | kwargs/dataclasses local to the module | kwargs-first avoids `configs.py` contention (only REV-01 and REV-04 touch it, in different waves); escalate to YAML sections later if dnallmmark needs them |

**Trade-offs:** kwargs-first means dnallmmark calls functions programmatically — which is exactly how its lanes (F4/F5/F8) work today. YAML sections are only needed for human-facing workflows (`dnallm train`), and REV-04 is the only one of those.

### Pattern 2: Dispatch-chain respect (REV-06)

**What:** `load_model_and_tokenizer` is a guarded chain: special-family handlers may return early (evo2/evo1/megadna/enformer/space/borzoi at :807-863), then crossdna → dnabert2 → generic `_load_model_by_task_type` run as a first-resolved-wins guarded chain (:898-918) with shared post-processing (:920-941) that must never be skipped.

**REV-06 insertion:** add `random_init` as a kwarg threaded into `load_args`; implement the config-only load **inside `_load_model_by_task_type`** as an alternative construction branch per task type (`AutoConfig.from_pretrained` + `AutoModelForX.from_config` + `init_weights()`). Early-returning special families are simply out of scope for random_init — document it, don't fight it (R2-5 baselines run on standard BERT/GPT/Mamba-family HF models). Preserve all post-processing (`_configure_model_padding`, device placement) — the anti-pattern "overwriting handler results in the load chain" (`codebase/ARCHITECTURE.md`) applies.

### Pattern 3: PEFT adapter symmetry (REV-04/05)

**What:** The LoRA branch at trainer.py:153-170 is: optional kbit-prep → `LoraConfig(**config["lora"].dict())` → `peft_forward_compatiable(model)` → `get_peft_model`. IA³ is the same shape with `peft.IA3Config`. Both produce a `PeftModel`, so save (`model.save_pretrained`) and reload (`DNAInference(..., lora_adapter=path)` → `PeftModel.from_pretrained`, inference.py:111-131) work identically — the reload path only needs its parameter name/docs generalized (it does not inspect adapter type).

**Trade-offs:** one shared "adapter" concept instead of parallel lora/ia3 code paths; QLoRA+IA³ combination is naturally excluded (IA³ has no rank) — guard with a clear ValueError if both flags are set.

### Pattern 4: Composition over modification for new capabilities (REV-07/08/09/10)

**What:** New modules are built as thin orchestrators over existing public seams, never by editing them: probing → `DNAInference.get_embeddings`; vep → `Mutagenesis.clm_evaluate/mlm_evaluate` kernels (instantiating Mutagenesis or extracting the kernels as reusable functions — prefer calling the class, extraction is a refactor); sweep → `DNATrainer` + `extra_args={"seed": s}`; motifs → `Mutagenesis.prepare_tfmodisco_inputs` output shape.

**When to use:** always, given the constraint "no refactors beyond what correctness/coverage requires" and the coverage gate that forces new code to be tested anyway.

### Pattern 5: Conventions every new module must follow

- Relative imports inside `dnallm` (`from ..inference.inference import DNAInference`); absolute in tests (`from dnallm.tasks.metric_registry import resolve_metric`).
- Function-local imports for heavy deps (sklearn in probing, peft in trainer branch — existing pattern).
- Optional-dep guards: none of the new modules introduces a currently-absent dependency (verified: sklearn, numpy, peft, pyyaml, click all present). `pysam`/biopython are NOT used — VCF and PWM parsing are hand-rolled minimal readers on stdlib+numpy.
- `logger = get_logger("dnallm.<sub>.<mod>")`; no bare `print` (T20).
- ValueError with matchable message at boundaries; Google docstrings; PEP 604 unions; ruff line-length 100.
- Registry module (`metric_registry.py`) must be import-light and side-effect-free: dnallmmark CI imports just the table — keep it to a dict of canonical → (aliases, lazy callable refs over numpy/sklearn).

## Data Flow

### REV-01 (training/eval semantics)

```
YAML finetune.allow_test_as_eval ──► TrainingConfig ──► DNATrainer.set_up_trainer
  splits from DatasetDict keys ("train"/"test"/"val" — data.py split_data)
  case dev+test → eval=val (unchanged)
  case test-only, allow=False  → eval_dataset=None, eval_strategy="no",
                                 load_best_model_at_end=False, WARN log     [NEW DEFAULT]
  case test-only, allow=True   → legacy behavior + WARN
  case train-only              → eval_strategy="no" (unchanged)
evaluate(split=None|"val"|"test"|...) → split=None: trainer.evaluate()
                                    → split given: trainer.predict(dataset[split])  [NEW]
infer() kept as-is (predict on test, :502) — no deprecation this milestone
```

### REV-04/05 (IA³ + presets)

```
YAML ia3: ──► Ia3Config ──► DNALLMConfig["ia3"]
DNATrainer(use_ia3=True): Ia3Config(target_modules=presets.resolve(family,"ia3") if None)
   → peft.IA3Config(**...) → peft_forward_compatiable(model) → get_peft_model
save: model.save_pretrained(output_dir)            [shared with LoRA]
reload: DNAInference(..., lora_adapter=dir) → PeftModel.from_pretrained  [shared path]
presets.resolve(family, kind): packaged YAML via importlib.resources;
   family inferred from model config.json module names (never guessed — dry-run error)
```

### REV-07 (probing)

```
config dict (task+inference sections) → DNAInference(model, tokenizer, config)
extract_embeddings(dataset, layer, pooling) → engine.get_embeddings(...)   [REUSE]
   → npz cache keyed (model, dataset, layer, pooling)
fit_probe(X_train, y_train, kind="logistic"|"mlp", val for early stop) → sklearn, fixed HPs
metrics via metric_registry.resolve("AUROC")(y_true, y_score)              [REV-02 dep]
output: probe row joinable with finetune results table (F4 lane contract)
```

### REV-08 (zero-shot VEP)

```
VCF row (chrom,pos,ref,alt) + reference window → align_variant(tokenizer)
   → tokenize ref-window & alt-window; variant must land in the SAME token slot index
   → slot mismatch (BPE/k-mer multi-token alleles) → explicit skip, counted   [R1-3e① answer]
score_variant: CLM Δlog-lik = clm_evaluate(alt) − clm_evaluate(ref)          [kernel reuse]
               MLM log-odds = mlm_evaluate at masked slot (ref vs alt id)     [kernel reuse]
evaluate_vcf → per-variant scores + skip tally + AUROC/AUPRC via registry    [REV-02 dep]
CLI: dnallm vep -c config.yaml --vcf in.vcf --output scores.tsv  (lazy import)
```

### REV-09 (multi-seed sweep)

```
run_seeds(fn, seeds=[42,43,44], out_root):
   for s: config copy with finetune.seed=s (or extra_args={"seed": s})
          fn(config) → results JSON at {out_root}/{model}/{task}/seed_{s}/
aggregate_seeds(values) → {mean, sd, ci95_bootstrap}   [PURE FUNCTION — build first]
final: results JSON gains a "statistics" block (schema documented in sweep.py)
```

### REV-10 (motifs)

```
Mutagenesis ISM run → hotspots (windows) + per-position hyp scores
   (prepare_tfmodisco_inputs emits (one_hot, hypothetical_scores) — same data shape)
motifs.match(window_seqs/scores, pwm_dir_or_dict) → log-odds scan vs JASPAR/CIS-BP PWMs
   → threshold (log-odds) + shuffled-background FDR → table(motif_id, coords, e_value)
```

### REV-11 (MCP)

```
client tool call → _with_timeout_wrapper → handler:
  ism_scan / hotspots → ModelManager engine → Mutagenesis(model, tokenizer, config)
  zero_shot_score → ModelManager engine → vep.score_variant / align_variant   [REV-08 dep]
  → error dicts across the boundary (never raises); single-flight via _infer_thread_lock
main(): CLI --host/--port take precedence over yaml (currently silently overridden — fix + test)
```

## Test Layering

The gate mechanics (verified in ci.yml): **fast PR leg runs `pytest -m "not slow" --cov` — the >90% gate is enforced on the fast lane alone**; the nightly GPU census runs the full suite including `slow` (also gated). Consequence: **every new `dnallm/` module must reach the coverage bar via mocked fast-lane tests; slow tests are acceptance, not gate-protection.**

| Layer | Marker/lane | What lives there (per REV) |
|-------|------------|---------------------------|
| Fast unit (PR leg, mocked) | unmarked, `not slow` | REV-01 split-semantics matrix: 3 split combos × default/explicit override, mocked DatasetDict + stub Trainer (`tests/finetune/test_trainer.py` extension). REV-02 registry contract: all task-type key spellings, alias resolution both directions, unknown-name error. REV-05 preset schema/dry-run tests (wrong module name → error; family coverage count). REV-04 Ia3Config validation + branch wiring with a stub model; adapter save/reload round-trip with a tiny in-memory PEFT-able model. REV-06 config-only init on a tiny HF-style config (e.g. 2-layer BERT config constructed in-test, no download); param-hash differs from seeded path. REV-07 extract_embeddings with mocked engine + fake hidden states; cache hit second call; probe fit on synthetic arrays. REV-08 align_variant with a hand-built char/k-mer/BPE fake tokenizer (same-slot vs skip cases); score math on synthetic logprobs; VCF reader parsing fixtures. REV-09 aggregate_seeds on constructed arrays (known mean/sd/CI); dir-protocol layout test. REV-10 PWM scan on synthetic PWM + sequence with planted site; FDR monotonicity; coordinates 0/1-based round-trip. REV-11 three tools with mocked ModelManager (precedent: `tests/mcp/test_mutagenesis_tool.py`); `--host/--port` precedence test on `main()` arg parsing. |
| Slow / network (nightly GPU census) | `@pytest.mark.slow` | REV-04 acceptance: 1 transformer + 1 mamba model, one task, adapter save/reload consistency. REV-06: two architectures, pretrained-vs-random param-hash assertion + divergent loss sanity. REV-07: one real model × one binary task end-to-end. REV-08: ClinVar 1k-sample × ≥5 models AUROC (acceptance criterion) — guard with the existing `network-unavailable:` typed-skip pattern if the runner can't reach the source; any skip MUST be registered in `tests/expected_skips.yaml` or CI fails. REV-09: ≥3-seed small-task full-chain trial run. REV-10: HBG1 BCL11A coordinate agreement with Fig 4a. REV-11: live server → client → 3-tool handshake (stdio transport) JSON assertions. |
| Giants lane | unchanged | none of the REVs touches evo-class models; no `giants` additions. |

Also: new `dnallm/` files automatically join the coverage denominator (`source_pkgs=["dnallm"]`) and the D-03 collected-test census counts will move — expect `pyproject` census-pinning docs/tests that hard-assert collected counts (v1.1 re-pin precedent, `c99fa9d`) to need a same-change bump; assign that to whichever agent lands the last test file of each wave, or to the wave-integration commit.

## Build Order for 4–5 Parallel Agents (waves)

Dependency chain from the intake (honored): REV-02 → benchmark F3 gate; REV-01 independent; REV-04 → REV-05; REV-07 reuses scoring embedding path (+ REV-02 for metrics); REV-08 reuses mutagenesis kernels; REV-09 pure functions first; REV-10/11 freestanding — **plus REV-08 → REV-11 (zero_shot_score wraps vep)**, and REV-07 → REV-02 (registry metrics).

**Shared-file conflict analysis (the constraint that shapes the waves):**

| Hot file | Touched by | Resolution |
|----------|-----------|------------|
| `dnallm/configuration/configs.py` | REV-01 (TrainingConfig field), REV-04 (Ia3Config + section) | Different waves (W1 vs W2) |
| `dnallm/finetune/trainer.py` | REV-01 (:234-241, evaluate), REV-04 (:153-170) | Different waves (W1 vs W2) |
| `dnallm/inference/inference.py` | REV-04 (:111-131 adapter param) only | W2 sole owner; probing/vep never edit it |
| `dnallm/__init__.py` / subpackage inits | nobody (no new re-exports rule) | avoided by convention |
| `dnallm/tasks/metrics.py` | REV-02 only | W1 sole owner |
| `dnallm/mcp/server.py` | REV-11 only | W3 sole owner |
| `dnallm/cli/cli.py` | REV-08 only | W2 sole owner |

### Wave 1 — Protocol & contract layer (P0, ~0.5 day, 4 agents)

| Agent | Scope | Files owned | Dependencies |
|-------|-------|-------------|--------------|
| W1-A | **REV-01** eval-semantics guard + `evaluate(split=)` + docstrings | `finetune/trainer.py`, `configuration/configs.py`, `tests/finetune/` | none |
| W1-B | **REV-02** metric registry + metrics.py rewire + contract tests | `tasks/metric_registry.py` (new), `tasks/metrics.py`, `tests/tasks/` | none; unblocks REV-07 |
| W1-C | **REV-03** terminology/comparability docs, validate_sequences docstring, CHANGELOG skeleton | `docs/**`, `README.md`, `CHANGELOG.md`, docstring in `datahandling/data.py` | none (IA³/preset usage chapter is a W3 addendum — REV-03 acceptance completes at W3) |
| W1-D | **REV-08 core** — `align_variant` + scoring kernels + unit tests (long-pole start; 2-day item begun early) | `inference/vep.py` (new), `tests/inference/test_vep.py` | reads mutagenesis kernels only |

No file overlaps. W1-B is the gate for the benchmark re-run (F3) — it must land first regardless.

### Wave 2 — Adaptation & evaluation capabilities (P1, ~1 day, 5 agents)

| Agent | Scope | Files owned | Dependencies |
|-------|-------|-------------|--------------|
| W2-A | **REV-04 + REV-05** (one agent: intake dependency REV-04→REV-05 + shared files) | `configuration/configs.py` (Ia3Config), `configuration/presets/lora_targets.yaml`, `finetune/presets.py`, `finetune/trainer.py` (IA³ branch), `inference/inference.py` (:111-131), `pyproject.toml` (package-data), `tests/` | W1 merged (trainer.py/configs.py free) |
| W2-B | **REV-06** random_init loading | `models/model.py`, `tests/models/` | none |
| W2-C | **REV-07** probing | `inference/probing.py` (new), `tests/inference/test_probing.py` | W1-B registry |
| W2-D | **REV-09** sweep | `finetune/sweep.py` (new), `tests/finetune/test_sweep.py` | none (pure functions first; trainer untouched) |
| W2-E | **REV-08 completion** — `evaluate_vcf`, CLI command, ClinVar slow test | `inference/vep.py` (continues W1-D), `cli/cli.py`, slow test | W1-D |

W2-A is the heaviest single assignment (REV-04 0.5d + REV-05 0.5d per intake); it is one agent precisely because its two REVs share every hot file. W2-C and W2-E both add files under `inference/` but never the same file.

### Wave 3 — Narrative surface & integration (P2 + closeout, ~0.5 day, 3 agents)

| Agent | Scope | Files owned | Dependencies |
|-------|-------|-------------|--------------|
| W3-A | **REV-10** motifs | NEW `interpret/` subpackage, `tests/interpret/` | none (freestanding) |
| W3-B | **REV-11** MCP tools + host/port fix | `mcp/server.py`, `tests/mcp/`, `dnallm/mcp/tests/` | W2-E (vep functions exist) — ism_scan/hotspots can land even if vep slips; zero_shot_score follows |
| W3-C | **Integration & REV-03 completion** — IA³/LoRA/preset usage chapter, terminology final sweep, CHANGELOG v0.7.2 finalization, census-count re-pin, coverage-expectation docs update | `docs/**`, `CHANGELOG.md`, census docs | all waves merged |

Parallel-safety check per wave: within every wave, no two agents own the same file (verified against the conflict matrix). Cross-wave, the two hot files (`configs.py`, `trainer.py`) change ownership W1→W2 cleanly; waves are sequential merges (orchestrator commits between waves).

**Calendar:** W1 0.5d → W2 1.0d → W3 0.5d ≈ 2 days elapsed at 4–5 concurrent agents — matching the owner's ~1–1.5-day compression once wave merges are pipelined (W3-A can start during W2 since it touches nothing W2 owns; pulling it forward yields the 1.5-day figure).

**Phase-mapping suggestion for the roadmapper:** Wave 1 ≈ Phase A (contract gate — milestone-defining, hard requirement), Wave 2 ≈ Phase B (two parallel phase-groups B1: adaptation [REV-04/05/06], B2: evaluation [REV-07/08/09]), Wave 3 ≈ Phase C (narrative [REV-10/11] + docs closeout). The intake's own Phase A/B/C split matches; the wave table adds the agent-level file ownership the phase planners need.

## Anti-Patterns

### Anti-Pattern 1: Placing the metric registry inside the vendored `dnallm/tasks/metrics/` directory

**What people do:** follow the intake path literally (`dnallm/tasks/metrics/registry.py`).
**Why it's wrong:** that glob is coverage-omitted (pyproject :542) and ruff-excluded (:302) because it mirrors upstream HF `evaluate`. The registry is the cross-repo contract — the one file that most needs the gate, lint, and mypy. It would also sit beside ~90 vendored metric dirs, inviting "hand-edit vendored code" drift.
**Do this instead:** `dnallm/tasks/metric_registry.py` (sibling of `metrics.py`); dnallmmark F3 imports that path. If the intake path is retained, the omit/exclude lists must be narrowed — denominator churn.

### Anti-Pattern 2: Presets YAML at repo-root `configs/presets/`

**What people do:** follow the intake path literally.
**Why it's wrong:** root `configs/` is user-facing example territory and is not in the wheel; dnallmmark (and any `pip install dnallm` user) could not resolve it.
**Do this instead:** `dnallm/configuration/presets/lora_targets.yaml` + `[tool.setuptools.package-data]` entry (precedent: `model_info.yaml`, `configuration/evo/*.yml`).

### Anti-Pattern 3: Building IA³ as a parallel adapter stack

**What people do:** new save/load/reload code paths "for IA³".
**Why it's wrong:** both LoRA and IA³ produce `PeftModel`s; the existing save (`save_pretrained`) and reload (`inference.py:111-131`) are adapter-type-agnostic.
**Do this instead:** one branch shape in `DNATrainer.__init__` beside LoRA, shared `peft_forward_compatiable`, reload parameter renamed/generalized once. Reject `use_ia3 + use_lora` together with ValueError; IA³ has no QLoRA pairing (no rank).

### Anti-Pattern 4: `random_init` fighting the special-family handlers

**What people do:** thread random_init through evo/enformer/space/borzoi handlers.
**Why it's wrong:** those handlers return early with their own construction paths; forcing them in doubles surface area for a baseline experiment (R2-5) that targets standard HF families.
**Do this instead:** implement in `_load_model_by_task_type` only; log + document the scope; from-scratch on special families is out of scope for v1.2.

### Anti-Pattern 5: New skips without registry entries

**What people do:** `pytest.skip("no network")` in the ClinVar/VEP slow tests.
**Why it's wrong:** `scripts/audit_skips.py` fails CI on any skip whose message is not in `tests/expected_skips.yaml`.
**Do this instead:** use the typed `network-unavailable:`/`optional-dep:` helper pattern and register the prefix.

### Anti-Pattern 6: Slow tests as the coverage vehicle for new modules

**What people do:** write the real-model acceptance tests and call coverage done.
**Why it's wrong:** the PR-leg gate runs `-m "not slow" --cov`; nightly-only tests do not protect merges.
**Do this instead:** every new module ships mocked fast-lane unit tests reaching the bar; slow tests are the acceptance layer (ClinVar AUROC, HBG1 coordinates, adapter round-trips).

### Anti-Pattern 7: Moving `DNAInterpret` into the new `dnallm/interpret/` subpackage

**What people do:** "tidy" the naming collision while creating the subpackage.
**Why it's wrong:** refactor beyond correctness needs; breaks `dnallm.inference.interpret` import path and the facade `__all__`.
**Do this instead:** leave `dnallm/inference/interpret.py` untouched; note the distinction in REV-03 docs.

## Integration Points

### External (cross-repo — coordination, not implementation, in this repo)

| Counterparty | Contract | Notes |
|--------------|----------|-------|
| dnallmmark F3 (exporter) | imports `dnallm.tasks.metric_registry` (or the intake path if owner overrides); shared alias table; contract test in both CIs | open intake question #3 (version locking) — recommend the registry module be dependency-light (numpy/sklearn only) so a pinned `dnallm` import is cheap in dnallmmark CI |
| dnallmmark F4 (adaptation lane) | consumes `Ia3Config`, presets resolver, probing outputs (joinable table row) | probing output schema must be agreed when F4 is written — document the emitted columns in probing.py |
| dnallmmark F5 (zero-shot lane) | consumes `vep.evaluate_vcf` + skip counts; scoring formulas documented in docstrings (protocol declaration for the reply letter) | GPN-class models are tokenizer-less — intake open question #4: keep v1.2 scope to tokenizer-based models; slot rule extension deferred |
| dnallmmark F8 (learning curve) | consumes `random_init=True` + `DNADataset.sampling` (verified signature at data.py:1058) | kwarg API, no YAML surface needed |
| JASPAR/CIS-BP | PWM files read at runtime (user-supplied path); no bundled data | keeps license/bulk out of the repo; HBG1 test uses a committed tiny PWM fixture |

### Internal boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| probing ↔ inference | calls `DNAInference.get_embeddings` (public) | never edits inference.py |
| vep ↔ mutagenesis | calls `clm_evaluate`/`mlm_evaluate` on a Mutagenesis instance | mutagenesis.py stays read-only; if kernels need a variant-conditional variant, prefer a small public wrapper in vep.py over editing kernels |
| sweep ↔ finetune | `run_seeds(fn, ...)` — trainer as a callable | sweep.py never imports trainer internals |
| motifs ↔ inference | consumes `prepare_tfmodisco_inputs`/hotspots output shape | new subpackage imports `..inference.mutagenesis` (relative) |
| mcp ↔ inference | ModelManager engines + vep functions | tools follow the error-dict + timeout-wrapper convention |

## Sources

All findings verified directly against the working tree (branch `revision`, 2026-10-09) — confidence HIGH unless noted:

- `/home/forrest/Github/DNALLM/dnallm/finetune/trainer.py` — `__init__`/LoRA branch :127-175, eval-split selection :234-241, extra_args merge :195-196, `search` :422, `evaluate` :489, `infer` :502
- `/home/forrest/Github/DNALLM/dnallm/tasks/metrics.py` — metric key emission sites :128-150, :181-219, :261-318
- `/home/forrest/Github/DNALLM/dnallm/configuration/configs.py` — TrainingConfig :263-337, LoraConfig :340-371, `DNALLMConfig` :495-511, `load_config` :513-557
- `/home/forrest/Github/DNALLM/dnallm/models/model.py` — `load_model_and_tokenizer` :753-941 (special handlers :807-863, guarded chain :898-918, post-processing :920-941), `_load_model_by_task_type` :549-659, `peft_forward_compatiable` :1007-1026
- `/home/forrest/Github/DNALLM/dnallm/inference/inference.py` — adapter-reuse path :111-131, `scoring` :1746-1819, `get_embeddings` :1978-2077
- `/home/forrest/Github/DNALLM/dnallm/inference/mutagenesis.py` — `mlm_evaluate` :257-309, `clm_evaluate` :311-347, hotspots :575, `prepare_tfmodisco_inputs` :583
- `/home/forrest/Github/DNALLM/dnallm/datahandling/data.py` — `split_data` :798 (split names `"train"/"test"/"val"`), `sampling` :1058
- `/home/forrest/Github/DNALLM/dnallm/mcp/server.py` — `_register_tools` :238-292, `_with_timeout_wrapper` :295, `main` :1928; test precedents `tests/mcp/test_{mutagenesis,interpret}_tool.py`
- `/home/forrest/Github/DNALLM/pyproject.toml` — coverage omit :539-549, `fail_under=90` :553, ruff exclude incl. `dnallm/tasks/metrics/` :302, pytest markers :330-345
- `/home/forrest/Github/DNALLM/.github/workflows/ci.yml` — fast leg `pytest -m "not slow" --cov` :106; skip audit :108-111
- `/home/forrest/Github/DNALLM/dnallm/__init__.py` — facade `__all__` and import-order constraint (utils before models)
- `/home/forrest/Github/DNALLM/.planning/research/261009-paper-revision-suite-plan.md` — REV definitions, dependency chain, D1-D4 rulings (intake, owner-aligned)
- `/home/forrest/Github/DNALLM/.planning/codebase/ARCHITECTURE.md` — existing architecture map (2026-10-05 refresh)

---
*Architecture research for: DNALLM v1.2 Paper Revision Suite Support*
*Researched: 2026-10-09*
