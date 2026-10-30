---
title: Architecture Overview
---

<!-- generated-by: gsd-doc-writer -->

# Architecture Overview

This document describes how the DNALLM package is organized at the code level: the major modules, how a configuration flows from a YAML file through model loading, dataset preparation, training, and inference, and how the MCP server exposes trained models as tools. It is aimed at contributors who want to understand where a change lands, and at users who want to know what actually runs when they call `DNATrainer` or `DNAInference`.

## 1. System Overview

DNALLM is a **configuration-driven Python toolkit** for fine-tuning and running DNA large language models. Its primary input is a YAML configuration file plus DNA sequence data (local files, or preset datasets hosted on Hugging Face / ModelScope); its outputs are fine-tuned model checkpoints with training metrics, prediction results (labels, scores, embeddings, attention maps), and — through the MCP layer — standardized prediction services consumable by AI assistants and web clients.

Architecturally, DNALLM is a **layered framework built on the Hugging Face ecosystem**: model loading wraps `transformers` auto-classes, datasets wrap the `datasets` library, and training delegates the optimization loop to the Hugging Face `Trainer`. DNALLM's own value sits in the layers above this foundation — a model registry (39 families plus 236 typed registry entries) whose transformers-5 compatibility is *measured*, not guessed, by a fail-closed census; task-aware model heads; PEFT adapter presets; honest evaluation semantics; in-silico mutagenesis and interpretation tooling; an MCP server for serving predictions; and a pre-flight path (`dnallm/preflight` + the `dnallm preflight` terminal wizard) that takes a user from raw data to a validated, model-compatible config file before any training starts.

The single YAML config file is the source of truth for every workflow: each top-level section (`task`, `model`, `finetune`, `lora`, `ia3`, `inference`, `vep`, `sweep`, `benchmark`) is parsed into a typed Pydantic configuration object by `dnallm.configuration.load_config`, and downstream components consume only the sections they need.

## 2. Component Diagram

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           Entry points                               │
│   dnallm CLI + console scripts  ·  dnallm preflight (TUI)        │
│   ·  Python API  ·  MCP clients                                  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ YAML config (--config / load_config)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Configuration — dnallm/configuration                                │
│  Pydantic section models: TaskConfig, TrainingConfig, LoraConfig,    │
│  Ia3Config, InferenceConfig, VepConfig, SweepConfig, BenchmarkConfig │
└─────────┬────────────────────────────────────────────┬───────────────┘
          ▼                                            ▼
┌───────────────────────┐  tokenizer   ┌───────────────────────────┐
│ Models                │─────────────►│ Datahandling               │
│ dnallm/models         │              │ dnallm/datahandling        │
│ MODEL_INFO registry,  │              │ DNADataset (local files,   │
│ load_model_and_       │              │ HF Hub, ModelScope presets)│
│ tokenizer(), heads    │              └─────────────┬─────────────┘
└─────────┬─────────────┘                            │ encoded splits
          │ model + tokenizer                        ▼
          ▼                          ┌───────────────────────────────┐
┌───────────────────────┐ checkpoint │ Inference — dnallm/inference   │
│ Finetune              │───────────►│ DNAInference · Benchmark ·     │
│ dnallm/finetune       │            │ Mutagenesis · DNAInterpret ·   │
│ DNATrainer (HF        │            │ VEP · probing · plotting       │
│ Trainer) + sweep      │            └─────────────┬─────────────────┘
└───────────────────────┘                          │ predictions
                                                   ▼
                                    ┌───────────────────────────────┐
                                     │ MCP server — dnallm/mcp        │
                                     │ ModelManager wraps multiple    │
                                     │ DNAInference engines; exposes  │
                                     │ FastMCP tools over stdio/SSE/  │
                                     │ streamable-HTTP                │
                                     └───────────────────────────────┘

  Cross-cutting: dnallm/tasks (TaskType enum + task metrics)
                 dnallm/utils (logging, transformers/CUDA compatibility
                 shims, genomic coordinate helpers, training plots)
                 dnallm/preflight + dnallm/tui (read-only pre-flight
                 checks; Textual wizard renders the reports)
                 dnallm/models/support (plain-language support words
                 derived from the census verdicts in the registry)
```

## 3. Data Flow

### 3.1 Fine-Tuning: config → model → dataset → trainer

The canonical fine-tuning path, as shown in the [Quick Start](../getting_started/quick_start.md):

1. **Load configuration.** `load_config("finetune_config.yaml")` (in `dnallm/configuration/configs.py`) parses the YAML file and returns a dictionary of typed config objects — one key per section present in the file.
2. **Load model and tokenizer.** `load_model_and_tokenizer(model_name, task_config=...)` (in `dnallm/models/model.py`) resolves the model name through the `MODEL_INFO` registry in `dnallm/models/modeling_auto.py`, downloads weights from Hugging Face or ModelScope (a mirror can be configured), and prepares the model for the configured task — including attaching a classification head (`DNALLMforSequenceClassification`) and label mappings when the task requires them.
3. **Build the dataset.** `DNADataset.load_local_data(...)` reads local files (CSV, TSV, JSON, Parquet, FASTA, TXT, or pickle — a single file or a pre-split dict such as `{"train": ..., "test": ...}`), or `load_preset_dataset(...)` fetches a named preset dataset from Hugging Face or ModelScope. `DNADataset.encode_sequences()` tokenizes the sequences so the trainer receives ready-to-consume input IDs.
4. **Train.** `DNATrainer(model, config, datasets, use_lora=...)` (in `dnallm/finetune/trainer.py`) applies the requested PEFT method (LoRA, QLoRA with 4-bit quantization, or IA³, using family-specific target-module presets), maps `TrainingConfig` fields onto Hugging Face `TrainingArguments`, selects task metrics from `dnallm/tasks/metrics.py` and the right data collator per task type, then delegates the loop to the Hugging Face `Trainer`. `train()` returns metrics and saves the checkpoint (and tokenizer) to `output_dir`.
5. **Optional outer protocols.** `dnallm/finetune/sweep.py` drives *N* independent seed runs from outside `DNATrainer` and aggregates mean / SD / 95% CI into a `statistics.json`; `dnallm/finetune/megatron.py` provides a Megatron-LM training path for NPU deployments of Mamba-2 models.

Evaluation follows deliberate honesty rules enforced inside `DNATrainer`: the test split is held out from per-step evaluation and best-model selection unless `finetune.allow_test_as_eval: true` is set explicitly, and `load_best_model_at_end` / early stopping fail loudly when no evaluation split exists.

### 3.2 Inference: trained model → predictions

`DNAInference` (in `dnallm/inference/inference.py`) is the inference engine. Constructed with a model, a tokenizer, and the config dictionary, its main entry point `infer()` dispatches to `infer_seqs()` (in-memory sequences) or `infer_file()` (input file), which:

1. Build batched inputs (padding and attention masks handled per model family, including models with non-standard forward signatures),
2. Run the forward pass on the selected device (CPU / CUDA / MPS),
3. Convert logits to predictions (`logits_to_preds`) for classification and regression tasks,
4. Optionally capture **hidden states** and **attention weights** for interpretability, and
5. Return formatted results, or metrics when `evaluate=True` and labels are available.

Sibling engines build on the same model + tokenizer + config pattern:

| Engine | File | Purpose |
|---|---|---|
| `Benchmark` | `dnallm/inference/benchmark.py` | Multi-model evaluation across tasks and model sources |
| `Mutagenesis` | `dnallm/inference/mutagenesis.py` | In-silico mutagenesis (substitutions, deletions, insertions) to find sequence positions that drive predictions |
| `DNAInterpret` | `dnallm/inference/interpret.py` | Captum-based attribution (e.g., LayerIntegratedGradients) |
| VEP pipeline | `dnallm/inference/vep.py` | Zero-shot variant effect prediction with token-aligned ref/alt alleles and ClinVar filtering |
| Probing | `dnallm/inference/probing.py` | Layer-wise embedding extraction (with caching) and probe training |
| Plotting | `dnallm/inference/plot.py` | Shared result-visualization helpers |

`dnallm/interpret/motifs.py` adds FIMO-convention motif scanning against JASPAR / CIS-BP PWMs with exact dynamic-programming p-value calibration.

### 3.3 MCP Service: predictions as tools

The MCP layer (`dnallm/mcp/`) turns inference engines into a network service, described in depth in [MCP Concepts](../concepts/mcp.md):

1. `MCPConfigManager` reads `mcp_server_config.yaml`, which lists one inference config file per enabled model.
2. `ModelManager` loads each enabled model as its own `DNAInference` engine and exposes async predict operations (single, batch, multi-model).
3. `DNALLMMCPServer` (in `dnallm/mcp/server.py`) registers these operations as FastMCP tools — `dna_sequence_predict`, `dna_batch_predict`, `dna_multi_model_predict`, their streaming (`dna_stream_*`) variants with real-time progress, plus model-management and health-check tools. The model-listing tool annotates every entry with the same plain-language support word the TUI shows (see 3.5).
4. The server runs over **stdio**, **SSE**, or **streamable-HTTP** transports, started via the `dnallm-mcp-server` console script or `dnallm mcp-server`.

### 3.4 Pre-flight: raw data → validated config

Before any of the flows above, a user can run the pre-flight path to find out whether their data and chosen model are compatible *before* downloading weights or writing a config by hand:

1. **Validate data.** `validate_data(sequences)` (in `dnallm/preflight/`) checks sequences for length, alphabet, and degenerate-base issues and returns a typed report — it is pure, read-only, and offline-safe (no file is ever rewritten).
2. **Check compatibility.** `check_compatibility(report, models=[...], task_type=...)` scores each registry model against the data report; every verdict fact flows through the typed registry in `dnallm/models/registry.py` (REG-01). `probe_availability` is the only network-capable call — opt-in, metadata-only, and every failure degrades to an honest unknown state.
3. **Generate a config.** `generate_config(model)` and `edit_config(path)` produce a config document; `render_yaml` returns the exact text that will be saved, and nothing is written unless an explicit `output_path` is passed.

The terminal wizard (`dnallm preflight`, package `dnallm/tui/`, Textual, installed via the `tui` extra) is a presentation-only shell over these functions: it renders the reports verbatim — every OK / WARN / FAIL rule lives in `dnallm.preflight`, never in the screens. See the [Pre-flight Wizard guide](../user_guide/tui/preflight_wizard.md).

### 3.5 Compatibility census: measured transformers-5 verdicts

Which preset models actually work under transformers 5 is **measured, not assumed**. `scripts/census_compat.py` runs a fail-closed census over the 49 pretrained registry entries: it fetches each repo's *code and config files only* (never weights), prefers ModelScope with the HF mirror pinned, attempts a real load plus one short forward pass, and stamps the verdict back into the packaged `dnallm/models/model_info.yaml` as `tf_compat_status` / `tf_compat_evidence` on pretrained entries. The six verdicts — `verified-tf5`, `import-verified-tf5`, `shim-required-tf5` (works only via DNALLM's built-in compat fixes), `extra-required-tf5` (works only after installing a curated optional extra), `remote-code-broken`, `census-blocked` — are the only compatibility claims the project makes; finetuned entries inherit their base model's verdict through `base_model`. The census runs manually (GitHub `workflow_dispatch` on the self-hosted GPU runner, `.github/workflows/census_compat.yml`) — never on push, PR, or schedule — and emits both a JSON artifact (`tests/models/compat_census.json`) and the generated user-facing page [Model Compatibility](../user_guide/model_compatibility.md).

The verdicts surface to users through one shared vocabulary: `support_word()` in `dnallm/models/support.py` maps each census status to a plain-language word ("works", "works with built-in fixes", "works with an optional extra", "not working with transformers 5", "untested"), follows `base_model` chains so finetuned entries inherit, and applies exact external-package overrides for the handful of repos a generic word cannot describe (for example CrossDNA needs the `flash-attn` extra, and the GENA-LM BigBird sparse model needs `deepspeed`). The TUI listing, the MCP model-listing tool, and the generated docs page all render this same word, so a user sees one consistent answer everywhere. The curated extras behind `extra-required-tf5` verdicts ship as the optional `deepspeed` and `flash-attn` install extras (see [Optional Dependencies](../user_guide/optional_dependencies.md)); a separate `extras_compat.yml` workflow re-verifies those installs stay green.

## 4. Key Abstractions

| Abstraction | Location | Role |
|---|---|---|
| `load_config` / `DNALLMConfig` | `dnallm/configuration/configs.py` | Parses the YAML config into typed Pydantic section objects; single source of truth for every workflow |
| `MODEL_INFO` registry | `dnallm/models/modeling_auto.py` | Catalog of 39 DNA LLM families with architecture, tokenization tags, Hugging Face / ModelScope IDs, and defaults |
| Typed model registry | `dnallm/models/registry.py` | Sole parser of the packaged `model_info.yaml` (236 entries: 49 pretrained + 187 finetuned); frozen `RegistryEntry` objects with accessors `resolve`, `try_resolve`, `get_by_task_type`, `get_by_base_model`, `all_entries`, `is_lab_model`, `effective_max_length`, `effective_tokenizer_type`; read-only by design (no write or serialize-back API) |
| Support vocabulary | `dnallm/models/support.py` | Maps census `tf_compat_status` verdicts to one plain-language support word per entry (finetuned entries inherit from their base); shared verbatim by the TUI listing, the MCP listing, and the generated model-compatibility docs page |
| `load_model_and_tokenizer` | `dnallm/models/model.py` | Family-aware model loading: source resolution (HF / ModelScope / local), task-specific head attachment, optional quantization and random-init gating |
| `load_preset_model` | `dnallm/models/model.py` | Preset loading: resolves a name in two tiers (registry entry name → repo id, then family table), validates `task_type` strictly (never patches it), and copies head fields (`num_labels`, `label_names`, `threshold`) from the registry entry |
| Pre-flight API | `dnallm/preflight/` | Pure, read-only validation before training: `validate_data`, `check_compatibility`, `generate_config`, `edit_config`, plus opt-in `probe_availability` / offline `local_availability` |
| `DNADataset` | `dnallm/datahandling/data.py` | Dataset wrapper: local loading, preset datasets, tokenization, augmentation, statistics |
| `TaskType` / `TaskConfig` | `dnallm/tasks/task.py` | Enum of the 8 supported task types (embedding, mask, generation, binary / multi-class / multi-label classification, regression, token classification) and its config |
| `compute_metrics` factory | `dnallm/tasks/metrics.py` | Builds the task-metrics callable handed to the Hugging Face `Trainer` |
| `DNATrainer` | `dnallm/finetune/trainer.py` | Unified training facade over the Hugging Face `Trainer` with PEFT presets, early stopping, Optuna search, and honest evaluation semantics |
| `DNAInference` | `dnallm/inference/inference.py` | Inference engine handling non-standard model APIs, hidden states, and attention capture |
| `DNALLMMCPServer` / `ModelManager` | `dnallm/mcp/server.py`, `dnallm/mcp/model_manager.py` | FastMCP server and the multi-model manager that wraps `DNAInference` engines as MCP tools |
| Compatibility shims | `dnallm/utils/transformers_compat.py`, `dnallm/utils/cuda_compat.py` | Defensive patches around upstream library behavior (transformers device probing, CUDA 13 wheel libraries) so the model layer stays clean |

Custom model families that need code beyond `transformers` auto-classes live in `dnallm/models/special/` (Enformer, Borzoi, Basenji2, Evo, GPN, megaDNA, Mamba-2 NPU, and others). Task heads used for fine-tuning (`BasicMLPHead`, `BasicCNNHead`, `BasicLSTMHead`, `BasicUNet1DHead`, and family-specific heads) live in `dnallm/models/head.py`, with `FocalLoss` in `dnallm/models/losses.py` and a `DNAOneHotTokenizer` fallback in `dnallm/models/tokenizer.py`.

## 5. Directory Structure

```text
dnallm/
├── cli/            # Click command groups behind the console scripts
├── configuration/  # Pydantic config models, load_config, PEFT target presets
├── datahandling/   # DNADataset: loading, encoding, preset datasets
├── finetune/       # DNATrainer, multi-seed sweep, Megatron path
├── inference/      # DNAInference, Benchmark, Mutagenesis, DNAInterpret, VEP,
│                   # embedding probes, plotting
├── interpret/      # Motif scanning (MEME/JASPAR/CIS-BP, FIMO conventions)
├── mcp/            # FastMCP server, model + config managers, tests
├── models/         # Typed registry + MODEL_INFO family table, support words,
│                   # model loading, heads, losses, tokenizer fallback,
│                   # special families
├── preflight/      # Read-only pre-flight validation (data, compatibility,
│                   # config generation) — imports no torch/transformers
├── tasks/          # TaskType, TaskConfig, metric registry
├── tui/            # Terminal pre-flight wizard (Textual, tui extra) —
│                   # presentation-only shell over dnallm.preflight
└── utils/          # Logging, transformers/CUDA compat, genomic coordinates
                    # and intervals, env facts, training plots
```

The layout follows the pipeline: each top-level package corresponds to one stage (configure → model → data → train / infer), so a change to fine-tuning logic lands in `finetune/` without touching the model registry, and a new model family is added by extending `models/` alone. `tasks/` and `utils/` are deliberately cross-cutting — shared by both training and inference.

Supporting directories at the repository root: `configs/` holds example YAML configs, `example/` holds runnable notebooks and MCP examples, `ui/` contains Gradio web app frontends (config generator, generation task app), `scripts/` holds CI and maintenance tooling (including `census_compat.py`, the transformers compatibility census that regenerates the model-compatibility page), and `tests/` mirrors the package layout.

## 6. Design Principles

- **Config as contract.** Workflows are driven entirely by YAML config files validated into Pydantic models; CLI invocations require `--config` and flag-only runs are rejected rather than guessed.
- **Delegate, don't reimplement.** Optimization goes to the Hugging Face `Trainer`, data handling to `datasets`, adapters to `peft` — DNALLM adds the DNA-specific and family-specific logic those libraries cannot know.
- **Fail loudly on dishonest evaluation.** Test-split leakage guards and PEFT dry-run validation are designed to make silent mistakes impossible, and Ruff checks run before tests in CI.
- **One registry, many sources.** Every model family is resolved through `MODEL_INFO` with Hugging Face, ModelScope, and local paths treated uniformly.
- **Measure, then claim.** Compatibility statements come from a fail-closed census stamped into the registry — never from assumptions — and every listing (TUI, MCP, docs) renders the same support word derived from that verdict, so a claim cannot drift between surfaces.

## 7. Related Documentation

- [Quick Start](../getting_started/quick_start.md) — runnable end-to-end examples of the flows described above
- [Model Architectures](../concepts/architecture/model_architectures.md) — the model families behind the `MODEL_INFO` registry
- [Model Compatibility](../user_guide/model_compatibility.md) — per-model transformers-5 census results (generated page, do not edit by hand)
- [Optional Dependencies](../user_guide/optional_dependencies.md) — install extras, including the curated `deepspeed` and `flash-attn` extras behind the census verdicts
- [Fine-Tuning Guide](../user_guide/fine_tuning/index.md) — configuration and PEFT adapters in depth
- [Inference Guide](../user_guide/inference/getting_started.md) — running predictions, embeddings, and attention extraction
- [Benchmark Guide](../user_guide/benchmark/index.md) — multi-model evaluation
- [MCP Concepts](../concepts/mcp.md) — the protocol layer and DNALLM's server integration
- [CLI Usage](../user_guide/cli/usage.md) — all console scripts and their options
- [Pre-flight Wizard](../user_guide/tui/preflight_wizard.md) — the `dnallm preflight` terminal walkthrough
- [Preset Model Loading](../user_guide/preset_loading.md) — `load_preset_model` and the registry resolution tiers
