---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
<!-- refreshed: 2026-10-05 -->

# Architecture

**Analysis Date:** 2026-10-05

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Entry Layer                                     │
├───────────────────┬────────────────────┬─────────────────────────────────────┤
│ Click CLI         │ MCP server         │ Web UI / notebooks                  │
│ `dnallm/cli/`     │ `dnallm/mcp/`      │ `ui/`, `example/`, `run_cli.py`     │
│ console scripts   │ `dnallm-mcp-server`│ Gradio + Jupyter/marimo             │
└────────┬──────────┴─────────┬──────────┴──────────────────┬─────────────────┘
         │                    │                           │
         ▼                    ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Facade Classes (config-dict driven)                       │
│  DNATrainer   DNAInference   DNAInterpret   Mutagenesis   Benchmark          │
│  `dnallm/finetune/trainer.py`  `dnallm/inference/*.py`                       │
│  All five accept `config: Mapping[str, Any]` (keys: task/finetune/inference/ │
│  lora/model/benchmark) produced by `load_config()`                           │
└────────┬───────────────────────────────┬────────────────────────────────────┘
         │                               │
         ▼                               ▼
┌──────────────────────────┐  ┌──────────────────────────────────────────────┐
│ Models layer             │  │ Data layer                                   │
│ `dnallm/models/`         │  │ `dnallm/datahandling/`                       │
│ load_model_and_tokenizer │  │ DNADataset (HF Dataset wrapper)              │
│ registry + special/      │  │ PRESET_DATASETS (`dataset_auto.py`)          │
│ handlers + heads/losses  │  └──────────────────────────────────────────────┘
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Tasks/Metrics (`dnallm/tasks/`)  │  Utils (`dnallm/utils/`)                 │
│  TaskType, compute_metrics        │  logger, sequence, genomic_coords,       │
│  vendored HF evaluate metrics     │  cuda_compat + transformers_compat shims │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Package facade | Single import surface re-exporting all core classes (`__all__`) | `dnallm/__init__.py` |
| `cli` (Click group) | `dnallm train/inference/benchmark/mutagenesis/config-generator/mcp-server` subcommands; lazy-imports core classes inside command bodies | `dnallm/cli/cli.py` |
| `load_config` | YAML → `DNALLMConfig` TypedDict of validated Pydantic sections (task, model, finetune, lora, inference, benchmark) | `dnallm/configuration/configs.py:513` |
| Model registry | `PRETRAIN_MODEL_MAPS` (35 families) + `MODEL_INFO` (HF/ModelScope repo ids); `model_info.yaml` registry of 219 pre-trained models | `dnallm/models/modeling_auto.py`, `dnallm/models/model_info.yaml` |
| `load_model_and_tokenizer` | Central loader: source resolution (local/huggingface/modelscope), special-family dispatch chain, task-type Auto* selection, quantization, device placement | `dnallm/models/model.py:748` |
| Special model handlers | Per-family load quirks (EVO-1/2, DNABERT-2, GPN, megaDNA, Enformer, SPACE, Borzoi, MutBERT, Basenji2, CrossDNA, OmniDNA) | `dnallm/models/special/*.py` |
| Classification wrapper | `DNALLMforSequenceClassification`: backbone + pluggable head + pooling + loss selection | `dnallm/models/model.py:49` |
| Task heads | MLP / CNN / LSTM / U-Net1D / MegaDNA multi-scale / EVO layer heads | `dnallm/models/head.py` |
| `DNADataset` | HF `Dataset`/`DatasetDict` wrapper: local file loading (csv/tsv/json/parquet/fasta/txt/pkl), HF/ModelScope loaders, tokenization, reverse-complement augmentation, stats, splitting | `dnallm/datahandling/data.py:31` |
| Preset dataset registry | ModelScope-hosted benchmark datasets | `dnallm/datahandling/dataset_auto.py:1` |
| `DNATrainer` | HF `Trainer` wrapper: task metrics binding, LoRA/QLoRA via PEFT, early stopping, Optuna search (`search()`, `dnallm/finetune/trainer.py:422`), training plots | `dnallm/finetune/trainer.py:66` |
| Megatron trainer | Standalone Megatron-LM loop for Mamba2 (excluded from lint/mypy) | `dnallm/finetune/megatron.py` |
| `DNAInference` | Inference engine: `batch_infer`/`infer_seqs`/`infer_file`/`infer`, logits→predictions, embeddings/attention, `generate`, `scoring`, `get_embeddings`, plotting, model info/memory estimation | `dnallm/inference/inference.py:60` |
| `DNAInterpret` | Captum attribution (IntegratedGradients, LayerIntegratedGradients, DeepLift, LayerDeepLift) over token ids/embeds | `dnallm/inference/interpret.py:95` |
| `Mutagenesis` | In-silico saturation mutagenesis + effect visualization | `dnallm/inference/mutagenesis.py:30` |
| `Benchmark` | Multi-model × multi-dataset evaluation driven by `BenchmarkConfig` | `dnallm/inference/benchmark.py:30` |
| Metrics dispatcher | `compute_metrics` per task type (binary/multiclass/multilabel/regression/token), `preprocess_logits_for_metrics` | `dnallm/tasks/metrics.py` |
| Task types | `TaskType` enum + `TaskConfig`/`TaskHead` definitions | `dnallm/tasks/task.py` |
| Vendored metrics | Ported HF `evaluate` implementations (BLEU, ROUGE, F1, seqeval, ...) — treat as upstream | `dnallm/tasks/metrics/*/` |
| MCP server | `DNALLMMCPServer` on FastMCP: tools registered in `_register_tools` (`dnallm/mcp/server.py:238`) — dna_sequence_predict, dna_batch_predict, dna_multi_model_predict, list_loaded_models, get_model_info, filtering, health_check, stream variants, dna_mutagenesis, dna_interpret; transports stdio/SSE/streamable-HTTP via uvicorn | `dnallm/mcp/server.py:69` |
| `MCPConfigManager` | Parses `dnallm/mcp/configs/mcp_server_config.yaml` (server, models, multi_model groups) | `dnallm/mcp/config_manager.py:20` |
| `ModelManager` | Lazy async load/unload of `DNAInference` engines per configured model; predict routing; single-flight inference | `dnallm/mcp/model_manager.py:22` |
| `DNALLMMCPClient` | Client wrapper for talking to the MCP server | `dnallm/mcp/client.py:47` |
| Logger | loguru-based `get_logger`/`setup_logging` singleton | `dnallm/utils/logger.py` |
| Genomic coords | Single source for coordinate (BED ↔ GFF3) and chromosome-name (TAIR ↔ Ensembl) normalization; pyfastx FASTA fetch; GFF3 attribute parsing and locus slicing — raises loudly, never returns silent-empty | `dnallm/utils/genomic_coords.py` |
| Compat shims | Import-time monkey patches: transformers/bitsandbytes fixes, CUDA 13 wheel library preloading | `dnallm/utils/transformers_compat.py`, `dnallm/utils/cuda_compat.py` |
| Example-execution harness | Private nbclient execution machinery: NOTEBOOK_EXEC_SPECS budgets, sandbox seeding, isolated kernel lanes, typed-skip helpers | `tests/examples/_execution.py` |
| Skip gate | Every junit `<skipped message>` must match an allowlist entry or CI fails | `scripts/audit_skips.py` + `tests/expected_skips.yaml` |

## Pattern Overview

**Overall:** Config-driven facade wrappers over the Hugging Face ecosystem.

**Key Characteristics:**
- YAML config files are the primary user interface; every workflow starts from a `DNALLMConfig` dict produced by `load_config()` (`dnallm/configuration/configs.py`)
- Model loading is a dispatch chain: special-family handlers get first crack, then generic task-type-based `Auto*` loading (`dnallm/models/model.py:748-933`)
- Everything wraps HF objects: `DNADataset` wraps `Dataset`/`DatasetDict`, `DNATrainer` wraps `Trainer`, `DNALLMforSequenceClassification` wraps `PreTrainedModel`
- Defensive compatibility: import-time patch modules no-op when their target library is absent; retry loops on model download (`dnallm/models/model.py:317`)
- Heavy use of function-local imports to keep startup light and avoid import cycles (CLI commands import `DNATrainer`/`DNAInference` inside the command body, `dnallm/cli/cli.py`)
- Vendor directories (`dnallm/tasks/metrics/`, `dnallm/models/special/enformer_model/`) are excluded from ruff/mypy — treat as upstream code

## Layers

**Entry Layer:**
- Purpose: User-facing commands; argument parsing; wiring config → core classes
- Location: `dnallm/cli/`, `dnallm/mcp/server.py:main` (line 1928), `ui/`
- Contains: Click commands, argparse mains, Gradio apps
- Depends on: configuration, finetune, inference, models, mcp
- Used by: console scripts in `pyproject.toml` `[project.scripts]` (lines 263-269)

**Configuration Layer:**
- Purpose: Validate and structure YAML into typed config objects
- Location: `dnallm/configuration/configs.py`
- Contains: Pydantic `BaseModel` config classes (HeadConfig, TaskConfig, TrainingConfig, LoraConfig, InferenceConfig, BenchmarkConfig, ...); `DNALLMConfig` TypedDict (line 495); `load_config()` (line 513); Evo architecture YAMLs in `dnallm/configuration/evo/`
- Depends on: pydantic, pyyaml only (no internal deps)
- Used by: every layer above it

**Models Layer:**
- Purpose: Registry + loading of DNA LLMs; task heads; losses; tokenizer fallback
- Location: `dnallm/models/`
- Contains: `model.py` (loader, 1111 lines), `modeling_auto.py` (registry), `head.py`, `tokenizer.py`, `losses.py`, `special/` (family handlers), `model_info.yaml`
- Depends on: transformers, huggingface_hub, modelscope (lazy), torch, peft (lazy), configuration
- Used by: finetune, inference, mcp, cli

**Data Layer:**
- Purpose: Load, tokenize, augment, split DNA sequence datasets
- Location: `dnallm/datahandling/` (`data.py` 1785 lines, `dataset_auto.py`)
- Depends on: datasets, pandas, transformers tokenizer, utils
- Used by: finetune, inference (via `generate_dataset`), tests

**Fine-tune Layer:**
- Purpose: Training orchestration
- Location: `dnallm/finetune/trainer.py` (+ standalone `megatron.py`)
- Depends on: transformers Trainer, peft, optuna (optional), datahandling, tasks/metrics
- Used by: cli, notebooks

**Inference Layer:**
- Purpose: Prediction, interpretation, mutagenesis, benchmarking, plotting
- Location: `dnallm/inference/` (`inference.py` 2149 lines)
- Depends on: models, datahandling, configuration, tasks/metrics, captum, altair
- Used by: cli, mcp ModelManager, ui, notebooks

**MCP Layer:**
- Purpose: Expose models as MCP tools for LLM clients with streaming
- Location: `dnallm/mcp/` (`server.py` 2113 lines); own test suite in `dnallm/mcp/tests/`
- Depends on: mcp (FastMCP), starlette/uvicorn, inference, models, configuration
- Used by: `dnallm-mcp-server` console script, external MCP clients

**Utils Layer:**
- Purpose: Logging, sequence biology helpers, genomic-coordinate normalization, hardware/library compat patches, plots
- Location: `dnallm/utils/`
- Depends on: loguru, torch (for shims); pyfastx only lazily inside `fetch_sequence`
- Used by: all layers; shims execute at package import via `dnallm/utils/__init__.py`

**Test/QA Infrastructure (outside the shipped package):**
- Purpose: Suite census, example execution, skip accounting, model pinning
- Location: `tests/` (mirrors package layout), `tests/examples/_execution.py` (1134-line private harness), `scripts/audit_skips.py`, `tests/expected_skips.yaml`, `models.lock`
- Depends on: pytest, nbclient, nbformat
- Used by: `.github/workflows/ci.yml` jobs (test, coverage-gate, coverage-nightly, example-nightly, mamba-nightly)

## Data Flow

### Primary Request Path (Training)

1. CLI `train` command parses `--config/-c` (`dnallm/cli/cli.py:45`)
2. `load_config(path)` → `DNALLMConfig` with `task`/`model`/`finetune`/`lora` sections (`dnallm/configuration/configs.py:513`)
3. `DNADataset` loads + tokenizes data (`dnallm/datahandling/data.py`)
4. `load_model_and_tokenizer(model_name, task_config, source)` resolves model + tokenizer (`dnallm/models/model.py:748`)
5. `DNATrainer(dataset, model, tokenizer, config).set_up_trainer()` builds HF `Trainer` with `compute_task_metrics` (`dnallm/finetune/trainer.py:177`)
6. `.train()` runs the HF loop; `.plot_history()` renders training plots

### Inference Path

1. `load_config` → `DNAInference(model, tokenizer, config)` (`dnallm/inference/inference.py:79`)
2. `infer`/`infer_seqs`/`infer_file` dispatch to `batch_infer` (`inference.py:928`)
3. `generate_dataset` tokenizes inputs; `_get_accepted_forward_args` introspects the backbone's forward signature (`inference.py:375`)
4. `logits_to_preds` → `format_output` → `save_predictions`/`save_metrics` (`inference.py:523,576,2118,2135`)

### MCP Request Path

1. `dnallm-mcp-server --config ... --transport stdio|sse|streamable-http` → `main()` (`dnallm/mcp/server.py:1928`)
2. `MCPConfigManager` parses server + model configs (`dnallm/mcp/config_manager.py`)
3. `ModelManager.load_model(name)` loads `DNAInference` in an executor thread (`dnallm/mcp/model_manager.py:121` area — `_load_model_sync`)
4. Tool call → `_with_timeout_wrapper` (`dnallm/mcp/server.py:282` area, defined at line 265 region) → predict under `_infer_thread_lock` single-flight → error dict on failure (never raises across the protocol boundary)

### Benchmark Path

1. Config with `benchmark` section → `BenchmarkConfig` → `Benchmark` (`dnallm/inference/benchmark.py:30`)
2. Iterates models × datasets, calling `load_model_and_tokenizer` + `DNAInference` per cell; results collected per `EvaluationConfig`/`OutputConfig`

### Example Execution Census (nightly)

1. `tests/examples/test_notebook_execution.py` walks its `ACTIVE_NOTEBOOKS` list (line 74)
2. `tests/examples/_execution.py` `seed_sandbox` copies `example/` into pytest `tmp_path`; `run_notebook` executes via nbclient with per-cell timeouts from `NOTEBOOK_EXEC_SPECS`
3. Isolated kernel lanes (`dnallm-mcp-langchain`, `dnallm-megadna`, `dnallm-evo-kernel`) pin `VIRTUAL_ENV` to throwaway `.scratch/` venvs so example install cells never touch the project venv
4. `assert_tree_clean` proves execution never dirtied `example/` or `docs/example/`
5. junit XML → `scripts/audit_skips.py pytest-junit.xml tests/expected_skips.yaml` fails CI on any skip whose message is not allowlisted (exact/prefix/reason_like matchers)
6. Remote artifacts keyed by `models.lock` (`<hf|ms>  <repo-id>@<revision-sha>` pins) feed the `actions/cache` model cache

**State Management:**
- Core library is stateless per workflow: config dict + class instances; no module-level mutable singletons in core layers
- `ModelManager` (MCP) is the intentional long-lived state holder: `loaded_models: dict[str, DNAInference]` + status map + `asyncio.Lock` for loading and `threading.Lock` (`_infer_thread_lock`) for single-flight inference spanning worker-thread lifetime (`dnallm/mcp/model_manager.py:30-56`)
- Gradio app keeps engine in module-global `GLOBAL_STATE` dict (`ui/generation_task_app.py:10`)
- Compat shims mutate process-wide library behavior once at import (`dnallm/utils/__init__.py` imports `cuda_compat` and `transformers_compat` eagerly)

## Key Abstractions

**Config-dict convention (Mapping-typed):**
- Purpose: All core classes accept `config: Mapping[str, Any]` with keys `task`, `inference`/`finetune`, `lora`, `model`; access via `config["task"]`, `config["inference"]`
- Examples: `dnallm/inference/inference.py:83`, `dnallm/finetune/trainer.py:130`, `dnallm/inference/interpret.py:114`, `dnallm/inference/mutagenesis.py:50`, `dnallm/inference/benchmark.py:52`
- Pattern: `DNALLMConfig` is a `TypedDict(total=False)` so `Mapping[str, Any]` parameter types keep TypedDict assignability sound; `model` section deliberately stays a plain dict

**Special-family handler chain:**
- Purpose: Isolate per-model-family loading quirks behind a uniform `_handle_<family>_models(...) -> tuple | None` contract; returning `None` falls through to generic loading
- Examples: `dnallm/models/special/evo.py` (`_handle_evo1_models`, `_handle_evo2_models`), `dnallm/models/special/enformer.py`, `dnallm/models/special/dnabert2.py`
- Pattern: chain-of-responsibility dispatched from `load_model_and_tokenizer` (`dnallm/models/model.py:796-858`); late stages (crossdna → dnabert2 → `_load_model_by_task_type`) are a guarded chain where each stage runs only when the previous left model or tokenizer None

**Head-on-backbone composition:**
- Purpose: Bolt customizable prediction heads onto any backbone
- Examples: `DNALLMforSequenceClassification` (`dnallm/models/model.py:49`) + heads in `dnallm/models/head.py`; head type selected by suffix matching in `_determine_classifier` (`model.py:132`)
- Pattern: config-driven composition; pooling strategy auto-inferred (`_determine_pooling_strategy`, `model.py:149`)

**Reflection-based capability detection:**
- Purpose: Uniform inference over heterogeneous backbones (HF models, EVO, mamba with fp32/CUDA-only constraints)
- Examples: `_get_accepted_forward_args` (`dnallm/inference/inference.py:375`), attention/hidden-states support probes (`_check_attention_support`, `_check_hidden_states_support`)
- Pattern: introspection instead of per-model subclasses

**Metrics factory:**
- Purpose: One `compute_metrics(task_type)` factory used as the HF Trainer `compute_metrics` callback
- Examples: `dnallm/tasks/metrics.py`; vendored packages `dnallm/tasks/metrics/<name>/{app.py,<name>.py}`
- Pattern: sklearn + vendored HF evaluate integration behind a task-type switch

**Coordinate normalization seam:**
- Purpose: Single module converting between the three genomics conventions (0-based half-open BED ↔ 1-based closed GFF3/FASTA; TAIR `Chr1` ↔ Ensembl `1` chromosome names)
- Examples: `dnallm/utils/genomic_coords.py` (`normalize_chrom`, `gff1_to_half_open`, `half_open_to_gff1`, `fetch_sequence`, `parse_gff_attributes`, `slice_gff_rows`)
- Pattern: validate-everything, raise `ValueError` on undocumented forms — silent-empty results are structurally impossible; consumed by the PlantHelixSeek showcase notebooks' shared data scripts (`example/notebooks/plant_helixseek_shared/`)

**Private test-harness seam:**
- Purpose: `_`-prefixed helper module living beside its consumer tests, never imported by root conftest or `dnallm/`, never shipped in the wheel
- Examples: `tests/examples/_execution.py`, `dnallm/mcp/tests/_network_skip.py`
- Pattern: harness + specs + typed-skip helpers with stable junit-greppable prefixes registered in `tests/expected_skips.yaml`

## Entry Points

**Console scripts (`pyproject.toml` `[project.scripts]`, lines 263-269):**
- `dnallm` → `dnallm.cli.cli:cli` — Click group (train, inference, benchmark, mutagenesis, model_config_generator, mcp_server)
- `dnallm-train` → `dnallm.cli.train:main`
- `dnallm-inference` → `dnallm.cli.inference:main`
- `dnallm-model-config-generator` → `dnallm.cli.model_config_generator:main` (interactive YAML builder; examples in `cli/examples/`)
- `dnallm-mcp-server` → `dnallm.mcp.server:main` (argparse: `--config`, `--host`, `--port`, `--transport stdio|sse|streamable-http`)
- `dnallm-mutagenesis` → `dnallm.cli.mutagenesis:main`

**Python API:**
- `from dnallm import DNATrainer, DNAInference, DNADataset, DNAInterpret, Mutagenesis, Benchmark, load_config, load_model_and_tokenizer, get_logger, setup_logging, cli` (`dnallm/__init__.py`)

**Legacy shims (backward compat only — do not extend):**
- `run_cli.py` → root `cli/cli.py` shims delegating into `dnallm.cli`

**UI apps:**
- `ui/generation_task_app.py` (interactive generation), `ui/model_config_generator_app.py` + `ui/run_config_app.py` (config web UI)

**Out-of-band scripts:**
- `scripts/finetune_mamba2_megatron.py`, `scripts/infer_mamba2_megatron.py`, `scripts/inference_mamba2_npu.py` (Megatron / Ascend NPU paths outside the HF Trainer flow)
- `scripts/audit_skips.py` (skip gate), docs-validation scripts (`scripts/check_docs_sync.py` etc.)

## Architectural Constraints

- **Threading:** Core library is synchronous PyTorch. The MCP layer is asyncio (FastMCP + uvicorn); blocking model loads bridge via `ModelManager._load_model_sync` in an executor (`dnallm/mcp/model_manager.py`). MCP tools are wrapped in timeout guards (`_with_timeout_wrapper`, `dnallm/mcp/server.py`). All predict traffic serializes through `ModelManager._infer_thread_lock` (a `threading.Lock` acquired inside the executor callable) because DataLoader worker forks plus hub-cache filelocks are fork-unsafe — single-flight spans the worker-thread lifetime, not the coroutine lifetime.
- **Global state:** Import-time monkey patches in `dnallm/utils/cuda_compat.py` (RTLD_GLOBAL library preload) and `dnallm/utils/transformers_compat.py` (bitsandbytes init fix) apply process-wide because `dnallm/utils/__init__.py` imports them eagerly. MCP `ModelManager` and `ui/generation_task_app.py GLOBAL_STATE` hold live models.
- **Device heterogeneity:** Device auto-selection order CUDA → MPS → XPU → CPU (`dnallm/models/model.py:667` area, `_get_device`); mamba models force CUDA/CPU + fp32; Ascend NPU paths live in `dnallm/models/special/mamba_npu.py` and `scripts/inference_mamba2_npu.py` (both excluded from lint/mypy).
- **Circular imports:** Avoided via function-local imports throughout. No known cycle, but `dnallm/__init__.py` imports every subpackage, so submodules must not import the package root at module level.
- **Dependency optionality:** modelscope, optuna, captum, bitsandbytes, megatron, gradio, mcp/langchain, pyfastx are optional/lazy — always guard imports with try/except or import inside functions (existing pattern in `dnallm/finetune/trainer.py:46-49`, `dnallm/models/model.py:421-448`, `dnallm/utils/genomic_coords.py:160`).
- **Transformers version span:** Must work across transformers 4.49–5.x; compat fixes concentrate in `dnallm/utils/transformers_compat.py` and fallback paths in `dnallm/models/model.py:543-568`.
- **Lint/mypy exclusions:** `dnallm/tasks/metrics/`, `dnallm/models/special/mamba_npu.py`, `dnallm/finetune/megatron.py`, `example/`, `.planning/` are vendored/vendor-style — do not hold them to project conventions (see `[tool.ruff] exclude` in `pyproject.toml`).
- **Skip discipline:** Any new skip path in tests must emit a typed message (stable prefix) and be registered in `tests/expected_skips.yaml`, or `scripts/audit_skips.py` fails CI. Cell errors/timeouts in example execution are always re-raised, never converted to skips.
- **Runner isolation:** Example kernels that install packages run under dedicated kernelspecs pinning `VIRTUAL_ENV` to gitignored `.scratch/` venvs (`tests/examples/_execution.py:80-106`); the nightly runner's ollama service is loopback-only by design (`scripts/runner/ollama.service` — binding IS the access control).

## Anti-Patterns

### New code in root `cli/` instead of `dnallm/cli/`

**What happens:** Root `cli/cli.py` and `run_cli.py` look like the CLI but are backward-compat shims that `sys.path`-hack the project root.
**Why it's wrong:** The shims are not packaged in the wheel; code added there is unreachable for installed users.
**Do this instead:** Add commands to the Click group in `dnallm/cli/cli.py` with lazy imports of core classes.

### Overwriting handler results in the load chain

**What happens:** Returning early from a mid-chain special handler skips tokenizer post-processing (MutBERT/Basenji2 tokenizers, `_configure_model_padding`, device placement).
**Why it's wrong:** A resolved handler result still needs the shared post-processing in `load_model_and_tokenizer`.
**Do this instead:** Follow the guarded-dispatch pattern at `dnallm/models/model.py:872-883`: each stage runs only when the previous left `model` or `tokenizer` None, and the chain falls through to shared post-processing.

### Hand-editing vendored metric code

**What happens:** `dnallm/tasks/metrics/` mirrors upstream HF `evaluate` implementations.
**Why it's wrong:** Local edits make upstream diffs impossible to track and are invisible to ruff/mypy.
**Do this instead:** Add new metric behavior in `dnallm/tasks/metrics.py` dispatch code; vendor a new metric as `dnallm/tasks/metrics/<name>/{app.py,<name>.py}` verbatim.

### Silent-empty returns in genomics data paths

**What happens:** Hand-scattered coordinate/name conversion code historically returned empty results or off-by-one bases instead of erroring.
**Why it's wrong:** Empty slices propagate as bogus downstream data.
**Do this instead:** Route every coordinate/chromosome conversion through `dnallm/utils/genomic_coords.py`, which validates inputs and raises `ValueError` on undocumented forms (including `require_nonempty=True` for locus slicing).

### Skipping instead of failing in execution tests

**What happens:** Broad `pytest.skip` on flaky infrastructure hides real breakage and trips the skip audit.
**Why it's wrong:** The nightly census's value is honest pass/fail signal; unregistered skips fail CI anyway.
**Do this instead:** Use the typed helpers in `tests/examples/_execution.py` (`environment_unavailable_skip`, `optional_dep_skip`, `network_unavailable_skip`) and register the message prefix in `tests/expected_skips.yaml`.

## Error Handling

**Strategy:** Fail loud at boundaries, degrade gracefully for optional dependencies.

**Patterns:**
- Retry loop with reason classification for model downloads (`dnallm/models/model.py:317-377`, `download_model(..., max_try=3)`)
- Try/except fallback chains: AutoTokenizer → PreTrainedTokenizerFast → DNAOneHotTokenizer (`dnallm/models/model.py:543-568`)
- `raise ValueError(f"Failed to load model: {e}") from e` wrapping at the `load_model_and_tokenizer` boundary (`dnallm/models/model.py:896` area)
- Pydantic field validators reject invalid config early in `dnallm/configuration/configs.py`
- MCP tools return error dicts rather than raising across the protocol boundary; every tool call is timeout-wrapped (`dnallm/mcp/server.py:_with_timeout_wrapper`)
- Compat shims no-op on absent libraries so import never breaks (`dnallm/utils/transformers_compat.py`, `dnallm/utils/cuda_compat.py`)
- `genomic_coords` validates every input and raises `ValueError` — never a silent empty result (`dnallm/utils/genomic_coords.py`)

## Cross-Cutting Concerns

**Logging:** loguru-backed `get_logger("dnallm.<module>")` singleton via `dnallm/utils/logger.py`; convenience functions `log_info`/`log_error`/`log_success`; MCP server adds structured logging (`_structured_log`, `dnallm/mcp/server.py`).
**Validation:** Pydantic v2 `Field(pattern=...)`, `field_validator`, `model_validator`, `model_post_init` at config-load time (`dnallm/configuration/configs.py`); input re-validation in `genomic_coords`.
**Authentication:** None in package code; no env-var auth; HuggingFace mirror toggle via `HF_ENDPOINT` set programmatically (`dnallm/models/model.py:380-391`).
**Testing:** pytest with `testpaths = ["tests", "dnallm/mcp/tests"]`, strict markers (`slow` et al.), 300s default timeout, asyncio auto mode (`pyproject.toml` `[tool.pytest.ini_options]`); root `conftest.py` handles global cleanup (multiprocessing + torch resources).
**CI enforcement:** ruff + mypy advisory + fast census + skip audit on PRs; coverage gate on fast leg; full slow census + model-cache restore keyed on `models.lock` on the self-hosted nightly; example execution census in its own `example-nightly` job with staged-serial ordering (stage 0 install/probes → 1 torch-heavy notebooks → 1.5 kernel pkill/VRAM settle → 2 MCP live-server probes on :8000 → 2.5 settle → 3 ollama batch → 4 junit skip audit + artifacts) in `.github/workflows/ci.yml`.

---

*Architecture analysis: 2026-10-05*
