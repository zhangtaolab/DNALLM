---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# Codebase Structure

**Analysis Date:** 2026-10-05

## Directory Layout

```
DNALLM/
├── dnallm/                     # The installable package (everything ships from here)
│   ├── cli/                    # Click commands: cli.py (group), train, inference, mutagenesis, model_config_generator
│   ├── configuration/          # Pydantic config classes + load_config; evo/ (24 Evo architecture YAMLs)
│   ├── datahandling/           # data.py (DNADataset), dataset_auto.py (PRESET_DATASETS)
│   ├── finetune/               # trainer.py (DNATrainer), megatron.py (standalone, lint-excluded)
│   ├── inference/              # inference.py (DNAInference), interpret.py, mutagenesis.py, benchmark.py, plot.py
│   ├── mcp/                    # server.py, model_manager.py, config_manager.py, config_validators.py, client.py,
│   │                           #   configs/ (server + per-model YAMLs), tests/ (own pytest suite)
│   ├── models/                 # model.py (loader), modeling_auto.py (registry), head.py, tokenizer.py, losses.py,
│   │                           #   model_info.yaml (219-model registry), special/ (family handlers + enformer_model/ vendored)
│   ├── tasks/                  # task.py (TaskType/TaskConfig/TaskHead), metrics.py (dispatcher), metrics/ (vendored HF evaluate)
│   ├── utils/                  # logger.py, sequence.py, genomic_coords.py, cuda_compat.py, transformers_compat.py,
│   │                           #   support.py (FA/FP8 capability), training_plots.py
│   └── version.py, __init__.py # facade re-exports
├── tests/                      # Pytest suite mirroring the package layout
│   ├── examples/               # _execution.py (private harness) + example census tests
│   ├── mcp/                    # MCP server/client tests (complement dnallm/mcp/tests/)
│   ├── models/                 # incl. test_special/, PlantHelixSeek registry/fla/smoke tests
│   ├── test_data/              # fixture datasets per task type (csv + labels.txt)
│   ├── expected_skips.yaml     # skip-audit allowlist (CI gate)
│   └── TESTING.md              # suite documentation
├── example/                    # notebooks/, marimo/, mcp_example/ — execution-census targets
├── docs/                       # MkDocs Markdown: api/, concepts/, example/, faq/, getting_started/, user_guide/, notes/
├── configs/                    # User-facing example YAMLs (finetune, inference, lora, ner)
├── scripts/                    # Dev/CI scripts; runner/ (ollama.service + README); feasibility/
├── ui/                         # Gradio apps (generation_task_app, model_config_generator_app, run_config_app)
├── cli/                        # LEGACY root CLI shims — backward compat only, do not extend
├── .github/workflows/          # ci.yml, docs-validation.yml, feasibility.yml, publish.yml
├── .planning/                  # GSD artifacts (codebase/, phases/, milestones/, graphs/)
├── models.lock                 # Remote-artifact pins; cache key for slow/nightly CI jobs
├── pyproject.toml              # Single source of truth: deps, extras, ruff, mypy, pytest
├── conftest.py                 # Root pytest hooks (global cleanup)
├── run_cli.py                  # Legacy launcher shim
├── setup.py                    # Backward-compat stub
├── mkdocs.yml                  # Docs site config
└── install_deps.sh             # Dependency bootstrap helper
```

## Directory Purposes

**`dnallm/`:**
- Purpose: The shipped package; console scripts and Python API all resolve here
- Contains: 8 subpackages (`cli`, `configuration`, `datahandling`, `finetune`, `inference`, `mcp`, `models`, `tasks`, `utils`)
- Key files: `dnallm/__init__.py` (facade `__all__`), `dnallm/models/model.py` (1111 lines), `dnallm/inference/inference.py` (2149), `dnallm/mcp/server.py` (2113), `dnallm/datahandling/data.py` (1785), `dnallm/configuration/configs.py` (550)

**`dnallm/models/special/`:**
- Purpose: Per-family model loading quirks, one module per family
- Contains: `evo.py`, `dnabert2.py`, `gpn.py`, `megadna.py`, `enformer.py`, `space.py`, `borzoi.py`, `mutbert.py`, `basenji2.py`, `crossdna.py`, `omnidna.py`, `lucaone.py`, `mamba_npu.py` (lint-excluded), `enformer_model/` (vendored ported Enformer/SPACE modeling code)

**`dnallm/configuration/evo/`:**
- Purpose: Evo 1/2 architecture YAMLs selected at load time by flash-attention/FP8 capability (via `dnallm/utils/support.py`)
- Contains: 24 files named `evo[1|2]-<size>-<ctx>[-noFA][-noFP8].yml`

**`dnallm/mcp/configs/`:**
- Purpose: MCP server runtime configuration
- Contains: `mcp_server_config.yaml` (host/port/transport/model registry/SSE), `mcp_server_config_2.yaml`, per-model `*_inference_config.yaml` (promoter, conservation, open_chromatin, h3k27ac, h3k27me3)

**`tests/`:**
- Purpose: Main pytest suite; mirrors `dnallm/` subpackage layout (`tests/models/`, `tests/inference/`, `tests/mcp/`, ...)
- Contains: `test_<module>.py` files; `_real_model.py` suffix for real-model integration tests; `tests/examples/_execution.py` private harness (1134 lines); `expected_skips.yaml` allowlist; `TESTING.md`

**`tests/examples/`:**
- Purpose: Example-artifact census: static content checks (`test_examples.py`), notebook execution (`test_notebook_execution.py` with `ACTIVE_NOTEBOOKS` at line 74), marimo execution (`test_marimo_execution.py`), example scripts (`test_script_execution.py`), PlantHelixSeek showcase lane (`test_plant_helixseek_showcase.py` — never joins `ACTIVE_NOTEBOOKS`)
- Contains: `_execution.py` (NOTEBOOK_EXEC_SPECS, MARIMO_EXEC_SPECS, seed_sandbox, run_notebook, kernel-lane provisioning, typed-skip helpers, assert_tree_clean)

**`example/`:**
- Purpose: User-facing example notebooks, marimo apps, and MCP client examples — the execution-census targets
- Contains: `notebooks/` (topic dirs: finetune_*, generation*, inference*, embedding_attention, interpretation, in_silico_mutagenesis, lora_finetune_inference, benchmark, data_prepare, inference_for_tRNA, plant_helixseek_{anno,cre,shared}), `marimo/`, `mcp_example/` (LLM client notebooks hitting loopback ollama)

**`scripts/`:**
- Purpose: Development, CI, and docs tooling
- Contains: `audit_skips.py` (skip gate), `check_code.py`/`.sh` (ruff+mypy wrapper), docs validators (`check_docs_sync.py`, `validate_docs_snippets.py`, `verify_docs.py`, `check_docs_code.py`, `check_notebook_md_sync.py`), notebook→md generators, megatron/NPU scripts, `install_mamba.sh`, `setup_uv.sh`, `publish.sh`, `ci_checks.sh`
- Key files: `scripts/runner/ollama.service` + `scripts/runner/README.md` (self-hosted runner's loopback ollama service, in-repo for auditability); `scripts/feasibility/spike_families.py`

**`ui/`:**
- Purpose: Gradio web apps
- Contains: `generation_task_app.py` (module-global `GLOBAL_STATE` engine), `model_config_generator_app.py`, `run_config_app.py`, `requirements.txt`, `inference_config.yaml`

**`docs/`:**
- Purpose: MkDocs Material site (published to GitHub Pages)
- Contains: `api/`, `concepts/`, `example/` (mirrors `example/` markdown exports), `faq/`, `getting_started/`, `user_guide/`, `notes/`, `pic/`, `resources/`, `index.md`

**`configs/`:**
- Purpose: User-facing example configuration files referenced by README/docs
- Contains: `finetune_config.yaml`, `inference_config.yaml`, `lora_config.yaml`, `ner_config.yaml`

**`cli/` (root):**
- Purpose: Backward-compat shims delegating into `dnallm.cli` — NOT part of the wheel
- Contains: `cli.py`, `train.py`, `inference.py`, `model_config_generator.py`, `examples/`

**`.github/workflows/`:**
- Purpose: CI/CD pipelines
- Contains: `ci.yml` (test matrix py3.11/3.12/3.13 × numpy 1.26.4/2.2.0, test-windows, gpu tests, mamba-nightly, coverage-gate, coverage-nightly, example-nightly with staged-serial D-07 ordering), `docs-validation.yml`, `feasibility.yml` (dispatch-only self-hosted), `publish.yml` (PyPI release)

## Key File Locations

**Entry Points:**
- `dnallm/cli/cli.py`: Click group `cli` + subcommands (train, inference, benchmark, mutagenesis, model_config_generator, mcp_server)
- `dnallm/mcp/server.py:1928`: `main()` for `dnallm-mcp-server`
- `dnallm/__init__.py`: Python API facade (`__all__` with 12 exports)
- `run_cli.py` / `cli/cli.py`: legacy launchers (do not extend)
- `ui/run_config_app.py`, `ui/generation_task_app.py`: Gradio app entries

**Configuration:**
- `pyproject.toml`: deps/extras, `[project.scripts]` (line 263), `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` (line 498), `[tool.coverage.run]` omit list
- `dnallm/configuration/configs.py`: all Pydantic config classes + `load_config` (line 513) + `DNALLMConfig` TypedDict (line 495)
- `dnallm/models/model_info.yaml`: registry of 219 pre-trained models (name/model repo/task_type)
- `dnallm/mcp/configs/mcp_server_config.yaml`: MCP server config
- `models.lock`: remote-artifact pins (`<hf|ms>  <repo-id>@<sha>`) keying the CI model cache
- `tests/expected_skips.yaml`: skip-audit allowlist (exact/prefix/reason_like matchers)

**Core Logic:**
- `dnallm/models/model.py`: `load_model_and_tokenizer` (line 748), `download_model` (line 317), `DNALLMforSequenceClassification` (line 49)
- `dnallm/models/modeling_auto.py`: `PRETRAIN_MODEL_MAPS` (line 4), `MODEL_INFO` (line 43)
- `dnallm/datahandling/data.py`: `DNADataset` (line 31)
- `dnallm/finetune/trainer.py`: `DNATrainer` (line 66), `set_up_trainer` (177), `train` (381), `search` (422)
- `dnallm/inference/inference.py`: `DNAInference` (line 60), `batch_infer` (928), `generate` (1609), `scoring` (1746), `get_embeddings` (1978)
- `dnallm/tasks/metrics.py`: `compute_metrics` dispatcher
- `dnallm/utils/genomic_coords.py`: coordinate/chrom-name normalization (6 public functions)
- `dnallm/utils/__init__.py`: eager compat-shim imports + utils re-exports

**Testing:**
- `conftest.py` (root): global cleanup hooks (multiprocessing, torch resources)
- `tests/conftest.py`: shared fixtures (`SimpleDNATokenizer` deterministic tokenizer)
- `tests/examples/_execution.py`: private nbclient execution harness
- `scripts/audit_skips.py`: skip gate (tested by `tests/scripts/test_audit_skips.py`)
- `tests/TESTING.md`: suite documentation

## Naming Conventions

**Files:**
- `snake_case.py` modules: `dnallm/utils/sequence.py`, `dnallm/inference/mutagenesis.py`
- Tests: `test_<module>.py` (`tests/utils/test_sequence.py`); real-model integration tests suffixed `_real_model.py` (`tests/finetune/test_trainer_real_model.py`)
- Private helper modules prefixed `_`: `tests/examples/_execution.py`, `dnallm/mcp/tests/_network_skip.py`
- Vendored metric dirs: `dnallm/tasks/metrics/<name>/{app.py,<name>.py}`
- Notebook-collection dirs use human topic names: `example/notebooks/finetune_binary/`, `plant_helixseek_cre/`

**Directories:**
- Package subpackages are singular-domain lowercase: `models`, `datahandling`, `finetune`, `inference`
- Test dirs mirror package dirs: `tests/models/test_special/` ↔ `dnallm/models/special/`

**Code entities:**
- Classes: `PascalCase` with domain prefix — `DNAInference`, `DNADataset`, `DNATrainer`, `DNALLMMCPServer`, `DNALLMMCPClient`, `DNALLMforSequenceClassification`, `Mutagenesis`, `Benchmark`, `DNAInterpret`
- Functions/variables: `snake_case`; private with leading underscore (`_handle_evo2_models`, `_get_model_path_and_imports`)
- Test functions: `test_*` with descriptive behavior names; test classes `Test*` group one function under test
- Constants: `UPPER_SNAKE_CASE` (`PRETRAIN_MODEL_MAPS`, `PRESET_DATASETS`, `NOTEBOOK_EXEC_SPECS`, `ACTIVE_NOTEBOOKS`, `GLOBAL_STATE`)
- Special-family handlers: `_handle_<family>_models`

## Where to Add New Code

**New Feature (end-to-end workflow):**
- Config section: Pydantic `BaseModel` in `dnallm/configuration/configs.py` + new key on `DNALLMConfig` TypedDict + branch in `load_config`
- Facade class: appropriate subpackage (`dnallm/finetune/` or `dnallm/inference/`), constructor takes `config: Mapping[str, Any]`
- CLI subcommand: `@cli.command()` in `dnallm/cli/cli.py` with lazy imports in the command body
- Re-export: add to `__all__` in `dnallm/__init__.py`
- Example config: `configs/<feature>_config.yaml`
- Tests: `tests/<subpackage>/test_<module>.py`

**New model family support:**
- Handler module: `dnallm/models/special/<family>.py` exposing `_handle_<family>_models(...) -> tuple | None`
- Wire into the dispatch chain in `dnallm/models/model.py:load_model_and_tokenizer` (follow the guarded-dispatch pattern; never overwrite a resolved earlier stage)
- Registry: family entry in `PRETRAIN_MODEL_MAPS` (`dnallm/models/modeling_auto.py`) + model rows in `dnallm/models/model_info.yaml`
- Tests: `tests/models/test_special/test_<family>.py` (+ `tests/models/test_family_handlers.py`)

**New metric:**
- Dispatch logic: `dnallm/tasks/metrics.py`
- New vendored metric: `dnallm/tasks/metrics/<name>/` with `app.py` + `<name>.py` (upstream-verbatim; excluded from ruff/mypy)

**New MCP tool:**
- Implementation method on `DNALLMMCPServer` (`dnallm/mcp/server.py`), wrapped via `self._with_timeout_wrapper(self._method, "tool_name")` in `_register_tools` (line 238)
- Tests: `dnallm/mcp/tests/test_mcp_functionality.py` or `tests/mcp/`

**New utility:**
- Module: `dnallm/utils/<name>.py` + re-export in `dnallm/utils/__init__.py` `__all__`
- Tests: `tests/utils/test_<name>.py`

**New example notebook:**
- Notebook: `example/notebooks/<topic>/<name>.ipynb`
- Execution coverage: add an entry to `NOTEBOOK_EXEC_SPECS` in `tests/examples/_execution.py` (per-cell timeout + sandbox inputs) and to `ACTIVE_NOTEBOOKS` in `tests/examples/test_notebook_execution.py:74`
- Remote model pins: add/verify rows in `models.lock` (pinned form `<hf|ms>  <repo-id>@<sha>`; prefix must match the notebook's `source=` route)
- Docs mirror: generated via `scripts/generate_md_from_notebook.py` into `docs/example/`

**New test skip path:**
- Emit a typed message with a stable prefix from a helper (pattern: `tests/examples/_execution.py` skip helpers, `dnallm/mcp/tests/_network_skip.py`)
- Register the prefix/exact/reason_like entry in `tests/expected_skips.yaml` with a category — never add an empty or wildcard entry

**Scripts:**
- CI/dev tooling: `scripts/<name>.py` (+ test under `tests/scripts/` when it is a gate)

## Special Directories

**`dnallm/tasks/metrics/`:**
- Purpose: Vendored Hugging Face `evaluate` metric implementations
- Generated: No (hand-vendored, upstream-tracked)
- Committed: Yes
- Excluded from ruff/mypy — treat as upstream code

**`dnallm/models/special/enformer_model/`:**
- Purpose: Ported Enformer/SPACE modeling code
- Excluded from lint; treat as upstream

**`dnallm/finetune/megatron.py` and `dnallm/models/special/mamba_npu.py`:**
- Purpose: Standalone Megatron-LM loop and Ascend NPU paths
- Excluded from ruff/mypy

**`example/`:**
- Purpose: Example notebooks/apps/scripts — also CI execution-census targets
- Generated: No; docs mirrors in `docs/example/` ARE generated (`scripts/generate_md_from_notebook.py`, `scripts/generate_md_from_marimo.py`)
- Committed: Yes; execution tests must leave the tree clean (`assert_tree_clean`)

**`.scratch/`:**
- Purpose: Throwaway venvs for isolated kernel lanes (`.scratch/mcp-example-venvs/`, `.scratch/megadna-venvs/`, `.scratch/evo-venvs/`)
- Generated: Yes (provisioned out-of-band, never auto-installed from a test)
- Committed: No (gitignored)

**`outputs/`, `results/`, `logs/`:**
- Purpose: Run artifacts (checkpoints, benchmark results, log files)
- Generated: Yes
- Committed: No

**`.planning/`:**
- Purpose: GSD workflow artifacts (this map, phases, milestones, graphs)
- Generated: Yes (by GSD commands)
- Committed: Yes; excluded from ruff

**`dnallm.egg-info/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`:**
- Purpose: Build/tool caches
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-10-05*
