---
phase: 07-docs-examples-sync
research: true
---

# Phase 07 Research: docs-examples-sync

## Research Goal

Identify all inconsistencies between `docs/` and `example/` and the current `dnallm` codebase, establish the pattern-based traversal strategy for systematic fixes, and determine what validation infrastructure is needed.

## Methodology

Pattern-based traversal of the codebase following the dependency chain:
1. Source code (`dnallm/`) — actual APIs, signatures, configs
2. Example configs (`example/**/*.yaml`) — validate against Pydantic models
3. Example code (`example/**/*.py`, `*.ipynb`) — imports, signatures, config keys
4. Documentation (`docs/**/*.md`) — must match examples and source
5. API docs (`docs/api/**/*.md`) — mkdocstrings configuration
6. CI/validation — automated checks

## Key Findings

### 1. benchmark.py Critical Bug (REQ-01)

**File:** `dnallm/inference/benchmark.py`
**Lines:** 93, 104, 142, 150
**Issue:** Assigns `InferenceConfig` and `TaskConfig` **classes** instead of instances.

```python
# Line 93 (BROKEN)
self.inference_config = InferenceConfig  # Class, not instance!
# Line 104 (BROKEN)
self.task_config = TaskConfig  # Class, not instance!
```

Same pattern at lines 142, 150 in `run()` method.

**Impact:** All benchmark usage breaks with `TypeError` or attribute errors because code expects instances with attributes, not classes.

**Fix:** Change to `InferenceConfig()` and `TaskConfig()` (instantiate).

### 2. Example YAML Configs (REQ-02)

**21 YAML files under `example/`** — all must validate via `load_config()`.

**Issues found:**
- `example/notebooks/benchmark/benchmark_config.yaml` line 15: typo `"postive"` (should be `"positive"`)
- `example/marimo/benchmark/config.yaml`: lacks `benchmark:` section but loads OK as non-benchmark config

**Validation:** All 21 YAML files currently pass `load_config()` — the typo does not cause a validation error (it's a string value, not a schema violation).

### 3. Example Notebooks (REQ-03)

**18 notebooks under `example/notebooks/`**

**Issues found:**
- `example/notebooks/data_prepare/predict/predict_data.ipynb`: uses `data_path=` parameter — actual API uses `file_path=`
- `example/notebooks/finetune_generation/finetune_generation.ipynb`: compatibility issues with current code (needs review)
- `example/notebooks/benchmark/benchmark.ipynb`: depends on broken `Benchmark` class (fixed in REQ-01)

### 4. Marimo Examples (REQ-04)

**3 marimo demo files under `example/marimo/`**

**Issues found:**
- `example/marimo/inference/inference_demo.py`: typo `Tokenzier` (2 occurrences, lines 61, 137) — should be `Tokenizer`
- `example/marimo/finetune/finetune_demo.py`: passes `task=configs['task'].task_type` to `encode_sequences` — `task` parameter expects `"SequenceClassification"` style string, not task_type like `"binary"`
- `example/marimo/benchmark/benchmark_demo.py`: depends on broken `Benchmark` class

### 5. MCP Examples (REQ-05)

**2 MCP client notebooks under `example/mcp_example/`**

**Issues found:**
- `mcp_client_ollama_pydantic_ai.ipynb`: uses `_dna_multi_model_predict` (with underscore prefix) — actual exposed tool name is `dna_multi_model_predict` (NO underscore)
- `mcp_client_ollama_langchain_agents.ipynb`: uses `"transport": "streamable_http"` — actual CLI accepts `"streamable-http"` (hyphen, not underscore)

**Source of truth for tool names:** `dnallm/mcp/server.py` lines 250-267:
```python
self._with_timeout_wrapper(self._dna_sequence_predict, "dna_sequence_predict")
# Second arg is the EXPOSED name — no underscore prefix
```

### 6. API Documentation Structure (REQ-06)

**10 API doc files lack `show_root_heading: true`**:
- `docs/api/finetune/trainer.md`
- `docs/api/datahandling/data.md`
- `docs/api/inference/mutagenesis.md`
- `docs/api/inference/inference.md`
- `docs/api/inference/plot.md`
- `docs/api/inference/benchmark.md`
- `docs/api/utils/sequence.md`
- `docs/api/mcp/server.md`
- `docs/api/index.md`
- `docs/api/configuration/configs.md` (has it already — use as reference)

**4 source modules have no API docs** (may be out of scope for this phase):
- `dnallm/inference/ensemble.py`
- `dnallm/inference/interpret.py`
- `dnallm/datahandling/dataset.py`
- `dnallm/utils/loss.py`

**TaskType enum mismatch:** `TaskType.BINARY == "binary"` but docs reference `"binary_classification"` in some places.

### 7. User Guide Import Paths (REQ-07)

**~20 files** document `from dnallm import load_model_and_tokenizer`.

**Reality check:** `dnallm/__init__.py` DOES export `load_model_and_tokenizer`:
```python
from .models import load_model_and_tokenizer
__all__ = [..., "load_model_and_tokenizer", ...]
```

**Verdict:** These imports ARE valid. However, for consistency with the pattern of using submodule imports (`from dnallm.models import...`), and per SPEC acceptance criteria, we will standardize on submodule imports.

**Files affected:** 17 occurrences across docs/ and example/ notebooks.

### 8. User Guide API Signatures (REQ-08)

**DNATrainer signature (actual):**
```python
def __init__(self, model, config, datasets=None, extra_args=None, use_lora=False):
```
**Wrong in docs:** `DNATrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, config=config)` — `tokenizer` and `train_dataset` params don't exist.

**Mutagenesis signature (actual):**
```python
def __init__(self, model, tokenizer, config):
```
**Status:** Mostly correct in docs.

**DNAInference signature (actual):**
```python
def __init__(self, model, tokenizer, config, lora_adapter=None, **kwargs):
```
**Status:** Mostly correct in docs.

### 9. User Guide Config Keys (REQ-09)

**Top-level config key:** `finetune:` (not `training:` or `training_args:`)

**Invalid keys found in docs:**
- `greater_is_better`: 13 files — NOT in TrainingConfig
- `early_stopping_patience`: 2 files — NOT in TrainingConfig
- `gradient_checkpointing`: 3 files — NOT in TrainingConfig

**Task type values:** Use `"binary"`, `"multiclass"`, `"token"` (short names) — not `"sequence_classification"`, `"token_classification"`.

### 10. User Guide CLI Commands (REQ-10)

**pyproject.toml entry points:**
```toml
[project.scripts]
dnallm-train = "dnallm.cli.cli:train_cli"
dnallm-inference = "dnallm.cli.cli:inference_cli"
dnallm-mcp-server = "dnallm.cli.cli:mcp_server_cli"
```

**AND subcommands:**
```python
# dnallm/cli/cli.py
def main():
    app = typer.Typer()
    app.command("train")(train_cli)
    app.command("inference")(inference_cli)
    app.command("mcp-server")(mcp_server_cli)
```

**Verdict:** `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` ARE valid standalone scripts. `dnallm-benchmark` is NOT a valid standalone script (only available as `dnallm benchmark` subcommand).

**Fix scope:** Only fix `dnallm-benchmark ` references (3 files).

### 11. MCP Documentation (REQ-11)

**Tool names:** Exposed names have NO underscore prefix:
- `dna_sequence_predict` (not `_dna_sequence_predict`)
- `dna_batch_predict` (not `_dna_batch_predict`)
- `dna_multi_model_predict` (not `_dna_multi_model_predict`)

**Default host:** `127.0.0.1` (not `0.0.0.0`)
- `dnallm/cli/cli.py` line 297: `mcp_server` default host is `127.0.0.1`
- `dnallm/mcp/server.py` `start_server` default is `127.0.0.1`

**Fake endpoints:** `/health` and `/predict` are NOT real MCP endpoints. Remove from docs.

### 12. docs/example/ Sync and CI Validation (REQ-12)

**Current state:** `docs/example/` is a byte-identical mirror of `example/` (except runtime artifacts).
**Verification:** `filecmp.dircmp(example/, docs/example/)` shows only `__pycache__`, `logs`, `outputs` differ.

**Data file differences:** 2 data files differ (expected — they contain runtime-generated data):
- `example/notebooks/NER_task/data/...` vs `docs/example/...`
- These are runtime artifacts, not source files.

**CI gap:** No workflow validates docs code snippets or example sync.

## Pattern Counts Summary

| Pattern | Count | Files | Action |
|---------|-------|-------|--------|
| `from dnallm import load_model_and_tokenizer` | 17 | docs/ + notebooks | Change to `from dnallm.models import` |
| `from dnallm import DNADataset` | 3 | docs/ + notebooks | Change to `from dnallm.datahandling import` |
| `training:` (top-level key) | ~10 | docs/ | Change to `finetune:` |
| `training_args:` (top-level key) | ~5 | docs/ | Change to `finetune:` |
| `greater_is_better` | 13 | docs/ | Remove from config examples |
| `early_stopping_patience` | 2 | docs/ | Remove from config examples |
| `gradient_checkpointing` | 3 | docs/ | Remove from config examples |
| `dnallm-benchmark ` | 3 | docs/ | Change to `dnallm benchmark ` |
| `dnallm-train `, `dnallm-inference `, `dnallm-mcp-server ` | Valid | — | DO NOT FIX |
| `0.0.0.0` in MCP docs | ~8 | docs/ | Change to `127.0.0.1` |
| `/health`, `/predict` | 2 | docs/ | Remove |
| Missing `show_root_heading` | 9 | docs/api/ | Add `show_root_heading: true` |
| `Tokenzier` typo | 2 | example/marimo/ | Fix to `Tokenizer` |
| `postive` typo | 1 | example/ | Fix to `positive` |
| `_dna_multi_model_predict` | 1 | example/mcp/ | Remove underscore |
| `streamable_http` | 1 | example/mcp/ | Change to `streamable-http` |
| `data_path=` | 1 | example/notebooks/ | Change to `file_path=` |

## Validation Strategy

### Scripts Needed
1. `scripts/validate_yaml.py` — iterate all `.yaml` under `example/`, call `load_config()`, report failures
2. `scripts/check_docs_sync.py` — verify `docs/example/` mirrors `example/` (ignore runtime artifacts)
3. `scripts/validate_docs_snippets.py` — extract Python code blocks from `.md` files, run `ast.parse()`

### CI Workflow
- Trigger: PR to main/master/dev
- Steps: sync check → snippet validation → YAML validation → example tests
- Runtime target: < 2 minutes
- Status: informational (not blocking) per D-01

## Dependency Chain for Fixes

```
REQ-01 (benchmark.py bug)
  → unblocks REQ-03 (notebooks), REQ-04 (marimo), REQ-06 (API docs)

REQ-02 (YAML configs)
  → standalone, validates config schema

REQ-03 (notebooks)
  → depends on REQ-01 for benchmark.ipynb

REQ-04 (marimo)
  → depends on REQ-01 for benchmark_demo.py

REQ-05 (MCP)
  → standalone

REQ-06 (API docs)
  → partially depends on REQ-01 for benchmark.md

REQ-07 (imports)
  → standalone

REQ-08 (signatures)
  → standalone

REQ-09 (config keys)
  → standalone

REQ-10 (CLI)
  → standalone (only dnallm-benchmark to fix)

REQ-11 (MCP docs)
  → standalone

REQ-12 (sync + CI)
  → depends on all above being fixed first
```

## Version Compatibility Matrix

| Old Pattern | New Pattern | Since | Notes |
|-------------|-------------|-------|-------|
| `from dnallm import X` | `from dnallm.models import X` | v0.5.x | Both work; standardizing on submodule |
| `training_args:` | `finetune:` | v0.5.x | Config schema changed |
| `sequence_classification` | `binary`/`multiclass` | v0.5.x | TaskConfig regex uses short names |
| `dnallm-xxx` standalone only | Both standalone AND `dnallm xxx` | v0.5.x | pyproject.toml registers both styles |
| `model_post_init` (pydantic v1 style) | `model_validator(mode="after")` (pydantic v2) | v0.5.2 | `model_post_init` still works but is deprecated; expanding regex requires updating the post-init logic |
| `training_args:` top-level key | `finetune:` top-level key | v0.5.x | Config schema changed; docs still reference old key |
| `sequence_classification` task type | `binary`/`multiclass` short names | v0.5.x | TaskConfig regex uses short names; docs still reference old names |
| `dnallm-xxx` standalone only | Both standalone AND `dnallm xxx` subcommands | v0.5.x | pyproject.toml registers both styles |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `from dnallm import load_model_and_tokenizer` is valid because `dnallm/__init__.py` exports it | REQ-07 | If `__init__.py` changes, imports break; but per D-03, we fix docs to match code |
| A2 | `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` are valid standalone scripts per `pyproject.toml` | REQ-10 | If entry points change, CLI docs break; but we fix docs to match current entry points |
| A3 | MCP tool names are exposed WITHOUT underscore prefix (e.g., `dna_sequence_predict`) | REQ-05, REQ-11 | If server registration changes, MCP docs break; verified by reading `_with_timeout_wrapper` calls |
| A4 | `docs/example/` and `example/` are currently in sync for tracked files | REQ-12 | Verified by `filecmp.dircmp` ignoring runtime artifacts; only 2 data files differ |
| A5 | All 21 YAML files under `example/` currently pass `load_config()` validation | REQ-02 | Verified by running `load_config()` on each file; 0 errors returned |

## Open Questions (RESOLVED)

1. **REQ-07 Import paths: No-op or not?** [RESOLVED]
   - Decision: Standardize on submodule imports per SPEC acceptance criteria. `from dnallm.models import load_model_and_tokenizer` and `from dnallm.datahandling import DNADataset`.
   - Rationale: Both top-level and submodule imports work, but SPEC acceptance criteria requires zero top-level imports for these names. Plan 07-03 Task 2 implements this.

2. **REQ-04 Marimo finetune_demo.py encode_sequences task parameter** [RESOLVED]
   - Decision: Runtime test during Plan 07-02 implementation; map task_type to valid task parameter value if needed.
   - Rationale: `encode_sequences` accepts `task: str | None = "SequenceClassification"` but the call passes `configs['task'].task_type` (e.g., `"binary"`). The fix will be verified at execution time in Plan 07-02 Task 3.

3. **REQ-03 Notebook functional testing scope** [RESOLVED]
   - Decision: Per D-01, CI validation is informational and static-only (imports, syntax, API signatures). Plan 07-02 may attempt local execution for core cells where feasible, but CI remains static-only.
   - Rationale: D-01 explicitly states "CI only validates imports + initialization (does not execute notebooks requiring external data/models)."

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.13.11 | — |
| pydantic | TaskConfig validation | ✓ | 2.13.4 | — |
| PyYAML | YAML config parsing | ✓ | 6.0.2 | — |
| pytest | Test framework | ✓ | 9.0.3 | — |
| mkdocs | Docs build verification | ✗ | — | Skip `mkdocs build` verification; rely on grep for `show_root_heading` |
| mkdocstrings-python | API doc generation | ✗ | — | Skip mkdocs build; verify API doc structure via grep |
| uv | Package management (CI) | ✓ | (in CI) | — |

**Missing dependencies with no fallback:**
- mkdocs / mkdocstrings-python not installed locally — cannot run `mkdocs build` for verification

**Missing dependencies with fallback:**
- mkdocs build verification can be replaced with grep-based checks for `show_root_heading` and file existence
