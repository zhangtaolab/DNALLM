# Phase 07: docs-examples-sync — Specification

**Created:** 2026-05-15
**Ambiguity score:** 0.16 (gate: ≤ 0.20)
**Requirements:** 12 locked

## Goal

All documentation (`docs/`) and examples (`example/`) are consistent with the current `dnallm` codebase — every code snippet, API reference, config key, import path, and CLI command documented matches the actual implementation, and automated checks prevent future drift.

## Background

The codebase recently underwent a major upgrade (type checking fixes, lint fixes, API refactoring in v0.5.2). A comprehensive review identified 150+ files with inconsistencies across `docs/` and `example/`:

- **`benchmark.py` library bug:** Assigns `InferenceConfig` and `TaskConfig` **classes** instead of instances, breaking all benchmark usage
- **Wrong import paths:** `from dnallm import load_model_and_tokenizer` documented in ~20 files, but the function is NOT exported from top-level `__init__.py`
- **Wrong signatures:** `DNATrainer`, `Mutagenesis`, `DNAInference` documented with non-existent parameters or wrong parameter order
- **Wrong config keys:** `training:` / `training_args:` used instead of `finetune:`, plus many invalid keys (`greater_is_better`, `early_stopping_patience`, etc.)
- **Wrong task_type values:** `sequence_classification`, `token_classification` used instead of `binary`, `token`, etc.
- **Wrong CLI commands:** `dnallm-train`, `dnallm-inference` documented as standalone scripts, but actual CLI uses subcommands
- **MCP tool names:** Documented with underscore prefix (`_dna_sequence_predict`) but actual server registers without prefix (`dna_sequence_predict`)
- **API docs incomplete:** Missing `show_root_heading`, missing module docs, `TaskType` enum values don't match `TaskConfig` regex
- **Mirror drift risk:** `docs/example/` mirrors `example/` but can diverge silently
- No automated validation exists to catch docs/examples drift in CI

## Requirements

1. **Fix benchmark.py critical bug**
   - Current: `dnallm/inference/benchmark.py` lines 93, 104, 142, 150 assign `InferenceConfig` and `TaskConfig` classes (not instances), causing shared mutable state
   - Target: All four lines instantiate with `InferenceConfig()` and `TaskConfig()`
   - Acceptance: `Benchmark(config=configs)` runs without `TypeError` or attribute errors; `example/notebooks/benchmark/benchmark.ipynb` executes cell-by-cell without error

2. **Fix example YAML configs**
   - Current: `example/marimo/benchmark/config.yaml` lacks required `benchmark:` section; `benchmark_config.yaml` has typo `"postive"`
   - Target: All 22 YAML files under `example/` are valid per current `load_config()` schema
   - Acceptance: A script iterates all `.yaml` under `example/` and calls `load_config()` on each — zero exceptions

3. **Fix example notebooks**
   - Current: `predict_data.ipynb` uses `data_path=` instead of `file_path=`; `finetune_generation.ipynb` has compatibility issues with current code
   - Target: All 18 notebooks have valid imports, correct API calls, and correct parameters
   - Acceptance: Extract all code cells from each `.ipynb`, filter out magic commands (`!`, `%`), run `ast.parse()` — zero `SyntaxError` or `NameError` on dnallm API usage

4. **Fix marimo examples**
   - Current: `benchmark_demo.py` depends on broken `Benchmark`; `finetune_demo.py` passes wrong task parameter to `encode_sequences`; `inference_demo.py` has typo `Tokenzier`
   - Target: All 3 marimo demos have correct code
   - Acceptance: `python -m py_compile` passes on all `.py` files; grep confirms `Tokenzier` typo is fixed

5. **Fix MCP examples**
   - Current: `mcp_client_ollama_pydantic_ai.ipynb` uses wrong tool names (underscore prefix); `mcp_client_ollama_langchain_agents.ipynb` has wrong connection config
   - Target: Both notebooks reference correct MCP tool names and valid connection configs
   - Acceptance: Grep confirms tool names match server registration (`dna_sequence_predict`, etc.); connection config uses valid `MCPServerStreamableHTTP` format

6. **Fix API documentation structure**
   - Current: 10 API doc files lack `show_root_heading: true`; 4 source modules have no API docs; `TaskType` enum values mismatch `TaskConfig` regex
   - Target: All API docs have consistent `show_root_heading`; new modules documented; `TaskType` values match regex
   - Acceptance: `mkdocs build` completes without warnings for API doc pages; grep confirms `TaskType.BINARY == "binary"` (not `"binary_classification"`)

7. **Fix user guide import paths**
   - Current: ~20 files document `from dnallm import load_model_and_tokenizer`; ~8 files document `from dnallm import DNADataset`
   - Target: All docs use correct import paths: `from dnallm.models import load_model_and_tokenizer`, `from dnallm.datahandling import DNADataset`
   - Acceptance: Grep returns zero matches for `from dnallm import load_model_and_tokenizer` or `from dnallm import DNADataset` in `docs/`

8. **Fix user guide API signatures**
   - Current: `DNATrainer` documented with `tokenizer`, `train_dataset`, `eval_dataset` params that don't exist; `Mutagenesis` and `DNAInference` have wrong parameter order
   - Target: All docs reflect actual signatures: `DNATrainer(model, config, datasets, ...)`, `Mutagenesis(model, tokenizer, config)`, `DNAInference(model, tokenizer, config)`
   - Acceptance: For each affected doc file, the documented signature matches the actual `__init__` signature (verified by reading source)

9. **Fix user guide config keys**
   - Current: `training:` / `training_args:` used instead of `finetune:`; invalid keys like `greater_is_better`, `early_stopping_patience`, `gradient_checkpointing` documented
   - Target: All config examples use `finetune:` key and only valid `TrainingConfig` / `InferenceConfig` / `TaskConfig` fields
   - Acceptance: Grep returns zero matches for `training_args:` or `training:` (as top-level config key) in `docs/`; zero matches for invalid keys listed in review

10. **Fix user guide CLI commands**
    - Current: `dnallm-benchmark` documented as standalone script (it is not registered in pyproject.toml)
    - Target: All docs use `dnallm benchmark` subcommand format for benchmark
    - Acceptance: Grep returns zero matches for `dnallm-benchmark ` (with trailing space) in `docs/`

11. **Fix MCP documentation**
    - Current: Tool names documented with underscore prefix; default host shown as `0.0.0.0` (actual: `127.0.0.1`); fake HTTP endpoints (`/health`, `/predict`) documented
    - Target: Tool names have NO underscore prefix (match exposed server names); host is `127.0.0.1`; only valid endpoints documented
    - Acceptance: Grep confirms `dna_sequence_predict` (without underscore) in MCP docs; no `/health` or `/predict` endpoints in MCP usage docs

12. **Establish docs/example sync and CI validation**
    - Current: `docs/example/` mirrors `example/` but no automated check prevents drift; no CI validates docs code snippets
    - Target: A sync check script exists; CI workflow validates docs code snippets syntax
    - Acceptance: Running sync check script returns `0` when directories match, non-zero when they differ; CI workflow runs on PR and fails if docs code snippets have syntax errors

## Boundaries

**In scope:**
- All `.md` files under `docs/` (API docs, user guides, concepts, FAQ, getting started, resources)
- All `.ipynb`, `.py`, `.yaml` under `example/` and `docs/example/`
- Source code fixes that unblock docs/examples (e.g., `benchmark.py` bug, `TaskType` enum mismatch)
- Sync verification script between `docs/example/` and `example/`
- CI workflow for docs code snippet validation

**Out of scope:**
- Rewriting or restructuring documentation content ( prose, explanations, tutorials ) — only fix factual/code inaccuracies
- Adding new features or new examples — only fix existing ones
- Fixing upstream dependencies (torch, transformers) deprecation warnings
- Translating documentation to other languages
- Visual redesign of documentation theme or layout
- Deep functional testing of every notebook end-to-end (only static validation: imports, syntax, API signatures)

## Constraints

- `docs/example/` must remain a byte-identical mirror of `example/` (except runtime artifacts like `__pycache__/`, `logs/`, `outputs/`)
- Changes to source code (`dnallm/`) must be minimal and only fix bugs that block docs/examples from working
- All config examples must validate against current Pydantic models in `dnallm/configuration/configs.py`
- CI validation must run fast (< 2 minutes) — extract and parse code snippets, not execute notebooks
- Backward compatibility: fix docs to match code, not vice versa (except the benchmark.py bug which is a clear code defect)

## Acceptance Criteria

- [ ] `benchmark.py` bug fixed — `Benchmark` instantiates `InferenceConfig()` and `TaskConfig()` (not classes)
- [ ] All 22 YAML files under `example/` load successfully via `load_config()`
- [ ] `docs/example/` and `example/` are byte-identical for all tracked files
- [ ] Zero `from dnallm import load_model_and_tokenizer` in `docs/`
- [ ] Zero `from dnallm import DNADataset` in `docs/`
- [ ] Zero `training_args:` or `training:` (as top-level config key) in `docs/`
- [ ] Zero `dnallm-benchmark ` standalone command references in `docs/`
- [ ] All MCP tool names in docs match exposed names (no underscore prefix: `dna_sequence_predict`, etc.)
- [ ] `TaskType` enum values match `TaskConfig` regex (`binary`, `multiclass`, etc.)
- [ ] All API doc `.md` files have `show_root_heading: true`
- [ ] Sync check script exists and passes when directories match
- [ ] CI workflow exists and passes on current docs state

## Ambiguity Report

| Dimension          | Score | Min  | Status | Notes                              |
|--------------------|-------|------|--------|------------------------------------|
| Goal Clarity       | 0.90  | 0.75 | ✓      | Specific and measurable            |
| Boundary Clarity   | 0.85  | 0.70 | ✓      | Explicit in/out of scope           |
| Constraint Clarity | 0.75  | 0.65 | ✓      | Mirror constraint, CI time limit   |
| Acceptance Criteria| 0.80  | 0.70 | ✓      | 12 pass/fail checkboxes            |
| **Ambiguity**      | 0.16  | ≤0.20| ✓      |                                    |

Status: ✓ = met minimum, ⚠ = below minimum (planner treats as assumption)

## Interview Log

| Round | Perspective    | Question summary                          | Decision locked                                   |
|-------|----------------|------------------------------------------|---------------------------------------------------|
| —     | —              | User chose to skip Socratic interview    | Requirements derived from PLAN.md + review report |
| —     | —              | Ambiguity pre-assessment at 0.16         | Gate passed without interview rounds              |

---

*Phase: 07-docs-examples-sync*
*Spec created: 2026-05-15*
*Next step: /gsd:discuss-phase 07 — implementation decisions (how to build what's specified above)*
