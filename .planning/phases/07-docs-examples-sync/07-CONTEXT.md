# Phase 07: docs-examples-sync - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix all inconsistencies between `docs/`, `example/`, and the current `dnallm` codebase (imports, API signatures, config keys, CLI commands, task types), fix the `benchmark.py` critical bug and `TaskConfig` regex mismatch, and establish automated validation to prevent future drift.

</domain>

<spec_lock>
## Requirements (locked via SPEC.md)

**12 requirements are locked.** See `07-SPEC.md` for full requirements, boundaries, and acceptance criteria.

Downstream agents MUST read `07-SPEC.md` before planning or implementing. Requirements are not duplicated here.

**In scope (from SPEC.md):**
- All `.md` files under `docs/` (API docs, user guides, concepts, FAQ, getting started, resources)
- All `.ipynb`, `.py`, `.yaml` under `example/` and `docs/example/`
- Source code fixes that unblock docs/examples (e.g., `benchmark.py` bug, `TaskConfig` regex mismatch)
- Sync verification script between `docs/example/` and `example/`
- CI workflow for docs code snippet validation

**Out of scope (from SPEC.md):**
- Rewriting or restructuring documentation content (prose, explanations, tutorials) — only fix factual/code inaccuracies
- Adding new features or new examples — only fix existing ones
- Fixing upstream dependencies (torch, transformers) deprecation warnings
- Translating documentation to other languages
- Visual redesign of documentation theme or layout
- Deep functional testing of every notebook end-to-end (only static validation: imports, syntax, API signatures)

</spec_lock>

<decisions>
## Implementation Decisions

### D-01 — Validation Strategy (Area A)
- **Strict validation for local development:** Attempt to run core cells of all notebooks during local verification
- **CI downgrade for environment constraints:** CI only validates imports + initialization (does not execute notebooks requiring external data/models). CI validation is **informational, not blocking** for PRs
- **External-data-dependent notebooks:** Only validate imports and API signature calls, skip data-dependent cells
- **YAML configs:** Execute-level validation via `load_config()` (Pydantic strict validation already catches schema errors)

### D-02 — Batch Modification Strategy (Area B)
- **All fixes are file-by-file reviewed:** No script-based bulk replacements without per-file context review
- **Workflow:** `grep` generates candidate list → open each file → confirm context → modify → commit
- **Rationale:** Documentation contains prose that may legitimately reference strings being fixed (e.g., "binary classification" in explanatory text). Per-file review prevents accidental prose corruption.

### D-03 — TaskType Enum / TaskConfig Regex Fix (Area C)
- **Fix `TaskConfig` regex, not `TaskType` enum:** Expand `task_type` regex to accept BOTH short names (`binary`, `multiclass`, `multilabel`, `token`) AND full names (`binary_classification`, `multi_class_classification`, `multi_label_classification`, `token_classification`)
- **Update `model_post_init`:** Sync condition checks to recognize both forms (e.g., `if self.task_type in {"binary", "binary_classification"}`)
- **Keep `TaskType` enum values unchanged:** Backward compatibility preserved; existing code using `TaskType.BINARY` continues to work
- **Rationale:** Zero code uses `TaskType` enum directly (confirmed by grep), but expanding regex is safer than changing enum values that may be referenced in user configs

### D-04 — Execution Order (Area D)
- **Pattern-based traversal, not directory-based:** Process all files for one problem pattern before moving to the next pattern
- **Order:**
  1. Source code fixes (`benchmark.py` bug + `TaskConfig` regex)
  2. YAML config fixes (22 files, validation script)
  3. Import path pattern (`from dnallm import` → `from dnallm.models import` etc.)
  4. Config key pattern (`training:` → `finetune:`, remove invalid keys)
  5. API signature pattern (DNATrainer, Mutagenesis, DNAInference params)
  6. CLI command pattern (`dnallm-xxx` → `dnallm xxx`)
  7. Task type value pattern (`sequence_classification` → `binary` etc.)
  8. MCP-related fixes (tool names, host, endpoints)
  9. Notebook and Marimo specific fixes
  10. API doc structure fixes (`show_root_heading`, missing modules)
  11. Tutorial and case study fixes
  12. CI / validation scripts
- **Rationale:** Building context memory on one pattern (e.g., correct `DNATrainer` signature) makes sequential file processing more efficient

### Claude's Discretion
- None — all key decisions were explicitly made during discussion

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Specifications
- `.planning/phases/07-docs-examples-sync/07-SPEC.md` — Locked requirements, boundaries, and acceptance criteria
- `.planning/phases/07-docs-examples-sync/PLAN.md` — Task breakdown with 13 batches

### Source of Truth (Code)
- `dnallm/inference/benchmark.py` — Critical bug at lines 93, 104, 142, 150 (class vs instance)
- `dnallm/configuration/configs.py` — `TaskConfig` regex at line 102; `model_post_init` at line 116
- `dnallm/tasks/task.py` — `TaskType` enum values (lines 51-62)
- `dnallm/__init__.py` — Top-level exports (lines 1-30; `load_model_and_tokenizer` and `DNADataset` ARE exported)
- `dnallm/cli/cli.py` — CLI command structure (Click group with subcommands)
- `pyproject.toml` — Entry points: `dnallm`, `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server`
- `dnallm/mcp/server.py` — MCP tool registration (internal methods use `_` prefix; exposed names do not)

### Review Report
- Comprehensive review report generated by 5 parallel subagents covering API docs, user guides, notebooks, marimo/MCP examples, YAML configs, and concepts/FAQ/resources

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `dnallm/configuration/configs.py:load_config()` — Already validates YAML configs against Pydantic models; use for YAML validation script
- `scripts/check_code.py` — Existing local CI simulation script; can be extended for docs validation
- `.github/workflows/` — Existing CI infrastructure for adding docs validation workflow

### Established Patterns
- `docs/example/` mirrors `example/` — Any fix in `example/` must be mirrored to `docs/example/`
- API docs use `::: dnallm.module.Class` mkdocstrings format — Adding `show_root_heading: true` is a consistent pattern
- Notebooks follow consistent structure: load config → load model → run inference/training → visualize

### Integration Points
- `dnallm/inference/benchmark.py` — Fix here unblocks `example/notebooks/benchmark/benchmark.ipynb` and `example/marimo/benchmark/benchmark_demo.py`
- `dnallm/configuration/configs.py` — Regex fix here affects all YAML configs and docs referencing task types
- `pyproject.toml` entry points — CLI docs reference both `dnallm-xxx` standalone scripts AND `dnallm xxx` subcommands (both are valid)

### Scout Findings (Affecting Requirements)
- `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset` — Requirement 7 (fix import paths) may be a no-op
- `pyproject.toml` registers `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` — Requirement 10 (fix CLI commands) needs re-evaluation; some `dnallm-xxx` references may be correct
- MCP server exposes tool names WITHOUT underscore prefix (e.g., `"dna_sequence_predict"`) — Requirement 11 (MCP tool names) may be a no-op; internal methods use `_` prefix but exposed names do not
- No code in `dnallm/` uses `TaskType` enum values directly (outside `task.py` itself) — Safe to modify either enum or regex

</code_context>

<specifics>
## Specific Ideas

- Import path fix may be partially no-op: `dnallm/__init__.py` already exports `load_model_and_tokenizer` and `DNADataset`. Verify each occurrence before modifying.
- CLI command fix needs re-evaluation: `pyproject.toml` registers both `dnallm` (Click group) AND `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` (standalone scripts). Both invocation styles are valid. Only fix references to non-existent commands.
- MCP tool names: Server exposes `dna_sequence_predict` (no underscore) to clients. Internal Python method is `_dna_sequence_predict`. Docs referencing client-side tool names should NOT add underscore.
- `benchmark.py` bug: Lines 93, 104, 142, 150 assign `InferenceConfig` and `TaskConfig` classes instead of instances. Fix: add `()` to instantiate.
- `TaskConfig` regex expansion: Current regex `^(embedding|mask|generation|binary|multiclass|multilabel|regression|token)$` → expand to also accept `binary_classification`, `multi_class_classification`, `multi_label_classification`, `token_classification`.
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-docs-examples-sync*
*Context gathered: 2026-05-15*
