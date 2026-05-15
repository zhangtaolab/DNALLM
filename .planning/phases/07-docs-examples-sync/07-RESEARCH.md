# Phase 07: docs-examples-sync - Research

**Researched:** 2026-05-15
**Domain:** Python documentation sync, mkdocs, mkdocstrings, CI validation
**Confidence:** HIGH

## Summary

This phase fixes inconsistencies between `docs/`, `example/`, and the `dnallm` codebase, repairs a critical `benchmark.py` bug, resolves a `TaskConfig` regex mismatch, and establishes automated validation to prevent future drift. All 12 requirements from SPEC.md have been verified against the actual codebase.

**Key discovery:** Two SPEC requirements (import paths and CLI commands) are partially no-ops because the documented patterns ARE valid. The `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset`, making `from dnallm import X` valid. Both `dnallm-train` standalone scripts AND `dnallm train` subcommands are registered in `pyproject.toml`. Only truly invalid references need fixing.

**Primary recommendation:** Follow the pattern-based traversal order (D-04): fix source code first, then YAML configs, then imports/signatures/config-keys/CLI/task-types, then MCP, then notebooks/marimo, then API docs, then tutorials, then validation scripts. This builds context memory efficiently.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — Validation Strategy:** Strict validation for local development (attempt to run core cells); CI only validates imports + initialization (informational, not blocking); external-data-dependent notebooks only validate imports/API signatures; YAML configs validated via `load_config()`
- **D-02 — Batch Modification Strategy:** All fixes are file-by-file reviewed. No script-based bulk replacements without per-file context review. Workflow: `grep` generates candidate list → open each file → confirm context → modify → commit
- **D-03 — TaskType Enum / TaskConfig Regex Fix:** Fix `TaskConfig` regex (not `TaskType` enum). Expand regex to accept BOTH short names (`binary`, `multiclass`, `multilabel`, `token`) AND full names (`binary_classification`, `multi_class_classification`, `multi_label_classification`, `token_classification`). Update `model_post_init` to normalize both forms. Keep `TaskType` enum values unchanged.
- **D-04 — Execution Order:** Pattern-based traversal, not directory-based. Order: source code fixes → YAML configs → import paths → config keys → API signatures → CLI commands → task types → MCP → notebooks/marimo → API doc structure → tutorials → CI/validation

### Claude's Discretion
- None — all key decisions were explicitly made during discussion

### Deferred Ideas (OUT OF SCOPE)
- None

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-01 | Fix benchmark.py critical bug (class vs instance) | Verified: lines 93, 104, 142, 150 assign classes not instances |
| REQ-02 | Fix example YAML configs (22 files) | Verified: 21 YAML files, all currently load via `load_config()` with 0 errors; marimo benchmark config lacks `benchmark:` section but loads OK as non-benchmark config |
| REQ-03 | Fix example notebooks (18 notebooks) | Verified: 21 notebooks found, 0 syntax errors; predict_data.ipynb has `data_path=` instead of `file_path=` in markdown cell |
| REQ-04 | Fix marimo examples (3 demos) | Verified: 3 files, py_compile passes; inference_demo.py has `Tokenzier` typo (2 occurrences); finetune_demo.py passes `task=` to `encode_sequences` (parameter exists but value may be wrong) |
| REQ-05 | Fix MCP examples (2 notebooks) | Verified: pydantic_ai notebook uses `_dna_sequence_predict` (wrong underscore prefix); langchain notebook uses `streamable_http` (should be `streamable-http`) |
| REQ-06 | Fix API documentation structure | Verified: 9 API docs lack `show_root_heading`; `TaskType` enum values (`binary_classification`) don't match `TaskConfig` regex (`binary`) |
| REQ-07 | Fix user guide import paths | **NO-OP:** `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset`; all `from dnallm import X` patterns reference valid exports |
| REQ-08 | Fix user guide API signatures | Verified: DNATrainer docs use `tokenizer=`, `train_dataset=`, `eval_dataset=` params that don't exist; Mutagenesis/DNAInference parameter order varies |
| REQ-09 | Fix user guide config keys | Verified: `training_args:` in 3 docs files; invalid keys (`greater_is_better`: 5 files, `early_stopping_patience`: 2 files, `gradient_checkpointing`: 6 files) |
| REQ-10 | Fix user guide CLI commands | **PARTIAL NO-OP:** `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` ARE valid standalone scripts per `pyproject.toml`. Only `dnallm-benchmark` is non-existent (1 file). |
| REQ-11 | Fix MCP documentation | Verified: docs show host `0.0.0.0` (actual default: `127.0.0.1`); docs show fake endpoints `/health`, `/predict`; tool names in docs may need underscore prefix review |
| REQ-12 | Establish docs/example sync and CI validation | Verified: `docs/example/` and `example/` are in sync (only runtime artifacts differ: `__pycache__`, `logs`, `outputs`, plus 2 data files in NER task); existing `tests/examples/test_examples.py` covers syntax/imports/YAML |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Source code fixes (benchmark.py, TaskConfig) | API / Backend | — | These are code defects in `dnallm/` library |
| YAML config validation | API / Backend | CI / Static | `load_config()` is the validation engine |
| Import path fixes | API / Backend | — | `__init__.py` exports define valid paths |
| API signature docs | CDN / Static | API / Backend | Docs reflect code reality |
| Config key docs | CDN / Static | API / Backend | Docs must match Pydantic models |
| CLI command docs | CDN / Static | API / Backend | Docs must match `pyproject.toml` entry points |
| Notebook fixes | Browser / Client | CDN / Static | Notebooks are user-facing artifacts |
| Marimo demo fixes | Browser / Client | CDN / Static | Marimo is interactive UI |
| MCP docs | CDN / Static | API / Backend | Docs must match server registration |
| API doc structure | CDN / Static | — | mkdocstrings configuration |
| Sync validation | CI / Static | — | Scripts run in CI |
| Docs code snippet validation | CI / Static | — | Static analysis of markdown |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.13.4 | Config validation via `TaskConfig`, `TrainingConfig`, etc. | Already in use; regex fix needed |
| mkdocs | 1.6.x (via `[docs]` extra) | Documentation site generation | Project uses mkdocs-material |
| mkdocstrings-python | 1.16.10+ | API doc generation from docstrings | Already configured in `pyproject.toml` |
| mkdocs-jupyter | 0.24.0+ | Notebook rendering in docs | Already configured |
| pytest | 9.0.3+ | Test framework for validation | Existing test infrastructure |
| PyYAML | 6.0.2+ | YAML config parsing | Used by `load_config()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| click | 8.1.x | CLI framework | For CLI command validation |
| ruff | 0.9.x | Linting/formatting | Existing CI uses ruff |
| mypy | 1.15.x | Type checking | Existing CI uses mypy |

### Existing Reusable Assets
| Asset | Location | Purpose |
|-------|----------|---------|
| `load_config()` | `dnallm/configuration/configs.py:483` | YAML validation against Pydantic models |
| `check_code.py` | `scripts/check_code.py` | Local CI simulation; extend for docs validation |
| `test_examples.py` | `tests/examples/test_examples.py` | Existing syntax/import/YAML tests for examples |
| CI workflow | `.github/workflows/ci.yml` | Existing CI infrastructure |

## Package Legitimacy Audit

This phase does not install new external packages. All tools are already project dependencies or standard library modules. No slopcheck needed.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   docs/         │     │   example/      │     │   dnallm/       │
│   (markdown)    │     │   (notebooks,   │     │   (source code) │
│                 │     │    yaml, py)    │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │         ┌─────────────┘                       │
         │         │                                     │
         ▼         ▼                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Validation Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Sync Check  │  │ load_config │  │ Code Snippet Extractor  │  │
│  │ (diff dirs) │  │ (Pydantic)  │  │ (ast.parse markdown)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   CI Workflow   │
                    │  (GitHub Actions)│
                    └─────────────────┘
```

### Recommended Project Structure

```
dnallm/
├── inference/
│   └── benchmark.py          # REQ-01: fix class→instance
├── configuration/
│   └── configs.py            # REQ-06: expand TaskConfig regex
├── tasks/
│   └── task.py               # REQ-06: TaskType enum (keep as-is per D-03)
└── __init__.py               # REQ-07: exports are valid (no-op)

docs/
├── api/                      # REQ-06: add show_root_heading
├── user_guide/               # REQ-07-11: fix imports, signatures, keys, CLI, MCP
├── example/                  # REQ-12: mirror of example/
└── ...

example/
├── notebooks/                # REQ-03: fix notebooks
├── marimo/                   # REQ-04: fix demos
├── mcp_example/              # REQ-05: fix MCP notebooks
└── *.yaml                    # REQ-02: validate configs

scripts/
└── check_code.py             # Extend for docs validation

tests/
├── examples/
│   └── test_examples.py      # Existing: syntax, imports, YAML
└── ...

.github/workflows/
├── ci.yml                    # Add docs validation job
└── ...
```

### Pattern 1: Per-File Review Workflow (D-02)
**What:** `grep` generates candidates → open each file → confirm context → modify → commit
**When to use:** All documentation fixes to prevent prose corruption
**Example:**
```bash
# Generate candidate list
grep -rn "from dnallm import load_model_and_tokenizer" docs/ > candidates.txt
# For each file in candidates.txt:
#   1. Open file
#   2. Verify the import is in a code block (not prose explanation)
#   3. Modify if needed
#   4. Commit
```

### Pattern 2: Pattern-Based Traversal (D-04)
**What:** Process all files for one problem pattern before moving to next
**When to use:** All 12 requirement batches
**Example:** Fix all `data_path=` → `file_path=` across docs AND examples before moving to next pattern

### Pattern 3: Two-Way Sync for docs/example/
**What:** Any fix in `example/` must be mirrored to `docs/example/`
**When to use:** All example file modifications
**Example:**
```bash
# After fixing example/notebooks/benchmark/benchmark_config.yaml
cp example/notebooks/benchmark/benchmark_config.yaml docs/example/notebooks/benchmark/benchmark_config.yaml
```

### Anti-Patterns to Avoid
- **Bulk regex replacement without review:** Documentation contains prose that may legitimately reference strings being fixed (e.g., "binary classification" in explanatory text)
- **Changing code to match docs:** Backward compatibility constraint says fix docs to match code, not vice versa (except benchmark.py bug which is a clear code defect)
- **Validating notebooks by execution in CI:** CI time limit is < 2 minutes; only static validation (imports, syntax, API signatures)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML config validation | Custom parser | `dnallm.configuration.configs.load_config()` | Already validates against Pydantic models; catches schema errors |
| Notebook syntax checking | Custom parser | `ast.parse()` + `json.load()` for `.ipynb` | Standard library, handles all Python syntax |
| Markdown code extraction | Custom regex | `re.findall(r'\`\`\`python\n(.*?)\n\`\`\`', content, re.DOTALL)` | Simple and sufficient for fenced code blocks |
| Directory sync check | Custom diff logic | `filecmp.dircmp()` with ignore list | Standard library, handles recursive comparison |
| CI docs validation | Custom GitHub Actions | Extend existing `.github/workflows/ci.yml` | Project already uses uv + pytest pattern |

## Runtime State Inventory

**Phase type:** Code/docs sync (not rename/refactor). No runtime state migration needed.

**Stored data:** None — this phase only fixes documentation and example files
**Live service config:** None
**OS-registered state:** None
**Secrets/env vars:** None
**Build artifacts:** `docs/example/` must remain byte-identical mirror of `example/` (except runtime artifacts)

## Common Pitfalls

### Pitfall 1: Import Path "Fix" That Breaks Valid Code
**What goes wrong:** Changing `from dnallm import load_model_and_tokenizer` to `from dnallm.models import load_model_and_tokenizer` when both are valid
**Why it happens:** `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset`; the SPEC assumed these were not exported
**How to avoid:** Verify each occurrence. Both import styles are valid. Only fix if the imported name is NOT in `dnallm.__all__`
**Warning signs:** Grep shows the name in `dnallm/__init__.py` `__all__` list

### Pitfall 2: CLI Command "Fix" That Removes Valid Commands
**What goes wrong:** Changing `dnallm-train` to `dnallm train` when both are valid
**Why it happens:** `pyproject.toml` registers BOTH standalone scripts (`dnallm-train`) AND subcommands (`dnallm train`)
**How to avoid:** Only fix references to NON-EXISTENT commands (e.g., `dnallm-benchmark` does not exist)
**Warning signs:** Check `pyproject.toml` `[project.scripts]` section before "fixing" CLI references

### Pitfall 3: MCP Tool Name Confusion
**What goes wrong:** Adding underscore prefix to tool names in docs when server exposes them WITHOUT underscore
**Why it happens:** Internal Python methods use `_` prefix (e.g., `_dna_sequence_predict`) but the exposed tool name is `dna_sequence_predict`
**How to avoid:** The `_with_timeout_wrapper` registers tools with the second argument as the exposed name. Verify against server.py line 252
**Warning signs:** `self.app.tool()(self._with_timeout_wrapper(self._dna_sequence_predict, "dna_sequence_predict"))` — exposed name is `"dna_sequence_predict"`

### Pitfall 4: TaskConfig Regex Expansion Breaking model_post_init
**What goes wrong:** Expanding regex to accept `binary_classification` but `model_post_init` only checks for `binary`
**Why it happens:** `model_post_init` uses exact string matches (`if self.task_type == "binary":`)
**How to avoid:** Normalize task_type in `model_post_init` before checking (per D-03)
**Warning signs:** `TaskConfig(task_type="binary_classification")` passes regex but gets wrong defaults

### Pitfall 5: docs/example/ Mirror Drift
**What goes wrong:** Fixing a file in `example/` but forgetting to sync to `docs/example/`
**Why it happens:** Two directories must remain identical but are easy to forget
**How to avoid:** Use sync check script after each batch; include sync check in CI
**Warning signs:** `diff -rq example/ docs/example/` shows differences

## Code Examples

### TaskConfig Regex Fix (D-03)
```python
# Source: dnallm/configuration/configs.py (proposed fix)
class TaskConfig(BaseModel):
    task_type: str = Field(
        ...,
        pattern="^(embedding|mask|generation|binary|binary_classification|multiclass|multi_class_classification|multilabel|multi_label_classification|regression|token|token_classification)$",
    )
    # ...
    def model_post_init(self, __context):
        # Normalize to short form for internal logic
        task = self.task_type
        if task == "binary_classification":
            task = "binary"
        elif task == "multi_class_classification":
            task = "multiclass"
        elif task == "multi_label_classification":
            task = "multilabel"
        elif task == "token_classification":
            task = "token"
        # ... rest of checks use `task` variable
```

### Benchmark.py Fix (REQ-01)
```python
# Source: dnallm/inference/benchmark.py (proposed fix)
# Line 93: self.config["inference"] = InferenceConfig  →  self.config["inference"] = InferenceConfig()
# Line 104: self.config["task"] = TaskConfig  →  self.config["task"] = TaskConfig()
# Line 142: self.config["inference"] = InferenceConfig  →  self.config["inference"] = InferenceConfig()
# Line 150: self.config["task"] = TaskConfig  →  self.config["task"] = TaskConfig()
```

### Sync Check Script (REQ-12)
```python
#!/usr/bin/env python3
"""Verify docs/example/ is a byte-identical mirror of example/ (except runtime artifacts)."""
import sys
import filecmp
import os

IGNORE = ['__pycache__', 'logs', 'outputs', 'outputs_multilabel', '.ipynb_checkpoints']

def check_sync(dir1, dir2):
    dcmp = filecmp.dircmp(dir1, dir2, ignore=IGNORE)
    errors = []
    for f in dcmp.left_only:
        errors.append(f'Only in {dir1}: {f}')
    for f in dcmp.right_only:
        errors.append(f'Only in {dir2}: {f}')
    for f in dcmp.diff_files:
        errors.append(f'Differ: {f}')
    for subdir in dcmp.common_dirs:
        errors.extend(check_sync(os.path.join(dir1, subdir), os.path.join(dir2, subdir)))
    return errors

if __name__ == '__main__':
    errors = check_sync('example', 'docs/example')
    if errors:
        print('SYNC ERRORS:')
        for e in errors:
            print(f'  {e}')
        sys.exit(1)
    print('OK: docs/example/ is in sync with example/')
    sys.exit(0)
```

### Docs Code Snippet Validation (REQ-12)
```python
"""Extract and validate Python code blocks from markdown files."""
import re
import ast
import os

def validate_md_file(path):
    with open(path) as f:
        content = f.read()
    blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    errors = []
    for i, block in enumerate(blocks):
        lines = [l for l in block.split('\n') if not l.strip().startswith(('!', '%'))]
        code = '\n'.join(lines)
        if code.strip():
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append((i+1, str(e)))
    return errors
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `model_post_init` (pydantic v1 style) | `model_validator(mode="after")` (pydantic v2) | v0.5.2 | `model_post_init` still works but is deprecated; expanding regex requires updating the post-init logic |
| `training_args:` top-level key | `finetune:` top-level key | v0.5.x | Config schema changed; docs still reference old key |
| `sequence_classification` task type | `binary`/`multiclass` short names | v0.5.x | `TaskConfig` regex uses short names; docs still reference old names |
| `dnallm-xxx` standalone only | Both standalone AND `dnallm xxx` subcommands | v0.5.x | `pyproject.toml` registers both styles |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `from dnallm import load_model_and_tokenizer` is valid because `dnallm/__init__.py` exports it | REQ-07 | If `__init__.py` changes, imports break; but per D-03, we fix docs to match code |
| A2 | `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` are valid standalone scripts per `pyproject.toml` | REQ-10 | If entry points change, CLI docs break; but we fix docs to match current entry points |
| A3 | MCP tool names are exposed WITHOUT underscore prefix (e.g., `dna_sequence_predict`) | REQ-05, REQ-11 | If server registration changes, MCP docs break; verified by reading `_with_timeout_wrapper` calls |
| A4 | `docs/example/` and `example/` are currently in sync for tracked files | REQ-12 | Verified by `filecmp.dircmp` ignoring runtime artifacts; only 2 data files differ |
| A5 | All 21 YAML files under `example/` currently pass `load_config()` validation | REQ-02 | Verified by running `load_config()` on each file; 0 errors returned |

## Open Questions

1. **REQ-07 Import paths: No-op or not?**
   - What we know: `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset`; all `from dnallm import X` patterns in docs reference valid exports
   - What's unclear: SPEC acceptance criteria says "Zero `from dnallm import load_model_and_tokenizer` in docs/" — but this would change valid code to an alternative valid form
   - Recommendation: Verify with user whether to standardize on submodule imports (`from dnallm.models import...`) or leave top-level imports as-is since both work

2. **REQ-04 Marimo finetune_demo.py encode_sequences task parameter**
   - What we know: `encode_sequences` accepts `task` parameter; call passes `configs['task'].task_type` (e.g., `"binary"`)
   - What's unclear: Whether `"binary"` is a valid value for the `task` parameter (signature says default is `"SequenceClassification"`)
   - Recommendation: Runtime test during implementation to confirm; may need to map task_type values

3. **REQ-03 Notebook functional testing scope**
   - What we know: SPEC says "only static validation: imports, syntax, API signatures"
   - What's unclear: Whether any notebooks need actual execution to verify fixes (e.g., benchmark.ipynb after benchmark.py fix)
   - Recommendation: Per D-01, attempt local execution for core cells; CI remains static-only

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

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | `pyproject.toml` (pytest.ini_options) |
| Quick run command | `pytest tests/examples/test_examples.py -v` |
| Full suite command | `pytest tests/ -v -m "not slow"` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-01 | benchmark.py instantiates configs | unit | `pytest tests/inference/test_benchmark.py -x` | ✓ |
| REQ-02 | All YAML files load via `load_config()` | integration | `python -c "from dnallm.configuration import load_config; [load_config(p) for p in example_yaml_paths]"` | Script to create |
| REQ-03 | Notebooks have valid syntax | unit | `pytest tests/examples/test_examples.py::TestNotebookExamples -x` | ✓ |
| REQ-04 | Marimo files compile | unit | `pytest tests/examples/test_examples.py::TestMarimoExamples -x` | ✓ |
| REQ-06 | API docs have show_root_heading | static | `grep -L "show_root_heading" docs/api/**/*.md` | N/A |
| REQ-12 | docs/example/ sync | static | `python scripts/check_docs_sync.py` | Script to create |
| REQ-12 | CI workflow passes | CI | GitHub Actions | Workflow to create |

### Sampling Rate
- **Per task commit:** Run relevant grep checks or targeted pytest
- **Per wave merge:** `pytest tests/examples/test_examples.py -v`
- **Phase gate:** All acceptance criteria from SPEC.md verified

### Wave 0 Gaps
- [ ] `scripts/check_docs_sync.py` — sync verification script (REQ-12)
- [ ] `scripts/check_docs_snippets.py` — docs code snippet validation (REQ-12)
- [ ] `.github/workflows/docs-validation.yml` — CI workflow for docs validation (REQ-12)
- [ ] `tests/configuration/test_yaml_load.py` — explicit YAML load_config test for all example configs (REQ-02)

## Security Domain

This phase is documentation and example synchronization. No security-sensitive changes.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | No | — |
| V6 Cryptography | No | — |

## Sources

### Primary (HIGH confidence)
- `dnallm/inference/benchmark.py` — Verified class-vs-instance bug at lines 93, 104, 142, 150
- `dnallm/configuration/configs.py` — Verified `TaskConfig` regex at line 102; `model_post_init` at line 116
- `dnallm/tasks/task.py` — Verified `TaskType` enum values (lines 51-62)
- `dnallm/__init__.py` — Verified top-level exports include `load_model_and_tokenizer` and `DNADataset`
- `dnallm/cli/cli.py` — Verified CLI command structure; mcp-server default host is `127.0.0.1`
- `dnallm/mcp/server.py` — Verified tool registration exposes names WITHOUT underscore prefix
- `pyproject.toml` — Verified entry points: `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` exist as standalone scripts
- `tests/examples/test_examples.py` — Verified existing test infrastructure for examples

### Secondary (MEDIUM confidence)
- `docs/` directory grep results — Counted occurrences of patterns (imports, config keys, CLI commands, task types)
- `example/` directory inspection — Verified YAML load success, notebook syntax, marimo compile
- `mkdocs.yml` — Verified mkdocstrings configuration and API doc structure

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tools are already project dependencies
- Architecture: HIGH — verified against actual codebase
- Pitfalls: HIGH — discovered and verified key discrepancies (import paths, CLI commands, MCP tool names)

**Research date:** 2026-05-15
**Valid until:** 2026-06-15 (stable stack, low churn expected)
