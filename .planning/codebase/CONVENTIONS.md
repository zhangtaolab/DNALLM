---
last_mapped_commit: 9ec6bf532fca3d999dfb42329ac6c5cfacff21fb
last_mapped_at: 2026-10-05
---
# Coding Conventions

**Analysis Date:** 2026-10-05

## Language & Runtime Baseline

- Python **3.10+** minimum (`requires-python = ">=3.10"` in `pyproject.toml`; ruff `target-version = "py310"`; mypy `python_version = "3.10"`). CI tests 3.11/3.12/3.13 (`.github/workflows/ci.yml`).
- Use PEP 604 unions and built-in generics: `str | None`, `list[int]`, `dict[str, float]`. Do not add new `Optional[...]`/`List[...]` imports.
- Newer modules start with `from __future__ import annotations` (see `tests/examples/_execution.py`, `tests/mcp/test_client_sdk.py`) — follow that pattern in new files.
- Comments in legacy files are mixed English/Chinese; write all new comments in English.

## Naming Patterns

**Files:**
- `snake_case.py` modules: `dnallm/utils/sequence.py`, `dnallm/inference/inference.py`
- Test files: `test_<module>.py` mirroring the package layout (`tests/utils/test_sequence.py`, `tests/models/test_special/test_evo.py`)
- Real-model integration tests: suffix `_real_model.py` (`tests/finetune/test_trainer_real_model.py`, `tests/inference/test_inference_real_model.py`)
- Private helper modules prefixed `_`: test-harness seams `tests/examples/_execution.py`, `dnallm/mcp/tests/_network_skip.py`; private functions `_compute_mean_embeddings` in `dnallm/inference/plot.py`

**Functions:**
- `snake_case`; private with leading underscore (`_setup_handlers`, `_get_model_path_and_imports` in `dnallm/models/model.py`)
- Test functions: `test_*` with descriptive behavior names (`test_download_success_after_retries`, `test_random_generate_sequences_padding_backs_off_to_maxl`)
- Fixture functions: `snake_case` nouns, factories suffixed `_factory` (`inference_config_factory`, `tiny_model_factory` in `tests/conftest.py`)

**Variables:**
- `snake_case`; throwaway/unused prefixed `_` (ruff `dummy-variable-rgx` allows `_`)

**Types/Constants:**
- Classes `PascalCase`, usually prefixed with the domain acronym: `DNAInference`, `DNADataset`, `DNATrainer`, `DNALLMMCPServer`, `DNALLMforSequenceClassification`
- Test classes: `Test*` grouping one area under test (`class TestDownloadModel:`, `class TestSeedSandbox:`)
- Constants: `UPPER_SNAKE_CASE`, module-level (`NOTEBOOK_EXEC_SPECS`, `ACTIVE_NOTEBOOKS`, `_ENV_OVERRIDES` in `tests/examples/_execution.py`; `COLORS: ClassVar[dict[str, str]]` in `dnallm/utils/logger.py`)
- Type hints on all new function signatures, `ClassVar[...]` for class-level constants, `if TYPE_CHECKING:` guard for import-only types

## Code Style

**Formatting:**
- Tool: **ruff format** (Black-compatible), configured in `[tool.ruff.format]` in `pyproject.toml`
- Line length **100** (`[tool.ruff]`), indent 4 spaces, double quotes, magic trailing commas respected, auto line endings
- Run: `ruff format .` / check: `ruff format --check .`
- NOTE: `CONTRIBUTING.md` still says "line length 79" — stale; the enforced ruff config is 100. The 79-char limit only remains for the MCP module via legacy `.flake8`.

**Linting:**
- Primary: **ruff** 0.16.9 pinned (`[tool.ruff.lint]`), `preview = true`
- Rules selected: `E4, E7, E9, F, W, B, C4, UP, N, S, T20, PT, Q, RUF`; `fixable = ["ALL"]`
- Ruff excludes vendored/legacy code: `dnallm/tasks/metrics/`, `example/`, `dnallm/models/special/mamba_npu.py`, `dnallm/finetune/megatron.py`, `.planning/` — do not hold those to project conventions
- Per-file ignores for `tests/**/*.py` relax: relative imports, unused imports/variables, star imports, magic-value comparison, asserts, hardcoded temp/password strings
- Inline suppressions use the comment form `# ruff: ignore[suspicious-subprocess-import]` (see `tests/examples/_execution.py`, `scripts/check_code.py`)
- Secondary: **flake8** (`.flake8`, max-line-length 79) applies to the MCP module only — historical, do not extend

**Type checking:**
- **ty (astral)** configured at `[tool.ty.src]` in `pyproject.toml`: excludes mirror the ruff/mypy vendored-code excludes (`dnallm/tasks/metrics/**`, `dnallm/models/special/mamba_npu.py`, `dnallm/finetune/megatron.py`); deliberately NO severity overrides — every remaining diagnostic must stay visible signal. Run flagless: `uvx ty check dnallm/`. Owner decision (recorded in recent closeout commits): **ty becomes the hard gate; mypy is retired**. Current baseline is 165 diagnostics; new code must not add to it.
- mypy is relaxed mode (`check_untyped_defs = true`, `no_implicit_optional = true`, `warn_unreachable = true`) and advisory in CI (`|| true`) — do not rely on it; it is on the way out.

**Pre-commit:**
- `.pre-commit-config.yaml` local hooks run in order: `ruff format` → `ruff check` → `mypy dnallm/ --show-error-codes --pretty`
- One-command wrapper matching CI: `python scripts/check_code.py` (options `--with-tests`, `--fix`, `--ci`, `--docs`, `--verbose`); shell variant `scripts/check_code.sh`; CI-aggregate variant `scripts/ci_checks.sh` (also runs `scripts/check_notebook_md_sync.py`)

## Import Organization

**Order:**
1. `from __future__ import annotations` (newer modules)
2. Standard library (`import os`, `from pathlib import Path`, `from typing import TYPE_CHECKING`)
3. Third party (`import pytest`, `import torch`, `from unittest.mock import Mock, patch`)
4. First party / local

**Rules:**
- Inside the `dnallm` package use **relative imports**: `from ..datahandling.data import DNADataset`, `from ..utils import get_logger` (`dnallm/inference/inference.py`)
- In tests and scripts use **absolute imports**: `from dnallm.tasks.metrics import ...`, `from tests.examples._execution import run_notebook` (`tests/examples/test_notebook_execution.py`)
- Heavy/optional dependencies are imported inside function bodies to keep startup fast and avoid cycles: `from ..finetune import DNATrainer` inside the `train()` command in `dnallm/cli/cli.py`; optional deps guarded with try/except (`dnallm/finetune/trainer.py`, `dnallm/models/model.py`)
- isort config: `known-first-party = ["dnallm"]`

**Path Aliases:**
- None. Pure package-relative / absolute imports.

## Error Handling

**Patterns:**
- Raise `ValueError` with a descriptive, regex-matchable message for invalid input/config/sequence — the dominant raise type in `dnallm/` (e.g. `download_model` raises `ValueError(f"Model {model_name} download failed.")` in `dnallm/models/model.py`)
- `ImportError` for missing optional dependencies (`"gpn package is required"` in `dnallm/models/special/`); `RuntimeError` for environment/device problems (provisioning helpers in `tests/examples/_execution.py`); `NotImplementedError` for unimplemented head/task paths
- Wrap and re-raise at boundaries: `raise ValueError(f"Failed to load model: {e}") from e` (`dnallm/models/model.py`)
- Retry-with-backoff for network operations (`download_model(..., max_try=3)` with `time.sleep` between attempts — tests patch `time.sleep`)
- Pydantic `Field(pattern=...)`, `field_validator`, `model_validator`, `model_post_init` reject invalid config early in `dnallm/configuration/configs.py` — validate at config-load time, never deep in training loops
- MCP tools return error dicts rather than raising across the protocol boundary; every tool call timeout-wrapped (`dnallm/mcp/server.py`)
- Compat shims no-op when their target library is absent so import never breaks (`dnallm/utils/transformers_compat.py`, `dnallm/utils/cuda_compat.py`)
- `warnings` module for non-fatal degradation (`dnallm/inference/inference.py`)
- Tests assert errors with `pytest.raises(ValueError, match=r"...")` regex matching (16 uses in `tests/configuration/test_configs.py` alone)

## Logging

**Framework:** loguru via `dnallm/utils/logger.py` (`get_logger`/`setup_logging`; convenience `log_info`, `log_error`, `log_success`, ...)

**Patterns:**
- Module-level logger at import time: `logger = get_logger("dnallm.inference.inference")`
- No bare `print()` in library code (ruff `T20` selected); tests routinely `patch("builtins.print")` to silence legacy output
- MCP server uses loguru directly (`dnallm/mcp/`)

## Comments

**When to Comment:**
- Inline comments only for complex/non-obvious logic (e.g. the 3D-logit flattening note in `tests/tasks/test_metrics.py`)
- Decision-trail comments citing the governing decision/task ID are the established style in test infrastructure: `# D-06`, `# 08-06`, `# 261003-csd, T-mcp1-01`, `# REPAIR-01 (08-02)` (see `tests/examples/_execution.py`, `pyproject.toml` extras, `.github/workflows/ci.yml`). When changing behavior governed by such a comment, keep the citation accurate.
- Section separators using box-drawing characters in larger test/script files: `# ──────────────────────────────────────────────────────────────────────────────` with a title line (see `tests/examples/test_examples.py`)

**Docstrings — Google style everywhere:**
- Module: summary paragraph + numbered feature list + `Example:` code block (`dnallm/inference/inference.py`, `tests/examples/_execution.py`)
- Class: summary + `Attributes:` section
- Function: `Args:` / `Returns:` / `Raises:` sections; use untyped-param style `name: description` (types live in the signature). Older modules use `name (str): ...` — do not copy that into new code.
- Test functions and fixtures carry docstrings too (`tests/conftest.py` fixtures document their contract)

## Function Design

**Size:** Small focused helpers; harness logic decomposed into named functions with one job (`seed_sandbox`, `run_notebook`, `assert_tree_clean` in `tests/examples/_execution.py`)

**Parameters:** Keyword-friendly with defaults; factory fixtures take overrides as keyword args (`inference_config_factory(task_type="binary", num_labels=None, ...)`)

**Return Values:** Typed returns; helpers that can fail return evidence tuples `(bool, str)` for gate probes (`megadna_prerequisites_installed`, `evo_prerequisites_installed` in `tests/examples/_execution.py`); raising helpers document `Raises:` explicitly

## Module Design

**Exports:**
- Public API re-exported in `dnallm/__init__.py` with explicit `__all__`: `Benchmark`, `DNADataset`, `DNAInference`, `DNAInterpret`, `DNATrainer`, `Mutagenesis`, `load_config`, `load_model_and_tokenizer`, `get_logger`, `setup_logging`, `cli`
- One facade class per subpackage; console entry points declared in `[project.scripts]`
- All tunables live in Pydantic `BaseModel` classes in `dnallm/configuration/configs.py` using `Field(default=..., description=...)`; configs load from YAML via `load_config`; engines accept `Mapping`-typed config dicts (not just `dict`) — the v1.1 Phase-8 fix for ty TypedDict assignability across all 5 facades

**Barrel Files:** Not used beyond the package facade `dnallm/__init__.py`.

**Private harness-seam modules:** `_`-prefixed helper modules that live beside their consumer tests, are never imported by any root conftest or `dnallm/` module, and never ship in the wheel: `tests/examples/_execution.py`, `dnallm/mcp/tests/_network_skip.py`. Follow this pattern for new shared test machinery.

## Repo-Hygiene Conventions (owner rules)

- **Same-change pytest rule:** any `dnallm/` code modification ships with pytest coverage in the same change — no exceptions across phases.
- **Atomic notebook commits:** a notebook, its Markdown mirror in `docs/`, and any wrapper changes commit together; `scripts/check_notebook_md_sync.py` (AST-based statement matching, run via `scripts/ci_checks.sh`) fails when they drift.
- **Vendored code is untouchable:** `dnallm/tasks/metrics/` (HF evaluate ports), `dnallm/models/special/enformer_model/` (ported Enformer), `dnallm/finetune/megatron.py`, `dnallm/models/special/mamba_npu.py` are excluded from lint/type/coverage — treat as upstream.
- **Coverage gate is global:** `fail_under = 90` in `[tool.coverage.report]` applies to every `--cov` invocation; use `--no-cov` for scoped runs.

---

*Convention analysis: 2026-10-05*
