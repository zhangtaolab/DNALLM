# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-05-15

### Fixed

- Type safety: resolve all mypy errors across 70 source files
  - Remove invalid `# type: ignore[Any]` comments (`Any` is not a valid mypy error code)
  - Replace `any`/`callable` builtins with proper `Any`/`Callable` type annotations
  - Add `@overload` signatures for tokenizer methods with union return types
  - Add explicit `None` checks to narrow union types for mypy
- Code quality: resolve all ruff lint issues
  - Replace `assert` with explicit `if` + `raise ValueError` (S101 bandit rule)
  - Fix `Callable` import from `typing` instead of `collections.abc` (UP035)
  - Fix module-level import not at top of file (E402)
  - Fix dummy variable `_plot_dir` accessed after prefix (RUF052)
- Bug fixes discovered during type checking:
  - Fix `run_noise_tunnel()` call missing required `base_method` argument
  - Fix `DNAInference.__init__` not storing `self.config`
  - Fix CLI config key mismatch (`training_args` -> `finetune`)
  - Fix `Mutagenesis.evaluate()` return type (`list[dict]` -> `dict[str, Any]`)
  - Fix `LoggingContext.__exit__` crash when `original_level` is `None`
- Ruff formatting: reformat 4 files after line-length changes

## [0.5.1] - 2026-05-09

### Added

- MCP: Streamable HTTP transport support (client + server) — MCP Streamable HTTP migration (Phase 5)
- MCP: `transport="streamable-http"` support in `DNALLMMCPClient` using `mcp.client.streamable_http.streamablehttp_client()` (Phase 5)
- MCP: `StreamableHTTPConfig` configuration block with `host`, `port`, and `path` fields (Phase 5)
- MCP: Streamable HTTP integration tests in `test_streamable_http_client.py` (Phase 5)
- MCP: Client SDK unit tests for `transport="streamable-http"` initialization and connection (Phase 5)

### Changed

- MCP: `DNALLMMCPClient` transport type expanded from `Literal["sse", "stdio"]` to `Literal["streamable-http", "sse", "stdio"]` (Phase 5)
- MCP: Server docstrings and CLI help text now recommend `streamable-http` as the primary remote transport (Phase 5)
- README: MCP Server section now shows `streamable-http` as the primary example with SSE noted as legacy (Phase 5)
- Recommended remote transport changed from SSE to Streamable HTTP per MCP spec 2025-11-25

### Deprecated

- SSE transport marked as legacy; still supported for backward compatibility

## [0.5.0] - 2026-05-08

### Overview

Version 0.5.0 is a major release that transforms DNALLM from a basic fine-tuning toolkit into a production-ready platform for DNA language model research. This release introduces five major capability areas:

1. **Quality Assurance Infrastructure** (Phase 1) — Restored CI/CD, unified testing, and modern tooling
2. **Advanced Training Features** (Phase 2) — Early stopping, hyperparameter search, QLoRA, and visualization
3. **MCP Server & Client SDK** (Phase 3) — Full Model Context Protocol integration with 13 tools
4. **Dependency Modernization & UX** (Phase 4) — Updated dependencies and Click-based CLI

### Added

#### Training & Fine-tuning (Phase 2)

- Fine-tuning: Early stopping callback in `DNATrainer` with configurable `patience` and `threshold`
- Fine-tuning: Optuna hyperparameter search via YAML-configurable search space with auto-inferred distributions
- Fine-tuning: Training visualization with loss curves and learning rate schedule plotting utilities
- Fine-tuning: QLoRA 4-bit quantization support via `bitsandbytes` with automatic fix for improperly quantized layers
- Fine-tuning: Configurable gradient clipping integrated into `TrainingConfig`

#### MCP Server & Client (Phase 3)

- MCP: `dna_mutagenesis` tool exposing `Mutagenesis` class with 5 mutation types
- MCP: `dna_interpret` tool exposing `DNAInterpret` class with 8 Captum attribution methods
- MCP: Python client SDK (`DNALLMMCPClient`) with dual sync/async API for all 13 server tools
- MCP: Request timeout handling on all 13 tools with configurable `tool_timeout_seconds`
- MCP: Structured JSON/text dual-format logging for production observability

#### CLI & User Experience (Phase 4)

- CLI: `dnallm-mutagenesis` standalone command for in-silico mutation analysis
- CLI: `dnallm-train` migrated from `sys.argv` to Click framework with typed options and help text
- CLI: `dnallm-inference` migrated from `sys.argv` to Click framework with typed options and help text

#### Infrastructure & Tooling (Phase 1)

- QA: CI test execution restored with unified pytest configuration
- QA: Shared fixtures centralized in `tests/conftest.py`
- QA: GPU/CUDA tests re-enabled in CI
- QA: Mamba tests restored
- QA: Pre-commit hooks with ruff and mypy
- QA: Dependabot config for pip and GitHub Actions
- CHANGELOG.md initialized

### Changed

- Dependencies: numpy constraint relaxed to `>=1.26.0` (supports 1.x and 2.x)
- Dependencies: transformers upper bound removed (`>=4.49.0`)
- Dependencies: torch upper bound relaxed from `<=2.7` to `<2.12` for RTX 5090 support
- README: Installation instructions moved to prominent position near the top

### Fixed

- Architecture: `FocalLoss` moved to module level for proper importability
- Architecture: `.flake8` removed, fully migrated to ruff for linting
- Architecture: mypy ignore list pruned for stub-covered packages
- Type safety: `load_model_and_tokenizer` return type changed from `tuple[Any, Any]` to `tuple[PreTrainedModel, PreTrainedTokenizer]`

### Removed

- `.flake8` configuration file

## [0.4.0] - 2025-12-01

### Overview

Last stable release before the 0.5.x development cycle. Provided core fine-tuning and inference capabilities for DNA language models with support for multiple model architectures (DNABERT2, Nucleotide Transformer, GPN, HyenaDNA, etc.).

### Added

- Core fine-tuning pipeline with `DNATrainer` supporting classification, regression, and masked language modeling
- Multi-model architecture support via `AutoModel` integration
- Basic inference engine with batch processing
- Model zoo with 20+ pre-trained DNA language models

### Changed

- Initial project structure and package layout

### Fixed

- Various stability improvements for model loading and tokenization
