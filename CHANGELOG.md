# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-05-08

### Added

- Fine-tuning: Early stopping callback in DNATrainer with configurable patience and threshold (Phase 2)
- Fine-tuning: Optuna hyperparameter search via YAML-configurable search space with auto-inferred distributions (Phase 2)
- Fine-tuning: Training visualization with loss curves and learning rate schedule plotting utilities (Phase 2)
- Fine-tuning: QLoRA 4-bit quantization support via bitsandbytes with automatic fix for improperly quantized layers (Phase 2)
- Fine-tuning: Configurable gradient clipping integrated into training config (Phase 2)
- MCP: `dna_mutagenesis` tool exposing Mutagenesis class with 5 mutation types (Phase 3)
- MCP: `dna_interpret` tool exposing DNAInterpret class with 8 Captum attribution methods (Phase 3)
- MCP: Python client SDK (`DNALLMMCPClient`) with dual sync/async API for all 13 server tools (Phase 3)
- MCP: Request timeout handling on all 13 tools with configurable `tool_timeout_seconds` (Phase 3)
- MCP: Structured JSON/text dual-format logging for production observability (Phase 3)
- CLI: `dnallm-mutagenesis` standalone command for in-silico mutation analysis (Phase 4)
- CHANGELOG.md initialized (Phase 4)

### Changed

- Dependencies: numpy constraint relaxed to `>=1.26.0` (supports 1.x and 2.x) (Phase 4)
- Dependencies: transformers upper bound removed (`>=4.49.0`) (Phase 4)
- CLI: `dnallm-train` migrated from `sys.argv` to Click framework with typed options and help text (Phase 4)
- CLI: `dnallm-inference` migrated from `sys.argv` to Click framework with typed options and help text (Phase 4)
- README: Installation instructions moved to prominent position near the top (Phase 4)

### Fixed

- QA: CI test execution restored with unified pytest configuration (Phase 1)
- QA: Shared fixtures centralized in `tests/conftest.py` (Phase 1)
- QA: GPU/CUDA tests re-enabled in CI (Phase 1)
- QA: Mamba tests restored (Phase 1)
- Architecture: `FocalLoss` moved to module level for proper importability (Phase 1)
- Architecture: `.flake8` removed, fully migrated to ruff for linting (Phase 1)
- Architecture: mypy ignore list pruned for stub-covered packages (Phase 1)
- Type safety: `load_model_and_tokenizer` return type changed from `tuple[Any, Any]` to `tuple[PreTrainedModel, PreTrainedTokenizer]` (Phase 4)

### Removed

- `.flake8` configuration file (Phase 1)

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
