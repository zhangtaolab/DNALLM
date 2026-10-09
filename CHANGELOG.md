# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Metric registry contract at `dnallm.tasks.metric_registry`: a single {canonical: (fn, aliases)} registry with resolve()/canonical_name(); `dnallm.tasks.metrics` now emits exclusively canonical registry names, historical aliases (eval_auroc, eval_spearman_r, ...) are recognized but never emitted (REV-02, R1-2d)

### Changed

- Evaluation semantics: the trainer no longer silently evaluates on the test split when no dev split exists; evaluation is disabled with a loud warning unless `finetune.allow_test_as_eval: true` is set, and the new `DNATrainer.evaluate(split=...)` evaluates any split through the predict path writing a result JSON (REV-01, R1-2c)

## [0.7.1] - 2026-10-08

### Overview

Patch release recording the stabilization that landed on `dev` immediately after 0.7.0 — CI first-exposure fixes, a verifier-driven docs accuracy repair, and a tooling refresh. No API changes.

### Fixed

- Windows installs failing on `pybigwig` — `pygenometracks` gated behind a non-Windows platform marker in the notebook extra
- transformers >= 5.19 device-type query crashing the import chain on CUDA-built torch without a visible GPU — compat shim answering both observed signatures (torch >= 2.6 `RuntimeError` and torch <= 2.5 `AttributeError`)
- OS-native test assertions replacing Unix-only fd//proc assumptions
- CI ruff-format gate tripping on over-long fenced Python blocks in 3 docs pages — reformatted
- docs: 49-page verifier-driven accuracy repair retiring the stale `VERIFICATION_REPORT` snapshot

### Changed

- ruff dev dependency 0.16.9 → 0.16.10 (dependabot)

## [0.7.0] - 2026-10-07

### Overview

Example execution release: every artifact under `example/` now executes for real on the nightly GPU runner — 21 notebooks, 3 marimo apps, the helper script, every YAML config through real `load_config()` — with every surfaced error fixed by a same-change regression test. The PlantHelixSeek showcase notebooks reproduce frozen truth-agreement metrics over committed Arabidopsis loci, and the execution-test layer is formally nightly-gated. Milestone v1.1: 32/32 requirements, 5/5 phases verified, 0 audit blockers.

### Added

- Tests: private nbclient execution harness (tmp-sandbox cwd isolation, kernel-kill proof, partial-failure artifacts) executing the entire example tree — final census 196 passed / 1 benign skip / 0 failed on the nightly GPU runner
- CI: example-nightly job — staged-serial execution, hard census collection gate (197/206 pinned), >=35GiB hygiene floors between stages, `if: always()` artifact uploads, measured runtime budgets
- CI: `models.lock` grown to 24 revision-pinned rows with a fast-leg consistency guard proven by drift injection; `giants` marker deselects evo-class notebooks from the scheduled census
- Models: `PlantHelixSeek-CRE`/`-Anno` registry entries (frozen label order, generic route); `dnallm.utils.genomic_coords` coordinate/chrom-name normalization helpers (6 functions, 100% statement coverage)
- Data: committed <=200kb Arabidopsis showcase loci with truth slices, negative controls, and a frozen selection contract (floors + tolerance bands parsed by the tests at startup)
- Docs: executed showcase notebooks written back byte-identically into the docs mirror; coverage-expectation page documenting that example execution runs in kernel subprocesses and does not move the coverage gate

### Fixed

- `DNATokenizer` unknown-character crash in the megaDNA handler (library fix, RED-proven regression test)
- numpy 2.x `np.fromstring` binary-mode regression via a probe-gated shim (`dnallm.utils.transformers_compat`)
- transformers >= 5.19 device-type query crashing the import chain on CUDA-built torch without a visible GPU — both observed signatures (torch >= 2.6 `RuntimeError`, torch <= 2.5 missing `torch.accelerator` `AttributeError`) answered honestly via compat shim
- Windows installs failing on `pybigwig` (no Windows wheel): `pygenometracks` gated behind a non-Windows platform marker in the notebook extra
- Both v1 false-green CI gates closed together with the docs-mirror drift they hid; docs-validation now a required check on dev and main
- MCP server: single-flight concurrent-inference deadlock (fork-unsafe filelock) and the `dna_interpret` mamba-model crash (captum backward SIGKILLs the server)

## [0.6.0] - 2026-10-01

### Overview

Quality-engineering release: the pytest suite was audited end to end, test gaps closed, and line coverage driven from 45.92% to 96.30% behind a CI-enforced >90% hard gate. Every test result and coverage number from this repo is now trustworthy and cannot silently regress.

### Added

- CI: `fail_under = 90` coverage hard gate enforced through the pytest exit code — red-proven end-to-end via a synthetic-drop probe PR (all tests green, only the floor red)
- CI: two-job gated coverage pipeline — `coverage-gate` fast PR leg (push/PR) + `coverage-nightly` slow full-suite census on a self-hosted GPU runner with `models.lock`-keyed HF model cache; required-check branch protection on `dev` and `main`
- CI: `test-mamba` job moved to the self-hosted GPU runner on nightly cadence (schedule/workflow-dispatch only) with a 180-minute timeout backstop — it now actually executes instead of no-op skipping on GPU-less hosted runners
- CI: windows-latest fast-test leg (py3.12)
- Tests: ~1,000 new behavior-verifying tests (suite grew 464 → 1,657), coverage 45.92% → 96.30% on the agreed denominator (vendored code excluded)
- Tests: typed network skips with an expected-skip allowlist (`tests/expected_skips.yaml`) and a fail-closed skip audit (`scripts/audit_skips.py`) wired into 4 CI jobs — an unexpected skip fails the run instead of passing silently

### Fixed

- Test harness: removed the root `conftest.py` exit-code mask that made every failing run exit 0 — cleanup now propagates status via `pytest_sessionfinish`, with a permanent CI canary proving failing runs fail the job
- Test harness: removed `tests/pytest.ini` so both test roots (`tests/` and `dnallm/mcp/tests/`) are collected under the single `pyproject.toml` pytest config
- `compute_metrics` multiclass AUROC crash on absent-class batches — presence guard plus `labels=expected_classes` anchoring
- CrossDNA handler result was overwritten in the `load_model_and_tokenizer` dispatch chain — guarded first-resolved-wins chain with a sentinel regression test
- Five latent crashes in `inference/plot.py` / `inference/benchmark.py` (multilabel curve scalars, dict annotations, entropy shape, pydantic `Benchmark` init, `StratifiedKFold` labels)
- MCP server `_format_multi_model_results` misclassified every successful dict prediction as a failure
- PDF-generating tests now write artifacts under `tmp_path` (working tree stays clean); fixed `.gitignore` pdf-path typo

### Known Issues

Latent bugs discovered by the audit, pinned by tests, deferred to the next cycle: `raw_reverse_complement` is a no-op (`datahandling/data.py:983`); `cosine_similarity` loss raises TypeError (`models/model.py:264`); generate-from-DataLoader returns an empty list (`inference/inference.py:1643`); mutagenesis `max` strategy raises AttributeError (`inference/mutagenesis.py:429`).

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
