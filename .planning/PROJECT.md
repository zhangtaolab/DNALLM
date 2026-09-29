# DNALLM — Test Suite Audit & Coverage Hardening

## What This Is

DNALLM (`dnallm` v0.5.2) is a Python toolkit for fine-tuning, inference, and benchmarking of DNA language models (150+ pretrained models from HF/ModelScope), plus an MCP server exposing them to LLM agents. This cycle is a quality-engineering milestone on that existing codebase: audit the pytest suite end to end, close test gaps, and drive code coverage above 90% with a CI-enforced gate.

## Core Value

A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## Requirements

### Validated

Inferred from the existing codebase (see `.planning/codebase/`):

- ✓ YAML → Pydantic config loading (`dnallm/configuration/configs.py`)
- ✓ Registry-dispatch model loading for 35 model families (`dnallm/models/model.py`, `modeling_auto.py`, `special/`)
- ✓ Dataset handling: local files, HF/ModelScope, tokenization, augmentation, splitting (`dnallm/datahandling/`)
- ✓ Fine-tuning via HF Trainer + LoRA/QLoRA + Optuna (`dnallm/finetune/trainer.py`)
- ✓ Inference engine, interpretability, mutagenesis, benchmarking (`dnallm/inference/`)
- ✓ MCP server with 11 tools over stdio/SSE/streamable-HTTP (`dnallm/mcp/`)
- ✓ Existing pytest suite: 464 tests across `tests/` and `dnallm/mcp/tests/`
- ✓ Published to PyPI; CI matrix Python 3.11–3.13

### Active

- [ ] Run the full pytest suite (including `slow` tests) and produce a pass/fail/skip audit report
- [ ] Measure line coverage with pytest-cov over the whole `dnallm/` package, excluding vendored code (`dnallm/tasks/metrics/`, `dnallm/models/special/enformer_model/`) and unimportable adapters (`dnallm/finetune/megatron.py`, `dnallm/models/special/mamba_npu.py`)
- [ ] Produce a per-module coverage gap report (`term-missing`) identifying what to test
- [ ] Fix failing tests and real code bugs blocking coverage (known: multiclass AUROC crash in `dnallm/tasks/metrics.py:283`, CrossDNA handler result overwritten in `dnallm/models/model.py:873-887`)
- [ ] Write new tests until coverage exceeds 90% on the agreed denominator
- [ ] Enforce the gate in CI: `--cov-fail-under=90` on a run that includes `slow` tests (network model downloads accepted)

### Out of Scope

- Vendored code coverage (`dnallm/tasks/metrics/`, `enformer_model/`) — upstream HF `evaluate` / ported Enformer, excluded from lint/mypy by design
- `megatron.py` / `mamba_npu.py` test coverage — require Megatron-LM / Ascend NPU toolchains that cannot import in CI
- Root `cli/` legacy launcher cleanup — packaging concern (CONCERNS.md), not needed for coverage
- mypy `|| true` CI fix, dependency lockfile, other CONCERNS items — separate quality work
- Performance optimization (e.g. `attn_implementation` hardcoding) — record, don't fix

## Context

- Current test config lives in `pyproject.toml [tool.pytest.ini_options]` (NOT pytest.ini): `--asyncio-mode=auto`, `--timeout=300`, `--strict-markers`; markers `slow`, `pdf`, `performance`, `integration`; testpaths `tests/` + `dnallm/mcp/tests/`
- `pytest-cov>=6.0.0` is already a dev dependency but no `--cov*` flags or threshold exist anywhere
- Coverage measurement must include `slow` tests (real model downloads) per owner decision; CI gate run therefore needs network access and accepts long runtimes
- Known skip: multiclass AUROC test explicitly skipped (`tests/tasks/test_metrics.py:761`) because `compute_metrics` crashes — fixing this is in scope
- transformers compatibility spans 4.49–5.x via `dnallm/utils/transformers_compat.py`; installed dev env uses transformers 5.17, torch 2.11 cu130
- Shared mock fixtures exist in `tests/conftest.py` (mock_model/mock_tokenizer/mock_config) — reuse these patterns for new tests
- Codebase map with full concerns list: `.planning/codebase/` (STACK, ARCHITECTURE, TESTING, CONCERNS)

## Constraints

- **Tech stack**: pytest + pytest-cov; coverage configured via `[tool.coverage.run]` omit list in `pyproject.toml` — no new test frameworks
- **Compatibility**: suite must keep passing on the CI matrix (Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0); tests must not pin to a single transformers minor version
- **CI**: coverage-gated run includes `slow` tests — requires network for model downloads; runtime cost accepted by owner
- **Scope**: bug fixes limited to what correctness/coverage requires; no refactors beyond that

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Coverage denominator: whole `dnallm/` excluding vendored dirs and unimportable adapters | Vendored code is upstream and excluded from lint/mypy; adapters cannot import in CI — including them makes 90% unattainable | — Pending |
| Audit first, then fix | Gap report drives test-writing priorities and surfaces real bugs before mass test authoring | — Pending |
| CI hard gate `--cov-fail-under=90`, run includes slow tests | Prevents coverage regression; owner accepts network downloads and longer CI runs for real coverage | — Pending |
| Fix real code bugs encountered during audit (AUROC, CrossDNA) | Skipped-crash tests hide real defects; unskipping them is required for honest coverage | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-29 after initialization*
