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

Shipped in Phase 1 (Harness Integrity & Measured Baseline, 2026-09-30):

- ✓ Full-suite audit report with pass/fail/skip census (625/0/0/9, both roots, `slow` included) — Phase 1 (`01-AUDIT-REPORT.md`)
- ✓ Measured line coverage on the agreed denominator (whole `dnallm/` minus vendored dirs, unimportable adapters, packaged test files; 7-entry omit list in `pyproject.toml`) — Phase 1
- ✓ Per-module coverage gap report (`term-missing` + `coverage.json`, 43-row ranked worklist) — Phase 1
- ✓ Honest harness: single pytest config (`pyproject.toml` only), real exit codes (mask removed, permanent CI canary), measured baseline **45.92%** — Phase 1

Shipped in Phase 2 (Suite Hygiene & Known-Bug Fixes, 2026-09-30):

- ✓ Multiclass AUROC fixed and unskipped — presence-guard ValueError + `labels=expected_classes`, both crash-skips removed (38/0 census) — Phase 2
- ✓ CrossDNA handler result survives dispatch — guarded first-resolved-wins chain, sentinel regression test, 12-handler audit (1 bug) — Phase 2
- ✓ Every skip typed and allowlisted — `network-unavailable:`-prefixed typed network skips, `expected_skips.yaml` + `scripts/audit_skips.py` CI gate, full run 623/7/0 with audit exit 0 — Phase 2
- ✓ PDF tests leave the tree clean — autouse tmp_path rebind, gitignore fixed, 9 strays deleted — Phase 2

### Active

None — all milestone requirements delivered (see Validated, Phases 3–4).

Shipped in Phase 3 (Coverage Waves, 2026-10-01):

- ✓ Write new tests until coverage exceeds 90% on the agreed denominator — **96.30%** (7,131/7,405 stmts, verifier-reproduced at HEAD d152d12; ~1,000 behavior tests across 5 ranked-worklist waves; pragma held at 3; 7 allowlisted skips; 8 latent source bugs fixed en route)

Shipped in Phase 4 (CI Gate Enforcement, 2026-10-01):

- ✓ Enforce the gate in CI: `fail_under=90` in `[tool.coverage.report]` enforced through the pytest exit code; two-job CI (fast-leg `coverage-gate` on push/PR + slow-census `coverage-nightly` on a self-hosted GPU runner, both bare `--cov` against the same pyproject); gate green at 96.27–96.30%, red-proven via probe PR #39 (78.91% < floor, 1261 tests otherwise green); required-check branch protection on dev+main; nightly census verified green end to end (run 36811033498: 1656 passed / 7 allowlisted / 0 failed); re-verification passed 10/10

### Out of Scope

- Vendored code coverage (`dnallm/tasks/metrics/`, `enformer_model/`) — upstream HF `evaluate` / ported Enformer, excluded from lint/mypy by design
- `megatron.py` / `mamba_npu.py` test coverage — require Megatron-LM / Ascend NPU toolchains that cannot import in CI
- Root `cli/` legacy launcher cleanup — packaging concern (CONCERNS.md), not needed for coverage
- mypy `|| true` CI fix, dependency lockfile, other CONCERNS items — separate quality work
- Performance optimization (e.g. `attn_implementation` hardcoding) — record, don't fix

## Context

- Current test config lives in `pyproject.toml [tool.pytest.ini_options]` (NOT pytest.ini): `--asyncio-mode=auto`, `--timeout=300`, `--strict-markers`; markers `slow`, `pdf`, `performance`, `integration`; testpaths `tests/` + `dnallm/mcp/tests/`
- `pytest-cov>=6.0.0` is a dev dependency; since Phase 4 the enforcement surface is `fail_under = 90` in `[tool.coverage.report]` (pyproject.toml) — a bare `--cov` on any invocation activates it; CI census jobs run exactly that
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
| Coverage denominator: whole `dnallm/` excluding vendored dirs and unimportable adapters | Vendored code is upstream and excluded from lint/mypy; adapters cannot import in CI — including them makes 90% unattainable | ✓ Landed Phase 1 (7-entry omit list; baseline 45.92% on 7,383 stmts) |
| Audit first, then fix | Gap report drives test-writing priorities and surfaces real bugs before mass test authoring | ✓ Landed Phase 1 (43-row ranked worklist from measured artifacts) |
| CI hard gate `--cov-fail-under=90`, run includes slow tests | Prevents coverage regression; owner accepts network downloads and longer CI runs for real coverage | ✓ Landed Phase 4 (fail_under=90 native via pyproject; green 96.27–96.30%; red-proven PR #39; branch protection on dev+main) |
| GATE-02 amended: PR gate = fast leg; slow census = nightly on self-hosted GPU runner | Hosted 360-min cap killed the 6h CPU census; org cannot use larger hosted runners | ✓ Landed Phase 4 (census ~15 min warm on `dnallm-nightly`; dispatch/cron only, fork PRs can't reach it) |
| Fix real code bugs encountered during audit (AUROC, CrossDNA) | Skipped-crash tests hide real defects; unskipping them is required for honest coverage | ✓ Landed Phase 2 (both fixed, regression-tested, unskipped) |
| Subprocess coverage: start minimal, escalate only on canary evidence (Phase 1) | pytest-cov 7 removed `.pth` subprocess auto-measurement; no collected test spawns subprocesses | ✓ Landed Phase 1 (AUDIT-04; escalation trigger recorded) |

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
*Last updated: 2026-10-01 after Phase 4*
