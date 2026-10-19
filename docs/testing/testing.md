---
title: Testing
---

<!-- generated-by: gsd-doc-writer -->
# Testing Overview

DNALLM's test suite is a single pytest suite with a single configuration: collection roots, markers, timeouts, async mode, and the coverage floor all live in `[tool.pytest.ini_options]` and `[tool.coverage.*]` in [`pyproject.toml`](https://github.com/zhangtaolab/DNALLM/blob/main/pyproject.toml). The suite divides into a **fast lane** (`-m "not slow"`) that runs on every push and pull request, and a **slow lane** of real-model tests (real Hugging Face / ModelScope downloads) exercised by the nightly coverage census. Every skip the suite produces is audited against a typed allowlist, and every `--cov` run is held to a 90% floor.

This page is the overview. Two companion documents carry the authoritative detail:

- [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md) — the canonical in-repo testing guide: the full marker table, lane commands, configuration rationale, and troubleshooting.
- [Continuous Integration](../user_guide/continuous_integration.md) — the CI lane story: nightly topology, coverage-gate semantics, the giants policy, and the typed-skip audit contract.

## Test framework and setup

The framework is **pytest** (`pytest>=8.4`, enforced by `minversion = "8.4"`) with a small plugin set, installed through the `test` extra:

```bash
pip install -e ".[test]"   # the pytest toolchain only
pip install -e ".[full]"   # what CI installs: dev + test + notebook + mcp extras
```

| Tool | Role |
|------|------|
| `pytest-asyncio` | `--asyncio-mode=auto` is baked into `addopts`, so async tests need no marker |
| `pytest-cov` + `coverage[toml]` | coverage rides the pytest exit code via `fail_under` (see below) |
| `pytest-timeout` | global `--timeout=300`; per-test `@pytest.mark.timeout` marks raise it for long example executions |
| `pytest-progress` | progress rendering (CI gate legs disable it with `-p no:progress`) |
| `nbclient` | executes example notebooks in IPython kernel subprocesses |

Configuration facts that surprise newcomers:

- **Single source of truth.** There is deliberately no `pytest.ini`, `tox.ini`, `setup.cfg`, or `.coveragerc` anywhere in the tree — a config file found at a test root takes precedence over `pyproject.toml` and would silently hijack the settings. Never (re)create `tests/pytest.ini`; edit `pyproject.toml`.
- **Two collection roots.** `testpaths = ["tests", "dnallm/mcp/tests"]` — the packaged MCP suite (including the live-server probes) is collected alongside the main tree.
- **Strict by default.** `--strict-markers --strict-config` make an unregistered marker or bad config an error, and `addopts` already includes `-v --tb=short`, so a plain `pytest` is fully configured.
- **Session cleanup.** The root [`conftest.py`](https://github.com/zhangtaolab/DNALLM/blob/main/conftest.py) kills stray multiprocessing children, releases PyTorch resources, and garbage-collects at session end while preserving the real exit status. It prints emoji at session start, which is why the Windows CI leg sets `PYTHONUTF8=1`.

## Suite layout

`tests/` mirrors the `dnallm/` package, plus a few suite-specific directories:

- Module directories: `benchmark/`, `cli/`, `configuration/`, `datahandling/`, `finetune/`, `inference/`, `interpret/`, `mcp/`, `models/`, `tasks/`, `utils/`
- `examples/` — the example/notebook/script execution census (nightly lane; per-test timeout marks in the 3600–7200s range)
- `scripts/` — tests for the CI scripts themselves (`audit_skips.py`, `check_docs_sync.py`)
- Root contract tests — `test_models_lock_contracts.py`, `test_extras_guard.py`, `test_config_mapping_contracts.py`, `test_runner_infra_contracts.py`, `test_dispatch_direction.py`, `test_frozen_pairs.py`, `test_genomic_intervals.py`
- `test_data/` and `fixtures/` — fixture datasets per task type (binary, multiclass, multilabel, regression, token classification, embedding, language modeling) and genomic-interval fixtures
- `expected_skips.yaml` — the typed skip allowlist consumed by the skip audit
- Second root `dnallm/mcp/tests/` — MCP client/server/model-manager tests plus the six slow-marked live-server probes

The full annotated tree is in [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md).

## Running tests

Run from the project root:

```bash
# Fast lane — the selection CI runs on push/PR (no slow real-model tests)
pytest -m "not slow"

# Full suite — includes the slow lane (real model downloads; this is the nightly census)
pytest

# Scoped runs
pytest tests/inference/test_plot.py                          # one file
pytest tests/utils/test_sequence.py::test_reverse_complement # one test
pytest tests/inference/ --no-cov                             # subset, no coverage gate

# Coverage runs (the 90% floor applies to every one of these)
pytest -m "not slow" --cov               # CI-shaped invocation
pytest --cov=dnallm --cov-report=html    # local HTML report
```

Marker selection uses standard pytest syntax (`-m "data and not slow"`, `-m pdf`, …). Markers are registered in `pyproject.toml`; of the registered set, only `slow`, `pdf`, `data`, and `giants` currently select tests — the full marker table and combination examples live in [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md).

Two lane-specific behaviors worth knowing:

- The MCP live-server probes (`dnallm/mcp/tests/test_sse_client.py`, `test_streamable_http_client.py`) target a `dnallm-mcp-server` on `localhost:8000`. No CI job starts one there, so they skip deterministically with the typed `network-unavailable:` reason; they are local-only.
- Giant-model (evo-class) tests carry the `giants` marker, which the example nightly deselects by owner policy; run them explicitly with `pytest -m giants`.

## Writing new tests

Conventions:

- **Naming**: `test_*.py` files, `Test*` classes, `test_*` functions (matching `python_files` / `python_classes` / `python_functions`).
- **Placement**: the directory mirroring the module under test — a test for `dnallm/utils/sequence.py` goes in `tests/utils/`.
- **Markers**: any new marker must be registered in the `markers` list in `pyproject.toml` — `--strict-markers` turns an unregistered marker into an error.
- **Timeouts**: the global default is 300s; tests expected to run longer carry their own `@pytest.mark.timeout` mark.

Shared fixtures live in [`tests/conftest.py`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/conftest.py). The suite's philosophy is deterministic *real* objects over pure mocks:

- `simple_dna_tokenizer` — a character-level DNA tokenizer with real encode/decode, an HF-compatible call signature, and `transformers.BatchEncoding` output (maps `N` to the mask id so MLM paths are exercised deterministically).
- `tiny_real_model` / `tiny_model_factory` — `TinyDNAModel`, a seeded, autograd-capable embedding+linear backbone with `.logits` / `.hidden_states`, shapeable for sequence-level (`pooled=True`) or token-level tasks.
- `inference_config_factory` — writes real YAML under `tmp_path` and loads it through `load_config`, so every test exercises the real Pydantic validation and alias normalization.
- `mock_model`, `mock_tokenizer`, `mock_config`, `mock_inference_engine`, `mock_dataset`, `sample_dna_sequence` — Mock-based and plain-data fixtures for paths where a real object is not needed.

**Skips are a contract.** Every skip in the junit artifact must match exactly one typed entry in [`tests/expected_skips.yaml`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/expected_skips.yaml) (typed prefixes such as `network-unavailable:`, `environment-unavailable:`, `optional-dep:`, `clinvar-unavailable:`), or [`scripts/audit_skips.py`](https://github.com/zhangtaolab/DNALLM/blob/main/scripts/audit_skips.py) fails the job. The audit is fail-closed in both directions: an unexpected skip fails, and a missing or malformed allowlist file fails rather than widening the allowlist. A new intentional skip means a new typed allowlist entry with a category — never a wildcard or empty entry.

## Coverage requirements

Coverage is configured in `[tool.coverage.run]` / `[tool.coverage.report]` in `pyproject.toml`:

| Scope | Threshold | Enforcement |
|-------|-----------|-------------|
| Line coverage of the `dnallm` package | `fail_under = 90` | Rides the pytest exit code on **every** `--cov` invocation, local or CI |

Key semantics:

- The 90% value is a **ratchet floor, not the achievement** — the fast lane measured 96.72% at the v1.2 closeout (2026-10-10; it was 96.30% when the gate landed). Hold new work at or above the current total rather than at the floor.
- The floor applies regardless of how many tests ran — for scoped runs, drop `--cov` or pass `--no-cov`.
- Omitted from measurement (the `omit` list): the vendored HF evaluate copy (`dnallm/tasks/metrics/`), the ported Enformer, the unimportable adapters (`dnallm/finetune/megatron.py`, `dnallm/models/special/mamba_npu.py`), and the packaged MCP tests/helper scripts.
- **Kernel subprocesses are not measured, by design.** The example tests run notebooks in IPython kernel subprocesses, so the example lane certifies that notebooks executed end to end — it does not move the coverage number, and the coverage number says nothing about notebook executability. The full design note is in [Continuous Integration](../user_guide/continuous_integration.md).
- There is no codecov upload and no XML coverage report in CI; totals are reported in the job terminal only.

## CI integration

All test CI lives in [`.github/workflows/ci.yml`](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/ci.yml). The push/PR legs run on hosted runners:

- **`test`** — Ubuntu matrix over Python 3.11/3.12/3.13 (NumPy pinned to 2.2.0): `ruff format --check` and `ruff check` first, then `pytest -m "not slow" --cov --junitxml`, then the skip audit and an exit-code canary (an intentionally failing test must make pytest exit non-zero).
- **`test-windows`** — Python 3.12 fast leg mirroring `test` (with `PYTHONUTF8=1` and git `autocrlf` off); Linux-only checks (SONAME probe, `/proc` fd accounting) typed-skip here by allowlist entry.
- **`test-cuda`** — Python 3.11 against a CUDA 13.0 toolkit install, running `pytest tests/ -m "not slow"`.
- **`coverage-gate`** — Python 3.12 fast census with full junit and durations, the skip audit, and the pinned-stub `ty` type-check hard gate (static checks stay out of pytest).

The nightly legs run on the self-hosted `dnallm-nightly` GPU runner, gated to schedule and manual `workflow_dispatch` only (push/PR code never reaches that box): `coverage-nightly` (03:00 UTC — the full census including slow tests, under the 90% floor), `test-mamba` (03:00 UTC — source-built mamba-ssm/causal-conv1d kernel leg), and `example-nightly` (05:30 UTC — the staged example-execution census with `-m "not giants"`). The stage tables, timeout arithmetic, and per-job detail are documented in [Continuous Integration](../user_guide/continuous_integration.md) and [`.github/workflows/README.md`](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/README.md).

## Related pages

- [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md) — full testing guide (markers, commands, troubleshooting, skip-audit walkthrough)
- [Continuous Integration](../user_guide/continuous_integration.md) — CI lanes, coverage-gate semantics, giants policy
- [Architecture Overview](../architecture/overview.md) — the module layout the test tree mirrors
