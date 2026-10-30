---
title: Development Guide
---

<!-- generated-by: gsd-doc-writer -->

# Development Guide

This page is the on-ramp for working on the DNALLM codebase itself: setting up a
development environment, finding your way around the repository, running the test
suite and quality gates locally, and following the project's branching and PR
conventions. It gives you the shortest path to a green local check; the complete
pre-commit checklist and PR process live in
[`CONTRIBUTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md),
and the CI picture lives in
[Continuous Integration](../user_guide/continuous_integration.md).

## Local Setup

**Prerequisites**

- Python >= 3.11 (CI matrix covers 3.11, 3.12, and 3.13)
- Git
- [uv](https://docs.astral.sh/uv/) package manager (recommended; the project pins
  `ruff==0.17.1` and invokes `ty` through `uvx`, so uv is effectively required for
  the quality gates)
- A CUDA-compatible GPU is optional — the fast test lane runs on CPU

**Install**

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/your-username/DNALLM.git
cd DNALLM
git remote add upstream https://github.com/zhangtaolab/DNALLM.git

# 2. Create and activate a virtual environment
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install in editable mode with the dev extras
uv pip install -e '.[dev]'
```

If you will touch the MCP example tests, install the `mcp` extra too — it provides
the langchain/pydantic-ai imports those tests need (see the README testing section):

```bash
uv pip install -e '.[test,dev,mcp]'
```

`scripts/setup_uv.sh` bundles the same flow (uv install, `uv venv`,
`uv pip install -e ".[dev,notebook,mcp]"`, `uv lock`) if you prefer one command.
GPU-specific torch lanes (`cpu`, `cuda124`, `cuda126`, `cuda128`, `cuda130`) and the
`mamba`, `deepspeed`, and `flash-attn` extras are documented in the
[Installation guide](../getting_started/installation.md) and `pyproject.toml`.

**Verify the install**

```bash
pytest tests/ -m "not slow"   # fast lane, the same selection CI runs
```

### Development extras

Optional-dependency groups from `pyproject.toml` relevant to contributors:

| Extra | Contains |
|-------|----------|
| `dev` | `test` + `notebook` + `ruff==0.17.1`, `pre-commit`, plotting/data extras (logomaker, seaborn, pyfastx, pybedtools on non-Windows) |
| `test` | pytest stack (`pytest>=8.4`, `pytest-asyncio`, `pytest-cov`, `pytest-progress`, `pytest-timeout`), `coverage[toml]`, `nbclient`; pulls in `tui` (Textual) for the preflight-wizard tests |
| `notebook` | jupyter, marimo, `nbclient` (with the IPython < 9 and pygenometracks platform pins) |
| `docs` | mkdocs-material, mkdocs-jupyter, mkdocstrings-python |
| `tui` | Textual, for the `dnallm preflight` terminal wizard |
| `full` | `dev` + `test` + `notebook` + `mcp` + `isort` |
| `all` | `full` + `dev` + `test` + `notebook` + `docs` + `ui` + `mcp` + `tui` |
| `mamba` | native CUDA kernels (`mamba-ssm==2.3.2.post1`, `causal_conv1d==1.7.0`), the end-user torch < 2.12 fast path |
| `deepspeed` | `deepspeed>=0.19.6,<0.19.7` — GENA-LM-BigBird remote code; band-capped because 0.19.7 removed `deepspeed.ops.sparse_attention` upstream |
| `flash-attn` | `flash-attn>=2.8.3.post2,<2.9` — CrossDNA attention; source installs need `--no-build-isolation` |

The `dev` extra is enough for package work; `full` is the CI environment shape
(every `ci.yml` job installs `.[full]`).

Two v1.4 changes to know about:

- **The `fla` extra is retired.** `flash-linear-attention` (the KDA kernels
  PlantHelixSeek's remote code imports) is now a base dependency, so a default
  install works out of the box — no extra suffix names it.
- **`deepspeed` and `flash-attn` are new curated extras** for the two
  external-package models, and they are deliberately never members of
  `all`/`full` (Linux + CUDA compile packages; pip fails loudly where the build
  cannot run). CI builds the kernels through the sha256-pinned installer
  scripts (`scripts/install_mamba.sh`, `scripts/install_flash_attn.sh`) rather
  than the extras. See
  [Optional Dependencies](../user_guide/optional_dependencies.md) for the full
  story.

## Repository Layout

```
DNALLM/
├── dnallm/            # The package
│   ├── cli/           # CLI entry points (dnallm, dnallm-train, dnallm-inference, ...)
│   ├── configuration/ # YAML config loading (load_config)
│   ├── datahandling/  # DNADataset and data processing
│   ├── finetune/      # Training pipeline (DNATrainer)
│   ├── inference/     # Inference, VEP, mutagenesis, plotting
│   ├── interpret/     # Model interpretation tooling
│   ├── mcp/           # MCP server plus its packaged test suite
│   ├── models/        # Model registry, loading, presets
│   ├── preflight/     # Read-only pre-flight validation (data, compatibility,
│   │                  # config generation) — torch/transformers-free
│   ├── tasks/         # Task heads and metrics
│   ├── tui/           # Terminal pre-flight wizard (Textual, tui extra)
│   └── utils/         # Shared utilities
├── tests/             # Pytest suite mirroring the package, plus TESTING.md
│                      # and expected_skips.yaml (the typed skip allowlist)
├── example/           # Notebooks, marimo apps, MCP examples — executed by the
│                      # nightly example census
├── configs/           # Example YAML configurations
├── docs/              # This mkdocs site's source
├── scripts/           # Dev/CI scripts (check_code.py, audit_skips.py, ...)
├── ui/                # Gradio web interfaces
├── cli/               # Legacy CLI scripts (deprecated)
├── run_cli.py         # Local CLI launcher
├── conftest.py        # Root conftest shared by the suite
├── pyproject.toml     # Dependencies, extras, and all tool config (ruff, ty,
│                      # pytest, coverage)
└── mkdocs.yml         # Documentation site nav
```

Two layout facts worth knowing early:

- **All tool configuration lives in `pyproject.toml`** — ruff, ty, pytest
  (`[tool.pytest.ini_options]`), and coverage. Do not add a `pytest.ini`; it would
  take precedence over `pyproject.toml` and hijack the suite's settings.
- **`tests/` mirrors `dnallm/`**, and pytest collects a second root,
  `dnallm/mcp/tests/` (the packaged MCP suite). Root-level `test_*_contracts.py`
  files guard repository invariants such as the extras set and `models.lock`.

## Running the Test Suite Locally

The fast lane — everything except the `slow` (real-model download) tests — is what
CI runs on every push and PR, and it is what you should run before committing:

```bash
pytest tests/ -m "not slow" --cov=dnallm --cov-report=term-missing
```

Common variations:

```bash
pytest tests/inference/ --no-cov          # one module, no coverage
pytest tests/utils/test_sequence.py -v    # one file
pytest -m unit                            # by marker
pytest                                   # everything, including slow tests
                                          # (network + GPU heavy)
```

Three contract details that trip new contributors:

- **The 90% coverage floor applies to every `--cov` invocation**, regardless of how
  many tests ran (`fail_under = 90` in `[tool.coverage.report]`, a ratchet floor —
  the fast lane actually measures in the mid-90s). Scoped runs must drop `--cov` or
  pass `--no-cov`, or they fail on the floor.
- **On local Python 3.13 + numpy 2.5.x stacks, pytest-cov has been observed to
  break** (CI is unaffected — it pins `numpy==2.2.0`). The equivalent route is a
  plain coverage run, which reads the same `[tool.coverage.*]` config and enforces
  the same floor:

  ```bash
  coverage run -m pytest tests/ -m "not slow"
  coverage report          # same fail_under = 90 gate
  ```

- **Skips are audited.** Any skip the suite produces is checked by
  `scripts/audit_skips.py` against the typed allowlist in
  `tests/expected_skips.yaml`; an unexpected skip fails CI. If a test starts
  skipping in your environment, triage it — fix it or add a typed allowlist entry.

Markers (`slow`, `unit`, `integration`, `giants`, ...), fixture conventions, and
per-lane details are documented in
[`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md).

## Code Quality Gates

Three gates run on every PR; all are configured in `pyproject.toml` and all must
pass before you commit.

### Ruff — format and lint

```bash
ruff format .            # write
ruff format --check .    # verify
ruff check . --statistics
```

Ruff is pinned (`ruff==0.17.1`) in the `dev` extra so local and CI agree. Project
settings: line length 100, double quotes, Python 3.11 target, preview mode, with
rule families E4/E7/E9/F/W/B/C4/UP/N/S/T20/PT/Q/RUF enabled. The MCP module
additionally keeps a flake8 79-character rule via the legacy shell/batch check
scripts; CI itself runs no flake8 step.

### Pre-commit hooks

`.pre-commit-config.yaml` defines exactly two local hooks, both running the venv's
pinned ruff: `ruff-format` and `ruff-check`. Install them once per clone:

```bash
pre-commit install             # wire the hooks into git
pre-commit run --all-files     # run both hooks over the whole tree
```

The ty type gate is deliberately **not** a pre-commit hook — it must run through
the exact-pinned `uvx` invocation below, so it stays in `check_code.py` and CI
only.

### ty — the type gate

```bash
uvx ty@0.0.84 check dnallm/
```

The invocation is exact-pinned on purpose — never run a floating `ty`. This is a
**hard gate** (the sole type gate since 2026-10-17; mypy is retired), enforced by
the CI `coverage-gate` job against a pinned stub environment
(`.github/workflows/ty-stub-constraints.txt`) so upstream releases cannot drift
the result. Locally, the plain `uvx` invocation above is the equivalent check.

If a diagnostic is verified library dynamics, silence it with a line-level
`# ty: ignore[rule]` plus a root-cause comment. Do not introduce coded mypy-syntax
`# type: ignore[code]` comments — they are inert under ty. There are no severity
overrides; vendored-code excludes (`dnallm/tasks/metrics/`, `mamba_npu.py`,
`megatron.py`) are listed in `[tool.ty.src]`.

### check_code.py — one command before every push

[`scripts/check_code.py`](https://github.com/zhangtaolab/DNALLM/blob/main/scripts/check_code.py)
runs the CI-matching sequence in one shot — from the project root, with the venv
active:

```bash
python scripts/check_code.py          # ruff format + ruff lint + fast tests w/ coverage + ty
python scripts/check_code.py --fix    # auto-fix ruff issues first
python scripts/check_code.py --all    # include slow tests (~10 minutes)
python scripts/check_code.py --docs   # additionally run docs code verification
python scripts/check_code.py --ci     # exact CI simulation (no coverage extras)
python scripts/check_code.py --verbose  # show full output from every check
```

The four steps match the PR lane: ruff format check, ruff lint, fast tests with
coverage, and the pinned ty gate. `check_code.sh` / `check_code.bat` are the older
platform equivalents (they add the legacy flake8 MCP check). The full pre-commit
checklist, in the order CONTRIBUTING.md prescribes, is in
[`CONTRIBUTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md#pre-commit-checklist).

## Branching and Commits

The repository runs a two-branch integration model, visible in the merge history:

- **`main`** — the default branch; releases land here.
- **`dev`** — the integration branch; feature work accumulates here and is merged
  to `main` via PR (e.g. PR #58, the v1.3.1 release train).
- **Release branches** named `vX.Y` (currently `v1.4`) carry stabilized releases;
  tags `vX.Y.Z` mark published versions.
- **Topic branches** for features and fixes — CONTRIBUTING.md documents
  `feature/your-feature-name` and `fix/issue-description`; dependabot branches
  (`dependabot/pip/...`) follow the bot's own scheme.

Commit messages follow Conventional Commits with scopes, as the history shows:
`feat(models):`, `fix(utils):`, `test(finetune):`, `docs(...)`, `ci(...)`,
`chore(release):`, `type(...)`, `style(...)`, `deps(...)`. Match the pattern of
recent commits in the area you touch.

## Pull Requests

Before opening a PR, `python scripts/check_code.py` must pass — it reproduces what
CI will run on your PR. The push/PR-gated lanes in `ci.yml` are:

- **`test`** — the hosted matrix, Python 3.11/3.12/3.13 × `numpy==2.2.0`, fast
  lane with coverage plus the skip audit and an exit-code canary
- **`test-windows`** — a Python 3.12 Windows leg of the same shape
- **`test-cuda`** — a hosted Python 3.11 + CUDA 13.0 leg (`.[full,cuda130]`)
- **`coverage-gate`** — Python 3.12: gated fast census with the coverage floor,
  skip audit, and the pinned ty hard gate

Nightly and dispatch-only lanes never gate PRs: `coverage-nightly`,
`example-nightly`, and `test-mamba` run on the self-hosted `dnallm-nightly` GPU
box (03:00/05:30 UTC crons), `extras_compat.yml` owns the 04:00 UTC extras lane
(deepspeed + flash-attn probes for the two external-package models), and
`census_compat.yml` / `feasibility.yml` are manual-dispatch-only re-verification
lanes (the transformers 4/5 census among them). See
[Continuous Integration](../user_guide/continuous_integration.md) for what runs
where.

The PR description template, review process, and the full submission checklist
are maintained in
[`CONTRIBUTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md#pull-request-process) —
follow that document as the authority.

## Working on Documentation

This site is MkDocs (Material theme) with `mkdocs-jupyter` for the example
notebooks:

```bash
uv pip install -e '.[docs]'
mkdocs build     # build to site/
mkdocs serve     # live preview at http://127.0.0.1:8000
```

`mkdocs.yml` watches `example/notebooks`, `example/marimo`, and `example/mcp_example`
in addition to `docs/`. Code snippets and config references inside documentation
pages are themselves checked — `check_code.py --docs` runs `scripts/verify_docs.py`,
and the `docs-validation` workflow runs `scripts/check_docs_sync.py`,
`scripts/validate_docs_snippets.py`, and `scripts/validate_yaml.py` — so keep
examples runnable.

## Where to Go Next

- [`CONTRIBUTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/CONTRIBUTING.md) —
  the full pre-commit checklist, code style standards, and PR process
- [`tests/TESTING.md`](https://github.com/zhangtaolab/DNALLM/blob/main/tests/TESTING.md) —
  test suite structure, markers, and per-lane contracts
- [Continuous Integration](../user_guide/continuous_integration.md) — nightly
  topology, coverage gate semantics, giants policy, skip audit
- [`.github/workflows/README.md`](https://github.com/zhangtaolab/DNALLM/blob/main/.github/workflows/README.md) —
  the authoritative per-job workflow documentation
- [Architecture Overview](../architecture/overview.md) — how the package's modules
  fit together
- [Configuration System](../configuration/configuration.md) — how YAML configs
  flow through the codebase
