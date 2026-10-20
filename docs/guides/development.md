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
`mamba` / `fla` kernel extras are documented in the
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
| `test` | pytest stack (`pytest>=8.4`, `pytest-asyncio`, `pytest-cov`, `pytest-timeout`), `coverage[toml]`, `nbclient` |
| `notebook` | jupyter, marimo, `nbclient` (with the IPython < 9 and pygenometracks platform pins) |
| `docs` | mkdocs-material, mkdocs-jupyter, mkdocstrings-python |
| `full` | `dev` + `test` + `notebook` + `mcp` + isort (renamed from `base` in v1.3) |
| `all` | `full` + `docs` + `ui` + `fla` |

The `dev` extra is enough for package work; `full` is the CI environment shape.

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
│   ├── tasks/         # Task heads and metrics
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

Two contract details that trip new contributors:

- **The 90% coverage floor applies to every `--cov` invocation**, regardless of how
  many tests ran (`fail_under = 90` in `[tool.coverage.report]`, a ratchet floor —
  the fast lane actually measures in the mid-90s). Scoped runs must drop `--cov` or
  pass `--no-cov`, or they fail on the floor.
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
- **Release branches** named `vX.Y` (currently `v1.3`) carry stabilized releases;
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
CI will run on your PR (the fast pytest lane with the coverage floor on a Python
3.11–3.13 × numpy 2.2.0 matrix plus a Windows leg, the skip audit, and the ty hard
gate). Nightly-only lanes (`coverage-nightly`, `example-nightly`, `test-mamba`)
never gate PRs; see
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
