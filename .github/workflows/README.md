# DNALLM CI/CD Workflow

This directory contains the GitHub Actions workflows for continuous integration and deployment of the DNALLM project.

## 🚀 Overview

The CI/CD pipeline automatically runs comprehensive tests and quality checks whenever code is pushed to the main branches or pull requests are created. This ensures code quality, compatibility, and reliability across different environments.

## 📋 Workflow Triggers

The workflows are triggered on:

- **Push events** to `main`, `master`, and `dev` branches
- **Pull request events** targeting `main`, `master`, and `dev` branches
- **Scheduled nightly runs** — two cron entries, and BOTH trigger the whole workflow (GitHub offers no per-cron routing), so each nightly job carries its own cron-string gate (`github.event.schedule == '<cron>'`, D-19) selecting exactly one entry:
  - **03:00 UTC** (cron `0 3 * * *`) — fires `coverage-nightly` (full coverage census) and `test-mamba` (kernel-build leg)
  - **05:30 UTC** (cron `30 5 * * *`) — fires `example-nightly` (example-execution census; D-06 stagger, queue-serialized on the single `dnallm-nightly` runner)
  - GitHub runs cron schedules only from the default branch
- **Manual workflow dispatch** — each nightly job's gate admits `workflow_dispatch` (on-demand census/calibration runs), and dispatch is the ONLY trigger of the `feasibility.yml` spike (runner confirmation for the GB10 feasibility verdicts; never push/PR/schedule, so PR-authored code cannot reach the self-hosted box)

## 🔧 Jobs

### 1. Test Job (`test`)

**Purpose**: Core testing across multiple Python versions with comprehensive quality checks.

**Matrix Strategy**:
- Python versions: 3.11, 3.12, 3.13
- NumPy versions: 1.26.4, 2.2.0
- Operating system: Ubuntu Latest

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Python Setup**: Installs specified Python version
3. **UV Installation**: Installs the UV package manager
4. **Dependency Installation**: Installs base dependencies plus the matrix NumPy pin
5. **Code Quality Checks** (Ruff):
   - **Ruff format**: formatting validation (`ruff format --check .`)
   - **Ruff check**: linting (`ruff check . --statistics`)
6. **Test Execution**: Runs the fast test suite (`pytest -m "not slow" --cov`); the coverage total is enforced against the `fail_under = 90` floor from `pyproject.toml [tool.coverage.report]` — dropping below it fails the job
7. **Skip Audit**: `scripts/audit_skips.py` against the junit — unexpected skips fail the job
8. **Exit-Code Canary**: an intentionally failing test must make pytest exit non-zero (guards against exit-code masking regressions)
9. **Type Checking**: Runs MyPy for static type analysis (advisory — does not fail the job)

### 2. Windows Test Job (`test-windows`)

**Purpose**: Windows platform leg for the fast suite — the package claims "Operating System :: OS Independent" and the primary dev machine is Windows 11. The compiled mamba kernels (`.[mamba]`) stay on the linux nightly box.

**Runner**: `windows-latest` (push/PR only), Python 3.12, 60-minute timeout

**Steps**: mirrors the `test` job — Ruff format/lint checks, fast tests with the coverage floor, skip audit, exit-code canary, advisory MyPy. Sets `PYTHONUTF8=1` (the root conftest prints emoji, which would crash non-UTF8 pipes on Windows) and disables git `autocrlf` so `ruff format` line endings match.

### 3. CUDA Test Job (`test-cuda`)

**Purpose**: GPU-enabled testing for CUDA-specific functionality.

**Matrix Strategy**:
- Python version: 3.11
- CUDA versions: 12.1, 12.4
- Operating system: Ubuntu Latest

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Python Setup**: Installs Python 3.11
3. **CUDA Installation**: Installs NVIDIA CUDA toolkit
4. **UV Installation**: Installs the UV package manager
5. **CUDA Dependency Installation**: Installs CUDA-specific dependencies
6. **GPU Test Execution**: Runs tests excluding slow tests

### 4. Mamba Test Job (`test-mamba`)

**Purpose**: Compiles and exercises the native mamba-ssm/causal_conv1d CUDA kernels (the `.[mamba]` extra). Scheduled nightly (03:00 UTC entry only, via its D-19 cron-string gate) / manual dispatch only — PR code (including forks) never runs on this runner.

**Runner**: `self-hosted` GPU box (`dnallm-nightly`), Python 3.11, 180-minute timeout (the recurring kernel source build is the long pole)

**Steps**:
1. **Code Checkout**: Clones the repository
2. **GPU Check**: Detects `nvidia-smi`; if the box ever loses its GPU the remaining steps are skipped as a fail-safe no-op
3. **Python Setup**: Installs Python 3.11
4. **UV Installation**: Installs the UV package manager
5. **Mamba Dependency Installation**: Installs `.[base]` — the same extras set the other legs use (`dev,test,notebook,mcp`; the `mcp` extra is required by the not-slow census: the `mcp_example` notebook-import tests and the `exceptiongroup` backport) — plus `.[mamba]` (kernel source build)
6. **Mamba Test Execution**: Runs tests excluding slow tests — a failing test fails the job; test logs are uploaded as an artifact on failure

### 5. Coverage Gate Job (`coverage-gate`)

**Purpose**: Single-leg coverage enforcement on every push and pull request (Python 3.12, NumPy 2.2.0). Runs the gated fast census (`pytest -m "not slow" --cov`) — the `fail_under = 90` floor from `pyproject.toml [tool.coverage.report]` rides the pytest exit code, so a total below 90 fails the job with `Coverage failure: total of N is less than fail-under=90`. Also audits its own junit for unexpected skips.

**Timeout**: 90 minutes

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Free Disk Space**: Removes unneeded preinstalled toolchains
3. **Python Setup**: Installs Python 3.12
4. **UV Installation + Dependency Cache**: Installs the UV package manager, restores the shared uv cache
5. **Dependency Installation**: Installs base dependencies plus NumPy 2.2.0
6. **Gated Fast Census**: Runs the fast census with coverage; a total below the 90 floor fails the job
7. **Skip Audit**: `scripts/audit_skips.py` against the gate junit — unexpected skips fail the job

### 6. Nightly Coverage Job (`coverage-nightly`)

**Purpose**: Coverage census including the `slow` tests (real HF/ModelScope model downloads) under the same `fail_under = 90` floor. Runs only on the 03:00 UTC schedule entry and via manual workflow dispatch — its cron-string gate (D-19) keeps it out of the push/PR loop and off the 05:30 entry.

**Census scope**: 27 tests carry the `slow` mark; 21 of them execute in this job. The remaining 6 — the MCP live-server probes in `dnallm/mcp/tests/test_sse_client.py` and `test_streamable_http_client.py` — target `localhost:8000`, which no CI job starts, so they skip deterministically as typed `network-unavailable:` skips (allowlisted in `tests/expected_skips.yaml`). Those probes are local-only: run them against a manually started `dnallm-mcp-server`.

**Timeout**: 900 minutes (per-test `@pytest.mark.timeout` ceilings across the slow suite sum to 840min — 600min from the 7 phase marks plus 240min from the download/real-inference/MCP marks, where the 1800s class mark on `TestRealModelInference` applies to all 5 of its items; the kill sits above that sum so a hung test fails via its own mark, with junit and the skip audit still produced. Note that GitHub-hosted runners hard-cap a single job at 360min, so the platform cap binds before this figure — the per-test marks are the primary protection, the job-level number is a backstop, and the census itself is projected at 4-7.5h on 4-core CPU runners, i.e. a slow night can still hit the platform cap)

**Model Caches**: none — the `models.lock`-keyed hub cache layer was removed (D-11, owner decision 2026-10-05): cold pulls are proven (a 65-min all-cold example stage 1 against a 2700-min budget) and the lock-only cache (~15.2GiB) exceeded the 10GB Actions cache quota without ever successfully saving. The runner box's local `$HOME` hub caches remain the warm path and are never cleaned (owner rule).

**Steps**:
1. **Code Checkout** / **Free Disk Space** / **Python 3.12 Setup** / **UV + Cache**: shared uv cache only (the hub cache layer is gone — D-11)
2. **Dependency Installation**: Installs base dependencies plus NumPy 2.2.0
3. **Gated Full Census**: Runs the census of record with slow tests included (`-ra --durations=0 --cov`), minus the 6 MCP live-server probes that typed-skip without a local server (see Census scope above); per-test `@pytest.mark.timeout` marks override the global 300s timeout for the long network-bound tests (trainer, real-download, and MCP integration)
4. **Skip Audit**: `scripts/audit_skips.py` against the nightly junit — unexpected skips fail the job

### 7. Example Nightly Job (`example-nightly`)

**Purpose**: Real execution census for everything under `example/` — notebooks (nbclient), marimo apps, the helper script, and every YAML config through real `load_config()`. Split out of coverage-nightly once the example census outgrew its budget (CI-06 pre-authorization).

**Trigger**: the **05:30 UTC** schedule entry only (cron-string gate, D-19) plus manual `workflow_dispatch` — never push/PR, so PR-authored code (including forks) never reaches the self-hosted box.

**Runner**: `self-hosted` GPU box (`dnallm-nightly`), Python 3.12, 2700-minute timeout (per-test `@pytest.mark.timeout` marks are the primary hang protection; the job kill is the whole-job backstop).

**Staged-serial topology** (stages run strictly in order inside one job — a server-binding stage may never start while the previous stage still holds GPU memory or kernel processes):

| Stage | Content |
|-------|---------|
| 0 | venv + full extras + bedtools rootless prefix + mamba kernel wheelhouse (cached) + megaDNA pinned provisioning + runner inventory probe. **No evo provisioning (D-04), no model cache (D-11)** |
| 0.5 | **Census collection assertion** (D-03) — hard gate: the collect-only triple for the exact stage-1 selector set (`-m "not giants" -k "not mcp_example"`) is pinned in the workflow; any drift (collection breakage, marker typo, accidental over-marking) fails the job before stage 1 runs. The pinned literal is the deliberate census-growth bump-point |
| 1 | torch-heavy example execution: `pytest tests/examples` with the two `mcp_example` ids deselected (stage 3 owns them) and the `giants`-marked evo execution test deselected (D-01 owner-policy exclusion — the runner environment works, so a typed skip would be dishonest; the dispatch/manual lane runs it explicitly with `-m giants`), plus the fast YAML config leg |
| 1.5 | kernel pkill + VRAM settle |
| 2 | MCP live-server probes on :8000 (streamable-http probes, restart, sse probes) |
| 2.5 | kernel pkill + VRAM settle |
| 3 | fresh MCP server + the `mcp_example` pair (ollama loopback) |
| 4 | skip audit + junit artifact upload + fail-soft summary |

**Fail-soft contract** (D-08): every stage-1+ invocation records its exit code to `stage-results.txt` and its step exits 0 so later stages still run; the stage-4 summary exits non-zero when ANY recorded outcome failed. NO step carries `continue-on-error` — a forever-green job is prohibited. Stage-0-area failures (venv, provisioning, and the stage-0.5 assertion) fail the job directly.

**Caches**: uv dependency cache + bedtools rootless prefix + mamba kernel wheelhouse only. The `models.lock`-keyed hub cache layer is gone (D-11 — cold pulls by design; the box's local `$HOME` caches are the warm path), and the evo provisioning steps (isolated venv, flash-attention wheelhouse, giant-model prefetch) are deleted per D-04 — the flash-attn build-isolation failure of run 37278002681 dissolved with that deletion. The evo family's committed executed-notebook outputs remain the evidence, and the dispatch/manual lane keeps working from the local giants model tier (never cleaned, owner rule).

### 8. Feasibility Spike Job (`feas-spike`)

**Purpose**: Runner confirmation for the Phase 5 GB10 feasibility spike (FEAS-01, D-04) — re-runs the committed per-family spike runner (`scripts/feasibility/spike_families.py`) on the same hardware class the local verdicts were taken on and uploads the logs plus the verdict matrix as artifacts. The owner fills the matrix's Runner confirmation column from those artifacts; local verdicts become official only then.

**Trigger**: **manual `workflow_dispatch` ONLY** — never push, PR, or schedule. This job executes repo code on the self-hosted GPU box, so the dispatch-only gate preserves the invariant that PR-authored code (including forks) never reaches that runner.

**Runner**: `self-hosted` GPU box (`dnallm-nightly`), 240-minute timeout (evo-1's cold download alone is ~30GB)

**Steps**:
1. **GPU Check** (runs BEFORE checkout): on GPU absence writes a `gpu_absent` marker into the artifact path and FAILS the job — unlike `test-mamba`'s green fail-safe no-op, this job's deliverable is evidence, so zero-evidence green is forbidden (the `if: always()` upload still delivers the marker). **Code Checkout** follows, gated on the GPU check's `has_gpu` output — the check must precede checkout so the marker survives the skipped checkout
2. **UV + Dependency Installation**: `uv venv` + `uv pip install -e ".[base]"` (the coverage-nightly-proven extras set; their absence from this venv is deliberate evidence)
3. **Runner Identity**: records the `nvidia-smi` identity line (the D-04 parity claim)
4. **Spike Execution**: `--family all --fallback` (D-06 fallback variants only — `--fallback` replaces the notebook variant, it does not run both; notebook-variant verdicts rest on the committed local evidence in `spike-logs/`); expected failures for environment-unavailable families are carried as evidence text in the artifacts, not hidden
5. **Artifact Upload**: spike logs + `05-FEASIBILITY.md`, unconditionally (`if: always()`)

### 9. Deploy Job (`deploy`)

**Purpose**: Automatic documentation deployment to GitHub Pages.

**Dependencies**: `needs: [test, test-cuda]` — only those two jobs gate the deploy. `test-windows` and `coverage-gate` are **not** deploy gates. (`test-mamba` is deliberately excluded: it is event-gated to schedule/dispatch, and dependents of a skipped `needs` job are skipped, which would silently stop push deploys.)
**Trigger**: Only runs on `main` or `master` branch pushes

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Git Configuration**: Sets up GitHub Actions bot credentials
3. **Python Setup**: Installs Python 3.11
4. **MkDocs Cache**: Configures caching for documentation dependencies
5. **UV Installation**: Installs the UV package manager
6. **Documentation Dependencies**: Installs MkDocs and related packages
7. **Documentation Deployment**: Deploys to GitHub Pages

## 🧪 Testing Strategy

### Test Categories

The project uses pytest markers to categorize tests:

- **Unit Tests**: Fast, isolated tests for individual functions
- **Integration Tests**: Tests that verify component interactions
- **Performance Tests**: Tests that measure execution time and resource usage
- **PDF Tests**: Tests that generate PDF outputs
- **Slow Tests**: Tests that take longer to execute (real model downloads, long training runs); they run in the nightly `coverage-nightly` job — the PR gate deliberately runs the fast leg only

### Coverage Requirements

- All code changes must maintain or improve test coverage
- Coverage totals are reported in the terminal only — the junit XML artifact carries test results, not coverage, and no XML coverage report or codecov upload is produced in CI
- Enforced floor: `fail_under = 90` in `pyproject.toml [tool.coverage.report]` fails any `--cov` run — local or CI — whose total drops below 90; the threshold is identical everywhere

### Quality Standards

- **Code Formatting**: Must pass `ruff format --check .` (ruff's Black-compatible formatter)
- **Linting**: Must pass `ruff check .` (rule set configured in `pyproject.toml [tool.ruff.lint]`)
- **Type Safety**: MyPy type checking runs in CI as an advisory step (it does not fail the job)

Flake8 is not run in CI; it remains a local, MCP-module-only tool via the legacy `.flake8` config.

## 🔍 Monitoring and Reporting

### Test Results

- Test results are displayed in the GitHub Actions interface
- Failed tests provide detailed error information and stack traces
- Coverage reports show which code areas need additional testing

### Quality Metrics

- Code formatting compliance
- Linting violations count
- Type checking errors
- Test coverage percentage

### Deployment Status

- Documentation deployment status
- GitHub Pages availability
- Cache hit/miss rates for dependencies

## 🚨 Troubleshooting

### Common Issues

1. **Dependency Installation Failures**
   - Check Python version compatibility
   - Verify package availability in PyPI
   - Review dependency conflicts in pyproject.toml

2. **Test Failures**
   - Review test output for specific error messages
   - Check if new dependencies are required
   - Verify test data availability

3. **Quality Check Failures**
   - Run `ruff format .` to auto-format code
   - Run `ruff check . --fix` to auto-fix lint violations
   - Address MyPy type annotation issues

4. **CUDA Test Failures**
   - Verify CUDA toolkit installation
   - Check GPU driver compatibility
   - Review CUDA version requirements

### Local Testing

Before pushing code, run these commands locally:

```bash
# Install development dependencies
uv pip install -e ".[test,dev]"

# Run quality checks (what CI runs)
ruff format --check .
ruff check .
mypy dnallm/

# Run tests
pytest --cov

# Census of record (what coverage-nightly runs; enforces the 90 floor)
pytest -ra --durations=0 --junitxml=/tmp/census-junit.xml --cov

# Fast census (what the coverage gate runs)
pytest -m "not slow" -ra --durations=0 --junitxml=/tmp/gate-junit.xml --cov

# Scoped runs: coverage is floor-gated, so a scoped --cov run exits 1
# even with green tests (expected, not a bug). Drop --cov or pass --no-cov:
pytest tests/utils/test_sequence.py --no-cov

# Run specific test categories
pytest tests/ -m "not slow"
pytest tests/ -m "unit"
pytest tests/ -m "integration"
```

## 📚 Additional Resources

- [Project Testing Documentation](../tests/TESTING.md)
- [Pytest Configuration](../../pyproject.toml) (see `[tool.pytest.ini_options]`)
- [Project Dependencies](../../pyproject.toml)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🤝 Contributing

When contributing to the project:

1. Ensure all tests pass locally before pushing
2. Follow the established code quality standards
3. Add tests for new functionality
4. Update documentation as needed
5. Monitor CI/CD pipeline results

## 📊 Performance Considerations

- CI jobs run in parallel when possible
- Dependency caching reduces setup time
- Test matrix strategy balances coverage and execution time
- Slow tests run in the nightly `coverage-nightly` job; the PR gate runs the fast leg to keep feedback quick
