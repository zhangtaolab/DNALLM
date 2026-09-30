# DNALLM CI/CD Workflow

This directory contains the GitHub Actions workflows for continuous integration and deployment of the DNALLM project.

## 🚀 Overview

The CI/CD pipeline automatically runs comprehensive tests and quality checks whenever code is pushed to the main branches or pull requests are created. This ensures code quality, compatibility, and reliability across different environments.

## 📋 Workflow Triggers

The workflows are triggered on:

- **Push events** to `main`, `master`, and `develop` branches
- **Pull request events** targeting `main`, `master`, and `develop` branches
- **Scheduled nightly run** at 03:00 UTC — triggers the `coverage-nightly` full census (GitHub runs cron schedules only from the default branch)
- **Manual workflow dispatch** — runs the nightly census on demand (e.g. for calibration)

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
4. **Dependency Installation**: Installs test and development dependencies
5. **Code Quality Checks**:
   - **Black**: Code formatting validation
   - **isort**: Import sorting validation
   - **Flake8**: Linting and style checking
6. **Type Checking**: Runs MyPy for static type analysis
7. **Test Execution**: Runs the fast test suite (`pytest -m "not slow" --cov`); the coverage total is enforced against the `fail_under = 90` floor from `pyproject.toml [tool.coverage.report]` — dropping below it fails the job

### 2. CUDA Test Job (`test-cuda`)

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

### 3. Mamba Test Job (`test-mamba`)

**Purpose**: Testing for Mamba-specific functionality and dependencies.

**Matrix Strategy**:
- Python version: 3.11
- Operating system: Ubuntu Latest

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Python Setup**: Installs Python 3.11
3. **UV Installation**: Installs the UV package manager
4. **Mamba Dependency Installation**: Installs Mamba-specific dependencies
5. **Mamba Test Execution**: Runs tests excluding slow tests

### 4. Deploy Job (`deploy`)

**Purpose**: Automatic documentation deployment to GitHub Pages.

**Dependencies**: Requires all test jobs to pass
**Trigger**: Only runs on `main` or `master` branch pushes

**Steps**:
1. **Code Checkout**: Clones the repository
2. **Git Configuration**: Sets up GitHub Actions bot credentials
3. **Python Setup**: Installs Python 3.11
4. **MkDocs Cache**: Configures caching for documentation dependencies
5. **UV Installation**: Installs the UV package manager
6. **Documentation Dependencies**: Installs MkDocs and related packages
7. **Documentation Deployment**: Deploys to GitHub Pages

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

**Purpose**: Coverage census including the `slow` tests (real HF/ModelScope model downloads) under the same `fail_under = 90` floor. Runs only on the 03:00 UTC schedule and via manual workflow dispatch — event guards keep it out of the push/PR loop.

**Census scope**: 27 tests carry the `slow` mark; 21 of them execute in this job. The remaining 6 — the MCP live-server probes in `dnallm/mcp/tests/test_sse_client.py` and `test_streamable_http_client.py` — target `localhost:8000`, which no CI job starts, so they skip deterministically as typed `network-unavailable:` skips (allowlisted in `tests/expected_skips.yaml`). Those probes are local-only: run them against a manually started `dnallm-mcp-server`.

**Timeout**: 480 minutes (the slow suite is projected at 4-7.5h on 4-core CPU runners)

**Model Caches**: Both hub directories (`~/.cache/huggingface/hub`, `~/.cache/modelscope/hub`) are cached whole, keyed on `hashFiles('models.lock')` — editing a `models.lock` entry rotates the key; the cache saves only on job success.

**Steps**:
1. **Code Checkout** / **Free Disk Space** / **Python 3.12 Setup** / **UV + Caches**: shared uv cache plus the `models.lock`-keyed model caches
2. **Dependency Installation**: Installs base dependencies plus NumPy 2.2.0
3. **Gated Full Census**: Runs the census of record with slow tests included (`-ra --durations=0 --cov`), minus the 6 MCP live-server probes that typed-skip without a local server (see Census scope above); per-test `@pytest.mark.timeout` marks override the global 300s timeout for the long trainer tests
4. **Skip Audit**: `scripts/audit_skips.py` against the nightly junit — unexpected skips fail the job

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
- Coverage reports are generated in XML and terminal formats
- Enforced floor: `fail_under = 90` in `pyproject.toml [tool.coverage.report]` fails any `--cov` run — local or CI — whose total drops below 90; the threshold is identical everywhere

### Quality Standards

- **Code Formatting**: Must pass Black formatting checks
- **Import Organization**: Must pass isort import sorting
- **Linting**: Must pass Flake8 style and complexity checks
- **Type Safety**: Must pass MyPy type checking (with reasonable exceptions)

## 🔍 Monitoring and Reporting

### Test Results

- Test results are displayed in the GitHub Actions interface
- Failed tests provide detailed error information and stack traces
- Coverage reports show which code areas need additional testing

### Quality Metrics

- Code formatting compliance
- Import organization
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
   - Run `black .` to auto-format code
   - Run `isort .` to organize imports
   - Fix Flake8 violations manually
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

# Run quality checks
black --check .
isort --check-only .
flake8 .
mypy dnallm/

# Run tests
pytest --cov

# Census of record (what the coverage gate runs; enforces the 90 floor)
pytest -ra --durations=0 --junitxml=/tmp/census-junit.xml --cov

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
