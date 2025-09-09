# DNALLM CI/CD Workflow

This directory contains the GitHub Actions workflows for continuous integration and deployment of the DNALLM project.

## 🚀 Overview

The CI/CD pipeline automatically runs comprehensive tests and quality checks whenever code is pushed to the main branches or pull requests are created. This ensures code quality, compatibility, and reliability across different environments.

## 📋 Workflow Triggers

The workflows are triggered on:

- **Push events** to `main`, `master`, and `develop` branches
- **Pull request events** targeting `main`, `master`, and `develop` branches

## 🔧 Jobs

### 1. Test Job (`test`)

**Purpose**: Core testing across multiple Python versions with comprehensive quality checks.

**Matrix Strategy**:
- Python versions: 3.10, 3.11, 3.12
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
7. **Test Execution**: Runs pytest with coverage reporting
8. **Coverage Upload**: Uploads coverage reports to Codecov

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

## 🧪 Testing Strategy

### Test Categories

The project uses pytest markers to categorize tests:

- **Unit Tests**: Fast, isolated tests for individual functions
- **Integration Tests**: Tests that verify component interactions
- **Performance Tests**: Tests that measure execution time and resource usage
- **PDF Tests**: Tests that generate PDF outputs
- **Slow Tests**: Tests that take longer to execute (excluded from CI)

### Coverage Requirements

- All code changes must maintain or improve test coverage
- Coverage reports are generated in XML and terminal formats
- Results are automatically uploaded to Codecov for tracking

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
pytest tests/ -v --cov=dnallm

# Run specific test categories
pytest tests/ -m "not slow"
pytest tests/ -m "unit"
pytest tests/ -m "integration"
```

## 📚 Additional Resources

- [Project Testing Documentation](../tests/TESTING.md)
- [Pytest Configuration](../tests/pytest.ini)
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
- Slow tests are excluded from CI to maintain reasonable execution times
