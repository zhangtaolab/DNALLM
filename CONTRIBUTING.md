# Contributing to DNALLM

Thank you for your interest in contributing to DNALLM! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and inclusive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- CUDA-compatible GPU (optional, for GPU acceleration)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/DNALLM.git
   cd DNALLM
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/zhangtaolab/DNALLM.git
   ```

## Development Setup

### 1. Environment Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install DNALLM in development mode
uv pip install -e '.[dev,test]'
```

### 2. Pre-commit Hooks (Optional)

```bash
# Install pre-commit
uv pip install pre-commit

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Check code quality
black --check .
isort --check-only .
flake8 .
mypy dnallm/
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add new examples or tutorials

### Workflow

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run specific test categories
   pytest tests/ -m "not slow"
   pytest tests/ -m "unit"
   
   # Run with coverage
   pytest tests/ --cov=dnallm --cov-report=term-missing
   ```

4. **Check code quality**:
   ```bash
   # Format code with Ruff
   ruff format .
   
   # Check code formatting
   ruff format --check .
   
   # Lint code with Ruff
   ruff check . --statistics
   
   # Additional flake8 check for MCP module
   flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402
   
   # Type checking (relaxed settings)
   mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style and Standards

### Python Code Style

We follow the following standards:

- **Ruff**: Code formatting and linting (line length: 79 characters)
- **flake8**: Additional linting for MCP module compatibility
- **mypy**: Type checking (with relaxed settings for development)

### Pre-commit Checklist

**⚠️ IMPORTANT: Before committing any code, you MUST run the following checks:**

1. **Code Formatting**:
   ```bash
   # Format all code
   ruff format .
   
   # Verify formatting is correct
   ruff format --check .
   ```

2. **Code Quality Checks**:
   ```bash
   # Run Ruff linting
   ruff check . --statistics
   
   # Run flake8 for MCP module
   flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402
   ```

3. **Type Checking**:
   ```bash
   # Run mypy with relaxed settings
   mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped
   ```

4. **Test Suite**:
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run tests with coverage
   pytest tests/ --cov=dnallm --cov-report=term-missing
   ```

5. **Quick Validation Script**:
   ```bash
   # Option 1: Use the automated check script (recommended)
   python scripts/check_code.py
   
   # Option 2: Run all checks manually in one command
   ruff format --check . && ruff check . --statistics && flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402 && mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped
   ```

**All checks must pass before committing!** CI will run the same checks and will fail if any issues are found.

### Code Check Scripts

We provide automated scripts to run all code quality checks at once:

#### Python Script (Cross-platform, Recommended)
```bash
# Basic usage - run code quality checks only (default)
python scripts/check_code.py

# Auto-fix issues where possible
python scripts/check_code.py --fix

# Show detailed output
python scripts/check_code.py --verbose

# Include test suite execution
python scripts/check_code.py --with-tests

# Get help
python scripts/check_code.py --help
```

#### Shell Script (Linux/macOS)
```bash
# Make executable (first time only)
chmod +x scripts/check_code.sh

# Run all checks
./scripts/check_code.sh

# Auto-fix issues
./scripts/check_code.sh --fix

# Verbose output
./scripts/check_code.sh --verbose
```

#### Batch Script (Windows)
```cmd
# Run all checks
scripts\check_code.bat

# Auto-fix issues
scripts\check_code.bat --fix

# Verbose output
scripts\check_code.bat --verbose
```

#### What the Scripts Check
1. **Code Formatting** (Ruff)
2. **Code Quality** (Ruff linting)
3. **MCP Module Compatibility** (Flake8)
4. **Type Checking** (MyPy with relaxed settings)
5. **Test Suite** (Pytest)
6. **Test Coverage** (Pytest with coverage)

#### Example Usage
```bash
# Quick code quality check (default - no tests)
$ python scripts/check_code.py
[INFO] Starting DNALLM code quality checks...
==========================================
[INFO] 1. Code Formatting...
[SUCCESS] Code formatting check completed successfully

[INFO] 2. Code Quality (Ruff)...
[SUCCESS] Code quality check completed successfully

[INFO] 3. Flake8 (MCP Module)...
[SUCCESS] Flake8 check for MCP module completed successfully

[INFO] 4. Type Checking (MyPy)...
[SUCCESS] Type checking with MyPy completed successfully

==========================================
[SUCCESS] All checks passed! ✅
[INFO] Your code is ready for commit.

# Include test suite execution
$ python scripts/check_code.py --with-tests
[INFO] Starting DNALLM code quality checks...
[INFO] 1. Code Formatting...
[SUCCESS] Code formatting check completed successfully
...
[INFO] 5. Test Suite...
[SUCCESS] Test suite execution completed successfully

# Auto-fix issues
$ python scripts/check_code.py --fix
[INFO] Starting DNALLM code quality checks...
[INFO] 1. Code Formatting...
[SUCCESS] Code formatting (auto-fix) completed successfully
...
```

### Code Quality Standards

#### Ruff Configuration
- **Line length**: 79 characters (not 88 like Black)
- **Indentation**: 4 spaces
- **Quote style**: Double quotes
- **Import sorting**: Automatic with isort compatibility
- **Error codes enabled**: E4, E7, E9, F, W, B, C4, UP, N, S, T20, PT, Q, RUF

#### Flake8 Configuration (MCP Module)
- **Line length**: 79 characters
- **Ignored errors**: E203, W503, C901, E402
- **Purpose**: Ensure MCP module compatibility

#### MyPy Configuration
- **Strict mode**: Disabled for development
- **Missing imports**: Ignored
- **Optional types**: Not strictly enforced
- **Disabled error codes**: Multiple codes disabled for development flexibility

#### File Organization
- **Maximum file size**: < 1000 lines
- **Import order**: Standard library → Third party → Local imports
- **Docstring style**: Google-style
- **Type hints**: Required for all function parameters and return values

### Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

- Use **Google-style docstrings** for functions and classes
- Include type hints for all function parameters and return values
- Add inline comments for complex logic

Example:
```python
def predict_sequence(self, sequence: str, model_name: str) -> Dict[str, Any]:
    """Predict the properties of a DNA sequence.
    
    Args:
        sequence: DNA sequence string (A, T, G, C)
        model_name: Name of the model to use for prediction
        
    Returns:
        Dictionary containing prediction results and metadata
        
    Raises:
        ValueError: If sequence contains invalid characters
        ModelNotFoundError: If specified model is not available
    """
    # Implementation here
    pass
```

### File Organization

- Keep files focused and reasonably sized (< 1000 lines)
- Use meaningful file and directory names
- Group related functionality together
- Follow the existing project structure

## Testing

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── pytest.ini              # Test configuration
├── TESTING.md              # Detailed testing guide
├── inference/              # Inference module tests
├── utils/                  # Utility function tests
├── datahandling/           # Data handling tests
├── finetune/               # Training tests
└── test_data/              # Test data files
```

### Writing Tests

1. **Test file naming**: `test_*.py`
2. **Test class naming**: `Test*`
3. **Test method naming**: `test_*`
4. **Use descriptive test names** that explain what is being tested

Example:
```python
import pytest
from dnallm.utils.sequence import validate_dna_sequence

class TestSequenceValidation:
    """Test cases for DNA sequence validation."""
    
    def test_valid_sequence(self):
        """Test validation of valid DNA sequences."""
        assert validate_dna_sequence("ATCG") == True
        assert validate_dna_sequence("ATCGATCG") == True
    
    def test_invalid_characters(self):
        """Test validation rejects invalid characters."""
        with pytest.raises(ValueError):
            validate_dna_sequence("ATCGX")
    
    @pytest.mark.slow
    def test_large_sequence(self):
        """Test validation of large sequences."""
        large_seq = "ATCG" * 1000
        assert validate_dna_sequence(large_seq) == True
```

### Test Markers

Use appropriate markers for different test types:

- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.pdf`: Tests that generate PDF files

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m "unit"              # Only unit tests
pytest -m "integration"       # Only integration tests

# Run with coverage
pytest --cov=dnallm --cov-report=html

# Run specific test files
pytest tests/inference/test_predictor.py
```

## Documentation

### Documentation Structure

Documentation is organized in the `docs/` directory:

```
docs/
├── index.md                 # Main documentation page
├── getting_started/         # Installation and setup guides
├── tutorials/              # Step-by-step tutorials
├── api/                    # API reference
├── concepts/               # Core concepts
└── faq/                    # Frequently asked questions
```

### Writing Documentation

1. **Use Markdown** for all documentation
2. **Include code examples** that can be run
3. **Keep documentation up-to-date** with code changes
4. **Use clear, concise language**
5. **Include diagrams** for complex concepts

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -e '.[docs]'

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Pull Request Process

### Before Submitting

1. **Run the complete pre-commit checklist** (see [Pre-commit Checklist](#pre-commit-checklist) above):
   ```bash
   # Quick validation - all checks must pass
   ruff format --check . && ruff check . --statistics && flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402 && mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped
   ```

2. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v --cov=dnallm --cov-report=term-missing
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

5. **Update CHANGELOG.md** if applicable

6. **Verify CI compatibility**: Your local checks should match what CI runs

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition/improvement

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review**

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Environment information**:
   - Python version
   - Operating system
   - DNALLM version
   - Dependencies versions

5. **Minimal code example** (if applicable)
6. **Error messages** and stack traces

### Feature Requests

For feature requests, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** (if you have ideas)
4. **Alternatives considered**

### Issue Templates

Use the provided issue templates when creating issues on GitHub.

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Build and publish** to PyPI

## Development Tips

### Performance Optimization

- Use **profiling tools** to identify bottlenecks
- Consider **vectorization** for numerical operations
- Implement **caching** for expensive computations
- Use **async/await** for I/O operations

### Memory Management

- **Monitor memory usage** during development
- **Clean up resources** properly
- Use **context managers** for file operations
- Consider **lazy loading** for large datasets

### Debugging

- Use **logging** instead of print statements
- Add **debugging information** to error messages
- Use **breakpoints** in development
- **Test edge cases** thoroughly

## Quick Reference

### Essential Commands

```bash
# Setup development environment
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,test]'

# Pre-commit validation (run before every commit)
# Option 1: Use automated script (recommended, code quality only)
python scripts/check_code.py

# Option 2: Include test suite execution
python scripts/check_code.py --with-tests

# Option 3: Manual validation
ruff format --check . && ruff check . --statistics && flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402 && mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped

# Auto-fix code issues
python scripts/check_code.py --fix

# Format code manually
ruff format .

# Run tests
pytest tests/ -v --cov=dnallm --cov-report=term-missing

# Build documentation
mkdocs build
mkdocs serve
```

### Common Issues and Solutions

1. **Code check script fails**: 
   - Make sure you're in the DNALLM root directory
   - Ensure virtual environment is activated: `source .venv/bin/activate`
   - Install dependencies: `uv pip install -e '.[dev,test]'`

2. **Ruff formatting errors**: 
   - Auto-fix: `python scripts/check_code.py --fix`
   - Manual fix: `ruff format .`

3. **Import errors**: 
   - Check import order and use `# noqa: E402` for necessary late imports
   - Run `ruff check . --fix` to auto-fix some import issues

4. **Type checking errors**: 
   - Most are disabled in development, but fix critical ones
   - Check specific files: `mypy dnallm/specific_file.py`

5. **Test failures**: 
   - Run specific test files: `pytest tests/specific_test.py -v`
   - Run with verbose output: `python scripts/check_code.py --verbose`

6. **Script permission errors (Linux/macOS)**:
   - Make executable: `chmod +x scripts/check_code.sh`
   - Or use Python script: `python scripts/check_code.py`

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the docs/ directory first
- **Examples**: Look at the example/ directory for usage patterns

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes**
- **Project documentation**

Thank you for contributing to DNALLM! 🧬
