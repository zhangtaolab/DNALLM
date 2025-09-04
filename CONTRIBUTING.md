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
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 .
   
   # Type checking
   mypy dnallm/
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

- **Black**: Code formatting (line length: 88 characters)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

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

1. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

2. **Check code quality**:
   ```bash
   black --check .
   isort --check-only .
   flake8 .
   mypy dnallm/
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

5. **Update CHANGELOG.md** if applicable

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
