# DNALLM Test Suite

This directory contains the comprehensive test suite for the DNALLM project, organized by module and functionality.

## 🏗️ Test Structure

```
tests/
├── pytest.ini              # Project-level pytest configuration
├── README.md               # This file - overall test documentation
├── inference/              # Inference module tests
│   ├── test_plot.py        # Plot functionality tests
│   ├── pdf/                # PDF output directory
│   └── README.md           # Inference-specific documentation
├── utils/                   # Utility module tests
│   └── test_sequence.py    # Sequence utility tests
└── test_data/              # Test data files
    ├── binary_classification/
    ├── embedding/
    ├── language_model/
    ├── multiclass_classification/
    ├── multilabel_classification/
    ├── regression/
    └── token_classification/
```

## 🚀 Running Tests

### Project-Wide Testing

From the project root directory (`DNALLM/`):

```bash
# Run all tests in the project
pytest

# Run tests with verbose output
pytest -v

# Run tests from specific module
pytest tests/inference/
pytest tests/utils/

# Run tests with coverage
pytest --cov=dnallm --cov-report=html
```

### Module-Specific Testing

From individual test directories:

```bash
# Test inference module
cd tests/inference
pytest

# Test utils module
cd tests/utils
pytest
```

### Selective Testing

```bash
# Run tests by marker
pytest -m inference -v        # Only inference tests
pytest -m utils -v            # Only utility tests
pytest -m pdf -v              # Only PDF generation tests
pytest -m performance -v      # Only performance tests

# Run specific test files
pytest tests/inference/test_plot.py
pytest tests/utils/test_sequence.py

# Run specific test classes
pytest tests/inference/test_plot.py::TestPlotBars
pytest tests/utils/test_sequence.py::TestSequenceUtils

# Run specific test methods
pytest tests/inference/test_plot.py::TestPDFOutputQuality::test_demo_pdf_generation
```

## 🔧 Configuration

### Pytest Configuration (`pytest.ini`)

The project uses a centralized pytest configuration that applies to all test modules:

```ini
[tool:pytest]
# Test discovery and execution settings
testpaths = . inference utils test_data
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output and reporting
addopts = -v --tb=short --disable-warnings --strict-markers

# Markers for different test types
markers =
    slow: marks tests as slow
    pdf: marks tests that generate PDF files
    performance: marks performance-related tests
    integration: marks integration tests
    unit: marks unit tests
    inference: marks inference-related tests
    utils: marks utility function tests
    data: marks data handling tests
```

### Key Benefits

- **Unified Configuration**: Single configuration file for all tests
- **Consistent Behavior**: Same settings across all test modules
- **Easy Maintenance**: Centralized configuration management
- **CI/CD Friendly**: Consistent testing behavior in automated environments

## 🏷️ Test Markers

Use markers to organize and selectively run tests:

- **`@pytest.mark.slow`**: Tests that take longer to execute
- **`@pytest.mark.pdf`**: Tests that generate PDF files
- **`@pytest.mark.performance`**: Performance and benchmarking tests
- **`@pytest.mark.integration`**: Integration and end-to-end tests
- **`@pytest.mark.unit`**: Unit tests for individual functions
- **`@pytest.mark.inference`**: Tests specific to inference functionality
- **`@pytest.mark.utils`**: Tests for utility functions
- **`@pytest.mark.data`**: Tests for data handling functionality

### Using Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run inference tests but exclude slow ones
pytest -m "inference and not slow"

# Run multiple marker combinations
pytest -m "pdf or performance"
```

## 📊 Test Coverage

### Coverage Reporting

```bash
# Generate HTML coverage report
pytest --cov=dnallm --cov-report=html

# Generate XML coverage report for CI
pytest --cov=dnallm --cov-report=xml

# Show coverage in terminal
pytest --cov=dnallm --cov-report=term-missing
```

### Coverage Targets

- **Overall Coverage**: Aim for >80% code coverage
- **Critical Modules**: >90% for core functionality
- **New Features**: >95% for newly added code

## 🔄 Continuous Integration

### CI/CD Integration

The test suite is designed for automated testing:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=dnallm --cov-report=xml --junitxml=test-results.xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Test Artifacts

- **Coverage Reports**: XML and HTML formats
- **Test Results**: JUnit XML format
- **PDF Outputs**: Generated charts and visualizations
- **Performance Metrics**: Timing and memory usage data

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/DNALLM
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   pytest
   ```

2. **Missing Dependencies**
   ```bash
   pip install pytest pytest-cov pytest-xdist
   ```

3. **Configuration Issues**
   ```bash
   # Check pytest configuration
   pytest --collect-only
   
   # Debug test discovery
   pytest --collect-only -v
   ```

### Debug Mode

```bash
# Run with detailed output
pytest -s --tb=long

# Stop on first failure
pytest -x

# Run specific test with debug output
pytest -s -v tests/inference/test_plot.py::TestPlotBars
```

## 📝 Adding New Tests

### Test File Naming

- Use `test_*.py` naming convention
- Place in appropriate module directory
- Follow existing structure and patterns

### Test Class Naming

- Use `Test*` naming convention
- Group related tests together
- Use descriptive class names

### Test Method Naming

- Use `test_*` naming convention
- Be descriptive about what is being tested
- Include edge cases and error conditions

### Example Test Structure

```python
import pytest
from dnallm.module.function import function_to_test

class TestFunctionName:
    """Test cases for function_name function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test("input")
        assert result == "expected_output"
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test("")
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance characteristics."""
        # Performance test implementation
        pass
```

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Pytest Configuration](https://docs.pytest.org/en/stable/reference/customize.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

## 🤝 Contributing

When contributing to the test suite:

1. Follow existing naming conventions
2. Use appropriate markers
3. Include comprehensive test coverage
4. Add documentation for new test functionality
5. Ensure tests pass in CI environment
6. Update this README if adding new test categories

## 📄 License

This test suite is part of the DNALLM project and follows the same license terms.
