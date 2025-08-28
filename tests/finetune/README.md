# DNATrainer Tests

This directory contains comprehensive tests for the `DNATrainer` class from the DNALLM package.

## Test Files

### 1. `test_trainer.py` - Unit Tests
Comprehensive unit tests that cover all major functionality of the `DNATrainer` class using mocked objects and data.

**Features tested:**
- Basic initialization and setup
- Training configuration
- Model training, evaluation, and prediction
- LoRA integration
- Multi-GPU support
- Error handling
- Different task types (classification, regression, masked language modeling)
- Dataset handling (single dataset, DatasetDict)

**Requirements:**
- pytest
- torch
- datasets
- No external model downloads required

### 2. `test_trainer_real_model.py` - Integration Tests
Integration tests that use real models from ModelScope and real datasets to test end-to-end functionality.

**Features tested:**
- Loading real models and tokenizers from ModelScope
- Loading real datasets from ModelScope
- Training with real data
- Evaluation and prediction with real models
- Error handling with real data

**Models and datasets used:**
- Model: `zhangtaolab/plant-dnabert-BPE`
- Dataset: `zhangtaolab/plant-multi-species-core-promoters`

**Requirements:**
- pytest
- torch (with CUDA support recommended)
- modelscope
- transformers
- datasets
- Internet connection for downloading models and datasets

### 3. `run_tests.py` - Test Runner Script
A convenient script to run different types of tests with various options.

## Running the Tests

### Prerequisites

1. **Install required packages:**
   ```bash
   pip install pytest torch datasets transformers
   ```

2. **For integration tests, also install:**
   ```bash
   pip install modelscope
   ```

3. **Ensure you have access to ModelScope:**
   - Some models and datasets may require authentication
   - Check ModelScope documentation for setup instructions

### Running Unit Tests Only

```bash
# From the tests/finetune directory
python run_tests.py --type unit

# Or directly with pytest
python -m pytest test_trainer.py -v
```

### Running Integration Tests Only

```bash
# From the tests/finetune directory
python run_tests.py --type real

# Or directly with pytest
python -m pytest test_trainer_real_model.py -v
```

### Running All Tests

```bash
# From the tests/finetune directory
python run_tests.py --type all

# Or directly with pytest
python -m pytest . -v
```

### Running with Verbose Output

```bash
python run_tests.py --type unit --verbose
```

### Running Specific Test Markers

```bash
# Run only slow tests
python run_tests.py --type real --markers slow

# Run multiple markers
python run_tests.py --type all --markers slow not slow
```

## Test Configuration

### Unit Tests
Unit tests use mocked objects and don't require external resources. They run quickly and are suitable for:
- Development and debugging
- CI/CD pipelines
- Quick validation of changes

### Integration Tests
Integration tests use real models and data, making them more comprehensive but slower. They're suitable for:
- End-to-end validation
- Performance testing
- Real-world scenario validation

**Note:** Integration tests may take longer to run and require more computational resources.

## Test Structure

### Fixtures
Both test files use pytest fixtures to set up test data and configurations:

- **Mock objects:** For unit tests
- **Real models/datasets:** For integration tests
- **Temporary directories:** For output files
- **Configurations:** Test-specific settings

### Test Categories

1. **Initialization Tests:** Verify proper setup of the trainer
2. **Configuration Tests:** Test different configuration options
3. **Training Tests:** Test training functionality
4. **Evaluation Tests:** Test evaluation and prediction
5. **Error Handling Tests:** Test error conditions and edge cases
6. **Integration Tests:** Test with real data and models

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed
   - Verify Python path includes the project root

2. **ModelScope Access Issues:**
   - Check internet connection
   - Verify ModelScope authentication
   - Some models may require specific permissions

3. **CUDA Issues:**
   - Integration tests require CUDA for optimal performance
   - Tests will skip if CUDA is not available
   - Check CUDA installation and compatibility

4. **Memory Issues:**
   - Real model tests may require significant memory
   - Reduce batch sizes in test configurations if needed
   - Use smaller datasets for testing

### Debug Mode

Run tests with increased verbosity for debugging:

```bash
python -m pytest test_trainer.py -v -s --tb=long
```

### Skipping Tests

Tests automatically skip if dependencies are not available:

```bash
# Check what's being skipped
python -m pytest test_trainer_real_model.py -v -rs
```

## Contributing

When adding new tests:

1. **Follow the existing pattern** for test structure
2. **Use appropriate fixtures** for test data
3. **Add comprehensive assertions** to verify functionality
4. **Include error handling tests** for edge cases
5. **Document new test requirements** in this README

## Performance Notes

- **Unit tests:** Should complete in seconds
- **Integration tests:** May take minutes depending on:
  - Model size
  - Dataset size
  - Hardware capabilities
  - Network speed (for downloads)

## CI/CD Integration

These tests are designed to work in CI/CD environments:

- Unit tests run quickly and reliably
- Integration tests can be marked as optional
- Tests handle missing dependencies gracefully
- Output is suitable for automated reporting
