"""Pytest configuration for DNATrainer tests.

This file contains common fixtures and configuration settings
that are shared across all test files in this directory.
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def test_session_dir():
    """Create a temporary directory for the entire test session."""
    temp_dir = tempfile.mkdtemp(prefix="dnallm_test_session_")
    yield temp_dir
    # Cleanup after all tests
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def test_output_dir(test_session_dir):
    """Create a temporary output directory for individual tests."""
    test_dir = os.path.join(test_session_dir, f"test_{id(test_output_dir)}")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup individual test directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def mock_torch_device():
    """Mock torch device for testing."""
    device = Mock()
    device.type = "cuda"
    device.index = 0
    return device


@pytest.fixture(scope="function")
def sample_dna_sequences():
    """Sample DNA sequences for testing."""
    return [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA",
        "TATATATATATATATA",
        "CGCGCGCGCGCGCGCG"
    ]


@pytest.fixture(scope="function")
def sample_labels():
    """Sample labels for testing."""
    return [0, 1, 0, 1]


@pytest.fixture(scope="function")
def basic_task_config():
    """Basic task configuration for testing."""
    return {
        "task_type": "binary",
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "threshold": 0.5
    }


@pytest.fixture(scope="function")
def basic_finetune_config(test_output_dir):
    """Basic fine-tuning configuration for testing."""
    return {
        "output_dir": test_output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "logging_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "save_total_limit": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "report_to": "none",
        "save_safetensors": True,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False
    }


@pytest.fixture(scope="function")
def lora_config():
    """LoRA configuration for testing."""
    return {
        "task_type": "SEQUENCE_CLASSIFICATION",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "bias": "none"
    }


@pytest.fixture(scope="function")
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.cpu = Mock(return_value=model)
    model.cuda = Mock(return_value=model)
    return model


@pytest.fixture(scope="function")
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.save_pretrained = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
    tokenizer.decode = Mock(return_value="ATCG")
    return tokenizer


@pytest.fixture(scope="function")
def mock_datasets(mock_tokenizer, sample_dna_sequences, sample_labels):
    """Mock datasets for testing."""
    from datasets import Dataset, DatasetDict
    
    # Create sample data
    sample_data = {
        "sequence": sample_dna_sequences,
        "labels": sample_labels
    }
    dataset = Dataset.from_dict(sample_data)
    dataset_dict = DatasetDict({
        "train": dataset,
        "val": dataset
    })
    
    datasets = Mock()
    datasets.dataset = dataset_dict
    datasets.tokenizer = mock_tokenizer
    return datasets


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        if "real_model" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


# Environment setup
def pytest_sessionstart(session):
    """Set up test environment at the start of the session."""
    # Set environment variables for testing
    os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")
    
    # Disable some features during testing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"Test session started. Project root: {project_root}")


def pytest_sessionfinish(session, exitstatus):
    """Clean up test environment at the end of the session."""
    print(f"Test session finished with exit status: {exitstatus}")


# Skip certain tests based on environment
def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip integration tests if no internet connection
    if "integration" in item.keywords:
        try:
            import requests
            requests.get("https://www.google.com", timeout=5)
        except:
            pytest.skip("Internet connection required for integration tests")
    
    # Skip GPU tests if CUDA not available
    if "gpu" in item.keywords:
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for GPU tests")
        except ImportError:
            pytest.skip("PyTorch not available for GPU tests")
