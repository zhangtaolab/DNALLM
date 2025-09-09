"""Simple tests for MCP server components."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic_core import ValidationError

from dnallm.mcp.config_validators import (
    TaskConfig,
    InferenceConfig,
    validate_inference_model_config,
)

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTaskConfig:
    """Test TaskConfig validation."""

    def test_valid_binary_task(self):
        """Test valid binary classification task."""
        config = TaskConfig(
            task_type="binary",
            num_labels=2,
            label_names=["Not promoter", "Core promoter"],
            threshold=0.5,
            description="Test binary task",
        )
        assert config.task_type == "binary"
        assert config.num_labels == 2
        assert len(config.label_names) == 2

    def test_invalid_task_type(self):
        """Test invalid task type."""
        with pytest.raises(ValidationError):
            TaskConfig(
                task_type="invalid",
                num_labels=2,
                label_names=["A", "B"],
                description="Test",
            )


class TestInferenceConfig:
    """Test InferenceConfig validation."""

    def test_valid_inference_config(self):
        """Test valid inference configuration."""
        config = InferenceConfig(
            batch_size=16,
            max_length=512,
            device="auto",
            num_workers=4,
            precision="float16",
            output_dir=tempfile.mkdtemp(),
            save_predictions=True,
        )
        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.device == "auto"


class TestConfigFileValidation:
    """Test configuration file validation."""

    def test_validate_inference_model_config_file(self):
        """Test validation of inference model config file."""
        config_data = {
            "task": {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["Not promoter", "Core promoter"],
                "threshold": 0.5,
                "description": "Test promoter prediction",
            },
            "inference": {
                "batch_size": 16,
                "max_length": 512,
                "device": "auto",
                "num_workers": 4,
                "precision": "float16",
                "output_dir": tempfile.mkdtemp(),
                "save_predictions": True,
            },
            "model": {
                "name": "test_model",
                "path": "test/path",
                "source": "huggingface",
                "task_info": {
                    "architecture": "DNABERT",
                    "tokenizer": "BPE",
                    "species": "plant",
                    "task_category": "promoter_prediction",
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = validate_inference_model_config(config_path)
            assert config.task.task_type == "binary"
            assert config.model.name == "test_model"
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
