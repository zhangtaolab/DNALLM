"""Tests for configuration validators."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic_core import ValidationError

from ..config_validators import (
    TaskConfig,
    InferenceConfig,
    InferenceModelConfig,
    ModelInfoConfig,
    ModelConfig,
    ModelEntryConfig,
    MCPServerConfig,
    validate_mcp_server_config,
    validate_inference_model_config,
)


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
                threshold=0.5,
                description="Test",
            )

    def test_invalid_threshold(self):
        """Test invalid threshold value."""
        with pytest.raises(ValidationError):
            TaskConfig(
                task_type="binary",
                num_labels=2,
                label_names=["A", "B"],
                threshold=1.5,  # Invalid threshold
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
            use_fp16=False,
            use_bf16=False,
            output_dir=tempfile.mkdtemp(),
            save_predictions=True,
            save_hidden_states=False,
            save_attentions=False,
        )
        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.device == "auto"

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                batch_size=200,  # Too large
                max_length=512,
                device="auto",
                num_workers=4,
                precision="float16",
                use_fp16=False,
                use_bf16=False,
                output_dir=tempfile.mkdtemp(),
                save_predictions=True,
                save_hidden_states=False,
                save_attentions=False,
            )

    def test_invalid_device(self):
        """Test invalid device."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                batch_size=16,
                max_length=512,
                device="invalid_device",
                num_workers=4,
                precision="float16",
                use_fp16=False,
                use_bf16=False,
                output_dir=tempfile.mkdtemp(),
                save_predictions=True,
                save_hidden_states=False,
                save_attentions=False,
            )


class TestInferenceModelConfig:
    """Test InferenceModelConfig validation."""

    def test_valid_model_config(self):
        """Test valid model configuration."""
        task_config = TaskConfig(
            task_type="binary",
            num_labels=2,
            label_names=["A", "B"],
            threshold=0.5,
            description="Test task",
        )

        inference_config = InferenceConfig(
            batch_size=16,
            max_length=512,
            device="auto",
            num_workers=4,
            precision="float16",
            use_fp16=False,
            use_bf16=False,
            output_dir=tempfile.mkdtemp(),
            save_predictions=True,
            save_hidden_states=False,
            save_attentions=False,
        )

        model_info = ModelInfoConfig(
            architecture="DNABERT",
            tokenizer="BPE",
            species="plant",
            task_category="test",
        )

        model_config = ModelConfig(
            name="test_model",
            path="test/path",
            source="huggingface",
            task_info=model_info,
        )

        config = InferenceModelConfig(
            task=task_config,
            inference=inference_config,
            model=model_config,
        )
        assert config.task.task_type == "binary"
        assert config.inference.batch_size == 16
        assert config.model.name == "test_model"


class TestValidatorEdgeBranches:
    """Cover the validator rejection/acceptance pairs not yet exercised."""

    def test_use_fp16_falls_back_when_precision_invalid(self):
        """With precision failed, the fp16 before-validator keeps the input."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                batch_size=16,
                device="auto",
                precision="quantum",  # invalid: precision validation fails first
                use_fp16=True,
                output_dir="validator-output-dir",
            )

    def test_use_bf16_falls_back_when_precision_invalid(self):
        """With precision failed, the bf16 before-validator keeps the input."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                batch_size=16,
                device="auto",
                precision="quantum",
                use_bf16=True,
                output_dir="validator-output-dir",
            )

    def test_precision_drives_fp16_flag(self):
        """A valid precision auto-sets the matching half-precision flag."""
        config = InferenceConfig(
            batch_size=16,
            device="auto",
            precision="float16",
            use_fp16=False,
            output_dir="validator-output-dir",
        )
        assert config.use_fp16 is True

    def test_num_labels_must_match_label_names(self):
        """InferenceModelConfig rejects a num_labels/label_names mismatch."""
        task = TaskConfig(
            task_type="binary",
            num_labels=3,
            label_names=["a", "b"],
            description="mismatched",
        )
        inference = InferenceConfig(batch_size=16, device="auto", output_dir="validator-output-dir")
        model = ModelConfig(
            name="m",
            path="p",
            source="huggingface",
            task_info=ModelInfoConfig(
                architecture="A", tokenizer="T", species="s", task_category="c"
            ),
        )

        with pytest.raises(ValidationError, match="num_labels must match"):
            InferenceModelConfig(task=task, inference=inference, model=model)

    def test_whitespace_config_path_rejected(self):
        """A whitespace-only config_path is rejected as empty."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ModelEntryConfig(
                name="m",
                model_name="M",
                config_path="   ",
            )

    @staticmethod
    def _server_config_dict():
        """Minimal valid MCPServerConfig payload."""
        return {
            "server": {
                "host": "127.0.0.1",
                "port": 8123,
                "workers": 1,
                "log_level": "INFO",
                "debug": False,
            },
            "mcp": {"name": "n", "version": "0.1.0", "description": "d"},
            "models": {
                "m1": {
                    "name": "duplicate-name",
                    "model_name": "M1",
                    "config_path": "./a.yaml",
                    "enabled": True,
                    "priority": 1,
                }
            },
            "multi_model": {},
            "sse": {
                "heartbeat_interval": 30,
                "max_connections": 100,
                "connection_timeout": 300,
                "enable_compression": True,
            },
            "logging": {
                "level": "INFO",
                "format": "f",
                "file": "./logs/x.log",
                "max_size": "10MB",
                "backup_count": 5,
            },
        }

    def test_duplicate_model_names_rejected(self):
        """Two entries sharing one name are rejected."""
        config = self._server_config_dict()
        config["models"]["m2"] = {
            "name": "duplicate-name",
            "model_name": "M2",
            "config_path": "./b.yaml",
            "enabled": True,
            "priority": 2,
        }

        with pytest.raises(ValidationError, match="must be unique"):
            MCPServerConfig(**config)

    def test_multi_model_dangling_reference_rejected(self):
        """A multi-model group naming an undefined model is rejected."""
        config = self._server_config_dict()
        config["multi_model"] = {
            "group": {
                "name": "group",
                "description": "references m-ghost",
                "models": ["m1", "m-ghost"],
                "enabled": True,
            }
        }

        with pytest.raises(ValidationError, match="referenced in multi-model"):
            MCPServerConfig(**config)


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
                "use_fp16": False,
                "use_bf16": False,
                "output_dir": tempfile.mkdtemp(),
                "save_predictions": True,
                "save_hidden_states": False,
                "save_attentions": False,
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = validate_inference_model_config(config_path)
            assert config.task.task_type == "binary"
            assert config.model.name == "test_model"
        finally:
            Path(config_path).unlink()

    def test_validate_mcp_server_config_file(self):
        """Test validation of MCP server config file."""
        config_data = {
            "server": {
                "host": "0.0.0.0",  # ruff: ignore[hardcoded-bind-all-interfaces]
                "port": 8000,
                "workers": 1,
                "log_level": "INFO",
                "debug": False,
            },
            "mcp": {
                "name": "Test MCP Server",
                "version": "0.1.0",
                "description": "Test server",
            },
            "models": {
                "test_model": {
                    "name": "test_model",
                    "model_name": "Test Model",
                    "config_path": "./test_config.yaml",
                    "enabled": True,
                    "priority": 1,
                },
                "test_model2": {
                    "name": "test_model2",
                    "model_name": "Test Model 2",
                    "config_path": "./test_config2.yaml",
                    "enabled": True,
                    "priority": 2,
                },
            },
            "multi_model": {
                "test_multi": {
                    "name": "test_multi",
                    "description": "Test multi-model",
                    "models": ["test_model", "test_model2"],
                    "enabled": True,
                }
            },
            "sse": {
                "heartbeat_interval": 30,
                "max_connections": 100,
                "connection_timeout": 300,
                "enable_compression": True,
            },
            "logging": {
                "level": "INFO",
                "format": ("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "file": "./logs/test.log",
                "max_size": "10MB",
                "backup_count": 5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        # Initialize dummy paths before try block
        dummy_config_path = Path(config_path).parent / "test_config.yaml"
        dummy_config2_path = Path(config_path).parent / "test_config2.yaml"

        try:
            # Create dummy config files for the models
            dummy_config_path.write_text("dummy: config")
            dummy_config2_path.write_text("dummy: config2")

            config = validate_mcp_server_config(config_path)
            assert config.server.host == "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]
            assert config.mcp.name == "Test MCP Server"
            assert len(config.models) == 2
        finally:
            Path(config_path).unlink()
            if dummy_config_path.exists():
                dummy_config_path.unlink()
            if dummy_config2_path.exists():
                dummy_config2_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
