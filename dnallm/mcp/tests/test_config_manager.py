"""Tests for configuration manager."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ..config_manager import MCPConfigManager


class TestMCPConfigManager:
    """Test MCPConfigManager functionality."""

    def create_test_configs(self, temp_dir):
        """Create test configuration files."""
        # Create main server config
        server_config = {
            "server": {
                "host": "0.0.0.0",  # noqa: S104
                "port": 8000,
                "workers": 1,
                "log_level": "INFO",
                "debug": False,
            },
            "mcp": {
                "name": "Test MCP Server",
                "version": "0.1.0",
                "description": "Test server for DNA prediction",
            },
            "models": {
                "test_model": {
                    "name": "test_model",
                    "model_name": "Test Model",
                    "config_path": "./test_model_config.yaml",
                    "enabled": True,
                    "priority": 1,
                },
                "test_model2": {
                    "name": "test_model2",
                    "model_name": "Test Model 2",
                    "config_path": "./test_model2_config.yaml",
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
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/test.log",
                "max_size": "10MB",
                "backup_count": 5,
            },
        }

        # Create model config
        model_config = {
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

        # Write config files
        server_config_path = temp_dir / "mcp_server_config.yaml"
        with open(server_config_path, "w") as f:
            yaml.dump(server_config, f)

        model_config_path = temp_dir / "test_model_config.yaml"
        with open(model_config_path, "w") as f:
            yaml.dump(model_config, f)

        # Create second model config
        model_config2 = model_config.copy()
        model_config2["model"]["name"] = "test_model2"
        model_config2_path = temp_dir / "test_model2_config.yaml"
        with open(model_config2_path, "w") as f:
            yaml.dump(model_config2, f)

        return str(temp_dir)

    def test_config_manager_initialization(self):
        """Test MCPConfigManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)

            # Check that configurations were loaded
            assert manager.server_config is not None
            assert len(manager.model_configs) == 2
            assert "test_model" in manager.model_configs
            assert "test_model2" in manager.model_configs

    def test_get_server_config(self):
        """Test getting server configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            server_config = manager.get_server_config()

            assert server_config is not None
            assert server_config.mcp.name == "Test MCP Server"
            assert server_config.server.port == 8000

    def test_get_model_config(self):
        """Test getting model configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            model_config = manager.get_model_config("test_model")

            assert model_config is not None
            assert model_config.task.task_type == "binary"
            assert model_config.model.name == "test_model"

    def test_get_enabled_models(self):
        """Test getting enabled models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            enabled_models = manager.get_enabled_models()

            assert "test_model" in enabled_models
            assert "test_model2" in enabled_models
            assert len(enabled_models) == 2

    def test_get_model_priority(self):
        """Test getting model priority."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            priority = manager.get_model_priority("test_model")

            assert priority == 1

    def test_get_multi_model_configs(self):
        """Test getting multi-model configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            multi_configs = manager.get_multi_model_configs()

            assert "test_multi" in multi_configs
            assert multi_configs["test_multi"]["models"] == [
                "test_model",
                "test_model2",
            ]

    def test_get_sse_config(self):
        """Test getting SSE configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            sse_config = manager.get_sse_config()

            assert sse_config["heartbeat_interval"] == 30
            assert sse_config["max_connections"] == 100

    def test_get_logging_config(self):
        """Test getting logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            logging_config = manager.get_logging_config()

            assert logging_config["level"] == "INFO"
            assert logging_config["file"] == "./logs/test.log"

    def test_validate_model_references(self):
        """Test validating model references."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            errors = manager.validate_model_references()

            # Should have no errors for valid configuration
            assert len(errors) == 0

    def test_get_model_info_summary(self):
        """Test getting model information summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            manager = MCPConfigManager(config_path)
            summary = manager.get_model_info_summary()

            assert summary["total_models"] == 2
            assert summary["enabled_models"] == 2
            assert "test_model" in summary["models"]
            assert "test_model2" in summary["models"]
            assert summary["models"]["test_model"]["task_type"] == "binary"

    def test_config_manager_without_config_file(self):
        """Test MCPConfigManager without config file."""
        manager = MCPConfigManager("nonexistent_config.yaml")

        # Should handle missing config gracefully
        assert manager.server_config is None
        assert len(manager.model_configs) == 0
        assert manager.get_enabled_models() == []


if __name__ == "__main__":
    pytest.main([__file__])
