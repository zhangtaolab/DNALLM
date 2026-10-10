"""Tests for configuration manager."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic_core import ValidationError

from ..config_manager import MCPConfigManager


class TestMCPConfigManager:
    """Test MCPConfigManager functionality."""

    def create_test_configs(self, temp_dir):
        """Create test configuration files."""
        # Create main server config
        server_config = {
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
                "format": ("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
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
        model_config2["model"]["name"] = "test_model2"  # type: ignore[index]
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


class TestMCPConfigManagerBranches:
    """Cover the config-manager branches not exercised by the happy path."""

    def _minimal_server_config(self, **overrides):
        """Build a minimal valid server config dict with overrides applied."""
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": 8123,
                "workers": 1,
                "log_level": "INFO",
                "debug": False,
            },
            "mcp": {
                "name": "Branch Test",
                "version": "0.1.0",
                "description": "test",
            },
            "models": {
                "test_model": {
                    "name": "test_model",
                    "model_name": "Test Model",
                    "config_path": "./test_model_config.yaml",
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
                "format": "%(asctime)s - %(message)s",
                "file": "./logs/test.log",
                "max_size": "10MB",
                "backup_count": 5,
            },
        }
        config.update(overrides)
        return config

    def _write_model_config(self, temp_dir):
        """Write the referenced model config next to the server config."""
        model_config = {
            "task": {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["a", "b"],
                "description": "test",
            },
            "inference": {
                "batch_size": 16,
                "output_dir": str(temp_dir / "out"),
            },
            "model": {
                "name": "test_model",
                "path": "p",
                "source": "huggingface",
                "task_info": {
                    "architecture": "A",
                    "tokenizer": "T",
                    "species": "s",
                    "task_category": "c",
                },
            },
        }
        with open(temp_dir / "test_model_config.yaml", "w") as f:
            yaml.dump(model_config, f)

    def test_invalid_server_config_raises(self, tmp_path):
        """A malformed server config propagates the validation error."""
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump({"server": "not-a-mapping"}, f)

        with pytest.raises(ValidationError):
            MCPConfigManager(str(tmp_path))

    def test_load_model_configs_without_server_config(self):
        """_load_model_configs no-ops (with an error log) when unloaded."""
        manager = MCPConfigManager("nonexistent_config.yaml")
        manager._load_model_configs()  # must not raise
        assert manager.model_configs == {}

    def test_disabled_model_is_skipped(self, tmp_path):
        """A disabled model entry is not loaded."""
        config = self._minimal_server_config()
        config["models"]["test_model"]["enabled"] = False
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(config, f)

        manager = MCPConfigManager(str(tmp_path))

        assert manager.get_enabled_models() == []
        assert manager.model_configs == {}

    def test_get_model_configs_returns_copy(self, tmp_path):
        """get_model_configs returns a copy, not the internal dict."""
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(self._minimal_server_config(), f)
        self._write_model_config(tmp_path)
        manager = MCPConfigManager(str(tmp_path))

        returned = manager.get_model_configs()
        assert set(returned) == {"test_model"}
        returned.clear()
        assert set(manager.model_configs) == {"test_model"}

    def test_get_model_priority_unknown_defaults_to_one(self, tmp_path):
        """An unknown model (or no config) yields the default priority 1."""
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(self._minimal_server_config(), f)
        self._write_model_config(tmp_path)
        manager = MCPConfigManager(str(tmp_path))

        assert manager.get_model_priority("ghost") == 1

        bare = MCPConfigManager("nonexistent_config.yaml")
        assert bare.get_model_priority("anything") == 1

    def test_getters_without_config_return_defaults(self):
        """Every getter has a safe default with no server config loaded."""
        manager = MCPConfigManager("nonexistent_config.yaml")

        assert manager.get_multi_model_configs() == {}
        assert manager.get_sse_config() == {}
        assert manager.get_streamable_http_config() == {}
        assert manager.get_logging_config() == {}
        assert manager.get_timeout_config() == {"tool_timeout_seconds": 30}
        assert manager.validate_model_references() == ["Server configuration not loaded"]

    def test_get_streamable_http_config_defaults_without_block(self, tmp_path):
        """Without a streamable_http block the server host/port + /mcp apply."""
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(self._minimal_server_config(), f)
        self._write_model_config(tmp_path)
        manager = MCPConfigManager(str(tmp_path))

        config = manager.get_streamable_http_config()

        assert config == {"host": "127.0.0.1", "port": 8123, "path": "/mcp"}

    def test_get_streamable_http_config_from_block(self, tmp_path):
        """A streamable_http block supplies host/port/path directly."""
        config = self._minimal_server_config()
        config["streamable_http"] = {"host": "127.0.0.1", "port": 9000, "path": "/custom"}
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(config, f)
        self._write_model_config(tmp_path)
        manager = MCPConfigManager(str(tmp_path))

        assert manager.get_streamable_http_config() == {
            "host": "127.0.0.1",
            "port": 9000,
            "path": "/custom",
        }

    def test_reload_configurations_picks_up_changes(self, tmp_path):
        """reload_configurations re-reads the files from disk."""
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(self._minimal_server_config(), f)
        self._write_model_config(tmp_path)
        manager = MCPConfigManager(str(tmp_path))
        assert set(manager.model_configs) == {"test_model"}

        # Drop the model from the config file, then reload
        updated = self._minimal_server_config()
        updated["models"]["test_model"]["enabled"] = False
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(updated, f)
        manager.reload_configurations()

        assert manager.model_configs == {}
        assert manager.get_enabled_models() == []

    def test_validate_model_references_flags_dangling_and_unloaded(self, tmp_path):
        """Both dangling multi-model refs and unloaded configs are flagged."""
        config = self._minimal_server_config()
        # Point the model entry at a config file that does not exist so the
        # model stays enabled but never loads.
        config["models"]["test_model"]["config_path"] = "./missing.yaml"
        with open(tmp_path / "mcp_server_config.yaml", "w") as f:
            yaml.dump(config, f)
        manager = MCPConfigManager(str(tmp_path))

        # The parse-time validator forbids dangling refs in the file, so the
        # runtime guard is exercised by injecting one post-parse.
        from ..config_validators import MultiModelConfig

        manager.server_config.multi_model["dangling"] = MultiModelConfig(
            name="dangling",
            description="references missing models",
            models=["ghost-a", "ghost-b"],
            enabled=True,
        )

        errors = manager.validate_model_references()

        assert any("references non-existent model 'ghost-a'" in e for e in errors)
        assert any("Model configuration not loaded for 'test_model'" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__])
