"""Integration tests for MCP server."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from ..server import DNALLMMCPServer


class TestDNALLMMCPServer:
    """Test DNALLMMCPServer functionality."""

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
                }
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

        return str(server_config_path)

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            # Mock the model loading to avoid actual model loading
            with patch(
                "dnallm.mcp.model_manager.load_model_and_tokenizer"
            ) as mock_load:
                mock_model = Mock()
                mock_tokenizer = Mock()
                mock_load.return_value = (mock_model, mock_tokenizer)

                server = DNALLMMCPServer(config_path)
                await server.initialize()

                assert server._initialized is True
                assert server.app is not None
                assert server.sse_app is None  # SSE is built into FastMCP
                assert server.config_manager is not None
                assert server.model_manager is not None

    def test_server_info(self):
        """Test getting server information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            server = DNALLMMCPServer(config_path)
            info = server.get_server_info()

            assert "name" in info
            assert "version" in info
            assert "description" in info
            assert "host" in info
            assert "port" in info

    @pytest.mark.asyncio
    async def test_server_shutdown(self):
        """Test server shutdown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_configs(temp_path)

            server = DNALLMMCPServer(config_path)
            await server.initialize()

            # Shutdown should complete without errors
            await server.shutdown()
            assert server._initialized is False

    def test_server_without_config_file(self):
        """Test server with missing config file."""
        server = DNALLMMCPServer("nonexistent_config.yaml")

        # Should handle missing config gracefully
        info = server.get_server_info()
        assert "error" in info or info.get("name") is None


class TestMCPTools:
    """Test MCP tool functionality."""

    @pytest.mark.asyncio
    async def test_dna_sequence_predict_tool(self):
        """Test DNA sequence prediction tool."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create minimal config
            server_config = {
                "server": {
                    "host": "0.0.0.0",  # noqa: S104
                    "port": 8000,
                    "workers": 1,
                    "log_level": "INFO",
                    "debug": False,
                },
                "mcp": {
                    "name": "Test",
                    "version": "0.1.0",
                    "description": "Test",
                },
                "models": {},
                "multi_model": {},
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

            config_path = temp_path / "server_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(server_config, f)

            server = DNALLMMCPServer(str(config_path))
            await server.initialize()

            # Test tool registration (tools are registered during initialization)
            assert server.app is not None

    @pytest.mark.asyncio
    async def test_health_check_tool(self):
        """Test health check tool."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create minimal config
            server_config = {
                "server": {
                    "host": "0.0.0.0",  # noqa: S104
                    "port": 8000,
                    "workers": 1,
                    "log_level": "INFO",
                    "debug": False,
                },
                "mcp": {
                    "name": "Test",
                    "version": "0.1.0",
                    "description": "Test",
                },
                "models": {},
                "multi_model": {},
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

            config_path = temp_path / "server_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(server_config, f)

            server = DNALLMMCPServer(str(config_path))
            await server.initialize()

            # Health check should work even without loaded models
            info = server.get_server_info()
            assert "name" in info


if __name__ == "__main__":
    pytest.main([__file__])
