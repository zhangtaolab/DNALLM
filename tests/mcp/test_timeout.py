"""Tests for MCP tool timeout behavior.

This module tests the timeout wrapper and chunk-based timeout
for all MCP tools, ensuring proper error responses and
configurability.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dnallm.mcp.server import DNALLMMCPServer


class TestToolTimeout:
    """Test timeout behavior for MCP tools."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server with minimal setup for timeout tests."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            with patch("dnallm.mcp.server.ModelManager"):
                # Configure mock config manager
                mock_config = MagicMock()
                mock_config.mcp.name = "Test Server"
                mock_config.mcp.description = "Test"
                mock_config.mcp.version = "0.1.0"
                mock_config.server.host = "127.0.0.1"
                mock_config.server.port = 8000
                mock_config.tool_timeout_seconds = 30
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                mock_config.logging.file = "./logs/test.log"
                mock_config.logging.max_size = "10MB"
                mock_config.logging.backup_count = 5
                mock_config.logging.log_format = "text"

                mock_cm_instance = MagicMock()
                mock_cm_instance.get_server_config.return_value = mock_config
                mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 30}
                mock_cm_instance.get_logging_config.return_value = {"log_format": "text"}
                mock_cm.return_value = mock_cm_instance

                server = DNALLMMCPServer("dummy_config.yaml")
                server._tool_timeout_seconds = 30
                server._log_format = "text"
                return server

    @pytest.mark.asyncio
    async def test_tool_timeout_returns_error(self, mock_server):
        """Verify that a slow coroutine returns a timeout error dict."""
        async def slow_tool():
            await asyncio.sleep(100)  # Will definitely timeout
            return {"result": "ok"}

        wrapper = mock_server._with_timeout_wrapper(slow_tool, "slow_tool")
        result = await wrapper()

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert result["timeout_seconds"] == 30
        assert result["tool_name"] == "slow_tool"
        assert "suggestion" in result
        assert any("Timeout after 30s" in item["text"] for item in result["content"])

    @pytest.mark.asyncio
    async def test_tool_completes_within_timeout(self, mock_server):
        """Verify that a fast coroutine returns its normal result."""
        async def fast_tool():
            await asyncio.sleep(0.01)
            return {"result": "success", "data": "test", "isError": False}

        wrapper = mock_server._with_timeout_wrapper(fast_tool, "fast_tool")
        result = await wrapper()

        assert result.get("isError") is False
        assert result["result"] == "success"
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_timeout_configurable(self, mock_server):
        """Verify that timeout can be configured to a shorter value."""
        mock_server._tool_timeout_seconds = 0.1

        async def slow_tool():
            await asyncio.sleep(1)
            return {"result": "ok"}

        wrapper = mock_server._with_timeout_wrapper(slow_tool, "configurable_tool")
        result = await wrapper()

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert result["timeout_seconds"] == 0.1

    @pytest.mark.asyncio
    async def test_streaming_chunk_timeout(self, mock_server):
        """Verify chunk-based timeout in streaming tools."""
        mock_server._tool_timeout_seconds = 0.1

        async def slow_predict(*args, **kwargs):
            await asyncio.sleep(1)  # Slow prediction that will timeout
            return {"probabilities": [0.5, 0.5]}

        # Mock model_manager and context
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_sequence = slow_predict

        mock_context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test_model",
            stream_progress=True,
            context=mock_context,
        )

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert result["timeout_seconds"] == 0.1
        assert result["tool_name"] == "dna_stream_predict"

    def test_timeout_error_structure(self, mock_server):
        """Verify timeout error response contains all required fields."""
        # Create a mock coroutine that will timeout
        async def never_completes():
            await asyncio.sleep(1000)

        wrapper = mock_server._with_timeout_wrapper(never_completes, "test_tool")

        # Run the wrapper and get the result
        result = asyncio.run(wrapper())

        # Verify all required fields
        assert "isError" in result
        assert "content" in result
        assert "error_type" in result
        assert "timeout_seconds" in result
        assert "tool_name" in result
        assert "suggestion" in result

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert isinstance(result["timeout_seconds"], int)
        assert result["timeout_seconds"] == 30
        assert result["tool_name"] == "test_tool"
        assert isinstance(result["suggestion"], str)
        assert len(result["suggestion"]) > 0

        # Verify content is a list with text items
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "Timeout after" in result["content"][0]["text"]


class TestStreamingTimeout:
    """Test chunk-based timeout for streaming tools."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server with short timeout for streaming tests."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            with patch("dnallm.mcp.server.ModelManager"):
                mock_config = MagicMock()
                mock_config.mcp.name = "Test Server"
                mock_config.mcp.description = "Test"
                mock_config.mcp.version = "0.1.0"
                mock_config.server.host = "127.0.0.1"
                mock_config.server.port = 8000
                mock_config.tool_timeout_seconds = 0.2
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                mock_config.logging.file = "./logs/test.log"
                mock_config.logging.max_size = "10MB"
                mock_config.logging.backup_count = 5
                mock_config.logging.log_format = "text"

                mock_cm_instance = MagicMock()
                mock_cm_instance.get_server_config.return_value = mock_config
                mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 0.2}
                mock_cm_instance.get_logging_config.return_value = {"log_format": "text"}
                mock_cm.return_value = mock_cm_instance

                server = DNALLMMCPServer("dummy_config.yaml")
                server._tool_timeout_seconds = 0.2
                server._log_format = "text"
                return server

    @pytest.mark.asyncio
    async def test_stream_batch_timeout(self, mock_server):
        """Verify stream batch prediction times out on slow chunks."""
        async def slow_predict(*args, **kwargs):
            await asyncio.sleep(1)
            return {"probabilities": [0.5, 0.5]}

        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_sequence = slow_predict

        mock_context = AsyncMock()

        result = await mock_server._dna_stream_batch_predict(
            sequences=["ATCG", "GCTA"],
            model_name="test_model",
            stream_progress=True,
            context=mock_context,
        )

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert result["tool_name"] == "dna_stream_batch_predict"

    @pytest.mark.asyncio
    async def test_stream_multi_model_timeout(self, mock_server):
        """Verify stream multi-model prediction times out on slow chunks."""
        async def slow_predict(*args, **kwargs):
            await asyncio.sleep(1)
            return {"probabilities": [0.5, 0.5]}

        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.return_value = ["model1", "model2"]
        mock_server.model_manager.predict_sequence = slow_predict

        mock_context = AsyncMock()

        result = await mock_server._dna_stream_multi_model_predict(
            sequence="ATCG",
            model_names=["model1", "model2"],
            stream_progress=True,
            context=mock_context,
        )

        assert result["isError"] is True
        assert result["error_type"] == "timeout"
        assert result["tool_name"] == "dna_stream_multi_model_predict"

    @pytest.mark.asyncio
    async def test_streaming_completes_within_timeout(self, mock_server):
        """Verify streaming succeeds when operations are fast."""
        mock_server._tool_timeout_seconds = 5

        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_sequence = AsyncMock(
            return_value={"probabilities": [0.5, 0.5]}
        )

        mock_context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test_model",
            stream_progress=True,
            context=mock_context,
        )

        assert result.get("isError") is not True
        assert "content" in result
        assert result["streamed"] is True
