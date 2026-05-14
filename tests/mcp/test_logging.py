"""Tests for MCP structured JSON and text logging.

This module tests the _structured_log helper and its integration
with tool wrappers, verifying both JSON and text output formats.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dnallm.mcp.server import DNALLMMCPServer


class TestStructuredLogging:
    """Test structured logging output formats."""

    @pytest.fixture
    def json_server(self):
        """Create a mock server configured for JSON logging."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            with patch("dnallm.mcp.server.ModelManager"):
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
                mock_config.logging.log_format = "json"

                mock_cm_instance = MagicMock()
                mock_cm_instance.get_server_config.return_value = mock_config
                mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 30}
                mock_cm_instance.get_logging_config.return_value = {"log_format": "json"}
                mock_cm.return_value = mock_cm_instance

                server = DNALLMMCPServer("dummy_config.yaml")
                server._tool_timeout_seconds = 30
                server._log_format = "json"
                return server

    @pytest.fixture
    def text_server(self):
        """Create a mock server configured for text logging."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            with patch("dnallm.mcp.server.ModelManager"):
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

    def test_json_log_format(self, json_server, capsys):
        """Verify JSON log format produces valid JSON with required fields."""
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.opt.return_value.log = capture_log

            json_server._structured_log(
                "info",
                "Test message",
                tool_name="test_tool",
                request_id="req-123",
                duration_ms=42.5,
                status="success",
            )

            assert len(log_calls) == 1
            level, msg = log_calls[0]
            # loguru uses uppercase level names
            assert level == "INFO"

            # Parse the JSON output
            log_entry = json.loads(msg.strip())

            # Verify required fields
            assert "timestamp" in log_entry
            assert log_entry["level"] == "INFO"
            assert log_entry["message"] == "Test message"
            assert log_entry["tool_name"] == "test_tool"
            assert log_entry["request_id"] == "req-123"
            assert log_entry["duration_ms"] == 42.5
            assert log_entry["status"] == "success"

            # Verify timestamp format (ISO 8601 with Z suffix)
            assert log_entry["timestamp"].endswith("Z")
            assert "T" in log_entry["timestamp"]

    def test_text_log_format(self, text_server, capsys):
        """Verify text log format produces human-readable output."""
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.log = capture_log

            text_server._structured_log(
                "info",
                "Test message",
                tool_name="test_tool",
                duration_ms=42.5,
                status="success",
            )

            assert len(log_calls) == 1
            level, msg = log_calls[0]
            # loguru uses uppercase level names
            assert level == "INFO"
            assert "Test message" in msg
            assert "[tool=test_tool]" in msg
            assert "[duration=42.50ms]" in msg
            assert "[status=success]" in msg

    def test_log_contains_tool_name(self, json_server):
        """Verify tool_name field is present in JSON logs."""
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.opt.return_value.log = capture_log

            json_server._structured_log("info", "Tool executed", tool_name="dna_sequence_predict")

            assert len(log_calls) == 1
            log_entry = json.loads(log_calls[0][1].strip())
            assert log_entry["tool_name"] == "dna_sequence_predict"

    def test_log_contains_duration(self, json_server):
        """Verify duration_ms is a positive number in JSON logs."""
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.opt.return_value.log = capture_log

            json_server._structured_log("info", "Tool executed", duration_ms=123.456)

            assert len(log_calls) == 1
            log_entry = json.loads(log_calls[0][1].strip())
            assert "duration_ms" in log_entry
            assert isinstance(log_entry["duration_ms"], (int, float))
            assert log_entry["duration_ms"] > 0

    def test_error_log_status(self, json_server):
        """Verify failed tool logs status='error' in JSON format."""
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.opt.return_value.log = capture_log

            json_server._structured_log(
                "error",
                "Tool failed",
                tool_name="dna_mutagenesis",
                duration_ms=500.0,
                status="error",
            )

            assert len(log_calls) == 1
            log_entry = json.loads(log_calls[0][1].strip())
            assert log_entry["level"] == "ERROR"
            assert log_entry["status"] == "error"
            assert log_entry["tool_name"] == "dna_mutagenesis"


class TestLoggingIntegration:
    """Test logging integration with tool wrappers."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server for integration tests."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            with patch("dnallm.mcp.server.ModelManager"):
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
                mock_config.logging.log_format = "json"

                mock_cm_instance = MagicMock()
                mock_cm_instance.get_server_config.return_value = mock_config
                mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 30}
                mock_cm_instance.get_logging_config.return_value = {"log_format": "json"}
                mock_cm.return_value = mock_cm_instance

                server = DNALLMMCPServer("dummy_config.yaml")
                server._tool_timeout_seconds = 30
                server._log_format = "json"
                return server

    @pytest.mark.asyncio
    async def test_successful_tool_logs_duration(self, mock_server):
        """Verify successful tool execution logs positive duration."""

        async def fast_tool():
            await asyncio.sleep(0.01)
            return {"result": "ok"}

        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_opt(*args, **kwargs):
                class OptLogger:
                    def log(self, level, msg):
                        log_calls.append((level, msg))

                return OptLogger()

            mock_logger.opt = capture_opt

            wrapper = mock_server._with_timeout_wrapper(fast_tool, "fast_tool")
            result = await wrapper()

            assert result["result"] == "ok"

            # Find the success log entry (loguru uses uppercase)
            success_logs = [c for c in log_calls if c[0] == "INFO"]
            assert len(success_logs) >= 1

            log_entry = json.loads(success_logs[0][1].strip())
            assert log_entry["tool_name"] == "fast_tool"
            assert log_entry["status"] == "success"
            assert "duration_ms" in log_entry
            assert log_entry["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_timeout_tool_logs_error(self, mock_server):
        """Verify timed-out tool logs status='error'."""
        mock_server._tool_timeout_seconds = 0.05

        async def slow_tool():
            await asyncio.sleep(10)
            return {"result": "ok"}

        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_opt(*args, **kwargs):
                class OptLogger:
                    def log(self, level, msg):
                        log_calls.append((level, msg))

                return OptLogger()

            mock_logger.opt = capture_opt

            wrapper = mock_server._with_timeout_wrapper(slow_tool, "slow_tool")
            result = await wrapper()

            assert result["isError"] is True

            # Find the error log entry (loguru uses uppercase)
            error_logs = [c for c in log_calls if c[0] == "ERROR"]
            assert len(error_logs) >= 1

            log_entry = json.loads(error_logs[0][1].strip())
            assert log_entry["tool_name"] == "slow_tool"
            assert log_entry["status"] == "error"
            assert "duration_ms" in log_entry

    def test_text_format_no_json_fields(self, mock_server):
        """Verify text format does not produce JSON."""
        mock_server._log_format = "text"
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_log(level, msg):
                log_calls.append((level, msg))

            mock_logger.log = capture_log

            mock_server._structured_log("info", "Simple message", tool_name="test_tool")

            assert len(log_calls) == 1
            msg = log_calls[0][1]

            # Should not be valid JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(msg)

            # Should contain human-readable markers
            assert "Simple message" in msg
            assert "[tool=test_tool]" in msg

    def test_extra_fields_in_json(self, mock_server):
        """Verify extra kwargs are included in JSON logs."""
        mock_server._log_format = "json"
        with patch("dnallm.mcp.server.logger") as mock_logger:
            log_calls = []

            def capture_opt(*args, **kwargs):
                class OptLogger:
                    def log(self, level, msg):
                        log_calls.append((level, msg))

                return OptLogger()

            mock_logger.opt = capture_opt

            mock_server._structured_log(
                "info",
                "Extra test",
                custom_field="custom_value",
                sequence_length=100,
            )

            assert len(log_calls) == 1
            log_entry = json.loads(log_calls[0][1].strip())
            assert log_entry["custom_field"] == "custom_value"
            assert log_entry["sequence_length"] == 100
