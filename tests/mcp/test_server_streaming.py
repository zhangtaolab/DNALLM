"""Streaming-tool progress-contract tests for the DNALLM MCP server.

The three streaming tools are progress-reporting coroutines (research
Pattern 2), not generators: they call ``await context.report_progress(
progress, total, message)`` at defined stages and return a final dict.
These tests pin the ORDERED progress sequence for each tool, the final
result assembly, the fault dicts (None-return mid-batch; Nth-call raise),
the timeout-wrapper propagation contract (only ``asyncio.TimeoutError``
is converted to an error dict; generic exceptions propagate), and the
JSON structured-log branch.

The non-streaming tool bodies share the same mock-server seam and are
covered here as wave-gate objective work (see 03-03-SUMMARY): the eight
timeout-wrapped tool bodies were unexecuted by the suite before wave 3.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from dnallm.mcp.server import DNALLMMCPServer


@pytest.fixture
def mock_server():
    """Create a mock server with minimal setup (test_timeout.py strategy)."""
    with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
        with patch("dnallm.mcp.server.ModelManager"):
            mock_config = MagicMock()
            mock_config.mcp.name = "Test Server"
            mock_config.mcp.description = "Test"
            mock_config.mcp.version = "0.1.0"
            mock_config.server.host = "127.0.0.1"
            mock_config.server.port = 8000
            mock_config.tool_timeout_seconds = 30
            mock_config.logging.log_format = "text"

            mock_cm_instance = MagicMock()
            mock_cm_instance.get_server_config.return_value = mock_config
            mock_cm_instance.get_timeout_config.return_value = {"tool_timeout_seconds": 30}
            mock_cm_instance.get_logging_config.return_value = {"log_format": "text"}
            mock_cm_instance.get_enabled_models.return_value = ["test_model"]
            mock_cm.return_value = mock_cm_instance

            server = DNALLMMCPServer("dummy_config.yaml")
            server._tool_timeout_seconds = 30
            server._log_format = "text"
            return server


def _set_predict(server, **asyncmock_kwargs):
    """Install an AsyncMock predict_sequence on the server's model manager."""
    server.model_manager = MagicMock()
    server.model_manager.predict_sequence = AsyncMock(**asyncmock_kwargs)
    return server.model_manager


# ---------------------------------------------------------------------------
# Single-model streaming tool
# ---------------------------------------------------------------------------


class TestStreamPredictProgress:
    """Test _dna_stream_predict progress ordering and fault paths."""

    @pytest.mark.asyncio
    async def test_reports_ordered_progress_stages(self, mock_server):
        """A successful prediction reports 0, 25, 75, 100 in order."""
        _set_predict(mock_server, return_value={"probabilities": [0.4, 0.6]})
        context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert context.report_progress.call_args_list == [
            call(0, 100, "Starting prediction with model test-model"),
            call(25, 100, "Loading model and tokenizer..."),
            call(75, 100, "Processing prediction results..."),
            call(100, 100, "Prediction completed successfully"),
        ]
        assert result.get("isError") is not True
        assert result["model_name"] == "test-model"
        assert result["sequence"] == "ATCG"
        assert result["streamed"] is True
        assert result["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_none_result_reports_error_and_returns_error_dict(self, mock_server):
        """A None prediction result fails fast with the verbatim message."""
        _set_predict(mock_server, return_value=None)
        context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert result == {
            "error": "Model test-model not available or prediction failed",
            "isError": True,
        }
        # Progress stops after the loading stage, then reports the failure.
        assert context.report_progress.call_args_list == [
            call(0, 100, "Starting prediction with model test-model"),
            call(25, 100, "Loading model and tokenizer..."),
            call(100, 100, "Error: Model test-model not available or prediction failed"),
        ]

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_dict(self, mock_server):
        """A raised prediction error hits the streaming except arm."""
        _set_predict(mock_server, side_effect=RuntimeError("engine exploded"))
        context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "Streaming prediction failed. See server logs for details."
        )
        assert context.report_progress.call_args_list[-1] == call(
            100, 100, "Error: Streaming prediction failed"
        )


# ---------------------------------------------------------------------------
# Batch streaming tool
# ---------------------------------------------------------------------------


class TestStreamBatchPredict:
    """Test _dna_stream_batch_predict progress ordering and fault paths."""

    @pytest.mark.asyncio
    async def test_reports_per_item_progress_in_order(self, mock_server):
        """Batch progress walks 0 -> i/total percent -> 100 completion."""
        _set_predict(mock_server, return_value={"probabilities": [0.5, 0.5]}, side_effect=None)
        context = AsyncMock()

        result = await mock_server._dna_stream_batch_predict(
            sequences=["AAAA", "CCCC", "GGGG"],
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert context.report_progress.call_args_list == [
            call(0, 100, "Starting batch prediction with 3 sequences using model test-model"),
            call(0, 100, "Processing sequence 1/3"),
            call(33, 100, "Processing sequence 2/3"),
            call(66, 100, "Processing sequence 3/3"),
            call(100, 100, "Batch prediction completed: 3 successful, 0 failed"),
        ]
        assert result["sequence_count"] == 3
        assert result["successful_predictions"] == 3
        assert result["failed_predictions"] == 0
        assert [entry["index"] for entry in result["results"]] == [0, 1, 2]
        assert all(entry["result"] == {"probabilities": [0.5, 0.5]} for entry in result["results"])

    @pytest.mark.asyncio
    async def test_none_mid_batch_builds_verbatim_error_entry(self, mock_server):
        """A None result mid-list produces the exact per-sequence error entry."""
        _set_predict(
            mock_server,
            side_effect=[
                {"probabilities": [0.5, 0.5]},
                None,
                {"probabilities": [0.1, 0.9]},
            ],
        )
        context = AsyncMock()

        result = await mock_server._dna_stream_batch_predict(
            sequences=["AAAA", "CCCC", "GGGG"],
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert result["results"][1] == {
            "sequence": "CCCC",
            "result": None,
            "error": "Prediction failed for sequence 2",
            "index": 1,
        }
        assert result["successful_predictions"] == 2
        assert result["failed_predictions"] == 1
        assert context.report_progress.call_args_list[-1] == call(
            100, 100, "Batch prediction completed: 2 successful, 1 failed"
        )

    @pytest.mark.asyncio
    async def test_nth_call_raise_returns_error_dict(self, mock_server):
        """A raise on the second item abandons the batch with an error dict."""
        _set_predict(
            mock_server,
            side_effect=[{"probabilities": [0.5, 0.5]}, RuntimeError("mid-stream fault")],
        )
        context = AsyncMock()

        result = await mock_server._dna_stream_batch_predict(
            sequences=["AAAA", "CCCC"],
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "Streaming batch prediction failed. See server logs for details."
        )
        assert context.report_progress.call_args_list[-1] == call(
            100, 100, "Error: Streaming batch prediction failed"
        )
        mm = mock_server.model_manager
        assert mm.predict_sequence.await_count == 2
        assert mm.predict_sequence.await_args_list[1] == call("test-model", "CCCC")


# ---------------------------------------------------------------------------
# Multi-model streaming tool
# ---------------------------------------------------------------------------


class TestStreamMultiModelPredict:
    """Test _dna_stream_multi_model_predict routing and aggregation."""

    @pytest.mark.asyncio
    async def test_routes_each_model_with_ordered_progress(self, mock_server):
        """Each model is visited once with i/total progress, then 100."""
        mm = _set_predict(
            mock_server,
            side_effect=[
                {"probabilities": [0.9, 0.1]},
                {"probabilities": [0.2, 0.8]},
            ],
        )
        context = AsyncMock()

        result = await mock_server._dna_stream_multi_model_predict(
            sequence="ATCG",
            model_names=["model-a", "model-b"],
            stream_progress=True,
            context=context,
        )

        assert mm.predict_sequence.await_args_list == [
            call("model-a", "ATCG"),
            call("model-b", "ATCG"),
        ]
        assert context.report_progress.call_args_list == [
            call(0, 100, "Starting multi-model prediction with 2 models"),
            call(0, 100, "Processing with model 1/2: model-a"),
            call(50, 100, "Processing with model 2/2: model-b"),
            call(100, 100, "Multi-model prediction completed: 2 successful, 0 failed"),
        ]
        assert result["model_count"] == 2
        assert result["results"]["model-a"] == {"probabilities": [0.9, 0.1]}
        assert result["results"]["model-b"] == {"probabilities": [0.2, 0.8]}
        assert result["successful_predictions"] == 2
        assert result["failed_predictions"] == 0

    @pytest.mark.asyncio
    async def test_none_result_marks_model_failed(self, mock_server):
        """A None result for one model is aggregated as a failure entry."""
        _set_predict(
            mock_server,
            side_effect=[{"probabilities": [0.9, 0.1]}, None],
        )
        context = AsyncMock()

        result = await mock_server._dna_stream_multi_model_predict(
            sequence="ATCG",
            model_names=["model-a", "model-b"],
            stream_progress=True,
            context=context,
        )

        assert result["results"]["model-b"] == {
            "error": "Prediction failed with model model-b",
            "result": None,
        }
        assert result["successful_predictions"] == 1
        assert result["failed_predictions"] == 1
        assert context.report_progress.call_args_list[-1] == call(
            100, 100, "Multi-model prediction completed: 1 successful, 1 failed"
        )

    @pytest.mark.asyncio
    async def test_defaults_to_loaded_models_then_errors_when_none(self, mock_server):
        """With no model names the loaded list is used; empty list errors."""
        mm = MagicMock()
        mm.get_loaded_models.return_value = []
        mock_server.model_manager = mm
        context = AsyncMock()

        result = await mock_server._dna_stream_multi_model_predict(
            sequence="ATCG",
            model_names=None,
            stream_progress=True,
            context=context,
        )

        mm.get_loaded_models.assert_called_once_with()
        assert result == {"error": "No models available for prediction", "isError": True}
        context.report_progress.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_dict(self, mock_server):
        """A raised prediction error hits the multi-model except arm."""
        _set_predict(mock_server, side_effect=RuntimeError("group fault"))
        context = AsyncMock()

        result = await mock_server._dna_stream_multi_model_predict(
            sequence="ATCG",
            model_names=["model-a"],
            stream_progress=True,
            context=context,
        )

        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "Streaming multi-model prediction failed. See server logs for details."
        )
        assert context.report_progress.call_args_list[-1] == call(
            100, 100, "Error: Streaming multi-model prediction failed"
        )


# ---------------------------------------------------------------------------
# Timeout-wrapper contract completion
# ---------------------------------------------------------------------------


class TestTimeoutWrapperPropagation:
    """Complete the timeout-wrapper contract not covered by test_timeout.py."""

    @pytest.mark.asyncio
    async def test_generic_exception_propagates_out_of_wrapper(self, mock_server):
        """Only asyncio.TimeoutError is caught; a ValueError must raise."""

        async def failing_tool():
            await asyncio.sleep(0)
            raise ValueError("boom")

        wrapper = mock_server._with_timeout_wrapper(failing_tool, "failing_tool")
        with pytest.raises(ValueError, match="boom"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_timeout_error_is_not_swallowed_by_tool_except_arms(self, mock_server):
        """A timeout inside a streaming tool returns the timeout dict, not a
        generic streaming-failure dict (chunk-based timeout contract)."""
        mock_server._tool_timeout_seconds = 0.05

        async def slow_predict(*args, **kwargs):
            await asyncio.sleep(1)

        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_sequence = slow_predict
        context = AsyncMock()

        result = await mock_server._dna_stream_predict(
            sequence="ATCG",
            model_name="test-model",
            stream_progress=True,
            context=context,
        )

        assert result["error_type"] == "timeout"
        assert result["tool_name"] == "dna_stream_predict"


# ---------------------------------------------------------------------------
# Structured logging (JSON branch)
# ---------------------------------------------------------------------------


class TestStructuredLogJsonBranch:
    """Test _structured_log JSON serialization with a patched sink."""

    def test_json_branch_serializes_asserted_fields(self, mock_server):
        """The JSON branch emits level, message, and rounded duration."""
        mock_server._log_format = "json"
        with patch("dnallm.mcp.server.logger") as mock_logger:
            mock_server._structured_log(
                "info",
                "Tool t completed",
                tool_name="t",
                duration_ms=123.4567,
                status="success",
            )

        mock_logger.opt.assert_called_once_with(raw=True)
        log_call = mock_logger.opt.return_value.log.call_args
        assert log_call.args[0] == "INFO"
        entry = json.loads(log_call.args[1])
        assert entry["level"] == "INFO"
        assert entry["message"] == "Tool t completed"
        assert entry["tool_name"] == "t"
        assert entry["duration_ms"] == 123.46  # rounded to 2 decimals
        assert entry["status"] == "success"
        assert entry["timestamp"].endswith("Z")

    def test_json_branch_merges_extra_fields(self, mock_server):
        """Keyword extras and request_id land in the serialized entry."""
        mock_server._log_format = "json"
        with patch("dnallm.mcp.server.logger") as mock_logger:
            mock_server._structured_log(
                "error",
                "Tool t failed",
                request_id="req-42",
                custom_field="value",
            )

        entry = json.loads(mock_logger.opt.return_value.log.call_args.args[1])
        assert entry["request_id"] == "req-42"
        assert entry["custom_field"] == "value"


# ---------------------------------------------------------------------------
# Non-streaming tool bodies (wave-gate objective work; see module docstring)
# ---------------------------------------------------------------------------


class TestBasicToolBodies:
    """Cover the eight timeout-wrapped tool bodies on the shared seam."""

    @pytest.mark.asyncio
    async def test_sequence_predict_success(self, mock_server):
        """dna_sequence_predict returns the MCP content envelope."""
        _set_predict(mock_server, return_value={"probabilities": [0.5, 0.5]})
        result = await mock_server._dna_sequence_predict("ATCG", "test-model")
        assert result["model_name"] == "test-model"
        assert result["sequence"] == "ATCG"
        assert result["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_sequence_predict_none_returns_error(self, mock_server):
        """A None prediction result yields the not-available error."""
        _set_predict(mock_server, return_value=None)
        result = await mock_server._dna_sequence_predict("ATCG", "test-model")
        assert result == {
            "error": "Model test-model not available or prediction failed",
            "isError": True,
        }

    @pytest.mark.asyncio
    async def test_sequence_predict_exception_returns_error_dict(self, mock_server):
        """A raised error is caught and reported as a failed prediction."""
        _set_predict(mock_server, side_effect=RuntimeError("nope"))
        result = await mock_server._dna_sequence_predict("ATCG", "test-model")
        assert result["isError"] is True
        assert result["content"][0]["text"] == ("Prediction failed. See server logs for details.")

    @pytest.mark.asyncio
    async def test_batch_predict_success(self, mock_server):
        """dna_batch_predict reports the sequence count."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_batch = AsyncMock(return_value={"ok": True})
        result = await mock_server._dna_batch_predict(["ATCG", "GGCC"], "test-model")
        mock_server.model_manager.predict_batch.assert_awaited_once_with(
            "test-model", ["ATCG", "GGCC"]
        )
        assert result["sequence_count"] == 2
        assert result["model_name"] == "test-model"

    @pytest.mark.asyncio
    async def test_batch_predict_none_returns_error(self, mock_server):
        """A None batch result yields the not-available error."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_batch = AsyncMock(return_value=None)
        result = await mock_server._dna_batch_predict(["ATCG"], "test-model")
        assert result["isError"] is True
        assert result["error"] == "Model test-model not available or prediction failed"

    @pytest.mark.asyncio
    async def test_batch_predict_exception_returns_error_dict(self, mock_server):
        """A raised batch error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_batch = AsyncMock(side_effect=RuntimeError("x"))
        result = await mock_server._dna_batch_predict(["ATCG"], "test-model")
        assert result["content"][0]["text"] == (
            "Batch prediction failed. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_multi_model_predict_defaults_to_loaded_models(self, mock_server):
        """With model_names=None the loaded model list is used."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.return_value = ["m1", "m2"]
        mock_server.model_manager.predict_multi_model = AsyncMock(return_value={"m1": 1})
        result = await mock_server._dna_multi_model_predict("ATCG")
        mock_server.model_manager.predict_multi_model.assert_awaited_once_with(["m1", "m2"], "ATCG")
        assert result["model_count"] == 2

    @pytest.mark.asyncio
    async def test_multi_model_predict_no_models_returns_error(self, mock_server):
        """An empty loaded-model list fails before calling predictions."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.return_value = []
        result = await mock_server._dna_multi_model_predict("ATCG")
        assert result == {"error": "No models available for prediction", "isError": True}

    @pytest.mark.asyncio
    async def test_multi_model_predict_exception_returns_error_dict(self, mock_server):
        """A raised multi-model error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.predict_multi_model = AsyncMock(side_effect=RuntimeError("x"))
        result = await mock_server._dna_multi_model_predict("ATCG", model_names=["m1"])
        assert result["content"][0]["text"] == (
            "Multi-model prediction failed. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_list_loaded_models_assembles_info(self, mock_server):
        """list_loaded_models joins the loaded list with per-model info."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.return_value = ["m1", "m2"]
        mock_server.model_manager.get_model_info.side_effect = [
            {"name": "m1"},
            {"name": "m2"},
        ]
        result = await mock_server._list_loaded_models()
        assert result["loaded_count"] == 2
        assert result["models"] == {"m1": {"name": "m1"}, "m2": {"name": "m2"}}

    @pytest.mark.asyncio
    async def test_list_loaded_models_exception_returns_error_dict(self, mock_server):
        """A raised listing error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.side_effect = RuntimeError("x")
        result = await mock_server._list_loaded_models()
        assert result["content"][0]["text"] == (
            "Failed to list models. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_get_model_info_found(self, mock_server):
        """get_model_info returns the info envelope for a known model."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_model_info.return_value = {"name": "m1"}
        result = await mock_server._get_model_info("m1")
        assert result["model_name"] == "m1"
        assert result["info"] == {"name": "m1"}

    @pytest.mark.asyncio
    async def test_get_model_info_not_found(self, mock_server):
        """An unknown model yields the not-found error."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_model_info.return_value = None
        result = await mock_server._get_model_info("ghost")
        assert result == {"error": "Model ghost not found", "isError": True}

    @pytest.mark.asyncio
    async def test_get_model_info_exception_returns_error_dict(self, mock_server):
        """A raised info error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_model_info.side_effect = RuntimeError("x")
        result = await mock_server._get_model_info("m1")
        assert result["content"][0]["text"] == (
            "Failed to get model info. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_list_models_by_task_type_filters(self, mock_server):
        """Only models whose task_type matches are returned."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_all_models_info.return_value = {
            "m1": {"task_type": "binary"},
            "m2": {"task_type": "multiclass"},
        }
        result = await mock_server._list_models_by_task_type("binary")
        assert result["models"] == {"m1": {"task_type": "binary"}}
        assert result["model_count"] == 1
        assert result["task_type"] == "binary"

    @pytest.mark.asyncio
    async def test_list_models_by_task_type_exception_returns_error_dict(self, mock_server):
        """A raised filtering error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_all_models_info.side_effect = RuntimeError("x")
        result = await mock_server._list_models_by_task_type("binary")
        assert result["content"][0]["text"] == (
            "Failed to filter models. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_get_all_available_models(self, mock_server):
        """get_all_available_models reports every configured model."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_all_models_info.return_value = {"m1": {"task_type": "binary"}}
        result = await mock_server._get_all_available_models()
        assert result["total_models"] == 1
        assert result["models"] == {"m1": {"task_type": "binary"}}

    @pytest.mark.asyncio
    async def test_get_all_available_models_exception_returns_error_dict(self, mock_server):
        """A raised inventory error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_all_models_info.side_effect = RuntimeError("x")
        result = await mock_server._get_all_available_models()
        assert result["content"][0]["text"] == (
            "Failed to get available models. See server logs for details."
        )

    @pytest.mark.asyncio
    async def test_health_check_reports_counts(self, mock_server):
        """health_check reports loaded/configured counts and server identity."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.return_value = ["m1"]
        result = await mock_server._health_check()
        assert result["health"]["status"] == "healthy"
        assert result["health"]["loaded_models"] == 1
        assert result["health"]["total_configured_models"] == 1
        assert result["health"]["server_name"] == "Test Server"
        assert result["health"]["server_version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_check_exception_returns_error_dict(self, mock_server):
        """A raised health error is caught with the verbatim message."""
        mock_server.model_manager = MagicMock()
        mock_server.model_manager.get_loaded_models.side_effect = RuntimeError("x")
        result = await mock_server._health_check()
        assert result["content"][0]["text"] == ("Health check failed. See server logs for details.")
