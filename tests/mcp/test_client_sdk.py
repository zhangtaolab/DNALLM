"""Tests for the DNALLM MCP Client SDK.

This module provides comprehensive unit tests for the DNALLMMCPClient class,
covering initialization, transport configuration, typed methods, generic
fallback, and sync/async behavior.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dnallm.mcp.client import DNALLMMCPClient

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session():
    """Return a mock MCP ClientSession."""
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=MagicMock(
            isError=False,
            content=[MagicMock(text='{"result": "ok"}')],
        )
    )
    session.initialize = AsyncMock(return_value=None)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def mock_connect(mock_session):
    """Return a mock async context manager that yields mock_session."""

    class MockAsyncContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    return MockAsyncContextManager()


@pytest.fixture
def streamable_http_client():
    """Return a DNALLMMCPClient configured for Streamable HTTP transport."""
    return DNALLMMCPClient(
        transport="streamable-http", url="http://localhost:8000"
    )


@pytest.fixture
def sse_client():
    """Return a DNALLMMCPClient configured for SSE transport."""
    return DNALLMMCPClient(transport="sse", url="http://localhost:8000")


@pytest.fixture
def stdio_client():
    """Return a DNALLMMCPClient configured for stdio transport."""
    return DNALLMMCPClient(
        transport="stdio", command="dnallm-mcp-server"
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_client_init_streamable_http(streamable_http_client):
    """Initialize with transport='streamable-http', verify url stored."""
    assert streamable_http_client.transport == "streamable-http"
    assert streamable_http_client.url == "http://localhost:8000"
    assert streamable_http_client.command is None


def test_client_init_streamable_http_custom_url():
    """Initialize with custom URL for streamable-http transport."""
    client = DNALLMMCPClient(
        transport="streamable-http", url="http://example.com:9000"
    )
    assert client.transport == "streamable-http"
    assert client.url == "http://example.com:9000"


def test_client_init_sse(sse_client):
    """Initialize with transport='sse', verify url stored."""
    assert sse_client.transport == "sse"
    assert sse_client.url == "http://localhost:8000"
    assert sse_client.command is None


def test_client_init_stdio(stdio_client):
    """Initialize with transport='stdio', verify command stored."""
    assert stdio_client.transport == "stdio"
    assert stdio_client.command == "dnallm-mcp-server"
    assert stdio_client.url is None


def test_client_init_invalid_transport():
    """Raise ValueError for invalid transport with all options listed."""
    with pytest.raises(
        ValueError, match='"streamable-http", "sse", or "stdio"'
    ):
        DNALLMMCPClient(transport="http")


# ---------------------------------------------------------------------------
# Generic call tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_acall_generic(sse_client, mock_session, mock_connect):
    """Mock session, verify async generic acall() works."""
    sse_client._connect = lambda: mock_connect

    result = await sse_client.acall("test_tool", {"arg": 1})

    mock_session.call_tool.assert_awaited_once_with(
        "test_tool", {"arg": 1}
    )
    assert result == {"result": "ok"}


def test_client_call_generic(sse_client, mock_session, mock_connect):
    """Mock session.call_tool, verify generic call() works."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.call("test_tool", {"arg": 1})

    mock_session.call_tool.assert_awaited_once_with(
        "test_tool", {"arg": 1}
    )
    assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# Streamable HTTP connection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_streamable_http_connection(
    streamable_http_client, mock_session, mock_connect
):
    """Mock session, verify streamable-http acall() works."""
    streamable_http_client._connect = lambda: mock_connect

    result = await streamable_http_client.acall("test_tool", {"arg": 1})

    mock_session.call_tool.assert_awaited_once_with(
        "test_tool", {"arg": 1}
    )
    assert result == {"result": "ok"}


def test_client_streamable_http_dna_sequence_predict_mocked(
    streamable_http_client, mock_session, mock_connect
):
    """Mock session, verify dna_sequence_predict with streamable-http."""
    streamable_http_client._connect = lambda: mock_connect

    result = streamable_http_client.dna_sequence_predict("ATCG", "dnabert-2")

    mock_session.call_tool.assert_awaited_once_with(
        "dna_sequence_predict",
        {"sequence": "ATCG", "model_name": "dnabert-2"},
    )
    assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# Typed method signature tests
# ---------------------------------------------------------------------------


TYPED_METHODS = [
    ("dna_sequence_predict", ["sequence", "model_name"]),
    ("adna_sequence_predict", ["sequence", "model_name"]),
    ("dna_batch_predict", ["sequences", "model_name"]),
    ("adna_batch_predict", ["sequences", "model_name"]),
    ("dna_multi_model_predict", ["sequence", "model_names"]),
    ("adna_multi_model_predict", ["sequence", "model_names"]),
    ("dna_stream_predict", ["sequence", "model_name", "stream_progress"]),
    ("adna_stream_predict", ["sequence", "model_name", "stream_progress"]),
    (
        "dna_stream_batch_predict",
        ["sequences", "model_name", "stream_progress"],
    ),
    (
        "adna_stream_batch_predict",
        ["sequences", "model_name", "stream_progress"],
    ),
    (
        "dna_stream_multi_model_predict",
        ["sequence", "model_names", "stream_progress"],
    ),
    (
        "adna_stream_multi_model_predict",
        ["sequence", "model_names", "stream_progress"],
    ),
    (
        "dna_mutagenesis",
        [
            "sequence",
            "sequences",
            "mutation_type",
            "positions",
            "model_name",
        ],
    ),
    (
        "adna_mutagenesis",
        [
            "sequence",
            "sequences",
            "mutation_type",
            "positions",
            "model_name",
        ],
    ),
    (
        "dna_interpret",
        ["sequence", "model_name", "method", "target_class", "max_length"],
    ),
    (
        "adna_interpret",
        ["sequence", "model_name", "method", "target_class", "max_length"],
    ),
    ("list_loaded_models", []),
    ("alist_loaded_models", []),
    ("get_model_info", ["model_name"]),
    ("aget_model_info", ["model_name"]),
    ("list_models_by_task_type", ["task_type"]),
    ("alist_models_by_task_type", ["task_type"]),
    ("get_all_available_models", []),
    ("aget_all_available_models", []),
    ("health_check", []),
    ("ahealth_check", []),
]


@pytest.mark.parametrize(("method_name", "expected_params"), TYPED_METHODS)
def test_client_typed_method_signature(method_name, expected_params):
    """Verify each typed method exists and has correct signature."""
    client = DNALLMMCPClient(transport="sse")
    assert hasattr(client, method_name), f"Missing method: {method_name}"

    method = getattr(client, method_name)
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    # Remove 'self' from params list
    if params and params[0] == "self":
        params = params[1:]

    assert params == expected_params, (
        f"Method {method_name} params {params} != expected {expected_params}"
    )


# ---------------------------------------------------------------------------
# Sync wraps async tests
# ---------------------------------------------------------------------------


def test_client_sync_wraps_async(sse_client):
    """Verify sync method calls async internally."""
    with patch.object(
        sse_client,
        "adna_sequence_predict",
        new_callable=AsyncMock,
        return_value={"predictions": [0.1]},
    ) as mock_async:
        result = sse_client.dna_sequence_predict("ATCG", "dnabert-2")

    mock_async.assert_awaited_once_with("ATCG", "dnabert-2")
    assert result == {"predictions": [0.1]}


# ---------------------------------------------------------------------------
# Specific tool tests
# ---------------------------------------------------------------------------


def test_client_dna_sequence_predict_mocked(
    sse_client, mock_session, mock_connect
):
    """Mock session, verify dna_sequence_predict calls correct tool."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.dna_sequence_predict("ATCG", "dnabert-2")

    mock_session.call_tool.assert_awaited_once_with(
        "dna_sequence_predict",
        {"sequence": "ATCG", "model_name": "dnabert-2"},
    )
    assert result == {"result": "ok"}


def test_client_dna_mutagenesis_mocked(
    sse_client, mock_session, mock_connect
):
    """Mock session, verify dna_mutagenesis calls correct tool."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.dna_mutagenesis(
        sequence="ATCG",
        positions=[1, 2],
        model_name="dnabert-2",
    )

    mock_session.call_tool.assert_awaited_once_with(
        "dna_mutagenesis",
        {
            "sequence": "ATCG",
            "sequences": None,
            "mutation_type": "single_base_substitution",
            "positions": [1, 2],
            "model_name": "dnabert-2",
        },
    )
    assert result == {"result": "ok"}


def test_client_dna_interpret_mocked(
    sse_client, mock_session, mock_connect
):
    """Mock session, verify dna_interpret calls correct tool."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.dna_interpret(
        sequence="ATCG",
        model_name="dnabert-2",
        method="deeplift",
        target_class=1,
    )

    mock_session.call_tool.assert_awaited_once_with(
        "dna_interpret",
        {
            "sequence": "ATCG",
            "model_name": "dnabert-2",
            "method": "deeplift",
            "target_class": 1,
            "max_length": None,
        },
    )
    assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_client_parse_result_error():
    """Verify error responses are parsed correctly."""
    error_result = MagicMock()
    error_result.isError = True
    error_result.content = [MagicMock(text='{"error": "failed"}')]

    parsed = DNALLMMCPClient._parse_result(error_result)
    assert parsed == {"error": "failed"}


def test_client_parse_result_plain_text():
    """Verify plain text responses are wrapped correctly."""
    text_result = MagicMock()
    text_result.isError = False
    text_result.content = [MagicMock(text="plain text")]

    parsed = DNALLMMCPClient._parse_result(text_result)
    assert parsed == {"text": "plain text"}


def test_client_parse_result_empty_content():
    """Verify empty content returns empty dict."""
    empty_result = MagicMock()
    empty_result.isError = False
    empty_result.content = []

    parsed = DNALLMMCPClient._parse_result(empty_result)
    assert parsed == {}


# ---------------------------------------------------------------------------
# Async context safety test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_sync_from_async_raises(sse_client):  # noqa: RUF029
    """Verify sync method raises RuntimeError when called from async."""
    with pytest.raises(RuntimeError, match="async context"):
        sse_client.dna_sequence_predict("ATCG", "dnabert-2")
