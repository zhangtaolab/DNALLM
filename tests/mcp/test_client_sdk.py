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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    return DNALLMMCPClient(transport="streamable-http", url="http://localhost:8000")


@pytest.fixture
def sse_client():
    """Return a DNALLMMCPClient configured for SSE transport."""
    return DNALLMMCPClient(transport="sse", url="http://localhost:8000")


@pytest.fixture
def stdio_client():
    """Return a DNALLMMCPClient configured for stdio transport."""
    return DNALLMMCPClient(transport="stdio", command="dnallm-mcp-server")


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
    client = DNALLMMCPClient(transport="streamable-http", url="http://example.com:9000")
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
    with pytest.raises(ValueError, match='"streamable-http", "sse", or "stdio"'):
        DNALLMMCPClient(transport="http")


def test_client_init_streamable_http_custom_url_with_path():
    """Initialize with explicit /mcp path — verify stored as-is (no double-append)."""
    client = DNALLMMCPClient(transport="streamable-http", url="http://localhost:8000/mcp")
    assert client.url == "http://localhost:8000/mcp"


def test_client_streamable_http_lifecycle(streamable_http_client):
    """Verify client has async context manager and close methods."""
    assert hasattr(streamable_http_client, "close")
    assert hasattr(streamable_http_client, "__aenter__")
    assert hasattr(streamable_http_client, "__aexit__")


# ---------------------------------------------------------------------------
# Generic call tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_acall_generic(sse_client, mock_session, mock_connect):
    """Mock session, verify async generic acall() works."""
    sse_client._connect = lambda: mock_connect

    result = await sse_client.acall("test_tool", {"arg": 1})

    mock_session.call_tool.assert_awaited_once_with("test_tool", {"arg": 1})
    assert result == {"result": "ok"}


def test_client_call_generic(sse_client, mock_session, mock_connect):
    """Mock session.call_tool, verify generic call() works."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.call("test_tool", {"arg": 1})

    mock_session.call_tool.assert_awaited_once_with("test_tool", {"arg": 1})
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

    mock_session.call_tool.assert_awaited_once_with("test_tool", {"arg": 1})
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


def test_client_dna_sequence_predict_mocked(sse_client, mock_session, mock_connect):
    """Mock session, verify dna_sequence_predict calls correct tool."""
    sse_client._connect = lambda: mock_connect

    result = sse_client.dna_sequence_predict("ATCG", "dnabert-2")

    mock_session.call_tool.assert_awaited_once_with(
        "dna_sequence_predict",
        {"sequence": "ATCG", "model_name": "dnabert-2"},
    )
    assert result == {"result": "ok"}


def test_client_dna_mutagenesis_mocked(sse_client, mock_session, mock_connect):
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


def test_client_dna_interpret_mocked(sse_client, mock_session, mock_connect):
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
async def test_client_sync_from_async_raises(sse_client):  # ruff: ignore[unused-async]
    """Verify sync method raises RuntimeError when called from async."""
    with pytest.raises(RuntimeError, match="async context"):
        sse_client.dna_sequence_predict("ATCG", "dnabert-2")


# ---------------------------------------------------------------------------
# Default URL derivation
# ---------------------------------------------------------------------------


def test_client_default_urls_per_transport():
    """Each transport derives its documented default endpoint URL."""
    http_client = DNALLMMCPClient(transport="streamable-http")
    assert http_client.url == "http://localhost:8000/mcp"

    sse = DNALLMMCPClient(transport="sse")
    assert sse.url == "http://localhost:8000/sse"


# ---------------------------------------------------------------------------
# Connection establishment (mocked SDK transport factories)
# ---------------------------------------------------------------------------


class _FakeSession:
    """Minimal ClientSession stand-in recording initialize calls."""

    initialized = False

    def __init__(self, read, write):
        self.read, self.write = read, write
        self.call_tool = AsyncMock(
            return_value=MagicMock(isError=False, content=[MagicMock(text='{"result": "ok"}')])
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def initialize(self):
        _FakeSession.initialized = True


def _fake_transport_cm(yields):
    """Build an async CM yielding the given tuple (read/write streams)."""

    class _CM:
        async def __aenter__(self):
            return yields

        async def __aexit__(self, *args):
            return None

    return _CM()


@pytest.mark.asyncio
async def test_connect_streamable_http_initializes_session():
    """The streamable-http _connect branch yields an initialized session."""
    client = DNALLMMCPClient(transport="streamable-http", url="http://localhost:8000/mcp")
    _FakeSession.initialized = False
    with (
        patch(
            "mcp.client.streamable_http.streamable_http_client",
            return_value=_fake_transport_cm((Mock(), Mock(), Mock())),
        ),
        patch("mcp.ClientSession", _FakeSession),
    ):
        async with client._connect() as session:
            assert isinstance(session, _FakeSession)
    assert _FakeSession.initialized is True


@pytest.mark.asyncio
async def test_connect_sse_initializes_session():
    """The sse _connect branch yields an initialized session."""
    client = DNALLMMCPClient(transport="sse", url="http://localhost:8000/sse")
    _FakeSession.initialized = False
    with (
        patch("mcp.client.sse.sse_client", return_value=_fake_transport_cm((Mock(), Mock()))),
        patch("mcp.ClientSession", _FakeSession),
    ):
        async with client._connect() as session:
            assert isinstance(session, _FakeSession)
    assert _FakeSession.initialized is True


@pytest.mark.asyncio
async def test_connect_stdio_builds_server_parameters():
    """The stdio branch spawns with the configured command/args/env."""
    client = DNALLMMCPClient(
        transport="stdio", command="dnallm-mcp-server", args=["--x"], env={"K": "V"}
    )
    captured = []
    _FakeSession.initialized = False

    def fake_stdio_client(server_params):
        captured.append(server_params)
        return _fake_transport_cm((Mock(), Mock()))

    with (
        patch("mcp.client.stdio.stdio_client", side_effect=fake_stdio_client),
        patch("mcp.ClientSession", _FakeSession),
    ):
        async with client._connect() as session:
            assert isinstance(session, _FakeSession)

    assert captured[0].command == "dnallm-mcp-server"
    assert captured[0].args == ["--x"]
    assert captured[0].env == {"K": "V"}
    assert _FakeSession.initialized is True


# ---------------------------------------------------------------------------
# Persistent session context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_async_context_manages_persistent_session(mock_session, mock_connect):
    """__aenter__ opens a persistent session; __aexit__ closes it."""
    client = DNALLMMCPClient(transport="sse")
    client._connect = lambda: mock_connect

    async with client as entered:
        assert entered is client  # __aenter__ yields the client, not the session
        assert client._session is mock_session
        # Persistent-session branch of _call_tool (no per-call connect)
        result = await client.acall("health_check", {})
        mock_session.call_tool.assert_awaited_with("health_check", {})
        assert result == {"result": "ok"}

    assert client._session is None
    assert client._connect_cm is None


@pytest.mark.asyncio
async def test_client_close_closes_open_session(mock_session, mock_connect):
    """close() tears down the persistent session."""
    client = DNALLMMCPClient(transport="sse")
    client._connect = lambda: mock_connect

    await client.__aenter__()
    assert client._session is mock_session
    await client.close()

    assert client._session is None
    assert client._connect_cm is None


# ---------------------------------------------------------------------------
# Parse-result error corner
# ---------------------------------------------------------------------------


def test_client_parse_result_error_non_json_text():
    """Non-JSON error text is wrapped into an error dict, not dropped."""
    error_result = MagicMock()
    error_result.isError = True
    error_result.content = [MagicMock(text="plain failure text")]

    parsed = DNALLMMCPClient._parse_result(error_result)
    assert parsed == {"error": "plain failure text", "isError": True}


# ---------------------------------------------------------------------------
# Connection failure semantics (research Pitfall 11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_failure_surfaces_through_exception_group():
    """Connection failures escape as ExceptionGroups wrapping httpx errors.

    The MCP transports fail through httpx inside anyio TaskGroups, so the
    catchable shape is a group whose leaves are httpx transport errors —
    never a bare httpx exception type.
    """
    import httpx

    from dnallm.mcp.tests._network_skip import _network_leaves

    client = DNALLMMCPClient(transport="streamable-http", url="http://localhost:8000/mcp")
    group = ExceptionGroup("connection attempt", [httpx.ConnectError("refused")])

    class _FailingCM:
        async def __aenter__(self):
            raise group

        async def __aexit__(self, *args):
            return False

    with patch("mcp.client.streamable_http.streamable_http_client", return_value=_FailingCM()):
        with pytest.raises(ExceptionGroup) as excinfo:
            await client.acall("health_check", {})

    leaves = _network_leaves(excinfo.value)
    assert leaves, "exception group must carry leaves"
    assert all(isinstance(leaf, httpx.TransportError) for leaf in leaves)


# ---------------------------------------------------------------------------
# Remaining typed-method execution coverage
# ---------------------------------------------------------------------------

ASYNC_TYPED_CALLS = [
    (
        "adna_batch_predict",
        (["ATCG", "GCTA"], "m"),
        "dna_batch_predict",
        {"sequences": ["ATCG", "GCTA"], "model_name": "m"},
    ),
    (
        "adna_multi_model_predict",
        ("ATCG", ["m1"]),
        "dna_multi_model_predict",
        {"sequence": "ATCG", "model_names": ["m1"]},
    ),
    (
        "adna_stream_predict",
        ("ATCG", "m", False),
        "dna_stream_predict",
        {"sequence": "ATCG", "model_name": "m", "stream_progress": False},
    ),
    (
        "adna_stream_batch_predict",
        ((["ATCG"], "m", False)),
        "dna_stream_batch_predict",
        {"sequences": ["ATCG"], "model_name": "m", "stream_progress": False},
    ),
    (
        "adna_stream_multi_model_predict",
        ("ATCG", ["m1"], False),
        "dna_stream_multi_model_predict",
        {"sequence": "ATCG", "model_names": ["m1"], "stream_progress": False},
    ),
    ("alist_loaded_models", (), "list_loaded_models", {}),
    ("aget_model_info", ("m1",), "get_model_info", {"model_name": "m1"}),
    ("alist_models_by_task_type", ("binary",), "list_models_by_task_type", {"task_type": "binary"}),
    ("aget_all_available_models", (), "get_all_available_models", {}),
    ("ahealth_check", (), "health_check", {}),
]


@pytest.mark.parametrize(("method", "args", "tool", "payload"), ASYNC_TYPED_CALLS)
@pytest.mark.asyncio
async def test_async_typed_methods_call_correct_tool(
    method, args, tool, payload, mock_session, mock_connect
):
    """Each async typed method invokes its named MCP tool with the payload."""
    client = DNALLMMCPClient(transport="sse")
    client._connect = lambda: mock_connect

    result = await getattr(client, method)(*args)

    mock_session.call_tool.assert_awaited_once_with(tool, payload)
    assert result == {"result": "ok"}


SYNC_TYPED_CALLS = [
    (
        "dna_batch_predict",
        (["ATCG"], "m"),
        "dna_batch_predict",
        {"sequences": ["ATCG"], "model_name": "m"},
    ),
    (
        "dna_multi_model_predict",
        ("ATCG", ["m1"]),
        "dna_multi_model_predict",
        {"sequence": "ATCG", "model_names": ["m1"]},
    ),
    (
        "dna_stream_predict",
        ("ATCG", "m"),
        "dna_stream_predict",
        {"sequence": "ATCG", "model_name": "m", "stream_progress": True},
    ),
    (
        "dna_stream_batch_predict",
        (["ATCG"], "m"),
        "dna_stream_batch_predict",
        {"sequences": ["ATCG"], "model_name": "m", "stream_progress": True},
    ),
    (
        "dna_stream_multi_model_predict",
        ("ATCG", ["m1"]),
        "dna_stream_multi_model_predict",
        {"sequence": "ATCG", "model_names": ["m1"], "stream_progress": True},
    ),
    ("list_loaded_models", (), "list_loaded_models", {}),
    ("get_model_info", ("m1",), "get_model_info", {"model_name": "m1"}),
    ("list_models_by_task_type", ("binary",), "list_models_by_task_type", {"task_type": "binary"}),
    ("get_all_available_models", (), "get_all_available_models", {}),
    ("health_check", (), "health_check", {}),
]


@pytest.mark.parametrize(("method", "args", "tool", "payload"), SYNC_TYPED_CALLS)
def test_sync_typed_methods_call_correct_tool(
    method, args, tool, payload, mock_session, mock_connect
):
    """Each sync typed method bridges to its named MCP tool."""
    client = DNALLMMCPClient(transport="sse")
    client._connect = lambda: mock_connect

    result = getattr(client, method)(*args)

    mock_session.call_tool.assert_awaited_once_with(tool, payload)
    assert result == {"result": "ok"}
