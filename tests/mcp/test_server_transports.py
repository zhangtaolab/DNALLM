"""Transport and protocol tests for the DNALLM MCP server.

Two complementary strategies, both socket-free:

- An in-memory client/server pair connecting the official MCP
  ``streamablehttp_client`` to the real ``FastMCP.streamable_http_app()``
  through ``httpx.ASGITransport`` (research Pattern 1, live-verified against
  the installed ``mcp`` 1.30.0). This proves the full protocol round trip:
  initialize, list_tools, call_tool, and the terminating session DELETE.
- Construction-shape tests for the three transport starters with patched
  uvicorn — the assembled ``uvicorn.Config`` fields, the Starlette mounts,
  and the ``app.run`` dispatch are asserted without ever binding a port.

SSE is covered construction-only: the in-memory SSE protocol exchange
deadlocks under ASGITransport (research Pitfall 2, reproduced twice), so
live SSE behavior stays with the existing typed network skips.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import yaml
from starlette.routing import Mount

from dnallm.mcp import server as server_module
from dnallm.mcp.server import DNALLMMCPServer, initialize_mcp_server

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# The complete registration set, enumerated one-for-one from the
# _register_tools() calls in dnallm/mcp/server.py: ten
# timeout-wrapped tools plus the three streaming tools registered
# directly, plus the two Phase 12 analysis tools (ism_scan, hotspots).
# FastMCP derives each wire name from the function __name__
# (functools.update_wrapper inside _with_timeout_wrapper), so the names
# carry the leading underscore of the implementing method.
EXPECTED_TOOLS = {
    "_dna_sequence_predict",
    "_dna_batch_predict",
    "_dna_multi_model_predict",
    "_list_loaded_models",
    "_get_model_info",
    "_list_models_by_task_type",
    "_get_all_available_models",
    "_health_check",
    "_dna_stream_predict",
    "_dna_stream_batch_predict",
    "_dna_stream_multi_model_predict",
    "_dna_mutagenesis",
    "_dna_interpret",
    "_ism_scan",
    "_hotspots",
}

_UNSET = object()


def _write_server_config(
    temp_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8123,
    log_level: str = "INFO",
    streamable_http: dict[str, Any] | object | None = _UNSET,
    streamable_http_port: int = 8124,
    streamable_http_path: str = "/custom-mcp",
) -> Path:
    """Write a real mcp_server_config.yaml and return its path.

    Args:
        temp_dir: Directory the YAML file is written into.
        host: server.host value.
        port: server.port value.
        log_level: server.log_level value (case exercises the lowering).
        streamable_http: The streamable_http block content. Defaults to a
            block derived from the other arguments; pass ``None`` to omit
            the block entirely (exercising the ``/mcp`` default path).
        streamable_http_port: Port used by the default streamable_http block.
        streamable_http_path: Path used by the default streamable_http block.

    Returns:
        Path to the written configuration file.
    """
    config: dict[str, Any] = {
        "server": {
            "host": host,
            "port": port,
            "workers": 1,
            "log_level": log_level,
            "debug": False,
        },
        "mcp": {
            "name": "Transport Test Server",
            "version": "0.1.0",
            "description": "Server for transport tests",
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
        "tool_timeout_seconds": 30,
    }
    if streamable_http is _UNSET:
        config["streamable_http"] = {
            "host": host,
            "port": streamable_http_port,
            "path": streamable_http_path,
        }
    elif streamable_http is not None:
        config["streamable_http"] = streamable_http

    config_path = temp_dir / "mcp_server_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
async def real_server(tmp_path: Path) -> AsyncIterator[DNALLMMCPServer]:
    """Real-config initialized server with the model-load boundary patched.

    Zero models are configured, so initialization is fast; the load
    boundary patch mirrors dnallm/mcp/tests/test_server_integration.py:129.
    """
    config_path = _write_server_config(tmp_path)
    with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
        mock_load.return_value = (Mock(), Mock())
        server = DNALLMMCPServer(str(config_path))
        await server.initialize()
    yield server
    await server.shutdown()


# ---------------------------------------------------------------------------
# In-memory protocol round trip (research Pattern 1)
# ---------------------------------------------------------------------------


class TestInMemoryProtocolRoundTrip:
    """Drive the official MCP client through ASGI into the real server."""

    async def test_round_trip_initialize(self, real_server):
        """initialize() negotiates a protocol version and server info."""
        result = await self._handshake(real_server)
        assert result.serverInfo.name == "Transport Test Server"

    async def test_list_tools_returns_complete_registered_set(self, real_server):
        """list_tools enumerates every tool registered in server.py."""
        async with self._client_session(real_server) as session:
            result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == EXPECTED_TOOLS
        assert len(names) == 15

    async def test_call_tool_health_check(self, real_server):
        """call_tool executes a registered tool and returns its payload."""
        async with self._client_session(real_server) as session:
            result = await session.call_tool("_health_check", {})

        assert result.isError is False
        payload = json.loads(result.content[0].text)
        assert payload["health"]["status"] == "healthy"
        assert payload["health"]["loaded_models"] == 0
        assert payload["health"]["server_name"] == "Transport Test Server"

    async def test_round_trip_ism_scan_unconfigured_model_error_dict(self, real_server):
        """ism_scan: an unknown model returns the matchable error dict."""
        async with self._client_session(real_server) as session:
            result = await session.call_tool(
                "_ism_scan",
                {
                    "model_name": "ghost-model",
                    "sequence": "ATGC",
                    "mutation_type": "single_base_substitution",
                    "positions": [0],
                },
            )

        assert result.isError is False  # error dict, not a protocol raise
        payload = json.loads(result.content[0].text)
        assert payload["isError"] is True
        assert "not configured" in payload["error"]

    async def test_round_trip_ism_scan(self, real_server):
        """ism_scan: happy path through the full protocol with a mock engine."""
        with (
            patch.object(
                real_server.model_manager,
                "get_inference_engine",
                return_value=_mock_ism_engine(),
            ),
            patch.object(
                real_server.model_manager.config_manager,
                "get_model_config",
                return_value=Mock(),
            ),
            patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls,
        ):
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {"sequence": "ATGC", "pred": [0.1], "score": 0.0},
                "mut_0_A_T": {
                    "sequence": "TTGC",
                    "pred": [0.2],
                    "logfc": [0.5],
                    "diff": [0.1],
                    "score": 0.5,
                },
            }
            async with self._client_session(real_server) as session:
                result = await session.call_tool(
                    "_ism_scan",
                    {
                        "model_name": "test-model",
                        "sequence": "ATGC",
                        "mutation_type": "single_base_substitution",
                        "positions": [0],
                    },
                )

        payload = json.loads(result.content[0].text)
        assert payload["model_name"] == "test-model"
        assert payload["affected_positions"] == [0]
        assert payload["original_prediction"]["sequence"] == "ATGC"
        assert payload["mutated_prediction"]["count"] == 1

    async def test_round_trip_hotspots_missing_fasta_error_dict(self, real_server, tmp_path):
        """hotspots: a missing reference FASTA returns the matchable error dict."""
        async with self._client_session(real_server) as session:
            result = await session.call_tool(
                "_hotspots",
                {
                    "model_name": "test-model",
                    "coordinates": {"chrom": "chr1", "start": 0, "end": 50},
                    "fasta_path": str(tmp_path / "missing.fasta"),
                },
            )

        payload = json.loads(result.content[0].text)
        assert payload["isError"] is True
        assert "not found" in payload["error"]
        assert "hotspots" in payload["error"]

    async def test_round_trip_hotspots(self, real_server, tmp_path):
        """hotspots: windows derive from model x coordinates + fasta_path."""
        fasta = tmp_path / "ref.fasta"
        fasta.write_text(">chr1\n" + "ACGT" * 16)  # 64 bases
        with (
            patch.object(
                real_server.model_manager,
                "get_inference_engine",
                return_value=_mock_ism_engine(),
            ),
            patch.object(
                real_server.model_manager.config_manager,
                "get_model_config",
                return_value=Mock(),
            ),
            patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls,
        ):
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {"raw": {"sequence": "ACGT", "score": 0.0}}
            mock_mut.find_hotspots.return_value = [(10, 20)]
            async with self._client_session(real_server) as session:
                result = await session.call_tool(
                    "_hotspots",
                    {
                        "model_name": "test-model",
                        "coordinates": {"chrom": "chr1", "start": 4, "end": 40},
                        "fasta_path": str(fasta),
                    },
                )

        payload = json.loads(result.content[0].text)
        assert payload["hotspots"] == [[10, 20]]
        assert payload["hotspots_genomic"] == [{"chrom": "chr1", "start": 14, "end": 24}]
        assert payload["sequence_length"] == 36
        assert payload["model_name"] == "test-model"
        # The ISM slice comes from the requested coordinates (0-based, then
        # uppercased by the vep window convention).
        call = mock_mut.mutate_sequence.call_args
        assert call.args[0] == "ACGT" * 9  # 36 bases from offset 4

    async def test_session_delete_issued_on_close(self, real_server):
        """Client exit issues the terminating session DELETE request."""
        recorded: list[tuple[str, str]] = []
        asgi_app = real_server.app.streamable_http_app()

        async def recording_asgi(scope, receive, send):
            if scope["type"] == "http":
                recorded.append((scope["method"], scope["path"]))
            await asgi_app(scope, receive, send)

        async with asgi_app.router.lifespan_context(asgi_app):
            async with _streamable_client(recording_asgi) as (read, write, _):
                from mcp import ClientSession

                async with ClientSession(read, write) as session:
                    await session.initialize()

        methods = [method for method, _ in recorded]
        assert "POST" in methods  # initialize went through the pair
        assert ("DELETE", "/mcp") in recorded

    # -- helpers -----------------------------------------------------------

    @staticmethod
    @asynccontextmanager
    async def _client_session(real_server):
        """Yield an initialized in-memory ClientSession (full protocol stack).

        The lifespan context MUST run for the streamable-http session
        manager (research Pitfall 3); the caller uses ``async with`` and
        exiting closes the session (issuing the terminating DELETE).
        """
        from mcp import ClientSession

        asgi_app = real_server.app.streamable_http_app()
        async with asgi_app.router.lifespan_context(asgi_app):
            async with _streamable_client(asgi_app) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    @staticmethod
    async def _handshake(real_server):
        """Run initialize() in a fully-managed context and return its result."""
        async with TestInMemoryProtocolRoundTrip._client_session(real_server) as session:
            return await session.initialize()


def _mock_ism_engine():
    """Minimal mock DNAInference for the ISM-based tool happy paths.

    The flight lock lives on the (real) ModelManager, which the caller
    patches separately only when the manager itself is a Mock.
    """
    engine = Mock()
    engine.model = Mock()
    engine.tokenizer = Mock()
    engine.config = {"task": Mock(task_type="binary"), "inference": Mock(max_length=512)}
    return engine


def _streamable_client(asgi_app):
    """Build the in-memory streamable-http client context (Pattern 1).

    The factory replaces the httpx transport with ASGITransport while
    keeping a localhost base_url — a non-localhost Host header is rejected
    with 421 by mcp 1.30.0 DNS-rebinding protection (research Pitfall 1).
    """
    from mcp.client.streamable_http import streamablehttp_client

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            base_url="http://localhost:8000",
        )

    return streamablehttp_client("http://localhost:8000/mcp", httpx_client_factory=factory)


# ---------------------------------------------------------------------------
# Transport dispatch and construction (patched uvicorn — never a real bind)
# ---------------------------------------------------------------------------


class TestStartServerDispatch:
    """Test the start_server validation and dispatch logic."""

    def test_start_server_requires_initialization(self):
        """An uninitialized server refuses to start."""
        server = DNALLMMCPServer("nonexistent_config.yaml")
        with pytest.raises(RuntimeError, match="Server not initialized"):
            server.start_server(host="127.0.0.1", port=8000)

    def test_invalid_transport_rejected(self, real_server):
        """An unknown transport raises the matchable ValueError."""
        with pytest.raises(ValueError, match=r"Invalid transport: 'carrier-pigeon'"):
            real_server.start_server(host="127.0.0.1", port=8000, transport="carrier-pigeon")

    def test_stdio_dispatch_invokes_app_run(self, real_server):
        """The stdio branch runs the FastMCP app with the stdio transport."""
        mock_app = MagicMock()
        with patch.object(real_server, "app", mock_app):
            real_server.start_server(host="127.0.0.1", port=8000, transport="stdio")
        mock_app.run.assert_called_once_with(transport="stdio")


class TestStreamableHTTPConstruction:
    """Test _start_http_server assembly with patched uvicorn."""

    def test_config_fields_assembled_from_streamable_http_block(self, real_server, tmp_path):
        """An explicit CLI port beats the streamable_http block (REV-11 flip).

        start_server is called with an explicit port 8123 while the YAML
        carries server.port=8123 and streamable_http.port=8124 — under the
        Phase-12 CLI-precedence fix the explicit 8123 must win (this test
        codified the OPPOSITE, YAML-beats-CLI behavior before the fix). The
        app passed to uvicorn still comes from the streamable-http app
        factory.
        """
        del tmp_path  # fixture-managed by real_server; kept for signature clarity
        sentinel_app = Mock()
        mock_app = MagicMock()
        mock_app.streamable_http_app.return_value = sentinel_app
        with (
            patch.object(real_server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server") as mock_server_cls,
        ):
            real_server.start_server(host="127.0.0.1", port=8123, transport="streamable-http")

        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["app"] is sentinel_app
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8123  # explicit CLI beats YAML streamable_http 8124
        assert kwargs["access_log"] is False
        assert kwargs["loop"] == "asyncio"
        assert kwargs["timeout_keep_alive"] == 5
        assert kwargs["timeout_graceful_shutdown"] == 10
        mock_app.streamable_http_app.assert_called_once_with()
        mock_server_cls.assert_called_once_with(mock_config_cls.return_value)
        mock_server_cls.return_value.run.assert_called_once_with()

    def test_default_path_without_streamable_http_block(self, tmp_path):
        """Without a streamable_http block the endpoint defaults to /mcp."""
        config_path = _write_server_config(tmp_path, streamable_http=None)
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            server = DNALLMMCPServer(str(config_path))
            server._initialized = True  # skip model loading; construction only
        mock_app = MagicMock()
        with (
            patch.object(server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            server.start_server(host="127.0.0.1", port=8123, transport="streamable-http")

        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["port"] == 8123  # no block: server.port is kept
        assert kwargs["log_level"] == "info"  # server.log_level lowered
        # The endpoint path lives in the log line; assert via the app factory
        mock_app.streamable_http_app.assert_called_once_with()

    def test_requires_initialized_app(self, real_server):
        """A missing FastMCP app raises before uvicorn runs."""
        with patch.object(real_server, "app", None):
            with pytest.raises(RuntimeError, match="FastMCP app not initialized"):
                real_server._start_http_server("127.0.0.1", 8000)


class TestSSEConstruction:
    """Test _start_sse_server assembly (construction-only; see module docstring)."""

    def _run_sse_start(self, server, host=None, port=None):
        """Start SSE via start_server dispatch with patched uvicorn.

        Driving through ``start_server(transport="sse")`` exercises the
        dispatch line, not just the private starter. ``host``/``port``
        default to the None sentinels so the config-resolution chain runs
        (pass explicit values to exercise CLI precedence).
        """
        sentinel_sse_app = Mock()
        mock_app = MagicMock()
        mock_app.sse_app.return_value = sentinel_sse_app
        with (
            patch.object(server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server") as mock_server_cls,
        ):
            server.start_server(host=host, port=port, transport="sse")
        kwargs = mock_config_cls.call_args.kwargs
        return kwargs, kwargs["app"].routes, mock_server_cls, sentinel_sse_app

    def test_double_mount_and_config_fields(self, real_server):
        """The SSE app is mounted at the mount path AND at the root."""
        kwargs, routes, mock_server_cls, sentinel_sse_app = self._run_sse_start(real_server)

        mounts = [route for route in routes if isinstance(route, Mount)]
        assert [m.path for m in mounts] == ["/mcp", ""]
        assert all(m.app is sentinel_sse_app for m in mounts)
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8123
        assert kwargs["access_log"] is False
        assert kwargs["timeout_graceful_shutdown"] == 10
        mock_server_cls.return_value.run.assert_called_once_with()

    def test_custom_mount_path_from_config(self, real_server):
        """An sse config exposing mount_path customizes the mount."""
        mock_sse_config = MagicMock()
        mock_sse_config.mount_path = "/sse-endpoint"
        mock_config = real_server.config_manager.get_server_config()
        with patch.object(mock_config, "sse", mock_sse_config, create=True):
            _, routes, _, _ = self._run_sse_start(real_server)
        mounts = [route for route in routes if isinstance(route, Mount)]
        assert [m.path for m in mounts] == ["/sse-endpoint", ""]

    def test_log_level_lowered(self, tmp_path):
        """A WARNING server log_level reaches uvicorn lowercase."""
        config_path = _write_server_config(tmp_path, log_level="WARNING")
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            server = DNALLMMCPServer(str(config_path))
            server._initialized = True
        with (
            patch.object(server, "app", MagicMock()),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            server._start_sse_server("127.0.0.1", 8123)
        assert mock_config_cls.call_args.kwargs["log_level"] == "warning"

    def test_requires_initialized_app(self, real_server):
        """A missing FastMCP app raises before the Starlette app is built."""
        with patch.object(real_server, "app", None):
            with pytest.raises(RuntimeError, match="FastMCP app not initialized"):
                real_server._start_sse_server("127.0.0.1", 8000)


class TestHostPortPrecedence:
    """CLI-explicit > transport YAML > server YAML > documented default.

    Phase 12 REV-11 regression matrix: the same resolution chain is proven
    on BOTH HTTP transports via the patched-uvicorn construction pattern
    (research Pitfall 2 — the old bug lived in two places with different
    shapes and a one-sided fix desynced the transports).
    """

    # -- streamable-http transport ----------------------------------------

    def test_http_explicit_cli_beats_all_yaml(self, real_server):
        """Explicit CLI host/port win over both YAML blocks (http path)."""
        mock_app = MagicMock()
        with (
            patch.object(real_server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            real_server.start_server(host="10.9.8.7", port=9999, transport="streamable-http")
        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["host"] == "10.9.8.7"
        assert kwargs["port"] == 9999

    def test_http_yaml_only_uses_streamable_http_block(self, real_server):
        """No CLI flags: the streamable_http block supplies host/port."""
        mock_app = MagicMock()
        with (
            patch.object(real_server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            real_server.start_server(transport="streamable-http")
        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["host"] == "127.0.0.1"  # streamable_http.host
        assert kwargs["port"] == 8124  # streamable_http.port beats server.port 8123

    def test_http_yaml_only_without_block_uses_server_block(self, tmp_path):
        """No CLI flags and no streamable_http block: server block wins."""
        config_path = _write_server_config(tmp_path, streamable_http=None)
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            server = DNALLMMCPServer(str(config_path))
            server._initialized = True  # construction only
        with (
            patch.object(server, "app", MagicMock()),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            server.start_server(transport="streamable-http")
        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8123

    def test_http_no_config_uses_documented_default(self, real_server):
        """No CLI flags and no server config at all: 127.0.0.1:8000."""
        with patch.object(real_server.config_manager, "get_server_config", return_value=None):
            mock_app = MagicMock()
            with (
                patch.object(real_server, "app", mock_app),
                patch("uvicorn.Config") as mock_config_cls,
                patch("uvicorn.Server"),
            ):
                real_server.start_server(transport="streamable-http")
        kwargs = mock_config_cls.call_args.kwargs
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8000

    # -- sse transport ------------------------------------------------------

    def test_sse_explicit_cli_beats_server_yaml(self, real_server):
        """Explicit CLI host/port win over the server YAML block (sse path)."""
        kwargs = self._run_sse(real_server, host="10.9.8.7", port=9999)
        assert kwargs["host"] == "10.9.8.7"
        assert kwargs["port"] == 9999

    def test_sse_yaml_only_uses_server_block(self, real_server):
        """No CLI flags: the server YAML block supplies host/port."""
        kwargs = self._run_sse(real_server)
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8123

    def test_sse_no_config_uses_documented_default(self, real_server):
        """No CLI flags and no server config at all: 127.0.0.1:8000."""
        with patch.object(real_server.config_manager, "get_server_config", return_value=None):
            kwargs = self._run_sse(real_server)
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8000

    @staticmethod
    def _run_sse(server, host=None, port=None):
        """Run the SSE starter through dispatch and return uvicorn kwargs."""
        mock_app = MagicMock()
        with (
            patch.object(server, "app", mock_app),
            patch("uvicorn.Config") as mock_config_cls,
            patch("uvicorn.Server"),
        ):
            server.start_server(host=host, port=port, transport="sse")
        return mock_config_cls.call_args.kwargs


class TestStdioConstruction:
    """Test _start_stdio_server guard."""

    def test_requires_initialized_app(self, real_server):
        """A missing FastMCP app raises before app.run is invoked."""
        with patch.object(real_server, "app", None):
            with pytest.raises(RuntimeError, match="FastMCP app not initialized"):
                real_server._start_stdio_server()


# ---------------------------------------------------------------------------
# Server lifecycle guards
# ---------------------------------------------------------------------------


class TestInitializeGuards:
    """Test initialize() idempotency and configuration-failure guards."""

    async def test_initialize_is_idempotent(self, real_server):
        """A second initialize() returns early without re-registering tools."""
        with patch.object(real_server, "_register_tools") as mock_register:
            await real_server.initialize()
        mock_register.assert_not_called()
        assert real_server._initialized is True

    async def test_initialize_raises_without_server_config(self):
        """A config manager with no server configuration fails loudly."""
        with patch("dnallm.mcp.server.MCPConfigManager") as mock_cm:
            mock_cm_instance = MagicMock()
            mock_cm_instance.get_server_config.return_value = None
            mock_cm.return_value = mock_cm_instance

            server = DNALLMMCPServer("dummy_config.yaml")
            with pytest.raises(RuntimeError, match="Failed to load server configuration"):
                await server.initialize()

    def test_register_tools_requires_app(self):
        """_register_tools refuses to run without a FastMCP app."""
        with patch("dnallm.mcp.server.MCPConfigManager"), patch("dnallm.mcp.server.ModelManager"):
            server = DNALLMMCPServer("dummy_config.yaml")
        server.app = None
        with pytest.raises(RuntimeError, match="FastMCP app not initialized"):
            server._register_tools()


class TestServerLifespan:
    """Test the graceful-shutdown lifespan context manager."""

    async def test_lifespan_runs_shutdown_on_exit(self, real_server):
        """Exiting the lifespan context shuts the server down."""
        shutdown_mock = AsyncMock()
        with patch.object(real_server, "shutdown", shutdown_mock):
            lifespan = real_server._create_server_lifespan()
            async with lifespan(Mock()):
                pass  # server "running"
        shutdown_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# CLI entry point (server.main)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Test the dnallm-mcp-server argparse entry point in server.py."""

    def test_missing_config_exits_with_code_1(self, tmp_path):
        """A missing configuration file exits 1 before any server work."""
        missing = tmp_path / "missing_config.yaml"
        with patch("sys.argv", ["dnallm-mcp-server", "--config", str(missing)]):
            with pytest.raises(SystemExit) as excinfo:
                server_module.main()
        assert excinfo.value.code == 1

    def test_main_starts_configured_transport(self, tmp_path):
        """main() initializes the server and starts the chosen transport."""
        config_path = _write_server_config(tmp_path)
        mock_server = MagicMock()
        mock_server.get_server_info.return_value = {
            "name": "Transport Test Server",
            "version": "0.1.0",
            "loaded_models": [],
            "enabled_models": [],
        }
        with (
            patch(
                "sys.argv",
                [
                    "dnallm-mcp-server",
                    "--config",
                    str(config_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9001",
                    "--transport",
                    "streamable-http",
                ],
            ),
            patch(
                "dnallm.mcp.server.initialize_mcp_server",
                new_callable=AsyncMock,
                return_value=mock_server,
            ) as mock_init,
        ):
            server_module.main()

        mock_init.assert_awaited_once_with(str(config_path))
        mock_server.start_server.assert_called_once_with(
            host="127.0.0.1", port=9001, transport="streamable-http"
        )

    def test_main_keyboard_interrupt_shuts_down_cleanly(self, tmp_path):
        """A KeyboardInterrupt from start_server exits without a failure code."""
        config_path = _write_server_config(tmp_path)
        mock_server = MagicMock()
        mock_server.get_server_info.return_value = {
            "name": "T",
            "version": "0.1.0",
            "loaded_models": [],
            "enabled_models": [],
        }
        mock_server.start_server.side_effect = KeyboardInterrupt
        with (
            patch("sys.argv", ["dnallm-mcp-server", "--config", str(config_path)]),
            patch(
                "dnallm.mcp.server.initialize_mcp_server",
                new_callable=AsyncMock,
                return_value=mock_server,
            ),
        ):
            server_module.main()  # must not raise SystemExit

    def test_main_server_error_exits_with_code_1(self, tmp_path):
        """A generic server error prints the traceback and exits 1."""
        config_path = _write_server_config(tmp_path)
        mock_server = MagicMock()
        mock_server.get_server_info.return_value = {
            "name": "T",
            "version": "0.1.0",
            "loaded_models": [],
            "enabled_models": [],
        }
        mock_server.start_server.side_effect = RuntimeError("boom")
        with (
            patch("sys.argv", ["dnallm-mcp-server", "--config", str(config_path)]),
            patch(
                "dnallm.mcp.server.initialize_mcp_server",
                new_callable=AsyncMock,
                return_value=mock_server,
            ),
        ):
            with pytest.raises(SystemExit) as excinfo:
                server_module.main()
        assert excinfo.value.code == 1


class TestInitializeMCPServerHelper:
    """Test the async convenience initializer."""

    async def test_creates_and_initializes_server(self):
        """initialize_mcp_server builds the server and awaits initialize()."""
        with patch("dnallm.mcp.server.DNALLMMCPServer") as mock_cls:
            mock_cls.return_value.initialize = AsyncMock()
            result = await initialize_mcp_server("cfg.yaml")

        mock_cls.assert_called_once_with("cfg.yaml")
        mock_cls.return_value.initialize.assert_awaited_once()
        assert result is mock_cls.return_value
