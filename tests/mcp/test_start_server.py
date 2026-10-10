"""Tests for the dnallm-mcp-server start script (dnallm/mcp/start_server.py).

setup_logging writes ``logs/mcp_server.log`` under the CURRENT working
directory, so every test here runs inside ``tmp_path`` via a mandatory
autouse fixture (the twice-run tree-clean gate is the tripwire — research
Pitfall 5). The argparse ``main`` is driven with a patched server entry
point so no transport is ever started.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dnallm.mcp import start_server
from dnallm.mcp.start_server import initialize_server, main, setup_logging


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """MANDATORY: run every test under tmp_path (setup_logging writes logs/)."""
    monkeypatch.chdir(tmp_path)


class TestSetupLogging:
    """Test setup_logging handler behavior."""

    def test_creates_log_file_under_cwd(self):
        """The file handler writes logs/mcp_server.log under the cwd."""
        from loguru import logger

        setup_logging("INFO")
        logger.info("hello from the start_server test")

        log_file = Path.cwd() / "logs" / "mcp_server.log"
        assert log_file.exists()
        assert "hello from the start_server test" in log_file.read_text()

    def test_double_setup_replaces_instead_of_stacking(self):
        """Re-running setup_logging keeps exactly console + file handlers."""
        from loguru import logger

        setup_logging("INFO")
        setup_logging("DEBUG")

        assert len(logger._core.handlers) == 2  # console + file, not 4


class TestInitializeServer:
    """Test the async initializer helper."""

    @pytest.mark.asyncio
    async def test_constructs_and_initializes(self):
        """initialize_server builds the server and awaits initialize()."""
        with patch("dnallm.mcp.start_server.DNALLMMCPServer") as mock_cls:
            mock_cls.return_value.initialize = AsyncMock()
            result = await initialize_server("cfg.yaml")

        mock_cls.assert_called_once_with("cfg.yaml")
        mock_cls.return_value.initialize.assert_awaited_once()
        assert result is mock_cls.return_value


class TestMain:
    """Test the argparse main's parsing and error branches."""

    @staticmethod
    def _mock_server() -> MagicMock:
        server = MagicMock()
        server.get_server_info.return_value = {
            "name": "T",
            "version": "0.1.0",
            "loaded_models": [],
            "enabled_models": [],
        }
        server.shutdown = AsyncMock()
        return server

    def test_missing_config_exits_1(self, tmp_path):
        """A missing configuration file exits 1 before any server work."""
        missing = tmp_path / "missing.yaml"
        with patch("sys.argv", ["dnallm-mcp-server", "--config", str(missing)]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 1

    def test_starts_server_with_parsed_args(self, tmp_path):
        """main() forwards the parsed host/port/transport to start_server."""
        config = tmp_path / "config.yaml"
        config.write_text("dummy: true")
        server = self._mock_server()
        with (
            patch(
                "sys.argv",
                [
                    "dnallm-mcp-server",
                    "--config",
                    str(config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9001",
                    "--transport",
                    "streamable-http",
                ],
            ),
            patch(
                "dnallm.mcp.start_server.initialize_server",
                new_callable=AsyncMock,
                return_value=server,
            ) as mock_init,
        ):
            main()

        mock_init.assert_awaited_once_with(str(config))
        server.start_server.assert_called_once_with(
            host="127.0.0.1", port=9001, transport="streamable-http"
        )
        server.shutdown.assert_awaited_once()  # finally-block cleanup

    def test_defaults_are_forwarded(self, tmp_path):
        """Without flags, the None sentinels reach start_server (REV-11).

        The argparse defaults are ``None`` (Phase 12 CLI-precedence fix), so
        start_server resolves host/port from the YAML config itself — an
        argparse-side ``0.0.0.0``/``8000`` default would masquerade as an
        explicit CLI value and silently beat the config.
        """
        config = tmp_path / "config.yaml"
        config.write_text("dummy: true")
        server = self._mock_server()
        with (
            patch("sys.argv", ["dnallm-mcp-server", "--config", str(config)]),
            patch(
                "dnallm.mcp.start_server.initialize_server",
                new_callable=AsyncMock,
                return_value=server,
            ),
        ):
            main()

        server.start_server.assert_called_once_with(
            host=None,
            port=None,
            transport="stdio",
        )

    def test_keyboard_interrupt_shuts_down_cleanly(self, tmp_path):
        """A KeyboardInterrupt from start_server exits without a failure code."""
        config = tmp_path / "config.yaml"
        config.write_text("dummy: true")
        server = self._mock_server()
        server.start_server = MagicMock(side_effect=KeyboardInterrupt)
        with (
            patch("sys.argv", ["dnallm-mcp-server", "--config", str(config)]),
            patch(
                "dnallm.mcp.start_server.initialize_server",
                new_callable=AsyncMock,
                return_value=server,
            ),
        ):
            main()  # must not raise SystemExit

        server.shutdown.assert_awaited_once()

    def test_server_error_exits_1_after_shutdown(self, tmp_path):
        """A generic server error still runs shutdown, then exits 1."""
        config = tmp_path / "config.yaml"
        config.write_text("dummy: true")
        server = self._mock_server()
        server.start_server = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch("sys.argv", ["dnallm-mcp-server", "--config", str(config)]),
            patch(
                "dnallm.mcp.start_server.initialize_server",
                new_callable=AsyncMock,
                return_value=server,
            ),
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 1
        server.shutdown.assert_awaited_once()
