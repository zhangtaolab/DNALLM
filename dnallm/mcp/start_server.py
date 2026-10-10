"""Start script for DNALLM MCP Server."""

import asyncio
import argparse
import sys
from pathlib import Path
from loguru import logger

try:
    from .server import DNALLMMCPServer
except ImportError:
    # Handle direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from dnallm.mcp.server import DNALLMMCPServer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "mcp_server.log",
        level=log_level,
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        rotation="10 MB",
        retention="7 days",
    )


async def initialize_server(config_path: str) -> DNALLMMCPServer:
    """Initialize the server asynchronously."""
    server = DNALLMMCPServer(str(config_path))
    await server.initialize()
    return server


def main():
    """Main function to start the MCP server."""
    parser = argparse.ArgumentParser(description="Start DNALLM MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/mcp_server_config.yaml",
        help="Path to MCP server configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=(
            "Host to bind HTTP/SSE transports to. Resolution order when "
            "omitted: transport-specific YAML (streamable_http block) > "
            "server YAML > 127.0.0.1. An explicit flag always wins over "
            "the YAML config."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Port to bind HTTP/SSE transports to. Resolution order when "
            "omitted: transport-specific YAML (streamable_http block) > "
            "server YAML > 8000. An explicit flag always wins over the "
            "YAML config."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol to use (stdio, sse, or streamable-http)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create a configuration file or specify the correct path with --config")
        sys.exit(1)

    try:
        # Create and initialize server
        logger.info("Starting DNALLM MCP Server...")

        # Initialize server in asyncio context
        server = asyncio.run(initialize_server(str(config_path)))

        # Get server info
        info = server.get_server_info()
        logger.info(f"Server initialized: {info['name']} v{info['version']}")
        logger.info(f"Loaded models: {info['loaded_models']}")
        logger.info(f"Enabled models: {info['enabled_models']}")

        # Start server (this is blocking and runs outside asyncio). Host/port
        # stay None-sentinels when not passed so start_server resolves them
        # from the YAML config with CLI precedence (Phase 12 REV-11).
        logger.info(
            f"Starting server with {args.transport} transport "
            f"(host: {args.host if args.host is not None else 'from config'}, "
            f"port: {args.port if args.port is not None else 'from config'})"
        )
        server.start_server(host=args.host, port=args.port, transport=args.transport)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        if "server" in locals():
            # Shutdown server in asyncio context
            asyncio.run(server.shutdown())
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
