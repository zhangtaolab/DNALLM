"""DNALLM MCP Server Implementation.

This module implements the main MCP server using FastMCP framework
with SSE support for real-time DNA sequence prediction.
"""

from typing import Any
from pathlib import Path
from loguru import logger

# MCP SDK imports
from mcp.server.fastmcp import FastMCP
# tool decorator is available as app.tool() method

from .config_manager import MCPConfigManager
from .model_manager import ModelManager


class DNALLMMCPServer:
    """DNALLM MCP Server implementation using FastMCP + SSE."""

    def __init__(self, config_path: str):
        """Initialize the MCP server.

        Args:
            config_path: Path to the main MCP server configuration file
        """
        self.config_path = config_path
        # Extract directory and filename from config file path
        config_path_obj = Path(config_path)
        config_dir = config_path_obj.parent
        config_filename = config_path_obj.name
        self.config_manager = MCPConfigManager(
            str(config_dir), config_filename
        )
        self.model_manager = ModelManager(self.config_manager)
        self.app = None
        self.sse_app = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the server and load models."""
        if self._initialized:
            logger.info("Server already initialized")
            return

        logger.info("Initializing DNALLM MCP Server...")

        # Create FastMCP application
        server_config = self.config_manager.get_server_config()
        if not server_config:
            raise RuntimeError("Failed to load server configuration")

        self.app = FastMCP(
            name=server_config.mcp.name,
            instructions=server_config.mcp.description,
        )

        # Register tools
        self._register_tools()

        # Load models
        await self.model_manager.load_all_enabled_models()

        # SSE transport is built into FastMCP
        # No need for separate SSE app

        self._initialized = True
        logger.info("DNALLM MCP Server initialized successfully")

    def _register_tools(self) -> None:
        """Register MCP tools with the FastMCP application."""

        @self.app.tool()
        async def dna_sequence_predict(
            sequence: str, model_name: str
        ) -> dict[str, Any]:
            """Predict DNA sequence using a specific model.

            Args:
                sequence: DNA sequence to predict (A, T, G, C)
                model_name: Name of the model to use for prediction

            Returns:
                Dictionary containing prediction results
            """
            try:
                result = await self.model_manager.predict_sequence(
                    model_name, sequence
                )
                if result is None:
                    return {
                        "error": f"Model {model_name} not available or prediction failed",
                        "isError": True,
                    }

                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_name": model_name,
                    "sequence": sequence,
                }
            except Exception as e:
                logger.error(f"Error in dna_sequence_predict: {e}")
                return {
                    "content": [
                        {"type": "text", "text": f"Prediction error: {e!s}"}
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def dna_batch_predict(
            sequences: list[str], model_name: str
        ) -> dict[str, Any]:
            """Predict multiple DNA sequences using a specific model.

            Args:
                sequences: List of DNA sequences to predict
                model_name: Name of the model to use for prediction

            Returns:
                Dictionary containing batch prediction results
            """
            try:
                result = await self.model_manager.predict_batch(
                    model_name, sequences
                )
                if result is None:
                    return {
                        "error": f"Model {model_name} not available or prediction failed",
                        "isError": True,
                    }

                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_name": model_name,
                    "sequence_count": len(sequences),
                }
            except Exception as e:
                logger.error(f"Error in dna_batch_predict: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Batch prediction error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def dna_multi_model_predict(
            sequence: str, model_names: list[str] | None = None
        ) -> dict[str, Any]:
            """Predict DNA sequence using multiple models in parallel.

            Args:
                sequence: DNA sequence to predict
                model_names: List of model names to use (if None, uses all loaded models)

            Returns:
                Dictionary containing multi-model prediction results
            """
            try:
                if model_names is None:
                    model_names = self.model_manager.get_loaded_models()

                if not model_names:
                    return {
                        "error": "No models available for prediction",
                        "isError": True,
                    }

                result = await self.model_manager.predict_multi_model(
                    model_names, sequence
                )

                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_count": len(model_names),
                    "sequence": sequence,
                }
            except Exception as e:
                logger.error(f"Error in dna_multi_model_predict: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Multi-model prediction error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def list_loaded_models() -> dict[str, Any]:
            """List all currently loaded models.

            Returns:
                Dictionary containing information about loaded models
            """
            try:
                loaded_models = self.model_manager.get_loaded_models()
                models_info = {}

                for model_name in loaded_models:
                    info = self.model_manager.get_model_info(model_name)
                    if info:
                        models_info[model_name] = info

                return {
                    "content": [{"type": "text", "text": str(models_info)}],
                    "loaded_count": len(loaded_models),
                    "models": models_info,
                }
            except Exception as e:
                logger.error(f"Error in list_loaded_models: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error listing models: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def get_model_info(model_name: str) -> dict[str, Any]:
            """Get detailed information about a specific model.

            Args:
                model_name: Name of the model

            Returns:
                Dictionary containing detailed model information
            """
            try:
                info = self.model_manager.get_model_info(model_name)
                if info is None:
                    return {
                        "error": f"Model {model_name} not found",
                        "isError": True,
                    }

                return {
                    "content": [{"type": "text", "text": str(info)}],
                    "model_name": model_name,
                    "info": info,
                }
            except Exception as e:
                logger.error(f"Error in get_model_info: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error getting model info: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def list_models_by_task_type(task_type: str) -> dict[str, Any]:
            """List all available models filtered by task type.

            Args:
                task_type: Task type to filter by (binary, multiclass, regression, etc.)

            Returns:
                Dictionary containing filtered model information
            """
            try:
                all_models = self.model_manager.get_all_models_info()
                filtered_models = {
                    name: info
                    for name, info in all_models.items()
                    if info.get("task_type") == task_type
                }

                return {
                    "content": [
                        {"type": "text", "text": str(filtered_models)}
                    ],
                    "task_type": task_type,
                    "model_count": len(filtered_models),
                    "models": filtered_models,
                }
            except Exception as e:
                logger.error(f"Error in list_models_by_task_type: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error filtering models: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def get_all_available_models() -> dict[str, Any]:
            """Get information about all available models (from model_info.yaml).

            Returns:
                Dictionary containing all available model information
            """
            try:
                # This would integrate with model_info.yaml
                # For now, return configured models
                all_models = self.model_manager.get_all_models_info()

                return {
                    "content": [{"type": "text", "text": str(all_models)}],
                    "total_models": len(all_models),
                    "models": all_models,
                }
            except Exception as e:
                logger.error(f"Error in get_all_available_models: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error getting available models: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def health_check() -> dict[str, Any]:
            """Perform health check on the MCP server.

            Returns:
                Dictionary containing health status information
            """
            try:
                loaded_models = self.model_manager.get_loaded_models()
                server_config = self.config_manager.get_server_config()

                health_status = {
                    "status": "healthy",
                    "loaded_models": len(loaded_models),
                    "total_configured_models": len(
                        self.config_manager.get_enabled_models()
                    ),
                    "server_name": server_config.mcp.name
                    if server_config
                    else "Unknown",
                    "server_version": server_config.mcp.version
                    if server_config
                    else "Unknown",
                }

                return {
                    "content": [{"type": "text", "text": str(health_status)}],
                    "health": health_status,
                }
            except Exception as e:
                logger.error(f"Error in health_check: {e}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Health check error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def dna_stream_predict(
            sequence: str,
            model_name: str,
            stream_progress: bool = True,
            context: Any | None = None,
        ) -> dict[str, Any]:
            """Stream DNA sequence prediction with real-time progress updates via SSE.

            This tool is optimized for SSE transport and provides real-time updates
            during the prediction process using MCP progress reporting.

            Args:
                sequence: DNA sequence to predict (A, T, G, C)
                model_name: Name of the model to use for prediction
                stream_progress: Whether to stream progress updates (default: True)
                context: MCP Context object for progress reporting

            Returns:
                Dictionary containing streaming prediction results with progress updates
            """
            try:
                if stream_progress and context:
                    # Send initial progress update
                    await context.report_progress(
                        0, 100, f"Starting prediction with model {model_name}"
                    )

                # Send progress update for model loading
                if stream_progress and context:
                    await context.report_progress(
                        25, 100, "Loading model and tokenizer..."
                    )

                # Perform prediction
                result = await self.model_manager.predict_sequence(
                    model_name, sequence
                )

                if result is None:
                    error_msg = f"Model {model_name} not available or prediction failed"
                    if stream_progress and context:
                        await context.report_progress(
                            100, 100, f"Error: {error_msg}"
                        )
                    return {"error": error_msg, "isError": True}

                # Send progress update for prediction completion
                if stream_progress and context:
                    await context.report_progress(
                        75, 100, "Processing prediction results..."
                    )

                # Send final result
                if stream_progress and context:
                    await context.report_progress(
                        100, 100, "Prediction completed successfully"
                    )

                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_name": model_name,
                    "sequence": sequence,
                    "streamed": stream_progress,
                }

            except Exception as e:
                logger.error(f"Error in dna_stream_predict: {e}")
                if stream_progress and context:
                    await context.report_progress(100, 100, f"Error: {e!s}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Streaming prediction error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def dna_stream_batch_predict(
            sequences: list[str],
            model_name: str,
            stream_progress: bool = True,
            context: Any | None = None,
        ) -> dict[str, Any]:
            """Stream batch DNA sequence prediction with real-time progress updates via SSE.

            This tool provides real-time progress updates for batch predictions,
            showing progress for each sequence in the batch using MCP progress reporting.

            Args:
                sequences: List of DNA sequences to predict
                model_name: Name of the model to use for prediction
                stream_progress: Whether to stream progress updates (default: True)
                context: MCP Context object for progress reporting

            Returns:
                Dictionary containing streaming batch prediction results
            """
            try:
                if stream_progress and context:
                    await context.report_progress(
                        0,
                        100,
                        f"Starting batch prediction with {len(sequences)} sequences using model {model_name}",
                    )

                results = []
                total_sequences = len(sequences)

                for i, sequence in enumerate(sequences):
                    if stream_progress and context:
                        progress = int((i / total_sequences) * 100)
                        await context.report_progress(
                            progress,
                            100,
                            f"Processing sequence {i + 1}/{total_sequences}",
                        )

                    # Predict current sequence
                    result = await self.model_manager.predict_sequence(
                        model_name, sequence
                    )
                    if result is not None:
                        results.append(
                            {
                                "sequence": sequence,
                                "result": result,
                                "index": i,
                            }
                        )
                    else:
                        results.append(
                            {
                                "sequence": sequence,
                                "result": None,
                                "error": f"Prediction failed for sequence {i + 1}",
                                "index": i,
                            }
                        )

                # Send completion update
                successful_predictions = len(
                    [r for r in results if r.get("result") is not None]
                )
                failed_predictions = len(
                    [r for r in results if r.get("result") is None]
                )

                if stream_progress and context:
                    await context.report_progress(
                        100,
                        100,
                        f"Batch prediction completed: {successful_predictions} successful, {failed_predictions} failed",
                    )

                return {
                    "content": [{"type": "text", "text": str(results)}],
                    "model_name": model_name,
                    "sequence_count": len(sequences),
                    "successful_predictions": successful_predictions,
                    "failed_predictions": failed_predictions,
                    "results": results,
                    "streamed": stream_progress,
                }

            except Exception as e:
                logger.error(f"Error in dna_stream_batch_predict: {e}")
                if stream_progress and context:
                    await context.report_progress(100, 100, f"Error: {e!s}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Streaming batch prediction error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        @self.app.tool()
        async def dna_stream_multi_model_predict(
            sequence: str,
            model_names: list[str] | None = None,
            stream_progress: bool = True,
            context: Any | None = None,
        ) -> dict[str, Any]:
            """Stream multi-model DNA sequence prediction with real-time progress updates via SSE.

            This tool provides real-time progress updates for multi-model predictions,
            showing progress for each model in the prediction pipeline using MCP progress reporting.

            Args:
                sequence: DNA sequence to predict
                model_names: List of model names to use (if None, uses all loaded models)
                stream_progress: Whether to stream progress updates (default: True)
                context: MCP Context object for progress reporting

            Returns:
                Dictionary containing streaming multi-model prediction results
            """
            try:
                if model_names is None:
                    model_names = self.model_manager.get_loaded_models()

                if not model_names:
                    return {
                        "error": "No models available for prediction",
                        "isError": True,
                    }

                if stream_progress and context:
                    await context.report_progress(
                        0,
                        100,
                        f"Starting multi-model prediction with {len(model_names)} models",
                    )

                results = {}
                total_models = len(model_names)

                for i, model_name in enumerate(model_names):
                    if stream_progress and context:
                        progress = int((i / total_models) * 100)
                        await context.report_progress(
                            progress,
                            100,
                            f"Processing with model {i + 1}/{total_models}: {model_name}",
                        )

                    # Predict with current model
                    result = await self.model_manager.predict_sequence(
                        model_name, sequence
                    )
                    if result is not None:
                        results[model_name] = result
                    else:
                        results[model_name] = {
                            "error": f"Prediction failed with model {model_name}",
                            "result": None,
                        }

                # Send completion update
                successful_predictions = len(
                    [
                        r
                        for r in results.values()
                        if not isinstance(r, dict)
                        or r.get("result") is not None
                    ]
                )
                failed_predictions = len(
                    [
                        r
                        for r in results.values()
                        if isinstance(r, dict) and r.get("result") is None
                    ]
                )

                if stream_progress and context:
                    await context.report_progress(
                        100,
                        100,
                        f"Multi-model prediction completed: {successful_predictions} successful, {failed_predictions} failed",
                    )

                return {
                    "content": [{"type": "text", "text": str(results)}],
                    "model_count": len(model_names),
                    "sequence": sequence,
                    "successful_predictions": successful_predictions,
                    "failed_predictions": failed_predictions,
                    "results": results,
                    "streamed": stream_progress,
                }

            except Exception as e:
                logger.error(f"Error in dna_stream_multi_model_predict: {e}")
                if stream_progress and context:
                    await context.report_progress(100, 100, f"Error: {e!s}")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Streaming multi-model prediction error: {e!s}",
                        }
                    ],
                    "isError": True,
                }

        logger.info("Registered MCP tools successfully")

    def start_server(
        self, host: str = "0.0.0.0", port: int = 8000, transport: str = "stdio"  # noqa: S104
    ) -> None:
        """Start the MCP server.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            transport: Transport protocol to use ("stdio", "sse", or "streamable-http")
        """
        if not self._initialized:
            raise RuntimeError(
                "Server not initialized. Call initialize() first."
            )

        server_config = self.config_manager.get_server_config()
        if server_config:
            host = server_config.server.host
            port = server_config.server.port

        logger.info(
            f"Starting DNALLM MCP Server on {host}:{port} with {transport} transport"
        )

        # Start the FastMCP server with specified transport
        if transport == "sse":
            # For SSE transport, we need to run the Starlette app with uvicorn
            import uvicorn
            from starlette.applications import Starlette
            from starlette.routing import Mount

            sse_config = (
                server_config.sse
                if server_config and hasattr(server_config, "sse")
                else None
            )
            mount_path = (
                sse_config.mount_path
                if sse_config and hasattr(sse_config, "mount_path")
                else "/mcp"
            )
            logger.info(f"Using SSE transport with mount path: {mount_path}")

            # Get the Starlette app from FastMCP
            sse_app = self.app.sse_app()
            logger.info("SSE app created with routes:")
            logger.info("  - /sse: SSE connection endpoint")
            logger.info("  - /messages/: MCP protocol messages")

            # Create a new Starlette app that mounts the SSE app at the correct path
            # This ensures /mcp/messages/ is available for MCP clients
            main_app = Starlette(
                routes=[
                    Mount(mount_path, sse_app),
                    Mount("", sse_app),  # Also mount at root for /sse
                ]
            )

            logger.info("Main app created with mounted routes:")
            logger.info("  - /sse: SSE connection endpoint")
            logger.info(f"  - {mount_path}/messages/: MCP protocol messages")
            logger.info(f"Starting uvicorn server on {host}:{port}")

            # Run the main app with uvicorn
            uvicorn.run(main_app, host=host, port=port, log_level="info")

        elif transport == "streamable-http":
            # For Streamable HTTP transport, we need to run the Starlette app with uvicorn
            import uvicorn

            logger.info("Using Streamable HTTP transport")

            # Get the Streamable HTTP app from FastMCP
            http_app = self.app.streamable_http_app()
            logger.info(
                f"Streamable HTTP app created, starting uvicorn server on {host}:{port}"
            )

            # Run the Starlette app with uvicorn
            uvicorn.run(http_app, host=host, port=port, log_level="info")

        else:
            # Default to stdio transport
            logger.info("Using STDIO transport")
            self.app.run(transport="stdio")

    def get_server_info(self) -> dict[str, Any]:
        """Get server information.

        Returns:
            Dictionary containing server information
        """
        server_config = self.config_manager.get_server_config()
        if not server_config:
            return {"error": "Server configuration not loaded"}

        return {
            "name": server_config.mcp.name,
            "version": server_config.mcp.version,
            "description": server_config.mcp.description,
            "host": server_config.server.host,
            "port": server_config.server.port,
            "loaded_models": self.model_manager.get_loaded_models(),
            "enabled_models": self.config_manager.get_enabled_models(),
            "initialized": self._initialized,
        }

    async def shutdown(self) -> None:
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down DNALLM MCP Server...")

        # Unload all models
        unloaded_count = self.model_manager.unload_all_models()
        logger.info(f"Unloaded {unloaded_count} models during shutdown")

        self._initialized = False
        logger.info("DNALLM MCP Server shutdown complete")


def main():
    """Main entry point for the MCP server CLI."""
    import asyncio
    import argparse
    import sys
    import signal
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Start the DNALLM MCP (Model Context Protocol) server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dnallm-mcp-server --config dnallm/mcp/configs/mcp_server_config.yaml
  dnallm-mcp-server --config dnallm/mcp/configs/mcp_server_config_2.yaml --transport sse --port 8000
  dnallm-mcp-server --config dnallm/mcp/configs/mcp_server_config.yaml --host 127.0.0.1 --port 9000
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="dnallm/mcp/configs/mcp_server_config.yaml",
        help="Path to MCP server configuration file (default: %(default)s)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # noqa: S104
        help="Host to bind the server to (default: %(default)s)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: %(default)s)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: %(default)s)",
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: %(default)s)",
    )

    parser.add_argument(
        "--version", action="version", version="DNALLM MCP Server 1.0.0"
    )

    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(
            "Please create a configuration file or specify the correct path with --config"
        )
        sys.exit(1)

    # Global server variable for signal handling
    server = None

    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if server:
            try:
                # Try to shutdown the server
                asyncio.run(server.shutdown())
            except Exception as e:
                print(f"Error during shutdown: {e}")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("Starting DNALLM MCP Server...")
        print(f"Configuration: {config_path}")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Transport: {args.transport}")
        print(f"Log Level: {args.log_level}")
        print("-" * 50)

        # Initialize server in asyncio context
        server = asyncio.run(initialize_mcp_server(str(config_path)))

        # Get server info
        info = server.get_server_info()
        print(f"Server initialized: {info['name']} v{info['version']}")
        print(f"Loaded models: {info['loaded_models']}")
        print(f"Enabled models: {info['enabled_models']}")
        print("-" * 50)

        # Start server (this is blocking and runs outside asyncio)
        print(
            f"Starting server on {args.host}:{args.port} with {args.transport} transport"
        )
        print("Press Ctrl+C to stop the server")
        server.start_server(
            host=args.host, port=args.port, transport=args.transport
        )

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if "server" in locals():
            # Shutdown server in asyncio context
            asyncio.run(server.shutdown())
        print("Server shutdown complete")


async def initialize_mcp_server(config_path: str):
    """Initialize the MCP server asynchronously."""
    server = DNALLMMCPServer(str(config_path))
    await server.initialize()
    return server


if __name__ == "__main__":
    # This allows the server to be run directly
    main()
