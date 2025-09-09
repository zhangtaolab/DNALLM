"""DNALLM MCP Server Implementation.

This module implements the main MCP (Model Context Protocol) server using the FastMCP
framework with Server-Sent Events (SSE) support for real-time DNA sequence prediction.

The server provides a comprehensive set of tools for DNA sequence analysis, including:
- Single sequence prediction with specific models
- Batch processing of multiple sequences
- Multi-model prediction and comparison
- Real-time streaming predictions with progress updates
- Model management and health monitoring

Architecture:
    The server is built on top of the FastMCP framework, which provides MCP protocol
    implementation with multiple transport options (stdio, SSE, HTTP). The server
    manages DNA language models through a ModelManager and handles configuration
    through a ConfigManager.

Transport Protocols:
    - stdio: Standard input/output for CLI tools
    - sse: Server-Sent Events for real-time web applications
    - streamable-http: HTTP-based streaming protocol

Example:
    Basic server initialization:

    ```python
    server = DNALLMMCPServer("config/server_config.yaml")
    await server.initialize()
    server.start_server(host="127.0.0.1", port=8000, transport="sse")
    ```

Note:
    This server requires proper configuration files and model setup before
    initialization. See the configuration documentation for details.
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
    """DNALLM MCP Server implementation using FastMCP framework with SSE support.

    This class provides a comprehensive MCP server for DNA language model inference
    and analysis. It supports multiple transport protocols and provides real-time
    streaming capabilities for DNA sequence prediction tasks.

    The server manages multiple DNA language models and provides various prediction
    modes including single sequence prediction, batch processing, and multi-model
    comparison. All operations support progress reporting through SSE for real-time
    user feedback.

    Attributes:
        config_path (str): Path to the main server configuration file
        config_manager (MCPConfigManager): Handles configuration management
        model_manager (ModelManager): Manages model loading and prediction
        app (FastMCP | None): Main FastMCP application instance
        sse_app: SSE application instance (unused, FastMCP handles SSE internally)
        _initialized (bool): Server initialization status flag

    Example:
        Initialize and start the server:

        ```python
        # Create server instance
        server = DNALLMMCPServer("config/mcp_server_config.yaml")

        # Initialize asynchronously
        await server.initialize()

        # Start with SSE transport
        server.start_server(host="0.0.0.0", port=8000, transport="sse")
        ```

    Note:
        The server must be initialized before starting. Configuration files
        must contain valid model and server settings.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the MCP server instance.

        Sets up the server with configuration and model managers, but does not
        load models or start the server. Call initialize() and start_server()
        separately for complete setup.

        Args:
            config_path (str): Absolute or relative path to the main MCP server
                configuration file. This file should contain server settings,
                model configurations, and transport options.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ConfigurationError: If the configuration file is invalid

        Example:
            ```python
            server = DNALLMMCPServer("/path/to/config.yaml")
            ```

        Note:
            The configuration directory and filename are extracted separately
            to support the MCPConfigManager's directory-based configuration
            loading strategy.
        """
        self.config_path = config_path
        # Extract directory and filename from config file path for ConfigManager
        # MCPConfigManager requires separate directory and filename parameters
        config_path_obj = Path(config_path)
        config_dir = config_path_obj.parent
        config_filename = config_path_obj.name
        # Initialize core components
        self.config_manager = MCPConfigManager(
            str(config_dir), config_filename
        )
        self.model_manager = ModelManager(self.config_manager)

        # FastMCP application instances
        self.app: FastMCP | None = None  # Main MCP application
        self.sse_app = None  # Not used - FastMCP handles SSE internally

        # Server state tracking
        self._initialized = False  # Prevents double initialization

    async def initialize(self) -> None:
        """Initialize the server and load all enabled models.

        This method performs the complete server initialization process:
        1. Checks if already initialized (idempotent operation)
        2. Loads and validates server configuration
        3. Creates the FastMCP application instance
        4. Registers all MCP tools
        5. Loads all enabled DNA language models

        The initialization is asynchronous because model loading can be
        time-consuming, especially for large transformer models.

        Raises:
            RuntimeError: If server configuration cannot be loaded
            ModelLoadError: If critical models fail to load
            ConfigurationError: If configuration is invalid

        Example:
            ```python
            server = DNALLMMCPServer("config.yaml")
            await server.initialize()  # Required before starting
            ```

        Note:
            This method is idempotent - calling it multiple times has no
            additional effect after the first successful initialization.
        """
        # Check for duplicate initialization
        if self._initialized:
            logger.info("Server already initialized")
            return

        logger.info("Initializing DNALLM MCP Server...")

        # Load and validate server configuration
        server_config = self.config_manager.get_server_config()
        if not server_config:
            raise RuntimeError("Failed to load server configuration")

        # Create FastMCP application with configuration
        self.app = FastMCP(
            name=server_config.mcp.name,
            instructions=server_config.mcp.description,
        )

        # Register all available MCP tools
        self._register_tools()

        # Load all enabled models asynchronously
        await self.model_manager.load_all_enabled_models()

        # SSE transport is built into FastMCP framework
        # No need for separate SSE application setup

        # Mark server as initialized
        self._initialized = True
        logger.info("DNALLM MCP Server initialized successfully")

    def _register_tools(self) -> None:
        """Register MCP tools with the FastMCP application.

        This method registers all available DNA sequence prediction tools
        with the FastMCP framework. Each tool is implemented as a separate
        method to maintain low complexity and high maintainability.

        Tools are organized into categories:
        - Basic prediction: single sequence, batch, multi-model
        - Model management: listing, info retrieval, filtering
        - Streaming: real-time prediction with progress updates
        - Health monitoring: server status and diagnostics

        Raises:
            RuntimeError: If FastMCP app is not initialized

        Note:
            This method should only be called after FastMCP app initialization
            and before starting the server. Tools are registered using the
            app.tool() decorator pattern.
        """
        # Validate FastMCP application is ready
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")

        # Register basic prediction tools
        self.app.tool()(self._dna_sequence_predict)
        self.app.tool()(self._dna_batch_predict)
        self.app.tool()(self._dna_multi_model_predict)

        # Register model management tools
        self.app.tool()(self._list_loaded_models)
        self.app.tool()(self._get_model_info)
        self.app.tool()(self._list_models_by_task_type)
        self.app.tool()(self._get_all_available_models)

        # Register monitoring and streaming tools
        self.app.tool()(self._health_check)
        self.app.tool()(self._dna_stream_predict)
        self.app.tool()(self._dna_stream_batch_predict)
        self.app.tool()(self._dna_stream_multi_model_predict)

        logger.info("Registered MCP tools successfully")

    async def _dna_sequence_predict(
        self, sequence: str, model_name: str
    ) -> dict[str, Any]:
        """Predict DNA sequence using a specific model.

        This tool performs single DNA sequence prediction using a specified
        pre-loaded model. It's the most basic prediction operation and serves
        as the foundation for more complex prediction tasks.

        Args:
            sequence (str): DNA sequence to predict, containing only valid
                nucleotide characters (A, T, G, C). Case insensitive.
            model_name (str): Name of the model to use for prediction.
                Must be one of the loaded models.

        Returns:
            dict[str, Any]: Prediction results in MCP format:
                - On success: Contains 'content', 'model_name', 'sequence'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            result = await server._dna_sequence_predict(
                sequence="ATCGATCG",
                model_name="dnabert-2"
            )
            ```

        Note:
            This method handles all exceptions internally and returns
            error information in the response rather than raising exceptions.
        """
        try:
            # Perform prediction through model manager
            result = await self.model_manager.predict_sequence(
                model_name, sequence
            )

            # Check if prediction was successful
            if result is None:
                return {
                    "error": f"Model {model_name} not available or prediction failed",
                    "isError": True,
                }

            # Return successful prediction in MCP format
            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_name": model_name,
                "sequence": sequence,
            }
        except Exception as e:
            # Log error for debugging and return error response
            logger.error(f"Error in dna_sequence_predict: {e}")
            return {
                "content": [
                    {"type": "text", "text": f"Prediction error: {e!s}"}
                ],
                "isError": True,
            }

    async def _dna_batch_predict(
        self, sequences: list[str], model_name: str
    ) -> dict[str, Any]:
        """Predict multiple DNA sequences using a specific model.

        This tool performs batch prediction on multiple DNA sequences using
        a single model. It's optimized for processing multiple sequences
        efficiently by leveraging model batching capabilities.

        Args:
            sequences (list[str]): List of DNA sequences to predict.
                Each sequence should contain only valid nucleotide
                characters (A, T, G, C). Case insensitive.
            model_name (str): Name of the model to use for all predictions.
                Must be one of the loaded models.

        Returns:
            dict[str, Any]: Batch prediction results in MCP format:
                - On success: Contains 'content', 'model_name', 'sequence_count'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            result = await server._dna_batch_predict(
                sequences=["ATCGATCG", "GCTAGCTA", "TTAACCGG"],
                model_name="zhangtaolab/plant-dnabert-BPE-open_chromatin"
            )
            ```

        Note:
            Batch processing is generally more efficient than individual
            predictions for multiple sequences, especially with GPU acceleration.
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

    async def _dna_multi_model_predict(
        self, sequence: str, model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Predict DNA sequence using multiple models in parallel.

        This tool performs prediction on a single DNA sequence using multiple
        models simultaneously, enabling model comparison and ensemble analysis.
        It's particularly useful for understanding prediction consensus across
        different model architectures.

        Args:
            sequence (str): DNA sequence to predict, containing only valid
                nucleotide characters (A, T, G, C). Case insensitive.
            model_names (list[str] | None, optional): List of model names to use.
                If None, uses all currently loaded models. Defaults to None.

        Returns:
            dict[str, Any]: Multi-model prediction results in MCP format:
                - On success: Contains 'content', 'model_count', 'sequence'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            # Use specific models
            result = await server._dna_multi_model_predict(
                sequence="ATCGATCG",
                model_names=["dnabert-2", "nucleotide-transformer"]
            )

            # Use all loaded models
            result = await server._dna_multi_model_predict(
                sequence="ATCGATCG"
            )
            ```

        Note:
            Models are processed in parallel for better performance.
            Individual model failures don't stop the entire operation.
        """
        try:
            # Use all loaded models if none specified
            if model_names is None:
                model_names = self.model_manager.get_loaded_models()

            # Validate that models are available
            if not model_names:
                return {
                    "error": "No models available for prediction",
                    "isError": True,
                }

            # Perform multi-model prediction through model manager
            result = await self.model_manager.predict_multi_model(
                model_names, sequence
            )

            # Return successful results in MCP format
            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_count": len(model_names),
                "sequence": sequence,
            }
        except Exception as e:
            # Log error and return error response
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

    async def _list_loaded_models(self) -> dict[str, Any]:
        """List all currently loaded models."""
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

    async def _get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get detailed information about a specific model."""
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

    async def _list_models_by_task_type(
        self, task_type: str
    ) -> dict[str, Any]:
        """List all available models filtered by task type."""
        try:
            all_models = self.model_manager.get_all_models_info()
            filtered_models = {
                name: info
                for name, info in all_models.items()
                if info.get("task_type") == task_type
            }

            return {
                "content": [{"type": "text", "text": str(filtered_models)}],
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

    async def _get_all_available_models(self) -> dict[str, Any]:
        """Get information about all available models."""
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

    async def _health_check(self) -> dict[str, Any]:
        """Perform health check on the MCP server."""
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

    async def _dna_stream_predict(
        self,
        sequence: str,
        model_name: str,
        stream_progress: bool = True,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Stream DNA sequence prediction with real-time progress updates.

        This tool provides real-time streaming prediction with progress updates
        via Server-Sent Events (SSE). It's designed for interactive applications
        where users need immediate feedback on prediction progress.

        The streaming capability is particularly useful for:
        - Long-running predictions on large sequences
        - Interactive web applications requiring real-time feedback
        - Progress monitoring for batch operations

        Args:
            sequence (str): DNA sequence to predict, containing only valid
                nucleotide characters (A, T, G, C). Case insensitive.
            model_name (str): Name of the model to use for prediction.
                Must be one of the loaded models.
            stream_progress (bool, optional): Whether to send progress updates
                via SSE. Defaults to True.
            context (Any | None, optional): MCP Context object for progress
                reporting. Required for progress updates. Defaults to None.

        Returns:
            dict[str, Any]: Streaming prediction results in MCP format:
                - On success: Contains 'content', 'model_name', 'sequence', 'streamed'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            # With progress streaming
            result = await server._dna_stream_predict(
                sequence="ATCGATCG",
                model_name="dnabert-2",
                stream_progress=True,
                context=mcp_context
            )
            ```

        Note:
            Progress updates are sent at key stages: initialization (0%),
            model loading (25%), processing (75%), and completion (100%).
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
                error_msg = (
                    f"Model {model_name} not available or prediction failed"
                )
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

    async def _dna_stream_batch_predict(
        self,
        sequences: list[str],
        model_name: str,
        stream_progress: bool = True,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Stream batch DNA sequence prediction with real-time progress updates."""
        return await self._process_batch_prediction(
            sequences, model_name, stream_progress, context
        )

    async def _process_batch_prediction(
        self,
        sequences: list[str],
        model_name: str,
        stream_progress: bool,
        context: Any | None,
    ) -> dict[str, Any]:
        """Process batch prediction with progress reporting."""
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

    async def _dna_stream_multi_model_predict(
        self,
        sequence: str,
        model_names: list[str] | None = None,
        stream_progress: bool = True,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """Stream multi-model DNA sequence prediction with real-time progress updates."""
        return await self._process_multi_model_prediction(
            sequence, model_names, stream_progress, context
        )

    async def _process_multi_model_prediction(
        self,
        sequence: str,
        model_names: list[str] | None,
        stream_progress: bool,
        context: Any | None,
    ) -> dict[str, Any]:
        """Process multi-model prediction with progress reporting."""
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

            results = await self._predict_with_multiple_models(
                model_names, sequence, stream_progress, context
            )

            # Send completion update
            result_dict = self._format_multi_model_results(
                results, model_names, sequence, stream_progress
            )

            if stream_progress and context:
                successful = result_dict.get("successful_predictions", 0)
                failed = result_dict.get("failed_predictions", 0)
                await context.report_progress(
                    100,
                    100,
                    f"Multi-model prediction completed: {successful} successful, {failed} failed",
                )

            return result_dict

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

    async def _predict_with_multiple_models(
        self,
        model_names: list[str],
        sequence: str,
        stream_progress: bool,
        context: Any | None,
    ) -> dict[str, Any]:
        """Predict with multiple models and report progress."""
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

        return results

    def _format_multi_model_results(
        self,
        results: dict[str, Any],
        model_names: list[str],
        sequence: str,
        stream_progress: bool,
    ) -> dict[str, Any]:
        """Format multi-model prediction results."""
        # Count successful and failed predictions
        successful_predictions = len(
            [
                r
                for r in results.values()
                if not isinstance(r, dict) or r.get("result") is not None
            ]
        )
        failed_predictions = len(
            [
                r
                for r in results.values()
                if isinstance(r, dict) and r.get("result") is None
            ]
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

    def _create_server_lifespan(self):
        """Create lifespan context manager for server graceful startup/shutdown.

        This method creates an async context manager that handles server
        lifecycle events. It ensures proper startup logging and graceful
        shutdown with resource cleanup when the server receives termination
        signals.

        The lifespan context manager is used by Starlette/FastAPI applications
        to handle application startup and shutdown events properly.

        Returns:
            AsyncContextManager: Context manager for server lifecycle

        Note:
            This follows modern async application lifecycle patterns and
            ensures proper cleanup of models and resources during shutdown.
        """
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app):
            # Startup phase: log successful initialization
            logger.info("Server startup complete")
            yield  # Server is running
            # Shutdown phase: cleanup resources gracefully
            logger.info("Starting graceful shutdown...")
            await self.shutdown()
            logger.info("Graceful shutdown complete")

        return lifespan

    def start_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "stdio",
    ) -> None:
        """Start the MCP server with the specified transport protocol.

        This method starts the server using one of the supported transport protocols.
        The server must be initialized before calling this method. The transport
        protocol determines how the server communicates with clients:

        - stdio: Standard input/output for CLI tools and automation
        - sse: Server-Sent Events for real-time web applications
        - streamable-http: HTTP-based streaming for REST API integration

        Args:
            host (str, optional): Host address to bind the server to.
                Defaults to "127.0.0.1". Use "0.0.0.0" for all interfaces.
            port (int, optional): Port number to bind the server to.
                Defaults to 8000. Only used for HTTP-based transports.
            transport (str, optional): Transport protocol to use.
                Choices: "stdio", "sse", "streamable-http".
                Defaults to "stdio".

        Raises:
            RuntimeError: If server is not initialized before starting
            OSError: If port is already in use or host is invalid
            ConfigurationError: If transport configuration is invalid

        Example:
            ```python
            # Start with SSE for real-time web apps
            server.start_server(
                host="0.0.0.0",
                port=8000,
                transport="sse"
            )

            # Start with stdio for CLI tools
            server.start_server(transport="stdio")
            ```

        Note:
            This method is blocking and will run until the server is stopped.
            For SSE and HTTP transports, uvicorn handles graceful shutdown
            on SIGINT/SIGTERM signals.
        """
        # Validate server initialization state
        if not self._initialized:
            raise RuntimeError(
                "Server not initialized. Call initialize() first."
            )

        # Override host/port from configuration if available
        server_config = self.config_manager.get_server_config()
        if server_config:
            host = server_config.server.host
            port = server_config.server.port

        logger.info(
            f"Starting DNALLM MCP Server on {host}:{port} with {transport} transport"
        )

        # Dispatch to appropriate transport handler
        if transport == "sse":
            self._start_sse_server(host, port)
        elif transport == "streamable-http":
            self._start_http_server(host, port)
        else:
            # Default to stdio transport
            self._start_stdio_server()

    def _start_sse_server(self, host: str, port: int) -> None:
        """Start SSE server."""
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        server_config = self.config_manager.get_server_config()
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
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")
        sse_app = self.app.sse_app()
        logger.info("SSE app created with routes:")
        logger.info("  - /sse: SSE connection endpoint")
        logger.info("  - /messages/: MCP protocol messages")

        # Create a new Starlette app that mounts the SSE app at the correct path
        main_app = Starlette(
            routes=[
                Mount(mount_path, sse_app),
                Mount("", sse_app),  # Also mount at root for /sse
            ],
            lifespan=self._create_server_lifespan(),
        )

        logger.info("Main app created with mounted routes:")
        logger.info("  - /sse: SSE connection endpoint")
        logger.info(f"  - {mount_path}/messages/: MCP protocol messages")
        logger.info(f"Starting uvicorn server on {host}:{port}")

        # Run the main app with uvicorn with proper signal handling
        config = uvicorn.Config(
            app=main_app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,  # Reduce log noise
            loop="asyncio",
            timeout_keep_alive=5,  # Keep-alive timeout
            timeout_graceful_shutdown=10,  # Graceful shutdown timeout
        )

        uvicorn_server = uvicorn.Server(config)
        uvicorn_server.run()

    def _start_http_server(self, host: str, port: int) -> None:
        """Start Streamable HTTP server."""
        import uvicorn

        logger.info("Using Streamable HTTP transport")

        # Get the Streamable HTTP app from FastMCP
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")
        http_app = self.app.streamable_http_app()

        logger.info(
            f"Streamable HTTP app created, starting uvicorn server on {host}:{port}"
        )

        # Run the Starlette app with uvicorn with proper signal handling
        config = uvicorn.Config(
            app=http_app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,  # Reduce log noise
            loop="asyncio",
            timeout_keep_alive=5,  # Keep-alive timeout
            timeout_graceful_shutdown=10,  # Graceful shutdown timeout
        )

        uvicorn_server = uvicorn.Server(config)
        uvicorn_server.run()

    def _start_stdio_server(self) -> None:
        """Start STDIO server."""
        logger.info("Using STDIO transport")
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")
        self.app.run(transport="stdio")

    def get_server_info(self) -> dict[str, Any]:
        """Get server information."""
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
    """Main entry point for the DNALLM MCP server CLI.

    This function provides a command-line interface for starting the DNALLM
    MCP server with various configuration options. It handles argument parsing,
    configuration validation, server initialization, and graceful error handling.

    The CLI supports multiple transport protocols and comprehensive configuration
    options for production deployment. It includes proper error handling and
    logging for troubleshooting.

    Command Line Arguments:
        --config: Path to server configuration file
        --host: Host address to bind to (default: 0.0.0.0)
        --port: Port number to bind to (default: 8000)
        --transport: Protocol (stdio/sse/streamable-http, default: stdio)
        --log-level: Logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        --version: Display version information

    Example Usage:
        ```bash
        # Start with SSE transport
        python server.py --config config.yaml --transport sse --port 8000

        # Start with stdio (default)
        python server.py --config config.yaml

        # Start with debug logging
        python server.py --config config.yaml --log-level DEBUG
        ```

    Exit Codes:
        0: Successful execution
        1: Configuration file not found or server error

    Note:
        The server runs in blocking mode. Use Ctrl+C to stop gracefully.
        For SSE/HTTP transports, uvicorn handles signal processing automatically.
    """
    import asyncio
    import argparse
    import sys
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
        logger.error(f"Configuration file not found: {config_path}")
        logger.error(
            "Please create a configuration file or specify the correct path with --config"
        )
        sys.exit(1)

    try:
        logger.info("Starting DNALLM MCP Server...")
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Transport: {args.transport}")
        logger.info(f"Log Level: {args.log_level}")
        logger.info("-" * 50)

        # Initialize server in asyncio context
        server = asyncio.run(initialize_mcp_server(str(config_path)))

        # Get server info
        info = server.get_server_info()
        logger.info(f"Server initialized: {info['name']} v{info['version']}")
        logger.info(f"Loaded models: {info['loaded_models']}")
        logger.info(f"Enabled models: {info['enabled_models']}")
        logger.info("-" * 50)

        # Start server - let uvicorn handle signals for HTTP/SSE transports
        logger.info(
            f"Starting server on {args.host}:{args.port} with {args.transport} transport"
        )
        logger.info("Press Ctrl+C to stop the server")

        # Start server (uvicorn will handle signals properly)
        server.start_server(
            host=args.host, port=args.port, transport=args.transport
        )

    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Server stopped")


async def initialize_mcp_server(config_path: str) -> DNALLMMCPServer:
    """Initialize the MCP server asynchronously.

    This is a convenience function that creates and initializes a
    DNALLMMCPServer instance. It's designed to be called from
    asyncio.run() or other async contexts.

    Args:
        config_path (str): Path to the server configuration file

    Returns:
        DNALLMMCPServer: Fully initialized server instance ready to start

    Raises:
        ConfigurationError: If configuration is invalid
        ModelLoadError: If model loading fails
        RuntimeError: If initialization fails

    Example:
        ```python
        server = await initialize_mcp_server("config.yaml")
        server.start_server(transport="sse")
        ```

    Note:
        This function combines server creation and initialization
        for convenience in async entry points.
    """
    server = DNALLMMCPServer(str(config_path))
    await server.initialize()
    return server


if __name__ == "__main__":
    # This allows the server to be run directly
    main()
