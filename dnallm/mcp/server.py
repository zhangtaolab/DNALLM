"""DNALLM MCP Server Implementation.

This module implements the main MCP (Model Context Protocol) server using
the FastMCP framework with Server-Sent Events (SSE) support for real-time
DNA sequence prediction.

The server provides a comprehensive set of tools for DNA sequence analysis,
including:
- Single sequence prediction with specific models
- Batch processing of multiple sequences
- Multi-model prediction and comparison
- Real-time streaming predictions with progress updates
- Model management and health monitoring

Architecture:
    The server is built on top of the FastMCP framework, which provides MCP
    protocol implementation with multiple transport options (stdio, SSE,
    HTTP). The server manages DNA language models through a ModelManager and
    handles configuration through a ConfigManager.

Transport Protocols:
    - stdio: Standard input/output for CLI tools
    - streamable-http: HTTP-based streaming protocol (recommended for
      remote connections, per MCP spec 2025-11-25)
    - sse: Server-Sent Events for real-time web applications (legacy,
      deprecated in MCP spec 2025-11-25 but retained for backward
      compatibility)

Example:
    Basic server initialization:

    ```python
    server = DNALLMMCPServer("config/server_config.yaml")
    await server.initialize()
    server.start_server(host="127.0.0.1", port=8000,
                        transport="streamable-http")
    ```

Note:
    This server requires proper configuration files and model setup before
    initialization. See the configuration documentation for details.
"""

import asyncio
import functools
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

import numpy as np
from loguru import logger

# MCP SDK imports
from mcp.server.fastmcp import FastMCP

# tool decorator is available as app.tool() method

from .config_manager import MCPConfigManager
from .model_manager import ModelManager
from ..inference.mutagenesis import Mutagenesis
from ..inference.interpret import DNAInterpret


class DNALLMMCPServer:
    """DNALLM MCP Server implementation using FastMCP framework with SSE
    support.

    This class provides a comprehensive MCP server for DNA language model
    inference and analysis. It supports multiple transport protocols and
    provides real-time streaming capabilities for DNA sequence prediction
    tasks.

    The server manages multiple DNA language models and provides various
    prediction modes including single sequence prediction, batch processing,
    and multi-model comparison. All operations support progress reporting
    through streaming transports for real-time user feedback.

    Attributes:
        config_path (str): Path to the main server configuration file
        config_manager (MCPConfigManager): Handles configuration management
        model_manager (ModelManager): Manages model loading and prediction
        app (FastMCP | None): Main FastMCP application instance
        sse_app: SSE application instance (unused, FastMCP handles SSE
            internally)
        _initialized (bool): Server initialization status flag

    Example:
        Initialize and start the server:

        ```python
        # Create server instance
        server = DNALLMMCPServer("config/mcp_server_config.yaml")

        # Initialize asynchronously
        await server.initialize()

        # Start with Streamable HTTP transport (recommended)
        server.start_server(host="0.0.0.0", port=8000,
                           transport="streamable-http")

        # Start with SSE transport (legacy, backward compatible)
        server.start_server(host="0.0.0.0", port=8000,
                           transport="sse")
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
            config_path (str): Absolute or relative path to the main MCP
                server configuration file. This file should contain server
                settings, model configurations, and transport options.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ConfigurationError: If the configuration file is invalid

        Example:
            ```python
            server = DNALLMMCPServer("/path/to/config.yaml")
            ```

        Note:
            The configuration directory and filename are extracted
            separately to support the MCPConfigManager's directory-based
            configuration loading strategy.
        """
        self.config_path = config_path
        # Extract directory and filename from config file path for
        # ConfigManager. MCPConfigManager requires separate directory and
        # filename parameters
        config_path_obj = Path(config_path)
        config_dir = config_path_obj.parent
        config_filename = config_path_obj.name
        # Initialize core components
        self.config_manager = MCPConfigManager(str(config_dir), config_filename)
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

        # Load timeout and logging configuration
        timeout_config = self.config_manager.get_timeout_config()
        self._tool_timeout_seconds = timeout_config.get("tool_timeout_seconds", 30)

        logging_config = self.config_manager.get_logging_config()
        self._log_format = logging_config.get("log_format", "text")

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

        # Register basic prediction tools (wrapped with timeout)
        self.app.tool()(
            self._with_timeout_wrapper(self._dna_sequence_predict, "dna_sequence_predict")
        )
        self.app.tool()(self._with_timeout_wrapper(self._dna_batch_predict, "dna_batch_predict"))
        self.app.tool()(
            self._with_timeout_wrapper(self._dna_multi_model_predict, "dna_multi_model_predict")
        )

        # Register model management tools (wrapped with timeout)
        self.app.tool()(self._with_timeout_wrapper(self._list_loaded_models, "list_loaded_models"))
        self.app.tool()(self._with_timeout_wrapper(self._get_model_info, "get_model_info"))
        self.app.tool()(
            self._with_timeout_wrapper(self._list_models_by_task_type, "list_models_by_task_type")
        )
        self.app.tool()(
            self._with_timeout_wrapper(self._get_all_available_models, "get_all_available_models")
        )

        # Register monitoring and streaming tools
        self.app.tool()(self._with_timeout_wrapper(self._health_check, "health_check"))
        # Streaming tools handle timeout internally (chunk-based)
        self.app.tool()(self._dna_stream_predict)
        self.app.tool()(self._dna_stream_batch_predict)
        self.app.tool()(self._dna_stream_multi_model_predict)

        # Register mutagenesis and interpretation tools (wrapped with timeout)
        self.app.tool()(self._with_timeout_wrapper(self._dna_mutagenesis, "dna_mutagenesis"))
        self.app.tool()(self._with_timeout_wrapper(self._dna_interpret, "dna_interpret"))

        logger.info("Registered MCP tools successfully")

    def _with_timeout_wrapper(self, tool_func, tool_name: str):
        """Create a timeout wrapper for a tool function.

        Wraps an async tool function with timeout handling and structured
        logging. Non-streaming tools use asyncio.wait_for for the entire
        call. Streaming tools handle timeout internally via chunk-based
        timers.

        Args:
            tool_func: The async tool function to wrap
            tool_name: Name of the tool for logging and error reporting

        Returns:
            Wrapped async function with timeout and logging
        """

        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    tool_func(*args, **kwargs),
                    timeout=self._tool_timeout_seconds,
                )
                duration_ms = (time.perf_counter() - start) * 1000
                self._structured_log(
                    "info",
                    f"Tool {tool_name} completed",
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    status="success",
                )
                return result
            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - start) * 1000
                self._structured_log(
                    "error",
                    f"Tool {tool_name} timed out",
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    status="error",
                )
                return {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": (f"Timeout after {self._tool_timeout_seconds}s"),
                        }
                    ],
                    "error_type": "timeout",
                    "timeout_seconds": self._tool_timeout_seconds,
                    "tool_name": tool_name,
                    "suggestion": (
                        "Try with fewer positions, smaller sequence, or increase timeout in config"
                    ),
                }

        # Preserve the original function's signature for FastMCP
        functools.update_wrapper(wrapper, tool_func)
        return wrapper

    def _structured_log(
        self,
        level: str,
        message: str,
        tool_name: str | None = None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        status: str | None = None,
        **extra,
    ) -> None:
        """Emit a structured log entry.

        Supports both JSON and text log formats. JSON format is designed
        for log aggregation in production deployments. Text format is
        backward-compatible human-readable output.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            tool_name: Optional tool name for context
            request_id: Optional request identifier for tracing
            duration_ms: Optional operation duration in milliseconds
            status: Optional operation status (success, error, etc.)
            **extra: Additional fields for JSON format
        """
        # Normalize level for loguru (requires uppercase)
        loguru_level = level.upper()
        if self._log_format == "json":
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "level": loguru_level,
                "message": message,
            }
            if tool_name is not None:
                log_entry["tool_name"] = tool_name
            if request_id is not None:
                log_entry["request_id"] = request_id
            if duration_ms is not None:
                log_entry["duration_ms"] = round(duration_ms, 2)  # type: ignore[assignment]
            if status is not None:
                log_entry["status"] = status
            log_entry.update(extra)
            logger.opt(raw=True).log(loguru_level, json.dumps(log_entry))
        else:
            parts = [message]
            if tool_name:
                parts.append(f"[tool={tool_name}]")
            if duration_ms is not None:
                parts.append(f"[duration={duration_ms:.2f}ms]")
            if status:
                parts.append(f"[status={status}]")
            logger.log(loguru_level, " ".join(parts))

    async def _dna_sequence_predict(self, sequence: str, model_name: str) -> dict[str, Any]:
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
            result = await self.model_manager.predict_sequence(model_name, sequence)

            # Check if prediction was successful
            if result is None:
                return {
                    "error": (f"Model {model_name} not available or prediction failed"),
                    "isError": True,
                }

            # Return successful prediction in MCP format
            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_name": model_name,
                "sequence": sequence,
            }
        except Exception as e:
            logger.error(f"Error in dna_sequence_predict: {e}", exc_info=True)
            return {
                "content": [
                    {"type": "text", "text": "Prediction failed. See server logs for details."}
                ],
                "isError": True,
            }

    async def _dna_batch_predict(self, sequences: list[str], model_name: str) -> dict[str, Any]:
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
                - On success: Contains 'content', 'model_name',
                  'sequence_count'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            result = await server._dna_batch_predict(
                sequences=["ATCGATCG", "GCTAGCTA", "TTAACCGG"],
                model_name=(
                    "zhangtaolab/plant-dnabert-BPE-open_chromatin"
                )
            )
            ```

        Note:
            Batch processing is generally more efficient than individual
            predictions for multiple sequences, especially with GPU
            acceleration.
        """
        try:
            result = await self.model_manager.predict_batch(model_name, sequences)
            if result is None:
                return {
                    "error": (f"Model {model_name} not available or prediction failed"),
                    "isError": True,
                }

            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_name": model_name,
                "sequence_count": len(sequences),
            }
        except Exception as e:
            logger.error(f"Error in dna_batch_predict: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Batch prediction failed. See server logs for details.",
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
            model_names (list[str] | None, optional): List of model names
                to use. If None, uses all currently loaded models.
                Defaults to None.

        Returns:
            dict[str, Any]: Multi-model prediction results in MCP format:
                - On success: Contains 'content', 'model_count', 'sequence'
                - On error: Contains 'error', 'isError' fields

        Example:
            ```python
            # Use specific models
            result = await server._dna_multi_model_predict(
                sequence="ATCGATCG",
                model_names=[
                    "dnabert-2", "nucleotide-transformer"
                ]
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
            result = await self.model_manager.predict_multi_model(model_names, sequence)

            # Return successful results in MCP format
            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_count": len(model_names),
                "sequence": sequence,
            }
        except Exception as e:
            logger.error(f"Error in dna_multi_model_predict: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Multi-model prediction failed. See server logs for details.",
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
            logger.error(f"Error in list_loaded_models: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Failed to list models. See server logs for details.",
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
            logger.error(f"Error in get_model_info: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Failed to get model info. See server logs for details.",
                    }
                ],
                "isError": True,
            }

    async def _list_models_by_task_type(self, task_type: str) -> dict[str, Any]:
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
            logger.error(f"Error in list_models_by_task_type: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Failed to filter models. See server logs for details.",
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
            logger.error(f"Error in get_all_available_models: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Failed to get available models. See server logs for details.",
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
                "total_configured_models": len(self.config_manager.get_enabled_models()),
                "server_name": (server_config.mcp.name if server_config else "Unknown"),
                "server_version": (server_config.mcp.version if server_config else "Unknown"),
            }

            return {
                "content": [{"type": "text", "text": str(health_status)}],
                "health": health_status,
            }
        except Exception as e:
            logger.error(f"Error in health_check: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Health check failed. See server logs for details.",
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
        via Server-Sent Events (SSE). It's designed for interactive
        applications
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
                - On success: Contains 'content', 'model_name',
                  'sequence', 'streamed'
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
            Uses chunk-based timeout that resets after each progress update.
        """
        tool_name = "dna_stream_predict"
        start = time.perf_counter()
        try:
            async with asyncio.timeout(  # type: ignore[attr-defined]
                self._tool_timeout_seconds
            ):
                if stream_progress and context:
                    # Send initial progress update
                    await context.report_progress(
                        0, 100, f"Starting prediction with model {model_name}"
                    )

                # Send progress update for model loading
                if stream_progress and context:
                    await context.report_progress(25, 100, "Loading model and tokenizer...")

                # Perform prediction
                result = await self.model_manager.predict_sequence(model_name, sequence)

                if result is None:
                    error_msg = f"Model {model_name} not available or prediction failed"
                    if stream_progress and context:
                        await context.report_progress(100, 100, f"Error: {error_msg}")
                    duration_ms = (time.perf_counter() - start) * 1000
                    self._structured_log(
                        "error",
                        f"Tool {tool_name} failed",
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        status="error",
                    )
                    return {"error": error_msg, "isError": True}

                # Send progress update for prediction completion
                if stream_progress and context:
                    await context.report_progress(75, 100, "Processing prediction results...")

                # Send final result
                if stream_progress and context:
                    await context.report_progress(100, 100, "Prediction completed successfully")

            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "info",
                f"Tool {tool_name} completed",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="success",
            )
            return {
                "content": [{"type": "text", "text": str(result)}],
                "model_name": model_name,
                "sequence": sequence,
                "streamed": stream_progress,
            }

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} timed out",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            if stream_progress and context:
                await context.report_progress(100, 100, "Error: Timeout - prediction took too long")
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Timeout after {self._tool_timeout_seconds}s",
                    }
                ],
                "error_type": "timeout",
                "timeout_seconds": self._tool_timeout_seconds,
                "tool_name": tool_name,
                "suggestion": (
                    "Try with fewer positions, smaller sequence, or increase timeout in config"
                ),
            }
        except Exception as e:
            if stream_progress and context:
                await context.report_progress(100, 100, "Error: Streaming prediction failed")
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} failed: {e}",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Streaming prediction failed. See server logs for details.",
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
        """Stream batch DNA sequence prediction with real-time progress
        updates."""
        return await self._process_batch_prediction(sequences, model_name, stream_progress, context)

    async def _process_batch_prediction(
        self,
        sequences: list[str],
        model_name: str,
        stream_progress: bool,
        context: Any | None,
    ) -> dict[str, Any]:
        """Process batch prediction with progress reporting."""
        tool_name = "dna_stream_batch_predict"
        start = time.perf_counter()
        try:
            async with asyncio.timeout(  # type: ignore[attr-defined]
                self._tool_timeout_seconds
            ):
                if stream_progress and context:
                    await context.report_progress(
                        0,
                        100,
                        (
                            f"Starting batch prediction with "
                            f"{len(sequences)} sequences using model "
                            f"{model_name}"
                        ),
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
                    result = await self.model_manager.predict_sequence(model_name, sequence)
                    if result is not None:
                        results.append({
                            "sequence": sequence,
                            "result": result,
                            "index": i,
                        })
                    else:
                        results.append({
                            "sequence": sequence,
                            "result": None,
                            "error": (f"Prediction failed for sequence {i + 1}"),
                            "index": i,
                        })

                # Send completion update
                successful_predictions = len([r for r in results if r.get("result") is not None])
                failed_predictions = len([r for r in results if r.get("result") is None])

                if stream_progress and context:
                    await context.report_progress(
                        100,
                        100,
                        (
                            f"Batch prediction completed: "
                            f"{successful_predictions} successful, "
                            f"{failed_predictions} failed"
                        ),
                    )

            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "info",
                f"Tool {tool_name} completed",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="success",
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

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} timed out",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            if stream_progress and context:
                await context.report_progress(
                    100, 100, "Error: Timeout - batch prediction took too long"
                )
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Timeout after {self._tool_timeout_seconds}s",
                    }
                ],
                "error_type": "timeout",
                "timeout_seconds": self._tool_timeout_seconds,
                "tool_name": tool_name,
                "suggestion": (
                    "Try with fewer sequences, smaller sequence length, or "
                    "increase timeout in config"
                ),
            }
        except Exception as e:
            if stream_progress and context:
                await context.report_progress(100, 100, "Error: Streaming batch prediction failed")
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} failed: {e}",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Streaming batch prediction failed. See server logs for details.",
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
        """Stream multi-model DNA sequence prediction with real-time
        progress updates."""
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
        tool_name = "dna_stream_multi_model_predict"
        start = time.perf_counter()
        try:
            async with asyncio.timeout(  # type: ignore[attr-defined]
                self._tool_timeout_seconds
            ):
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
                        (f"Starting multi-model prediction with {len(model_names)} models"),
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
                        (
                            f"Multi-model prediction completed: "
                            f"{successful} successful, {failed} failed"
                        ),
                    )

            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "info",
                f"Tool {tool_name} completed",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="success",
            )
            return result_dict

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} timed out",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            if stream_progress and context:
                await context.report_progress(
                    100, 100, "Error: Timeout - multi-model prediction took too long"
                )
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Timeout after {self._tool_timeout_seconds}s",
                    }
                ],
                "error_type": "timeout",
                "timeout_seconds": self._tool_timeout_seconds,
                "tool_name": tool_name,
                "suggestion": (
                    "Try with fewer models, smaller sequence, or increase timeout in config"
                ),
            }
        except Exception as e:
            if stream_progress and context:
                await context.report_progress(
                    100, 100, "Error: Streaming multi-model prediction failed"
                )
            duration_ms = (time.perf_counter() - start) * 1000
            self._structured_log(
                "error",
                f"Tool {tool_name} failed: {e}",
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Streaming multi-model prediction failed. See server logs for details.",
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
                    (f"Processing with model {i + 1}/{total_models}: {model_name}"),
                )

            # Predict with current model
            result = await self.model_manager.predict_sequence(model_name, sequence)
            if result is not None:
                results[model_name] = result
            else:
                results[model_name] = {
                    "error": (f"Prediction failed with model {model_name}"),
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
        successful_predictions = len([
            r for r in results.values() if not isinstance(r, dict) or r.get("result") is not None
        ])
        failed_predictions = len([
            r for r in results.values() if isinstance(r, dict) and r.get("result") is None
        ])

        return {
            "content": [{"type": "text", "text": str(results)}],
            "model_count": len(model_names),
            "sequence": sequence,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "results": results,
            "streamed": stream_progress,
        }

    async def _dna_mutagenesis(
        self,
        model_name: str,
        sequence: str | None = None,
        sequences: list[str] | None = None,
        mutation_type: str = "single_base_substitution",
        positions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Perform in silico mutagenesis on DNA sequences.

        This tool evaluates the impact of sequence mutations on model
        predictions, supporting single base substitutions, multi-base
        substitutions, deletions, insertions, and exhaustive combinations.

        Args:
            sequence (str | None): Single DNA sequence to mutate. If provided,
                it is processed as a one-element list internally.
            sequences (list[str] | None): List of DNA sequences to mutate.
                Either sequence or sequences must be provided.
            mutation_type (str): Type of mutation to perform. One of:
                "single_base_substitution", "multi_base_substitution",
                "deletion", "insertion", "combo". Defaults to
                "single_base_substitution".
            positions (list[int] | None): 0-based positions to mutate.
                Required and must be non-empty.
            model_name (str): Name of the model to use for prediction.

        Returns:
            dict[str, Any]: Mutagenesis results in MCP format:
                - On success: Contains 'content', 'original_prediction',
                  'mutated_prediction', 'delta', 'affected_positions',
                  'mutation_type', 'model_name'
                - On error: Contains 'error', 'isError' fields
        """
        try:
            # Validate model_name is non-empty
            if not model_name:
                return {
                    "error": "model_name is required",
                    "isError": True,
                }

            # Validate mutation type
            allowed_types = {
                "single_base_substitution",
                "multi_base_substitution",
                "deletion",
                "insertion",
                "combo",
            }
            if mutation_type not in allowed_types:
                return {
                    "error": (
                        f"Invalid mutation_type: {mutation_type}. Must be one of: {allowed_types}"
                    ),
                    "isError": True,
                }

            # Validate positions
            if positions is None or len(positions) == 0:
                return {
                    "error": "positions must be a non-empty list of integers",
                    "isError": True,
                }

            # Validate sequence input
            if sequence is None and sequences is None:
                return {
                    "error": "Either sequence or sequences must be provided",
                    "isError": True,
                }

            # Wrap single sequence as list
            if sequences is None:
                if sequence is None:
                    raise ValueError("Either sequence or sequences must be provided")
                sequences = [sequence]

            # Validate DNA sequence content
            dna_pattern = re.compile(r"^[ACGTacgtNn]+$")
            for i, seq in enumerate(sequences):
                if not dna_pattern.match(seq):
                    return {
                        "error": (
                            f"Sequence at index {i} contains invalid "
                            f"characters. Only A, C, G, T, N "
                            f"(case-insensitive) are allowed."
                        ),
                        "isError": True,
                    }

            # Enforce combo limit: n <= 5 (4^5 = 1024 max combos)
            if mutation_type == "combo" and len(positions) > 5:
                return {
                    "error": (
                        f"Combo mutation supports at most 5 positions "
                        f"(got {len(positions)}). "
                        f"Limit: 4^5 = 1024 combinations."
                    ),
                    "isError": True,
                }

            # Get model inference engine
            inference_engine = self.model_manager.get_inference_engine(model_name)
            if inference_engine is None:
                return {
                    "error": f"Model {model_name} not loaded",
                    "isError": True,
                }

            model = inference_engine.model
            tokenizer = inference_engine.tokenizer
            config = inference_engine.config

            # Prepare mutagenesis parameters based on mutation type
            replace_mut = mutation_type in {
                "single_base_substitution",
                "multi_base_substitution",
                "combo",
            }
            delete_size = 1 if mutation_type == "deletion" else 0
            insert_seq = "N" if mutation_type == "insertion" else None

            results = []
            for seq in sequences:
                mutagenesis = Mutagenesis(model, tokenizer, config)
                mutagenesis.mutate_sequence(
                    seq,
                    replace_mut=replace_mut,
                    delete_size=delete_size,
                    insert_seq=insert_seq,
                )
                eval_result = mutagenesis.evaluate(do_pred=True)

                # Extract original and mutated predictions
                raw = eval_result.get("raw", {})
                original_prediction = {
                    "sequence": raw.get("sequence", seq),
                    "prediction": raw.get("pred", {}),
                    "score": raw.get("score", 0.0),
                }

                # Aggregate mutated predictions
                mutated_entries = [v for k, v in eval_result.items() if k != "raw"]
                mutated_prediction = {
                    "count": len(mutated_entries),
                    "predictions": [
                        {
                            "sequence": e.get("sequence", ""),
                            "prediction": e.get("pred", {}),
                            "logfc": e.get("logfc", 0.0),
                            "diff": e.get("diff", 0.0),
                            "score": e.get("score", 0.0),
                        }
                        for e in mutated_entries
                    ],
                }

                # Compute delta (average logfc and diff)
                if mutated_entries:
                    avg_logfc = float(
                        np.mean([
                            float(np.mean(e.get("logfc", 0)))
                            if hasattr(e.get("logfc", 0), "__len__")
                            else float(e.get("logfc", 0))
                            for e in mutated_entries
                        ])
                    )
                    avg_diff = float(
                        np.mean([
                            float(np.mean(e.get("diff", 0)))
                            if hasattr(e.get("diff", 0), "__len__")
                            else float(e.get("diff", 0))
                            for e in mutated_entries
                        ])
                    )
                else:
                    avg_logfc = 0.0
                    avg_diff = 0.0

                delta = {
                    "average_logfc": avg_logfc,
                    "average_diff": avg_diff,
                }

                results.append({
                    "original_prediction": original_prediction,
                    "mutated_prediction": mutated_prediction,
                    "delta": delta,
                })

            # Format response
            result_payload: dict[str, Any]
            if len(results) == 1:
                result_payload = results[0]
            else:
                result_payload = {
                    "batch_results": results,
                    "sequence_count": len(sequences),
                }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Mutagenesis complete: {mutation_type} "
                            f"at positions {positions} using "
                            f"model {model_name}"
                        ),
                    }
                ],
                **result_payload,
                "affected_positions": positions,
                "mutation_type": mutation_type,
                "model_name": model_name,
            }
        except Exception as e:
            logger.error(f"Error in dna_mutagenesis: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Mutagenesis failed. See server logs for details.",
                    }
                ],
                "isError": True,
            }

    async def _dna_interpret(
        self,
        sequence: str,
        model_name: str,
        method: str = "lig",
        target_class: int | None = None,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """Interpret model predictions using attribution methods.

        This tool provides model interpretability by computing attribution
        scores for each token in a DNA sequence using various Captum methods.

        Args:
            sequence (str): DNA sequence to interpret.
            model_name (str): Name of the model to use.
            method (str): Attribution method. One of: "lig",
                "deeplift", "occlusion", "feature_ablation",
                "layer_conductance", "gradient_shap", "noise_tunnel",
                "integrated_gradients". Defaults to "lig".
            target_class (int | None): Target class index for attribution.
                If None, auto-selects the class with maximum probability.
            max_length (int | None): Maximum token length for tokenizer.

        Returns:
            dict[str, Any]: Interpretation results in MCP format:
                - On success: Contains 'content', 'attributions',
                  'tokens', 'method', 'target_class', 'model_name',
                  'sequence'
                - On error: Contains 'error', 'isError' fields
        """
        try:
            # Validate DNA sequence content
            dna_pattern = re.compile(r"^[ACGTacgtNn]+$")
            if not dna_pattern.match(sequence):
                return {
                    "error": (
                        "Sequence contains invalid characters. "
                        "Only A, C, G, T, N (case-insensitive) are allowed."
                    ),
                    "isError": True,
                }

            # Validate and map method names
            allowed_methods = {
                "lig",
                "deeplift",
                "occlusion",
                "feature_ablation",
                "layer_conductance",
                "gradient_shap",
                "noise_tunnel",
                "integrated_gradients",
            }
            if method not in allowed_methods:
                return {
                    "error": (f"Invalid method: {method}. Must be one of: {allowed_methods}"),
                    "isError": True,
                }

            # Map external method names to internal dispatch names
            method_map = {
                "gradient_shap": "gradshap",
                "integrated_gradients": "lig",
            }
            mapped_method = method_map.get(method, method)

            # Get model inference engine
            inference_engine = self.model_manager.get_inference_engine(model_name)
            if inference_engine is None:
                return {
                    "error": f"Model {model_name} not loaded",
                    "isError": True,
                }

            model = inference_engine.model
            tokenizer = inference_engine.tokenizer
            config = inference_engine.config

            # Auto-select target class if not provided
            if target_class is None:
                pred_result = await self.model_manager.predict_sequence(model_name, sequence)
                if pred_result is not None:
                    # Try to extract probabilities and find max
                    probs = pred_result.get("probabilities", [])
                    if probs:
                        target_class = int(np.argmax(probs))
                    else:
                        # Fallback: use class 0
                        target_class = 0
                else:
                    target_class = 0

            # Instantiate interpreter
            interpreter = DNAInterpret(model, tokenizer, config)  # type: ignore[arg-type]

            # Handle layer_conductance: auto-detect embedding layer
            kwargs: dict[str, Any] = {}
            if mapped_method == "layer_conductance":
                target_layer = interpreter._find_embedding_layer()
                kwargs["target_layer"] = target_layer

            # Run interpretation
            tokens, attr_scores = interpreter.interpret(
                input_seq=sequence,
                method=mapped_method,
                target=target_class,
                max_length=max_length,
                **kwargs,  # type: ignore[arg-type]
            )

            # Normalize attribution scores
            attr_min = float(np.min(attr_scores))
            attr_max = float(np.max(attr_scores))
            attr_range = attr_max - attr_min
            if attr_range > 1e-12:
                normalized = ((attr_scores - attr_min) / (attr_range + 1e-8)).tolist()
            else:
                normalized = np.zeros_like(attr_scores).tolist()

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Interpretation complete: {method} "
                            f"for class {target_class} using "
                            f"model {model_name}"
                        ),
                    }
                ],
                "attributions": {
                    "raw": attr_scores.tolist(),
                    "normalized": normalized,
                },
                "tokens": tokens,
                "method": method,
                "target_class": target_class,
                "model_name": model_name,
                "sequence": sequence,
            }
        except Exception as e:
            logger.error(f"Error in dna_interpret: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Interpretation failed. See server logs for details.",
                    }
                ],
                "isError": True,
            }

    def _create_server_lifespan(self):
        """Create lifespan context manager for server graceful
        startup/shutdown.

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

        This method starts the server using one of the supported transport
        protocols. The server must be initialized before calling this method.
        The transport protocol determines how the server communicates with
        clients:

        - stdio: Standard input/output for CLI tools and automation
        - streamable-http: HTTP-based streaming for REST API integration
          (recommended per MCP spec 2025-11-25)
        - sse: Server-Sent Events (legacy, deprecated in MCP spec 2025-11-25
          but retained for backward compatibility)

        Session management for the ``streamable-http`` transport is handled
        entirely by the FastMCP SDK via ``StreamableHTTPSessionManager``.
        Callers do not need to create, track, or clean up ``MCP-Session-Id``
        headers manually.

        Args:
            host (str, optional): Host address to bind the server to.
                Defaults to "127.0.0.1". Use "0.0.0.0" for all interfaces.
            port (int, optional): Port number to bind the server to.
                Defaults to 8000. Only used for HTTP-based transports.
            transport (str, optional): Transport protocol to use.
                Choices: "stdio", "streamable-http", "sse".
                Defaults to "stdio".

        Raises:
            RuntimeError: If server is not initialized before starting
            OSError: If port is already in use or host is invalid
            ConfigurationError: If transport configuration is invalid

        Example:
            ```python
            # Start with Streamable HTTP (recommended)
            server.start_server(
                host="0.0.0.0",
                port=8000,
                transport="streamable-http"
            )

            # Start with SSE (legacy, backward compatible)
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
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Override host/port from configuration if available
        server_config = self.config_manager.get_server_config()
        if server_config:
            host = server_config.server.host
            port = server_config.server.port

        logger.info(f"Starting DNALLM MCP Server on {host}:{port} with {transport} transport")

        # Validate transport before dispatching
        valid_transports = ("stdio", "sse", "streamable-http")
        if transport not in valid_transports:
            raise ValueError(
                f"Invalid transport: {transport!r}. Must be one of: {valid_transports}"
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
        sse_config = server_config.sse if server_config and hasattr(server_config, "sse") else None
        mount_path = (
            sse_config.mount_path if sse_config and hasattr(sse_config, "mount_path") else "/mcp"
        )
        logger.info(f"Using SSE transport with mount path: {mount_path}")

        # Read configured log level from server config
        log_level = (
            server_config.server.log_level.lower()
            if server_config and hasattr(server_config.server, "log_level")
            else "info"
        )

        # Get the Starlette app from FastMCP
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")
        sse_app = self.app.sse_app()
        logger.info("SSE app created with routes:")
        logger.info("  - /sse: SSE connection endpoint")
        logger.info("  - /messages/: MCP protocol messages")

        # Create a new Starlette app that mounts the SSE app at the
        # correct path
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
            log_level=log_level,
            access_log=False,  # Reduce log noise
            loop="asyncio",
            timeout_keep_alive=5,  # Keep-alive timeout
            timeout_graceful_shutdown=10,  # Graceful shutdown timeout
        )

        uvicorn_server = uvicorn.Server(config)
        uvicorn_server.run()

    def _start_http_server(self, host: str, port: int) -> None:
        """Start Streamable HTTP server.

        This method creates and runs a uvicorn server using the Streamable
        HTTP application provided by FastMCP's ``streamable_http_app()``.
        Session management (including ``MCP-Session-Id`` header handling)
        is performed entirely by the FastMCP SDK via
        ``StreamableHTTPSessionManager`` — callers do not need to manage
        session state manually.

        The uvicorn configuration mirrors the SSE server for consistency:
        asyncio loop, no access logs, keep-alive and graceful-shutdown
        timeouts enabled.
        """
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        logger.info("Using Streamable HTTP transport")

        # Read streamable_http config if available
        server_config = self.config_manager.get_server_config()
        streamable_http_config = (
            server_config.streamable_http
            if server_config and hasattr(server_config, "streamable_http")
            else None
        )

        # Use streamable_http host/port only if server config doesn't specify
        # them (to avoid overriding CLI args which are passed as parameters)
        if streamable_http_config:
            if server_config and server_config.server.host == host:
                host = streamable_http_config.host
            if server_config and server_config.server.port == port:
                port = streamable_http_config.port
            http_path = streamable_http_config.path
        else:
            http_path = "/mcp"

        # Get the Streamable HTTP app from FastMCP
        if self.app is None:
            raise RuntimeError("FastMCP app not initialized")
        http_app = self.app.streamable_http_app()

        logger.info(f"Streamable HTTP endpoint: http://{host}:{port}{http_path}")

        # Read configured log level from server config
        log_level = (
            server_config.server.log_level.lower()
            if server_config and hasattr(server_config.server, "log_level")
            else "info"
        )

        # Wrap HTTP app in Starlette with lifespan to ensure shutdown cleanup
        main_app = Starlette(
            routes=[Mount("", http_app)],
            lifespan=self._create_server_lifespan(),
        )

        # Run the Starlette app with uvicorn with proper signal handling
        config = uvicorn.Config(
            app=main_app,
            host=host,
            port=port,
            log_level=log_level,
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
    configuration validation, server initialization, and graceful error
    handling.

    The CLI supports multiple transport protocols and comprehensive
    configuration
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
        For SSE/HTTP transports, uvicorn handles signal processing
        automatically.
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
  dnallm-mcp-server --config dnallm/mcp/configs/mcp_server_config_2.yaml \
      --transport sse --port 8000
  dnallm-mcp-server --config dnallm/mcp/configs/mcp_server_config.yaml \
      --host 127.0.0.1 --port 9000
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
        help=(
            "Transport protocol (streamable-http=recommended per MCP "
            "2025-11-25, sse=legacy, stdio=default) (default: %(default)s)"
        ),
    )

    parser.add_argument("--version", action="version", version="DNALLM MCP Server 1.0.0")

    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create a configuration file or specify the correct path with --config")
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
        logger.info(f"Starting server on {args.host}:{args.port} with {args.transport} transport")
        logger.info("Press Ctrl+C to stop the server")

        # Start server (uvicorn will handle signals properly)
        server.start_server(host=args.host, port=args.port, transport=args.transport)

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
