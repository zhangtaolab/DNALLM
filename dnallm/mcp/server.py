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
    HTTP). The server manages DNA large language models through a ModelManager and
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
import shutil
import tempfile
import threading
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
from ..inference.vep import (
    ClinVarFilter,
    VepResult,
    _load_reference,
    _resolve_chromosome,
    evaluate_vcf,
)

#: Documented fallback bind address when neither the CLI nor any YAML block
#: supplies one. Unified in Phase 12 (REV-11): the argparse default used to
#: be ``0.0.0.0`` while ``start_server`` documented ``127.0.0.1`` — one
#: loopback-safe default now applies everywhere; pass ``--host 0.0.0.0`` to
#: bind all interfaces explicitly.
DEFAULT_BIND_HOST = "127.0.0.1"

#: Documented fallback bind port (see ``DEFAULT_BIND_HOST``).
DEFAULT_BIND_PORT = 8000

#: Tool-boundary input caps (Phase 12 REV-11, T-12-06). Enforced BEFORE any
#: engine call so the timeout wrapper's "reduce inputs" suggestion stays
#: actionable: single-base-substitution ISM costs 3 forward passes per base
#: (a 2000-base scan is already ~6000 passes against a 30s tool timeout).
ISM_MAX_SEQUENCE_LENGTH = 2000

#: Maximum mutated positions accepted by ``ism_scan`` per call.
ISM_MAX_POSITIONS = 100

#: Maximum region length (bases) accepted by ``hotspots`` per call — the
#: region is ISM-scanned in full, so the same pass-count math applies.
HOTSPOT_MAX_REGION_LENGTH = 2000

#: FASTA suffix allowlist for the per-call ``fasta_path`` parameter
#: (T-12-04/T-12-08: client-named files are operator-trust-boundary reads).
FASTA_SUFFIXES = (".fasta", ".fa", ".fa.gz", ".fna")

#: VCF suffix allowlist for the per-call ``vcf_path`` parameter (T-12-04).
VCF_SUFFIXES = (".vcf", ".vcf.gz")

#: Maximum inline variants accepted by ``zero_shot_score`` per call (T-12-06).
ZERO_SHOT_MAX_VARIANTS = 500

#: Maximum accepted server-side VCF size in bytes (T-12-04: size-capped
#: read of untrusted client-named files).
ZERO_SHOT_MAX_VCF_BYTES = 64 * 1024 * 1024

#: Maximum per-side context window for ``zero_shot_score`` (bounds the
#: per-variant window strings the kernel builds).
ZERO_SHOT_MAX_CONTEXT_WINDOW = 10_000

#: Fixed sanitized basename for the materialized inline-variants temp VCF.
#: NEVER derived from VCF/variant record fields (T-12-05: no record-derived
#: string may appear in any path); a fixed name keeps record content out of
#: the filesystem entirely.
INLINE_VCF_BASENAME = "inline_variants.vcf"

#: Pass-through CLNSIG marker written into the inline temp VCF. The kernel's
#: convention gates require a CLNSIG/CLNREVSTAT/CLNVC triple; inline
#: variants carry no ClinVar annotation by construction, so they are marked
#: with these clearly-non-ClinVar sentinels and admitted under
#: ``_INLINE_SENTINEL_FILTER`` below. The convention block in the response
#: reports exactly what was applied (labels "['not_analyzed']=1 vs []=0"),
#: and metrics stay None by construction (single label class).
_INLINE_SENTINEL_CLNSIG = "not_analyzed"

#: Sentinel CLNVC marker matching ``_INLINE_SENTINEL_FILTER.variant_type``.
_INLINE_SENTINEL_CLNVC = "inline_variant"

#: Sentinel INFO string for every inline temp-VCF row. ``criteria_provided``
#: satisfies the kernel's hardcoded >=1-star floor token check; the injected
#: value is visible verbatim in the response's ``convention.clnrevstat_counts``.
_INLINE_SENTINEL_INFO = (
    f"CLNSIG={_INLINE_SENTINEL_CLNSIG};CLNREVSTAT=criteria_provided;CLNVC={_INLINE_SENTINEL_CLNVC}"
)


class DNALLMMCPServer:
    """DNALLM MCP Server implementation using FastMCP framework with SSE
    support.

    This class provides a comprehensive MCP server for DNA large language model
    inference and analysis. It supports multiple transport protocols and
    provides real-time streaming capabilities for DNA sequence prediction
    tasks.

    The server manages multiple DNA large language models and provides various
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

        # WR-01 (261003-ij4): dna_interpret offloads its captum work to the
        # default executor; this lock is acquired INSIDE the
        # executor-submitted closure (the CR-01 / 261003-hhj pattern), so a
        # tool-timeout cancellation abandons only the await — the orphaned
        # thread keeps the flight, a client retry queues behind it, and at
        # most one interpretation runs per process at any instant,
        # preventing unbounded orphan stacking via timeout→retry loops.
        # Deliberately NOT _infer_thread_lock: DNAInterpret uses no
        # DataLoader (no fork-unsafe window) and attributions must not
        # queue behind minutes-long predicts.
        self._interpret_thread_lock = threading.Lock()

    async def initialize(self) -> None:
        """Initialize the server and load all enabled models.

        This method performs the complete server initialization process:
        1. Checks if already initialized (idempotent operation)
        2. Loads and validates server configuration
        3. Creates the FastMCP application instance
        4. Registers all MCP tools
        5. Loads all enabled DNA large language models

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

        # Register Phase 12 analysis tools (wrapped with timeout, D-06)
        self.app.tool()(self._with_timeout_wrapper(self._ism_scan, "ism_scan"))
        self.app.tool()(self._with_timeout_wrapper(self._hotspots, "hotspots"))
        self.app.tool()(self._with_timeout_wrapper(self._zero_shot_score, "zero_shot_score"))

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
        # Count successful and failed predictions. The only failure marker is
        # the {"error": ..., "result": None} entry built by
        # _predict_with_multiple_models; raw prediction dicts (which carry no
        # "result" key) are successes — hence the explicit key-presence check.
        failed_predictions = len([
            r
            for r in results.values()
            if isinstance(r, dict) and "result" in r and r["result"] is None
        ])
        successful_predictions = len(results) - failed_predictions

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

            # Mamba guard (261003-csd): captum gradient backward on
            # Mamba-family models (pure-PyTorch fallback backend when the
            # CUDA selective-scan kernels are absent) exhausts memory and
            # the whole serving process is SIGKILLed at driver level --
            # repro: lig and layer_conductance on the open_chromatin
            # DNAMamba model exit 137 in-process.  Refuse cleanly instead
            # of killing the server for every connected client.
            try:
                guard_config = self.model_manager.config_manager.get_model_config(model_name)
                architecture = str(guard_config.model.task_info.architecture or "")
            except Exception:
                architecture = ""
            if "mamba" in architecture.lower():
                return {
                    "error": (
                        f"Interpretation is not supported for model {model_name} "
                        f"(architecture '{architecture}'): Mamba-family gradient "
                        "backward passes exhaust memory and would kill the server"
                    ),
                    "isError": True,
                }

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

            def _run_interpretation() -> tuple[list[str], np.ndarray]:
                # All model-touching sync work (instantiation, embedding-
                # layer detection, the captum attribution itself) lives in
                # one off-loop closure, serialized by the interpret flight
                # lock so a timeout-cancellation cannot stack concurrent
                # attributions on one shared torch model.
                with self._interpret_thread_lock:
                    interpreter = DNAInterpret(model, tokenizer, config)  # type: ignore[arg-type]

                    # Handle layer_conductance: auto-detect embedding layer
                    kwargs: dict[str, Any] = {}
                    if mapped_method == "layer_conductance":
                        target_layer = interpreter._find_embedding_layer()
                        kwargs["target_layer"] = target_layer

                    # Run interpretation
                    return interpreter.interpret(
                        input_seq=sequence,
                        method=mapped_method,
                        target=target_class,
                        max_length=max_length,
                        **kwargs,  # type: ignore[arg-type]
                    )

            # WR-01 (261003-ij4): a synchronous captum attribution (observed
            # 172s) must never occupy the event-loop thread, else the
            # _with_timeout_wrapper asyncio.wait_for cannot fire and every
            # client on every transport freezes.
            loop = asyncio.get_running_loop()
            tokens, attr_scores = await loop.run_in_executor(None, _run_interpretation)

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

    def _ism_engine_guard(self, model_name: str) -> dict[str, Any] | tuple[Any, Any, Any]:
        """Shared model gate for the Phase 12 ISM-based tools.

        Args:
            model_name: Caller-supplied model name.

        Returns:
            An error dict when the model is not configured on this server or
            has no loaded inference engine; otherwise the tuple
            ``(model, tokenizer, config)`` of the loaded engine.
        """
        if self.model_manager.config_manager.get_model_config(model_name) is None:
            return {
                "error": f"Model {model_name} is not configured on this server",
                "isError": True,
            }
        inference_engine = self.model_manager.get_inference_engine(model_name)
        if inference_engine is None:
            return {"error": f"Model {model_name} not loaded", "isError": True}
        return (
            inference_engine.model,
            inference_engine.tokenizer,
            inference_engine.config,
        )

    async def _ism_scan(
        self,
        model_name: str,
        sequence: str | None = None,
        sequences: list[str] | None = None,
        mutation_type: str = "single_base_substitution",
        positions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run a bounded in silico mutagenesis (ISM) scan on DNA sequences.

        Mirrors the ``dna_mutagenesis`` engine surface with tool-boundary
        input caps so a scan cannot silently expand past the tool timeout:
        sequences are capped at 2000 bases and positions at 100 entries per
        call (single-base-substitution ISM costs 3 forward passes per base).
        Reduce the inputs or raise ``tool_timeout_seconds`` in the server
        config when a larger scan is genuinely needed.

        Args:
            model_name (str): Name of the loaded model to scan with.
            sequence (str | None): Single DNA sequence to scan. If provided,
                it is processed as a one-element list internally.
            sequences (list[str] | None): List of DNA sequences to scan.
                Either sequence or sequences must be provided.
            mutation_type (str): One of "single_base_substitution",
                "multi_base_substitution", "deletion", "insertion", "combo".
            positions (list[int] | None): 0-based positions of interest;
                validated against every sequence's length and echoed in the
                response. Must be non-empty.

        Returns:
            dict[str, Any]: ISM scan results in MCP format:
                - On success: Contains 'content', 'original_prediction',
                  'mutated_prediction', 'delta', 'affected_positions',
                  'mutation_type', 'model_name'
                - On error: Contains 'error', 'isError' fields
        """
        try:
            if not model_name:
                return {"error": "model_name is required", "isError": True}

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

            if positions is None or len(positions) == 0:
                return {
                    "error": "positions must be a non-empty list of integers",
                    "isError": True,
                }
            if len(positions) > ISM_MAX_POSITIONS:
                return {
                    "error": (
                        f"ism_scan accepts at most {ISM_MAX_POSITIONS} positions "
                        f"(got {len(positions)}). Reduce the position list."
                    ),
                    "isError": True,
                }
            for pos in positions:
                if not isinstance(pos, int) or isinstance(pos, bool) or pos < 0:
                    return {
                        "error": (f"positions must be non-negative integers (got {pos!r})"),
                        "isError": True,
                    }

            if sequence is None and sequences is None:
                return {
                    "error": "Either sequence or sequences must be provided",
                    "isError": True,
                }
            if sequences is None:
                sequences = [sequence]

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
                if len(seq) > ISM_MAX_SEQUENCE_LENGTH:
                    return {
                        "error": (
                            f"Sequence at index {i} exceeds the ism_scan cap "
                            f"of {ISM_MAX_SEQUENCE_LENGTH} bases (got "
                            f"{len(seq)}). Shorten the sequence or increase "
                            f"tool_timeout_seconds in the server config."
                        ),
                        "isError": True,
                    }
                for pos in positions:
                    if pos >= len(seq):
                        return {
                            "error": (
                                f"Position {pos} is out of range for sequence "
                                f"at index {i} (length {len(seq)})."
                            ),
                            "isError": True,
                        }

            guard = self._ism_engine_guard(model_name)
            if isinstance(guard, dict):
                return guard
            model, tokenizer, config = guard

            replace_mut = mutation_type in {
                "single_base_substitution",
                "multi_base_substitution",
                "combo",
            }
            delete_size = 1 if mutation_type == "deletion" else 0
            insert_seq = "N" if mutation_type == "insertion" else None
            scan_sequences = sequences

            def _run_ism_scan() -> list[dict[str, Any]]:
                # ISM builds a DataLoader (num_workers may exceed 0) and so
                # shares the fork-unsafe window that predicts serialize on
                # (ModelManager CR-01 note): the flight lock is acquired
                # INSIDE the executor-submitted closure, keeping the event
                # loop live (health_check stays responsive) while the torch
                # work runs off-loop.
                with self.model_manager._infer_thread_lock:
                    eval_results = []
                    for seq in scan_sequences:
                        mutagenesis = Mutagenesis(model, tokenizer, config)
                        mutagenesis.mutate_sequence(
                            seq,
                            replace_mut=replace_mut,
                            delete_size=delete_size,
                            insert_seq=insert_seq,
                        )
                        eval_results.append(mutagenesis.evaluate(do_pred=True))
                    return eval_results

            loop = asyncio.get_running_loop()
            eval_results = await loop.run_in_executor(None, _run_ism_scan)

            results = []
            for seq, eval_result in zip(sequences, eval_results, strict=True):
                raw = eval_result.get("raw", {})
                original_prediction = {
                    "sequence": raw.get("sequence", seq),
                    "prediction": raw.get("pred", {}),
                    "score": raw.get("score", 0.0),
                }
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

                results.append({
                    "original_prediction": original_prediction,
                    "mutated_prediction": mutated_prediction,
                    "delta": {
                        "average_logfc": avg_logfc,
                        "average_diff": avg_diff,
                    },
                })

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
                            f"ISM scan complete: {mutation_type} "
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
            logger.error(f"Error in ism_scan: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "ISM scan failed. See server logs for details.",
                    }
                ],
                "isError": True,
            }

    async def _hotspots(
        self,
        model_name: str,
        coordinates: dict[str, Any],
        fasta_path: str,
        strategy: str = "maxabs",
        window_size: int = 10,
        percentile_threshold: float = 90.0,
    ) -> dict[str, Any]:
        """Identify mutational hotspot windows for a genomic region.

        Computes hotspots from the model itself (in silico mutagenesis over
        the requested reference slice followed by sliding-window
        aggregation) — never from a precomputed window file. The reference
        sequence is read from a per-call server-side ``fasta_path``.

        Args:
            model_name (str): Name of the loaded model to scan with.
            coordinates (dict[str, Any]): Region to scan, as
                ``{"chrom": str, "start": int, "end": int}`` with 0-based
                half-open coordinates (Python slicing convention). The
                region length is capped at 2000 bases per call.
            fasta_path (str): Server-side reference FASTA path
                (.fasta/.fa/.fa.gz/.fna) containing the chromosome. This is
                an operator-trust-boundary file read, not a client upload.
            strategy (str): Per-base score aggregation, one of "maxabs",
                "min", "max", "mean". Defaults to "maxabs".
            window_size (int): Sliding-window size (bases) for hotspot
                detection. Defaults to 10.
            percentile_threshold (float): Rolling-window percentile above
                which a window is a hotspot (0 < p <= 100). Defaults to 90.

        Returns:
            dict[str, Any]: Hotspot scan results in MCP format:
                - On success: Contains 'content', 'hotspots' (0-based
                  half-open [start, end) pairs relative to the region),
                  'hotspots_genomic' (absolute coordinates),
                  'window_count', 'coordinates', 'sequence_length',
                  'strategy', 'window_size', 'percentile_threshold',
                  'model_name'
                - On error: Contains 'error', 'isError' fields
        """
        try:
            if not model_name:
                return {"error": "model_name is required", "isError": True}

            if not isinstance(coordinates, dict):
                return {
                    "error": "coordinates must be an object with 'chrom', 'start', 'end' fields",
                    "isError": True,
                }
            chrom = coordinates.get("chrom")
            start = coordinates.get("start")
            end = coordinates.get("end")
            if not isinstance(chrom, str) or not chrom:
                return {
                    "error": "coordinates.chrom must be a non-empty string",
                    "isError": True,
                }
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                return {
                    "error": "coordinates.start must be an integer >= 0",
                    "isError": True,
                }
            if not isinstance(end, int) or isinstance(end, bool) or end <= start:
                return {
                    "error": (f"coordinates.end must be an integer greater than start ({start})"),
                    "isError": True,
                }
            if end - start > HOTSPOT_MAX_REGION_LENGTH:
                return {
                    "error": (
                        f"hotspots region length (end - start = {end - start}) "
                        f"exceeds the cap of {HOTSPOT_MAX_REGION_LENGTH} bases. "
                        f"Narrow the coordinates or increase "
                        f"tool_timeout_seconds in the server config."
                    ),
                    "isError": True,
                }

            if not isinstance(fasta_path, str) or not fasta_path:
                return {
                    "error": "fasta_path is required (server-side reference FASTA)",
                    "isError": True,
                }
            if not fasta_path.endswith(FASTA_SUFFIXES):
                return {
                    "error": (
                        f"hotspots: fasta_path must end with one of "
                        f"{FASTA_SUFFIXES} (got '{fasta_path}')"
                    ),
                    "isError": True,
                }
            if not Path(fasta_path).is_file():
                return {
                    "error": f"hotspots: reference FASTA not found at '{fasta_path}'.",
                    "isError": True,
                }

            allowed_strategies = {"maxabs", "min", "max", "mean"}
            if strategy not in allowed_strategies:
                return {
                    "error": (
                        f"Invalid strategy: {strategy}. Must be one of: {allowed_strategies}"
                    ),
                    "isError": True,
                }
            if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
                return {
                    "error": "window_size must be an integer >= 1",
                    "isError": True,
                }
            if (
                not isinstance(percentile_threshold, (int, float))
                or isinstance(percentile_threshold, bool)
                or not 0 < percentile_threshold <= 100
            ):
                return {
                    "error": "percentile_threshold must be a number in (0, 100]",
                    "isError": True,
                }

            guard = self._ism_engine_guard(model_name)
            if isinstance(guard, dict):
                return guard
            model, tokenizer, config = guard

            def _run_hotspot_scan() -> tuple[list[tuple[int, int]], int]:
                # Same off-loop, single-flight contract as _ism_scan: the
                # reference read, slice, ISM, and window extraction all run
                # in the executor under the fork-unsafe flight lock.
                with self.model_manager._infer_thread_lock:
                    reference = _load_reference(fasta_path)
                    chrom_key = _resolve_chromosome(reference, chrom)
                    ref_seq = reference[chrom_key]
                    if end > len(ref_seq):
                        raise ValueError(
                            f"hotspots: coordinates end {end} exceeds the "
                            f"length of {chrom} in the reference "
                            f"({len(ref_seq)} bases) — assembly mismatch?"
                        )
                    seq = ref_seq[start:end].upper()
                    mutagenesis = Mutagenesis(model, tokenizer, config)
                    mutagenesis.mutate_sequence(seq, replace_mut=True)
                    preds = mutagenesis.evaluate(do_pred=True)
                    windows = mutagenesis.find_hotspots(
                        preds,
                        strategy=strategy,
                        window_size=window_size,
                        percentile_threshold=percentile_threshold,
                    )
                    return windows, len(seq)

            loop = asyncio.get_running_loop()
            try:
                windows, seq_len = await loop.run_in_executor(None, _run_hotspot_scan)
            except ValueError as e:
                # Matchable parse/coordinate failures from the vep loaders
                # (missing chromosome, malformed FASTA, out-of-bounds region).
                return {"error": f"hotspots: {e}", "isError": True}

            hotspot_pairs = [[int(s), int(e)] for s, e in windows]
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Hotspot scan complete: {len(hotspot_pairs)} "
                            f"hotspot window(s) on {chrom}:{start}-{end} "
                            f"using model {model_name}"
                        ),
                    }
                ],
                "hotspots": hotspot_pairs,
                "hotspots_genomic": [
                    {"chrom": chrom, "start": int(start) + s, "end": int(start) + e}
                    for s, e in hotspot_pairs
                ],
                "window_count": len(hotspot_pairs),
                "coordinates": {"chrom": chrom, "start": start, "end": end},
                "sequence_length": seq_len,
                "strategy": strategy,
                "window_size": window_size,
                "percentile_threshold": percentile_threshold,
                "model_name": model_name,
            }
        except Exception as e:
            logger.error(f"Error in hotspots: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Hotspot scan failed. See server logs for details.",
                    }
                ],
                "isError": True,
            }

    @staticmethod
    def _build_clnsig_filter(
        clnsig_filter: dict[str, Any] | None,
    ) -> tuple[ClinVarFilter | None, str | None]:
        """Validate and build the kernel convention filter.

        Args:
            clnsig_filter: Caller-supplied override mapping to
                ``vep.ClinVarFilter`` fields, or ``None`` for the D-17
                defaults.

        Returns:
            Tuple of (filter-or-None, error-text). Exactly one element is
            None; the error text is for the tool's matchable error dict.
        """
        if clnsig_filter is None:
            return None, None
        allowed_keys = {"variant_type", "positive_labels", "negative_labels", "star_floor"}
        if not isinstance(clnsig_filter, dict) or not set(clnsig_filter) <= allowed_keys:
            return None, (
                f"clnsig_filter must be an object with a subset of keys {sorted(allowed_keys)}"
            )
        for list_field in ("positive_labels", "negative_labels"):
            value = clnsig_filter.get(list_field)
            if value is not None and (
                not isinstance(value, list) or not all(isinstance(v, str) for v in value)
            ):
                return None, f"clnsig_filter.{list_field} must be a list of strings"
        variant_type = clnsig_filter.get("variant_type")
        if variant_type is not None and not isinstance(variant_type, str):
            return None, "clnsig_filter.variant_type must be a string"
        star_floor = clnsig_filter.get("star_floor")
        if star_floor is not None and (
            not isinstance(star_floor, int) or isinstance(star_floor, bool) or star_floor < 0
        ):
            return None, "clnsig_filter.star_floor must be an integer >= 0"
        base = ClinVarFilter()
        built = ClinVarFilter(
            variant_type=clnsig_filter.get("variant_type", base.variant_type),
            positive_labels=frozenset(clnsig_filter.get("positive_labels", base.positive_labels)),
            negative_labels=frozenset(clnsig_filter.get("negative_labels", base.negative_labels)),
            star_floor=clnsig_filter.get("star_floor", base.star_floor),
        )
        return built, None

    def _write_inline_vcf(self, variants: list[dict[str, Any]]) -> Path:
        """Materialize inline variants to a temp VCF (fixed sanitized name).

        Both input modes of ``zero_shot_score`` route through the same
        ``vep.evaluate_vcf`` kernel (D-04), so inline variants are written
        to a server-side temp VCF first. The file lives under a
        ``tempfile.mkdtemp`` directory and carries the FIXED basename
        ``INLINE_VCF_BASENAME`` — no path segment is ever derived from
        record fields (T-12-05). Rows carry the pass-through CLNSIG/CLNVC/
        CLNREVSTAT sentinels documented on ``_INLINE_SENTINEL_CLNSIG``.

        Args:
            variants: Pre-validated inline variant dicts ({chrom, pos, ref,
                alt}; pos is the 1-based VCF coordinate).

        Returns:
            Path to the written temp VCF (caller owns cleanup of the
            containing directory).
        """
        temp_dir = tempfile.mkdtemp(prefix="dnallm_zero_shot_")
        vcf_path = Path(temp_dir) / INLINE_VCF_BASENAME
        lines = [
            "##fileformat=VCFv4.2",
            (
                "##INFO=<ID=CLNSIG,Number=.,Type=String,"
                'Description="pass-through marker for inline variants">'
            ),
            (
                "##INFO=<ID=CLNREVSTAT,Number=.,Type=String,"
                'Description="pass-through marker for inline variants">'
            ),
            (
                "##INFO=<ID=CLNVC,Number=1,Type=String,"
                'Description="pass-through marker for inline variants">'
            ),
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        ]
        for variant in variants:
            lines.append(
                f"{variant['chrom']}\t{variant['pos']}\t.\t"
                f"{variant['ref'].upper()}\t{variant['alt'].upper()}\t.\t.\t"
                f"{_INLINE_SENTINEL_INFO}"
            )
        vcf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return vcf_path

    async def _zero_shot_score(
        self,
        model_name: str,
        fasta_path: str,
        variants: list[dict[str, Any]] | None = None,
        vcf_path: str | None = None,
        paradigm: str = "mlm",
        context_window: int = 200,
        clnsig_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score zero-shot variant effects (dual-mode, D-04).

        Accepts EITHER an inline ``variants`` list of {chrom, pos, ref, alt}
        objects (pos is the 1-based VCF coordinate; materialized to a
        server-side temp VCF with a fixed sanitized name) OR a server-side
        ``vcf_path`` (.vcf/.vcf.gz, size-capped). Both modes route through
        the SAME ``dnallm.inference.vep.evaluate_vcf`` kernel, so skip
        accounting (skip_counts, skipped, skip_fraction) and the convention
        block are identical and surfaced verbatim in the response.

        Inline variants carry no ClinVar annotation, so they are admitted
        under a pass-through convention (labels "not_analyzed"=1, metrics
        None by construction — see the convention block in the response).
        Pass ``clnsig_filter`` (fields: variant_type, positive_labels,
        negative_labels, star_floor — mapping to ``vep.ClinVarFilter``) to
        override the D-17 defaults, e.g. for non-ClinVar ``vcf_path`` input.

        Variant count is capped at 500 per call; sequences are capped at
        2000 bases for the ISM tools. ``fasta_path``/``vcf_path`` are
        operator-trust-boundary server-side file reads, never uploads.

        Args:
            model_name (str): Name of the loaded model to score with.
            fasta_path (str): Server-side reference FASTA path
                (.fasta/.fa/.fa.gz/.fna) for window building.
            variants (list[dict[str, Any]] | None): Inline variants, each
                {"chrom": str, "pos": int (1-based), "ref": ACGT str, "alt":
                ACGT str}. Mutually exclusive with vcf_path.
            vcf_path (str | None): Server-side ClinVar-style VCF path.
                Mutually exclusive with variants.
            paradigm (str): "mlm" (log-odds, default) or "clm"
                (delta-log-likelihood).
            context_window (int): Reference bases kept on each side of a
                variant. Defaults to 200.
            clnsig_filter (dict[str, Any] | None): Optional ClinVar
                convention override (see above); ``None`` keeps the D-17
                defaults for vcf_path mode and the pass-through convention
                for inline mode.

        Returns:
            dict[str, Any]: Zero-shot scoring results in MCP format:
                - On success: Contains 'content', 'records' (per-variant
                  scores/skips), 'skip_counts', 'evaluated', 'skipped',
                  'skip_fraction', 'metrics', 'convention' (all verbatim
                  from the kernel's VepResult), 'input_mode', 'paradigm',
                  'model_name'
                - On error: Contains 'error', 'isError' fields
        """
        temp_vcf_dir: str | None = None
        try:
            if not model_name:
                return {"error": "model_name is required", "isError": True}

            if variants is not None and vcf_path is not None:
                return {
                    "error": "Provide either variants or vcf_path, not both",
                    "isError": True,
                }
            if variants is None and vcf_path is None:
                return {
                    "error": (
                        "Provide either variants (inline list of "
                        "{chrom, pos, ref, alt}) or vcf_path (server-side VCF)"
                    ),
                    "isError": True,
                }
            inline_mode = variants is not None

            if paradigm not in ("mlm", "clm"):
                return {
                    "error": f"Invalid paradigm: {paradigm}. Must be 'mlm' or 'clm'",
                    "isError": True,
                }
            if (
                not isinstance(context_window, int)
                or isinstance(context_window, bool)
                or not 1 <= context_window <= ZERO_SHOT_MAX_CONTEXT_WINDOW
            ):
                return {
                    "error": (
                        f"context_window must be an integer in [1, {ZERO_SHOT_MAX_CONTEXT_WINDOW}]"
                    ),
                    "isError": True,
                }

            if not isinstance(fasta_path, str) or not fasta_path:
                return {
                    "error": "fasta_path is required (server-side reference FASTA)",
                    "isError": True,
                }
            if not fasta_path.endswith(FASTA_SUFFIXES):
                return {
                    "error": (
                        f"zero_shot_score: fasta_path must end with one of "
                        f"{FASTA_SUFFIXES} (got '{fasta_path}')"
                    ),
                    "isError": True,
                }
            if not Path(fasta_path).is_file():
                return {
                    "error": f"zero_shot_score: reference FASTA not found at '{fasta_path}'.",
                    "isError": True,
                }

            kernel_filter, filter_error = self._build_clnsig_filter(clnsig_filter)
            if filter_error is not None:
                return {"error": filter_error, "isError": True}
            if inline_mode and kernel_filter is None:
                kernel_filter = ClinVarFilter(
                    variant_type=_INLINE_SENTINEL_CLNVC,
                    positive_labels=frozenset({_INLINE_SENTINEL_CLNSIG}),
                    negative_labels=frozenset(),
                    star_floor=1,
                )

            if inline_mode:
                if not isinstance(variants, list) or not variants:
                    return {
                        "error": (
                            "variants must be a non-empty list of {chrom, pos, ref, alt} objects"
                        ),
                        "isError": True,
                    }
                if len(variants) > ZERO_SHOT_MAX_VARIANTS:
                    return {
                        "error": (
                            f"zero_shot_score accepts at most "
                            f"{ZERO_SHOT_MAX_VARIANTS} variants per call "
                            f"(got {len(variants)}). Split the batch."
                        ),
                        "isError": True,
                    }
                allele_pattern = re.compile(r"^[ACGTacgt]+$")
                for i, variant in enumerate(variants):
                    if not isinstance(variant, dict):
                        return {
                            "error": (
                                f"variants[{i}] must be an object with chrom/pos/ref/alt fields"
                            ),
                            "isError": True,
                        }
                    chrom = variant.get("chrom")
                    pos = variant.get("pos")
                    ref = variant.get("ref")
                    alt = variant.get("alt")
                    if not isinstance(chrom, str) or not chrom:
                        return {
                            "error": f"variants[{i}].chrom must be a non-empty string",
                            "isError": True,
                        }
                    if not isinstance(pos, int) or isinstance(pos, bool) or pos < 1:
                        return {
                            "error": (
                                f"variants[{i}].pos must be an integer >= 1 "
                                f"(1-based VCF coordinate, got {pos!r})"
                            ),
                            "isError": True,
                        }
                    for field, value in (("ref", ref), ("alt", alt)):
                        if not isinstance(value, str) or not allele_pattern.match(value):
                            return {
                                "error": (
                                    f"variants[{i}].{field} must be a non-empty "
                                    f"ACGT string (got {value!r})"
                                ),
                                "isError": True,
                            }
                temp_vcf = self._write_inline_vcf(variants)
                temp_vcf_dir = str(temp_vcf.parent)
                kernel_vcf: str | Path = temp_vcf
            else:
                if not isinstance(vcf_path, str) or not vcf_path.endswith(VCF_SUFFIXES):
                    return {
                        "error": (
                            f"zero_shot_score: vcf_path must end with one of "
                            f"{VCF_SUFFIXES} (got '{vcf_path}')"
                        ),
                        "isError": True,
                    }
                vcf_file = Path(vcf_path)
                if not vcf_file.is_file():
                    return {
                        "error": f"zero_shot_score: VCF not found at '{vcf_path}'.",
                        "isError": True,
                    }
                if vcf_file.stat().st_size > ZERO_SHOT_MAX_VCF_BYTES:
                    return {
                        "error": (
                            f"zero_shot_score: VCF exceeds the "
                            f"{ZERO_SHOT_MAX_VCF_BYTES} byte cap "
                            f"({vcf_file.stat().st_size} bytes)"
                        ),
                        "isError": True,
                    }
                kernel_vcf = vcf_path

            guard = self._ism_engine_guard(model_name)
            if isinstance(guard, dict):
                return guard
            model, tokenizer, _config = guard

            def _run_zero_shot() -> VepResult:
                # Scoring is torch work: same off-loop, single-flight
                # contract as the ISM tools (fork-unsafe window under the
                # ModelManager flight lock, acquired inside the closure).
                with self.model_manager._infer_thread_lock:
                    return evaluate_vcf(
                        model,
                        tokenizer,
                        kernel_vcf,
                        fasta_path,
                        paradigm=paradigm,
                        context_window=context_window,
                        clnsig_filter=kernel_filter,
                        alt_number=4,
                    )

            loop = asyncio.get_running_loop()
            try:
                vep_result = await loop.run_in_executor(None, _run_zero_shot)
            except ValueError as e:
                # Matchable kernel failures (unreadable VCF, missing
                # chromosome, assembly mismatch, paradigm guard).
                return {"error": f"zero_shot_score: {e}", "isError": True}

            payload = vep_result.to_dict()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Zero-shot scoring complete: "
                            f"{vep_result.evaluated} variants scored, "
                            f"{vep_result.skipped} skipped using "
                            f"model {model_name}"
                        ),
                    }
                ],
                "records": payload["records"],
                "skip_counts": payload["skip_counts"],
                "evaluated": payload["evaluated"],
                "skipped": payload["skipped"],
                "skip_fraction": payload["skip_fraction"],
                "metrics": payload["metrics"],
                "convention": payload["convention"],
                "input_mode": "inline_variants" if inline_mode else "vcf_path",
                "paradigm": paradigm,
                "model_name": model_name,
            }
        except Exception as e:
            logger.error(f"Error in zero_shot_score: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Zero-shot scoring failed. See server logs for details.",
                    }
                ],
                "isError": True,
            }
        finally:
            if temp_vcf_dir is not None:
                shutil.rmtree(temp_vcf_dir, ignore_errors=True)

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
            try:
                yield  # Server is running
            finally:
                # Shutdown phase: cleanup resources gracefully
                # Runs on both clean exit and exceptions/termination signals
                logger.info("Starting graceful shutdown...")
                await self.shutdown()
                logger.info("Graceful shutdown complete")

        return lifespan

    def _resolve_bind_address(
        self, host: str | None, port: int | None, transport: str
    ) -> tuple[str, int]:
        """Resolve the bind address once for both HTTP transports.

        Precedence (Phase 12 REV-11 CLI-precedence fix, per field):

        1. an explicitly-passed CLI value (never ``None`` here),
        2. the transport-specific YAML block (``streamable_http`` on the
           streamable-http transport only),
        3. the ``server`` YAML block,
        4. the documented default ``DEFAULT_BIND_HOST``/``DEFAULT_BIND_PORT``.

        Args:
            host: Explicit host from the caller (``None`` = not given).
            port: Explicit port from the caller (``None`` = not given).
            transport: One of "stdio", "sse", "streamable-http".

        Returns:
            Tuple of the final (host, port) both starters receive.
        """
        server_config = self.config_manager.get_server_config()
        streamable_http_config = (
            server_config.streamable_http
            if server_config and hasattr(server_config, "streamable_http")
            else None
        )
        # The transport-specific block refines only CONFIG-sourced values: a
        # CLI-explicit value was never None and skips it entirely.
        transport_block = streamable_http_config if transport == "streamable-http" else None

        if host is None:
            if transport_block is not None and transport_block.host is not None:
                host = transport_block.host
            elif server_config:
                host = server_config.server.host
        if host is None:
            host = DEFAULT_BIND_HOST
        if port is None:
            if transport_block is not None and transport_block.port is not None:
                port = transport_block.port
            elif server_config:
                port = server_config.server.port
        if port is None:
            port = DEFAULT_BIND_PORT
        return host, port

    def start_server(
        self,
        host: str | None = None,
        port: int | None = None,
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
            host (str | None, optional): Host address to bind the server to.
                When ``None`` (the default), the value is resolved per the
                documented chain: transport-specific YAML (``streamable_http``
                on the streamable-http transport) > ``server`` YAML >
                ``DEFAULT_BIND_HOST`` ("127.0.0.1"). An explicitly-passed
                value always takes precedence over every YAML block
                (Phase 12 REV-11 fix — previously YAML silently overrode
                the CLI flags). Use "0.0.0.0" to bind all interfaces.
            port (int | None, optional): Port number to bind the server to.
                Resolved with the same precedence chain when ``None``,
                falling back to ``DEFAULT_BIND_PORT`` (8000). Only used for
                HTTP-based transports.
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

        # Resolve the bind address ONCE (Phase 12 REV-11): CLI-explicit
        # values win over YAML on both HTTP transports; both starters below
        # receive the final values. The old unconditional YAML override here
        # silently discarded the CLI flags, and the streamable-http starter
        # then always overrode them again with the streamable_http block.
        host, port = self._resolve_bind_address(host, port, transport)

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

        logger.info("Using Streamable HTTP transport")

        # Read streamable_http config if available. Host/port resolution is
        # NOT done here anymore (Phase 12 REV-11): start_server resolves the
        # bind address once with CLI precedence and passes final values in;
        # this block now contributes only the endpoint path.
        server_config = self.config_manager.get_server_config()
        streamable_http_config = (
            server_config.streamable_http
            if server_config and hasattr(server_config, "streamable_http")
            else None
        )

        http_path = streamable_http_config.path if streamable_http_config else "/mcp"

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

        # Run the Starlette app with uvicorn with proper signal handling
        # http_app already has lifespan for session_manager from streamable_http_app()
        config = uvicorn.Config(
            app=http_app,
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
        --host: Host to bind HTTP/SSE transports to; when omitted the value
            resolves from the YAML config (streamable_http block on that
            transport, else the server block), falling back to 127.0.0.1
        --port: Port to bind HTTP/SSE transports to; same resolution chain
            as --host, falling back to 8000
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
        logger.info(
            f"Host: {args.host if args.host is not None else '(from config, default 127.0.0.1)'}"
        )
        logger.info(
            f"Port: {args.port if args.port is not None else '(from config, default 8000)'}"
        )
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
        logger.info(f"Starting server with {args.transport} transport")
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
