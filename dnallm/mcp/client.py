"""DNALLM MCP Client SDK.

This module provides a Python client SDK for interacting with the DNALLM MCP
server. It supports both SSE and stdio transports, and provides typed methods
for all server tools with both sync and async variants.

Example:
    Sync usage::

        client = DNALLMMCPClient(transport="sse", url="http://localhost:8000")
        result = client.dna_sequence_predict("ATCGATCG", "dnabert-2")

    Async usage::

        client = DNALLMMCPClient(transport="stdio", command="dnallm-mcp-server")
        result = await client.adna_sequence_predict("ATCGATCG", "dnabert-2")

    Generic fallback::

        result = client.call("dna_sequence_predict", {
            "sequence": "ATCGATCG",
            "model_name": "dnabert-2"
        })
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp import ClientSession


class DNALLMMCPClient:
    """Client for the DNALLM MCP server.

    Provides typed methods for all DNA prediction and analysis tools,
    with both synchronous and asynchronous APIs. Supports SSE and stdio
    transports.

    Args:
        transport: Transport protocol to use. Either "sse" or "stdio".
        url: SSE endpoint URL. Defaults to "http://localhost:8000".
            Only used when transport is "sse".
        command: Command to spawn the stdio server. Defaults to
            "dnallm-mcp-server". Only used when transport is "stdio".
        args: Additional arguments to pass to the stdio command.
        env: Environment variables for the stdio server process.

    Raises:
        ValueError: If transport is not "sse" or "stdio".
        ConnectionError: If the server connection fails.
        RuntimeError: If called from an async context without using
            the async variant.
    """

    def __init__(
        self,
        transport: Literal["sse", "stdio"] = "stdio",
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if transport not in ("sse", "stdio"):
            raise ValueError(
                f"Invalid transport: {transport!r}. "
                'Must be "sse" or "stdio".'
            )
        self.transport = transport
        self.url = url or (
            "http://localhost:8000" if transport == "sse" else None
        )
        self.command = command or (
            "dnallm-mcp-server" if transport == "stdio" else None
        )
        self.args = args or []
        self.env = env or {}

    @asynccontextmanager
    async def _connect(self):
        """Establish a connection to the MCP server.

        Yields an initialized ClientSession ready for tool calls.

        Yields:
            ClientSession: An initialized MCP client session.
        """
        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(self.url) as (read_stream, write_stream):
                from mcp import ClientSession

                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:
                    await session.initialize()
                    yield session
        else:
            from mcp.client.stdio import stdio_client, StdioServerParameters

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )
            async with stdio_client(server_params) as (
                read_stream,
                write_stream,
            ):
                from mcp import ClientSession

                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:
                    await session.initialize()
                    yield session

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool and parse the result.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dictionary.

        Returns:
            Parsed tool result as a dictionary.
        """
        async with self._connect() as session:
            result = await session.call_tool(tool_name, arguments)
            return self._parse_result(result)

    @staticmethod
    def _parse_result(result: Any) -> dict:
        """Parse a CallToolResult into a dictionary.

        Args:
            result: CallToolResult from the MCP session.

        Returns:
            Parsed result dictionary.
        """
        if result.isError:
            text = result.content[0].text if result.content else "Unknown error"
            try:
                return json.loads(text)
            except (json.JSONDecodeError, IndexError):
                return {"error": text, "isError": True}
        if not result.content:
            return {}
        text = result.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def call(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool by name (sync).

        Generic fallback for calling any MCP tool by name.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dictionary.

        Returns:
            Tool result as a dictionary.

        Raises:
            RuntimeError: If called from an async context.
        """
        return self._run_async(self.acall(tool_name, arguments))

    async def acall(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool by name (async).

        Generic fallback for calling any MCP tool by name.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dictionary.

        Returns:
            Tool result as a dictionary.
        """
        return await self._call_tool(tool_name, arguments)

    # ------------------------------------------------------------------
    # Internal helper for sync/async bridging
    # ------------------------------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync code.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine result.

        Raises:
            RuntimeError: If called from an async context.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "Cannot call sync method from an async context. "
            "Use the async variant (prefix with 'a') instead."
        )

    # ------------------------------------------------------------------
    # 1. dna_sequence_predict
    # ------------------------------------------------------------------

    def dna_sequence_predict(self, sequence: str, model_name: str) -> dict:
        """Predict DNA sequence using a specific model (sync)."""
        return self._run_async(
            self.adna_sequence_predict(sequence, model_name)
        )

    async def adna_sequence_predict(
        self, sequence: str, model_name: str
    ) -> dict:
        """Predict DNA sequence using a specific model (async)."""
        return await self._call_tool(
            "dna_sequence_predict",
            {"sequence": sequence, "model_name": model_name},
        )

    # ------------------------------------------------------------------
    # 2. dna_batch_predict
    # ------------------------------------------------------------------

    def dna_batch_predict(
        self, sequences: list[str], model_name: str
    ) -> dict:
        """Predict multiple DNA sequences using a specific model (sync)."""
        return self._run_async(
            self.adna_batch_predict(sequences, model_name)
        )

    async def adna_batch_predict(
        self, sequences: list[str], model_name: str
    ) -> dict:
        """Predict multiple DNA sequences using a specific model (async)."""
        return await self._call_tool(
            "dna_batch_predict",
            {"sequences": sequences, "model_name": model_name},
        )

    # ------------------------------------------------------------------
    # 3. dna_multi_model_predict
    # ------------------------------------------------------------------

    def dna_multi_model_predict(
        self, sequence: str, model_names: list[str] | None = None
    ) -> dict:
        """Predict DNA sequence using multiple models (sync)."""
        return self._run_async(
            self.adna_multi_model_predict(sequence, model_names)
        )

    async def adna_multi_model_predict(
        self, sequence: str, model_names: list[str] | None = None
    ) -> dict:
        """Predict DNA sequence using multiple models (async)."""
        return await self._call_tool(
            "dna_multi_model_predict",
            {"sequence": sequence, "model_names": model_names},
        )

    # ------------------------------------------------------------------
    # 4. dna_stream_predict
    # ------------------------------------------------------------------

    def dna_stream_predict(
        self,
        sequence: str,
        model_name: str,
        stream_progress: bool = True,
    ) -> dict:
        """Stream DNA sequence prediction with progress updates (sync)."""
        return self._run_async(
            self.adna_stream_predict(sequence, model_name, stream_progress)
        )

    async def adna_stream_predict(
        self,
        sequence: str,
        model_name: str,
        stream_progress: bool = True,
    ) -> dict:
        """Stream DNA sequence prediction with progress updates (async)."""
        return await self._call_tool(
            "dna_stream_predict",
            {
                "sequence": sequence,
                "model_name": model_name,
                "stream_progress": stream_progress,
            },
        )

    # ------------------------------------------------------------------
    # 5. dna_stream_batch_predict
    # ------------------------------------------------------------------

    def dna_stream_batch_predict(
        self,
        sequences: list[str],
        model_name: str,
        stream_progress: bool = True,
    ) -> dict:
        """Stream batch DNA sequence prediction (sync)."""
        return self._run_async(
            self.adna_stream_batch_predict(
                sequences, model_name, stream_progress
            )
        )

    async def adna_stream_batch_predict(
        self,
        sequences: list[str],
        model_name: str,
        stream_progress: bool = True,
    ) -> dict:
        """Stream batch DNA sequence prediction (async)."""
        return await self._call_tool(
            "dna_stream_batch_predict",
            {
                "sequences": sequences,
                "model_name": model_name,
                "stream_progress": stream_progress,
            },
        )

    # ------------------------------------------------------------------
    # 6. dna_stream_multi_model_predict
    # ------------------------------------------------------------------

    def dna_stream_multi_model_predict(
        self,
        sequence: str,
        model_names: list[str] | None = None,
        stream_progress: bool = True,
    ) -> dict:
        """Stream multi-model DNA sequence prediction (sync)."""
        return self._run_async(
            self.adna_stream_multi_model_predict(
                sequence, model_names, stream_progress
            )
        )

    async def adna_stream_multi_model_predict(
        self,
        sequence: str,
        model_names: list[str] | None = None,
        stream_progress: bool = True,
    ) -> dict:
        """Stream multi-model DNA sequence prediction (async)."""
        return await self._call_tool(
            "dna_stream_multi_model_predict",
            {
                "sequence": sequence,
                "model_names": model_names,
                "stream_progress": stream_progress,
            },
        )

    # ------------------------------------------------------------------
    # 7. dna_mutagenesis
    # ------------------------------------------------------------------

    def dna_mutagenesis(
        self,
        sequence: str | None = None,
        sequences: list[str] | None = None,
        mutation_type: str = "single_base_substitution",
        positions: list[int] | None = None,
        model_name: str = "",
    ) -> dict:
        """Perform in silico mutagenesis on DNA sequences (sync)."""
        return self._run_async(
            self.adna_mutagenesis(
                sequence, sequences, mutation_type, positions, model_name
            )
        )

    async def adna_mutagenesis(
        self,
        sequence: str | None = None,
        sequences: list[str] | None = None,
        mutation_type: str = "single_base_substitution",
        positions: list[int] | None = None,
        model_name: str = "",
    ) -> dict:
        """Perform in silico mutagenesis on DNA sequences (async)."""
        return await self._call_tool(
            "dna_mutagenesis",
            {
                "sequence": sequence,
                "sequences": sequences,
                "mutation_type": mutation_type,
                "positions": positions,
                "model_name": model_name,
            },
        )

    # ------------------------------------------------------------------
    # 8. dna_interpret
    # ------------------------------------------------------------------

    def dna_interpret(
        self,
        sequence: str,
        model_name: str,
        method: str = "lig",
        target_class: int | None = None,
        max_length: int | None = None,
    ) -> dict:
        """Interpret model predictions using attribution methods (sync)."""
        return self._run_async(
            self.adna_interpret(
                sequence, model_name, method, target_class, max_length
            )
        )

    async def adna_interpret(
        self,
        sequence: str,
        model_name: str,
        method: str = "lig",
        target_class: int | None = None,
        max_length: int | None = None,
    ) -> dict:
        """Interpret model predictions using attribution methods (async)."""
        return await self._call_tool(
            "dna_interpret",
            {
                "sequence": sequence,
                "model_name": model_name,
                "method": method,
                "target_class": target_class,
                "max_length": max_length,
            },
        )

    # ------------------------------------------------------------------
    # 9. list_loaded_models
    # ------------------------------------------------------------------

    def list_loaded_models(self) -> dict:
        """List all currently loaded models (sync)."""
        return self._run_async(self.alist_loaded_models())

    async def alist_loaded_models(self) -> dict:
        """List all currently loaded models (async)."""
        return await self._call_tool("list_loaded_models", {})

    # ------------------------------------------------------------------
    # 10. get_model_info
    # ------------------------------------------------------------------

    def get_model_info(self, model_name: str) -> dict:
        """Get detailed information about a specific model (sync)."""
        return self._run_async(self.aget_model_info(model_name))

    async def aget_model_info(self, model_name: str) -> dict:
        """Get detailed information about a specific model (async)."""
        return await self._call_tool(
            "get_model_info", {"model_name": model_name}
        )

    # ------------------------------------------------------------------
    # 11. list_models_by_task_type
    # ------------------------------------------------------------------

    def list_models_by_task_type(self, task_type: str) -> dict:
        """List all available models filtered by task type (sync)."""
        return self._run_async(
            self.alist_models_by_task_type(task_type)
        )

    async def alist_models_by_task_type(self, task_type: str) -> dict:
        """List all available models filtered by task type (async)."""
        return await self._call_tool(
            "list_models_by_task_type", {"task_type": task_type}
        )

    # ------------------------------------------------------------------
    # 12. get_all_available_models
    # ------------------------------------------------------------------

    def get_all_available_models(self) -> dict:
        """Get information about all available models (sync)."""
        return self._run_async(self.aget_all_available_models())

    async def aget_all_available_models(self) -> dict:
        """Get information about all available models (async)."""
        return await self._call_tool("get_all_available_models", {})

    # ------------------------------------------------------------------
    # 13. health_check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Perform health check on the MCP server (sync)."""
        return self._run_async(self.ahealth_check())

    async def ahealth_check(self) -> dict:
        """Perform health check on the MCP server (async)."""
        return await self._call_tool("health_check", {})
