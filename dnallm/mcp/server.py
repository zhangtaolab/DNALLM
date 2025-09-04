"""DNALLM MCP Server Implementation.

This module implements the main MCP server using FastMCP framework
with SSE support for real-time DNA sequence prediction.
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger

# MCP SDK imports
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
# tool decorator is available as app.tool() method

from config_manager import MCPConfigManager
from model_manager import ModelManager


class DNALLMMCPServer:
    """DNALLM MCP Server implementation using FastMCP + SSE."""
    
    def __init__(self, config_path: str):
        """Initialize the MCP server.
        
        Args:
            config_path: Path to the main MCP server configuration file
        """
        self.config_path = config_path
        # Extract directory from config file path
        config_dir = Path(config_path).parent
        self.config_manager = MCPConfigManager(str(config_dir))
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
            instructions=server_config.mcp.description
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
        async def dna_sequence_predict(sequence: str, model_name: str) -> Dict[str, Any]:
            """Predict DNA sequence using a specific model.
            
            Args:
                sequence: DNA sequence to predict (A, T, G, C)
                model_name: Name of the model to use for prediction
                
            Returns:
                Dictionary containing prediction results
            """
            try:
                result = await self.model_manager.predict_sequence(model_name, sequence)
                if result is None:
                    return {
                        "error": f"Model {model_name} not available or prediction failed",
                        "isError": True
                    }
                
                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_name": model_name,
                    "sequence": sequence
                }
            except Exception as e:
                logger.error(f"Error in dna_sequence_predict: {e}")
                return {
                    "content": [{"type": "text", "text": f"Prediction error: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def dna_batch_predict(sequences: List[str], model_name: str) -> Dict[str, Any]:
            """Predict multiple DNA sequences using a specific model.
            
            Args:
                sequences: List of DNA sequences to predict
                model_name: Name of the model to use for prediction
                
            Returns:
                Dictionary containing batch prediction results
            """
            try:
                result = await self.model_manager.predict_batch(model_name, sequences)
                if result is None:
                    return {
                        "error": f"Model {model_name} not available or prediction failed",
                        "isError": True
                    }
                
                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_name": model_name,
                    "sequence_count": len(sequences)
                }
            except Exception as e:
                logger.error(f"Error in dna_batch_predict: {e}")
                return {
                    "content": [{"type": "text", "text": f"Batch prediction error: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def dna_multi_model_predict(sequence: str, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
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
                        "isError": True
                    }
                
                result = await self.model_manager.predict_multi_model(model_names, sequence)
                
                return {
                    "content": [{"type": "text", "text": str(result)}],
                    "model_count": len(model_names),
                    "sequence": sequence
                }
            except Exception as e:
                logger.error(f"Error in dna_multi_model_predict: {e}")
                return {
                    "content": [{"type": "text", "text": f"Multi-model prediction error: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def list_loaded_models() -> Dict[str, Any]:
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
                    "models": models_info
                }
            except Exception as e:
                logger.error(f"Error in list_loaded_models: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error listing models: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def get_model_info(model_name: str) -> Dict[str, Any]:
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
                        "isError": True
                    }
                
                return {
                    "content": [{"type": "text", "text": str(info)}],
                    "model_name": model_name,
                    "info": info
                }
            except Exception as e:
                logger.error(f"Error in get_model_info: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error getting model info: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def list_models_by_task_type(task_type: str) -> Dict[str, Any]:
            """List all available models filtered by task type.
            
            Args:
                task_type: Task type to filter by (binary, multiclass, regression, etc.)
                
            Returns:
                Dictionary containing filtered model information
            """
            try:
                all_models = self.model_manager.get_all_models_info()
                filtered_models = {
                    name: info for name, info in all_models.items()
                    if info.get("task_type") == task_type
                }
                
                return {
                    "content": [{"type": "text", "text": str(filtered_models)}],
                    "task_type": task_type,
                    "model_count": len(filtered_models),
                    "models": filtered_models
                }
            except Exception as e:
                logger.error(f"Error in list_models_by_task_type: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error filtering models: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def get_all_available_models() -> Dict[str, Any]:
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
                    "models": all_models
                }
            except Exception as e:
                logger.error(f"Error in get_all_available_models: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error getting available models: {str(e)}"}],
                    "isError": True
                }
        
        @self.app.tool()
        async def health_check() -> Dict[str, Any]:
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
                    "total_configured_models": len(self.config_manager.get_enabled_models()),
                    "server_name": server_config.mcp.name if server_config else "Unknown",
                    "server_version": server_config.mcp.version if server_config else "Unknown"
                }
                
                return {
                    "content": [{"type": "text", "text": str(health_status)}],
                    "health": health_status
                }
            except Exception as e:
                logger.error(f"Error in health_check: {e}")
                return {
                    "content": [{"type": "text", "text": f"Health check error: {str(e)}"}],
                    "isError": True
                }
        
        logger.info("Registered MCP tools successfully")
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the MCP server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")
        
        server_config = self.config_manager.get_server_config()
        if server_config:
            host = server_config.server.host
            port = server_config.server.port
        
        logger.info(f"Starting DNALLM MCP Server on {host}:{port}")
        
        # Start the FastMCP server
        # FastMCP.run() is a blocking call, not async
        # Note: FastMCP uses stdio transport by default for MCP protocol
        self.app.run()
    
    def get_server_info(self) -> Dict[str, Any]:
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
            "initialized": self._initialized
        }
    
    async def shutdown(self) -> None:
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down DNALLM MCP Server...")
        
        # Unload all models
        unloaded_count = self.model_manager.unload_all_models()
        logger.info(f"Unloaded {unloaded_count} models during shutdown")
        
        self._initialized = False
        logger.info("DNALLM MCP Server shutdown complete")
