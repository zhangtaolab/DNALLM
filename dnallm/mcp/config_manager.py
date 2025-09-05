"""Configuration Manager for MCP Server.

This module provides configuration management functionality for the MCP server,
including loading, validating, and managing both main server configurations
and individual model configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from .config_validators import (
    MCPServerConfig,
    InferenceModelConfig,
    validate_mcp_server_config,
    validate_inference_model_config
)


class MCPConfigManager:
    """Manages MCP server configurations and model configurations."""
    
    def __init__(self, config_dir: str, server_config_file: str = "mcp_server_config.yaml"):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Path to the configuration directory
            server_config_file: Name of the server configuration file
        """
        self.config_dir = Path(config_dir)
        self.server_config_path = self.config_dir / server_config_file
        self.server_config: Optional[MCPServerConfig] = None
        self.model_configs: Dict[str, InferenceModelConfig] = {}
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load and validate all configurations."""
        if self.server_config_path.exists():
            self._load_server_config()
            self._load_model_configs()
        else:
            logger.warning(f"Configuration file not found: {self.server_config_path}")
    
    def _load_server_config(self) -> None:
        """Load and validate the main server configuration."""
        try:
            self.server_config = validate_mcp_server_config(str(self.server_config_path))
            logger.info(f"Successfully loaded server configuration from {self.server_config_path}")
        except Exception as e:
            logger.error(f"Failed to load server configuration: {e}")
            raise
    
    def _load_model_configs(self) -> None:
        """Load and validate all model configurations."""
        if not self.server_config:
            logger.error("Server configuration not loaded")
            return
        
        for model_name, model_entry in self.server_config.models.items():
            if not model_entry.enabled:
                logger.info(f"Skipping disabled model: {model_name}")
                continue
            
            try:
                config_path = Path(model_entry.config_path)
                if not config_path.is_absolute():
                    # Make path relative to the config directory
                    config_path = self.config_dir / config_path
                
                model_config = validate_inference_model_config(str(config_path))
                self.model_configs[model_name] = model_config
                logger.info(f"Successfully loaded model configuration: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model configuration for {model_name}: {e}")
                # Continue loading other models even if one fails
    
    def get_server_config(self) -> Optional[MCPServerConfig]:
        """Get the main server configuration.
        
        Returns:
            MCPServerConfig object or None if not loaded
        """
        return self.server_config
    
    def get_model_config(self, model_name: str) -> Optional[InferenceModelConfig]:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            InferenceModelConfig object or None if not found
        """
        return self.model_configs.get(model_name)
    
    def get_model_configs(self) -> Dict[str, InferenceModelConfig]:
        """Get all loaded model configurations.
        
        Returns:
            Dictionary of model configurations
        """
        return self.model_configs.copy()
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names.
        
        Returns:
            List of enabled model names
        """
        if not self.server_config:
            return []
        
        return [
            name for name, entry in self.server_config.models.items()
            if entry.enabled
        ]
    
    def get_model_priority(self, model_name: str) -> int:
        """Get priority of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Priority value (1-10, higher is more important)
        """
        if not self.server_config or model_name not in self.server_config.models:
            return 1
        
        return self.server_config.models[model_name].priority
    
    def get_multi_model_configs(self) -> Dict[str, Any]:
        """Get multi-model parallel prediction configurations.
        
        Returns:
            Dictionary of multi-model configurations
        """
        if not self.server_config:
            return {}
        
        return {
            name: {
                "name": config.name,
                "description": config.description,
                "models": config.models,
                "enabled": config.enabled
            }
            for name, config in self.server_config.multi_model.items()
        }
    
    def get_sse_config(self) -> Dict[str, Any]:
        """Get SSE configuration.
        
        Returns:
            Dictionary of SSE configuration parameters
        """
        if not self.server_config:
            return {}
        
        sse_config = self.server_config.sse
        return {
            "heartbeat_interval": sse_config.heartbeat_interval,
            "max_connections": sse_config.max_connections,
            "connection_timeout": sse_config.connection_timeout,
            "enable_compression": sse_config.enable_compression
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Dictionary of logging configuration parameters
        """
        if not self.server_config:
            return {}
        
        logging_config = self.server_config.logging
        return {
            "level": logging_config.level,
            "format": logging_config.format,
            "file": logging_config.file,
            "max_size": logging_config.max_size,
            "backup_count": logging_config.backup_count
        }
    
    def reload_configurations(self) -> None:
        """Reload all configurations from files."""
        logger.info("Reloading configurations...")
        self.model_configs.clear()
        self._load_configurations()
        logger.info("Configurations reloaded successfully")
    
    def validate_model_references(self) -> List[str]:
        """Validate that all model references are valid.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.server_config:
            errors.append("Server configuration not loaded")
            return errors
        
        # Check that all multi-model configurations reference existing models
        available_models = set(self.server_config.models.keys())
        
        for multi_name, multi_config in self.server_config.multi_model.items():
            for model_name in multi_config.models:
                if model_name not in available_models:
                    errors.append(
                        f"Multi-model config '{multi_name}' references non-existent model '{model_name}'"
                    )
        
        # Check that all model config files exist and are valid
        for model_name, model_entry in self.server_config.models.items():
            if model_name not in self.model_configs:
                errors.append(f"Model configuration not loaded for '{model_name}'")
        
        return errors
    
    def get_model_info_summary(self) -> Dict[str, Any]:
        """Get summary information about all loaded models.
        
        Returns:
            Dictionary containing model information summary
        """
        summary = {
            "total_models": len(self.model_configs),
            "enabled_models": len(self.get_enabled_models()),
            "models": {}
        }
        
        for model_name, model_config in self.model_configs.items():
            summary["models"][model_name] = {
                "task_type": model_config.task.task_type,
                "num_labels": model_config.task.num_labels,
                "label_names": model_config.task.label_names,
                "model_path": model_config.model.path,
                "model_source": model_config.model.source,
                "architecture": model_config.model.task_info.architecture,
                "tokenizer": model_config.model.task_info.tokenizer,
                "species": model_config.model.task_info.species,
                "task_category": model_config.model.task_info.task_category
            }
        
        return summary
