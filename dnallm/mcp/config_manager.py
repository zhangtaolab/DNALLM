"""
Configuration Manager for MCP Server

This module handles loading and validation of MCP server configurations.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """服务器配置模型"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    cors_origins: List[str] = ["*"]


class MCPConfig(BaseModel):
    """MCP 配置模型"""
    name: str = "DNALLM MCP Server"
    version: str = "1.0.0"
    description: str = "DNA sequence prediction server using MCP protocol"


class ModelConfig(BaseModel):
    """模型配置模型"""
    name: str
    model_name: str
    config_path: str
    enabled: bool = True
    max_concurrent_requests: int = 10
    task_type: str
    description: str
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        allowed_types = ['binary', 'multiclass', 'multilabel', 'regression']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of {allowed_types}')
        return v


class MultiModelConfig(BaseModel):
    """多模型配置模型"""
    enabled: bool = True
    max_parallel_models: int = 8
    default_model_sets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class SSEConfig(BaseModel):
    """SSE 配置模型"""
    heartbeat_interval: int = 30
    max_connections: int = 100
    buffer_size: int = 1000


class LoggingConfig(BaseModel):
    """日志配置模型"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/mcp_server.log"


class MCPServerConfig(BaseModel):
    """MCP 服务器完整配置模型"""
    server: ServerConfig
    mcp: MCPConfig
    models: List[ModelConfig]
    multi_model: MultiModelConfig
    sse: SSEConfig
    logging: LoggingConfig


class InferenceConfig(BaseModel):
    """推理配置模型"""
    task: Dict[str, Any]
    inference: Dict[str, Any]
    model: Dict[str, Any]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.mcp_config: Optional[MCPServerConfig] = None
        self.inference_configs: Dict[str, InferenceConfig] = {}
    
    def load_mcp_config(self, config_path: str) -> MCPServerConfig:
        """加载 MCP 服务器配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 验证配置
            self.mcp_config = MCPServerConfig(**config_data)
            logger.info(f"Loaded MCP config from {config_path}")
            return self.mcp_config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Invalid config format: {e}")
    
    def load_inference_config(self, config_path: str) -> InferenceConfig:
        """加载推理配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 验证配置
            inference_config = InferenceConfig(**config_data)
            logger.info(f"Loaded inference config from {config_path}")
            return inference_config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Inference config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Invalid inference config format: {e}")
    
    def load_all_inference_configs(self) -> Dict[str, InferenceConfig]:
        """加载所有模型的推理配置"""
        if not self.mcp_config:
            raise ValueError("MCP config must be loaded first")
        
        self.inference_configs = {}
        
        for model_config in self.mcp_config.models:
            if model_config.enabled:
                try:
                    inference_config = self.load_inference_config(model_config.config_path)
                    self.inference_configs[model_config.name] = inference_config
                except Exception as e:
                    logger.warning(f"Failed to load inference config for {model_config.name}: {e}")
        
        logger.info(f"Loaded {len(self.inference_configs)} inference configs")
        return self.inference_configs
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if not self.mcp_config:
            return None
        
        for model_config in self.mcp_config.models:
            if model_config.name == model_name or model_config.model_name == model_name:
                return model_config
        return None
    
    def get_inference_config(self, model_name: str) -> Optional[InferenceConfig]:
        """获取推理配置"""
        return self.inference_configs.get(model_name)
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """获取启用的模型列表"""
        if not self.mcp_config:
            return []
        
        return [model for model in self.mcp_config.models if model.enabled]
    
    def get_models_by_task_type(self, task_type: str) -> List[ModelConfig]:
        """根据任务类型获取模型列表"""
        if not self.mcp_config:
            return []
        
        return [model for model in self.mcp_config.models 
                if model.enabled and model.task_type == task_type]
    
    def get_model_set_config(self, set_name: str) -> Optional[Dict[str, Any]]:
        """获取模型集配置"""
        if not self.mcp_config:
            return None
        
        return self.mcp_config.multi_model.default_model_sets.get(set_name)
    
    def validate_config_paths(self) -> List[str]:
        """验证配置文件路径是否存在"""
        if not self.mcp_config:
            return []
        
        missing_paths = []
        
        for model_config in self.mcp_config.models:
            if model_config.enabled:
                if not os.path.exists(model_config.config_path):
                    missing_paths.append(model_config.config_path)
        
        return missing_paths
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        if not self.mcp_config:
            return {}
        
        return {
            "name": self.mcp_config.mcp.name,
            "version": self.mcp_config.mcp.version,
            "description": self.mcp_config.mcp.description,
            "host": self.mcp_config.server.host,
            "port": self.mcp_config.server.port,
            "enabled_models": len(self.get_enabled_models()),
            "total_models": len(self.mcp_config.models)
        }
    
    def get_model_capabilities(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型能力信息"""
        model_config = self.get_model_config(model_name)
        inference_config = self.get_inference_config(model_name)
        
        if not model_config or not inference_config:
            return None
        
        return {
            "model_name": model_config.model_name,
            "task_type": model_config.task_type,
            "description": model_config.description,
            "max_concurrent_requests": model_config.max_concurrent_requests,
            "enabled": model_config.enabled,
            "task_info": inference_config.model.get("task_info", {}),
            "inference_config": inference_config.inference
        }
    
    def reload_config(self, config_path: str) -> MCPServerConfig:
        """重新加载配置"""
        logger.info(f"Reloading config from {config_path}")
        return self.load_mcp_config(config_path)
    
    def save_config(self, config_path: str, config: MCPServerConfig) -> None:
        """保存配置到文件"""
        config_dict = config.dict()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"Config saved to {config_path}")


def create_default_config() -> MCPServerConfig:
    """创建默认配置"""
    return MCPServerConfig(
        server=ServerConfig(),
        mcp=MCPConfig(),
        models=[],
        multi_model=MultiModelConfig(),
        sse=SSEConfig(),
        logging=LoggingConfig()
    )


def validate_config_file(config_path: str) -> bool:
    """验证配置文件是否有效"""
    try:
        manager = ConfigManager()
        manager.load_mcp_config(config_path)
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False
