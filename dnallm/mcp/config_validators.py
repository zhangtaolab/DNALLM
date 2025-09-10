"""Configuration validators for MCP Server.

This module provides Pydantic models for validating MCP server configurations
and inference model configurations.
"""

from pydantic import BaseModel, Field, field_validator


class TaskConfig(BaseModel):
    """Task configuration for DNA prediction models."""

    task_type: str = Field(
        ..., pattern="^(binary|multiclass|multilabel|regression)$"
    )
    num_labels: int = Field(..., ge=1)
    label_names: list[str] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    description: str = Field(..., min_length=1)


class InferenceConfig(BaseModel):
    """Inference configuration for model prediction."""

    batch_size: int = Field(16, ge=1, le=128)
    max_length: int = Field(512, ge=64, le=4096)
    device: str = Field("auto", pattern="^(auto|cpu|cuda|mps)$")
    num_workers: int = Field(4, ge=1, le=16)
    precision: str = Field("float16", pattern="^(float16|float32|bfloat16)$")
    use_fp16: bool = Field(False)  # Whether to use half precision
    output_dir: str = Field(..., min_length=1)
    save_predictions: bool = Field(True)
    save_hidden_states: bool = Field(False)
    save_attentions: bool = Field(False)

    @field_validator("use_fp16", mode="before")
    @classmethod
    def set_use_fp16(cls, v, info):  # noqa: N805
        """Set use_fp16 based on precision setting."""
        if info.data and "precision" in info.data:
            return info.data["precision"] == "float16"
        return v


class ModelInfoConfig(BaseModel):
    """Model information configuration."""

    architecture: str = Field(..., min_length=1)
    tokenizer: str = Field(..., min_length=1)
    species: str = Field(..., min_length=1)
    task_category: str = Field(..., min_length=1)
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Individual model configuration."""

    name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    source: str = Field("modelscope", pattern="^(huggingface|modelscope)$")
    task_info: ModelInfoConfig


class InferenceModelConfig(BaseModel):
    """Complete inference model configuration."""

    task: TaskConfig
    inference: InferenceConfig
    model: ModelConfig

    @field_validator("task")
    @classmethod
    def validate_task_labels(cls, v):  # noqa: N805
        """Validate that num_labels matches label_names length."""
        if v.num_labels != len(v.label_names):
            raise ValueError("num_labels must match the length of label_names")
        return v


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field("0.0.0.0", pattern="^[0-9.]+$")  # noqa: S104
    port: int = Field(8000, ge=1024, le=65535)
    workers: int = Field(1, ge=1, le=16)
    log_level: str = Field(
        "INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    debug: bool = Field(False)


class MCPConfig(BaseModel):
    """MCP protocol configuration."""

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class ModelEntryConfig(BaseModel):
    """Model entry in the main configuration."""

    name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    config_path: str = Field(..., min_length=1)
    enabled: bool = Field(True)
    priority: int = Field(1, ge=1, le=10)

    @field_validator("config_path")
    @classmethod
    def validate_config_path(cls, v):  # noqa: N805
        """Validate config path format."""
        if not v or not v.strip():
            raise ValueError("Config path cannot be empty")
        return v


class MultiModelConfig(BaseModel):
    """Multi-model parallel prediction configuration."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=2)
    enabled: bool = Field(True)


class SSEConfig(BaseModel):
    """SSE (Server-Sent Events) configuration."""

    heartbeat_interval: int = Field(30, ge=5, le=300)
    max_connections: int = Field(100, ge=1, le=1000)
    connection_timeout: int = Field(300, ge=60, le=3600)
    enable_compression: bool = Field(True)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(..., min_length=1)
    file: str = Field(..., min_length=1)
    max_size: str = Field("10MB", pattern="^[0-9]+(MB|GB)$")
    backup_count: int = Field(5, ge=1, le=20)


class MCPServerConfig(BaseModel):
    """Complete MCP server configuration."""

    server: ServerConfig
    mcp: MCPConfig
    models: dict[str, ModelEntryConfig]
    multi_model: dict[str, MultiModelConfig]
    sse: SSEConfig
    logging: LoggingConfig

    @field_validator("models")
    @classmethod
    def validate_model_names(cls, v):  # noqa: N805
        """Validate that model names are unique."""
        names = [entry.name for entry in v.values()]
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique")
        return v

    @field_validator("multi_model")
    @classmethod
    def validate_multi_model_references(cls, v, info):  # noqa: N805
        """Validate that multi-model configurations reference existing models."""
        if info.data and "models" in info.data:
            available_models = set(info.data["models"].keys())
            for config in v.values():
                for model_name in config.models:
                    if model_name not in available_models:
                        raise ValueError(
                            f"Model '{model_name}' referenced in multi-model config but not defined in models"
                        )
        return v


def validate_mcp_server_config(config_path: str) -> MCPServerConfig:
    """Validate MCP server configuration file."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return MCPServerConfig(**config_dict)


def validate_inference_model_config(config_path: str) -> InferenceModelConfig:
    """Validate inference model configuration file."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return InferenceModelConfig(**config_dict)
