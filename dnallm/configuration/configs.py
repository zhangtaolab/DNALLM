import os
import yaml
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class HeadConfig(BaseModel):
    """Configuration for the classification/regression head"""

    head: str = Field(
        default="mlp",
        description="Type of head to use",
    )
    frozen: bool = Field(
        default=False,
        description=("Whether to freeze the model except the head during training."),
    )
    task_type: str = Field(default="binary", description="Task type (default is binary)")
    hidden_dims: list[int] | None = Field(
        default=None,
        description=(
            "List of hidden layer dimensions for the head MLP. If None, "
            "a single linear layer is used."
        ),
    )
    activation_fn: str = Field(
        default="relu",
        description=(
            "Activation function to use in the head MLP. Options include "
            "'relu', 'gelu', 'silu', 'tanh', 'sigmoid'."
        ),
    )
    use_normalization: bool = Field(
        default=True,
        description="Whether to use normalization layers in the head MLP.",
    )
    norm_type: str = Field(
        default="layernorm",
        pattern="^(layernorm|batchnorm)$",
        description="Type of normalization to use: 'layernorm' or 'batchnorm'.",
    )
    num_filters: int = Field(
        default=128,
        description=(
            "Number of filters for convolutional layers in the head. "
            "If None, no convolutional layers are used."
        ),
    )
    kernel_sizes: list[int] | None = Field(
        default=None,
        description=(
            "List of kernel sizes for convolutional layers in the head. "
            "Must match the length of num_filters. If None, no convolutional "
            "layers are used."
        ),
    )
    hidden_size: int = Field(default=256, description="Hidden size for RNN heads.")
    num_layers: int = Field(default=2, description="Number of layers for RNN heads.")
    bidirectional: bool = Field(default=True, description="Whether to use bidirectional RNNs.")
    initial_filters: int | None = Field(
        default=None, description="Initial number of filters for U-Net."
    )
    dropout: float = Field(default=0.1, description="Dropout probability for the head.")
    pooling_strategy: str | None = Field(
        default=None,
        description=(
            "Pooling strategy to convert token embeddings to a single "
            "vector. Options: 'mean', 'max', 'cls', 'last'."
        ),
    )
    embedding_dims: list[int] | None = Field(
        default=None,
        description=("List of embedding dimensions for model with multi-scale features."),
    )
    custom_head: Any | None = (
        Field(
            default=None,
            description=(
                "Custom head class. If provided, this will override other "
                "head configuration parameters."
            ),
        ),
    )
    loss_function: str | None = Field(
        default=None,
        description=(
            "Custom loss function for the head. If None, default loss "
            "functions will be used based on the task."
        ),
    )
    loss_fn_kwargs: dict | None = Field(
        default=None,
        description=("Additional keyword arguments for the custom loss function."),
    )


class TaskConfig(BaseModel):
    """Configuration for different fine-tuning tasks"""

    task_type: str = Field(
        ...,
        pattern="^(embedding|mask|generation|binary|binary_classification|multiclass|multi_class_classification|multilabel|multi_label_classification|regression|token|token_classification)$",
    )
    num_labels: int | None = Field(default=2, description="Number of labels (default 2)")
    label_names: list[str] | None = None
    threshold: float = Field(default=0.5, description="Threshold for binary/multilabel tasks")
    mlm_probability: float | None = Field(
        default=0.15, description="Masking probability for MLM tasks"
    )
    # Add HeadConfig as a nested model
    head_config: HeadConfig | None = Field(
        default=None,
        description="Configuration for the classification/regression head",
    )

    def model_post_init(self, __context):
        task = self.task_type
        if task == "binary_classification":
            task = "binary"
        elif task == "multi_class_classification":
            task = "multiclass"
        elif task == "multi_label_classification":
            task = "multilabel"
        elif task == "token_classification":
            task = "token"

        if task == "binary":
            self.num_labels = self.num_labels or 2
            self.label_names = self.label_names or ["negative", "positive"]

        elif task == "multiclass":
            if not self.num_labels or self.num_labels < 2:
                raise ValueError("num_labels must be at least 2 for multiclass classification")
            if not self.label_names or len(self.label_names) != self.num_labels:
                self.label_names = [f"class_{i}" for i in range(self.num_labels)]

        elif task == "multilabel":
            if not self.num_labels or self.num_labels < 2:
                raise ValueError("num_labels must be at least 2 for multilabel classification")
            if not self.label_names or len(self.label_names) != self.num_labels:
                self.label_names = [f"label_{i}" for i in range(self.num_labels)]

        elif task == "regression":
            self.num_labels = 1
            self.label_names = ["value"]

        elif task in {"mask", "generation"}:
            self.num_labels = None
            self.label_names = None


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping callback."""

    patience: int | None = Field(
        default=None,
        description="Number of evaluation steps with no improvement before stopping. None disables early stopping.",
    )
    threshold: float = Field(
        default=0.0,
        description="Improvement threshold. An improvement of less than this is considered no improvement.",
    )

    @field_validator("patience")
    @classmethod
    def validate_patience(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("patience must be non-negative")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0:
            raise ValueError("threshold must be non-negative")
        return v


class SearchSpaceDistribution(BaseModel):
    """A single hyperparameter distribution for Optuna search.

    Supports both explicit type declaration and concise HF-like format.
    """

    type: str | None = Field(
        default=None,
        pattern="^(float|int)$",
        description="Distribution type. Auto-inferred from other fields if not set.",
    )
    low: float | int
    high: float | int
    log: bool = Field(default=False, description="Use log scale for sampling.")
    step: int | None = Field(default=None, description="Step size for int distributions.")

    @model_validator(mode="after")
    def infer_and_validate(self):
        # Infer type if not specified
        if self.type is None:
            if isinstance(self.low, float) or isinstance(self.high, float):
                self.type = "float"
            else:
                self.type = "int"

        # Auto-enable log scale for float ranges spanning >10x
        if self.type == "float" and not self.log:
            if self.low > 0 and self.high / self.low > 10:
                self.log = True

        # Validate step only for int
        if self.type == "float" and self.step is not None:
            raise ValueError("'step' is only valid for int distributions")

        if self.log and self.low <= 0:
            raise ValueError("low must be positive when log=True")

        if self.low >= self.high:
            raise ValueError("low must be less than high")

        return self


class HyperparameterSearchConfig(BaseModel):
    """Configuration for Optuna hyperparameter search."""

    search_space: dict[str, SearchSpaceDistribution] = Field(
        default_factory=dict,
        description="Mapping from hyperparameter name to its search distribution.",
    )
    n_trials: int = Field(
        default=0,
        ge=0,
        description="Number of Optuna trials. 0 disables hyperparameter search.",
    )
    direction: str = Field(
        default="minimize",
        pattern="^(minimize|maximize)$",
        description="Optimization direction for the objective metric.",
    )
    study_name: str | None = Field(
        default=None,
        description="Name for the Optuna study.",
    )
    metric: str = Field(
        default="eval_loss",
        description="Metric to optimize. Must be returned by compute_metrics.",
    )


class CallbackConfig(BaseModel):
    """Configuration for training callbacks.

    Only predefined callbacks are supported in Phase 2.
    Custom callbacks are deferred to v2.
    """

    early_stopping: EarlyStoppingConfig | None = Field(
        default_factory=EarlyStoppingConfig,
        description="Early stopping configuration.",
    )


class TrainingConfig(BaseModel):
    """Configuration for training"""

    _VALID_REPORT_TO: set[str] = {"tensorboard", "wandb", "none", "all"}

    output_dir: str | None = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int | None = -1
    logging_strategy: str = "steps"
    logging_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 20
    save_safetensors: bool = True
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = Field(default=1.0, ge=0)
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict | None = None
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    report_to: list[str] = Field(
        default_factory=lambda: ["tensorboard"],
        description="List of experiment trackers to report to. Options: 'tensorboard', 'wandb', 'none', 'all'.",
    )
    resume_from_checkpoint: str | None = None
    callbacks: CallbackConfig | None = Field(
        default_factory=CallbackConfig,
        description="Callback configuration for training lifecycle controls.",
    )
    hyperparameter_search: HyperparameterSearchConfig | None = Field(
        default_factory=HyperparameterSearchConfig,
        description="Hyperparameter search configuration. Disabled when n_trials=0.",
    )
    use_qlora: bool = Field(
        default=False,
        description="Whether to use 4-bit quantized LoRA (QLoRA). Requires bitsandbytes.",
    )
    quantization_config: dict | None = Field(
        default=None,
        description="Custom BitsAndBytesConfig parameters. If None, uses QLoRA defaults.",
    )

    @field_validator("report_to", mode="before")
    @classmethod
    def coerce_report_to(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("report_to")
    @classmethod
    def validate_report_to(cls, v: list[str]) -> list[str]:
        valid = {"tensorboard", "wandb", "none", "all"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid report_to values: {invalid}. Valid: {valid}")
        if "none" in v and len(v) > 1:
            raise ValueError("'none' cannot be combined with other trackers")
        if "all" in v and len(v) > 1:
            raise ValueError("'all' cannot be combined with other trackers")
        return v


class LoraConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation).

    Compatible with both standard LoRA and QLoRA (4-bit quantization).
    When use_qlora=True, the base model is loaded in 4-bit and LoRA
    adapters are applied via prepare_model_for_kbit_training().
    """

    r: int = Field(default=8, description="LoRA attention dimension (rank).")
    lora_alpha: int = Field(default=16, description="The alpha parameter for LoRA scaling.")
    target_modules: list[str] | None = Field(  # type: ignore
        default=None,
        description=(
            "The names of the modules to apply LoRA to.",
            "E.g., ['query', 'value'] or ['q_proj', 'v_proj'].",
        ),
    )
    lora_dropout: float = Field(default=0.1, description="The dropout probability for LoRA layers.")
    bias: str = Field(
        default="none",
        pattern="^(none|all|lora_only)$",
        description="Bias type for LoRA. Can be 'none', 'all' or 'lora_only'.",
    )
    lora_bias: bool = Field(default=False, description="Whether to use bias in LoRA layers.")
    inference_mode: bool = Field(
        default=False,
        description="Whether to set the model in inference mode.",
    )
    task_type: str | None = Field(
        default="SEQ_CLS",
        description="The task type for PEFT. E.g., 'CAUSAL_LM', 'TOKEN_CLS'.",
    )


class InferenceConfig(BaseModel):
    """Configuration for model inference"""

    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"  # cpu, cuda, rocm, mps, tpu, ipex, auto
    num_workers: int = 4
    use_fp16: bool = False  # Whether to use half precision
    use_bf16: bool = False  # Whether to use bfloat16 precision
    output_dir: str | None = None


class BenchmarkInfoConfig(BaseModel):
    """Configuration for the benchmark's metadata."""

    name: str = Field(..., description="The overall name for the benchmark.")
    description: str | None = Field(
        None, description="A brief description of the benchmark's purpose."
    )


class ModelConfig(BaseModel):
    """Configuration for a single model to be benchmarked."""

    name: str = Field(..., description="A unique name for the model in the benchmark.")
    path: str = Field(
        ...,
        description="Path to the model, can be a local path or a Hugging Face model identifier.",
    )
    lora_adapter_path: str | None = Field(
        default=None,
        description="Optional path to a trained LoRA adapter for inference.",
    )
    source: str | None = "huggingface"
    task_type: str | None = "classification"
    revision: str | None = "main"
    trust_remote_code: bool = True
    torch_dtype: str | None = "float32"


class DatasetConfig(BaseModel):
    """Configuration for a single dataset used in the benchmark."""

    name: str = Field(..., description="A unique name for the dataset.")
    path: str = Field(..., description="Path to the dataset file (e.g., .csv, .json).")
    task: str = Field(
        ...,
        description="The primary task associated with this dataset (e.g., binary_classification).",
    )
    format: str | None = "csv"
    text_column: str = "sequence"
    label_column: str | None = "label"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    test_size: float | None = 0.2
    val_size: float | None = 0.1
    random_state: int | None = 42
    threshold: float | None = 0.5
    num_labels: int | None = 2
    label_names: list[str] | None = None


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation phase of the benchmark."""

    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"
    num_workers: int = 4
    use_fp16: bool = False
    use_bf16: bool = False
    mixed_precision: bool = True
    pin_memory: bool = True
    memory_efficient_attention: bool = False
    seed: int = 42
    deterministic: bool = True


class OutputConfig(BaseModel):
    """Configuration for generating benchmark reports and artifacts."""

    path: str = "benchmark_results"
    format: str = "html"
    save_predictions: bool = True
    save_embeddings: bool = False
    save_attention_maps: bool = False
    generate_plots: bool = True
    report_title: str = "DNA Model Benchmark Report"
    include_summary: bool = True
    include_details: bool = True
    include_recommendations: bool = True


class BenchmarkConfig(BaseModel):
    """
    Top-level configuration for the DNA Language Model benchmark.
    This class validates and structures the entire YAML configuration file,
    where each top-level key in the YAML corresponds to an attribute of this
    class.
    """

    benchmark: BenchmarkInfoConfig = Field(
        ...,
        description="General metadata and information about the benchmark.",
    )
    models: list[ModelConfig] = Field(..., description="A list of models to be evaluated.")
    datasets: list[DatasetConfig] = Field(
        ..., description="A list of datasets to run the benchmark on."
    )
    metrics: list[str] | None = Field(None, description="A list of evaluation metrics to compute.")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Configuration for the evaluation phase.",
    )
    output: OutputConfig = Field(
        ..., description="Configuration for benchmark outputs and reports."
    )
    config_path: str | None = None


def load_config(config_path: str) -> dict[str, BaseModel]:
    """Load configuration from a YAML file and return a dictionary of
    configuration objects.
    Args:
        config_path (str): Path to the YAML configuration file.
    """

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        config_dict["config_path"] = os.path.abspath(config_path)

    configs: dict[str, BaseModel] = {}

    # Configurations for tasks
    if "task" in config_dict:
        configs["task"] = TaskConfig(**config_dict["task"])

    # Configurations for inference
    if "inference" in config_dict:
        configs["inference"] = InferenceConfig(**config_dict["inference"])

    # Configurations for models
    if "model" in config_dict:
        configs["model"] = config_dict["model"]  # 保持为字典格式

    # Configurations for training (Optional)
    if "finetune" in config_dict:
        configs["finetune"] = TrainingConfig(**config_dict["finetune"])

    # Configurations for LoRA (Optional)
    if "lora" in config_dict:
        configs["lora"] = LoraConfig(**config_dict["lora"])

    # Configurations for benchmark (Optional)
    if "benchmark" in config_dict:
        configs["benchmark"] = BenchmarkConfig(**config_dict)

    return configs
