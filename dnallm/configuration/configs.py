import os
import yaml
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Configuration for different fine-tuning tasks"""

    task_type: str = Field(
        ...,
        pattern="^(embedding|mask|generation|binary|multiclass|"
        "multilabel|regression|token)$",
    )
    num_labels: int | None = Field(
        default=2, description="Number of labels (default 2)"
    )
    label_names: list[str] | None = None
    threshold: float = Field(
        default=0.5, description="Threshold for binary/multilabel tasks"
    )
    mlm_probability: float | None = Field(
        default=0.15, description="Masking probability for MLM tasks"
    )

    def model_post_init(self, __context):
        if self.task_type == "binary":
            self.num_labels = self.num_labels or 2
            self.label_names = self.label_names or ["negative", "positive"]

        elif self.task_type == "multiclass":
            if not self.num_labels or self.num_labels < 2:
                raise ValueError(
                    "num_labels must be at least 2 for multiclass "
                    "classification"
                )
            if (
                not self.label_names
                or len(self.label_names) != self.num_labels
            ):
                self.label_names = [
                    f"class_{i}" for i in range(self.num_labels)
                ]

        elif self.task_type == "multilabel":
            if not self.num_labels or self.num_labels < 2:
                raise ValueError(
                    "num_labels must be at least 2 for multilabel "
                    "classification"
                )
            if (
                not self.label_names
                or len(self.label_names) != self.num_labels
            ):
                self.label_names = [
                    f"label_{i}" for i in range(self.num_labels)
                ]

        elif self.task_type == "regression":
            self.num_labels = 1
            self.label_names = ["value"]

        elif self.task_type in {"mask", "generation"}:
            self.num_labels = None
            self.label_names = None


class TrainingConfig(BaseModel):
    """Configuration for training"""

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
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict | None = None
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    report_to: str = "all"
    resume_from_checkpoint: str | None = None


class InferenceConfig(BaseModel):
    """Configuration for model inference"""

    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"  # cpu, cuda, rocm, mps, tpu, ipex, auto
    num_workers: int = 4
    use_fp16: bool = False  # Whether to use half precision
    output_dir: str | None = None


class BenchmarkInfoConfig(BaseModel):
    """Configuration for the benchmark's metadata."""

    name: str = Field(..., description="The overall name for the benchmark.")
    description: str | None = Field(
        None, description="A brief description of the benchmark's purpose."
    )


class ModelConfig(BaseModel):
    """Configuration for a single model to be benchmarked."""

    name: str = Field(
        ..., description="A unique name for the model in the benchmark."
    )
    path: str = Field(
        ...,
        description="Path to the model, can be a local path or a Hugging "
        "Face model identifier.",
    )
    source: str | None = "huggingface"
    task_type: str | None = "classification"
    revision: str | None = "main"
    trust_remote_code: bool = True
    torch_dtype: str | None = "float32"


class DatasetConfig(BaseModel):
    """Configuration for a single dataset used in the benchmark."""

    name: str = Field(..., description="A unique name for the dataset.")
    path: str = Field(
        ..., description="Path to the dataset file (e.g., .csv, .json)."
    )
    task: str = Field(
        ...,
        description="The primary task associated with this dataset "
        "(e.g., binary_classification).",
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
    models: list[ModelConfig] = Field(
        ..., description="A list of models to be evaluated."
    )
    datasets: list[DatasetConfig] = Field(
        ..., description="A list of datasets to run the benchmark on."
    )
    metrics: list[str] | None = Field(
        None, description="A list of evaluation metrics to compute."
    )
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

    # 根据配置文件内容动态创建配置
    configs = {}

    # 任务配置
    if "task" in config_dict:
        configs["task"] = TaskConfig(**config_dict["task"])

    # 推理配置
    if "inference" in config_dict:
        configs["inference"] = InferenceConfig(**config_dict["inference"])

    # 模型配置
    if "model" in config_dict:
        configs["model"] = config_dict["model"]  # 保持为字典格式

    # 训练配置(可选)
    if "finetune" in config_dict:
        configs["finetune"] = TrainingConfig(**config_dict["finetune"])

    if "benchmark" in config_dict:
        configs["benchmark"] = BenchmarkConfig(**config_dict)

    return configs
