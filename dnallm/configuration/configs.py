import yaml
from pydantic import BaseModel, Field
from typing import Optional, List, Union


class TaskConfig(BaseModel):
    """Configuration for different fine-tuning tasks"""
    task_type: str = Field(..., pattern="^(embedding|mask|generation|binary|multiclass|multilabel|regression|token)$")
    num_labels: int = 2
    label_names: Optional[List] = None
    threshold: float = 0.5  # For binary classification and multi label classification


class TrainingConfig(BaseModel):
    """Configuration for training"""
    output_dir: str = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = -1
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
    lr_scheduler_kwargs: Union[dict, None]
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    report_to: str = "all"
    resume_from_checkpoint: Union[str, None]


class InferenceConfig(BaseModel):
    """Configuration for model inference"""
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"  # cpu, cuda, rocm, mps, tpu, ipex, auto
    num_workers: int = 4
    use_fp16: bool = False  # Whether to use half precision
    output_dir: Optional[str] = None


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # 根据配置文件内容动态创建配置
    configs = {}
    
    # 任务配置
    if 'task' in config_dict:
        configs['task'] = TaskConfig(**config_dict['task'])
    
    # 推理配置
    if 'inference' in config_dict:
        configs['inference'] = InferenceConfig(**config_dict['inference'])
    
    # 训练配置（可选）
    if 'finetune' in config_dict:
        configs['finetune'] = TrainingConfig(**config_dict['finetune'])
    
    return configs

