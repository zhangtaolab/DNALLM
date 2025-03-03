from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    model_path: str
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda"  # cuda or cpu
    num_workers: int = 4
    use_fp16: bool = False  # Whether to use half precision
    output_dir: Optional[str] = None 