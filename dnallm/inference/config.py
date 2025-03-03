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

"""
DNA语言模型推理配置模块

本模块定义了模型推理时的配置参数，包括：

1. InferenceConfig类
   - 模型路径配置
   - 批处理大小设置
   - 序列最大长度限制
   - 计算设备选择
   - 并行处理线程数
   - 半精度推理选项
   - 输出目录设置

2. 配置参数说明：
   - model_path: 训练好的模型路径
   - batch_size: 推理批次大小
   - max_length: 序列最大长度
   - device: 使用的计算设备(CPU/GPU)
   - num_workers: 数据加载线程数
   - use_fp16: 是否使用半精度推理
   - output_dir: 结果保存目录

使用示例：
    config = InferenceConfig(
        model_path="models/dna_model",
        batch_size=32,
        device="cuda"
    )
""" 