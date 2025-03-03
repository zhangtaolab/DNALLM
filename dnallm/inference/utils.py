from typing import List, Dict, Iterator
from pathlib import Path
import torch
import json

def batch_sequences(sequences: List[str], batch_size: int) -> Iterator[List[str]]:
    """Yield batches of sequences"""
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def save_predictions(predictions: Dict[str, torch.Tensor], output_dir: Path) -> None:
    """Save predictions to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    json_predictions = {
        k: v.cpu().tolist() for k, v in predictions.items()
    }
    
    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(json_predictions, f)

"""
DNA语言模型推理工具模块

本模块提供了推理过程中需要的各种工具函数，包括：

1. 序列处理工具：
   - 序列批处理生成器
   - 序列格式转换
   - 序列验证和清理

2. 结果处理工具：
   - 预测结果保存
   - JSON格式转换
   - 张量处理
   - 文件操作

3. 主要功能：
   - batch_sequences: 生成序列批次
   - save_predictions: 保存预测结果
   - 张量到列表的转换
   - 文件路径处理

使用示例：
    batches = batch_sequences(sequences, batch_size=32)
    save_predictions(predictions, output_dir)
""" 