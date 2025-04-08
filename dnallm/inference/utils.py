from typing import List, Dict, Iterator
from pathlib import Path
import random
import json
import torch

from utils import check_sequence


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


def load_seqfile(file, batch_size=1, sample=10000000, seed=None):
    # 从文件加载序列
    # Load sequences from a file
    seqs = []  # 存储有效序列的列表
    batch = []  # 当前批次的序列
    filtered = 0  # 过滤的序列计数

    if file.endswith(".fa") or file.endswith(".fasta"):
        # 处理FA或FASTA格式文件
        # Handle FA or FASTA format files
        with open(file) as fi:
            for line in fi:
                if line.startswith(">"):
                    continue  # 跳过描述行
                    # Skip header lines
                seq = line.strip().upper()  # 读取并标准化序列
                if check_sequence(seq):
                    batch.append(seq)  # 添加有效序列到当前批次
                else:
                    filtered += 1  # 计数无效序列
                if len(batch) == batch_size:
                    seqs.append(batch)  # 添加当前批次到序列列表
                    batch = []  # 重置当前批次
    else:
        # 处理CSV或其他分隔符文件
        # Handle CSV or other delimiter files
        if file.endswith(".csv"):
            sep = ","  # 设置分隔符为逗号
        else:
            sep = "\t"  # 设置分隔符为制表符
        cnt = 0  # 行计数
        seq_idx = -1  # 序列索引
        with open(file) as fi:
            for line in fi:
                info = line.strip().split(sep)  # 按分隔符分割行
                if cnt == 0:
                    if "sequence" in info:
                        seq_idx = info.index("sequence")  # 找到序列列的索引
                        continue
                if seq_idx == -1:
                    seq = info[-1].upper()  # 默认取最后一列作为序列
                    if check_sequence(seq):
                        batch.append(seq)  # 添加有效序列到当前批次
                else:
                    seq = info[seq_idx].upper()  # 根据索引获取序列
                    if check_sequence(seq):
                        batch.append(seq)  # 添加有效序列到当前批次
                    else:
                        filtered += 1  # 计数无效序列
                if len(batch) == batch_size:
                    seqs.append(batch)  # 添加当前批次到序列列表
                    batch = []  # 重置当前批次
    if filtered > 0:
        print(f"Filtered {filtered} sequence(s) due to unsupported chars or length.")
        # 打印过滤的序列数量
        # Print the number of filtered sequences

    len_seqs = sum([len(batch) for batch in seqs])  # 计算有效序列的总数
    if len_seqs > sample:
        seqs = [seq for batch in seqs for seq in batch]  # 展平序列列表
        if seed is not None:
            random.seed(seed)  # 设置随机种子
        random.shuffle(seqs)  # 随机打乱序列
        seqs = [seqs[i:min(i+batch_size, sample)] for i in range(0, sample, batch_size)]  # 按批次大小分割序列

    return seqs  # 返回加载的序列


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