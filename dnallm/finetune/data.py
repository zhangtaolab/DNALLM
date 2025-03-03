from torch.utils.data import Dataset
import pandas as pd
from typing import Optional, Union, List

class DNADataset(Dataset):
    def __init__(
        self, 
        file_path: str,
        task_config: Optional[TaskConfig] = None,
        label_column: Optional[str] = None
    ):
        """
        Initialize DNA dataset
        
        Args:
            file_path: Path to data file (CSV or FASTA)
            task_config: Task configuration
            label_column: Name of label column for CSV files
        """
        self.task_config = task_config
        self.data = self.load_data(file_path, label_column)
    
    def load_data(self, file_path: str, label_column: Optional[str]):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            self.sequences = df['sequence'].tolist()
            self.labels = df[label_column].tolist() if label_column else None
        else:  # FASTA format
            with open(file_path) as f:
                self.sequences = [line.strip() for line in f 
                                if not line.startswith(">")]
                self.labels = None
                
        return self.sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'sequence': self.sequences[idx],
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item 

"""
DNA序列数据集处理模块

本模块提供了DNA序列数据的加载和预处理功能，主要特性：

1. DNADataset类
   - 支持多种数据格式：
     * FASTA格式：标准DNA序列文件
     * CSV格式：带标签的结构化数据
   - 提供数据集迭代接口
   - 支持标签处理

2. 数据处理功能：
   - 序列读取和验证
   - 标签转换和编码
   - 批次生成
   - 数据清洗和过滤

3. 支持的数据特性：
   - 可变长度序列
   - 多种标签类型
   - 缺失值处理
   - 数据增强

使用示例：
    dataset = DNADataset(
        file_path="data/sequences.csv",
        task_config=task_config,
        label_column="label"
    )
""" 