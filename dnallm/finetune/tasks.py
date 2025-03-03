from enum import Enum
from typing import Optional, List, Union
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

"""
DNA语言模型微调任务定义模块

本模块定义了DNA语言模型微调时支持的各种任务类型和相关组件，包括：

1. TaskType: 任务类型枚举
   - 二分类(BINARY)：如启动子预测、增强子识别等
   - 多分类(MULTICLASS)：如蛋白质家族分类、功能区域分类等
   - 回归(REGRESSION)：如表达水平预测、结合强度预测等

2. TaskConfig: 任务配置类
   - 配置任务类型、标签数量、标签名称等
   - 为二分类任务提供阈值设置

3. TaskHead: 任务特定的预测头
   - 为不同任务类型提供专门的神经网络层
   - 支持特征降维和dropout以防止过拟合
   - 根据任务类型自动选择输出维度

4. compute_metrics: 评估指标计算
   - 二分类：准确率、F1分数
   - 多分类：准确率、宏观F1、加权F1
   - 回归：均方误差、R方值

使用示例：
    task_config = TaskConfig(
        task_type=TaskType.BINARY,
        num_labels=2,
        label_names=["negative", "positive"]
    )
"""

class TaskType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

class TaskConfig:
    """Configuration for different fine-tuning tasks"""
    def __init__(
        self,
        task_type: TaskType,
        num_labels: int = 2,
        label_names: Optional[List[str]] = None,
        threshold: float = 0.5,  # For binary classification
    ):
        self.task_type = task_type
        self.num_labels = num_labels
        self.label_names = label_names or [f"class_{i}" for i in range(num_labels)]
        self.threshold = threshold

class TaskHead(nn.Module):
    """Task-specific prediction head"""
    def __init__(self, config: TaskConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        if config.task_type == TaskType.REGRESSION:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, config.num_labels)
            )
            
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

def compute_metrics(task_config: TaskConfig, predictions: torch.Tensor, 
                   labels: torch.Tensor) -> dict:
    """Compute metrics based on task type"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    metrics = {}
    
    if task_config.task_type == TaskType.BINARY:
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        preds = (probs > task_config.threshold).astype(int)
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1"] = f1_score(labels, preds)
        
    elif task_config.task_type == TaskType.MULTICLASS:
        preds = predictions.argmax(axis=1)
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_macro"] = f1_score(labels, preds, average="macro")
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted")
        
    else:  # Regression
        metrics["mse"] = mean_squared_error(labels, predictions)
        metrics["r2"] = r2_score(labels, predictions)
        
    return metrics 