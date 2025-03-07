from enum import Enum
from typing import Optional, List


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
    EMBEDDING = "embedding"                   # Get embeddings, attention map and token probability for downstream analysis
    GENERATION = "generation"                 # Generation task, for Causal Language Model
    BINARY = "binary_classification"          # Binary classification task with two labels
    MULTICLASS = "multi_class_classification" # Multi-class classification task that specific the input belongs to which class (more than two)
    MULTILABEL = "multi_label_classification" # Multi-label classification task with multiple binary labels
    REGRESSION = "regression"                 # Regression task which return a score for the input
    NER = "token_classification"              # Token classification task which is usually for Named Entity Recognition


class TaskConfig:
    """Configuration for different fine-tuning tasks"""
    def __init__(
        self,
        task_type: TaskType,
        num_labels: int = 2,
        label_names: Optional[List[str]] = None,
        threshold: float = 0.5,  # For binary classification and multi label classification
    ):
        self.task_type = task_type
        self.num_labels = num_labels
        self.label_names = label_names or [f"class_{i}" for i in range(num_labels)]
        self.threshold = threshold



