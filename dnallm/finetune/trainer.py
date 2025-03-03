import os
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from .models.base import BaseDNAModel
from .data import DNADataset
from .config import TrainingConfig
from .tasks import TaskType, TaskConfig, TaskHead, compute_metrics

"""
DNA语言模型训练器模块

本模块实现了DNA语言模型的训练流程管理，主要功能包括：

1. DNALLMTrainer类
   - 统一管理模型训练、评估和预测流程
   - 支持多种任务类型（分类、回归）
   - 集成任务特定的预测头
   - 提供训练参数配置
   - 实现训练过程监控和模型保存

2. 核心功能：
   - 模型初始化和设备管理
   - 训练参数配置
   - 训练循环控制
   - 评估指标计算
   - 模型保存和加载
   - 预测结果生成

3. 支持的训练特性：
   - 自动评估和保存最佳模型
   - 训练日志记录
   - 灵活的批次大小设置
   - 学习率和权重衰减配置
   - 分布式训练支持

使用示例：
    trainer = DNALLMTrainer(
        model=model,
        config=training_config,
        task_config=task_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    metrics = trainer.train()
"""

class DNALLMTrainer:
    """DNA Language Model Trainer class that supports multiple model types"""
    
    def __init__(
        self,
        model: BaseDNAModel,
        config: TrainingConfig,
        task_config: TaskConfig,
        train_dataset: Optional[DNADataset] = None,
        eval_dataset: Optional[DNADataset] = None,
    ):
        self.model = model
        self.config = config
        self.task_config = task_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Add task-specific head
        base_model = model.get_model()
        self.task_head = TaskHead(
            task_config,
            hidden_size=base_model.config.hidden_size
        )
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model_with_head(),
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_task_metrics,
        )
    
    def model_with_head(self):
        """Combine base model with task head"""
        class CombinedModel(nn.Module):
            def __init__(self, base_model, task_head):
                super().__init__()
                self.base_model = base_model
                self.task_head = task_head
                
            def forward(self, **inputs):
                outputs = self.base_model(**inputs)
                logits = self.task_head(outputs.last_hidden_state[:, 0])
                return logits
                
        return CombinedModel(self.model.get_model(), self.task_head)
    
    def compute_task_metrics(self, eval_pred):
        """Compute task-specific metrics"""
        predictions, labels = eval_pred
        return compute_metrics(self.task_config, predictions, labels)
    
    def train(self) -> Dict[str, float]:
        """Train the model and return metrics"""
        train_result = self.trainer.train()
        metrics = train_result.metrics
        
        # Save the model
        self.trainer.save_model()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model and return metrics"""
        return self.trainer.evaluate()
    
    def compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        # Implement your metric computation here
        metrics = {
            "accuracy": 0.0,  # Replace with actual metric computation
        }
        return metrics 