import os
from typing import Optional, Dict
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

from ..datasets.data import DNADataset
from ..tasks.metrics import compute_metrics

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
    trainer = DNATrainer(
        model=model,
        config=config,
        datasets=datasets
    )
    metrics = trainer.train()
"""


class DNATrainer:
    """DNA Language Model Trainer class that supports multiple model types"""
    
    def __init__(
        self,
        model,
        config: dict,
        datasets: Optional[DNADataset] = None,
        extra_args: Optional[Dict] = None,
    ):

        self.model = model
        self.task_config = config['task']
        self.train_config = config['finetune']
        self.datasets = datasets
        self.extra_args = extra_args
        
        self.set_up_trainer()
    
    def set_up_trainer(self):
        # Setup training arguments
        training_args = self.train_config.model_dump()
        if self.extra_args:
            training_args.update(self.extra_args)
        self.training_args = TrainingArguments(
            **training_args,
        )
        # Check if the dataset has been split
        if isinstance(self.datasets.dataset, DatasetDict):        
            self.data_split = self.datasets.dataset.keys()
        else:
            self.data_split = [None]
        # Get datasets
        if "train" in self.data_split:
            train_dataset = self.datasets.dataset["train"]
        else:
            if len(self.data_split) == 1:
                train_dataset = self.datasets.dataset
            else:
                raise KeyError("Cannot find train data.")
        eval_key = [x for x in self.data_split if x not in ['train', 'test']]
        if eval_key:
            eval_dataset = self.datasets.dataset[eval_key[0]]
        elif "test" in self.data_split:
            eval_dataset = self.datasets.dataset['test']
        else:
            eval_dataset = None
        
        # Get compute metrics
        compute_metrics = self.compute_task_metrics()
        # Set data collator
        if self.task_config.task_type == "mask":
            from transformers import DataCollatorForLanguageModeling
            mlm_probability = self.task_config.mlm_probability
            mlm_probability = mlm_probability if mlm_probability else 0.15
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=True, mlm_probability=mlm_probability
            )
        elif self.task_config.task_type == "generation":
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.datasets.tokenizer,
                mlm=False
            )
        else:
            data_collator = None
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    def compute_task_metrics(self):
        """Compute task-specific metrics"""
        return compute_metrics(self.task_config)

    def train(self, save_tokenizer: bool = False) -> Dict[str, float]:
        """Train the model and return metrics"""
        self.model.train()
        train_result = self.trainer.train()
        metrics = train_result.metrics
        # Save the model
        self.trainer.save_model()
        if save_tokenizer:
            self.datasets.tokenizer.save_pretrained(self.train_config.output_dir)
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model and return metrics"""
        self.model.eval()
        result = self.trainer.evaluate()
        return result
    
    def predict(self) -> Dict[str, float]:
        """Predict the model and return metrics"""
        self.model.eval()
        result = {}
        if "test" in self.data_split:
            test_dataset = self.datasets.dataset['test']
            result = self.trainer.predict(test_dataset)
        return result
