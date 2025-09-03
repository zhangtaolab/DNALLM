"""
Task Router for MCP Server

This module provides task routing functionality to handle different types of DNA prediction tasks,
including binary classification, multiclass classification, multilabel classification, and regression.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    REGRESSION = "regression"


@dataclass
class TaskConfig:
    """任务配置"""
    task_type: TaskType
    num_labels: int
    label_names: List[str]
    threshold: Optional[float] = None
    description: str = ""


@dataclass
class PredictionResult:
    """预测结果"""
    sequence: str
    task_type: TaskType
    model_name: str
    prediction: Union[int, float, List[int]]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskRouter:
    """任务路由器"""
    
    def __init__(self):
        self.task_handlers = {
            TaskType.BINARY: self._handle_binary_task,
            TaskType.MULTICLASS: self._handle_multiclass_task,
            TaskType.MULTILABEL: self._handle_multilabel_task,
            TaskType.REGRESSION: self._handle_regression_task
        }
    
    def get_task_type(self, task_type_str: str) -> TaskType:
        """获取任务类型枚举"""
        try:
            return TaskType(task_type_str.lower())
        except ValueError:
            raise ValueError(f"Unknown task type: {task_type_str}")
    
    def create_task_config(self, config_dict: Dict[str, Any]) -> TaskConfig:
        """从配置字典创建任务配置"""
        task_type = self.get_task_type(config_dict.get('task_type', 'binary'))
        
        return TaskConfig(
            task_type=task_type,
            num_labels=config_dict.get('num_labels', 2),
            label_names=config_dict.get('label_names', []),
            threshold=config_dict.get('threshold', 0.5),
            description=config_dict.get('describe', '')
        )
    
    async def route_prediction(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str, 
        task_config: TaskConfig
    ) -> PredictionResult:
        """路由预测结果到相应的处理器"""
        handler = self.task_handlers.get(task_config.task_type)
        if not handler:
            raise ValueError(f"No handler for task type: {task_config.task_type}")
        
        return await handler(raw_result, sequence, model_name, task_config)
    
    async def _handle_binary_task(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str, 
        task_config: TaskConfig
    ) -> PredictionResult:
        """处理二分类任务"""
        # 提取预测值
        prediction = raw_result.get('prediction', 0)
        if isinstance(prediction, (list, np.ndarray)):
            prediction = int(prediction[0]) if len(prediction) > 0 else 0
        else:
            prediction = int(prediction)
        
        # 提取概率
        probabilities = raw_result.get('probabilities', {})
        if isinstance(probabilities, (list, np.ndarray)):
            # 转换为字典格式
            if len(probabilities) >= 2 and len(task_config.label_names) >= 2:
                probabilities = {
                    task_config.label_names[0]: float(probabilities[0]),
                    task_config.label_names[1]: float(probabilities[1])
                }
            else:
                probabilities = {"Class 0": 0.5, "Class 1": 0.5}
        
        # 计算置信度
        if probabilities:
            confidence = max(probabilities.values()) if probabilities else 0.5
        else:
            confidence = 0.5
        
        # 应用阈值
        threshold = task_config.threshold or 0.5
        if confidence >= threshold:
            final_prediction = 1
        else:
            final_prediction = 0
        
        return PredictionResult(
            sequence=sequence,
            task_type=TaskType.BINARY,
            model_name=model_name,
            prediction=final_prediction,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "threshold": threshold,
                "raw_prediction": prediction
            }
        )
    
    async def _handle_multiclass_task(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str, 
        task_config: TaskConfig
    ) -> PredictionResult:
        """处理多分类任务"""
        # 提取预测值
        prediction = raw_result.get('prediction', 0)
        if isinstance(prediction, (list, np.ndarray)):
            prediction = int(prediction[0]) if len(prediction) > 0 else 0
        else:
            prediction = int(prediction)
        
        # 提取概率
        probabilities = raw_result.get('probabilities', {})
        if isinstance(probabilities, (list, np.ndarray)):
            # 转换为字典格式
            if len(probabilities) > 0 and len(task_config.label_names) > 0:
                probabilities = {
                    task_config.label_names[i]: float(probabilities[i]) 
                    for i in range(min(len(probabilities), len(task_config.label_names)))
                }
            else:
                probabilities = {}
        
        # 计算置信度
        if probabilities:
            confidence = max(probabilities.values()) if probabilities else 0.0
        else:
            confidence = 0.0
        
        return PredictionResult(
            sequence=sequence,
            task_type=TaskType.MULTICLASS,
            model_name=model_name,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "num_classes": task_config.num_labels
            }
        )
    
    async def _handle_multilabel_task(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str, 
        task_config: TaskConfig
    ) -> PredictionResult:
        """处理多标签任务"""
        # 提取预测值
        prediction = raw_result.get('predictions', [])
        if isinstance(prediction, (list, np.ndarray)):
            prediction = [int(x) for x in prediction]
        else:
            prediction = [int(prediction)] if prediction is not None else []
        
        # 提取概率
        probabilities = raw_result.get('probabilities', {})
        if isinstance(probabilities, (list, np.ndarray)):
            # 转换为字典格式
            if len(probabilities) > 0 and len(task_config.label_names) > 0:
                probabilities = {
                    task_config.label_names[i]: float(probabilities[i]) 
                    for i in range(min(len(probabilities), len(task_config.label_names)))
                }
            else:
                probabilities = {}
        
        # 计算置信度（多标签任务使用平均置信度）
        if probabilities:
            confidence = sum(probabilities.values()) / len(probabilities) if probabilities else 0.0
        else:
            confidence = 0.0
        
        # 应用阈值
        threshold = task_config.threshold or 0.5
        final_prediction = []
        if probabilities:
            for label, prob in probabilities.items():
                if prob >= threshold:
                    final_prediction.append(1)
                else:
                    final_prediction.append(0)
        else:
            final_prediction = prediction
        
        return PredictionResult(
            sequence=sequence,
            task_type=TaskType.MULTILABEL,
            model_name=model_name,
            prediction=final_prediction,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "threshold": threshold,
                "raw_prediction": prediction
            }
        )
    
    async def _handle_regression_task(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str, 
        task_config: TaskConfig
    ) -> PredictionResult:
        """处理回归任务"""
        # 提取预测值
        prediction = raw_result.get('prediction', 0.0)
        if isinstance(prediction, (list, np.ndarray)):
            prediction = float(prediction[0]) if len(prediction) > 0 else 0.0
        else:
            prediction = float(prediction)
        
        # 回归任务通常没有概率，但可能有置信度
        confidence = raw_result.get('confidence', 1.0)
        if not isinstance(confidence, (int, float)):
            confidence = 1.0
        
        return PredictionResult(
            sequence=sequence,
            task_type=TaskType.REGRESSION,
            model_name=model_name,
            prediction=prediction,
            confidence=float(confidence),
            probabilities=None,
            metadata={
                "task_type": "regression"
            }
        )
    
    def format_prediction_result(self, result: PredictionResult) -> Dict[str, Any]:
        """格式化预测结果"""
        formatted = {
            "sequence": result.sequence,
            "sequence_length": len(result.sequence),
            "task_type": result.task_type.value,
            "model_name": result.model_name,
            "prediction": result.prediction,
            "confidence": result.confidence,
            "metadata": result.metadata or {}
        }
        
        if result.probabilities:
            formatted["probabilities"] = result.probabilities
        
        return formatted
    
    def get_task_summary(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """获取任务摘要"""
        if not results:
            return {"total_predictions": 0}
        
        task_type = results[0].task_type
        total_predictions = len(results)
        
        summary = {
            "total_predictions": total_predictions,
            "task_type": task_type.value,
            "model_name": results[0].model_name
        }
        
        if task_type == TaskType.BINARY:
            # 统计正负样本
            positive_count = sum(1 for r in results if r.prediction == 1)
            negative_count = total_predictions - positive_count
            summary.update({
                "positive_predictions": positive_count,
                "negative_predictions": negative_count,
                "positive_rate": positive_count / total_predictions
            })
        
        elif task_type == TaskType.MULTICLASS:
            # 统计各类别分布
            class_counts = {}
            for result in results:
                pred = result.prediction
                class_counts[pred] = class_counts.get(pred, 0) + 1
            summary["class_distribution"] = class_counts
        
        elif task_type == TaskType.MULTILABEL:
            # 统计标签分布
            label_counts = {}
            for result in results:
                if isinstance(result.prediction, list):
                    for i, pred in enumerate(result.prediction):
                        if pred == 1:
                            label_counts[i] = label_counts.get(i, 0) + 1
            summary["label_distribution"] = label_counts
        
        elif task_type == TaskType.REGRESSION:
            # 统计回归值
            predictions = [float(r.prediction) for r in results]
            summary.update({
                "mean_prediction": np.mean(predictions),
                "std_prediction": np.std(predictions),
                "min_prediction": np.min(predictions),
                "max_prediction": np.max(predictions)
            })
        
        # 计算平均置信度
        confidences = [r.confidence for r in results]
        summary["average_confidence"] = np.mean(confidences)
        summary["min_confidence"] = np.min(confidences)
        summary["max_confidence"] = np.max(confidences)
        
        return summary


class TaskRouterManager:
    """任务路由器管理器"""
    
    def __init__(self):
        self.router = TaskRouter()
        self.task_configs: Dict[str, TaskConfig] = {}
    
    def register_task_config(self, model_name: str, config_dict: Dict[str, Any]):
        """注册任务配置"""
        task_config = self.router.create_task_config(config_dict)
        self.task_configs[model_name] = task_config
        logger.info(f"Registered task config for {model_name}: {task_config.task_type.value}")
    
    def get_task_config(self, model_name: str) -> Optional[TaskConfig]:
        """获取任务配置"""
        return self.task_configs.get(model_name)
    
    async def process_prediction(
        self, 
        raw_result: Dict[str, Any], 
        sequence: str, 
        model_name: str
    ) -> PredictionResult:
        """处理预测结果"""
        task_config = self.get_task_config(model_name)
        if not task_config:
            raise ValueError(f"No task config found for model: {model_name}")
        
        return await self.router.route_prediction(raw_result, sequence, model_name, task_config)
    
    def format_prediction_result(self, result: PredictionResult) -> Dict[str, Any]:
        """格式化预测结果"""
        return self.router.format_prediction_result(result)
    
    def get_task_summary(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """获取任务摘要"""
        return self.router.get_task_summary(results)
    
    def get_registered_models(self) -> List[str]:
        """获取已注册的模型列表"""
        return list(self.task_configs.keys())
    
    def get_models_by_task_type(self, task_type: TaskType) -> List[str]:
        """根据任务类型获取模型列表"""
        return [
            model_name for model_name, config in self.task_configs.items()
            if config.task_type == task_type
        ]
