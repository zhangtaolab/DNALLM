"""
DNA Predictor Adapter for MCP Server

This module provides an adapter layer between the MCP server and the existing DNAPredictor class,
handling model loading, prediction, and result formatting for different task types.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import numpy as np
from pathlib import Path

# Import existing DNALLM components
from dnallm.inference.predictor import DNAPredictor
from dnallm.models.model import load_model_and_tokenizer
from dnallm.configuration.configs import load_config

logger = logging.getLogger(__name__)


class DNAPredictorAdapter:
    """适配器类，将 DNAPredictor 集成到 MCP 服务器中"""
    
    def __init__(self, model_name: str, config_path: str):
        self.model_name = model_name
        self.config_path = config_path
        self.predictor: Optional[DNAPredictor] = None
        self.config: Optional[Dict[str, Any]] = None
        self.is_loaded = False
        self._lock = asyncio.Lock()
    
    async def load_model(self) -> bool:
        """异步加载模型"""
        async with self._lock:
            if self.is_loaded:
                return True
            
            try:
                logger.info(f"Loading model {self.model_name} from {self.config_path}")
                
                # 在线程池中加载配置和模型
                loop = asyncio.get_event_loop()
                self.config = await loop.run_in_executor(None, load_config, self.config_path)
                
                # 加载模型和分词器
                model, tokenizer = await loop.run_in_executor(
                    None, 
                    load_model_and_tokenizer,
                    self.config['model']['path'],
                    self.config['task'],
                    self.config['model']['source']
                )
                
                # 创建预测器
                self.predictor = DNAPredictor(model, tokenizer, self.config)
                self.is_loaded = True
                
                logger.info(f"Successfully loaded model {self.model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                return False
    
    async def predict_single(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """单序列预测"""
        if not self.is_loaded:
            await self.load_model()
        
        if not self.predictor:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        try:
            # 在线程池中执行预测
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._predict_single_sync,
                sequence,
                **kwargs
            )
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {self.model_name}: {e}")
            raise
    
    def _predict_single_sync(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """同步单序列预测"""
        # 使用 DNAPredictor 的 predict_seqs 方法
        predictions = self.predictor.predict_seqs(sequence, **kwargs)
        
        # 格式化结果
        return self._format_prediction_result(predictions, sequence)
    
    async def predict_batch(self, sequences: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.is_loaded:
            await self.load_model()
        
        if not self.predictor:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        try:
            # 在线程池中执行批量预测
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._predict_batch_sync,
                sequences,
                **kwargs
            )
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error for {self.model_name}: {e}")
            raise
    
    def _predict_batch_sync(self, sequences: List[str], **kwargs) -> List[Dict[str, Any]]:
        """同步批量预测"""
        # 使用 DNAPredictor 的 predict_seqs 方法
        predictions = self.predictor.predict_seqs(sequences, **kwargs)
        
        # 格式化结果
        formatted_results = []
        for i, sequence in enumerate(sequences):
            # 提取单个序列的预测结果
            single_prediction = self._extract_single_prediction(predictions, i)
            formatted_result = self._format_prediction_result(single_prediction, sequence)
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _extract_single_prediction(self, predictions: Dict[str, Any], index: int) -> Dict[str, Any]:
        """从批量预测结果中提取单个序列的预测"""
        single_pred = {}
        
        for key, value in predictions.items():
            if isinstance(value, (list, np.ndarray, torch.Tensor)):
                if len(value) > index:
                    single_pred[key] = value[index]
                else:
                    single_pred[key] = value[0] if len(value) > 0 else None
            else:
                single_pred[key] = value
        
        return single_pred
    
    def _format_prediction_result(self, prediction: Dict[str, Any], sequence: str) -> Dict[str, Any]:
        """格式化预测结果"""
        task_type = self.config['task']['task_type']
        
        result = {
            "sequence": sequence,
            "sequence_length": len(sequence),
            "task_type": task_type,
            "model_name": self.model_name
        }
        
        if task_type == "binary":
            result.update(self._format_binary_result(prediction))
        elif task_type == "multiclass":
            result.update(self._format_multiclass_result(prediction))
        elif task_type == "multilabel":
            result.update(self._format_multilabel_result(prediction))
        elif task_type == "regression":
            result.update(self._format_regression_result(prediction))
        else:
            # 默认处理
            result.update(prediction)
        
        return result
    
    def _format_binary_result(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """格式化二分类结果"""
        result = {}
        
        # 提取预测值和概率
        if 'predictions' in prediction:
            pred_value = prediction['predictions']
            if isinstance(pred_value, (list, np.ndarray)):
                pred_value = pred_value[0] if len(pred_value) > 0 else 0
            result['prediction'] = int(pred_value)
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            if isinstance(probs, (list, np.ndarray)):
                probs = probs[0] if len(probs) > 0 else [0.5, 0.5]
            
            # 确保概率是列表格式
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            
            # 获取标签名称
            label_names = self.config['task'].get('label_names', ['Class 0', 'Class 1'])
            if len(probs) >= 2 and len(label_names) >= 2:
                result['probabilities'] = {
                    label_names[0]: float(probs[0]),
                    label_names[1]: float(probs[1])
                }
                result['confidence'] = float(max(probs))
            else:
                result['probabilities'] = {"Class 0": 0.5, "Class 1": 0.5}
                result['confidence'] = 0.5
        
        # 添加阈值
        threshold = self.config['task'].get('threshold', 0.5)
        result['threshold'] = threshold
        
        return result
    
    def _format_multiclass_result(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """格式化多分类结果"""
        result = {}
        
        # 提取预测值
        if 'predictions' in prediction:
            pred_value = prediction['predictions']
            if isinstance(pred_value, (list, np.ndarray)):
                pred_value = pred_value[0] if len(pred_value) > 0 else 0
            result['prediction'] = int(pred_value)
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            if isinstance(probs, (list, np.ndarray)):
                probs = probs[0] if len(probs) > 0 else []
            
            # 确保概率是列表格式
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            
            # 获取标签名称
            label_names = self.config['task'].get('label_names', [])
            if len(probs) > 0 and len(label_names) > 0:
                result['probabilities'] = {
                    label_names[i]: float(probs[i]) 
                    for i in range(min(len(probs), len(label_names)))
                }
                result['confidence'] = float(max(probs))
            else:
                result['probabilities'] = {}
                result['confidence'] = 0.0
        
        return result
    
    def _format_multilabel_result(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """格式化多标签结果"""
        result = {}
        
        # 提取预测值
        if 'predictions' in prediction:
            pred_value = prediction['predictions']
            if isinstance(pred_value, (list, np.ndarray)):
                pred_value = pred_value[0] if len(pred_value) > 0 else []
            result['predictions'] = pred_value.tolist() if isinstance(pred_value, np.ndarray) else pred_value
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            if isinstance(probs, (list, np.ndarray)):
                probs = probs[0] if len(probs) > 0 else []
            
            # 确保概率是列表格式
            if isinstance(probs, np.ndarray):
                probs = probs.tolist()
            
            # 获取标签名称
            label_names = self.config['task'].get('label_names', [])
            if len(probs) > 0 and len(label_names) > 0:
                result['probabilities'] = {
                    label_names[i]: float(probs[i]) 
                    for i in range(min(len(probs), len(label_names)))
                }
            else:
                result['probabilities'] = {}
        
        # 添加阈值
        threshold = self.config['task'].get('threshold', 0.5)
        result['threshold'] = threshold
        
        return result
    
    def _format_regression_result(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """格式化回归结果"""
        result = {}
        
        # 提取预测值
        if 'predictions' in prediction:
            pred_value = prediction['predictions']
            if isinstance(pred_value, (list, np.ndarray)):
                pred_value = pred_value[0] if len(pred_value) > 0 else 0.0
            result['prediction'] = float(pred_value)
        
        # 回归任务通常没有概率，但可能有置信度
        if 'confidence' in prediction:
            result['confidence'] = float(prediction['confidence'])
        else:
            result['confidence'] = 1.0  # 默认置信度
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.config:
            return {}
        
        return {
            "model_name": self.model_name,
            "task_type": self.config['task']['task_type'],
            "num_labels": self.config['task']['num_labels'],
            "label_names": self.config['task'].get('label_names', []),
            "description": self.config['model'].get('task_info', {}).get('describe', ''),
            "model_path": self.config['model']['path'],
            "is_loaded": self.is_loaded
        }
    
    def unload_model(self):
        """卸载模型"""
        if self.predictor:
            # 清理模型和分词器
            if hasattr(self.predictor, 'model'):
                del self.predictor.model
            if hasattr(self.predictor, 'tokenizer'):
                del self.predictor.tokenizer
            del self.predictor
        
        self.predictor = None
        self.is_loaded = False
        logger.info(f"Unloaded model {self.model_name}")


class DNAPredictorPool:
    """DNA 预测器池管理器"""
    
    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self.predictors: Dict[str, DNAPredictorAdapter] = {}
        self._lock = asyncio.Lock()
    
    async def get_predictor(self, model_name: str, config_path: str) -> DNAPredictorAdapter:
        """获取预测器实例"""
        async with self._lock:
            if model_name not in self.predictors:
                # 检查是否超过最大模型数
                if len(self.predictors) >= self.max_models:
                    # 移除最少使用的模型
                    await self._remove_least_used_model()
                
                # 创建新的预测器
                predictor = DNAPredictorAdapter(model_name, config_path)
                self.predictors[model_name] = predictor
            
            return self.predictors[model_name]
    
    async def _remove_least_used_model(self):
        """移除最少使用的模型"""
        if not self.predictors:
            return
        
        # 简单实现：移除第一个模型
        model_name = next(iter(self.predictors))
        predictor = self.predictors[model_name]
        predictor.unload_model()
        del self.predictors[model_name]
        logger.info(f"Removed model {model_name} from pool")
    
    async def predict_single(self, model_name: str, config_path: str, sequence: str, **kwargs) -> Dict[str, Any]:
        """单序列预测"""
        predictor = await self.get_predictor(model_name, config_path)
        return await predictor.predict_single(sequence, **kwargs)
    
    async def predict_batch(self, model_name: str, config_path: str, sequences: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量预测"""
        predictor = await self.get_predictor(model_name, config_path)
        return await predictor.predict_batch(sequences, **kwargs)
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return [name for name, predictor in self.predictors.items() if predictor.is_loaded]
    
    def get_pool_info(self) -> Dict[str, Any]:
        """获取池信息"""
        return {
            "total_models": len(self.predictors),
            "loaded_models": len(self.get_loaded_models()),
            "max_models": self.max_models,
            "models": list(self.predictors.keys())
        }
    
    async def shutdown(self):
        """关闭预测器池"""
        async with self._lock:
            for predictor in self.predictors.values():
                predictor.unload_model()
            self.predictors.clear()
        logger.info("DNA predictor pool shutdown complete")
