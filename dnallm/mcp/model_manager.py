"""
Model Manager for MCP Server

This module manages model loading, caching, and prediction tasks.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Import existing DNALLM components
from dnallm.inference.predictor import DNAPredictor
from dnallm.models.model import load_model_and_tokenizer
from dnallm.configuration.configs import load_config

# Import MCP components
from .dna_predictor_adapter import DNAPredictorAdapter, DNAPredictorPool

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    model_name: str
    predictor: Optional[DNAPredictor]
    config: Dict[str, Any]
    loaded_at: float
    last_used: float
    usage_count: int
    is_loading: bool = False


class ModelManager:
    """模型管理器"""
    
    def __init__(self, max_models: int = 10, max_concurrent_requests: int = 100):
        self.max_models = max_models
        self.max_concurrent_requests = max_concurrent_requests
        self.models: Dict[str, ModelInfo] = {}
        self.loading_models: Dict[str, asyncio.Event] = {}
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
        # 使用新的 DNA 预测器池
        self.predictor_pool = DNAPredictorPool(max_models=max_models)
    
    async def load_model(self, model_name: str, config_path: str) -> bool:
        """异步加载模型"""
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        # 检查是否正在加载
        if model_name in self.loading_models:
            logger.info(f"Model {model_name} is already being loaded")
            await self.loading_models[model_name].wait()
            return model_name in self.models
        
        # 创建加载事件
        loading_event = asyncio.Event()
        self.loading_models[model_name] = loading_event
        
        try:
            logger.info(f"Loading model {model_name} from {config_path}")
            
            # 在线程池中加载模型
            loop = asyncio.get_event_loop()
            predictor = await loop.run_in_executor(
                self.executor, 
                self._load_model_sync, 
                model_name, 
                config_path
            )
            
            if predictor:
                # 创建模型信息
                model_info = ModelInfo(
                    name=model_name,
                    model_name=model_name,
                    predictor=predictor,
                    config={},  # 可以在这里存储配置信息
                    loaded_at=time.time(),
                    last_used=time.time(),
                    usage_count=0
                )
                
                with self._lock:
                    self.models[model_name] = model_info
                
                logger.info(f"Successfully loaded model {model_name}")
                return True
            else:
                logger.error(f"Failed to load model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
        finally:
            # 清理加载事件
            if model_name in self.loading_models:
                del self.loading_models[model_name]
            loading_event.set()
    
    def _load_model_sync(self, model_name: str, config_path: str) -> Optional[DNAPredictor]:
        """同步加载模型（在线程池中执行）"""
        try:
            # 加载配置
            config = load_config(config_path)
            
            # 加载模型和分词器
            model, tokenizer = load_model_and_tokenizer(config)
            
            # 创建预测器
            predictor = DNAPredictor(model, tokenizer, config)
            
            return predictor
            
        except Exception as e:
            logger.error(f"Error in sync model loading for {model_name}: {e}")
            return None
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                logger.info(f"Unloaded model {model_name}")
                return True
            return False
    
    async def get_model(self, model_name: str) -> Optional[DNAPredictor]:
        """获取模型预测器"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded")
            return None
        
        # 更新使用统计
        with self._lock:
            if model_name in self.models:
                self.models[model_name].last_used = time.time()
                self.models[model_name].usage_count += 1
        
        return self.models[model_name].predictor
    
    async def predict(self, model_name: str, sequence: str, config_path: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """使用指定模型进行预测"""
        async with self.request_semaphore:
            try:
                # 如果没有提供配置路径，尝试从已加载的模型中获取
                if not config_path:
                    model_info = self.get_model_info(model_name)
                    if not model_info:
                        logger.error(f"Model {model_name} not found and no config path provided")
                        return None
                    # 这里需要从配置中获取路径，暂时使用默认路径
                    config_path = f"./configs/generated/{model_name}_config.yaml"
                
                # 使用新的预测器池进行预测
                result = await self.predictor_pool.predict_single(
                    model_name, config_path, sequence, **kwargs
                )
                return result
                
            except Exception as e:
                logger.error(f"Error in prediction for {model_name}: {e}")
                return None
    
    def _predict_sync(self, predictor: DNAPredictor, sequence: str, **kwargs) -> Dict[str, Any]:
        """同步预测（在线程池中执行）"""
        try:
            # 使用现有的预测方法
            result = predictor.predict(sequence, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in sync prediction: {e}")
            return {}
    
    async def batch_predict(self, model_name: str, sequences: List[str], config_path: str = None, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """批量预测"""
        async with self.request_semaphore:
            try:
                # 如果没有提供配置路径，尝试从已加载的模型中获取
                if not config_path:
                    model_info = self.get_model_info(model_name)
                    if not model_info:
                        logger.error(f"Model {model_name} not found and no config path provided")
                        return None
                    # 这里需要从配置中获取路径，暂时使用默认路径
                    config_path = f"./configs/generated/{model_name}_config.yaml"
                
                # 使用新的预测器池进行批量预测
                results = await self.predictor_pool.predict_batch(
                    model_name, config_path, sequences, **kwargs
                )
                return results
                
            except Exception as e:
                logger.error(f"Error in batch prediction for {model_name}: {e}")
                return None
    
    def _batch_predict_sync(self, predictor: DNAPredictor, sequences: List[str], **kwargs) -> List[Dict[str, Any]]:
        """同步批量预测（在线程池中执行）"""
        try:
            results = []
            for sequence in sequences:
                result = predictor.predict(sequence, **kwargs)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in sync batch prediction: {e}")
            return []
    
    async def multi_model_predict(self, model_names: List[str], sequence: str, config_paths: Dict[str, str] = None, **kwargs) -> Dict[str, Any]:
        """多模型并行预测"""
        # 创建预测任务
        tasks = []
        for model_name in model_names:
            config_path = config_paths.get(model_name) if config_paths else None
            task = self.predict(model_name, sequence, config_path, **kwargs)
            tasks.append((model_name, task))
        
        # 并行执行预测
        results = {}
        for model_name, task in tasks:
            try:
                result = await task
                if result:
                    results[model_name] = result
                else:
                    results[model_name] = {"error": "Prediction failed"}
            except Exception as e:
                logger.error(f"Error in multi-model prediction for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        with self._lock:
            if model_name not in self.models:
                return None
            
            model_info = self.models[model_name]
            return {
                "name": model_info.name,
                "model_name": model_info.model_name,
                "loaded_at": model_info.loaded_at,
                "last_used": model_info.last_used,
                "usage_count": model_info.usage_count,
                "is_loaded": True
            }
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        with self._lock:
            return list(self.models.keys())
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        with self._lock:
            total_models = len(self.models)
            total_usage = sum(model.usage_count for model in self.models.values())
            
            return {
                "total_loaded_models": total_models,
                "max_models": self.max_models,
                "total_predictions": total_usage,
                "models": {
                    name: {
                        "usage_count": model.usage_count,
                        "last_used": model.last_used,
                        "loaded_at": model.loaded_at
                    }
                    for name, model in self.models.items()
                }
            }
    
    async def cleanup_unused_models(self, max_age: float = 3600) -> int:
        """清理未使用的模型"""
        current_time = time.time()
        models_to_remove = []
        
        with self._lock:
            for name, model_info in self.models.items():
                if current_time - model_info.last_used > max_age:
                    models_to_remove.append(name)
        
        # 卸载模型
        for model_name in models_to_remove:
            await self.unload_model(model_name)
        
        logger.info(f"Cleaned up {len(models_to_remove)} unused models")
        return len(models_to_remove)
    
    async def preload_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """预加载模型"""
        results = {}
        
        for config in model_configs:
            model_name = config.get('name')
            config_path = config.get('config_path')
            
            if model_name and config_path:
                success = await self.load_model(model_name, config_path)
                results[model_name] = success
        
        return results
    
    def shutdown(self):
        """关闭模型管理器"""
        logger.info("Shutting down model manager")
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 关闭预测器池
        asyncio.create_task(self.predictor_pool.shutdown())
        
        # 清理模型
        with self._lock:
            self.models.clear()
            self.loading_models.clear()
        
        logger.info("Model manager shutdown complete")


class ModelPool:
    """模型池管理器"""
    
    def __init__(self, max_models_per_type: int = 3):
        self.max_models_per_type = max_models_per_type
        self.model_managers: Dict[str, ModelManager] = {}
        self._lock = threading.Lock()
    
    def get_manager(self, task_type: str) -> ModelManager:
        """获取指定任务类型的模型管理器"""
        with self._lock:
            if task_type not in self.model_managers:
                self.model_managers[task_type] = ModelManager(
                    max_models=self.max_models_per_type
                )
            return self.model_managers[task_type]
    
    async def predict_with_best_model(self, task_type: str, sequence: str, **kwargs) -> Optional[Dict[str, Any]]:
        """使用最佳模型进行预测"""
        manager = self.get_manager(task_type)
        loaded_models = manager.get_loaded_models()
        
        if not loaded_models:
            logger.warning(f"No models loaded for task type {task_type}")
            return None
        
        # 选择使用次数最少的模型
        best_model = min(loaded_models, key=lambda m: manager.get_model_info(m)['usage_count'])
        return await manager.predict(best_model, sequence, **kwargs)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取模型池统计信息"""
        with self._lock:
            stats = {}
            for task_type, manager in self.model_managers.items():
                stats[task_type] = manager.get_model_stats()
            return stats
    
    async def shutdown(self):
        """关闭模型池"""
        with self._lock:
            for manager in self.model_managers.values():
                manager.shutdown()
            self.model_managers.clear()
