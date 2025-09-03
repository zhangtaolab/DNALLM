"""
Model Pool Manager for MCP Server

This module provides advanced model pool management functionality, including:
- Concurrent model loading and unloading
- Model resource monitoring
- Load balancing across multiple models
- Model health checking
- Automatic model scaling
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ModelResourceUsage:
    """模型资源使用情况"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    disk_usage_mb: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ModelHealth:
    """模型健康状态"""
    status: ModelStatus
    last_health_check: float
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    resource_usage: ModelResourceUsage = field(default_factory=ModelResourceUsage)
    error_message: Optional[str] = None


@dataclass
class ModelInstance:
    """模型实例"""
    model_id: str
    model_name: str
    config_path: str
    status: ModelStatus
    health: ModelHealth
    created_at: float
    last_used: float
    usage_count: int
    max_concurrent_requests: int
    current_requests: int = 0
    predictor: Optional[Any] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelPoolManager:
    """模型池管理器"""
    
    def __init__(
        self,
        max_models: int = 10,
        max_concurrent_requests_per_model: int = 100,
        health_check_interval: int = 60,
        resource_check_interval: int = 30,
        auto_scaling: bool = True,
        min_models: int = 1,
        max_models_per_type: int = 5
    ):
        self.max_models = max_models
        self.max_concurrent_requests_per_model = max_concurrent_requests_per_model
        self.health_check_interval = health_check_interval
        self.resource_check_interval = resource_check_interval
        self.auto_scaling = auto_scaling
        self.min_models = min_models
        self.max_models_per_type = max_models_per_type
        
        self.models: Dict[str, ModelInstance] = {}
        self.model_types: Dict[str, List[str]] = {}  # task_type -> model_ids
        self.loading_models: Set[str] = set()
        self.health_check_task: Optional[asyncio.Task] = None
        self.resource_check_task: Optional[asyncio.Task] = None
        self.auto_scaling_task: Optional[asyncio.Task] = None
        
        self._lock = asyncio.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
    
    async def start(self):
        """启动模型池管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动后台任务
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.resource_check_task = asyncio.create_task(self._resource_check_loop())
        
        if self.auto_scaling:
            self.auto_scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        logger.info("Model Pool Manager started")
    
    async def stop(self):
        """停止模型池管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止后台任务
        tasks = [self.health_check_task, self.resource_check_task, self.auto_scaling_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 卸载所有模型
        await self.unload_all_models()
        
        # 关闭线程池
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Model Pool Manager stopped")
    
    async def load_model(self, model_name: str, config_path: str, task_type: str = "binary") -> bool:
        """加载模型"""
        model_id = f"{model_name}_{int(time.time() * 1000)}"
        
        async with self._lock:
            # 检查是否超过最大模型数
            if len(self.models) >= self.max_models:
                if not await self._make_room_for_new_model():
                    logger.warning(f"Cannot load model {model_name}: maximum models reached")
                    return False
            
            # 检查是否超过该类型的最大模型数
            if task_type in self.model_types:
                if len(self.model_types[task_type]) >= self.max_models_per_type:
                    logger.warning(f"Cannot load model {model_name}: maximum models for type {task_type} reached")
                    return False
            
            # 创建模型实例
            model_instance = ModelInstance(
                model_id=model_id,
                model_name=model_name,
                config_path=config_path,
                status=ModelStatus.LOADING,
                health=ModelHealth(
                    status=ModelStatus.LOADING,
                    last_health_check=time.time()
                ),
                created_at=time.time(),
                last_used=time.time(),
                usage_count=0,
                max_concurrent_requests=self.max_concurrent_requests_per_model
            )
            
            self.models[model_id] = model_instance
            self.loading_models.add(model_id)
            
            # 添加到类型映射
            if task_type not in self.model_types:
                self.model_types[task_type] = []
            self.model_types[task_type].append(model_id)
        
        try:
            # 在线程池中加载模型
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._thread_pool,
                self._load_model_sync,
                model_instance
            )
            
            async with self._lock:
                if success:
                    model_instance.status = ModelStatus.LOADED
                    model_instance.health.status = ModelStatus.LOADED
                    self.loading_models.discard(model_id)
                    logger.info(f"Model {model_name} loaded successfully with ID {model_id}")
                else:
                    model_instance.status = ModelStatus.ERROR
                    model_instance.health.status = ModelStatus.ERROR
                    model_instance.health.error_message = "Failed to load model"
                    self.loading_models.discard(model_id)
                    logger.error(f"Failed to load model {model_name}")
            
            return success
            
        except Exception as e:
            async with self._lock:
                model_instance.status = ModelStatus.ERROR
                model_instance.health.status = ModelStatus.ERROR
                model_instance.health.error_message = str(e)
                self.loading_models.discard(model_id)
            
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def _load_model_sync(self, model_instance: ModelInstance) -> bool:
        """同步加载模型"""
        try:
            # 这里应该调用实际的模型加载逻辑
            # 暂时模拟加载过程
            import time
            time.sleep(1)  # 模拟加载时间
            
            # 创建预测器实例
            # predictor = DNAPredictorAdapter(model_instance.model_name, model_instance.config_path)
            # model_instance.predictor = predictor
            
            return True
            
        except Exception as e:
            logger.error(f"Error in sync model loading: {e}")
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        async with self._lock:
            if model_id not in self.models:
                return False
            
            model_instance = self.models[model_id]
            model_instance.status = ModelStatus.UNLOADING
        
        try:
            # 在线程池中卸载模型
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_pool,
                self._unload_model_sync,
                model_instance
            )
            
            async with self._lock:
                # 从类型映射中移除
                for task_type, model_ids in self.model_types.items():
                    if model_id in model_ids:
                        model_ids.remove(model_id)
                        break
                
                # 删除模型实例
                del self.models[model_id]
                self.loading_models.discard(model_id)
            
            logger.info(f"Model {model_instance.model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    def _unload_model_sync(self, model_instance: ModelInstance):
        """同步卸载模型"""
        try:
            # 清理预测器
            if model_instance.predictor:
                # model_instance.predictor.unload_model()
                model_instance.predictor = None
            
            # 清理其他资源
            model_instance.status = ModelStatus.UNLOADED
            
        except Exception as e:
            logger.error(f"Error in sync model unloading: {e}")
    
    async def get_model(self, model_name: str, task_type: str = None) -> Optional[ModelInstance]:
        """获取可用的模型实例"""
        async with self._lock:
            # 如果指定了任务类型，优先从该类型中选择
            if task_type and task_type in self.model_types:
                available_models = [
                    model_id for model_id in self.model_types[task_type]
                    if model_id in self.models and 
                    self.models[model_id].status == ModelStatus.LOADED and
                    self.models[model_id].current_requests < self.models[model_id].max_concurrent_requests
                ]
            else:
                # 从所有模型中查找
                available_models = [
                    model_id for model_id, model in self.models.items()
                    if model.model_name == model_name and
                    model.status == ModelStatus.LOADED and
                    model.current_requests < model.max_concurrent_requests
                ]
            
            if not available_models:
                return None
            
            # 选择使用次数最少的模型
            best_model_id = min(available_models, key=lambda x: self.models[x].usage_count)
            model_instance = self.models[best_model_id]
            
            # 更新使用统计
            model_instance.usage_count += 1
            model_instance.last_used = time.time()
            model_instance.current_requests += 1
            
            return model_instance
    
    async def release_model(self, model_id: str):
        """释放模型实例"""
        async with self._lock:
            if model_id in self.models:
                model_instance = self.models[model_id]
                model_instance.current_requests = max(0, model_instance.current_requests - 1)
    
    async def predict(self, model_name: str, sequence: str, task_type: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """使用模型进行预测"""
        model_instance = await self.get_model(model_name, task_type)
        if not model_instance:
            return None
        
        try:
            # 这里应该调用实际的预测逻辑
            # result = await model_instance.predictor.predict_single(sequence, **kwargs)
            
            # 暂时返回模拟结果
            result = {
                "sequence": sequence,
                "model_name": model_name,
                "prediction": 1,
                "confidence": 0.95,
                "model_id": model_instance.model_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for model {model_name}: {e}")
            return None
        finally:
            await self.release_model(model_instance.model_id)
    
    async def _make_room_for_new_model(self) -> bool:
        """为新模型腾出空间"""
        # 查找最久未使用的模型
        unused_models = [
            (model_id, model) for model_id, model in self.models.items()
            if model.status == ModelStatus.LOADED and model.current_requests == 0
        ]
        
        if not unused_models:
            return False
        
        # 按最后使用时间排序，卸载最久未使用的
        unused_models.sort(key=lambda x: x[1].last_used)
        model_id_to_unload = unused_models[0][0]
        
        return await self.unload_model(model_id_to_unload)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        async with self._lock:
            for model_id, model_instance in self.models.items():
                if model_instance.status == ModelStatus.LOADED:
                    # 检查模型是否响应
                    try:
                        # 这里应该执行实际的健康检查
                        # health_status = await model_instance.predictor.health_check()
                        
                        # 暂时模拟健康检查
                        health_status = True
                        
                        if health_status:
                            model_instance.health.consecutive_failures = 0
                            model_instance.health.status = ModelStatus.LOADED
                        else:
                            model_instance.health.consecutive_failures += 1
                            if model_instance.health.consecutive_failures >= 3:
                                model_instance.status = ModelStatus.ERROR
                                model_instance.health.status = ModelStatus.ERROR
                                model_instance.health.error_message = "Health check failed"
                        
                        model_instance.health.last_health_check = time.time()
                        
                    except Exception as e:
                        model_instance.health.consecutive_failures += 1
                        model_instance.health.error_message = str(e)
                        if model_instance.health.consecutive_failures >= 3:
                            model_instance.status = ModelStatus.ERROR
                            model_instance.health.status = ModelStatus.ERROR
    
    async def _resource_check_loop(self):
        """资源检查循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.resource_check_interval)
                await self._check_resource_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource check loop: {e}")
    
    async def _check_resource_usage(self):
        """检查资源使用情况"""
        async with self._lock:
            for model_id, model_instance in self.models.items():
                if model_instance.status == ModelStatus.LOADED:
                    # 获取系统资源使用情况
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    memory_mb = memory.used / 1024 / 1024
                    
                    # 更新资源使用情况
                    model_instance.health.resource_usage.cpu_percent = cpu_percent
                    model_instance.health.resource_usage.memory_mb = memory_mb
                    model_instance.health.resource_usage.last_updated = time.time()
    
    async def _auto_scaling_loop(self):
        """自动扩缩容循环"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                await self._perform_auto_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto scaling loop: {e}")
    
    async def _perform_auto_scaling(self):
        """执行自动扩缩容"""
        # 这里可以实现自动扩缩容逻辑
        # 例如：根据负载情况自动加载或卸载模型
        pass
    
    async def unload_all_models(self):
        """卸载所有模型"""
        async with self._lock:
            model_ids = list(self.models.keys())
        
        for model_id in model_ids:
            await self.unload_model(model_id)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """获取模型池状态"""
        # 同步版本，避免异步事件循环问题
        total_models = len(self.models)
        loaded_models = sum(1 for m in self.models.values() if m.status == ModelStatus.LOADED)
        loading_models = len(self.loading_models)
        error_models = sum(1 for m in self.models.values() if m.status == ModelStatus.ERROR)
        
        total_requests = sum(m.current_requests for m in self.models.values())
        total_usage = sum(m.usage_count for m in self.models.values())
        
        model_types_status = {}
        for task_type, model_ids in self.model_types.items():
            model_types_status[task_type] = {
                "total": len(model_ids),
                "loaded": sum(1 for mid in model_ids if mid in self.models and self.models[mid].status == ModelStatus.LOADED),
                "loading": sum(1 for mid in model_ids if mid in self.loading_models),
                "error": sum(1 for mid in model_ids if mid in self.models and self.models[mid].status == ModelStatus.ERROR)
            }
        
        return {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "loading_models": loading_models,
            "error_models": error_models,
            "total_requests": total_requests,
            "total_usage": total_usage,
            "model_types": model_types_status,
            "is_running": self.is_running,
            "auto_scaling": self.auto_scaling
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_id not in self.models:
            return None
        
        model_instance = self.models[model_id]
        return {
            "model_id": model_instance.model_id,
            "model_name": model_instance.model_name,
            "config_path": model_instance.config_path,
            "status": model_instance.status.value,
            "health": {
                "status": model_instance.health.status.value,
                "last_health_check": model_instance.health.last_health_check,
                "consecutive_failures": model_instance.health.consecutive_failures,
                "total_requests": model_instance.health.total_requests,
                "successful_requests": model_instance.health.successful_requests,
                "average_response_time": model_instance.health.average_response_time,
                "resource_usage": {
                    "cpu_percent": model_instance.health.resource_usage.cpu_percent,
                    "memory_mb": model_instance.health.resource_usage.memory_mb,
                    "gpu_memory_mb": model_instance.health.resource_usage.gpu_memory_mb,
                    "disk_usage_mb": model_instance.health.resource_usage.disk_usage_mb,
                    "last_updated": model_instance.health.resource_usage.last_updated
                },
                "error_message": model_instance.health.error_message
            },
            "created_at": model_instance.created_at,
            "last_used": model_instance.last_used,
            "usage_count": model_instance.usage_count,
            "max_concurrent_requests": model_instance.max_concurrent_requests,
            "current_requests": model_instance.current_requests
        }
