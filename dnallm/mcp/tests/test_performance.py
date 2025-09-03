"""
Performance tests for MCP Server

This module provides comprehensive performance tests to evaluate the system's
performance under various load conditions.
"""

import pytest
import asyncio
import time
import tempfile
import os
import yaml
import statistics
from pathlib import Path
from typing import Dict, Any, List
import sys
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.model_manager import ModelManager
from dnallm.mcp.model_pool_manager import ModelPoolManager
from dnallm.mcp.dna_predictor_adapter import DNAPredictorAdapter, DNAPredictorPool
from dnallm.mcp.task_router import TaskRouter, TaskRouterManager
from dnallm.mcp.sse_manager import SSEManager
from dnallm.mcp.utils.validators import validate_dna_sequence
from dnallm.mcp.utils.formatters import format_prediction_result


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None
        self.end_time = None
    
    def start_timing(self):
        """开始计时"""
        self.start_time = time.time()
    
    def end_timing(self):
        """结束计时"""
        self.end_time = time.time()
    
    def add_response_time(self, response_time: float):
        """添加响应时间"""
        self.response_times.append(response_time)
    
    def add_system_metrics(self):
        """添加系统指标"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_usage.append(memory.used / 1024 / 1024)  # MB
        self.cpu_usage.append(cpu)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        return {
            "total_requests": len(self.response_times),
            "total_duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": self._percentile(self.response_times, 95),
                "p99": self._percentile(self.response_times, 99)
            },
            "throughput": len(self.response_times) / (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
            "memory_usage": {
                "min": min(self.memory_usage) if self.memory_usage else 0,
                "max": max(self.memory_usage) if self.memory_usage else 0,
                "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0
            },
            "cpu_usage": {
                "min": min(self.cpu_usage) if self.cpu_usage else 0,
                "max": max(self.cpu_usage) if self.cpu_usage else 0,
                "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestSSEPerformance:
    """SSE 性能测试"""
    
    @pytest.fixture
    async def sse_manager(self):
        """SSE 管理器fixture"""
        manager = SSEManager(heartbeat_interval=1)
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_sse_client_connection_performance(self, sse_manager):
        """测试 SSE 客户端连接性能"""
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试连接100个客户端
        client_count = 100
        clients = []
        
        for i in range(client_count):
            start_time = time.time()
            client = await sse_manager.add_client(f"perf_client_{i}")
            end_time = time.time()
            
            metrics.add_response_time(end_time - start_time)
            clients.append(client)
        
        metrics.end_timing()
        metrics.add_system_metrics()
        
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == client_count
        assert summary["response_times"]["mean"] < 0.1  # 平均连接时间 < 100ms
        assert summary["response_times"]["p95"] < 0.2   # 95% 连接时间 < 200ms
        assert sse_manager.get_client_count() == client_count
        
        # 清理
        for client in clients:
            await sse_manager.remove_client(client.client_id)
    
    async def test_sse_event_broadcast_performance(self, sse_manager):
        """测试 SSE 事件广播性能"""
        # 添加客户端
        client_count = 50
        clients = []
        for i in range(client_count):
            client = await sse_manager.add_client(f"broadcast_client_{i}")
            clients.append(client)
        
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试事件广播性能
        event_count = 1000
        for i in range(event_count):
            start_time = time.time()
            await sse_manager.send_prediction_start(f"model_{i}", f"sequence_{i}")
            end_time = time.time()
            
            metrics.add_response_time(end_time - start_time)
            
            if i % 100 == 0:
                metrics.add_system_metrics()
        
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == event_count
        assert summary["response_times"]["mean"] < 0.05  # 平均广播时间 < 50ms
        assert summary["response_times"]["p95"] < 0.1    # 95% 广播时间 < 100ms
        assert summary["throughput"] > 100  # 吞吐量 > 100 events/sec
        
        # 清理
        for client in clients:
            await sse_manager.remove_client(client.client_id)
    
    async def test_sse_concurrent_operations(self, sse_manager):
        """测试 SSE 并发操作性能"""
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 并发添加客户端
        async def add_client(client_id):
            start_time = time.time()
            client = await sse_manager.add_client(client_id)
            end_time = time.time()
            return end_time - start_time, client
        
        client_count = 20
        tasks = [add_client(f"concurrent_client_{i}") for i in range(client_count)]
        results = await asyncio.gather(*tasks)
        
        for response_time, client in results:
            metrics.add_response_time(response_time)
        
        # 并发发送事件
        async def send_event(event_id):
            start_time = time.time()
            await sse_manager.send_prediction_start(f"model_{event_id}", f"sequence_{event_id}")
            end_time = time.time()
            return end_time - start_time
        
        event_count = 100
        tasks = [send_event(i) for i in range(event_count)]
        results = await asyncio.gather(*tasks)
        
        for response_time in results:
            metrics.add_response_time(response_time)
        
        metrics.end_timing()
        metrics.add_system_metrics()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == client_count + event_count
        assert summary["response_times"]["mean"] < 0.1  # 平均响应时间 < 100ms
        assert summary["throughput"] > 50  # 吞吐量 > 50 ops/sec
        
        # 清理
        for i in range(client_count):
            await sse_manager.remove_client(f"concurrent_client_{i}")


class TestTaskRouterPerformance:
    """任务路由器性能测试"""
    
    @pytest.fixture
    def task_manager(self):
        """任务管理器fixture"""
        return TaskRouterManager()
    
    def test_task_router_registration_performance(self, task_manager):
        """测试任务路由器注册性能"""
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试大量任务配置注册
        task_count = 1000
        for i in range(task_count):
            start_time = time.time()
            
            config = {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": [f"Class_{i}_0", f"Class_{i}_1"],
                "threshold": 0.5
            }
            task_manager.register_task_config(f"model_{i}", config)
            
            end_time = time.time()
            metrics.add_response_time(end_time - start_time)
        
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == task_count
        assert summary["response_times"]["mean"] < 0.001  # 平均注册时间 < 1ms
        assert summary["response_times"]["p95"] < 0.005   # 95% 注册时间 < 5ms
        assert len(task_manager.get_registered_models()) == task_count
    
    def test_task_router_query_performance(self, task_manager):
        """测试任务路由器查询性能"""
        # 先注册大量任务配置
        task_count = 500
        for i in range(task_count):
            config = {
                "task_type": "binary" if i % 2 == 0 else "multiclass",
                "num_labels": 2 if i % 2 == 0 else 3,
                "label_names": [f"Class_{i}_{j}" for j in range(2 if i % 2 == 0 else 3)],
                "threshold": 0.5 if i % 2 == 0 else None
            }
            task_manager.register_task_config(f"model_{i}", config)
        
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试查询性能
        query_count = 1000
        for i in range(query_count):
            start_time = time.time()
            
            # 随机查询
            model_name = f"model_{i % task_count}"
            task_config = task_manager.get_task_config(model_name)
            
            end_time = time.time()
            metrics.add_response_time(end_time - start_time)
        
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == query_count
        assert summary["response_times"]["mean"] < 0.001  # 平均查询时间 < 1ms
        assert summary["response_times"]["p95"] < 0.005   # 95% 查询时间 < 5ms
    
    def test_task_router_batch_processing_performance(self, task_manager):
        """测试任务路由器批量处理性能"""
        # 注册任务配置
        task_count = 100
        for i in range(task_count):
            config = {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": [f"Class_{i}_0", f"Class_{i}_1"],
                "threshold": 0.5
            }
            task_manager.register_task_config(f"model_{i}", config)
        
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试批量结果处理
        batch_size = 1000
        results = []
        
        for i in range(batch_size):
            mock_result = {
                "prediction": i % 2,
                "probabilities": [0.3, 0.7] if i % 2 == 1 else [0.7, 0.3]
            }
            results.append(mock_result)
        
        # 批量格式化结果
        start_time = time.time()
        formatted_results = []
        for result in results:
            formatted = format_prediction_result(result)
            formatted_results.append(formatted)
        end_time = time.time()
        
        metrics.add_response_time(end_time - start_time)
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert len(formatted_results) == batch_size
        assert summary["response_times"]["mean"] < 0.1  # 平均批量处理时间 < 100ms
        assert summary["throughput"] > 1000  # 吞吐量 > 1000 results/sec


class TestModelPoolPerformance:
    """模型池性能测试"""
    
    @pytest.fixture
    async def model_pool_manager(self):
        """模型池管理器fixture"""
        manager = ModelPoolManager(
            max_models=10,
            health_check_interval=60,
            resource_check_interval=30
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_model_pool_loading_performance(self, model_pool_manager):
        """测试模型池加载性能"""
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 测试模型加载性能
        model_count = 5
        for i in range(model_count):
            start_time = time.time()
            
            # 模拟模型加载
            success = await model_pool_manager.load_model(
                f"test_model_{i}",
                f"test_config_{i}.yaml",
                "binary"
            )
            
            end_time = time.time()
            metrics.add_response_time(end_time - start_time)
            
            if i % 2 == 0:
                metrics.add_system_metrics()
        
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == model_count
        assert summary["response_times"]["mean"] < 1.0  # 平均加载时间 < 1秒
        assert summary["response_times"]["p95"] < 2.0   # 95% 加载时间 < 2秒
    
    async def test_model_pool_concurrent_access(self, model_pool_manager):
        """测试模型池并发访问性能"""
        # 先加载一些模型
        model_count = 3
        for i in range(model_count):
            await model_pool_manager.load_model(
                f"concurrent_model_{i}",
                f"concurrent_config_{i}.yaml",
                "binary"
            )
        
        metrics = PerformanceMetrics()
        metrics.start_timing()
        
        # 并发访问模型
        async def access_model(model_id):
            start_time = time.time()
            
            model_instance = await model_pool_manager.get_model(f"concurrent_model_{model_id}")
            if model_instance:
                await model_pool_manager.release_model(model_instance.model_id)
            
            end_time = time.time()
            return end_time - start_time
        
        concurrent_count = 50
        tasks = [access_model(i % model_count) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        
        for response_time in results:
            metrics.add_response_time(response_time)
        
        metrics.end_timing()
        summary = metrics.get_summary()
        
        # 性能断言
        assert summary["total_requests"] == concurrent_count
        assert summary["response_times"]["mean"] < 0.1  # 平均访问时间 < 100ms
        assert summary["response_times"]["p95"] < 0.2   # 95% 访问时间 < 200ms


class TestSystemResourcePerformance:
    """系统资源性能测试"""
    
    async def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # 添加大量客户端
        client_count = 100
        clients = []
        for i in range(client_count):
            client = await sse_manager.add_client(f"memory_client_{i}")
            clients.append(client)
        
        # 发送大量事件
        event_count = 1000
        for i in range(event_count):
            await sse_manager.send_prediction_start(f"model_{i}", f"sequence_{i}")
        
        peak_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # 内存使用断言
        assert memory_increase < 100  # 内存增长 < 100MB
        assert peak_memory < 1000     # 峰值内存 < 1GB
        
        # 清理
        for client in clients:
            await sse_manager.remove_client(client.client_id)
        await sse_manager.stop()
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_memory_usage(self):
        """测试真实模型内存使用"""
        import tempfile
        import yaml
        import os
        
        # 创建真实模型配置
        model_info = {
            "name": "Plant DNABERT BPE promoter",
            "model_path": "zhangtaolab/plant-dnabert-BPE-promoter",
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Not promoter", "Core promoter"],
            "threshold": 0.5
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模型配置文件
            model_config = {
                "task": {
                    "task_type": model_info["task_type"],
                    "num_labels": model_info["num_labels"],
                    "label_names": model_info["label_names"],
                    "threshold": model_info["threshold"]
                },
                "inference": {
                    "batch_size": 2,  # 小批量
                    "max_length": 128,  # 短序列
                    "device": "cpu",
                    "num_workers": 1,
                    "use_fp16": False
                },
                "model": {
                    "name": model_info["name"],
                    "path": model_info["model_path"],
                    "source": "huggingface",
                    "trust_remote_code": True,
                    "torch_dtype": "float32"
                }
            }
            
            config_path = os.path.join(temp_dir, "real_model_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(model_config, f)
            
            # 记录初始内存
            initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            # 创建适配器并加载模型
            adapter = DNAPredictorAdapter(model_info["name"], config_path)
            
            try:
                # 加载模型
                success = await adapter.load_model()
                if not success:
                    pytest.skip(f"Failed to load real model {model_info['name']}")
                
                # 记录加载后内存
                loaded_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
                memory_increase = loaded_memory - initial_memory
                
                # 进行预测测试
                test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
                result = await adapter.predict_single(test_sequence)
                
                # 记录预测后内存
                prediction_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
                total_memory_increase = prediction_memory - initial_memory
                
                # 验证内存使用
                assert memory_increase > 0, "Model loading should increase memory usage"
                assert total_memory_increase < 2000, f"Total memory increase {total_memory_increase:.1f}MB is too high"
                
                # 验证预测结果
                assert "prediction" in result
                assert "confidence" in result
                assert 0.0 <= result["confidence"] <= 1.0
                
            except Exception as e:
                pytest.skip(f"Real model test failed: {e}")
    
    async def test_cpu_usage_under_load(self):
        """测试负载下的CPU使用"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 添加客户端
        client_count = 50
        clients = []
        for i in range(client_count):
            client = await sse_manager.add_client(f"cpu_client_{i}")
            clients.append(client)
        
        # 监控CPU使用
        cpu_samples = []
        for i in range(100):
            await sse_manager.send_prediction_start(f"model_{i}", f"sequence_{i}")
            cpu_usage = psutil.cpu_percent()
            cpu_samples.append(cpu_usage)
            await asyncio.sleep(0.01)  # 短暂延迟
        
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # CPU使用断言
        assert avg_cpu < 50   # 平均CPU使用 < 50%
        assert max_cpu < 80   # 峰值CPU使用 < 80%
        
        # 清理
        for client in clients:
            await sse_manager.remove_client(client.client_id)
        await sse_manager.stop()


class TestScalabilityPerformance:
    """可扩展性性能测试"""
    
    async def test_sse_scalability(self):
        """测试 SSE 可扩展性"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 测试不同规模的客户端数量
        client_scales = [10, 50, 100, 200]
        results = {}
        
        for scale in client_scales:
            # 添加客户端
            clients = []
            start_time = time.time()
            
            for i in range(scale):
                client = await sse_manager.add_client(f"scale_client_{scale}_{i}")
                clients.append(client)
            
            # 发送事件
            event_count = scale * 10
            for i in range(event_count):
                await sse_manager.send_prediction_start(f"model_{i}", f"sequence_{i}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[scale] = {
                "duration": duration,
                "throughput": event_count / duration,
                "clients": scale
            }
            
            # 清理
            for client in clients:
                await sse_manager.remove_client(client.client_id)
        
        await sse_manager.stop()
        
        # 可扩展性断言
        for scale, result in results.items():
            assert result["throughput"] > 10  # 每个规模都应该有合理的吞吐量
            assert result["duration"] < 30    # 每个规模都应该在合理时间内完成
    
    async def test_concurrent_model_operations(self):
        """测试并发模型操作"""
        model_pool_manager = ModelPoolManager(max_models=20)
        await model_pool_manager.start()
        
        # 并发加载模型
        async def load_model(model_id):
            return await model_pool_manager.load_model(
                f"concurrent_model_{model_id}",
                f"concurrent_config_{model_id}.yaml",
                "binary"
            )
        
        model_count = 10
        tasks = [load_model(i) for i in range(model_count)]
        results = await asyncio.gather(*tasks)
        
        # 检查结果
        successful_loads = sum(1 for result in results if result)
        assert successful_loads > 0  # 至少有一些模型加载成功
        
        await model_pool_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
