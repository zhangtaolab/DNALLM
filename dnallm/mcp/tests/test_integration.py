"""
Integration tests for MCP Server

This module provides comprehensive integration tests that verify the interaction
between different components of the MCP server system.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.model_manager import ModelManager
from dnallm.mcp.model_pool_manager import ModelPoolManager
from dnallm.mcp.dna_predictor_adapter import DNAPredictorAdapter, DNAPredictorPool
from dnallm.mcp.task_router import TaskRouter, TaskRouterManager
from dnallm.mcp.sse_manager import SSEManager
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
from dnallm.mcp.utils.validators import validate_dna_sequence
from dnallm.mcp.utils.formatters import format_prediction_result


class TestMCPIntegration:
    """MCP 服务器集成测试"""
    
    @pytest.fixture
    async def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    async def sample_config(self, temp_config_dir):
        """创建示例配置"""
        config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info"
            },
            "mcp": {
                "name": "Test MCP Server",
                "version": "1.0.0",
                "description": "Test server"
            },
            "models": [
                {
                    "name": "test_promoter_model",
                    "model_name": "Plant DNABERT BPE promoter",
                    "config_path": f"{temp_config_dir}/test_promoter_config.yaml",
                    "enabled": True,
                    "max_concurrent_requests": 5,
                    "task_type": "binary",
                    "description": "Test promoter model"
                }
            ],
            "sse": {
                "heartbeat_interval": 30,
                "max_connections": 10
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 创建模型配置文件
        model_config = {
            "task": {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["Not promoter", "Core promoter"],
                "threshold": 0.5
            },
            "inference": {
                "batch_size": 16,
                "max_length": 512,
                "device": "cpu",
                "num_workers": 1
            },
            "model": {
                "name": "Plant DNABERT BPE promoter",
                "path": "zhangtaolab/plant-dnabert-BPE-promoter",
                "source": "huggingface",
                "trust_remote_code": True
            }
        }
        
        config_path = f"{temp_config_dir}/mcp_server_config.yaml"
        model_config_path = f"{temp_config_dir}/test_promoter_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        return config_path
    
    async def test_config_manager_integration(self, sample_config):
        """测试配置管理器集成"""
        config_manager = ConfigManager(sample_config)
        
        # 测试配置加载
        assert config_manager.config is not None
        assert config_manager.config.server.host == "0.0.0.0"
        assert config_manager.config.server.port == 8000
        
        # 测试模型配置获取
        models = config_manager.get_models()
        assert len(models) == 1
        assert models[0].name == "test_promoter_model"
        
        # 测试模型能力查询
        capabilities = config_manager.get_model_capabilities("test_promoter_model")
        assert capabilities is not None
        assert capabilities["task_type"] == "binary"
    
    async def test_model_manager_integration(self, sample_config):
        """测试模型管理器集成"""
        config_manager = ConfigManager(sample_config)
        model_manager = ModelManager(config_manager)
        
        # 测试模型管理器初始化
        assert model_manager.config_manager == config_manager
        
        # 测试模型信息获取
        models = model_manager.get_available_models()
        assert len(models) == 1
        assert models[0]["name"] == "test_promoter_model"
    
    async def test_task_router_integration(self):
        """测试任务路由器集成"""
        task_router = TaskRouter()
        task_manager = TaskRouterManager()
        
        # 测试任务类型识别
        binary_type = task_router.get_task_type("binary")
        assert binary_type.value == "binary"
        
        # 测试任务配置创建
        config_dict = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Class 0", "Class 1"],
            "threshold": 0.5
        }
        
        task_config = task_router.create_task_config(config_dict)
        assert task_config.task_type.value == "binary"
        assert task_config.num_labels == 2
        assert len(task_config.label_names) == 2
        
        # 测试任务管理器注册
        task_manager.register_task_config("test_model", config_dict)
        registered_config = task_manager.get_task_config("test_model")
        assert registered_config is not None
        assert registered_config.task_type.value == "binary"
    
    async def test_sse_manager_integration(self):
        """测试 SSE 管理器集成"""
        sse_manager = SSEManager(heartbeat_interval=5)
        
        # 启动 SSE 管理器
        await sse_manager.start()
        assert sse_manager.is_running
        
        # 测试客户端连接
        client = await sse_manager.add_client("test_client")
        assert client.client_id == "test_client"
        assert sse_manager.get_client_count() == 1
        
        # 测试事件发送
        await sse_manager.send_prediction_start("test_model", "ATCGATCG")
        
        # 测试客户端信息获取
        client_info = sse_manager.get_client_info()
        assert len(client_info) == 1
        assert client_info[0]["client_id"] == "test_client"
        
        # 清理
        await sse_manager.remove_client("test_client")
        await sse_manager.stop()
    
    async def test_model_pool_manager_integration(self, sample_config):
        """测试模型池管理器集成"""
        config_manager = ConfigManager(sample_config)
        pool_manager = ModelPoolManager(max_models=2, health_check_interval=10)
        
        # 启动模型池管理器
        await pool_manager.start()
        assert pool_manager.is_running
        
        # 测试模型池状态
        status = pool_manager.get_pool_status()
        assert status["total_models"] == 0
        assert status["is_running"] is True
        
        # 清理
        await pool_manager.stop()
    
    async def test_dna_predictor_adapter_integration(self, sample_config):
        """测试 DNA 预测器适配器集成"""
        # 创建预测器适配器
        adapter = DNAPredictorAdapter(
            "Plant DNABERT BPE promoter",
            f"{Path(sample_config).parent}/test_promoter_config.yaml"
        )
        
        # 测试模型信息获取（不加载模型）
        model_info = adapter.get_model_info()
        assert model_info["model_name"] == "Plant DNABERT BPE promoter"
        assert model_info["is_loaded"] is False
        
        # 测试预测器池
        pool = DNAPredictorPool(max_models=2)
        
        # 测试池信息
        pool_info = pool.get_pool_info()
        assert pool_info["total_models"] == 0
        assert pool_info["max_models"] == 2
        
        # 清理
        await pool.shutdown()
    
    async def test_end_to_end_prediction_flow(self, sample_config):
        """测试端到端预测流程"""
        # 初始化所有组件
        config_manager = ConfigManager(sample_config)
        task_manager = TaskRouterManager()
        sse_manager = SSEManager()
        
        # 启动 SSE 管理器
        await sse_manager.start()
        
        # 注册任务配置
        task_config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Not promoter", "Core promoter"],
            "threshold": 0.5
        }
        task_manager.register_task_config("test_promoter_model", task_config)
        
        # 测试序列验证
        test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        is_valid = validate_dna_sequence(test_sequence)
        assert is_valid
        
        # 模拟预测结果格式化
        mock_result = {
            "sequence": test_sequence,
            "task_type": "binary",
            "model_name": "test_promoter_model",
            "prediction": 1,
            "confidence": 0.85,
            "probabilities": {
                "Not promoter": 0.15,
                "Core promoter": 0.85
            }
        }
        
        formatted_result = format_prediction_result(mock_result)
        assert formatted_result["sequence"] == test_sequence
        assert formatted_result["prediction"] == 1
        assert formatted_result["confidence"] == 0.85
        
        # 测试 SSE 事件发送
        await sse_manager.send_prediction_start("test_promoter_model", test_sequence)
        await sse_manager.send_prediction_complete("test_promoter_model", formatted_result)
        
        # 清理
        await sse_manager.stop()
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_end_to_end_flow(self, temp_config_dir):
        """测试真实模型端到端预测流程"""
        # 创建真实模型配置
        model_info = {
            "name": "Plant DNABERT BPE promoter",
            "model_path": "zhangtaolab/plant-dnabert-BPE-promoter",
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Not promoter", "Core promoter"],
            "threshold": 0.5
        }
        
        # 创建模型配置文件
        model_config = {
            "task": {
                "task_type": model_info["task_type"],
                "num_labels": model_info["num_labels"],
                "label_names": model_info["label_names"],
                "threshold": model_info["threshold"]
            },
            "inference": {
                "batch_size": 4,
                "max_length": 256,
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
        
        config_path = os.path.join(temp_config_dir, "real_model_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        # 创建服务器配置
        server_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info"
            },
            "mcp": {
                "name": "Test MCP Server",
                "version": "1.0.0",
                "description": "Test server"
            },
            "models": [
                {
                    "name": "real_promoter_model",
                    "model_name": model_info["name"],
                    "config_path": config_path,
                    "enabled": True,
                    "max_concurrent_requests": 5,
                    "task_type": "binary",
                    "description": "Real promoter model"
                }
            ],
            "sse": {
                "heartbeat_interval": 30,
                "max_connections": 10
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        server_config_path = os.path.join(temp_config_dir, "server_config.yaml")
        with open(server_config_path, 'w') as f:
            yaml.dump(server_config, f)
        
        # 初始化组件
        config_manager = ConfigManager(server_config_path)
        task_manager = TaskRouterManager()
        sse_manager = SSEManager()
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        try:
            # 启动 SSE 管理器
            await sse_manager.start()
            
            # 注册任务配置
            task_config = {
                "task_type": model_info["task_type"],
                "num_labels": model_info["num_labels"],
                "label_names": model_info["label_names"],
                "threshold": model_info["threshold"]
            }
            task_manager.register_task_config("real_promoter_model", task_config)
            
            # 加载真实模型
            success = await adapter.load_model()
            if not success:
                pytest.skip(f"Failed to load real model {model_info['name']}")
            
            # 测试序列
            test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
            is_valid = validate_dna_sequence(test_sequence)
            assert is_valid
            
            # 发送预测开始事件
            await sse_manager.send_prediction_start("real_promoter_model", test_sequence)
            
            # 进行真实预测
            result = await adapter.predict_single(test_sequence)
            
            # 验证预测结果
            assert "sequence" in result
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["sequence"] == test_sequence
            assert result["task_type"] == model_info["task_type"]
            assert result["model_name"] == model_info["name"]
            assert isinstance(result["prediction"], int)
            assert 0 <= result["prediction"] <= 1
            assert 0.0 <= result["confidence"] <= 1.0
            
            # 通过任务路由器处理结果
            processed_result = await task_manager.process_prediction(
                result, test_sequence, "real_promoter_model"
            )
            
            # 验证处理后的结果
            assert processed_result.sequence == test_sequence
            assert processed_result.task_type.value == model_info["task_type"]
            assert processed_result.model_name == "real_promoter_model"
            
            # 发送预测完成事件
            await sse_manager.send_prediction_complete("real_promoter_model", result)
            
        finally:
            await sse_manager.stop()
    
    async def test_multi_model_integration(self, sample_config):
        """测试多模型集成"""
        config_manager = ConfigManager(sample_config)
        task_manager = TaskRouterManager()
        
        # 注册多个任务配置
        task_configs = [
            {
                "name": "binary_model",
                "config": {
                    "task_type": "binary",
                    "num_labels": 2,
                    "label_names": ["Class 0", "Class 1"],
                    "threshold": 0.5
                }
            },
            {
                "name": "multiclass_model", 
                "config": {
                    "task_type": "multiclass",
                    "num_labels": 3,
                    "label_names": ["Class A", "Class B", "Class C"],
                    "threshold": None
                }
            },
            {
                "name": "regression_model",
                "config": {
                    "task_type": "regression",
                    "num_labels": 1,
                    "label_names": [],
                    "threshold": None
                }
            }
        ]
        
        for task_config in task_configs:
            task_manager.register_task_config(task_config["name"], task_config["config"])
        
        # 测试按任务类型获取模型
        binary_models = task_manager.get_models_by_task_type(task_manager.router.get_task_type("binary"))
        assert len(binary_models) == 1
        assert "binary_model" in binary_models
        
        multiclass_models = task_manager.get_models_by_task_type(task_manager.router.get_task_type("multiclass"))
        assert len(multiclass_models) == 1
        assert "multiclass_model" in multiclass_models
        
        regression_models = task_manager.get_models_by_task_type(task_manager.router.get_task_type("regression"))
        assert len(regression_models) == 1
        assert "regression_model" in regression_models
    
    async def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 测试无效配置
        with pytest.raises(Exception):
            ConfigManager("nonexistent_config.yaml")
        
        # 测试无效任务类型
        task_router = TaskRouter()
        with pytest.raises(ValueError):
            task_router.get_task_type("invalid_task_type")
        
        # 测试无效序列
        invalid_sequences = [
            "",  # 空序列
            "INVALID_CHARS",  # 无效字符
            "A" * 10000,  # 过长序列
        ]
        
        for seq in invalid_sequences:
            is_valid = validate_dna_sequence(seq)
            assert not is_valid
        
        # 测试 SSE 管理器错误处理
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 发送错误事件
        await sse_manager.send_prediction_error("test_model", "Test error message")
        
        await sse_manager.stop()
    
    async def test_concurrent_operations(self, sample_config):
        """测试并发操作"""
        config_manager = ConfigManager(sample_config)
        sse_manager = SSEManager()
        task_manager = TaskRouterManager()
        
        await sse_manager.start()
        
        # 注册任务配置
        task_config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Class 0", "Class 1"],
            "threshold": 0.5
        }
        task_manager.register_task_config("test_model", task_config)
        
        # 并发添加多个客户端
        async def add_client(client_id):
            return await sse_manager.add_client(client_id)
        
        client_ids = [f"client_{i}" for i in range(5)]
        clients = await asyncio.gather(*[add_client(client_id) for client_id in client_ids])
        
        assert len(clients) == 5
        assert sse_manager.get_client_count() == 5
        
        # 并发发送事件
        async def send_event(client_id):
            await sse_manager.send_prediction_start("test_model", "ATCG", client_id)
        
        await asyncio.gather(*[send_event(client_id) for client_id in client_ids])
        
        # 清理
        for client_id in client_ids:
            await sse_manager.remove_client(client_id)
        
        await sse_manager.stop()
    
    async def test_config_generator_integration(self):
        """测试配置生成器集成"""
        # 使用相对路径到 model_info.yaml
        model_info_path = "../../models/model_info.yaml"
        
        if os.path.exists(model_info_path):
            generator = MCPModelConfigGenerator(model_info_path)
            
            # 测试模型信息加载
            assert generator.model_info is not None
            assert "finetuned" in generator.model_info
            
            # 测试按任务类型获取模型
            binary_models = generator.get_models_by_task_type("binary")
            assert len(binary_models) > 0
            
            multiclass_models = generator.get_models_by_task_type("multiclass")
            assert len(multiclass_models) > 0
            
            regression_models = generator.get_models_by_task_type("regression")
            assert len(regression_models) > 0
            
            # 测试配置生成
            selected_models = [
                "Plant DNABERT BPE promoter",
                "Plant DNABERT BPE conservation"
            ]
            
            config = generator.generate_mcp_server_config(selected_models)
            assert "server" in config
            assert "models" in config
            assert len(config["models"]) == 2
        else:
            pytest.skip("model_info.yaml not found, skipping config generator integration test")


class TestPerformanceIntegration:
    """性能集成测试"""
    
    async def test_sse_performance(self):
        """测试 SSE 性能"""
        sse_manager = SSEManager(heartbeat_interval=1)
        await sse_manager.start()
        
        # 添加多个客户端
        client_count = 10
        clients = []
        for i in range(client_count):
            client = await sse_manager.add_client(f"perf_client_{i}")
            clients.append(client)
        
        # 测试并发事件发送性能
        start_time = time.time()
        
        async def send_events():
            for i in range(100):
                await sse_manager.send_prediction_start(f"model_{i}", f"sequence_{i}")
        
        await send_events()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能断言：100个事件应该在合理时间内完成
        assert duration < 5.0  # 5秒内完成
        
        # 清理
        for client in clients:
            await sse_manager.remove_client(client.client_id)
        
        await sse_manager.stop()
    
    async def test_task_router_performance(self):
        """测试任务路由器性能"""
        task_manager = TaskRouterManager()
        
        # 注册大量任务配置
        task_count = 100
        for i in range(task_count):
            config = {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": [f"Class_{i}_0", f"Class_{i}_1"],
                "threshold": 0.5
            }
            task_manager.register_task_config(f"model_{i}", config)
        
        # 测试批量处理性能
        start_time = time.time()
        
        # 模拟批量预测结果处理
        results = []
        for i in range(1000):
            mock_result = {
                "prediction": i % 2,
                "probabilities": [0.3, 0.7] if i % 2 == 1 else [0.7, 0.3]
            }
            results.append(mock_result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能断言：1000个结果处理应该在合理时间内完成
        assert duration < 1.0  # 1秒内完成
        assert len(results) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
