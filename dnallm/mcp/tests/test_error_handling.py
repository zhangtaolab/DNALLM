"""
Error handling tests for MCP Server

This module provides comprehensive error handling tests to ensure the system
gracefully handles various error conditions and edge cases.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
from unittest.mock import Mock, patch, AsyncMock

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


class TestConfigManagerErrorHandling:
    """配置管理器错误处理测试"""
    
    def test_invalid_config_file_path(self):
        """测试无效配置文件路径"""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent_config.yaml")
    
    def test_invalid_yaml_format(self):
        """测试无效YAML格式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            
            try:
                with pytest.raises(Exception):  # YAML解析错误
                    ConfigManager(f.name)
            finally:
                os.unlink(f.name)
    
    def test_missing_required_fields(self):
        """测试缺少必需字段"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # 缺少server字段
            config = {
                "mcp": {
                    "name": "Test Server"
                }
            }
            yaml.dump(config, f)
            f.flush()
            
            try:
                with pytest.raises(Exception):  # Pydantic验证错误
                    ConfigManager(f.name)
            finally:
                os.unlink(f.name)
    
    def test_invalid_model_config(self):
        """测试无效模型配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 1,
                    "log_level": "info"
                },
                "mcp": {
                    "name": "Test Server",
                    "version": "1.0.0",
                    "description": "Test"
                },
                "models": [
                    {
                        "name": "invalid_model",
                        # 缺少必需字段
                        "enabled": True
                    }
                ]
            }
            yaml.dump(config, f)
            f.flush()
            
            try:
                with pytest.raises(Exception):  # Pydantic验证错误
                    ConfigManager(f.name)
            finally:
                os.unlink(f.name)


class TestTaskRouterErrorHandling:
    """任务路由器错误处理测试"""
    
    def test_invalid_task_type(self):
        """测试无效任务类型"""
        task_router = TaskRouter()
        
        with pytest.raises(ValueError, match="Unknown task type"):
            task_router.get_task_type("invalid_task_type")
    
    def test_invalid_task_config(self):
        """测试无效任务配置"""
        task_router = TaskRouter()
        
        # 缺少必需字段
        invalid_config = {
            "task_type": "binary"
            # 缺少num_labels
        }
        
        with pytest.raises(Exception):
            task_router.create_task_config(invalid_config)
    
    def test_missing_task_config(self):
        """测试缺少任务配置"""
        task_manager = TaskRouterManager()
        
        # 尝试处理未注册的模型
        mock_result = {"prediction": 1, "probabilities": [0.3, 0.7]}
        
        with pytest.raises(ValueError, match="No task config found"):
            asyncio.run(task_manager.process_prediction(
                mock_result, "ATCGATCG", "nonexistent_model"
            ))
    
    def test_invalid_prediction_result_format(self):
        """测试无效预测结果格式"""
        task_router = TaskRouter()
        task_manager = TaskRouterManager()
        
        # 注册任务配置
        config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Class 0", "Class 1"],
            "threshold": 0.5
        }
        task_manager.register_task_config("test_model", config)
        
        # 测试无效结果格式
        invalid_results = [
            {},  # 空结果
            {"prediction": "invalid"},  # 无效预测值
            {"prediction": 1, "probabilities": "invalid"},  # 无效概率格式
            {"prediction": [1, 2, 3]},  # 二分类任务的多值预测
        ]
        
        for invalid_result in invalid_results:
            with pytest.raises(Exception):
                asyncio.run(task_manager.process_prediction(
                    invalid_result, "ATCGATCG", "test_model"
                ))


class TestSSEManagerErrorHandling:
    """SSE 管理器错误处理测试"""
    
    async def test_client_connection_errors(self):
        """测试客户端连接错误"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 测试重复客户端ID
        client1 = await sse_manager.add_client("duplicate_id")
        client2 = await sse_manager.add_client("duplicate_id")
        
        # 应该只有一个客户端
        assert sse_manager.get_client_count() == 1
        
        # 测试移除不存在的客户端
        await sse_manager.remove_client("nonexistent_client")
        # 不应该抛出异常
        
        await sse_manager.stop()
    
    async def test_event_sending_errors(self):
        """测试事件发送错误"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 测试发送事件到不存在的客户端
        await sse_manager.send_prediction_start("model", "sequence", "nonexistent_client")
        # 不应该抛出异常
        
        # 测试无效事件数据
        await sse_manager.send_prediction_complete("model", None)
        # 不应该抛出异常
        
        await sse_manager.stop()
    
    async def test_sse_manager_lifecycle_errors(self):
        """测试 SSE 管理器生命周期错误"""
        sse_manager = SSEManager()
        
        # 测试重复启动
        await sse_manager.start()
        await sse_manager.start()  # 应该被忽略
        
        # 测试重复停止
        await sse_manager.stop()
        await sse_manager.stop()  # 应该被忽略
        
        # 测试在停止状态下发送事件
        await sse_manager.send_prediction_start("model", "sequence")
        # 不应该抛出异常


class TestModelPoolErrorHandling:
    """模型池错误处理测试"""
    
    async def test_model_loading_errors(self):
        """测试模型加载错误"""
        model_pool_manager = ModelPoolManager(max_models=2)
        await model_pool_manager.start()
        
        # 测试加载不存在的模型配置
        success = await model_pool_manager.load_model(
            "nonexistent_model",
            "nonexistent_config.yaml",
            "binary"
        )
        assert not success
        
        # 测试超过最大模型数限制
        for i in range(3):  # 超过max_models=2
            success = await model_pool_manager.load_model(
                f"model_{i}",
                f"config_{i}.yaml",
                "binary"
            )
            # 前两个应该成功，第三个应该失败
            if i < 2:
                assert success
            else:
                assert not success
        
        await model_pool_manager.stop()
    
    async def test_model_access_errors(self):
        """测试模型访问错误"""
        model_pool_manager = ModelPoolManager()
        await model_pool_manager.start()
        
        # 测试获取不存在的模型
        model_instance = await model_pool_manager.get_model("nonexistent_model")
        assert model_instance is None
        
        # 测试释放不存在的模型
        await model_pool_manager.release_model("nonexistent_model_id")
        # 不应该抛出异常
        
        await model_pool_manager.stop()
    
    async def test_model_pool_lifecycle_errors(self):
        """测试模型池生命周期错误"""
        model_pool_manager = ModelPoolManager()
        
        # 测试重复启动
        await model_pool_manager.start()
        await model_pool_manager.start()  # 应该被忽略
        
        # 测试重复停止
        await model_pool_manager.stop()
        await model_pool_manager.stop()  # 应该被忽略


class TestDNAPredictorErrorHandling:
    """DNA 预测器错误处理测试"""
    
    def test_invalid_model_path(self):
        """测试无效模型路径"""
        adapter = DNAPredictorAdapter("test_model", "nonexistent_config.yaml")
        
        # 测试获取模型信息（未加载）
        model_info = adapter.get_model_info()
        assert model_info["loaded"] is False
    
    async def test_prediction_without_loaded_model(self):
        """测试未加载模型时的预测"""
        adapter = DNAPredictorAdapter("test_model", "nonexistent_config.yaml")
        
        # 测试单序列预测
        with pytest.raises(RuntimeError, match="Model.*not loaded"):
            await adapter.predict_single("ATCGATCG")
        
        # 测试批量预测
        with pytest.raises(RuntimeError, match="Model.*not loaded"):
            await adapter.predict_batch(["ATCGATCG", "GCTAGCTA"])
    
    async def test_invalid_sequence_input(self):
        """测试无效序列输入"""
        adapter = DNAPredictorAdapter("test_model", "nonexistent_config.yaml")
        
        # 测试空序列
        with pytest.raises(Exception):
            await adapter.predict_single("")
        
        # 测试无效字符
        with pytest.raises(Exception):
            await adapter.predict_single("INVALID_CHARS")
        
        # 测试过长序列
        long_sequence = "A" * 10000
        with pytest.raises(Exception):
            await adapter.predict_single(long_sequence)
    
    async def test_predictor_pool_errors(self):
        """测试预测器池错误"""
        pool = DNAPredictorPool(max_models=1)
        
        # 测试获取不存在的预测器
        predictor = await pool.get_predictor("nonexistent_model", "nonexistent_config.yaml")
        assert predictor is not None  # 应该创建新的预测器
        
        await pool.shutdown()


class TestValidatorErrorHandling:
    """验证器错误处理测试"""
    
    def test_dna_sequence_validation_errors(self):
        """测试DNA序列验证错误"""
        invalid_sequences = [
            "",  # 空序列
            "INVALID_CHARS",  # 无效字符
            "ATCGX",  # 包含无效字符
            "123456",  # 数字
            "ATCG " * 1000,  # 包含空格的长序列
            None,  # None值
        ]
        
        for seq in invalid_sequences:
            if seq is None:
                with pytest.raises(TypeError):
                    validate_dna_sequence(seq)
            else:
                result = validate_dna_sequence(seq)
                assert not result['is_valid']
    
    def test_dna_sequence_validation_edge_cases(self):
        """测试DNA序列验证边界情况"""
        edge_cases = [
            "A",  # 单字符
            "ATCG",  # 最短有效序列
            "ATCG" * 1000,  # 长序列
            "atcg",  # 小写字母
            "ATCGatcg",  # 混合大小写
        ]
        
        for seq in edge_cases:
            assert validate_dna_sequence(seq)
    
    def test_formatter_error_handling(self):
        """测试格式化器错误处理"""
        invalid_results = [
            None,  # None结果
            {},  # 空结果
            {"prediction": "invalid"},  # 无效预测值
            {"prediction": 1, "probabilities": "invalid"},  # 无效概率
        ]
        
        for result in invalid_results:
            with pytest.raises(Exception):
                format_prediction_result(result)


class TestConcurrentErrorHandling:
    """并发错误处理测试"""
    
    async def test_concurrent_config_access(self):
        """测试并发配置访问"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 1,
                    "log_level": "info"
                },
                "mcp": {
                    "name": "Test Server",
                    "version": "1.0.0",
                    "description": "Test"
                },
                "models": []
            }
            yaml.dump(config, f)
            f.flush()
            
            try:
                # 并发创建配置管理器
                async def create_config_manager():
                    return ConfigManager(f.name)
                
                tasks = [create_config_manager() for _ in range(10)]
                config_managers = await asyncio.gather(*tasks)
                
                # 所有配置管理器都应该成功创建
                assert len(config_managers) == 10
                for cm in config_managers:
                    assert cm.config is not None
                    
            finally:
                os.unlink(f.name)
    
    async def test_concurrent_sse_operations(self):
        """测试并发SSE操作"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 并发添加和移除客户端
        async def add_and_remove_client(client_id):
            client = await sse_manager.add_client(client_id)
            await asyncio.sleep(0.1)
            await sse_manager.remove_client(client_id)
            return client_id
        
        tasks = [add_and_remove_client(f"client_{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # 所有操作都应该成功
        assert len(results) == 20
        assert sse_manager.get_client_count() == 0
        
        await sse_manager.stop()
    
    async def test_concurrent_model_operations(self):
        """测试并发模型操作"""
        model_pool_manager = ModelPoolManager(max_models=5)
        await model_pool_manager.start()
        
        # 并发加载模型
        async def load_model(model_id):
            return await model_pool_manager.load_model(
                f"concurrent_model_{model_id}",
                f"concurrent_config_{model_id}.yaml",
                "binary"
            )
        
        tasks = [load_model(i) for i in range(10)]  # 超过max_models
        results = await asyncio.gather(*tasks)
        
        # 一些应该成功，一些应该失败
        successful_loads = sum(1 for result in results if result)
        assert successful_loads <= 5  # 不超过max_models
        assert successful_loads > 0   # 至少有一些成功
        
        await model_pool_manager.stop()


class TestResourceExhaustionErrorHandling:
    """资源耗尽错误处理测试"""
    
    async def test_memory_exhaustion_handling(self):
        """测试内存耗尽处理"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 添加大量客户端（模拟内存压力）
        clients = []
        try:
            for i in range(1000):  # 大量客户端
                client = await sse_manager.add_client(f"memory_client_{i}")
                clients.append(client)
                
                # 每100个客户端检查一次内存
                if i % 100 == 0:
                    import psutil
                    memory = psutil.virtual_memory()
                    if memory.percent > 90:  # 内存使用超过90%
                        break
        except Exception as e:
            # 应该优雅地处理内存不足
            assert "memory" in str(e).lower() or "resource" in str(e).lower()
        
        # 清理
        for client in clients:
            try:
                await sse_manager.remove_client(client.client_id)
            except:
                pass
        
        await sse_manager.stop()
    
    async def test_connection_limit_handling(self):
        """测试连接限制处理"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 尝试添加超过限制的客户端
        clients = []
        max_clients = 100
        
        for i in range(max_clients + 10):  # 超过限制
            try:
                client = await sse_manager.add_client(f"limit_client_{i}")
                clients.append(client)
            except Exception as e:
                # 应该优雅地处理连接限制
                assert "limit" in str(e).lower() or "connection" in str(e).lower()
                break
        
        # 清理
        for client in clients:
            try:
                await sse_manager.remove_client(client.client_id)
            except:
                pass
        
        await sse_manager.stop()


class TestNetworkErrorHandling:
    """网络错误处理测试"""
    
    async def test_network_timeout_handling(self):
        """测试网络超时处理"""
        sse_manager = SSEManager(heartbeat_interval=1)
        await sse_manager.start()
        
        # 添加客户端
        client = await sse_manager.add_client("timeout_client")
        
        # 模拟网络延迟
        with patch('asyncio.sleep', side_effect=asyncio.TimeoutError):
            try:
                await sse_manager.send_prediction_start("model", "sequence", "timeout_client")
            except asyncio.TimeoutError:
                # 应该优雅地处理超时
                pass
        
        await sse_manager.remove_client("timeout_client")
        await sse_manager.stop()
    
    async def test_connection_drop_handling(self):
        """测试连接断开处理"""
        sse_manager = SSEManager()
        await sse_manager.start()
        
        # 添加客户端
        client = await sse_manager.add_client("drop_client")
        
        # 模拟连接断开
        client.connected = False
        
        # 尝试发送事件到断开的客户端
        await sse_manager.send_prediction_start("model", "sequence", "drop_client")
        # 不应该抛出异常
        
        await sse_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
