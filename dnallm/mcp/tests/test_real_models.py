"""
Real Model Tests for MCP Server

This module provides tests using real DNA prediction models instead of mocks.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.model_manager import ModelManager
from dnallm.mcp.dna_predictor_adapter import DNAPredictorAdapter, DNAPredictorPool
from dnallm.mcp.task_router import TaskRouter, TaskRouterManager
from dnallm.mcp.sse_manager import SSEManager
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
from dnallm.mcp.utils.validators import validate_dna_sequence


class TestRealModels:
    """真实模型测试"""
    
    @pytest.fixture
    def real_model_configs(self):
        """真实模型配置"""
        return [
            {
                "name": "Plant DNABERT BPE promoter",
                "model_path": "zhangtaolab/plant-dnabert-BPE-promoter",
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["Not promoter", "Core promoter"],
                "threshold": 0.5
            },
            {
                "name": "Plant DNABERT BPE conservation", 
                "model_path": "zhangtaolab/plant-dnabert-BPE-conservation",
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["Not conserved", "Conserved"],
                "threshold": 0.5
            }
        ]
    
    @pytest.fixture
    async def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def create_real_model_config(self, model_info: Dict[str, Any], temp_dir: str) -> str:
        """创建真实模型配置文件"""
        config = {
            "task": {
                "task_type": model_info["task_type"],
                "num_labels": model_info["num_labels"],
                "label_names": model_info["label_names"],
                "threshold": model_info["threshold"]
            },
            "inference": {
                "batch_size": 4,  # 小批量以节省内存
                "max_length": 256,  # 较短序列以节省内存
                "device": "cpu",  # 使用CPU以避免GPU依赖
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
        
        config_path = os.path.join(temp_dir, f"{model_info['name'].lower().replace(' ', '_')}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_loading(self, real_model_configs, temp_config_dir):
        """测试真实模型加载"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        # 测试模型加载
        success = await adapter.load_model()
        assert success, f"Failed to load model {model_info['name']}"
        assert adapter.is_loaded, "Model should be marked as loaded"
        
        # 测试模型信息获取
        model_info_result = adapter.get_model_info()
        assert model_info_result["model_name"] == model_info["name"]
        assert model_info_result["task_type"] == model_info["task_type"]
        assert model_info_result["is_loaded"] is True
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_prediction(self, real_model_configs, temp_config_dir):
        """测试真实模型预测"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        # 加载模型
        success = await adapter.load_model()
        assert success, f"Failed to load model {model_info['name']}"
        
        # 测试序列
        test_sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",  # 64bp
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",  # 64bp
        ]
        
        for sequence in test_sequences:
            # 验证序列
            assert validate_dna_sequence(sequence), f"Invalid sequence: {sequence}"
            
            # 进行预测
            result = await adapter.predict_single(sequence)
            
            # 验证结果
            assert "sequence" in result
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["sequence"] == sequence
            assert result["task_type"] == model_info["task_type"]
            assert result["model_name"] == model_info["name"]
            assert isinstance(result["prediction"], int)
            assert 0 <= result["prediction"] <= 1  # 二分类结果
            assert 0.0 <= result["confidence"] <= 1.0
            assert len(result["probabilities"]) == model_info["num_labels"]
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_batch_prediction(self, real_model_configs, temp_config_dir):
        """测试真实模型批量预测"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        # 加载模型
        success = await adapter.load_model()
        assert success, f"Failed to load model {model_info['name']}"
        
        # 测试序列
        test_sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        ]
        
        # 批量预测
        results = await adapter.predict_batch(test_sequences)
        
        # 验证结果
        assert len(results) == len(test_sequences)
        
        for i, result in enumerate(results):
            assert result["sequence"] == test_sequences[i]
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert isinstance(result["prediction"], int)
            assert 0 <= result["prediction"] <= 1
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_with_task_router(self, real_model_configs, temp_config_dir):
        """测试真实模型与任务路由器集成"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        task_manager = TaskRouterManager()
        
        # 注册任务配置
        task_config = {
            "task_type": model_info["task_type"],
            "num_labels": model_info["num_labels"],
            "label_names": model_info["label_names"],
            "threshold": model_info["threshold"]
        }
        task_manager.register_task_config(model_info["name"], task_config)
        
        # 加载模型
        success = await adapter.load_model()
        assert success, f"Failed to load model {model_info['name']}"
        
        # 测试序列
        test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        
        # 进行预测
        raw_result = await adapter.predict_single(test_sequence)
        
        # 通过任务路由器处理结果
        processed_result = await task_manager.process_prediction(
            raw_result, test_sequence, model_info["name"]
        )
        
        # 验证处理后的结果
        assert processed_result.sequence == test_sequence
        assert processed_result.task_type.value == model_info["task_type"]
        assert processed_result.model_name == model_info["name"]
        assert isinstance(processed_result.prediction, int)
        assert 0 <= processed_result.prediction <= 1
        assert 0.0 <= processed_result.confidence <= 1.0
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_pool(self, real_model_configs, temp_config_dir):
        """测试真实模型池"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        pool = DNAPredictorPool(max_models=2)
        
        try:
            # 测试获取预测器
            predictor = await pool.get_predictor(model_info["name"], config_path)
            assert predictor is not None
            
            # 测试预测
            test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
            result = await pool.predict_single(model_info["name"], config_path, test_sequence)
            
            assert result is not None
            assert "sequence" in result
            assert "prediction" in result
            assert "confidence" in result
            
            # 测试池信息
            pool_info = pool.get_pool_info()
            assert pool_info["total_models"] >= 1
            assert pool_info["max_models"] == 2
            
        finally:
            await pool.shutdown()
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_multiple_real_models(self, real_model_configs, temp_config_dir):
        """测试多个真实模型"""
        results = []
        
        for model_info in real_model_configs[:2]:  # 只测试前两个模型
            config_path = self.create_real_model_config(model_info, temp_config_dir)
            adapter = DNAPredictorAdapter(model_info["name"], config_path)
            
            try:
                # 加载模型
                success = await adapter.load_model()
                if not success:
                    print(f"Warning: Failed to load model {model_info['name']}")
                    continue
                
                # 测试预测
                test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
                result = await adapter.predict_single(test_sequence)
                
                results.append({
                    "model": model_info["name"],
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                print(f"Error testing model {model_info['name']}: {e}")
                results.append({
                    "model": model_info["name"],
                    "error": str(e),
                    "success": False
                })
        
        # 验证至少有一个模型成功
        successful_models = [r for r in results if r["success"]]
        assert len(successful_models) > 0, "At least one model should work"
        
        # 验证成功模型的结果
        for result in successful_models:
            assert "prediction" in result["result"]
            assert "confidence" in result["result"]
            assert "probabilities" in result["result"]
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_with_sse(self, real_model_configs, temp_config_dir):
        """测试真实模型与SSE集成"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        sse_manager = SSEManager()
        
        try:
            # 启动SSE管理器
            await sse_manager.start()
            
            # 加载模型
            success = await adapter.load_model()
            assert success, f"Failed to load model {model_info['name']}"
            
            # 添加客户端
            client = await sse_manager.add_client("test_client")
            assert client is not None
            
            # 测试序列
            test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
            
            # 发送预测开始事件
            await sse_manager.send_prediction_start(model_info["name"], test_sequence, "test_client")
            
            # 进行预测
            result = await adapter.predict_single(test_sequence)
            
            # 发送预测完成事件
            await sse_manager.send_prediction_complete(model_info["name"], result, "test_client")
            
            # 验证结果
            assert "prediction" in result
            assert "confidence" in result
            
        finally:
            await sse_manager.stop()
    
    @pytest.mark.slow
    @pytest.mark.real_model
    async def test_real_model_performance(self, real_model_configs, temp_config_dir):
        """测试真实模型性能"""
        model_info = real_model_configs[0]  # 使用第一个模型
        config_path = self.create_real_model_config(model_info, temp_config_dir)
        
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        # 加载模型
        success = await adapter.load_model()
        assert success, f"Failed to load model {model_info['name']}"
        
        # 测试序列
        test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        
        # 性能测试
        num_requests = 5  # 减少请求数以节省时间
        start_time = time.time()
        
        results = []
        for i in range(num_requests):
            result = await adapter.predict_single(test_sequence)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        
        # 验证性能
        assert total_time < 60, f"Total time {total_time:.2f}s is too long"
        assert avg_time < 10, f"Average time {avg_time:.2f}s is too long"
        
        # 验证所有结果
        for result in results:
            assert "prediction" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0


class TestRealModelConfigGenerator:
    """真实模型配置生成器测试"""
    
    @pytest.mark.slow
    @pytest.mark.real_model
    def test_generate_real_configs(self):
        """测试生成真实模型配置"""
        model_info_path = "../../models/model_info.yaml"
        
        if not os.path.exists(model_info_path):
            pytest.skip("model_info.yaml not found")
        
        generator = MCPModelConfigGenerator(model_info_path)
        
        # 测试获取真实模型
        binary_models = generator.get_models_by_task_type("binary")
        assert len(binary_models) > 0, "Should have binary models"
        
        # 选择第一个模型进行测试
        test_model = binary_models[0]
        assert "name" in test_model
        assert "model" in test_model
        assert "task" in test_model
        
        # 测试生成配置
        selected_models = [test_model["name"]]
        config = generator.generate_mcp_server_config(selected_models)
        
        assert "server" in config
        assert "models" in config
        assert len(config["models"]) == 1
        assert config["models"][0]["model_name"] == test_model["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "real_model"])
