"""
Test model configuration generator.
"""

import pytest
import os
import tempfile
from pathlib import Path
import yaml

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_config_generator import MCPModelConfigGenerator


class TestMCPModelConfigGenerator:
    """测试模型配置生成器"""
    
    def test_init(self):
        """测试初始化"""
        import os
        model_info_path = os.path.join(os.path.dirname(__file__), "../../models/model_info.yaml")
        generator = MCPModelConfigGenerator(model_info_path)
        assert generator.model_info_path == model_info_path
        assert generator.model_info is not None
        assert "finetuned" in generator.model_info
    
    def test_get_models_by_task_type(self):
        """测试按任务类型获取模型"""
        generator = MCPModelConfigGenerator("../../models/model_info.yaml")
        
        # 测试二分类模型
        binary_models = generator.get_models_by_task_type("binary")
        assert len(binary_models) > 0
        for model in binary_models:
            assert model["task"]["task_type"] == "binary"
        
        # 测试多分类模型
        multiclass_models = generator.get_models_by_task_type("multiclass")
        assert len(multiclass_models) > 0
        for model in multiclass_models:
            assert model["task"]["task_type"] == "multiclass"
        
        # 测试回归模型
        regression_models = generator.get_models_by_task_type("regression")
        assert len(regression_models) > 0
        for model in regression_models:
            assert model["task"]["task_type"] == "regression"
    
    def test_get_all_task_types(self):
        """测试获取所有任务类型"""
        generator = MCPModelConfigGenerator("../../models/model_info.yaml")
        task_types = generator.get_all_task_types()
        
        expected_types = ["binary", "multiclass", "regression"]
        for task_type in expected_types:
            assert task_type in task_types
    
    def test_generate_mcp_server_config(self):
        """测试生成 MCP 服务器配置"""
        generator = MCPModelConfigGenerator("../../models/model_info.yaml")
        
        # 选择一些模型
        selected_models = [
            "Plant DNABERT BPE promoter",
            "Plant DNABERT BPE conservation"
        ]
        
        config = generator.generate_mcp_server_config(selected_models)
        
        # 验证配置结构
        assert "server" in config
        assert "mcp" in config
        assert "models" in config
        assert "multi_model" in config
        assert "sse" in config
        assert "logging" in config
        
        # 验证模型配置
        assert len(config["models"]) == 2
        for model_config in config["models"]:
            assert "name" in model_config
            assert "model_name" in model_config
            assert "config_path" in model_config
            assert "enabled" in model_config
            assert "task_type" in model_config
    
    def test_generate_inference_configs(self):
        """测试生成推理配置"""
        generator = MCPModelConfigGenerator("../../models/model_info.yaml")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_files = generator.generate_inference_configs(temp_dir)
            
            # 验证生成了文件
            assert len(generated_files) > 0
            
            # 验证文件内容
            for file_path in generated_files[:3]:  # 只检查前3个文件
                assert os.path.exists(file_path)
                
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 验证配置结构
                assert "task" in config
                assert "inference" in config
                assert "model" in config
                
                # 验证任务配置
                assert "task_type" in config["task"]
                assert "num_labels" in config["task"]
                assert "label_names" in config["task"]
    
    def test_get_model_capabilities(self):
        """测试获取模型能力信息"""
        generator = MCPModelConfigGenerator("../../models/model_info.yaml")
        
        # 测试存在的模型
        capabilities = generator.get_model_capabilities("Plant DNABERT BPE promoter")
        assert capabilities is not None
        assert "model_name" in capabilities
        assert "task_type" in capabilities
        assert "num_labels" in capabilities
        assert "label_names" in capabilities
        
        # 测试不存在的模型
        capabilities = generator.get_model_capabilities("Non-existent Model")
        assert capabilities is None


if __name__ == "__main__":
    pytest.main([__file__])
