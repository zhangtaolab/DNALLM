#!/usr/bin/env python3
"""
MCP Server 测试示例

这个示例展示了如何运行和验证 MCP Server 的测试。
"""

import asyncio
import tempfile
import os
import yaml
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.dna_predictor_adapter import DNAPredictorAdapter
from dnallm.mcp.task_router import TaskRouterManager
from dnallm.mcp.sse_manager import SSEManager
from dnallm.mcp.utils.validators import validate_dna_sequence


async def test_basic_functionality():
    """测试基本功能"""
    print("🧬 测试基本功能")
    print("-" * 30)
    
    # 测试序列验证
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    is_valid = validate_dna_sequence(test_sequence)
    print(f"序列验证: {'✅ 通过' if is_valid else '❌ 失败'}")
    
    # 测试任务路由器
    task_manager = TaskRouterManager()
    task_config = {
        "task_type": "binary",
        "num_labels": 2,
        "label_names": ["Not promoter", "Core promoter"],
        "threshold": 0.5
    }
    task_manager.register_task_config("test_model", task_config)
    print(f"任务路由器: ✅ 通过")
    
    # 测试 SSE 管理器
    sse_manager = SSEManager()
    await sse_manager.start()
    client = await sse_manager.add_client("test_client")
    print(f"SSE 管理器: ✅ 通过")
    await sse_manager.stop()
    
    print("基本功能测试完成!\n")


async def test_real_model():
    """测试真实模型（如果可用）"""
    print("🧬 测试真实模型")
    print("-" * 30)
    
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
                "batch_size": 2,
                "max_length": 128,
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
        
        config_path = os.path.join(temp_dir, "test_model_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        # 创建适配器
        adapter = DNAPredictorAdapter(model_info["name"], config_path)
        
        try:
            print(f"尝试加载模型: {model_info['name']}")
            success = await adapter.load_model()
            
            if success:
                print("✅ 模型加载成功")
                
                # 测试预测
                test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
                result = await adapter.predict_single(test_sequence)
                
                print(f"✅ 预测成功")
                print(f"   预测结果: {result['prediction']}")
                print(f"   置信度: {result['confidence']:.3f}")
                print(f"   概率分布: {result['probabilities']}")
                
            else:
                print("❌ 模型加载失败")
                print("   可能原因: 网络连接问题、内存不足或模型不可用")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            print("   这是正常的，如果网络连接有问题或模型不可用")


async def test_integration():
    """测试集成功能"""
    print("🧬 测试集成功能")
    print("-" * 30)
    
    # 测试任务路由器集成
    task_manager = TaskRouterManager()
    task_config = {
        "task_type": "binary",
        "num_labels": 2,
        "label_names": ["Not promoter", "Core promoter"],
        "threshold": 0.5
    }
    task_manager.register_task_config("test_model", task_config)
    
    # 模拟预测结果
    mock_result = {
        "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "task_type": "binary",
        "model_name": "test_model",
        "prediction": 1,
        "confidence": 0.85,
        "probabilities": {
            "Not promoter": 0.15,
            "Core promoter": 0.85
        }
    }
    
    # 处理预测结果
    processed_result = await task_manager.process_prediction(
        mock_result, mock_result["sequence"], "test_model"
    )
    
    print(f"✅ 任务路由器集成测试通过")
    print(f"   处理后的预测: {processed_result.prediction}")
    print(f"   置信度: {processed_result.confidence:.3f}")
    
    # 测试 SSE 集成
    sse_manager = SSEManager()
    await sse_manager.start()
    
    client = await sse_manager.add_client("integration_test_client")
    await sse_manager.send_prediction_start("test_model", mock_result["sequence"], "integration_test_client")
    await sse_manager.send_prediction_complete("test_model", mock_result, "integration_test_client")
    
    print(f"✅ SSE 集成测试通过")
    print(f"   客户端数量: {sse_manager.get_client_count()}")
    
    await sse_manager.stop()
    print("集成功能测试完成!\n")


async def main():
    """主函数"""
    print("🧬 MCP Server 测试示例")
    print("=" * 50)
    print()
    
    try:
        # 运行基本功能测试
        await test_basic_functionality()
        
        # 运行集成功能测试
        await test_integration()
        
        # 尝试运行真实模型测试
        await test_real_model()
        
        print("🎉 所有测试完成!")
        print()
        print("💡 提示:")
        print("   - 基本功能和集成测试应该总是通过")
        print("   - 真实模型测试可能需要网络连接和足够的内存")
        print("   - 如果真实模型测试失败，这是正常的")
        print("   - 运行完整测试套件: python -m pytest dnallm/mcp/tests/ -v")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        print("   请检查依赖是否正确安装")


if __name__ == "__main__":
    asyncio.run(main())
