#!/usr/bin/env python3
"""
Test script for Task Router
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.task_router import TaskRouter, TaskRouterManager, TaskType, TaskConfig


async def test_task_router():
    """测试任务路由器"""
    print("🧪 Testing Task Router...")
    
    try:
        # 创建任务路由器
        router = TaskRouter()
        print("✓ Task Router created successfully")
        
        # 测试任务类型枚举
        task_type = router.get_task_type("binary")
        assert task_type == TaskType.BINARY
        print("✓ Task type enum working correctly")
        
        # 测试任务配置创建
        config_dict = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Negative", "Positive"],
            "threshold": 0.5,
            "describe": "Binary classification task"
        }
        task_config = router.create_task_config(config_dict)
        assert task_config.task_type == TaskType.BINARY
        assert task_config.num_labels == 2
        print("✓ Task config creation working correctly")
        
        # 测试二分类结果处理
        raw_result = {
            "prediction": 1,
            "probabilities": {"Negative": 0.2, "Positive": 0.8}
        }
        sequence = "ATCGATCGATCGATCG"
        model_name = "test_model"
        
        result = await router.route_prediction(raw_result, sequence, model_name, task_config)
        assert result.task_type == TaskType.BINARY
        assert result.prediction == 1
        assert result.confidence == 0.8
        print("✓ Binary task routing working correctly")
        
        # 测试结果格式化
        formatted = router.format_prediction_result(result)
        assert "prediction" in formatted
        assert "confidence" in formatted
        assert "probabilities" in formatted
        print("✓ Result formatting working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Task Router: {e}")
        return False


async def test_task_router_manager():
    """测试任务路由器管理器"""
    print("\n🧪 Testing Task Router Manager...")
    
    try:
        # 创建管理器
        manager = TaskRouterManager()
        print("✓ Task Router Manager created successfully")
        
        # 注册任务配置
        config_dict = {
            "task_type": "multiclass",
            "num_labels": 3,
            "label_names": ["Class A", "Class B", "Class C"],
            "describe": "Multiclass classification task"
        }
        manager.register_task_config("test_model", config_dict)
        print("✓ Task config registration working correctly")
        
        # 获取任务配置
        task_config = manager.get_task_config("test_model")
        assert task_config is not None
        assert task_config.task_type == TaskType.MULTICLASS
        print("✓ Task config retrieval working correctly")
        
        # 测试预测处理
        raw_result = {
            "prediction": 1,
            "probabilities": {"Class A": 0.1, "Class B": 0.7, "Class C": 0.2}
        }
        sequence = "ATCGATCGATCGATCG"
        
        result = await manager.process_prediction(raw_result, sequence, "test_model")
        assert result.task_type == TaskType.MULTICLASS
        assert result.prediction == 1
        assert result.confidence == 0.7
        print("✓ Prediction processing working correctly")
        
        # 测试格式化
        formatted = manager.format_prediction_result(result)
        assert "task_type" in formatted
        assert formatted["task_type"] == "multiclass"
        print("✓ Result formatting working correctly")
        
        # 测试已注册模型列表
        models = manager.get_registered_models()
        assert "test_model" in models
        print("✓ Registered models list working correctly")
        
        # 测试按任务类型获取模型
        multiclass_models = manager.get_models_by_task_type(TaskType.MULTICLASS)
        assert "test_model" in multiclass_models
        print("✓ Models by task type working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Task Router Manager: {e}")
        return False


async def test_different_task_types():
    """测试不同任务类型"""
    print("\n🧪 Testing Different Task Types...")
    
    try:
        manager = TaskRouterManager()
        
        # 测试二分类
        binary_config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["No", "Yes"],
            "threshold": 0.5
        }
        manager.register_task_config("binary_model", binary_config)
        
        binary_result = await manager.process_prediction(
            {"prediction": 1, "probabilities": {"No": 0.3, "Yes": 0.7}},
            "ATCG", "binary_model"
        )
        assert binary_result.task_type == TaskType.BINARY
        print("✓ Binary classification working correctly")
        
        # 测试多分类
        multiclass_config = {
            "task_type": "multiclass",
            "num_labels": 3,
            "label_names": ["A", "B", "C"]
        }
        manager.register_task_config("multiclass_model", multiclass_config)
        
        multiclass_result = await manager.process_prediction(
            {"prediction": 2, "probabilities": {"A": 0.1, "B": 0.2, "C": 0.7}},
            "ATCG", "multiclass_model"
        )
        assert multiclass_result.task_type == TaskType.MULTICLASS
        print("✓ Multiclass classification working correctly")
        
        # 测试多标签
        multilabel_config = {
            "task_type": "multilabel",
            "num_labels": 2,
            "label_names": ["Label1", "Label2"],
            "threshold": 0.5
        }
        manager.register_task_config("multilabel_model", multilabel_config)
        
        multilabel_result = await manager.process_prediction(
            {"predictions": [1, 0], "probabilities": {"Label1": 0.8, "Label2": 0.3}},
            "ATCG", "multilabel_model"
        )
        assert multilabel_result.task_type == TaskType.MULTILABEL
        print("✓ Multilabel classification working correctly")
        
        # 测试回归
        regression_config = {
            "task_type": "regression",
            "num_labels": 1,
            "label_names": ["value"]
        }
        manager.register_task_config("regression_model", regression_config)
        
        regression_result = await manager.process_prediction(
            {"prediction": 0.75, "confidence": 0.9},
            "ATCG", "regression_model"
        )
        assert regression_result.task_type == TaskType.REGRESSION
        print("✓ Regression working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing different task types: {e}")
        return False


async def test_task_summary():
    """测试任务摘要功能"""
    print("\n🧪 Testing Task Summary...")
    
    try:
        manager = TaskRouterManager()
        
        # 注册二分类模型
        binary_config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["No", "Yes"],
            "threshold": 0.5
        }
        manager.register_task_config("binary_model", binary_config)
        
        # 创建多个预测结果
        results = []
        for i in range(5):
            result = await manager.process_prediction(
                {"prediction": i % 2, "probabilities": {"No": 0.3, "Yes": 0.7}},
                f"ATCG{i}", "binary_model"
            )
            results.append(result)
        
        # 获取任务摘要
        summary = manager.get_task_summary(results)
        assert summary["total_predictions"] == 5
        assert summary["task_type"] == "binary"
        assert "positive_predictions" in summary
        assert "negative_predictions" in summary
        print("✓ Task summary working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing task summary: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting Task Router Tests\n")
    
    tests = [
        test_task_router,
        test_task_router_manager,
        test_different_task_types,
        test_task_summary
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # 总结结果
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
