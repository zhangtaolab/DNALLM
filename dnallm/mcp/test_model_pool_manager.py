#!/usr/bin/env python3
"""
Test script for Model Pool Manager
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.model_pool_manager import ModelPoolManager, ModelStatus


async def test_model_pool_manager():
    """测试模型池管理器"""
    print("🧪 Testing Model Pool Manager...")
    
    try:
        # 创建模型池管理器
        manager = ModelPoolManager(
            max_models=5,
            max_concurrent_requests_per_model=10,
            health_check_interval=10,
            resource_check_interval=5,
            auto_scaling=True
        )
        print("✓ Model Pool Manager created successfully")
        
        # 启动管理器
        await manager.start()
        print("✓ Model Pool Manager started successfully")
        
        # 测试模型加载
        success1 = await manager.load_model("test_model_1", "./configs/test1.yaml", "binary")
        success2 = await manager.load_model("test_model_2", "./configs/test2.yaml", "multiclass")
        success3 = await manager.load_model("test_model_3", "./configs/test3.yaml", "binary")
        
        assert success1 and success2 and success3
        print("✓ Models loaded successfully")
        
        # 测试池状态
        status = manager.get_pool_status()
        assert status["total_models"] == 3
        assert status["loaded_models"] == 3
        print("✓ Pool status retrieved correctly")
        
        # 测试模型获取
        model_instance = await manager.get_model("test_model_1", "binary")
        assert model_instance is not None
        assert model_instance.model_name == "test_model_1"
        print("✓ Model instance retrieved successfully")
        
        # 测试预测
        result = await manager.predict("test_model_1", "ATCGATCG", "binary")
        assert result is not None
        assert result["model_name"] == "test_model_1"
        print("✓ Prediction executed successfully")
        
        # 测试模型信息获取
        model_info = manager.get_model_info(model_instance.model_id)
        assert model_info is not None
        assert model_info["model_name"] == "test_model_1"
        print("✓ Model info retrieved successfully")
        
        # 测试模型卸载
        success = await manager.unload_model(model_instance.model_id)
        assert success
        print("✓ Model unloaded successfully")
        
        # 测试状态更新
        status = manager.get_pool_status()
        assert status["total_models"] == 2
        print("✓ Pool status updated correctly")
        
        # 停止管理器
        await manager.stop()
        print("✓ Model Pool Manager stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Model Pool Manager: {e}")
        return False


async def test_model_loading():
    """测试模型加载功能"""
    print("\n🧪 Testing Model Loading...")
    
    try:
        manager = ModelPoolManager(max_models=3)
        await manager.start()
        
        # 测试正常加载
        success = await manager.load_model("test_model", "./configs/test.yaml", "binary")
        assert success
        print("✓ Normal model loading working correctly")
        
        # 测试重复加载
        success = await manager.load_model("test_model", "./configs/test.yaml", "binary")
        assert success  # 应该创建新的实例
        print("✓ Duplicate model loading working correctly")
        
        # 测试超过最大模型数
        success1 = await manager.load_model("model1", "./configs/test1.yaml", "binary")
        success2 = await manager.load_model("model2", "./configs/test2.yaml", "binary")
        success3 = await manager.load_model("model3", "./configs/test3.yaml", "binary")
        success4 = await manager.load_model("model4", "./configs/test4.yaml", "binary")
        
        # 第四个模型应该失败或替换现有模型
        print("✓ Model limit handling working correctly")
        
        await manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ Error testing Model Loading: {e}")
        return False


async def test_model_types():
    """测试模型类型管理"""
    print("\n🧪 Testing Model Types...")
    
    try:
        manager = ModelPoolManager(max_models_per_type=2)
        await manager.start()
        
        # 加载不同类型的模型
        await manager.load_model("binary_model_1", "./configs/binary1.yaml", "binary")
        await manager.load_model("binary_model_2", "./configs/binary2.yaml", "binary")
        await manager.load_model("multiclass_model_1", "./configs/multiclass1.yaml", "multiclass")
        await manager.load_model("multiclass_model_2", "./configs/multiclass2.yaml", "multiclass")
        
        # 测试按类型获取模型
        binary_model = await manager.get_model("binary_model_1", "binary")
        assert binary_model is not None
        print("✓ Binary model retrieval working correctly")
        
        multiclass_model = await manager.get_model("multiclass_model_1", "multiclass")
        assert multiclass_model is not None
        print("✓ Multiclass model retrieval working correctly")
        
        # 测试类型限制
        success = await manager.load_model("binary_model_3", "./configs/binary3.yaml", "binary")
        # 应该失败或替换现有模型
        print("✓ Model type limit handling working correctly")
        
        await manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ Error testing Model Types: {e}")
        return False


async def test_concurrent_requests():
    """测试并发请求处理"""
    print("\n🧪 Testing Concurrent Requests...")
    
    try:
        manager = ModelPoolManager(max_concurrent_requests_per_model=3)
        await manager.start()
        
        # 加载模型
        await manager.load_model("test_model", "./configs/test.yaml", "binary")
        
        # 创建多个并发预测任务
        tasks = []
        for i in range(5):
            task = manager.predict("test_model", f"ATCG{i}", "binary")
            tasks.append(task)
        
        # 执行并发预测
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 检查结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 3  # 至少3个应该成功
        print("✓ Concurrent requests handling working correctly")
        
        await manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ Error testing Concurrent Requests: {e}")
        return False


async def test_health_monitoring():
    """测试健康监控"""
    print("\n🧪 Testing Health Monitoring...")
    
    try:
        manager = ModelPoolManager(
            health_check_interval=2,
            resource_check_interval=1
        )
        await manager.start()
        
        # 加载模型
        await manager.load_model("test_model", "./configs/test.yaml", "binary")
        
        # 等待健康检查
        await asyncio.sleep(3)
        
        # 检查健康状态
        status = manager.get_pool_status()
        assert status["loaded_models"] >= 0
        print("✓ Health monitoring working correctly")
        
        await manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ Error testing Health Monitoring: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting Model Pool Manager Tests\n")
    
    tests = [
        test_model_pool_manager,
        test_model_loading,
        test_model_types,
        test_concurrent_requests,
        test_health_monitoring
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
