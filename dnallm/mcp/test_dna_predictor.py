#!/usr/bin/env python3
"""
Test script for DNA Predictor Adapter
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.dna_predictor_adapter import DNAPredictorAdapter, DNAPredictorPool


async def test_dna_predictor_adapter():
    """测试 DNA 预测器适配器"""
    print("🧪 Testing DNA Predictor Adapter...")
    
    # 测试配置路径
    config_path = "./configs/generated/promoter_configs/plant_dnabert_bpe_promoter_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please run the model config generator first:")
        print("python model_config_generator.py --model-info ../models/model_info.yaml --output-dir ./configs/generated --generate-inference")
        return False
    
    try:
        # 创建适配器
        adapter = DNAPredictorAdapter("Plant DNABERT BPE promoter", config_path)
        print("✓ DNA Predictor Adapter created successfully")
        
        # 测试模型信息获取
        model_info = adapter.get_model_info()
        print(f"✓ Model info retrieved: {model_info.get('task_type', 'unknown')} task")
        
        # 测试序列
        test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        print(f"✓ Test sequence: {test_sequence[:20]}... (length: {len(test_sequence)})")
        
        # 注意：这里不实际加载模型，因为需要下载模型文件
        print("⚠️  Skipping actual model loading (requires model download)")
        print("✓ DNA Predictor Adapter test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DNA Predictor Adapter: {e}")
        return False


async def test_dna_predictor_pool():
    """测试 DNA 预测器池"""
    print("\n🧪 Testing DNA Predictor Pool...")
    
    try:
        # 创建预测器池
        pool = DNAPredictorPool(max_models=3)
        print("✓ DNA Predictor Pool created successfully")
        
        # 测试池信息
        pool_info = pool.get_pool_info()
        print(f"✓ Pool info: {pool_info}")
        
        # 测试获取预测器
        config_path = "./configs/generated/promoter_configs/plant_dnabert_bpe_promoter_config.yaml"
        if os.path.exists(config_path):
            predictor = await pool.get_predictor("Plant DNABERT BPE promoter", config_path)
            print("✓ Predictor retrieved from pool successfully")
        else:
            print("⚠️  Config file not found, skipping predictor retrieval test")
        
        # 关闭池
        await pool.shutdown()
        print("✓ DNA Predictor Pool shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DNA Predictor Pool: {e}")
        return False


async def test_model_manager_integration():
    """测试模型管理器集成"""
    print("\n🧪 Testing Model Manager Integration...")
    
    try:
        from dnallm.mcp.model_manager import ModelManager
        
        # 创建模型管理器
        manager = ModelManager(max_models=3)
        print("✓ Model Manager created successfully")
        
        # 测试预测器池
        pool_info = manager.predictor_pool.get_pool_info()
        print(f"✓ Predictor pool info: {pool_info}")
        
        # 关闭管理器
        manager.shutdown()
        print("✓ Model Manager shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Model Manager Integration: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting DNA Predictor Adapter Tests\n")
    
    tests = [
        test_dna_predictor_adapter,
        test_dna_predictor_pool,
        test_model_manager_integration
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
