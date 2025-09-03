#!/usr/bin/env python3
"""
MCP Server 基本使用示例

本示例展示了如何使用 MCP Server 进行 DNA 序列预测的基本操作。
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List


class MCPClient:
    """MCP Server 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def list_models(self, task_type: str = None) -> Dict[str, Any]:
        """获取模型列表"""
        url = f"{self.base_url}/models"
        if task_type:
            url += f"?task_type={task_type}"
        
        async with self.session.get(url) as response:
            return await response.json()
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        async with self.session.get(f"{self.base_url}/models/{model_name}") as response:
            return await response.json()
    
    async def predict_single(self, model_name: str, sequence: str, task_type: str = None) -> Dict[str, Any]:
        """单序列预测"""
        data = {
            "model_name": model_name,
            "sequence": sequence
        }
        if task_type:
            data["task_type"] = task_type
        
        async with self.session.post(f"{self.base_url}/predict", json=data) as response:
            return await response.json()
    
    async def predict_batch(self, model_name: str, sequences: List[str], task_type: str = None) -> Dict[str, Any]:
        """批量预测"""
        data = {
            "model_name": model_name,
            "sequences": sequences
        }
        if task_type:
            data["task_type"] = task_type
        
        async with self.session.post(f"{self.base_url}/batch_predict", json=data) as response:
            return await response.json()
    
    async def predict_multi(self, sequence: str, models: List[str]) -> Dict[str, Any]:
        """多模型预测"""
        data = {
            "sequence": sequence,
            "models": models
        }
        
        async with self.session.post(f"{self.base_url}/multi_predict", json=data) as response:
            return await response.json()
    
    async def get_model_pool_status(self) -> Dict[str, Any]:
        """获取模型池状态"""
        async with self.session.get(f"{self.base_url}/model_pool/status") as response:
            return await response.json()


async def example_health_check():
    """健康检查示例"""
    print("=== 健康检查示例 ===")
    
    async with MCPClient() as client:
        try:
            health = await client.health_check()
            print(f"服务器状态: {health['status']}")
            print(f"版本: {health['version']}")
            print(f"运行时间: {health['uptime']} 秒")
            print(f"组件状态: {health['components']}")
        except Exception as e:
            print(f"健康检查失败: {e}")


async def example_list_models():
    """模型列表示例"""
    print("\n=== 模型列表示例 ===")
    
    async with MCPClient() as client:
        try:
            # 获取所有模型
            all_models = await client.list_models()
            print(f"总模型数: {all_models['total']}")
            
            # 获取二分类模型
            binary_models = await client.list_models("binary")
            print(f"二分类模型数: {len(binary_models['models'])}")
            
            # 显示前几个模型
            for i, model in enumerate(all_models['models'][:3]):
                print(f"  {i+1}. {model['name']} ({model['task_type']})")
                
        except Exception as e:
            print(f"获取模型列表失败: {e}")


async def example_single_prediction():
    """单序列预测示例"""
    print("\n=== 单序列预测示例 ===")
    
    # 测试序列
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async with MCPClient() as client:
        try:
            # 获取可用模型
            models = await client.list_models("binary")
            if not models['models']:
                print("没有可用的二分类模型")
                return
            
            model_name = models['models'][0]['name']
            print(f"使用模型: {model_name}")
            print(f"测试序列: {test_sequence}")
            
            # 进行预测
            result = await client.predict_single(model_name, test_sequence, "binary")
            
            print(f"预测结果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"概率分布: {result['probabilities']}")
            print(f"处理时间: {result['processing_time']:.3f} 秒")
            
        except Exception as e:
            print(f"单序列预测失败: {e}")


async def example_batch_prediction():
    """批量预测示例"""
    print("\n=== 批量预测示例 ===")
    
    # 测试序列列表
    test_sequences = [
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    ]
    
    async with MCPClient() as client:
        try:
            # 获取可用模型
            models = await client.list_models("binary")
            if not models['models']:
                print("没有可用的二分类模型")
                return
            
            model_name = models['models'][0]['name']
            print(f"使用模型: {model_name}")
            print(f"测试序列数: {len(test_sequences)}")
            
            # 进行批量预测
            result = await client.predict_batch(model_name, test_sequences, "binary")
            
            print(f"批量预测结果:")
            for i, pred_result in enumerate(result['results']):
                print(f"  序列 {i+1}: 预测={pred_result['prediction']}, 置信度={pred_result['confidence']:.3f}")
            
            print(f"总处理时间: {result['summary']['processing_time']:.3f} 秒")
            print(f"平均置信度: {result['summary']['average_confidence']:.3f}")
            
        except Exception as e:
            print(f"批量预测失败: {e}")


async def example_multi_model_prediction():
    """多模型预测示例"""
    print("\n=== 多模型预测示例 ===")
    
    # 测试序列
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async with MCPClient() as client:
        try:
            # 获取不同类型的模型
            binary_models = await client.list_models("binary")
            multiclass_models = await client.list_models("multiclass")
            
            # 选择模型
            selected_models = []
            if binary_models['models']:
                selected_models.append(binary_models['models'][0]['name'])
            if multiclass_models['models']:
                selected_models.append(multiclass_models['models'][0]['name'])
            
            if not selected_models:
                print("没有可用的模型")
                return
            
            print(f"使用模型: {selected_models}")
            print(f"测试序列: {test_sequence}")
            
            # 进行多模型预测
            result = await client.predict_multi(test_sequence, selected_models)
            
            print(f"多模型预测结果:")
            for model_name, pred_result in result['predictions'].items():
                print(f"  {model_name}:")
                print(f"    任务类型: {pred_result['task_type']}")
                print(f"    预测结果: {pred_result['prediction']}")
                print(f"    置信度: {pred_result['confidence']:.3f}")
                if pred_result.get('probabilities'):
                    print(f"    概率分布: {pred_result['probabilities']}")
            
            print(f"总处理时间: {result['summary']['processing_time']:.3f} 秒")
            print(f"平均置信度: {result['summary']['average_confidence']:.3f}")
            
        except Exception as e:
            print(f"多模型预测失败: {e}")


async def example_model_pool_status():
    """模型池状态示例"""
    print("\n=== 模型池状态示例 ===")
    
    async with MCPClient() as client:
        try:
            status = await client.get_model_pool_status()
            
            print(f"模型池状态:")
            print(f"  总模型数: {status['total_models']}")
            print(f"  已加载模型数: {status['loaded_models']}")
            print(f"  加载中模型数: {status['loading_models']}")
            print(f"  错误模型数: {status['error_models']}")
            print(f"  总请求数: {status['total_requests']}")
            print(f"  总使用次数: {status['total_usage']}")
            print(f"  是否运行: {status['is_running']}")
            print(f"  自动扩缩容: {status['auto_scaling']}")
            
            print(f"按任务类型分布:")
            for task_type, info in status['model_types'].items():
                print(f"  {task_type}: 总数={info['total']}, 已加载={info['loaded']}, 加载中={info['loading']}, 错误={info['error']}")
                
        except Exception as e:
            print(f"获取模型池状态失败: {e}")


async def example_performance_test():
    """性能测试示例"""
    print("\n=== 性能测试示例 ===")
    
    # 测试序列
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async with MCPClient() as client:
        try:
            # 获取可用模型
            models = await client.list_models("binary")
            if not models['models']:
                print("没有可用的二分类模型")
                return
            
            model_name = models['models'][0]['name']
            print(f"使用模型: {model_name}")
            
            # 性能测试
            num_requests = 10
            start_time = time.time()
            
            tasks = []
            for i in range(num_requests):
                task = client.predict_single(model_name, test_sequence, "binary")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_requests
            throughput = num_requests / total_time
            
            print(f"性能测试结果:")
            print(f"  请求数: {num_requests}")
            print(f"  总时间: {total_time:.3f} 秒")
            print(f"  平均响应时间: {avg_time:.3f} 秒")
            print(f"  吞吐量: {throughput:.2f} 请求/秒")
            
            # 统计置信度
            confidences = [result['confidence'] for result in results]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  平均置信度: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"性能测试失败: {e}")


async def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    async with MCPClient() as client:
        try:
            # 测试无效序列
            print("测试无效序列:")
            try:
                result = await client.predict_single("test_model", "INVALID_CHARS", "binary")
                print(f"结果: {result}")
            except Exception as e:
                print(f"预期错误: {e}")
            
            # 测试不存在的模型
            print("\n测试不存在的模型:")
            try:
                result = await client.predict_single("nonexistent_model", "ATCGATCG", "binary")
                print(f"结果: {result}")
            except Exception as e:
                print(f"预期错误: {e}")
            
            # 测试空序列
            print("\n测试空序列:")
            try:
                result = await client.predict_single("test_model", "", "binary")
                print(f"结果: {result}")
            except Exception as e:
                print(f"预期错误: {e}")
                
        except Exception as e:
            print(f"错误处理测试失败: {e}")


async def main():
    """主函数"""
    print("MCP Server 基本使用示例")
    print("=" * 50)
    
    # 运行所有示例
    await example_health_check()
    await example_list_models()
    await example_single_prediction()
    await example_batch_prediction()
    await example_multi_model_prediction()
    await example_model_pool_status()
    await example_performance_test()
    await example_error_handling()
    
    print("\n示例完成!")


if __name__ == "__main__":
    asyncio.run(main())
