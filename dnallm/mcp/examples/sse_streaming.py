#!/usr/bin/env python3
"""
MCP Server SSE 流式预测示例

本示例展示了如何使用 Server-Sent Events (SSE) 进行实时流式预测。
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List, Callable


class SSEStreamClient:
    """SSE 流式客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.event_handlers: Dict[str, Callable] = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def on_event(self, event_type: str):
        """注册事件处理器装饰器"""
        def decorator(handler: Callable):
            self.event_handlers[event_type] = handler
            return handler
        return decorator
    
    async def stream_prediction(self, model_name: str, sequence: str, task_type: str = None):
        """流式预测"""
        url = f"{self.base_url}/stream_predict"
        params = {
            "model_name": model_name,
            "sequence": sequence
        }
        if task_type:
            params["task_type"] = task_type
        
        print(f"开始流式预测: {model_name}")
        print(f"序列: {sequence}")
        print("-" * 50)
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    print(f"连接失败: {response.status}")
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('event: '):
                        event_type = line[7:]
                    elif line.startswith('data: '):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            await self._handle_event(event_type, data)
                        except json.JSONDecodeError:
                            print(f"JSON 解析错误: {data_str}")
                    elif line == '':
                        # 空行表示事件结束
                        pass
                    else:
                        # 其他行
                        pass
                        
        except Exception as e:
            print(f"流式预测错误: {e}")
    
    async def _handle_event(self, event_type: str, data: Dict[str, Any]):
        """处理事件"""
        print(f"[{event_type}] {data}")
        
        # 调用注册的事件处理器
        if event_type in self.event_handlers:
            try:
                if asyncio.iscoroutinefunction(self.event_handlers[event_type]):
                    await self.event_handlers[event_type](data)
                else:
                    self.event_handlers[event_type](data)
            except Exception as e:
                print(f"事件处理器错误: {e}")


async def example_basic_streaming():
    """基本流式预测示例"""
    print("=== 基本流式预测示例 ===")
    
    # 测试序列
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async with SSEStreamClient() as client:
        # 注册事件处理器
        @client.on_event("prediction_start")
        def on_prediction_start(data):
            print(f"🚀 预测开始: {data['model_name']}")
            print(f"   序列长度: {data['sequence_length']}")
            print(f"   序列预览: {data['sequence_preview']}")
        
        @client.on_event("prediction_progress")
        def on_prediction_progress(data):
            progress = data['progress_percent']
            print(f"⏳ 预测进度: {progress}%")
        
        @client.on_event("prediction_complete")
        def on_prediction_complete(data):
            print(f"✅ 预测完成: {data['model_name']}")
            result = data['result']
            print(f"   预测结果: {result['prediction']}")
            print(f"   置信度: {result['confidence']:.3f}")
            print(f"   概率分布: {result['probabilities']}")
        
        @client.on_event("prediction_error")
        def on_prediction_error(data):
            print(f"❌ 预测错误: {data['model_name']}")
            print(f"   错误信息: {data['error']}")
        
        @client.on_event("heartbeat")
        def on_heartbeat(data):
            print(f"💓 心跳: {data['timestamp']}, 客户端数: {data['client_count']}")
        
        # 开始流式预测
        try:
            await client.stream_prediction("Plant DNABERT BPE promoter", test_sequence, "binary")
        except Exception as e:
            print(f"流式预测失败: {e}")


async def example_multiple_streams():
    """多流并发示例"""
    print("\n=== 多流并发示例 ===")
    
    # 测试序列列表
    test_sequences = [
        "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
    ]
    
    async def stream_single_prediction(sequence_id: int, sequence: str):
        """单个流式预测"""
        async with SSEStreamClient() as client:
            @client.on_event("prediction_start")
            def on_start(data):
                print(f"[流 {sequence_id}] 🚀 开始预测")
            
            @client.on_event("prediction_complete")
            def on_complete(data):
                result = data['result']
                print(f"[流 {sequence_id}] ✅ 完成: 预测={result['prediction']}, 置信度={result['confidence']:.3f}")
            
            @client.on_event("prediction_error")
            def on_error(data):
                print(f"[流 {sequence_id}] ❌ 错误: {data['error']}")
            
            try:
                await client.stream_prediction("Plant DNABERT BPE promoter", sequence, "binary")
            except Exception as e:
                print(f"[流 {sequence_id}] 流式预测失败: {e}")
    
    # 并发运行多个流
    tasks = []
    for i, sequence in enumerate(test_sequences):
        task = stream_single_prediction(i + 1, sequence)
        tasks.append(task)
    
    await asyncio.gather(*tasks)


async def example_stream_with_timeout():
    """带超时的流式预测示例"""
    print("\n=== 带超时的流式预测示例 ===")
    
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async with SSEStreamClient() as client:
        @client.on_event("prediction_start")
        def on_start(data):
            print(f"🚀 预测开始: {data['model_name']}")
        
        @client.on_event("prediction_complete")
        def on_complete(data):
            result = data['result']
            print(f"✅ 预测完成: 预测={result['prediction']}, 置信度={result['confidence']:.3f}")
        
        @client.on_event("prediction_error")
        def on_error(data):
            print(f"❌ 预测错误: {data['error']}")
        
        try:
            # 设置超时
            await asyncio.wait_for(
                client.stream_prediction("Plant DNABERT BPE promoter", test_sequence, "binary"),
                timeout=10.0  # 10秒超时
            )
        except asyncio.TimeoutError:
            print("⏰ 流式预测超时")
        except Exception as e:
            print(f"流式预测失败: {e}")


async def example_stream_statistics():
    """流式预测统计示例"""
    print("\n=== 流式预测统计示例 ===")
    
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    class StreamStatistics:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.events_received = 0
            self.event_types = {}
        
        def on_event(self, event_type: str, data: Dict[str, Any]):
            if event_type == "prediction_start":
                self.start_time = time.time()
            elif event_type == "prediction_complete":
                self.end_time = time.time()
            
            self.events_received += 1
            self.event_types[event_type] = self.event_types.get(event_type, 0) + 1
        
        def get_summary(self):
            duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
            return {
                "duration": duration,
                "events_received": self.events_received,
                "event_types": self.event_types
            }
    
    stats = StreamStatistics()
    
    async with SSEStreamClient() as client:
        @client.on_event("prediction_start")
        def on_start(data):
            stats.on_event("prediction_start", data)
            print(f"🚀 预测开始: {data['model_name']}")
        
        @client.on_event("prediction_progress")
        def on_progress(data):
            stats.on_event("prediction_progress", data)
            print(f"⏳ 进度: {data['progress_percent']}%")
        
        @client.on_event("prediction_complete")
        def on_complete(data):
            stats.on_event("prediction_complete", data)
            result = data['result']
            print(f"✅ 预测完成: 预测={result['prediction']}, 置信度={result['confidence']:.3f}")
        
        @client.on_event("heartbeat")
        def on_heartbeat(data):
            stats.on_event("heartbeat", data)
            print(f"💓 心跳")
        
        try:
            await client.stream_prediction("Plant DNABERT BPE promoter", test_sequence, "binary")
            
            # 显示统计信息
            summary = stats.get_summary()
            print(f"\n📊 统计信息:")
            print(f"   总耗时: {summary['duration']:.3f} 秒")
            print(f"   接收事件数: {summary['events_received']}")
            print(f"   事件类型分布: {summary['event_types']}")
            
        except Exception as e:
            print(f"流式预测失败: {e}")


async def example_stream_reconnection():
    """流式预测重连示例"""
    print("\n=== 流式预测重连示例 ===")
    
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    async def stream_with_retry(max_retries: int = 3):
        """带重试的流式预测"""
        for attempt in range(max_retries):
            try:
                print(f"尝试 {attempt + 1}/{max_retries}")
                
                async with SSEStreamClient() as client:
                    @client.on_event("prediction_start")
                    def on_start(data):
                        print(f"🚀 预测开始: {data['model_name']}")
                    
                    @client.on_event("prediction_complete")
                    def on_complete(data):
                        result = data['result']
                        print(f"✅ 预测完成: 预测={result['prediction']}, 置信度={result['confidence']:.3f}")
                    
                    @client.on_event("prediction_error")
                    def on_error(data):
                        print(f"❌ 预测错误: {data['error']}")
                    
                    await client.stream_prediction("Plant DNABERT BPE promoter", test_sequence, "binary")
                    return  # 成功，退出重试循环
                    
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 2 秒后重试...")
                    await asyncio.sleep(2)
                else:
                    print(f"所有重试都失败了")
    
    await stream_with_retry()


async def example_custom_event_handler():
    """自定义事件处理器示例"""
    print("\n=== 自定义事件处理器示例 ===")
    
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    class CustomEventHandler:
        def __init__(self):
            self.predictions = []
            self.start_times = {}
        
        async def handle_prediction_start(self, data):
            model_name = data['model_name']
            self.start_times[model_name] = time.time()
            print(f"🚀 开始预测 {model_name}")
        
        async def handle_prediction_complete(self, data):
            model_name = data['model_name']
            result = data['result']
            
            if model_name in self.start_times:
                duration = time.time() - self.start_times[model_name]
                print(f"✅ {model_name} 完成 (耗时: {duration:.3f}s)")
            
            self.predictions.append({
                'model': model_name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        
        async def handle_prediction_error(self, data):
            model_name = data['model_name']
            print(f"❌ {model_name} 错误: {data['error']}")
        
        def get_summary(self):
            if not self.predictions:
                return "没有预测结果"
            
            avg_confidence = sum(p['confidence'] for p in self.predictions) / len(self.predictions)
            return f"预测数: {len(self.predictions)}, 平均置信度: {avg_confidence:.3f}"
    
    handler = CustomEventHandler()
    
    async with SSEStreamClient() as client:
        @client.on_event("prediction_start")
        async def on_start(data):
            await handler.handle_prediction_start(data)
        
        @client.on_event("prediction_complete")
        async def on_complete(data):
            await handler.handle_prediction_complete(data)
        
        @client.on_event("prediction_error")
        async def on_error(data):
            await handler.handle_prediction_error(data)
        
        try:
            await client.stream_prediction("Plant DNABERT BPE promoter", test_sequence, "binary")
            
            # 显示摘要
            print(f"\n📊 摘要: {handler.get_summary()}")
            
        except Exception as e:
            print(f"流式预测失败: {e}")


async def main():
    """主函数"""
    print("MCP Server SSE 流式预测示例")
    print("=" * 50)
    
    # 运行所有示例
    await example_basic_streaming()
    await example_multiple_streams()
    await example_stream_with_timeout()
    await example_stream_statistics()
    await example_stream_reconnection()
    await example_custom_event_handler()
    
    print("\n示例完成!")


if __name__ == "__main__":
    asyncio.run(main())
