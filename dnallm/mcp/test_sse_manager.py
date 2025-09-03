#!/usr/bin/env python3
"""
Test script for SSE Manager
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.sse_manager import SSEManager, EventType, SSEEvent, sse_stream_generator


async def test_sse_manager():
    """测试 SSE 管理器"""
    print("🧪 Testing SSE Manager...")
    
    try:
        # 创建 SSE 管理器
        manager = SSEManager(heartbeat_interval=5)
        print("✓ SSE Manager created successfully")
        
        # 启动管理器
        await manager.start()
        print("✓ SSE Manager started successfully")
        
        # 添加客户端
        client1 = await manager.add_client("client1")
        client2 = await manager.add_client("client2")
        print("✓ Clients added successfully")
        
        # 测试事件发送
        await manager.send_prediction_start("test_model", "ATCGATCG")
        print("✓ Prediction start event sent")
        
        await manager.send_prediction_complete("test_model", {"result": "success"})
        print("✓ Prediction complete event sent")
        
        await manager.send_model_loaded("test_model")
        print("✓ Model loaded event sent")
        
        # 测试客户端信息
        client_info = manager.get_client_info()
        assert len(client_info) == 2
        print("✓ Client info retrieved correctly")
        
        # 测试心跳
        await manager.send_heartbeat()
        print("✓ Heartbeat sent")
        
        # 移除客户端
        await manager.remove_client("client1")
        assert manager.get_client_count() == 1
        print("✓ Client removed successfully")
        
        # 停止管理器
        await manager.stop()
        print("✓ SSE Manager stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing SSE Manager: {e}")
        return False


async def test_sse_events():
    """测试 SSE 事件"""
    print("\n🧪 Testing SSE Events...")
    
    try:
        manager = SSEManager()
        
        # 测试事件创建
        event = manager.create_event(
            EventType.PREDICTION_START,
            {"model_name": "test_model", "sequence": "ATCG"}
        )
        assert event.event_type == EventType.PREDICTION_START
        assert event.data["model_name"] == "test_model"
        print("✓ SSE Event created successfully")
        
        # 测试事件格式化
        from dnallm.mcp.sse_manager import format_sse_event
        formatted = format_sse_event(event)
        assert "event: prediction_start" in formatted
        assert "data:" in formatted
        print("✓ SSE Event formatted correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing SSE Events: {e}")
        return False


async def test_sse_stream_generator():
    """测试 SSE 流生成器"""
    print("\n🧪 Testing SSE Stream Generator...")
    
    try:
        manager = SSEManager()
        
        # 添加客户端
        client = await manager.add_client("test_client")
        
        # 发送一些事件
        await manager.send_prediction_start("test_model", "ATCG")
        await manager.send_prediction_complete("test_model", {"result": "success"})
        
        # 测试流生成器
        events_received = []
        async for event_data in sse_stream_generator(client):
            events_received.append(event_data)
            if len(events_received) >= 3:  # 连接确认 + 2个事件
                break
        
        assert len(events_received) >= 3
        print("✓ SSE Stream Generator working correctly")
        
        # 清理
        await manager.remove_client("test_client")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing SSE Stream Generator: {e}")
        return False


async def test_event_types():
    """测试所有事件类型"""
    print("\n🧪 Testing Event Types...")
    
    try:
        manager = SSEManager()
        
        # 测试所有事件类型
        event_types = [
            EventType.PREDICTION_START,
            EventType.PREDICTION_PROGRESS,
            EventType.PREDICTION_COMPLETE,
            EventType.PREDICTION_ERROR,
            EventType.MODEL_LOADED,
            EventType.MODEL_UNLOADED,
            EventType.MODEL_STATUS_UPDATE,
            EventType.SERVER_STATUS,
            EventType.HEARTBEAT
        ]
        
        for event_type in event_types:
            event = manager.create_event(event_type, {"test": "data"})
            assert event.event_type == event_type
            print(f"✓ Event type {event_type.value} working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Event Types: {e}")
        return False


async def test_client_subscription():
    """测试客户端订阅功能"""
    print("\n🧪 Testing Client Subscription...")
    
    try:
        manager = SSEManager()
        
        # 添加客户端
        client = await manager.add_client("subscriber")
        
        # 订阅特定事件
        client.subscribe_to_events([EventType.PREDICTION_START, EventType.PREDICTION_COMPLETE])
        
        # 发送不同类型的事件
        await manager.send_prediction_start("test_model", "ATCG")
        await manager.send_model_loaded("test_model")
        await manager.send_prediction_complete("test_model", {"result": "success"})
        
        # 检查订阅状态
        assert client.is_subscribed_to(EventType.PREDICTION_START)
        assert client.is_subscribed_to(EventType.PREDICTION_COMPLETE)
        assert not client.is_subscribed_to(EventType.MODEL_LOADED)
        print("✓ Client subscription working correctly")
        
        # 取消订阅
        client.unsubscribe_from_events([EventType.PREDICTION_START])
        assert not client.is_subscribed_to(EventType.PREDICTION_START)
        print("✓ Client unsubscription working correctly")
        
        # 清理
        await manager.remove_client("subscriber")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Client Subscription: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting SSE Manager Tests\n")
    
    tests = [
        test_sse_manager,
        test_sse_events,
        test_sse_stream_generator,
        test_event_types,
        test_client_subscription
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
