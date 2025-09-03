"""
SSE (Server-Sent Events) Manager for MCP Server

This module provides SSE functionality for real-time streaming of prediction results,
model status updates, and other server events to clients.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    PREDICTION_START = "prediction_start"
    PREDICTION_PROGRESS = "prediction_progress"
    PREDICTION_COMPLETE = "prediction_complete"
    PREDICTION_ERROR = "prediction_error"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_STATUS_UPDATE = "model_status_update"
    SERVER_STATUS = "server_status"
    HEARTBEAT = "heartbeat"


@dataclass
class SSEEvent:
    """SSE 事件数据结构"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float
    event_id: str
    retry: Optional[int] = None


class SSEClient:
    """SSE 客户端连接"""
    
    def __init__(self, client_id: str, queue: asyncio.Queue):
        self.client_id = client_id
        self.queue = queue
        self.connected = True
        self.subscribed_events: Set[EventType] = set()
        self.last_heartbeat = time.time()
        self.created_at = time.time()
    
    async def send_event(self, event: SSEEvent):
        """发送事件到客户端"""
        if self.connected:
            try:
                await self.queue.put(event)
            except Exception as e:
                logger.error(f"Error sending event to client {self.client_id}: {e}")
                self.connected = False
    
    def subscribe_to_events(self, event_types: List[EventType]):
        """订阅特定类型的事件"""
        self.subscribed_events.update(event_types)
    
    def unsubscribe_from_events(self, event_types: List[EventType]):
        """取消订阅特定类型的事件"""
        self.subscribed_events.difference_update(event_types)
    
    def is_subscribed_to(self, event_type: EventType) -> bool:
        """检查是否订阅了特定类型的事件"""
        return event_type in self.subscribed_events or not self.subscribed_events
    
    def should_receive_event(self, event: SSEEvent) -> bool:
        """检查是否应该接收该事件"""
        return self.connected and self.is_subscribed_to(event.event_type)
    
    def disconnect(self):
        """断开连接"""
        self.connected = False


class SSEEventBroadcaster:
    """SSE 事件广播器"""
    
    def __init__(self):
        self.clients: Dict[str, SSEClient] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def add_client(self, client_id: str) -> SSEClient:
        """添加客户端"""
        async with self._lock:
            if client_id in self.clients:
                # 如果客户端已存在，直接移除旧连接（不调用 remove_client 避免死锁）
                old_client = self.clients[client_id]
                del self.clients[client_id]
                # 关闭旧客户端的队列
                try:
                    old_client.queue.put_nowait(None)  # 发送结束信号
                except asyncio.QueueFull:
                    pass
            
            queue = asyncio.Queue(maxsize=100)
            client = SSEClient(client_id, queue)
            self.clients[client_id] = client
            
            logger.info(f"SSE client {client_id} connected. Total clients: {len(self.clients)}")
            return client
    
    async def remove_client(self, client_id: str):
        """移除客户端"""
        async with self._lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.disconnect()
                del self.clients[client_id]
                logger.info(f"SSE client {client_id} disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_event(self, event: SSEEvent):
        """广播事件到所有相关客户端"""
        async with self._lock:
            if not self.clients:
                return
            
            # 发送到所有订阅了该事件类型的客户端
            tasks = []
            for client in self.clients.values():
                if client.should_receive_event(event):
                    tasks.append(client.send_event(event))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_client(self, client_id: str, event: SSEEvent):
        """发送事件到特定客户端"""
        async with self._lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                if client.should_receive_event(event):
                    await client.send_event(event)
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def handle_event(self, event: SSEEvent):
        """处理事件"""
        # 广播事件
        await self.broadcast_event(event)
        
        # 调用注册的处理器
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type}: {e}")
    
    def get_client_count(self) -> int:
        """获取客户端数量"""
        return len(self.clients)
    
    def get_client_info(self) -> List[Dict[str, Any]]:
        """获取客户端信息"""
        return [
            {
                "client_id": client_id,
                "connected": client.connected,
                "subscribed_events": [et.value for et in client.subscribed_events],
                "created_at": client.created_at,
                "last_heartbeat": client.last_heartbeat
            }
            for client_id, client in self.clients.items()
        ]


class SSEManager:
    """SSE 管理器"""
    
    def __init__(self, heartbeat_interval: int = 30):
        self.broadcaster = SSEEventBroadcaster()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self):
        """启动 SSE 管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("SSE Manager started")
    
    async def stop(self):
        """停止 SSE 管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 断开所有客户端
        await self.broadcaster.remove_client("*")
        logger.info("SSE Manager stopped")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def send_heartbeat(self):
        """发送心跳事件"""
        event = self.create_event(
            EventType.HEARTBEAT,
            {"timestamp": time.time(), "client_count": self.broadcaster.get_client_count()}
        )
        await self.broadcaster.broadcast_event(event)
    
    def create_event(self, event_type: EventType, data: Dict[str, Any], retry: Optional[int] = None) -> SSEEvent:
        """创建 SSE 事件"""
        return SSEEvent(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            event_id=str(uuid.uuid4()),
            retry=retry
        )
    
    async def send_prediction_start(self, model_name: str, sequence: str, client_id: Optional[str] = None):
        """发送预测开始事件"""
        event = self.create_event(
            EventType.PREDICTION_START,
            {
                "model_name": model_name,
                "sequence_length": len(sequence),
                "sequence_preview": sequence[:50] + "..." if len(sequence) > 50 else sequence
            }
        )
        
        if client_id:
            await self.broadcaster.send_to_client(client_id, event)
        else:
            await self.broadcaster.broadcast_event(event)
    
    async def send_prediction_progress(self, model_name: str, progress: float, client_id: Optional[str] = None):
        """发送预测进度事件"""
        event = self.create_event(
            EventType.PREDICTION_PROGRESS,
            {
                "model_name": model_name,
                "progress": progress,
                "progress_percent": int(progress * 100)
            }
        )
        
        if client_id:
            await self.broadcaster.send_to_client(client_id, event)
        else:
            await self.broadcaster.broadcast_event(event)
    
    async def send_prediction_complete(self, model_name: str, result: Dict[str, Any], client_id: Optional[str] = None):
        """发送预测完成事件"""
        event = self.create_event(
            EventType.PREDICTION_COMPLETE,
            {
                "model_name": model_name,
                "result": result,
                "success": True
            }
        )
        
        if client_id:
            await self.broadcaster.send_to_client(client_id, event)
        else:
            await self.broadcaster.broadcast_event(event)
    
    async def send_prediction_error(self, model_name: str, error: str, client_id: Optional[str] = None):
        """发送预测错误事件"""
        event = self.create_event(
            EventType.PREDICTION_ERROR,
            {
                "model_name": model_name,
                "error": error,
                "success": False
            }
        )
        
        if client_id:
            await self.broadcaster.send_to_client(client_id, event)
        else:
            await self.broadcaster.broadcast_event(event)
    
    async def send_model_loaded(self, model_name: str):
        """发送模型加载完成事件"""
        event = self.create_event(
            EventType.MODEL_LOADED,
            {"model_name": model_name, "status": "loaded"}
        )
        await self.broadcaster.broadcast_event(event)
    
    async def send_model_unloaded(self, model_name: str):
        """发送模型卸载事件"""
        event = self.create_event(
            EventType.MODEL_UNLOADED,
            {"model_name": model_name, "status": "unloaded"}
        )
        await self.broadcaster.broadcast_event(event)
    
    async def send_model_status_update(self, model_name: str, status: Dict[str, Any]):
        """发送模型状态更新事件"""
        event = self.create_event(
            EventType.MODEL_STATUS_UPDATE,
            {"model_name": model_name, "status": status}
        )
        await self.broadcaster.broadcast_event(event)
    
    async def send_server_status(self, status: Dict[str, Any]):
        """发送服务器状态事件"""
        event = self.create_event(
            EventType.SERVER_STATUS,
            status
        )
        await self.broadcaster.broadcast_event(event)
    
    async def add_client(self, client_id: str) -> SSEClient:
        """添加客户端"""
        return await self.broadcaster.add_client(client_id)
    
    async def remove_client(self, client_id: str):
        """移除客户端"""
        await self.broadcaster.remove_client(client_id)
    
    def get_client_count(self) -> int:
        """获取客户端数量"""
        return self.broadcaster.get_client_count()
    
    def get_client_info(self) -> List[Dict[str, Any]]:
        """获取客户端信息"""
        return self.broadcaster.get_client_info()
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        self.broadcaster.register_event_handler(event_type, handler)


def format_sse_event(event: SSEEvent) -> str:
    """格式化 SSE 事件为字符串"""
    lines = []
    
    # 事件类型
    lines.append(f"event: {event.event_type.value}")
    
    # 事件 ID
    lines.append(f"id: {event.event_id}")
    
    # 重试间隔
    if event.retry is not None:
        lines.append(f"retry: {event.retry}")
    
    # 数据
    data_str = json.dumps(event.data, ensure_ascii=False)
    lines.append(f"data: {data_str}")
    
    # 时间戳
    lines.append(f"timestamp: {event.timestamp}")
    
    # 空行分隔
    lines.append("")
    
    return "\n".join(lines)


async def sse_stream_generator(client: SSEClient):
    """SSE 流生成器"""
    try:
        # 发送连接确认
        yield "data: {\"type\": \"connected\", \"client_id\": \"" + client.client_id + "\"}\n\n"
        
        while client.connected:
            try:
                # 等待事件，设置超时
                event = await asyncio.wait_for(client.queue.get(), timeout=1.0)
                
                if event:
                    yield format_sse_event(event)
                
            except asyncio.TimeoutError:
                # 超时，发送心跳
                heartbeat_event = SSEEvent(
                    event_type=EventType.HEARTBEAT,
                    data={"timestamp": time.time()},
                    timestamp=time.time(),
                    event_id=str(uuid.uuid4())
                )
                yield format_sse_event(heartbeat_event)
                
            except Exception as e:
                logger.error(f"Error in SSE stream for client {client.client_id}: {e}")
                break
    
    except Exception as e:
        logger.error(f"SSE stream error for client {client.client_id}: {e}")
    finally:
        # 清理客户端
        client.disconnect()
