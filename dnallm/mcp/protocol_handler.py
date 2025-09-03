"""
MCP Protocol Handler for MCP Server

This module provides MCP (Model Context Protocol) protocol handling functionality,
including tool definitions, request/response processing, and protocol compliance.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP 消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPToolType(Enum):
    """MCP 工具类型"""
    DNA_PREDICT = "dna_predict"
    DNA_BATCH_PREDICT = "dna_batch_predict"
    DNA_MULTI_PREDICT = "dna_multi_predict"
    DNA_STREAM_PREDICT = "dna_stream_predict"
    LIST_MODELS = "list_models"
    GET_MODEL_INFO = "get_model_info"
    LIST_MODELS_BY_TASK = "list_models_by_task"
    GET_MODEL_CAPABILITIES = "get_model_capabilities"
    HEALTH_CHECK = "health_check"


@dataclass
class MCPMessage:
    """MCP 消息结构"""
    id: str
    type: MCPMessageType
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None


class MCPProtocolHandler:
    """MCP 协议处理器"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.request_handlers: Dict[str, Callable] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        
        # 注册默认工具
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        tools = [
            MCPTool(
                name="dna_predict",
                description="Predict DNA sequence using specified model",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model name from model_info.yaml (e.g., 'Plant DNABERT BPE promoter')"
                        },
                        "sequence": {
                            "type": "string",
                            "description": "DNA sequence to predict"
                        },
                        "task_type": {
                            "type": "string",
                            "enum": ["binary", "multiclass", "multilabel", "regression"],
                            "description": "Task type: binary (promoter, conservation, lncRNAs, H3K27ac, H3K4me3, H3K27me3), multiclass (open chromatin), regression (promoter strength)"
                        }
                    },
                    "required": ["model_name", "sequence"]
                }
            ),
            MCPTool(
                name="dna_batch_predict",
                description="Batch predict multiple DNA sequences",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model name from model_info.yaml"
                        },
                        "sequences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of DNA sequences to predict"
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Task type for the model"
                        }
                    },
                    "required": ["model_name", "sequences"]
                }
            ),
            MCPTool(
                name="dna_multi_predict",
                description="Predict DNA sequence using multiple models in parallel",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "DNA sequence to predict"
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model names to use for prediction"
                        }
                    },
                    "required": ["sequence", "models"]
                }
            ),
            MCPTool(
                name="list_models",
                description="List all available models",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["binary", "multiclass", "regression"],
                            "description": "Optional task type filter"
                        }
                    }
                }
            ),
            MCPTool(
                name="get_model_info",
                description="Get detailed information about a specific model",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model name from model_info.yaml"
                        }
                    },
                    "required": ["model_name"]
                }
            ),
            MCPTool(
                name="list_models_by_task",
                description="List available models by task type",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["binary", "multiclass", "regression"],
                            "description": "Task type to filter models"
                        }
                    },
                    "required": ["task_type"]
                }
            ),
            MCPTool(
                name="get_model_capabilities",
                description="Get model capabilities and task information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model name from model_info.yaml"
                        }
                    },
                    "required": ["model_name"]
                }
            ),
            MCPTool(
                name="health_check",
                description="Check server health and status",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
        
        for tool in tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")
    
    def register_request_handler(self, method: str, handler: Callable):
        """注册请求处理器"""
        self.request_handlers[method] = handler
        logger.info(f"Registered request handler: {method}")
    
    def register_notification_handler(self, method: str, handler: Callable):
        """注册通知处理器"""
        self.notification_handlers[method] = handler
        logger.info(f"Registered notification handler: {method}")
    
    async def handle_message(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理 MCP 消息"""
        try:
            message = self._parse_message(message_data)
            
            if message.type == MCPMessageType.REQUEST:
                return await self._handle_request(message)
            elif message.type == MCPMessageType.NOTIFICATION:
                await self._handle_notification(message)
                return None
            elif message.type == MCPMessageType.RESPONSE:
                await self._handle_response(message)
                return None
            else:
                return self._create_error_response(
                    message.id,
                    -32600,
                    "Invalid Request",
                    "Unknown message type"
                )
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            message_id = message_data.get("id", str(uuid.uuid4()))
            return self._create_error_response(
                message_id,
                -32603,
                "Internal Error",
                str(e)
            )
    
    def _parse_message(self, message_data: Dict[str, Any]) -> MCPMessage:
        """解析消息"""
        message_id = message_data.get("id", str(uuid.uuid4()))
        method = message_data.get("method")
        params = message_data.get("params")
        result = message_data.get("result")
        error = message_data.get("error")
        
        if "method" in message_data:
            if "result" in message_data or "error" in message_data:
                message_type = MCPMessageType.RESPONSE
            else:
                message_type = MCPMessageType.REQUEST
        else:
            message_type = MCPMessageType.NOTIFICATION
        
        return MCPMessage(
            id=message_id,
            type=message_type,
            method=method,
            params=params,
            result=result,
            error=error
        )
    
    async def _handle_request(self, message: MCPMessage) -> Dict[str, Any]:
        """处理请求"""
        if not message.method:
            return self._create_error_response(
                message.id,
                -32600,
                "Invalid Request",
                "Missing method"
            )
        
        handler = self.request_handlers.get(message.method)
        if not handler:
            return self._create_error_response(
                message.id,
                -32601,
                "Method Not Found",
                f"Unknown method: {message.method}"
            )
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message.params or {})
            else:
                result = handler(message.params or {})
            
            return self._create_success_response(message.id, result)
            
        except Exception as e:
            logger.error(f"Error handling request {message.method}: {e}")
            return self._create_error_response(
                message.id,
                -32603,
                "Internal Error",
                str(e)
            )
    
    async def _handle_notification(self, message: MCPMessage):
        """处理通知"""
        if not message.method:
            logger.warning("Received notification without method")
            return
        
        handler = self.notification_handlers.get(message.method)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message.params or {})
                else:
                    handler(message.params or {})
            except Exception as e:
                logger.error(f"Error handling notification {message.method}: {e}")
        else:
            logger.warning(f"No handler for notification: {message.method}")
    
    async def _handle_response(self, message: MCPMessage):
        """处理响应"""
        async with self._lock:
            if message.id in self.pending_requests:
                future = self.pending_requests.pop(message.id)
                if message.error:
                    future.set_exception(Exception(f"MCP Error: {message.error}"))
                else:
                    future.set_result(message.result)
    
    def _create_success_response(self, message_id: str, result: Any) -> Dict[str, Any]:
        """创建成功响应"""
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": result
        }
    
    def _create_error_response(self, message_id: str, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """创建错误响应"""
        error = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data
        
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": error
        }
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """发送请求并等待响应"""
        message_id = str(uuid.uuid4())
        message = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params or {}
        }
        
        # 创建等待响应的 Future
        future = asyncio.Future()
        async with self._lock:
            self.pending_requests[message_id] = future
        
        try:
            # 这里应该发送消息到客户端
            # 暂时模拟响应
            await asyncio.sleep(0.1)
            
            # 模拟成功响应
            return {"status": "success", "method": method, "params": params}
            
        except Exception as e:
            async with self._lock:
                if message_id in self.pending_requests:
                    del self.pending_requests[message_id]
            raise e
    
    async def send_notification(self, method: str, params: Dict[str, Any] = None):
        """发送通知"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        # 这里应该发送消息到客户端
        logger.info(f"Sending notification: {method}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具定义"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            }
            for tool in self.tools.values()
        ]
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取特定工具"""
        return self.tools.get(name)
    
    def validate_tool_input(self, tool_name: str, input_data: Dict[str, Any]) -> bool:
        """验证工具输入"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        
        # 简单的验证逻辑
        required_fields = tool.input_schema.get("required", [])
        for field in required_fields:
            if field not in input_data:
                return False
        
        return True


class MCPToolExecutor:
    """MCP 工具执行器"""
    
    def __init__(self, protocol_handler: MCPProtocolHandler):
        self.protocol_handler = protocol_handler
        self.execution_context: Dict[str, Any] = {}
    
    async def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具"""
        tool = self.protocol_handler.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        if not self.protocol_handler.validate_tool_input(tool_name, input_data):
            raise ValueError(f"Invalid input for tool: {tool_name}")
        
        if tool.handler:
            try:
                if asyncio.iscoroutinefunction(tool.handler):
                    result = await tool.handler(input_data, self.execution_context)
                else:
                    result = tool.handler(input_data, self.execution_context)
                
                return {
                    "tool": tool_name,
                    "success": True,
                    "result": result,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
        else:
            return {
                "tool": tool_name,
                "success": False,
                "error": "No handler registered for tool",
                "timestamp": time.time()
            }
    
    def set_context(self, key: str, value: Any):
        """设置执行上下文"""
        self.execution_context[key] = value
    
    def get_context(self, key: str) -> Any:
        """获取执行上下文"""
        return self.execution_context.get(key)
    
    def clear_context(self):
        """清除执行上下文"""
        self.execution_context.clear()


class MCPProtocolValidator:
    """MCP 协议验证器"""
    
    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """验证 MCP 消息格式"""
        if not isinstance(message, dict):
            return False
        
        # 检查 JSON-RPC 版本
        if message.get("jsonrpc") != "2.0":
            return False
        
        # 检查必需字段
        if "method" not in message and "result" not in message and "error" not in message:
            return False
        
        # 检查 ID 字段（请求和响应必须有 ID）
        if "method" in message or "result" in message or "error" in message:
            if "id" not in message:
                return False
        
        return True
    
    @staticmethod
    def validate_tool_schema(schema: Dict[str, Any]) -> bool:
        """验证工具模式"""
        if not isinstance(schema, dict):
            return False
        
        # 检查必需字段
        required_fields = ["type", "properties"]
        for field in required_fields:
            if field not in schema:
                return False
        
        # 检查类型
        if schema["type"] != "object":
            return False
        
        # 检查属性
        if not isinstance(schema["properties"], dict):
            return False
        
        return True
    
    @staticmethod
    def validate_error_code(code: int) -> bool:
        """验证错误代码"""
        # JSON-RPC 2.0 标准错误代码
        standard_codes = {
            -32700,  # Parse error
            -32600,  # Invalid Request
            -32601,  # Method not found
            -32602,  # Invalid params
            -32603,  # Internal error
        }
        
        # 允许标准错误代码和自定义错误代码（-32000 到 -32099）
        return code in standard_codes or (-32099 <= code <= -32000)


if __name__ == "__main__":
    # 测试协议处理器
    async def test_protocol_handler():
        handler = MCPProtocolHandler()
        
        # 测试工具注册
        tools = handler.get_tools()
        print(f"Registered {len(tools)} tools")
        
        # 测试消息处理
        test_message = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "list_models",
            "params": {}
        }
        
        response = await handler.handle_message(test_message)
        print(f"Response: {response}")
    
    asyncio.run(test_protocol_handler())
