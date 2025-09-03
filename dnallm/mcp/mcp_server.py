"""
Main MCP Server Implementation

This module implements the core MCP server functionality for DNA sequence prediction.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# Import MCP components (simplified for now)
# from mcp.server import Server
# from mcp.server.models import InitializationOptions
# from mcp import stdio_server
# from mcp import (
#     Resource, Tool, TextContent, ImageContent, EmbeddedResource,
#     CallToolRequest, CallToolResult, ListResourcesRequest, ListResourcesResult,
#     ListToolsRequest, ListToolsResult, ReadResourceRequest, ReadResourceResult
# )

# Import local components
from .config_manager import ConfigManager, MCPServerConfig
from .model_manager import ModelManager, ModelPool
from .model_pool_manager import ModelPoolManager
from .task_router import TaskRouterManager, TaskType
from .sse_manager import SSEManager, EventType, sse_stream_generator
from .utils.validators import validate_prediction_request, validate_dna_sequence
from .utils.formatters import (
    format_prediction_result, format_multi_model_result, 
    format_error_response, format_health_check_response,
    format_model_list_response, format_model_info_response,
    format_sse_event
)

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP 服务器主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_manager = ConfigManager()
        self.model_manager = ModelManager()
        self.model_pool = ModelPool()
        self.model_pool_manager = ModelPoolManager()
        self.task_router = TaskRouterManager()
        self.sse_manager = SSEManager()
        self.app = FastAPI(title="DNALLM MCP Server", version="1.0.0")
        self.start_time = time.time()
        # self.mcp_server = None  # Temporarily disabled
        
        # 设置日志
        self._setup_logging()
        
        # 设置 FastAPI 应用
        self._setup_fastapi()
        
        # 设置 MCP 服务器 (temporarily disabled)
        # self._setup_mcp_server()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_fastapi(self):
        """设置 FastAPI 应用"""
        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 添加路由
        self._add_routes()
    
    def _add_routes(self):
        """添加 API 路由"""
        
        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            uptime = time.time() - self.start_time
            loaded_models = len(self.model_manager.get_loaded_models())
            total_models = len(self.config_manager.get_enabled_models()) if self.config_manager.mcp_config else 0
            
            return format_health_check_response(
                status="healthy",
                models_loaded=loaded_models,
                total_models=total_models,
                uptime=uptime
            )
        
        @self.app.get("/models")
        async def list_models():
            """列出所有模型"""
            if not self.config_manager.mcp_config:
                raise HTTPException(status_code=503, detail="Server not configured")
            
            models = []
            for model_config in self.config_manager.get_enabled_models():
                model_info = self.model_manager.get_model_info(model_config.name)
                model_data = {
                    "name": model_config.name,
                    "model_name": model_config.model_name,
                    "task_type": model_config.task_type,
                    "description": model_config.description,
                    "enabled": model_config.enabled,
                    "max_concurrent_requests": model_config.max_concurrent_requests,
                    "is_loaded": model_info is not None,
                    "usage_count": model_info.get("usage_count", 0) if model_info else 0,
                    "last_used": model_info.get("last_used") if model_info else None
                }
                models.append(model_data)
            
            return format_model_list_response(models)
        
        @self.app.get("/models/{model_name}")
        async def get_model_info(model_name: str):
            """获取模型详细信息"""
            capabilities = self.config_manager.get_model_capabilities(model_name)
            if not capabilities:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            return format_model_info_response(capabilities)
        
        @self.app.get("/models/task/{task_type}")
        async def list_models_by_task(task_type: str):
            """按任务类型列出模型"""
            if not self.config_manager.mcp_config:
                raise HTTPException(status_code=503, detail="Server not configured")
            
            models = self.config_manager.get_models_by_task_type(task_type)
            model_data = []
            
            for model_config in models:
                model_info = self.model_manager.get_model_info(model_config.name)
                model_data.append({
                    "name": model_config.name,
                    "model_name": model_config.model_name,
                    "task_type": model_config.task_type,
                    "description": model_config.description,
                    "enabled": model_config.enabled,
                    "max_concurrent_requests": model_config.max_concurrent_requests,
                    "is_loaded": model_info is not None,
                    "usage_count": model_info.get("usage_count", 0) if model_info else 0,
                    "last_used": model_info.get("last_used") if model_info else None
                })
            
            return format_model_list_response(model_data, task_type)
        
        @self.app.get("/task-types")
        async def list_task_types():
            """列出所有支持的任务类型"""
            task_types = {}
            
            for task_type in TaskType:
                models = self.task_router.get_models_by_task_type(task_type)
                task_types[task_type.value] = {
                    "name": task_type.value,
                    "description": f"{task_type.value.title()} classification/regression task",
                    "model_count": len(models),
                    "models": models
                }
            
            return {
                "task_types": task_types,
                "total_task_types": len(task_types),
                "timestamp": time.time()
            }
        
        @self.app.get("/pool/status")
        async def get_pool_status():
            """获取模型池状态"""
            return self.model_pool_manager.get_pool_status()
        
        @self.app.get("/pool/models/{model_id}")
        async def get_model_info(model_id: str):
            """获取模型实例信息"""
            model_info = self.model_pool_manager.get_model_info(model_id)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            return model_info
        
        @self.app.post("/predict")
        async def predict(request: Dict[str, Any]):
            """单序列预测"""
            # 验证请求
            validation = validate_prediction_request(request)
            if not validation["is_valid"]:
                return format_error_response(
                    f"Invalid request: {'; '.join(validation['errors'])}",
                    "VALIDATION_ERROR",
                    {"errors": validation["errors"], "warnings": validation["warnings"]}
                )
            
            cleaned_request = validation["cleaned_request"]
            model_name = cleaned_request.get("model_name")
            sequence = cleaned_request["sequence"]
            
            if not model_name:
                return format_error_response("model_name is required", "MISSING_MODEL_NAME")
            
            # 检查模型是否已加载
            if not self.model_manager.get_model_info(model_name):
                return format_error_response(f"Model {model_name} not loaded", "MODEL_NOT_LOADED")
            
            # 获取模型配置路径
            model_config = self.config_manager.get_model_config(model_name)
            config_path = model_config.config_path if model_config else None
            
            # 发送预测开始事件
            await self.sse_manager.send_prediction_start(model_name, sequence)
            
            # 执行预测
            start_time = time.time()
            try:
                # 使用模型池管理器进行预测
                result = await self.model_pool_manager.predict(model_name, sequence, task_type="binary")
                processing_time = time.time() - start_time
                
                if result is None:
                    return format_error_response(f"Prediction failed for model {model_name}", "PREDICTION_FAILED")
                
                # 使用任务路由器处理结果
                try:
                    prediction_result = await self.task_router.process_prediction(result, sequence, model_name)
                    formatted_result = self.task_router.format_prediction_result(prediction_result)
                    formatted_result["processing_time"] = processing_time
                    formatted_result["timestamp"] = time.time()
                    
                    # 发送预测完成事件
                    await self.sse_manager.send_prediction_complete(model_name, formatted_result)
                    
                    return formatted_result
                except Exception as e:
                    logger.error(f"Task routing error: {e}")
                    # 回退到原始格式化方法
                    model_config = self.config_manager.get_model_config(model_name)
                    task_type = model_config.task_type if model_config else "binary"
                    formatted_result = format_prediction_result(result, model_name, sequence, task_type, processing_time)
                    
                    # 发送预测完成事件
                    await self.sse_manager.send_prediction_complete(model_name, formatted_result)
                    
                    return formatted_result
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                # 发送预测错误事件
                await self.sse_manager.send_prediction_error(model_name, str(e))
                return format_error_response(f"Prediction error: {str(e)}", "PREDICTION_ERROR")
        
        @self.app.post("/batch_predict")
        async def batch_predict(request: Dict[str, Any]):
            """批量预测"""
            # 验证请求
            validation = validate_prediction_request(request)
            if not validation["is_valid"]:
                return format_error_response(
                    f"Invalid request: {'; '.join(validation['errors'])}",
                    "VALIDATION_ERROR",
                    {"errors": validation["errors"], "warnings": validation["warnings"]}
                )
            
            cleaned_request = validation["cleaned_request"]
            model_name = cleaned_request.get("model_name")
            sequences = cleaned_request["sequences"]
            
            if not model_name:
                return format_error_response("model_name is required", "MISSING_MODEL_NAME")
            
            # 检查模型是否已加载
            if not self.model_manager.get_model_info(model_name):
                return format_error_response(f"Model {model_name} not loaded", "MODEL_NOT_LOADED")
            
            # 获取模型配置路径
            model_config = self.config_manager.get_model_config(model_name)
            config_path = model_config.config_path if model_config else None
            
            # 执行批量预测
            start_time = time.time()
            try:
                results = await self.model_manager.batch_predict(model_name, sequences, config_path)
                processing_time = time.time() - start_time
                
                if results is None:
                    return format_error_response(f"Batch prediction failed for model {model_name}", "BATCH_PREDICTION_FAILED")
                
                # 格式化结果
                formatted_results = []
                for i, (sequence, result) in enumerate(zip(sequences, results)):
                    model_config = self.config_manager.get_model_config(model_name)
                    task_type = model_config.task_type if model_config else "binary"
                    formatted_result = format_prediction_result(result, model_name, sequence, task_type)
                    formatted_result["sequence_index"] = i
                    formatted_results.append(formatted_result)
                
                return {
                    "model_name": model_name,
                    "total_sequences": len(sequences),
                    "processing_time": processing_time,
                    "results": formatted_results,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                return format_error_response(f"Batch prediction error: {str(e)}", "BATCH_PREDICTION_ERROR")
        
        @self.app.post("/multi_predict")
        async def multi_predict(request: Dict[str, Any]):
            """多模型并行预测"""
            # 验证请求
            validation = validate_prediction_request(request)
            if not validation["is_valid"]:
                return format_error_response(
                    f"Invalid request: {'; '.join(validation['errors'])}",
                    "VALIDATION_ERROR",
                    {"errors": validation["errors"], "warnings": validation["warnings"]}
                )
            
            cleaned_request = validation["cleaned_request"]
            models = cleaned_request.get("models", [])
            sequence = cleaned_request["sequence"]
            
            if not models:
                return format_error_response("models list is required", "MISSING_MODELS")
            
            # 检查模型是否已加载并收集配置路径
            missing_models = []
            config_paths = {}
            for model_name in models:
                if not self.model_manager.get_model_info(model_name):
                    missing_models.append(model_name)
                else:
                    # 获取模型配置路径
                    model_config = self.config_manager.get_model_config(model_name)
                    if model_config:
                        config_paths[model_name] = model_config.config_path
            
            if missing_models:
                return format_error_response(
                    f"Models not loaded: {', '.join(missing_models)}",
                    "MODELS_NOT_LOADED",
                    {"missing_models": missing_models}
                )
            
            # 执行多模型预测
            start_time = time.time()
            try:
                results = await self.model_manager.multi_model_predict(models, sequence, config_paths)
                processing_time = time.time() - start_time
                
                return format_multi_model_result(results, sequence, processing_time)
                
            except Exception as e:
                logger.error(f"Multi-model prediction error: {e}")
                return format_error_response(f"Multi-model prediction error: {str(e)}", "MULTI_MODEL_PREDICTION_ERROR")
        
        @self.app.get("/stream_predict")
        async def stream_predict(
            model_name: str,
            sequence: str,
            background_tasks: BackgroundTasks
        ):
            """流式预测（SSE）"""
            # 验证序列
            seq_validation = validate_dna_sequence(sequence)
            if not seq_validation["is_valid"]:
                return format_error_response(
                    f"Invalid sequence: {'; '.join(seq_validation['errors'])}",
                    "INVALID_SEQUENCE"
                )
            
            # 检查模型是否已加载
            if not self.model_manager.get_model_info(model_name):
                return format_error_response(f"Model {model_name} not loaded", "MODEL_NOT_LOADED")
            
            # 获取模型配置路径
            model_config = self.config_manager.get_model_config(model_name)
            config_path = model_config.config_path if model_config else None
            
            async def generate_events():
                try:
                    # 发送开始事件
                    yield format_sse_event("prediction_start", {
                        "model_name": model_name,
                        "sequence_length": len(sequence),
                        "timestamp": time.time()
                    })
                    
                    # 执行预测
                    result = await self.model_manager.predict(model_name, sequence, config_path)
                    
                    if result is None:
                        yield format_sse_event("prediction_error", {
                            "error": "Prediction failed",
                            "model_name": model_name
                        })
                        return
                    
                    # 发送结果事件
                    model_config = self.config_manager.get_model_config(model_name)
                    task_type = model_config.task_type if model_config else "binary"
                    formatted_result = format_prediction_result(result, model_name, sequence, task_type)
                    
                    yield format_sse_event("prediction_result", formatted_result)
                    
                except Exception as e:
                    logger.error(f"Stream prediction error: {e}")
                    yield format_sse_event("prediction_error", {
                        "error": str(e),
                        "model_name": model_name
                    })
            
            return StreamingResponse(
                generate_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        @self.app.get("/stream/events")
        async def stream_events(
            background_tasks: BackgroundTasks,
            client_id: Optional[str] = None
        ):
            """SSE 事件流"""
            if not client_id:
                client_id = f"client_{int(time.time() * 1000)}"
            
            # 添加客户端到 SSE 管理器
            client = await self.sse_manager.add_client(client_id)
            
            async def cleanup():
                await self.sse_manager.remove_client(client_id)
            
            background_tasks.add_task(cleanup)
            
            return StreamingResponse(
                sse_stream_generator(client),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
    
    def _setup_mcp_server(self):
        """设置 MCP 服务器 (temporarily disabled)"""
        # self.mcp_server = Server("dnallm-mcp-server")
        
        # 注册工具
        # self._register_tools()
        
        # 注册资源
        # self._register_resources()
        pass
    
    def _register_tools(self):
        """注册 MCP 工具 (temporarily disabled)"""
        pass
    
    # MCP tool handlers (temporarily disabled)
    pass
    
    def _register_resources(self):
        """注册 MCP 资源 (temporarily disabled)"""
        pass
    
    async def initialize(self, config_path: str):
        """初始化服务器"""
        logger.info(f"Initializing MCP server with config: {config_path}")
        
        # 加载配置
        self.config_manager.load_mcp_config(config_path)
        
        # 加载推理配置
        self.config_manager.load_all_inference_configs()
        
        # 注册任务配置
        self._register_task_configs()
        
        # 启动 SSE 管理器
        await self.sse_manager.start()
        
        # 启动模型池管理器
        await self.model_pool_manager.start()
        
        # 预加载模型
        enabled_models = self.config_manager.get_enabled_models()
        model_configs = []
        
        for model_config in enabled_models:
            model_configs.append({
                "name": model_config.name,
                "config_path": model_config.config_path
            })
        
        logger.info(f"Preloading {len(model_configs)} models...")
        
        # 使用模型池管理器加载模型
        for model_config in model_configs:
            model_name = model_config["name"]
            config_path = model_config["config_path"]
            
            # 获取任务类型
            inference_config = self.config_manager.get_inference_config(model_name)
            task_type = "binary"  # 默认任务类型
            if inference_config and inference_config.model.get('task_info'):
                task_type = inference_config.model['task_info'].get('task_type', 'binary')
            
            # 加载模型到池中
            success = await self.model_pool_manager.load_model(model_name, config_path, task_type)
            if success:
                logger.info(f"Model {model_name} loaded to pool successfully")
            else:
                logger.error(f"Failed to load model {model_name} to pool")
        
        # 获取池状态
        pool_status = self.model_pool_manager.get_pool_status()
        logger.info(f"Model pool status: {pool_status['loaded_models']}/{pool_status['total_models']} models loaded")
        
        # 启动清理任务
        asyncio.create_task(self._cleanup_task())
    
    def _register_task_configs(self):
        """注册任务配置"""
        logger.info("Registering task configurations...")
        
        for model_config in self.config_manager.get_enabled_models():
            # 获取推理配置
            inference_config = self.config_manager.get_inference_config(model_config.name)
            if inference_config:
                # 从推理配置中提取任务信息
                task_info = inference_config.model.get('task_info', {})
                if task_info:
                    self.task_router.register_task_config(model_config.name, task_info)
                    logger.info(f"Registered task config for {model_config.name}: {task_info.get('task_type', 'unknown')}")
        
        logger.info(f"Registered {len(self.task_router.get_registered_models())} task configurations")
    
    async def _cleanup_task(self):
        """定期清理未使用的模型"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                await self.model_manager.cleanup_unused_models()
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def run_fastapi(self, host: str = "0.0.0.0", port: int = 8000):
        """运行 FastAPI 服务器"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run_mcp(self):
        """运行 MCP 服务器 (temporarily disabled)"""
        pass
    
    def shutdown(self):
        """关闭服务器"""
        logger.info("Shutting down MCP server")
        
        # 停止 SSE 管理器
        asyncio.create_task(self.sse_manager.stop())
        
        # 停止模型池管理器
        asyncio.create_task(self.model_pool_manager.stop())
        
        # 关闭模型管理器
        self.model_manager.shutdown()
        
        # 关闭模型池
        self.model_pool.shutdown()


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DNALLM MCP Server")
    parser.add_argument("--config", default="./configs/mcp_server_config.yaml",
                       help="Path to MCP server config file")
    parser.add_argument("--mode", choices=["fastapi", "mcp"], default="fastapi",
                       help="Server mode: fastapi or mcp")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # 创建服务器
    server = MCPServer()
    
    try:
        # 初始化服务器
        await server.initialize(args.config)
        
        # 运行服务器
        if args.mode == "fastapi":
            await server.run_fastapi(args.host, args.port)
        else:
            await server.run_mcp()
            
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
