#!/usr/bin/env python3
"""
Simple script to run the MCP server for testing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.mcp_server import MCPServer


async def main():
    """主函数"""
    # 配置文件路径
    config_path = "./configs/generated/mcp_server_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please run the model config generator first:")
        print("python model_config_generator.py --model-info ../models/model_info.yaml --output-dir ./configs/generated --generate-inference")
        return
    
    # 创建服务器
    server = MCPServer()
    
    try:
        print("Initializing MCP server...")
        await server.initialize(config_path)
        
        print("Starting FastAPI server on http://0.0.0.0:8000")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /models - List all models")
        print("  GET  /models/{model_name} - Get model info")
        print("  POST /predict - Single sequence prediction")
        print("  POST /batch_predict - Batch prediction")
        print("  POST /multi_predict - Multi-model prediction")
        print("  GET  /stream_predict - Stream prediction (SSE)")
        print("\nPress Ctrl+C to stop the server")
        
        await server.run_fastapi("0.0.0.0", 8000)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
