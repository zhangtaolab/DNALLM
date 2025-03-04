"""
DNA语言模型MCP支持模块

本模块提供了与Model Context Protocol(MCP)集成的功能，用于DNA语言模型的推理。主要内容包括：

1. 服务器组件：
   - MCP服务器实现
   - 工具注册与处理
   - 序列分析能力

2. 客户端组件：
   - MCP客户端接口
   - 与服务器交互

3. 与Claude等LLM集成：
   - 通过MCP协议提供DNA分析能力
   - 启用智能对话能力

此模块使DNA语言模型能够通过MCP协议与大型语言模型集成，为生物信息学研究提供强大的交互式分析工具。
"""

from .server import DNALLMMCPServer, run_server, main as server_main
from .client import DNALLMMCPClient

__all__ = ["DNALLMMCPServer", "run_server", "DNALLMMCPClient", "server_main"] 