"""
DNA语言模型MCP客户端模块

本模块提供了与DNA语言模型MCP服务器交互的客户端实现，主要功能包括：

1. 核心功能：
   - 连接到DNA语言模型MCP服务器
   - 发送序列预测请求
   - 处理模型响应

2. 客户端能力：
   - 序列预测
   - 获取模型信息
   - 基序分析

3. 特点：
   - 简单易用的API
   - 异步通信支持
   - 与Claude等LLM集成能力

使用示例:
    client = DNALLMMCPClient()
    await client.connect_to_server("python -m dnallm.mcp.server")
    results = await client.predict_sequences(["ATCG", "GCTA"])
"""

import asyncio
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class DNALLMMCPClient:
    """与DNA语言模型MCP服务器交互的客户端"""
    
    def __init__(self):
        """初始化客户端"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_server(self, command: str, args: List[str] = None):
        """
        连接到MCP服务器
        
        Args:
            command: 服务器启动命令
            args: 命令行参数
        """
        if args is None:
            args = []
        
        server_params = StdioServerParameters(
            command=command.split()[0],
            args=[*command.split()[1:], *args],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print(f"已连接到服务器，可用工具: {[tool.name for tool in tools]}")
    
    async def predict_sequences(self, sequences: List[str], return_probabilities: bool = False) -> Dict[str, Any]:
        """
        预测DNA序列
        
        Args:
            sequences: 需要预测的DNA序列列表
            return_probabilities: 是否返回概率分布
            
        Returns:
            预测结果
        """
        if self.session is None:
            raise ValueError("未连接到服务器")
        
        params = {
            "sequences": sequences,
            "return_probabilities": return_probabilities
        }
        
        result = await self.session.call_tool("predict-sequence", params)
        return result
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        if self.session is None:
            raise ValueError("未连接到服务器")
        
        result = await self.session.call_tool("get-model-info", {})
        return result
    
    async def analyze_motifs(self, sequence: str, min_length: int = 3, max_length: int = 8) -> Dict[str, Any]:
        """
        分析DNA序列中的基序
        
        Args:
            sequence: DNA序列
            min_length: 最小基序长度
            max_length: 最大基序长度
            
        Returns:
            基序分析结果
        """
        if self.session is None:
            raise ValueError("未连接到服务器")
        
        params = {
            "sequence": sequence,
            "min_length": min_length,
            "max_length": max_length
        }
        
        result = await self.session.call_tool("analyze-motifs", params)
        return result
    
    async def close(self):
        """关闭客户端连接"""
        if self.session is not None:
            await self.exit_stack.aclose()
            self.session = None

async def main():
    """客户端示例"""
    client = DNALLMMCPClient()
    
    try:
        # 连接到服务器
        await client.connect_to_server("python -m dnallm.mcp.server")
        
        # 获取模型信息
        model_info = await client.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 预测序列
        sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
        results = await client.predict_sequences(sequences)
        print(f"预测结果: {results}")
        
        # 分析基序
        motif_results = await client.analyze_motifs("ATCGATCGATCGATCG")
        print(f"基序分析结果: {motif_results}")
    
    finally:
        # 关闭连接
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 