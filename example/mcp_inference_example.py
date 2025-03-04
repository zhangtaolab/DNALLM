"""
MCP推理示例

本示例展示如何使用MCP协议与DNA语言模型交互。这包括:
1. 启动MCP服务器
2. 连接客户端
3. 发送DNA序列进行推理
4. 分析结果
"""

import asyncio
import subprocess
import time
import os

from dnallm.mcp.client import DNALLMMCPClient

async def main():
    # 启动服务器进程
    server_process = subprocess.Popen(
        ["python", "-m", "dnallm.mcp.server", 
         "--model-type", "plant_dna", 
         "--model-path", "zhangtaolab/plant-dnabert-BPE"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("启动MCP服务器...")
    # 给服务器一些启动时间
    time.sleep(2)
    
    # 创建客户端
    client = DNALLMMCPClient()
    
    try:
        # 连接到服务器
        await client.connect_to_server("python -m dnallm.mcp.server")
        
        # 获取模型信息
        print("\n获取模型信息...")
        model_info = await client.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 预测单个序列
        print("\n预测单个序列...")
        single_result = await client.predict_sequences(["ATCGATCGATCG"])
        print(f"单个序列预测结果: {single_result}")
        
        # 预测多个序列
        print("\n预测多个序列...")
        sequences = [
            "ATCGATCGATCG", 
            "GCTAGCTAGCTA",
            "TTTTAAAACCCC",
            "NNNNNNNNNNNN"
        ]
        
        batch_results = await client.predict_sequences(sequences)
        print(f"批量预测结果: {batch_results}")
        
        # 带概率分布的预测
        print("\n获取概率分布...")
        prob_results = await client.predict_sequences(
            sequences=["ATCGATCGATCG"], 
            return_probabilities=True
        )
        print(f"概率分布预测结果: {prob_results}")
        
        # 基序分析
        print("\n分析DNA基序...")
        motif_results = await client.analyze_motifs(
            sequence="ATCGATCGATCGATCGATCGATCG",
            min_length=3,
            max_length=6
        )
        print(f"基序分析结果: {motif_results}")
        
    finally:
        # 关闭客户端
        await client.close()
        
        # 终止服务器进程
        server_process.terminate()
        server_process.wait()
        
        print("\nMCP示例完成")

if __name__ == "__main__":
    asyncio.run(main()) 