"""
DNA语言模型MCP服务器模块

本模块实现了Model Context Protocol(MCP)服务器，提供DNA语言模型的推理能力，包括：

1. 核心功能：
   - 暴露DNA序列分析工具
   - 提供模型预测结果
   - 支持批量序列处理

2. MCP工具：
   - predict_sequence: 对单个或多个DNA序列进行预测
   - get_model_info: 获取当前加载模型的信息
   - analyze_motifs: 分析DNA序列中的基序

3. 特点：
   - 轻量级服务器，支持本地运行
   - 与Claude等LLM应用无缝集成
   - 安全可靠的DNA序列处理

使用示例: dnallm-mcp-server --model-path models/dna_model
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from mcp import (
    ServerSession,
    initialize_server,
    run_stdio_server,
    ListToolsResponse,
    ToolDefinition,
    Tool,
)

from ..inference.predictor import DNAPredictor
from ..inference.config import InferenceConfig

class DNALLMMCPServer:
    """MCP服务器，提供DNA语言模型的推理功能"""
    
    def __init__(self, model_type: str = "plant_dna", model_path: str = None):
        """
        初始化DNA MCP服务器
        
        Args:
            model_type: 模型类型，默认为plant_dna
            model_path: 模型路径，如果为None则使用默认模型
        """
        self.model_type = model_type
        self.model_path = model_path or "zhangtaolab/plant-dnabert-BPE"
        
        # 初始化推理配置
        self.inference_config = InferenceConfig(
            model_path=self.model_path,
            batch_size=8,
            device="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
        )
        
        # 延迟初始化predictor，直到真正需要时才加载模型
        self.predictor = None

    def get_predictor(self):
        """懒加载预测器"""
        if self.predictor is None:
            self.predictor = DNAPredictor(
                model_type=self.model_type,
                model_path=self.model_path,
                config=self.inference_config
            )
        return self.predictor

    async def initialize_session(self, session: ServerSession):
        """初始化服务器会话"""
        await session.initialize()
        print("MCP服务器已初始化，等待连接...")
        
        # 注册工具
        predict_tool = Tool(
            name="predict-sequence",
            description="对DNA序列进行预测，可接受单个序列或多个序列",
            parameters={
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "一个或多个DNA序列"
                    },
                    "return_probabilities": {
                        "type": "boolean",
                        "description": "是否返回概率分布",
                        "default": False
                    }
                },
                "required": ["sequences"]
            },
            handler=self.predict_handler
        )
        
        info_tool = Tool(
            name="get-model-info",
            description="获取当前加载模型的信息",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self.model_info_handler
        )
        
        motif_tool = Tool(
            name="analyze-motifs",
            description="分析DNA序列中的基序",
            parameters={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "需要分析的DNA序列"
                    },
                    "min_length": {
                        "type": "integer",
                        "description": "最小基序长度",
                        "default": 3
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "最大基序长度",
                        "default": 8
                    }
                },
                "required": ["sequence"]
            },
            handler=self.motif_handler
        )
        
        await session.register_tool(predict_tool)
        await session.register_tool(info_tool)
        await session.register_tool(motif_tool)
    
    async def predict_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理序列预测请求"""
        sequences = params.get("sequences", [])
        return_probs = params.get("return_probabilities", False)
        
        if not sequences:
            return {"error": "未提供序列"}
        
        # 获取预测器
        predictor = self.get_predictor()
        
        # 执行预测
        try:
            results = predictor.predict(sequences)
            
            # 处理预测结果
            if return_probs:
                # 返回完整概率分布
                predictions = {
                    "probabilities": results["logits"].softmax(dim=-1).cpu().tolist(),
                    "sequences": sequences
                }
            else:
                # 只返回最高概率的类别
                predictions = {
                    "predictions": results["logits"].argmax(dim=-1).cpu().tolist(),
                    "sequences": sequences
                }
            
            return predictions
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}
    
    async def model_info_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """返回模型信息"""
        predictor = self.get_predictor()
        model = predictor.model.get_model()
        
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "hidden_size": model.config.hidden_size if hasattr(model.config, "hidden_size") else "unknown",
            "vocabulary_size": model.config.vocab_size if hasattr(model.config, "vocab_size") else "unknown",
            "device": self.inference_config.device
        }
    
    async def motif_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """分析DNA序列中的基序"""
        sequence = params.get("sequence", "")
        min_length = params.get("min_length", 3)
        max_length = params.get("max_length", 8)
        
        if not sequence:
            return {"error": "未提供序列"}
        
        # 简单的基序分析实现
        motifs = {}
        
        for length in range(min_length, min(max_length + 1, len(sequence))):
            motif_counts = {}
            for i in range(len(sequence) - length + 1):
                motif = sequence[i:i+length]
                motif_counts[motif] = motif_counts.get(motif, 0) + 1
            
            # 筛选出现次数最多的前5个基序
            top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            motifs[f"{length}mer"] = [{"motif": m, "count": c} for m, c in top_motifs]
        
        return {
            "sequence": sequence,
            "motifs": motifs
        }

async def run_server(model_type: str = "plant_dna", model_path: str = None):
    """运行MCP服务器"""
    server = DNALLMMCPServer(model_type=model_type, model_path=model_path)
    
    # 注册会话初始化函数
    async def session_initializer(session: ServerSession):
        await server.initialize_session(session)
    
    # 运行服务器
    await run_stdio_server(session_initializer)

def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动DNA语言模型MCP服务器")
    parser.add_argument("--model-type", type=str, default="plant_dna", 
                       help="模型类型: plant_dna/dnabert/nucleotide")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径，如果不提供则使用默认模型")
    parser.add_argument("--device", type=str, default="cuda",
                       help="运行设备: cuda/cpu")
    
    args = parser.parse_args()
    
    # 设置设备环境变量
    os.environ["USE_CUDA"] = "1" if args.device.lower() == "cuda" else "0"
    
    # 运行服务器
    asyncio.run(run_server(args.model_type, args.model_path))

if __name__ == "__main__":
    main() 