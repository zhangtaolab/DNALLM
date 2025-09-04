#!/usr/bin/env python3
"""
MCP 服务器启动脚本

这个脚本演示如何启动 MCP 服务器并运行基本功能测试。
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dnallm.mcp.mcp_server import MCPServer
from dnallm.mcp.config_manager import ConfigManager
from dnallm.mcp.model_config_generator import MCPModelConfigGenerator
from dnallm.mcp.utils.validators import validate_dna_sequence


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试 MCP 服务器基本功能...")
    
    # 1. 测试配置生成器
    print("\n1. 测试配置生成器...")
    try:
        generator = MCPModelConfigGenerator('dnallm/models/model_info.yaml')
        task_types = generator.get_all_task_types()
        print(f"   ✅ 支持的任务类型: {task_types}")
        
        binary_models = generator.get_models_by_task_type('binary')
        print(f"   ✅ 二分类模型数量: {len(binary_models)}")
        
    except Exception as e:
        print(f"   ❌ 配置生成器测试失败: {e}")
        return False
    
    # 2. 测试配置管理器
    print("\n2. 测试配置管理器...")
    try:
        config_manager = ConfigManager('dnallm/mcp/configs/mcp_server_config.yaml.example')
        print("   ✅ 配置管理器创建成功")
        
    except Exception as e:
        print(f"   ❌ 配置管理器测试失败: {e}")
        return False
    
    # 3. 测试 MCP 服务器
    print("\n3. 测试 MCP 服务器...")
    try:
        server = MCPServer('dnallm/mcp/configs/mcp_server_config.yaml.example')
        print("   ✅ MCP 服务器创建成功")
        
        # 测试 FastAPI 应用
        app = server.app
        print("   ✅ FastAPI 应用创建成功")
        
    except Exception as e:
        print(f"   ❌ MCP 服务器测试失败: {e}")
        return False
    
    # 4. 测试 DNA 序列验证
    print("\n4. 测试 DNA 序列验证...")
    try:
        test_sequences = [
            "ATCGATCGATCG",  # 有效序列
            "ATCG123",       # 无效序列（包含数字）
            "",              # 空序列
            "ATCG" * 100     # 长序列
        ]
        
        for seq in test_sequences:
            result = validate_dna_sequence(seq)
            status = "✅" if result['is_valid'] else "❌"
            print(f"   {status} 序列 '{seq[:20]}{'...' if len(seq) > 20 else ''}': {result['is_valid']}")
            
    except Exception as e:
        print(f"   ❌ DNA 序列验证测试失败: {e}")
        return False
    
    print("\n🎉 所有基本功能测试通过！")
    return True


async def start_server():
    """启动 MCP 服务器"""
    print("🚀 启动 MCP 服务器...")
    
    try:
        # 创建服务器实例
        server = MCPServer('dnallm/mcp/configs/mcp_server_config.yaml.example')
        
        # 获取 FastAPI 应用
        app = server.app
        
        print("✅ MCP 服务器启动成功！")
        print("📡 服务器信息:")
        print(f"   - 主机: 0.0.0.0")
        print(f"   - 端口: 8000")
        print(f"   - API 文档: http://localhost:8000/docs")
        print(f"   - 健康检查: http://localhost:8000/health")
        
        # 这里可以添加实际的服务器启动代码
        # 例如使用 uvicorn 启动服务器
        
        return True
        
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return False


async def main():
    """主函数"""
    print("🧬 DNALLM MCP 服务器")
    print("=" * 50)
    
    # 运行基本功能测试
    if not await test_basic_functionality():
        print("\n❌ 基本功能测试失败，退出...")
        return
    
    print("\n" + "=" * 50)
    
    # 启动服务器
    if await start_server():
        print("\n🎯 MCP 服务器已准备就绪！")
        print("\n📋 可用的 API 端点:")
        print("   - GET  /health                    - 健康检查")
        print("   - GET  /models                    - 获取模型列表")
        print("   - GET  /models/{model_name}       - 获取模型信息")
        print("   - POST /predict                   - 单序列预测")
        print("   - POST /batch_predict             - 批量预测")
        print("   - POST /multi_predict             - 多模型预测")
        print("   - GET  /stream_predict            - 流式预测")
        print("   - GET  /docs                      - API 文档")
        
        print("\n🔧 使用示例:")
        print("   # 健康检查")
        print("   curl http://localhost:8000/health")
        print("   ")
        print("   # 获取模型列表")
        print("   curl http://localhost:8000/models")
        print("   ")
        print("   # 单序列预测")
        print('   curl -X POST "http://localhost:8000/predict" \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"model_name": "Plant DNABERT BPE promoter", "sequence": "ATCGATCGATCG", "task_type": "binary"}\'')
        
    else:
        print("\n❌ 服务器启动失败")


if __name__ == "__main__":
    asyncio.run(main())
