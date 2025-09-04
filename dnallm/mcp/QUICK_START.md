# DNALLM MCP Server 快速开始指南

## 1. 环境准备

```bash
# 激活虚拟环境
cd /Users/forrest/GitHub/DNALLM
source .venv/bin/activate

# 进入MCP目录
cd dnallm/mcp
```

## 2. 启动服务器

```bash
python start_server.py --config ./configs/mcp_server_config.yaml
```

等待模型加载完成（首次运行需要下载模型，约1-2分钟）。

## 3. 测试功能

```bash
# 运行功能测试
python test_mcp_functionality.py
```

## 4. 集成到 Claude Desktop

编辑 `~/.config/claude-desktop/config.json`：

```json
{
  "mcpServers": {
    "dnallm": {
      "command": "python",
      "args": [
        "/Users/forrest/GitHub/DNALLM/dnallm/mcp/start_server.py",
        "--config",
        "/Users/forrest/GitHub/DNALLM/dnallm/mcp/configs/mcp_server_config.yaml"
      ]
    }
  }
}
```

重启 Claude Desktop 即可使用 DNA 预测功能。

## 5. 基本使用

在 Claude 中，你可以直接说：

- "请分析这个DNA序列是否为启动子：ATCGATCGATCG"
- "使用多个模型预测这个序列的功能"
- "列出所有可用的DNA预测模型"

## 故障排除

- **模型加载失败**: 检查网络连接，确保能访问 ModelScope
- **服务器启动失败**: 检查配置文件路径和格式
- **预测结果为空**: 确保序列只包含 A, T, G, C 字符
