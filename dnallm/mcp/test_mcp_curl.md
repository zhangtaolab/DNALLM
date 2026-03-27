# MCP服务器测试指南

## 重要说明

DNALLM MCP服务器使用FastMCP框架，通过stdio传输协议与MCP客户端通信，**不支持直接的HTTP请求**。

## 启动MCP服务器

在独立终端中启动服务器：

```bash
cd /Users/forrest/GitHub/DNALLM
source .venv/bin/activate
cd dnallm/mcp
python start_server.py --config ./configs/mcp_server_config.yaml
```

## 测试方法

### 方法1: 功能测试（推荐）

运行完整的功能测试脚本：

```bash
cd /Users/forrest/GitHub/DNALLM/dnallm/mcp
python test_mcp_functionality.py
```

这个脚本会：
- 启动服务器
- 加载所有模型
- 测试所有预测功能
- 显示详细的预测结果
- 自动关闭服务器

### 方法2: 单元测试

运行单元测试：

```bash
cd /Users/forrest/GitHub/DNALLM/dnallm/mcp
python run_tests.py
```

### 方法3: 集成到Claude Desktop

将服务器添加到Claude Desktop配置中：

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

重启Claude Desktop后，可以直接在对话中使用DNA预测功能。

## 测试结果示例

成功运行 `test_mcp_functionality.py` 后，你会看到类似以下的输出：

```
2025-09-04 17:03:55 | INFO | Creating DNALLM MCP Server...
2025-09-04 17:03:55 | INFO | Initializing server...
🚀 Starting to load 3 enabled models:
   1. promoter_model
   2. conservation_model
   3. open_chromatin_model

🔄 Loading model: promoter_model
   Model path: zhangtaolab/plant-dnabert-BPE-promoter
   Source: modelscope
   Task type: binary
   Architecture: DNABERT
   📥 Downloading/loading model and tokenizer...
   ✅ Model and tokenizer loaded in 10.93 seconds
   🎉 Successfully loaded model: promoter_model (total: 11.01s)

... (其他模型加载信息)

2025-09-04 17:04:30 | INFO | Testing promoter prediction...
2025-09-04 17:04:30 | INFO | Promoter prediction result:
2025-09-04 17:04:30 | INFO |   Label: Core promoter
2025-09-04 17:04:30 | INFO |   Scores: {'Not promoter': 0.1234, 'Core promoter': 0.8766}
2025-09-04 17:04:30 | INFO |   Confidence: 0.8766

... (其他预测结果)

============================================================
PREDICTION SUMMARY
============================================================
DNA Sequence: AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGTGCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCGCTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGAGGCCATTGCCGGCGGCGCCCCCGGTGAGCCGCTGCACCACGTCCCTGA...
Sequence Length: 250 bp
Promoter Prediction: Core promoter (confidence: 0.8766)
Conservation Prediction: Conserved (confidence: 0.7234)
Open Chromatin Prediction: Active (confidence: 0.6543)
2025-09-04 17:04:35 | INFO | Test completed successfully!
```

## 可用的MCP工具

服务器提供以下工具：

- `health_check`: 健康检查
- `list_loaded_models`: 列出已加载的模型
- `get_model_info`: 获取模型详细信息
- `dna_sequence_predict`: 单序列预测
- `dna_batch_predict`: 批量序列预测
- `dna_multi_model_predict`: 多模型并行预测
- `list_models_by_task_type`: 按任务类型列出模型
- `get_all_available_models`: 获取所有可用模型

## 测试序列示例

```python
test_sequence = "AGAAAAAACATGACAAGAAATCGATAATAATACAAAAGCTATGATGGTGTGCAATGTCCGTGTGCATGCGTGCACGCATTGCAACCGGCCCAAATCAAGGCCCATCGATCAGTGAATACTCATGGGCCGGCGGCCCACCACCGCTTCATCTCCTCCTCCGACGACGGGAGCACCCCCGCCGCATCGCCACCGACGAGGAGGAGGCCATTGCCGGCGGCGCCCCCGGTGAGCCGCTGCACCACGTCCCTGA"
```

## 故障排除

- **模型加载失败**: 检查网络连接，确保能访问 ModelScope
- **预测结果为空**: 确保序列只包含 A, T, G, C 字符
- **服务器启动失败**: 检查配置文件路径和格式
- **测试超时**: 首次运行需要下载模型，请耐心等待