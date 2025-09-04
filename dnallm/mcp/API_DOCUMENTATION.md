# DNALLM MCP Server API 文档

## 概述

DNALLM MCP Server 提供了一套完整的 DNA 序列预测工具，通过 MCP (Model Context Protocol) 协议与客户端通信。服务器支持多种 DNA 预测任务，包括启动子预测、保守性分析、开放染色质预测等。

## 服务器信息

- **名称**: DNALLM MCP Server
- **版本**: 0.1.0
- **协议**: MCP (Model Context Protocol)
- **传输**: stdio
- **支持的任务类型**: binary, multiclass, multilabel, regression

## 可用工具

### 1. dna_sequence_predict

单序列预测工具，使用指定模型预测单个DNA序列。

**参数:**
- `sequence` (string, 必需): DNA序列，只包含 A, T, G, C 字符
- `model_name` (string, 必需): 模型名称

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"0\": {\"sequence\": \"ATCG...\", \"label\": \"Core promoter\", \"scores\": {\"Not promoter\": 0.1234, \"Core promoter\": 0.8766}}}"
    }
  ],
  "model_name": "promoter_model",
  "sequence": "ATCG..."
}
```

**示例:**
```python
result = await session.call_tool("dna_sequence_predict", {
    "sequence": "GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT",
    "model_name": "promoter_model"
})
```

### 2. dna_batch_predict

批量序列预测工具，使用指定模型预测多个DNA序列。

**参数:**
- `sequences` (array[string], 必需): DNA序列列表
- `model_name` (string, 必需): 模型名称

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"0\": {\"sequence\": \"ATCG...\", \"label\": \"Core promoter\", \"scores\": {...}}, \"1\": {...}}"
    }
  ],
  "model_name": "promoter_model",
  "sequence_count": 2
}
```

**示例:**
```python
result = await session.call_tool("dna_batch_predict", {
    "sequences": [
        "ATCGATCGATCG",
        "GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT"
    ],
    "model_name": "promoter_model"
})
```

### 3. dna_multi_model_predict

多模型并行预测工具，使用多个模型同时预测DNA序列。

**参数:**
- `sequence` (string, 必需): DNA序列
- `model_names` (array[string], 可选): 模型名称列表，默认使用所有已加载模型

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"promoter_model\": {\"0\": {...}}, \"conservation_model\": {\"0\": {...}}, \"open_chromatin_model\": {\"0\": {...}}}"
    }
  ],
  "model_count": 3,
  "sequence": "ATCG..."
}
```

**示例:**
```python
result = await session.call_tool("dna_multi_model_predict", {
    "sequence": "ATCGATCGATCG",
    "model_names": ["promoter_model", "conservation_model", "open_chromatin_model"]
})
```

### 4. list_loaded_models

列出当前已加载的所有模型及其信息。

**参数:** 无

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"promoter_model\": {\"name\": \"promoter_model\", \"task_type\": \"binary\", \"status\": \"loaded\", ...}, ...}"
    }
  ],
  "loaded_count": 3,
  "models": {...}
}
```

**示例:**
```python
result = await session.call_tool("list_loaded_models", {})
```

### 5. get_model_info

获取指定模型的详细信息。

**参数:**
- `model_name` (string, 必需): 模型名称

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"name\": \"promoter_model\", \"task_type\": \"binary\", \"num_labels\": 2, \"label_names\": [\"Not promoter\", \"Core promoter\"], \"model_path\": \"zhangtaolab/plant-dnabert-BPE-promoter\", \"model_source\": \"modelscope\", \"architecture\": \"DNABERT\", \"tokenizer\": \"BPE\", \"species\": \"plant\", \"task_category\": \"promoter_prediction\", \"performance_metrics\": {\"accuracy\": 0.85, \"f1_score\": 0.82}, \"status\": \"loaded\", \"loaded\": true}"
    }
  ],
  "model_name": "promoter_model",
  "info": {...}
}
```

**示例:**
```python
result = await session.call_tool("get_model_info", {
    "model_name": "promoter_model"
})
```

### 6. list_models_by_task_type

按任务类型列出所有可用模型。

**参数:**
- `task_type` (string, 必需): 任务类型（binary, multiclass, multilabel, regression）

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"promoter_model\": {...}, \"conservation_model\": {...}}"
    }
  ],
  "task_type": "binary",
  "model_count": 2,
  "models": {...}
}
```

**示例:**
```python
result = await session.call_tool("list_models_by_task_type", {
    "task_type": "binary"
})
```

### 7. get_all_available_models

获取所有可用模型的信息（从model_info.yaml）。

**参数:** 无

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"promoter_model\": {...}, \"conservation_model\": {...}, ...}"
    }
  ],
  "total_models": 3,
  "models": {...}
}
```

**示例:**
```python
result = await session.call_tool("get_all_available_models", {})
```

### 8. health_check

检查服务器健康状态。

**参数:** 无

**返回格式:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\": \"healthy\", \"loaded_models\": 3, \"total_configured_models\": 3, \"server_name\": \"DNALLM MCP Server\", \"server_version\": \"0.1.0\"}"
    }
  ],
  "health": {...}
}
```

**示例:**
```python
result = await session.call_tool("health_check", {})
```

## 错误处理

所有工具在出错时会返回包含 `isError` 字段的响应：

```json
{
  "content": [
    {
      "type": "text",
      "text": "错误信息"
    }
  ],
  "isError": true
}
```

常见错误：
- 模型未加载
- 序列格式错误
- 模型名称不存在
- 预测失败

## 支持的模型

### 当前已配置模型

1. **promoter_model** (二分类)
   - 任务: 启动子预测
   - 标签: ["Not promoter", "Core promoter"]
   - 模型: zhangtaolab/plant-dnabert-BPE-promoter

2. **conservation_model** (二分类)
   - 任务: 保守性预测
   - 标签: ["Not conserved", "Conserved"]
   - 模型: zhangtaolab/plant-dnabert-BPE-conservation

3. **open_chromatin_model** (多分类)
   - 任务: 开放染色质预测
   - 标签: ["Closed", "Open", "Active"]
   - 模型: zhangtaolab/plant-dnabert-BPE-open_chromatin

## 性能说明

- 模型加载时间: 首次启动需要下载模型（约30-60秒）
- 预测响应时间: 单序列预测 < 1秒
- 内存使用: 约2-4GB（3个模型）
- 支持设备: CPU, CUDA, MPS (Apple Silicon)

## 最佳实践

1. **序列长度**: 建议序列长度在64-512bp之间
2. **批量预测**: 对于多个序列，使用 `dna_batch_predict` 比多次调用 `dna_sequence_predict` 更高效
3. **多模型预测**: 使用 `dna_multi_model_predict` 进行综合分析
4. **错误处理**: 始终检查返回结果中的 `isError` 字段
5. **模型管理**: 定期使用 `health_check` 检查服务器状态
