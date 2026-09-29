# transformers 5.x tokenizer 兼容性分析（临时笔记）

> **⚠️ 临时文档：问题修复（模型仓库元数据修正或代码侧多兼容方案实施）后可以删除本文件。**

日期：2026-09-29 ｜ 分析环境：transformers 5.17.0 / torch 2.7.0 / macOS (MPS) ｜ 相关提交：`4bd943e`

## 现象

加载 DNALLM 模型（如 `zhangtaolab/plant-dnabert-BPE`）时出现 WARNING：

```
AutoTokenizer failed; loaded fast tokenizer from tokenizer.json.
```

（来自 `dnallm/models/model.py` 中 `4bd943e` 加入的兜底逻辑）

## 根因

**模型仓库元数据与实际 tokenizer 文件不匹配**：

| 文件 | 内容 |
|---|---|
| `tokenizer_config.json` | 声明 `"tokenizer_class": "DebertaV2Tokenizer"` —— **slow 版 Unigram (SentencePiece) 类** |
| `tokenizer.json` | 实际是 **BPE 模型**（`model.type: BPE`，vocab 为 `dict{token: id}`，8000 条 DNA k-mer） |

复现的真实异常（被 `except Exception` 吞掉）：

```
File ".../transformers/models/deberta_v2/tokenization_deberta_v2.py", line 112, in __init__
    Unigram(self._vocab, unk_id=unk_id, byte_fallback=False)
TypeError: 'dict' object is not an instance of 'Sequence'
```

版本行为差异：

- **transformers 4.x**：`AutoTokenizer` 自动把声明的 slow 类映射到对应 fast 类，直接加载 `tokenizer.json`，从不构建 Unigram → 不触发
- **transformers 5.x**：忠实构建声明的 slow `DebertaV2Tokenizer`，其内部 `tokenizers.Unigram()` 要求 `(token, score)` 元组序列，收到 BPE 的 dict → `TypeError`

## 现状评估：无害

`4bd943e` 的 try/catch 级联（`AutoTokenizer` → `PreTrainedTokenizerFast` → `DNAOneHotTokenizer`）：

- `PreTrainedTokenizerFast` 直接加载 `tokenizer.json` —— 与 4.x 的 fast 类加载的是**同一个文件、同一个 BPE 模型**，编码行为一致（已验证 DNA 序列编码/解码正常）
- 全量测试（562 passed）+ slow 真实模型测试均通过
- **WARNING 无功能损失，不阻塞任何工作**

## 待实施的修复方案（多兼容，2026-09-29 决定暂缓）

### 设计原则

优先**能力探测**（try/except 级联、pop-kwarg-再换算），避免版本号 if/else（`transformers.__version__` 判断会随版本腐烂，patch 级行为变化也无法覆盖）。

### 具体改动

1. **抽公共 helper**（三处复用）：

```python
def _load_tokenizer_with_fallback(model_name, *, trust_remote_code=True, add_prefix_space=False):
    """Load a tokenizer compatible with transformers 4.x/5.x.

    v5 cannot rebuild slow tokenizers (e.g. Unigram classes named in
    tokenizer_config.json) from BPE tokenizer.json state; fall back to
    PreTrainedTokenizerFast, which matches AutoTokenizer on v4.
    """
    kwargs = {"trust_remote_code": trust_remote_code}
    if add_prefix_space:
        kwargs["add_prefix_space"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception as e:
        logger.debug(f"AutoTokenizer failed for {model_name}: {e!r}")  # 保留原始异常，可诊断
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, **kwargs)  # 不丢 kwargs
        logger.warning(f"AutoTokenizer failed for {model_name} ({type(e).__name__}); "
                       "loaded fast tokenizer from tokenizer.json.")
        return tokenizer
    except Exception as e2:
        logger.debug(f"PreTrainedTokenizerFast also failed: {e2!r}")
    logger.warning(f"All tokenizer loading failed for {model_name}; using DNAOneHotTokenizer.")
    return DNAOneHotTokenizer()
```

2. **修复 `model.py:544` 现有级联的缺口**：兜底未透传 `trust_remote_code` / `add_prefix_space`；原始异常被吞（本次排查 WARNING 原因时绕了弯路的原因）

3. **补 `dnallm/models/special/crossdna.py:506`**：目前完全没有兜底，CrossDNA checkpoint 若有同样元数据问题会在 v5 直接崩溃

4. **可选**：`dnallm/tasks/metrics/perplexity/perplexity.py:126`（一般加载标准 NLP checkpoint，低风险）

### 治本方案（模型仓库侧）

修正 ModelScope/HuggingFace 上模型仓库的 `tokenizer_config.json`：删除 `tokenizer_class` 字段（让自动推断），或改为与 BPE 匹配的类。需逐个仓库修改。

## 验证清单（实施修复时）

- [ ] 全量 `pytest tests/ -m "not slow"` 通过
- [ ] slow 真实模型测试通过（覆盖真实 tokenizer 加载路径）
- [ ] transformers 4.x 与 5.x 双版本下无 WARNING
- [ ] **完成后删除本文件**
