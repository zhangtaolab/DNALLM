# Phase 23: Pre-flight Core — Pure Validation / Suitability / Config API - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-20
**Phase:** 23-Pre-flight Core — Pure Validation / Suitability / Config API
**Areas discussed:** 数据输入面, 判定校准, 可用性探测范围, generate_config 形状

---

## 数据输入面

### Q1: validate_data() 接受什么类型的输入?

| Option | Description | Selected |
|--------|-------------|----------|
| 文件路径 + 序列列表 | str\|Path 文件路径(含 dict-of-splits,与 load_local_data 对齐)+ list[str] 纯序列;hub 数据集名不进 validate_data | ✓ |
| 仅本地文件路径 | 只接受 str\|Path,格式面与 DNADataset.load_local_data 完全一致;测试/壳层必须先落盘 | |
| 路径 + 列表 + hub 名 | 再加 HF/ModelScope 数据集名;把网络拉进 validate_data,离线语义变复杂 | |

**User's choice:** 文件路径 + 序列列表
**Notes:** 问题被一次自由输入打断 —— 用户提出接入 pi agent 用大模型做识别(见 Deferred Ideas);澄清后按推荐选项确认。

### Q(随处置确认): 序列列怎么定位?

**User's choice:** 确定性启发式自动识别 + 显式参数覆盖
**Notes:** 作为「Agent 壳」处置的一部分一并锁定:启发式层未来供 agent 壳复用,LLM 不进此环。

### Q2: 底层怎么读文件?

| Option | Description | Selected |
|--------|-------------|----------|
| 独立轻量读取器 | pandas/json/pyarrow + 手写 fasta,不 import datasets/torch/transformers;结构测试锁与 DNADataset 双读一致 | ✓ |
| 复用 DNADataset 加载 | 单一格式真源零分叉,但拖 HF datasets(+torch 链),TUI 首屏慢 | |
| 规划时再定 | CONTEXT 只记两条路线取舍 | |

**User's choice:** 独立轻量读取器

### Q3: 支持哪些文件格式?

| Option | Description | Selected |
|--------|-------------|----------|
| 除 pkl 外全对齐 | csv/tsv/json/parquet/arrow/fasta/txt;pkl/pickle 给明确 reason code(不对不可信文件执行 pickle.load) | ✓ |
| 完全对齐含 pkl | 与 DNADataset 一字不差,但校验层反序列化不可信文件 | |
| 最小核心子集 | csv/tsv/json/fasta/txt;parquet 用户不少 | |

**User's choice:** 除 pkl 外全对齐
**Notes:** xlsx(需 openpyxl,新依赖)记 deferred。

---

## 判定校准

**总原则(随文提出,后续三问均在其下确认):** FAIL 只留给硬不兼容;可由配置或数据清理缓解的问题最多 WARN + 计数。

### Q1: 截断判定怎么定级?

| Option | Description | Selected |
|--------|-------------|----------|
| 截断永不 FAIL | 长度维度只 OK/WARN;报告 p50/p95/p99/max vs max_length(tokenizer_type 折算字符预算)+ 截断条数/比例;阈值 planner 定 | ✓ |
| 比例阈值三档 | <10% OK,10–50% WARN,>50% FAIL;FAIL 语义变重 | |
| planner 定阈值 | 只锁「不可逆丢数据至少 WARN」 | |

**User's choice:** 截断永不 FAIL

### Q2: 字符集不兼容怎么定级?

| Option | Description | Selected |
|--------|-------------|----------|
| 硬不兼容才 FAIL | ⊆ → OK;N/歧义码被接受 → WARN 逐字符计数;字符集外 → 该模型 FAIL(其他模型继续);报告不丢行 | ✓ |
| 字符集永不 FAIL | 一律 WARN;对无法 tokenize 的模型仍标可用会误导 | |
| planner 定 | 分级规则结合词表定 | |

**User's choice:** 硬不兼容才 FAIL

### Q3: 任务不匹配怎么定级?

| Option | Description | Selected |
|--------|-------------|----------|
| finetuned 硬匹配/base 宽容 | finetuned task_type ≠ 用户任务 → FAIL(reason code);base 任何监督任务 OK;generation/MLM 底座 → WARN | ✓ |
| 任务永不 FAIL | 一律 WARN;违背 D-3 意图路由初衷 | |
| planner 定矩阵 | 结合 TaskConfig 任务枚举定 | |

**User's choice:** finetuned 硬匹配/base 宽容

### Q4(随文声明,未单独问): 统计口径默认

**User's choice:** (默认随文确认,用户未反对)
**Notes:** upper() 归一(小写软屏蔽同碱基,小写比例 info);空序列/超短(<20 可配)error 级;全 N warn;severity 仅标注,不丢行不阻断。

---

## 可用性探测范围

**随文默认(未单独问,用户未反对):** 进程内 memo 缓存;单探默认超时 ≈5s 可配;一切异常 → UNKNOWN(PFC-03/06 已锁)。

### Q1: 调用粒度?

| Option | Description | Selected |
|--------|-------------|----------|
| 单模型+可选短名单 | probe_availability(model) 主 API;check_compatibility 列表默认零联网,可选短名单(前 N≈20)批量;不探 236 全量 | ✓ |
| 仅单模型 | 批量由壳层循环;限流策略每壳重写 | |
| 列表自动探测 | 默认联网,违背零网络默认姿势 | |

**User's choice:** 单模型+可选短名单

### Q2: hub 回退顺序?

| Option | Description | Selected |
|--------|-------------|----------|
| source 优先+另一 hub 兜底 | 按条目 source 字段先探(实验室模型天然 ModelScope-first),未命中再探另一 hub;双源都在报告 ModelScope 为主源 | ✓ |
| 永远 ModelScope→HF | 字面落实 ModelScope-first,但 HF-only 模型多一次注定失败的请求 | |
| 仅探声明源 | 请求最少,但镜像延迟时误报不可用 | |

**User's choice:** source 优先+另一 hub 兜底

### Q3: cached 状态本地检测覆盖?

| Option | Description | Selected |
|--------|-------------|----------|
| 三路全覆盖 | HF scan_cache_dir + ModelScope 双布局解析器(spike)+ 本地路径直接判存在 | ✓ |
| 先 HF+本地路径 | 首版降险,但实验室主场景拿不到 cached | |
| planner 定 | 结合 spike 结论定 | |

**User's choice:** 三路全覆盖

---

## generate_config 形状

### Q1: 生成哪些 section、默认值从哪来?

| Option | Description | Selected |
|--------|-------------|----------|
| 完整可跑配置 | task+model+finetune+inference+lora 全 section,模型推导默认值(effective_max_length;finetuned 预填 num_labels/label_names/threshold);未知模型=字段级错误 | ✓ |
| 最小骨架 | 只生成 task+model;把不懂 ML 的用户推回手写配置 | |
| 两档参数 | minimal=True/False;两套输出双倍测试负担 | |

**User's choice:** 完整可跑配置

### Q2: 往返输入形态?

| Option | Description | Selected |
|--------|-------------|----------|
| 路径 + dict 双入口 | YAML 路径或已解析 dict;编辑以补丁 dict 作用后重验证;TUI/agent/测试零临时文件 | ✓ |
| 仅 YAML 路径 | 与 load_config 同构;测试/agent 先落盘 | |
| 仅 dict | 最纯;每壳重写读+parse 错误处理 | |

**User's choice:** 路径 + dict 双入口

### Q3: 输出形态?

| Option | Description | Selected |
|--------|-------------|----------|
| 返回对象+可选写盘 | 返回完整配置对象(含验证报告);可选 output_path 才写 YAML | ✓ |
| 总是写文件 | 与向导体验一致,但纯函数核心带强制副作用 | |
| 仅返回对象 | 最纯;每壳重写 dump/格式化 | |

**User's choice:** 返回对象+可选写盘

---

## Claude's Discretion

- 报告/verdict 对象类名与字段表;reason code 枚举命名(agent 友好:结构化、自描述、机器可读)
- 截断 WARN 阈值(分位数/比例)、BPE 保守折算比例、短名单 N、超时默认值、探测并发上限
- 独立读取器内部结构;双读一致性结构测试的具体锁点
- 生成/回归/多标签任务的 section 差异、多标签分隔符默认(循 TaskConfig 既有约定)
- preflight 是否进根 facade —— 倾向不进(启动轻,从 dnallm.preflight 导入),planner 定

## Deferred Ideas

- **Agent 壳·麻瓜数据引导**(未来 Track,非 v1.4):对话式 agent(Pi / Claude / pydantic-ai;参考 https://pi.dev/docs/latest)经 D-6(MCP 工具暴露 pre-flight 核心)消费纯函数核心,自动识别序列列/标签列/推断任务类型,把不懂生信/ML 用户的原始数据整理成 DNALLM 可识别格式与任务;判定逻辑永远在核心,不进 LLM。阶段 23 呼应:报告对象 agent 友好 + 确定性列检测启发式。
- **xlsx/Excel 输入支持**(需 openpyxl 新依赖,超出 v1.4 依赖预算)。
