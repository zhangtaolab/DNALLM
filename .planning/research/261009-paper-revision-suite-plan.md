# 论文修订：DNALLM 套件侧修订计划（对齐整合版）

**日期**: 2026-10-09
**基线**: `revision` 分支 `c99fa9d`（= dev，v0.7.1）；基准仓库 dnallmmark @ `808d61e`（审稿人引用 `a44d310`，已 diff）
**输入文档**:
1. 会话评估 A（2026-10-09，本仓库逐文件核查：trainer 评估语义、指标命名、N 过滤、JASPAR 缺失、FLOPs、v1.1 证据）
2. 《代码审查与功能补充方案：DNALLM》（评估 B，S1–S10 方案 + 两仓 E#/F# 互锁表）
3. 审稿意见全文（Editor ×8 + R1 ×3 组 + R2 ×7+3）

**用途**: `/gsd-new-milestone` 的 intake 材料。REV-01…REV-11 为候选 requirements，Phase A/B/C 为候选 roadmap 切分。

---

## 1. 对齐结论

### 1.1 一致项（两份评估互相印证，不再重证）

| 议题 | 共同结论 | 关键证据 |
|---|---|---|
| LoRA/QLoRA 已在库 | `use_lora`（trainer.py:133）/`use_qlora`（configs.py:310）/4-bit `bnb_config`（model.py:558-612）/`prepare_model_for_kbit_training`（trainer.py:157-164）均存在 | R2-2 的真实缺口是 IA³、探针、预设表与基准侧暴露，不是 LoRA 本身 |
| IA³ 缺失 | 全包无 `IA3Config` 引用 | 需新增（REV-04） |
| 零样本 VEP 只有零件 | `scoring()`（inference.py:1746，score_type=embedding/logits/probability/loss）+ `mlm_evaluate`/`clm_evaluate`（mutagenesis.py:258/312）可算分，但无变异-token 对齐、VCF 读取、聚合协议 | 需模块化（REV-08） |
| JASPAR/CIS-BP 无实现 | 全仓库（py/md/yaml）零命中；链路止于 `prepare_tfmodisco_inputs`（mutagenesis.py:583） | 先 B 案（手稿改人工/外部注释措辞），模块作提交后兑现（REV-10） |
| 指标命名需契约机制 | metrics.py 发 `AUROC`/`AUPRC`/`spearmanr`/`pearsonr`；两仓当前已对齐（见 1.2-D2 事实更正） | 需注册表契约（REV-02） |
| 多 seed 缺编排 | `TrainingConfig.seed`（configs.py:291）仅是单值配置 | 需协议组件（REV-09） |
| R2-7 基本已解决 | v1.1 里程碑（2026-10-07）全量真实模型执行修复 `example/`（census 196P/1S/0F）+ MCP live 夜间门控（run 37432001711/37550730293）+ Windows/macOS/多后端 | 回复信引用 CI 证据 + 指出审稿基线 `81bbd02` 早于修复 |
| from-scratch 路径缺失 | `random_init` 全包零命中；`load_model_and_tokenizer` 必载预训练权重 | 需新增（REV-06） |

### 1.2 矛盾与边界分歧——重新评估后的裁决（共 4 处 + 1 处事实更正）

**D1 · R1-2c 评估语义：suite 侧有没有活？**
评估 B 将 split 修复全部划归基准侧（F1），suite 侧无对应项。**裁决：suite 侧补一个小项（REV-01）**。理由：泄漏的机制在 `dnallm/finetune/trainer.py:234-241`——无 dev 时 **test 静默成为 eval 集**（`eval_strategy="no"` 仅在 dev、test 都不存在时设置），叠加基准配置 `load_best_model_at_end: True` + 每任务 `metric_for_best_model`，18 个无 dev 任务等于在测试集上做 checkpoint 选择。即便 F1 修好管线，任何 suite 用户仍可能无声复现同一错误。REV-01 = 默认防泄漏（test 不得自动充当 eval 集，需显式 opt-in）+ `evaluate(split=...)` 显式入口 + 语义文档。F1（管线改用 test 的 `predict`）仍是主修复，两者互补；`infer()`（trainer.py:502）已证明正确路径存在于库中。

**D2 · 指标注册表优先级：P2 还是 P0？**
评估 A 定 P2（两仓当前已对齐），评估 B 定 P0。**裁决：P0**。理由：R1-2b/2c 修复后全部结果将重新生成——历史上正是"键名漂移 → 导出器取到空串 → 人工补 JSON → 溯源断裂"这条链制造了 R1-2d。重跑前把契约（注册表 + 两仓共享单测）立起来，是防止复发的机制层修复，必须在开跑之前完成。
**附带事实更正（对评估 B）**：B 中"管线期望 eval_auroc/eval_spearman_r"是审稿时旧态（`a44d310`）；dnallmmark HEAD 已修复为 `eval_AUROC`/`eval_spearmanr`（diff 仅 4 行键名）。注册表应锚定**当前**管线拼写，alias 表收录旧拼写用于识别历史 JSON。

**D3 · N 过滤统一子集：suite 改 API 还是管线侧解决？**
评估 A 提议给 `validate_sequences`（data.py:842，经 `check_sequence` sequence.py:89 整条丢弃含 N 序列）加统一子集模式（P0）；评估 B 仅在基准侧列 F7 审计。**裁决：降级为文档项（并入 REV-03），主体走基准侧**。理由：统一子集用现有 API 即可在管线侧实现——对全部模型先按最严格字符集（`models_no_char_n` 的 13 模型对应的 `"ACGTacgt|"`）预过滤一次、所有模型共用同一 eval 集，suite 无需新 API。suite 侧保留两件事：(a) `validate_sequences` docstring 写明跨模型可比性警告；(b) 若 F7 审计发现含 N 数据，重跑必须走统一子集（写入 F7 验收）。

**D4 · R2-5 from-scratch：实验项还是 suite 功能？**
评估 A 原将 R2-5 归为"手稿/实验、非本仓库"；评估 B 的 S3 正确指出同一 harness 跑 from-scratch 基线需要 `random_init=True` 加载路径，且已验证缺失。**裁决：采纳评估 B（REV-06）**。学习曲线另有底层支撑：`DNADataset.sampling`（分层 ratio+seed）可直接驱动标注比例子采样。

**其余差异**：评估 B 的 S9（MCP 工具扩展）为评估 A 未覆盖的合理新增，保留为 P2；S2（per-model PEFT 预设表）细化了评估 A 的 P1-3，保留。

---

## 2. 修订计划（suite 侧，REV-01…REV-11）

> 优先级：P0 = 基准重跑开闸前必须；P1 = 修稿窗口内尽量；P2 = 回复信承诺的后续版本。
> 分工边界：本计划只改 DNALLM 套件；dnallmmark 侧 F1–F10 与手稿侧 E# 见配套文档，不在此列。

### Phase A · 协议与契约层（P0，≈2 天）

#### REV-01 评估语义防泄漏（R1-2c）【P0，0.5 天】
- **现状**: trainer.py:234-241 无 dev 时 test 静默成为 eval 集；`evaluate()`（:489）只评配置的 eval 集；正确路径 `infer()`（:502）未被管线使用。
- **改动**: (a) 无 dev 且存在 test 时默认 `eval_strategy="no"`、`load_best_model_at_end=False`，仅显式参数（如 `allow_test_as_eval=True`）可覆盖，覆盖时打 WARN；(b) 新增 `evaluate(split="test"|"dev"|...)` 显式入口（内部走 predict 路径）；(c) trainer docstring 写明 eval 集选择规则与 held-out 语义。
- **验收**: 单测覆盖 3 种 split 组合（dev+test / 仅 test / 仅 train）× 默认与显式覆盖；无 dev 时默认不再于 test 上评估/选 checkpoint。

#### REV-02 指标注册表契约（R1-2d）【P0，0.5 天】
- **现状**: metrics.py 发 `AUROC`/`AUPRC`（:134/:135/:299/:306）、`spearmanr`/`pearsonr`（:181-219）；与管线导出器之间无契约，历史漂移已实际发生。
- **改动**: `dnallm/tasks/metrics/registry.py`——单一 `{canonical_name: (fn, aliases)}` 注册表，aliases 收录 `eval_AUROC`/`eval_spearmanr`（当前管线拼写）与 `eval_auroc`/`eval_spearman_r`（历史拼写，仅识别用）；`resolve(name)` API；metrics.py 全面改走注册表；契约单测覆盖 47 任务用到的全部指标键。
- **验收**: dnallmmark 导出器（F3）import 本表对齐键名并有单测；两仓 CI 各自引用同一表；注册表单测绿。

#### REV-03 文档、术语与可比性警告（Ed-2/Ed-6/R1-3c/联动）【P0，0.5–1 天】
- **改动**: (a) `validate_sequences` docstring + docs 加跨模型 valid_chars 可比性警告（D3 裁决落地）；(b) README/docs 补 LoRA/QLoRA/IA³（REV-04 后）用法章；(c) 术语统一 "DNA large language models"（Ed-2 联动）；(d) CHANGELOG 记 v0.7.1→v0.7.2 修稿条目（回复信证据链：commit 可点开验证）。
- **验收**: docs 构建绿（docs-validation 门控）；新增警告出现在 API 文档；CHANGELOG 条目与实际 commit 对应。

### Phase B · 适配与评估能力（P1，≈5.5 天，可两人并行 ≈3 天）

#### REV-04 IA³ 适配器（R2-2）【P1，0.5 天】
- **改动**: `configs.py` 加 `Ia3Config`（target_modules 默认取 REV-05 预设）；`TrainingConfig.use_ia3`；trainer 初始化分支对齐 LoRA（trainer.py:153-170 处）；与 LoRA 共享 adapter 保存/加载路径（含 inference.py:112-130 的 PeftModel 复用路径）。
- **验收**: 1 个 transformer 模型 + 1 个 Mamba 模型各跑通一个任务；adapter save/reload 往返一致。

#### REV-05 per-model PEFT 预设表（R2-2）【P1，0.5 天】
- **改动**: `configs/presets/lora_targets.yaml`：按架构族（BERT/GPT/Mamba/Gemma/Llama/hybrid）记录 target_modules 与推荐 r，从各模型 config.json 模块名核对生成（不臆测）；`target_modules=None` 时按族自动选并打印日志。
- **验收**: 基准全模型表覆盖（~44）；错误模块名 dry-run 报错；预设表单测防回归。

#### REV-06 from-scratch 加载（R2-5）【P1，0.5 天】
- **改动**: `load_model_and_tokenizer(..., random_init=True)`：载 config/tokenizer 后跳过权重加载、`torch.nn.init` 重初始化；日志打印 "randomly initialized" 与参数哈希（证明未载预训练权重，供回复信引用）。
- **验收**: 同模型 random_init 与预训练两路参数哈希不同；下游训练 loss 曲线显著不同；单测两架构覆盖。

#### REV-07 冻结探针组件（R2-2）【P1，1 天】
- **改动**: `dnallm/inference/probing.py`：`extract_embeddings(...)`（复用 scoring embedding 路径，pooling/layers 可选）+ `fit_probe(kind='logistic'|'mlp')`（sklearn 固定超参、dev 早停）+ 指标输出（走 REV-02 注册表）+ 嵌入 npz 缓存。
- **验收**: 任一模型 × 任一二分类任务端到端；缓存二次运行命中；探针指标与全量微调结果可同表导出（供 F4 lane）。

#### REV-08 零样本 VEP 模块（R2-3，兼答 R1-3e①）【P1，2 天】
- **改动**: `dnallm/inference/vep.py`：① `align_variant(seq,pos,ref,alt,tokenizer)` → 同槽位可评性判定（BPE/k-mer 的 ref/alt 必须落同一 token 槽位，否则记 skip——这是对 R1-3e① token 级 likelihood 可比性质疑的正面回应）；② `score_variant(paradigm='clm'|'mlm')`（CLM Δlog-lik / MLM log-odds，复用 mutagenesis.py:258/312 内核）；③ `evaluate_vcf(...)` → 逐变异分数 + skip 计数 + AUROC/AUPRC（注册表指标）；④ CLI 入口；评分公式写入 docstring 与 README（协议声明）。
- **验收**: ClinVar 抽样 1k × ≥5 模型（CLM/MLM 各若干）出 AUROC，与文献量级一致（GPN/DNABERT-2 参照）；槽位不可评变异被显式 skip 并计数。

#### REV-09 多 seed 协议组件（R1-2a/R2-4）【P1，1 天】
- **改动**: `dnallm/finetune/sweep.py`：`run_seeds(fn, seeds, out_root)` 目录协议 `{model}/{task}/seed_{s}/` + `aggregate_seeds(...) -> mean/sd/ci95_bootstrap`（纯函数）+ 结果 JSON `statistics` 块规范。
- **验收**: 聚合单测（构造已知数组）通过；目录协议与基准 F2 一致；≥3 seed 试跑一个小任务全链路。

### Phase C · 叙事扩展（P2，≈3–4 天，修稿窗口外、回复信承诺）

#### REV-10 JASPAR/CIS-BP 比对模块（R1-3d）【P2，1–2 天】
- **改动**: `dnallm/interpret/motifs.py`：hotspot 窗口 ↔ JASPAR PWM 相似度扫描（log-odds 阈值 + FDR），输出 motif ID/坐标/E 值表。修稿窗口内手稿走 B 案（人工/外部注释措辞），本模块为提交后兑现项。
- **验收**: HBG1 案例 BCL11A motif 命中坐标与论文 Fig 4a 标注一致。

#### REV-11 MCP 工具扩展（R2 叙事/Ed-6）【P2，1–2 天】
- **改动**: `mcp/server.py` 增 `ism_scan`/`hotspots`/`zero_shot_score` 三 tool（包装既有类）；补三者握手回归测试。残余已知项顺手修：`--host/--port` 被 yaml 静默覆盖（v1.1 audit W 项）。
- **验收**: server 起动 → client 调用 3 工具 → JSON 断言；host/port 覆盖行为有测试。

---

## 3. 两仓联动表（suite ↔ dnallmmark ↔ 手稿）

| REV#（本仓） | F#（基准仓） | E#/审稿条目 | 优先级 |
|---|---|---|---|
| REV-01 评估语义 | F1 dev-split 修复 + test predict | R1-2c / E1' | P0 |
| REV-02 指标注册表 | F3 统一导出器 | R1-2d / E1' | P0 |
| —（D3：管线侧统一子集） | F7 N 审计 | R1-3c / E7 | P0/P1 |
| REV-03 文档 | F10 退役标注 | Ed-2/Ed-6/W1 | P0 |
| REV-04+05 IA³+预设 | F4 适配 lane | R2-2 / E3' | P1 |
| REV-07 探针 | F4 探针 lane | R2-2 / E3' | P1 |
| REV-08 VEP | F5 零样本 lane | R2-3/R1-3e① / E5 | P1 |
| REV-09 多 seed | F2 sweep runner（含 G1 梯度累积修复） | R1-2a/R2-4 / E1'/E2' | P0→P1 |
| REV-06 from-scratch | F8 学习曲线 lane | R2-5 / E6' | P1 |
| —（species 按数据集标注，在 F3） | F3/F6 | R1-2d/Table S3 | P0 |
| REV-10 JASPAR | — | R1-3d（B 案先行） | P2 |
| REV-11 MCP | F9 CI | R2-7/Ed-6 | P2 |

> 注：R1-2b（梯度累积累乘，`dnallmmark_pipeline.py:990-994`，configs 于 :814 循环外加载）与 species 溯源（:1229 用 `model_row.get("species")`，`datasets_info.json` 无 species 字段；实证：同一数据集 EPI_GM12878 在不同模型 JSON 中标 "human"/"Microbe"）均为基准仓修复项，列入 F2/F3，此处仅存证供回复信引用。

**suite 侧工作量**: P0 ≈ 2 天（REV-01/02/03）；P1 ≈ 5.5 天（REV-04…09，两人并行 ≈3 天）；P2 ≈ 3–4 天（REV-10/11）。
**依赖链**: REV-02 → 基准 F3（重跑开闸前置）；REV-01 → F1；REV-09 → F2 → E2'（多种子重跑开闸）；REV-04 → REV-05 → F4；REV-07 → F4；REV-08 → F5；REV-06 → F8。
**关键路径**: Phase A（契约层）必须先于基准全量重跑；Phase B 与重跑可流水并行。

---

## 4. GSD 接手指引

- 建议里程碑: **v1.2 "Paper Revision Suite Support"**（`/gsd-new-milestone`，以本文档为 context）。
- Requirements 输入: REV-01…REV-11（Phase A 三项为门槛需求，建议定义为 milestone 定义完成的硬条件）。
- Roadmap 切分: Phase A（2 计划）/ Phase B（4–6 计划，可并行波次）/ Phase C（2 计划，窗口外可延后）。
- 明确 out-of-scope: dnallmmark 侧全部（F#）、手稿文本（E#）、基准重跑机时——由配套文档与修稿计划 v1/v2 承接。
- 开放问题（需 owner 决策后再开 Phase B 对应计划）:
  1. 全量重跑机时与修稿截止（决定 P1 取舍与多种子规模）；
  2. VEP 基准数据集选型（ClinVar + 植物 AraGWAS/Cri-MG? 与 MarinDNA/BEND/dart-eval 重叠度）；
  3. 跨仓契约测试的 CI 接线方式（dnallmmark import dnallm 的版本锁定策略）；
  4. GPN 类显式基因组模型是否纳入 VEP 统一打分协议（架构非 tokenizer，槽位判定规则需扩展）。

---

*对齐整合自评估 A（会话核查）与评估 B（S1–S10 方案）；裁决 D1–D4 的证据均为本仓库/dnallmmark 本地实测（2026-10-09）。*
