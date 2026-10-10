# Phase 8: Full Execution Rollout & Repair Loop - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-04
**Phase:** 8-Full Execution Rollout & Repair Loop
**Areas discussed:** 修复节奏与验收线, Nightly 作业布局与共存, ollama 基础设施, 巨型模型/锁与结构家族, marimo 断言深度, 回归测试归属, wrapper 同步策略, 版本戳策略

---

## 修复节奏与验收线

### 推进分组方式

| Option | Description | Selected |
|--------|-------------|----------|
| 按 census 类别波次 | NT-结构性→BPE→OTHER，同类根因一次修复多点受益 | |
| 按模型家族 | evo 系/NT 系/小模型群分组，家族内 notebook+marimo+script 一起跑修 | ✓ |
| 逐件串行 | census 清单逐项闭环 | |

**User's choice:** 按模型家族
**Notes:** 未选推荐的 census 类别波次。

### notebook 内容修改边界

| Option | Description | Selected |
|--------|-------------|----------|
| 仅修错误 | 保持教学内容/单元格结构不变 | |
| 错误+表述同步 | 顺带更新过时表述，不动结构 | |
| 含结构重构 | 允许单元格合并/拆分/重排，可执行性优先 | ✓ |

**User's choice:** 含结构重构（最宽边界）
**Notes:** planner 需为成套镜像+wrapper 重导出预算成本。

### dev-box 验收时点

| Option | Description | Selected |
|--------|-------------|----------|
| 家族复验+末尾全量 | 家族内复验，phase 末一次全量 census | |
| 逐项全量对账 | 每个修复项后跑全量 census+快道 | ✓ |
| 最简验收 | 只跑被修项，全量推给 runner | |

**User's choice:** 逐项全量对账（最重验收）
**Notes:** 成本接受；runner 确认仍按 Phase-5 D-04 合并后。

### UPSTREAM 不可修类处置

| Option | Description | Selected |
|--------|-------------|----------|
| typed skip | 保留 cell 原样+证据化 skip（05-04 模式） | ✓ |
| 重写数据路径 | 换可工作的等效路径（偏离上游教程） | |
| 逐项判断 | 能修就修，不行才 skip | |

**User's choice:** typed skip

---

## Nightly 作业布局与共存

### 作业归属

| Option | Description | Selected |
|--------|-------------|----------|
| 独立 example job | 启用 CI-06 预授权，与 coverage-nightly 并行 | ✓ |
| 并入 coverage-nightly | 单 job 全量，~970min vs 900min cap 风险 | |
| 先并后拆 | 超时实涨再拆 | |

**User's choice:** 独立 example job

### 调度关系

| Option | Description | Selected |
|--------|-------------|----------|
| 错峰串行 | coverage 03:00 UTC、example 05:30 UTC 起步 | ✓ |
| 同时调度 | 自然竞争 | |
| 依赖链串行 | workflow_run 触发链 | |

**User's choice:** 错峰串行

### VRAM/端口共存（MCP-02）

| Option | Description | Selected |
|--------|-------------|----------|
| 阶段化串行 | 重 torch → MCP :8000 批 → ollama 批，阶段间清理 | ✓ |
| 并行混跑 | 靠自然并发 | |
| ollama 优先 | 小模型快探针先行 | |

**User's choice:** 阶段化串行

### 失败语义

| Option | Description | Selected |
|--------|-------------|----------|
| fail-soft+汇总非零 | 单件不阻后续，job 末尾按失败数非零退出 | ✓ |
| fail-soft 永绿 | 失败只在日志 | |
| fail-fast | 首错即停 | |

**User's choice:** fail-soft+汇总非零

### 模型缓存

| Option | Description | Selected |
|--------|-------------|----------|
| 共享 lock-keyed cache | 复用 coverage-nightly 的 cache key/volume | ✓ |
| 独立 cache | 空间隔离但双份存储 | |
| 先共享后拆 | 记阈值监控 | |

**User's choice:** 共享 lock-keyed cache

### 环境安装

| Option | Description | Selected |
|--------|-------------|----------|
| 全 extras | .[base,fla,dev,mcp] + 系统包 | ✓ |
| 最小逐步加 | 遇缺再加 | |
| 复用现有安装线 | 一字不差 | |

**User's choice:** 全 extras

---

## ollama 基础设施（MCP-01）

### 预拉模型

| Option | Description | Selected |
|--------|-------------|----------|
| notebook 原模型 | 保持教程内容（研究阶段核实体积） | ✓ |
| 换更小模型 | 同步改 notebook 引用 | |
| 研究后定 | 实测权衡 | |

**User's choice:** notebook 原模型

### systemd 定义归属

| Option | Description | Selected |
|--------|-------------|----------|
| 定义进仓 | infra/ 或 scripts/runner/ + README | ✓ |
| 手工不进仓 | 仅计划文档记步骤 | |
| 进仓+自检脚本 | 加 systemctl+curl 自检 | |

**User's choice:** 定义进仓

### 就绪探针

| Option | Description | Selected |
|--------|-------------|----------|
| 测试内探针 | curl /api/tags 重试窗口；失败→typed skip 带证据 | ✓ |
| job 步骤级 | 独立 step 失败即 fail job | |
| 双层探针 | job 快探+测试内保底 | |

**User's choice:** 测试内探针

### 回退线范围

| Option | Description | Selected |
|--------|-------------|----------|
| 仅基础设施缺失 | 服务/模型/端口层；输出异常必须修 | ✓ |
| 含输出不稳 | 模型输出偶发超时也可 skip | |

**User's choice:** 仅基础设施缺失

---

## 巨型模型、锁与结构家族

### evo-1 拉取（CI-05）

| Option | Description | Selected |
|--------|-------------|----------|
| safetensors+本地巨仓 | allow_patterns ~12.9GB + ~/models-giants 不进 quota cache | ✓ |
| 全进 cache | 10GB 配额必溢出 | |
| 直接换 8k 变体 | 避开巨拉但改 notebook 引用 | |

**User's choice:** safetensors+本地巨仓

### models.lock 粒度（CI-04）

| Option | Description | Selected |
|--------|-------------|----------|
| id+sha+对齐 source= | 逐 id、ms 前缀、revision pin、注释 | ✓ |
| 不 pin sha | 每次拉 latest | |
| 粗略记账 | 只记大类别 | |

**User's choice:** id+sha+对齐 source=

### NT-REMOTE-STRUCTURAL 处置

| Option | Description | Selected |
|--------|-------------|----------|
| 小变体→shim→skip | 阶梯推进 | |
| 直接 typed skip | 三项一个处置 | |
| 研究后定 | 先评估 shim 深度再定 | ✓ |

**User's choice:** 研究后定（唯一交给研究阶段的处置）

### lock 覆盖范围

| Option | Description | Selected |
|--------|-------------|----------|
| 全部执行 id | census 全部真实执行过的都入锁 | ✓ |
| 仅新增 id | 现有不动 | |
| 仅 nightly 常驻 | 一次性验证不占锁 | |

**User's choice:** 全部执行 id

---

## 追加灰区

### marimo 断言深度

| Option | Description | Selected |
|--------|-------------|----------|
| 标准三件套 | 无头+默认值+退出码 | |
| 加状态导出 | +app.run 结果关键值断言 | |
| 再加 HTML 校验 | +export-html 产物生成+关键内容校验（不入仓） | ✓ |

**User's choice:** 再加 HTML 校验（最深层）

### 回归测试归属

| Option | Description | Selected |
|--------|-------------|----------|
| 按层归属 | example 修复→tests/examples/；库修复→模块 tests/ | ✓ |
| 全进 examples | 集中但离模块远 | |

**User's choice:** 按层归属

### wrapper 同步策略

| Option | Description | Selected |
|--------|-------------|----------|
| 重导出+同提交 | 原子提交含 notebook+镜像+wrapper+叙述 | ✓ |
| 重导出+分步提交 | docs 单独提交 | |
| 最小修改 | 只保 AST 同步绿 | |

**User's choice:** 重导出+同提交

### 版本戳策略

| Option | Description | Selected |
|--------|-------------|----------|
| 逐本真实戳 | 各自记录实际执行环境版本 | ✓ |
| 批次统一戳 | 整齐但可能失真 | |

**User's choice:** 逐本真实戳

---

## Claude's Discretion

- 家族划分清单与修复顺序（从 census+ACTIVE/GATED 推导）
- 错峰调度的具体时刻
- evo-1 巨仓路径与指向机制（env var vs symlink）
- ollama systemd 单元进仓目录命名
- 结构重构的具体单元格编排

## Deferred Ideas

None — discussion stayed within phase scope
