# Phase 5: Execution Harness, Honest Gates & Runner Feasibility - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-02
**Phase:** 5-Execution Harness, Honest Gates & Runner Feasibility
**Areas discussed:** WR-08 翻真范围, notebook 陈旧输出处置, spike 执行位置, spike 判决深度

---

## WR-08 翻真范围

### Q1: 去掉 continue-on-error 时,翻真范围选哪个?

| Option | Description | Selected |
|--------|-------------|----------|
| 全部 5 步翻真 | 漂移关闭后把 5 步全部去掉 continue-on-error;若 snippets/YAML 校验暴露 latent 失败,在本阶段内修到绿 | ✓ |
| 只翻确信能绿的三步 | 只翻 docs-sync、example 测试、YAML 加载;snippets/YAML 校验暂留 advisory | |

**User's choice:** 全部 5 步翻真(推荐)
**Notes:** 无隐藏的 advisory 残留;未知红状态由本阶段吸收。

### Q2: docs-validation 要不要升格为分支保护必需检查?

| Option | Description | Selected |
|--------|-------------|----------|
| 加为必需检查 | 像 coverage-gate 一样加为 dev+main 的 required check,假绿问题彻底无回归可能 | ✓ |
| 先非必需观察 | 诚实但非阻塞 —— 失败可见但不拦合并 | |

**User's choice:** 加为必需检查(推荐)

---

## notebook 陈旧输出处置

### Q1: 19/21 个 notebook 的陈旧 committed outputs 怎么处置?

| Option | Description | Selected |
|--------|-------------|----------|
| 保持原样,修复时再生 | 保持 notebook 原样(含陈旧输出)byte-identical 重同步镜像;Phase 8 修复到哪个 notebook 时才刷新它的输出 | ✓ |
| 现在全部清空 | 本阶段就把 21 个 notebook 的 outputs 全部清空;GitHub 上暂时渲染为空 | |
| 只清不一致的 | 只清与源码不一致的(如 megaDNA),其余保持;需先做一致性盘点 | |

**User's choice:** 保持原样,修复时再生(推荐)
**Notes:** GitHub 渲染观感不变;镜像短期携陈旧输出可接受。

---

## spike 执行位置

### Q1: GB10 可行性 spike 在哪里执行?

| Option | Description | Selected |
|--------|-------------|----------|
| 本机先行+runner 复核 | 判决矩阵先在本机 dev 盒迭代,结论在自托管 runner 上用 workflow_dispatch 复核一次固化 | ✓ |
| 只在 runner 上跑 | 全部 spike 通过 workflow_dispatch 在自托管 runner 上跑;判决即终判 | |
| 只在本机跑 | 本机判决即终判,不再上 runner 复核 | |

**User's choice:** 本机先行+runner 复核(推荐)
**Notes:** 本机已验证与 runner 同为 GB10 硬件;runner 维持 schedule/dispatch-only 安全姿态。

---

## spike 判决深度

### Q1: 各家族的可行性判决要到什么深度才算数?

| Option | Description | Selected |
|--------|-------------|----------|
| 分级深度 | pyBigWig=import+读写;evo 家族=最小变体真跑前向 | |
| 统一 import 级 | 只做 import+烟测加载,不跑前向 | |
| 全用 notebook 变体 | 直接用 notebook 实际引用的变体(如 evo-1-131k-base 7B)跑前向 —— 证据最强但成本高 | ✓ |

**User's choice:** 全用 notebook 变体(自定义,非推荐项)
**Notes:** 用户明确要最强证据;下载/时长成本接受。

### Q2: 若某家族的 notebook 变体在 GB10 上跑不通,spike 要不要降级试小变体?

| Option | Description | Selected |
|--------|-------------|----------|
| 失败后降级小变体 | 同一 spike 内再试该家族最小变体;小变体能跑则 Phase 8 用小变体真跑(notebook 改引用),都不行才 typed skip | ✓ |
| 二元判决,不降级 | notebook 变体不行 = 家族 typed skip | |

**User's choice:** 失败后降级小变体(推荐)

---

## Claude's Discretion

- Pilot notebook 选择(健康、快、无巨大下载)
- typed-skip 前缀命名细节(前提:都注册进 expected_skips.yaml 且审计绿)
- 沙箱 fixture 机制(在锁定的 tmp-sandbox + cwd-redirect 模式内)
- 判决矩阵文档格式与位置

## Deferred Ideas

None — 讨论保持在阶段范围内
