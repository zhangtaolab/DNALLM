# Phase 9: CI Wiring & Census Verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-05
**Phase:** 9-CI Wiring & Census Verification
**Areas discussed:** evo 退出机制, ty 硬门路径, re-dispatch 验证策略, 预算与卫生步骤

---

## evo 退出机制

| Option | Description | Selected |
|--------|-------------|----------|
| giants marker (Recommended) | 新增 pytest marker,example-nightly 跑 `-m "not giants"`;语义自文档、任何调用可用、不是 skip(诚实:环境可用按策略排除) | ✓ |
| job 级 deselect | 仅 example-nightly 的 pytest 调用加 deselect 表达式(stage-1 mcp 先例);排除藏在 job YAML,本地全量仍含 evo | |
| 降级 GATED | NOTEBOOK_EXEC_SPECS 降 GATED(venv 探针 skip);runner 环境明明可用,typed-skip 造成"环境不可用"假象,污染 skip 审计 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 基线 census 重建 (Recommended) | Phase 9 执行期跑一次含全部削减的基线 census,新计数进 09 rollup,兼做超时实测 | ✓ |
| 只改注释 | 196P 保留历史,新 count 只写 ci.yml 注释;省一次 ~2.5h 运行但超时只能估算 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 加硬计数断言 (Recommended) | collect-only 计数 == 期望值否则红;防 marker 写错导致整类测试静默消失 | ✓ |
| 不加 | 仅逐测试 pass/fail;静默丢失采集风险 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 全部移除 (Recommended) | giants prefetch + evo venv + flash-attn wheelhouse 步骤全删;flash-attn bug(37278002681)随之消解;本地 ~/models-giants 保留(永不清理) | ✓ |
| 保留步骤 | 夜间多付构建/拉取时间,无消费方 | |

**User's choice:** giants marker / 基线重建 / 硬断言 / 全部移除(4/4 推荐项)
**Notes:** 用户全选推荐;明确反对降级 GATED 的"假 skip"语义。

---

## ty 硬门路径

| Option | Description | Selected |
|--------|-------------|----------|
| 分阶段 (Recommended) | Phase 9 接 advisory step;E-family 165→0 归零为后续 quick task(继承 261003-0p0 分诊清单);归零后翻门 | ✓ |
| 全部进 Phase 9 | 165 条分诊并入本相计划;一步到位但相变重 | |
| 硬门+重抑制 | 立即硬门 + 大量 suppression;审计负担后置,抑制质量有风险 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 快腿独立 step (Recommended) | coverage-gate 内独立 step,hosted 秒级,随 push/PR 生效;ruff --statistics 同层 | ✓ |
| 独立 job | 多一个 job 维护面,收益不明显 | |
| 仅 pre-commit | 不拦 CI,门强度不够 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 同翻同退 (Recommended) | mypy 退出与 ty 翻硬门同一原子变更;不存在双无门窗口 | ✓ |
| 先退后翻 | 存在无类型门窗口 | |

**User's choice:** 分阶段 / 快腿独立 step / 同翻同退(3/3 推荐项)
**Notes:** owner 此前(16:48)已拍板 ty 升硬门、mypy 退休的政策;本次定的是路径与时序。

---

## re-dispatch 验证策略

| Option | Description | Selected |
|--------|-------------|----------|
| 逐计划 dispatch (Recommended) | 每计划落地其 ci.yml 部分后立即 dispatch + 盯结果;first-dispatch consumption 先例 | ✓ |
| 最后一次跑 | 全部落地后一次性验证;叠错难定位 | |
| 等自然 cron | 反馈最慢,周期被 05:30 拖长 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 绿跑为门 (Recommended) | 终计划 verify 门包含一次绿色完整 example-nightly dispatch(criterion 1 正式确认) | ✓ |
| 不作为硬门 | 仅靠各步验证与历史证据 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 本相重发 (Recommended) | test-mamba + coverage-nightly 在出口稳定后 re-dispatch;便宜、确认恢复、不留悬案 | ✓ |
| 等自然调度 | 可能又碰出口抖动,悬案留着 | |

**User's choice:** 逐计划 dispatch / 绿跑为门 / 本相重发(3/3 推荐项)
**Notes:** 讨论中确认:evo 步骤全删使 flash-attn 修复从"torch-first 构建改造"简化为"删步骤"。

---

## 预算与卫生步骤

| Option | Description | Selected |
|--------|-------------|----------|
| 实测后定 (Recommended) | 基线 census 实测值直接写回 budgets/sum-of-ceilings 注释 | ✓ |
| 比例估算 | 用削减比例估算;快但可能与实际信封不吻合 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 阶段间命名步 (Recommended) | kernel pkill + VRAM 断言 + 前后值日志;stage 2.5 扩展为正式命名步;criterion 3 "可观察"字面满足 | ✓ |
| 仅末尾 | 仅 job 末尾清理;不满足可观察判据 | |

| Option | Description | Selected |
|--------|-------------|----------|
| 全量上传 (Recommended) | if:always() 上传所有阶段日志 + census 输出 + server 日志 | ✓ |
| 最小化 | 仅失败时退出日志 | |

| Option | Description | Selected |
|--------|-------------|----------|
| docs 落地 (Recommended) | docs/ CI/测试页 + AUDIT-04 交叉引用(kernel 子进程不计入 96.30% 门) | ✓ |
| 仅 planning | 团队外不可见,判据存疑 | |

**User's choice:** 实测后定 / 阶段间命名步 / 全量上传 / docs 落地(4/4 推荐项)
**Notes:** 14/14 推荐项全中 — owner 对推荐路径全面认可。

---

## Claude's Discretion

marker 具体命名与放置;ty 调用形式与版本 pin;采集数断言实现形式;docs 页结构;D-05 沙箱补丁缝隙;D-06 num_ctx 注入机制。

## Deferred Ideas

7 个潜伏库 bug(CONCERNS.md 行号清单,独立质量批次);torch 上界放宽 PR #42(内核重建周期,v1.1 外);WINDOWS.md 两处陈旧项(随 ship 清账);TypedDict cli 消费者剩余(typing 特项)。
