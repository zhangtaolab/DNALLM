# Phase 9: CI Wiring & Census Verification - Context

**Gathered:** 2026-10-05
**Status:** Ready for planning

<domain>
## Phase Boundary

The nightly census formally gates the finished execution-test layer — verified end to end on the real runner for collection, skip audit, runtime budget, hygiene steps, lock consistency, and the documented coverage expectation. Carries the owner closeout decisions from Phase 8 (2026-10-05) into concrete CI wiring: evo/giants exit the pytest census, the models-cache layer is removed, runtime cuts (epochs/num_ctx) land test-side, and ty/mypy swap enforcement roles.

</domain>

<decisions>
## Implementation Decisions

### evo/giants 退出 pytest census(owner 2026-10-05 政策 + 本次机制定案)
- **D-01:** evo 退出机制 = **`giants` pytest marker**(注册进 `pyproject.toml [tool.pytest.ini_options] markers`),example-nightly 跑 `-m "not giants"`。不是 typed-skip — runner 环境明明可用,是按 owner 策略排除;marker 语义自文档,本地/CI 任何调用都可用。
- **D-02:** 退出后的新 census 基线 = **Phase 9 执行期重建**:跑一次含全部削减(evo 退出 + epochs 1 + num_ctx 8k)的基线 census,新权威计数(预计 190P±)记进 09 rollup;**同一次跑兼做超时实测**(见 D-12)。
- **D-03:** 夜间 job 加**采集数硬断言**:`pytest --collect-only` 计数 == 期望值,否则红 — 防 marker 写错导致整类测试静默消失(criterion 1 的采集完整性与 criterion 4 的 lock 守卫互补)。
- **D-04:** example-nightly 里 evo 专属 CI 步骤**全部移除**(giants prefetch、evo venv 构建、flash-attn wheelhouse 缓存)。**flash-attn build-isolation bug(run 37278002681)随之消解 — 修复=删除,不做 torch-first 构建改造**。本地 `~/models-giants` 资产保留(dispatch/手动道仍可用,永不清理 per owner rule)。

### 运行时间削减(owner A/A + 本次落地)
- **D-05:** finetune_custom_head epochs 3→1 经**测试沙箱专用 YAML 补丁**(harness sandbox-patch 步骤,committed notebook 内容不变;循环体一致,可执行性证明不变)。~31→~11 分钟。
- **D-06:** mcp_example 对经 D-13/探针层注入**每请求 `options.num_ctx` ~8k**(同模型 per D-11、同轮数、notebook 不动;今天实测 256k kv-cache 占 36GB 且拖慢每轮 — 顺手解决显存贴边)。
- **D-07:** 两项削减均需**同船测试**(sandbox 补丁只影响沙箱副本、num_ctx 注入点有契约测试)。

### ty 硬门 / mypy 退役(owner 政策 + 本次路径定案)
- **D-08:** **分阶段**:Phase 9 先接 ty 为 coverage-gate 快腿的 **advisory 独立 step**(hosted 秒级,ruff `--statistics` step 同层);E-family 165→0 分诊作为**后续 quick task**(继承 261003-0p0 的 E-family 分诊清单);归零后一次性翻硬门。
- **D-09:** **mypy 退役与 ty 翻硬门同一原子变更**(pre-commit + CI + `scripts/check_code.py` + `[tool.mypy]` 配置一起动)— 不存在"双无类型门"窗口。
- **D-10:** 静态检查不进 pytest(lint 道与行为道分开)。

### 缓存层与清理(owner 2026-10-05)
- **D-11:** **模型缓存层从 CI 移除**(冷拉已实证:65 分钟全冷 stage 1 vs 2700 分钟预算;lock-only 15.2GiB > 10GB 配额且从未成功存过)— 本相落 ci.yml 编辑;uv wheelhouse / bedtools prefix 等小件缓存保留。本地四处缓存(hf/ms/giants/ollama)**永不清理**(owner rule,记忆文件 no-routine-cache-cleanup)。

### 预算与卫生(本次定案)
- **D-12:** 削减后的超时预算**实测后定**:D-02 的基线 census 实测值直接写回 budgets/sum-of-ceilings 注释 — 不吃比例估算风险。
- **D-13:** 夜间卫生步骤 = **阶段间命名步骤**(kernel pkill + VRAM 断言 + 前后值日志输出;现有 stage 2.5 cleanup 扩展为正式命名步)— criterion 3 "可观察" 的字面满足。
- **D-14:** `if:always()` 工件上传**全量**:所有阶段日志 + census 输出 + server 日志(fail-soft job 的失败现场保全)。
- **D-15:** 覆盖率预期(criterion 5)**写进 docs/**(CI/测试页 + AUDIT-04 交叉引用:kernel 子进程不计入 96.30% 门的设计说明),不止 planning 文档。

### re-dispatch 验证策略(本次定案)
- **D-16:** **逐计划 dispatch**:Phase 9 每个计划落地其 ci.yml 部分后立即 dispatch example-nightly 并盯结果(增量验证;first-dispatch consumption 先例)。
- **D-17:** **绿跑为门**:终计划的 verify 门包含**一次绿色完整 example-nightly dispatch**(criterion 1 的正式确认)。
- **D-18:** 两个瞬态失败腿(test-mamba / coverage-nightly)在**本相范围内 re-dispatch**(出口稳定后;便宜、确认恢复、不留悬案)。

### 其他随相决定
- **D-19:** example-nightly 05:30 cron 双触发 coverage/test-mamba 的**一行 job-gate 修复**纳入本相(08-06 遗留 open flag,owner 已默认随相处理)。
- **D-20:** dependabot torch ignore 规则已落(16a9ffb);PR #42 保持开放作记录,不合并。

### Claude's Discretion
- `giants` marker 的具体命名与放置(测试函数 vs NOTEBOOK_EXEC_SPECS 派生标记)
- ty 在 CI 的调用形式与版本 pin(uvx vs dev-dep 精确 pin,ruff ==0.16.9 先例)
- 采集数断言的实现形式(env var 期望值 vs 内联字面量 vs 从 rollup 派生)
- docs 页面结构与 AUDIT-04 交叉引用的具体措辞
- D-05 沙箱补丁的实现缝隙(seed_sandbox 钩子 vs spec-env 扩展)
- D-06 num_ctx 注入的具体机制(环境变量 vs 测试夹具改写客户端参数)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 8 交接(直接输入)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CONTEXT.md` — D-01..D-21(修复节奏/nightly 布局/ollama/巨模锁/marimo/wrapper/版本戳全链)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-CENSUS-ROLLUP.md` — census 台账、缓存配额实测、08-09 继承注记(≥35Gi 纪律)
- `.planning/phases/08-full-execution-rollout-repair-loop/08-VERIFICATION.md` — 通过基线与已知开放项

### 基础设施与契约
- `.github/workflows/ci.yml` — example-nightly 现状(staged-serial D-07、05:30 cron、待删的 evo 步骤在 ~660-779 区)
- `models.lock` — 24 行/14 pin;criterion 4 一致性守卫的对象
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — typed-skip 白名单门(criterion 1)
- `.github/dependabot.yml` — torch ABI 上界 ignore 理由(PR #42 处置背景)
- `tests/examples/_execution.py` — NOTEBOOK_EXEC_SPECS、spec-env 缝隙(D-05/D-06 的接入点)
- `scripts/runner/ollama.service` + README — D-06 的 num_ctx 注入面

### 新鲜知识库(2026-10-05 重建)
- `.planning/codebase/TESTING.md` — 测试全景与 typed-skip 契约
- `.planning/codebase/CONCERNS.md` — open flag 全录 + 7 个潜伏库 bug 行号(后者不在本相范围)
- `.planning/codebase/STACK.md` — extras 结构与原生内核栈约束

### 历史决策链
- `.planning/phases/07-planthelixseek-showcase-notebooks/07-CONTEXT.md` — D-13 nightly-only 执行道、D-14 超时算术交接(+2400s CRE / +5400s Anno 的 sum-of-ceilings 评论在本相重写)
- `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CONTEXT.md` — D-04 runner 确认边界、D-08 全树底线

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- stage-1 mcp deselect 先例(ci.yml)— D-01 的 `-m "not giants"` 直接类比
- spec-env 缝隙(08-08 lora 镜像端点同款)— D-05/D-06 注入点的现成模式
- D-13 重试探针(~60s×2s 证据入消息)— D-06 num_ctx 的注入层
- wheelhouse/bedtools 缓存模式 — 保留小件缓存的参照
- timeout 阶梯(600–7200s cell_timeout)— D-12 重写对象

### Established Patterns
- fail-soft + 汇总非零(D-08/08);honest gates(audit_skips fail-closed)
- 逐项全量对账(D-03/08)— D-02 基线 census 沿用分块模式
- 原子提交 + 同船测试 — D-05/D-06/D-08 全部适用

### Integration Points
- example-nightly job 的 pytest 调用行(marker + collect-only 断言)
- `pyproject.toml` markers 注册 + `[tool.ty.*]` + `[tool.mypy]` 移除点
- `.pre-commit-config.yaml`(mypy 钩子退出)
- `scripts/check_code.py`(ty step 接入/mypy 段移除)
- docs/ CI 页(覆盖预期)

</code_context>

<specifics>
## Specific Ideas

- owner 明确反对把 evo 退出做成 typed-skip("环境可用却被 skip 不诚实")— marker 是唯一认可机制
- "绿跑为门"(D-17)是硬性完成条件,不是建议
- 缓存清理在本项目是敏感操作:任何涉及删除缓存的计划步骤都需要 owner 事先批准

</specifics>

<deferred>
## Deferred Ideas

- CONCERNS.md 记录的 7 个潜伏库 bug(inference.py:1645 DataLoader no-op、mutagenesis.py:429、model.py:264/276 cosine、data.py:983 反向互补、metrics.py:612 嵌套 r2、configs.py:118 别名、benchmark.py:296)— 修复属独立质量批次,本相不碰
- torch 上界放宽(PR #42)— 需协调的内核重建+夜间验证周期,v1.1 里程碑外
- WINDOWS.md 两处陈旧项(#13 mcp_example 已闭环、ship-triage WR-01 与现行 ci.yml 矛盾)— 清账随 ship 流程
- TypedDict 消费者剩余(cli 层)— 与 typing 特项合并

</deferred>

---

*Phase: 9-CI Wiring & Census Verification*
*Context gathered: 2026-10-05*
