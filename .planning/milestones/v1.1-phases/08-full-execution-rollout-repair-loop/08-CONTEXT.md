# Phase 8: Full Execution Rollout & Repair Loop - Context

**Gathered:** 2026-10-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Roll out real execution of the ENTIRE `example/` tree on the nightly GPU runner — all 21 census notebooks, the 3 marimo apps, `generate_bpe_dataset.py`, every example YAML, the two ollama-backed mcp_example notebooks, and the two Phase-7 showcase notebooks — and fix every error real execution surfaces, across example code, the docs mirror, and the dnallm library. Also lands the enabling infrastructure: the CI-06-pre-authorized separate example-execution nightly job, models.lock coverage for every executed model id, the evo-1 giant-tier cache strategy (CI-05), and loopback-only ollama on the runner (MCP-01/02). Delivers EXEC-02, EXEC-03 (re-verify), EXEC-04, EXEC-05, REPAIR-01, REPAIR-04, CI-04, CI-05, MCP-01, MCP-02.

</domain>

<decisions>
## Implementation Decisions

### 修复节奏与验收线
- **D-01:** 推进分组按**模型家族**（evo 系 / NT 系 / 小模型群 / mcp+ollama 批等），家族内 notebook+marimo+script 一起跑一起修 — 不按 census 错误类别波次、不逐件串行。家族清单由 planner 从 census+ACTIVE/GATED 清单推导。
- **D-02:** 修复时对 notebook 内容的修改边界 = **允许结构重构**（单元格合并/拆分/重排），以可执行性优先；不受"仅修错误"约束。镜像随之字节同步（D-11 沿袭）。— **Reversibility:** costly — 结构重构后 wrapper 摘录与镜像需成套重导出，回退需恢复整本 notebook。
- **D-03:** dev-box 侧验收 = **逐项全量对账**：每个修复项落地后跑一次全量 census（21+3+1+YAML）对账 + 全快道，每步都有全量基线（成本接受）。runner 官方确认仍按 Phase-5 D-04 合并后进行。
- **D-04:** 上游不可修类（UPSTREAM 根因，如 BPE tokenizer 工件本身损坏且 dnallm 侧无修复点）= 保留 cell 原样 + **证据化 typed skip**（复用 05-04 模式，证据进 skip 消息）；不重写数据路径偏离上游教程。

### Nightly 作业布局与共存
- **D-05:** example 执行落**独立 example-execution nightly job**（启用 CI-06 预授权），与 coverage-nightly 并行调度、时长预算互不挤占 — **Reversibility:** costly — job 拆分后合并需重排全部 nightly 预算与顺序约定。
- **D-06:** 两 nightly **错峰串行**（如 coverage 03:00 UTC、example 05:30 UTC 起步；具体时刻 planner 定），避免同时占 GPU/磁盘带宽；job 内部串行。
- **D-07:** VRAM/端口共存（MCP-02）= **阶段化串行**：job 内先重 torch 执行（含 example notebooks）→ 后 MCP live-server 探针批（:8000）→ 最后 ollama 批；阶段间显式清理 VRAM/进程；排序写入 ci.yml 注释与计划文档。
- **D-08:** example job 失败语义 = **fail-soft + 汇总非零退出**：单件失败不阻后续项，job 末尾按失败数非零退出（honest-gates 原则；不允许永绿）。
- **D-09:** 模型缓存 = **共享 coverage-nightly 的 models.lock-keyed hub cache**（同 key 只读复用）；巨型模型单独 tier 不进此 cache（见 D-13）。
- **D-10:** 环境安装 = **全 extras** `.[base,fla,dev,mcp]`（系统依赖如 bedtools 在 job 步骤装），对齐 runner 现有 nightly 腿安装线。

### ollama 基础设施（MCP-01）
- **D-11:** 预拉模型 = **notebook 原引用模型**（保持教程内容不变；前提体积/VRAM 可行——研究阶段核实具体 id 与大小）。
- **D-12:** systemd 服务**定义文件进仓**（infra/ 或 scripts/runner/，含 README），runner 上一次手工 enable；可审计可重建。loopback-only（127.0.0.1）。
- **D-13（探针/回退）:** 就绪探针 = **测试内探针**：ollama 批测试前置 `curl http://127.0.0.1:11434/api/tags` 重试窗口（~60s×2s）；失败 → typed `network-unavailable:` skip，curl 输出+重试日志作证据写进 skip 消息。回退线**仅限基础设施缺失**（服务未起/模型未拉/端口不可达）；模型在但输出内容异常属测试断言问题，必须修、不许 skip。

### 巨型模型、锁与结构家族
- **D-14:** evo-1（CI-05）= snapshot_download **allow_patterns=safetensors-only（~12.9GB）** + 巨型模型放 runner **本地巨仓目录**（如 ~/models-giants/，经 env/symlink 指向；具体机制研究阶段定），不进 10GB-quota cache、永不驱逐热缓存。— **Reversibility:** costly — 巨仓布局与 cache 策略进入 job 定义后重排成本高。
- **D-15:** models.lock（CI-04）新条目 = **逐 id + ms 前缀 + revision pin（commit sha）+ 用途注释**，对齐现有两行格式；notebook 的 `source=` 与前缀对齐（不一致处改 notebook）。
- **D-16:** lock 覆盖范围 = **census 全部真实执行过的模型 id**（ACTIVE×13 + GATED×8 已覆盖的 + 本阶段新跑的）；~8+ 只是下限。
- **D-17:** NT-REMOTE-STRUCTURAL 家族（v2-100m-promoter 等 3 项、05-04 梯子遗留 owner 处置）= **研究阶段先评估 shim 深度**（离 HEAD 多远、vendor 面多大、小变体是否同坎）再定：可修则小变体→加深 shim 真实执行，不可修才证据化 typed skip。处置结论必须逐项记账。

### 追加决策（marimo / wrapper / 回归 / 版本戳）
- **D-18:** marimo 断言深度 = 标准三件套（无头跑通 + UI 元素默认值断言 + 退出码断言）**+ export-html 产物生成与关键内容校验**（工件不入仓，仅临时断言后丢弃）。
- **D-19:** 回归测试归属 = **按层归属**：example 内容修复的回归测试进 `tests/examples/`；dnallm 库修复的进对应模块 `tests/`（对齐 owner 的库改动同船 pytest 规则）。
- **D-20:** wrapper 同步 = 修复提交为**一个原子提交**（notebook + 字节镜像 + 重导出的 wrapper 摘录 + 必要的叙述更新），`check_notebook_md_sync` AST 校验绿是提交门槛。
- **D-21:** 版本戳 = **逐本真实戳**：每本 notebook 修复重执行时更新自己的 provenance 版本行（transformers/torch/fla key=value），不追求跨本一致。

### Claude's Discretion
- 家族划分的具体清单（从 census + ACTIVE/GATED 推导）与修复顺序。
- 错峰调度的具体时刻（错开即可）。
- evo-1 本地巨仓的具体路径与指向机制（env var vs symlink）。
- ollama systemd 单元的进仓目录命名（infra/ vs scripts/runner/）。
- 结构重构的具体单元格编排（可执行性优先原则下）。

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 执行基础设施（Phase 5）
- `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CONTEXT.md` — harness 契约、D-01..D-09（含 D-04 dev-box 先行/runner 确认、D-05/D-06 变体规则、D-08 全树底线）
- `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md` — 25 项 census：11 PASS / 12 class-tagged FAIL / 2 deferred（已补绿）——修复队列的证据基线
- `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-FEASIBILITY.md` — GB10 可行性判定（evo-1 8k、evo2 noFP8、megaDNA pinned clone、pyBigWig env-unavailable、marimo export-html）
- `tests/examples/_execution.py` — nbclient harness、NOTEBOOK_EXEC_SPECS、seed_sandbox、assert_tree_clean
- `tests/examples/test_notebook_execution.py` — 执行测试形态与 typed-skip 前缀契约

### Showcase 与注册表（Phase 6/7）
- `.planning/phases/07-planthelixseek-showcase-notebooks/07-CONTEXT.md` — D-13 nightly-only 执行道、D-14 超时预算、D-16 fla 硬守卫等沿袭决策
- `example/notebooks/plant_helixseek_shared/data/selection.md` — 冻结选择契约（showcase 断言源）
- `models.lock` — 现有 lock 行格式（ms 前缀 + 注释）与 CI cache key 机制
- `.planning/phases/06-model-registry-showcase-data-curation/06-CONTEXT.md` — 注册表条目、fla 决策链

### CI 与契约
- `.github/workflows/ci.yml` — coverage-nightly 现状（03:00 UTC + dispatch-only、models.lock-keyed cache）、WR-08/09 修复后形态
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — typed-skip 白名单门
- `scripts/check_docs_sync.py` / `scripts/check_notebook_md_sync.py` / `scripts/validate_docs_snippets.py` — 镜像与 wrapper 同步契约
- `.planning/REQUIREMENTS.md` — EXEC-02..05、REPAIR-01/04、CI-04/05、MCP-01/02 需求原文

### 运维记忆
- `dnallm-nightly runner ops`（owner memory）：runner 重启需 sanitized env（env -i），否则 uv 装错 venv

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/examples/_execution.py` 全套 harness（tmp-sandbox、per-cell timeout、kernel-kill、tree-clean、NOTEBOOK_EXEC_SPECS 预算表、typed-skip 前缀）——example job 直接复用
- ACTIVE×13 / GATED×8 车道接线（05-05/06 + 261003-csd/sl7）已存在；marimo export-html flavor 已实证
- `dnallm/utils/transformers_compat.py` absence-gated shim 模式（05-04 D-07 的 NT shim 先例；D-17 若加深沿此模式）
- `models.lock` 两行先例 + coverage-nightly 的 lock-keyed cache 动作——example job 共享同一 key（D-09）

### Established Patterns
- 修复 = 原子提交（代码+回归测试同船；notebook 修复含镜像+wrapper，D-20）
- typed `environment-unavailable:`/`network-unavailable:` skip 带证据进消息，白名单门审计
- fail-soft 逐件 + 汇总非零（夜间 census 诚实语义，D-08）

### Integration Points
- 新 example-execution job 加入 ci.yml（schedule 错峰 + dispatch；分支保护不覆盖 nightly）
- ollama systemd 单元进仓 + runner enable（D-12）
- evo-1 巨仓路径机制接入模型下载路径（D-14）

</code_context>

<specifics>
## Specific Ideas

- 用户明确选了最宽的内容边界（允许结构重构）与最重的验收（逐项全量对账）——planner 应据此预算时长，家族划分宜小步多轮。
- "研究后定"仅一处（D-17 NT 结构家族）：researcher 需给出 shim 深度评估（vendor 面多大、离 HEAD 距离、小变体是否同坎）作为处置依据。

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 8-Full Execution Rollout & Repair Loop*
*Context gathered: 2026-10-04*
