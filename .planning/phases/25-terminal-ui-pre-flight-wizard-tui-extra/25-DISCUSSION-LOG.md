# Phase 25: Terminal UI — Pre-flight Wizard & `tui` Extra - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-21
**Phase:** 25-Terminal UI — Pre-flight Wizard & tui Extra
**Areas discussed:** 启动速度与环境面板 (Startup & env panel), 可用性探测时机 (Availability probing UX), 流程形态 (Wizard flow shape), 配置面板字段面 (Config panel surface)
**Format note:** AskUserQuestion unavailable in this harness — discussion ran in the Phase-24 numbered-options-table style; the user answered by option number.

---

## 启动速度与环境面板 (Startup & env panel) — inherits CR-01

Measured evidence presented first: `import dnallm.preflight` = 3.7 s warm on the dev box, pulling torch+transformers via the package-init chain; root init contains the load-bearing utils→models ordering constraint.

### Q1: 向导启动策略 (CR-01 runtime purity — Phase 25 owns the decision)

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 接受现状 — 等待启动 | torch/transformers load at launch (~3.7 s warm, 10–30 s cold); wizard appears after; zero refactor risk | |
| 2. 根门面懒加载 — UI 先行 | PEP 562 `__getattr__` lazy root facade; wizard starts instantly, version rows via `importlib.metadata`, device/VRAM from a background worker; two research traps flagged (ordering constraint moves into models' own init; utils shim chain may need lazy too) | ✓ |
| 3. 先研究再定 | lock the goal only; researcher picks the mechanism | |

**User's choice:** 2
**Notes:** Claude recommended 2 (muggle minute-zero UX; small code, strong test net; permanently upgrades one-core-many-shells), with 1 as the scope-protecting fallback. Owner accepted the recommendation after seeing the trade-off framed.

### Q2: 环境信息收集代码的位置 (TUI-03 "from the core")

| Option | Description | Selected |
|--------|-------------|----------|
| 1. `dnallm/utils/env_info.py` 新模块 | core-owned collector, function-local torch import, worker-invoked; screens render only | ✓ |
| 2. `dnallm/preflight/env.py` + 修订 Phase-23 AST 规则 | keeps domain logic in one package but weakens the zero-torch structure invariant | |
| 3. Planner 定 | lock only "screens render, never collect" | |

**User's choice:** 1

### Q3: 环境面板的行集合

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 精简固定行 | dnallm / Python / torch (+CUDA、设备名、显存) / transformers / numpy / datasets / 缓存目录磁盘余量 — exactly TUI-03's enumeration | ✓ |
| 2. 体检式全探测 | + GPU 驱动/CUDA 版本、平台、CPU/内存 (dnallmmark env_check style) | |
| 3. Claude 定 | | |

**User's choice:** 1

---

## 可用性探测时机 (Availability probing UX) — inherits WR-01 + D-10

Context noted: cached-state detection is purely local (no network), so 已缓存 can render for all rows for free; only remote confirmation costs network.

### Q4: 远程探测的触发时机

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 短名单自动 + 其余按需 | local cached states for all visible rows (free, offline) + auto remote probe of top N≈20 (lab-first); rest UNKNOWN with per-row on-demand probe | ✓ |
| 2. 全按需 | zero network by default; remote probing strictly on keypress | |
| 3. Claude 定 | lock only "never probe all 236" | |

**User's choice:** 1

### Q5: WR-01 未证实托管的第三方名称如何展示

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 如实标注 | listed normally (lab-first puts them lower); availability column shows UNKNOWN/"hosting not verified"; probe on demand reveals truth | ✓ |
| 2. 默认隐藏 | screen-layer filter excludes third-party until toggled | |
| 3. 推迟 Phase 26 | v1.4 suitability panel doesn't expose them | |

**User's choice:** 1

### Q6: UNKNOWN 模型能否直接下载

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 始终可下载 | UNKNOWN ≠ unavailable (PFC-06); download has its own retry; never dead-ended | ✓ |
| 2. 仅 cached/remote-confirmed 可下载 | UNKNOWN rows' download key greyed until probed | |

**User's choice:** 1

---

## 流程形态 (Wizard flow shape)

Signature verified before asking: `check_compatibility(data=None, ...)` already supports the no-data mode (length/charset degrade to NO_DATA; task dimension computes).

### Q7: 入口模式 (新建 vs 编辑现有配置)

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 首屏选择 | first interaction: 从数据开始(新建) or 编辑现有配置 | |
| 2. CLI 参数直达 | `--config` jumps to config panel; default is full flow | |
| 3. 两者都要 | first-screen choice + `--config` shortcut | ✓ |

**User's choice:** 3

### Q8: 数据面板是否必需

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 数据可选 | suitability usable with no data (task filter + registry facts); data-driven dims show 未提供数据 | ✓ |
| 2. 数据必需 | suitability gated until data loaded | |

**User's choice:** 1

### Q9: 下载的位置

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 配置完成后的可选终步 | final screen after validated config emission: progress + cancel + summary; quit anytime | ✓ |
| 2. 适配面板内即可下载 | download as soon as model selected | |
| 3. 两处都提供 | download during config editing saves wall-clock on 12 GB models | |

**User's choice:** 1

### Q10: 向导文案语言

| Option | Description | Selected |
|--------|-------------|----------|
| 1. English（大白话） | consistent with repo/docs/PyPI; obeys plain-language directive | ✓ |
| 2. 中英双语 | domestic-user friendly; screen space + maintenance ×2 | |
| 3. 中文 | lab/domestic users first | |

**User's choice:** 1

---

## 配置面板字段面 (Config panel surface — absorbs survey R4)

### Q11: 结构化编辑的 section 范围

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 核心五节编辑，其余只读透传 | task/model/finetune/inference/lora form fields; ia3/vep/sweep preserved verbatim, shown read-only | ✓ |
| 2. 全部 section 都可编辑 | seven-section full editor | |
| 3. 生成窄、编辑宽 | generate mode five sections; edit mode all-present sections | |

**User's choice:** 1

### Q12: 注册表自动填充字段的编辑性

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 架构事实只读，策略字段可编辑 | num_labels/label_names/threshold read-only + note (D-04 registry wins at load); max_length prefilled but editable (D-06 truncation is policy); hyperparams fully editable | ✓ |
| 2. 全部可编辑 | registry values as mere defaults; edits silently overridden at load | |
| 3. Claude 定 | | |

**User's choice:** 1

### Q13: 实时验证的呈现与导出门控

| Option | Description | Selected |
|--------|-------------|----------|
| 1. 常驻错误面板 + 验证通过才可导出 | split pane form/live error list (`section.field → 大白话`) + read-only YAML preview; export disabled until clean | ✓ |
| 2. 保存时一次性校验 | modal on save attempt only | |
| 3. Claude 定 | | |

**User's choice:** 1

---

## Claude's Discretion

Shortlist size N, probe concurrency, placeholder wording; screen/widget/CSS structure; back-navigation mechanics (default: always allowed, state kept); worker-seam mechanism + cancel semantics (roadmap research flags); `tui`-extra vs alias-removal ordering; Pilot batch structure/wall-time budget; env-row ordering; YAML-preview placement; default export filename; summary-screen contents; lazy-facade mechanism details beyond D-01's contract.

## Deferred Ideas

None — discussion stayed within phase scope.
