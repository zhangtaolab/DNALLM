# Phase 27: Transformers 4/5 Model Compatibility Census - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-27
**Phase:** 27-Transformers 4/5 Model Compatibility Census
**Areas discussed:** 证据深度 (evidence depth), 版本锚点 (version anchors), 修复策略 (fix strategy), 矩阵落点 (matrix home), 复跑策略 (recurrence)

---

## Pre-discussion clarification (scope)

**User's choice:** 只检查预训练模型框架 — census covers only the `pretrained:` section (49 entries); the 187 `finetuned:` entries are excluded and inherit base-model compatibility.
**Notes:** User also flagged garbled characters in the first AskUserQuestion rendering; subsequent questions were re-sent clean.

## 证据深度 (evidence depth)

### Q1: 49 个预训练模型的普查证据做到什么深度?

| Option | Description | Selected |
|--------|-------------|----------|
| 分层:全量 import + 抽样加载 (推荐) | All 49 get import-level evidence; real loads only for import-broken models (shim verification) + one representative per model family | ✓ |
| 全量加载级 | All 49 really loaded under both versions; strongest evidence, largest download surface | |
| 仅 import 级 | Remote-code file import resolution only; cheapest, misses runtime-only failures | |

**User's choice:** 分层:全量 import + 抽样加载
**Notes:** —

### Q2: 加载级证据验证到哪一步算"过"?

| Option | Description | Selected |
|--------|-------------|----------|
| 加载+短序列 forward (推荐) | Load succeeds AND one short-sequence real forward runs; catches forward-layer breakage seen historically | ✓ |
| 仅加载成功 | Construction + weights only; faster, misses forward-layer failures | |
| 加载+forward+输出断言 | Additionally assert output shape/type; strongest, uniform assertions over 49 heterogeneous models cost work | |

**User's choice:** 加载+短序列 forward
**Notes:** —

### Q3: giants 级模型(evo-1、evo-2)在加载抽样里怎么处理?

| Option | Description | Selected |
|--------|-------------|----------|
| 纳入,限有缓存处 (推荐) | Giants load only where cached (runner), typed skip + honest label elsewhere | |
| 排除加载,仅 import | Giants get import-level evidence only; matrix labels import-verified only | ✓ |
| 全部同等对待 | Same standard as everything else, re-download as needed | |

**User's choice:** 排除加载,仅 import
**Notes:** evo loads through `special/evo.py` (stripedhyena), not the trust_remote_code surface.

## 版本锚点 (version anchors)

### Q1: 矩阵里的 4 和 5 分别锚定到哪个具体版本?

| Option | Description | Selected |
|--------|-------------|----------|
| 4.49 + 5.17 (推荐) | Floor + installed; matches vendored-rung reference and verified shims | |
| 最新 4.x + 最新 5.x | Freshest for current users; all shims need re-verification | |
| 双侧双点 (4 个环境) | Most complete evidence; double the environments/cost | |

**User's choice:** (free text) 其实就用 5.17 检测即可,如果不兼容 4, 自然会报错
**Notes:** Simplified beyond all offered options: single-environment census under installed 5.17.0, no separate 4.x environment. Reasoning: all 49 checkpoints are 4.x-era; breakage only runs 5-removes-4-API; pass-under-5.17 implies 4-side OK. Confirmed via follow-up: matrix labels 5 side "tested on 5.17.0", 4 side "inferred (4-era code)".

## 修复策略 (fix strategy) — user-added area

### Q1: 遇到不兼容时的修复手法优先级怎么定?

| Option | Description | Selected |
|--------|-------------|----------|
| 先 5.x 别名,后 vendor (推荐) | Prefer aliasing 5.x equivalents back onto old names; vendor 4.49.0 implementations only when 5.x has no equivalent | |
| 一律 vendor 4.49 | Always vendor the reference implementation; most controlled, shim grows heavy | |
| 逐案决定 | No preset priority; decide per failure after census | |

**User's choice:** (free text) 我记得就是类/函数 的名字改了,修起来不是很麻烦的,先查吧,查出来问题再看解决方案。
**Notes:** Census-first, per-failure solutions on evidence. Owner expects renames (PreTrained*/Pretrained* casing class) to dominate and be cheap. Fix-over-skip per 2026-10-02 precedent. Alias mapping acknowledged as an expected fix shape (from the user's original area description: use the transformers 5 package to substitute removed 4.x names via if/else or try/except).

## 矩阵落点 (matrix home)

### Q1: 普查结果的数据落点和 docs 表格怎么做?

| Option | Description | Selected |
|--------|-------------|----------|
| 字段+生成表格+gate (推荐) | New fields in model_info.yaml pretrained section (status+evidence+version stamp, REG-02 pattern); docs table generated from registry; CI consistency gate (Phase 26 precedent) | ✓ |
| 独立产物+生成表格 | Standalone census artifact file; docs generated from it; registry not extended | |
| 仅手写 docs 表格 | Hand-written Markdown table; cheapest, no machine readability or drift protection | |

**User's choice:** 字段+生成表格+gate
**Notes:** Field naming/schema left to planner.

## 复跑策略 (recurrence)

### Q1: 普查工具的复跑形态选哪档?

| Option | Description | Selected |
|--------|-------------|----------|
| dispatch lane+漂移即红 (推荐) | Committed census tool + dispatch-only lane on self-hosted runner (feasibility.yml precedent); re-run on transformers bumps; committed-field vs re-run mismatch turns red (fail-closed) | ✓ |
| 一次性,仅本地 | Run once locally, commit evidence, tool in repo but no CI lane | |
| nightly 常驻重验 | Nightly re-verification with version-stamp mismatch failing; most timely, nightly cost | |

**User's choice:** dispatch lane+漂移即红
**Notes:** Operationalizes the dependabot "bump must re-verify" requirement.

---

## Claude's Discretion

- Census tool implementation shape (script vs pytest lane, evidence artifact format)
- Registry field names/schema and version-stamp representation
- Docs page placement and table layout
- Family-representative selection for load sampling
- Alias-vs-vendor choice per individual failure (decided on census evidence)
- ROADMAP "238" count correction path (actual: 49 pretrained + 187 finetuned = 236)

## Deferred Ideas

- Load-time compat notice: `load_model_and_tokenizer` consulting the new registry fields to warn/raise with the exact failing import when a 4-only model is requested under transformers 5 (noted by Claude, not owner-raised — future phase).
