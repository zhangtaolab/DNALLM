# Phase 24: Preset Loading — `load_preset_model` Registry Rewire - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-21
**Phase:** 24-Preset Loading — load_preset_model Registry Rewire
**Areas discussed:** 解析优先级与名称碰撞, 补丁语义, base↔finetuned 运行时轴 (模型家谱), 回归锁定面

---

## 解析优先级与名称碰撞 (Resolution order)

### Q1: 查找顺序

| Option | Description | Selected |
|--------|-------------|----------|
| 条目名 → 家族名 → return 0 | registry entry name first-class; family keys unshadowed (lowercase vs Title Case); unknown keeps the int contract | ✓ |
| 家族名 → 条目名 → return 0 | absolutely conservative; weakens "entry name is the primary entry point" | |
| 条目名 → 家族名 → 直传通用加载 | loosest; breaks the return-0 contract | |

**User's choice:** 1
**Notes:** 未知名维持现 int 契约(return 0)不变。

### Q2: repo id 直接传入

| Option | Description | Selected |
|--------|-------------|----------|
| 注册表 repo-id 反查 | hit → patched entry load; miss → return 0; total over "every name the registry knows" | ✓ |
| repo id 不在契约内 | names are the only currency; repo ids belong to `load_model_and_tokenizer` | |

**User's choice:** 1
**Notes:** repo id 在注册表中跨节唯一,反查无歧义。

---

## 补丁语义 (Patch semantics)

### Q3: 注册表字段 vs 用户 TaskConfig 冲突

| Option | Description | Selected |
|--------|-------------|----------|
| 三字段注册表全覆盖 + 差异 info 日志 | checkpoint head shape is architectural fact; guardrail against wrong num_labels | ✓ |
| num_labels/label_names 覆盖, threshold 用户优先 | split architectural fact from decision policy | |
| 仅填空缺 | no guardrail against wrong user num_labels (wrong-shaped head) | |

**User's choice:** 1

### Q4: task_type 不一致时

| Option | Description | Selected |
|--------|-------------|----------|
| 只校验不补丁:不一致 → ValueError(大白话消息) | additive safety gate; stays within the three-field patch scope | ✓ |
| task_type 一并覆盖 | exceeds requirement scope; user intent swallowed | |
| 不管 | binary/multilabel mismatch silently picks the wrong loss | |

**User's choice:** 1
**Notes:** 本问期间 owner 下达语言指令(已存 agent 记忆):面向用户的文案必须让生物学家/生物信息学家看得懂——该 ValueError 的消息文本也须遵守。

### Q5: 底座(pretrained)条目传入

| Option | Description | Selected |
|--------|-------------|----------|
| 照常加载,不补丁 | three fields are None for bases; uniform rule for all registry names | ✓ |
| 拒绝并提示选微调条目 | guards novices, limits flexibility | |

**User's choice:** 1

---

## base↔finetuned 运行时轴 (模型家谱)

### Q6: "axis live at runtime" 的具体含义

| Option | Description | Selected |
|--------|-------------|----------|
| 加载时 info 日志报家谱 + 现有查询 API | user-visible, tiny change, no interface changes | ✓ |
| 加载返回值携带谱系 | big surface change affecting all callers | |
| 什么都不加 | zero runtime visibility | |

**User's choice:** 1

---

## 回归锁定面 (Regression scope)

### Q7: 家族路径回归的严格程度

| Option | Description | Selected |
|--------|-------------|----------|
| 全量 35 家族逐个测(mock) | literal "every family unchanged" | |
| 挑代表测 | fastest, promise diluted | |
| 存档对照文件 | strongest evidence, ongoing maintenance burden | |

**User's choice:** 只测 zhangtaolab 的模型(free-text)
**Notes:** Owner narrowed the regression suite: parametrize ONLY over families whose default repo is `zhangtaolab/*`. The phase's own additions (entry-name resolution, repo-id reverse lookup, pretrained no-patch, task-type ValueError, unknown→0) are still fully tested — they are deliverables, not regressions.

---

## Claude's Discretion

- D-03: exact case-sensitive matching for both lookups; collision resolved by D-01 order (no current collision).
- D-07: the dead `preset`-list branch is replaced (deleted), not kept alongside; registry-data filename never named literally under `dnallm/`.
- Patched TaskConfig built as a copy, never mutating the caller's object.

## Deferred Ideas

None — discussion stayed within phase scope.
