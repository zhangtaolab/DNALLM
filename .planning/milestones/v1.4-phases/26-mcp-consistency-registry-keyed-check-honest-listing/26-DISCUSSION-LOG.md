# Phase 26: MCP Consistency — Registry-Keyed Check & Honest Listing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-22
**Phase:** 26-MCP Consistency — Registry-Keyed Check & Honest Listing
**Areas discussed:** 判定语义 (verdict semantics), 报告形态 (finding shape), MCP-02 列表覆盖层 (listing overlay shape), CI 落点 (CI placement), 文本载荷与残余细节 (text payload & residuals)

---

## 报告形态 (finding shape)

| Option | Description | Selected |
|--------|-------------|----------|
| 复用 preflight Finding | Reuse `dnallm/preflight/report.py`'s `Finding` (severity + reason code + plain message). Phase 23 D-05 designated the finding conventions a cross-shell contract; the Phase 25 TUI already consumes it. | ✓ |
| MCP 自建报告类型 | New MCP-local dataclass. Decouples from preflight, but creates a third warning vocabulary against D-05's contract. | |
| 纯 dict，不建类型 | Return plain dicts, no type. Fastest, but no stable field contract for CI or future MCP tools. | |

**User's choice:** 复用 preflight Finding
**Notes:** — (选择即理由)

---

## MCP-02 列表覆盖层 (listing overlay shape)

| Option | Description | Selected |
|--------|-------------|----------|
| 注册表全量 + 状态标记 | All 236 registry entries as the body, each carrying configured / enabled / loaded status fields. Closes #64's name/reality mismatch. | ✓ |
| 注册表全量 + 保留配置键名 | Registry body keyed by registry name, but configured rows also expose the server-config key. Two namespaces in one payload. | |
| 现有配置列表 + 注册表摘要 | Keep the configured list as the body, add a registry_summary. Minimal change, but the client menu still cannot see 236 models. | |

**User's choice:** 注册表全量 + 状态标记
**Notes:** Follow-up confirmed keys = registry entry names, with the server-config key retained as a field on configured rows (CONTEXT.md D-14).

---

## 判定语义 (verdict semantics)

| Option | Description | Selected |
|--------|-------------|----------|
| source 由 lab 身份推导 | Derive the expected `source` via `is_lab_model()` — lab ⇒ modelscope, third-party ⇒ undetermined. Gives the source check a real basis without false positives. | ✓ |
| source 不做硬比对，仅提示 | Exclude `source` from the comparison; match only the four core fields. Most conservative, but MCP-01 names `source` explicitly. | ✓ |
| 额外比对 max_length | Beyond MCP-01's five fields, also compare `effective_max_length` vs `inference.max_length`. Catches real misconfiguration. | ✓ |
| description 只提示不失败 | Compare `task.description` vs registry `task.describe` as informational only. Hand-written descriptions legitimately differ in wording. | ✓ |

**User's choice:** All four selected.
**Notes:** Two `source` options were selected together. Interpreted as: DERIVE the expectation from lab identity (so the check has a real basis), but report a mismatch as WARN only and never let it participate in the `strict_registry` failure. Recorded as CONTEXT.md D-05. `max_length` took the same WARN-only tier, grounded in already-locked decisions (Phase 23 D-06 truncation never FAILs; Phase 25 D-12 `max_length` is an editable user policy). `description` is informational, never ERROR and never PART of strict.

---

## CI 落点 (CI placement)

| Option | Description | Selected |
|--------|-------------|----------|
| pytest 测试即 CI 闸 | A test under `tests/mcp/` calling the checker with `strict_registry=True` over the config dir. Follows the repo's tests-mirror-package + pytest-only-CI convention; zero new CI steps. | ✓ |
| ci.yml 独立步骤 | An explicit step in `ci.yml`. Clearer logs, but a second failure channel for one constraint. | |
| pytest 测试 + CI 独立步骤 | Both. Double coverage, two sources of truth for one constraint. | |

**User's choice:** pytest 测试即 CI 闸
**Notes:** Follow-up: `mcp_server_config_2.yaml` is included in the checked set (CONTEXT.md D-03), since it references the same per-model files.

---

## 文本载荷与残余细节 (text payload & residuals)

| Option | Description | Selected |
|--------|-------------|----------|
| 紧凑行式摘要 | One line per model in `content[0].text` (name \| repo id \| task_type \| status); full structured rows in the `models` field. | ✓ |
| 保留 str(dict) 全量 | Keep `str(dict)`. Minimal change, but a 236-entry dict stringified is very long for an LLM client. | |
| 只给统计摘要 | Totals and task_type distribution only; clients drill in via `list_models_by_task_type`. Cheapest, but no visible menu. | |

**User's choice:** 紧凑行式摘要
**Notes:** —

---

## Claude's Discretion

- Checker module location (`dnallm/mcp/consistency.py` vs a section inside `config_validators.py`); the public symbol name `check_config_registry_consistency` is fixed by MCP-01.
- Exact function signature, parameter names, and the shape of the machine-readable pass/fail signal.
- How configs are parsed for comparison (direct YAML vs `MCPConfigManager`/Pydantic) and what the checker does when a YAML fails Pydantic validation before it can be compared.
- The description-similarity measure and threshold.
- Compact-summary line format details, and whether the text block truncates for very large listings.
- Whether the registry-full listing gains optional `task_type` / `intent` filters.
- Test file organization under `tests/mcp/` and fixture style.

## Deferred Ideas

- Root `configs/*.yaml` consistency checking — excluded: those templates carry no `model.path`, so there is nothing to key on.
- Renaming `get_all_available_models` to `get_configured_models` — rejected because the listing now honestly is all available models.
- MCP tools exposing the pre-flight core (`dnallm_preflight_check` class) — REQUIREMENTS.md Out of Scope D-6.
- Defect halves of issues #62/#63/#64 — filed and fixed outside v1.4 by owner disposition 2026-10-20.
