# Phase 07: docs-examples-sync - Discussion Log

**Date:** 2026-05-15
**Participants:** User, Claude
**Phase:** 07-docs-examples-sync

---

## Summary

4 gray areas discussed. All decisions locked. No scope creep. No deferred ideas.

---

## Area A — Validation Strategy

**Question:** How should we validate that docs/examples fixes are correct?

**Options presented:**
1. Static check only (grep + ast.parse) — fast, CI-friendly
2. Static + import execution — medium effort, catches constructor errors
3. Full execution — run all notebook cells (strictest)

**User selected:** Full execution for local, CI downgrade

**Follow-up question:** How to handle CI environment constraints?

**Options presented:**
1. CI降级：本地完整执行，CI只做import+初始化
2. CI也完整执行
3. 分两层CI

**User selected:** CI降级 — local full execution, CI import+initialization only (non-blocking)

**Decision locked (D-01):**
- Local: run core notebook cells
- CI: import + initialization validation only (informational, not blocking)
- External-data notebooks: import + API signature only

---

## Area B — Batch Modification Strategy

**Question:** How to execute the ~100+ pattern-based fixes across 150+ files?

**Options presented:**
1. Layered: script for low-risk, per-file for high-risk
2. All script: generate diffs, review, batch apply
3. All per-file: open each file, review context, modify

**User selected:** All per-file review

**Decision locked (D-02):**
- No script-based bulk replacements
- Workflow: grep → open file → confirm context → modify → commit

---

## Area C — TaskType Enum / TaskConfig Regex

**Question:** TaskType enum values don't match TaskConfig regex. Fix enum or regex?

**Options presented:**
1. Fix enum values (short names)
2. Fix regex (accept full names)
3. Both

**User selected:** Fix regex — expand to accept both short and full names

**Scout finding:** No code uses TaskType enum directly (outside task.py). Safe to modify either.

**Decision locked (D-03):**
- Expand TaskConfig regex to accept both forms
- Update model_post_init condition checks
- Keep TaskType enum values unchanged

---

## Area D — Execution Order

**Question:** How to sequence 35 tasks across 13 batches?

**Options presented:**
1. Pattern-based traversal (recommended)
2. Directory-based traversal
3. Strict batch order

**User selected:** Pattern-based traversal

**Decision locked (D-04):**
- Order: source code → YAML → imports → config keys → API signatures → CLI → task types → MCP → notebooks/marimo → API docs → tutorials → CI

---

## Key Scout Findings That Changed Requirements

| Finding | Impact |
|---------|--------|
| `dnallm/__init__.py` exports `load_model_and_tokenizer` and `DNADataset` | Requirement 7 (fix import paths) may be no-op for some files |
| `pyproject.toml` registers `dnallm-train`, `dnallm-inference`, `dnallm-mcp-server` | Requirement 10 (fix CLI commands) needs re-evaluation |
| MCP server exposes tool names WITHOUT underscore prefix | Requirement 11 (MCP tool names) may be no-op |
| No code uses `TaskType` enum directly | Safe to modify either enum or regex |

---

## Deferred Ideas

None

---

*Phase: 07-docs-examples-sync*
*Discussion completed: 2026-05-15*
