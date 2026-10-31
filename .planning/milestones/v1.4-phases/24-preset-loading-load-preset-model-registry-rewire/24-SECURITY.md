---
phase: "24"
slug: "preset-loading-load-preset-model-registry-rewire"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-21"
---

# Phase 24 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Caller → `load_preset_model` | user-supplied `model_name` (arbitrary string) and a pattern-validated `TaskConfig` cross into the resolution chain | Model names, task/label config |
| Packaged data → runtime | frozen registry entries (packaged, Pydantic-validated at first parse) flow into TaskConfig fields and log lines | Frozen task metadata (labels, thresholds, base-model ids) |

No new input boundary is created: the function signature, its arguments, and the downstream
`load_model_and_tokenizer` call are unchanged; `model_name` remains an exact-match key against
frozen in-package data — never interpolated into paths or queries.

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-24-01 | Tampering | new D-04/D-08 `logger.info` lines embedding user-supplied model_name | low | mitigate | Hit-path logs embed only frozen entry values (`entry.name`, `entry.base_model`, `entry.task` fields), reachable solely after exact-match `try_resolve` (model.py:1669-1675); the miss path logs the raw name at debug over the same channel as the pre-rewire miss log; f-string rendering matches the 100+ existing call sites — no new sink or format channel | closed |
| T-24-02 | Tampering | registry entry values shaping the patched TaskConfig | low | accept | Registry is packaged data hard-validated into frozen Pydantic models at first parse (REG-01/REG-03); `_patched_task_config` injects exactly the three typed `_PATCH_FIELDS` via `model_copy(update=...)` (model.py:1619-1634) — no dynamic attributes, no revalidation bypass; registry module untouched this phase (phase diff: model.py + tests only; 41-test guard suite green incl. sole-parser + no-write-API) | closed |
| T-24-SC | Tampering | package installs | high | accept | Not applicable — the phase installed zero packages: no install tasks existed and the phase diff touches no dependency manifests (`git diff ff529ee..HEAD` — only `dnallm/models/model.py`, `tests/models/test_model.py`, planning artifacts). Any future install task in this phase must re-open this row | closed |

---

## ASVS Mapping (level 1)

| Category | Applies | Evidence |
|----------|---------|----------|
| V2 Authentication | no | no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | no privilege surface |
| V5 Input Validation | yes | `model_name` is an exact-match key against frozen in-package indexes (`try_resolve` name/repo, `MODEL_INFO` dict membership) — never interpolated into paths or queries; `TaskConfig` is pattern-validated Pydantic (configs.py:107-111) |
| V6 Cryptography | no | none |
