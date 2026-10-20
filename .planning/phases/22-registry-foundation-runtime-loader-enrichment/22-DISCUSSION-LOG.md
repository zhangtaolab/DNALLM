# Phase 22: Registry Foundation — Runtime Loader & Enrichment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-20
**Phase:** 22-Registry Foundation — Runtime Loader & Enrichment
**Areas discussed:** enrichment value source, max_length coverage, third-party enrichment scope, loader validation strength

> Interaction note: AskUserQuestion popups rendered garbled in the owner's terminal (recurring harness issue this session); the discussion switched to plain-text numbered options after the first attempt was rejected with that feedback. All decisions below were captured via plain text.

---

## Enrichment value source (max_length provenance)

| Option | Description | Selected |
|--------|-------------|----------|
| a. Name derivation only | Derive tokenizer_type from name suffixes; no max_length in registry | |
| b. Probe-then-commit | Batch-pull each model's config.json via HF/ModelScope metadata APIs (no weights), commit measured values into the registry | ✓ |
| c. Hybrid + runtime fallback | Derive what's derivable, probe the rest lazily at runtime with cache | |

**User's choice:** b（探测固化）— "全部按推荐"
**Notes:** Recommended for SSH/offline usability: one-time probe cost buys permanent offline availability.

---

## max_length coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Pretrained bases only + inheritance | All 49 bases get max_length; finetuned entries inherit via base_model unless explicitly overridden | ✓ |
| All 236 entries | Also probe and commit per-finetuned-entry values (heads may change context) | |

**User's choice:** 基座+继承 — "全部按推荐"
**Notes:** Inheritance resolved by the loader at query time; explicit override possible in YAML.

---

## Third-party enrichment scope

| Option | Description | Selected |
|--------|-------------|----------|
| Lab-first, best-effort third-party | 174 zhangtaolab entries mandatory; 62 third-party probed best-effort, failures recorded as missing values, never blocking | ✓ |
| Lab only | Skip third-party entirely | |
| All mandatory | Block phase until all 236 enriched | |

**User's choice:** 推荐（lab 优先 + 第三方尽力而为）— "全部按推荐"
**Notes:** Aligns with owner's lab-models-first directive; third-party API instability was flagged MEDIUM by research.

---

## Loader validation strength

| Option | Description | Selected |
|--------|-------------|----------|
| Pydantic hard validation | Malformed entries fail loudly at first registry access (early exposure) | ✓ |
| Lightweight types | Frozen dataclasses; bad entries surface only at lookup | |

**User's choice:** Pydantic 硬校验 — "全部按推荐"
**Notes:** "The registry is the truth source" — fail early. Optional fields are X | None; core fields required.

---

## Claude's Discretion

- Pydantic model field granularity (stay close to existing YAML schema)
- reload() / cache-invalidation mechanics
- is_lab_model as function vs entry property
- Probe script location (.scratch/ vs scripts/) — committed output is what matters

## Deferred Ideas

None — discussion stayed within phase scope.
