---
phase: "23"
slug: "pre-flight-core-pure-validation-suitability-config-api"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-21"
---

# Phase 23 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| User data → pure core | `validate_data()` reads user-supplied files (7 formats) read-only; no mutation, no row drop | Local file contents (may contain sensitive genomic data) |
| Core → model hubs | `probe_availability()` queries ModelScope/HF metadata endpoints; no weight downloads | Repo ids, sizes, revisions (public metadata) |
| Config round-trip | `generate_config`/`edit_config` read/write user-named YAML paths only when `output_path` given | Validated config sections (user-authored) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-23-01 | Tampering / Elevation of Privilege | validation.py reader (pkl path) | high | mitigate | `.pkl` inputs return `UNSUPPORTED_FORMAT_PKL` finding; `grep -rn "import pickle" dnallm/preflight/` = 0 hits; never unpickled (report.py:101, validation.py:22-93) | closed |
| T-23-02 | Information Disclosure | validation.py logging | low | mitigate | `_log_split_summary` logs counts/reason codes only; 0 bare `print(` under dnallm/preflight/ (ASVS V8 data minimization) | closed |
| T-23-03 | Denial of Service | validation.py reader (oversized file) | low | accept | Local tool reading the user's own local file; same memory profile as `DNADataset` today; no remote trigger surface | closed |
| T-23-04 | Tampering (verdict integrity) | compatibility.py budget/charset tables | medium | mitigate | k-mer trap neutralized by name-regex classifier; live-registry pins in tests (6-mer → 517 = 512+k−1, BPE ×3 = 1536, char = 512; tests/preflight/test_compatibility.py:12-71) | closed |
| T-23-05 | Tampering (SSRF/endpoint manipulation) | availability.py adapters | medium | mitigate | Endpoints from pinned library constants only (`huggingface_hub.constants.ENDPOINT`, modelscope default; availability.py:33-163); zero user-controlled URL construction | closed |
| T-23-06 | Spoofing (cache-poisoned "cached") | local cache resolvers | low | accept | Presence-based detection; size/revision surfaced with the verdict so shells can flag anomalies; no code executes from cache paths | closed |
| T-23-07 | Tampering (verdict integrity via misread failures) | failure taxonomy | medium | mitigate | RATE_LIMITED / OFFLINE / PROBE_TIMEOUT / NOT_FOUND_ON_EITHER_HUB / PROBE_ERROR all map to unknown+reason with pinned tests — 429 can never masquerade as "unavailable" (tests/preflight/test_availability.py:439-469) | closed |
| T-23-08 | Tampering / Elevation of Privilege (YAML deserialization) | config_gen.py round-trip load | high | mitigate | Safe loaders only (`CSafeLoader`→`SafeLoader` fallback, config_gen.py:296-311); python/object-tag YAML refused by pinned test (ConstructorError) | closed |
| T-23-09 | Tampering (surprise writes / path traversal) | _dump_yaml | medium | mitigate | Exactly one write site (config_gen.py:196) to the explicitly-passed `output_path`; read site (config_gen.py:310) opens the user-named existing config; nothing written when `output_path` omitted (pinned test) | closed |
| T-23-10 | Tampering (coverage-gate bypass) | phase closeout | medium | mitigate | Closeout asserted per-module coverage ≥96% with zero pyproject omit entries (no `preflight` hits in pyproject.toml); omit-guard test in tests/preflight/test_structure.py | closed |
| T-23-SC | Tampering | package installs | high | mitigate | `git diff 70544b7..HEAD -- pyproject.toml` empty — zero dependency changes across the phase; nothing to gate | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-23-01 | T-23-03 | Oversized-file memory use mirrors existing `DNADataset` behavior on a local tool with the user's own file; no remote trigger | Phase 23 plan (D-discussion), owner-approved scope | 2026-10-21 |
| AR-23-02 | T-23-06 | Cache presence ≠ integrity; size/revision are surfaced alongside the state so any anomaly is visible; no code execution from cache paths | Phase 23 plan (D-discussion), owner-approved scope | 2026-10-21 |

---

## Security Audit

| Metric | Value |
|--------|-------|
| Threats found | 11 |
| Closed | 11 |
| Open | 0 |
| Method | L1 grep-depth verification (ASVS L1 short-circuit: plan-time register, threats_open: 0) |
| Register origin | PLAN.md `<threat_model>` blocks (authored at plan time) |

---
