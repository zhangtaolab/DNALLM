---
phase: "06"
slug: "model-registry-showcase-data-curation"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-10-03"
---

# Phase 06 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| HF/ModelScope hub → dev box | checkpoint remote code execution during smoke/verification loads (trust_remote_code) | model weights + code, owner-org repos only |
| plantdhs.org → scratch | the one network fetch: external zip becoming committed truth data | GFF text (parsed as data, never executed) |
| owner-local TAIR10/PlantDHS files → selection math | truth files parsed and sliced into committed artifacts | GFF3/FASTA text |
| committed data → Phase 7/8 | downstream notebooks and nightly tests treat these files as ground truth | FASTA/GFF artifacts ≤200kb/set |
| scratch tooling → repo | intentionally-uncommitted one-shot code; .gitignore boundary | none (must never cross) |
| PyPI → .venv (post-hoc) | flash-linear-attention 0.5.2 install (owner-approved at blocking-human gate) | Triton kernel package |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-06-01 | E (Elevation of Privilege) | smoke loads, trust_remote_code | high | mitigate | Only owner-org zhangtaolab repos loaded via registry; checkpoint shas recorded in committed yaml provenance (CRE 7093de3b..., Anno 6d39386a...); HF fallback stays in same org/repo ids; verified in tests/models/test_plant_helixseek_smoke.py (loads only the two frozen ids) and model_info.yaml provenance comments | closed |
| T-06-02 | T (Tampering) | checkpoint-metadata consumption + registry append | medium | mitigate | Hub metadata asserted, never transcribed: placeholder-pattern + head-shape assertions fail closed; append-only byte-identical-prefix gate; yaml.safe_load re-parse; committed fast-leg tests carry duplicate-append + LABEL_-placeholder guards (06-REVIEW verified all) | closed |
| T-06-03 | T (Tampering) | parse_gff_attributes / normalize_chrom on external GFF3 text | low | mitigate | Column 9 opaque data — split/strip only, no eval/format-string/shell; unknown forms raise ValueError; verified in code + 20 unit tests (06-02 SUMMARY threat section) | closed |
| T-06-04 | T (Tampering) / V5 | coordinate + slice guards | low | mitigate | Every conversion validates ints/ranges; require_nonempty + empty-fetch guard raise ValueError (silent-empty structurally impossible); exercised by the test suite | closed |
| T-06-05 | T (Tampering) | plantdhs.org zip fetch | medium | mitigate | Zip magic-byte check, inner GFF structural check, 50 MB decompression bound, row-count band [30k,50k] before parsing; evidence recorded in selection.md provenance (493,051 B zip, 39,523 rows validated); data parsed, never executed; committed artifacts human-reviewed (UAT 1 pass) | closed |
| T-06-06 | T (Tampering) / V12 | scratch tool write surface | low | mitigate | Writes confined to gitignored .scratch/ + three designated data/ dirs; atomic temp-then-replace; scoped tree-clean assertion post-commit (verified green in Task 3 + orchestrator spot-check + code review); no executable content in committed data; git ls-files emptiness under .scratch/ holds | closed |
| T-06-07 | E (Elevation of Privilege) | verification-stage model loads | high | mitigate | Same class as T-06-01: only registry-frozen owner-org repo ids, shas in selection.md provenance; no user-controllable id reaches a load call (select_loci reads registry only) | closed |
| T-06-SC (06-01) | T (supply chain) | package installs | high | accept | Plan-time basis "no installs" held for the plan itself; see Accepted Risks Log for the post-hoc owner-directed install | closed (accepted) |
| T-06-SC (06-02) | T (supply chain) | package installs | high | accept | No installs in plan; pyfastx pre-existing dev extra, function-local import | closed (accepted) |
| T-06-SC (06-03) | T (supply chain) | package installs / subprocess | high | accept | No installs in plan; bedtools pre-existing system binary (2.31.1) invoked with fixed arguments | closed (accepted) |

*Severity: only open threats at or above workflow.security_block_on (high) count toward threats_open — all rows are closed or accepted; none open.*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-06-01 | T-06-SC ×3 | No package installs during plan execution; all tooling pre-installed and pinned | plan-time (owner-reviewed plans) | 2026-10-02 |
| AR-06-02 | T-06-SC (post-hoc) | ONE package installed post-plan by owner decision B+ (2026-10-03 16:40 CST, blocking-human package gate): flash-linear-attention==0.5.2, installed bare into .venv. Provenance verified against the official fla-org GitHub org and PyPI at approval time; upstream PlantHelixSeek pins 0.4.1 (chosen 0.5.2 for torch-2.11 compat after a probe table through the unmodified dnllm route); pyproject declares fla extra >=0.5.2,<0.6 wired into all; guard tests committed (import path + declaration); docs carry the bare-install/downgrade-hazard warning. Residual risk: chunk_kda semantics drift within 0.5.x minors — bounded range + import guard test surface it loudly. | owner (checkpoint), assistant verified provenance | 2026-10-03 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-03 | 10 | 10 | 0 | orchestrator (L1 grep-depth short-circuit: register_authored_at_plan_time, asvs_level 1, threats_open 0) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
