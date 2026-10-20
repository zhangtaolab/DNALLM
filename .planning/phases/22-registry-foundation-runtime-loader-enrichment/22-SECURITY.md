---
phase: "22"
slug: "22-registry-foundation-runtime-loader-enrichment"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-20"
---

# Phase 22 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| packaged data -> runtime | model_info.yaml ships inside the wheel; a tampered or corrupted artifact crosses into process memory at first registry access | YAML config data (model registry) |
| dnallm/ source tree -> registry file | any future module could become a second parser or a writer; this phase's guards police that boundary | source-code references to packaged YAML |
| remote hub metadata -> committed registry | config.json / tokenizer_config.json fetched from ModelScope/HF are untrusted remote data feeding a committed artifact | public model metadata (read-only, never executed) |
| one-off scripts -> tracked source | .scratch/ tooling (untrusted execution surface) writes into git-tracked dnallm/models/model_info.yaml | generated YAML field insertions |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-22-01 | Tampering | dnallm/models/registry.py + model_info.yaml | high | mitigate | Loader born with zero write/serialize API (function surface verified read-only); entries frozen Pydantic models (`ConfigDict(frozen=True, extra="forbid")`); no-write structure tests make adding a mutation path CI-red (`tests/models/test_registry.py:560` verb + emitter-token scans) | closed |
| T-22-02 | Denial of Service | registry parse path (supply-chain-tampered package data) | medium | mitigate | Pydantic hard validation at first access: malformed entries, duplicates, missing sections fail loudly as descriptive ValueError naming the offender (`dnallm/models/registry.py:144-220`); broken-YAML and missing-module failure modes under test (`tests/models/test_registry.py:257,269`) | closed |
| T-22-03 | Tampering | dnallm/ at large | medium | mitigate | Sole-parser grep guard: any new mentioner of model_info.yaml in dnallm/ outside the EXPECTED_REGISTRY_MENTIONERS allowlist fails CI (`tests/models/test_registry.py:541`; the one pre-existing server.py comment exception is documented and scheduled for the Phase 26 MCP rewire) | closed |
| T-22-04 | Tampering | dnallm/models/model_info.yaml insertion path | high | mitigate | Scripted insertion-only path gated at execution time (unique-line keyed insertion, numstat zero-deleted, sentinel greps — recorded in 22-02-SUMMARY); permanent guards: section counts pinned 49/187 with field-level insertion-only check (`tests/models/test_model_info_registry.py:28,203,317`) and strict Pydantic parse over the packaged result | closed |
| T-22-05 | Spoofing | remote hub metadata (renamed/typosquatted repo serving wrong config.json) | medium | mitigate | Values are reversible data (D-01); load-bearing entries spot-pinned in tests (hyenadna 32770, DebertaV2Tokenizer — `tests/models/test_model_info_registry.py:221-233`); config.json vs tokenizer_config.json cross-check recorded per repo in .scratch/enrichment_results.json; derived-constant fallback carries per-entry cited derivations; probe fetches only metadata files, never executes fetched content; remote-vs-committed provenance independently re-verified live (2026-10-20 UAT, 10 repos, 0 mismatches) | closed |
| T-22-SC | Tampering | package installs | high | mitigate | Zero new dependencies this phase (pyproject.toml untouched across the phase window — verified via git log); loader needs only existing pyyaml + pydantic; legitimacy gate re-applies if any dependency is ever added | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|

*Accepted risks do not resurface in future audit runs.*

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-20 | 6 | 6 | 0 | Claude (verify-work inline dispatch, ASVS L1 short-circuit: threats_open=0, register authored at plan time) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-20
