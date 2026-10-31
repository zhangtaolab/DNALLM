---
phase: "26"
slug: "mcp-consistency-registry-keyed-check-honest-listing"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-22"
verified: "2026-10-22"
---

# Phase 26 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
>
> The register was authored at plan time — both plans carry `<threat_model>` blocks
> (T-26-01-01…T-26-01-05, T-26-02-01…T-26-02-04). This audit re-verified each mitigation
> against the tree as committed (b09ebf2…f9fba20).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| committed config dir → checker | YAML files on disk (repo-committed today, any directory tomorrow) are parsed and compared — parsed values never execute | config field values (task type, labels, thresholds) |
| packaged registry → checker / listing | frozen Pydantic entries flow into comparisons, finding messages, and listing rows | repo ids, task types, label names |
| checker → CI | the `passed` bool is the only CI-consumed signal; CI never parses message prose | boolean + reason codes |
| MCP client → listing tool | zero-argument tool call crossing the protocol boundary; no client-controlled input enters the listing path | none (no parameters) |
| server state → response | config manager (parsed committed YAML), model manager (live engine map), registry compose a public payload | repo ids, task types, configured/enabled/loaded flags, server-config keys |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-26-01-01 | Tampering | checker mutates configs or registry while scanning | high | mitigate | D-10 never-writes: behavioral test (byte-identical inputs after a drifted run, `TestNeverWrites`) + source-scan test (no yaml.dump/write_text/write-mode open — verified: only `read_text` + 2× `yaml.safe_load` in the module) + read-only registry accessors only | closed |
| T-26-01-02 | Elevation of Privilege / Supply chain | malicious YAML object construction during parse | medium | mitigate | `yaml.safe_load` exclusively at both parse sites (consistency.py:158, :344 — verified by grep: no `yaml.load`, no unsafe/full loaders); parsed doc treated as plain data | closed |
| T-26-01-03 | Information Disclosure | finding messages leak sensitive paths or secrets | low | mitigate | messages carry config file names (`path.name`), repo ids, and field values only (verified: no absolute-path interpolation in any of the 12 Finding constructions) | closed |
| T-26-01-04 | Denial of Service | pathological config dir stalls CI | low | accept | scope is the committed 7-file directory; the walk is glob-bounded, no recursion, no network — milliseconds | closed |
| T-26-01-05 | Spoofing | checker reports green while drift exists (false-negative gate) | high | mitigate | the red-when-drift matrix proves every tier fires (21 tests incl. same-members label reorder and 0.500001 near-miss); strict semantics tested per code; the gate test asserts the `passed` signal, never message strings | closed |
| T-26-02-01 | Information Disclosure | listing exposes repo ids, task types, configured/enabled/loaded state, server-config keys | low | accept | all values are repo-committed public data (configs and registry ship in the repo); no secrets, paths, or user data enter the payload | closed |
| T-26-02-02 | Spoofing | menu over-claims callability (issue #64 class: name says available, reality says configured-only) | high | mitigate | the overlay is the fix — every row carries configured/enabled/loaded plus a status word; `unmatched_configured` keeps registry-unknown configured models visible instead of silently dropped (pinned by the overlay matrix) | closed |
| T-26-02-03 | Tampering | listing path mutates config or manager state | medium | mitigate | pure composition — the body reads exactly three states (`get_server_config()`, `model_configs.get()`, `get_loaded_models()`, verified by grep) and writes nothing; no new server state | closed |
| T-26-02-04 | Denial of Service | 236-row payload built per call, unbounded growth | low | mitigate | bounded by the packaged registry size; assembly is one sort + dict build (milliseconds); the timeout wrapper still wraps the tool (server.py:410, verified); truncation decision recorded (revisit at ~10x growth) | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-26-01 | T-26-01-04 | CI runtime exposure bounded by the glob over a committed 7-file directory; no recursion, no network | owner (plan-time disposition) | 2026-10-22 |
| AR-26-02 | T-26-02-01 | Payload contents are repo-committed public data; ASVS L1 documented acceptance | owner (plan-time disposition) | 2026-10-22 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-22 | 9 | 9 | 0 | gsd-execute-phase inline audit (L1) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-22
