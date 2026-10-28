---
phase: "27"
slug: "transformers-4-5-model-compatibility-census"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-28"
---

# Phase 27 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| GitHub Actions runner -> package index | census lane's install step fetches the mamba sdist + causal_conv1d from the TUNA mirror on the self-hosted box | pip packages / sdist bytes (integrity-gated by sha256 pin) |
| Runner -> model hubs (ModelScope first, HF mirror) | census fetches remote modeling code + configs at recorded revisions; loads family-representative weights | remote code executed under trust_remote_code (dispatch-only lane, dev box, never PR-triggered) |
| registry -> docs page | the generated page is the public compatibility claim; plain-language failure leads are public honesty claims | verdicts + evidence strings (read-only derivation) |
| workflow_dispatch -> self-hosted runner | repo code executes on the dnallm-nightly GPU box only on owner dispatch (D-08 posture) | event payload (write-access-gated) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-27-01 | Elevation of Privilege / Tampering | remote modeling code in census import probes | high | mitigate | probes run in `python -c` subprocesses, no CWD dependence; dispatch-only self-hosted lane, never PR-triggered, never inside pytest — lane posture pinned by the 13 runner-contract tests (re-run green 2026-10-28) | closed |
| T-27-02 | Tampering | hub mirror divergence (HF vs ModelScope) | medium | mitigate | every evidence record stamps `hub` + `revision` (verified: 49/49 records carry both keys in tests/models/compat_census.json); ModelScope-first, consumers read the stamp | closed |
| T-27-03 | Denial of Service | gigabyte-scale weight fetches | medium | mitigate | CODE_PATTERNS allowlist (`*.py`, `config.json`, tokenizer metadata) in `--run`; weights only in explicit `--with-loads` for family representatives (census_compat.py:23,1262) | closed |
| T-27-04 | Tampering | catch-all `__getattr__` restore surface | high | mitigate | closed-map discipline only — the sole `__getattr__` is the instance-level closed default map; rung test `test_no_module_level_getattr_on_patched_modules` (test_transformers_compat_census.py:374-376) pins it | closed |
| T-27-05 | Tampering | vendored code drift from upstream v4.49.0 | medium | mitigate | verbatim-copy + citation convention (16 `v4.49.0` citations in transformers_compat.py); behavior tests pin semantics per rung | closed |
| T-27-06 | Tampering | rung overwrites stock transformers attributes | high | mitigate | absence-gated attach-only-missing rungs (hasattr guards at transformers_compat.py:186,260,333); never-overwrite test per rung; no-op on 4.x | closed |
| T-27-07 | Tampering / Repudiation | forged census evidence in registry | medium | mitigate | write-back reads only the committed artifact; Task-3 consistency test re-derives every status independently; provenance block records method + date (88 registry+rung tests green, verifier re-ran 2026-10-28) | closed |
| T-27-08 | Denial of Service | schema/data drift breaks package-wide registry access | high | mitigate | strict commit ordering (schema+loader tests before data); loader assertion in same change; pytest ships with every dnallm/ change (owner rule) | closed |
| T-27-09 | Tampering | YAML round-trip destroys provenance/comments | high | mitigate | writer performs anchored text insertion only (census_compat.py:1293,1412-1427); comment-preservation + idempotency + provenance-sentinel tests | closed |
| T-27-10 | Repudiation | hand-edited docs table contradicts registry | medium | mitigate | GENERATED header (line 1 of docs/user_guide/model_compatibility.md); drift gate re-derives all 49 rows independently (68 tests green) | closed |
| T-27-11 | Elevation of Privilege | lane triggered by push/PR/schedule | high | mitigate | `on: workflow_dispatch` + event-name guard + `permissions: contents: read` — byte-pinned by TestCensusLanePosture (13 contract tests) | closed |
| T-27-12 | Tampering | stale verdicts after transformers bump | medium | mitigate | `--verify` fail-closed diff incl. VERSION DRIFT; no `\|\| true` masking (contract-pinned); dependabot KNOWN RISK comment names the re-run duty | closed |
| T-27-13 | Denial of Service / Info Spoofing | lane greens with zero evidence | medium | mitigate | GPU-absent writes marker + exit 1 (census_compat.yml:44-54); unreachable repos become census-blocked records that diff UNEXPECTED — both paths red | closed |
| T-27-14 | Tampering / DoS | editorial revert / masked install failure re-creates permanently-red lane | high | mitigate | TestCensusLaneInstallSequence pins bootstrap + installer + ordering + no-masking + no-continue-on-error + posture (13 tests; deliberate-red proven); post-checkout artifact-dir fix (53d0017) closes the checkout-clean hole | closed |
| T-27-15 | Information Spoofing | failure-class lead mislabels the cause | medium | mitigate | per-class mapping over distinctive evidence substrings, fail-closed ValueError on unmapped shapes (verifier live-tested); gate re-derives + byte-pins the six rows + lead jargon blocklist | closed |
| T-27-SC-01..04 | Tampering | package supply chain (plans 27-01..27-04) | high | accept | each of those plans installs zero new packages; census runs against the pinned project venv — see Accepted Risks Log | closed |
| T-27-SC-05 | Tampering | package supply chain (pip in census lane, plan 27-05) | high | mitigate | lane installs mamba-ssm 2.3.2.post1 + causal_conv1d 1.7.0 exclusively via sha256-pinned scripts/install_mamba.sh (index hash cross-check + local sha256 gate before any build; fetch bypasses pip resolution); TUNA route integrity-neutral by the hash gate; no new package enters the repo dependency set | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-27-1 | T-27-SC-01..04 | Plans 27-01 through 27-04 install zero new packages (RESEARCH Package Legitimacy Audit found none to gate); the census executes against the already-pinned project venv — no install tasks exist to gate. The residual is the project's standing pinned-dependency posture, not a phase-introduced risk. | owner (plan-time dispositions) | 2026-10-28 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-28 | 18 | 18 | 0 | execute-phase orchestrator (L1 grep-depth pass; ASVS L1 short-circuit — register authored at plan time, zero open threats; mitigation anchors re-verified on disk: evidence stamps, fetch allowlist, closed-map guard + its pin test, absence gates, anchored writer, GENERATED header, lane posture pins, sha256 installer gate) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-28
