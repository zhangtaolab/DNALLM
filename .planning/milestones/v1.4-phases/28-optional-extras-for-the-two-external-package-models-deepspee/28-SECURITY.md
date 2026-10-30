---
phase: "28"
slug: "optional-extras-for-the-two-external-package-models-deepspee"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-30"
---

# Phase 28 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| package index -> installer | A remote sdist crosses into a build that will execute upstream setup code | flash-attn sdist bytes (integrity-critical) |
| pyproject/CI -> developer and runner environments | Dependency declarations select code installed into trusted Python environments | extras metadata, workflow definitions |
| remote model repositories -> census subprocess | trust_remote_code model implementations execute during load and forward probes | remote code execution during probing |
| measured registry evidence -> generated public compatibility page | A generic or incorrect wording map can misrepresent the measured load/prediction split | registry/census records -> public docs |
| README dependency table -> user installation choice | Missing or overclaiming rows can send a biologist toward an unusable prediction workflow | docs text -> user decisions |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-28-01 | Tampering | scripts/install_flash_attn.sh fetch/build pipeline | high | mitigate | Exact sdist filename plus pinned sha256, index-fragment precheck, local hash gate before extraction, occurrence-counted patch checks, and a corrupted-file negative test | closed |
| T-28-02 | Tampering | deepspeed dependency resolution | high | mitigate | Cap at <0.19.7 (sparse_attention removed upstream in 0.19.7); assert installed 0.19.6 and prove the real BigBird forward | closed |
| T-28-03 | Elevation | remote model load probes | medium | mitigate | Only the two owner-selected repositories through the existing census subprocess route at recorded revisions; replicas isolated; no workflow PR-reachable | closed |
| T-28-04 | Tampering | census seed and registry write-back | high | mitigate | Plan-01 green load+forward records required; exactly two repos flipped through the sanctioned writer; 49 verdicts independently re-derived; GPN/Omni-DNA pinned unchanged | closed |
| T-28-05 | Tampering | model.py and crossdna.py exception interception | medium | mitigate | Match only deepspeed ModuleNotFoundError names or the exact Flash Attention ImportError; cycle-safe cause traversal; nonmatching exception controls tested | closed |
| T-28-06 | Information disclosure | generated model_compatibility.md | low | mitigate | Evidence uses only hub/repo@revision plus package/rung names, never local cache paths (0 hits verified); fail-closed docs consistency gate + byte-identity check retained | closed |
| T-28-07 | Tampering | optional_dependencies.md install guidance | medium | mitigate | Exact curated-extra commands and --no-build-isolation pinned; Linux + CUDA constraints stated; GPN/Omni-DNA sections preserved by sha256; strict MkDocs | closed |
| T-28-08 | Tampering | six notebooks and three rendered mirrors | low | mitigate | Only install strings changed; every notebook parsed as JSON; retired form asserted absent; check_notebook_md_sync.py over the complete pair set | closed |
| T-28-09 | Elevation | ci.yml reusable call and extras_compat.yml triggers | high | mitigate | No push/pull_request triggers (verified: on.schedule + workflow_dispatch only); manual caller requires inputs.run_extras_compat; contents:read, no secrets; offline trigger/gate contracts | closed |
| T-28-10 | Tampering | extras lane probe and verification steps | high | mitigate | Real load+forward assertion separated from --verify; zero continue-on-error occurrences; fail-closed without GPU; evidence always uploaded | closed |
| T-28-11 | Information disclosure | extras-fresh-evidence artifact | medium | mitigate | Upload only the census JSON and GPU-absence marker; no environment dumps, secrets, or pip config; only GPU name/compute capability/driver recorded | closed |
| T-28-12 | Tampering | README, installation, troubleshooting, and Windows guidance | medium | mitigate | Retired install forms asserted absent (19 guard assertions); explicit base/built-in wording on every affected surface; Windows marker and KDA probe semantics preserved; strict MkDocs for guides | closed |
| T-28-13 | Tampering | BigBird lead mapping and README model rows | medium | mitigate | One evidence-specific split class; generated cell independently re-derived; registry/evidence pins preserved; byte-identical regeneration required; README assertions scoped to the two rows | closed |
| T-28-SC | Tampering | package-manager installs (none added) | high | mitigate | Presentation-only plan 28-06 adds no dependency or install task (pyproject untouched in its window); existing package-legitimacy and pinned-installer controls unchanged | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-30 | 14 | 14 | 0 | orchestrator (L1 grep-depth, asvs_level 1 short-circuit; every mitigation re-verified against the implementation) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-30
