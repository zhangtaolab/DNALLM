---
phase: "13"
slug: "ollama-surface-disposition"
status: verified
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: "2026-10-11"
---

# Phase 13 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| repo commits (push/PR) → ci.yml execution on the self-hosted dnallm-nightly runner | workflow YAML is attacker-reachable via PR-authored code; the example-nightly job never runs on push/PR (D-19 cron-string gate) but the fast-leg tests that READ ci.yml run everywhere | workflow YAML / test code (public repo surface) |
| gate probes → localhost services | the typed skip's honesty depends on real probe evidence reaching the allowlist matcher | probe HTTP results over localhost (:11434, :8000) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-13-01 | Tampering | ci.yml stage-0.5 census gate | medium | mitigate | Re-pin discipline: only the two triple-bearing lines changed (c99fa9d shape); Task 1 verify extracts the pinned literal and proves the measured line matches; gate stays job-fatal | closed |
| T-13-02 | Repudiation | ci.yml stage-4 audit loop + fail-soft summary | medium | mitigate | Exactly one list member removed; audit loop, upload globs, and summary greps untouched; 4-member junit list pinned by acceptance; end-anchored summary grep intact | closed |
| T-13-03 | Information Disclosure | runner-host ollama service (loopback access control) | low | mitigate | Committed unit file's loopback pin + TestOllamaUnitPins stay green; stage-3 deletion REMOVES a network-listener start from CI (net surface reduction); README live-drift notice (0.0.0.0, 2026-10-05) retained | closed |
| T-13-04 | Tampering | tests/expected_skips.yaml allowlist | high | mitigate | Comment-only edit enforced mechanically: matcher-diff gate proves the exact/prefix/reason_like set byte-identical to HEAD; prohibition P2 records the must-NOT; fail-closed audit demands per-skip prefix matching with probe evidence in-message | closed |
| T-13-SC | Tampering | package installs | high | mitigate | No package-manager installs in this phase (zero new dependencies; RESEARCH Package Legitimacy Audit: "No gate applicable: zero packages"); any [ASSUMED]/[SUS] package introduced later requires a blocking human checkpoint | closed |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-13-01 | T-13-03 | The LIVE runner box's ollama unit has drifted to OLLAMA_HOST=0.0.0.0:11434 (LAN-reachable, unauthenticated) with the loopback re-apply pending — documented in scripts/runner/README.md's live-drift notice. The COMMITTED unit file pin and TestOllamaUnitPins are intact; Phase 13 removed the last CI consumer, leaving the exposure an owner ops action (re-apply pin or disable service). Surfaced as code-review WR-02. | Claude (orchestrator, ASVS L1 audit) on behalf of owner | 2026-10-11 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-11 | 5 | 5 | 0 | Claude (verify-work → secure-phase, ASVS L1, short-circuit: threats_open 0 at plan-time register) |

Audit evidence basis (all independently re-verified 2026-10-11 by the phase verifier and code reviewer, live on this tree): census `216/217 tests collected (1 deselected)` idempotent across two collects matching both pinned ci.yml gate lines (T-13-01); stage-4 audit list exactly 4 junit members with `Stage 2.5` consumer suite passing 25/25 (T-13-02); stage 3 deleted removes the CI-side network-listener start; TestOllamaUnitPins green with the README drift notice retained (T-13-03); expected_skips.yaml diff is 3 comment lines only, matcher set byte-identical to HEAD via the MATCHERS-UNCHANGED gate (T-13-04); no dependency manifests touched by any phase commit (T-13-SC).

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-11
