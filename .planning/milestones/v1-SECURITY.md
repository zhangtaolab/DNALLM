---
milestone: v1
audited: 2026-10-01T14:40:00Z
verdict: SECURED
threats_open: 0
asvs_level: 1
block_on: high
scope: "65 non-planning files in fe1d4b4..HEAD; threat register from 14 phase PLANs + 1 quick-task PLAN"
auditor: gsd-security-auditor (ship-time retroactive audit, /gsd-ship)
---

# Milestone v1 Security Review

**Verdict: SECURED — 0 unresolved findings at high or above** (config: `security_asvs_level=1`, `security_block_on=high`).

All **37 registered threats** across the 14 phase plans and the self-hosted-runner quick task resolve to **CLOSED** with code-level evidence. Highlights of the blocking tier:

| Threat | Severity | Evidence |
|--------|----------|----------|
| PR-code-never-on-self-hosted-box (T-261001-01) | critical | Both `dnallm-nightly` jobs (`test-mamba` ci.yml:270, `coverage-nightly` ci.yml:410) gated `schedule \|\| workflow_dispatch`; every push/PR job hosted (test:24, test-windows:127, test-cuda:201, coverage-gate:345, deploy:500); repo-wide grep confirms no other self-hosted jobs exist |
| Exit-code mask removal (T-01-02) | high | `conftest.py:27-32` `pytest_sessionfinish` cleanup, no forced exit; permanent canary steps ci.yml:98-114/177-193 re-prove failing runs exit non-zero |
| Skip-allowlist suppression vector (T-02-05) | high | `audit_skips.py:40-45` exactly-one/no-empty-matcher gate, malformed → exit 1; narrow allowlist entries; typed httpx-only skip helper (`_network_skip.py:43-62`) |
| Third-party uploader removal (T-04-03) | high | `codecov-action@v3` + coverage.xml export deleted; only first-party actions remain |
| Coverage metric gaming (T-3-12) | high | pragma count exactly 3 (compat shims), `fail_under=90` with vendored-only omit list |
| Zero new packages/actions (8× T-SC) | high | pyproject delta is floor bumps on existing deps; new `.github/dependabot.yml` is update automation, not an install |

Cache poisoning (T-04-04): `models.lock`-keyed cache is save-on-success and GitHub cache branch-scoping makes PR-branch saves invisible to main-branch scheduled runs — no PR→self-hosted poisoning path. The 8 milestone source fixes are behavior corrections with no validation/injection surface added; the vendored-metrics switch (`evaluate.load` on a local path) reduces network exposure vs hub fetch. The ~1,000 new tests contain no credentials, no unexpected hosts, no unmocked `trust_remote_code` loads, no subprocess, no writes outside `tmp_path`.

## Advisory findings (non-blocking, below `block_on`)

1. **[medium] ci.yml:270,410 — event guard is bypassable by a PR that edits the workflow file itself.** The declared mitigation is implemented as specified and this residual is inherent to hosting a self-hosted runner beside `pull_request` triggers. *Recommended owner action: restrict the `dnallm-nightly` runner group (Settings → Actions → Runner groups: disable fork-PR access / require approval for all external runs) and/or protect `.github/workflows/**` paths.*
2. [low] ci.yml (6 sites) — `curl-pipe-sh` uv installer retained (accepted T-04-06); consider pinned `astral-sh/setup-uv` when convenient.
3. [low] `dnallm/models/tokenizer.py:295-299` — stage-2 fast-tokenizer fallback now forwards `trust_remote_code=True` (stage-1 already ran with it; net expansion negligible); consider explicit `False` for least privilege.
4. [low] `docs-validation.yml:41-66` — masked-red `continue-on-error` steps confirmed reliability-only (hosted, no secrets, no `pull_request_target`); matches audit tech-debt WR-08.
5. [low] ci.yml:517 — `actions/cache@v3` in deploy (first-party, legacy major); dependabot github-actions ecosystem will surface the bump.

---
*Ship gate: `security_enforcement` ship:pre gate satisfied via `threats_open: 0`. Full 37-entry verification table in the auditor's working output; this file is the durable record.*
