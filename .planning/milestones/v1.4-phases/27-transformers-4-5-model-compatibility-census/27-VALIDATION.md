---
phase: "27"
slug: "transformers-4-5-model-compatibility-census"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-10-28"
---

# Phase 27 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Reconstructed from artifacts 2026-10-28 (State B) after phase completion — every claim below was re-checked against collected/executed tests in this run.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (pytest-asyncio auto, pytest-timeout 300s) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py tests/models/test_compat_docs_consistency.py tests/models/test_census_compat.py -q` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ dnallm/mcp/tests/ -q` (CI additionally gates coverage ≥90) |
| **Estimated runtime** | ~1 s quick lane surface (164 tests); full offline suite minutes-scale |

---

## Sampling Rate

- **After every task commit:** quick lane-surface command above (owner execution-scope rule 2026-10-09: targeted verify + lane files, no repo-wide lanes)
- **After every plan wave:** the phase's six test files together (252 tests collected)
- **Before `/gsd-verify-work`:** lane surface + registry suites green
- **Max feedback latency:** ~2 s (quick lane surface)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 27-01-01 census tool (--run/--with-loads/--artifact) | 01 | 1 | census evidence producer | T-27-01/03 | fetch allowlist; weights gated behind --with-loads | unit | `.venv/bin/python -m pytest tests/models/test_census_compat.py -q` | ✅ | ✅ green (83) |
| 27-01-02 committed evidence artifact | 01 | 1 | 49-record compat_census.json | T-27-02 | every record stamps hub + revision | unit | `pytest tests/models/test_census_compat.py -q` (artifact integrity cases) | ✅ | ✅ green |
| 27-01-03 census-blocked honest verdicts | 01 | 1 | unreachable ≠ green | T-27-13 | unreachable repos surface as census-blocked records | unit | `pytest tests/models/test_census_compat.py -q` | ✅ | ✅ green |
| 27-02-01 compat rungs (TDD) | 02 | 2 | restore fixable failures | T-27-04/05/06 | closed-map, absence-gated, never-overwrite rungs | unit | `.venv/bin/python -m pytest tests/utils/test_transformers_compat_census.py -q` | ✅ | ✅ green (19; includes no-module-`__getattr__` + never-overwrite pins) |
| 27-02-02 post-shim census re-run | 02 | 2 | recovered-green records carry no disposition | T-27-02 | re-run maps onto committed statuses | unit | `pytest tests/models/test_census_compat.py -q` | ✅ | ✅ green |
| 27-02-03 disposition record (fix/skip) | 02 | 2 | every failure dispositioned | T-27-07 | dispositions ride merge_records only | unit | `pytest tests/models/test_census_compat.py -q` | ✅ | ✅ green |
| 27-03-01 RegistryEntry schema + loader | 03 | 3 | 3 optional-nullable compat fields | T-27-08 | extra=forbid schema ships with loader tests in same change | unit | `.venv/bin/python -m pytest tests/models/test_registry.py -q` | ✅ | ✅ green (47) |
| 27-03-02 registry write-back (anchored) | 03 | 3 | 49 pretrained rows gain tf_compat_* | T-27-09 | anchored insertion preserves comments/provenance | unit | `.venv/bin/python -m pytest tests/models/test_model_info_registry.py -q` | ✅ | ✅ green (22; provenance sentinel incl.) |
| 27-03-03 independent consistency gate | 03 | 3 | artifact ↔ registry agree | T-27-07 | re-derives every status independent of writer | unit | `pytest tests/models/test_model_info_registry.py -q` | ✅ | ✅ green |
| 27-04-01 generator + docs page | 04 | 4 | plain-language compatibility table | T-27-10 | GENERATED header; page never hand-edited | unit | `.venv/bin/python -m pytest tests/models/test_compat_docs_consistency.py -q` | ✅ | ✅ green (68) |
| 27-04-02 offline drift gate | 04 | 4 | page ↔ registry byte-pinned | T-27-10 | gate re-derives all 49 rows, no generator import | unit | `pytest tests/models/test_compat_docs_consistency.py -q` | ✅ | ✅ green |
| 27-04-03 dispatch-only lane | 04 | 4 | D-08 fail-closed lane | T-27-11/12/13 | workflow_dispatch only; contents:read; no masking | contract | `.venv/bin/python -m pytest tests/test_runner_infra_contracts.py -q` | ✅ | ✅ green (posture pins) |
| 27-05-01 lane mamba install + contract | 05 | 5 | G-27-1 workflow half | T-27-14/SC-05 | sanctioned sha256-pinned installer; ordering; no masking; no continue-on-error | contract | `pytest tests/test_runner_infra_contracts.py -q` | ✅ | ✅ green (13, TestCensusLaneInstallSequence + posture) |
| 27-05-02 plain-language failure cells (TDD) | 05 | 5 | G-27-2 | T-27-15 | per-class honest leads; fail-closed map; 300-char cap; jargon blocklist | unit | `pytest tests/models/test_compat_docs_consistency.py -q` | ✅ | ✅ green (68; RED evidence at .scratch/p27-uat/tdd-27-05-red-evidence.json) |
| 27-05-03 replica green run | 05 | 5 | G-27-1 environmental proof | T-27-14 | lane's exact command exits 0 on unchanged inputs | run-evidence | `grep -c "^  matched " .scratch/p27-uat/lane-sim/census-green.log` → 49; `grep LANE-EXIT-CODE` → 0 (asserts encoded in plan `<verify>`; the lane itself re-verifies via `--verify` on dispatch) | ✅ | ✅ green (302 s run, log committed to scratch) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements (pytest suite + committed workflow-content contract tests + census artifact already in place at phase start).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live `gh workflow run census_compat.yml` dispatch end-to-end green | 27-04-03 / 27-05-01 (D-08 lane) | Structurally impossible before the v1.4→main merge (GitHub registers dispatchable workflows from the default branch only; main frozen) — deferred operational act, not a code gap | After merge: `gh workflow run census_compat.yml --ref main`, expect 49 matched / exit 0 / artifact upload; interim evidence = runner-box replica run (census-green.log) |

---

## Validation Audit

| Audit Date | Gaps Found | Resolved | Escalated |
|------------|-----------|----------|-----------|
| 2026-10-28 | 0 | 0 | 0 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none MISSING)
- [x] No watch-mode flags
- [x] Feedback latency < 2 s (quick lane surface)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-28 (validate-phase State B reconstruction; 252 tests collected / 164 lane-surface tests re-run green in this audit)
