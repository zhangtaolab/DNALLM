---
milestone: v1
audited: 2026-10-01T12:46:00Z
status: tech_debt
scores:
  requirements: 23/23
  phases: 4/4
  integration: 8/8
  flows: 2/2
gaps:  # Critical blockers
  requirements: []
  integration: []
  flows: []
tech_debt:  # Non-critical, deferred
  - phase: 01-harness-integrity-measured-baseline
    items:
      - "WR-09 (warning, open): README \"Local Testing\" still prescribes the retired `.[test,dev]` install that CR-03 proved fails the documented census commands"
      - "WR-08 (warning, open): the WR-01 multilabel-curve guard fix has no regression test — guarded path unreachable from the suite"
      - "IN-03..IN-09, IN-10, IN-11, IN-12 (info, 10 items open in 01-REVIEW-DISPOSITION.md)"
  - phase: 02-suite-hygiene-known-bug-fixes
    items:
      - "WR-05 (warning, open): metrics_for_dnabert2(\"regression\") returns nested {\"r2\": {\"r2\": float}} — pinned by a phase-03 test as contract"
      - "WR-06 (warning, open): verbose task_type aliases validate but store the verbose spelling downstream dispatchers reject; pinned by tests"
      - "WR-07 (warning, open): README \"Local Testing\" `.[test,dev]` (same defect class as 01-WR-09, separate ledger entry)"
      - "WR-08 (warning, open): docs-validation.yml example-test step deterministically red, masked by continue-on-error: true"
      - "IN-02..IN-12 (info, 11 items open in 02-REVIEW-DISPOSITION.md)"
  - phase: 03-test-authoring-to-90-coverage
    items:
      - "WR-05 (warning, open): multilabel AUROC/AUPRC absent-summary guard added in fix round has no test"
      - "Deferred (deferred-items.md): logs/dnallm.log import-time sink recreated under pytest cwd every run (gitignored but plan-level Path('logs') gates unsatisfiable)"
      - "Deferred (deferred-items.md): test_timeout.py spends a fixed ~60s per census on two full-timeout waits"
      - "Deferred (deferred-items.md): raw_reverse_complement is a no-op (ds.map result discarded, data.py:983) — latent bug pinned as-is by test"
      - "Deferred (decisions): cosine_similarity loss TypeError recorded as latent bug, not fixed"
      - "IN-01, IN-02, IN-04..IN-08 (info, open in 03-REVIEW-DISPOSITION.md)"
  - phase: 04-ci-gate-enforcement
    items:
      - "Warning: nightly test-mamba leg's first live rehearsal of the de4b5cc `.[base]` install pending (next 03:00 UTC schedule; derivation + masked-red basis verified; non-gate lane)"
      - "Warning: models.lock stale entry — plant-dnamamba-BPE-open_chromatin provenance comment names the config swapped to plant-dnagpt-BPE-promoter at 95c9ba0 (one-line fix)"
      - "IN-01..IN-10 (info, open in 04-REVIEW-DISPOSITION.md)"
  - phase: cross-phase (integration checker, informational)
    items:
      - "W-2 (informational): test-cuda GPU leg runs `pytest tests/` (single root, no --cov) — neither dual-root nor floor-enforced; outside milestone single-config scope, noted as ungated leg"
      - "W-1 (informational): census statement counts wobble ±2 across artifacts (7405→7407) from post-phase-3 source commits; denominator config byte-stable — cite config, not absolute counts"
  - phase: milestone-close (STATE.md follow-ups)
    items:
      - "WINDOWS.md ledger triage (~10 entries + IN-01..07 info findings)"
      - "Optional: install dnallm-nightly runner as systemd service (svc.sh install) for reboot resilience"
---

# Milestone v1 Audit — DNALLM Test Suite Audit & Coverage Hardening

**Audited:** 2026-10-01T12:46:00Z · **Milestone definition of done:** a fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

**Verdict: tech_debt** — all 23 requirements satisfied, all 4 phases verified passed at HEAD, cross-phase integration fully wired (8/8), both E2E flows complete, zero critical blockers; accumulated open review findings (7 warning-tier + ~31 info-tier + 3 deferred engineering items) need review before/alongside milestone completion.

## Requirements Coverage (3-Source Cross-Reference)

Sources: phase VERIFICATION.md requirement tables × SUMMARY `requirements-completed` frontmatter × REQUIREMENTS.md traceability table.

| REQ-ID | Phase | VERIFICATION | SUMMARY frontmatter | Traceability | Final Status |
|--------|-------|--------------|--------------------|--------------|--------------|
| HARN-01 | 1 | SATISFIED | 01-01 | [x] | **satisfied** |
| HARN-02 | 1 | SATISFIED | 01-01 | [x] | **satisfied** |
| HARN-03 | 1 | SATISFIED | 01-01 | [x] | **satisfied** |
| HARN-04 | 1 | SATISFIED | 01-01 | [x] | **satisfied** |
| AUDIT-01 | 1 | SATISFIED | 01-02 | [x] | **satisfied** |
| AUDIT-02 | 1 | SATISFIED | 01-02 | [x] | **satisfied** |
| AUDIT-03 | 1 | SATISFIED | 01-02 | [x] | **satisfied** |
| AUDIT-04 | 1 | SATISFIED | 01-02 | [x] | **satisfied** |
| FIX-01 | 2 | SATISFIED | 02-01 | [x] | **satisfied** |
| FIX-02 | 2 | SATISFIED | 02-01 | [x] | **satisfied** |
| FIX-03 | 2 | SATISFIED | 02-03 | [x] | **satisfied** |
| FIX-04 | 2 | SATISFIED | 02-02 | [x] | **satisfied** |
| TEST-01 | 3 | SATISFIED | 03-02 | [x] | **satisfied** |
| TEST-02 | 3 | SATISFIED | 03-03 | [x] | **satisfied** |
| TEST-03 | 3 | SATISFIED | 03-01 `[]` (coverage block: D1–D4 pass) | [x] | **satisfied** (manually verified) |
| TEST-04 | 3 | SATISFIED | 03-04 | [x] | **satisfied** |
| TEST-05 | 3 | SATISFIED | 03-05 | [x] | **satisfied** |
| TEST-06 | 3 | SATISFIED | 03-05 | [x] | **satisfied** |
| GATE-01 | 4 | SATISFIED | 04-01 | [x] | **satisfied** |
| GATE-02 | 4 | SATISFIED (amended shape) | 04-01, 04-02 | [x] | **satisfied** |
| GATE-03 | 4 | SATISFIED (removed) | 04-02 | [x] | **satisfied** |
| GATE-04 | 4 | SATISFIED | 04-03 | [x] | **satisfied** |
| GATE-05 | 4 | SATISFIED (enforced) | 04-03 | [x] | **satisfied** |

**TEST-03 note:** the status matrix flagged it `partial` because 03-01-SUMMARY's top-level `requirements-completed:` is `[]`. Manual verification: the same SUMMARY's `coverage:` block carries 4 evidence entries (D1–D4, all `status: pass`) tagged `requirement: TEST-03`, and the Phase 03 VERIFICATION table satisfies it on a fresh verifier-owned census (inference area 175 missing ≤ 250 gate). Frontmatter omission only — recorded as a documentation nit, not a gap.

**Orphan detection:** none — all 23 traceability REQ-IDs appear in at least one phase VERIFICATION table; every phase table reports "orphaned requirements: none".

**FAIL gate:** 0 unsatisfied → not triggered.

## Phase Verification Summary

| Phase | Verified at | Status | Score | Gaps |
|-------|-------------|--------|-------|------|
| 01 Harness Integrity & Measured Baseline | 2026-10-01T12:33:05Z (final pass) | passed | 15/15 | none |
| 02 Suite Hygiene & Known-Bug Fixes | 2026-10-01T12:31:30Z | passed | 17/17 | none (4 advisory groups) |
| 03 Test Authoring to >90% Coverage | 2026-10-01T12:25:00Z | passed | 12/12 | none |
| 04 CI Gate Enforcement | 2026-10-01T12:40:00Z | passed | 10/10 | none (2 warnings, non-blocking) |

All four digests fresh at HEAD 2ab708a (commits after each verification's HEAD are `.planning/`-docs only — no digest staling).

## Integration Check (gsd-integration-checker, 2026-10-01)

**Score: 8/8** — 6/6 seams WIRED, 2/2 E2E flows COMPLETE, 0 blockers.

| Seam | From → To | Verdict | Key evidence |
|------|-----------|---------|--------------|
| 1 | HARN-02 → GATE-01 (exit-code fix carries the gate) | WIRED | conftest `pytest_sessionfinish` returns without forced exit; local probe: 43/43 tests pass yet exit 1 with `FAIL Required test coverage of 90.0%... 16.36%`; PR #39 `coverage-gate` check FAILURE live |
| 2 | HARN-03 → TEST-06/GATE-01 (denominator stability) | WIRED | `[tool.coverage.run]` byte-identical Phase-1 commit 5cf935f → HEAD; live TOTAL 7407 = Phase-3 figure; only coverage change ever = `fail_under = 90` (Phase 4, ae5c8ba) |
| 3 | FIX-03 → GATE-02 (skip audit in gate jobs) | WIRED | audit_skips.py blocking in 4 jobs with matching junit filenames; full prefix→junit→allowlist→audit chain traced; fail-closed probes exit 1 |
| 4 | FIX-01/FIX-02 → TEST-* (fixes load-bearing for Phase 3) | WIRED | guard at metrics.py:283–297 + regression test :315; guarded dispatch model.py:856–874 + tests :739/:1166; scoped run 109 passed; no test re-pins broken behavior |

| E2E Flow | Verdict | Key evidence |
|----------|---------|--------------|
| Developer: bare `pytest` / `pytest --cov` from root | COMPLETE | configfile: pyproject.toml; both roots in collection tree (1664); honest exit both directions; no local/CI fork |
| CI: PR gate + nightly census | COMPLETE | branch protection on dev AND main requires exactly `coverage-gate (py3.12, fast leg)` (live API); nightly self-hosted slow census green 96.30%; PR #39 red proof with zero residue |

Integration warnings (informational, mapped): W-1 census-count wobble (TEST-06/GATE-01 docs), W-2 test-cuda GPU leg ungated/single-root (HARN-01 scope note), W-3 matrix legs redundantly enforce fail_under (positive).

## Tech Debt by Phase

**Phase 01** — WR-09 README `.[test,dev]` install doc (warning); WR-08 WR-01-fix untested guard (warning); IN-03..IN-12 (10 info).

**Phase 02** — WR-05 nested r2 dict pinned as contract (warning); WR-06 verbose task_type alias storage (warning); WR-07 README `.[test,dev]` (warning); WR-08 docs-validation.yml masked-red example-test step (warning); IN-02..IN-12 (11 info).

**Phase 03** — WR-05 untested multilabel absent-summary guard (warning); deferred: logs/ import-time sink, ~60s timeout-test cost, raw_reverse_complement no-op (latent bug, pinned), cosine_similarity loss TypeError (latent bug, recorded); IN-01..IN-08 (7 info).

**Phase 04** — test-mamba nightly `.[base]` first live rehearsal pending (warning, non-gate lane); models.lock stale provenance entry (warning, one-line fix); IN-01..IN-10 (10 info).

**Cross-phase / milestone-close** — W-2 test-cuda ungated leg (informational); WINDOWS.md ledger triage; optional runner systemd install.

**Total: 7 warning-tier + ~31 info-tier review items + 3 deferred engineering items across 4 phases** — none block any requirement; all dispositioned open by owner decision and concentrated in the `/gsd-ship` triage path already recorded in STATE.md.

## Conclusion

The milestone's definition of done is met and defended in depth: the suite is fully passing (1657 passed / 7 allowlisted skips locally; 1656 in nightly census), coverage is 96.30% (7,133/7,407) on a denominator byte-stable since Phase 1, and the >90% gate is live in both directions (red proof PR #39, green nightly + fast leg) with branch protection enforcing it on dev and main. Cross-phase wiring is complete — the audit found no integration blockers and no orphaned or unsatisfied requirements. The remaining debt is reviewed-and-dispositioned findings (documentation staleness, two latent library bugs pinned by tests, CI-lane hygiene), appropriate for backlog triage during `/gsd-ship` rather than a closure phase.

---
*Audit: 2026-10-01T12:46:00Z · Orchestrator: Claude (gsd-audit-milestone) · Integration check: gsd-integration-checker subagent*
