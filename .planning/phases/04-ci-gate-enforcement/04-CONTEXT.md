# Phase 4: CI Gate Enforcement - Context

**Gathered:** 2026-09-30
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — no grey areas)

<domain>
## Phase Boundary

Coverage cannot regress — the gate goes live green and provably fails CI when coverage drops.

In scope (GATE-01..GATE-05, ROADMAP success criteria 1-5):
1. `fail_under = 90` active in `[tool.coverage.report]`, enforced through the pytest exit code; identical command locally and in CI
2. Dedicated single-leg CI job (py3.12, full suite incl. `slow`) with HF model cache keyed on `models.lock`, per-test timeout marks, job-level `timeout-minutes` backstop — passes green
3. A synthetic regression (deliberately coverage-dropping change) demonstrably fails the CI job — end-to-end exercise of the Phase-1 exit-code fix
4. Codecov step: codecov-action v7 as reporting-only, or removed — no dead/failing step remains
5. Gated coverage job triggers on PRs to both `dev` and `main`

Landing state at phase entry (Phase 3, verified): **96.30%** (7,131/7,405) — 6.3 points of headroom over the 90 gate; census command of record: `.venv/bin/python -m pytest -ra --durations=0 --junitxml=<junit> --cov -p no:cacheprovider -p no:progress`; 7 allowlisted skips enforced by `scripts/audit_skips.py` (CI step exists from Phase 2); pragma exactly 3.

Out of scope: two-lane CI split, patch coverage, nightly drift (v2 backlog per REQUIREMENTS.md).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP goal, success criteria, GATE-01..05, and codebase conventions.

Pre-locked by project decisions (do not re-litigate):
- Gate ratchet: enabled only now that the suite is above it (never permanently red) — 90, not 96
- The gated run includes `slow` tests (network accepted by owner)
- Codecov is reporting-only at most, never the gate
- GATE-02's `models.lock` cache key: the repo has no lockfile committed — the plan must either create the models.lock manifest or pick the equivalent cache-key mechanism; ratchet semantics stay
- Deferred from Phase-1 review (WR-02/WR-05/WR-06 dispositions): mamba no-op GPU leg, workflows-README broad staleness, uv installer pinning are OWNER decisions visible at this phase's CI rework — surface them in the plan as explicit decision points or leave untouched with rationale; do not silently expand scope

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ci.yml` current state: bare-pytest fast leg + junit + skip-audit step + exit-code canary (Phase 1/2), least-privilege permissions + mamba upload fix (Phase 1 review), codecov-action@v3 upload with `fail_ci_if_error: false` at line ~88 (the GATE-03 target)
- `pyproject.toml [tool.coverage.report]`: `show_missing`, no `fail_under` yet (by design until now)
- Proven census command + audit tooling; `tests/expected_skips.yaml`

### Established Patterns
- CI steps use `source .venv/bin/activate` preamble; YAML validated by PyYAML in tests; canary pattern (generate → run → assert rc) for behavioral CI proofs
- GATE-04's synthetic regression: commit a deliberately coverage-dropping change on a branch/PR, observe the job fail, then revert — the plan must encode a safe, local-first rehearsal plus the CI-observable proof

### Integration Points
- `.github/workflows/ci.yml` (new job + triggers), `pyproject.toml` (fail_under), possibly `.github/workflows/README.md` (WR-05 disposition)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discuss skipped (infrastructure phase).

</deferred>
