# Roadmap: DNALLM — Test Suite Audit & Coverage Hardening

## Overview

This milestone turns an untrustworthy 464-test suite into an enforced >90% coverage gate. Today the pipeline lies at two reproduced points — a root `conftest.py` exit-code mask that makes every failing run exit 0, and a `tests/pytest.ini` that silently drops the `dnallm/mcp/tests/` root from CI collection — so no pass/fail result or coverage number from this repo can be trusted. The journey runs strictly in dependency order, matching the research consensus: first make results honest (harness fix + measured baseline), then make the suite honest (unskip crash-skipped tests, fix known bugs, type the skips), then close the measured gaps to >90% with behavior-verifying tests, and only then lock the door with a CI gate whose first blocking day is its first green day.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Harness Integrity & Measured Baseline** - Make every test result and coverage number trustworthy: one pytest config, real exit codes, agreed denominator, measured baseline (completed 2026-09-30)
- [ ] **Phase 2: Suite Hygiene & Known-Bug Fixes** - Fix the known code defects hiding behind skips and make every remaining skip typed and intentional
- [ ] **Phase 3: Test Authoring to >90% Coverage** - Close the measured gaps biggest-first with tests that assert observable behavior until coverage exceeds 90%
- [ ] **Phase 4: CI Gate Enforcement** - Turn 90% into a ratcheted CI hard gate that provably fails when coverage drops

## Phase Details

### Phase 1: Harness Integrity & Measured Baseline

**Goal**: Every test result and coverage number from this repo is trustworthy — failing runs actually fail, both test roots are collected under a single config, and the distance to 90% is a measured number instead of a guess
**Depends on**: Nothing (first phase)
**Requirements**: HARN-01, HARN-02, HARN-03, HARN-04, AUDIT-01, AUDIT-02, AUDIT-03, AUDIT-04
**Success Criteria** (what must be TRUE):
  1. Bare `pytest` from the repo root collects both test roots (`tests/` and `dnallm/mcp/tests/`) with timeout and asyncio auto-mode active; `tests/pytest.ini` no longer exists and `pyproject.toml` is the only pytest config source
  2. A pytest run containing a failing test exits non-zero — proven by a permanent CI canary step — so the `os._exit(0)` exit-code mask can never return silently
  3. A coverage run driven purely by `pyproject.toml` config (no CLI cov flags) reports over the agreed denominator: vendored dirs, unimportable adapters, and packaged test files appear in no report row, and no `fail_under` exists yet
  4. An audit report exists with pass/fail/skip counts by skip reason (both roots, `slow` included), a ranked per-module gap worklist (`term-missing` + machine-readable artifact), the measured baseline coverage %, and cold/warm slow-test wall-clock timings
  5. The subprocess-coverage scope decision is recorded with canary evidence (start minimal; escalate to `patch = ["subprocess"]` only on proof)

**Plans**: 2/2 plans complete

Plans:
**Wave 1**
- [x] 01-01-PLAN.md — Single pytest config + honest exit codes + coverage config/floors + CI canary (HARN-01..04)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 01-02-PLAN.md — Full-suite audit: census by skip reason, ranked gap worklist, baseline %, cold/warm timings, subprocess decision (AUDIT-01..04)

### Phase 2: Suite Hygiene & Known-Bug Fixes

**Goal**: The suite reports true code behavior — no test is skipped because the code crashes, and every remaining skip is a typed, intentional network skip
**Depends on**: Phase 1
**Requirements**: FIX-01, FIX-02, FIX-03, FIX-04
**Success Criteria** (what must be TRUE):
  1. The multiclass AUROC test at `tests/tasks/test_metrics.py:761` runs unskipped and passes; `compute_metrics` handles multiclass targets without crashing
  2. CrossDNA handler results are returned instead of overwritten — a regression test asserts the correct handler's result survives the dispatch chain
  3. Every skip in the suite is typed (specific network exceptions) and matches an expected-skip allowlist; a new unexpected skip fails the run instead of passing silently
  4. Running the PDF-marked tests leaves the git working tree clean (artifacts written under `tmp_path`), and `.gitignore` ignores `tests/inference/pdf/` correctly

**Plans**: 3 plans

Plans:
**Wave 1**
- [ ] 02-01-PLAN.md — Multiclass AUROC presence-guard fix + CrossDNA dispatch fix, proven by unskipped and sentinel regression tests (FIX-01, FIX-02)
- [ ] 02-02-PLAN.md — PDF test artifacts under tmp_path, pdf marker applied, .gitignore typo fixed, strays deleted (FIX-04)

**Wave 2** *(blocked on Wave 1 — 02-01 owns tests/models/test_model.py and changes the skip census)*
- [ ] 02-03-PLAN.md — Typed network skips (httpx tuple + group unwrapping), dead-skip deletion, expected-skip allowlist + CI junit audit step (FIX-03)

### Phase 3: Test Authoring to >90% Coverage

**Goal**: Line coverage on the agreed denominator exceeds 90%, closed biggest-gap-first with tests that verify observable behavior rather than merely executing lines
**Depends on**: Phase 1 (gap report orders the work), Phase 2 (bug fixes land before tests that would enshrine the broken paths)
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06
**Success Criteria** (what must be TRUE):
  1. A single full-suite coverage run (both roots, `slow` included, config-only) reports total line coverage above 90% on the agreed denominator, reproducible with one local command
  2. `models/model.py` + `special/*` dispatch, retry/reason-classification, and tokenizer-fallback branches are exercised by fault-injection tests that assert which path was selected
  3. `mcp/server.py` transports, streaming generators, and timeout-wrapper error paths are covered by tests
  4. The `inference`, `datahandling`/`finetune`, and `cli` + compat-shim waves close their ranked gaps, with `transformers_compat` verified as a behavior contract (e.g. idempotent `apply_patches`), not line completion
  5. Every new test holds at least one observable-behavior assertion, and the pragma count stays at the recorded baseline (3)

**Plans**: TBD

> Sizing note (research flag): the distance to 90% is unknown until Phase 1's baseline lands. If the measured gap makes this phase too large, split it via `/gsd-phase` after Phase 1 — preserving the wave order (models → mcp → inference → datahandling/finetune → cli/shims) and keeping all waves after the Phase 2 bug fixes.

### Phase 4: CI Gate Enforcement

**Goal**: Coverage cannot regress — the gate goes live green and provably fails CI when coverage drops
**Depends on**: Phase 3
**Requirements**: GATE-01, GATE-02, GATE-03, GATE-04, GATE-05
**Success Criteria** (what must be TRUE):
  1. `fail_under = 90` is active in `[tool.coverage.report]` and enforced through the pytest exit code; the identical command runs locally and in CI
  2. A dedicated single-leg CI job (py3.12, full suite including `slow`) runs with the HF model cache keyed on `models.lock`, per-test timeout marks, and a job-level `timeout-minutes` backstop — and passes green
  3. A synthetic regression (a deliberately coverage-dropping change) demonstrably fails the CI job, exercising the Phase 1 exit-code fix end to end
  4. The coverage reporting step either runs on codecov-action v7 as reporting-only or is removed — no dead or failing reporting step remains
  5. PRs targeting both `dev` and `main` trigger the gated coverage job

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Harness Integrity & Measured Baseline | 2/2 | Complete    | 2026-09-30 |
| 2. Suite Hygiene & Known-Bug Fixes | 0/? | Not started | - |
| 3. Test Authoring to >90% Coverage | 0/? | Not started | - |
| 4. CI Gate Enforcement | 0/? | Not started | - |
