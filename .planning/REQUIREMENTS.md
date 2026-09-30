# Requirements: DNALLM — Test Suite Audit & Coverage Hardening

**Defined:** 2026-09-29
**Core Value:** A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Harness & Measurement Config

- [x] **HARN-01**: Delete `tests/pytest.ini`; `pyproject.toml [tool.pytest.ini_options]` is the single pytest config source; CI invokes bare `pytest` so both test roots (`tests/`, `dnallm/mcp/tests/`) are collected
- [x] **HARN-02**: Fix exit-code masking in root `conftest.py` (cleanup via `pytest_sessionfinish` with status propagation, no unconditional `os._exit(0)`); add a permanent CI canary step proving a failing test run fails the job
- [x] **HARN-03**: Add `[tool.coverage.run]`/`[tool.coverage.report]` to `pyproject.toml`: `source = ["dnallm"]`, omit list (vendored `dnallm/tasks/metrics/`, `enformer_model/`, unimportable `megatron.py`/`mamba_npu.py`, test files `dnallm/mcp/tests/*`, `run_tests.py`, `example_sse_usage.py`), `show_missing` — no `fail_under` yet
- [x] **HARN-04**: Bump test dependency floors: `pytest-cov>=7.0`, `pytest-asyncio>=1.0`, `pytest-timeout>=2.3.1,<2.5`, `coverage[toml]>=7.10.6`

### Audit & Baseline

- [x] **AUDIT-01**: Full-suite audit census (both roots, `slow` included): pass/fail/skip counts by skip reason
- [x] **AUDIT-02**: Ranked per-module coverage gap report (`term-missing` + machine-readable artifact) as the test-authoring worklist
- [x] **AUDIT-03**: Measured baseline coverage % on the agreed denominator + slow-test wall-clock timings (cold and warm cache)
- [x] **AUDIT-04**: Explicit subprocess-coverage scope decision (canary-driven: start minimal, escalate to `patch = ["subprocess"]` only on evidence)

### Known-Defect Fixes

- [x] **FIX-01**: Fix multiclass AUROC crash in `dnallm/tasks/metrics.py:283` and unskip `tests/tasks/test_metrics.py:761`
- [x] **FIX-02**: Fix CrossDNA handler result overwrite in `dnallm/models/model.py:873-887` (early-return chain-of-responsibility pattern) and add regression test
- [x] **FIX-03**: Replace broad `except Exception: pytest.skip` with typed network skips; enforce an expected-skip allowlist
- [x] **FIX-04**: Point PDF test artifacts at `tmp_path`; fix `.gitignore` typo (`test/inference/pdf/` → `tests/inference/pdf/`)

### Test Authoring

- [x] **TEST-01**: Tests for `models/model.py` + `special/*` (dispatch-chain fault-injection, retry/reason-classification branches, tokenizer fallback chain)
- [x] **TEST-02**: Tests for `mcp/server.py` (transports, streaming generators, timeout-wrapper error paths)
- [x] **TEST-03**: Tests for `inference/*` (engine paths, logits→predictions, interpret/mutagenesis/benchmark)
- [ ] **TEST-04**: Tests for `datahandling`/`finetune` (dataset loading/tokenization/augmentation, trainer wiring)
- [ ] **TEST-05**: Tests for `cli/` + utils compat shims (CliRunner; `transformers_compat` as behavior contract, not line completion)
- [ ] **TEST-06**: Coverage >90% on the agreed denominator, with every new test holding at least one observable-behavior assertion; pragma budget held at baseline (3)

### CI Gate

- [ ] **GATE-01**: `fail_under = 90` in `[tool.coverage.report]` — enabled only after the suite first crosses 90% (ratchet, never permanently red)
- [ ] **GATE-02**: Dedicated single-leg slow-inclusive coverage CI job (py3.12, full suite, HF model cache keyed on `models.lock`, per-test timeout marks + job-level `timeout-minutes` backstop)
- [ ] **GATE-03**: Fix or remove the dead `codecov-action@v3` step (→ `@v7` or drop); reporting only, never the gate
- [ ] **GATE-04**: Synthetic-regression proof that the gate actually fails CI when coverage drops (end-to-end exercise of HARN-02)
- [ ] **GATE-05**: Gate runs on PRs to every protected branch (`dev` and `main`), not just `main`

## v2 Requirements

Deferred decay-prevention capabilities (research P2 — after the gate is green):

### Patch Coverage & Trends

- **PATCH-01**: PR patch (diff) coverage status on every pull request (Codecov status or diff-cover)
- **PATCH-02**: Coverage trend tracking with ratchet automation (threshold file that only moves up)
- **FLAKY-01**: Selective `@pytest.mark.flaky` reruns for demonstrated network flakes only
- **CTX-01**: `--cov-context=test` in the authoring loop to accelerate gap-closing
- **LANE-01**: Two-lane CI split (fast un-gated matrix + slow gated lane) if gate runtime hurts; nightly drift report

## Out of Scope

| Feature | Reason |
|---------|--------|
| Vendored code coverage (`dnallm/tasks/metrics/`, `enformer_model/`) | Upstream HF `evaluate` / ported Enformer; excluded from lint/mypy by design |
| `megatron.py` / `mamba_npu.py` coverage | Cannot import without Megatron-LM / Ascend NPU toolchains (~3.9 dead points of denominator) |
| Branch coverage (`branch = true`) | Reads 5–10pp lower; stage-2 metric after line gate is green (research recommendation) |
| New test frameworks (Hypothesis, mutmut, pytest-xdist) | PROJECT.md constraint — no new frameworks; xdist is a measured decision post-gate |
| Root `cli/` legacy launcher cleanup | Packaging concern, not needed for coverage |
| mypy `|| true` CI fix, dependency lockfile | Separate quality work outside this milestone |
| Performance optimization (e.g. `attn_implementation` hardcoding) | Record, don't fix |
| Mutation testing as a CI gate | Anti-feature per research (spot-checks only, not enforced) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HARN-01 | Phase 1 | Complete |
| HARN-02 | Phase 1 | Complete |
| HARN-03 | Phase 1 | Complete |
| HARN-04 | Phase 1 | Complete |
| AUDIT-01 | Phase 1 | Complete |
| AUDIT-02 | Phase 1 | Complete |
| AUDIT-03 | Phase 1 | Complete |
| AUDIT-04 | Phase 1 | Complete |
| FIX-01 | Phase 2 | Complete |
| FIX-02 | Phase 2 | Complete |
| FIX-03 | Phase 2 | Complete |
| FIX-04 | Phase 2 | Complete |
| TEST-01 | Phase 3 | Complete |
| TEST-02 | Phase 3 | Complete |
| TEST-03 | Phase 3 | Complete |
| TEST-04 | Phase 3 | Pending |
| TEST-05 | Phase 3 | Pending |
| TEST-06 | Phase 3 | Pending |
| GATE-01 | Phase 4 | Pending |
| GATE-02 | Phase 4 | Pending |
| GATE-03 | Phase 4 | Pending |
| GATE-04 | Phase 4 | Pending |
| GATE-05 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-09-29*
*Last updated: 2026-09-29 after roadmap creation (traceability mapped)*
