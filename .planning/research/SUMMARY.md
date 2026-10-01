# Project Research Summary

**Project:** DNALLM — Test Suite Audit & Coverage Hardening (`dnallm` v0.5.2)
**Domain:** Coverage measurement and >90% CI enforcement for an existing layered Python ML library (pytest ecosystem)
**Researched:** 2026-09-29
**Confidence:** HIGH — top findings were reproduced empirically on this repository, and tool semantics verified against official docs of the exact installed toolchain (pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, pytest-asyncio 1.4.0, pytest-timeout 2.4.0)

## Executive Summary

This is not a greenfield build — it is a hardening program over an existing 464-test suite on a layered ML library (configuration → models → datahandling → finetune → inference → MCP). Experts structure such programs as a pipeline: test tiers → measurement → gate → CI, where each boundary can silently lie. Research found DNALLM's pipeline lies at two points, both reproduced locally on 2026-09-29. First, the repo-root `conftest.py` registers an `atexit` handler ending in `os._exit(0)`, which masks **every** pytest failure exit code (test failures=1, coverage gate=2, no-tests-collected=5 all become 0) — any gate wired today would be permanently green on day one, and the current CI "Run fast tests" step cannot actually fail on test failures. Second, `tests/pytest.ini` shadows `pyproject.toml` whenever CI invokes `pytest tests/` (pytest configfile discovery is first-match-wins, never merged), silently dropping the packaged `dnallm/mcp/tests/` root from collection entirely, along with `--timeout=300`, `--asyncio-mode=auto` (falls back to STRICT), and the `legacy` marker. Until both are fixed, no pass/fail result and no coverage number from this repo is trustworthy.

The recommended approach is strictly ordered: **fix the harness, then measure, then fix known bugs, then write tests, then gate last.** Land a `[tool.coverage]` section in `pyproject.toml` (none exists today — the denominator is currently defined by an ad-hoc CI flag) pinning `source = ["dnallm"]` and an `omit` list covering vendored code (`dnallm/tasks/metrics/`, `enformer_model/`), the two unimportable adapters (`megatron.py` 184 stmts + `mamba_npu.py` 141 stmts ≈ 3.9 dead percentage points of the 8,344-statement denominator), **and** the packaged test files (`dnallm/mcp/tests/*`, `run_tests.py`, `example_sse_usage.py`) that currently inflate the numerator. Take a measured slow-inclusive baseline and a ranked per-module gap report; close gaps biggest-denominator-first (models → mcp → inference → datahandling/finetune → cli/shims); only then flip `fail_under = 90` in a dedicated single-leg CI job whose failure mode is proven with a canary. The installed toolchain needs no new frameworks — only dependency-floor bumps (`pytest-cov>=7.0`, `pytest-asyncio>=1.0`, `pytest-timeout>=2.3.1,<2.5`, `coverage[toml]>=7.10.6`) and one dead CI step (`codecov-action@v3` → `@v7` or drop; it fails on current runners).

Key risks: enabling the gate before the suite reaches 90% (a permanently-red gate trains bypass behavior within weeks); coverage theater — assertion-free tests that execute code without verifying behavior, the cheapest cheat a numeric mandate invites; and network flakiness from the 16 slow tests inside a blocking gate (HF/ModelScope downloads, plus skip-on-error storms that make the measured number wobble with network weather). Mitigations are concrete and researched: ratchet semantics with the gate's first blocking day being its first green day, an observable-behavior assertion standard enforced in review with a pragma budget (baseline: 3), typed network skips with an expected-skip allowlist, HF model caching keyed on a lockfile, per-test timeout overrides with the `signal` method preserved, and enforcement via the pytest exit code in-job — never via Codecov.

## Key Findings

### Reproduced Harness Defects (the two findings everything else depends on)

1. **Exit-code masking — the gate that can never fail.** Root `conftest.py`'s `atexit` handler calls `os._exit(0)` unconditionally after pytest has raised `SystemExit(exitstatus)`; the OS sees 0. Reproduced twice (architecture and pitfalls researchers, independently): a run that should exit 5 exited 0. Consequence: `--cov-fail-under`, test failures, and collection errors are all invisible to CI. Fix in Phase 1: move cleanup to `pytest_sessionfinish(session, exitstatus)` and propagate the real status (`os._exit(exitstatus)` only if a hard exit must stay). Add a permanent canary: CI runs one deliberately failing throwaway test and asserts a non-zero shell exit.
2. **Config split-brain.** `pytest tests/` selects `tests/pytest.ini` as configfile (first match wins from the args' common ancestor; configs never merge) even though `.planning/codebase/TESTING.md` calls references to that file stale — the file exists on disk. CI runs `pytest tests/ -m "not slow"` at `ci.yml:81`, so CI never collects `dnallm/mcp/tests/`, runs under STRICT asyncio mode (currently latent: async MCP tests pass only via class-level `@pytest.mark.asyncio` markers), and has no timeout. Fix: delete `tests/pytest.ini` (do **not** port its `--disable-warnings` — it suppresses the skip warnings that reveal silently-skipped tests); CI invokes bare `pytest` so `testpaths = ["tests", "dnallm/mcp/tests"]` applies.

### Recommended Stack

No new frameworks. The installed stack is correct and was verified working locally, including `fail_under` enforcement from config alone with bare `--cov` (a local run printed the failure and exited accordingly). Required floor bumps and pins go in `[project.optional-dependencies].test`.

**Core technologies:**
- **coverage[toml] 7.16.2** (floor `>=7.10.6`) — measurement engine, config, omit policy; 7.10 added the `patch = subprocess/_exit` options that replace pytest-cov 7's removed `.pth` mechanism (escalation ladder, not default)
- **pytest-cov 7.1.0** (floor `>=7.0`, up from `>=6.0.0`) — the gate; 7.1.0 fixed the `--cov-fail-under` total-computation inconsistency; threshold lives **only** in config `fail_under`, never duplicated on the CLI
- **pytest 9.1.1** (floor `>=8.4`) — 9.1.0 fixed a 9.0 regression that silently ignored `--strict-markers`/`--strict-config` set via `addopts` (dnallm uses both); do NOT migrate to the new native `[tool.pytest]` table (incompatible with existing `[tool.pytest.ini_options]`)
- **pytest-asyncio 1.4.0** (floor `>=1.0`, up from `>=0.21.1`) — keeps `asyncio_mode=auto`; a fresh resolve must never pair 0.x asyncio with pytest 9
- **pytest-timeout 2.4.0** (pin `>=2.3.1,<2.5` — 2.5.0 was yanked) — keep the `signal` method (Linux default): it fails only the timed-out test and preserves the coverage report; `thread` kills the whole process and the data
- **actions/cache@v4** for `~/.cache/huggingface` keyed on a `models.lock` file (model ids + revisions, NOT `github.sha`), with `restore-keys` prefix fallback — makes the slow-inclusive gate run practical
- **codecov-action v3 → v7** (v7.1.1 verified current) or delete the step — reporting only (`fail_ci_if_error: false` stays); the gate is the pytest exit code, never Codecov

**Explicitly avoided:** `branch = true` for this milestone (line coverage per owner decision; branch reads 5–10pp lower — stage it after the gate is green); `exclude_lines` (replaces the default exclusion set — use `exclude_also`); CLI `--cov-fail-under` (second source of truth); `parallel = true` (documented as pointless under pytest-cov); new test frameworks (Hypothesis/mutmut — PROJECT.md constraint); pytest-timeout 2.5.0; the `thread` timeout method.

### Expected Features

**Must have (table stakes, P1 — the honest 90%):**
- Single pytest config source of truth — retire `tests/pytest.ini`
- `[tool.coverage.run]/[report]` in pyproject.toml: `source`, the agreed `omit`, `fail_under` placement, `show_missing`, `skip_covered`
- Full-suite audit census: pass/fail/skip **by reason**, both test roots, `slow` included — the milestone's first deliverable
- Ranked per-module gap report (`term-missing` + machine-readable artifact) — the test-authoring worklist
- Bug fixes with unskips: multiclass AUROC (`dnallm/tasks/metrics.py:283`, skipped at `tests/tasks/test_metrics.py:761`) and CrossDNA handler overwrite (`dnallm/models/model.py:873-887`)
- Tests to >90% under an observable-behavior assertion standard (values/shapes/keys/ranges/`pytest.raises(match=)`)
- CI gate over the full denominator (both roots, slow included) — from the pytest exit code
- Updated `tests/TESTING.md`/`CONTRIBUTING.md`: assertion rule, skip hygiene, marker policy

**Should have (P2 — decay prevention):** patch coverage on PRs (Codecov status or diff-cover); two-lane CI + HF cache (if gate runtime hurts); selective `@pytest.mark.flaky` on demonstrated network flakes; `--cov-context=test` during authoring; trend tracking.

**Defer (v3+):** quarantine lane with expiry; branch coverage as stage-2 metric; nightly drift report/ratchet automation; duration-budget policy.

### Architecture Approach

One canonical pytest invocation feeds a single measurement pipeline (pytest-cov → coverage data → combined report → `fail_under` → exit code → CI step); the existing two test roots map cleanly onto a test pyramid — L0 unit/mock (`tests/` with `tests/conftest.py` fixtures), L1 in-process integration (`dnallm/mcp/tests/`, CliRunner), L2 network/slow (`*_real_model.py`) — and each library layer has a designated primary tier (e.g. `models/` dispatch chain via L0 fault-injection on Mock loaders + L2 sample of real families; `mcp/` via L0 mocked-ModelManager tools + L1 real server methods; `cli/` — currently the classic zero-coverage layer — via CliRunner). Gap magnets are known in advance: import-time compat shims, the 12-handler × 35-family model-loading dispatch chain, MCP transports/streaming/timeout error paths, and CLI lazy imports.

**Major components:**
1. **Harness config** — `[tool.pytest.ini_options]` in pyproject.toml only; one invocation defines pass/fail, coverage, and gate
2. **Session lifecycle conftest** — cleanup that propagates exit codes (currently the blocker)
3. **Coverage config** — the denominator contract: `source` (what counts) + `omit` (what doesn't), with a CI grep check asserting forbidden rows never reappear
4. **Gap report → authoring loop** — pick biggest uncovered module, write L0 tests, re-run scoped, next module
5. **Gate** — `fail_under` added LAST; measures the same command CI runs, reproducible locally with one command
6. **CI jobs** — existing fast matrix (`not slow`, un-gated) + one dedicated slow-inclusive gated coverage job (single py/numpy leg, HF cache, timeout backstop)

### Critical Pitfalls

1. **Exit-code masking (`os._exit(0)` atexit)** — fix in Phase 1 + permanent canary step; nothing downstream works without it (see reproduced defect 1)
2. **Wrong denominator** — unimportable adapters counted at 0% (~3.9 unreachable points), packaged test files counted in the numerator, no coverage config at all; a subtlety: vendored `tasks/metrics/` subdirs have no `__init__.py` (0 of 55), so their exclusion today is *accidental* — one direct import re-pollutes the denominator. Fix: config omit list + a "denominator contract" CI check that greps the report for forbidden rows
3. **Gate enabled too early** — a permanently-red gate gets bypassed ("temporarily" removed, pragma scatter, assertion-free tests) within weeks; ratchet from the measured baseline, gate last, police escape hatches (pragma baseline = 3, skip sites = 7 — record both in Phase 1)
4. **Flaky network tests inside the gate** — HF 429s/DNS hiccups fail the gate for non-code reasons, and skip-on-network-error storms make the percentage flap ±0.5pp run-to-run; mitigation: single dedicated gate job, HF cache + optional `HF_TOKEN`, typed skips (catch specific network exceptions, fail on everything else) with an expected-skip allowlist, one automatic job retry that never auto-passes
5. **Coverage theater** — assertion-free/mock-only tests that execute without verifying; every new test needs at least one observable-outcome assertion (call-count assertions are fine *in addition*, never instead); for fallback chains, assert *which* fallback was selected; spot-check with mutation on `utils/sequence.py`/`tasks/metrics.py`
6. **Timeout budget & method** — global `--timeout=300` was sized for mocks; cold downloads can exceed it (per-test `@pytest.mark.timeout(1800)` marks, not a raised global); `signal` method preserves coverage data, `thread` destroys it; job-level `timeout-minutes` backstop for non-main-thread hangs SIGALRM can't reach
7. **Subprocess coverage silently missing** — pytest-cov 7.0 removed `.pth` subprocess support; child-process lines (Trainer dataloader workers) are invisible without `patch = ["subprocess"]`; make the scope decision explicitly in Phase 1 with a canary, don't discover it as phantom missing lines in Phase 3

## Implications for Roadmap

Four phases. All four research files independently converge on this ordering (measurement before improvement; bug fixes before tests that would lock bugs in; tests before gates; gate last so it never blocks mid-program).

### Phase 1: Harness Integrity & Measured Baseline
**Rationale:** Everything downstream is meaningless until the measured test set and exit semantics are deterministic. The two reproduced defects live here, and these are the cheapest fixes in the whole program.
**Delivers:** Exit-code fix in root `conftest.py` (`pytest_sessionfinish` + status propagation) + permanent failing-run canary; `tests/pytest.ini` deleted and CI collecting both test roots; `[tool.coverage.run]/[report]` in pyproject.toml (source, omit, `show_missing` — **no `fail_under` yet**); full-suite audit census (pass/fail/skip by reason, slow included, both roots); ranked per-module gap report as the working backlog; measured baseline number; slow-test wall-clock timings (cold and warm); pragma/skip baselines; denominator-contract check; subprocess-scope decision.
**Addresses:** Table-stakes config, audit census, gap report (FEATURES P1).
**Avoids:** Pitfalls 1, 2, 6 (decision), 5 (timing data), 3 (baseline recorded).
**Note:** Test-floor bumps (`pytest-cov>=7.0`, `pytest-asyncio>=1.0`, `pytest-timeout>=2.3.1,<2.5`, `coverage[toml]>=7.10.6`) land here with the config work.

### Phase 2: Suite Hygiene & Known-Bug Fixes
**Rationale:** The audit confirms scope; the fixes change control flow — tests written before the CrossDNA fix would enshrine the broken path. A skipped-because-it-crashes test is a known defect wearing a disguise.
**Delivers:** Multiclass AUROC fix + unskip; CrossDNA handler-overwrite fix + test; typed network skips replacing broad `except Exception: pytest.skip`; PDF test artifacts pointed at `tmp_path` + the `.gitignore` typo fixed (`test/inference/pdf/` → `tests/inference/pdf/`); skip-reason allowlist enforcement.
**Addresses:** Bug-fix table stakes; skip hygiene (FEATURES P1).
**Avoids:** Pitfall 4 (hygiene half), 7 (routing around bugs), maintainer-UX pitfalls (dirty working tree — `tests/inference/pdf/` is already showing up untracked in git status).

### Phase 3: Test Authoring to >90% (waves, biggest-denominator first)
**Rationale:** The Phase 1 gap report orders the work by module size × gap; L0 mock tests deliver the most branch-reach per CI second, with L1/L2 only where mocks cannot reach.
**Delivers:** Waves in order: `models/model.py` + `special/*` (dispatch-chain fault-injection, retry/reason-classification branches, tokenizer fallback chain) → `mcp/server.py` (transports, streaming generators, timeout-wrapper error paths) → `inference/*` → `datahandling`/`finetune` → `cli/` + utils compat shims. Assertion standard enforced per test; `transformers_compat` tested as behavior contract (idempotent `apply_patches`, nested-quantizer tolerance) not line completion; pragma budget held.
**Addresses:** "Tests to >90%" — the bulk of the milestone's effort (FEATURES P1).
**Avoids:** Pitfalls 7, 8; anti-features (assertion-free tests, deep mocking, pragma abuse, refactors-for-testability).
**Note for roadmapper:** This is the only phase whose size is unknown until Phase 1's baseline lands (distance to 90% unmeasured). Consider splitting into two phases (e.g., Wave A: models + inference; Wave B: mcp + datahandling/finetune + cli/shims) — but preserve the wave order and keep them after bug fixes.

### Phase 4: CI Gate Enforcement (LAST)
**Rationale:** A gate is a lock, not a generator, of coverage. The gate's first blocking day must be its first green day; enabling it earlier trains bypass behavior.
**Delivers:** `fail_under = 90` in config (optionally landed at the measured baseline first and ratcheted up via a checked-in threshold file that only moves up); dedicated single-leg coverage job (py3.12, full suite incl. slow) separate from the untouched fast matrix; HF model cache keyed on `models.lock` + warm-up step option; per-test timeout marks + job-level `timeout-minutes`; codecov `@v3 → @v7` or dropped, informational only; **synthetic-regression proof that the gate actually fails CI** (exercises the Phase 1 exit-code fix end to end); gate runs on PRs to every protected branch including `dev`, not just `main`.
**Addresses:** CI gate table stakes + P2 decay prevention (two-lane CI, caching) (FEATURES P1/P2).
**Avoids:** Pitfalls 3, 4 (CI half), 5 (marks/backstop), 9 (matrix-gating), 10 (Codecov as gate).

### Phase Ordering Rationale

- **Harness before measurement:** a denominator is only meaningful if the collected test set is deterministic — CI currently measures a different suite than the docs describe, with different semantics (no timeout, STRICT asyncio, one root missing)
- **Exit-code fix before anything:** the audit's own pass/fail counts are unreliable until it lands; it is also why the current `deploy`-gating on `test` is an illusion today
- **Bug fixes before test authoring:** the CrossDNA fix changes control flow in the largest module; tests written first would enshrine the broken path and need rewriting
- **Tests before gate:** flipping `fail_under` early produces a permanently-red gate — the documented fast path to a bypassed, meaningless gate
- **Denominator contract spans all phases:** the CI grep check from Phase 1 mechanically prevents both accidental re-inclusion and numerator inflation in every later phase

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (specifically the MCP wave):** transport-level testing of `server.py:1718+` against the pinned `mcp>=1.3.0,<2` SDK — the in-memory client-server pattern from FastMCP docs could not be verified against the installed SDK (LOW confidence); also the `sys.modules` patch + `importlib.reload` pattern for import-time compat shims is community practice, not officially documented (MEDIUM). Run `/gsd-plan-phase --research-phase` for this phase, or at minimum for the mcp wave.
- **Phase 3 sizing:** the baseline number is unknown until Phase 1 executes — plan Phase 3 after Phase 1's gap report exists, and expect the roadmapper to revisit wave granularity then.

Phases with standard patterns (skip research-phase):
- **Phase 1:** prescriptive — STACK.md contains the exact working config and commands, empirically verified; remaining work is repo-specific execution, not knowledge gaps
- **Phase 2:** bugs already located and characterized (file:line given for both)
- **Phase 4:** the CI job pattern is fully specified in STACK.md (YAML sketch included); codecov action major verified v7.1.1 as of 2026-09

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions/config semantics verified against official docs of the exact installed toolchain AND empirical local runs (`fail_under` from config with bare `--cov` confirmed; floor/pin requirements registry-checked) |
| Features | HIGH (project-grounded) / MEDIUM (ecosystem) | P1 list derives from PROJECT.md requirements + direct repo reads (pytest.ini, ci.yml); P2/P3 ecosystem patterns cross-checked across independent sources |
| Architecture | HIGH | Both critical harness findings reproduced in-repo 2026-09-29 with commands recorded; layer/tier mapping read from the actual suite |
| Pitfalls | HIGH | Top findings reproduced empirically (exit-code experiment, denominator rows, 325 dead statements, suite counts: 16 slow, 7 skips, 3 pragmas); tool behavior docs-verified |

**Overall confidence:** HIGH — unusually strong for research synthesis because the researchers ran experiments on this repository rather than relying on web sources alone. The two load-bearing facts (exit-code masking, config shadowing) are not predictions; they are reproduced observations.

### Gaps to Address

- **Subprocess coverage scope (unresolved config conflict):** STACK recommends starting minimal (no `parallel`/`patch`/`concurrency` — escalate on evidence, in the order concurrency → `patch=subprocess` → `patch=_exit`); ARCHITECTURE's Pattern 3 sketch sets `parallel = true` + `patch = ["subprocess"]` + `concurrency = ["multiprocessing", "thread"]` preemptively. Both cite official docs. **Resolution: start minimal (STACK), let a Phase 1 subprocess canary decide**, escalate only on evidence — coverage 7.16's stricter option checks make preemptive enabling costly.
- **Baseline number unknown:** the distance to 90% on the agreed denominator is unmeasured until Phase 1 runs; Phase 3 sizing (and the split-vs-single-phase decision) depends on it.
- **MCP transport test pattern:** unverified against the installed `mcp` SDK (LOW confidence) — the Phase 3 research flag above.
- **Codecov action major:** verified v7.1.1 as of 2026-09 (FEATURES' "v5" recollection is superseded); reconfirm at Phase 4 implementation since the marketplace moves.
- **Skip-storm magnitude:** how often the 7 skip-on-network sites actually fire (warm vs cold cache) is unknown — the Phase 1 audit must count skips by reason across two runs; a skip storm silently shrinks the exercised denominator.
- **`--disable-warnings` from `tests/pytest.ini`:** deliberate decision required — do NOT port it into pyproject (it masks skip warnings); record as a decision so it doesn't get cargo-culted back.
- **HF cache hit-rate / gate runtime:** estimates only until the first slow-inclusive CI run; the two-lane split and any xdist adoption are measured decisions, not defaults.

## Sources

### Primary (HIGH confidence — local reproduction + official docs of installed versions)
- In-repo reproduction, 2026-09-29 (exit-code masking: expected 5 → got 0; `pytest tests/` configfile = `tests/pytest.ini`; `--cov=dnallm` denominator rows: `megatron.py` 184 stmts 0%, `mamba_npu.py` 141 stmts 0%, `mcp/tests/*` present, TOTAL 8,344 stmts; vendored metrics dirs 0 `__init__.py` in 55; suite counts 464 tests / 16 slow / 7 skip sites / 3 pragmas; `fail_under`-from-config and bare-`--cov`+`source` verification)
- coverage.py official docs — config reference, changelog (7.16.2 current; `patch` options since 7.10), subprocess support pages — https://coverage.readthedocs.io
- pytest-cov official docs + changelog — 7.x, `.pth` removal in 7.0.0, `--cov-context`, xdist support — https://pytest-cov.readthedocs.io
- pytest official docs — rootdir/configfile discovery (first-match-wins, no merge), changelog 9.0/9.1 (`--strict-*` addopts fix, `[tool.pytest]` exclusivity) — https://docs.pytest.org
- pytest-timeout README (2.4.0) — method semantics, priority order (ini < env < flag < marker)
- PyPI registry metadata — pytest 9.1.1, pytest-asyncio 1.4.0 (`pytest>=8.4,<10`), pytest-timeout 2.4.0 (2.5.0 yanked), pytest-xdist 3.8.0
- Project files (direct reads) — `.planning/PROJECT.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/CONCERNS.md`, `pyproject.toml`, `tests/pytest.ini`, `conftest.py`, `.github/workflows/ci.yml`, `dnallm/utils/transformers_compat.py`

### Secondary (MEDIUM confidence)
- codecov-action releases (v7.1.1 current; v3 dead on Node-24 runners) — https://github.com/codecov/codecov-action/releases
- HF Hub caching/env docs + community `actions/cache` patterns — `HF_HOME`, `HF_HUB_OFFLINE`, 10 GB repo cache limit
- HF transformers testing conventions — `@slow` default-deselected, tiny-model fixtures, `is_flaky` retry — https://huggingface.co/docs/transformers/en/testing
- Coverage-gate ratchet practice — SonarSource community, Azure/missionlz, kokil.com.np (multiple independent sources agree)
- Coverage anti-pattern literature — Optivem, Codecov blog, eyas.sh, jasonrudolph.com (100% mandates, assertion-free tests)
- pytest-rerunfailures docs; flaky-quarantine consensus (trunk.io, buildpulse.io, harness.io); diff-cover (PyPI + Marketplace action)

### Tertiary (LOW confidence — validate during implementation)
- MCP in-memory client-server testing pattern vs pinned `mcp>=1.3.0,<2` (v2-rewrite README mismatch) — Phase 3 research flag
- `sys.modules` patch + `importlib.reload` for import-time shim coverage — community practice
- CI coverage-tracer overhead estimate (20–30% on torch-heavy tests) — web-only
- CI wall-clock / 429 flake-rate forecasts — unmeasured until first gated run

---
*Research completed: 2026-09-29*
*Ready for roadmap: yes*
