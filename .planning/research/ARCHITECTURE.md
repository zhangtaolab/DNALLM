# Architecture Research

**Domain:** Test-coverage hardening architecture for a layered Python ML library (DNALLM: config → models → datahandling → finetune → inference → MCP)
**Researched:** 2026-09-29
**Confidence:** HIGH for harness/data-flow findings (reproduced locally against the repo + official docs); MEDIUM for ecosystem conventions (HF test-suite patterns, ratchet gating); LOW for a few strategy details noted inline.

## Standard Architecture

A coverage-hardening program on a layered library is itself a small system: **test layers** (which exercise code layers) feed a **measurement pipeline** (pytest-cov → coverage.py data → combined report), which feeds a **gate** (fail_under → exit code), which feeds **CI** (step success/failure). Each boundary can silently lie. Two lies were found in DNALLM's current wiring and are reproduced below — they must be fixed before any baseline number is trusted.

### System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│  TEST LAYERS (pyramid) — which tests exercise which code layers           │
│                                                                           │
│  L2  network/slow  (@pytest.mark.slow, real HF/ModelScope downloads)      │
│      → model-loading dispatch chain end-to-end, download retry, presets   │
│  L1  integration in-process (real classes, mocked transport/network)      │
│      → MCP server tools, config manager/validators, trainer wiring, CLI   │
│  L0  unit/mock (default fast run: Mock/patch + shared conftest fixtures)  │
│      → config, utils+compat shims, metrics, datahandling, inference       │
│        engine, model.py branch logic, MCP tools with mocked ModelManager  │
├───────────────────────────────────────────────────────────────────────────┤
│  MEASUREMENT PIPELINE                                                     │
│  single canonical pytest invocation (both test roots)                     │
│    → pytest-cov plugin (coverage engine starts before collection)         │
│    → [tool.coverage.run] source_pkgs=dnallm, omit=vendored+unimportable   │
│    → subprocess shards: .coverage.<host>.<pid>.<rand>  (if parallel)      │
│    → coverage combine → .coverage                                         │
├───────────────────────────────────────────────────────────────────────────┤
│  REPORTING                        ┌─ GATE                                 │
│  term-missing (per-module gaps)   │  [tool.coverage.report] fail_under    │
│  coverage.xml → Codecov (info)    │  → exit code 2 on miss                │
│  html (local authoring)           │  → GHA step fails (hard gate)         │
│                                   │  ⚠ exit code must survive conftest    │
│                                   │    atexit — currently masked to 0     │
├───────────────────────────────────────────────────────────────────────────┤
│  CI: fast PR job (not slow, no gate) + coverage job (slow incl., gated)   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation (DNALLM mapping) |
|-----------|----------------|----------------------------------------|
| Test harness config | One source of truth for markers, asyncio mode, timeout, testpaths | `[tool.pytest.ini_options]` in pyproject.toml only — delete `tests/pytest.ini` |
| Session lifecycle conftest | Cleanup of multiprocessing/CUDA; must propagate exit codes | Root `conftest.py` — currently breaks the gate (see Anti-Patterns) |
| Unit/mock layer (L0) | Fast, deterministic, CPU-only branch coverage of every layer except real network/loading | `tests/` with `unittest.mock` + `tests/conftest.py` fixtures (`mock_model`, `mock_tokenizer`, …) |
| Integration layer (L1) | Real classes wired together in-process; only transport/network mocked | `dnallm/mcp/tests/` (real server methods), `tests/mcp/`, CliRunner for CLI |
| Slow/network layer (L2) | Real checkpoints through the dispatch chain; download/retry/fallback paths | `*_real_model.py` files, `@pytest.mark.slow`; HF Hub cache in CI |
| Coverage config | Denominator policy: what counts (source_pkgs) and what doesn't (omit) | `[tool.coverage.run]` in pyproject.toml — **does not exist yet** |
| Combiner | Merge parallel/subprocess shards before reporting | `coverage combine` (auto-invoked by pytest-cov when parallel=true) |
| Gap report | Per-module missing-lines list driving the test-writing backlog | `--cov-report=term-missing` |
| Gate | Fail CI below threshold | `[tool.coverage.report] fail_under = 90` (exit code 2), added LAST |
| CI jobs | Fast feedback (PR) separate from expensive gated measurement (merge/nightly) | `.github/workflows/ci.yml` |

## Recommended Project Structure

Keep the existing two test roots; they already map cleanly onto pyramid layers. Additions in brackets.

```
pyproject.toml                      # [tool.pytest.ini_options] (single config)
                                    # [tool.coverage.run]/[tool.coverage.report] (NEW)
conftest.py                         # root: cleanup ONLY, exit-code-preserving (FIX)
tests/
├── conftest.py                     # shared mock fixtures (extend, don't re-roll)
├── cli/                            # (NEW) CliRunner tests for dnallm/cli/*
├── configuration/ datahandling/ models/ inference/ finetune/ tasks/ utils/
│                                    # L0 unit/mock tests mirroring package layout
│                                    # + *_real_model.py for L2 (existing pattern)
└── mcp/                            # L0/L1 MCP tests with mocked ModelManager
dnallm/mcp/tests/                   # L1 in-process server integration (real methods)
                                     # + (NEW) transport-level tests (sse/streamable-http
                                     #   startup paths in server.py:1718+)
```

### Structure Rationale

- **tests/ vs dnallm/mcp/tests/:** keep both — the packaged suite is the natural home of in-process server integration tests — but CI must collect both. Today it collects only `tests/`.
- **Mirror-the-package layout:** per-module `term-missing` output maps 1:1 onto test directories, so gap triage is mechanical.
- **`*_real_model.py` suffix:** the existing convention already encodes "L2 network tier"; keep it rather than inventing a new marker.

## Architectural Patterns

### Pattern 1: Test-pyramid mapping to library layers

**What:** Each code layer is covered primarily by one test tier, with cheaper tiers doing most of the work.
**When to use:** Always — the pyramid is what keeps a 90% target affordable on a 35-family model registry.

| Code layer | Primary tier | Secondary tier | Notes |
|-----------|-------------|----------------|-------|
| `configuration/` | L0 real Pydantic objects | — | Never mock validators; construct real configs (existing convention) |
| `utils/` incl. compat shims | L0 with mocked patch targets | matrix job (transformers 4.49/5.x) | See Pattern 4 |
| `models/model.py` dispatch chain + `special/*` | L0 fault-injection on Mock loaders | L2 real checkpoints for a sample of families | Biggest denominator; known bug (CrossDNA overwrite) hid here |
| `datahandling/` | L0 local CSV fixtures (real tokenization) | L2 HF/ModelScope loaders | Local files are already well covered (46 tests) |
| `finetune/trainer.py` | L0 mocked Trainer/PEFT wiring | L2 tiny real training run | Optuna paths are optional-import guarded |
| `inference/` | L0 mock model/tokenizer engine | L2 real model | Engine is ~the largest file; mock-driven tests scale |
| `mcp/` | L0 tools with mocked ModelManager | L1 in-process real server methods (`dnallm/mcp/tests/`) | Async generators + timeout wrappers need explicit error-path tests |
| `cli/` | L1 CliRunner | — | Currently the classic zero-coverage layer |

**Trade-offs:** L0 maximizes branch reach per second of CI; it cannot prove the dispatch chain works with real weights — that's L2's job, and per the owner decision L2 (`slow`) is included in the gated measurement.

### Pattern 2: Single canonical invocation (harness unification)

**What:** Exactly one pytest command defines pass/fail, coverage, and the gate; every other invocation (docs, scripts) must be an alias of it.
**When to use:** Before taking the baseline. A denominator is only meaningful if the set of collected tests is deterministic.

Reproduced ground truth (2026-09-29, this repo):

- `pytest tests/` → configfile **`tests/pytest.ini`** (pytest searches upward from the args' common ancestor; first match wins; configs are never merged — official docs). Consequences vs pyproject.toml: no `--asyncio-mode=auto` (pytest-asyncio falls back to STRICT), no `--timeout=300`, marker set lacks `legacy`, **`dnallm/mcp/tests/` is never collected** (`testpaths` only applies when no path args are given).
- `pytest` (bare, repo root) → configfile `pyproject.toml`, both test roots, root `conftest.py` loaded.
- CI runs `pytest tests/ …` — i.e. the *first* column, minus the MCP packaged suite, minus timeout, in strict asyncio mode.

Current async exposure is latent, not active: `tests/mcp/` async tests pass under STRICT mode only because `@pytest.mark.asyncio` is applied at class level (verified: `TestDNAInterpretTool`, `TestDNAMutagenesisTool`). Any new async test relying on the *documented* auto-mode behavior would silently not run under the CI invocation.

**Fix:** delete `tests/pytest.ini` (fold its `--disable-warnings` decision into pyproject deliberately — do NOT keep it, it suppresses the skip warnings that would reveal silently-skipped tests), make CI invoke bare `pytest` or `pytest tests dnallm/mcp/tests`.

### Pattern 3: Coverage data flow with subprocess support

**What:** pytest-cov starts the coverage engine before test collection, so module-level code (compat shims) executed at first `import dnallm` during collection is measured in-process. Anything that executes in *child processes* is not, unless configured.

DNALLM-specific subprocess surfaces: HF Trainer dataloader workers (tokenization `map()` in `datahandling/`), and any test that shells out. With the installed `coverage 7.16.2` / `pytest-cov 7.1.0`:

```toml
[tool.coverage.run]
source_pkgs = ["dnallm"]
omit = [
  "dnallm/tasks/metrics/*",        # vendored HF evaluate
  "dnallm/models/special/enformer_model/*",  # ported Enformer
  "dnallm/finetune/megatron.py",   # unimportable in CI (Megatron-LM)
  "dnallm/models/special/mamba_npu.py",      # unimportable in CI (Ascend NPU)
]
concurrency = ["multiprocessing", "thread"]  # else "very wrong results" (official docs)
sigterm = true                      # save data when workers are SIGTERM'd
parallel = true                     # .coverage.<host>.<pid>.<rand> shards, auto-combined
patch = ["subprocess"]              # coverage>=7.10; pytest-cov 7 removed its .pth mechanism

[tool.coverage.report]
show_missing = true
# fail_under = 90  ← added LAST, after the number is real (build order, below)
```

**Trade-offs:** `parallel = true` + combine is harmless for single-process runs and required for workers; `branch = true` deliberately NOT set (owner decision is line coverage; switching mid-program changes the denominator). Keep `relative_files = true` if Codecov-reported paths differ between install modes — verify at baseline.

### Pattern 4: Testing the three known gap magnets

**What:** In layered ML libraries, gaps concentrate exactly where DNALLM's architecture doc predicts:

1. **Import-time compat shims** (`dnallm/utils/transformers_compat.py`, `cuda_compat.py`). Module bodies execute once at first import and are cached in `sys.modules`; version-conditional branches (transformers 4.49 vs 5.x) cannot all execute on one installed version. Strategy: `unittest.mock.patch.dict(sys.modules)` + `importlib.reload` with mocked patch targets for the no-op/absent-library paths (MEDIUM confidence — community practice, not officially documented); transformers-version matrix is already a CI constraint, so per-version branch coverage comes from the matrix, not from tricks; residual unreachable branches get `exclude_also` (keeps `pragma: no cover` defaults — do not use `exclude_lines`, which replaces them) or honest targeted excludes.
2. **Model-loading dispatch chain** (`model.py:736-910` + `special/*`). Chain-of-responsibility with 12 handlers × 35 families: L0 covers each handler with Mock downloaders/Auto classes (fault-injection for retry/reason-classification branches at `model.py:317-377` and the tokenizer fallback chain at `543-568` — patch the collaborator to raise); L2 covers a sample of real families. Device branches (MPS/XPU) are unreachable in CI → mock-driven tests or exclusion. The CrossDNA bug (`model.py:873-887`, handler result overwritten) is precisely the class of defect that gap-hiding produces — fix before writing tests that would enshrine the broken path.
3. **MCP async layer** (`server.py`, ~1900 lines). Tools are covered in-process (L1) but transports (uvicorn/Starlette mounts, `server.py:1718+`), streaming generators, timeout-wrapper error paths, and client fallbacks are the typical holes. Async generators are exercised by iterating them (`async for` over the stream result); timeout paths by patching `ModelManager` methods with `AsyncMock(side_effect=slow)`. In-memory client-server pairs (no spawned process) are the standard FastMCP pattern but were NOT verifiable in the fetched SDK README (v2 rewrite) against the pinned `mcp>=1.3.0,<2` — verify against the installed SDK before relying on it (LOW confidence).

Plus the classic fourth: **CLI entry points** (`dnallm/cli/*`) — `click.testing.CliRunner` for the Click group, direct `main()` calls with argv for argparse mains. Lazy imports inside command bodies mean coverage requires actually invoking each command path.

### Pattern 5: Gate-last with ratchet semantics

**What:** Introduce `fail_under` only once the measured number is ≥ target, then treat it as a one-way ratchet (community consensus, MEDIUM): never set an aspirational unmet threshold — it gets bypassed; never let the number regress.

For DNALLM the owner has already chosen the endpoint (`--cov-fail-under=90` on a slow-inclusive run), so the ratchet is only the transition mechanism: optionally land the gate at the measured baseline first, raise to 90 when reached. The gate must measure the *same command CI runs* (Pattern 2).

## Data Flow

### Coverage data flow (direction: tests → data → report → gate → CI)

```
pytest (single canonical invocation, both roots, slow included in coverage job)
   │  pytest-cov engine starts pre-collection → import-time shim code counted in-process
   ▼
.coverage  (+ .coverage.<host>.<pid>.<rand> shards from dataloader workers / subprocesses)
   │  coverage combine   (pytest-cov does this automatically when parallel=true)
   ▼
combined .coverage
   ├─→ term-missing  → per-module gap list → test-writing backlog (authoring feedback loop)
   ├─→ coverage.xml  → Codecov (trend, PR comments; informational, fail_ci_if_error stays false)
   ├─→ html/         → local authoring view
   └─→ fail_under check → exit code 2 → GHA step exit status → red X on the commit
          ⚠ BLOCKED TODAY: root conftest.py atexit handler calls os._exit(0)
            unconditionally — reproduced: a run that should exit 5 exited 0.
            Every failure exit code (pytest 1, coverage 2) is masked whenever
            the root conftest loads (bare pytest). The gate cannot work until
            this propagates the real exit status (os._exit(code)) or is removed.
```

### Authoring feedback loop

```
gap report (term-missing) → pick biggest uncovered module → write L0 tests
   → re-run scoped: pytest tests/models --cov=dnallm.models --cov-report=term-missing
   → module turns green → next module (fast inner loop; full-suite run only per wave)
```

### State Management

The coverage program adds no product state. One caution: `tests/pytest.ini` and root `conftest.py` are process-global configuration — changing them changes every developer's local run simultaneously, so harness unification must land as one atomic, communicated change.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Today (464 tests, fast job ~minutes) | Single job; sequential is fine; slow tier only in gated coverage job |
| ~2× tests after gap-closing | Keep fast PR job un-gated and `not slow` for feedback latency; gate runs on the coverage job only. `pytest-xdist` (`-n auto`) is compatible with pytest-cov (workers must have pytest-cov installed — it is a dev dep) if wall-clock grows |
| Slow tier with real downloads | Cache `~/.cache/huggingface` via actions/cache keyed on hash of the slow-test model list; expect 429/rate-limit flakiness → retry-once wrapper or `is_flaky`-style retry decorator (HF convention, MEDIUM) |

### Scaling Priorities

1. **First bottleneck: CI wall-clock on the gated slow-inclusive run.** Fix with HF Hub caching + keeping the fast job separate; parallelize with xdist only if needed.
2. **Second bottleneck: network flakiness corrupting the gate.** A flaky download failing a gated run is a false alarm; retries/caching are the mitigation, not loosening the gate.

## Anti-Patterns

### Anti-Pattern 1: Split pytest configs (present in this repo — reproduced)

**What people do:** A `pytest.ini` inside `tests/` alongside `[tool.pytest.ini_options]` in pyproject.toml, with CI passing `tests/` as an argument.
**Why it's wrong:** First match wins, configs never merge. CI silently loses the second test root, asyncio auto-mode, timeout, and a registered marker. The documented run commands and the CI commands measure different test suites.
**Do this instead:** Exactly one config (pyproject.toml). Delete `tests/pytest.ini`. CI invokes the canonical command that collects both roots.

### Anti-Pattern 2: Exit-code-hostile session cleanup (present in this repo — reproduced)

**What people do:** `atexit.register(fn)` where `fn` ends in unconditional `os._exit(0)` to kill stray workers.
**Why it's wrong:** It converts every failure exit code (pytest 1, coverage gate 2, "no tests collected" 5) into success (verified: expected 5, got 0). A CI gate wired on top of it can never fail — the hardest possible failure mode because everything looks green.
**Do this instead:** Propagate the real status: capture pytest's `exitstatus` in `pytest_sessionfinish` and `os._exit(exitstatus)` from the atexit handler (or drop `os._exit` and terminate children without overriding the code). Add a regression check that a deliberately failing gated run actually fails CI.

### Anti-Pattern 3: Assert-free tests that execute code

**What people do:** To close `term-missing` lines fast, write tests that call a function and assert nothing (or assert only "did not raise").
**Why it's wrong:** Coverage measures execution, not verification; the gate then certifies nothing. The known skipped-crash (multiclass AUROC at `tasks/metrics.py:283`, skipped at `tests/tasks/test_metrics.py:761`) shows how assertion-free/skipped tests hide real defects.
**Do this instead:** Every new test asserts on observable behavior (values, keys, ranges, raised exceptions). Un-skip the AUROC test by fixing the bug, not by deleting the test.

### Anti-Pattern 4: Excluding hard modules instead of testing them

**What people do:** Grow the `omit` list whenever a module resists testing until the threshold is trivially reachable.
**Why it's wrong:** Omit policy is a denominator decision. The agreed excludes (vendored `tasks/metrics/`, `enformer_model/`, unimportable `megatron.py`/`mamba_npu.py`) are principled; anything else is gaming.
**Do this instead:** Omit only vendored/unimportable code (already decided in PROJECT.md). Device-unreachable or version-unreachable branches get surgical `exclude_also` patterns with comments, never file-level omits.

### Anti-Pattern 5: Measuring one command, gating another

**What people do:** Gate on the slow-inclusive coverage job while the fast job is what developers iterate on; or gate in Codecov while a different pytest command produces the artifact.
**Why it's wrong:** Coverage differs between `not slow` and full runs; a gate referencing a number produced by a different invocation is unfalsifiable locally.
**Do this instead:** The gate reads the combined data produced by the canonical invocation; developers can reproduce the gate number locally with the same one command.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GitHub Actions | Two-tier: fast job (`not slow`, lint, advisory mypy) + coverage job (slow-inclusive, `--cov-fail-under`) | Matrix py3.11–3.13 × numpy must all pass the gate run or gate on one designated leg (decide at gate phase) |
| Codecov | Upload `coverage.xml` from the gated job; keep `fail_ci_if_error: false` — the hard gate is pytest-cov's exit code | `codecov-action@v3` is deprecated/sunset (MEDIUM — verify current action major at implementation); informational trending only |
| Hugging Face Hub | Slow tests download small public checkpoints | Cache `~/.cache/huggingface`; expect 429 flakiness; keep downloads small (HF policy: big downloads belong in slow tier) |
| ModelScope | Alternate model source for preset datasets/models | Network availability in CI runners is the flakiness source; retry pattern already exists in `model.py:317` |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| tests/ ↔ dnallm/mcp/tests/ | Disjoint test roots, single pytest invocation | Must be collected together post-unification; packaged suite must not import `tests/` fixtures |
| Coverage config ↔ pytest config | Both in pyproject.toml; pytest-cov reads coverage config implicitly | `--cov` flags on CLI and `[tool.coverage]` config combine; keep the denominator policy in config, not in CI command text |
| Gate ↔ session cleanup | Exit-code propagation across `atexit` | The masking bug lives exactly on this boundary |
| Fast job ↔ gated job | Same canonical command with/without `-m "not slow"` | Gate only on the slow-inclusive run per owner decision |

## Build Order (dependencies for roadmap phasing)

Strictly sequential where marked; the later steps consume the earlier ones' outputs.

1. **Harness unification (blocking prerequisite).** Single config, single invocation, both roots; fix root-conftest exit-code propagation. *Everything downstream is meaningless until the measured test set and exit semantics are deterministic.*
2. **Baseline + audit.** Add `[tool.coverage.run]` (source/omit), run full suite including `slow`, produce pass/fail/skip audit + per-module `term-missing` gap report. Depends on 1. Output: the gap backlog that orders step 4, plus honest skip accounting.
3. **Known-bug fixes + un-skips.** AUROC multiclass crash, CrossDNA handler overwrite. Depends on 2 (audit confirms scope); precedes 4 because these change control flow — tests written before the CrossDNA fix would enshrine the broken path.
4. **Test-writing waves, biggest-denominator first.** Order by (module size × gap): `models/model.py` + `special/*` → `mcp/server.py` → `inference/*` → `datahandling`/`finetune` → `cli` + utils shims. Within each module: L0 mock tests first, L1 integration second, L2 slow only where mocks cannot reach (real weights, real downloads). Depends on 2 (backlog) and 3 (control flow stable).
5. **Gate last.** Add `fail_under = 90` (optionally land at baseline first and ratchet), wire the slow-inclusive gated CI job, verify the gate actually fails CI on a synthetic regression (exercises the Anti-Pattern 2 fix end to end). Depends on 4 reaching ≥90.

**Phase-ordering rationale:** measurement before improvement (you cannot manage what the harness measures inconsistently), bug fixes before tests that would lock bugs in, tests before gates (a gate is only a lock, not a generator, of coverage), gate last so it never blocks development mid-program.

## Sources

- pytest docs, "Customize/Configuration" (rootdir + configfile discovery; first-match-wins; no merging) — **HIGH**, cross-verified by local reproduction in this repo.
- coverage.py config docs (`fail_under` exit 2, `exclude_also`, `concurrency`, `parallel`, `sigterm`, `[run] patch=subprocess` ≥7.10) — **MEDIUM** (official docs via fetch).
- pytest-cov README (`--cov-append` semantics, data file erased per run, pytest-cov 7 removed `.pth` subprocess support, xdist workers need the plugin) — **MEDIUM**.
- pytest-asyncio concepts (strict = marker-only, auto = auto-mark; class-level markers valid) — **HIGH** with local verification (STRICT mode + class-level markers reproduced).
- Hugging Face testing conventions (`@slow` default-deselected + `RUN_SLOW=1`, tiny-model fixtures, `is_flaky` retry, "no large downloads in default run" — [testing docs](https://huggingface.co/docs/transformers/en/testing), [issue #7250](https://github.com/huggingface/transformers/issues/7250)) — **MEDIUM** (via search aggregation).
- Coverage-gate ratchet practice ([SonarSource community](https://community.sonarsource.com/t/ratcheting-quality-gate-conditions-fail-quality-gate-if-coverage-decreases-from-last-analysis/15317), [Azure/missionlz #1295](https://github.com/Azure/missionlz/issues/1295), [kokil.com.np guide](https://kokil.com.np/blog/code-coverage-gates-in-ci)) — **MEDIUM** (community consensus, multiple sources).
- Gap-concentration strategies (sys.modules patching + reload for import-time code, ImportError-branch handling, CliRunner, in-memory MCP testing) — **LOW/MEDIUM** (sparse search results; partly training-knowledge synthesis — flagged for phase-level verification).
- Local repo verification (config shadowing, exit-code masking, class-level async markers, CI collection gap) — **HIGH** (reproduced 2026-09-29; commands in research-store digest `dnallm-arch-local-verification-gate`).

---
*Architecture research for: pytest coverage hardening of DNALLM (layered Python ML library)*
*Researched: 2026-09-29*
