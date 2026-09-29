# Feature Research

**Domain:** Test-suite audit & coverage hardening for an existing Python ML library (pytest + pytest-cov, 464 tests, slow/network tests, vendored code)
**Researched:** 2026-09-29
**Confidence:** HIGH for project-grounded items (read directly from repo); MEDIUM for ecosystem patterns (cross-checked across independent web sources); LOW where individually noted

"Users" of this program are the DNALLM maintainer, CI, and future contributors. A feature here is a *capability of the coverage-hardening program*, not a library feature.

## Feature Landscape

### Table Stakes (Users Expect These)

Without these, the 90% gate is dishonest, unenforceable, or unmaintainable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Authoritative coverage config in `pyproject.toml` (`[tool.coverage.run]` + `[tool.coverage.report]`: `source = ["dnallm"]`, `omit` for vendored dirs and unimportable adapters, `fail_under`) | No `[tool.coverage]` section exists today; the denominator must be pinned in config or every run measures something different. Whitelisting `source` is preferred over blacklisting everything else | LOW | Confirmed via repo read: zero coverage config exists. `omit`: `dnallm/tasks/metrics/*`, `dnallm/models/special/enformer_model/*`, `dnallm/finetune/megatron.py`, `dnallm/models/special/mamba_npu.py` (per PROJECT.md decision) |
| Single pytest config source of truth — retire `tests/pytest.ini` | The file exists on disk (contradicting `.planning/codebase/TESTING.md`, which calls references to it stale) with `testpaths = .` and its own marker list missing `legacy`. pytest's rootdir discovery can select it over `pyproject.toml` depending on invocation, silently dropping `--timeout=300`, `--asyncio-mode=auto`, and marker registrations — a config-shadowing hazard that makes any gate unreliable | LOW | Verified by direct read of `/home/forrest/Github/DNALLM/tests/pytest.ini`. Delete it or reduce it to a comment pointing at `pyproject.toml`; add a regression check that `pytest --co` from repo root picks up pyproject settings |
| Full-suite audit report (pass/fail/skip census, both test roots, `slow` included) | The milestone's first Active requirement. You cannot harden what you have not censused; skips and xfails hide exactly the defects (AUROC crash) this milestone exists to fix | MEDIUM | Long runtime + network model downloads. Produce machine-readable output (JUnit XML or `--json-report`-style) so the census is diffable across runs. Count skips *by reason* |
| Gap report that drives test-writing priorities: `--cov-report=term-missing` plus a machine-readable artifact (JSON/XML) ranked by missing lines per module | The milestone's explicit deliverable; without a ranked gap list, test authoring is random | LOW | `term-missing` is one flag; the ranking/sorting of modules by uncovered lines is a small script or Codecov file-level view. Store the baseline gap report in `.planning/` as the working checklist |
| Gate on the *agreed denominator*: CI job running `--cov-fail-under=90` over **both** testpaths **including** `slow` tests | Today's CI coverage run is `pytest tests/ -v -m "not slow" --cov=dnallm` — it excludes the entire `dnallm/mcp/tests` root and all slow tests, so the measured denominator differs from the decided one. A gate on a different denominator than the audit is a dishonest gate | MEDIUM (config LOW, CI runtime MEDIUM) | Verified in `.github/workflows/ci.yml:81`. Requires network + HF model cache (see Differentiators) to be practical. Keep the threshold in exactly one place (config `fail_under` or CLI flag) to avoid divergence |
| Fix real bugs blocking honest coverage: unskip and fix multiclass AUROC (`dnallm/tasks/metrics.py:283`, skipped at `tests/tasks/test_metrics.py:761`) and CrossDNA handler overwrite (`dnallm/models/model.py:873-887`) | A skipped-because-it-crashes test is a known defect wearing a disguise; coverage numbers that include the skip are dishonest | MEDIUM | In PROJECT.md Active list. Fix code, then unskip — never delete the test or weaken the assertion to make it pass |
| Skip hygiene policy: every `skip`/`skipif`/`xfail` carries a reason string; audit enumerates them; no bare `except: pass` in tests | Skips silently erode the effective suite; 464 tests minus an uncounted skip set is not 464 tests | LOW | Enforce via review convention + the audit census. `--strict-markers` is already on; consider `xfail_strict = true` so stale xfails fail when they start passing |
| Assertion standard for every new test: assert observable behavior (values, shapes, keys, ranges, `pytest.raises` with `match=`) | Coverage measures execution, not verification — the canonical coverage failure mode is suites full of "calls it, checks nothing" tests that hit 90% while verifying zero behavior | LOW (convention) | Document in `tests/TESTING.md` + `CONTRIBUTING.md`; reuse the existing `tests/conftest.py` mock fixtures and the canonical TestClass pattern from `.planning/codebase/TESTING.md`. Make "no assertion-free tests" an explicit review rule for this milestone |
| Working PR coverage feedback | The repo already uploads `coverage.xml` to Codecov on every CI run — contributors expect the PR comment to appear. Today it pins legacy `codecov-action@v3` with `fail_ci_if_error: false` (upload advisory-only) | LOW | Verified in `ci.yml:83-87`. Minimum bar: PRs show project + patch coverage. The wrapper major is outdated (current documented major is v5) — verify wrapper status when touching this (LOW confidence on exact v3 sunset details, not verified this session) |

### Differentiators (Competitive Advantage)

Not required for an honest gate, but they keep 90% *true over time* instead of decaying the week after the milestone closes.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Enforced diff-based (patch) coverage on PRs — "if you touch it, you must test it" | Project-wide 90% erodes through new code; patch coverage holds the line on every PR with a clear, achievable review standard. diff-cover runs entirely in CI from `coverage.xml` + git diff (no signup/token/external service, listed as a companion tool in coverage.py docs); Codecov patch coverage gives the same metric hosted with status checks and thresholds | LOW-MEDIUM | diff-cover PyPI + Diff Cover Action on GitHub Marketplace; Codecov equivalent documented in Codecov FAQ ("patch coverage = percentage of only the changed lines"). Since Codecov is already wired, enabling patch status there is cheapest; diff-cover is the no-SaaS fallback. Can gate at e.g. patch >= 80% without demanding 100% |
| Coverage trend tracking + ratchet | A single number proves nothing about direction; trends show drift early and let `fail_under` ratchet upward (90 -> 92 -> ...) instead of being renegotiated | LOW | Codecov already receives uploads — trends/charts are nearly free. Badge optional (shields.io endpoint or marketplace badge action; no third-party service needed). Alternative self-hosted: Codecov is open source (docker-compose) — overkill here |
| Two-lane CI with HF model caching | Gate must include `slow` tests (owner decision), but nobody will tolerate a 40-minute push-blocking run. Fast lane (`-m "not slow"`) on every push; gated full lane on main/nightly/label with `actions/cache` on `~/.cache/huggingface` keyed by lockfile | MEDIUM | Transforms the gate from aspirational to practical; caching also reduces network flake. The two lanes must use the *same coverage config*; slow lane uses `--cov-append` semantics or a single full invocation so data files combine correctly (pytest-cov plugin manages parallel/combine internally) |
| Flaky-test management: selective reruns + quarantine lane | Slow tests download models over the network — HF Hub hiccups *will* flake the gated run. pytest-rerunfailures `@pytest.mark.flaky(reruns=N)` (marker priority > CLI > config) with `--rerun-delay`; quarantine = registered marker, merge gate runs `-m "not quarantine"`, scheduled job runs only quarantined tests with an expiry policy | MEDIUM | Consensus pattern (pytest-rerunfailures docs; trunk.io/buildpulse/harness flaky-test guides): quarantine > skip > delete, rerun selectively (blanket `--reruns` masks real bugs). Start simpler: selective flaky markers on known network tests; full quarantine lane only if flakes persist |
| Per-test coverage contexts (`--cov-context=test`) | During hardening, "which tests already exercise this line?" answers make gap-closing dramatically faster; `coverage json` then shows contexts per missing line | LOW | Documented pytest-cov flag (`Dynamic contexts to use. 'test' for now`). Use ad hoc during the gap-closing phase; consider leaving off in CI (context data is larger) |
| Branch coverage as stage-2 metric (`branch = true` / `--cov-branch`) | Line coverage misses untaken branches — `if/else` counting one line covered while one path never runs. Raising the honesty bar beyond the milestone's line-coverage target | MEDIUM | Flip on *after* 90% line is green; expect the effective percentage to drop and set a separate, lower branch target. Don't gate on branch in the same phase as the line gate (confounding) |
| Test duration discipline (`--durations=25` in CI, documented slow-marker grant policy) | Slow tiers rot: everything gets marked slow, fast lane shrinks. Duration reports make marker drift visible | LOW | `--durations` is built into pytest. Add to CI output; review "new slow marker" like you'd review a dependency addition |
| Nightly full-gate + drift report on main | Catches coverage regressions that slip through PR lanes (e.g. flaky-skipped slow tests that stop running) | MEDIUM | Depends on two-lane CI. The nightly run is also the trend data source |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|----------|---------------|-----------------|-------------|
| 100% coverage mandate / ratchet to 100 | "90 is arbitrary, 100 is honest" | Documented perverse incentives (Optivem, Codecov's own blog, eyas.sh retrospective, jasonrudolph.com): teams game the metric with trivial tests, delete valuable defensive error handling as "untestable", and burn weeks on the long tail. Coverage measures *execution*, not *verification* — 100% executed-and-unverified is worth less than 90% executed-and-asserted | Hold 90% project-wide with assertion standards; enforce *patch* coverage on PRs so new code is well-covered without punishing the existing long tail |
| Assertion-free / smoke tests to close gaps ("call the function, assert nothing") | Fastest way to move the number | Zero regression protection while consuming the budget; the single most common way coverage programs produce worse-than-nothing suites | Every test asserts observable behavior; error paths via `pytest.raises(..., match=...)`; ranges/shapes/keys on numeric outputs (existing suite already models this — `assert 0 <= metrics["accuracy"] <= 1`) |
| Deep mocking that severs real behavior | Mock everything to keep tests fast and deterministic | Over-mocked tests verify mock configuration, not the system; they break on every refactor and pass through real bugs (tests the mock, not the code). Also hides exactly the integration defects this suite's `*_real_model.py` lane exists to catch | Mock at boundaries only (model/tokenizer/network/`time.sleep`), reusing `tests/conftest.py` fixtures; keep real Pydantic config objects, real CSV fixtures, real MCP client/server classes with only transport mocked — the suite's documented "What NOT to Mock" list is already correct; follow it |
| `# pragma: no cover` to erase hard lines | Quick denominator relief | Pragma abuse launders untested code as "uncovered-by-design"; within a milestone it is indistinguishable from cheating | Reserve pragma for genuinely unreachable/platform-specific code only (coverage.py docs guidance: never for "hard to test"); the decided `omit` list handles vendored/unimportable code as *policy*, visible in one config block, reviewable in one place |
| Excluding `slow` tests from the gated run to keep CI fast | Gate must be fast, right? | The gate then measures a different suite than the one shipped; slow tests exercise the real model-loading paths that give the number meaning. Owner has explicitly decided slow tests are IN | Two-lane CI + HF cache + scheduled full lane (see Differentiators) |
| Blanket `--reruns=N` on the whole suite | "Flakes fixed, CI green" | Masks real regressions as flakiness; every genuine failure gets N free passes | Selective `@pytest.mark.flaky` on demonstrated-flaky tests + quarantine lane with expiry; measure flake rates, fix root causes |
| Refactoring production code to make it testable | "We can't reach 90% without restructuring" | Out of scope per PROJECT.md (bug fixes only); refactors during a coverage push churn the diff under the measurement | Test the code as it is; record desired refactors as follow-up issues (e.g. `attn_implementation` hardening — "record, don't fix") |
| New test frameworks/plugins beyond pytest + pytest-cov | "Property testing / new runner would fix this" | Constraint in PROJECT.md: no new test frameworks; added plugins widen the Python 3.11-3.13 x numpy 1.26/2.2 matrix surface for marginal gain | pytest parametrize, existing fixtures, `--durations`, built-in markers. (pytest-rerunfailures, if adopted for flake mitigation, is the one defensible addition — flag it as a decision) |

## Feature Dependencies

```
[Coverage config: source/omit/fail_under]
    └──requires──> [Single pytest config source of truth (retire tests/pytest.ini)]

[Full-suite audit census] ──enables──> [Per-module gap report]
        │                                     │
        └──requires──> [Bug fixes: AUROC + CrossDNA]   gap report ──drives──> [Test authoring to >90%]
                                                                    │
[CI gate on full denominator] <──requires──────────────────────────┘
        ├──requires──> [Coverage config]  (threshold lives here)
        ├──benefits──> [HF model cache + two-lane CI]
        └──enables──> [Trend tracking + badge]

[PR coverage feedback] ──requires──> [coverage.xml from CI] ──enhances──> [Patch coverage enforcement]

[Flaky markers/quarantine] ──requires──> [Audit skip/flake census]
[Nightly drift report] ──requires──> [Two-lane CI + scheduled full run]
[--cov-context=test] ──enhances──> [Test authoring to >90%] (ad hoc, no CI dependency)
[Branch coverage gate] ──requires──> [Line gate green first]  (deliberately staged)
```

### Dependency Notes

- **CI gate requires coverage config + bug fixes:** the gate is meaningless until the denominator is pinned and the suite it measures actually passes; the AUROC fix must precede the gate (a skipped-crashing test inside a gated run is a red gate).
- **Config-shadowing fix precedes everything:** if `tests/pytest.ini` can hijack rootdir resolution, every downstream measurement is environment-dependent. Cheapest fix in the whole program; do it first.
- **Gap report enhances/depends on audit:** the census tells you which tests *run*; the gap report tells you what they *reach*. Both derive from the same full-suite invocation.
- **Patch coverage conflicts with nothing but requires a PR-lane coverage.xml:** today's fast-lane run already produces it — patch coverage from the fast lane is valid PR feedback even while the full gate lives on the slow lane.
- **Flaky management conflicts with gate strictness:** reruns and hard gates fight each other (a rerun-passed test counts as pass but hides instability). Resolve by policy: reruns allowed only on the slow/network lane, never to make a unit-lane failure green.
- **Branch coverage deliberately staged after line gate:** enabling both at once makes the initial target unattainable and the failure signal unattributable.

## MVP Definition

### Launch With (v1)

This *is* the milestone's Active requirement list — the minimum for an honest, enforced 90%.

- [ ] Single pytest config source of truth (retire/neutralize `tests/pytest.ini`) — config-shadowing makes everything else nondeterministic
- [ ] `[tool.coverage.run]`/`[tool.coverage.report]` in `pyproject.toml` with `source`, `omit` (vendored + unimportable), `fail_under = 90`, `show_missing = true` — pins the denominator
- [ ] Full-suite audit census (pass/fail/skip by reason, both test roots, `slow` included) — the milestone's first deliverable
- [ ] Per-module gap report (`term-missing` + JSON artifact, ranked) — the working checklist for test authoring
- [ ] Bug fixes: multiclass AUROC crash, CrossDNA overwrite; unskip their tests — honesty requirement
- [ ] New tests to >90% under the assertion standard — the bulk of the effort
- [ ] CI job enforcing `--cov-fail-under=90` over the full denominator — the gate
- [ ] Updated `tests/TESTING.md`/`CONTRIBUTING.md` conventions (assertion rule, skip hygiene, marker policy) — keeps 90% maintainable after the push

### Add After Validation (v1.x)

- [ ] PR patch-coverage enforcement (Codecov status or diff-cover) — trigger: gate green on main for a full week; prevents decay
- [ ] Two-lane CI + HF model cache — trigger: full-gate CI runtime actually hurting (measure first; cache before splitting if runtime is the only pain)
- [ ] Selective `@pytest.mark.flaky` on demonstrated network flakes — trigger: first flaky red gate
- [ ] `--cov-context=test` during any follow-on gap-closing — trigger: next round of test authoring
- [ ] Coverage badge + trend review habit — trigger: gate stable

### Future Consideration (v2+)

- [ ] Quarantine lane with scheduled quarantine-only run + expiry policy — only if flake volume justifies the machinery
- [ ] Branch coverage target (`branch = true`, separate threshold) — after line target is comfortably held
- [ ] Nightly drift report / coverage ratchet — when contributor count or PR rate makes PR-lane enforcement insufficient
- [ ] Test duration budget policy — when the fast lane's runtime starts creeping

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Single pytest config (retire `tests/pytest.ini`) | HIGH | LOW | P1 |
| Coverage config (`source`/`omit`/`fail_under`) | HIGH | LOW | P1 |
| Full-suite audit census | HIGH | MEDIUM | P1 |
| Per-module gap report | HIGH | LOW | P1 |
| Bug fixes (AUROC, CrossDNA) + unskips | HIGH | MEDIUM | P1 |
| Tests to >90% w/ assertion standard | HIGH | HIGH | P1 |
| CI gate on full denominator | HIGH | MEDIUM | P1 |
| Assertion/skip-hygiene conventions documented | MEDIUM | LOW | P1 |
| Working PR coverage feedback (fix action version) | MEDIUM | LOW | P1/P2 |
| Patch coverage enforcement | HIGH | LOW-MEDIUM | P2 |
| Two-lane CI + HF cache | MEDIUM-HIGH | MEDIUM | P2 |
| Selective flaky reruns | MEDIUM | LOW | P2 |
| Trend tracking + badge | MEDIUM | LOW | P2 |
| `--cov-context=test` for authoring | MEDIUM | LOW | P2 |
| Nightly drift report / ratchet | MEDIUM | MEDIUM | P3 |
| Branch coverage stage 2 | MEDIUM | MEDIUM | P3 |
| Quarantine lane | LOW-MEDIUM | MEDIUM | P3 |
| Duration discipline reports | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for the milestone (an honest, enforced 90%)
- P2: Should have — decay prevention once the gate exists
- P3: Nice to have — operational maturity beyond this milestone

## Reference Program Analysis

How comparable programs handle this, and what DNALLM should take from each.

| Capability | HF transformers (tests tree verified via GitHub API) | Typical Codecov-gated OSS Python lib | DNALLM plan |
|-----------|--------------------|--------------------|-------------|
| Suite organization | `tests/models/<family>/` inheriting shared tester base classes (`test_modeling_common.py`, `test_configuration_common.py`, ...) so every model family gets uniform standard-behavior coverage; domain dirs (`pipelines`, `quantization`, `integrations`) | `tests/` mirroring package layout | Already mirrors package layout; keep, and consider common-tester inheritance only if model-family tests show drift (out of scope this milestone) |
| Slow-tier gating | Slow tests opt-in (marker + env/CLI gate); CI selects suites via parameterized jobs; per-model files runnable directly | Usually `-m "not slow"` fast lane + nightly full | Two-lane CI; gate lane includes slow per owner decision |
| Coverage % gate | Not enforced repo-wide — relies on breadth of parameterized common tests | `fail_under` + Codecov project/patch statuses | Enforce 90% line on agreed denominator — *stricter* than transformers; justified by smaller codebase |
| PR feedback | N/A (no hosted patch gate) | Codecov PR comment: project + patch coverage | Keep Codecov (already uploading), verify wrapper version, enable patch status |
| Flake handling | Decorator-based flaky marking historically used | `pytest-rerunfailures` selective reruns | Selective flaky markers on network tests; quarantine later if needed |
| Vendored code | N/A | `omit` in coverage config — standard practice for `*/vendor/*` | `omit` for `dnallm/tasks/metrics/`, `enformer_model/`, unimportable adapters (matches lint/mypy exclusions) |

## Sources

Project-grounded (HIGH confidence, direct reads):
- `/home/forrest/Github/DNALLM/.planning/PROJECT.md` — requirements, decisions, scope
- `/home/forrest/Github/DNALLM/pyproject.toml` `[tool.pytest.ini_options]` (no `[tool.coverage*]` section exists)
- `/home/forrest/Github/DNALLM/tests/pytest.ini` — exists on disk; shadowing hazard
- `/home/forrest/Github/DNALLM/.github/workflows/ci.yml` — coverage command excludes `dnallm/mcp/tests` and slow tests; `codecov-action@v3`, `fail_ci_if_error: false`
- `/home/forrest/Github/DNALLM/.planning/codebase/TESTING.md` — suite anatomy, mock policy, markers

Ecosystem (MEDIUM confidence, cross-checked across multiple independent sources):
- coverage.py docs — excluding code (`pragma: no cover`, `exclude_lines`, omit) and config reference: https://coverage.readthedocs.io
- pytest-cov config docs — `--cov-fail-under`, `--cov-append`, plugin overrides `parallel` option, `--cov-config` subprocess caveat, `--cov-context`, `--cov-reset`: https://pytest-cov.readthedocs.io/en/latest/config.html
- diff-cover (PyPI) + Diff Cover Action (GitHub Marketplace, "no signup, no token, no external service"); coverage.py docs list diff-cover as companion: https://pypi.org/project/diff_cover , https://github.com/marketplace/actions/diff-cover-action
- Codecov FAQ (patch coverage definition) and PR-comments docs: https://docs.codecov.com/docs/frequently-asked-questions , https://docs.codecov.com/docs/pull-request-comments
- pytest-rerunfailures (pytest-dev) — marker/CLI/config priority, reruns/delay: https://github.com/pytest-dev/pytest-rerunfailures
- Flaky quarantine consensus — trunk.io, buildpulse.io, harness.io flaky-test guides (quarantine > skip > delete; selective retries)
- Coverage anti-patterns — Optivem journal "Code Coverage Targets: Recipe for Disaster"; Codecov blog "The Case Against 100% Coverage"; blog.eyas.sh "Unexpected Lessons from 100% Test Coverage"; jasonrudolph.com "Testing anti-patterns: how to fail with 100% coverage"; testim.io
- Badge/trend alternatives — Codecov open-source self-hosting announcement (Aug 2023): https://about.codecov.io ; coverallsapp/github-action; marketplace badge actions
- HF transformers `tests/` taxonomy — verified via GitHub Contents API (tests/models, tests/pipelines, `test_modeling_common.py` et al., `conftest_tests/`); slow-gating details (RUN_SLOW-era mechanism) from long-standing HF contributing docs — treat exact current flag as MEDIUM/LOW, verify at implementation time

LOW confidence (flagged, not load-bearing):
- Exact `codecov-action@v3` sunset status and current recommended major (asserted v5 from memory, not verified this session) — verify when touching CI
- pytest-cov x sigterm/atexit interaction with DNALLM's root-conftest `os._exit(0)` cleanup — plausible data-loss hazard (coverage writes at process exit), needs a 10-minute empirical check in Phase 1, not web research

---
*Feature research for: pytest coverage hardening of a Python ML library*
*Researched: 2026-09-29*
