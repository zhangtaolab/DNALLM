# Pitfalls Research

**Domain:** Coverage hardening (>90% mandate + CI hard gate) of an existing Python ML library wrapping Hugging Face transformers/Trainer, shipping an asyncio MCP server
**Project:** DNALLM (`dnallm` v0.5.2) — Test Suite Audit & Coverage Hardening
**Researched:** 2026-09-29
**Confidence:** HIGH (top findings reproduced empirically in this repo; tool behavior verified against official docs of the exact installed versions: pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, pytest-timeout 2.4.0, pytest-asyncio 1.4.0)

## How this was verified

Findings marked **[REPRODUCED]** were reproduced on this repository on 2026-09-29 (commands included). Findings marked **[DOCS]** are verified against the official documentation of the exact tool versions installed in the dev environment. Web-only community findings are marked **[WEB]** and carry lower confidence.

---

## Critical Pitfalls

### Pitfall 1: The root `conftest.py` `os._exit(0)` atexit hook silently disables any CI gate — the gate that can never fail

**What goes wrong:** **[REPRODUCED]** The repo-root `conftest.py` registers `atexit.register(force_cleanup_and_exit)` at `pytest_sessionstart`; the handler terminates the process with `os._exit(0)`. Because `atexit` handlers run during interpreter shutdown *after* pytest has raised `SystemExit(exitstatus)`, the hard `os._exit(0)` discards pytest's real exit code and the OS sees 0. Reproduced on 2026-09-29:

```
$ python -m pytest tmp_exitcheck -q          # single test: assert False
1 failed in 0.02s
$ echo $?
0                                            # <-- failure completely masked
```

Consequence for this milestone: `--cov-fail-under=90`, test failures, and even collection errors (exit 2) all exit 0. The "hard gate" would be permanently green on day one. This also means the *current* `ci.yml` "Run fast tests" step cannot fail on test failures today, and `deploy` gating on `test` is an illusion.

**Why it happens:** The hook was added to kill hanging multiprocessing/CUDA teardown. `os._exit` cannot be interrupted and takes its argument as the final status — someone cargo-culted `0` (the "success" they observed when the hang went away) instead of propagating pytest's status.

**How to avoid:**
- Fix before enabling any gate. Preferred: delete the `atexit` registration and move cleanup into a `pytest_sessionfinish(session, exitstatus)` hook (which receives the real status and runs before exit). If a hard exit must stay, capture the status — `atexit.register(lambda: os._exit(saved_status))` where `saved_status` is stashed in `pytest_sessionfinish`.
- Add a permanent canary: a CI step (or meta-test) that runs pytest on one deliberately failing throwaway test and asserts the shell exit code is non-zero (`pytest ... ; test $? -ne 0`). Cheap, catches any future regression of this class.

**Warning signs:**
- `pytest <anything> ; echo $?` prints 0 while the summary says `N failed`.
- Coverage gate "passes" on a branch where you know coverage is below threshold.
- The `🧹 Force cleaning up resources... / 🚪 Forcing exit...` prints at the end of every run.

**Phase to address:** Phase 1 (Audit & Measurement Setup) — must land before any `--cov-fail-under` exists anywhere; the audit's pass/fail counts are themselves unreliable until fixed.

---

### Pitfall 2: Wrong coverage denominator — unimportable adapters counted at 0%, packaged test files counted in the numerator, and no `[tool.coverage.run]` exists at all

**What goes wrong:** **[REPRODUCED]** Three separate denominator traps, all confirmed by running `--cov=dnallm` on this repo:

1. `--cov=dnallm` is `source` semantics: coverage.py reports *never-imported* files at 0%. The report contains `dnallm/finetune/megatron.py` (184 stmts, 0%) and `dnallm/models/special/mamba_npu.py` (141 stmts, 0%) — 325 dead statements in an 8,344-statement denominator (~3.9 percentage points that no test can ever reach).
2. `dnallm/mcp/tests/test_*.py` (the packaged test suite), `dnallm/mcp/run_tests.py`, and `dnallm/mcp/example_sse_usage.py` appear as *source* rows. When the full suite runs, the test modules execute and their lines count toward the numerator — test code inflating the "product coverage" number. PROJECT.md's denominator decision (vendored dirs + adapters) does not yet exclude these.
3. There is **no `[tool.coverage.run]` section in pyproject.toml** — the `--cov=dnallm` flag lives only in the CI command line. Local runs without the flag measure nothing; runs with different flags measure differently. The decided omit list exists nowhere yet.

A subtlety found on inspection: the vendored `dnallm/tasks/metrics/` subdirectories have **no `__init__.py`** (0 of 55), which is the only reason they don't currently flood the denominator as 0% rows — that protection is accidental. The day a test imports any vendored metric module directly, those files enter the measured set.

**Why it happens:** `source`-style measurement is the correct choice for a package, but every exclusion has to be explicit. People assume `--cov=dnallm` "just measures the code my tests exercise"; it also measures code nothing imports, and code that is itself tests.

**How to avoid:**
- Create a single source of truth in `pyproject.toml`:
  ```toml
  [tool.coverage.run]
  source = ["dnallm"]
  omit = [
    "dnallm/tasks/metrics/*",        # vendored HF evaluate
    "dnallm/models/special/enformer_model/*",  # ported Enformer
    "dnallm/finetune/megatron.py",   # unimportable without Megatron-LM
    "dnallm/models/special/mamba_npu.py",      # unimportable without torch_npu
    "dnallm/mcp/tests/*",            # packaged test suite is not product code
    "dnallm/mcp/run_tests.py",
    "dnallm/mcp/example_sse_usage.py",
  ]
  ```
  Note omit globs are matched against file paths as shown in the report (`dnallm/...` relative form) — verify by grepping the term report, not by assumption.
- Keep `branch = true` OFF for this milestone. PROJECT.md mandates *line* coverage; branch coverage would silently redefine 90% to something much harder.
- Add a "denominator contract" CI check: after the coverage run, assert the term report contains zero rows matching `mcp/tests/`, `tasks/metrics/`, `megatron.py`, `mamba_npu.py` (grep on the report output). This prevents both accidental re-inclusion and numerator inflation.
- Prefer config over CLI flags: `--cov` bare in CI plus pyproject `source`, so local and CI measure identically.

**Warning signs:**
- Coverage % jumps or drops several points with no product-code change.
- `coverage report` lists any `*/tests/*` file or a file you know cannot import in CI.
- Team members' local coverage numbers don't match CI (CLI-vs-config split).

**Phase to address:** Phase 1 (config lands with the audit tooling); verified mechanically in every subsequent phase via the denominator-contract check.

---

### Pitfall 3: Enabling the 90% gate before the suite reaches 90% — permanently-red gate trains bypass behavior

**What goes wrong:** The mandate is "write tests until >90%, *then* enforce." The classic failure is flipping `--cov-fail-under=90` on early (or on a shared branch) while true coverage is far below: every PR is red for reasons unrelated to its diff. Within weeks the team responds by removing the flag "temporarily," scattering `# pragma: no cover`, skipping hard tests, or writing assertion-free tests (Pitfall 7). The gate then measures nothing. **[WEB]** Community practice is consistent: ratchet from a measured baseline, or make the gate blocking only after the target is first reached.

**Why it happens:** The gate feels like the "enforcement" deliverable, so it gets built first; or a ratchet number is guessed ("we're probably at ~80%") instead of measured.

**How to avoid:**
- Phase 1 must produce the measured number on the agreed denominator (fast suite AND full suite including `slow` — they differ, see Pitfall 4).
- Sequence per PROJECT.md: audit → fix → author tests past 90% → *then* flip `--cov-fail-under=90` in the final phase. The gate's first day blocking is the first day it's green.
- If an interim ratchet is wanted, gate at `floor(measured)` via a checked-in threshold file the CI reads, with a rule (or bot) that it may only move up. Never hard-code a guess.
- Police the escape hatches: baseline `# pragma: no cover` count in `dnallm/` is **3 today** — record it in Phase 1 and require any new pragma to carry a reason + issue link in review. Same for new `pytest.skip` (7 call sites today).

**Warning signs:**
- PRs whose only change is lowering/removing the threshold or adding pragmas/skips.
- Coverage graph rises while open bug count doesn't move (coverage theater, Pitfall 7).
- Developers running `-m "not slow"` locally then being surprised by the gate number.

**Phase to address:** Phase 1 records baseline; the gate itself is the *last* phase of the milestone.

---

### Pitfall 4: Flaky network tests (HF/ModelScope downloads) inside a blocking gate — red builds for non-code reasons, and silent skips that make coverage wobble

**What goes wrong:** The owner decision is that the gate run *includes* `slow` tests with real model downloads (16 `@pytest.mark.slow` tests; models such as `zhangtaolab/plant-dnabert-BPE`, `ZhejiangLab-LifeScience/DNA_bert_4`, DialoGPT-small). Three failure modes:

1. **Transient network failures fail the gate.** Anonymous HF hub downloads get rate-limited (429) and CI runners suffer DNS hiccups; the gate goes red on a Tuesday because of hub load, not code. **[WEB]**
2. **Skip-on-network-failure hides both bugs and lines.** The suite has 7 `pytest.skip(...)` call sites inside broad `except Exception` blocks (e.g. `tests/models/test_model.py:106,128`). In a coverage gate, a skip removes that test's unique covered lines from the numerator — so coverage percentage *varies run to run* with network weather, and the 90.0% gate flaps around the threshold. The skip also permanently masks real breakage (already documented in CONCERNS.md).
3. **Cold-cache downloads blow the time budget** (see Pitfall 5).

**Why it happens:** Tests were written to be "polite" (skip on network problems) for a non-gating CI. A hard gate inverts the incentives: now every non-deterministic skip/failure is a build breaker.

**How to avoid:**
- Run the gate in **one dedicated job** (single Python/numpy pin), not across the 6-leg matrix. Cache `~/.cache/huggingface/hub` with `actions/cache` keyed on a hash of the slow-test model list (`restore-keys` fallback); optionally add a warm-up step (`huggingface-cli download` / `snapshot_download`) and set `HF_HOME` explicitly. Consider an `HF_TOKEN` repo secret (higher rate-limit quota; keep it out of logs).
- Make skips deterministic and typed: replace broad `except Exception: pytest.skip` with catching specific network exceptions (e.g. `requests.ConnectionError`, `huggingface_hub` errors) and *fail* on everything else. In the gate job, treat unexpected skips as failures — assert the `-ra` skip summary matches an expected allowlist (or run with `--strict` skip accounting).
- Allow one automatic retry of the gate job (re-run on failure) so a single 429 doesn't demand human attention — but never auto-pass.
- Matrix legs keep running `-m "not slow"` without a coverage threshold; they protect the version matrix, the gate job protects coverage.

**Warning signs:**
- Gate green on immediate re-run with zero changes.
- Skip count differs between two runs of the same commit (`pytest -ra` output).
- Gate coverage number moves ±0.5pp between runs of identical code.

**Phase to address:** Test-hygiene part (typed skips, AUROC/CrossDNA unskips) in the bug-fix phase; caching + retry + single-job design in the CI-gate phase.

---

### Pitfall 5: Global `--timeout=300` collides with slow model downloads; the wrong timeout method destroys the whole coverage run

**What goes wrong:** **[DOCS — pytest-timeout 2.4.0 README]** Two distinct mechanisms:

1. **Budget collision:** the ini-level `--timeout=300` applies to every test including `slow` ones. A cold download + first tokenization of a real model on a 2-core GitHub runner can exceed 300s, so the gate fails on runner speed. Today's `slow` tests are deselected in CI, so nobody has ever seen this bind.
2. **Method collision:** on Linux the default method is `signal` (SIGALRM): on expiry it fails *only that test* and the run (and coverage report) completes — the good case. But if someone "hardens" the config with `--timeout-method=thread` (or runs on a platform where thread is default), a timeout **terminates the entire pytest process with a hard `os._exit`**: no teardown, no report, no coverage data — the gate fails with no diagnostics. Inverse trap: the signal method *cannot interrupt a hang in a non-main thread* (an MCP streamable-HTTP handler stuck in a worker thread, a wedged dataloader), so the run can hang until the job-level timeout.

**Why it happens:** `--timeout=300` was sized for the mocked fast suite. The gate run changes the workload; timeout configuration must change with it.

**How to avoid:**
- Keep the global 300s for fast tests; override per-test on real-download tests with `@pytest.mark.timeout(1800)` (marker priority beats ini — verified in README: ini < env < flag < marker; `timeout=0` disables). Do not raise the global timeout — that hides fast-suite regressions.
- Keep the `signal` method (Linux default) for the gate job — it preserves the coverage report on timeout.
- Warm the HF cache (Pitfall 4) so 300s rarely binds even for `slow` tests.
- Set a job-level `timeout-minutes` (e.g. 60–90) on the gate job as the backstop for non-main-thread hangs; GH Actions default (360 min) wastes six hours of CI when it triggers.

**Warning signs:**
- `Timeout >300.0s` failures that pass on re-run.
- Gate job killed at the Actions time limit with no test summary.
- Coverage report absent after a timeout event.

**Phase to address:** Phase 1 audit records wall-clock times of every `slow` test (cold and warm); timeout marks land with the gate job in the final phase.

---

### Pitfall 6: Subprocess/multiprocessing coverage is silently unmeasured — pytest-cov 7 removed subprocess support

**What goes wrong:** **[DOCS — pytest-cov official docs]** Subprocess support was **removed in pytest-cov 7.0**; this repo's dev environment has **pytest-cov 7.1.0** installed (pyproject allows `>=6.0.0`, so CI also resolves 7.x). Code executed in child processes — HF `Trainer` dataloader workers (`num_workers > 0`), `dnallm/mcp/run_tests.py` (`subprocess.run`), any stdio MCP server spawn — is invisible to measurement unless you configure coverage.py's native `patch = subprocess` under `[tool.coverage.run]` (which auto-enables `parallel = true` and per-process `.coverage.*` data files that must be combined).

Two ways this bites: (a) you chase "missing" lines that a passing test demonstrably exercises (they ran in a child), burning days; or (b) you enable `patch = subprocess` naively and hit stale-`.coverage.*` inflation (leftover data files from crashed runs get combined in) or fork+threads deadlocks under `concurrency = multiprocessing`.

**Why it happens:** Everyone's mental model of pytest-cov subprocess measurement is from 6.x (cov-core/.pth mechanism). The 7.0 removal is recent and easy to miss; the failure is silent (numbers just look wrong).

**How to avoid:**
- Make an explicit denominator decision in Phase 1: is child-process code in scope? Most of `dnallm`'s logic-under-test runs in the main process; dataloader-worker lines are framework glue. If out of scope, document it and don't chase those lines. If in scope, add `patch = ["subprocess"]` to `[tool.coverage.run]` and verify with a canary (a test that calls a function in a subprocess, asserted covered).
- Delete stale `.coverage*` files before authoritative runs (`rm -f .coverage .coverage.*` in the CI step) — combine is only trustworthy from a clean slate.
- Note: `os._exit` anywhere in a child (including copies of the Pitfall 1 pattern) skips the child's atexit flush and loses that child's data file. Fix Pitfall 1 everywhere it appears.

**Warning signs:**
- Lines reported missing despite a green test that demonstrably executes them.
- `.coverage.<hostname>.<pid>` files accumulating in the repo root.
- Coverage numbers differing between `pytest` and `pytest -p xdist` style runs.

**Phase to address:** Phase 1 (decision + config); verification canary in the CI-gate phase.

---

### Pitfall 7: Coverage theater — assertion-free and mock-only tests that prove nothing

**What goes wrong:** The pressure of a 90% mandate on a mock-heavy ML-wrapper codebase produces tests that *execute* code without *checking* behavior: call a function with `MagicMock` collaborators, assert nothing (or only `mock.assert_called_once_with`), and collect the lines. This domain is especially prone because real behavior needs real models: the cheapest path to a covered line in `dnallm/models/model.py` is to mock `AutoTokenizer.from_pretrained` and just... run the function. Broad `except Exception` fallback chains (the tokenizer triple-fallback at `model.py:540-568` that can silently degrade to one-hot tokenization) get "fully covered" while the actual degradation behavior is unverified. The current suite is clean by grep (no assertion-free test files found; shared mock fixtures exist in `tests/conftest.py`) — the risk is entirely in the *new* tests this milestone adds.

**Why it happens:** Writing a behavioral assertion requires understanding the behavior; executing a function requires only importing it. Under a numeric gate, the second is 10x cheaper and the gate can't tell the difference.

**How to avoid:**
- Review checklist rule for every new test: at least one *observable-outcome* assertion — returned value/type/shape, config mutation, file written, raised exception with `match=` — not call-count-only. (`call_count` assertions are fine *in addition*, never instead.)
- Follow the repo's own TESTING.md "What NOT to Mock" section: real Pydantic configs, real `dnallm/utils/sequence.py` values, real MCP wire behavior with mocked transport.
- For fallback chains, assert *which* fallback was selected (e.g. the returned tokenizer's type), not merely that the log line fired.
- Unskip the known-broken tests (multiclass AUROC at `tests/tasks/test_metrics.py:761`, CrossDNA at `model.py:873-887`) by *fixing the code* (already in scope per PROJECT.md) — do not write adjacent tests that route around the bug.
- Optional spot check: run `mutmut` (or hand-mutate a few predicates) on `dnallm/utils/sequence.py` and `dnallm/tasks/metrics.py`; if mutants survive your new tests, the tests are decorative.
- `pytest.raises(Exception)` without `match=` is a smell — it passes on any error, including the wrong one.

**Warning signs:**
- New test files where `grep -c assert` ≈ 0 relative to test count.
- Coverage climbing steadily on `server.py`/`inference.py` god-files with no new bug discoveries.
- Tests that never fail during refactors of the code they "cover".

**Phase to address:** Every test-authoring phase; enforce via review checklist and the pragma/skip police from Pitfall 3.

---

### Pitfall 8: The import-time transformers monkey-patch shim (`transformers_compat.py`) — untestable-by-construction lines and fake version gates

**What goes wrong:** `dnallm/utils/__init__.py` imports `transformers_compat`, whose module-level `apply_patches()` runs *once at first import* and patches `PreTrainedModel` methods; the bitsandbytes branches (`_swap_to_fp32`, `_restore_quantized`) import bnb lazily and manipulate `Params4bit` internals (`quant_state`, `_is_hf_initialized`). Under coverage pressure this module produces three failure modes:

1. **Chasing unreachable lines:** the bnb swap/restore lines cannot execute without bitsandbytes + a real 4-bit model in CI. The tempting "fix" is constructing fake `Params4int`-shaped objects just to touch the lines — a test that proves only that Python executes lines.
2. **Fake version gates:** patches are gated by `hasattr` duck-typing, not version strings. A test that fakes `transformers.__version__` changes *nothing* and can assert nonsense confidently.
3. **Patch-state pollution:** tests that `mock.patch` the same `PreTrainedModel` methods interact with the already-applied compat patches; a test asserting "patch applied" by re-importing the module is a no-op (module cached) and covers nothing new.

**Why it happens:** The module is *compatibility-contract* code whose essence is "behave differently per environment." Uniform line-coverage targets fit it badly.

**How to avoid:**
- Treat this module as behavior-contract testing, not line completion: verify the patched method *tolerates nested quantizer keys on a synthetic model* (observable behavior), and that `apply_patches()` is idempotent (safe to call multiple times — its docstring promises this; test it).
- For bnb-interior and hardware-only lines, prefer an explicit `# pragma: no cover` with a reason comment (counts against the Pitfall 3 police budget — that's what the budget is *for*) over fake-object tests. Same for `megatron.py`/`mamba_npu.py` (already omitted per Pitfall 2).
- Do not try to force 90% on every file. The gate is on the package; per-file gaps with documented pragmas are the honest outcome for environment-gated code.

**Warning signs:**
- Tests importing `types`/building `SimpleNamespace` objects shaped like `Params4bit` just to reach lines.
- `transformers_compat` coverage stuck just below target, driving pragma debates.
- Tests that mutate `transformers.__version__` expecting gate changes.

**Phase to address:** The utils test-authoring phase; the pragma-budget decision lands with the Phase 1 baseline.

---

### Pitfall 9: Gate on the full CI matrix including slow tests — unusably long CI

**What goes wrong:** The existing matrix is 6 legs (py 3.11/3.12/3.13 × numpy 1.26.4/2.2.0). Running the *slow-inclusive, coverage-instrumented* suite on all 6 legs means 6 × (full model downloads + ~20-30% coverage-tracer overhead on torch-heavy tests + long tail). Result: hour-plus queues per PR, `push` builds on `dev` backing up, developers starting to skip CI (`[ci skip]`) — which defeats the gate. **[WEB]** for overhead estimates; the multiplier arithmetic is plain.

**Why it happens:** Copying the existing matrix job and appending `--cov-fail-under` to its pytest line — the one-line-diff version of the gate.

**How to avoid:**
- New dedicated `coverage-gate` job: one Python version, one numpy version (pick the combination closest to the dev env — py3.12/numpy 2.2.0), full suite including `slow`, `--cov-fail-under=90`, HF cache (Pitfall 4), `timeout-minutes` backstop (Pitfall 5).
- Matrix legs keep `-m "not slow"` and no threshold; they answer "does it work on the version matrix", the gate job answers "is coverage ≥90". Two questions, two jobs.
- Sequence: fast matrix first (fail fast on lint/unit), gate job in parallel or after. Upload the coverage XML as an artifact from the gate job only.

**Warning signs:**
- PR CI wall-clock > 45-60 min.
- Developers batching merges to avoid CI waits.
- `timeout-minutes` hit regularly on the gate job.

**Phase to address:** CI-gate phase (job design is a first-class deliverable there, not an edit to the existing job).

---

### Pitfall 10: Relying on codecov (action v3, `fail_ci_if_error`) as the enforcement mechanism

**What goes wrong:** **[WEB]** `ci.yml` pins `codecov/codecov-action@v3` with `fail_ci_if_error: false`. v3 is unsupported (no bug/security fixes; v1's bash uploader was sunset Feb 2022; v4/v5 moved to the CLI uploader and generally require `CODECOV_TOKEN`). Two traps: (a) trying to enforce via codecov's threshold/status checks couples pass/fail of every build to a third-party service — codecov incidents become your red builds; (b) upgrading the action without adding a token silently stops uploads (coverage stats freeze at an old commit and nobody notices for weeks).

**Why it happens:** The upload step *looks* like a coverage gate, so "turn it on" seems like a one-flag change.

**How to avoid:**
- Enforce locally: the gate job's pytest invocation carries `--cov-fail-under=90` and the *job* fails on the pytest exit code (only trustworthy after Pitfall 1 is fixed — this is why the exit-code fix precedes everything).
- Keep codecov informational only: upgrade to `@v5` with `CODECOV_TOKEN` secret or drop the step entirely; never let a codecov status be required for merge.
- If you want trend enforcement beyond the gate, the ratchet file from Pitfall 3 is in-repo and has no external dependency.

**Warning signs:**
- Codecov badges/comments stale relative to recent commits.
- Red builds whose only failing step is the codecov upload.
- Anyone proposing codecov patch/status checks as *required* checks.

**Phase to address:** CI-gate phase (one line of intent: gate = pytest exit code; codecov = reporting only).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keeping the `atexit os._exit(0)` and adding the gate anyway | No conftest rework; gate "green" day one | The entire gate is cosmetic; regressions ship silently | Never — this is the one blocker that must be fixed first |
| `# pragma: no cover` on hard-to-test lines | Fast path to 90% | Pragmas never expire; real gaps hide behind them | Environment-gated code only (bnb internals, NPU/Megatron), each with reason + issue link; budget tracked from baseline of 3 |
| Skipping flaky tests instead of fixing them | Green build today | Coverage numerator shrinks unpredictably; gate flaps; bugs hide | Never in the gate job; quarantine list with owners and expiry only |
| Raising the global `--timeout` to make slow tests pass | One-line fix | Real hangs in fast tests no longer fail promptly | Never; use per-test `@pytest.mark.timeout` overrides instead |
| `filterwarnings` blanket ignores (already present: DeprecationWarning/UserWarning) | Quiet runs | transformers 5.x→6.x deprecation signals invisible; next compat break lands mid-milestone | Acceptable now (out of scope), but do not widen the list during this milestone |
| Coverage config as CLI flags in CI only | No pyproject edit | Local and CI measure different things; denominator disputes | Never — move to `[tool.coverage.run]` in Phase 1 |
| Deleting the skipped AUROC/CrossDNA tests instead of fixing the bugs | Smaller suite, faster to 90% | Known defects (PROJECT.md explicitly lists them) remain shipped | Never — fixing them is in scope and required for honest coverage |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Hugging Face Hub | Cold-download in the gate job every run; anonymous requests only | `actions/cache` on `~/.cache/huggingface/hub` keyed on the slow-test model list; `HF_HOME` set explicitly; optional `HF_TOKEN` secret for quota; warm-up step |
| Hugging Face Hub | Treating every network error as skippable | Catch specific network exceptions; fail on others; expected-skip allowlist enforced in the gate job |
| ModelScope | Forgetting the `slow` suite also pulls from ModelScope (OSS CDN) — different failure domain than HF | Include ModelScope-backed tests in the audit's flake assessment; same typed-skip rules |
| GitHub Actions cache | Assuming the 10 GB repo cache always holds all models | Key on the model list; `restore-keys` partial fallback; measure hit rate; prune large models from `slow` tests if evicted |
| Codecov | v3 + `fail_ci_if_error: true` as "the gate" | pytest `--cov-fail-under` exit code is the gate; codecov v5 + token informational only |
| pytest-timeout | `--timeout-method=thread` for "reliability" | Keep `signal` (Linux default) in the gate job — thread kills the process and the coverage report |
| pytest-cov 7 | Assuming 6.x subprocess behavior | `patch = ["subprocess"]` in `[tool.coverage.run]` if child-process code is in the denominator; clean `.coverage*` before runs |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Coverage tracer overhead on torch-heavy tests | Gate job 20-30% slower than the same suite uninstrumented | Accept it in the single gate job; keep matrix legs uninstrumented | 464-test suite + slow tests on 2-core runner |
| Cold HF cache on first gate run | First run after cache-key change times out or takes 3-5x longer | Warm-up step + generous first-run per-test timeouts; treat cache-miss runs as expected-slow | Every change to the slow-test model list |
| `--timeout=300` sized for mocks, applied to downloads | `Timeout >300.0s` on `slow` tests, pass on re-run | Per-test `@pytest.mark.timeout(1800)` marks on download tests | First CI run that includes `slow` (ever, in this repo's history) |
| Job-level GH Actions default timeout (360 min) | Six hours of queued CI burned by one hung non-main-thread test | `timeout-minutes: 60-90` on the gate job | Any MCP streamable-HTTP / dataloader hang (SIGALRM can't reach it) |
| Matrix × gate multiplication | 6 legs × downloads; PR queues > 1h | Dedicated single gate job (Pitfall 9) | The day the gate is added to the matrix |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Leaking `HF_TOKEN` into build logs | Token abuse for hub quota | Use `secrets.` context only; never `echo` env in debug steps; token needs read scope only |
| Gate only on `main` while `dev` merges bypass it (ci.yml triggers on push to dev) | Coverage regressions enter via dev PRs un-gated | Gate job must run on `pull_request` targeting every protected branch, incl. dev→main |
| Relaxing `trust_remote_code=True` paths in new slow tests to "make them pass" | Arbitrary code execution from a typosquatted model repo (already a documented CONCERNS item) | Slow tests pin exact vetted model IDs; no string-built model names from fixtures |
| Test subprocess execution expanding scope (`run_tests.py` runs pytest as subprocess) | Command built from test-controlled strings | Keep list-form `subprocess.run`, no shell; no new subprocess tests without fixed argv |

## Maintainer-Experience Pitfalls (UX of the test suite)

| Pitfall | Impact | Better Approach |
|---------|--------|-----------------|
| Local coverage command differs from CI (flags vs config) | "Works on my machine" coverage disputes | One command in CONTRIBUTING: `pytest --cov` with all config in pyproject |
| Fast-suite coverage vs gate coverage (with `slow`) treated as one number | Developers chase gaps that only exist in the other selection | Publish both numbers in the audit; the gate number is authoritative; document both commands |
| PDF test artifacts written into the repo tree (`tests/inference/pdf/`, .gitignore typo `test/inference/pdf/`) | Dirty working tree mid-milestone; accidental commits muddy the coverage PRs | Point plot tests at `tmp_path`; fix the .gitignore typo while touching tests |
| Stale docs referencing `tests/pytest.ini` (config actually in pyproject) | New contributors edit the wrong file | Refresh `tests/TESTING.md` + CONTRIBUTING in the same phase as the config work |

## "Looks Done But Isn't" Checklist

- [ ] **Gate actually gates:** a deliberately failing test / sub-threshold coverage run makes the CI job red (canary from Pitfall 1) — verify, don't assume
- [ ] **Exit code fixed:** `pytest <failing-test>; echo $?` returns non-zero with the root conftest in place
- [ ] **Denominator clean:** `coverage report` contains zero rows for `dnallm/mcp/tests/*`, `tasks/metrics/`, `megatron.py`, `mamba_npu.py` — enforced by CI check, not by review memory
- [ ] **Gate stable:** three consecutive scheduled runs of the gate job green with identical coverage ±0.2pp (network weather proven handled)
- [ ] **Skips accounted:** `-ra` summary in the gate job shows exactly the expected allowlist; the AUROC and CrossDNA tests are unskipped and asserting correct behavior
- [ ] **Timeouts sized:** every `@pytest.mark.slow` download test carries a per-test timeout mark; gate job has `timeout-minutes`
- [ ] **Subprocess decision made:** child-process coverage either configured (`patch = ["subprocess"]` + canary test) or explicitly documented out of the denominator
- [ ] **Pragma budget intact:** `grep -rc "pragma: no cover" dnallm/` equals baseline (3) + documented additions only
- [ ] **One command:** local `pytest --cov` reproduces the CI gate number on the same selection
- [ ] **Matrix vs gate split:** matrix legs still fast and green; the gate runs in exactly one job

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Exit-code masking discovered after gate ships | LOW | Fix conftest (sessionfinish hook); add canary; audit anything merged during the blind period |
| Wrong denominator discovered late | LOW | Add/fix omit globs; re-measure; the percentage moves but tests need no rework |
| Gate permanently red (enabled too early) | LOW | Lower `--cov-fail-under` to measured baseline; re-raise via ratchet file |
| Coverage theater discovered late (assertion-free tests) | HIGH | Identify by grep/mutation spot-check; rewrite tests with behavioral assertions — the lines are already "covered", so tooling won't help; only review finds them |
| Flaky-network gate (429/skip flapping) | MEDIUM | Add cache + token + typed skips; quarantine genuinely flaky tests with owners and expiry; re-enable |
| Timeout thread-method data loss | LOW | Revert to signal method; re-run; no code damage (data was simply not written) |
| Subprocess coverage missing | MEDIUM | Configure `patch = ["subprocess"]`, clean `.coverage*`, verify canary; or formally descoped |
| CI unusably long | MEDIUM | Split gate job from matrix; warm caches; prune slow tests that duplicate coverage paths |

## Pitfall-to-Phase Mapping

Assumes the milestone structure implied by PROJECT.md: **Phase 1 — Audit & Measurement Setup** (fix exit code, coverage config, denominator contract, baseline + ratchet, slow-test timing), **Phase 2 — Suite Hygiene & Bug Fixes** (unskip AUROC/CrossDNA, typed skips, PDF artifacts), **Phase 3 — Test Authoring to >90%**, **Phase 4 — CI Gate Enforcement** (dedicated job, caching, timeouts, codecov demotion).

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Exit-code masking (os._exit) | Phase 1 | Canary step: failing pytest run must exit non-zero |
| 2. Denominator (omit/tests-in-numerator/config split) | Phase 1 | CI grep of coverage report for forbidden rows; local `pytest --cov` == CI number |
| 3. Gate enabled too early / ratchet | Phase 1 (baseline) → Phase 4 (flip) | Gate's first blocking run is green; threshold file only moves up |
| 4. Flaky network in gate | Phase 2 (typed skips) + Phase 4 (cache/retry) | 3 consecutive green scheduled runs; skip allowlist enforced |
| 5. Timeout vs downloads / method choice | Phase 1 (timing data) + Phase 4 (marks, timeout-minutes) | No `Timeout >300s` failures across scheduled runs; report present after any timeout |
| 6. Subprocess coverage (pytest-cov 7) | Phase 1 (decision) + Phase 4 (canary) | Canary subprocess test covered (or descope documented) |
| 7. Assertion-free tests | Phase 3 (+ review gate all phases) | Mutation spot-check on utils/metrics; review checklist enforced |
| 8. transformers_compat fake-coverage | Phase 3 | Behavior-contract tests present; pragmas documented; no fake bnb objects |
| 9. Gate on matrix / CI length | Phase 4 | Gate = 1 job; matrix unchanged; PR wall-clock < 60 min |
| 10. Codecov as gate | Phase 4 | Gate fails only on pytest exit code; codecov informational |

## Sources

- **[REPRODUCED in-repo, 2026-09-29]** exit-code masking experiment (`pytest` failing run → `EXIT_CODE=0` with root `conftest.py`); coverage denominator analysis (`python -m coverage report` on a `--cov=dnallm` run: megatron.py 184/0%, mamba_npu.py 141/0%, `dnallm/mcp/tests/*` rows present, TOTAL 8344 stmts, vendored metrics dirs have 0 `__init__.py` in 55); installed versions (pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, pytest-timeout 2.4.0, pytest-asyncio 1.4.0); suite counts (16 slow tests, 7 pytest.skip sites, 3 pragmas, no assert-free test files)
- **[DOCS]** pytest-cov subprocess-support removal in 7.0 and migration to coverage `patch = subprocess`: pytest-cov.readthedocs.io/en/latest/subprocess-support.html (fetched 2026-09-29)
- **[DOCS]** pytest-timeout method semantics and mark priority: github.com/pytest-dev/pytest-timeout README (fetched 2026-09-29)
- **[DOCS + REPRODUCED]** coverage.py `source` includes unexecuted files at 0% — coverage.readthedocs.io "Specifying source files"; pytest-cov PyPI notes; empirically confirmed in-repo
- **[WEB]** HF Hub cache layout and `HF_HOME`: huggingface.co/docs caching docs; HF Hub 429/rate-limit and `HF_HUB_OFFLINE`/token mitigations, actions/cache patterns for `~/.cache/huggingface/hub` (community discussions, Feb 2025+) — MEDIUM confidence
- **[WEB]** codecov-action v3 unsupported / v1 bash uploader sunset (Feb 2022) / v4-v5 token requirements: github.com/codecov/codecov-action, about.codecov.io January product update, docs.codecov.com/docs/codecov-tokens — MEDIUM confidence
- **[WEB]** ratchet-vs-big-bang threshold practice and pragma/assertion policing: pytest-with-eric.com coverage guidance (Sep 2024) — LOW confidence (community guidance, directionally consistent)
- Project context: `.planning/PROJECT.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/CONCERNS.md`, `conftest.py`, `pyproject.toml`, `.github/workflows/ci.yml`, `dnallm/utils/transformers_compat.py`

---
*Pitfalls research for: DNALLM coverage hardening milestone*
*Researched: 2026-09-29*
