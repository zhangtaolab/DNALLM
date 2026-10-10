---
quick_task: 261003-hhj-fix-cr-01-mcp-single-flight-inference-as
phase: quick-261003-hhj
plan: 01
status: complete
started: 2026-10-03T04:42:25Z
completed: "2026-10-03T04:47:20Z"
duration_min: 5
branch: phs
push: manual-only
estimate:
  tokens: 34000
actuals:
  tokens: 2192    # chars/4 over git diff 84841ea..032b308 (realized changes)
  tasks: 2
  commits: 1      # git rev-list --count 84841ea..HEAD
plan_head_before: 84841ea
plan_head_after: 032b308
tags: [mcp, single-flight, cr-01, threading, regression-test, code-review-fix]
key-files:
  modified:
    - dnallm/mcp/model_manager.py
    - tests/mcp/test_model_manager.py
commits:
  - 032b308 "fix(quick-261003-hhj): single-flight spans orphaned infer_seqs thread lifetime — timeout cancellation no longer releases the flight (CR-01)"
---

# Quick Task 261003-hhj: Close CR-01 — single-flight inference spans the orphaned-thread lifetime

**One-liner:** ModelManager single-flight moved from an `asyncio.Lock` around the executor await to a `threading.Lock` acquired inside the executor-submitted closure, so a tool-timeout cancellation abandons only the result while the orphaned thread keeps the flight — a client retry can no longer start a second fork-unsafe `infer_seqs` concurrently.

## Outcome

CR-01 from the Phase 05 incremental review (05-REVIEW.md, commit a943411) is closed:

| Truth | Result | Proof |
|---|---|---|
| Cancellation abandons only the result; at most one `infer_seqs` in the process at any instant, counted across the cancellation boundary | HELD | new `test_timeout_cancellation_does_not_release_single_flight`: retry `infer_seqs` `call_count == 0` while orphan mid-flight, `registry["max_active"] == 1` over the whole sequence (RED pre-fix: `assert 1 == 0` — retry ran concurrently, exactly the reviewer's max=2 repro) |
| Well-behaved concurrent predicts still serialize | HELD | both pre-existing `TestSingleFlightInference` tests pass unchanged |
| External behavior otherwise unchanged (verbatim results, exceptions→None, CancelledError propagates) | HELD | `except Exception` handler and `return result  # type: ignore` kept verbatim in both methods; all 26 other tests in the file green |

## The hole and the fix

- **Hole (pre-fix):** `_with_timeout_wrapper` (server.py:301) cancels the tool coroutine via `asyncio.wait_for` (default 30s; real DNA predicts take minutes). The `async with self._infer_lock` block exited on cancellation and released the lock while the uncancellable executor thread kept running the orphaned `infer_seqs` — an immediate client retry started a second `infer_seqs` concurrently, reintroducing the `os.fork`-unsafe / hung-server incident class the 261003-csd single-flight fix shipped to prevent.
- **Fix:** `self._infer_lock = asyncio.Lock()` → `self._infer_thread_lock = threading.Lock()`, acquired inside a per-call `_single_flight_infer()` closure submitted via `loop.run_in_executor(None, _single_flight_infer)` in both `predict_sequence` and `predict_batch`. Single-flight now spans the worker-thread lifetime, not the coroutine lifetime; `threading` added to the module imports; nothing else touched (no server.py / executor / timeout changes, per the minimal-scope rule).

## Deviations from Plan

None — plan executed exactly as written (RED evidence at the predicted assertion, atomic test+fix commit, no trailers).

Out-of-scope observation (not fixed, pre-existing): `mypy` in this venv fails inside numpy 2.x stubs (`type` statement vs configured `python_version = 3.10`) before any project file is checked — reproduces identically on untouched modules (e.g. `config_manager.py`); CI runs mypy advisory (`|| true`) so no gate is affected.

## Threat model dispositions

- **T-quick-cr01-01 (DoS, high, mitigate):** mitigated as planned — regression test pins the cancellation boundary (above).
- **T-quick-cr01-02 (executor-pool saturation, medium, accept):** accepted as planned — each queued predict occupies one default-executor thread while blocked on the flight; timeout errors still reach clients promptly. Revisit only if serving storms are observed.
- **T-quick-cr01-SC (supply chain, accept):** honored — stdlib `threading` only, zero installs, no package tasks.

## Verification summary

- RED (pre-fix): `tests/mcp/test_model_manager.py::TestSingleFlightInference::test_timeout_cancellation_does_not_release_single_flight` FAILED at the retry-unstarted assertion (`assert 1 == 0`); the two pre-existing single-flight tests passed — the hole was real and specifically the cancellation path.
- GREEN (post-fix): `TestSingleFlightInference` 3 passed; full `tests/mcp/test_model_manager.py` **28 passed**.
- Broader sanity: full `tests/mcp` **220 passed** in 66.4s (includes the known ~60s test_timeout.py deferred cost).
- Structural: no `async with self._infer_lock` remains; `_infer_thread_lock` is a `threading.Lock` created in `__init__` and acquired only inside the two executor-submitted closures; the only remaining `async with` in the module is the unrelated `_loading_lock`.
- Style gates: `ruff format --check` + `ruff check` clean on both changed files (two of the new test's lines were joined by the formatter — cosmetic, included in the atomic commit).
- Commit `032b308`: exactly the two planned files, no deletions, no attribution trailers, no stray untracked artifacts.

## Known Stubs

None.

## Self-Check: PASSED

- Files exist: `dnallm/mcp/model_manager.py` (modified, threading.Lock fix verified by grep), `tests/mcp/test_model_manager.py` (modified, 3-test single-flight class verified by pytest).
- Commit `032b308` present on `phs` (`git log --oneline` match), parent `84841ea` as recorded in frontmatter.
