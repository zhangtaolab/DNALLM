---
phase: 261003-ij4
plan: 01
subsystem: mcp
tags: [mcp, dna_interpret, event-loop, executor, single-flight, wr-01, serving-liveness]
requires:
  - CR-01 single-flight pattern (261003-hhj, model_manager.py:260-264)
provides:
  - dna_interpret event-loop liveness under long captum attributions (WR-01 closed)
  - dedicated _interpret_thread_lock (interpret single-flight across timeout-cancellation boundaries)
affects:
  - dnallm/mcp/server.py
  - tests/mcp/test_interpret_tool.py
tech-stack:
  added: []  # stdlib threading only — zero new dependencies
  patterns:
    - lock-inside-executor-closure (CR-01 / 261003-hhj pattern, applied to interpret)
key-files:
  created: []
  modified:
    - dnallm/mcp/server.py
    - tests/mcp/test_interpret_tool.py
decisions:
  - Dedicated _interpret_thread_lock, NOT _infer_thread_lock — DNAInterpret uses no DataLoader (no fork-unsafe window) and attributions must not queue behind minutes-long predicts (planning-time decision, implemented as planned)
  - asyncio.get_running_loop() instead of model_manager's get_event_loop — inside a running coroutine, avoids the 3.12+ deprecation trajectory (planning-time decision, implemented as planned)
  - Test-side (deviation): poll window 7s instead of the plan's 2s — Python 3.12+ asyncio.wait_for drives raw coroutines inline in the caller's task, so a 2s window reds at the poll (TimeoutError) instead of the plan-named in_flight assertion
  - Test-side (deviation): mock_server._log_format = "text" required — _with_timeout_wrapper's success path calls _structured_log, which reads _log_format set only in initialize(); followed the established tests/mcp/test_timeout.py:48 pattern
metrics:
  duration: 13 min
  completed: 2026-10-03T05:47:35Z
status: complete
actuals:
  tokens: 3838        # chars/4 over the realized diff (15353 diff chars)
  tasks: 2
  commits: 1          # measured: git rev-list --count 0d267a0..3fe80bf
  files: 2
plan_head_before: 0d267a0
plan_head_after: 3fe80bf
---

# Quick Task 261003-ij4: Fix WR-01 — dna_interpret runs blocking captum work on the event loop Summary

One-liner: `_dna_interpret` now submits its whole captum body to the default executor behind a dedicated
single-flight `threading.Lock`, restoring serving liveness and making the 30s tool timeout real for
dna_interpret — pinned by 3 regression tests that were red on the old code (loop frozen, timeout wrapper
dead, flight lost on retry) and green after.

## What Was Built

**dnallm/mcp/server.py** (fix):

1. `import threading` added to the stdlib block (between `re` and `time`).
2. `self._interpret_thread_lock = threading.Lock()` created in `__init__` (exists before any tool can be
   served) with the full contract comment: lock acquired INSIDE the executor-submitted closure (CR-01 /
   261003-hhj pattern); timeout cancellation abandons only the await, the orphaned thread keeps the
   flight, retries queue behind it; deliberately NOT `_infer_thread_lock`.
3. `_dna_interpret`'s blocking block (DNAInterpret instantiation + layer_conductance
   `_find_embedding_layer` detection + `interpreter.interpret`) moved into a sync closure
   `_run_interpretation()` that enters `with self._interpret_thread_lock:` and returns the interpret
   result; submitted via `loop = asyncio.get_running_loop()` +
   `await loop.run_in_executor(None, _run_interpretation)` with the WR-01 rationale comment
   (172s attribution must never occupy the loop thread).
4. Everything else byte-for-byte in behavior: sequence/method validation, the mamba guard
   (fa19675, now at server.py:1532-1552) untouched and still ahead of the submission, the
   target_class auto-select await, normalization, response dicts, outer try/except.

**tests/mcp/test_interpret_tool.py** (+243 lines): new class `TestDNAInterpretLoopOffloading` with 3
regression tests, all with explicit `target_class=0` (no predict mock), asyncio.sleep-based polling
(never blocking the probed loop on a threading wait), bounded `release.wait(timeout=5)` fakes:

1. `test_concurrent_tool_completes_while_interpretation_blocked` — `_health_check` heartbeat (with
   `get_loaded_models.return_value = []`) completes WHILE the attribution is blocked
   (`state["in_flight"] is True`).
2. `test_timeout_wrapper_fires_while_interpretation_blocked` — wrapped tool at
   `_tool_timeout_seconds = 0.5` returns the `error_type: "timeout"` dict in <2s while the attribution
   is still in flight; after `release.set()` the server stays usable (retry with plain return_value
   succeeds).
3. `test_timeout_cancelled_interpret_holds_single_flight_for_retry` (CR-01 mirror) — after a timed-out
   call, the immediate retry does NOT enter `interpret` while the orphan holds it
   (`counts["active"] == 1`), and `counts["max_active"] == 1` after the retry completes.

## Verification

| Check | Result |
|-------|--------|
| New class RED on pre-fix code | 3/3 FAILED at exactly the plan-named assertions (in_flight-during-heartbeat; error_type=="timeout" ×2 — old code returned a success dict after the 5s block) |
| `tests/mcp/test_interpret_tool.py` after fix | 23 passed (20 pre-existing unchanged + 3 new) |
| `tests/mcp/test_timeout.py` | 8 passed (wrapper untouched) |
| Full `tests/mcp` sanity | 223 passed in 67.64s (1 pre-existing ResourceWarning) |
| `ruff check` / `ruff format --check` both files | clean |
| `grep run_in_executor server.py` | line 1597, submitting `_run_interpretation` |
| `grep _interpret_thread_lock server.py` | created in `__init__` (:170), acquired only inside the closure (:1574) |
| Mamba guard position | guard at :1532-1552, submission at :1597 — guard ahead, `mock_interp_cls.assert_not_called()` tests green |

Commit: `3fe80bf` — `fix(quick-261003-ij4): dna_interpret runs captum work in executor — event loop stays
responsive and the tool timeout can fire (WR-01)` (test + fix atomic, per the owner rule that every
dnallm/ change ships with pytest coverage in the same change; no attribution trailers).

## Deviations from Plan

Two test-side adjustments (deviation rules 1/3 — fix inline, no architectural change; all planning-time
decisions implemented exactly as planned):

**1. [Rule 3 - Blocking issue] `_log_format` harness attribute missing**
- **Found during:** Task 1 RED run — tests 2/3 failed with `AttributeError: 'DNALLMMCPServer' object has
  no attribute '_log_format'` instead of the intended assertions
- **Issue:** `_with_timeout_wrapper`'s success path calls `_structured_log`, which reads
  `self._log_format` — set only in `initialize()`, which the `mock_server` fixture never runs. The
  harness gap would have broken the GREEN side too (success path logs on both sides of the timeout).
- **Fix:** `mock_server._log_format = "text"` alongside each `_tool_timeout_seconds` override — the
  established pattern from `tests/mcp/test_timeout.py:47-48`.
- **Files modified:** tests/mcp/test_interpret_tool.py
- **Commit:** 3fe80bf

**2. [Rule 1 - Bug] Poll window widened 2s → 7s so the RED lands at the plan-named assertion**
- **Found during:** Task 1 RED run — test 1 failed with `TimeoutError` at the poll line, not at the
  in_flight assertion
- **Issue:** since Python 3.12 `asyncio.wait_for` drives a raw coroutine **inline in the caller's
  task** (verified empirically: the poll coroutine runs before the created interpret task is ever
  scheduled; an expired timeout callback queued during the loop block preempts the resumption). With a
  2s window, the pre-fix failure surfaces at the poll instead of proving the heartbeat property.
- **Fix:** poll window 7s (> the fake's 5s bounded block), with an explanatory comment in the test.
  On pre-fix code the poll now succeeds at ~5s, the heartbeat runs after the block, and the test fails
  exactly at `state["in_flight"] is True` as the plan's done criteria requires. On fixed code the poll
  returns in milliseconds.
- **Files modified:** tests/mcp/test_interpret_tool.py
- **Commit:** 3fe80bf

## Threat Model Disposition

- **T-quick-wr01-01 (DoS, high) — mitigated:** executor offload + Tests 1-2 pin it.
- **T-quick-wr01-02 (DoS, medium) — mitigated:** `_interpret_thread_lock` inside the closure + Test 3
  pins it (retry queues behind the orphan; max_active == 1).
- **T-quick-wr01-03 (executor pool occupancy) — accepted** per plan (same acceptance as
  T-quick-cr01-02); every client still gets its timeout error promptly.
- **T-quick-wr01-SC — accepted:** zero new dependencies (stdlib threading only).

No new security-relevant surface beyond the plan's threat model.

## Self-Check: PASSED

- `dnallm/mcp/server.py` modified (commit 3fe80bf): FOUND
- `tests/mcp/test_interpret_tool.py` modified (commit 3fe80bf): FOUND
- Commit `3fe80bf` in `git log`: FOUND
- 23/23 tests in test_interpret_tool.py green; 223/223 in tests/mcp green (run post-commit)
