---
phase: 261003-hhj
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/mcp/model_manager.py
  - tests/mcp/test_model_manager.py
autonomous: true
requirements:
  - CR-01
estimate:
  tokens: 34000
  raw_tokens: 17000
  tasks: 2
  confidence: med

must_haves:
  truths:
    - Cancelling the await of an in-flight predict (asyncio.wait_for timeout, the _with_timeout_wrapper shape at dnallm/mcp/server.py:301) abandons only the result — a predict begun after the cancellation blocks in the executor until the orphaned infer_seqs returns, so at most one infer_seqs runs in the process at any instant, counted across the cancellation boundary.
    - Well-behaved concurrent predicts (multi-model gather; sequence + batch together) still serialize — the two existing TestSingleFlightInference tests pass unchanged.
    - predict_sequence/predict_batch external behavior is otherwise unchanged: results returned verbatim, exceptions from infer_seqs swallowed into None, and asyncio.CancelledError still propagates to the caller (CancelledError is BaseException and is never caught by the existing except Exception handler — do not widen that handler).
  artifacts:
    - dnallm/mcp/model_manager.py — _infer_thread_lock (threading.Lock) created in __init__, acquired inside the executor-submitted callable in both predict_sequence and predict_batch; the asyncio.Lock _infer_lock removed.
    - tests/mcp/test_model_manager.py — TestSingleFlightInference gains a timeout-cancellation regression test plus a blocking-engine helper.
  key_links:
    - server.py _with_timeout_wrapper asyncio.wait_for cancellation → predict_sequence/predict_batch executor closure → _infer_thread_lock held for the full orphaned-thread lifetime of infer_seqs (the fork-unsafe window of DataLoader worker spawn).
---

<objective>
Close CR-01 from the Phase 05 incremental review (05-REVIEW.md, commit a943411): the
single-flight inference guarantee in ModelManager currently lives in an asyncio.Lock
held around `await loop.run_in_executor(None, inference_engine.infer_seqs, ...)`.
When the tool timeout wrapper cancels that await (default tool_timeout_seconds=30 per
config_validators.py:180; real DNA predicts take minutes), the `async with` block
exits and releases the lock while the uncancellable executor thread keeps running the
orphaned infer_seqs — a client retry then starts a second infer_seqs concurrently,
reintroducing the os.fork-unsafe / hung-server incident the single-flight fix
(quick 261003-csd) shipped to prevent.

Purpose: make single-flight span the worker-thread lifetime, not the coroutine
lifetime, so the original fork-deadlock incident class cannot recur on the
slow-predict-then-retry path.

Output: dnallm/mcp/model_manager.py fixed (threading.Lock inside the
executor-submitted callable), tests/mcp/test_model_manager.py regression test that
fails on the old code and passes on the new code, committed together per the project
rule that every dnallm/ change ships with pytest coverage in the same change.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@dnallm/mcp/model_manager.py
@dnallm/mcp/server.py
@tests/mcp/test_model_manager.py
@.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-REVIEW.md

Key code points (verified at planning time):
- dnallm/mcp/model_manager.py:40 — `self._infer_lock = asyncio.Lock()` with the
  261003-csd rationale comment (lines 35-40).
- dnallm/mcp/model_manager.py:244-253 (predict_sequence) and 276-284 (predict_batch) —
  `async with self._infer_lock:` around `loop.run_in_executor(None, inference_engine.infer_seqs, ...)`.
- dnallm/mcp/server.py:298-341 — `_with_timeout_wrapper` wraps every non-streaming
  tool in `asyncio.wait_for(..., timeout=self._tool_timeout_seconds)`; on
  TimeoutError it returns the isError timeout dict to the client, which is exactly
  when a client retry fires.
- All predict entry points (dna_sequence_predict, dna_batch_predict, streams,
  mutagenesis at server.py:428/488/792/927/1171/1543) route through
  ModelManager.predict_sequence / predict_batch — fixing those two methods covers
  the whole infer_seqs surface (verified by grep: no other run_in_executor of
  infer_seqs in dnallm/mcp/).
- tests/mcp/test_model_manager.py:346 — TestSingleFlightInference with
  `_counting_engine` / `_registry` helpers (registry tracks active/max_active under
  a threading.Lock); its two tests only exercise well-behaved concurrent awaits,
  which is why the cancellation hole was invisible.
- `threading` is NOT currently imported in dnallm/mcp/model_manager.py (only
  asyncio, typing.Any, loguru, torch, time) — the fix must add it.
- `_infer_lock` has no references outside model_manager.py (verified by grep) —
  safe to rename.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: RED — regression test proving the timeout-cancellation hole</name>
  <files>tests/mcp/test_model_manager.py</files>
  <behavior>
    - RED against current code: after `asyncio.wait_for(manager.predict_sequence("model-a", "ATCG"), timeout=0.2)` raises TimeoutError with the orphan's infer_seqs observable mid-flight (a started threading.Event is set), an immediate retry predict for model-b enters its own infer_seqs right away — the retry engine's infer_seqs call_count is 1 (not 0) at the checkpoint — and registry max_active reaches 2. That failure at the retry-unstarted assertion is the proof the hole exists.
    - The orphaned first infer_seqs remains alive after the cancellation (started Event set, blocking on a release Event with a bounded wait of 10s so a red run cannot wedge pytest-timeout).
  </behavior>
  <action>
Add to class TestSingleFlightInference in tests/mcp/test_model_manager.py (module already imports asyncio, threading, time, Mock — no new imports needed):

1. A `_blocking_engine(registry, started, release)` static helper following the existing `_counting_engine` pattern: the infer_seqs side_effect increments registry active/max_active under registry lock, sets the started Event, blocks on `release.wait(timeout=10)`, decrements active, and returns {"probabilities": [0.5, 0.5]}. This is the controllable stand-in for a slow real predict (minutes-long infer_seqs with DataLoader worker forks).

2. Test `test_timeout_cancellation_does_not_release_single_flight(self, manager)` reproducing the tool-timeout shape end to end:
   - Load model-a with the blocking engine and model-b with the existing `_counting_engine(registry)`, both sharing one `self._registry()`.
   - Await predict #1 under `pytest.raises(asyncio.TimeoutError)` via `asyncio.wait_for(..., timeout=0.2)` — this is the same cancellation `_with_timeout_wrapper` delivers at server.py:301. Then assert the orphan started (`orphan_started.is_set()`).
   - Schedule the client's immediate retry as `asyncio.create_task(manager.predict_sequence("model-b", "ATCG"))`, `await asyncio.sleep(0.2)` to let its executor submission run, then assert the retry engine's infer_seqs call_count equals 0 — the single-flight contract must keep the retry out of infer_seqs while the orphan is still inside it. This is the assertion that fails on current code.
   - `release.set()` to let the orphan return, then `await asyncio.wait_for(retry_task, timeout=5)` and assert: retry result equals {"probabilities": [0.5, 0.5]}, retry engine call_count equals 1, and `registry["max_active"] == 1` — serialization held across the cancellation boundary.

Docstring must cite CR-01 and explain WHY the existing two tests missed this (they cancel nothing; asyncio.Lock only misbehaves when the await is cancelled).

Run the new test and confirm it fails at the retry-unstarted assertion (expect `call_count` of 1 vs 0, i.e. the retry ran concurrently — the reviewer reproduced max concurrent = 2). Do NOT commit yet — the red run is evidence, and the fix lands as one atomic commit with the test in Task 2.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest "tests/mcp/test_model_manager.py::TestSingleFlightInference::test_timeout_cancellation_does_not_release_single_flight" -q</automated>
  </verify>
  <done>
New test exists and FAILS on current code specifically at the retry-engine-unentered assertion (retry infer_seqs call_count 1 where 0 was required), while the two existing TestSingleFlightInference tests still pass. Nothing committed yet. Never use `uv run pytest` (known resolver failure) — always `.venv/bin/python -m pytest` from the repo root.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: GREEN — thread-lifetime single-flight in ModelManager</name>
  <files>dnallm/mcp/model_manager.py, tests/mcp/test_model_manager.py</files>
  <behavior>
    - GREEN: the Task 1 regression test passes — the retry predict's executor thread blocks on the thread lock until the orphaned infer_seqs returns, retry result is delivered afterward, and registry max_active stays 1 for the whole sequence including the cancellation boundary.
    - test_multi_model_predicts_execute_single_flight and test_sequence_and_batch_predicts_share_the_flight pass unchanged (threading-based serialization covers well-behaved concurrency too).
    - The full tests/mcp/test_model_manager.py file passes.
  </behavior>
  <action>
In dnallm/mcp/model_manager.py, move the single-flight guarantee from the coroutine to the worker thread, per the reviewer's fix shape (05-REVIEW.md CR-01):

1. Add `import threading` to the module-top imports.

2. In `__init__` (line 40): replace `self._infer_lock = asyncio.Lock()` with `self._infer_thread_lock = threading.Lock()`. Rewrite the preceding comment block (lines 35-40) to state the CR-01-hardened contract: the lock is acquired inside the executor-submitted callable, so cancelling the awaiting coroutine (tool timeout via _with_timeout_wrapper, server.py) abandons only the result — the orphaned executor thread keeps the flight until infer_seqs returns, and the next predict waits in the executor until the DataLoader-fork window closes. Keep the 261003-csd fork-unsafe rationale and add the CR-01 (261003-hhj) note.

3. In `predict_sequence` (lines 244-253): drop the `async with self._infer_lock:` wrapper. Define a per-call closure `def _single_flight_infer():` that acquires `with self._infer_thread_lock:` and returns `inference_engine.infer_seqs(sequence, **kwargs)` from inside the with-block; submit that closure: `result = await loop.run_in_executor(None, _single_flight_infer)`. Keep the existing `return result  # type: ignore` and the surrounding try/except Exception → None handler EXACTLY as-is (CancelledError must keep propagating — it is BaseException and is not caught by `except Exception`).

4. Apply the identical change in `predict_batch` (lines 276-284) with `sequences` in place of `sequence`, updating its inline comment to match.

5. Touch nothing else: no server.py changes, no executor changes, no timeout changes — minimal fix per the project scope rule.

Then run the regression test (green), the full test file, and commit BOTH files (test + fix) as one atomic commit following the repo's quick-task convention, message:
`fix(quick-261003-hhj): single-flight spans orphaned infer_seqs thread lifetime — timeout cancellation no longer releases the flight (CR-01)`
No attribution trailers of any kind.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest tests/mcp/test_model_manager.py -q</automated>
  </verify>
  <done>
The regression test and every other test in tests/mcp/test_model_manager.py pass; grep of dnallm/mcp/model_manager.py shows no remaining `_infer_lock` asyncio usage and `_infer_thread_lock` acquired inside the executor-submitted closures of both predict_sequence and predict_batch; both files committed together in one fix(quick-261003-hhj) commit with no attribution trailers.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| MCP client → server tools | Untrusted request timing: a client can let a predict hit tool_timeout_seconds (30s default) and immediately retry, deliberately or accidentally overlapping requests |
| event loop → default executor threads | asyncio cancellation crosses this boundary one-way: the coroutine dies, the thread does not |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-quick-cr01-01 | Denial of Service | ModelManager.predict_sequence/predict_batch single-flight lock | high | mitigate | threading.Lock acquired inside the executor-submitted callable (Task 2): timeout cancellation releases only the coroutine, the orphaned thread holds the flight until infer_seqs returns, so the retry-into-concurrent-DataLoader-forks window (os.fork-unsafe crash / hung pt_data_worker children) stays closed; regression test (Task 1) pins the cancellation boundary |
| T-quick-cr01-02 | Denial of Service | default executor pool shared by predicts and _load_model_sync | medium | accept | With the fix, each queued predict occupies one default-executor thread while blocked on the flight; mass timeout+retry storms can saturate the pool and delay model loads. Accepted for this minimal CR-01 fix: clients still receive their timeout error promptly, and a dedicated inference executor / queue-depth guard is beyond the "bug fixes limited to what correctness requires" scope rule — revisit if serving storms are observed |
| T-quick-cr01-SC | Tampering | package installs | high | accept | No npm/pip/cargo install tasks in this plan — stdlib threading only, zero new dependencies, so no supply-chain surface is introduced |
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/mcp/test_model_manager.py -q` — full file green (single-flight class now 3 tests).
- Regression test specifically: `.venv/bin/python -m pytest "tests/mcp/test_model_manager.py::TestSingleFlightInference" -q` — 3 passed.
- Structural check: no `async with self._infer_lock` remains in dnallm/mcp/model_manager.py; `_infer_thread_lock` is a threading.Lock created in `__init__` and acquired only inside the two executor-submitted closures.
- Optional broader sanity if wall-clock allows: `.venv/bin/python -m pytest tests/mcp -q` (known: tests/mcp/test_timeout.py carries a ~60s deferred cost).
</verification>

<success_criteria>
- CR-01 closed: cancelling an in-flight predict's await can no longer allow a second infer_seqs to start while the orphan runs — proven by a test that was red on the old code and is green on the new code (max_active == 1 across the cancellation boundary).
- Existing well-behaved single-flight tests unchanged and green.
- predict_sequence/predict_batch signatures, return values, and error-swallowing semantics unchanged; CancelledError propagation preserved.
- Test and fix committed atomically per the owner rule (dnallm/ change ships with pytest coverage in the same change).
</success_criteria>

<output>
Create `.planning/quick/261003-hhj-fix-cr-01-mcp-single-flight-inference-as/261003-hhj-SUMMARY.md` when done
</output>
