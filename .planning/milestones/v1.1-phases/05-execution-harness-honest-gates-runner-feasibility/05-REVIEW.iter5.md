---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-03T12:30:00Z
depth: standard
iteration: 5
files_reviewed: 56
files_reviewed_list:
  - dnallm/configuration/configs.py
  - dnallm/datahandling/data.py
  - dnallm/finetune/trainer.py
  - dnallm/inference/benchmark.py
  - dnallm/inference/inference.py
  - dnallm/inference/interpret.py
  - dnallm/inference/mutagenesis.py
  - dnallm/inference/plot.py
  - dnallm/mcp/model_manager.py
  - dnallm/mcp/server.py
  - dnallm/models/model.py
  - dnallm/models/special/borzoi.py
  - dnallm/models/special/crossdna.py
  - dnallm/models/special/enformer_model/configuration_enformer.py
  - dnallm/models/special/enformer_model/configuration_space.py
  - dnallm/models/special/enformer_model/data.py
  - dnallm/models/special/enformer_model/modeling_enformer.py
  - dnallm/models/special/enformer_model/modeling_space.py
  - dnallm/models/special/enformer_model/modules.py
  - dnallm/models/special/evo.py
  - dnallm/models/special/gpn.py
  - dnallm/models/special/lucaone.py
  - dnallm/models/special/megadna.py
  - dnallm/models/special/mutbert.py
  - dnallm/models/special/omnidna.py
  - dnallm/models/tokenizer.py
  - dnallm/utils/support.py
  - dnallm/utils/transformers_compat.py
  - docs/example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - docs/example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - docs/example/notebooks/benchmark/benchmark.ipynb
  - docs/example/notebooks/data_prepare/finetune/dev.csv
  - docs/example/notebooks/data_prepare/finetune/finetune_data.ipynb
  - docs/example/notebooks/data_prepare/finetune/test.csv
  - docs/example/notebooks/data_prepare/finetune/train.csv
  - docs/example/notebooks/embedding_attention.ipynb
  - docs/example/notebooks/finetune_generation/finetune_generation.ipynb
  - docs/example/notebooks/inference/inference.ipynb
  - example/mcp_example/mcp_client_ollama_langchain_agents.ipynb
  - example/mcp_example/mcp_client_ollama_pydantic_ai.ipynb
  - example/notebooks/benchmark/benchmark.ipynb
  - example/notebooks/data_prepare/finetune/dev.csv
  - example/notebooks/data_prepare/finetune/finetune_data.ipynb
  - example/notebooks/data_prepare/finetune/test.csv
  - example/notebooks/data_prepare/finetune/train.csv
  - example/notebooks/embedding_attention.ipynb
  - example/notebooks/finetune_generation/finetune_generation.ipynb
  - example/notebooks/inference/inference.ipynb
  - pyproject.toml
  - tests/configuration/test_configs.py
  - tests/examples/_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_script_execution.py
  - tests/mcp/test_interpret_tool.py
  - tests/mcp/test_model_manager.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 1
  warning: 1
  info: 3
  total: 5
status: issues_found
---

# Phase 5 (incremental, post-iter4 quick tasks): Code Review Report

**Reviewed:** 2026-10-03T12:30:00Z
**Depth:** standard
**Files Reviewed:** 56 (diff base `774aa61`)
**Status:** issues_found

## Summary

Incremental review of the five quick tasks landed since the iter4 report
(diff base `774aa61`): the transformers-5.x compat rungs in
`dnallm/utils/transformers_compat.py` (+743 lines: get_head_mask,
init_weights bookkeeping, config legacy defaults, MambaCache, DebertaV2
dict-vocab), the typing/ty-suppression pass across 27 `dnallm/` files
(`load_config` `DNALLMConfig` TypedDict, `PretrainedConfig` →
`PreTrainedConfig` renames, dead `# type: ignore` removals), the two MCP
serving fixes (single-flight inference in `ModelManager`, mamba interpret
refusal in `server.py`), the execution-harness changes (timeout/flavor
spec-field removal — the iter4 WR-02/WR-03 fixes — plus tuple extra
inputs, 4xx-honest probes, the ollama/MCP execute-state gate, and the
isolated langchain kernelspec lane), and the committed notebook run
outputs with `docs/` mirrors.

Verified against ground truth, not just the diff:

- Every transformers-5.x shim claim was re-checked against the LIVE
  installed transformers 5.17.0: `post_init` sets `all_tied_weights_keys`
  BEFORE calling `init_weights` (so the wrapper's depth-2 delegation is
  sound, not recursive); `TokenizersBackend.convert_to_native_format` has
  exactly the `(cls, trust_remote_code=False, **kwargs)` signature the
  wrapper mirrors and is called keyword-only from `from_pretrained`;
  `PretrainedConfig` on 5.17 has no `__getattr__` and no `is_decoder`
  default (gate works), and after `import dnallm` all nine patches attach,
  `to_dict()` stays free of the injected defaults, deepcopy round-trips,
  and the dict vocab normalizes to a pair list. No
  `hasattr(config, "is_decoder")` branch exists anywhere in transformers
  5.17 that the restored default could flip (only `getattr(..., False)`
  reads, which return the same value).
- All changed test files pass on the live env: 76 (compat), 129
  (configs + MCP), 12 fast-lane execution tests — all green; `ruff check`
  and `ruff format --check` clean; the 166 remaining `ty` diagnostics
  match the owner-accepted baseline from the typing triage, none
  attributable to these hunks.
- Harness invariants hold: every ACTIVE (7200 class mark vs max 3600
  cell) and GATED (incl. the three `_TIMEOUT_7200_GATED` overrides — the
  iter4 WR-02 fix) entry keeps the outer mark strictly above the cell
  budget; no consumer of the removed `flavor`/`test_timeout` spec fields
  remains; `example/notebooks/inference/test.csv` exists for the new
  tuple seeding; `.scratch/` is gitignored and outside
  `assert_tree_clean`'s watched paths; all `example/` ↔ `docs/example/`
  mirrors are byte-identical; the committed CSVs are synthetic
  sequence/label data with no credentials; no secret-like strings in any
  changed notebook.
- The `seed_sandbox` traversal guard blocks absolute destinations and
  symlink escapes too (`Path / absolute` → absolute, caught by the same
  `resolve()`-under-`tmp_path` check).

One Critical finding. The single-flight inference fix serializes
`infer_seqs` behind an asyncio lock around the `run_in_executor` await —
but `_with_timeout_wrapper` cancels that await on timeout
(`asyncio.wait_for`, default `tool_timeout_seconds` = 30s, max 300s),
cancellation releases the asyncio lock, and the executor THREAD keeps
running the orphaned `infer_seqs` (threads are not cancellable). The next
request then acquires the lock and forks DataLoader workers concurrently
with the orphan — the exact `os.fork is unsafe` / hung-server incident
the fix ships to prevent, reachable through the server's own default
timeout on any predict slower than 30s. Demonstrated empirically in this
review (max concurrent infer_seqs = 2 under timeout+retry). One Warning
(interpret tool blocks the event loop, making its own timeout wrapper
dead) and three Info items follow.

## Critical Issues

### CR-01: Single-flight inference lock releases on tool timeout while the orphaned infer_seqs thread keeps running

**File:** `dnallm/mcp/model_manager.py:249-252` (and `280-283`)
**Issue:** `_infer_lock` is an `asyncio.Lock` held around
`await loop.run_in_executor(None, inference_engine.infer_seqs, ...)`.
The tool wrapper cancels the awaiting coroutine on timeout
(`dnallm/mcp/server.py:301`, `asyncio.wait_for(..., timeout=
self._tool_timeout_seconds)`, default 30s per
`dnallm/mcp/config_validators.py:180` — the shipped
`mcp_server_config.yaml` does not override it). Cancellation exits the
`async with self._infer_lock` block and releases the lock, but the
default-executor thread cannot be cancelled and keeps running the
orphaned `infer_seqs` — including its `DataLoader(num_workers>0)` worker
forks. The client's immediate retry (or any concurrent client) acquires
the lock and starts a second `infer_seqs` concurrently. The single-flight
invariant ("at most one ``infer_seqs`` in the process at any instant" —
the test docstring and the fix rationale) is therefore violated exactly
on the slow-predict-then-retry path, which is the realistic serving case
for DNA models whose inference routinely exceeds the 30s default (the
campaign's own notebook cell budgets are 600–1800s). This was
demonstrated empirically during this review with a reproduction of the
lock/timeout/thread pattern: max concurrent infer_seqs = 2. Consequence
is the original failure mode: concurrent forks under threaded serving
raise `os.fork is unsafe ...` / hang `pt_data_worker` children holding
the serving socket.
**Fix:** Hold the single-flight guarantee for the lifetime of the worker
thread, not the coroutine — acquire a `threading.Lock` inside the
executor-submitted callable:

```python
import threading  # module top

# __init__:
self._infer_thread_lock = threading.Lock()

# predict_sequence / predict_batch:
def _single_flight_infer():
    with self._infer_thread_lock:
        return inference_engine.infer_seqs(sequence, **kwargs)

result = await loop.run_in_executor(None, _single_flight_infer)
```

Coroutine cancellation then only abandons the result; the orphaned
thread still holds the thread lock until `infer_seqs` returns, so the
next predict blocks in the executor until the fork window closes. Add a
regression test that asserts serialization survives a
`asyncio.wait_for`-cancelled predict (the current
`TestSingleFlightInference` exercises only well-behaved concurrent
awaits, which is why this hole was invisible).

## Warnings

### WR-01: dna_interpret runs blocking captum work on the event-loop thread — its timeout wrapper can never fire

**File:** `dnallm/mcp/server.py:1565-1571`
**Issue:** `_dna_interpret` calls `interpreter.interpret(...)`
(synchronous captum attribution, potentially minutes) directly inside the
async tool. `asyncio.wait_for` in `_with_timeout_wrapper`
(`server.py:301`) can only fire at an `await` point; while the single
loop thread executes the blocking interpretation, the timeout timer
cannot run and every other client on every transport is frozen. The
`dna_interpret` timeout wrapper is therefore effectively dead code, and
for non-mamba models an unbounded attribution blocks the whole server —
the same serving-liveness class of failure the mamba guard added in this
diff (server.py:1519-1539) exists to prevent, and the guard only covers
the mamba case. (Pre-existing behavior, but the function was modified in
this diff to address exactly this liveness hazard.)
**Fix:** Route the blocking call through the executor like the predict
paths:

```python
tokens, attr_scores = await asyncio.get_event_loop().run_in_executor(
    None,
    lambda: interpreter.interpret(
        input_seq=sequence,
        method=mapped_method,
        target=target_class,
        max_length=max_length,
        **kwargs,
    ),
)
```

This both unblocks the loop and makes the tool's timeout wrapper
functional.

## Info

### IN-01: Three new patch installers omit the try/except transformers-import guard the module contract promises

**File:** `dnallm/utils/transformers_compat.py:553` (`_patch_pretrained_config_legacy_defaults`), `:764` (`_patch_mamba_cache`), `:990` (`_patch_legacy_init_weights_bookkeeping`)
**Issue:** Every other patch in this module wraps its transformers
import in `try/except ... return` ("so importing DNALLM never breaks an
otherwise working environment" per the module docstring); the three new
installers import `transformers.configuration_utils` /
`transformers.cache_utils` / `transformers.modeling_utils` bare, so a
future transformers that renames one of these submodules would crash
`import dnallm` at `apply_patches()` instead of no-oping. transformers
is a hard dependency today, so this is consistency/robustness, not an
active bug.
**Fix:** Wrap each import in the same `try: import ... except Exception:
return` guard used by `_patch_remote_code_pruning_helpers` (lines
363-375).

### IN-02: Port bind-close-probe race in TestProbeHonesty unbound-port test

**File:** `tests/examples/test_notebook_execution.py:360-370`
**Issue:** `test_unbound_port_probes_down_with_verbatim_evidence` binds
a socket to an ephemeral port, closes it, then probes — another process
on the machine can bind that port inside the window, making the probe
see a live server and the `ok is False` assertion flake. Rare on a
loopback ephemeral port but a real TOCTOU in a test that runs in the
fast lane.
**Fix:** Retry once on an unexpected `ok is True`, or assert on the
evidence shape (transport-error prefix) instead of the boolean when a
server answers.

### IN-03: langchain notebook ensure-cell spawns a detached MCP server that is never shut down

**File:** `example/mcp_example/mcp_client_ollama_langchain_agents.ipynb` cell 3 (source line ~632, `subprocess.Popen(..., start_new_session=True)`)
**Issue:** The probe-then-ensure guard starts `dnallm-mcp-server` on
port 8000 detached, with cwd inside the (later deleted) pytest tmp
sandbox, and no cell ever terminates it. Deliberate by design ("If the
server is already running ... this cell only detects it"), but in the
gated test lane the leaked process outlives the run with a vanished
cwd, and subsequent runs silently reuse it. The `docs/` mirror carries
the same cell.
**Fix:** Track the PID in the log/notebook state and offer a shutdown
cell (`mcp_server_proc.terminate()` when this cell started it), or have
the gated lane's teardown kill servers it spawned.

---

_Reviewed: 2026-10-03T12:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Previous iterations: 05-REVIEW.iter2.md, 05-REVIEW.iter3.md, 05-REVIEW.iter4.md_
