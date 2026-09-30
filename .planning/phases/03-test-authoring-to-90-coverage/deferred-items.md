## Deferred Items

- dnallm's import-time file sink creates `logs/dnallm.log` under the pytest launch cwd on every suite run
  status: open
  **Status:** open
  **What:** `DNALLMLogger._setup_handlers` (dnallm/utils/logger.py:57-60) does `Path("logs").mkdir(exist_ok=True)` and binds a `FileHandler` relative to the CWD at the first `get_logger()` call — which happens at module import (e.g. `dnallm/mcp/model_manager.py:19`). Every pytest run therefore (re)creates `logs/dnallm.log` in the repo root before any test executes. The path is gitignored, so git-status tree-clean gates never see it, but plan-level `Path('logs').exists()` checks are unsatisfiable by construction. Found during 03-03 Task 3 (start_server logs/ gate). Out of scope for wave 3 (pre-existing library behavior, not caused by the wave's changes); wave 5's test_logger work could pin the sink to a tmp_path via an autouse fixture if desired.
- test_timeout.py spends a fixed ~60s per run on two tests that wait out the full 30s default timeout
  status: open
  **Status:** open
  **What:** `TestToolTimeout::test_tool_timeout_returns_error` and `test_timeout_error_structure` wrap coroutines sleeping 100s in `asyncio.wait_for(..., timeout=30)` — the wrapper's timeout fires at the full 30s wall clock each time (the sleep is not shortened). Pre-existing (out of scope for wave 3); shortening the server's `_tool_timeout_seconds` in those tests the way `test_timeout_configurable` already does would save ~60s per census.
