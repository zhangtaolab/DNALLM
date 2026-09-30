---
phase: 03-test-authoring-to-90-coverage
plan: 03
subsystem: testing
tags: [pytest, coverage, mcp, httpx-asgitransport, fastmcp, asyncio, fault-injection]

requires:
  - phase: 03-test-authoring-to-90-coverage
    provides: wave-2 models closeout (88/240), suite-wide coverage growth to 79.90%, coverage-wave2-missing.txt worklist
  - phase: 01-harness-integrity-measured-baseline
    provides: measured baseline + coverage tooling of record
provides:
  - mcp area closed from 449 missing to 6 (gate ≤ 110 — passed with 104 lines of slack; research residual estimate was 69)
  - 154 new behavior tests (in-memory MCP protocol round trip, ordered progress contracts, transport construction shapes, ModelManager lifecycle, start_server CLI, client corners, config branches)
  - One Rule 1 source fix (multi-model success-count misclassification) with regression test
  - coverage-wave3-missing.txt — re-ranked worklist input for wave 4
  - Suite 85.89% (6,355 / 7,399) at wave-3 census, up from 79.90% post-wave-2
affects: [03-test-authoring-to-90-coverage, 04-coverage-gate-ci]

actuals:
  tokens: 29075  # chars/4 over the realized diff (116,301 chars / 4); estimate was 50,000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "In-memory MCP pair: httpx.ASGITransport factory (localhost base_url — 421 otherwise) + manual router.lifespan_context + streamablehttp_client(httpx_client_factory=...) — zero sockets, full protocol round trip incl. session DELETE"
    - "Ordered report_progress assertions: streaming tools are progress-reporting coroutines — assert call_args_list sequences (0/25/75/100; i/total), never generator exhaustion"
    - "Construction-shape transport tests: patch uvicorn.Config/Server and app.run, assert assembled fields + Starlette double Mount — never a port bind (SSE in-memory protocol deadlocks by design)"
    - "SDK-factory mocking for client _connect: patch mcp.client.{streamable_http,sse,stdio}.* factories + mcp.ClientSession — covers all three transport bodies with zero I/O"

key-files:
  created:
    - tests/mcp/test_server_transports.py
    - tests/mcp/test_server_streaming.py
    - tests/mcp/test_model_manager.py
    - tests/mcp/test_start_server.py
    - .planning/phases/03-test-authoring-to-90-coverage/coverage-wave3-missing.txt
  modified:
    - dnallm/mcp/server.py
    - tests/mcp/test_client_sdk.py
    - tests/mcp/test_mutagenesis_tool.py
    - tests/mcp/test_interpret_tool.py
    - dnallm/mcp/tests/test_config_manager.py
    - dnallm/mcp/tests/test_config_validators.py

key-decisions:
  - "Wire tool names carry the leading underscore (FastMCP derives names from __name__ via functools.update_wrapper) — the 13-tool registration set is pinned with the underscores, matching what real clients must call"
  - "Rule 1 fix in _format_multi_model_results: failure is now keyed on the explicit {result: None} marker entry the same file constructs; .get('result') is None misclassified every successful dict-shaped prediction ('0 successful, N failed' on fully successful runs)"
  - "Objective-required coverage beyond the plan's task text (03-02 precedent): the eight timeout-wrapped tool bodies landed in test_server_streaming.py on the shared mock-server seam, the mutagenesis/interpret corners in their existing tool files, and server.main() CLI rows in test_server_transports.py — without them the ≤110 area gate is arithmetically unreachable"
  - "SSE construction test drives through start_server(transport='sse') rather than the private starter so the dispatch line itself is executed"
  - "The plan's literal 'not logs.exists()' gate is unsatisfiable: dnallm's import-time file sink (utils/logger.py:57-60) creates logs/dnallm.log at the pytest launch cwd on every suite run — substituted with 'no logs/mcp_server.log at repo root' (the Pitfall 5 tripwire proper); sink recorded in deferred-items.md"

patterns-established:
  - "Recording ASGI wrapper: intercept (method, path) pairs around the real app to assert protocol-observable requests (session DELETE) without sockets"
  - "ExceptionGroup-aware client failure assertions: build the group from the exceptiongroup backport (py310-portable), flatten leaves via _network_skip._network_leaves, assert all leaves are httpx.TransportError"

requirements-completed: [TEST-02]

coverage:
  - id: D1
    description: "tests/mcp/test_server_transports.py (24): in-memory streamable-http round trip (initialize / list_tools with the complete 13-tool set / call_tool health / session DELETE via a recording ASGI wrapper), transport dispatch + construction shapes for stdio / streamable-http / SSE with patched uvicorn, app-None RuntimeError guards, initialize idempotency + config-failure guard, lifespan shutdown, server.main CLI rows, initialize_mcp_server"
    requirement: TEST-02
    verification:
      - kind: integration
        ref: "tests/mcp/test_server_transports.py (24 passed, 3.9s; no unpatched uvicorn.Server construction — grep-gated)"
        status: pass
    human_judgment: false
  - id: D2
    description: "tests/mcp/test_server_streaming.py (34): ordered report_progress sequences for all three streaming tools, verbatim fault dicts (None mid-batch, Nth-call raise, no-models early return), generic-exception propagation out of the timeout wrapper, json _structured_log field assembly, plus the eight basic tool bodies (objective-required)"
    requirement: TEST-02
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_streaming.py (34 passed) + test_timeout.py regression (42 combined)"
        status: pass
    human_judgment: false
  - id: D3
    description: "tests/mcp/test_model_manager.py (25): load/unload/status routing on real YAML configs with the load boundary patched — lazy single-load, error/in-flight short-circuits, executor-bridge passthrough, aggregate loading with exception capture, predict routing, model-info memory degradation"
    requirement: TEST-02
    verification:
      - kind: unit
        ref: "tests/mcp/test_model_manager.py (25 passed)"
        status: pass
    human_judgment: false
  - id: D4
    description: "tests/mcp/test_start_server.py (8): setup_logging writes logs/mcp_server.log strictly under tmp_path (mandatory autouse chdir — no repo-root leak), argparse main rows (missing config exit 1, arg pass-through with defaults, KeyboardInterrupt clean exit, error exit 1 with shutdown), initialize_server helper"
    requirement: TEST-02
    verification:
      - kind: unit
        ref: "tests/mcp/test_start_server.py (8 passed); post-run gate: no logs/mcp_server.log at repo root"
        status: pass
    human_judgment: false
  - id: D5
    description: "tests/mcp/test_client_sdk.py extended 37 -> 73: all three _connect transport bodies via mocked SDK factories, persistent-session branch, async-context lifecycle + close, non-JSON error parse, ExceptionGroup-wrapped httpx connection failures (Pitfall 11), 20 typed-method executions, default per-transport URLs"
    requirement: TEST-02
    verification:
      - kind: unit
        ref: "tests/mcp/test_client_sdk.py (73 passed)"
        status: pass
    human_judgment: false
  - id: D6
    description: "dnallm/mcp/tests/{test_config_manager.py 11->21, test_config_validators.py 7->16} + tests/mcp/{test_mutagenesis_tool.py 12->16, test_interpret_tool.py 14->18}: invalid-config propagation, disabled-model skip, no-config getter defaults, streamable_http both branches, reload, dangling/unloaded reference flags, validator rejection pairs, mutagenesis/interpret corner arms"
    requirement: TEST-02
    verification:
      - kind: unit
        ref: "dnallm/mcp/tests/test_config_manager.py (21), test_config_validators.py (16), tests/mcp/test_mutagenesis_tool.py (16), test_interpret_tool.py (18) — all passing"
        status: pass
    human_judgment: false
  - id: D7
    description: "Wave-3 gate: full census 1380 passed / 7 allowlisted skips / audit exit 0 / 896s; mcp-area missing 6 <= 110 (104 slack); suite 85.89% (6355/7399); coverage-wave3-missing.txt committed; pragma budget still exactly 3; pyproject coverage/pytest config untouched"
    requirement: TEST-02
    verification:
      - kind: command
        ref: "pytest --junitxml --cov full census (exit 0) -> coverage json -> sum(missing_lines over dnallm/mcp/) = 6; scripts/audit_skips.py exit 0"
        status: pass
    human_judgment: false

duration: 49 min
completed: 2026-09-30
status: complete
commits: 3
plan_head_before: 440b21789ec1e3956b28532d13126c31ea8260c3
plan_head_after: 38ea916266d56615b312927e0284743df426bf95
---

# Phase 3 Plan 3: MCP Wave — Test Authoring Summary

**154 protocol and behavior tests closing the mcp area from 449 to 6 missing lines (gate ≤ 110) via a socket-free in-memory MCP round trip, ordered progress-contract assertions, patched-uvicorn transport construction shapes, and full lifecycle coverage — plus one Rule 1 fix to the multi-model success counter**

## Performance

- **Duration:** 49 min (incl. 15-min full census)
- **Started:** 2026-09-30T10:41:25Z
- **Completed:** 2026-09-30T11:31:08Z
- **Tasks:** 3/3
- **Files modified:** 11 (4 new test files, 5 extended test files, 1 source fix, 1 artifact)

## Post-Wave Measurement (full census, both roots, slow included)

- **Census:** 1380 passed / 7 skipped (all allowlisted) / 0 failed / exit 0 — 896s
- **Suite coverage:** **85.89%** (6,355 covered / 2,044 missing on 7,399 stmts) — was 79.90% after wave 2
- **mcp-area missing sum: 6 (gate ≤ 110 — passed with 104 lines of slack; research residual estimate was 69, and the +41 slack proved unnecessary)**
  - server.py 2 · start_server.py 4 · client.py / config_manager.py / config_validators.py / model_manager.py / __init__.py 0 each
- `scripts/audit_skips.py` exit 0 on the census junit; zero new skips introduced
- Fast mcp leg at task boundaries: 181 → 267 passed, exit 0 each time
- Pragma budget: still exactly 3 occurrences under `dnallm/`; `pyproject.toml` coverage/pytest config untouched; no stray pdf/log artifacts in git status; no `logs/mcp_server.log` at the repo root

## Accomplishments

- The MCP seam is proven end to end in memory: the official `streamablehttp_client` drives the real `FastMCP.streamable_http_app()` through `httpx.ASGITransport` with a manually-run lifespan — initialize, list_tools, call_tool, and the terminating session DELETE all observed with zero sockets (research Pattern 1 realized as 4 tests, ~4s)
- The complete 13-tool registration set is pinned by name — including the discovery that wire names carry the leading underscore (`_dna_sequence_predict`, ...), derived from `__name__` through `functools.update_wrapper`; a client calling `dna_sequence_predict` would be calling a tool that does not exist
- Every transport constructor is pinned by shape with patched uvicorn: stdio `app.run(transport="stdio")`, streamable-http Config fields (host/port/path from the config block, `access_log=False`, graceful-shutdown 10) with the app-factory provenance asserted, SSE double `Mount(mount_path, sse_app)` + `Mount("", sse_app)` with custom mount path and log-level lowering, plus all three app-None RuntimeError guards — no port ever bound, and no in-memory SSE protocol attempt (deadlock boundary respected)
- Streaming contracts realized as progress-coroutine assertions: ordered `report_progress` sequences for single (0→25→75→100), batch (i/total + completion message), and multi-model (per-model + aggregation), with the verbatim fault dicts (None mid-batch entry, Nth-call-raise arms, no-models early return) and generic-exception propagation out of the timeout wrapper (only `asyncio.TimeoutError` converts)
- **Rule 1 fix:** `_format_multi_model_results` counted every successful dict-shaped prediction as a failure (`r.get("result") is None` is true for absent keys) — a fully successful multi-model run reported "0 successful, N failed"; failure is now keyed on the explicit `{result: None}` marker entry, with the ordering test as its regression
- ModelManager lifecycle fully covered on real YAML configs: lazy single-load, error/in-flight short-circuits, the executor bridge, aggregate loading with exception capture, predict routing for all three modes, memory-estimate degradation, unload counting
- start_server pinned under tmp_path discipline (mandatory autouse chdir — the tree stayed clean of `mcp_server.log`), argparse rows including both shutdown-after-error paths
- Client SDK corners: all three `_connect` bodies via mocked SDK factories, persistent-session branch, context lifecycle, non-JSON error parse, and ExceptionGroup-wrapped httpx connection failures asserted by leaf flattening (Pitfall 11), plus 20 typed-method executions
- Config managers and validators: invalid-config propagation, disabled-model skip, safe defaults without a server config, both streamable_http branches, reload, dangling multi-model reference flags (runtime guard exercised post-parse because the pydantic validator already rejects them at load), and the six validator rejection pairs

## Task Commits

1. **Task 1: in-memory round trip + transport construction (tracer)** — `dd04d2d` (test)
2. **Task 2: streaming progress contracts + timeout/log wrappers + Rule 1 fix** — `4a8508b` (fix)
3. **Task 3: model manager + start_server + client + config corners + wave-3 re-measure** — `38ea916` (test + artifact)

**Plan metadata:** this commit (docs)

## Files Created/Modified

- `tests/mcp/test_server_transports.py` — NEW, 24 tests
- `tests/mcp/test_server_streaming.py` — NEW, 34 tests
- `tests/mcp/test_model_manager.py` — NEW, 25 tests
- `tests/mcp/test_start_server.py` — NEW, 8 tests
- `tests/mcp/test_client_sdk.py` — 37 → 73 collected tests (extended)
- `tests/mcp/test_mutagenesis_tool.py` — 12 → 16 collected tests (extended, objective-required)
- `tests/mcp/test_interpret_tool.py` — 14 → 18 collected tests (extended, objective-required)
- `dnallm/mcp/tests/test_config_manager.py` — 11 → 21 collected tests (extended)
- `dnallm/mcp/tests/test_config_validators.py` — 7 → 16 collected tests (extended)
- `dnallm/mcp/server.py` — Rule 1 fix in `_format_multi_model_results` (5 lines)
- `.planning/phases/03-test-authoring-to-90-coverage/coverage-wave3-missing.txt` — re-ranked worklist for wave 4

## Decisions Made

- Tool-name assertion pins the underscore-prefixed wire names (what the protocol actually exposes) rather than the pretty names passed to `_with_timeout_wrapper` — registration completeness is pinned against reality, not intent
- SSE construction goes through `start_server(transport="sse")` so the dispatch line executes, while remaining a pure construction test
- Client connection-failure groups are built from the `exceptiongroup` backport (py310-portable; ruff-clean against target py310) and flattened with the existing `_network_skip._network_leaves` helper — the assertion survives anyio's re-wrapping
- `test_logging.py` already covered the `_structured_log` json branch; the two new field-assembly tests pin it from the streaming seam as the plan requested, accepting the small duplication

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Multi-model success counter classified every dict prediction as a failure**
- **Found during:** Task 2 (multi-model ordering test)
- **Issue:** `_format_multi_model_results` used `r.get("result") is not None` as the success predicate; raw prediction dicts carry no `result` key, so `.get()` returned None and every successful multi-model run reported "0 successful, N failed" in both the progress message and the result counts
- **Fix:** failure is now keyed on the explicit `{"error": ..., "result": None}` marker entry that `_predict_with_multiple_models` constructs (key-presence check), with `successful = len(results) - failed`
- **Files modified:** dnallm/mcp/server.py
- **Verification:** the ordering test (2 dict results → "2 successful, 0 failed") failed before the fix and passes after; full mcp roots and census green
- **Committed in:** 4a8508b

### Objective-required coverage beyond the plan's task text (03-02 precedent)

- The eight timeout-wrapped tool bodies (server.py:426-714, ~56 statements) were assigned to no task but are arithmetically required by the ≤110 gate → covered in `test_server_streaming.py` on the shared mock-server seam
- The mutagenesis/interpret corner arms (~11 statements) → extended `tests/mcp/test_mutagenesis_tool.py` and `tests/mcp/test_interpret_tool.py` (two files beyond the plan's `files_modified` list — same extension pattern the plan itself uses for the config files)
- `server.main()` (~55 statements) and `initialize_mcp_server` → CLI rows in `test_server_transports.py`

### Verify-command adjustments (no code impact)

- **`grep -c "::"` tripwires:** pytest 9.1.1's `--collect-only -q` emits no `::` separators (same as waves 1-2); the equivalent "N tests collected" summary lines were used (transports 24 ≥ 10; streaming 34 ≥ 12)
- **`assert not pathlib.Path('logs').exists()`:** unsatisfiable by construction — `DNALLMLogger._setup_handlers` (dnallm/utils/logger.py:57-60) creates `logs/dnallm.log` at the pytest launch cwd at import time on every suite run, before any test executes (empirically reproduced: delete `logs/`, run the suite, it reappears containing only `dnallm.log`). The gate's intent (T-3-08 / Pitfall 5) was enforced as "no `logs/mcp_server.log` at the repo root" (passed) plus the git-status stray-artifact grep (passed); the import-time sink itself is out of scope (pre-existing library behavior) and is recorded in `deferred-items.md`

---

**Total deviations:** 1 auto-fix (Rule 1) + 3 objective-required coverage placements + 2 verify-command substitutions
**Impact on plan:** No scope creep — every addition maps to uncovered statements inside the plan's own area gate; no new dependencies; pragma budget intact at 3; allowlist untouched.

## Accepted-Uncovered Residual Ledger (mcp area, documented — never pragma'd)

| File | Missing | Lines | Justification |
|------|---------|-------|---------------|
| dnallm/mcp/server.py | 2 | 1286, 2065 | 1286 is a defensive `raise ValueError` behind an earlier same-condition validation return (unreachable); 2065 is `main()` under the `if __name__ == "__main__":` guard |
| dnallm/mcp/start_server.py | 4 | 11, 13-14, 126 | 11/13-14 are the direct-script ImportError import fallback (`sys.path.insert` + absolute re-import — reachable only when executed outside the installed package); 126 is the `__main__` guard |

The anticipated "SSE live-protocol residual" did not materialize as dnallm lines: the SSE protocol handlers live inside the MCP SDK, not in `dnallm/mcp/`; `_start_sse_server` is fully covered construction-wise, and live SSE protocol behavior remains exercised only by the existing typed network skips (which skip cleanly without a live server, as the census allowlist shows).

## Known Stubs

None — every new test asserts observable behavior (protocol results, ordered call sequences, call counts, verbatim error shapes, constructed field values); no placeholder logic introduced.

## Issues Encountered

- First draft of the in-memory session helper omitted the Starlette lifespan (research Pitfall 3) and failed with stream/session errors — fixed by wrapping the transport in `router.lifespan_context`; the verify loop caught it before commit
- `logs/` appearing at the repo root after runs was traced to the library's import-time sink, not the start_server tests (see deviation above)

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 3 (mcp) complete at 6/110; wave 4 should start from `coverage-wave3-missing.txt` — the remaining ranked gaps are datahandling/data.py (~324 missing) and cli (225) / utils (transformers_compat 45, logger 39) / tasks-metrics (31) / finetune trainer (81) / configuration (9)
- Suite at 85.89%; gap to the 90.5% target = 343 lines (2,044 missing − 1,701 allowed at the 7,399-statement denominator)
- Suite runtime: census 896s (was 856s; +144 tests), fast mcp leg ~67s; no CI timeout risk. Note: `test_timeout.py` contributes a fixed ~60s per census via two tests that wait out the full 30s default timeout — recorded in deferred-items.md
- Windows ledger: one deviation entry appended (the unsatisfiable logs/ plan gate, so waves 4-5 plans do not repeat it)

---
*Phase: 03-test-authoring-to-90-coverage*
*Completed: 2026-09-30*

## Self-Check: PASSED

All 6 created files exist on disk; all three task commits (dd04d2d, 4a8508b, 38ea916) present in history; commits measured from the plan ledger (440b217 -> 38ea916 = 3).
