# Phase 3: Test Authoring to >90% Coverage - Research

**Researched:** 2026-09-30
**Domain:** pytest test authoring against an existing Python ML codebase; MCP SDK in-memory testing; mock/fault-injection strategy
**Confidence:** HIGH (all load-bearing claims verified live against the installed environment this session; estimates marked separately)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Sizing**
- Single phase, ONE PLAN PER WAVE (~5 plans): models → mcp → inference → datahandling/finetune → cli/compat — the wave structure segments execution; fresh executor context per plan preserves fidelity; NO /gsd-phase split, NO roadmap renumbering
- Ordering authority: the ranked worklist (biggest-gap-first), NOT the ROADMAP's illustrative wave order — close inference's 1,505 before models' 1,210 if plan composition allows; wave dependency edges must stay acyclic
- Coverage re-measured after EACH wave (one command per Phase-1 tooling); course-correct early
- Landing target: >90.5% (buffer so minor refactors don't immediately trip the Phase-4 gate)

**MCP wave research (recorded STATE flag)**
- Run a dedicated `gsd-plan-phase 3 --research-phase` pass BEFORE full planning — verifies transport/streaming/timeout patterns against the INSTALLED `mcp` 1.30.0 (the `server.py:1718+` seam) — **THIS RUN; findings below**
- Transport tests: in-memory client/server with mocked transport (established `test_server_integration.py` pattern); NO live-network transport tests beyond existing typed network skips
- Streaming generator tests: consume generators to exhaustion, assert yielded sequence + final status, with fault-injection mid-stream

**Models-wave mock strategy**
- Mock at the handler's own load calls (AutoModel.from_pretrained etc.) — exercises real dispatch/argument logic with lightweight fakes; no real tiny models
- Dispatch-chain coverage via sentinel fault-injection matrix (per-family selection + patch-all-but-one fall-through to generic loading)
- Tokenizer-fallback chain via staged failures (AutoTokenizer → PreTrainedTokenizerFast → DNAOneHotTokenizer), asserting which tier served

**Verification discipline & pragmas**
- Every new test carries ≥1 observable-behavior assertion (TEST-06) — plan acceptance criteria + code review enforce; NO new enforcement tooling (framework ban)
- `# pragma: no cover` budget held at the recorded baseline of 3; additions require written justification in the wave SUMMARY
- Environment-bound branches (CUDA-only etc.): mock-based tests where behavior is assertable; otherwise documented as accepted-uncovered in the wave SUMMARY — never pragma'd
- Wave definition of done: area's ranked gaps closed AND full fast leg passes AND `scripts/audit_skips.py` exits 0 (allowlist absorbs any new intentional skips first)

### Claude's Discretion
Test names, file organization within the tests/ mirror, mock helper shapes — per codebase conventions.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEST-01 | Tests for `models/model.py` + `special/*` (dispatch-chain fault-injection, retry/reason-classification branches, tokenizer fallback chain) | Retry fault-injection matrix (see Code Examples); dispatch chain mapped post-FIX-02 at `dnallm/models/model.py:772-878` with handler-by-handler missing lines; fallback chain in `dnallm/models/tokenizer.py:256-315` with 4-stage failure matrix |
| TEST-02 | Tests for `mcp/server.py` (transports, streaming generators, timeout-wrapper error paths) | In-memory client/server pair VERIFIED working against installed mcp 1.30.0 (streamable-http); SSE in-memory deadlock characterized — construction-level SSE tests instead; streaming tools are progress-reporting coroutines (see Patterns) |
| TEST-03 | Tests for `inference/*` (engine paths, logits→predictions, interpret/mutagenesis/benchmark) | File-by-file missing-line map; captum on tiny real CPU models measured at ~28ms/run; existing shallow coverage inventoried to avoid duplication |
| TEST-04 | Tests for `datahandling`/`finetune` (dataset loading/tokenization/augmentation, trainer wiring) | `data.py` method map — local file round-trips (csv/tsv/json/parquet/fasta/txt/pkl) are cheap real-behavior tests via tmp_path; trainer wiring mockable at Trainer boundary |
| TEST-05 | Tests for `cli/` + utils compat shims (CliRunner; `transformers_compat` as behavior contract, not line completion) | click.testing.CliRunner available (click installed); transformers_compat contract surface fully mapped + idempotency verified live |
| TEST-06 | Coverage >90% on the agreed denominator, every new test ≥1 observable-behavior assertion; pragma budget held at baseline (3) | Arithmetic: need +3,292 covered lines (82.4% of the 3,993 missing); per-area achievable estimates with ~183 lines of slack at 90.5%; exactly 3 pragmas located verbatim |
</phase_requirements>

## Summary

The gap to >90.5% is 3,292 statements on a 7,383-statement denominator (45.92% baseline, 3,390 covered at the Phase-1 census; ~3,993 missing concentrated in 43 files). The work decomposes into five wave areas plus two small orphans (tasks 51 + configuration 9 lines; wave-assigned to the cli/compat wave — see Open Questions (RESOLVED) #1). Every area is testable with the existing stack — no new frameworks — using three verified mechanisms: (1) mock-at-the-load-call for model dispatch (with sys.modules stubbing for the seven special families whose heavy deps are not installed), (2) a LIVE-VERIFIED in-memory MCP client/server pair for `mcp/server.py` connecting the official `streamablehttp_client` to `FastMCP.streamable_http_app()` via `httpx.ASGITransport` (full protocol round trip in 0.65s wall, no sockets), and (3) real-tiny-torch-module tests for captum interpret and the seven `head.py` classes (~28ms per attribution).

The recorded research flag is resolved with two decisive corrections to the CONTEXT's wording. First, the `server.py:1718+` seam is COMPATIBLE with installed mcp 1.30.0 — `sse_app()`, `streamable_http_app()`, and `run(transport=...)` all exist and return/accept what DNALLM expects [VERIFIED: live SDK probe]. But in-memory SSE protocol tests are NOT viable: the SSE GET stream never yields response headers under `httpx.ASGITransport` (deadlock reproduced twice), so SSE coverage comes from construction-shape tests plus the existing typed live-network skips. Second, the "streaming generators" are not generators at all — they are coroutines that report progress via `context.report_progress(progress, total, message)` and return final dicts; the decision's intent (ordered sequence + final status + mid-stream fault injection) is fully implementable by asserting the ordered `report_progress` call list plus the returned dict, with `side_effect` sequences injecting per-item failures.

**Primary recommendation:** Plan five waves in ranked-worklist order (inference first at 1,505 missing, then models 1,210, mcp 449, datahandling/finetune 441, cli/compat 328 + the 60-line orphans), re-measure coverage before wave 1 (Phase 2 added ~25 tests and moved the numbers), and hold the ~183-line slack at the 90.5% target by treating models-special sys.modules stubbing depth as the course-correct lever the per-wave re-measurement will expose.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Unit tests for library modules | Test tier (`tests/`, `dnallm/mcp/tests/`) | — | All production code is library-side; no UI/backend split exists |
| MCP transport verification | In-process ASGI (httpx.ASGITransport) | Mocked uvicorn construction | Full protocol round trip proven in-memory for streamable-http; SSE construction-only (deadlock) |
| Fault injection (download retry, dispatch) | Test tier via `unittest.mock.patch` at call sites | sys.modules stubs for absent optional deps | Established repo idiom (`tests/models/test_model.py`) |
| Real numerical behavior (heads, captum, logits) | Test tier with tiny real torch modules on CPU | — | Mocks cannot exercise autograd/shape semantics captum and heads need |
| Coverage measurement | Tooling tier (`pytest --cov` + `coverage json`, config in pyproject) | — | Phase-1 command of record; no new tooling allowed |
| Skip discipline | `tests/expected_skips.yaml` + `scripts/audit_skips.py` | CI junit matching | Fail-closed audit from FIX-03; wave DoD depends on it exiting 0 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | 9.1.1 (installed) | test runner | Project constraint — no new frameworks |
| pytest-cov | 7.1.0 | coverage activation (`--cov` only; scope from pyproject) | Phase-1 route A decision |
| coverage | 7.16.2 | `coverage report -m` / `coverage json` re-measurement per wave | Phase-1 tooling of record |
| pytest-asyncio | 1.4.0, `--asyncio-mode=auto` | async tests (MCP) with NO marker needed | Existing config |
| pytest-timeout | 2.4.0 (`--timeout=300`) | per-test backstop | Existing config |
| unittest.mock | stdlib | patch/AsyncMock/MagicMock at load calls | Established idiom |
| click.testing.CliRunner | click (installed) | CLI invocation tests for `dnallm/cli/*` | Official click testing utility |
| httpx | 0.28.1 | `ASGITransport` for the in-memory MCP pair | Installed; mcp clients accept `httpx_client_factory` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| captum | 0.9.0 | real attributions on tiny torch modules for `interpret.py` tests | Only where autograd semantics matter (~28ms/run measured) |
| transformers | 5.17.0 | patched-class targets for `transformers_compat` contract tests | Patch already applied at import — assert against real `PreTrainedModel` |
| torch | 2.11.0+cu130 | real tiny modules for `head.py` forwards | Always for head/interpret tests |

**Installation:** none — the no-new-framework constraint is absolute; everything above is installed in `.venv` and on CI.

**Version verification:** performed live via import/importlib.metadata this session (mcp 1.30.0, httpx 0.28.1, captum 0.9.0, transformers 5.17.0, torch 2.11.0+cu130, bitsandbytes 0.50.2, modelscope 1.34.0, optuna 5.0.0, peft 0.21.1, altair 6.3.0) [VERIFIED: live probe].

## Package Legitimacy Audit

> This phase installs NO external packages (PROJECT constraint: no new test frameworks; all tooling already present). No registry checks required. **Packages removed due to SLOP verdict:** none. **Packages flagged as suspicious:** none.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌────────────────────────────────────────────────┐
                    │  Ranked worklist (01-AUDIT-REPORT.md 43 rows)  │
                    │  coverage.json / coverage-term-missing.txt     │
                    └───────────────┬────────────────────────────────┘
                                    │ biggest-gap-first ordering
                                    ▼
   Wave A: inference (1,505) ──► Wave B: models (1,210) ──► Wave C: mcp (449)
        │                            │                          │
        │  mock_model/mock_tokenizer │  patch at handler load   │  in-memory ASGI pair
        │  + tiny real torch modules │  + sys.modules stubs     │  + AsyncMock contexts
        ▼                            ▼                          ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Per-wave re-measure: pytest --cov → coverage json → re-rank        │◄─┐
   └───────────────┬─────────────────────────────────────────────────────┘  │
                   ▼                                                            │
   Wave D: datahandling/finetune (441) ──► Wave E: cli/compat (328+60) ───────┘
        (tmp_path file round-trips)         (CliRunner + compat contract)
                   │
                   ▼
   Final gate run: full suite, both roots, slow included, config-only --cov
   → percent_covered > 90.5 on the 7,383-statement denominator
   → pragma count == 3 · audit_skips.py exit 0 · every test has ≥1 behavior assert
```

### Recommended Project Structure
```
tests/
├── inference/
│   ├── test_inference.py          # extend: engine branches, embeddings, generate, scoring
│   ├── test_mutagenesis.py        # NEW (269-line gap, zero existing tests)
│   └── test_interpret.py          # NEW (261-line gap, zero existing tests)
│   └── test_plot.py               # extend (pdf-marker discipline from FIX-04)
├── models/
│   ├── test_model.py              # extend: dispatch matrix, retry matrix (partially exists)
│   ├── test_tokenizer.py          # NEW (fallback chain + DNAOneHotTokenizer, 115 lines)
│   ├── test_head.py               # NEW (7 real torch head classes, 196 lines)
│   ├── test_losses.py             # NEW (12 lines)
│   └── test_special/              # NEW: one file per family or grouped
│       ├── test_crossdna.py       # 249 lines — real torch forwards
│       └── ...                    # evo/borzoi/megadna/... via sys.modules stubs
├── mcp/
│   ├── test_server_transports.py  # NEW: construction + dispatch + in-memory pair
│   ├── test_server_streaming.py   # NEW: progress ordering + mid-stream faults
│   ├── test_model_manager.py      # NEW (55 lines)
│   └── test_start_server.py       # NEW (53 lines, 100% missing)
├── datahandling/test_dna_dataset.py  # extend: file formats, tokenization, augmentation
├── finetune/
│   └── test_trainer.py            # NEW fast unit tests (81 lines; real-model file exists)
├── cli/
│   └── test_cli.py                # NEW (CliRunner; 225 lines across 5 files)
└── utils/
    ├── test_transformers_compat.py # NEW behavior contract (45 lines)
    └── test_logger.py             # NEW (39 lines)
```

### Pattern 1: In-memory MCP client/server pair (VERIFIED WORKING)
**What:** Connect the official MCP client to the FastMCP Starlette app with no sockets.
**When to use:** `mcp/server.py` tool-registration + protocol round-trip tests (TEST-02).
**Verified facts:** on installed mcp 1.30.0, `streamablehttp_client(url, headers=None, timeout=..., sse_read_timeout=..., terminate_on_close=True, httpx_client_factory=..., auth=None)` accepts a factory; `FastMCP.streamable_http_app()` returns a `Starlette` whose lifespan is `lambda app: self.session_manager.run()` — httpx `ASGITransport` does NOT run lifespans, so drive it manually via `asgi.router.lifespan_context(asgi)` [VERIFIED: live probe — full initialize/list_tools/call_tool/DELETE round trip, exit 0, 0.65s wall].

```python
# Source: live-verified probe against installed mcp 1.30.0 this session
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

def factory(**kwargs):  # mcp passes headers/timeout/auth; replace transport
    kwargs.pop("transport", None)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=asgi_app),
        base_url="http://localhost:8000",  # MUST be localhost/127.0.0.1 — see Pitfall 1
    )

async with asgi_app.router.lifespan_context(asgi_app):  # starts session manager
    async with streamablehttp_client(
        "http://localhost:8000/mcp", httpx_client_factory=factory
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("echo", {"text": "hello"})
```

### Pattern 2: Streaming tools are progress-reporting coroutines (not generators)
**What:** `_dna_stream_predict` / `_dna_stream_batch_predict` / `_dna_stream_multi_model_predict` call `await context.report_progress(progress, total, message)` at stages inside `async with asyncio.timeout(self._tool_timeout_seconds):` and return a final dict [VERIFIED: dnallm/mcp/server.py:724-1148 read this session].
**When to use:** TEST-02 "streaming" tests. The CONTEXT's "consume generators to exhaustion, assert yielded sequence + final status" translates to: await the coroutine with an `AsyncMock` context, assert the ORDERED `report_progress.call_args_list` (e.g. 0→25→75→100 for single predict; per-item `i/total` for batch), then assert the final result dict. Mid-stream fault injection = `side_effect` list on `model_manager.predict_sequence` (raise on the Nth call) or a `None` return mid-batch (the `result is None` branch builds `{"sequence": ..., "result": None, "error": f"Prediction failed for sequence {i + 1}", "index": i}` — verbatim from server.py:935-940).

### Pattern 3: Sentinel fault-injection dispatch matrix (models wave)
**What:** `load_model_and_tokenizer` is a chain: early-return handlers (evo2:773, evo1:778, megadna:786, enformer:799, space:810, borzoi:821 — each `if result is not None: return`), import-gate discards (`_ = _handle_gpn_models(...)` :783, `_ = _handle_omnidna_models(...)` :796), then the guarded first-resolved-wins chain (crossdna:863-874 → dnabert2:875-876 → generic `_load_model_by_task_type`:877-878 — each stage only when the previous left model or tokenizer None), then mutbert/basenji2 tokenizer post-processing (:880-883) [VERIFIED: dnallm/models/model.py:772-901 read this session].
**When to use:** per-family selection tests patch the ONE handler under test to return a sentinel `(Mock(), Mock())` pair and assert it was selected AND that `_load_model_by_task_type` was NOT called; fall-through tests patch all-but-one handler to return None and assert the generic loader served. The FIX-02 regression test `test_load_model_crossdna_result_not_overwritten` (tests/models/test_model.py:487) is the established template.

### Pattern 4: sys.modules stubbing for absent special-family deps
**What:** `evo`, `evo2`, `enformer`, `borzoi`, `seqmodels`, `megadna`, `gpn`, `multiomics` are all NOT installed (ModuleNotFoundError verified live); their imports are function-local inside the handlers, so the handlers currently return None and fall through.
**When to use:** to reach handler bodies, inject `types.ModuleType("evo2")` with shaped fakes (e.g. an `Evo2` class with the attributes the handler reads) via `monkeypatch.setitem(sys.modules, "evo2", fake)` — monkeypatch auto-restores, preventing cross-test contamination. Weight the effort by gap size: crossdna (249) needs no stubs (module-level imports are torch/transformers only — verified), evo (168) needs `evo2`/`evo`/`stripedhyena`/`vortex` stubs, the rest are 14-68 lines each.

### Pattern 5: transformers_compat as behavior contract
**What:** `apply_patches()` (called at module import, server.py of the module: line 223) installs two guarded wrappers on the real `transformers.PreTrainedModel`; idempotency via class flags — verified LIVE this session: after `import dnallm.utils.transformers_compat`, a second explicit `apply_patches()` leaves `PreTrainedModel.get_parameter_or_buffer` object-identical (`before is after` → True) and both `_dnallm_quant_key_patch` / `_dnallm_quant_init_patch` are True on transformers 5.17.0 [VERIFIED: live probe].
**Contract surface to test:** (a) idempotency (identity across second call); (b) patched `get_parameter_or_buffer(dummy_self, "x.weight.absmax")` returns a `_QuantStatProxy` when the original raises AttributeError and a parent is found via `get_parameter`/`get_buffer`; (c) `_QuantStatProxy` forwards attribute reads, forwards regular `__setattr__`, and silently DROPS `_is_hf_initialized` sets (dnallm/utils/transformers_compat.py:209-212 — read verbatim); (d) `_iter_uninitialized_quantized_weights` candidate/marked classification over fake modules; (e) patched `initialize_weights` passthrough when `not marked or not candidates`, bnb-import-failure passthrough, and the swap→original→restore flow with `bitsandbytes.functional.dequantize_4bit`/`quantize_4bit` monkeypatched (bitsandbytes 0.50.2 imports fine on this box — verified).

### Anti-Patterns to Avoid
- **Building a socket server in tests** (uvicorn on a port) for MCP coverage — the in-memory ASGI pair is faster and deterministic; live-network shapes stay in the existing typed-skip files.
- **Hand-rolling skip logic** (`except Exception: pytest.skip`) — use `dnallm/mcp/tests/_network_skip.py::skip_if_unreachable`; the audit is fail-closed.
- **Mocking `model.forward` when a real tiny module works** — heads/captum/losses tests need autograd; Mocks cannot produce gradients.
- **Pragma'ing environment-bound branches** — locked decision: document as accepted-uncovered in the wave SUMMARY instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| In-memory MCP transport | custom ASGI bridge/socket server | `httpx.ASGITransport` + mcp's `httpx_client_factory` + `router.lifespan_context` | Verified working; SSE variant deadlocks — don't reinvent |
| Fake HF models for engine tests | per-test model dummies | `tests/conftest.py` `mock_model` / `mock_tokenizer` fixtures (`.to()` self-return already shaped) | Established, shared, keeps tests short |
| CLI invocation harness | subprocess calls to `dnallm` | `click.testing.CliRunner` | Official; captures output/exit codes; no subprocess-coverage caveat (AUDIT-04) |
| Network-failure detection | broad except → skip | `_network_skip.py` leaf-flattening helper | ExceptionGroup-aware (httpx inside anyio TaskGroups) |
| Skip allowlisting | editing junit post-hoc | append to `tests/expected_skips.yaml` before adding the skip | audit_skips.py fails closed on unmatched skips |

**Key insight:** the repo already encodes every hard-won testing lesson from Phases 1-2 (typed skips, exit-code canary, tmp_path artifacts, sys.modules object-form rebinding); reusing those seams is cheaper and safer than new scaffolding.

## Common Pitfalls

### Pitfall 1: Host-header 421 in in-memory MCP tests
**What goes wrong:** client gets `421 Misdirected Request` before any protocol exchange.
**Why:** mcp 1.30.0 auto-enables `TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"])` when FastMCP's `host` is in `("127.0.0.1", "localhost", "::1")` — DNALLM constructs `FastMCP(name=..., instructions=...)` without host, so the default applies; any other Host header is rejected [VERIFIED: live — reproduced then fixed by switching base_url to `http://localhost:8000`].
**How to avoid:** always use a localhost/127.0.0.1 base_url in the factory. **Warning signs:** 421 status in the first POST.

### Pitfall 2: SSE in-memory deadlock (do not chase)
**What goes wrong:** `sse_client` (or even a raw streaming GET) against `FastMCP.sse_app()` over `httpx.ASGITransport` hangs forever — response headers never arrive; faulthandler shows the loop idle in select [VERIFIED: live — reproduced twice, including a raw GET with no mcp client].
**How to avoid:** full SSE protocol tests are out of reach in-memory; cover `_start_sse_server` (server.py:1716-1772) by patching `uvicorn.Config`/`uvicorn.Server` and asserting the assembled Starlette routes (`Mount(mount_path, sse_app)`, `Mount("", sse_app)`), config fields, and `.run()` invocation. SSE protocol behavior remains covered by the existing typed live-network skips.

### Pitfall 3: Missing lifespan for streamable-http
**What goes wrong:** requests fail with session errors — `streamable_http_app()`'s lifespan (`session_manager.run()`) never started.
**How to avoid:** wrap the whole client exchange in `async with asgi_app.router.lifespan_context(asgi_app):` (verified pattern). **Warning signs:** "session manager not running" style errors.

### Pitfall 4: Literal reading of "streaming generators"
The tools are coroutines with `report_progress` side effects, not Python generators (the only `yield` in server.py is the lifespan at :1615 — verified by grep). Write progress-sequence assertions, not generator exhaustion.

### Pitfall 5: `start_server.py` tests dirty the repo
`setup_logging` does `logger.remove()` then adds a file handler under `logs/mcp_server.log` (start_server.py:34-43) — tests must `monkeypatch.chdir(tmp_path)` to avoid creating `logs/` in the repo tree (the twice-run tree-clean gate from Phase 2 is the tripwire).

### Pitfall 6: Retry tests that actually sleep
The retry loop calls `time.sleep(1)` (model.py:372). Established idiom: `with patch("time.sleep")` — tests/models/test_model.py:50 already does this; every new retry-matrix test must too, or the fast leg gains seconds.

### Pitfall 7: Pragma budget is exactly 3 — and all in one file
[VERIFIED: grep this session] `dnallm/utils/transformers_compat.py:87` `except Exception:  # pragma: no cover - transformers not installed`, `:156` same, `:181` `except Exception:  # pragma: no cover - bitsandbytes not installed`. Any addition requires written justification in the wave SUMMARY (locked decision). Since transformers AND bitsandbytes are installed here, those three lines are already excluded from the denominator's missing count (16 excluded lines in coverage.json).

### Pitfall 8: MagicMock pickling in benchmark tests
The audit recorded `dill PicklingWarning`s from `tests/benchmark/test_benchmark.py` (MagicMock pickling). New benchmark/trainer tests that cross a pickling boundary should use simple fakes or `SpeccableMock`-style objects rather than bare MagicMock attributes that get pickled.

### Pitfall 9: Phase-1 coverage artifacts are pre-Phase-2
`coverage.json` / `coverage-term-missing.txt` predate FIX-01..04 (fast leg went 602→622 passed; collection is now 650 tests — verified this session). The ranking is still directionally correct, but the covered baseline has drifted UP. Re-measure before wave 1 (the locked per-wave re-measure decision naturally covers this if wave 1 starts with it).

### Pitfall 10: sys.modules stub leakage
Injected fake `evo2`/`evo`/`enformer` modules leak between tests if set via `sys.modules[...] =` directly. Always use `monkeypatch.setitem(sys.modules, ...)` so pytest restores the real (absent) state.

### Pitfall 11: anyio ExceptionGroup escape in client tests
MCP client context managers fail through httpx inside TaskGroups — plain `except httpx.ConnectError` misses. The `_network_skip._network_leaves` flattening exists for this; new client tests that intentionally exercise error paths should expect `ExceptionGroup` semantics.

### Pitfall 12: `asyncio.timeout` is 3.11+
The streaming tools use `asyncio.timeout` (with `# type: ignore[attr-defined]`) — unavailable on Python 3.10 despite `requires-python >=3.10`. CI tests 3.11/3.12/3.13 so tests will pass everywhere they run; record, do not fix (bug-fix scope discipline).

## Code Examples

### Retry/reason-classification fault-injection matrix (TEST-01)
```python
# Source: dnallm/models/model.py:343-377 (read verbatim this session)
#         while True:
#             if cnt >= max_try: break
#             cnt += 1
#             try:
#                 status = downloader(model_name, revision=revision)
#                 if status != "incomplete": ... break
#             except Exception as e:
#                 if "connection" in str(e):
#                     reason = "unstable network connection."
#                 elif "not found" in str(e).lower():
#                     reason = "repo is not found."; break      # NO retry
#                 elif "response [404]" in str(e).lower():
#                     reason = "repo is not existed."; break    # NO retry
#                 else:
#                     reason = str(e)
#                     if "no revision" in reason.lower(): revision = None
#                 logger.warning(...); time.sleep(1)
#             ...
#             if status == "incomplete":
#                 raise ValueError(f"Model {model_name} download failed.")
```

| Injection (downloader fake) | Asserts which classification | Observable behavior |
|---|---|---|
| returns `"/cache/path"` first call | success | returns path; downloader called once; sleep never called |
| raises `Exception("connection reset")` then succeeds | "unstable network connection." | retries; succeeds on 2nd; `time.sleep` called once |
| raises `Exception("Repository not found")` | "repo is not found." | downloader called EXACTLY once (break, no retry); ends in `ValueError("Model ... download failed.")` |
| raises `Exception("HTTP Response [404] invoked...")` (no "not found" substring) | "repo is not existed." | same single-call shape as above — NOTE: message must contain "response [404]" (case-insensitive) but NOT "not found", or the earlier branch wins |
| raises `Exception("no revision found: xyz")` repeatedly, `max_try=2` | else-branch + revision reset | `downloader` 2nd call receives `revision=None`; exhausts → ValueError |
| raises `Exception("connection ...")` every call, `max_try=3` | exhaustion | downloader called 3×; `pytest.raises(ValueError, match=r"download failed")` |
| returns `"incomplete"` every call | no-exception loop | exhausts max_try WITHOUT sleeping (sleep is inside except); ValueError |

`tests/models/test_model.py::TestDownloadModel` already covers rows 1, 2, 3-partial, 5-partial, 6 — the matrix completion targets the 404 branch, the revision reset, the "incomplete"-return loop, and per-row call-count/sleep assertions.

### Tokenizer-fallback staged-failure matrix (TEST-01)
```python
# Source: dnallm/models/tokenizer.py:286-315 (read verbatim this session)
# tier 1: auto_tokenizer_cls.from_pretrained(...)  -> return on success
# tier 2: from transformers import PreTrainedTokenizerFast
#         PreTrainedTokenizerFast.from_pretrained(...) -> warning "loaded fast tokenizer from tokenizer.json."
# tier 3: logger.warning(f"All tokenizer loading failed for {model_name}; using DNAOneHotTokenizer.")
#         return DNAOneHotTokenizer()
```
- Tier-1 success via `auto_tokenizer_cls` sentinel; tier-1 success via the default `from transformers import AutoTokenizer` path (patch the class attr — function-local import resolves the same object).
- Tier-1 fail → tier-2 success: assert returned sentinel AND the `"loaded fast tokenizer"` warning.
- Both fail → tier-3: assert `isinstance(tok, DNAOneHotTokenizer)` AND the `"using DNAOneHotTokenizer"` warning.
- `DNAOneHotTokenizer` itself (lines 13-253) is pure Python — `vocab_size` property returns `6` (verbatim, tokenizer.py:201-202); cover `__call__` tensor/dict shapes, `convert_tokens_to_ids` str-vs-list overloads, encode/decode/batch_decode round-trips, `save_pretrained`/`from_pretrained` round-trip via `tmp_path`.

### Timeout-wrapper error contract (TEST-02)
```python
# Source: dnallm/models/../mcp/server.py:323-337 (read verbatim this session)
# return {
#     "isError": True,
#     "content": [{"type": "text", "text": f"Timeout after {self._tool_timeout_seconds}s"}],
#     "error_type": "timeout",
#     "timeout_seconds": self._tool_timeout_seconds,
#     "tool_name": tool_name,
#     "suggestion": "Try with fewer positions, smaller sequence, or increase timeout in config",
# }
```
`tests/mcp/test_timeout.py` already asserts this shape for wrapper + all three streaming tools. Gaps the new tests must close: (a) generic non-timeout exceptions PROPAGATE out of the wrapper (only `asyncio.TimeoutError` is caught — a `ValueError` from a tool must raise, not return a dict); (b) `_structured_log` JSON format branch (`self._log_format == "json"`) and its field assembly (server.py:370-385); (c) the streaming `except Exception` branches (server.py:857-876, 1004-1023, 1127-1148) via mid-stream `side_effect` raises.

### Transport dispatch and construction (TEST-02)
```python
# Source: dnallm/mcp/server.py:1700-1714 (read verbatim this session)
# valid_transports = ("stdio", "sse", "streamable-http")
# if transport not in valid_transports:
#     raise ValueError(f"Invalid transport: {transport!r}. Must be one of: {valid_transports}")
# if transport == "sse": self._start_sse_server(host, port)
# elif transport == "streamable-http": self._start_http_server(host, port)
# else: self._start_stdio_server()
```
- invalid transport → `pytest.raises(ValueError, match=r"Invalid transport")` (also covers the uninitialized-server RuntimeError at :1689-1690).
- `_start_stdio_server` → patch `app.run`, assert called with `transport="stdio"` (FastMCP.run signature verified: `run(self, transport: Literal['stdio','sse','streamable-http'] = 'stdio', mount_path: str | None = None) -> None`).
- `_start_http_server` → patch `uvicorn.Config`/`uvicorn.Server`; assert host/port/http_path from `streamable_http` config block, `access_log=False`, `timeout_graceful_shutdown=10`; assert app came from `app.streamable_http_app()`.
- `_start_sse_server` → same uvicorn mock; assert double Mount and `log_level.lower()` handling; the `sse_app is None → RuntimeError` guard.

### Real-behavior test cost reference (measured this session)
- In-memory streamable-http round trip (initialize + list_tools + call_tool + DELETE): **0.65s wall including imports** — keep unmarked (fast leg).
- Captum `LayerIntegratedGradients` on a 12-token tiny CPU module, n_steps=8: **~28ms/run**; import cost ~0.68s (already paid at suite start). Interpret tests: use tiny REAL torch modules (autograd required) with `(batch, n_classes)` outputs — scalar-summed outputs crash captum's target selection (IndexError, reproduced live).

## Runtime State Inventory

> Not a rename/refactor/migration phase — test authoring only. SKIPPED per protocol. (One adjacent note: new tests must not create state — Pitfall 5 covers the `logs/` trap in start_server tests, and FIX-4's tmp_path discipline covers artifact-writing tests.)

## Common Pitfalls (consolidated budget view)

See the twelve pitfalls above; the planner should turn Pitfalls 1-3, 5-7 into explicit plan verification steps.

## Coverage Reachability Arithmetic (TEST-06 budget)

[VERIFIED: recomputed from `.planning/phases/01-harness-integrity-measured-baseline/coverage.json` this session]

Denominator 7,383 statements; 3,390 covered (45.92%); 3,993 missing; 16 excluded. Target 90.5% ⇒ ceil(0.905 × 7,383) = 6,682 covered ⇒ **+3,292 net lines (82.4% of all missing)**; ≤701 may remain uncovered.

| Area | Missing | Dominant files (missing/total stmts) | Est. achievable [ASSUMED] | Likely residual |
|------|---------|--------------------------------------|---------------------------|-----------------|
| inference | 1,505 | inference.py 523/826 · plot.py 332/740 · mutagenesis 269/297 · interpret 261/291 · benchmark 120/258 | ~1,325 (~88%) | deep `generate`/`scoring` branches, altair spec corners |
| models-core | 557 | model.py 234/462 · head.py 196/221 · tokenizer.py 115/149 · losses.py 12/18 | ~505 (~91%) | DNALLMforSequenceClassification exotic pooling corners |
| models-special | 653 | crossdna 249/270 · evo 168/194 · borzoi 68/76 · megadna 56/65 · 8 more ≤30 | ~520 (~80%) | stub-shape limits in evo/deep handler bodies |
| mcp | 449 | server.py 250/522 · client 58/144 · model_manager 55/176 · start_server 53/53 · config_manager 27/103 · config_validators 6/138 | ~380 (~85%) | SSE live-protocol lines (deadlock-bound), client error corners |
| datahandling | 360 | data.py 359/732 · dataset_auto 1/1 | ~325 (~90%) | exotic format/header combos |
| finetune | 81 | trainer.py 81/183 | ~70 | optuna/megatron-adjacent wiring |
| cli | 225 | cli.py 108/152 · mutagenesis 71/93 · inference 19/30 · config_generator 14/17 · train 13/24 | ~205 | interactive generator paths |
| utils | 103 | transformers_compat 45/90 · logger 39/109 · sequence 7 · support 7 · cuda_compat 4 · training_plots 1 | ~90 | cuda_compat preload lines (env-bound, mockable in principle) |
| tasks | 51 | metrics.py 51/271 | ~45 | AUROC/seqeval edge branches |
| configuration | 9 | configs.py 9/254 | 9 | — |
| **Total** | **3,993** | | **~3,475 (~87%)** | **~518** |

Projected landing: 3,390 + ~3,475 = ~6,865 / 7,383 ≈ **93.0%** ⇒ ~183 lines of slack above the 6,682 needed for 90.5%. The slack is thin: if models-special stubbing yields only ~60% instead of ~80% (−130 lines) AND inference's plot corners resist (−100), the target is missed. Mitigations already locked: per-wave re-measurement course-corrects early; environment-bound lines are documented, not pragma'd, so they consume residual budget honestly. **Environment-bound reality:** CI is `ubuntu-latest` CPU-only (verified in `.github/workflows/ci.yml`) while local dev has CUDA (GB10) — CUDA-only lines (e.g. `_get_device` cuda return) may cover locally but NOT in the CI gate environment; prefer `torch.cuda.is_available` patching so coverage holds on CPU. MPS/XPU branches are mock-only everywhere.

## Suite Runtime Impact

- Current: fast leg 622 passed / 78s; full leg 894s; 650 tests collected at HEAD (verified this session).
- Expected new tests: ~250-400 (3,292 lines at a conservative 8-13 behavior-covered lines/test).
- Per-test cost anchors (measured): mocked unit tests 30-150ms; captum interpret ~30-300ms; in-memory MCP round trip ~0.3-0.5s of test body; CliRunner 50-200ms; datahandling file round-trips 50-300ms.
- Estimate: +35-60s on the fast leg (→ ~115-140s), +same on full (→ ~930-955s). Acceptable; no CI timeout risk with the 300s per-test backstop.
- **Slow-marking strategy: unchanged.** New tests must not need network or real model downloads (all mocking strategies verified above), so none warrant `slow`. The only marker interaction: plot tests that write PDFs join the class-level `pdf` marker discipline (9 existing classes; FIX-04) and write to `tmp_path`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Per-area achievable percentages (80-91%) | Reachability arithmetic | Under-delivery vs 90.5%; per-wave re-measure catches it — the swing lever is models-special stub depth |
| A2 | ~250-400 new tests at 8-13 covered lines/test | Runtime impact | Test-count estimate off ±40%; runtime conclusion robust either way |
| A3 | `router.lifespan_context` remains the right manual-lifespan entry on Starlette 1.7.0 across the milestone | Pattern 1 | Verified working now; if a Starlette upgrade lands mid-phase, tests break loudly and the pattern needs a one-line adjustment |
| A4 | tasks/(51) + configuration/(9) orphans should ride the cli/compat wave (planner's call) | Worklist | Mis-assignment only affects plan sizing, not reachability |
| A5 | Interpret tests use tiny real torch modules (within "Claude's discretion" mock-helper shapes) | TEST-03 | Mocks break autograd — the measured 28ms anchor says real modules are affordable |
| A6 | CI gate (Phase 4) measures on CPU runners, so CUDA-only lines stay uncovered there | Reachability | If GATE-02 ever runs on GPU runners, a few more lines cover — upside only |
| A7 | Post-Phase-2 covered-baseline drift is upward (more covered than 3,390) | Pitfall 9 | If somehow downward, the +3,292 requirement grows; the pre-wave-1 re-measure resolves this cheaply |

## Open Questions (RESOLVED)

All three questions were resolved at plan time (2026-09-30); each records the adopting plan and task below.

1. **Wave assignment of the 60-line orphans (tasks/metrics 51 + configuration/configs 9)** — RESOLVED
   - What we know: no wave in the locked structure names them; they must land somewhere.
   - Recommendation: attach to the cli/compat wave (smallest marginal planning cost) or distribute to the nearest domain wave.
   - Resolution: adopted the first option — the orphans ride the cli/compat wave as extensions of tests/tasks/test_metrics.py (51 lines) and tests/configuration/test_configs.py (9 lines) in 03-05 Task 2.
2. **Depth of evo handler stubbing** — RESOLVED
   - What we know: 168 missing lines behind `evo2`/`evo`/`stripedhyena`/`vortex` function-local imports; stubs must satisfy real attribute reads.
   - Recommendation: timebox; if the stub shapes exceed ~1 plan-task effort, take the documented-uncovered residual and bank the slack elsewhere (inference/models-core are higher-yield per effort).
   - Resolution: the timebox is adopted verbatim in 03-02 Task 3 — if it fires, a per-file residual ledger is mandatory in 03-02-SUMMARY.md and the slack banks into models-core.
3. **Whether `mcp/server.py` health/mutagenesis/interpret tool bodies (missing lines 487-714) need the in-memory pair or AsyncMock model_manager** — RESOLVED
   - What we know: both work; the pair additionally proves registration/schema correctness.
   - Recommendation: one in-memory smoke test asserting ALL 13 registered tools are listable, plus per-tool AsyncMock unit tests for behavior branches.
   - Resolution: both, as recommended — 03-03 Task 1's in-memory smoke asserts the complete registered-tool set (enumerated from source; research count 13) and 03-03 Task 2's AsyncMock units pin the per-tool behavior branches.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest / pytest-cov / coverage | all waves | ✓ | 9.1.1 / 7.1.0 / 7.16.2 | — |
| mcp SDK | MCP wave | ✓ | 1.30.0 (pin `>=1.3.0,<2`) | — |
| httpx (ASGITransport) | in-memory pair | ✓ | 0.28.1 | — |
| click (CliRunner) | cli wave | ✓ | installed | — |
| captum | interpret tests | ✓ | 0.9.0 | mock-based (loses autograd fidelity) |
| torch CPU | heads/interpret | ✓ | 2.11.0+cu130 | — |
| transformers | compat contract | ✓ | 5.17.0 | — |
| bitsandbytes | compat swap/restore | ✓ | 0.50.2 | monkeypatch bnb.functional |
| modelscope | data tests (presets) | ✓ | 1.34.0 | mock |
| evo/evo2/enformer/borzoi/gpn/megadna/multiomics | special handlers | ✗ (not installed — verified) | — | sys.modules stubs (Pattern 4) |
| GPU (local) / CPU (CI) | env-bound branches | local ✓ / CI ✗ | — | mock `torch.cuda.is_available` |

**Missing dependencies with no fallback:** none — all absent packages are optional-by-design with the stubbing strategy.
**Missing dependencies with fallback:** the seven special-family packages (stub via monkeypatch).

## Security Domain

`security_enforcement: true` (level 1). This phase authors tests only — no new production attack surface, no new deps, no network listeners beyond in-process ASGI apps that bind nothing.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | unchanged (MCP auth settings untouched; tests use no-auth apps mirroring production default) |
| V3 Session Management | no | StreamableHTTPSessionManager behavior exercised in-memory, not modified |
| V4 Access Control | no | — |
| V5 Input Validation | yes (tests only) | assertions validate existing ValueError/Pydantic rejection paths (invalid transport, bad config, invalid sequence) rather than bypassing them |
| V6 Cryptography | no | — |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Tests binding real sockets / 0.0.0.0 | Information Disclosure | none — in-memory ASGI transport binds nothing; any accidental real-port test would be a review finding |
| Hardcoded credentials in new tests | Spoofing | ruff `S` rules already selected; per-file-ignores for tests are narrow (S101/S106/S108 in MCP tests only) |
| Tempfile artifacts outside tmp_path | Tampering | FIX-4 discipline: `tmp_path` for all artifacts; start_server tests `monkeypatch.chdir(tmp_path)` |

## Sources

### Primary (HIGH confidence)
- `dnallm/mcp/server.py` (full read of lines 1-1860: transports 1625-1846, streaming 724-1148, timeout wrapper 282-341) — this session
- Live probes against installed SDK in `.venv`: FastMCP method surface + signatures + sources (mcp 1.30.0); in-memory streamable-http round trip (exit 0, 0.65s); SSE deadlock reproduction (raw GET + sse_client); 421 host-validation root cause (mcp/server/fastmcp/server.py:191-197, transport_security.py:119-127)
- `dnallm/models/model.py:300-679, 736-914` and `dnallm/models/tokenizer.py:254-315` — this session
- `dnallm/utils/transformers_compat.py` (full, 223 lines) + live idempotency verification on transformers 5.17.0 / bitsandbytes 0.50.2
- `.planning/phases/01-harness-integrity-measured-baseline/{01-AUDIT-REPORT.md, coverage.json}` — all per-file numbers recomputed from coverage.json this session
- `tests/models/test_model.py`, `tests/conftest.py`, `tests/mcp/test_timeout.py`, `dnallm/mcp/tests/test_server_integration.py`, `dnallm/mcp/tests/_network_skip.py`, `tests/expected_skips.yaml` — this session
- `pyproject.toml [tool.pytest.ini_options]`, `.github/workflows/ci.yml` — this session

### Secondary (MEDIUM confidence)
- Per-area achievable-coverage estimates and runtime-impact projections (analysis over the verified inputs — see Assumptions Log)

### Tertiary (LOW confidence)
- None — no training-data-only claims are load-bearing in this document

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all installed versions probed live
- MCP seam (the recorded flag): HIGH — every claim reproduced against the installed SDK, including the two negative results (SSE deadlock, 421)
- Worklist decomposition: HIGH — mechanical recompute of Phase-1 coverage.json (with the pre-Phase-2 caveat, Pitfall 9)
- Reachability/runtime estimates: MEDIUM — analysis-based, bounded by the per-wave re-measure loop

**Research date:** 2026-09-30
**Valid until:** 2026-10-30 (stable domain; revisit if transformers/mcp majors land mid-phase)
