# Phase 1: Harness Integrity & Measured Baseline - Audit Report

**Audited:** 2026-09-30 (local); run window 2026-09-29T17:51Z–2026-09-29T18:37Z (UTC)
**Environment:** local dev machine (Python 3.13.15, GPU NVIDIA GB10, warm HF cache 21G at `~/.cache/huggingface`, 2.5T free on `/` — `/tmp` is the same filesystem). Tooling: pytest 9.1.1, pytest-cov 7.1.0 (enabling flag only), coverage 7.16.2, pytest-timeout 2.4.0 (`timeout: 300.0s` active), pytest-asyncio 1.4.0 (mode=auto). Plugins loaded (audit log header): `logfire-5.1.1, platformdirs-4.12.1, langsmith-0.14.1, cov-7.1.0, timeout-2.4.0, anyio-4.15.1, asyncio-1.4.0` (`progress` disabled via `-p no:progress` for clean artifacts). Per RESEARCH Pitfall 7: local-vs-CI plugin deltas are expected (CI installing only `.[base]` loads fewer plugins) — recorded, not normalized.
**Commands of record** (verbatim; all exit codes captured by redirect-then-read, never a pipe):

1. Warm full census (both roots, `slow` included, coverage on — single enabling `--cov`, scope from `pyproject.toml`):
   `.venv/bin/python -m pytest -ra --durations=0 --junitxml=.planning/phases/01-harness-integrity-measured-baseline/junit-full.xml --cov -p no:cacheprovider -p no:progress > /tmp/audit-full.log 2>&1` — **exit 0**
   then `.venv/bin/coverage report -m > .planning/phases/01-harness-integrity-measured-baseline/coverage-term-missing.txt` and `.venv/bin/coverage json -o .planning/phases/01-harness-integrity-measured-baseline/coverage.json`
2. Warm slow timing leg (no coverage): `.venv/bin/python -m pytest -m slow --durations=0 --junitxml=.planning/phases/01-harness-integrity-measured-baseline/junit-slow-warm.xml -p no:cacheprovider -p no:progress > /tmp/audit-slow-warm.log 2>&1` — **exit 0**
3. Cold slow timing leg (isolated scratch HF cache, deleted afterwards): `HF_HOME=/tmp/dnallm-hf-audit-cold .venv/bin/python -m pytest -m slow --durations=0 --junitxml=.planning/phases/01-harness-integrity-measured-baseline/junit-slow-cold.xml -p no:cacheprovider -p no:progress > /tmp/audit-slow-cold.log 2>&1` — **exit 0**. **HF route of record: direct `huggingface.co`** (hf-mirror fallback was not needed). Scratch cache grew to 2.0G during the leg and was removed after.

Machine evidence (this directory, committed): `junit-full.xml`, `coverage.json`, `coverage-term-missing.txt`, `junit-slow-warm.xml`, `junit-slow-cold.xml`. Every number below is a mechanical transform of those files (parse script: junit via `xml.etree`, coverage via `json`); artifacts are treated as untrusted input when parsed (T-01-04).

## Config-resolution probe matrix (success criteria 1–2 evidence)

Post-HARN-01 state, recomputed this session (header lines are pytest's own resolution verdict):

| Invocation shape | rootdir | configfile | collected |
|---|---|---|---|
| bare `pytest --collect-only -q` | `/home/forrest/Github/DNALLM` | `pyproject.toml` | **625 tests** |
| `pytest tests/utils/test_sequence.py --collect-only -q` | `/home/forrest/Github/DNALLM` | `pyproject.toml` | 5 tests |
| `pytest dnallm/mcp/tests --collect-only -q` | `/home/forrest/Github/DNALLM` | `pyproject.toml` | **39 tests** |

All three shapes resolve to the single config source (`pyproject.toml`, `testpaths: tests, dnallm/mcp/tests`); the `tests/pytest.ini` hijack shape (`configfile: pytest.ini`, rootdir `<repo>/tests`) is gone. Bare-collection total 625 matches the grounded planning number exactly (586 `tests/` + 39 `dnallm/mcp/tests/`). The census run itself (junit-full.xml) executed 625 tests — both roots, `slow` included.

## AUDIT-01: Census (pass/fail/skip by reason, both roots, slow included)

Totals — equal to the `junit-full.xml` testsuite attributes:

| tests | failures | errors | skipped | wall time |
|---|---|---|---|---|
| **625** | **0** | **0** | **9** | 909.355s (15:09) |

Passed = 616. Exit code 0. Skip reasons parsed from the junit `<skipped message>` elements, grouped with per-reason counts (all reasons occurred; zero-occurrence rows for the categories Phase 2's skip-typing worklist (FIX-03) anticipates are rendered explicitly as 0 rather than dropped — specless edge "empty" disposition):

| # | Skip reason (parsed message) | Count | Root |
|---|---|---|---|
| 1 | `SSE connection failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry of mcp client tests) |
| 2 | `Health check test failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry) |
| 3 | `DNA prediction test failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry) |
| 4 | `Streamable HTTP connection failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry) |
| 5 | `Streamable HTTP session reuse test failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry) |
| 6 | `Streamable HTTP custom URL test failed: unhandled errors in a TaskGroup (1 sub-exception)` | 1 | `tests/` (re-entry) |
| 7 | `Multi-class plotting with AUROC requires complex implementation` (`tests/tasks/test_metrics.py:298`) | 1 | `tests/` |
| 8 | `Multiclass AUROC implementation has issues` (`tests/tasks/test_metrics.py:761` — the known AUDIT-blocker crash skip) | 1 | `tests/` |
| 9 | `No import statements found` (`tests/examples/test_examples.py:254`) | 1 | `tests/` |
| — | *network-offline skip* | **0** | — |
| — | *missing-optional-dependency skip* | **0** | — |
| — | *platform/GPU-specific skip* | **0** | — |

Per-root skip breakdown: `tests/` = 3 native skips + 6 re-entries of the mcp client tests via `tests/benchmark/../../dnallm/mcp/tests/...` (the same 6 mcp-root skips appear again inside the `tests/` root; counted in the junit where they executed) = 9 total skip events; `dnallm/mcp/tests/` contributed the TaskGroup skip family. **Phase 2 skip-typing worklist (FIX-03):** the six TaskGroup skips are *untyped conditional skips* (live-network probes failing with `ExceptionGroup`), two are *crash-skips* tied to the known multiclass-AUROC defect (`metrics.py:283`, FIX-01), one is a content-based example skip (benign, likely typed `legacy`).

**Timeout triage** (RESEARCH Pitfall 3 — the 300s per-test timeout is newly active after Plan 01): log-repr scan of `/tmp/audit-full.log` gives **0 failures marked `Failed: Timeout`** and **0 ordinary failures** (0 `FAILED` lines, 0 `ERROR` lines). No tripped tests to record as Phase 2/3 follow-ups; the longest test (`test_complete_training_workflow`, 194.2s) ran well inside the 300s budget.

Warnings delta (Pitfall 6): the deleted ini's `--disable-warnings` is gone; the census showed **3 warnings total** — two `dill` `PicklingWarning`s from `tests/benchmark/test_benchmark.py::test_run_benchmark_flow` (MagicMock pickling) and one `RuntimeWarning: coroutine 'DNALLMMCPClient.adna_sequence_predict' was never awaited` at `tests/mcp/test_client_sdk.py:405`. No flood occurred; **no `filterwarnings` entry added** (the only permitted remedy, and it was not needed). The pytest-asyncio unset-`asyncio_default_fixture_loop_scope` warning did not materialize as a counted warning (header shows `asyncio_default_fixture_loop_scope=None`; research A5 confirmed cosmetic this cycle).

## AUDIT-02: Ranked per-module gap worklist

Machine artifacts: `coverage.json` (per-file `missing_lines`) and `coverage-term-missing.txt` (human worklist). Ranking rule: missing-line count **descending**, module path **ascending** tie-break (specless edge "ordering" disposition — equal-count modules have a stable, specified order). Full ranked table (all 43 files with gaps; 14 measured files have zero missing lines and are listed after):

| Rank | Missing lines | Module |
|---|---|---|
| 1 | 523 | `dnallm/inference/inference.py` |
| 2 | 359 | `dnallm/datahandling/data.py` |
| 3 | 332 | `dnallm/inference/plot.py` |
| 4 | 269 | `dnallm/inference/mutagenesis.py` |
| 5 | 261 | `dnallm/inference/interpret.py` |
| 6 | 250 | `dnallm/mcp/server.py` |
| 7 | 249 | `dnallm/models/special/crossdna.py` |
| 8 | 234 | `dnallm/models/model.py` |
| 9 | 196 | `dnallm/models/head.py` |
| 10 | 168 | `dnallm/models/special/evo.py` |
| 11 | 120 | `dnallm/inference/benchmark.py` |
| 12 | 115 | `dnallm/models/tokenizer.py` |
| 13 | 108 | `dnallm/cli/cli.py` |
| 14 | 81 | `dnallm/finetune/trainer.py` |
| 15 | 71 | `dnallm/cli/mutagenesis.py` |
| 16 | 68 | `dnallm/models/special/borzoi.py` |
| 17 | 58 | `dnallm/mcp/client.py` |
| 18 | 56 | `dnallm/models/special/megadna.py` |
| 19 | 55 | `dnallm/mcp/model_manager.py` |
| 20 | 53 | `dnallm/mcp/start_server.py` |
| 21 | 51 | `dnallm/tasks/metrics.py` |
| 22 | 45 | `dnallm/utils/transformers_compat.py` |
| 23 | 39 | `dnallm/utils/logger.py` |
| 24 | 30 | `dnallm/models/special/dnabert2.py` |
| 25 | 27 | `dnallm/mcp/config_manager.py` |
| 26 | 23 | `dnallm/models/special/lucaone.py` |
| 27 | 19 | `dnallm/cli/inference.py` |
| 28 | 14 | `dnallm/cli/model_config_generator.py` |
| 29 | 14 | `dnallm/models/special/enformer.py` |
| 30 | 14 | `dnallm/models/special/mutbert.py` |
| 31 | 14 | `dnallm/models/special/space.py` |
| 32 | 13 | `dnallm/cli/train.py` |
| 33 | 12 | `dnallm/models/losses.py` |
| 34 | 9 | `dnallm/configuration/configs.py` |
| 35 | 7 | `dnallm/models/special/gpn.py` |
| 36 | 7 | `dnallm/models/special/omnidna.py` |
| 37 | 7 | `dnallm/utils/sequence.py` |
| 38 | 7 | `dnallm/utils/support.py` |
| 39 | 6 | `dnallm/mcp/config_validators.py` |
| 40 | 4 | `dnallm/utils/cuda_compat.py` |
| 41 | 3 | `dnallm/models/special/basenji2.py` |
| 42 | 1 | `dnallm/datahandling/dataset_auto.py` |
| 43 | 1 | `dnallm/utils/training_plots.py` |

Files measured at 100% (14): all nine package `__init__.py` files, `dnallm/models/modeling_auto.py`, `dnallm/models/special/__init__.py`, `dnallm/tasks/task.py`, `dnallm/version.py` (complete report in `coverage-term-missing.txt`). The ordering recomputes identically from `coverage.json` (spot-check: rows 1–3 = 523/359/332, missing-descending; rows 28–31 = the four 14-missing files in path-ascending order: `cli/model_config_generator` < `special/enformer` < `special/mutbert` < `special/space`).

**Phase 3 wave-ordering input** (ROADMAP wave order models → mcp → inference → datahandling/finetune → cli/shims): by subpackage, missing lines concentrate in `inference/` (1,505), `models/` (1,210, of which the `special/` subtree 653), `mcp/` (449), `datahandling/`+`finetune/` (441), `cli/`+`utils/` shims (328), `tasks/` (51), `configuration/` (9) — sums recomputed from `coverage.json`, totaling 3,993. Denominator health: zero report rows match any of the seven pre-locked omit entries (vendored `dnallm/tasks/metrics/` dir, `enformer_model/`, `megatron.py`, `mamba_npu.py`, `dnallm/mcp/tests/`, `run_tests.py`, `example_sse_usage.py`), while the measured neighbors `dnallm/tasks/metrics.py` (51 missing) and `dnallm/utils/sequence.py` (7 missing) remain in the denominator — boundary exact, no neighbor spill.

## AUDIT-03: Measured baseline coverage and cold/warm slow timings

**Baseline (the number Phase 3 sizing waits on):** `coverage.json` `totals.percent_covered` = **45.92%** — 3,390 covered / 7,383 statements (3,993 missing, 16 excluded) across **57 measured files**. Distance to the 90% gate: **44.08 percentage points**, i.e. roughly 3,255 additional statements to cover (0.90 × 7,383 = 6,645 needed vs 3,390 covered). STATE.md Phase-3 sizing blocker now has its input: at ~44 points of gap over ~4.0k missing statements concentrated in 43 files, the single-phase Phase 3 plan stays viable but is near the split threshold flagged in the ROADMAP sizing note — revisit split via `/gsd-phase` with this table as the evidence.

**Cold vs warm slow-test wall clock** (per-testcase `time` attributes of the two slow junit files; 27 collected = 21 passed + 6 skipped each leg; delta = cold − warm):

| Test | Warm (s) | Cold (s) | Δ (s) |
|---|---|---|---|
| `trainer_real_model::test_download_real_huggingface_connection` (HF download) | 0.271 | 61.402 | **+61.131** |
| `test_inference_real_model::test_basic_inference` (HF model download) | 5.569 | 29.895 | **+24.326** |
| `trainer_real_model::test_complete_training_workflow` | 194.203 | 194.603 | +0.400 |
| `trainer_real_model::test_with_config_file` | 189.790 | 189.988 | +0.198 |
| `trainer_real_model::test_training` | 182.598 | 182.362 | −0.236 |
| `trainer_real_model::test_early_stopping_stops_before_full_epochs` | 94.853 | 95.276 | +0.423 |
| `trainer_real_model::test_no_early_stopping_runs_full_epochs` | 51.219 | 51.197 | −0.022 |
| `trainer_real_model::test_qlora_training` | 26.135 | 26.823 | +0.688 |
| `mcp_functionality::test_mcp_functionality` | 14.031 | 14.500 | +0.469 |
| `trainer_real_model::test_prediction` | 15.102 | 15.180 | +0.078 |
| remaining 17 slow tests (each < 10s in both legs) | — | — | each |Δ| < 0.7 |

**Leg totals:** warm 819.58s (13:39), cold 906.86s (15:06) — total delta **+87.28s (+10.6%)**. The two genuinely cold-HF tests account for +85.5s of that delta (model downloads into the scratch cache, which reached 2.0G). The dominant trainer tests are **ModelScope-sourced** (`zhangtaolab/plant-dnabert-BPE`, `source="modelscope"`, 12 call sites) and therefore ran against the warm `~/.cache/modelscope` cache in *both* legs — their ~±0.5s deltas are run-to-run noise, not cache effects. This is an honest environmental boundary of the cold-leg method (HF_HOME isolates the HF cache only; ModelScope cache is untouched) and is recorded here rather than normalized away. HF route of record: direct `huggingface.co` (mirror not needed).

## AUDIT-04: Subprocess-coverage scope decision record

**Decision: start minimal — no subprocess patching in the coverage config.** `[tool.coverage.run]` carries no `patch` entry; nothing measures child processes. This is the pre-locked starting decision, now closed on canary evidence.

**Static evidence** (`grep -rn "subprocess\|Popen" tests/ dnallm/mcp/tests/`): **0 hits** across both collected roots. The only `dnallm/` files mentioning subprocess are `dnallm/mcp/run_tests.py` (the omitted helper script) and `dnallm/tasks/metrics/code_eval/execute.py` (inside the omitted vendored `dnallm/tasks/metrics/` directory) — neither is in the measured denominator. No collected test spawns a subprocess, so unmeasured child execution cannot distort today's numbers.

**Dynamic evidence** (ephemeral `tests/test_zz_subprocess_canary.py`, run with the single enabling `--cov`, deleted afterwards):
- Child-only execution: a child spawned via `subprocess.run` executed `import dnallm.utils.sequence` (test asserted `returncode == 0` — the child genuinely ran the module) and the parent-side coverage export responded **"No data to report"** — the child's execution of `dnallm` code left zero measurable trace.
- Parent-import control: with the parent importing the same module in-process, `dnallm/utils/sequence.py` appeared in the report with exactly the parent's 7 import-level lines executed and **68 function-body lines missing** — the child's identical execution of that module contributed nothing on top of the parent's own import.

Conclusion: under the minimal config, child-process-side execution is entirely unmeasured — precisely the semantics pytest-cov 7 documents (the `.pth` auto-measurement was removed; subprocess coverage is strictly opt-in via coverage's `patch`).

**Escalation trigger** (the only condition under which the minimal scope is revisited): a future test whose *assertions depend on child-process-side code paths* — i.e. a test that would pass/fail based on code executing inside a `subprocess.run`/`Popen` child. Only then add `patch = ["subprocess"]` to `[tool.coverage.run]`, which additionally forces `parallel = True` and requires a `coverage combine` step before reporting (per coverage 7.16 docs; floor `coverage[toml]>=7.10.6` guarantees the lever exists). Mere *use* of subprocesses by tests (whose assertions live in the parent) does not trigger escalation.

**STATE.md blocker closure:** "Phase 1: subprocess-coverage scope is an unresolved config conflict (start minimal; let a canary decide)" — **resolved by this record: start minimal, canary evidence above, escalation trigger stated.**

## Flagged Assumptions

Unresolved specless edges and inherited research assumptions, carried forward explicitly (none silently dropped):

- **[HARN-03/concurrency — specless edge, unresolved by design]** Coverage and audit runs execute strictly serially. All Task-1 legs ran one at a time (census → coverage export → warm leg → cold leg → canary). Concurrent pytest-with-coverage invocations sharing one `.coverage` data file are out of scope; they would require parallel data files plus a combine step, which only enters if AUDIT-04 ever escalates.
- **[HARN-01/concurrency — carried from Plan 01]** The SIGINT-exits-non-zero guarantee remains a backstop truth authored in 01-01-PLAN.md; this audit did not re-probe SIGINT behavior, so it stays unconfirmed-by-explicit-evidence (the Plan-01 probes covered failing-test rc=1 and SIGINT rc=2; no new evidence was produced this cycle).
- **[Research A4 — partially confirmed]** The cold leg fit local disk (2.0G scratch, 2.5T free) and completed in 15:06. The planning extrapolation "cold may take hours" proved pessimistic for this suite: only two slow tests are genuinely HF-cold here (+85.5s combined), because the dominant slow tests are ModelScope-sourced. Per-test durations recorded above make any future stall visible.
- **[Research A5 — confirmed cosmetic]** The pytest-asyncio unset-`asyncio_default_fixture_loop_scope` warning did not appear as a counted warning in either audit log (3 warnings total, none pytest-asyncio). No fixture-scope failures occurred.
- **[Cold-leg method boundary — new, discovered during execution]** `HF_HOME` isolation covers the HuggingFace cache only. ModelScope-sourced tests (`test_trainer_real_model.py`, the longest slow tests) ran warm-cache in both legs; a fully-cold timing for them would require isolating the ModelScope cache (`MODELSCOPE_CACHE`) — out of scope for this audit's command of record, recorded as a fact about the numbers above.
