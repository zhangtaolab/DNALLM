---
phase: 08-full-execution-rollout-repair-loop
verified: 2026-10-05T07:58:00Z
status: passed
score: 8/8 must-haves verified (5 roadmap success criteria + 3 phase-level gates)
covered_files:
  - .github/workflows/ci.yml
  - .planning/phases/08-full-execution-rollout-repair-loop/08-01-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-01-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-02-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-02-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-03-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-03-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-04-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-04-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-05-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-05-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-06-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-06-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-07-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-07-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-08-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-08-SUMMARY.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-09-PLAN.md
  - .planning/phases/08-full-execution-rollout-repair-loop/08-09-SUMMARY.md
  - dnallm/models/model.py
  - dnallm/models/special/evo.py
  - dnallm/models/special/megadna.py
  - dnallm/utils/transformers_compat.py
  - models.lock
  - pyproject.toml
  - scripts/runner/README.md
  - scripts/runner/ollama.service
  - tests/examples/_execution.py
  - tests/examples/test_examples.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
  - tests/examples/test_script_execution.py
  - tests/models/test_model.py
  - tests/models/test_special/test_evo.py
  - tests/models/test_special/test_megadna.py
  - tests/test_extras_guard.py
  - tests/utils/test_transformers_compat.py
  - tests/utils/test_transformers_compat_np.py
covered_digest: "v2:sha256:3b59c6f774cb91335019b2b53c9d2d42edcfc2f319ac6d182484193f062a17bc"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 8: Full Execution Rollout & Repair Loop Verification Report

**Phase Goal:** Every example artifact executes for real on the nightly GPU runner — all notebooks, the marimo apps, the helper script, every YAML, and the two ollama-backed mcp_example notebooks — and every error that surfaces is fixed with regression tests across example code, the docs mirror, and the dnallm library
**Verified:** 2026-10-05T07:58:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

All 5 roadmap success criteria and all 3 phase-level gates verified against the codebase, live test runs, on-disk census logs, and git history — not SUMMARY claims. The verifier re-ran the full fast lane live (1807 passed / 1 pre-existing skip / 53 deselected, exit 0, 97.55 s — count-identical to the 08-09 census baseline), re-ran one marimo D-18 quadruple test live (1 passed, 9.93 s), re-ran all 26 named repair-regression tests live (26 passed), re-validated the YAML leg both ways (21/21), re-ran both docs-sync gates (exit 0), and read the raw pytest tails of the four surviving census/family logs under /tmp (`08_09_examples.log`: 196 passed / 1 skipped in 2:59:34; `08_08_mcp.log`: 2 passed / 0 skipped; `08_08_lora.log`: 3 passed / 0 skipped; `08_05_family.log`: 12 passed / 0 skipped). Per the operational contract the 3-hour slow census was NOT re-run; its wiring was spot-checked instead (NOTEBOOK_EXEC_SPECS coverage, ACTIVE/GATED registries, expected_skips allowlist) and its raw log evidence read directly.

### Roadmap Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | All 21 notebooks execute all code cells end-to-end with real models (fail-at-first-error per notebook, fail-soft across); 3 marimo apps headless with default-value + exit-code assertions; generate_bpe_dataset.py produces its artifact in-sandbox | ✓ VERIFIED | Census wiring: `NOTEBOOK_EXEC_SPECS` (tests/examples/_execution.py:127-260) carries all 21 census notebooks + 2 showcase budget entries; `ACTIVE_NOTEBOOKS` (13) + `GATED_NOTEBOOKS` (8, tests/examples/test_notebook_execution.py:1102-1115) = exactly the 21 non-showcase ipynb in `example/` (verified by tree enumeration). Execution evidence: /tmp/08_09_examples.log tail read by verifier — `196 passed, 1 skipped in 10774.89s (2:59:34)`, single skip = benign no-imports (predict_data, tests/examples/test_examples.py:257). Marimo D-18 quadruple (headless + defaults + exit-code-via-error-artifact-absence + key-content marker, tests/examples/test_marimo_execution.py:86-125) re-run live by verifier: 1 passed in 9.93 s. Script lane: artifact assertions (`out_pkl.is_file()`, size>0, mtime>=run_start, tests/examples/test_script_execution.py:151-153) green in census (rollup: 4 passed, artifact 11,299,268 B fresh). Runner side: example-nightly job complete in ci.yml (all prereq steps + stages); dispatch 2 executed the then-current set on the real runner; completed job's first dispatch 37278002681 in flight — Phase-5 D-04 post-merge boundary, Phase 9 scope |
| 2 | Every example YAML passes real `load_config()` Pydantic validation on the fast leg, zero new fast-leg skips | ✓ VERIFIED (live) | `validate_yaml.py`: "All YAML files passed validation." (21 files); `tests/configuration/test_yaml_load.py`: 21 passed. Full fast lane re-run live: **1807 passed / 1 skipped / 53 deselected, exit 0** — count-identical to the 08-08/08-09 baselines; the single skip is the permanent benign no-imports entry. Zero new fast-leg skips |
| 3 | Every surfaced error fixed with regression test; harness-bug vs content-bug triage explicit (no cwd false-repairs); langchain-ollama in mcp extra; docs mirror regenerated per repair | ✓ VERIFIED | 13 repair classes each land with same-commit (or paired RED-commit) regression tests — all verified in git and green live: ipython<9 pin (439370f + TestNotebookExtraMembers/TestNotebookKernelPlotCompat), rice cache seeding (e90ff21/c5f0916 + 6 unit tests), combined sibling seeding (c5f0916 + TestCombinedSiblingSeeding), DNATokenizer unknown→id 1 (69cc108 RED + c547969 + 6 tests), megaDNA column drop (54892b2 + contract test), allow_patterns passthrough (fe37b85 RED + 5ed8c9e + 4 kwargs tests), np.fromstring probe-gated shim (bedc118 RED + c820780 + 12 tests), lora mirror endpoint (bc942c4 + TestLoraMirrorEndpoint), langchain-ollama>=1.1.0 in mcp extra (52f9d06 + TestMcpExtraMembers), pyBigWig LDFLAGS (f71c091, proven by dispatch-2 install pass), bedtools rootless (ea1bfd6, ci step), OPTIONAL_IMPORT_MODULES seam (cd6debd), D-17 NT shim disposition (08-D17-DISPOSITION.md, pristine-snapshot proof). Verifier ran all 26 named regression tests live: 26 passed in 3.72 s. Triage explicit per failure in 08-CENSUS-ROLLUP.md repair tables (library/declared-dep, infrastructure, harness/content, runner system dep, transient); **zero `os.chdir` anywhere in the harness** — no cwd false-repairs. Every notebook-repair commit carries its docs mirror (a4cbcb4, 081a886, a7ed221 file lists); both sync gates exit 0 live |
| 4 | models.lock carries all newly-executed ids (~8+) with ms-first prefixes aligned to each notebook's source= route and revision pins; evo-1 safetensors-only via allow_patterns with giants tiered outside the 10GB-quota cache | ✓ VERIFIED (live) | models.lock counted live: **24 `^(hf\|ms)` rows, 14 `@sha`-pinned (9 ms + 5 hf)**, header documents the D-15 pin format; evo-1 row (line 33) carries the GIANTS-tier comment. Prefix-vs-source alignment checked per-id against every notebook's ACTIVE source= line: dnagpt-BPE/6mer/singlebase/NT-BPE/tRNA×2/promoter_strength×2/PlantHelixSeek×2 → ms + `source="modelscope"` active; megaDNA_updated, PlantCAD2, evo2, evo-1-8k → hf + `source="huggingface"` active; benchmark NT-100m → ms via `benchmark_config.yaml: source: modelscope`. evo-1 safetensors-only: `_EVO1_SAFETENSORS_ONLY_PATTERNS` (evo.py:33-38) forwarded at the evo-1 hub fetch only (evo.py:389 — evo2 unfiltered per prohibition); ci.yml giants prefetch uses the same patterns into `~/models-giants/hub` which is OUTSIDE both cached paths (`~/.cache/huggingface/hub`, `~/.cache/modelscope/hub`). A4 load-time proof recorded in the rollup: offline load (HF_HUB_OFFLINE=1), evo-1 snapshot zero `.pt`, blob store unchanged 14,889 MiB |
| 5 | Both mcp_example notebooks execute end-to-end against loopback-only ollama on the nightly runner (systemd unit in-repo, pre-pulled model, readiness probe); port/VRAM coexistence planned against the 6 MCP :8000 probes; typed network-unavailable skip is fallback only | ✓ VERIFIED | `scripts/runner/ollama.service` in-repo with `OLLAMA_HOST=127.0.0.1:11434` loopback pin + security rationale; `scripts/runner/README.md` documents the one-time owner install/enable/`ollama pull qwen3.8:latest` (17.74GB)/verify steps. D-13 probe: `_probe_http_with_retry` (~60s × 2s window, patchable sleep, full attempt log in message, test_notebook_execution.py:960-984) wired into `_gate_ollama_stack`; skip is infra-missing only. Both-up execution: /tmp/08_08_mcp.log — `2 passed, 57 deselected in 382.62s` (langchain under isolated dnallm-mcp-langchain kernelspec, still wired); pair green again in the final census (196P/1S/0F includes both). Live probe by verifier: `curl 127.0.0.1:11434/api/tags` answers with qwen3.8:latest (17,741,872,154 B) on this shared runner/dev box. D-07 coexistence: stage contract in-module (test_notebook_execution.py:922-935) and implemented in ci.yml — stage 1 deselects the mcp pair (`-k "not mcp_example"`), stage 2 runs the 6 :8000 probes (3 streamable-http, sse restart, 3 sse), stage 2.5 kernel pkill + VRAM settle, stage 3 fresh streamable-http server + pair, server stopped after |

### Phase-Level Gates

| Gate | Status | Evidence |
|------|--------|----------|
| Regression gate: fast lane green | ✓ PASS | Re-run live by verifier: 1807 passed / 1 skipped / 53 deselected, exit 0, 97.55 s (`pytest tests/ -m "not slow" -q`) |
| Census rollup closure | ✓ PASS | 08-CENSUS-ROLLUP.md carries the final D-03 census (196P/1S/0F, 2:59:34, both endpoints up), the per-item table (every notebook/marimo/script/showcase/YAML row PASS), all four family closures, the First-Dispatch Consumption table (every runner failure classified + dispositioned with commit shas), and the cache-quota measurement. Raw log /tmp/08_09_examples.log matches the recorded counts exactly |
| STATE / ROADMAP / REQUIREMENTS consistency | ✓ PASS | REQUIREMENTS.md marks all 11 Phase 8 requirements `[x]` Complete; ROADMAP shows 9/9 plans executed; STATE.json Phase 8 `in_progress` — the correct pre-verification state this report resolves |

### Observable Truths (per-plan, load-bearing)

All 31 phase-8 commits verified present on `phs`. Every plan's load-bearing must-have truths were verified; the ones proven LIVE by the verifier (not from logs) are marked (live).

**08-01** — example-nightly job: staggered cron `30 5 * * *` (ci.yml:18) + dispatch, `[self-hosted, dnallm-nightly]`, verbatim event gate (ci.yml:525), D-07 staged layout with contract comments (526-563), fail-soft stage-results.txt + summary exit-1 (796-804, 933-946), zero `continue-on-error: true` anywhere, shared models.lock cache key identical to coverage-nightly (both `${{ runner.os }}-models-${{ hashFiles('models.lock') }}`), full extras `.[base,fla,dev,mcp]` (627); langchain-ollama>=1.1.0 in mcp extra (pyproject:132) + guard (live); marimo D-18 quadruple (live: 1 passed 9.93 s); runner-inventory probes (781-790); fast lane green (live).

**08-02** — D-17 ledger exists with per-item verdicts + pristine-snapshot reproducibility proof; EXEC-04 script lane healed with artifact freshness assertions; EXEC-05 YAML 21/21 (live, both validators); D-03 baseline rollup created; ipython>=8.31,<9 pin + static/dynamic guards (live); rice seeding unit tests (live); combined sibling seeding + contract tests (live).

**08-03** — `allow_patterns: list[str] | None` on `download_model` (model.py:322) and `_get_model_path_and_imports` (model.py:412), forwarded only when not None, kwargs rebuilt per attempt (360-361, 442-443); evo-1 pattern set wired (evo.py:389); np.fromstring shim probe-gated, sentinel-idempotent, registered in apply_patches (transformers_compat.py:1038-1112; 12 tests live); spec env override sandwiched with the evo giants `HF_HUB_CACHE` + `HF_HUB_OFFLINE` entry (_execution.py:153-156; TestSpecEnvOverrides live).

**08-04** — DNATokenizer `UNKNOWN_TOKEN_ID = 1` + `_convert_token_to_id` fallback (megadna.py:58, 91-92; 6 tests live); finetune_generation repaired (pinned clone cell, D-21 stamp, present-filter column drop) with byte-synced mirror in the same commit (a4cbcb4); isolated `dnllm-megadna` kernelspec lane + venv-targeted `_gate_megadna_isolated`; first real execution 837 s 0 skips (family log).

**08-05** — both megaDNA siblings repaired with D-21 stamps + executable pinned installs + mirrors (081a886); family lane 12 passed / 0 skipped in 2885.10 s (log read); megaDNA family CLOSED in rollup; reversible pinned prereqs recorded and reproduced by the ci.yml step.

**08-06** — giants tier safetensors-only at pinned revision `a9be7b6…` outside every cached path; isolated `dnallm-evo-kernel` lane (`evo_prerequisites_installed` probes the venv, `ensure_evo_kernel` never installs); evo notebook repaired (8k reference, noFP8 override cell, D-21 stamp) with mirrors (a7ed221); first real execution green, zero new rungs.

**08-07** — zero-rung confirmation (5P/0S, 49.45 s); census reconciled 188P/5S/0F; A4/CI-05 load-time proof (offline load, evo-1 `.pt`-free, blob store unchanged); OPTIONAL_IMPORT_MODULES seam extended (cd6debd).

**08-08** — lora pair real green (log: 3 passed / 0 skipped, 661.87 s; `_gate_mamba`); ollama unit + README in-repo (read in full: loopback pin, owner steps); D-13 retry probe + extended TestProbeHonesty (green in fast lane); D-07 stage contract in-module; family-close census 196P/1S/0F.

**08-09** — models.lock 24/14/prefix-aligned (live count); job completion with cache-before-consumer ordering (bedtools cache step ci.yml:630-634 precedes install 636), wheelhouses, giants prefetch, stages 2/2.5/3 wired, job-level HF_ENDPOINT=hf-mirror with network-reality comment; first-dispatch consumption (every failure dispositioned, nothing dropped); final census + fast lane (logs read); cache-quota measurement recorded as an owner hand-off.

**Score:** 8/8 must-haves verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci.yml` (example-nightly job) | Complete staged job: prereqs, caches, giants prefetch, stages 0-4, fail-soft | ✓ VERIFIED | Lines 514-946 read in full; structure + ordering + zero continue-on-error confirmed |
| `models.lock` | 24 rows, 14 pinned, prefix-aligned, giants comment | ✓ VERIFIED (live) | Counted + per-id alignment checked against notebook ACTIVE source= lines |
| `pyproject.toml` | langchain-ollama>=1.1.0 in mcp extra; ipython>=8.31,<9 in notebook extra | ✓ VERIFIED (live) | Lines 132, 110; guard tests green |
| `dnallm/models/special/megadna.py` | DNATokenizer unknown→id 1 | ✓ VERIFIED (live) | Lines 58, 91-92; 6 tests green |
| `dnallm/models/special/evo.py` | safetensors-only patterns on the evo-1 fetch only | ✓ VERIFIED | Lines 33-38, 389; single allow_patterns occurrence |
| `dnallm/models/model.py` | conditional allow_patterns passthrough, both layers | ✓ VERIFIED | Lines 322, 360-361, 412, 442-443 |
| `dnallm/utils/transformers_compat.py` | probe-gated np.fromstring binary-mode shim | ✓ VERIFIED (live) | Lines 1038-1112; 12 tests green |
| `scripts/runner/ollama.service` + `README.md` | loopback-only unit + owner steps | ✓ VERIFIED | OLLAMA_HOST=127.0.0.1:11434 pinned; enable/pull/verify documented |
| `tests/examples/_execution.py` | specs for all 21 + gates wiring + env overrides | ✓ VERIFIED (live) | 27 spec entries; evo/megadna/langchain kernelspecs; lora mirror env |
| `tests/examples/test_notebook_execution.py` | ACTIVE/GATED registries, D-13 probe, D-07 contract, contracts | ✓ VERIFIED (live) | 13 ACTIVE + 8 GATED = 21; probe + contract tests green |
| `tests/examples/test_marimo_execution.py` | D-18 quadruple | ✓ VERIFIED (live) | 1 app re-run live: passed 9.93 s |
| `tests/examples/test_script_execution.py` | healed script lane + rice seeding | ✓ VERIFIED (live) | Artifact freshness assertions; seeding unit tests green |
| 08-CENSUS-ROLLUP.md / 08-D17-DISPOSITION.md | census ledger + D-17 per-item ledger | ✓ VERIFIED | Both read; final census + quota record + disposition tables present |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| example-nightly stage 1 | all 21 execution tests | `pytest tests/examples -k "not mcp_example"` + junit | ✓ WIRED |
| example-nightly stages 2/3 | MCP :8000 probes + mcp pair | dnallm-mcp-server background + readiness curl + `-k mcp_example` | ✓ WIRED |
| lora specs | hf-mirror endpoint | spec `env.HF_ENDPOINT` through the run_notebook sandwich | ✓ WIRED (TestLoraMirrorEndpoint live) |
| evo spec | giants tier | spec `env.HF_HUB_CACHE`/`HF_HUB_OFFLINE` + kernelspec venv | ✓ WIRED (TestSpecEnvOverrides live) |
| ci.yml giants prefetch | evo-1 safetensors set | `allow_patterns` + `cache_dir=~/models-giants/hub` | ✓ WIRED |
| ci.yml models cache | models.lock | `hashFiles('models.lock')` key, identical to coverage-nightly | ✓ WIRED |
| notebook repairs | docs mirrors | same-commit mirror files; check_docs_sync + check_notebook_md_sync | ✓ WIRED (both exit 0 live) |

### Data-Flow Trace (Level 4)

No rendered-value chains end in static returns. The census metrics flow from real model executions (raw pytest logs with real timings/skips); the lock prefixes flow from the notebooks' actual active routes (verified per-id); the marimo default assertions read the real exported HTML text; the fast-lane counts were regenerated live by this verifier and match the committed baseline exactly.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full fast lane (regression gate) | `pytest tests/ -m "not slow" -q` | 1807 passed, 1 skipped, 53 deselected, exit 0, 97.55 s | ✓ PASS |
| Named repair-regression tests (5 modules) | `pytest TestLoraMirrorEndpoint TestSpecEnvExtras… megadna np-shim extras_guard` | 26 passed in 3.72 s | ✓ PASS |
| Marimo D-18 quadruple (1 app, live re-execution) | `pytest ...test_marimo_execution[inference_demo]` | 1 passed in 9.93 s | ✓ PASS |
| YAML leg (both validators) | `validate_yaml.py` + `pytest tests/configuration/test_yaml_load.py` | 21/21 + 21 passed | ✓ PASS |
| Docs sync gates | `check_docs_sync.py` / `check_notebook_md_sync.py` | OK exit 0 / 24-24 pairs exit 0 | ✓ PASS |
| ollama readiness (SC5 infra, live) | `curl 127.0.0.1:11434/api/tags` | qwen3.8:latest, 17.74 GB, HTTP 200 | ✓ PASS |
| Final census (committed log evidence) | read /tmp/08_09_examples.log tail | 196 passed, 1 skipped in 10774.89 s | ✓ PASS (log-verified) |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared; the phase's runnable checks are the pytest lanes and CI dispatches above.

### Requirements Coverage

| Requirement | Source Plans | Description (abridged) | Status | Evidence |
|-------------|--------------|------------------------|--------|----------|
| EXEC-02 | 08-01..09 | all 21 notebooks real execution, fail-soft | ✓ SATISFIED | Census log 196P/1S/0F; wiring 13+8; job complete |
| EXEC-03 | 08-01 | 3 marimo apps headless + defaults + exit codes | ✓ SATISFIED | D-18 quadruple; 1 re-run live |
| EXEC-04 | 08-02 | generate_bpe_dataset.py artifact in-sandbox | ✓ SATISFIED | Artifact assertions; census row 4 passed, 11.3 MB fresh |
| EXEC-05 | 08-02 | YAML real load_config on fast leg, zero new skips | ✓ SATISFIED | 21/21 live ×2; fast lane 1807/1 identical |
| REPAIR-01 | 08-02..09 | every surfaced error fixed + regression test, triaged | ✓ SATISFIED | 13 repair classes w/ tests; 26 named tests live green |
| REPAIR-03 | 08-02..04 | dnallm library bugs fixed with tests | ✓ SATISFIED | DNATokenizer, np.fromstring, allow_patterns, D-17 shims |
| REPAIR-04 | 08-01 | langchain-ollama declared in mcp extra | ✓ SATISFIED | pyproject:132 + guard test live |
| CI-04 | 08-09 | models.lock extended, ms-first, revision-pinned, source-aligned | ✓ SATISFIED | 24/14/prefix-aligned (live) |
| CI-05 | 08-03/06/07/09 | giants strategy: safetensors-only, outside quota cache | ✓ SATISFIED | Patterns + prefetch + cache-path exclusion + A4 proof |
| MCP-01 | 08-08 | ollama loopback systemd + pre-pulled model + probe; pair executes | ✓ SATISFIED | Unit + README in-repo; endpoint live; pair 2P/0S |
| MCP-02 | 08-01/08/09 | port/VRAM coexistence planned vs 6 :8000 probes | ✓ SATISFIED | D-07 stage contract in-module + ci.yml stages |

Orphaned requirements: none — REQUIREMENTS.md maps exactly the 11 IDs to Phase 8 and the 9 plans' `requirements` fields cover all 11 (union).

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| test_extras_guard.py | REPAIR-04/01 | 5 | 0 | 0 | Value (tomllib membership, installed-pair probe) | OK |
| test_transformers_compat_np.py | REPAIR-03 | 12 | 0 | 0 | Value (historical parity bytes/count/writability) | OK |
| test_megadna.py | REPAIR-03 | 6 | 0 | 0 | Value (exact ids, round-trips) | OK |
| test_notebook_execution.py contracts | REPAIR-01 | all | 0 | 0 | Value (spec env, mirror pin, JSON content) | OK |
| test_marimo_execution.py | EXEC-03 | 3 | 0 | 0 | Behavioral (export + literals + exit code) | OK |

Disabled tests on requirements: 0. Circular patterns: none (band/expected values come from committed contracts — selection.md precedent — or historical numpy semantics, not from the system under test). Size assertion in the marimo test is one of four assertions (D-18), not alone.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/runner/ollama.service` (host state) | — | `systemctl is-active ollama` reports "inactive" on this box while the endpoint answers on 127.0.0.1:11434 with qwen3.8 loaded (likely user-level unit or name mismatch on the shared host) | ℹ️ Info | None for MCP-01/D-13, which probe the HTTP endpoint; recorded so the owner can reconcile the unit name during the Phase-9 runner checks |
| `tests/examples/test_marimo_execution.py` | 101 | `st_size > 1000` present | ℹ️ Info | Not a violation — one of the four D-18 assertions (prohibition was size-ALONE); defaults/exit-code/marker all asserted and live-proven |

Debt-marker gate: zero `TBD`/`FIXME`/`XXX` markers across all 21 phase-modified files. No stub or empty-return shapes: library fixes are real implementations, all gates execute rather than silently pass, and the fail-soft summary fails loudly (exit 1) on any failure.

## Human Verification

N/A — infrastructure/testing phase with no user-facing elements; all acceptance criteria verified programmatically (live lanes, logs, git, structure checks). Owner hand-offs below are recorded decisions, not unverified phase truths.

### Open Owner Items (handed back by the executor — recorded, not gaps)

1. **Cache-quota decision** (08-09 D5): GitHub cache store measured at 10.37 GB (threshold) with zero models-cache entries ever saved; clean lock-only cache ≈15.2 GiB > 10 GB quota. Options (a) pay-as-you-go, (b) drop the models-cache layer (de-facto, proven green cold at 65-min stage 1), (c) partial ms-only — measured numbers and the prerequisite $HOME cleanup are in 08-CENSUS-ROLLUP.md. The nightly is green regardless; the decision is Phase-9 CI-06/07 planning input.
2. **Box-side $HOME cleanup**: `rm -rf ~/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base` (28 GB leftover inside a cached path; executor deletion correctly denied by the permission policy) plus the flagged 19 GB Qwen / 38 GB legacy-blob trees.
3. **Hand-off dispatch 37278002681** (completed job's first real dispatch, fired 2026-10-05T07:29:14Z, `in_progress` at verification time): outcome lands per the Phase-5 D-04 post-merge runner-confirmation boundary — Phase 9 scope.
4. **08-06 cron double-trigger job-gate** (the `30 5 * * *` schedule fires coverage-nightly/test-mamba a second time; queue-serialized on the single runner): one-line job-gate fix available if the owner wants it — separately flagged, dispositioned open.

### Gaps Summary

None. All 5 roadmap success criteria, all 3 phase-level gates, all 11 requirement IDs, all load-bearing plan truths, all artifacts (existence + substance + wiring + data flow), and all key links verified — against live runs by this verifier (fast lane, 26 named regression tests, one marimo re-execution, YAML ×2, sync gates, ollama probe), raw on-disk census logs, git history (31/31 commits present), and structure checks on ci.yml / models.lock / the harness. The four open owner items above are recorded hand-offs per the executor's protocol, not failed truths.

---

_Verified: 2026-10-05T07:58:00Z_
_Verifier: Claude (gsd-verifier)_
