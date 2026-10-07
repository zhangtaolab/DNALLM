---
phase: 08-full-execution-rollout-repair-loop
verified: 2026-10-06T23:59:39Z
status: passed
score: 8/8 must-haves verified (5 roadmap success criteria + 3 phase-level gates)
covered_files:
  - .github/workflows/ci.yml
  - .github/workflows/publish.yml
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
  - docs/example/notebooks/generation_evo_models/inference.ipynb
  - docs/example/notebooks/inference_evo_models.md
  - example/notebooks/generation_evo_models/inference.ipynb
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
covered_digest: "v3:sha256:77c54045bb608592ee4cea807e3cad2a72a6fd40750bd2dbb85c0ddd30a874ff"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 8/8
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 8: Full Execution Rollout & Repair Loop Verification Report

**Phase Goal:** Every example artifact executes for real on the nightly GPU runner — all notebooks, the marimo apps, the helper script, every YAML, and the two ollama-backed mcp_example notebooks — and every error that surfaces is fixed with regression tests across example code, the docs mirror, and the dnallm library
**Verified:** 2026-10-06T23:59:39Z (regenerated at HEAD `1531eea`)
**Status:** passed
**Re-verification:** Yes — stale-verification regeneration at HEAD after the phase's first code review (08-REVIEW.md: 1 critical + 6 warnings + 5 infos) and its CR-01 repair chain; prior report verified 2026-10-05T07:58:00Z at 8/8 with no gaps

## Goal Achievement

All 5 roadmap success criteria and all 3 phase-level gates re-verified against the live codebase at HEAD `1531eea` — not SUMMARY claims. The material delta since the prior pass is the CR-01 chain (5dc18c6 → bc601af → cd90e73 → ceea260 → 3352eb8) plus 11 review fixes (WR-01..06, IN-01..05), all dispositioned `fixed` in 08-REVIEW-DISPOSITION.md (0/12 open). This verifier re-ran live at HEAD: the **evo giants lane itself** (`-k generation_evo`: 1 passed in 106.06 s — the exact test whose subject CR-01 was), the **full fast lane** (1881 passed / 1 pre-existing skip / 53 deselected, exit 0, 97.27 s), **all 3 marimo D-18 apps** (3 passed in 20.89 s), the **YAML leg both ways** (21/21), **both docs-sync gates** (exit 0), the **ollama loopback endpoint** (live curl: qwen3.5:4b 3,324,173,934 B + qwen3.8:latest still present), and the named repair-regression sets (extras guard + np shim + megaDNA: 40 passed; model + evo fast: 201 passed; harness contracts: 12 passed; evo/megaDNA content contracts: 5 passed). Nightly runner evidence re-checked via `gh`: run 37432001711 — example-nightly, test-mamba, and coverage-nightly jobs all `success` with zero failed steps. Committed notebook outputs read directly: the EVO1 section now carries **real evo-1 evidence** (load at giants snapshot `a9be7b6…`, generation `TACCCCGCACC…` distinct from the evo2 section's `TCCATTGTACG…`, distinct scores) — CR-01 closed honestly.

### Roadmap Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | All 21 notebooks execute all code cells end-to-end with real models (fail-at-first-error per notebook, fail-soft across); 3 marimo apps headless with default-value + exit-code assertions; generate_bpe_dataset.py produces its artifact in-sandbox | ✓ VERIFIED | Census wiring live-checked: `ACTIVE_NOTEBOOKS` (13, test_notebook_execution.py:76-90) + `GATED_NOTEBOOKS` (8, :1321-1341) = exactly the 21 census ipynb (tree-enumerated: 24 files − 1 checkpoint − 2 mcp pair). **Evo giants lane re-run live by this verifier at HEAD: 1 passed in 106.06 s** (the CR-01 subject; commit 3352eb8 recorded 105.58 s) with committed outputs proving the restored evo-1 load (cell 14 `load_model_and_tokenizer("togethercomputer/evo-1-8k-base", source="huggingface")` → giants snapshot a9be7b6…, 438-tensor safetensors, distinct generation/scoring from evo2, D-21 stamp transformers 5.18.0/torch 2.11.0+cu130/flash_attn 2.8.3.post1, noFA override cell with rationale). **Marimo D-18 quadruple re-run live: 3 passed in 20.89 s** (defaults + error-artifact-absence exit-code proof + key content, test_marimo_execution.py:86-125). Script lane artifact assertions intact (out_pkl.is_file + st_size>0 + mtime>=run_start, test_script_execution.py:151-153), green in the committed final census. Phase-close census: 196 passed / 1 benign skip / 0 failed in 2:59:34 (rollup, both endpoints up, zero family skips). Runner-side: example-nightly job in run 37432001711 `success`, stage-0.5 census hard-gate pins 193/202 collected. **Scope note (post-close owner policy, not a gap):** Phase 9's D-04 directive deselects the giants-marked evo lane from the *scheduled* census (`-m "not giants"`; marker registered in pyproject; `_GIANTS_GATED` = the evo notebook) — its execution evidence is the dispatch/manual lane, tonight's gated run, the committed outputs, and this verifier's live 106.06 s run at HEAD |
| 2 | Every example YAML passes real `load_config()` Pydantic validation on the fast leg, zero new fast-leg skips | ✓ VERIFIED (live) | `validate_yaml.py`: "All YAML files passed validation." (21 files); `tests/configuration/test_yaml_load.py`: 21 passed. Full fast lane re-run live at HEAD: **1881 passed / 1 skipped / 53 deselected, exit 0, 97.27 s** — the single skip is the permanent benign no-imports entry (predict_data), zero new fast-leg skips vs every prior baseline (1807 at phase close + later-phase review-fix additions, same 1 skip) |
| 3 | Every surfaced error fixed with regression test; harness-bug vs content-bug triage explicit (no cwd false-repairs); langchain-ollama in mcp extra; docs mirror regenerated per repair | ✓ VERIFIED | 13 phase repair classes + the 12 review findings all carry same-commit (or RED-then-GREEN) regression tests — re-run live by this verifier: extras guard (langchain-ollama>=1.1.0 in mcp extra, tomllib-proven + ipython>=8.31,<9 pin), np.fromstring shim (**15 tests** at HEAD incl. cd90e73's str-encode), megaDNA (**20 tests** incl. WR-02 TestMegadnaCheckpointSelection 9, WR-03 TestMegadnaExtraDoesNotMutateModuleList, WR-05 TestMegadnaLoadHardening 4, plus the original DNATokenizer 6), model.py (WR-04 TestDispatchChain partial-result-survival, WR-06 unknown-head ValueError), evo (IN-01 .pt-less ValueError, IN-03 mixed-case source), harness contracts (TestSpecEnvOverrides, TestLoraMirrorEndpoint, TestEvoIsolatedLane, TestProbeHonesty — 12 passed), CR-01 content contracts (TestEvoNotebookContentContracts 2 + megadna siblings 3 — 5 passed). Triage explicit per failure in the rollup repair tables; **zero `os.chdir` in the harness** (grep-verified). Every notebook repair carries its docs mirror same-commit; both sync gates exit 0 live (24/24 pairs) and the evo mirror is raw byte-identical to the notebook (`cmp`) |
| 4 | models.lock carries all newly-executed ids with ms-first prefixes aligned to each notebook's source= route and revision pins; evo-1 safetensors-only via allow_patterns with giants tiered outside the 10GB-quota cache | ✓ VERIFIED (live) | models.lock counted live at HEAD: **24 `^(hf\|ms)` rows, 14 `@sha`-pinned**; evo-1 row carries the GIANTS-tier comment (safetensors-only into ~/models-giants, outside the quota cache, D-14/CI-05); header documents the D-15 pin format (D-11 provenance-only note added by Phase 7's review). Prefix-vs-source alignment spot-checked against ACTIVE source= lines: plant-dnagpt-BPE/finetune_generation → ms + `source="modelscope"`; lingxusb/megaDNA_updated, PlantCAD2 → hf + `source="huggingface"`. Library half intact at HEAD: conditional allow_patterns passthrough (model.py), `_EVO1_SAFETENSORS_ONLY_PATTERNS` (evo.py:46-52) now including `*.py` (bc601af — auto_map cross-repo code resolution; `.pt` still excluded by omission, in-code comment proves intent), forwarded at the evo-1 hub fetch only (evo.py:417 — evo2 unfiltered per prohibition). Giants-stays-outside holds by construction: dedicated `~/models-giants/hub` dir; Phase 9's D-11 owner decision additionally removed the models-cache layer outright (recorded in ci.yml + lock header), so the never-evict clause is satisfied vacuously and the sanctioned giants copy remains the local tier |
| 5 | Both mcp_example notebooks execute end-to-end against loopback-only ollama on the nightly runner (systemd unit in-repo, pre-pulled model, readiness probe); port/VRAM coexistence planned against the 6 MCP :8000 probes; typed network-unavailable skip is fallback only | ✓ VERIFIED | `scripts/runner/ollama.service` in-repo with `OLLAMA_HOST=127.0.0.1:11434` (loopback pin + security rationale) and the WR-01-reconciled num_ctx deferral narrative; `scripts/runner/README.md` documents install/enable/pull (now qwen3.5:4b ~3.3GB after the owner's 261006 model swap — swept consistently through unit, README, probe docstring, and ci.yml comments), the live-drift notice, and the D-13 fallback semantics. D-13 probe `_probe_http_with_retry` (~60 s × 2 s, patchable sleep, full attempt log in message) wired into `_gate_ollama_stack`; skip is infra-missing only (TestProbeHonesty green). Live probe by verifier: `curl 127.0.0.1:11434/api/tags` answers with qwen3.5:4b (3,324,173,934 B) — and the runner shares this box's owner-enabled service. D-07 coexistence implemented end-to-end in ci.yml (stages 1/1.5/2/2.5/3/4 read at HEAD: stage-1 `-k "not mcp_example"` deselect, hard-gate hygiene floors ≥35 Gi with fail-closed parse, 3 streamable + sse-restart + 3 sse probes, fresh streamable-http server for the pair, server stopped after); run 37432001711 example-nightly `success` proves the nightly executes the pair (fail-soft summary requires zero failures for that conclusion) |

### Phase-Level Gates

| Gate | Status | Evidence |
|------|--------|----------|
| Regression gate: fast lane green | ✓ PASS | Re-run live by this verifier at HEAD: **1881 passed / 1 skipped / 53 deselected, exit 0, 97.27 s** (`pytest tests/ -m "not slow" -q`) — superset of the phase-close 1807/1 baseline (later-phase review fixes added tests), same single benign skip, zero new skips |
| Census rollup closure | ✓ PASS | 08-CENSUS-ROLLUP.md carries the final D-03 census (196P/1S/0F, 2:59:34, both endpoints up), the per-item PASS table (33 PASS rows), all four family closures, the First-Dispatch Consumption table (every runner failure classified + dispositioned with commit shas), and the cache-quota measurement + owner decision request |
| STATE / ROADMAP / REQUIREMENTS consistency | ✓ PASS | REQUIREMENTS.md marks all 11 Phase-8 requirements `[x]` Complete; ROADMAP shows 9/9 plans executed; state.json Phase 8 `complete` (Phase 9 also complete — milestone consistent) |
| Code-review closure (this regeneration's trigger) | ✓ PASS | 08-REVIEW-DISPOSITION.md: 12/12 findings `fixed`, 0 open — CR-01 via 5dc18c6 + gate-by-gate chain bc601af/cd90e73/cea260/3352eb8; WR-01..06 and IN-01..05 each with cited commits (WR-05/IN-01/IN-02/IN-04 via 08-REVIEW-FIX.md a7baff1/66650c7/5af5929/40280e3); every dnallm/ change ships same-commit tests (owner rule honored — verifier re-ran them live, all green) |

### CR-01 Chain Verification (the regeneration's material delta)

The critical finding was that the evo notebook's EVO1 section never loaded evo-1 (call dropped in a7ed221; committed outputs were evo2 output). Verified fixed at HEAD, gate by gate:

1. **5dc18c6** — load call restored into the cell after `model_name = "togethercomputer/evo-1-8k-base"`; `TestEvoNotebookContentContracts` added (2 tests: load-cell pin; load-precedes-second-DNAInference-build with model=model/tokenizer=tokenizer). Live: green.
2. **bc601af** — giants fetch pattern set gained `*.py` (evo-1 auto_map cross-repo code; `.pt` still excluded by omission — in-code comment at evo.py:30-45 records the full rationale including the evo-1-131k-base sibling-repo prefetch caveat). Live: `test_hub_fetch_is_safetensors_only` green in the 201-test model+evo run.
3. **cd90e73** — `_np_fromstring` encodes bare `str` to utf-8 before `frombuffer` (stripedhyena's CharLevelTokenizer passes bare str). Live: 15 shim tests green.
4. **ceea260** — noFA capability override cell in the notebook (flash-attn wheel imports under torch 2.11.0+cu130 but its C++ op has no registered torch schema; the import-only probe cannot see that; noFP8-precedent pattern). Verified present as visible notebook content with rationale.
5. **3352eb8** — full honest re-execution: committed outputs carry real evo-1 evidence (verified directly by this verifier: pinned snapshot a9be7b6…, distinct generation `TACCCCGCACCGGGGCAGG…` vs evo2's `TCCATTGTACGAATTG…`, distinct scores −1.4132/−1.3698, timestamps 07:44-07:45 matching the commit, 183,347-byte notebook within the 2 MB budget); gated-lane green 105.58 s per commit, **reconfirmed live at HEAD: 1 passed in 106.06 s**; mirrors `cmp`-identical.

The chain is the phase goal's own repair loop operating: every surfaced error (unreachable offline load, str-vs-bytes tokenizer input, flash-attn schema mismatch, stale outputs) was fixed with same-change tests and honest re-execution — the success criterion for this regeneration.

### Observable Truths (per-plan, load-bearing)

All 31 phase-8 commits plus the 21 review/CR-01 commits verified present on `phs`. Plan truths re-confirmed at HEAD (live-proven items marked):

**08-01** — example-nightly job present (ci.yml:592) with cron-gated event filter (`workflow_dispatch || schedule && '30 5 * * *'` — fork/PR can never trigger), D-07 staged layout, D-08 fail-soft (stage-results.txt ledger + end-anchored non-zero grep summary; **zero actual `continue-on-error` steps — YAML-parsed**, the only textual match is the comment prohibiting it), full extras `.[base,fla,dev,mcp]`, runner-inventory probes; langchain-ollama>=1.1.0 in mcp extra (tomllib-live) + guard test (live); marimo D-18 quadruple (3 apps live).

**08-02** — D-17 ledger on disk with per-item verdicts + pristine-snapshot proof; EXEC-04 script lane healed (artifact freshness assertions intact); EXEC-05 YAML 21/21 (live, both validators); D-03 baseline rollup; ipython>=8.31,<9 pin (tomllib-live); rice/combined seeding repairs (green in fast lane).

**08-03** — conditional allow_patterns passthrough at both layers (model.py), kwargs rebuilt per attempt; evo-1 pattern set wired (evo.py:417, now with `*.py`); probe-gated np.fromstring shim (15 tests live, incl. the CR-01 str-encode); spec env override sandwich (TestSpecEnvOverrides live).

**08-04/08-05** — DNATokenizer unknown→id 1 (6 original tests + WR-02/03/05 hardening = 20 megadna tests live); finetune_generation + both siblings repaired with D-21 stamps, pinned installs, byte-synced mirrors (sync gates exit 0 live); megaDNA family CLOSED in rollup.

**08-06/08-07** — giants tier + isolated dnallm-evo-kernel lane (`TestEvoIsolatedLane` live); evo notebook repaired and — post CR-01 — **executing its evo-1 leg for real with honest committed outputs (verifier-rerun green at HEAD)**; evo2 native `.pt` scoped note; OPTIONAL_IMPORT_MODULES seam (cd6debd) green in fast lane.

**08-08** — lora pair green (TestLoraMirrorEndpoint live; mirror endpoint pinned at the spec-env seam); ollama unit + README in-repo and internally consistent post WR-01/qwen3.5:4b swap; D-13 retry probe + TestProbeHonesty (live); D-07 stage contract in-module and implemented in ci.yml.

**08-09** — models.lock 24/14/prefix-aligned (live count); first-dispatch consumption + quota record in rollup; job completion with cache-before-consumer ordering; nightly runner evidence re-verified via gh (37432001711 all-green).

**Score:** 8/8 must-haves verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci.yml` (example-nightly job) | Staged job: prereqs, census hard-gate, stages 0-4, fail-soft | ✓ VERIFIED | Job at :592-1093 read in full; YAML-parsed: zero continue-on-error steps, cron-gated event filter, all stage tokens present; Phase-9 evolutions (D-11 no-cache, D-04 giants deselect, stage-0.5 census gate) recorded with owner-directive comments |
| `models.lock` | 24 rows, 14 pinned, prefix-aligned, giants comment | ✓ VERIFIED (live) | Counted + per-id alignment spot-checked against notebook ACTIVE source= lines |
| `pyproject.toml` | langchain-ollama>=1.1.0 in mcp extra; ipython>=8.31,<9 in notebook extra | ✓ VERIFIED (live) | tomllib-parsed both; guard tests green |
| `dnallm/models/special/megadna.py` | DNATokenizer unknown→id 1; WR-02 checkpoint map; WR-03 no list mutation; WR-05 revision pin + weights_only-first | ✓ VERIFIED (live) | 20 tests green incl. all four review-fix classes |
| `dnallm/models/special/evo.py` | safetensors-only(+`*.py`) patterns on the evo-1 fetch only; IN-01/IN-03 hardening | ✓ VERIFIED (live) | Patterns at :46-52, single allow_patterns call site :417; 201 model+evo fast tests green |
| `dnallm/models/model.py` | conditional allow_patterns passthrough; WR-04 per-half chain merge; WR-06 ValueError | ✓ VERIFIED (live) | Tests green in the 201-test run |
| `dnallm/utils/transformers_compat.py` | probe-gated np.fromstring binary-mode shim (str-encoding) | ✓ VERIFIED (live) | 15 tests green |
| `example/notebooks/generation_evo_models/inference.ipynb` + docs mirrors | evo-1 load restored, noFA cell, D-21 stamp, honest outputs, byte-synced mirrors | ✓ VERIFIED (live) | Outputs read directly (distinct evo-1 evidence at pinned snapshot); lane re-run green; `cmp`-identical mirrors; sync gates exit 0 |
| `scripts/runner/ollama.service` + `README.md` | loopback-only unit + owner steps (one consistent story) | ✓ VERIFIED | Loopback pin; num_ctx deferral narrative reconciled; qwen3.5:4b pull documented; live-drift notice |
| `tests/examples/_execution.py` + `test_notebook_execution.py` | specs for all 21, gates, env sandwich, D-13 probe, D-07 contract, CR-01 content contracts | ✓ VERIFIED (live) | 13 ACTIVE + 8 GATED; 12 harness-contract tests + 5 content-contract tests green |
| `tests/examples/test_marimo_execution.py` | D-18 quadruple | ✓ VERIFIED (live) | 3 apps re-run live: passed in 20.89 s |
| `tests/examples/test_script_execution.py` | healed script lane + rice seeding | ✓ VERIFIED | Artifact freshness assertions intact; green in census + nightly |
| 08-CENSUS-ROLLUP.md / 08-D17-DISPOSITION.md / 08-REVIEW-DISPOSITION.md | census ledger + D-17 ledger + review closure | ✓ VERIFIED | All three read; final census + quota + first-dispatch tables; 12/12 fixed, 0 open |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| example-nightly stage 1 | example census tests | `pytest tests/examples -k "not mcp_example" -m "not giants"` + census hard-gate (193/202) | ✓ WIRED |
| example-nightly stages 2/3 | MCP :8000 probes + mcp pair | dnallm-mcp-server background + readiness curl (status-code shape for sse) + `-k mcp_example` | ✓ WIRED (job success in 37432001711) |
| lora specs | hf-mirror endpoint | spec `env.HF_ENDPOINT` through the run_notebook sandwich | ✓ WIRED (TestLoraMirrorEndpoint live) |
| evo spec | giants tier | spec `env.HF_HUB_CACHE`/`HF_HUB_OFFLINE` + dnallm-evo-kernel kernelspec | ✓ WIRED (TestEvoIsolatedLane live + 106.06 s lane run) |
| evo-1 hub fetch | safetensors+code-only set | `_EVO1_SAFETENSORS_ONLY_PATTERNS` (incl. `*.py`, excl. `.pt`) via allow_patterns | ✓ WIRED (wiring test live) |
| notebook repairs | docs mirrors | same-commit mirror files; check_notebook_md_sync + check_docs_sync | ✓ WIRED (both exit 0 live; evo mirror `cmp`-identical) |
| ollama unit/README | live runner service | owner enable step; D-13 probe treats down-service as evidence-backed typed skip | ✓ WIRED (endpoint live; probe tests green) |

### Data-Flow Trace (Level 4)

No rendered-value chains end in static returns. The census metrics flow from real model executions; the lock prefixes flow from the notebooks' actual ACTIVE routes (spot-checked per-id); the marimo default assertions read the real exported HTML text; the evo-1 outputs flow from a real load of the pinned giants snapshot (path + distinct sequences verified in the committed outputs); the fast-lane counts were regenerated live by this verifier.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Evo giants lane (CR-01 subject) | `pytest tests/examples/test_notebook_execution.py -k generation_evo -q` | 1 passed in 106.06 s, 67 deselected | ✓ PASS |
| Full fast lane (regression gate) | `pytest tests/ -m "not slow" -q` | 1881 passed, 1 skipped, 53 deselected, exit 0, 97.27 s | ✓ PASS |
| Marimo D-18 quadruple (all 3 apps) | `pytest tests/examples/test_marimo_execution.py -q` | 3 passed in 20.89 s | ✓ PASS |
| Repair-regression set A (extras/np/megadna) | `pytest tests/test_extras_guard.py tests/utils/test_transformers_compat_np.py tests/models/test_special/test_megadna.py -q` | 40 passed in 3.76 s | ✓ PASS |
| Repair-regression set B (model/evo fast) | `pytest tests/models/test_model.py tests/models/test_special/test_evo.py -m "not slow and not giants" -q` | 201 passed, 2 deselected | ✓ PASS |
| Harness contracts (spec env / lora mirror / evo lane / probe honesty) | `pytest ::TestSpecEnvOverrides ::TestLoraMirrorEndpoint ::TestEvoIsolatedLane ::TestProbeHonesty -q` | 12 passed in 5.41 s | ✓ PASS |
| Notebook content contracts (evo CR-01 + megadna siblings) | `pytest ::TestEvoNotebookContentContracts ::TestMegadnaSiblingContentContracts -q` | 5 passed in 0.85 s | ✓ PASS |
| YAML leg (both validators) | `validate_yaml.py` + `pytest tests/configuration/test_yaml_load.py` | 21/21 + 21 passed | ✓ PASS |
| Docs sync gates | `check_notebook_md_sync.py` / `check_docs_sync.py` | 24/24 pairs exit 0 / exit 0 | ✓ PASS |
| ollama readiness (SC5 infra, live) | `curl 127.0.0.1:11434/api/tags` | qwen3.5:4b (3,324,173,934 B) + qwen3.8:latest, HTTP 200 | ✓ PASS |
| example-nightly structure (YAML parse) | python yaml parse + token asserts | crons [03:00, 05:30]; zero continue-on-error; all tokens present | ✓ PASS |
| Nightly runner runs (gh) | `gh run view 37432001711` | conclusion success; example-nightly/test-mamba/coverage-nightly jobs success, 0 failed steps | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared; the phase's runnable checks are the pytest lanes and CI dispatches above.

### Requirements Coverage

| Requirement | Source Plans | Description (abridged) | Status | Evidence |
|-------------|--------------|------------------------|--------|----------|
| EXEC-02 | 08-01..09 | all 21 notebooks real execution, fail-soft | ✓ SATISFIED | Wiring 13+8; final census 196P/1S/0F; evo lane verifier-rerun green at HEAD; nightly 37432001711 success |
| EXEC-03 | 08-01 | 3 marimo apps headless + defaults + exit codes | ✓ SATISFIED | D-18 quadruple; 3 apps re-run live at HEAD |
| EXEC-04 | 08-02 | generate_bpe_dataset.py artifact in-sandbox | ✓ SATISFIED | Artifact freshness assertions; census row PASS; unchanged and green since phase close |
| EXEC-05 | 08-02 | YAML real load_config on fast leg, zero new skips | ✓ SATISFIED | 21/21 live ×2 at HEAD; fast lane same single benign skip |
| REPAIR-01 | 08-02..09 | every surfaced error fixed + regression test, triaged | ✓ SATISFIED | 13 repair classes + 12 review findings, all with tests (re-run live); no cwd false-repairs; rollup triage tables |
| REPAIR-03 | 08-02..04 | dnallm library bugs fixed with tests | ✓ SATISFIED | DNATokenizer, np.fromstring, allow_patterns, WR-02..06, IN-01..03 — 256+ tests across the touched modules green live |
| REPAIR-04 | 08-01 | langchain-ollama declared in mcp extra | ✓ SATISFIED | tomllib-live + guard test green |
| CI-04 | 08-09 | models.lock extended, ms-first, revision-pinned, source-aligned | ✓ SATISFIED | 24/14/prefix-aligned (live count at HEAD) |
| CI-05 | 08-03/06/07/09 | giants strategy: safetensors-only, outside quota cache | ✓ SATISFIED | Patterns (+`*.py`, −`.pt`) + dedicated giants dir + D-14 lock comment + live evo-1 offline load proof |
| MCP-01 | 08-08 | ollama loopback systemd + pre-pulled model + probe; pair executes | ✓ SATISFIED | Unit + README consistent (qwen3.5:4b); endpoint live; pair green in nightly + census |
| MCP-02 | 08-01/08/09 | port/VRAM coexistence planned vs 6 :8000 probes | ✓ SATISFIED | D-07 stage contract in-module + implemented stages 1.5/2/2.5/3 with hard-gate hygiene floors |

Orphaned requirements: none — REQUIREMENTS.md maps exactly the 11 IDs to Phase 8; the 9 plans' `requirements` fields cover all 11 (union), and all 11 are marked `[x]` Complete.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| test_extras_guard.py | REPAIR-04/01 | 5 | 0 | 0 | Value (tomllib membership, installed-pair probe) | OK |
| test_transformers_compat_np.py | REPAIR-03 | 15 | 0 | 0 | Value (historical parity bytes/count/writability, str-encode) | OK |
| test_megadna.py | REPAIR-03 | 20 | 0 | 0 | Value (exact ids, checkpoint filenames, mutation-unchanged, pin/weights_only order) | OK |
| test_model.py | REPAIR-03/CI-05 | 167+ | 0 | 0 | Value (kwargs forward/omit, chain survival, ValueError match) | OK |
| test_evo.py | CI-05/REPAIR-03 | 34 | 0 (2 giants-deselected in fast filter) | 0 | Value (pattern set, revision selection, ValueError) | OK |
| test_notebook_execution.py contracts | REPAIR-01/EXEC-02 | all | 0 | 0 | Value (spec env, mirror pin, JSON content incl. CR-01 load-cell pin) | OK |
| test_marimo_execution.py | EXEC-03 | 3 | 0 | 0 | Behavioral (export + literals + exit-code artifact) | OK |

Disabled tests on requirements: 0 (grep for skip/skipif/xit across the regression files: none). Circular patterns: none — expected values come from committed contracts, historical numpy semantics, or live HF repo listings fetched during the fix (documented in 08-REVIEW-FIX.md), never from the system under test.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `dnallm/models/model.py` | 865 | `# TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (git-blamed to cd7f6c6, 2026-03-27 — predates this milestone); not phase-introduced |
| `dnallm/utils/transformers_compat.py` | 671 | `# TODO (joao): remove …` | ℹ️ Info | Vendored upstream transformers provenance comment — intentional |
| `tests/examples/test_marimo_execution.py` | 101 | `st_size > 1000` present | ℹ️ Info | Not a violation — one of the four D-18 assertions (prohibition was size-ALONE); defaults/exit-code/marker all asserted and live-proven |

Debt-marker gate: zero `TBD`/`FIXME`/XXX markers across all phase-modified files (the two TODOs above are warning-level infos with provenance). No stub or empty-return shapes: library fixes are real implementations (verifier-rerun green), the fail-soft summary fails loudly, and the giants-deselect is an owner-policy marker with the dispatch lane retained — not a silent skip.

### Decision Coverage

Decision coverage gate (non-blocking): 21/21 trackable CONTEXT.md decisions honored by shipped artifacts (`check.decision-coverage-verify`: honored 21, total 21, not_honored []). Re-verification evidence gate: no new-scope Step-7 blockers arose — the only post-prior-pass findings are the review chain's own, all fixed with cited commits and live-rerun tests.

### Advisory (New Scope, Unevidenced)

None — every concern surfaced during this regeneration resolved to either live-verified evidence or an already-dispositioned review finding.

### Human Verification

N/A — infrastructure/testing phase with no user-facing elements; all acceptance criteria verified programmatically (live lanes by this verifier, committed census logs, gh-verified nightly runs, git history, structure checks). The items below are recorded owner hand-offs and post-close policy decisions, not unverified phase truths.

### Open Owner Items (recorded hand-offs / deferred follow-ups — not gaps)

1. **Cache-quota decision** (08-09 D5, still the standing open item): measured 10.37 GB store at threshold, lock-only cache ≈15.2 GiB > quota; options (a) pay-as-you-go (b) drop the models-cache layer (de-facto since Phase-9 D-11) (c) partial ms-only. Numbers in 08-CENSUS-ROLLUP.md.
2. **Box-side $HOME cleanup**: the 28 GB evo-1 full-repo leftover inside `~/.cache/huggingface/hub` plus the flagged 19 GB Qwen / 38 GB legacy-blob trees (owner action; never cleaned by default per the runner-ops rule).
3. **CR-01 deferred follow-ups (recorded in-code)**: (a) `is_flash_attention_capable` is import-only — a live-probe upgrade would have caught the flash-attn wheel/torch schema mismatch the notebook's noFA cell now documents; (b) a manual giants-lane prefetch must include the evo-1-131k-base sibling repo's `*.py` (evo.py comment); (c) rebuilding the flash-attn wheel against torch 2.11 would remove the need for the noFA override — optional.
4. **WR-01 residual**: the qwen3.5:4b Modelfile num_ctx re-probe (`ollama show qwen3.5:4b --modelfile`) remains an owner action before the inert `OLLAMA_CONTEXT_LENGTH` pin can be trusted to govern; the live-runner loopback re-apply step in the README closes the recorded 0.0.0.0 drift.
5. **08-06 cron double-trigger job-gate** (the 05:30 entry re-fires coverage-nightly/test-mamba; queue-serialized): one-line fix available if the owner wants it.

### Gaps Summary

None. All 5 roadmap success criteria, all 3 phase-level gates (plus the code-review closure gate that triggered this regeneration), all 11 requirement IDs, all load-bearing plan truths, all artifacts (existence + substance + wiring + data flow), and all key links verified at HEAD `1531eea` — against live runs by this verifier (evo giants lane 106.06 s, fast lane 1881P/1S, marimo ×3, YAML ×2, sync gates, ollama probe, 250+ named regression tests), the committed honest evo-1 outputs, gh-verified nightly runs, git history, and structure checks. The CR-01 repair chain is the phase goal's own repair loop operating as designed: honestly fixed, tested, and executed.

---

_Verified: 2026-10-06T23:59:39Z (regenerated at HEAD 1531eea)_
_Verifier: Claude (gsd-verifier)_
