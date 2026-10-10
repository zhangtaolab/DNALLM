---
phase: 08-full-execution-rollout-repair-loop
verified: 2026-10-07T00:28:53Z
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
covered_digest: "v3:sha256:e30d1826c1ebb832377802a367bf99fc82d10126f45419678e0e99a4a5315115"
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
**Verified:** 2026-10-07T00:28:53Z (regenerated at HEAD `acb675c`)
**Status:** passed
**Re-verification:** Yes — stale-fingerprint regeneration at HEAD after the census-ratchet re-pin (commit `86c1fe8`, ci.yml stage-0.5 triple 193/202 → 197/206, closing phase-09's D-03 finding that the 05/08 review-repair cycles' 4 fast tests had outgrown the pin); prior report verified 2026-10-06T23:59:39Z at HEAD `1531eea` with 8/8 and no gaps

## Regeneration Delta Audit (this round's scope)

`git diff 1531eea..HEAD` on the covered set is **exactly the claimed two-line census-literal re-pin** in `.github/workflows/ci.yml` (stage-0.5 grep + FAIL-echo lines, `193/202` → `197/206`, commit `86c1fe8`) — every other covered file is **byte-identical** to the prior verified state, so the prior pass's live evidence transfers unchanged. Remaining delta 1531eea..HEAD is planning-only and outside the covered set: `.planning/PROJECT.md`, `.planning/STATE.md`, phase-05/07/09 VERIFICATION regenerations, and `09-CENSUS-ROLLUP.md` (which records the re-pin at its line 375: "197/206 tests collected (9 deselected) — was 193/202").

Live checks run by this verifier this round (all fast, no heavy lanes re-executed — their evidence from the prior pass at this code state stands):

1. **Census collection at HEAD: `197/206 tests collected (9 deselected) in 0.78s`** — exact stage-0.5 command (`pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"`) re-run live; the re-pinned gate literal matches measured reality (the prior pin had fallen behind the tree: the 4 growth tests already predated `1531eea`, phase-09's convergence verification reported the ratchet gap, `86c1fe8` closed it).
2. **Provenance of the +4 verified**: `git show` on `374e8e6` (+1: `test_non_mapping_override_section_raises`), `9d44cbf` (+1: `TestVenvProbeTimeoutContract::test_hung_probe_returns_false_with_timeout_evidence`), `5dc18c6` (+2: `TestEvoNotebookContentContracts`) — all in `tests/examples/test_notebook_execution.py`, all ancestors of `1531eea`; 193+4=197, 202+4=206, arithmetic exact.
3. **All 4 growth tests re-run live at HEAD: 4 passed** (2 + 2 across two invocations, 1.54 s + 1.66 s) — the pin's growth is real passing tests, not phantom collection entries.
4. **Re-pinned workflow still valid**: YAML-parsed at HEAD — crons [03:00, 05:30], `workflow_dispatch` present, **0 continue-on-error steps**, stage-0.5 step present carrying the 197/206 literal, zero `193/202` residue anywhere in the file. No debt markers in the changed lines.
5. **Planning-state consistency re-checked**: REQUIREMENTS.md still marks all 11 Phase-8 IDs `[x]` Complete; state.json Phase 8 `complete`; ROADMAP.md untouched by the delta.
6. **Decision coverage re-run**: all trackable CONTEXT.md decisions honored (`not_honored: []`).
7. **Nightly dispatch 37550730293**: in flight (`queued`, headSha `86c1fe8` — the re-pin HEAD) at verification time; cited as in-flight only — its result is phase-09's evidence item, not this report's.

## Goal Achievement

All 5 roadmap success criteria and all 3 phase-level gates verified — carried from the prior pass at byte-identical code state (covered set unchanged except the census literal), with this round's delta live-checked above and the prior pass's live evidence summarized here. The prior pass (2026-10-06T23:59:39Z, HEAD `1531eea`) re-ran live: the **evo giants lane itself** (`-k generation_evo`: 1 passed in 106.06 s — the exact test whose subject CR-01 was), the **full fast lane** (1881 passed / 1 pre-existing skip / 53 deselected, exit 0, 97.27 s), **all 3 marimo D-18 apps** (3 passed in 20.89 s), the **YAML leg both ways** (21/21), **both docs-sync gates** (exit 0), the **ollama loopback endpoint** (live curl: qwen3.5:4b 3,324,173,934 B + qwen3.8:latest present), and the named repair-regression sets (extras guard + np shim + megaDNA: 40 passed; model + evo fast: 201 passed; harness contracts: 12 passed; evo/megaDNA content contracts: 5 passed). Nightly runner evidence re-checked via `gh`: run 37432001711 — example-nightly, test-mamba, and coverage-nightly jobs all `success` with zero failed steps. Committed notebook outputs read directly: the EVO1 section carries **real evo-1 evidence** (load at giants snapshot `a9be7b6…`, generation `TACCCCGCACC…` distinct from the evo2 section's `TCCATTGTACG…`, distinct scores) — CR-01 closed honestly.

### Roadmap Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | All 21 notebooks execute all code cells end-to-end with real models (fail-at-first-error per notebook, fail-soft across); 3 marimo apps headless with default-value + exit-code assertions; generate_bpe_dataset.py produces its artifact in-sandbox | ✓ VERIFIED | Census wiring live-checked: `ACTIVE_NOTEBOOKS` (13, test_notebook_execution.py:76-90) + `GATED_NOTEBOOKS` (8, :1321-1341) = exactly the 21 census ipynb (tree-enumerated: 24 files − 1 checkpoint − 2 mcp pair). **Evo giants lane re-run live by this verifier at 1531eea: 1 passed in 106.06 s** (the CR-01 subject; commit 3352eb8 recorded 105.58 s) with committed outputs proving the restored evo-1 load (cell 14 `load_model_and_tokenizer("togethercomputer/evo-1-8k-base", source="huggingface")` → giants snapshot a9be7b6…, 438-tensor safetensors, distinct generation/scoring from evo2, D-21 stamp transformers 5.18.0/torch 2.11.0+cu130/flash_attn 2.8.3.post1, noFA override cell with rationale) — covered files byte-identical at HEAD, evidence carries. **Marimo D-18 quadruple re-run live: 3 passed in 20.89 s** (defaults + error-artifact-absence exit-code proof + key content, test_marimo_execution.py:86-125). Script lane artifact assertions intact (out_pkl.is_file + st_size>0 + mtime>=run_start, test_script_execution.py:151-153), green in the committed final census. Phase-close census: 196 passed / 1 benign skip / 0 failed in 2:59:34 (rollup, both endpoints up, zero family skips). Runner-side: example-nightly job in run 37432001711 `success`; **stage-0.5 census hard-gate now pins 197/206 collected — re-pinned 193/202 → 197/206 by 86c1fe8 after the 4 review-cycle fast tests (374e8e6/9d44cbf/5dc18c6) grew the collection; collection re-measured live at HEAD: 197/206 in 0.78 s**. **Scope note (post-close owner policy, not a gap):** Phase 9's D-04 directive deselects the giants-marked evo lane from the *scheduled* census (`-m "not giants"`; marker registered in pyproject; `_GIANTS_GATED` = the evo notebook) — its execution evidence is the dispatch/manual lane, the nightly gated run, the committed outputs, and the verifier's live 106.06 s run |
| 2 | Every example YAML passes real `load_config()` Pydantic validation on the fast leg, zero new fast-leg skips | ✓ VERIFIED (live) | `validate_yaml.py`: "All YAML files passed validation." (21 files); `tests/configuration/test_yaml_load.py`: 21 passed. Full fast lane re-run live at 1531eea: **1881 passed / 1 skipped / 53 deselected, exit 0, 97.27 s** — the single skip is the permanent benign no-imports entry (predict_data), zero new fast-leg skips vs every prior baseline (1807 at phase close + later-phase review-fix additions, same 1 skip); tests/ byte-identical at HEAD |
| 3 | Every surfaced error fixed with regression test; harness-bug vs content-bug triage explicit (no cwd false-repairs); langchain-ollama in mcp extra; docs mirror regenerated per repair | ✓ VERIFIED | 13 phase repair classes + the 12 review findings all carry same-commit (or RED-then-GREEN) regression tests — re-run live by this verifier at 1531eea: extras guard (langchain-ollama>=1.1.0 in mcp extra, tomllib-proven + ipython>=8.31,<9 pin), np.fromstring shim (**15 tests** incl. cd90e73's str-encode), megaDNA (**20 tests** incl. WR-02 TestMegadnaCheckpointSelection 9, WR-03 TestMegadnaExtraDoesNotMutateModuleList, WR-05 TestMegadnaLoadHardening 4, plus the original DNATokenizer 6), model.py (WR-04 TestDispatchChain partial-result-survival, WR-06 unknown-head ValueError), evo (IN-01 .pt-less ValueError, IN-03 mixed-case source), harness contracts (TestSpecEnvOverrides, TestLoraMirrorEndpoint, TestEvoIsolatedLane, TestProbeHonesty — 12 passed), CR-01 content contracts (TestEvoNotebookContentContracts 2 + megadna siblings 3 — 5 passed, and the 2 evo contracts re-run green at HEAD this round). Triage explicit per failure in the rollup repair tables; **zero `os.chdir` in the harness** (grep-verified). Every notebook repair carries its docs mirror same-commit; both sync gates exit 0 live (24/24 pairs) and the evo mirror is raw byte-identical to the notebook (`cmp`) |
| 4 | models.lock carries all newly-executed ids with ms-first prefixes aligned to each notebook's source= route and revision pins; evo-1 safetensors-only via allow_patterns with giants tiered outside the 10GB-quota cache | ✓ VERIFIED (live) | models.lock counted live at 1531eea: **24 `^(hf\|ms)` rows, 14 `@sha`-pinned** (file byte-identical at HEAD); evo-1 row carries the GIANTS-tier comment (safetensors-only into ~/models-giants, outside the quota cache, D-14/CI-05); header documents the D-15 pin format. Prefix-vs-source alignment spot-checked against ACTIVE source= lines: plant-dnagpt-BPE/finetune_generation → ms + `source="modelscope"`; lingxusb/megaDNA_updated, PlantCAD2 → hf + `source="huggingface"`. Library half intact at HEAD: conditional allow_patterns passthrough (model.py), `_EVO1_SAFETENSORS_ONLY_PATTERNS` (evo.py:46-52) including `*.py` (bc601af — auto_map cross-repo code resolution; `.pt` still excluded by omission, in-code comment proves intent), forwarded at the evo-1 hub fetch only (evo.py:417 — evo2 unfiltered per prohibition). Giants-stays-outside holds by construction: dedicated `~/models-giants/hub` dir; Phase 9's D-11 owner decision additionally removed the models-cache layer outright, so the never-evict clause is satisfied vacuously and the sanctioned giants copy remains the local tier |
| 5 | Both mcp_example notebooks execute end-to-end against loopback-only ollama on the nightly runner (systemd unit in-repo, pre-pulled model, readiness probe); port/VRAM coexistence planned against the 6 MCP :8000 probes; typed network-unavailable skip is fallback only | ✓ VERIFIED | `scripts/runner/ollama.service` in-repo with `OLLAMA_HOST=127.0.0.1:11434` (loopback pin + security rationale) and the WR-01-reconciled num_ctx deferral narrative; `scripts/runner/README.md` documents install/enable/pull (qwen3.5:4b ~3.3GB after the owner's 261006 model swap — swept consistently through unit, README, probe docstring, and ci.yml comments), the live-drift notice, and the D-13 fallback semantics. D-13 probe `_probe_http_with_retry` (~60 s × 2 s, patchable sleep, full attempt log in message) wired into `_gate_ollama_stack`; skip is infra-missing only (TestProbeHonesty green). Live probe by verifier at 1531eea: `curl 127.0.0.1:11434/api/tags` answers with qwen3.5:4b (3,324,173,934 B). D-07 coexistence implemented end-to-end in ci.yml (stages 1/1.5/2/2.5/3/4; stage-1 `-k "not mcp_example"` deselect, hard-gate hygiene floors ≥35 Gi with fail-closed parse, 3 streamable + sse-restart + 3 sse probes, fresh streamable-http server for the pair, server stopped after); run 37432001711 example-nightly `success` proves the nightly executes the pair; a further dispatch (37550730293) is in flight at the re-pin HEAD — phase-09's evidence item |

### Phase-Level Gates

| Gate | Status | Evidence |
|------|--------|----------|
| Regression gate: fast lane green | ✓ PASS | Re-run live by this verifier at 1531eea: **1881 passed / 1 skipped / 53 deselected, exit 0, 97.27 s** (`pytest tests/ -m "not slow" -q`) — superset of the phase-close 1807/1 baseline, same single benign skip, zero new skips; tests/ byte-identical at HEAD so the count carries |
| Census rollup closure | ✓ PASS | 08-CENSUS-ROLLUP.md carries the final D-03 census (196P/1S/0F, 2:59:34, both endpoints up), the per-item PASS table (33 PASS rows), all four family closures, the First-Dispatch Consumption table, and the cache-quota measurement; the post-close stage-0.5 literal ratchet (193/202 → 197/206) is recorded in phase-09's 09-CENSUS-ROLLUP.md line 375 and live-verified by this round's collection run |
| STATE / ROADMAP / REQUIREMENTS consistency | ✓ PASS | REQUIREMENTS.md marks all 11 Phase-8 requirements `[x]` Complete (re-checked at HEAD this round); ROADMAP shows 9/9 plans executed (ROADMAP.md untouched by the delta); state.json Phase 8 `complete` (re-checked at HEAD) |
| Code-review closure (prior regeneration's trigger) | ✓ PASS | 08-REVIEW-DISPOSITION.md: 12/12 findings `fixed`, 0 open — CR-01 via 5dc18c6 + gate-by-gate chain bc601af/cd90e73/cea260/3352eb8; WR-01..06 and IN-01..05 each with cited commits; every dnallm/ change ships same-commit tests (owner rule honored — verifier re-ran them live, all green) |

### CR-01 Chain Verification (prior regeneration's material delta — carries byte-identical)

The critical finding was that the evo notebook's EVO1 section never loaded evo-1 (call dropped in a7ed221; committed outputs were evo2 output). Verified fixed, gate by gate:

1. **5dc18c6** — load call restored into the cell after `model_name = "togethercomputer/evo-1-8k-base"`; `TestEvoNotebookContentContracts` added (2 tests: load-cell pin; load-precedes-second-DNAInference-build with model=model/tokenizer=tokenizer). Live: green (re-run green at HEAD this round).
2. **bc601af** — giants fetch pattern set gained `*.py` (evo-1 auto_map cross-repo code; `.pt` still excluded by omission — in-code comment at evo.py:30-45 records the full rationale). Live: `test_hub_fetch_is_safetensors_only` green in the 201-test model+evo run.
3. **cd90e73** — `_np_fromstring` encodes bare `str` to utf-8 before `frombuffer` (stripedhyena's CharLevelTokenizer passes bare str). Live: 15 shim tests green.
4. **ceea260** — noFA capability override cell in the notebook (flash-attn wheel imports under torch 2.11.0+cu130 but its C++ op has no registered torch schema; the import-only probe cannot see that; noFP8-precedent pattern). Verified present as visible notebook content with rationale.
5. **3352eb8** — full honest re-execution: committed outputs carry real evo-1 evidence (verified directly by this verifier: pinned snapshot a9be7b6…, distinct generation `TACCCCGCACCGGGGCAGG…` vs evo2's `TCCATTGTACGAATTG…`, distinct scores −1.4132/−1.3698, timestamps 07:44-07:45 matching the commit, 183,347-byte notebook within the 2 MB budget); gated-lane green 105.58 s per commit, reconfirmed live at 1531eea: 1 passed in 106.06 s; mirrors `cmp`-identical.

The chain is the phase goal's own repair loop operating: every surfaced error (unreachable offline load, str-vs-bytes tokenizer input, flash-attn schema mismatch, stale outputs) was fixed with same-change tests and honest re-execution — the success criterion for this verification.

### Observable Truths (per-plan, load-bearing)

All 31 phase-8 commits plus the 21 review/CR-01 commits verified present on `phs`. Plan truths re-confirmed (live-proven items at 1531eea, files byte-identical at HEAD unless noted):

**08-01** — example-nightly job present with cron-gated event filter (`workflow_dispatch || schedule && '30 5 * * *'` — fork/PR can never trigger; re-parsed at HEAD this round: crons [03:00, 05:30]), D-07 staged layout, D-08 fail-soft (stage-results.txt ledger + end-anchored non-zero grep summary; **zero actual `continue-on-error` steps — re-parsed at HEAD**), full extras `.[base,fla,dev,mcp]`, runner-inventory probes; langchain-ollama>=1.1.0 in mcp extra (tomllib-live) + guard test (live); marimo D-18 quadruple (3 apps live).

**08-02** — D-17 ledger on disk with per-item verdicts + pristine-snapshot proof; EXEC-04 script lane healed (artifact freshness assertions intact); EXEC-05 YAML 21/21 (live, both validators); D-03 baseline rollup; ipython>=8.31,<9 pin (tomllib-live); rice/combined seeding repairs (green in fast lane).

**08-03** — conditional allow_patterns passthrough at both layers (model.py), kwargs rebuilt per attempt; evo-1 pattern set wired (evo.py:417, with `*.py`); probe-gated np.fromstring shim (15 tests live, incl. the CR-01 str-encode); spec env override sandwich (TestSpecEnvOverrides live).

**08-04/08-05** — DNATokenizer unknown→id 1 (6 original tests + WR-02/03/05 hardening = 20 megadna tests live); finetune_generation + both siblings repaired with D-21 stamps, pinned installs, byte-synced mirrors (sync gates exit 0 live); megaDNA family CLOSED in rollup.

**08-06/08-07** — giants tier + isolated dnallm-evo-kernel lane (`TestEvoIsolatedLane` live); evo notebook repaired and — post CR-01 — **executing its evo-1 leg for real with honest committed outputs (verifier-rerun green)**; evo2 native `.pt` scoped note; OPTIONAL_IMPORT_MODULES seam (cd6debd) green in fast lane.

**08-08** — lora pair green (TestLoraMirrorEndpoint live; mirror endpoint pinned at the spec-env seam); ollama unit + README in-repo and internally consistent post WR-01/qwen3.5:4b swap; D-13 retry probe + TestProbeHonesty (live); D-07 stage contract in-module and implemented in ci.yml.

**08-09** — models.lock 24/14/prefix-aligned (live count); first-dispatch consumption + quota record in rollup; job completion with cache-before-consumer ordering; nightly runner evidence via gh (37432001711 all-green).

**Score:** 8/8 must-haves verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci.yml` (example-nightly job) | Staged job: prereqs, census hard-gate, stages 0-4, fail-soft | ✓ VERIFIED | Job structure re-parsed at HEAD after the re-pin: zero continue-on-error steps, cron-gated event filter, all stage tokens present, stage-0.5 carries the 197/206 triple (no 193/202 residue); Phase-9 evolutions (D-11 no-cache, D-04 giants deselect, stage-0.5 census gate) recorded with owner-directive comments |
| `models.lock` | 24 rows, 14 pinned, prefix-aligned, giants comment | ✓ VERIFIED (live) | Counted + per-id alignment spot-checked against notebook ACTIVE source= lines (byte-identical at HEAD) |
| `pyproject.toml` | langchain-ollama>=1.1.0 in mcp extra; ipython>=8.31,<9 in notebook extra | ✓ VERIFIED (live) | tomllib-parsed both; guard tests green |
| `dnallm/models/special/megadna.py` | DNATokenizer unknown→id 1; WR-02 checkpoint map; WR-03 no list mutation; WR-05 revision pin + weights_only-first | ✓ VERIFIED (live) | 20 tests green incl. all four review-fix classes |
| `dnallm/models/special/evo.py` | safetensors-only(+`*.py`) patterns on the evo-1 fetch only; IN-01/IN-03 hardening | ✓ VERIFIED (live) | Patterns at :46-52, single allow_patterns call site :417; 201 model+evo fast tests green |
| `dnallm/models/model.py` | conditional allow_patterns passthrough; WR-04 per-half chain merge; WR-06 ValueError | ✓ VERIFIED (live) | Tests green in the 201-test run |
| `dnallm/utils/transformers_compat.py` | probe-gated np.fromstring binary-mode shim (str-encoding) | ✓ VERIFIED (live) | 15 tests green |
| `example/notebooks/generation_evo_models/inference.ipynb` + docs mirrors | evo-1 load restored, noFA cell, D-21 stamp, honest outputs, byte-synced mirrors | ✓ VERIFIED (live) | Outputs read directly (distinct evo-1 evidence at pinned snapshot); lane re-run green; `cmp`-identical mirrors; sync gates exit 0 |
| `scripts/runner/ollama.service` + `README.md` | loopback-only unit + owner steps (one consistent story) | ✓ VERIFIED | Loopback pin; num_ctx deferral narrative reconciled; qwen3.5:4b pull documented; live-drift notice |
| `tests/examples/_execution.py` + `test_notebook_execution.py` | specs for all 21, gates, env sandwich, D-13 probe, D-07 contract, CR-01 content contracts | ✓ VERIFIED (live) | 13 ACTIVE + 8 GATED; 12 harness-contract tests + 5 content-contract tests green (evo contracts re-run green at HEAD this round) |
| `tests/examples/test_marimo_execution.py` | D-18 quadruple | ✓ VERIFIED (live) | 3 apps re-run live: passed in 20.89 s |
| `tests/examples/test_script_execution.py` | healed script lane + rice seeding | ✓ VERIFIED | Artifact freshness assertions intact; green in census + nightly |
| 08-CENSUS-ROLLUP.md / 08-D17-DISPOSITION.md / 08-REVIEW-DISPOSITION.md | census ledger + D-17 ledger + review closure | ✓ VERIFIED | All three read; final census + quota + first-dispatch tables; 12/12 fixed, 0 open |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| example-nightly stage 1 | example census tests | `pytest tests/examples -k "not mcp_example" -m "not giants"` + census hard-gate (197/206 — re-pinned from 193/202 by 86c1fe8; collection live-measured 197/206 in 0.78 s at HEAD) | ✓ WIRED |
| example-nightly stages 2/3 | MCP :8000 probes + mcp pair | dnallm-mcp-server background + readiness curl (status-code shape for sse) + `-k mcp_example` | ✓ WIRED (job success in 37432001711; 37550730293 in flight) |
| lora specs | hf-mirror endpoint | spec `env.HF_ENDPOINT` through the run_notebook sandwich | ✓ WIRED (TestLoraMirrorEndpoint live) |
| evo spec | giants tier | spec `env.HF_HUB_CACHE`/`HF_HUB_OFFLINE` + dnallm-evo-kernel kernelspec | ✓ WIRED (TestEvoIsolatedLane live + 106.06 s lane run) |
| evo-1 hub fetch | safetensors+code-only set | `_EVO1_SAFETENSORS_ONLY_PATTERNS` (incl. `*.py`, excl. `.pt`) via allow_patterns | ✓ WIRED (wiring test live) |
| notebook repairs | docs mirrors | same-commit mirror files; check_notebook_md_sync + check_docs_sync | ✓ WIRED (both exit 0 live; evo mirror `cmp`-identical) |
| ollama unit/README | live runner service | owner enable step; D-13 probe treats down-service as evidence-backed typed skip | ✓ WIRED (endpoint live; probe tests green) |

### Data-Flow Trace (Level 4)

No rendered-value chains end in static returns. The census metrics flow from real model executions; the census gate triple flows from a live collect-only run at HEAD (measured = pinned = 197/206, the guard's own bump-point contract); the lock prefixes flow from the notebooks' actual ACTIVE routes (spot-checked per-id); the marimo default assertions read the real exported HTML text; the evo-1 outputs flow from a real load of the pinned giants snapshot (path + distinct sequences verified in the committed outputs); the fast-lane counts were regenerated live by this verifier at 1531eea on byte-identical tests.

### Behavioral Spot-Checks

Rows marked *(this round)* were run live at HEAD `acb675c` by this regeneration; the remaining heavy-lane rows stand from this verifier's own prior-pass runs at `1531eea` — the covered files are byte-identical between the two HEADs except ci.yml's two census-literal lines (CI config, not imported by any test), so re-execution would reproduce identical results and was deliberately not repeated.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Stage-0.5 census collection (D-03 pin) *(this round)* | `.venv/bin/python -m pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | 197/206 tests collected (9 deselected) in 0.78s — matches re-pinned gate literal | ✓ PASS |
| Census-growth tests (the +4 behind the re-pin) *(this round)* | `pytest ::TestEvoNotebookContentContracts` + `-k "non_mapping_override_section_raises or hung_probe_returns_false_with_timeout"` | 2 passed in 1.54 s + 2 passed in 1.66 s (4/4 green) | ✓ PASS |
| Re-pinned workflow validity *(this round)* | python yaml parse + token asserts | crons [03:00, 05:30]; zero continue-on-error; stage-0.5 present; no 193/202 residue | ✓ PASS |
| Planning-state consistency *(this round)* | REQUIREMENTS.md grep + state.json read | 11/11 Phase-8 IDs `[x]` Complete; Phase 8 `complete` | ✓ PASS |
| Decision coverage *(this round)* | `gsd_run query check.decision-coverage-verify` | honored 21/21, not_honored [] | ✓ PASS |
| Nightly dispatch at re-pin HEAD *(this round)* | `gh run view 37550730293` | queued (in flight) at headSha 86c1fe8 — phase-09's evidence item | ℹ️ IN FLIGHT |
| Evo giants lane (CR-01 subject) *(prior pass, byte-identical)* | `pytest tests/examples/test_notebook_execution.py -k generation_evo -q` | 1 passed in 106.06 s, 67 deselected | ✓ PASS |
| Full fast lane (regression gate) *(prior pass, byte-identical)* | `pytest tests/ -m "not slow" -q` | 1881 passed, 1 skipped, 53 deselected, exit 0, 97.27 s | ✓ PASS |
| Marimo D-18 quadruple (all 3 apps) *(prior pass, byte-identical)* | `pytest tests/examples/test_marimo_execution.py -q` | 3 passed in 20.89 s | ✓ PASS |
| Repair-regression set A (extras/np/megadna) *(prior pass, byte-identical)* | `pytest tests/test_extras_guard.py tests/utils/test_transformers_compat_np.py tests/models/test_special/test_megadna.py -q` | 40 passed in 3.76 s | ✓ PASS |
| Repair-regression set B (model/evo fast) *(prior pass, byte-identical)* | `pytest tests/models/test_model.py tests/models/test_special/test_evo.py -m "not slow and not giants" -q` | 201 passed, 2 deselected | ✓ PASS |
| Harness contracts (spec env / lora mirror / evo lane / probe honesty) *(prior pass, byte-identical)* | `pytest ::TestSpecEnvOverrides ::TestLoraMirrorEndpoint ::TestEvoIsolatedLane ::TestProbeHonesty -q` | 12 passed in 5.41 s | ✓ PASS |
| Notebook content contracts (evo CR-01 + megadna siblings) *(prior pass; evo pair re-run this round)* | `pytest ::TestEvoNotebookContentContracts ::TestMegadnaSiblingContentContracts -q` | 5 passed in 0.85 s | ✓ PASS |
| YAML leg (both validators) *(prior pass, byte-identical)* | `validate_yaml.py` + `pytest tests/configuration/test_yaml_load.py` | 21/21 + 21 passed | ✓ PASS |
| Docs sync gates *(prior pass, byte-identical)* | `check_notebook_md_sync.py` / `check_docs_sync.py` | 24/24 pairs exit 0 / exit 0 | ✓ PASS |
| ollama readiness (SC5 infra, live) *(prior pass)* | `curl 127.0.0.1:11434/api/tags` | qwen3.5:4b (3,324,173,934 B) + qwen3.8:latest, HTTP 200 | ✓ PASS |
| example-nightly structure (YAML parse) *(re-run this round post re-pin)* | python yaml parse + token asserts | crons [03:00, 05:30]; zero continue-on-error; all tokens present | ✓ PASS |
| Nightly runner runs (gh) *(prior pass)* | `gh run view 37432001711` | conclusion success; example-nightly/test-mamba/coverage-nightly jobs success, 0 failed steps | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared; the phase's runnable checks are the pytest lanes and CI dispatches above.

### Requirements Coverage

| Requirement | Source Plans | Description (abridged) | Status | Evidence |
|-------------|--------------|------------------------|--------|----------|
| EXEC-02 | 08-01..09 | all 21 notebooks real execution, fail-soft | ✓ SATISFIED | Wiring 13+8; final census 196P/1S/0F; evo lane verifier-rerun green; nightly 37432001711 success; census gate now pinned to the true 197/206 collection |
| EXEC-03 | 08-01 | 3 marimo apps headless + defaults + exit codes | ✓ SATISFIED | D-18 quadruple; 3 apps re-run live at 1531eea |
| EXEC-04 | 08-02 | generate_bpe_dataset.py artifact in-sandbox | ✓ SATISFIED | Artifact freshness assertions; census row PASS; unchanged and green since phase close |
| EXEC-05 | 08-02 | YAML real load_config on fast leg, zero new skips | ✓ SATISFIED | 21/21 live ×2 at 1531eea; fast lane same single benign skip |
| REPAIR-01 | 08-02..09 | every surfaced error fixed + regression test, triaged | ✓ SATISFIED | 13 repair classes + 12 review findings, all with tests (re-run live); no cwd false-repairs; rollup triage tables |
| REPAIR-03 | 08-02..04 | dnallm library bugs fixed with tests | ✓ SATISFIED | DNATokenizer, np.fromstring, allow_patterns, WR-02..06, IN-01..03 — 256+ tests across the touched modules green live |
| REPAIR-04 | 08-01 | langchain-ollama declared in mcp extra | ✓ SATISFIED | tomllib-live + guard test green |
| CI-04 | 08-09 | models.lock extended, ms-first, revision-pinned, source-aligned | ✓ SATISFIED | 24/14/prefix-aligned (live count; byte-identical at HEAD) |
| CI-05 | 08-03/06/07/09 | giants strategy: safetensors-only, outside quota cache | ✓ SATISFIED | Patterns (+`*.py`, −`.pt`) + dedicated giants dir + D-14 lock comment + live evo-1 offline load proof |
| MCP-01 | 08-08 | ollama loopback systemd + pre-pulled model + probe; pair executes | ✓ SATISFIED | Unit + README consistent (qwen3.5:4b); endpoint live; pair green in nightly + census |
| MCP-02 | 08-01/08/09 | port/VRAM coexistence planned vs 6 :8000 probes | ✓ SATISFIED | D-07 stage contract in-module + implemented stages 1.5/2/2.5/3 with hard-gate hygiene floors |

Orphaned requirements: none — REQUIREMENTS.md maps exactly the 11 IDs to Phase 8 (re-checked at HEAD this round); the 9 plans' `requirements` fields cover all 11 (union), and all 11 are marked `[x]` Complete.

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

Disabled tests on requirements: 0 (grep for skip/skipif/xit across the regression files: none). Circular patterns: none — expected values come from committed contracts, historical numpy semantics, or live HF repo listings fetched during the fix (documented in 08-REVIEW-FIX.md), never from the system under test. The 4 census-growth tests added by the review cycles are value-level contract tests (re-run green this round), not collection padding.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `dnallm/models/model.py` | 865 | `# TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (git-blamed to cd7f6c6, 2026-03-27 — predates this milestone); not phase-introduced |
| `dnallm/utils/transformers_compat.py` | 671 | `# TODO (joao): remove …` | ℹ️ Info | Vendored upstream transformers provenance comment — intentional |
| `tests/examples/test_marimo_execution.py` | 101 | `st_size > 1000` present | ℹ️ Info | Not a violation — one of the four D-18 assertions (prohibition was size-ALONE); defaults/exit-code/marker all asserted and live-proven |

Debt-marker gate: zero `TBD`/`FIXME`/XXX markers across all phase-modified files (the two TODOs above are warning-level infos with provenance). The only covered-file change this round (ci.yml's two census-literal lines) introduces no markers, no stubs, no fail-soft weakening — the re-pin is the gate's own documented bump-point mechanism operating as designed ("census-growth PRs update it on purpose — that is the guard working, not friction"). Re-verification evidence gate (#3304): the only post-prior-pass covered-file change is the in-contract re-pin on a file modified since the prior `verified:` timestamp; no new-scope findings arose.

### Decision Coverage

Decision coverage gate (non-blocking): 21/21 trackable CONTEXT.md decisions honored by shipped artifacts (`check.decision-coverage-verify`, re-run at HEAD this round: honored 21, total 21, not_honored []).

### Advisory (New Scope, Unevidenced)

None — re-verification ran; the only covered-file delta is the evidenced, in-contract census re-pin (measured live: 197/206), and every concern surfaced during this regeneration resolved to live evidence or an already-dispositioned record.

### Human Verification

N/A — infrastructure/testing phase with no user-facing elements; all acceptance criteria verified programmatically (live lanes by this verifier, committed census logs, gh-verified nightly runs, git history, structure checks). The items below are recorded owner hand-offs and post-close policy decisions, not unverified phase truths.

### Open Owner Items (recorded hand-offs / deferred follow-ups — not gaps)

1. **Cache-quota decision** (08-09 D5, still the standing open item): measured 10.37 GB store at threshold, lock-only cache ≈15.2 GiB > quota; options (a) pay-as-you-go (b) drop the models-cache layer (de-facto since Phase-9 D-11) (c) partial ms-only. Numbers in 08-CENSUS-ROLLUP.md.
2. **Box-side $HOME cleanup**: the 28 GB evo-1 full-repo leftover inside `~/.cache/huggingface/hub` plus the flagged 19 GB Qwen / 38 GB legacy-blob trees (owner action; never cleaned by default per the runner-ops rule).
3. **CR-01 deferred follow-ups (recorded in-code)**: (a) `is_flash_attention_capable` is import-only — a live-probe upgrade would have caught the flash-attn wheel/torch schema mismatch the notebook's noFA cell now documents; (b) a manual giants-lane prefetch must include the evo-1-131k-base sibling repo's `*.py` (evo.py comment); (c) rebuilding the flash-attn wheel against torch 2.11 would remove the need for the noFA override — optional.
4. **WR-01 residual**: the qwen3.5:4b Modelfile num_ctx re-probe (`ollama show qwen3.5:4b --modelfile`) remains an owner action before the inert `OLLAMA_CONTEXT_LENGTH` pin can be trusted to govern; the live-runner loopback re-apply step in the README closes the recorded 0.0.0.0 drift.
5. **08-06 cron double-trigger job-gate** (the 05:30 entry re-fires coverage-nightly/test-mamba; queue-serialized): one-line fix available if the owner wants it.

### Gaps Summary

None. The covered-input delta since the prior 8/8 pass is exactly the two-line stage-0.5 census-literal re-pin in `.github/workflows/ci.yml` (commit 86c1fe8, 193/202 → 197/206) — confirmed by `git diff 1531eea..HEAD` on the covered set — and the re-pin is correct: the census collection re-measured live at HEAD returns 197/206 (9 deselected) in 0.78 s, its +4 growth provenance is exactly the three already-verified review-cycle commits (374e8e6/9d44cbf/5dc18c6, all ancestors of the prior verified HEAD), and all 4 growth tests pass live. Every other covered file is byte-identical to the prior verified state, so that pass's live evidence (evo giants lane 106.06 s, fast lane 1881P/1S, marimo ×3, YAML ×2, sync gates, ollama probe, 250+ named regression tests) carries; planning-state consistency (11/11 requirements complete, Phase 8 complete), decision coverage (21/21), and workflow structure (YAML re-parsed post-re-pin) re-checked at HEAD. All 5 roadmap success criteria, all 3 phase-level gates, all 11 requirement IDs, all load-bearing plan truths, all artifacts (existence + substance + wiring + data flow), and all key links verified at HEAD `acb675c`. The nightly dispatch 37550730293 was in flight (queued) at the re-pin HEAD at verification time — its result belongs to phase-09's evidence, not this report.

---

_Verified: 2026-10-07T00:28:53Z (regenerated at HEAD acb675c; covered delta since prior pass: ci.yml census re-pin 86c1fe8)_
_Verifier: Claude (gsd-verifier)_
