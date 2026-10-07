---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 06
subsystem: example full-tree census campaign + durable rollout layer (D-08)
tags: [census, example-execution, typed-skip, probe-then-execute, d-08, gap-closure, repair-queue]
requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: nbclient/marimo/script harness lanes + all-21 specs (05-01/05-05), census inventory skeleton (05-05), GAP-1 shim + ladder-terminal typed-skip pattern (05-04), feasibility verdict matrix + throwaway-venv recipe (05-03)
provides:
  - "Executed D-08 census: all 25 example/ items carry real-execution verdicts in the committed 05-CENSUS.md — 11 PASS / 12 FAIL (exact terminal traceback + class tag + log path each) / 2 deferred-owner (probe green; Phase-8 ollama plan required)"
  - "Resumable fail-soft census driver (.scratch/, never committed): per-item child processes with process-group wall-timeout kill, manifest.json evidence (outcomes, durations, cold/warm, probes), batch CLI light/heavy/gated"
  - "Durable rollout: ACTIVE_NOTEBOOKS grown pilot -> 8 census-green notebooks; TestGatedNotebookExecution (7 probe-then-execute tests, honest typed skips with live probe evidence, loud-fail on the forbidden both-endpoints-up state); MARIMO_EXEC_SPECS grown to all 3 census-green apps"
  - "network_unavailable_skip harness helper (registered prefix, MCP-01 convention)"
  - "Owner hand-off in 05-CENSUS.md: Phase 7-9 overlap flag, class-tagged transformers-5 repair queue (NT-REMOTE-STRUCTURAL / BPE-TOKENIZER / OLLAMA-ENV / OTHER), evo reference question, EXEC-03 complete / EXEC-04 partial, unpushed phs range"
affects: [Phase 8 repair/rescoping, Phase 9 nightly census wiring, example/ tree, tests/examples]
actuals: {tokens: 11435, tasks: 3, commits: 3}
plan_head_before: 2c70aad7147b19ff50e76b0b2e017ea91125e1e39
plan_head_after: bf504e1cc34f1fae732814ee301eafe62dbe5443
tech-stack:
  added: []  # zero installs; campaign ran on the existing venv + the surviving /tmp/feas-venv (T-05-SC)
  patterns:
    - "Per-item subprocess census driver: parent enforces wall timeouts via process-group kill so a wedged kernel/app cannot poison a multi-hour campaign; manifest.json is the resumability + evidence source of truth"
    - "Probe-then-execute durable gate mirroring _network_skip.py: skip fires only on genuine environment facts (endpoint unreachable / find_spec None) with the live probe results in the message; probe-green-but-forbidden fails loudly as an owner decision"
    - "Census verdict cells carry [CLASS] tags with exact terminal one-liner + file:line + log path — the census doubles as the repair worklist"
key-files:
  created:
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-06-SUMMARY.md
  modified:
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py
    - tests/examples/test_marimo_execution.py
    - .planning/WINDOWS.md
key-decisions:
  - "mcp pair = deferred-owner, not network-unavailable: the ollama probe is GREEN on this box (qwen3.8:latest), so the plan's probe-green branch fired — items never executed (T-05-16 install cells), manifest holds pending-owner, census cells use the collision-free deferred-owner wording; the durable tests skip on the genuinely-unreachable MCP server endpoint (honest network-unavailable) and fail loudly if both endpoints ever come up pre-Phase-8"
  - "Census FAIL classification followed the owner's taxonomy precisely: NT-REMOTE-STRUCTURAL (NER notebook + script + benchmark-third-model-behind-blocker), BPE-TOKENIZER (finetune_data raw-AutoTokenizer cell; loader route unaffected), OLLAMA-ENV (mcp pair), OTHER (tRNA MambaCache remote-API removal; embedding_attention pytorch_utils shim-scope gap; benchmark.py:296 labels KeyError; custom_head megaDNA demo cell; finetune_generation missing .fa.gz; lora pair mamba_ssm; evo/megaDNA install gates)"
  - "Driver seeded repo-relative trees for cross-dir references (benchmark ../inference/test.csv; finetune_data ../../../../tests/test_data) — a missing input would have been a harness artifact, not a notebook defect (Rule 3 prevention)"
  - "NT cache left in its orchestrator-patched state (documented in the benchmark row): the patched snapshot surfaces the forward-stage get_extended_attention_mask signature instead of the init-stage one; same FAIL verdict, richer evidence, backups at *.dnallm-bak"
patterns-established:
  - "Census-driven durable wiring: census-green -> ACTIVE parametrization; environment-gated -> probe-then-execute class; FAIL -> census row + repair queue only (never a red test, never silently omitted)"
requirements-completed: [EXEC-01, EXEC-03]
coverage:
  - id: D1
    description: "Resumable fail-soft census driver + probes + light batch: manifest carries outcomes for all 11 light items plus ollama/optional-dep probe evidence; guarded tree delta-zero"
    requirement: EXEC-01
    verification:
      - kind: command
        ref: ".venv/bin/python .scratch/census_driver.py --batch light (idempotent re-run: all SKIP-RESUME, exit 0) + manifest assertions (11 outcomes, ollama probe) -> PASS"
        status: pass
      - kind: command
        ref: "git status --porcelain -- example docs/example pyproject.toml == exactly the 4 owner-baseline lines; git ls-files .scratch/ empty"
        status: pass
    human_judgment: false
  - id: D2
    description: "Heavy batch + gated ladder complete: 28 manifest items (25 census + 3 spike legs), no None outcomes; variant-first everywhere; throwaway-venv legs PASS; isolation re-proven"
    requirement: EXEC-01
    verification:
      - kind: command
        ref: "heavy/gated manifest verify (15 ids + >=25 items) -> PASS; spike legs: evo1-8k (load 21.3s/forward 1.1s/generate OK 55.5s/13.88GB), evo2-noFP8 (generate OK 13.3s/2.32GB), megadna-pinned (forward OK, GENERATE_FAILED tokenizer bug reproduced)"
        status: pass
      - kind: command
        ref: "isolation: stripedhyena/evo2/MEGABYTE_pytorch/pyBigWig absent from .venv; langchain_ollama absent (T-05-16); pyproject porcelain-empty"
        status: pass
    human_judgment: false
  - id: D3
    description: "Committed census complete: 0 pending rows, 21/3/1 counts match the live tree, every row evidence-backed, hand-off section present"
    requirement: EXEC-01
    verification:
      - kind: command
        ref: "grep -c '| pending' == 0; awk Table A ipynb rows == 21; marimo == 3; script == 1; PASS/FAIL/deferred-owner == 11/12/2; census self-verify block ALL GREEN; 'pending-owner' string absent from census"
        status: pass
    human_judgment: false
  - id: D4
    description: "Durable layer green-or-typed-skipped with audit green and fast leg unchanged"
    requirement: EXEC-01
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/examples -q --junitxml=.scratch/census-junit.xml -> 107 passed, 9 skipped, exit 0, 47:26 on GB10"
        status: pass
      - kind: command
        ref: "scripts/audit_skips.py .scratch/census-junit.xml tests/expected_skips.yaml -> exit 0 (9/9 skips matched registered prefixes; no new prefixes)"
        status: pass
      - kind: command
        ref: "fast leg: 94 passed, 1 pre-existing skip, 21 deselected, zero new skips"
        status: pass
    human_judgment: false
  - id: D5
    description: "Owner hand-off decisions (D-09): Phase 7-9 rescoping, NT disposition (one decision covers NER notebook + script + benchmark third model), BPE-tokenizer upstream disposition, evo model-reference update, ollama coexistence plan"
    verification: []
    human_judgment: true
    rationale: "The census's 12 FAIL rows and 2 deferred-owner rows are owner decisions by design (D-08/D-09): environment/upstream dispositions and Phase 7-9 rescoping are flagged in the 05-CENSUS.md hand-off, not resolved inside this plan."
duration: 151min
completed: 2026-10-02
status: complete
---

# Phase 5 Plan 06: Full-Tree Census Campaign + Durable Rollout Summary

**Executed the ENTIRE example/ tree for real (25 items through the harness lanes: 11 PASS / 12 FAIL with class-tagged exact tracebacks / 2 deferred-owner on the probe-green ollama branch), re-proved all three gated families in the throwaway venv, wired the outcomes durably (8 active notebooks incl. two real trainings, 7 probe-then-execute gated tests, 3 marimo apps) with the suite fully green (107 passed / 9 audit-matched typed skips / 47:26) and the environment provably untouched.**

## Performance
- **Duration:** 151 min (campaign machine time dominates: ~2h5m of real executions across light+heavy+gated batches; the durable proof run alone is 47:26) / **Started:** 2026-10-02T06:02:54Z / **Tasks:** 3 / **Files modified:** 5 (+1 ledger)

## Accomplishments
- **D-08 closed:** every one of the 25 census items carries a real-execution verdict or evidence-backed disposition in the committed census — 11 PASS (incl. two full trainings: finetune_binary 484.7s, finetune_multi_labels 1421.0s; the rice pipeline data_generation_and_inference 779.8s cold; all 3 marimo apps), 12 FAIL each with [CLASS] tag + exact terminal one-liner + file:line + log path, 2 deferred-owner.
- **New failure classes found and precisely described** (the owner's transformers-5 adaptation worklist): tRNA remote code imports `MambaCache` (removed 5.x API, second remote-API family); embedding_attention's InstaDeepAI remote code imports the pruning helper from `transformers.pytorch_utils` (the 05-04 shim covers modeling_utils only, and the raw-AutoModel notebook path never imports dnallm); **a dnallm-side bug — `benchmark.py:296` hardcodes `dataset["labels"]` vs the config's `label_column: label`, failing the benchmark notebook before any model loads**; finetune_custom_head's training is fully green and only its megaDNA demo cell is install-gated; finetune_generation expects a gitignored external `.fa.gz` with no download cell; the PlantCAD lora pair needs `mamba_ssm`.
- **Gated ladder walked variant-first everywhere** and all three throwaway-venv fallback legs re-passed (evo1-8k generate OK 55.5s on 13.88GB; evo2-noFP8 generate OK 13.3s on 2.32GB; megadna-pinned forward OK with the known dnallm generate tokenizer bug faithfully reproduced).
- **Durable layer:** ACTIVE_NOTEBOOKS pilot → 8; TestGatedNotebookExecution adds 7 probe-then-execute tests whose skips carry live probe evidence (ollama/server HTTP results, find_spec results) and whose forbidden state (both mcp endpoints up pre-Phase-8) fails loudly; the mcp durable skips are honest — the MCP server endpoint is genuinely down on this box while the ollama-GREEN finding rides in the message; MARIMO_EXEC_SPECS covers all 3 apps.
- **Proof:** full tests/examples 107 passed / 9 skipped / exit 0 in 47:26; audit_skips exit 0 (every skip matched a registered prefix — zero new prefixes); fast leg 94+1 pre-existing, zero new; guarded tree delta-zero; `.scratch/` untracked; isolation re-proven (pyproject clean; stripedhyena/evo2/MEGABYTE_pytorch/pyBigWig/langchain_ollama/megaDNA/mamba_ssm all absent from `.venv`).

## Task Commits
1. **Task 1: census driver (scratch) + probes + light batch → census light verdicts** - `330134a` (feat)
2. **Task 2: heavy batch + gated ladder → census heavy/gated verdicts + ladder evidence section** - `147ea9a` (feat)
3. **Task 3: durable wiring + audit-green proof + owner hand-off** - `bf504e1` (feat)

Plan close-out (docs: SUMMARY + STATE + ROADMAP + WINDOWS) committed immediately after — see `git log bf504e1..` for the `docs(05-06)` commit.

## Files Created/Modified
- `.planning/.../05-CENSUS.md` — all 25 verdicts filled, gated-ladder evidence table, final Table C arithmetic (11/12/2), full owner hand-off section; in-file verification gates updated (0-pending via the concatenation trick)
- `tests/examples/_execution.py` — +`network_unavailable_skip`, +MARIMO_EXEC_SPECS entries for benchmark_demo/finetune_demo (3600/7200)
- `tests/examples/test_notebook_execution.py` — ACTIVE_NOTEBOOKS ×8, +probe helpers, +gates (_gate_ollama_stack/_gate_optional_deps), +GATED_NOTEBOOKS ×7, +TestGatedNotebookExecution
- `tests/examples/test_marimo_execution.py` — parametrization now covers all 3 apps; class mark raised 1500→7200 (spec test-timeouts; pilot's 1200s wall timeout still bounds it internally)
- `.planning/WINDOWS.md` — +3 open entries (deferred-owner rows, benchmark.py:296 defect, gated typed skips)
- `.scratch/census_driver.py` + `.scratch/census-out/` — the campaign machinery and evidence, NEVER committed (owner rule)

## Decisions Made
See key-decisions. Additionally: the driver records a `class_hint` heuristic but every committed classification was made by reading the actual traceback (the heuristic mislabelled tRNA as OTHER-family on first pass — the committed tag reflects the read evidence); the campaign survived two infrastructure kills (a 30-min background task cap mid-multi_labels and a monitor timeout) with zero lost evidence via manifest resumability — the fail-soft design earning its keep.

## Deviations from Plan

**[Rule 3 - Driver bug] sandbox cwd passed as mkdtemp root instead of the seeded copy**
- **Found during:** Task 1 light batch (first 6 items failed identically in ~4s on config open)
- **Fix:** capture seed_sandbox's return value; also fixed an inverted import-probe polarity in the same pass; all false results wiped from the manifest before the real run
- **Files:** .scratch/census_driver.py (never committed) | **Verification:** pilot re-passed 10.7s warm before relaunch | **Commit:** n/a (scratch)

**[Rule 3 - Blocking harness artifact] cross-directory notebook inputs**
- **Found during:** Task 1 prep (read_first survey)
- **Issue:** benchmark_config.yaml points at `../inference/test.csv` and finetune_data at `../../../../tests/test_data/regression/*.csv` — a flat sandbox seed would fail these on missing inputs (harness artifacts, not notebook defects)
- **Fix:** driver seeds mirrored repo-relative trees for exactly these two items
- **Files:** .scratch/census_driver.py | **Verification:** both items failed for REAL reasons instead (BPE tokenizer; NT rung) | **Commit:** n/a (scratch)

**Total deviations:** 2 auto-fixed (2× Rule 3, both scratch-side). Committed artifacts otherwise executed exactly as planned.

## Issues Encountered
- The 30-min default background-task cap killed the first heavy chain mid-multi_labels and one monitor; resumability made it a non-event (SKIP-RESUME on relaunch; multi_labels re-ran from scratch and passed).
- The ollama probe being GREEN (not the plan-expected fail) activated the deferred-owner branch — handled exactly per the plan's probe-green wording; the durable tests needed the both-endpoints-up loud-fail design to stay honest without ever executing install cells.
- predict_data.ipynb is documentation-only (0 code cells): PASS is honest (kernel ran, nothing to execute) — noted in its census row.
- The full durable run costs ~48 min on GB10 (two real trainings inside) — well inside the 900-min nightly; recorded for Phase 9 budget planning. Owner's pytest-notebook/xdist question answered in the hand-off: keep nbclient; xdist does not help a GPU-bound lane.
- The NT ModelScope snapshot remains in the orchestrator-patched state; the benchmark census row states this and which failure rung would surface (decision: keep patched — richer evidence, backups at `*.dnallm-bak`).

## Known Stubs
None. The 9 durable skips are sanctioned typed terminals, not stubs: 7 probe-then-execute gates (self-healing: they execute for real the moment prerequisites/endpoints land), 1 05-05 self-healing script skip, 1 pre-existing allowlisted structural skip. All matched by audit_skips; open ledger entries record them for ship-time visibility.

## User Setup Required
None - no external service configuration required. (Owner hand-off decisions, not setup: the 05-CENSUS.md Hand-off section — Phase 7-9 rescoping, the NT disposition covering three items, BPE-tokenizer upstream disposition, benchmark.py:296 repair, evo model-reference update, ollama coexistence plan.)

## Next Phase Readiness
- The census IS the Phase 8 repair worklist: 12 class-tagged FAIL rows with exact tracebacks, plus 2 deferred-owner rows gated on the Phase-8 ollama plan.
- The durable layer runs green-or-typed-skipped end to end (107/9/0 in 47:26); every typed skip self-heals on its environment fix.
- Environment isolation proven after the whole campaign; `.scratch/` evidence (manifest, 28 item logs, spike logs, junit) stays on the dev box, never committed.
- EXEC-01 complete (census + durable harness); EXEC-03 complete (3/3 marimo apps); EXEC-04 partial (script runs standalone but terminates at the NT structural rung — REQ open for Phase 8).

## Self-Check: PASSED
- 05-CENSUS.md — FOUND (25 verdicts: 11/12/2; hand-off section; ladder evidence; 0 pending)
- tests/examples/test_notebook_execution.py — FOUND (ACTIVE_NOTEBOOKS ×8, TestGatedNotebookExecution ×7 params, probe helpers)
- tests/examples/test_marimo_execution.py + _execution.py — FOUND (3 apps in specs; network_unavailable_skip)
- Commits 330134a / 147ea9a / bf504e1 — FOUND on phs
- Post-close verification: full run 107 passed/9 skipped exit 0; audit exit 0; fast leg 94+1; isolation green; git ls-files .scratch/ empty

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
