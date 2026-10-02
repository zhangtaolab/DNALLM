---
phase: 05-execution-harness-honest-gates-runner-feasibility
plan: 05
subsystem: example execution harness (marimo + script lanes, census inventory)
tags: [marimo-export-html, nbclient, census, example-execution, typed-skip, subprocess-runner, d-08]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: nbclient notebook lane + seed_sandbox/assert_tree_clean harness (05-01), export-html flavor verdict + spike evidence (05-03), GAP-1 shim + ladder-terminal typed-skip pattern (05-04)
provides:
  - "tests/examples/_execution.run_marimo_app — venv-resolved marimo CLI runs 'export html' with cwd=sandbox; returncode + >1000-byte-HTML gate, error artifacts, _ENV_OVERRIDES sandwich"
  - "tests/examples/_execution.MARIMO_EXEC_SPECS — export-html flavor spec (inference_demo: 1200s/1500s)"
  - "tests/examples/_execution.run_example_script — [sys.executable, script] with cwd=sandbox; ALWAYS-written run-log artifact; stderr tail in AssertionError"
  - "tests/examples/_execution.NOTEBOOK_EXEC_SPECS — all 21 example notebooks with per-class budgets (cell strictly below test; max 7200)"
  - "tests/examples/test_marimo_execution.py — durable slow marimo lane, proven green on the spike-proven app (67KB HTML, model cell executed on cuda)"
  - "tests/examples/test_script_execution.py — durable slow script lane; sandbox-only rice downloads; GAP-1-class self-healing typed skip with exact traceback"
  - ".planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md — committed D-08 census inventory skeleton (Table A 25 executables / Table B 34 inputs / Table C arithmetic, all verdicts pending, counts provably exhaustive)"
  - ".gitignore .scratch/ entry — owner rule enforced (one-off census/probe scripts never committable)"
affects: [05-06 census campaign, Phase 8 repair/rescoping, example/notebooks/finetune_NER_task, example/marimo]

actuals:
  tokens: 12900
  tasks: 3
  commits: 4
  plan_head_before: 85827aa4ffa0aa3038aba9c10880682a348c6d2
  plan_head_after: 311d9a6

tech-stack:
  added: []  # zero installs; existing venv tools only (marimo CLI, python, bedtools) — T-05-SC honored
  patterns:
    - "Third harness lane family: subprocess runners (marimo export-html / example script) joining the nbclient lane, all sharing seed_sandbox + assert_tree_clean + _ENV_OVERRIDES sandwich"
    - "Self-healing typed skip (05-04 pattern) applied to a second census item: real attempt first, documented structural marker converts to environment-unavailable with evidence, everything else fails loudly"
    - "Delta-zero tripwire: assert_tree_clean compares against import-time baseline (clean checkout degrades to the original absolute check)"
    - "Self-verifying markdown: census Table C assembles its own section-header strings by concatenation so its verification block cannot re-open the awk ranges it documents"

key-files:
  created:
    - tests/examples/test_marimo_execution.py
    - tests/examples/test_script_execution.py
    - .planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md
  modified:
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py
    - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
    - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
    - .gitignore

key-decisions:
  - "assert_tree_clean converted to delta-zero vs an import-time baseline: the owner's live IDE session keeps execution-output churn in tracked notebooks (3 at plan start, 4 by close — inference.ipynb joined mid-plan), which is not harness business; on a clean checkout (CI) the assert is byte-for-byte the original absolute check"
  - "generate_bpe_dataset.py carried a latent standalone-execution defect — it reads rice_annotation.bed but the bed-write block was dropped from the notebook export (FileNotFoundError proven) — repaired verbatim from data_generation_and_inference.ipynb cell 10, docs mirror resynced byte-identical"
  - "plant-nucleotide-transformer-BPE (the script's model) hits the exact GAP-1 terminal signature on transformers 5.17 (remote modeling_esm.py:335 config.is_decoder, AFTER the 05-04 import shim let the remote code load) — 05-04's ladder decision honored (no PretrainedConfig patch), script lane landed as the self-healing evidence-backed typed skip; owner disposition pending per D-09"
  - "marimo export-html honesty probed: a raising cell makes marimo 0.25.0 exit 1 ('Export was successful, but some cells failed to execute') — the lane's returncode gate genuinely detects census FAILs"
  "Census Table C live gates are logs-tolerant (transient gitignored dnallm.log artifacts appear under any example dir the owner runs locally); the exact-count gates are tracked-baseline (57) plus no-untracked-outside-logs"

patterns-established:
  - "Subprocess execution lanes reuse the sandbox contract: caller seeds, runner executes with cwd=sandbox, artifacts land under tmp_path, scoped tripwire guards the repo"
  - "Census inventory rows carry the specs-dict budgets and known GAP-1-class flags in the evidence column so 05-06 verdicts start from recorded facts"

requirements-completed: [EXEC-01]

coverage:
  - id: D1
    description: "Marimo export-html execution lane: run_marimo_app + MARIMO_EXEC_SPECS + durable slow test, proven green on the spike-proven inference_demo app"
    requirement: EXEC-01
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/examples/test_marimo_execution.py -q -> 1 passed (exit 0), twice (initial + tracer feedback gate)"
        status: pass
      - kind: command
        ref: "artifact evidence: 67,273-byte HTML with executed cell outputs (Available tasks / Current model / ModelScope load log lines); 'traceback' hit in HTML is the marimo show_tracebacks config key, not an error"
        status: pass
      - kind: command
        ref: "cell-error honesty probe (.scratch/05-05/failing_app.py): marimo export exits 1 on a raising cell -> the returncode gate detects census FAILs"
        status: pass
      - kind: command
        ref: "git status --porcelain -- example docs/example == owner baseline (no new dirt from the export)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Script execution lane: run_example_script + durable test with sandbox-only rice downloads; script's latent annotation-bed defect repaired; lane terminates in the sanctioned GAP-1-class typed skip (self-healing)"
    requirement: EXEC-01
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/examples/test_script_execution.py -q -> 1 skipped (environment-unavailable: with exact ValueError traceback in the message), exit 0"
        status: pass
      - kind: command
        ref: "run-log artifact always written (pytest-268 artifacts/generate_bpe_dataset.run.log carries the full failure chain incl. remote modeling_esm.py:335)"
        status: pass
      - kind: command
        ref: "git check-ignore -q .scratch/x.py -> exit 0; find example -name 'osa1_r7*' -> 0 (inputs never touched the repo)"
        status: pass
      - kind: command
        ref: "annotation-bed repair: FileNotFoundError proven pre-fix; block restored verbatim from notebook cell 10; docs mirror diff-clean; check_docs_sync DIFFERs remain exactly the owner-baseline notebooks"
        status: pass
    human_judgment: false
  - id: D3
    description: "Census inventory contract: 05-CENSUS.md committed with Table A (25 executables, all verdicts pending), Table B (34 input/config files with lanes), Table C (completeness arithmetic); counts provably exhaustive"
    requirement: EXEC-01
    verification:
      - kind: command
        ref: "awk-range ipynb rows == 21 == find example -name '*.ipynb' | wc -l; marimo rows == 3 == live; script rows == 1; pending verdicts == 25; Table A rows 25 + Table B rows 34; git ls-files example == 57; non-log live files == tracked (nothing untracked outside transient logs); every Table A path exists on disk"
        status: pass
      - kind: command
        ref: "plan's mandated verify: C=$(find example -name '*.ipynb' | wc -l) == awk count == 21"
        status: pass
    human_judgment: false
  - id: D4
    description: "All-21 NOTEBOOK_EXEC_SPECS + generalized notebook_sandbox fixture + ACTIVE_NOTEBOOKS rollout gate; pilot behavior unchanged"
    requirement: EXEC-01
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/examples/test_notebook_execution.py -q -> 3 passed (pilot + kill + partial-failure), exit 0; kill/partial-failure tests byte-unchanged (diff shows no +/- lines in them)"
        status: pass
      - kind: command
        ref: "spec sanity: 21 entries, every cell_timeout < test_timeout, max test_timeout 7200 == class mark timeout(7200), all keys exist on disk; grep -c 'str(EXAMPLE_DIR' == 22"
        status: pass
      - kind: command
        ref: "fast leg: pytest tests/examples -m 'not slow' -q -> 94 passed, 1 pre-existing allowlisted skip (test_examples.py:254), 5 deselected (all new tests slow-marked), exit 0 — zero new skips"
        status: pass
    human_judgment: false
  - id: D5
    description: "Owner hand-off: the GAP-1-class environment gap now covers three census items (benchmark third model via 05-04; generate_bpe_dataset.py + finetune_NER_task.ipynb via this plan's evidence) — one owner disposition decision"
    verification: []
    human_judgment: true
    rationale: "The three items share one root cause (zhangtaolab NT-family remote modeling_esm.py needs removed transformers-4.x PretrainedConfig defaults); 05-04's ladder explicitly rejected extending the shim with a PretrainedConfig legacy-defaults patch, and D-09 defers roadmap rescoping to the owner at hand-off. Options documented in 05-04-SUMMARY: structural PretrainedConfig patch, transformers pin for this consumer, or checkpoint re-export. The typed skips self-heal the moment any option lands."

duration: 29min
completed: 2026-10-02
status: complete
---

# Phase 5 Plan 05: Marimo/Script Lanes + Full-Tree Census Inventory Summary

**Built the D-08 machinery: the marimo export-html and example-script subprocess lanes joined the private harness (marimo proven green on the spike app, script lane terminating in an evidence-backed self-healing typed skip after the same GAP-1-class remote-code gap surfaced on a second checkpoint), NOTEBOOK_EXEC_SPECS grew to all 21 notebooks on the per-class ladder, the notebook fixture generalized behind an ACTIVE_NOTEBOOKS rollout gate, and the committed 05-CENSUS.md inventory now accounts for every one of the tree's 59 files with provably exhaustive arithmetic.**

## Performance
- **Duration:** 29 min / **Started:** 2026-10-02T05:19:19Z / **Completed:** 2026-10-02T05:48:07Z / **Tasks:** 3 / **Files modified:** 8

## Accomplishments
- The marimo lane is durable-green: `run_marimo_app` resolved the venv marimo CLI, exported `inference_demo.py` headlessly from a sandbox cwd (exit 0, 67KB HTML artifact, model cell genuinely executed — ModelScope load log lines in the artifact), and the scoped tree stayed clean. Cell-error honesty was probed, not assumed: marimo 0.25.0 export exits 1 on a raising cell, so the returncode gate detects census FAILs.
- The script lane exposed two real defects and handled both honestly: the script could never run standalone (it reads `rice_annotation.bed` without writing it — block dropped from the notebook export; repaired verbatim from the notebook, mirror resynced), and its model `plant-nucleotide-transformer-BPE` hits the exact GAP-1 terminal signature (`EsmConfig.is_decoder` at remote `modeling_esm.py:335`, after the 05-04 import shim let the remote code load). The lane landed as the 05-04-pattern self-healing typed skip carrying the traceback; the 389MB checkpoint is now cache-warm so the skip re-probes cheaply and self-heals when the owner lands an environment fix.
- `NOTEBOOK_EXEC_SPECS` covers all 21 notebooks on the per-class ladder (600–900 inference / 1800 evo / 3600 finetune; every cell timeout strictly below its test timeout; max 7200 == the raised class mark). The execution test generalized behind `ACTIVE_NOTEBOOKS` (pilot only — 05-06 grows it), with the kill and partial-failure tests byte-unchanged and still green.
- The committed census inventory accounts for every executable item (25) and every input/config file (34) under `example/` with lanes and budgets; Table C's arithmetic (57 tracked + 2 transient logs = 59 at planning time) is re-provable with the in-file commands, including the nothing-silently-omitted gates.

## Task Commits
1. **Task 1 (tracer): marimo export-html execution lane + spike-proven durable test** - `7769f35` (feat)
2. **Task 2: script execution lane + generate_bpe_dataset typed-skip durable test** - `a85bb42` (feat)
3. **Task 3: census inventory contract + all-21 specs + generalized fixture** - `311d9a6` (feat)

Plan close-out (docs: SUMMARY + STATE + ROADMAP) committed immediately after this list was finalized — see `git log 311d9a6..` for the `docs(05-05)` commit.

## Files Created/Modified
- `tests/examples/_execution.py` - +MARIMO_EXEC_SPECS, +run_marimo_app, +run_example_script, NOTEBOOK_EXEC_SPECS 1→21 entries, assert_tree_clean → delta-zero vs import-time baseline
- `tests/examples/test_marimo_execution.py` - TestMarimoAppExecution (slow, timeout 1500), module-local marimo_sandbox fixture, no conftest.py
- `tests/examples/test_script_execution.py` - TestExampleScriptExecution (slow, timeout 3600), sandbox-only rice downloads, typed network skip on URLError, GAP-1-class self-healing skip
- `tests/examples/test_notebook_execution.py` - ACTIVE_NOTEBOOKS rollout gate, generalized notebook_sandbox fixture, class mark timeout(7200)
- `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` + `docs/example/...` mirror - annotation-bed write block restored verbatim from the notebook
- `.planning/phases/05-execution-harness-honest-gates-runner-feasibility/05-CENSUS.md` - the D-08 acceptance artifact (Tables A/B/C)
- `.gitignore` - `.scratch/` under an owner-rule comment

## Decisions Made
See key-decisions. Additionally: the census's live verification gates were made logs-tolerant after a third transient `logs/dnallm.log` appeared mid-plan under `notebooks/inference/` (the owner's live session also churned `inference.ipynb` outputs — sources byte-identical, not my damage, left untouched per the dispatch note; the delta-zero tripwire absorbs both).

## Deviations from Plan

**[Rule 3 - Blocking] assert_tree_clean fired on the owner's pre-existing dirty baseline**
- **Found during:** Task 1 verify
- **Issue:** the absolute `git status` assert failed on the 3 owner-baseline notebooks (documentated in the dispatch note; 05-04 hit the same class via check_docs_sync), making every new lane unrunnable on this box
- **Fix:** delta-zero comparison against an import-time-captured baseline (the `_kernel_count` principle already in this test tree); on a clean checkout the assert is byte-for-byte the original absolute check; the narrow blind spot (further churn on an already-dirty file) is documented in the module — executions run in sandboxes so repo writes are prevented by construction
- **Files:** tests/examples/_execution.py | **Verification:** all lanes green; scoped status == owner baseline after every run | **Commit:** 7769f35

**[Rule 3 - Missing functionality] generate_bpe_dataset.py reads rice_annotation.bed it never writes**
- **Found during:** Task 2 read_first (pybedtools BedTool → FileNotFoundError proven)
- **Issue:** the annotation-bed build block present in notebook cell 10 was dropped from the script export; the dir's own .gitignore lists the artifact, proving the script silently assumed the notebook had run in the same directory — impossible in a sandbox (and wrong standalone)
- **Fix:** block restored verbatim from the notebook; docs mirror resynced (check_docs_sync DIFFERs remain exactly the owner-baseline notebooks)
- **Files:** example/notebooks/finetune_NER_task/generate_bpe_dataset.py, docs mirror | **Verification:** mirror diff-clean | **Commit:** a85bb42

**[Rule 3 - Environment gap] script's model hits the GAP-1 terminal signature**
- **Found during:** Task 2 verify (the real run)
- **Issue:** `plant-nucleotide-transformer-BPE` fails to construct on transformers 5.17 exactly like the benchmark third model (remote `modeling_esm.py:335` reads `config.is_decoder`; the 05-04 import shim works — failure is strictly the documented structural rung); the plan's green-execution truth is undeliverable without an owner decision 05-04 explicitly reserved
- **Fix:** the 05-04 ladder-terminal pattern — durable test attempts the real execution; only the documented structural marker converts to `environment-unavailable:` with the exact traceback; everything else stays loud; census rows flagged (script + finetune_NER_task.ipynb)
- **Files:** tests/examples/test_script_execution.py, 05-CENSUS.md | **Verification:** 1 skipped (typed, evidence-carrying), exit 0 | **Commit:** a85bb42
**Total deviations:** 3 auto-fixed (3x Rule 3). **Impact:** the marimo/census/spec/fixture deliverables landed exactly as planned; the script lane's execution truth is honestly delivered as lane + self-healing typed skip pending one owner disposition covering three census items.

## Issues Encountered
- The plan's `git status --porcelain -- example docs/example` verify leg cannot read "empty" on this box while the owner's session keeps the notebooks dirty; the honest form (used throughout) is delta-vs-baseline — zero new lines in every check.
- Census self-verification caught two defects in my own first draft before commit: the Table C code block's literal header strings re-opened the awk ranges it documented (inflating counts; even the plan's mandated verify would have read 22), fixed by concatenating the header strings; and the live file count drifted 59→60 when the owner's session added a transient log, fixed by logs-tolerant live gates over the exact tracked baseline.
- 05-FEASIBILITY records the inference_demo defaults as resolving to `plant-dnamamba-BPE-open_chromatin`; the live export log shows `plant-dnabert-BPE-open_chromatin` (both cache-warm; the app's own defaults are what the lane executes — D-05 faithful either way). Noted for 05-06 so the census reads the right model name in export logs.
- Session restart mid-plan (host event): disk state reconciled by the orchestrator; Task 1 commit verified present before continuing; no work lost.

## Known Stubs
- `tests/examples/test_script_execution.py` — the `environment-unavailable:` typed skip is the sanctioned D-08 terminal state for this census item (05-04 pattern), carries the exact traceback, is registered against the allowlisted prefix, and self-heals into the real green pkl-producing run when the owner lands an environment disposition. Recorded in the broken-windows ledger (open) for ship-time visibility.

## User Setup Required
None - no external service configuration required. (Owner hand-off decisions, not setup: one disposition for the GAP-1-class gap now covering three census items — options in 05-04-SUMMARY; Phase 7-9 rescoping per D-09.)

## Next Phase Readiness
- 05-06 inherits: three durable lanes (notebook/marimo/script) with specs for all 21 notebooks + the seeded marimo app, a committed census with every verdict pending and known flags pre-recorded, and the ACTIVE_NOTEBOOKS/MARIMO_EXEC_SPECS growth points.
- GAP-1-class items pre-flagged for the census: benchmark third model, generate_bpe_dataset.py, finetune_NER_task.ipynb — one owner disposition unblocks all three; the typed skip re-probes cheaply (checkpoint now cache-warm).
- Full tests/examples: 98 passed, 2 skipped (1 pre-existing allowlisted, 1 sanctioned typed), exit 0; fast leg 94+1 pre-existing, zero new; ruff check/format clean; tree delta-zero after all runs.

## Self-Check: PASSED
- tests/examples/test_marimo_execution.py — FOUND (TestMarimoAppExecution, slow + timeout(1500), artifact-size + tree-clean asserts)
- tests/examples/test_script_execution.py — FOUND (TestExampleScriptExecution, slow + timeout(3600), fresh-pkl asserts, typed-skip path)
- .planning/.../05-CENSUS.md — FOUND (Tables A/B/C; arithmetic gates re-run green at close)
- Commits 7769f35 / a85bb42 / 311d9a6 — FOUND on phs
- NOTEBOOK_EXEC_SPECS 21 entries / MARIMO_EXEC_SPECS 1 / ACTIVE_NOTEBOOKS pilot-only / no tests/examples/conftest.py — re-verified at close

---
*Phase: 05-execution-harness-honest-gates-runner-feasibility*
*Completed: 2026-10-02*
