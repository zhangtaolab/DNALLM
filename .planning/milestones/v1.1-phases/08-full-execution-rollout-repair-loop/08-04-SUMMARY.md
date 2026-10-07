---
phase: 08-full-execution-rollout-repair-loop
plan: 04
subsystem: testing
tags: [megadna, dnatokenizer, isolated-kernelspec, pinned-clone, nbclient, provenance-stamp, docs-mirror, transformers-5-span]

requires:
  - phase: 08-full-execution-rollout-repair-loop
    provides: D-03 census baseline + family view (08-02); per-notebook kernel env override machinery (08-03)
provides:
  - megaDNA DNATokenizer generate repair (unknown characters encode to id 1) with RED-proven regression tests
  - Self-sufficient finetune_generation notebook (ordering + pinned clone + D-21 stamp) with byte-synced mirror
  - Isolated dnallm-megadna kernelspec lane (ensure_megadna_kernel + venv-targeted prerequisite probe) — the family tracer
  - First real green execution of finetune_generation end-to-end (both trainings) in isolation, zero skips
  - check_notebook_md_sync gate healed to 24/24 (three pre-existing mirror drifts resynced)
affects: [08-05, 08-08, 08-09]

actuals:
  tokens: 9821   # chars/4 over the realized diff (39284 chars); estimate 56000 overshot — execution/download time dominates, not diff size
  tasks: 3
  commits: 5     # measured: git rev-list --count bcad078..HEAD (69cc108, c547969, 9c1d0a4, a4cbcb4, 54892b2)
plan_head_before: bcad0783c2688a65c0e8c301d7385db993ad1972
plan_head_after: 54892b29cefa5fa1a977aaebf0e3da1bb71a429b

tech-stack:
  added: []   # no new libraries; MEGABYTE_pytorch==0.2.1 + pinned megaDNA clone live ONLY in the throwaway venv
  patterns: [venv-targeted prerequisite probe (probe the kernel venv interpreter, not the running one), present-filter column drops for the transformers 4.49-5.x span, fast JSON-level notebook content contracts pinning real-execution invariants]

key-files:
  created:
    - tests/models/test_special/test_megadna.py
  modified:
    - dnallm/models/special/megadna.py
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py
    - example/notebooks/finetune_generation/finetune_generation.ipynb
    - docs/example/notebooks/finetune_generation/finetune_generation.ipynb
    - docs/example/notebooks/finetune_generation.md
    - docs/example/mcp_langchain.md
    - docs/example/mcp_pydantic_ai.md
    - docs/example/notebooks/data_prepare_finetune.md

key-decisions:
  - "DNATokenizer unknown characters encode to id 1 — upstream's own encode_sequence rule (megaDNA mutagenesis notebook); the checkpoint vocabulary stays exactly six tokens wide, so a new unk id would break the embedding lookup, and None (the old unk_token_id default) crashed tensor creation in transformers _call_one/_encode_plus"
  - "Isolated-lane un-gating is conservative: the gate probes the throwaway venv (not the running interpreter) and emits an optional-dep typed skip when cold — provisioning runs only after a green gate, so the nightly runner behavior is unchanged until 08-05/08-09 wire the family rollout deliberately"
  - "megadna_prerequisites_installed() greens only when megaDNA is importable AND MEGABYTE_pytorch is exactly 0.2.1 — a floating MEGABYTE is a FEASIBILITY-lock violation, and the probe refuses to pass on it"
  - "The three pre-existing mirror drifts (mcp_langchain, mcp_pydantic_ai, data_prepare_finetune) were resynced as a separate Rule-3 commit before the Task 2 gate: the plan's sync-script gate is binary and could never pass with them red (each was a mechanical wrapper-behind-notebook reconciliation)"
  - "The MEGA-DNA column drop filters to columns present on the running stack instead of a hardcoded list — transformers 5.x fast tokenizers no longer emit token_type_ids and the hardcode died mid-notebook on first real execution (REPAIR-01)"

patterns-established:
  - "Venv-targeted probe: for isolated-kernel lanes, prerequisites are probed through the kernel venv's interpreter — a running-interpreter probe reports absent forever by design"
  - "Content-contract tests: after a real notebook execution proves repaired invariants, pin them with fast JSON-level notebook parsing so an editorial revert fails in seconds, not at the next 14-minute execution"

requirements-completed: [EXEC-02, REPAIR-01, REPAIR-03]

coverage:
  - id: D1
    description: "DNATokenizer generate repair — non-vocab characters (IUPAC codes, lowercase) encode to valid in-vocab ids instead of crashing tensor creation (EXEC-02 / REPAIR-03 library half)"
    requirement: EXEC-02
    verification:
      - kind: unit
        ref: "tests/models/test_special/test_megadna.py — RED 69cc108 (3 intentional failures, exact census ValueError signature) -> GREEN c547969; full test_special 146 passed; real-path proof: pinned-clone venv loads the checkpoint, encodes 'acgN' and generates end-to-end"
        status: pass
    human_judgment: false
  - id: D2
    description: "finetune_generation content repair — download-before-load ordering, executable pinned clone cell (cb2f5ab4 + MEGABYTE_pytorch==0.2.1), D-21 stamp, mirror byte-synced in the same commit (D-20)"
    requirement: REPAIR-01
    verification:
      - kind: integration
        ref: "check_notebook_md_sync 24/24 + check_docs_sync green at commit a4cbcb4; TestFinetuneGenerationContentContracts (4 fast JSON contracts) green"
        status: pass
    human_judgment: false
  - id: D3
    description: "Isolated dnallm-megadna kernelspec lane + first real green execution of finetune_generation (both trainings) with zero typed skips"
    requirement: EXEC-02
    verification:
      - kind: integration
        ref: "pytest -k finetune_generation: 2 passed in 837s, 0 SKIPPED lines in the -rs log (run 2, after the REPAIR-01 fix); project venv pip list shows no megaDNA/MEGABYTE residue; kernelspec env VIRTUAL_ENV pinned to the throwaway venv"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every surfaced failure repaired same-commit with triage: token_type_ids column drop (content, transformers 5.x span) + three pre-existing mirror drifts (harness-gate blocker)"
    requirement: REPAIR-01
    verification:
      - kind: unit
        ref: "TestFinetuneGenerationContentContracts::test_megadna_column_drop_is_stack_version_robust + full fast lane 1797 passed / 1 pre-existing skip (baseline 1783 + 14 new tests)"
        status: pass
    human_judgment: false

status: complete
duration: 115min
completed: 2026-10-04
---

# Phase 8 Plan 4: megaDNA Family Tracer — DNATokenizer Repair + finetune_generation First Execution Summary

**megaDNA generate encode crash fixed at the library layer (unknown chars -> id 1, RED-proven), finetune_generation made self-sufficient (ordering + pinned clone + D-21 stamp + synced mirror), and executed end-to-end green under the new isolated dnallm-megadna kernelspec — the family's proven tracer for 08-05**

## Performance

- **Duration:** 115 min (16:32–18:27 UTC; ~75 min of it was a throttled ~70MB PyPI download into the throwaway venv — llvmlite/numba are direct dnallm deps and were cache misses)
- **Started:** 2026-10-04T16:32:10Z
- **Completed:** 2026-10-04T18:27:31Z
- **Tasks:** 3/3 (Task 1 TDD: RED then GREEN commits)
- **Files modified:** 10 (1 created test module, 1 library fix, 2 harness, 3 notebook+mirror set, 3 pre-existing mirror resyncs)

## Accomplishments

- **Defect signature captured and fixed.** The census's `GENERATE_FAILED megadna tokenizer single-string encode error` reproduced minimally: `unk_token=None` makes `unk_token_id` None, so ANY character outside the six-token vocabulary (IUPAC ambiguity codes, soft-masked lowercase) encodes to None and dies inside transformers `_call_one`/`_encode_plus` with `ValueError: type of None unknown` — before the model ever runs. The committed ath_cds.csv corpus holds 11 such characters (W/K/S/M/Y), so the notebook's megaDNA half hit exactly this class. Fix: unknowns map to id 1 per upstream's own `encode_sequence` rule; checkpoint vocabulary stays six wide.
- **finetune_generation self-sufficient.** D-21 stamp leads the code cells (transformers/torch/fla=not-used + the two FEASIBILITY pins); Ensembl wget cell documented-and-ordered before the Fasta load; the floating clone cell replaced by the executable pinned form (clone @ cb2f5ab4 + MEGABYTE_pytorch==0.2.1 via `uv pip install` targeting the kernel VIRTUAL_ENV); source= verified already D-15-aligned (zhangtaolab id active on modelscope, HF-only lingxusb id on huggingface); mirror + raw docs copy re-exported in the same commit, both sync gates green.
- **Isolated lane landed.** `ensure_megadna_kernel()` provisions the throwaway `.scratch/megadna-venvs/megadna` venv (dnallm `-e .[cuda130]`, ipykernel, pyfastx), installs the pinned prerequisites from a local pinned checkout, and registers the `dnllm-megadna` kernelspec with `VIRTUAL_ENV` pinned (langchain precedent). The gate `_gate_megadna_isolated` probes THAT venv's interpreter and greens only at the pinned MEGABYTE version.
- **First real execution green.** Attempt 1 died at the MEGA-DNA column-drop cell (transformers 5.x no longer emits `token_type_ids`) — repaired to a present-filter form with same-commit contract tests; attempt 2: **2 passed in 837s, zero skips** — genome download, stamp, DNAGPT training + generation, pinned install, megaDNA 145M load + training + generation all executed for real. Project venv provably untouched.
- **Gate hygiene.** The plan's binary sync gate was un-passable due to three pre-existing mirror drifts from earlier quick tasks; all three resynced mechanically in a separate commit — `check_notebook_md_sync` now 24/24.

## Task Commits

1. **Task 1: DNATokenizer generate library repair** — `69cc108` (test, RED: 3 intentional failures with the exact census ValueError) + `c547969` (fix, GREEN: test_special 146 passed)
2. **Task 2: finetune_generation content repair + mirror** — `9c1d0a4` (Rule-3 pre-commit: three drifted mirrors resynced) + `a4cbcb4` (notebook + wrapper + raw copy, both sync gates green)
3. **Task 3: isolated kernelspec lane + first execution** — `54892b2` (feat: provisioning + gate + spec wiring + REPAIR-01 content fix + 4 contract tests)

**Plan metadata:** (this commit)

## Files Created/Modified

- `dnallm/models/special/megadna.py` — DNATokenizer.UNKNOWN_TOKEN_ID=1; `_convert_token_to_id` maps unknowns to it
- `tests/models/test_special/test_megadna.py` — NEW: 6 tests (encode contract, unknown-id rule, generate-path shapes, decode round-trip) through the real handler class with fakes
- `tests/examples/_execution.py` — MEGADNA_* constants, `megadna_prerequisites_installed()`, `ensure_megadna_kernel()`; finetune_generation spec pinned to the isolated kernel
- `tests/examples/test_notebook_execution.py` — `_gate_megadna_isolated`, GATED registry switch, kernel routing for provisioning, TestMegadnaIsolatedLane (4) + TestFinetuneGenerationContentContracts (4)
- `example/notebooks/finetune_generation/finetune_generation.ipynb` — stamp cell, ordering note, pinned executable install cell, present-filter column drop
- `docs/example/notebooks/finetune_generation.md` + the raw `.ipynb` docs copy — re-exported excerpts + narrative (D-20)
- `docs/example/mcp_langchain.md`, `docs/example/mcp_pydantic_ai.md`, `docs/example/notebooks/data_prepare_finetune.md` — mechanical resync to their notebooks (qwen3.8, FastMCPClient/MCPToolset, ./train.csv paths)

## Decisions Made

- See key-decisions (frontmatter). The load-bearing one is the unknown->id-1 mapping: upstream's mutagenesis notebook pins the rule, and it is the only fix that cannot widen the checkpoint vocabulary.
- fla_version=not-used: the plan's stamp trio names fla, but this notebook has no flash-linear-attention dependency — the stamp records the honest `not-used` and carries the megaDNA pins alongside (D-21 "real per-notebook stamp" reading).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Three pre-existing docs-mirror drifts made the Task 2 gate un-passable**
- **Found during:** Task 2 (verify step)
- **Issue:** `check_notebook_md_sync.py` exited 1 on mcp_langchain.md (qwen3.6 vs notebook qwen3.8), mcp_pydantic_ai.md (stale MCPServerStreamableHTTP form + `result.usage()` call), data_prepare_finetune.md (test-data paths never in the notebook) — leftovers from earlier quick tasks; the plan's gate is binary so it could never pass with them red
- **Fix:** Mechanical wrapper-side reconciliation to the notebooks' current statements, committed separately BEFORE the Task 2 commit (9c1d0a4); gate now 24/24
- **Files modified:** docs/example/mcp_langchain.md, docs/example/mcp_pydantic_ai.md, docs/example/notebooks/data_prepare_finetune.md
- **Verification:** check_notebook_md_sync exit 0 (24/24 pairs); check_docs_sync exit 0
- **Committed in:** 9c1d0a4

**2. [Rule 1 - Bug] MEGA-DNA column-drop cell hardcoded token_type_ids (transformers 5.x span)**
- **Found during:** Task 3 (first real execution — fail-at-first-error)
- **Issue:** `remove_columns(["seq_id","sequence","token_type_ids","attention_mask"])` raised ValueError mid-notebook: transformers 5.x fast tokenizers no longer emit token_type_ids
- **Fix:** Present-filter form (`if column in data.dataset.column_names`) that runs on both sides of the 4.49–5.x span; mirror re-exported; fast contract test pins it
- **Files modified:** example/notebooks/finetune_generation/finetune_generation.ipynb, both docs mirrors of it, tests/examples/test_notebook_execution.py
- **Verification:** re-execution 2 passed / 0 skips in 837s; test_megadna_column_drop_is_stack_version_robust green
- **Committed in:** 54892b2 (triage: content)

**3. [Rule 3 - Blocking] Bandwidth-throttled prerequisite install (~70MB at 15–80KB/s)**
- **Found during:** Task 3 (venv provisioning)
- **Issue:** llvmlite/numba/fonttools/sqlalchemy were uv cache misses; the first two install attempts were killed (a wedged-looking silent phase + a background 30-min cap), and the pinned clone step never ran in the killed attempts
- **Fix:** Restarted with a 2h window; completed the pinned clone + MEGABYTE install manually with the exact pinned form; ensure_megadna_kernel now encodes the same steps idempotently with a 3600s per-step budget and an absolute `-e` path
- **Verification:** venv imports torch 2.11.0+cu130 / transformers 5.18.0 / dnallm / megaDNA; provisioning is a no-op fast path when already green
- **Committed in:** 54892b2 (harness hardening rode the Task 3 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking). **Impact:** All necessary for the plan's own gates; the mirror resync is the only scope extension and it is exactly the documented repair pattern for a binary gate blocked by pre-existing drift.

## Issues Encountered

- The Phase-5 spike venv (/tmp/feas-venv) has a broken transformers install today (PreTrainedConfig unimportable) — the census signature was re-derived from a minimal replica plus a live pinned-clone reproduction instead of reusing it
- The isolated venv resolved transformers 5.18.0 (project venv: 5.17.0) — same constraint span; the execution proves the repaired path on both minors
- uv cache lock contention with long-running owner `uvx` tooling made the first install attempt look wedged; killing and restarting with visible progress resolved it

## TDD Gate Compliance

- Task 1 (tdd="true"): RED `69cc108` — 3 intentional failures reproducing the census ValueError signature exactly (`_call_one -> _encode_plus -> ValueError: type of None unknown`), 3 contract tests passing pre-fix; GREEN `c547969` — 6/6 new + 146/146 test_special
- Task 1's RED commit was amended once for a ruff composite-assertion fix (same discipline as 08-03 Deviation 4)
- Tasks 2-3: not tdd-marked; content/harness work with same-commit verification (sync gates, fast contract tests, real execution)

## User Setup Required

None — no external service configuration required. (The throwaway venv and kernelspec are provisioned idempotently by the harness on boxes where the family lane is enabled.)

## Next Phase Readiness

- The megaDNA family has its proven tracer: 08-05 can move the two read-only siblings (generation_megaDNA, finetune_custom_head) onto the same isolated kernel + venv-targeted gate (`_gate_megadna_isolated` generalizes by spec entry) and run the D-03 reconciliation census including this notebook's row
- The pinned-clone venv is warm on the dev box (`megaDNA` importable, MEGABYTE at 0.2.1); runner-side provisioning stays deliberate (job steps per the example-job plan), never an implicit test-time install
- The D-15 lock entry for lingxusb/megaDNA_updated lands in 08-09 per the plan's own note (HF-only id — prefix direction differs from the zhangtaolab ms-first set)

## Self-Check: PASSED

- tests/models/test_special/test_megadna.py exists; all 10 key-files committed
- Commits 69cc108 / c547969 / 9c1d0a4 / a4cbcb4 / 54892b2 present on phs (5 measured from the bcad078 ledger)
- All three task verify commands re-run exit-0 (test_special 146; both sync scripts; finetune_generation 2 passed / 0 SKIPPED); plan-level fast lane 1797/1/exit 0

---
*Phase: 08-full-execution-rollout-repair-loop · Completed: 2026-10-04*
