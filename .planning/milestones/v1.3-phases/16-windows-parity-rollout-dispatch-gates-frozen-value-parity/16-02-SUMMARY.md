---
phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
plan: 02
subsystem: testing
tags: [windows-parity, dispatch, sys-platform, pybedtools, loj-intersect, notebook, gates, numpy]

requires:
  - phase: 15-windows-parity-core-numpy-interval-evaluators
    provides: dnallm.utils.loj_intersect (locked import-pure numpy evaluator, list[str] tab-joined, nulls verbatim)
  - phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
    provides: 16-01 dispatch shape (sys.platform equality predicate), gate-edit mechanics, tests/test_dispatch_direction.py reader/runtime-contract patterns
provides:
  - NER script platform dispatch (generate_bpe_dataset.py :7 import seam + :249 acquisition via inline LojRow + _loj_rows; consumer byte-identical both paths)
  - NER notebook dispatch (cells 0/1 install+verify conditionals, cell 2 import seam, cell 8 in-cell adapter + acquisition swap; only cells 0/1/2/8 touched)
  - Platform-aware NER gates (_gate_script_pybedtools, _gate_pybedtools — win32 early-return before the probes; registration untouched) completing the three-gate inventory
  - tests/test_dispatch_direction.py NER group (11 tests: runtime parity vs pinned + live pybedtools loj, POSIX ImportError fail-loud, seam/cell/gate structure + AS-1 textual-identity drift guard)
affects: [16-04, WPAR-03, WPAR-04, WPAR-06, WPAR-08]

actuals:
  tokens: 12023     # chars/4 over the realized diff (48091 diff chars across 8 files)
  tasks: 3
  commits: 4        # MEASURED: git rev-list --count 9aa95a7..HEAD

plan_head_before: 9aa95a7218fb28f1669899b2aeda599eb8fdef81
plan_head_after: 628926abc436672126f44cf184c7383577813a23

tech-stack:
  added: []          # no new packages — consumes the phase-15 locked surface only
  patterns:
    - "Duplicated-inline adapter with textual-identity drift guard: LojRow + _loj_rows live verbatim in script AND cell 8 (locked dnallm.utils surface untouched), pinned equal after whitespace normalization by structure test"
    - "Dual-truth parity contract: always-on frozen 12-column literals (bedtools v2.31.1-measured, provenance noted) + live pybedtools loj comparison wherever the truth chain is installed (find_spec + shutil.which gate) — no skip machinery"
    - "POSIX platform pin in gate-contract tests (monkeypatch sys.platform='linux') so the win32 early-return cannot shadow the probe contracts on test-windows"

key-files:
  created: []
  modified:
    - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
    - docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py
    - example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
    - docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb
    - tests/examples/test_script_execution.py
    - tests/examples/test_notebook_execution.py
    - tests/examples/test_examples.py
    - tests/test_dispatch_direction.py

key-decisions:
  - "Live-parity contract executes only where the full truth chain is installed (pybedtools importable AND bedtools binary on PATH: this box with the nightly prefix exported, coverage-nightly, example-nightly); everywhere else the frozen tmp-pair truth — bedtools v2.31.1-measured literals verified against live pybedtools before baking — carries the identical contract, keeping the file's platform-neutral no-skip posture honest on base-only legs and test-windows"
  - "Cell 8's adapter block was composed from the script's ast-extracted source segments at edit time (identity by construction), then pinned by the textual-identity structure test (AS-1)"
  - "Notebook JSON round-trip uses ensure_ascii=False: the committed file stores raw UTF-8 output bytes (braille spinners) with \\u escapes only for control characters; the default ensure_ascii rewrote ~800 unrelated output lines"
  - "The four gate-contract tests pin sys.platform to linux: the win32 early-return would otherwise shadow the probe contracts they pin on test-windows (they execute there)"

requirements-completed: [WPAR-03, WPAR-04, WPAR-06, WPAR-08]

coverage:
  - id: D1
    description: "generate_bpe_dataset.py acquisition dispatch — win32 import seam (dnallm.utils.loj_intersect) / POSIX BedTool verbatim, inline LojRow + _loj_rows at module level, call site swapped, consumer loop byte-identical, minutes-class Windows cost documented"
    requirement: WPAR-03
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_script_win32_rows_match_frozen_loj_truth"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_script_win32_rows_match_live_pybedtools_loj"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_script_posix_seam_import_fails_loud"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_script_import_seam_is_the_conditional_pair"
        status: pass
      - kind: other
        ref: "plan verify probe: SCRIPT SEAM PROBE OK (no-hit sextet ['.', '-1', '-1', '.', '-1', '.'], int start, str fields, 3/3 rows)"
        status: pass
    human_judgment: false
  - id: D2
    description: "data_generation_and_inference.ipynb cells 0/1/2/8 — install/verify magics POSIX-guarded (Windows installs pyfastx only; which-magic folded, Pitfall 5), cell 2 conditional import pair, cell 8 in-cell adapter + dispatched acquisition with consumer/docstring untouched; only cells 0/1/2/8 source changed"
    requirement: WPAR-03
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_nb_cell2_import_seam_matches_script_shape"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_adapter_textually_identical_script_and_cell8"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_ner_install_and_verify_cells_are_posix_guarded"
        status: pass
      - kind: unit
        ref: "tests/examples/test_examples.py fast lane: 106 passed, 1 skipped (OPTIONAL_IMPORT_MODULES tolerance holds — TRUTH-8)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both NER gates platform-aware — win32 early-return as first statement before the probes, POSIX probe + typed skip verbatim, _ACTIVE_NOTEBOOK_GATES registration untouched, no skip machinery reachable on the win32 path"
    requirement: WPAR-04
    verification:
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_script_gate_guard_precedes_probe"
        status: pass
      - kind: unit
        ref: "tests/test_dispatch_direction.py#test_notebook_gate_guard_precedes_probe_and_registration_untouched"
        status: pass
      - kind: unit
        ref: "tests/examples/test_script_execution.py + test_notebook_execution.py contract tests (POSIX-pinned) green in the full fast lane"
        status: pass
    human_judgment: false
  - id: D4
    description: "tests/test_dispatch_direction.py NER extension — 11 tests: runtime parity (frozen + live), POSIX ImportError fail-loud, no try-wrapped import ladder (script + cell 2), textual identity, null-score check intact, install-cell guards, gate structure; file total 19 passed"
    requirement: WPAR-06
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/test_dispatch_direction.py -q -> 19 passed (with the live bedtools prefix on PATH)"
        status: pass
      - kind: other
        ref: "census: live collect-only tail 216/217 (1 deselected) AND ci.yml triple grep equality '216/217 tests collected (1 deselected)' — zero census movement"
        status: pass
    human_judgment: false
  - id: D5
    description: "Docs mirrors byte-identical in the same commits as both example edits (script + notebook)"
    requirement: WPAR-08
    verification:
      - kind: other
        ref: "scripts/check_docs_sync.py -> OK exit 0 (after every task and at close-out)"
        status: pass
    human_judgment: false

duration: 21 min
completed: 2026-10-13
status: complete
---

# Phase 16 Plan 02: NER Rollout Summary

**The NER acquisition surface forks at acquisition only — script + notebook dispatch win32 to the numpy loj through the uniform row adapter (null sextet verbatim, consumer byte-identical), both NER gates gain the win32 early-return, and the dispatch-direction suite grows to 19 tests with live pybedtools parity and ImportError fail-loud contracts — mirrored same-commit, zero census movement.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-10-13T17:16:21Z
- **Completed:** 2026-10-13T17:37:02Z
- **Tasks:** 3 (1 TDD: RED -> GREEN, 2 auto)
- **Files modified:** 8

## Accomplishments
- `generate_bpe_dataset.py` dispatches at acquisition only: the import seam is the platform-conditional pair (win32: `from dnallm.utils import loj_intersect`; POSIX: `from pybedtools import BedTool` verbatim), inline `LojRow` + `_loj_rows` sit at module level above the consumer, the :249 call site binds `intersection = _loj_rows(tokens_bed, annotation_bed)`, and the consumer loop is byte-identical below the swap (git diff: zero edits) — A-file order, gene flush, and the `name == "-1"` count-6 null-score check untouched
- The notebook forks identically: cell 0 wraps both install magics in the inverse guard (POSIX keeps `!brew install bedtools` + `!uv pip install pyfastx pybedtools` verbatim; Windows installs pyfastx alone), cell 1's `!which intersectBed` is folded into the same POSIX guard (the discovered fourth seam, Pitfall 5), cell 2 carries the conditional import pair, and cell 8 defines the adapter + helper (textually identical to the script — composed from the script's AST segments at edit time) with only the acquisition statement swapped; the JSON round-trip touched exactly cells 0/1/2/8 source lists (outputs/ids/counts preserved, raw-UTF-8 bytes preserved via ensure_ascii=False)
- Both NER gates are platform-aware: the win32 early-return is the first statement (plain return, dispatch named in a one-line comment; no skip call reachable), the POSIX find_spec probe + typed skip are verbatim after it, `_ACTIVE_NOTEBOOK_GATES` is untouched — completing the three-gate inventory (showcase gate landed in 16-01)
- `tests/test_dispatch_direction.py` grows to 19 tests: the NER runtime contracts prove the win32 dispatch reproduces the pinned bedtools v2.31.1 12-column truth (>=2 consecutive no-hit null sextets, one multi-hit A row emitted twice, pybedtools row types) AND equals a live pybedtools loj where the truth chain is installed, while a POSIX platform with a poisoned `sys.modules["pybedtools"]` propagates the ImportError (WPAR-06); the structure group pins the seam pair shape, the no-try-ladder ban, the AS-1 textual identity, the install-cell guards, and both gates' guard-before-probe ordering
- Full fast lane green: 2550 passed (= 2539 at 16-03 baseline + 11 new), 1 pre-existing allowlisted skip; docs mirrors byte-identical in the same commits; census triple 216/217 (1 deselected) live and in ci.yml — zero movement

## Task Commits

Each task was committed atomically (Task 1 as a TDD RED -> GREEN pair):

1. **Task 1 RED: NER script runtime contracts (failing)** - `33f585a` (test)
2. **Task 1 GREEN: generate_bpe_dataset.py acquisition dispatch + docs mirror** - `3ffa658` (feat)
3. **Task 2: notebook cells 0/1/2/8 + docs mirror + magic-stripper Rule 3 fix** - `87b0c13` (feat)
4. **Task 3: NER gates platform-aware + dispatch-direction NER structure contracts** - `628926a` (test)

**Plan metadata:** (docs commit follows this SUMMARY)

### TDD cycle (Task 1, tdd="true")

- **RED** (`33f585a`): both target tests failed on the planned assertions — `test_ner_script_win32_rows_match_frozen_loj_truth` failed on "must define ['LojRow', '_loj_rows'] at module level above the consumer" (2 failed, 9 passed). The live-parity variant trivially passed via its availability gate on the binary-less PATH (its extraction helper IS the frozen test's, so its RED is carried by that test) — semantic assessment: the failures are the planned missing-feature assertions, not collection/fixture faults
- **GREEN** (`3ffa658`): the three-hunk script edit made all 11 dispatch-direction tests pass — verified twice, once with the nightly bedtools prefix exported (the live comparison branch executed: `live == numpy` proven element-wise before baking the literals) and once without (base-leg shape)
- **REFACTOR**: none needed — the adapter is already the minimal pinned contract

## Files Created/Modified
- `example/notebooks/finetune_NER_task/generate_bpe_dataset.py` - import seam, inline LojRow + _loj_rows, call-site swap; consumer byte-identical
- `docs/example/notebooks/finetune_NER_task/generate_bpe_dataset.py` - byte-identical mirror (same commit)
- `example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` - cells 0/1/2/8 only
- `docs/example/notebooks/finetune_NER_task/data_generation_and_inference.ipynb` - byte-identical mirror (same commit)
- `tests/examples/test_script_execution.py` - `_gate_script_pybedtools` win32 early-return + docstring third outcome; contract tests POSIX-pinned
- `tests/examples/test_notebook_execution.py` - `_gate_pybedtools` win32 early-return + docstring third outcome; contract tests POSIX-pinned; registration untouched
- `tests/examples/test_examples.py` - `_strip_magic_lines` placeholder substitution (Rule 3)
- `tests/test_dispatch_direction.py` - NER group: 3 runtime + 8 structure tests

## Decisions Made
- Dual-truth parity design: the live pybedtools comparison runs only where the truth chain is installed (pybedtools importable AND bedtools on PATH); the frozen tmp-pair literals — measured against live pybedtools 0.12.1 + bedtools v2.31.1 before baking, provenance noted in-test — carry the identical contract on every leg. This keeps the file's platform-neutral, no-skip-machinery posture intact on `.[base]`-only legs and test-windows (where pybedtools cannot even install) while satisfying the plan's live-parity precondition on this box
- Cell 8's adapter block composed from the script's ast-extracted segments (identity by construction), then pinned by the whitespace-normalized textual-identity structure test
- Notebook round-trip with `ensure_ascii=False` — the committed file's raw-UTF-8 output bytes (braille spinner chars in cell 0's brew outputs) would otherwise churn ~800 unrelated lines
- The four gate-contract tests pin `sys.platform` to linux so the probe contracts they pin stay truthful on test-windows after the win32 early-return (see Deviation 2)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `_strip_magic_lines` placeholder substitution**
- **Found during:** Task 2 (notebook cells 0/1)
- **Issue:** the plan-mandated conditional-install pattern puts `!` magics inside if/else branches; `test_notebook_cell_syntax` strips magic lines by DELETION, leaving bodyless branches -> false SyntaxError, failing the fast lane ("expected an indented block after 'if' statement")
- **Fix:** `_strip_magic_lines` now substitutes each magic line with a same-indentation `pass` placeholder instead of deleting it (empty-line behavior unchanged); repo scan confirmed no other notebook has indented magics, so no other cell's lane outcome changes
- **Files modified:** tests/examples/test_examples.py
- **Verification:** test_examples fast lane 106 passed / 1 pre-existing skip; census stays 216/217 (body-only edit, zero test-count change)
- **Committed in:** 87b0c13 (part of task commit)

**2. [Rule 3 - Blocking] POSIX platform pin in the four gate-contract tests**
- **Found during:** Task 3 (gate edits)
- **Issue:** with the win32 early-return as the gate's first statement, `TestScriptGateContracts` (x2) and `TestActiveNotebookGates` (x2) — which run on test-windows — would take the early return instead of the probe path and fail their `pytest.raises(Skipped)` / return contracts there
- **Fix:** each test pins `monkeypatch.setattr(sys, "platform", "linux")` before exercising the gate, making the POSIX-branch contract they pin explicit and platform-independent
- **Files modified:** tests/examples/test_script_execution.py, tests/examples/test_notebook_execution.py
- **Verification:** all four green in the full fast lane (2550 passed); census unchanged
- **Committed in:** 628926a (part of task commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes are required consequences of the planned shapes (guarded magics; win32 gate early-return) meeting existing fast-lane gates — no scope creep, no census movement, no behavior change on POSIX lanes.

## Issues Encountered
None beyond the two documented deviations. The first notebook write attempt used default `ensure_ascii=True` and was caught by the diff-size check (~1548 lines) before commit; reset and re-run with `ensure_ascii=False` (raw-UTF-8 preserved, 78-line diff).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All five phase seams dispatched (CRE cell 11 + showcase gate in 16-01; script :7/:249 + cells 0/1/2/8 + both NER gates here) and the frozen pairs + tolerance band landed (16-03); WPAR-06/08 close-out (phase-wide guard sweep + nightly workflow_dispatch evidence) is 16-04's scope
- REQUIREMENTS.md marking is shared-ID-gated: WPAR-03 and WPAR-04 marked complete here; WPAR-06 and WPAR-08 stay pending until 16-04 (their last declaring plan) finishes
- Linux nightly re-execution holds by construction (POSIX branches byte-equivalent; both nightlies provision pybedtools + bedtools), with live evidence to be captured by 16-04's manual dispatch
- The dispatch-direction suite now covers every phase seam with fail-loud + parity contracts on every CI leg including test-windows

## Threat Flags

None — no security-relevant surface beyond the plan's threat model. T-16-04 mitigated (platform-predicate-only dispatch, AST-banned ladder, ImportError contract, live-parity null-sextet pin), T-16-06 mitigated (install/verify guards pinned by structure test), T-16-05 accepted with the documented cost comment (AS-3), T-16-SC n/a (zero installs).

---
*Phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity*
*Completed: 2026-10-13*

## Self-Check: PASSED

- All 8 modified files exist on disk (FOUND each)
- All 4 commits (33f585a, 3ffa658, 87b0c13, 628926a) are ancestors of HEAD on this branch; zero file deletions across them; no untracked files left behind
- Re-ran acceptance criteria live at close-out: dispatch-direction suite 19 passed (with the bedtools prefix exported — live-parity branch executed), frozen-pairs + genomic-intervals alongside (62 passed combined), check_docs_sync exit 0, live collect-only 216/217 (1 deselected), ci.yml census triple equality, full fast lane 2550 passed / 1 pre-existing allowlisted skip
- Commit ledger measured: `git rev-list --count 9aa95a7..HEAD` = 4 (matches per-task commits; SUMMARY written after all four)
