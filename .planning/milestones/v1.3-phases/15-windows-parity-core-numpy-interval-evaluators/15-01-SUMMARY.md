---
phase: 15-windows-parity-core-numpy-interval-evaluators
plan: 01
subsystem: library
tags: [bedtools-parity, numpy, interval-arithmetic, jaccard, loj-intersect, bed-parsing, windows-parity, oracle-tests, seeded-corpus, import-purity, coverage-gate]

requires: []   # depends_on: [] — no prior v1.3 phase; the true provenance is the two parity quick-tasks (261010-wx2, 261011-0a3), whose spike scripts + report.txt oracles under .planning/quick/ are the ported source
provides:
  - dnallm/utils/genomic_intervals.py — pure-numpy bedtools-parity evaluators (parse_bed / jaccard / loj_intersect + BedRow + NULL_* constants + private _merge_intervals/_sweep_stats/_null_b_fields/_parses_int), import-pure numpy+stdlib only
  - The dnallm.utils package surface re-export of all three entry points (import block + sorted __all__); dnallm/__init__.py facade byte-stable
  - tests/test_genomic_intervals.py at tests/ root (owner lock) — 8+11 handcrafted baked oracles, 9-variant null-fill pin table, parse guard battery, seeded corpora (20261010/20261011) with 300+200 property cases + 2x12 baked subset literals, import-purity test, frozen-pair documented constants
  - The Phase-16 adapter contract on record: loj_intersect returns list[str] tab-joined lines (split on tab by the WPAR-03 row adapter)
affects: [16-windows-rollout-dispatch-gates (WPAR-03..08 consume the module + surface), WPAR-05 frozen-pair in-band gate, dnallm.utils package surface]

actuals:
  tokens: 20729       # chars/4 over the realized 3-file diff (82919 chars); plan estimate was 55000 (estimate covered conservative transcription overhead; verbatim porting from read-verified spike excerpts came in leaner)
  tasks: 3
  commits: 3          # MEASURED: git rev-list --count 088db69..e07deac

plan_head_before: 088db69486d1831996343acfd10f03c9e315d3d5
plan_head_after: e07deac29c2f5dc3f534dc83fe7450a8af752f4d

tech-stack:
  added: []            # zero new dependencies — numpy >=2.0.0 floor is a pinned core dep; module imports numpy + stdlib (pathlib/typing) only
  patterns:
    - "Spike-as-specification porting: algorithms copied verbatim from git-tracked parity spikes with file:line cites, never imported from .planning/ — the 5008-case bedtools v2.31.1 parity record IS the design authority"
    - "Two-layer seeded corpus: property invariants primary (independently re-derived in plain python, no numpy in the oracle) + one fully-baked 12-cycle subset per spike as a Generator-stream-divergence tripwire (AS-2)"
    - "Coverage proof via the coverage-CLI form only (WINDOWS #20): uv run --no-sync coverage run -m pytest ... && coverage report --include=<module> — the pytest --cov=<dotted> form crashes in the dev venv"
    - "Exact-key-set assertion + quoted-key grep count 0 as the behavioral lock for an EXCLUDED contract (WPAR-01's informational fourth key never spelled in the module)"

key-files:
  created:
    - dnallm/utils/genomic_intervals.py
    - tests/test_genomic_intervals.py
  modified:
    - dnallm/utils/__init__.py

key-decisions:
  - "Case 11 plan-prose corrected against the oracle: the plan's Task 2 wording placed the 6-column null vector inside the NER case-11 driver, but the parity-proven spike output (and independent manual derivation) shows case 11 is all-hits (8 lines, no null fill); the null vector is pinned line-exactly where it actually fires — case 7's two boundary-touching rows plus the 6-int/6-non-int pin-table rows — so every Task 2 acceptance criterion still locks verbatim"
  - "Baked subset = one full 12-cycle of the generator (indices 0-11) per spike: this covers every pattern x shuffle state the 6-pattern/4-shuffle cycle can actually produce (CRT: shuffled indices only ever land on patterns 1/3/5), which is the complete deterministic surface"
  - "Corpus property test derives the null fill from the actual B side (empty corpus B -> 3-column form, 6-column rows -> the NER vector) rather than assuming 6 columns — the corpus generator legally produces empty B sides (only both-empty is excluded)"
  - "The two spikes' duplicates generators factor through one _stagger helper (identical rng draw sequence per side; the loj variant just runs it on both sides) — draw-order fidelity preserved with less duplication"
  - "Task 3 (tdd=true) committed as a single test(15-01) commit: the task's files are tests-only and lock behavior built in Tasks 1-2 (tracer-first plan ordering), so the RED-GREEN cycle is structurally inapplicable — no new production behavior exists to drive RED; the coverage gate itself provided the red-green loop (extend battery until the gate passes, which it did at 100%)"

patterns-established:
  - "Oracle-pinned null-B fill table baked as a 9-variant public-path pin test (loop-driven, not pytest.mark.parametrize — the repo forbids marker machinery in this file)"
  - "Import-purity test targeting the SUBMODULE directly with pandas/pybedtools/torch sentinel-blocked (package-level purity impossible: dnallm.utils.__init__ imports torch via support.py)"
  - "Frozen-pair unrecoverable corpora recorded as documented constants + a 6-dp rounding anchor test, with the real-pair gate explicitly deferred to Phase 16 / WPAR-05"

requirements-completed: [WPAR-01, WPAR-02]

coverage:
  - id: D1
    description: "dnallm/utils/genomic_intervals.py with parse_bed/jaccard/loj_intersect carrying every oracle-pinned contract (null-B fill incl. -1 score slot, A-file-order unsorted emission, join-not-merge multiplicity, bookended-adjacency merge, excluded fourth key), re-exported on dnallm.utils"
    requirement: WPAR-01
    verification:
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_jaccard_identical_sets_is_one (+ 8 further jaccard oracle cases, exact literals)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_loj_null_fill_pin_table_all_nine_variants"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_loj_case11_ner_realistic_six_plus_six (+ 10 further loj line-exact cases)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_jaccard_returns_exactly_three_keys"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_package_surface_reexports_all_three_entry_points"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_module_import_is_pandas_pybedtools_torch_free (import purity)"
        status: pass
      - kind: other
        ref: "command: grep -cE quoted n-intersections key on module == 0; git diff dnallm/__init__.py empty"
        status: pass
    human_judgment: false
  - id: D2
    description: "Platform-neutral root-level test file (tests/test_genomic_intervals.py) with the full baked oracle battery + seeded property corpora, zero skip machinery, zero markers, no bedtools binary anywhere"
    requirement: WPAR-02
    verification:
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_jaccard_seeded_corpus_property_invariants (300 cases)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_loj_seeded_corpus_property_invariants (200 cases, independent re-derivation)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_jaccard_seeded_subset_baked_literals + #test_loj_seeded_subset_baked_literals (2x12 baked)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_parse_bed_rejects_fewer_than_three_columns (+ 3 further guard branches, CRLF, comment-skip)"
        status: pass
      - kind: unit
        ref: "tests/test_genomic_intervals.py#test_seeded_corpus_generators_are_deterministic"
        status: pass
      - kind: other
        ref: "command: .venv/bin/python -m pytest -m 'not slow' --no-cov -q -> 2526 passed, 2 skipped (pre-existing), exit 0"
        status: pass
      - kind: other
        ref: "command: grep -cE 'pytest.mark|importorskip|skipif|xfail' on test file == 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "Per-module coverage proof: module row visible in coverage report at >=96% (measured 100%), provably outside the pyproject 7-entry omit list, in the same change that adds the module"
    requirement: WPAR-01
    verification:
      - kind: other
        ref: "command: uv run --no-sync coverage run -m pytest tests/test_genomic_intervals.py -q && uv run --no-sync coverage report --include=dnallm/utils/genomic_intervals.py -> 110 stmts, 0 miss, 100%"
        status: pass
    human_judgment: false

duration: 26 min
completed: 2026-10-13
status: complete
---

# Phase 15 Plan 01: Windows Parity Core — Numpy Interval Evaluators Summary

**Pure-numpy bedtools-parity evaluators (parse_bed/jaccard/loj_intersect) ported verbatim from the 5008-case parity spikes, re-exported on dnallm.utils, and locked by a root-level oracle battery + seeded corpora at 100% module coverage**

## Performance

- **Duration:** 26 min
- **Started:** 2026-10-13T14:26:14Z
- **Completed:** 2026-10-13T14:52:36Z
- **Tasks:** 3 (tracer + expand + lock)
- **Files modified:** 3 (2 created, 1 edited; 1641 insertions, 0 deletions)

## Accomplishments
- Ported the two parity-proven spike cores into `dnallm/utils/genomic_intervals.py` (numpy + stdlib only): `parse_bed` (loud ValueError naming file+line, CRLF-safe, file-order preserving), `jaccard` (merge + boundary sweep with the load-bearing starts-before-ends tie-break; exactly 3 keys — WPAR-01's informational fourth key excluded and never spelled), `loj_intersect` (per-A-row broadcast join, strict half-open inequalities, join-not-merge multiplicity, oracle-pinned null-B fill)
- Re-exported all three names on the `dnallm.utils` surface (import block + sorted `__all__`); `dnallm/__init__.py` byte-stable throughout
- Locked every contract with baked oracles: 9 jaccard literals (8 spike cases + both-sides-empty guard), 11 loj line-exact cases, the 9-variant null-fill pin table through the public path, the parse guard battery, and the frozen-pair aggregates as documented constants with the 0.324727 6-dp anchor test
- Ported both seeded generators (20261010/20261011) driving two-layer corpora: 300 jaccard + 200 loj property-invariant cases (oracle re-derived in plain python) + 2x12 fully-baked subset literals as the stream-divergence tripwire; import-purity test sentinel-blocks pandas/pybedtools/torch
- Coverage gate: module row **visible at 100%** (110 stmts, 0 miss) via the coverage-CLI form; full fast lane green (2526 passed, 2 pre-existing skips, 75 deselected); census surface untouched (zero ci.yml / pyproject.toml / expected_skips.yaml edits — `git status` clean outside the three intended files)

## Task Commits

Each task was committed atomically:

1. **Task 1: Tracer — parse_bed + jaccard end-to-end through the dnallm.utils surface** - `c7c387b` (feat)
2. **Task 2: Expand — loj_intersect with the oracle-pinned null-B fill table** - `94dd395` (feat)
3. **Task 3: Lock — full oracle battery, seeded corpora, purity test, 100% coverage proof** - `e07deac` (test)

Tracer feedback gate: re-ran Task 1's `<automated>` verify end-to-end after its commit (interactive + end-of-phase + automated-only → row 3 auto-continue) — green, expanded to Task 2.

## Files Created/Modified
- `dnallm/utils/genomic_intervals.py` - NEW pure-numpy module: BedRow NamedTuple, parse_bed, NULL_* constants, _parses_int, _null_b_fields, _merge_intervals, _sweep_stats, jaccard, loj_intersect
- `dnallm/utils/__init__.py` - +3-name genomic_intervals import block + 3 sorted __all__ entries
- `tests/test_genomic_intervals.py` - NEW root-level (owner lock): 39 tests, zero markers, zero skips, zero subprocess

## Decisions Made
- Case-11 plan-prose corrected against the parity oracle (see Deviations) — the null vector pins in case 7 + pin table where it actually fires
- Baked subset taken as one full 12-cycle per spike (the complete pattern x shuffle surface the cycle can produce)
- Corpus null fill derived from the actual B side per case (empty B is legal in the corpus)
- Task 3's tdd=true realized as a single test commit (tests-only task locking Tasks-1/2 behavior; no new production behavior to drive RED — the plan's own tracer-first ordering governs; the coverage gate supplied the red-green loop)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan prose placed the 6-column null vector inside the case-11 NER driver; the oracle shows case 11 is all-hits**
- **Found during:** Task 2 (loj handcrafted battery authoring)
- **Issue:** The plan's Task 2 action reads "the NER-shaped 6+6-column driver (case 11) asserting the full 8 output lines with the B-side null vector '.', '-1', '-1', '.', '-1', '.' exact". Manual half-open derivation over the case-11 inputs (spike_loj_parity.py:1010-1032) shows all 8 A rows hit exactly one B row each — and the parity-proven spike implementation (which matched live bedtools byte-for-byte on this case, canonical+sequence PASS in report.txt:47) confirms: 8 hit lines, no null fill. Baking the plan's literal wording would have pinned output bedtools never produced.
- **Fix:** Baked the true 8 oracle lines for case 11; pinned the 6-column NER null vector line-exactly where it actually fires — case 7's two boundary-touching rows and the 6-int/6-non-int pin-table variants. Every Task 2 acceptance criterion ("all 9 pin-table variants asserted verbatim; the 6-column NER null vector pinned exactly; boundary case 2 lines both null-filled") holds verbatim.
- **Files modified:** tests/test_genomic_intervals.py
- **Verification:** test_loj_case11_ner_realistic_six_plus_six, test_loj_boundary_touching_rows_null_fill, test_loj_null_fill_pin_table_all_nine_variants all green; case-11 expected lines byte-identical to the spike oracle dump
- **Committed in:** 94dd395 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — plan-prose transcription error vs the oracle source)
**Impact on plan:** None on contracts — every pinned contract locks at the cases where it fires. Transcription-fidelity risk (the phase's only real risk) was caught exactly the way the plan's oracle-first design intended.

## Issues Encountered
None beyond the deviation above. Two authoring-time corrections absorbed before commit: the loj property test initially assumed 6-column B for the null-fill check (the corpus legally produces empty B sides — fixed to derive the fill per case), and ruff format/composite-assert adjustments (mechanical).

## TDD Gate Compliance

Task 3 carries `tdd="true"`; `task.is-behavior-adding` reports true, but the plan-level gate is inactive (`workflow.tdd_mode: false`; the orchestrator passed no `TDD_MODE=true`). The task's `<files>` contain only the test file — it locks behavior built in Tasks 1-2 per the plan's deliberate tracer-first ordering (tracer → expand → lock), so no RED phase is realizable: there is no unimplemented behavior to drive a failing test against, and a forced RED would mean deliberately baking wrong literals. Commit is `test(15-01)` (single test-scoped commit). The coverage gate provided the genuine red-green loop: battery extended until the module row passed at 100%.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 16 (Windows dispatch, WPAR-03..08) can `from dnallm.utils import loj_intersect, parse_bed` (and jaccard) — the list[str] tab-joined output contract for the WPAR-03 row adapter is documented in the loj_intersect docstring
- The frozen-pair real-pair in-band gate remains Phase 16 / WPAR-05 (aggregates recorded here as documented constants: 15872 / 48878 / 0.32472687098490116)
- CI legs (test, test-windows, coverage-gate, coverage-nightly) collect tests/ automatically; green there is verified post-push (out of plan scope)

## Self-Check: PASSED

- All key-files exist on disk (module, __init__, test file)
- All 3 task commits are ancestors of HEAD (c7c387b, 94dd395, e07deac)
- Plan-level verification rerun post-commit: fast lane green, coverage 100% visible, scope clean, facade byte-stable, negative greps 0, key-set assertion present

---
*Phase: 15-windows-parity-core-numpy-interval-evaluators*
*Completed: 2026-10-13*
