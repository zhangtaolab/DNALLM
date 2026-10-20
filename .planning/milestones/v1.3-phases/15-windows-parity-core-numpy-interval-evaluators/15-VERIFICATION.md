---
phase: 15-windows-parity-core-numpy-interval-evaluators
verified: 2026-10-13T15:08:10Z
status: passed
score: 10/10 must-haves verified
covered_files:
  - .planning/phases/15-windows-parity-core-numpy-interval-evaluators/15-01-PLAN.md
  - .planning/phases/15-windows-parity-core-numpy-interval-evaluators/15-01-SUMMARY.md
  - dnallm/utils/__init__.py
  - dnallm/utils/genomic_intervals.py
  - tests/test_genomic_intervals.py
covered_digest: "v3:sha256:ef1c6152184bda550f8678355f914f0846b55a39a7c8ac3957e98cbeedc2ac3a"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 15: Windows Parity Core — Numpy Interval Evaluators Verification Report

**Phase Goal:** The spike-validated numpy interval evaluators exist as covered, platform-neutral library code whose oracle-pinned contracts are locked by unit tests on every CI leg — the bedrock Phase 16's Windows dispatch imports
**Verified:** 2026-10-13T15:08:10Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All 10 PLAN must-have truths map onto the 3 ROADMAP success criteria; every one was re-proved independently of SUMMARY.md claims (oracles cross-checked against the spike `report.txt` artifacts, tests and coverage run by this verifier, git evidence inspected directly).

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | TRUTH-1 (SC1, import-pure): re-export of `jaccard`, `loj_intersect`, `parse_bed` on `dnallm.utils`; numpy+stdlib-only submodule import; `dnallm/__init__.py` byte-identical | ✓ VERIFIED | `dnallm/utils/__init__.py:17-21` import block + `__all__` entries at :57/:66/:68 (alphabetically correct); import probe `from dnallm.utils import jaccard, loj_intersect, parse_bed` exits 0; module imports only `pathlib`, `typing`, `numpy` (genomic_intervals.py:30-33); `test_module_import_is_pandas_pybedtools_torch_free` sentinel-blocks pandas/pybedtools/torch and re-imports the submodule — passed; `git diff 088db69..e0fdbed -- dnallm/__init__.py` = 0 lines |
| 2 | TRUTH-2 (SC1, pinned jaccard contracts): exact 3-key dict; 8 spike oracle values; bookended-adjacency merge; one-sided chroms count toward union; union==0 → 0.0 | ✓ VERIFIED | All 8 baked literals match spike `261010-wx2/report.txt:21-28` value-for-value (0/1500/0.0; 450/450/1.0; 700/1900/0.3684210526315789; 0/200/0.0 ×3; 200/300/0.6666666666666666; 220/900/0.24444444444444444) + both-sides-empty guard; `test_jaccard_returns_exactly_three_keys` asserts the exact key set; merge coalesces at `start <= merged[-1][1]` (:123) and sweep uses the `np.lexsort((-deltas, coords))` starts-before-ends tie-break (:162); 39/39 tests green in this verifier's run |
| 3 | TRUTH-3 (SC1, precision): int64 arrays, Python-int boundary, exact single division | ✓ VERIFIED | `dtype=np.int64` on every coordinate array (:120, :127, :155-156, :307, :311); `int(...)` at the return boundary (:167-168); single guarded division (:206); baked float literals compare with `==` and pass — behaviorally proven by the passing oracle tests |
| 4 | TRUTH-4 (SC1, pinned loj contracts): A-file-order emission on unsorted input, B-file-order hits, duplicates multiply, touching boundaries do not overlap | ✓ VERIFIED | Case 10 unsorted → 9 lines in A-file order (test lines 257-292); case 9 duplicates both sides → 8 lines (226-254); case 7 boundary-touching → both rows null-fill (205-223); mask uses strict `starts < a.end & ends > a.start` (:319); join path never sorts/dedupes — all green |
| 5 | TRUTH-5 (SC1, pinned null-B fill): 6-col NER vector `'.','-1','-1','.','-1','.'`; count-5 content dependence; count-6 always `-1`; count-7 `.`; empty-B 3-col form | ✓ VERIFIED | `PIN_TABLE` (test :125-138) matches spike `261011-0a3/report.txt:8-17` all 9 variants exactly (cross-checked row by row); `_null_b_fields` (:263-271) ports the spike encoder verbatim; 6-col vector pinned in case 7 lines + both 6-col pin rows; empty-B → `['.','-1','-1']` — all green |
| 6 | TRUTH-6 (SC1, parse contracts): loud ValueError naming file+line on 4 guard branches; blank/'#' skipped; file order, never sorted | ✓ VERIFIED | Guard battery test :500-538 with `pytest.raises(match=...)` anchoring `bad.bed: line N` for each branch (incl. line-4 numbering past skipped lines); comment/blank skip test :541-559 asserts physical `line_index`; CRLF test :562-568; `parse_bed` appends rows in read order (:99), no sort anywhere — green |
| 7 | TRUTH-7 (SC2): tests at tests/ ROOT, zero skip machinery, zero markers, no subprocess, no bedtools binary, 8+11 handcrafted baked cases | ✓ VERIFIED | File at `tests/test_genomic_intervals.py` (root, owner lock honored); `grep -cE "pytest\.mark"` = 0; skip-machinery grep: only docstring/test-name word matches, no `importorskip/skipif/xfail`; subprocess grep = 0 on both files; 39 tests pass in 3.63 s; zero platform conditionals (`sys.platform`/`os.name`/`win32` grep clean; "Windows" appears only in prose) |
| 8 | TRUTH-8 (SC2): seeded corpora ≥300 jaccard + ≥200 loj property cases + ~12 baked subset literals per spike; deterministic reruns | ✓ VERIFIED | `test_jaccard_seeded_corpus_property_invariants` iterates `range(300)`, `test_loj_seeded_corpus_property_invariants` `range(200)` with independently re-derived plain-python oracle; 2×12 baked subsets (test :978-1304); determinism test :908-918; generator port verified verbatim vs `spike_jaccard_parity.py:758-921` / `spike_loj_parity.py:1048-1240` including rng draw order (the `_stagger` factoring preserves the spike's exact draw sequence) |
| 9 | TRUTH-9 (SC3): module row visible at ≥96% in coverage report (omitted files never print) | ✓ VERIFIED | This verifier ran `uv run --no-sync coverage run -m pytest tests/test_genomic_intervals.py -q` + `coverage report --include="dnallm/utils/genomic_intervals.py"` → row **visible, 110 stmts, 0 miss, 100%**; pyproject omit list (:552-562) has exactly 7 entries, none matching the module; coverage measured in the same phase commits that add the module |
| 10 | TRUTH-10 (WPAR-01 exclusion): the spike's informational fourth return key appears nowhere in the module | ✓ VERIFIED | `grep -cE "['\"]n_intersections['\"]" dnallm/utils/genomic_intervals.py` = 0; spike's `_count_overlapping` + 4th dict key NOT ported (`_sweep_stats` returns a 2-tuple); exact-key-set assertion present (test :80-85 and in the corpus loop :831); exclusion documented in the module docstring without reproducing the key's spelling |

**Score:** 10/10 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `dnallm/utils/genomic_intervals.py` | NEW pure-numpy module: parse_bed / jaccard / loj_intersect public; BedRow, NULL_* constants, `_merge_intervals` / `_sweep_stats` / `_null_b_fields` / `_parses_int` private | ✓ VERIFIED | 327 lines; every named symbol present (grep of definitions at :36/:54/:103/:130/:172/:213-218/:221/:241/:274); substantive (verbatim port of the parity-proven spike cores — cross-compared line by line); wired via re-export + 39 passing tests |
| `dnallm/utils/__init__.py` | EDIT: 3-name import block + 3 sorted `__all__` entries | ✓ VERIFIED | :17-21 block; `jaccard`/`loj_intersect`/`parse_bed` in `__all__` at correct alphabetical positions; identity + call-through proven by `test_package_surface_reexports_all_three_entry_points` |
| `tests/test_genomic_intervals.py` | NEW root-level test file: oracle battery + pin table + guards + seeded corpora + purity test | ✓ VERIFIED | 1304 lines, 39 test functions, all green; zero markers/skips/subprocess |

### Key Link Verification

(gsd-tools maps nothing for the PLAN's string-form key_links; verified manually.)

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `dnallm.utils` package surface | `genomic_intervals` | re-export import + `__all__` | ✓ WIRED | Import block + `__all__` entries; identity assertion `pkg_jaccard is jaccard` etc. + calls through the surface (test :93-113) |
| `tests/test_genomic_intervals.py` | spike oracle literals (261010-wx2, 261011-0a3) | baked verbatim | ✓ WIRED | Every handcrafted literal cross-checked against the two `report.txt` artifacts by this verifier — 8/8 jaccard values, 9/9 pin-table variants, 11/11 loj line counts (0,3,9,2,2,3,2,1,8,9,8), case-11's 8 lines re-derived by hand |
| `coverage report --include` filter | visible module row ≥96% | coverage-CLI form | ✓ WIRED | Re-run by this verifier: visible, 100% (110 stmts, 0 miss) |
| `loj_intersect` list[str] output | Phase 16 WPAR-03 row adapter | documented downstream contract | ✓ WIRED | Docstring documents tab-joined lines + "the Phase-16 row adapter splits each line on tab" (:294-296); nothing imports the module yet by design (phase boundary; Phase 16 = WPAR-03..08) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `genomic_intervals.py` | `rows` / `a_chroms` / `b_chroms` / `out` | real file reads via `parse_bed` (`open(path)`, per-line parse) | Yes — every return value is computed from parsed file content; no static/hardcoded returns in any code path | ✓ FLOWING |
| `tests/test_genomic_intervals.py` | baked literals | spike `report.txt` live-bedtools parity records (external oracle) | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Package-surface re-export | `.venv/bin/python -c "from dnallm.utils import jaccard, loj_intersect, parse_bed"` | exit 0 | ✓ PASS |
| Full phase test file | `.venv/bin/python -m pytest tests/test_genomic_intervals.py -q --no-header` | 39 passed in 3.63 s | ✓ PASS |
| Per-module coverage gate (SC3) | `uv run --no-sync coverage run -m pytest tests/test_genomic_intervals.py -q && uv run --no-sync coverage report --include="dnallm/utils/genomic_intervals.py"` | 110 stmts, 0 miss, **100%**, row visible | ✓ PASS |
| Full fast lane | `.venv/bin/python -m pytest -m "not slow" --no-cov -q` | 2526 passed, 2 skipped, 75 deselected in 101 s | ✓ PASS |
| Fast-lane skips proven pre-existing | `pytest -m "not slow" -rs` | `tests/examples/test_examples.py:257` (no import statements) + `tests/models/test_plant_helixseek_fla_kernels.py:90` (fla not installed) — both unrelated to Phase 15 | ✓ PASS |
| Negative gates | quoted-key grep on module; marker/skip/subprocess greps on both files | 0 / 0 / 0 / 0 | ✓ PASS |
| Facade byte-stability | `git diff 088db69..e0fdbed -- dnallm/__init__.py` | 0 lines | ✓ PASS |
| Census triple untouched | `git diff 088db69..e0fdbed --name-only \| grep -E "ci.yml|pyproject.toml|expected_skips"` | 0 files | ✓ PASS |

### Probe Execution

No probes declared: the phase is a pure library port; PLAN/SUMMARY declare no `probe-*.sh` scripts and none exist under `scripts/*/tests/`. All phase checks are the pytest/coverage/git commands above. — N/A

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| WPAR-01 | 15-01-PLAN | Pure-numpy interval module with oracle-pinned contracts, outside coverage omit list, import-pure numpy-only | ✓ SATISFIED | TRUTH-1..6, 9, 10 — every pinned contract locked by a passing exact-literal test; 100% visible coverage; purity test green |
| WPAR-02 | 15-01-PLAN | Spike corpora + handcrafted edges as fast platform-neutral unit tests with baked oracle literals, no bedtools binary | ✓ SATISFIED | TRUTH-7, 8 — 39 tests / 3.6 s, zero skip/marker/subprocess machinery, 500 seeded property cases + 24 baked subset literals, binary-free by construction |

Orphaned requirements: none — REQUIREMENTS.md maps exactly WPAR-01/02 to Phase 15; both are claimed by 15-01-PLAN and satisfied.

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
| ----------- | ---------- | ------ | ------- | -------- | --------------- | ------- |
| `tests/test_genomic_intervals.py` | WPAR-01, WPAR-02 | 39 | 0 | 0 | Value (exact dict equality, `==` float literals, full-line string equality) | PASS |

- **Disabled tests on requirements:** 0 (zero skip machinery — grep-proven).
- **Circular patterns:** 0. Handcrafted oracle provenance is **VALID** (external source: the two spikes' live-bedtools v2.31.1 parity `report.txt` records — this verifier cross-checked every baked value against them). The 2×12 seeded-subset literals are a documented regression/determinism tripwire (AS-2), not a correctness oracle; correctness there is anchored by the verbatim port of code proven against live bedtools over 5008 cases, plus the independently re-derived property invariants — honest, not circular.
- **Coverage quantity:** 300 jaccard + 200 loj property cases confirmed in the test source (`range(300)` / `range(200)`); assertion strength is the strongest tier throughout.

### Decision Coverage

`check.decision-coverage-verify`: "No trackable decisions in CONTEXT.md." (skipped, non-blocking). The CONTEXT's substantive locks (tests/ ROOT placement, 3-name API surface, baked-oracle methodology, ≥96% standard) were each independently verified as honored above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| — | — | none found | — | Zero TBD/FIXME/XXX/TODO/HACK/placeholder matches across all three files; no stub returns; no hardcoded empty data; no console-log implementations; only "Windows" prose mentions, no platform conditionals |

### Deviation Check (case 11) — VERIFIED CORRECT

The one recorded deviation (plan prose placed the 6-column null vector inside the case-11 NER driver; executor baked the true spike oracle) was re-derived by hand by this verifier: over the case-11 inputs (spike_loj_parity.py:1010-1032, baked identically at test :299-325), every one of the 8 A rows satisfies the strict half-open inequalities against exactly one B row — all-hits, 8 lines, no null fill. The baked 8 lines match this derivation exactly, and the spike record (report.txt:47 "case 11 … lines 8 canonical PASS sequence PASS") confirms 8 lines. The 6-column NER null vector is pinned line-exactly where it actually fires: case 7's two boundary-touching rows (test :222-223) and the 6-int/6-non-int pin-table variants. Every Task-2 acceptance criterion locks verbatim. Baking the plan's literal wording would have pinned output bedtools never produced — the auto-fix was correct.

### Human Verification Required

N/A — Infrastructure/core-library phase with no user-facing elements. All acceptance criteria are verifiable programmatically, and every behavior-dependent contract (ordering, multiplicity, boundary, null-fill, guards, purity, determinism) is exercised by a passing behavioral test this verifier ran. Per the plan's own scoping, CI-leg greenness (test / test-windows / coverage-gate / coverage-nightly) is observed post-push; the structural wiring for it is verified here: `testpaths = ["tests", …]` auto-collects the root-level file on every leg, CI legs invoke bare `pytest -m "not slow" --cov` with no path filter, and the file is platform-neutral by construction (no platform conditionals, no binary, no subprocess; 39/39 green locally).

### Gaps Summary

None. All 10 must-have truths verified with behavioral evidence; all three artifacts exist, are substantive, wired, and data-flowing; all four key links hold; no anti-patterns; owner locks honored (tests/ ROOT placement, `dnallm/__init__.py` byte-stable across `c7c387b..e0fdbed`); the census triple untouched (zero ci.yml / pyproject.toml / expected_skips.yaml edits in the phase commits); fast lane 2526P/2S green with both skips proven pre-existing; the single deviation re-derived and confirmed correct against the spike oracle record.

---

_Verified: 2026-10-13T15:08:10Z_
_Verifier: Claude (gsd-verifier)_
