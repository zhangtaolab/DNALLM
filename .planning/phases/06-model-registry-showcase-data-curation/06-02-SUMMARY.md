---
phase: 06-model-registry-showcase-data-curation
plan: "02"
subsystem: testing
tags: [genomics, gff3, pyfastx, coordinates, chromosome-names, pytest]

# Dependency graph
requires: []
provides:
  - "dnallm.utils.genomic_coords — six shared helpers (normalize_chrom, gff1_to_half_open, half_open_to_gff1, fetch_sequence, parse_gff_attributes, slice_gff_rows) importable as dnallm.utils.*"
  - "20-test unit suite on tiny inline fixtures covering every guard, conversion, and import purity"
  - "In-helper non-emptiness guards (fetch_sequence empty-result, slice_gff_rows require_nonempty) making silent-empty results structurally impossible (SHOW-02)"
affects: ["06-03 select_loci.py (all coordinate math routes through this helper)", "Phase 7 showcase notebooks", "Phase 8 execution tests"]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
actuals:
  tokens: 6208    # chars/4 over the realized diff (3 files, base 39c73ad)
  tasks: 2
  commits: 3      # MEASURED: git rev-list --count 39c73ad..HEAD

plan_head_before: 39c73ad45f1dabc0f54616d6f9e3ad4e36033dae
plan_head_after: 975af3399f9786875b22ae3254fa8780d75d9ade

# Tech tracking
tech-stack:
  added: []       # no new libraries — pyfastx is a pre-existing dev extra, imported lazily
  patterns:
    - "Strict-parse utility module: undocumented input forms raise ValueError instead of best-effort coercion (Pitfall-11 silence guard)"
    - "Function-local optional-dep import (pyfastx inside fetch_sequence only) keeping the eagerly-imported utils package import-pure"
    - "In-helper non-emptiness guards rather than caller-side asserts (SHOW-02 structural guarantee)"

key-files:
  created:
    - dnallm/utils/genomic_coords.py
    - tests/utils/test_genomic_coords.py
  modified:
    - dnallm/utils/__init__.py

key-decisions:
  - "RED evidence produced via a NotImplementedError stub module committed inside the test commit: the suite fails at test level (RED_EVIDENCE_OK, 20 failed / exit 1) instead of at collection, where #3770 classifies an ImportError as INVALID_RED"
  - "normalize_chrom token vocabulary restricted to numeric tokens and organelle letters C/M after a case-insensitive chr prefix (plus bare digits): anything else — e.g. 'chromosome1' — raises; the first draft's [A-Za-z0-9]+ token over-accepted and the RED-authored tests caught it"
  - "fetch_sequence accepts an open pyfastx.Fasta OR a FASTA path, making the function-local pyfastx import load-bearing rather than decorative"
  - "half_open_to_gff1 rejects zero-width intervals (end0 <= start0) — an empty half-open interval has no 1-based closed representation; refusing beats coercing"
  - "Module landed at 237 lines vs the plan's soft '~150' guidance: Google-style Args/Returns/Raises docstrings for six public functions dominate; logic beyond the verified contract is zero"

patterns-established:
  - "Genomics coordinate/chrom conversion must route through dnallm.utils.genomic_coords — no inline minus-one adjustments (06-03 key_link)"
  - "Chromosome-name comparison is exact string equality after explicit style conversion; organelles ChrC/ChrM pass through both styles"

requirements-completed: [SHOW-02]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Shared, unit-tested coordinate/chrom-name normalization helper at dnallm/utils/genomic_coords.py (six functions, every ValueError guard tested, import-pure, 100% statement coverage)"
    requirement: SHOW-02
    verification:
      - kind: unit
        ref: "tests/utils/test_genomic_coords.py (20 tests: chrom both directions, organelle pass-through, round-trips, length-1 + touching features, trailing-semicolon/comma-Parent parse, all guards, import purity)"
        status: pass
      - kind: automated
        ref: "coverage gate: dnallm/utils/genomic_coords.py 82 stmts, 0 miss, 100% (>=90 required)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Public surface wired: all six functions re-exported from dnallm.utils with alphabetical __all__, pyfastx absent from sys.modules at package import, whole fast suite green with zero new skips"
    requirement: SHOW-02
    verification:
      - kind: integration
        ref: "import surface check ('surface-ok', pyfastx not in sys.modules) + ruff check/format clean on dnallm/utils/__init__.py"
        status: pass
      - kind: integration
        ref: "pytest tests/ -m 'not slow' -q: 1702 passed, 1 pre-existing typed skip (test_examples.py:254 'No import statements found'), 50 deselected, exit 0"
        status: pass
    human_judgment: false

# Metrics
duration: 16 min
completed: 2026-10-03
status: complete
---

# Phase 06 Plan 02: Genomic Coordinate Helper Summary

**Import-pure dnallm.utils.genomic_coords module (six strict-parse helpers: chrom-name normalization, half-open/closed conversion, pyfastx fetch with silent-empty guards, TAIR10-tolerant GFF3 attribute parsing, order-preserving locus slicing) at 100% statement coverage, re-exported on the dnallm.utils surface**

## Performance

- **Duration:** 16 min
- **Started:** 2026-10-03T07:11:09Z
- **Completed:** 2026-10-03T07:27:26Z
- **Tasks:** 2/2
- **Files modified:** 3 (2 created, 1 modified)

## Accomplishments
- Task 1 (TDD): RED → GREEN cycle with machine-verified RED evidence (`gsd check tdd-red-evidence` → RED_EVIDENCE_OK); 20 tests on tiny inline fixtures; module coverage 100% (82/82 statements)
- Task 2: public surface wired via `dnallm/utils/__init__.py` re-export block + six alphabetical `__all__` entries; whole fast suite green (1702 passed / 1 pre-existing typed skip / 0 failed)
- SHOW-02 satisfied end to end: non-emptiness guards live inside the helpers (fetch_sequence raises on empty fetch; slice_gff_rows raises on empty locus with require_nonempty), and the module is importable everywhere (06-03 script, Phase 7 notebooks, Phase 8 tests) without path hacks

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: failing tests + stub** - `3f7c851` (test)
2. **Task 1 GREEN: helper implementation** - `f98a30c` (feat)
3. **Task 2: dnallm.utils re-export** - `975af33` (feat)

**Plan metadata:** (see final docs commit)

_Note: TDD task produced the RED→GREEN pair; REFACTOR was skipped — the GREEN implementation is already in final form (no cleanup committed, per the TDD contract "only commit if changes made")._

## Files Created/Modified
- `dnallm/utils/genomic_coords.py` - the six-function helper (created; 237 lines, stdlib-only at module level, pyfastx lazily imported inside fetch_sequence)
- `tests/utils/test_genomic_coords.py` - 20 unit tests on inline fixtures (created; includes the pyfastx-blocked import-purity test)
- `dnallm/utils/__init__.py` - re-export block + six `__all__` entries (modified)

## TDD Cycle Record (Task 1, tdd="true")

- **RED:** `tests/utils/test_genomic_coords.py` written first; `dnallm/utils/genomic_coords.py` committed in the same test commit as a NotImplementedError stub so the suite fails at TEST level rather than at collection (an import-time failure is INVALID_RED per #3770). Run: `pytest tests/utils/test_genomic_coords.py -q` → **20 failed, exit 1**. Evidence record verified by `gsd check tdd-red-evidence` → **RED_EVIDENCE_OK** (target `tests.utils.test_genomic_coords`, pytest junitxml → Surefire class-level matching; record at /tmp/gsd-0602-red-record.json).
- **GREEN:** full implementation; two corrective iterations (see Deviations 1 and 3) → **20 passed**; coverage 100%.
- **REFACTOR:** none needed — no third commit.
- **Gate commits:** `test(06-02)` 3f7c851 precedes `feat(06-02)` f98a30c. ✓

## Decisions Made
See `key-decisions` in frontmatter. Additional execution notes:
- pyfastx error semantics probed live before GREEN: unknown sequence names raise `NameError` ("Sequence Chr9 does not exists"), so `fetch_sequence` catches `(KeyError, NameError)` and re-raises as ValueError — membership pre-check avoided to keep duck-typed stubs testable.
- pyfastx is CI-present: the `base` extra chains `dnallm[dev,...]` and dev carries `pyfastx>=2.2.0`, so the fetch tests need no skip guard (acceptance: zero new skips — confirmed).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] normalize_chrom over-accepted undocumented name forms**
- **Found during:** Task 1 (GREEN run)
- **Issue:** First-draft regex `[cC][hH][rR]([A-Za-z0-9]+)` accepted `"chromosome1"` as prefix `chr` + token `omosome1` — exactly the best-effort renaming the plan prohibits
- **Fix:** Token vocabulary restricted to `[0-9]+|[CM]` (numeric chromosomes + TAIR organelle letters); the RED-authored test listing `chromosome1` as invalid caught it
- **Files modified:** dnallm/utils/genomic_coords.py
- **Verification:** test_normalize_chrom_rejects_unknown_forms passes; all 20 green
- **Committed in:** f98a30c (Task 1 GREEN commit)

**2. [Rule 3 - Blocking] Plan's coverage-gate command crashes in this environment**
- **Found during:** Task 1 (verify step)
- **Issue:** `--cov=dnallm.utils.genomic_coords` reproducibly kills the run at conftest import: pandas → numpy raises "ImportError: cannot load module more than once per process" under the module-scoped coverage tracer (pre-existing pytest-cov/numpy 2.5.3 interaction; bare `--cov` and `--cov=dnallm` work fine — CI uses the bare form). Additionally, the plan's grep `'genomic_coords'` also matches the pytest progress line `tests/utils/test_genomic_coords.py ....`, whose `$4` is dots → the awk gate exits 1 on a passing run
- **Fix:** Ran the same gate on the package-scoped form with a table-row-anchored grep: `pytest --cov=dnallm --cov-report=term tests/utils/test_genomic_coords.py -q | grep -E 'genomic_coords\.py[[:space:]]+[0-9]+...%' | awk '...'` — identical intent and threshold
- **Files modified:** none (verification-command adaptation only)
- **Verification:** `dnallm/utils/genomic_coords.py 82 0 100%` → gate exit 0
- **Committed in:** n/a (no code change)

**3. [Rule 1 - Bug] Boundary-slice test expectation was wrong**
- **Found during:** Task 1 (first GREEN run)
- **Issue:** test_slice_gff_rows_closed_interval_boundaries expected the CDS row (3760-3913) inside locus [5899, 6788] — it does not overlap; the implementation was correct, the test fixture arithmetic was not
- **Fix:** Expectation corrected to the two edge-touching features (3631-5899 gene/mRNA, 6788-9130 gene); comment states the interior-CDS exclusion explicitly
- **Files modified:** tests/utils/test_genomic_coords.py
- **Verification:** all 20 green
- **Committed in:** f98a30c (Task 1 GREEN commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 3 blocking verify-command issue)
**Impact on plan:** All fixes were correctness/verification necessities inside task scope. No scope creep; the six-function interface contract implemented verbatim.

## Issues Encountered
- **Pre-existing mypy environment breakage (out of scope):** `mypy dnallm/` reports 38 errors in 26 files and aborts on numpy's stubs (`type _Falsy` statement needs Python 3.12 vs configured `python_version = "3.10"`) — "errors prevented further checking". Zero errors mention genomic_coords; CI runs mypy advisory (`|| true`). Not caused by this plan; not fixed (scope boundary). Note: pre-commit hooks are not installed in this checkout (`.git/hooks/` has only samples), so commits ran without hook execution — the equivalent gates (ruff format, ruff check) were run manually and pass on all three files.
- No auth gates, no package installs (T-06-SC: nothing to verify).

## Threat Mitigations Landed (per plan threat_model)
- **T-06-03:** column-9 attributes treated as opaque data — split/strip only, no eval/format execution; undocumented chrom forms raise ValueError (fail loud).
- **T-06-04:** every conversion validates ints/ranges; `require_nonempty` and the empty-fetch guard turn silent-empty into raised ValueError.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 06-03 `select_loci.py` can now route ALL coordinate math and GFF3 attribute parsing through `dnallm.utils.*` imports (key_link satisfied: `from dnallm.utils.genomic_coords import ...`, no inline minus-one adjustments)
- No blockers.

---
*Phase: 06-model-registry-showcase-data-curation*
*Completed: 2026-10-03*

## Self-Check: PASSED
