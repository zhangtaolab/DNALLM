# Phase 15: Windows Parity Core — Numpy Interval Evaluators - Pattern Map

**Mapped:** 2026-10-13
**Files analyzed:** 3 (2 new, 1 edit)
**Analogs found:** 3 / 3

All analog paths verified git-tracked (`git ls-files` non-empty for each).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dnallm/utils/genomic_intervals.py` (NEW) | utility (pure library) | file-I/O → transform | `dnallm/utils/genomic_coords.py` (structure) + `.planning/quick/261011-0a3-*/spike_loj_parity.py` + `.planning/quick/261010-wx2-*/spike_jaccard_parity.py` (algorithm source) | exact (algorithm); exact (module anatomy) |
| `dnallm/utils/__init__.py` (EDIT) | config/package surface | — | existing file itself; precedent is the genomic_coords block at lines 9-16 + 40-69 | exact |
| `tests/test_genomic_intervals.py` (NEW) | test | file-I/O + baked-oracle literals | `tests/utils/test_genomic_coords.py` | exact (style; path adaptation noted below) |

**Path adaptation (owner-locked):** CONTEXT locks the test path at `tests/test_genomic_intervals.py` (tests/ root). The style precedent file actually lives at `tests/utils/test_genomic_coords.py` — the parenthetical precedent description in CONTEXT is factually off, but the root-level path is the locked part. Do NOT "fix" the plan toward `tests/utils/`. Root-level `tests/test_*.py` is an established repo pattern (test_config_mapping_contracts.py, test_extras_guard.py, etc.); collection is identical (`testpaths = ["tests", ...]`, pyproject.toml:516).

## Pattern Assignments

### `dnallm/utils/genomic_intervals.py` (utility, file-I/O → transform)

**Analogs:**
- Module anatomy: `dnallm/utils/genomic_coords.py` (270 lines)
- Algorithms (port near-verbatim, strip harness): `.planning/quick/261011-0a3-spike-numpy-pandas-loj-intersect-parity-/spike_loj_parity.py` and `.planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/spike_jaccard_parity.py` (both git-tracked). Never `import` from `.planning/` — port the code ("task artifact — never library code").

**Module docstring + loud-ValueError philosophy** (`dnallm/utils/genomic_coords.py:1-19`):

```python
"""
Genomic coordinate and chromosome-name normalization helpers.

The single place where ... Every function validates its input and raises
ValueError on anything outside the documented forms: the characteristic
failure mode of hand-scattered conversion code is silence (empty results,
off-by-one bases), never a loud error, so silent-empty results are
structurally impossible here.
"""
```

**Module conventions** (from genomic_coords): private helpers prefixed `_` (`_validate_gff1` at line 31); Google-style `Args:/Returns:/Raises:` docstrings on every public function; stdlib-only top imports (numpy + `pathlib` + `typing` here); lazy imports for optional deps (not needed — numpy-only).

**`parse_bed` + `BedRow` — port from spike_loj_parity.py:98-163** (NamedTuple with chrom/start/end/extra/line_index; `open(path, encoding="utf-8")`, `rstrip("\r\n")` CRLF handling, skip blank/`#` lines, loud ValueError naming file + line on <3 columns / non-integer coords / negative start / empty interval). Key excerpt (lines 138-163):

```python
    rows: list[BedRow] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 3:
                raise ValueError(
                    f"{path}: line {lineno}: expected at least 3 tab-separated "
                    f"columns, got {len(columns)}: {line!r}"
                )
```

**Null-B constants + `_null_b_fields` — port from spike_loj_parity.py:76-81, 186-217 verbatim** (NULL_CHROM/START/END/NAME/SCORE/OTHER and the count-dependent encoder). The 6-column vector `['.', '-1', '-1', '.', '-1', '.']` is load-bearing for the NER consumer's `name == "-1"` branch.

**`loj_intersect` — port from spike_loj_parity.py:220-271** (per-A-row broadcast join, A-file order, B-file-order hits via `np.nonzero`, half-open strict inequalities, explicit `dtype=np.int64` per-chrom arrays, `"\t".join` output lines):

```python
        if a.chrom in grouped:
            mask = (starts_by_chrom[a.chrom] < a.end) & (ends_by_chrom[a.chrom] > a.start)
            hits = [grouped[a.chrom][int(i)] for i in np.nonzero(mask)[0]]
```

**`jaccard` — port from spike_jaccard_parity.py:128-147 (merge_intervals) and 199-249 (sweep + jaccard_np)**; drop `_count_overlapping` and `n_intersections` entirely (lines 150-174 — excluded by WPAR-01, dead contract). Load-bearing details: `order = np.lexsort((-deltas, coords))` (starts tie-break before ends so bookended intervals merge into the union), `sorted(set(a_chroms) | set(b_chroms))` (one-sided chroms count toward union), `jaccard = intersection / union if union else 0.0`, `int(...)` at the Python boundary, `dtype=np.int64` on every coordinate array (Windows int-dtype discipline).

---

### `dnallm/utils/__init__.py` (EDIT — add re-export)

**Analog:** the file itself; copy the genomic_coords block exactly (lines 9-16 import + `__all__` entries at 40-69, sorted).

Add after the genomic_coords block:

```python
from .genomic_intervals import (
    jaccard,
    loj_intersect,
    parse_bed,
)
```

and alphabetical `"jaccard"`, `"loj_intersect"`, `"parse_bed"` entries in `__all__`. `dnallm/__init__.py` facade stays byte-stable (v1.2 invariant).

---

### `tests/test_genomic_intervals.py` (test, file-I/O + baked oracles)

**Analog:** `tests/utils/test_genomic_coords.py` — style: module docstring (line 1), flat `test_*` functions (no classes), `──`-rule section-divider comments (line 42), inline literal fixtures at module top (lines 22-39), `pytest.raises(ValueError, match=...)` guards, `tmp_path` for file-based tests, NO custom markers, NO skips.

**Import-purity test precedent** (tests/utils/test_genomic_coords.py:344-368) — same shape, sentinel-blocking pandas/pybedtools/torch, and targeting `dnallm.utils.genomic_intervals` DIRECTLY (not `dnallm.utils` — the package init imports torch via support.py):

```python
    sys.modules.pop("dnallm.utils.genomic_coords", None)
    sys.modules["pyfastx"] = None  # any import attempt now raises ImportError
    try:
        mod = importlib.import_module("dnallm.utils.genomic_coords")
    finally:
        ...  # restore sys.modules entry AND the package attribute (single live module)
```

**Oracle literals to bake** (from RESEARCH.md Code Examples, verbatim-cited from the spike artifacts): 8 jaccard handcrafted cases (spike_jaccard_parity.py:697-754; report.txt:21-28), 11 loj handcrafted cases (spike_loj_parity.py:899-1033; report.txt:37-47), the null-fill pin table (261011-0a3 report.txt:8-17), frozen-pair aggregates as documented constants only (intersection=15872, union=48878, jaccard=0.32472687098490116 — the 51/94-row corpora are gitignored/unrecoverable; do NOT plan literal row fixtures).

**Seeded corpus generators to port:** spike_jaccard_parity.py:875-921 (seed 20261010) and spike_loj_parity.py:1185-1240 (seed 20261011) — PATTERNS 6-cycle, CHROM_POOL, 30-row cap, shuffle-every-4th. Two-layer testing: property invariants over a few hundred regenerated cases + fully baked literals for a small deterministic subset (~12 cases).

## Shared Patterns

### Loud ValueError input validation
**Source:** `dnallm/utils/genomic_coords.py:31-40, 252-268`; spike `parse_bed` guards (spike_loj_parity.py:144-161).
**Apply to:** `parse_bed`, and any coordinate validation in the new module. Always name the file and line number in the message; never silently skip malformed rows.

### Explicit dtype + Python-int boundary discipline
**Source:** spike_jaccard_parity.py:204-215 (`np.ones(..., dtype=np.int64)`, `np.full(..., -1, dtype=np.int64)`, `int(widths[...].sum())`); spike_loj_parity.py:250-255 (`np.array(..., dtype=np.int64)` per chrom).
**Apply to:** every coordinate array construction in the new module (Windows int32/float64 upcast defense).

### CRLF-safe file handling
**Source:** spike_loj_parity.py:141 (`line.rstrip("\r\n")`); genomic_coords.py:254.
**Apply to:** `parse_bed`; test fixtures written with `"\n".join(rows)` + `write_text(..., encoding="utf-8")`.

### Per-module coverage proof command (coverage-CLI form, NOT `--cov=`)
**Source:** `.planning/milestones/v1.2-phases/12-.../12-01-...-PLAN.md:227` (WINDOWS #20 workaround).
**Apply to:** this phase's ≥96% proof:
`uv run --no-sync coverage run -m pytest tests/test_genomic_intervals.py -q && uv run --no-sync coverage report --include="dnallm/utils/genomic_intervals.py"` — gate on module row ≥96% and VISIBLE (outside the pyproject omit list). Any `pytest --cov=<dotted>` form crashes in the dev venv.

## No Analog Found

None — all three files have exact in-repo analogs.

## Metadata

**Analog search scope:** `dnallm/utils/`, `tests/`, `.planning/quick/261010-wx2*/`, `.planning/quick/261011-0a3*/`
**Analogs read:** 5 files (genomic_coords.py, utils/__init__.py, tests/utils/test_genomic_coords.py sections, spike_loj_parity.py:70-270, spike_jaccard_parity.py:128-253)
**Pattern extraction date:** 2026-10-13
