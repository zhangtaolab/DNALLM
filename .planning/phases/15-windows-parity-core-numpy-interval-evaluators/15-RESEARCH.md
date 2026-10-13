# Phase 15: Windows Parity Core — Numpy Interval Evaluators - Research

**Researched:** 2026-10-13
**Domain:** pure-numpy genomic interval arithmetic (bedtools-parity evaluators) + repo test/coverage/CI integration
**Confidence:** HIGH

## Summary

Phase 15 is a **pure library addition** with an unusually strong evidence base: both parity spikes (261010-wx2 jaccard, 261011-0a3 loj-intersect) survived as complete, readable artifacts under `.planning/quick/` — including the full 1205-line and 1523-line spike scripts, the captured `report.txt` oracle outputs, and SUMMARY records. Every contract the phase must carry (null-B fill encoding, emission order, join multiplicity, bookended-adjacency merge, the excluded `n_intersections`) is pinned **empirically against live bedtools v2.31.1** in those artifacts, and the algorithm code to port is already written, docstringed, and parity-proven (5008 compared cases, 0 mismatches, both gates). The porting job is: lift the spike cores into `dnallm/utils/genomic_intervals.py` following the `genomic_coords.py` precedent (explicit re-export on `dnallm.utils`, root-`tests/` test file, loud-ValueError parsing), strip the harness layer (argparse, bedtools subprocess, report machinery), and bake the oracle literals as unit tests.

Three repo-mechanics facts shape the plan. First, the coverage proof: the ≥96% per-module owner rule is proven with the phase-11/12 pattern `coverage run -m pytest <file>` + `coverage report --include="dnallm/utils/genomic_intervals.py"` (the `pytest --cov=<dotted>` form crashes in the local dev venv — WINDOWS #20, numpy 2.5.x tracer interaction; the coverage-CLI form is the proven workaround). Second, the new test file is collected on every pytest leg automatically (`testpaths = ["tests", "dnallm/mcp/tests"]`, `pytest -m "not slow" --cov` on test/test-windows/coverage-gate), including test-windows — **zero CI workflow edits are needed**; the D-03 examples census is untouched because the file lands outside `tests/examples`. Third, the frozen wx2 selection pair (`pred.bed` 51 rows / `truth.bed` 94 rows) is **gitignored (`.gitignore:60 .scratch/`) and absent from disk** — its recorded oracle numbers (intersection=15872, union=48878, jaccard=0.32472687098490116) are preserved in the spike records but the row corpora are NOT recoverable in-repo; the unit layer's randomized corpora must come from porting the spike's seeded generators (seeds 20261010 / 20261011), which are fully deterministic and fully preserved.

**Primary recommendation:** Port the three spike cores (`parse_bed` from 0a3, `jaccard` from wx2 minus the `n_intersections` analogue, `loj_intersect` from 0a3) essentially verbatim into `dnallm/utils/genomic_intervals.py`, add the explicit re-export to `dnallm/utils/__init__.py`, and author `tests/test_genomic_intervals.py` in the genomic_coords flat-function style with (a) the 8+11 handcrafted spike cases baked as literal input/output pairs, (b) the ported seeded generators driving a property/invariant corpus, and (c) an import-purity test proving the module imports with numpy-only (pandas/pybedtools/torch sentinel-blocked).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (owner-accepted 2026-10-13)

- Test file at repo root: `tests/test_genomic_intervals.py` (mirrors the `tests/test_genomic_coords.py` precedent for dnallm/utils modules)
- API surface strictly `parse_bed` / `jaccard` / `loj_intersect`, re-exported on the `dnallm.utils` surface (genomic_coords precedent); private helpers stay private
- Oracle literals baked from the two spike corpora (261010-wx2: jaccard parity incl. 0.324727 frozen pair + randomized property cases; 261011-0a3: loj spike) plus handcrafted edge cases (boundary touching, chrom isolation, duplicates, empty B) — no runtime bedtools comparison anywhere in the unit layer
- Coverage: ≥96% per-module in the same change, module provably outside the 7-entry coverage omit list (pyproject.toml:554), green on every CI leg including test-windows

Contracts pinned by ROADMAP/WPAR-01 (carried, not re-decided):

- 6-column null-B fill including the `-1` score slot; A-file-order emission accepting unsorted input; join-not-merge multiplicity; bookended-adjacency merge; `n_intersections` excluded; import-pure numpy-only

### Claude's Discretion

Internal algorithm choices (vectorization strategy, dtype handling) at Claude's discretion within the pinned contracts.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WPAR-01 | Pure-numpy interval module `dnallm/utils/genomic_intervals.py` (`parse_bed`, `jaccard`, `loj_intersect`) carries the oracle-pinned spike contracts (6-column null-B fill incl. the `-1` score slot, A-file-order emission accepting unsorted input, join-not-merge multiplicity, bookended-adjacency merge; `n_intersections` excluded), sits provably outside the coverage omit list, import-pure numpy-only | Spike cores fully preserved and Read-verified (Code Examples section carries the verbatim algorithms); omit-list enumeration proves no glob matches; re-export precedent Read-verified in `dnallm/utils/__init__.py:9-16` |
| WPAR-02 | Spike corpora (261010-wx2 / 261011-0a3 seeds) + handcrafted edge cases run as fast, platform-neutral unit tests with baked oracle literals on every CI leg — no bedtools binary required | Seeded corpus generators fully preserved in spike scripts (deterministic: `default_rng(seed)`, byte-identical reruns recorded); handcrafted case inputs preserved verbatim; CI legs collect `tests/` automatically — no workflow edits |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| BED parsing + interval arithmetic (parse_bed / jaccard / loj_intersect) | Library (`dnallm/utils/genomic_intervals.py`) | — | Pure, platform-neutral computation; numpy is already a core dependency; this tier makes it importable by Phase 16's Windows dispatch without platform coupling |
| Oracle contract pinning | Unit test layer (`tests/test_genomic_intervals.py`) | — | Baked literals, no runtime oracle; every CI leg runs it identically |
| Package surface exposure | `dnallm/utils/__init__.py` (explicit re-export) | — | Locked decision; `dnallm/__init__.py` facade stays byte-stable (v1.2 invariant — NOT touched here) |
| bedtools truth chain | Linux example/nightly lanes (EXISTING — unchanged this phase) | Phase 16 (Windows dispatch) | WINDOWS #19 platform split: this phase ships only the evaluator; no example, gate, or workflow file changes |
| Coverage enforcement | pyproject `[tool.coverage.*]` (global 90 gate) + per-module ≥96% owner rule proven by coverage-CLI run | — | No config change needed; the proof is a command, documented below |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | `>=2.0.0` floor (pyproject.toml:48); dev venv 2.5.3; CI pins 2.2.0 on linux legs; test-windows resolves latest from `.[base]` | All interval arithmetic | Already a core dependency; the spikes ran numpy-only by design ("pandas would add import weight with no arithmetic to do") [VERIFIED: spike summaries] |

**No new packages are installed by this phase.** The module imports numpy + stdlib (`pathlib`, `typing`) only; the test file imports pytest + numpy + stdlib.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy-only evaluators | pybedtools / pyranges / rurunes | REJECTED by owner (WINDOWS #19 evaluation 2026-10-10): pyranges1 has no Windows wheels, rurunes 0.2.7 ships no win_amd64 wheels and requires-python >=3.12; the numpy evaluator is the lower-dependency path. Locked — do not revisit |
| pandas for the join | numpy broadcasting | Spike decision, carried: per-chrom int64 arrays carry the whole computation; string payloads ride by row index |

**Installation:** none — `pip install`/`uv pip install` steps are out of scope for this phase.

**Version verification:** `numpy>=2.0.0` is the declared floor [VERIFIED: pyproject.toml:47-48 — `# numpy 1.x retired 2026-10-10; 1.x users stay on the 0.8.x series` / `"numpy>=2.0.0",`]. Dev venv live-probed at 2.5.3 (all APIs used by the spikes execute). CI matrix read directly: `numpy-version: ['2.2.0']` [VERIFIED: .github/workflows/ci.yml:45].

## Package Legitimacy Audit

> This phase installs zero external packages. No registry lookups required.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                    ┌──────────────────────────────────────────────────────┐
                    │  dnallm/utils/genomic_intervals.py  (NEW, pure lib)  │
                    │                                                      │
  BED file A ──────►│  parse_bed(path) ──► list[BedRow]                    │
  BED file B ──────►│    (BED3+, file order preserved, never sorted,       │
                    │     exact chrom string key, loud ValueError)         │
                    │        │                                             │
                    │        ├─► jaccard(a, b)                             │
                    │        │     per-chrom: merge_intervals (sort by     │
                    │        │     (start,end), coalesce next.start <=     │
                    │        │     cur.end — bookended adjacency merges)   │
                    │        │     → sweep (+1/-1 boundaries, starts       │
                    │        │     tie-break first; coverage==2 = ∩,       │
                    │        │     coverage>=1 = ∪), sum across chroms     │
                    │        │     → {intersection, union, jaccard}        │
                    │        │       (n_intersections EXCLUDED)            │
                    │        │                                             │
                    │        └─► loj_intersect(a, b)                       │
                    │              per A row (A-file order, unsorted OK):  │
                    │              mask = (b_starts < a.end) &             │
                    │                     (b_ends > a.start)               │
                    │              hit? → one line per B hit, B-file order │
                    │                    (JOIN: duplicates multiply)       │
                    │              miss? → null-B fill per pinned table    │
                    │              → list[str] output lines                │
                    └──────────────────────┬───────────────────────────────┘
                                           │ explicit re-export
                    ┌──────────────────────▼───────────────────────────────┐
                    │  dnallm/utils/__init__.py  (+parse_bed, jaccard,     │
                    │  loj_intersect in imports + __all__)                 │
                    └──────────────────────┬───────────────────────────────┘
                                           │ imported by (Phase 16, NOT this phase)
                    ┌──────────────────────▼───────────────────────────────┐
                    │  Windows dispatch in generate_bpe_dataset.py / NER   │
                    │  notebook (WPAR-03..08 — out of scope here)          │
                    └──────────────────────────────────────────────────────┘

  Unit test oracle flow (tests/test_genomic_intervals.py, every CI leg incl.
  test-windows; NO bedtools anywhere):
    spike handcrafted cases (8 jaccard + 11 loj, verbatim literals)
    + ported seeded generators (seeds 20261010 / 20261011, deterministic)
      → assert baked oracle literals / invariants
    + import-purity test (pandas/pybedtools/torch sentinel-blocked)
```

### Recommended Project Structure

```
dnallm/utils/genomic_intervals.py   # NEW — the three public fns + private helpers
dnallm/utils/__init__.py            # EDIT — add re-export (imports + __all__)
tests/test_genomic_intervals.py     # NEW — root-level per locked decision
```

Three files. Nothing else in the repo changes (no ci.yml, no pyproject, no expected_skips.yaml — the tests never skip and land outside `tests/examples`).

### Pattern 1: Module anatomy (genomic_coords precedent, Read-verified)

**What:** `dnallm/utils/genomic_coords.py` (270 lines) is the structural template: module docstring stating the single-responsibility contract, private helpers prefixed `_`, Google-style `Args:/Returns:/Raises:` docstrings, loud `ValueError` on every malformed input ("silent-empty results are structurally impossible"), lazy imports for optional deps.
**When to use:** for the new module verbatim.
**Example** — the re-export mechanism [VERIFIED: dnallm/utils/__init__.py:9-16, 38]:

```python
from .genomic_coords import (
    fetch_sequence,
    gff1_to_half_open,
    half_open_to_gff1,
    normalize_chrom,
    parse_gff_attributes,
    slice_gff_rows,
)
```
plus matching alphabetical entries in `__all__` (which lists `"fetch_sequence"`, `"gff1_to_half_open"`, …). The new module adds the same import block + three `__all__` entries (`"jaccard"`, `"loj_intersect"`, `"parse_bed"` — `__all__` is sorted in the precedent). `dnallm/__init__.py` stays byte-stable.

### Pattern 2: jaccard = merge + boundary sweep (wx2 core, Read-verified)

**What:** per chromosome: merge (sort by `(start, end)`, coalesce while `next start <= current end` — overlaps AND bookended adjacency), then a combined boundary sweep where coverage-exactly-2 bases are the intersection and coverage>=1 the union. Both merged inputs are internally disjoint, so coverage never exceeds 2.
**When to use:** always — this is the parity-proven algorithm; do not redesign.
**Example** [VERIFIED: spike_jaccard_parity.py:139-147, 201-216]:

```python
    merged: list[list[int]] = []
    for start, end in sorted(pairs):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.array(merged, dtype=np.int64)
```
```python
    starts = np.concatenate([a_merged[:, 0], b_merged[:, 0]])
    ends = np.concatenate([a_merged[:, 1], b_merged[:, 1]])
    coords = np.concatenate([starts, ends])
    deltas = np.concatenate(
        [np.ones(starts.size, dtype=np.int64), np.full(ends.size, -1, dtype=np.int64)]
    )
    # Primary sort key is the coordinate; the secondary key -delta puts +1
    # (start) before -1 (end) when boundaries coincide.
    order = np.lexsort((-deltas, coords))
    coords_sorted = coords[order]
    coverage = np.cumsum(deltas[order])
    widths = np.diff(coords_sorted)
    segment_coverage = coverage[:-1]
    intersection = int(widths[segment_coverage == 2].sum())
    union = int(widths[segment_coverage >= 1].sum())
```

The starts-tie-break-before-ends detail (`-deltas` secondary key) is load-bearing: at a coordinate where one interval ends and another starts, coverage must not dip to 0 or the union splits bookended intervals. `jaccard_np` iterates `sorted(set(a_chroms) | set(b_chroms))` — a one-sided chromosome still counts toward the union — and returns `jaccard = intersection / union if union else 0.0` [VERIFIED: spike_jaccard_parity.py:241-249].

### Pattern 3: loj = per-A-row broadcast join (0a3 core, Read-verified)

**What:** every A row survives in A-file order; per A row, same-chrom B rows with half-open overlap emit one line per pair in B-file order; no-hit rows emit once with the pinned null fill.
**When to use:** always — parity-proven; note the spike's own scale comment ("spike scale is <= 30 rows per side — correctness and emission order over asymptotics").
**Example** [VERIFIED: spike_loj_parity.py:258-270]:

```python
    for a in a_rows:
        prefix = (a.chrom, str(a.start), str(a.end), *a.extra)
        hits: list[BedRow] = []
        if a.chrom in grouped:
            mask = (starts_by_chrom[a.chrom] < a.end) & (ends_by_chrom[a.chrom] > a.start)
            hits = [grouped[a.chrom][int(i)] for i in np.nonzero(mask)[0]]
        if hits:
            for b in hits:
                out.append(
                    "\t".join((*prefix, b.chrom, str(b.start), str(b.end), *b.extra))
                )
        else:
            out.append("\t".join((*prefix, *null_fields)))
```
The strict inequalities encode the half-open overlap (`b.start < a.end AND a.start < b.end` — touching boundaries do NOT overlap) and `np.nonzero` yields indices in B-file order (grouping preserved insertion order). B-column count is derived as `len(b_rows[0].extra) + 3 if b_rows else 3` — the empty-B-file default is the 3-column null form [VERIFIED: spike_loj_parity.py:243].

### Pattern 4: the oracle-pinned null-B fill table

**What:** the exact literals bedtools prints for a no-hit A row's B side, as a function of B column count (and, only at count 5, score-column int-parseability). This table is EMPIRICALLY pinned against live bedtools v2.31.1 — the official docs do not document the field-level encoding (web search confirmed; see Sources), and the spike's own deviation record shows the guessed map was WRONG at counts 5 and 7.
**Verbatim pins** [VERIFIED: 261011-0a3 report.txt:68-79]:

```text
  B columns 3: literals ['.', '-1', '-1'] numpy-emit PASS
  B columns 4: literals ['.', '-1', '-1', '.'] numpy-emit PASS
  B columns 5 score col '0': literals ['.', '-1', '-1', '.', '-1'] numpy-emit PASS
  B columns 5 score col 'exon': literals ['.', '-1', '-1', '.', '.'] numpy-emit PASS
  B columns 6 score col 'exon': literals ['.', '-1', '-1', '.', '-1', '.'] numpy-emit PASS
  B columns 6 score col '0': literals ['.', '-1', '-1', '.', '-1', '.'] numpy-emit PASS
  B columns 7 score col '0': literals ['.', '-1', '-1', '.', '.', '.', '.'] numpy-emit PASS
  B columns 7 score col 'exon': literals ['.', '-1', '-1', '.', '.', '.', '.'] numpy-emit PASS
  empty B file: ['.', '-1', '-1'] (column count unknowable — oracle emits the 3-column null form) numpy-emit PASS
```

Implementation constants [VERIFIED: spike_loj_parity.py:76-81]:

```python
NULL_CHROM = "."  # B column 1 (chromosome) under a null join
NULL_START = "-1"  # B column 2 (start) under a null join
NULL_END = "-1"  # B column 3 (end) under a null join
NULL_NAME = "."  # B column 4 (BED name slot) under a null join
NULL_SCORE = "-1"  # B column 5 (BED score slot, numeric) under a null join
NULL_OTHER = "."  # B column 6+ (strand and beyond) under a null join
```

and the encoder [VERIFIED: spike_loj_parity.py:209-217]:

```python
    if count <= 4:
        return [NULL_CHROM, NULL_START, NULL_END] + [NULL_OTHER] * (count - 3)
    if count == 5:
        slot = NULL_SCORE if numeric_score else NULL_OTHER
        return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME, slot]
    if count == 6:
        return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME, NULL_SCORE, NULL_OTHER]
    # count >= 7: beyond BED6 the score slot reverts to "." as well.
    return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME] + [NULL_OTHER] * (count - 4)
```

The NER 6-column consequence (why the `-1` score slot is load-bearing): "a no-hit A row against the 6-column annotation B carries B-side `.` `-1` `-1` `.` `-1` `.`; the port MUST emit `-1` in B column 5 because the consumer branches on `name == "-1"` where `name = iv.fields[10]`" [VERIFIED: 261011-0a3-SUMMARY.md:83]. Unit tests must pin this exact 6-column vector.

### Pattern 5: test authoring style (Read-verified precedent)

**What:** `tests/utils/test_genomic_coords.py` style: module docstring, flat `test_*` functions (no classes), section-divider comments, inline literal fixtures at module top, `pytest.raises(ValueError, match=...)` for guards, `tmp_path` for file-based tests, no custom markers (the `utils` marker exists in pyproject but the precedent file does not use it — follow the precedent).
**Import-purity test precedent** [VERIFIED: tests/utils/test_genomic_coords.py:344-400 — `test_module_import_is_pyfastx_free` and `test_import_purity_leaves_single_live_module`]: pop the module from `sys.modules`, set `sys.modules["pyfastx"] = None` (any import attempt now raises ImportError), `importlib.import_module("dnallm.utils.genomic_coords")`, restore state, and assert the package attribute and `sys.modules` entry point at one object. The new module needs the same shape with pandas/pybedtools/torch sentinels — and targeting `dnallm.utils.genomic_intervals` DIRECTLY, not `dnallm.utils` (the package `__init__` imports torch via `support.py` — `from torch.cuda import get_device_capability` [VERIFIED: dnallm/utils/support.py:2] — so package-level purity is impossible and not required).

### Anti-Patterns to Avoid

- **Sorting in `parse_bed` / `loj_intersect`:** the loj contract is A-file order with B hits in B-file order, unsorted inputs accepted (pin 2). Never sort rows in the loj path. Only `jaccard`'s internal merge sorts.
- **Deduplicating or coalescing in `loj_intersect`:** it is a join — duplicate rows on either side multiply per copy (pin: "exact-duplicate rows on either side join per copy — never merged, never coalesced").
- **Guessing the null-B fill from BED convention:** the obvious reading (all `.`) is wrong at the score slot; the plan's original example map was wrong at counts 5 and 7 (two of the spike's three auto-fixed bugs). Use the pinned table verbatim.
- **Porting `n_intersections`:** bedtools counts raw A rows (51 on the frozen pair) while the natural numpy analogue counts merged-A intervals (45) — "NOT portable as-is" [VERIFIED: 261010-wx2-SUMMARY.md:97]. WPAR-01 locks the exclusion: the public `jaccard` contract carries intersection/union/jaccard only. The spike's `_count_overlapping` helper (spike_jaccard_parity.py:150-174) fed only that informational field — recommend dropping it from the port entirely (fewer lines, less coverage surface, no dead contract).
- **Importing the spike artifacts from library/test code:** spike scripts are "task artifact — never library code" [VERIFIED: 261010-wx2-SUMMARY.md:50]; port the code, never `import` from `.planning/`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Interval overlap/join semantics | A novel re-derivation of bedtools behavior | The spike algorithms verbatim (Patterns 2-4) | 5008 compared cases, 0 mismatches; every re-design decision risks re-learning what the spikes already pinned (the null-fill table alone cost two spike bug-fixes) |
| Corpus generation | New ad-hoc random cases | The ported seeded generators (seeds 20261010/20261011, 6-pattern cycle, fixed pools) | Determinism is proven (byte-identical reruns recorded); the corpus shape (chrom-name styles, duplicates, adjacency, shuffle-every-4th) is exactly the adversarial surface the parity verdict covers |
| Coverage proof | Ad-hoc `pytest --cov=...` invocation | `coverage run -m pytest` + `coverage report --include=...` | `pytest --cov=<dotted>` crashes repo-wide in the local dev venv (WINDOWS #20); the coverage-CLI form is the in-repo proven workaround |
| BED parsing validation | Silent skip / best-effort coercion | Loud `ValueError` naming file + line (spike `parse_bed3`/`parse_bed` guards, genomic_coords philosophy) | Silence is the characteristic failure mode of hand-scattered conversion code (genomic_coords module docstring, Read-verified) |

**Key insight:** the spikes ARE the specification. The phase's risk is transcription fidelity, not algorithm design — which is why the handcrafted literals and pins belong in tests as verbatim copies.

## Common Pitfalls

### Pitfall 1: The frozen wx2 pair is NOT recoverable in-repo
**What goes wrong:** planning assumes `tests` can recompute jaccard=0.324727 over the frozen 51/94-row pair.
**Why it happens:** the pair lives at `example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed`, matched by `.gitignore:60` (`.scratch/`), and the directory does not exist on disk today (verified: `ls` empty, `git check-ignore` confirms the match). The spike read the files at runtime; only the aggregate oracle numbers are preserved.
**How to avoid:** bake the recorded aggregates (intersection=15872, union=48878, jaccard full-precision 0.32472687098490116, 6-dp anchor 0.324727, informational n_intersections divergence 51-vs-45) as documented literals where useful, but the *computational* unit oracles are the handcrafted cases + seeded randomized corpora. The frozen-pair in-band gate on a real regenerated pair is Phase 16's WPAR-05 concern (selection.md band parsing on both paths). See Open Questions Q1.
**Warning signs:** a plan task that writes 51/94-row literal BED fixtures into the test file "from the spike" — those rows are not in any repo artifact.

### Pitfall 2: Test-path precedent mismatch
**What goes wrong:** the CONTEXT decision says `tests/test_genomic_intervals.py` "mirrors the `tests/test_genomic_coords.py` precedent" — but the precedent file actually lives at `tests/utils/test_genomic_coords.py` (verified by `find`; no `tests/test_genomic_coords.py` exists).
**Why it happens:** the decision's explicit path (`tests/test_genomic_intervals.py`, tests/ root) is the locked part; the parenthetical precedent description is factually off. Root-level `tests/test_*.py` files are an established repo pattern (test_config_mapping_contracts.py, test_extras_guard.py, test_models_lock_contracts.py, test_runner_infra_contracts.py).
**How to avoid:** honor the locked path `tests/test_genomic_intervals.py`. Collection is identical either way (`testpaths = ["tests", "dnallm/mcp/tests"]` [VERIFIED: pyproject.toml:516]). Flagging the discrepancy here so the planner doesn't "fix" the decision toward `tests/utils/` or get confused verifying the precedent.
**Warning signs:** a plan that places the file at `tests/utils/test_genomic_intervals.py`.

### Pitfall 3: `pytest --cov=<module>` crashes in the local dev venv (WINDOWS #20)
**What goes wrong:** the per-module coverage proof written as `pytest tests/test_genomic_intervals.py --cov=dnallm.utils.genomic_intervals` fails in the shared dev venv: coverage 7.16.2 `source_pkgs` resolution imports dnallm under the active tracer, and the shims→torch→numpy 2.5.3 chain trips numpy's "cannot load module more than once per process" C-extension guard [VERIFIED: .planning/WINDOWS.md ledger #20].
**How to avoid:** use the proven phase-10/11/12 pattern — [VERIFIED: 12-01-PLAN.md:227] `uv run --no-sync coverage run -m pytest tests/interpret -q -m "not slow" && uv run --no-sync coverage report --include="dnallm/interpret/*"` — i.e. for this phase: `uv run --no-sync coverage run -m pytest tests/test_genomic_intervals.py -q` then `uv run --no-sync coverage report --include="dnallm/utils/genomic_intervals.py"`; gate on the module row >= 96% and VISIBLE. (The `uv run --no-sync` form is the repo's recorded workaround for the uv-resolver deferred item.)
**Warning signs:** any `--cov=<dotted module>` in a verification command; CI legs are unaffected (numpy 2.2.0 there) but the executor's local proof must use the coverage-CLI form.

### Pitfall 4: baked randomized-corpus literals depend on numpy Generator stream stability
**What goes wrong:** literals baked on the dev venv (numpy 2.5.3) diverge on a CI leg (2.2.0 pinned on linux; test-windows resolves the latest 2.x) because an `rng.integers/choice/permutation` stream changed.
**Why it happens:** numpy `Generator` streams are versioned-stable by policy (stream changes require major-version bumps / documented exceptions), so this SHOULD be safe across 2.2-2.5 — but the spikes only ever ran on 2.5.3 (their determinism proof is same-version rerun determinism, not cross-version).
**How to avoid:** two-layer corpus testing — (a) property invariants that need no baked per-case oracle (jaccard ∈ [0,1], jaccard(a,b)==jaccard(b,a), intersection ≤ min(union parts), every A row yields ≥1 loj line, loj line count == Σ per-A hits, hits respect the half-open rule), and (b) a SMALL deterministic subset (e.g. one case per pattern × shuffle state) with fully baked outputs. Any stream divergence then fails loudly on the first CI run rather than producing 2000 red herring diffs. Note the corpus draws use only `rng.integers`, `rng.choice`, `rng.permutation`, `rng.random` [VERIFIED: spike generator code, Read].
**Warning signs:** baking expected outputs for all 2000/3000 corpus cases.

### Pitfall 5: numpy default-int / dtype pitfalls on Windows
**What goes wrong:** coordinate arithmetic silently running int32 or float64.
**Why it happens:** platform default ints differ (though numpy 2.0 made Windows default int 64-bit [CITED: numpy 2.0.0 release notes]); `np.concatenate` of int64 with Python-int lists can upcast to float/object.
**How to avoid:** the spike pattern everywhere: explicit `dtype=np.int64` array construction (`np.array(..., dtype=np.int64)`, `np.ones(..., dtype=np.int64)`, `np.full(..., -1, dtype=np.int64)`), and `int(...)` conversion at the Python boundary (`intersection = int(widths[...].sum())`). Both spikes ran green on Linux/2.5.3 with exactly this discipline; port it as-is. Division only happens once (`intersection / union` on Python ints → float64, deterministic IEEE).
**Warning signs:** any array construction without an explicit dtype near coordinates.

### Pitfall 6: CRLF / encoding in test fixtures and parsing
**What goes wrong:** Windows-written fixture files carry `\r\n`, breaking byte-literal output comparisons.
**How to avoid:** the spike parser already does `line.rstrip("\r\n")` on read [VERIFIED: spike_loj_parity.py:141] — keep it. Tests write fixtures with explicit `"\n".join(rows)` + `write_text(..., encoding="utf-8")` (spike `write_case_bed` shape); test-windows disables git autocrlf anyway (ci.yml:150) and tmp_path files never pass through git.
**Warning signs:** fixture writers using `print`/`writelines` without terminator control.

### Pitfall 7: accidentally widening the census / skip surface
**What goes wrong:** the new tests introduce a skip, a marker, or a tests/examples change that trips the D-03 census assertion or the junit skip audit.
**How to avoid:** no skips at all (platform-neutral is the point), no new markers (precedent uses none), file at tests/ root (outside tests/examples — the stage-0.5 census triple `216/217 tests collected (1 deselected)` [VERIFIED: ci.yml:867] is untouched), and the root `tests/conftest.py` (imports torch/pandas/yaml — Read-verified) applies as-is with no changes.
**Warning signs:** `pytest.mark.skip*`, `importorskip`, or expected_skips.yaml edits appearing in the plan.

## Code Examples

### Handcrafted oracle literals — jaccard (wx2, 8 cases) [VERIFIED: spike_jaccard_parity.py:697-754 inputs; report.txt:21-28 oracles]

```python
# inputs (chrom, start, end); oracle: inter/union/jaccard
# 1 empty-side-A: a=[], b=[("Chr1",1000,2000),("Chr1",3000,3500)] → 0/1500/0.0
# 2 identical-sets: both [("Chr1",100,250),("Chr1",400,600),("Chr2",50,150)] → 450/450/1.0
# 3 strict-containment: a=[("Chr1",200,800),("Chr1",2100,2200)],
#   b=[("Chr1",100,1000),("Chr1",2000,3000)] → 700/1900/0.3684210526315789
# 4 bookended-adjacent-touching: a=[("Chr1",0,100)], b=[("Chr1",100,200)] → 0/200/0.0
# 5 fully-disjoint: a=[("Chr1",0,100)], b=[("Chr1",500,600)] → 0/200/0.0
# 6 same-coords-different-chrom-names: a=[("Chr1",100,200)], b=[("chr1",100,200)] → 0/200/0.0
# 7 duplicates-and-staggered: a=[("Chr1",100,300),("Chr1",100,300),("Chr1",200,400),("Chr1",205,210)],
#   b=[("Chr1",150,350)] → 200/300/0.6666666666666666
# 8 shuffled-row-order: a=[("Chr1",500,700),("Chr1",100,300),("Chr2",50,120),("Chr1",650,900)],
#   b=[("Chr2",100,150),("Chr1",200,600)] → 220/900/0.24444444444444444
```

Cases 4, 6, 7 pin the three contract clauses the phase success criteria name explicitly: bookended-adjacency merge (union 200 = coalesced), chrom isolation by exact string key, duplicate merge semantics.

### Handcrafted oracle literals — loj (0a3, 11 cases) [VERIFIED: spike_loj_parity.py:899-1033 inputs; report.txt:37-47 line counts]

Case names and output-line counts (full expected-line literals are derivable from the pinned contracts; case 11 "ner-realistic-per-gene-iteration-order" is the NER-shaped 6+6-column driver with token/gene/strand payloads — inputs at spike_loj_parity.py:1010-1032):

```text
case 1 empty-A → 0 lines          case 7 boundary-touching-no-hit → 2 lines
case 2 empty-B → 3 lines          case 8 same-coords-different-chrom-names → 1 line
case 3 all-overlap → 9 lines      case 9 duplicate-rows-both-sides → 8 lines
case 4 no-overlap → 2 lines       case 10 unsorted-A-B-multi-hit → 9 lines
case 5 strict-containment → 2     case 11 ner-realistic-per-gene-iteration-order → 8 lines
case 6 multiplicity-one-A-multi-B → 3 lines
```

Boundary-touching inputs (case 7) for the half-open pin: `a=[("Chr1",100,200,...),("Chr1",400,500,...)]`, `b=[("Chr1",200,300,...),("Chr1",300,400,...)]` — A.end==B.start and B.end==A.start both miss, both rows null-fill.

### The frozen-pair recorded oracle (documentation anchor; see Pitfall 1) [VERIFIED: 261010-wx2 report.txt:10-18]

```text
metric          numpy                 bedtools(live)        recorded-ref          note
intersection    15872                 15872                 15872
union           48878                 48878                 48878
jaccard         0.32472687098490116   0.324727              0.324727              bedtools prints 6 dp
n_intersections 45                    51                    51                    informational, never gated
informational delta vs 6-dp anchor 0.324727: 1.2901509882645712e-07 (bedtools print rounding)
```

The parity gate was jaccard delta ≤ 1e-9 vs live full precision (achieved 0.0). For unit-test float comparisons of baked jaccard literals: `intersection / union` on Python ints is exact IEEE-754 double division (deterministic cross-platform) — exact `==` against baked literals is safe; a tight `math.isclose(abs_tol=1e-12)` is the defensive alternative. The looser tolerance story (e.g. `0.3247270000001` trailing deltas) belongs to Phase 16's selection.md 6-dp comparison layer, not this unit layer.

### Corpus generator shape to port [VERIFIED: spike_jaccard_parity.py:875-921; spike_loj_parity.py:1185-1240]

```python
PATTERNS = ("mixed", "dense", "containment", "adjacency", "disjoint", "duplicates")
CHROM_POOL = ("Chr1", "Chr2", "Chr3", "Chr4", "chr1", "1", "scaffoldA")
WINDOW = 8000
# per case: pattern = PATTERNS[index % 6]; n_chroms = rng.integers(1, 5);
# cap = max(2, 30 // n_chroms); both sides truncated to 30 rows; never both empty;
# shuffled = (index % 4 == 3) → rng.permutation of both sides,
# else written canonically sorted by (chrom, start, end).
```
wx2 seeds 20261010 (3000 cases, BED3); 0a3 seeds 20261011 (2000 cases, 6+6 NER shape with `TOKEN_POOL`/`GENE_POOL`/`FEATURE_POOL`/`STRAND_POOL` payload draws — constants at spike_loj_parity.py:1039-1045). A unit-scale rerun (100-500 cases) executes in well under a second per the spike runtimes (3008 bedtools subprocess invocations took 5.1s; without subprocesses the compute is trivial).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CI numpy matrix `1.26.4 / 2.2.0` | `numpy-version: ['2.2.0']` only | 2026-10-10, quick task 261010-rpa (commit 834f9e2) | The phase objective's "1.26.4/2.2.0 per CI matrix" is STALE — verified directly in ci.yml:45. numpy>=2.0.0 floor [VERIFIED: pyproject.toml:48]; numpy 1.x compatibility concerns are moot |
| `pytest --cov=<module>` for per-module proofs | `coverage run -m pytest` + `coverage report --include=` | Phase 10 (WINDOWS #20), reused 11/12 | The only working per-module proof form in the dev venv |
| pyranges1 as bedtools replacement candidate | numpy-only evaluator (owner platform-split) | 2026-10-10 evaluation in WINDOWS #19 | Locked; do not research interval libraries again |

**Deprecated/outdated:** nothing in the module's dependency surface — numpy APIs used (`lexsort`, `cumsum`, `diff`, `nonzero`, `concatenate`, `ones`, `full`, `empty`, `array`) are core-stable; the numpy 2.0 release notes' removal list does not touch any of them [CITED: numpy 2.0.0 release notes via web search], and all were live-probed on the dev venv (numpy 2.5.3) this session.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | numpy `Generator` (`default_rng`) streams for `integers/choice/permutation/random` are stable across 2.2.0 ↔ 2.5.3, so generator-ported corpora produce identical draws on all CI legs | Pitfalls / Code Examples | Baked subset literals fail on a CI leg — loud failure at PR time; fallback is property-invariants-only for randomized cases (already the recommended primary layer) |
| A2 | `loj_intersect` returning `list[str]` tab-joined output lines (spike shape) is the right public signature for Phase 16's row adapter (which will `.split("\t")` or wrap) | Architecture Patterns | Phase 16 would need a small adapter change — but the API surface (3 names) is locked either way; signature details are within Claude's discretion per CONTEXT |
| A3 | Root `tests/conftest.py`'s heavy imports (torch/pandas/yaml) are an acceptable fixed cost for the new test file (as for every existing test) | Common Pitfalls 7 | None material — suite-wide status quo; test still runs "fast" by suite standards |
| A4 | The `n_intersections`-excluded `jaccard` return shape (dict of intersection/union/jaccard, dropping the spike's informational 4th key) satisfies "n_intersections excluded" | Patterns 2 / Anti-patterns | If the owner intended a different shape (e.g. NamedTuple), it's a same-phase rename of a return type — cheap; flag at plan review if uncertain |

## Open Questions

1. **Frozen-pair unit-layer treatment (owner expectation alignment)**
   - What we know: CONTEXT says oracle literals include "0.324727 frozen pair"; the pair's row corpora are gitignored and absent; aggregates are recorded; regenerating the pair requires running the torch-heavy CRE showcase (not unit-layer); Phase 16's WPAR-05 already owns the frozen-metric in-band gate via selection.md.
   - What's unclear: whether the owner expects a unit test that *computes* jaccard over a frozen 51/94-row pair in Phase 15.
   - Recommendation: treat the frozen pair in Phase 15 as documented oracle constants (the aggregates above) + handcrafted/seeded corpora for computation; defer the real-pair band check to Phase 16/WPAR-05 (it needs the regenerated pair anyway). If the owner wants it here, the pair must first be regenerated outside the unit layer and committed as a fixture — an explicit scope addition.
2. **Randomized-corpus depth**
   - What we know: full corpus is 3000+2000 cases with per-case live-oracle comparison — impossible without bedtools; determinism is same-version only (A1).
   - What's unclear: how many cases the owner expects in the unit layer.
   - Recommendation: property invariants over a few hundred regenerated cases + fully baked literals for a small deterministic subset (one per pattern × shuffle state = 12 cases per spike). Fast (<1s), platform-neutral, and regression-meaningful.
3. **`parse_bed` return-type exposure**
   - What we know: 0a3's `BedRow` NamedTuple (chrom/start/end/extra/line_index) is the natural public return; "private helpers stay private" is locked, but the row type is a return type, not a helper.
   - What's unclear: whether `BedRow` should be importable/documented or treated as an implementation detail.
   - Recommendation: keep `BedRow` as a module-level NamedTuple (public-by-visibility, undocumented in `dnallm.utils.__all__` — only the 3 functions are re-exported, matching the locked API surface).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| numpy | module + tests | ✓ (dev venv) | 2.5.3 (live-probed) | — |
| numpy on CI linux legs | test / coverage-gate / coverage-nightly | ✓ (pinned) | 2.2.0 [VERIFIED: ci.yml:45,415,542] | — |
| numpy on test-windows / test-mamba | tests via `.[base]` | ✓ (resolved, floor >=2.0.0) | latest 2.x at install time | — |
| bedtools | NOT required — prohibited in unit layer | ✓ locally (v2.31.1, homebrew) but unused | — | n/a |
| pytest / pytest-timeout / pytest-asyncio | test execution (addopts) | ✓ (existing suite infra) | per base/test extras | — |
| Python | module (>=3.11 floor) | ✓ | dev 3.13; CI 3.11/3.12/3.13; test-windows 3.12 | — |
| uv | coverage-proof invocation | ✓ | 0.12.x (recorded) | plain `coverage` CLI from the venv |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none.

## Validation Architecture

> SKIPPED — `workflow.nyquist_validation` is explicitly `false` in `.planning/config.json`.

(For orientation: the phase's verification is the repo's existing suite — `pytest tests/test_genomic_intervals.py -q` targeted lane; `pytest -m "not slow"` census on all legs; the coverage-CLI per-module proof from Pitfall 3.)

## Security Domain

### Applicable ASVS Categories (L1)

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | library function, no resources |
| V5 Input Validation | yes | Loud `ValueError` on malformed BED rows (fewer than 3 columns, non-integer coordinates, negative start, empty interval) naming file + line — ported verbatim from the spikes, matching the genomic_coords "silent-empty is structurally impossible" philosophy |
| V6 Cryptography | no | none |

### Known Threat Patterns for pure-numpy BED parsing

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/untrusted BED content | Tampering | Strict parse + loud ValueError; no eval, no exec, no subprocess (unlike the bedtools CLI path, no shell-injection surface exists at all in this module) |
| Path handling | — | Plain `open(path, encoding="utf-8")` reads only; no writes outside caller-supplied paths (module writes nothing) |

Note: the phase REMOVES attack surface relative to the subprocess path it will eventually replace on Windows (Phase 16) — no argv construction, no shell.

## Sources

### Primary (HIGH confidence — Read this session)

- `.planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/` — spike_jaccard_parity.py (core algorithms lines 74-255, handcrafted 681-755, generators 758-921), report.txt (frozen-pair table, corpus results), 261010-wx2-SUMMARY.md (porting notes incl. the n_intersections non-portability)
- `.planning/quick/261011-0a3-spike-numpy-pandas-loj-intersect-parity-/` — spike_loj_parity.py (BedRow/parse_bed 98-163, _null_b_fields 186-217, loj_intersect_np 220-271, NULL constants 76-81, handcrafted 880-1033, pools 1039-1045, generator 1185-1240), report.txt (all three pins verbatim), 261011-0a3-SUMMARY.md (NER port notes)
- `dnallm/utils/genomic_coords.py` (module precedent, full read), `dnallm/utils/__init__.py:9-16` (re-export), `dnallm/utils/support.py:2` (torch import in package init)
- `tests/utils/test_genomic_coords.py:1-120, 344-400` (test style, import-purity precedent)
- `pyproject.toml:47-48, 514-562` (numpy floor, pytest config, markers, coverage omit list — 7 entries enumerated), `.github/workflows/ci.yml` (full: test matrix 2.2.0-only, test-windows job, coverage-gate, census gate, bedtools provisioning — NOT used by unit layer)
- `.planning/WINDOWS.md` ledger #19 (platform split, owner directive) and #20 (cov-crash + `coverage run --include=` workaround)
- `.planning/milestones/v1.2-phases/12-.../12-01-...-PLAN.md:227` (per-module coverage proof command)
- `example/notebooks/plant_helixseek_shared/data/selection.md` (jaccard=0.3247 anchor, 4dp; floor >=0.3)
- `.gitignore:60` + `git check-ignore` (frozen pair unrecoverable)

### Secondary (MEDIUM confidence)

- [NumPy 2.0.0 release notes](https://numpy.org/doc/stable/release/2.0.0-notes.html) — removal list does not include lexsort/cumsum/diff/nonzero; Windows default int → 64-bit; NEP 50 promotion notes (checked via web search)
- [NEP 52 — Python API cleanup for NumPy 2.0](https://github.com/numpy/numpy/blob/main/doc/neps/nep-0052-python-api-cleanup.rst) — scope of 2.0 API cleanup
- bedtools `-loj` null-fill: NO authoritative field-level documentation found (official docs describe -loj as left outer join only; community tutorials cover other flags) — which is why the spike's empirical pins are the governing source for this contract

### Tertiary (LOW confidence)

- numpy Generator cross-version stream stability (2.2.0 ↔ 2.5.3) — policy-based inference, not tested on 2.2.0 this session (A1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies; numpy floor/matrix verified against pyproject/ci.yml directly
- Architecture: HIGH — module/re-export/test placement all Read-verified against in-repo precedent; spike algorithms Read-verified line-by-line
- Pitfalls: HIGH — all seven grounded in Read-verified repo facts (gitignore status, ledger #20, census gate, precedent path)
- Oracle contracts: HIGH — every pinned value quoted verbatim from spike artifacts with file:line cites

**Research date:** 2026-10-13
**Valid until:** 2026-11-13 (stable, repo-grounded; re-check only if the CI matrix or coverage config changes before planning lands)
