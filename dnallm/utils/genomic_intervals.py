"""
Genomic interval evaluators reproducing bedtools semantics with pure numpy.

The public entry points (``parse_bed`` / ``jaccard`` / ``loj_intersect``)
are ports of the two parity spikes 261010-wx2 (interval jaccard) and
261011-0a3 (left-outer-join intersect), which compared 5008 cases against
live bedtools v2.31.1 with zero mismatches. Every contract here — the
null-B fill literals, A-file-order emission, join-not-merge multiplicity,
bookended-adjacency merge, half-open touching boundaries — is pinned
empirically by those runs and locked by the baked oracle tests in
``tests/test_genomic_intervals.py``; none of it is re-derived from BED
convention, where the obvious reading is wrong at exactly the places the
spikes caught it.

The module exists so the Windows dispatch path can evaluate interval
arithmetic in-process (numpy is a core dependency) instead of shelling out
to the bedtools binary, which has no Windows build: no argv construction,
no subprocess, no shell. Imports are numpy + stdlib only.

The characteristic failure mode of hand-scattered interval code is silence
(empty results, quietly wrong bases), so ``parse_bed`` raises a loud
ValueError naming the offending file and line on anything malformed —
silent-empty results are structurally impossible here. The informational
overlap-count key excluded by requirement WPAR-01 is deliberately absent
from the ``jaccard`` return: bedtools' exact definition of that metric is
unspecified and its natural numpy analogue diverges (merged-interval vs
raw-row counting), so it was never parity-portable.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class BedRow(NamedTuple):
    """One parsed BED row with file-order metadata.

    Attributes:
        chrom: chromosome name as an exact string key ("Chr1" != "chr1").
        start: 0-based half-open start coordinate.
        end: half-open end coordinate (> start).
        extra: trailing columns verbatim (may be empty).
        line_index: 1-based line number in the source file.
    """

    chrom: str
    start: int
    end: int
    extra: list[str]
    line_index: int


def parse_bed(path: str | Path) -> list[BedRow]:
    """
    Parse a BED3+ file into rows, preserving file order.

    Trailing columns beyond the first three are carried verbatim as strings;
    rows are NEVER sorted here. Blank lines and '#' comment lines are
    skipped; anything else must parse. Grouping by chromosome happens later
    and always by exact string key.

    Args:
        path: BED file with at least 3 tab-separated columns (chromosome,
            0-based start, half-open end) and any number of trailing
            columns.

    Returns:
        list[BedRow]: rows in file order.

    Raises:
        ValueError: naming the offending file and line number when a row
            has fewer than 3 columns, non-integer coordinates, a negative
            start, or an empty (start >= end) interval.
    """
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
            try:
                start, end = int(columns[1]), int(columns[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: line {lineno}: non-integer coordinate(s) "
                    f"{columns[1]!r}, {columns[2]!r}"
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}: line {lineno}: invalid 0-based half-open interval ({start}, {end})"
                )
            rows.append(BedRow(columns[0], start, end, columns[3:], lineno))
    return rows


def _merge_intervals(pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Sort and coalesce intervals with bedtools merge semantics.

    Intervals are sorted by (start, end), then merged while the next start
    is <= the current end: overlapping AND bookended (adjacent) intervals
    coalesce, matching ``bedtools merge`` at its default distance 0.

    Args:
        pairs: intervals as (start, end) tuples; need not be sorted.

    Returns:
        np.ndarray: shape (k, 2) int64 merged intervals, sorted, pairwise
        disjoint and non-adjacent. Empty input returns an int64 (0, 2)
        array.
    """
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    merged: list[list[int]] = []
    for start, end in sorted(pairs):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.array(merged, dtype=np.int64)


def _sweep_stats(a_merged: np.ndarray, b_merged: np.ndarray) -> tuple[int, int]:
    """
    Compute intersection and union sizes by combined boundary sweep.

    A boundary sweep over the concatenated start/end events of both merged
    sets (+1 at starts, -1 at ends, starts tie-breaking before ends at the
    same coordinate) yields the cumulative coverage between consecutive
    boundaries. Because each input set is merged (disjoint and
    non-adjacent), coverage never exceeds 2, so bases at coverage exactly
    2 are |A n B| and bases at coverage >= 1 are |A u B|.

    Args:
        a_merged: merged intervals for side A (from _merge_intervals).
        b_merged: merged intervals for side B.

    Returns:
        tuple[int, int]: bases with coverage exactly 2 (|A n B|) and bases
        with coverage >= 1 (|A u B|), as Python ints.
    """
    if a_merged.shape[0] == 0 and b_merged.shape[0] == 0:
        return 0, 0
    starts = np.concatenate([a_merged[:, 0], b_merged[:, 0]])
    ends = np.concatenate([a_merged[:, 1], b_merged[:, 1]])
    coords = np.concatenate([starts, ends])
    deltas = np.concatenate([
        np.ones(starts.size, dtype=np.int64),
        np.full(ends.size, -1, dtype=np.int64),
    ])
    # Primary sort key is the coordinate; the secondary key -delta puts +1
    # (start) before -1 (end) when boundaries coincide, so the union never
    # splits bookended intervals (load-bearing tie-break, pinned by the
    # bookended-adjacent-touching oracle case).
    order = np.lexsort((-deltas, coords))
    coords_sorted = coords[order]
    coverage = np.cumsum(deltas[order])
    widths = np.diff(coords_sorted)
    segment_coverage = coverage[:-1]
    intersection = int(widths[segment_coverage == 2].sum())
    union = int(widths[segment_coverage >= 1].sum())
    return intersection, union


def jaccard(path_a: str | Path, path_b: str | Path) -> dict[str, int | float]:
    """
    Compute bedtools-jaccard-equivalent statistics with pure numpy.

    Both files are parsed, merged per chromosome (exact string key: "Chr1"
    never matches "chr1"), swept per chromosome and summed across
    chromosomes; a chromosome present on only one side still contributes
    its bases to the union.

    Args:
        path_a: BED3+ file for side A.
        path_b: BED3+ file for side B.

    Returns:
        dict[str, int | float]: exactly the keys ``intersection`` and
        ``union`` (ints) and ``jaccard`` (intersection / union as a float,
        0.0 when the union is empty). The informational overlap-count key
        excluded by WPAR-01 is intentionally absent.
    """
    a_chroms: dict[str, list[tuple[int, int]]] = {}
    for row in parse_bed(path_a):
        a_chroms.setdefault(row.chrom, []).append((row.start, row.end))
    b_chroms: dict[str, list[tuple[int, int]]] = {}
    for row in parse_bed(path_b):
        b_chroms.setdefault(row.chrom, []).append((row.start, row.end))
    intersection = 0
    union = 0
    for chrom in sorted(set(a_chroms) | set(b_chroms)):
        i, u = _sweep_stats(
            _merge_intervals(a_chroms.get(chrom, [])),
            _merge_intervals(b_chroms.get(chrom, [])),
        )
        intersection += i
        union += u
    value = intersection / union if union else 0.0
    return {"intersection": intersection, "union": union, "jaccard": value}


# Null-B literals bedtools prints for the B side of a no-hit output line,
# per B column (pinned empirically against live bedtools v2.31.1; the
# official docs do not document the field-level encoding).
NULL_CHROM = "."  # B column 1 (chromosome) under a null join
NULL_START = "-1"  # B column 2 (start) under a null join
NULL_END = "-1"  # B column 3 (end) under a null join
NULL_NAME = "."  # B column 4 (BED name slot) under a null join
NULL_SCORE = "-1"  # B column 5 (BED score slot, numeric) under a null join
NULL_OTHER = "."  # B column 6+ (strand and beyond) under a null join


def _parses_int(value: str) -> bool:
    """
    Report whether a BED payload field parses as a base-10 integer.

    Mirrors the int-parseability bedtools applies to the BED "score" column
    before upcasting a 5-column record to BED6.

    Args:
        value: the literal field content.

    Returns:
        bool: True when int(value) would succeed.
    """
    try:
        int(value)
    except ValueError:
        return False
    return True


def _null_b_fields(count: int, numeric_score: bool | None = None) -> list[str]:
    """
    Build the null B-side field vector for a no-hit A row.

    Follows the oracle-pinned count-dependent encoding: a null BED6 prints
    (".", -1, -1, ".", "-1", ".") and bedtools emits its first ``count``
    columns for B counts <= 6. The score slot (B column 5) is
    content-dependent ONLY at count 5 ("-1" when column 5 parses as int,
    else "."); at count 6 it is ALWAYS "-1" (verified with numeric and
    non-numeric content alike) and at count >= 7 ALWAYS "." (likewise) —
    so the 6-column NER annotation surface is content-independent.

    Args:
        count: B column count (>= 3; an empty B file means 3).
        numeric_score: for count 5, whether B's column-5 values parse as
            int (derived from content by loj_intersect); ignored for
            every other count.

    Returns:
        list[str]: the literals bedtools prints for the B side of a no-hit
        output line.
    """
    if count <= 4:
        return [NULL_CHROM, NULL_START, NULL_END] + [NULL_OTHER] * (count - 3)
    if count == 5:
        slot = NULL_SCORE if numeric_score else NULL_OTHER
        return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME, slot]
    if count == 6:
        return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME, NULL_SCORE, NULL_OTHER]
    # count >= 7: beyond BED6 the score slot reverts to "." as well.
    return [NULL_CHROM, NULL_START, NULL_END, NULL_NAME] + [NULL_OTHER] * (count - 4)


def loj_intersect(path_a: str | Path, path_b: str | Path) -> list[str]:
    """
    Compute ``bedtools intersect -a A -b B -loj`` output with pure numpy.

    Every A row survives in A-file order; for each A row, B rows on the
    exact-same chromosome string with a half-open overlap (b.start < a.end
    AND a.start < b.end — touching boundaries do NOT overlap) are emitted
    one output line per pair, in B-file order; duplicates on either side
    each produce their own lines. This is a JOIN, not a merge: rows are
    never sorted and never deduplicated in this path. A row with no hit is
    emitted once with the B side null-filled per _null_b_fields.
    Per-chromosome candidate search broadcasts int64 arrays (consumer
    scale is spike-scale, tens of rows per side — correctness and emission
    order over asymptotics).

    Args:
        path_a: BED3+ file for side A (the preserved side of the join).
        path_b: BED3+ file for side B (hits or nulls).

    Returns:
        list[str]: tab-joined output lines (no trailing newline) — the
        documented downstream contract: the Phase-16 row adapter splits
        each line on tab.
    """
    a_rows = parse_bed(path_a)
    b_rows = parse_bed(path_b)
    b_count = len(b_rows[0].extra) + 3 if b_rows else 3
    numeric_score = all(_parses_int(row.extra[1]) for row in b_rows) if b_count == 5 else None
    null_fields = _null_b_fields(b_count, numeric_score)
    grouped: dict[str, list[BedRow]] = {}
    for row in b_rows:
        grouped.setdefault(row.chrom, []).append(row)
    starts_by_chrom = {
        chrom: np.array([row.start for row in rows], dtype=np.int64)
        for chrom, rows in grouped.items()
    }
    ends_by_chrom = {
        chrom: np.array([row.end for row in rows], dtype=np.int64)
        for chrom, rows in grouped.items()
    }
    out: list[str] = []
    for a in a_rows:
        prefix = (a.chrom, str(a.start), str(a.end), *a.extra)
        hits: list[BedRow] = []
        if a.chrom in grouped:
            mask = (starts_by_chrom[a.chrom] < a.end) & (ends_by_chrom[a.chrom] > a.start)
            hits = [grouped[a.chrom][int(i)] for i in np.nonzero(mask)[0]]
        if hits:
            for b in hits:
                out.append("\t".join((*prefix, b.chrom, str(b.start), str(b.end), *b.extra)))
        else:
            out.append("\t".join((*prefix, *null_fields)))
    return out
