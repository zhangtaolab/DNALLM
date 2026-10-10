#!/usr/bin/env python3
"""
Spike: zero-dependency numpy interval-jaccard parity against bedtools.

Evidence-only artifact for quick task 261010-wx2 (windows-ledger id 19). It
proves — or refutes — that pure numpy interval arithmetic reproduces
`bedtools jaccard` exactly, de-risking the v1.3 plan to replace the CRE
showcase's `bedtools jaccard` subprocess (and later the NER example's
pybedtools loj intersect) with in-process numpy evaluation along the lines
of dnallm/utils/genomic_coords.py. This module provides:

- BED3 parsing grouped by exact chromosome-string key, with loud ValueError
  naming the offending file and line on malformed rows
- Per-chromosome interval merging with bedtools merge semantics (overlaps
  AND bookended adjacency coalesce: next start <= current end)
- A combined boundary sweep computing |A n B| and |A u B| from coverage
  levels, plus an informational n_intersections analogue
- Live `bedtools jaccard` oracle invocation via a fixed-argv subprocess
- The frozen-pair parity check against the recorded reference
  (intersection 15872, union 48878, jaccard 0.324727, n_intersections 51)
- A seeded randomized + handcrafted-edge-case property harness

Imports are stdlib + numpy ONLY: per-chromosome int64 boundary arrays carry
the entire computation, so pandas would add import weight with no work to
do. bedtools prints jaccard at 6 decimal places, so the live full-precision
jaccard is reconstructed as intersection / union from the live integers —
the exact rational bedtools rounded for display. The numpy side always
sorts internally; bedtools is probed with raw files first and re-invoked on
sorted copies when it rejects unsorted input.

Re-run from repo root (deterministic, fixed seed 20261010):
    .venv/bin/python \\
        .planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/spike_jaccard_parity.py

Exit code is 0 iff the verdict line reads VERDICT: PARITY.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --- fixed spike constants (never derived from the environment) -----------
SEED_DEFAULT = 20261010
CASES_DEFAULT = 3000
JACCARD_TOL = 1e-9

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
FROZEN_PRED = REPO_ROOT / "example/notebooks/plant_helixseek_shared/.scratch/pred.bed"
FROZEN_TRUTH = REPO_ROOT / "example/notebooks/plant_helixseek_shared/.scratch/truth.bed"
# Repo-relative display forms keep report lines free of absolute paths.
FROZEN_PRED_DISPLAY = "example/notebooks/plant_helixseek_shared/.scratch/pred.bed"
FROZEN_TRUTH_DISPLAY = "example/notebooks/plant_helixseek_shared/.scratch/truth.bed"

# Orchestrator-captured ground truth (bedtools v2.31.1 on this box); the
# jaccard anchor is bedtools' 6-dp print of the full-precision value.
REFERENCE: dict[str, int | float] = {
    "intersection": 15872,
    "union": 48878,
    "jaccard": 0.324727,
    "n_intersections": 51,
}


def parse_bed3(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """
    Parse a 3-column BED file into per-chromosome interval lists.

    Grouping is by exact chromosome-string key ("Chr1" and "chr1" are
    different chromosomes), preserving file order within each chromosome;
    rows are never sorted here. Blank lines and '#' comment lines are
    skipped; anything else must parse.

    Args:
        path: BED file with at least 3 tab-separated columns (chromosome,
            0-based start, half-open end).

    Returns:
        dict[str, list[tuple[int, int]]]: chromosome -> intervals in file
        order.

    Raises:
        ValueError: naming the offending file and line number when a row
            has fewer than 3 columns, non-integer coordinates, a negative
            start, or an empty (start >= end) interval.
    """
    per_chrom: dict[str, list[tuple[int, int]]] = {}
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
                    f"{path}: line {lineno}: invalid 0-based half-open interval "
                    f"({start}, {end})"
                )
            per_chrom.setdefault(columns[0], []).append((start, end))
    return per_chrom


def merge_intervals(pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Sort and coalesce intervals with bedtools merge semantics.

    Intervals are sorted by (start, end), then merged while the next start
    is <= the current end: overlapping AND bookended (adjacent) intervals
    coalesce, matching `bedtools merge` at its default distance 0.

    Args:
        pairs: intervals as (start, end) tuples; need not be sorted.

    Returns:
        np.ndarray: shape (k, 2) int64 merged intervals, sorted, pairwise
        disjoint and non-adjacent.
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


def _count_overlapping(a_merged: np.ndarray, b_merged: np.ndarray) -> int:
    """
    Count merged-A intervals that overlap any merged-B interval.

    Two-pointer sweep valid because both inputs are sorted and internally
    disjoint: a b-interval ending at or before the current a-start can
    never overlap any later a-interval either.

    Args:
        a_merged: merged intervals for side A (from merge_intervals).
        b_merged: merged intervals for side B.

    Returns:
        int: number of merged-A intervals with at least one half-open
        overlap against merged-B.
    """
    count = 0
    j = 0
    n_b = b_merged.shape[0]
    for start, end in a_merged:
        while j < n_b and b_merged[j, 1] <= start:
            j += 1
        if j < n_b and b_merged[j, 0] < end:
            count += 1
    return count


def sweep_stats(a_merged: np.ndarray, b_merged: np.ndarray) -> tuple[int, int, int]:
    """
    Compute intersection, union and an n_intersections analogue by sweep.

    A combined boundary sweep over the concatenated start/end events of
    both merged sets (+1 at starts, -1 at ends, starts tie-breaking before
    ends at the same coordinate) yields the cumulative coverage between
    consecutive boundaries. Because each input set is merged (disjoint and
    non-adjacent), coverage never exceeds 2, so bases at coverage exactly 2
    are |A n B| and bases at coverage >= 1 are |A u B|.

    Args:
        a_merged: merged intervals for side A (from merge_intervals).
        b_merged: merged intervals for side B.

    Returns:
        tuple[int, int, int]: bases with coverage exactly 2 (|A n B|),
        bases with coverage >= 1 (|A u B|), and the number of merged-A
        intervals overlapping any merged-B interval. The third value is an
        n_intersections analogue kept INFORMATIONAL ONLY: bedtools' exact
        definition is unspecified and it is never gated on.
    """
    if a_merged.shape[0] == 0 and b_merged.shape[0] == 0:
        return 0, 0, 0
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
    return intersection, union, _count_overlapping(a_merged, b_merged)


def jaccard_np(path_a: str | Path, path_b: str | Path) -> dict[str, int | float]:
    """
    Compute bedtools-jaccard-equivalent statistics with pure numpy.

    Both files are parsed, merged per chromosome (exact string key), swept
    per chromosome and summed across chromosomes; a chromosome present on
    only one side still contributes its bases to the union.

    Args:
        path_a: BED file for side A.
        path_b: BED file for side B.

    Returns:
        dict[str, int | float]: intersection and union as ints, jaccard as
        intersection / union (0.0 when the union is empty), and the
        informational n_intersections analogue.
    """
    a_chroms = parse_bed3(path_a)
    b_chroms = parse_bed3(path_b)
    intersection = 0
    union = 0
    n_analogue = 0
    for chrom in sorted(set(a_chroms) | set(b_chroms)):
        i, u, n = sweep_stats(
            merge_intervals(a_chroms.get(chrom, [])),
            merge_intervals(b_chroms.get(chrom, [])),
        )
        intersection += i
        union += u
        n_analogue += n
    jaccard = intersection / union if union else 0.0
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
        "n_intersections": n_analogue,
    }


def run_bedtools(a: str | Path, b: str | Path, bedtools_bin: str) -> dict[str, int | float]:
    """
    Run `bedtools jaccard` on two BED files and parse its report.

    Args:
        a: BED file passed as -a.
        b: BED file passed as -b.
        bedtools_bin: resolved bedtools executable.

    Returns:
        dict[str, int | float]: intersection and union as ints; jaccard as
        parsed from bedtools' 6-dp print; jaccard_full reconstructed at full
        precision as intersection / union (the exact rational bedtools
        rounded for display; 0.0 when the union is empty); n_intersections
        as int.

    Raises:
        subprocess.CalledProcessError: when bedtools exits nonzero (for
            example on input that is not sorted lexicographically).
        ValueError: when the output has no data line or a malformed one.
    """
    argv = [bedtools_bin, "jaccard", "-a", str(a), "-b", str(b)]
    proc = subprocess.run(argv, text=True, capture_output=True, check=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("bedtools jaccard emitted no data line")
    fields = lines[1].split("\t")
    if len(fields) != 4:
        raise ValueError(f"unexpected bedtools jaccard data line: {lines[1]!r}")
    intersection, union = int(fields[0]), int(fields[1])
    jaccard_full = intersection / union if union else 0.0
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": float(fields[2]),
        "jaccard_full": jaccard_full,
        "n_intersections": int(fields[3]),
    }


def resolve_bedtools(override: str | None = None) -> str:
    """
    Resolve the bedtools executable, honoring an explicit override.

    Args:
        override: explicit path from --bedtools; validated to exist and be
            executable. When None, resolved via shutil.which.

    Returns:
        str: the bedtools executable path.

    Raises:
        ValueError: when the override is not an executable file, or when no
            bedtools is found on PATH (pass --bedtools explicitly).
    """
    if override is not None:
        if not (Path(override).is_file() and os.access(override, os.X_OK)):
            raise ValueError(f"--bedtools override {override!r} is not an executable file.")
        return override
    resolved = shutil.which("bedtools")
    if resolved is None:
        raise ValueError("bedtools not found on PATH; pass --bedtools /path/to/bedtools.")
    return resolved


def _chrom_sequence(path: str | Path) -> list[str]:
    """
    Read the raw chromosome column in file order, one entry per data row.

    Args:
        path: BED file readable by parse_bed3.

    Returns:
        list[str]: chromosome names in file order (blank/comment rows
        skipped), for contiguity and sortedness analysis.
    """
    sequence: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            sequence.append(line.split("\t")[0])
    return sequence


def analyze_sortedness(path: str | Path) -> dict[str, object]:
    """
    Verify (not assume) the sortedness properties of a BED file.

    Args:
        path: BED file readable by parse_bed3.

    Returns:
        dict[str, object]: rows (int), chroms (first-appearance order),
        counts (per-chromosome row counts, same order), chrom_grouped
        (each chromosome appears in one contiguous run) and
        start_sorted_within_chrom (file-order starts non-decreasing per
        chromosome).
    """
    per_chrom = parse_bed3(path)
    sequence = _chrom_sequence(path)
    transitions = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])
    chrom_grouped = (transitions + 1 if sequence else 0) == len(per_chrom)
    start_sorted = all(
        pairs[i][0] <= pairs[i + 1][0] for pairs in per_chrom.values() for i in range(len(pairs) - 1)
    )
    return {
        "rows": len(sequence),
        "chroms": list(per_chrom.keys()),
        "counts": [len(pairs) for pairs in per_chrom.values()],
        "chrom_grouped": chrom_grouped,
        "start_sorted_within_chrom": start_sorted,
    }


def _print_sortedness(label: str, path: str | Path, display: str) -> None:
    """Print the sortedness finding line for one frozen BED file."""
    info = analyze_sortedness(path)
    layout = " ".join(f"{chrom}={count}" for chrom, count in zip(info["chroms"], info["counts"]))  # type: ignore[arg-type]
    print(
        f"{label}: rows={info['rows']} chroms={len(info['chroms'])} [{layout}] "
        f"chrom-grouped={info['chrom_grouped']} "
        f"start-sorted-within-chrom={info['start_sorted_within_chrom']} ({display})"
    )


def frozen_pair_report(bedtools_bin: str) -> bool:
    """
    Run the frozen-pair parity check (numpy vs live bedtools vs reference).

    Reads the two frozen BEDs read-only, reports sortedness findings for
    both, computes numpy statistics, invokes live bedtools on the pair, and
    gates: intersection and union exact integer equality vs live; numpy
    jaccard within JACCARD_TOL of the live full-precision jaccard. The
    recorded 6-dp reference is an informational anchor, never a gate.

    Args:
        bedtools_bin: resolved bedtools executable.

    Returns:
        bool: True when all frozen gates pass.
    """
    print("--- frozen pair ---")
    _print_sortedness("pred", FROZEN_PRED, FROZEN_PRED_DISPLAY)
    _print_sortedness("truth", FROZEN_TRUTH, FROZEN_TRUTH_DISPLAY)
    np_stats = jaccard_np(FROZEN_PRED, FROZEN_TRUTH)
    bt_stats = run_bedtools(FROZEN_PRED, FROZEN_TRUTH, bedtools_bin)
    print(
        f"{'metric':<16}{'numpy':<22}{'bedtools(live)':<22}{'recorded-ref':<22}"
        "note"
    )
    print(
        f"{'intersection':<16}{np_stats['intersection']!s:<22}"
        f"{bt_stats['intersection']!s:<22}{REFERENCE['intersection']!s:<22}"
    )
    print(
        f"{'union':<16}{np_stats['union']!s:<22}{bt_stats['union']!s:<22}"
        f"{REFERENCE['union']!s:<22}"
    )
    print(
        f"{'jaccard':<16}{np_stats['jaccard']!r:<22}{bt_stats['jaccard']!r:<22}"
        f"{REFERENCE['jaccard']!r:<22}bedtools prints 6 dp"
    )
    print(
        f"{'n_intersections':<16}{np_stats['n_intersections']!s:<22}"
        f"{bt_stats['n_intersections']!s:<22}{REFERENCE['n_intersections']!s:<22}"
        "informational, never gated"
    )
    gate_intersection = np_stats["intersection"] == bt_stats["intersection"]
    gate_union = np_stats["union"] == bt_stats["union"]
    delta_live = abs(np_stats["jaccard"] - bt_stats["jaccard_full"])  # type: ignore[arg-type]
    delta_anchor = abs(np_stats["jaccard"] - float(REFERENCE["jaccard"]))  # type: ignore[arg-type]
    gate_jaccard = delta_live <= JACCARD_TOL
    print(
        f"gate intersection-exact-vs-live: "
        f"{'PASS' if gate_intersection else 'FAIL'} "
        f"(delta {abs(np_stats['intersection'] - bt_stats['intersection'])})"  # type: ignore[arg-type]
    )
    print(
        f"gate union-exact-vs-live: {'PASS' if gate_union else 'FAIL'} "
        f"(delta {abs(np_stats['union'] - bt_stats['union'])})"  # type: ignore[arg-type]
    )
    print(
        f"gate jaccard-delta<= {JACCARD_TOL:g} vs live-full-precision: "
        f"{'PASS' if gate_jaccard else 'FAIL'} (achieved delta {delta_live!r} "
        f"vs live full {bt_stats['jaccard_full']!r})"
    )
    print(
        f"informational delta vs 6-dp anchor {REFERENCE['jaccard']!r}: "
        f"{delta_anchor!r} (bedtools print rounding)"
    )
    ok = gate_intersection and gate_union and gate_jaccard
    print(f"frozen gates: {'PASS' if ok else 'FAIL'}")
    return ok


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (argparse, not click: the artifact runs outside the package)."""
    parser = argparse.ArgumentParser(
        description=(
            "Spike: numpy interval-jaccard parity vs bedtools "
            "(quick-261010-wx2, windows-ledger id 19)."
        )
    )
    parser.add_argument(
        "--frozen-only",
        action="store_true",
        help="run only the frozen-pair parity check (skip the property corpus)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=CASES_DEFAULT,
        help=f"number of randomized property cases (default {CASES_DEFAULT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help=f"numpy default_rng seed (default {SEED_DEFAULT})",
    )
    parser.add_argument(
        "--bedtools",
        default=None,
        help="explicit bedtools binary (default: shutil.which lookup)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run the spike and print the report to stdout.

    Args:
        argv: command-line arguments (defaults to sys.argv[1:]).

    Returns:
        int: exit code — 0 iff the verdict line is VERDICT: PARITY.
    """
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    print("=" * 80)
    print("spike: numpy interval-jaccard parity vs bedtools (quick-261010-wx2)")
    print("=" * 80)
    print(
        f"python {sys.version.split()[0]} | numpy {np.__version__} | seed {args.seed} | "
        f"cases {args.cases} | jaccard tolerance {JACCARD_TOL:g}"
    )
    bedtools_bin = resolve_bedtools(args.bedtools)
    source = "override" if args.bedtools is not None else "PATH"
    version = subprocess.run(
        [bedtools_bin, "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    print(f"bedtools binary: {bedtools_bin} (source: {source})")
    print(f"bedtools version: {version}")
    frozen_ok = frozen_pair_report(bedtools_bin)
    if args.frozen_only:
        print("randomized harness: skipped (--frozen-only)")
    else:
        # The seeded randomized + handcrafted harness arrives with task 2.
        print("randomized harness: pending (arrives with task 2)")
    elapsed = time.perf_counter() - started
    print(f"RUNTIME: {elapsed:.1f}s")
    parity = frozen_ok
    print(f"VERDICT: {'PARITY' if parity else 'NO-PARITY'}")
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
