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
import tempfile
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


# --- property harness (task 2) ---------------------------------------------
WINDOW = 8000
CHROM_POOL = ("Chr1", "Chr2", "Chr3", "Chr4", "chr1", "1", "scaffoldA")
PATTERNS = ("mixed", "dense", "containment", "adjacency", "disjoint", "duplicates")
FAILURES_DIR = TASK_DIR / "failures"
FAILURES_DISPLAY = os.path.relpath(TASK_DIR, REPO_ROOT).replace(os.sep, "/") + "/failures"


def _first_line(text: str | None) -> str:
    """
    Extract the first non-empty line of subprocess stderr.

    Args:
        text: captured stderr (may be None or empty).

    Returns:
        str: the first non-empty line, stripped; a placeholder when absent.
    """
    if not text:
        return "<no stderr>"
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "<empty stderr>"


def _scrub(text: str, tmp_root: Path) -> str:
    """
    Replace the ephemeral temp-directory prefix in captured oracle text.

    Args:
        text: subprocess output that may embed temp file paths.
        tmp_root: the TemporaryDirectory root to redact.

    Returns:
        str: text with every occurrence of the temp root replaced by
        "<tmpdir>", keeping report lines byte-stable across re-runs.
    """
    return text.replace(str(tmp_root), "<tmpdir>")


def write_case_bed(path: str | Path, intervals: list[tuple[str, int, int]]) -> None:
    """
    Serialize one case side to a 3-column BED file.

    Args:
        path: destination file. Generated cases live inside a
            TemporaryDirectory, so they never land in the repo tree and
            their paths never enter report lines.
        intervals: (chromosome, start, end) rows in the exact write order.
    """
    lines = [f"{chrom}\t{start}\t{end}\n" for chrom, start, end in intervals]
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_sorted_bed(path: str | Path, intervals: list[tuple[str, int, int]]) -> None:
    """
    Serialize a case side in canonical bedtools-acceptable order.

    Args:
        path: destination file.
        intervals: (chromosome, start, end) rows in any order.

    Orders by chromosome (lexicographic), then start, then end — the sort
    bedtools demands of its inputs.
    """
    ordered = sorted(intervals, key=lambda row: (row[0], row[1], row[2]))
    write_case_bed(path, ordered)


def _case_gates(
    np_stats: dict[str, int | float], bt_stats: dict[str, int | float]
) -> tuple[bool, int, int, float, float]:
    """
    Evaluate the LOCKED per-case gates: exact integer equality on
    intersection and union, and jaccard within JACCARD_TOL of the live
    full-precision value. Never loosened — a failure is recorded evidence.

    Args:
        np_stats: statistics from jaccard_np.
        bt_stats: statistics from run_bedtools.

    Returns:
        tuple[bool, int, int, float, float]: overall pass, |delta
        intersection|, |delta union|, |delta jaccard| vs live full
        precision, |delta jaccard| vs bedtools' 6-dp print.
    """
    d_int = abs(int(np_stats["intersection"]) - int(bt_stats["intersection"]))
    d_uni = abs(int(np_stats["union"]) - int(bt_stats["union"]))
    d_jac = abs(float(np_stats["jaccard"]) - float(bt_stats["jaccard_full"]))
    d_print = abs(float(np_stats["jaccard"]) - float(bt_stats["jaccard"]))
    ok = d_int == 0 and d_uni == 0 and d_jac <= JACCARD_TOL
    return ok, d_int, d_uni, d_jac, d_print


def compare_case(  # noqa: PLR0912, PLR0913, PLR0915 — one explicit oracle protocol
    label: str,
    a_rows: list[tuple[str, int, int]],
    b_rows: list[tuple[str, int, int]],
    bedtools_bin: str,
    tmp_root: Path,
    failures_root: Path,
    state: dict[str, object],
    *,
    shuffled: bool,
    drop_if_rejected: bool = False,
    seed: int | None = None,
    pattern: str | None = None,
) -> dict[str, object]:
    """
    Compare numpy vs live bedtools jaccard on one written case pair.

    Oracle sort-handling protocol: the numpy side reads the files exactly as
    written (it always sorts internally). bedtools is probed on the raw
    files first; when it rejects unsorted input or reports values that
    disagree with numpy, it is re-invoked on sorted copies, the behavior is
    recorded in `state`, and later shuffled comparisons use pre-sorted
    input directly.

    Args:
        label: stable case identifier (temp filenames, failure dir name).
        a_rows: side A rows in write order.
        b_rows: side B rows in write order.
        bedtools_bin: resolved bedtools executable.
        tmp_root: the case's TemporaryDirectory root.
        failures_root: reproduction-artifact directory for mismatches.
        state: mutable harness bookkeeping (sort-policy flags, counters).
        shuffled: whether the written rows deliberately violate sortedness.
        drop_if_rejected: drop the case from gating when bedtools rejects
            the input (observed behavior recorded instead).
        seed: seed of the generating rng (for mismatch records).
        pattern: generating pattern name (for mismatch records).

    Returns:
        dict[str, object]: outcome ("pass" | "mismatch" | "dropped"), the
        oracle mode used, both statistic dicts, gate deltas and, on
        mismatch, the reproduction path and context.
    """
    a_raw = tmp_root / f"{label}_a.bed"
    b_raw = tmp_root / f"{label}_b.bed"
    write_case_bed(a_raw, a_rows)
    write_case_bed(b_raw, b_rows)
    np_stats = jaccard_np(a_raw, b_raw)
    sorted_copies: list[tuple[Path, str]] = []

    def sorted_oracle() -> dict[str, int | float]:
        a_or = tmp_root / f"{label}_a.sorted.bed"
        b_or = tmp_root / f"{label}_b.sorted.bed"
        write_sorted_bed(a_or, a_rows)
        write_sorted_bed(b_or, b_rows)
        sorted_copies.extend([(a_or, "a.sorted.bed"), (b_or, "b.sorted.bed")])
        return run_bedtools(a_or, b_or, bedtools_bin)

    mode = "raw"
    if shuffled and state["presort_policy"]:
        bt_stats = sorted_oracle()
        mode = "presorted(policy)"
    else:
        if shuffled:
            state["raw_probed"] = int(state["raw_probed"]) + 1
        try:
            bt_stats = run_bedtools(a_raw, b_raw, bedtools_bin)
        except subprocess.CalledProcessError as exc:
            note = _scrub(_first_line(exc.stderr), tmp_root)
            if drop_if_rejected:
                state["dropped_note"] = note
                return {
                    "outcome": "dropped",
                    "mode": "oracle-rejected",
                    "np": np_stats,
                    "bt": None,
                }
            state["presort_policy"] = True
            if shuffled:
                state["raw_errored"] = int(state["raw_errored"]) + 1
                if state["rejection_note"] is None:
                    state["rejection_note"] = note
                mode = "sorted-after-raw-error"
            else:
                state["nonshuffled_rejections"] = int(state["nonshuffled_rejections"]) + 1
                mode = "sorted-after-unexpected-rejection"
            bt_stats = sorted_oracle()
        else:
            ok_raw, _, _, _, _ = _case_gates(np_stats, bt_stats)
            if not ok_raw and shuffled:
                bt_sorted = sorted_oracle()
                ok_sorted, _, _, _, _ = _case_gates(np_stats, bt_sorted)
                state["presort_policy"] = True
                if ok_sorted:
                    state["raw_differed"] = int(state["raw_differed"]) + 1
                    mode = "sorted-after-raw-difference"
                else:
                    mode = "sorted(raw-also-mismatched)"
                bt_stats = bt_sorted
    ok, d_int, d_uni, d_jac, d_print = _case_gates(np_stats, bt_stats)
    if ok:
        return {
            "outcome": "pass",
            "mode": mode,
            "np": np_stats,
            "bt": bt_stats,
            "deltas": (d_int, d_uni, d_jac, d_print),
        }
    repro_dir = failures_root / f"case_{label}"
    repro_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a_raw, repro_dir / "a.bed")
    shutil.copy2(b_raw, repro_dir / "b.bed")
    for src, name in sorted_copies:
        shutil.copy2(src, repro_dir / name)
    return {
        "outcome": "mismatch",
        "mode": mode,
        "np": np_stats,
        "bt": bt_stats,
        "deltas": (d_int, d_uni, d_jac, d_print),
        "repro": f"{FAILURES_DISPLAY}/case_{label}",
        "context": {
            "seed": seed,
            "pattern": pattern,
            "shuffled": shuffled,
            "n_a": len(a_rows),
            "n_b": len(b_rows),
        },
    }


def handcrafted_cases() -> list[dict[str, object]]:
    """
    Build the eight deterministic handcrafted edge cases.

    Covers: (1) one side an empty file; (2) identical interval sets
    (jaccard 1.0); (3) strict containment of A within B; (4) bookended
    intervals touching at one boundary (half-open: zero intersection,
    coalesced union); (5) fully disjoint sides; (6) identical coordinates
    under different chromosome names (must not intersect — exact string
    key); (7) duplicate and staggered-overlap rows within one side (merge
    semantics); (8) deliberately shuffled row order.

    Returns:
        list[dict[str, object]]: one dict per case with name, a, b,
        shuffled and drop_if_rejected keys.
    """
    identical = [("Chr1", 100, 250), ("Chr1", 400, 600), ("Chr2", 50, 150)]
    return [
        {
            "name": "empty-side-A",
            "a": [],
            "b": [("Chr1", 1000, 2000), ("Chr1", 3000, 3500)],
            "shuffled": False,
            "drop_if_rejected": True,
        },
        {
            "name": "identical-sets",
            "a": list(identical),
            "b": list(identical),
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "strict-containment-A-in-B",
            "a": [("Chr1", 200, 800), ("Chr1", 2100, 2200)],
            "b": [("Chr1", 100, 1000), ("Chr1", 2000, 3000)],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "bookended-adjacent-touching",
            "a": [("Chr1", 0, 100)],
            "b": [("Chr1", 100, 200)],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "fully-disjoint",
            "a": [("Chr1", 0, 100)],
            "b": [("Chr1", 500, 600)],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "same-coords-different-chrom-names",
            "a": [("Chr1", 100, 200)],
            "b": [("chr1", 100, 200)],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "duplicates-and-staggered-overlaps",
            "a": [("Chr1", 100, 300), ("Chr1", 100, 300), ("Chr1", 200, 400), ("Chr1", 205, 210)],
            "b": [("Chr1", 150, 350)],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "shuffled-row-order",
            "a": [("Chr1", 500, 700), ("Chr1", 100, 300), ("Chr2", 50, 120), ("Chr1", 650, 900)],
            "b": [("Chr2", 100, 150), ("Chr1", 200, 600)],
            "shuffled": True,
            "drop_if_rejected": False,
        },
    ]


def _draw_intervals(
    rng: np.random.Generator,
    count: int,
    lo: int,
    hi: int,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]]:
    """
    Draw `count` random intervals inside a bounded window.

    Args:
        rng: seeded generator (deterministic draw order).
        count: number of intervals (0 draws nothing).
        lo: inclusive lower bound for starts.
        hi: upper bound for interval ends.
        min_len: minimum interval length.
        max_len: maximum interval length; callers keep max_len well below
            hi - lo so the start range never empties.

    Returns:
        list[tuple[int, int]]: (start, end) intervals, unsorted.
    """
    out: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, hi - lo - length))
        out.append((lo + start, lo + start + length))
    return out


def _gen_mixed(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """Plain independent draws on both sides (may leave a side empty)."""
    a = _draw_intervals(rng, int(rng.integers(0, cap)), 0, WINDOW, 20, 1200)
    b = _draw_intervals(rng, int(rng.integers(0, cap)), 0, WINDOW, 20, 1200)
    return a, b


def _gen_dense(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """Long intervals in a shared window: dense within- and cross-side overlap."""
    a = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 60, 900)
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 60, 900)
    return a, b


def _gen_containment(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """Strictly nested A intervals inside non-overlapping long B intervals."""
    n_b = int(rng.integers(2, max(3, cap + 1)))
    b: list[tuple[int, int]] = []
    pos = 0
    for _ in range(n_b):
        gap = int(rng.integers(10, 80))
        length = int(rng.integers(200, 500))
        b.append((pos + gap, pos + gap + length))
        pos += gap + length
    picks = rng.choice(len(b), size=int(rng.integers(1, len(b) + 1)), replace=False)
    a: list[tuple[int, int]] = []
    for idx in picks:
        start, end = b[int(idx)]
        pad_left = int(rng.integers(0, 40))
        pad_right = int(rng.integers(0, 40))
        if pad_left == 0 and pad_right == 0:
            pad_right = 1  # keep containment strict on at least one boundary
        a.append((start + pad_left, end - pad_right))
    return a, b


def _gen_adjacency(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """Tiled segments cycled A A B: sides touch at boundaries and A keeps bookended pairs."""
    n_cuts = int(rng.integers(3, 11))
    cuts = sorted({int(x) for x in rng.integers(1, WINDOW, size=n_cuts)})
    bounds = [0, *cuts, WINDOW]
    a: list[tuple[int, int]] = []
    b: list[tuple[int, int]] = []
    for j in range(len(bounds) - 1):
        if rng.random() < 0.15:
            continue  # segment owned by neither side (a gap)
        segment = (bounds[j], bounds[j + 1])
        (a if j % 3 != 2 else b).append(segment)
    return a, b


def _gen_disjoint(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """A confined below the split point, B above it: zero cross-side intersection."""
    mid = int(rng.integers(WINDOW // 4, 3 * WINDOW // 4))
    a = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, mid, 30, 300)
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), mid, WINDOW, 30, 300)
    return a, b


def _gen_duplicates(rng: np.random.Generator, cap: int) -> tuple[list, list]:
    """Side A carries exact duplicate rows and staggered self-overlaps."""
    base = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    a: list[tuple[int, int]] = list(base)
    for start, end in base:
        roll = rng.random()
        if roll < 0.35:
            a.append((start, end))  # exact duplicate row
        elif roll < 0.70:
            shift = int(rng.integers(1, 16))
            if rng.random() < 0.5 and start > 0:
                shift = -min(shift, start)  # keep the staggered start >= 0
            a.append((start + shift, end + shift))  # staggered overlap
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 80, 500)
    return a, b


_GENERATORS = {
    "mixed": _gen_mixed,
    "dense": _gen_dense,
    "containment": _gen_containment,
    "adjacency": _gen_adjacency,
    "disjoint": _gen_disjoint,
    "duplicates": _gen_duplicates,
}


def gen_random_case(
    rng: np.random.Generator, index: int
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]], str, bool]:
    """
    Generate one seeded randomized case.

    Chromosomes come from a fixed pool mixing naming styles (Chr1..Chr4,
    chr1, 1, scaffoldA) to prove grouping is by exact string key; the
    pattern cycles deterministically; a fixed quarter of cases (index % 4
    == 3) is emitted in shuffled non-coordinate order while the rest are
    written canonically sorted. Side totals are capped at 30 rows by
    deterministic truncation and both sides are never simultaneously
    empty (the empty side lives in handcrafted case 1).

    Args:
        rng: the corpus-wide seeded generator.
        index: 0-based case index driving pattern and shuffle selection.

    Returns:
        tuple: (a_rows, b_rows, pattern, shuffled).
    """
    pattern = PATTERNS[index % len(PATTERNS)]
    n_chroms = int(rng.integers(1, 5))
    picked = rng.choice(len(CHROM_POOL), size=n_chroms, replace=False)
    chroms = [CHROM_POOL[int(i)] for i in picked]
    cap = max(2, 30 // n_chroms)
    generator = _GENERATORS[pattern]
    a_rows: list[tuple[str, int, int]] = []
    b_rows: list[tuple[str, int, int]] = []
    for chrom in chroms:
        side_a, side_b = generator(rng, cap)
        a_rows.extend((chrom, start, end) for start, end in side_a)
        b_rows.extend((chrom, start, end) for start, end in side_b)
    if not a_rows and not b_rows:
        b_rows.append((chroms[0], 0, 10))
    a_rows = a_rows[:30]
    b_rows = b_rows[:30]
    shuffled = index % 4 == 3
    if shuffled:
        order_a = rng.permutation(len(a_rows))
        order_b = rng.permutation(len(b_rows))
        a_rows = [a_rows[int(i)] for i in order_a]
        b_rows = [b_rows[int(i)] for i in order_b]
    else:
        a_rows = sorted(a_rows, key=lambda row: (row[0], row[1], row[2]))
        b_rows = sorted(b_rows, key=lambda row: (row[0], row[1], row[2]))
    return a_rows, b_rows, pattern, shuffled


def run_corpus(bedtools_bin: str, cases: int, seed: int) -> dict[str, object]:
    """
    Run the handcrafted edge cases and the seeded randomized corpus.

    Args:
        bedtools_bin: resolved bedtools executable.
        cases: number of randomized cases.
        seed: corpus seed (np.random.default_rng).

    Returns:
        dict[str, object]: comparison counts, mismatch records, maxima and
        the sort-handling state, all consumed by the report and verdict.
    """
    failures_root = FAILURES_DIR
    # Artifacts must reflect THIS run exactly, so any previous run's
    # failures directory is removed first.
    shutil.rmtree(failures_root, ignore_errors=True)
    state: dict[str, object] = {
        "presort_policy": False,
        "raw_probed": 0,
        "raw_errored": 0,
        "raw_differed": 0,
        "nonshuffled_rejections": 0,
        "rejection_note": None,
        "dropped_note": None,
    }
    records: list[dict[str, object]] = []
    maxima = {"jac": 0.0, "jac_print": 0.0, "inter": 0, "union": 0}

    def account(result: dict[str, object]) -> None:
        if result["outcome"] != "pass":
            return
        d_int, d_uni, d_jac, d_print = result["deltas"]  # type: ignore[misc]
        maxima["jac"] = max(maxima["jac"], d_jac)  # type: ignore[typeddict-item]
        maxima["jac_print"] = max(maxima["jac_print"], d_print)  # type: ignore[typeddict-item]
        maxima["inter"] = max(maxima["inter"], d_int)  # type: ignore[typeddict-item]
        maxima["union"] = max(maxima["union"], d_uni)  # type: ignore[typeddict-item]

    print("--- handcrafted edge cases ---")
    hand_compared = 0
    hand_pass = 0
    hand_dropped = 0
    for number, case in enumerate(handcrafted_cases(), start=1):
        label = f"hand{number}"
        with tempfile.TemporaryDirectory() as tmp:
            result = compare_case(
                label,
                case["a"],  # type: ignore[arg-type]
                case["b"],  # type: ignore[arg-type]
                bedtools_bin,
                Path(tmp),
                failures_root,
                state,
                shuffled=case["shuffled"],  # type: ignore[arg-type]
                drop_if_rejected=case["drop_if_rejected"],  # type: ignore[arg-type]
                seed=seed,
                pattern="handcrafted",
            )
        account(result)
        if result["outcome"] == "dropped":
            hand_dropped += 1
            np_stats = result["np"]  # type: ignore[index]
            print(
                f"case {number} {case['name']}: DROPPED from gating — bedtools rejected "
                f"the input (observed: {state['dropped_note']}); numpy side reports "
                f"inter {np_stats['intersection']} union {np_stats['union']} "
                f"jaccard {np_stats['jaccard']!r}"
            )
            continue
        hand_compared += 1
        ok = result["outcome"] == "pass"
        if ok:
            hand_pass += 1
        else:
            records.append(
                {
                    "kind": "handcrafted",
                    "label": label,
                    "name": case["name"],
                    **{key: result[key] for key in ("mode", "deltas", "repro", "context")},
                }
            )
        np_stats = result["np"]  # type: ignore[index]
        bt_stats = result["bt"]  # type: ignore[index]
        suffix = "PASS" if ok else f"MISMATCH reproduction {result['repro']}"
        print(
            f"case {number} {case['name']}: mode={result['mode']} inter "
            f"{np_stats['intersection']}/{bt_stats['intersection']} union "
            f"{np_stats['union']}/{bt_stats['union']} jaccard "
            f"{np_stats['jaccard']!r}/{bt_stats['jaccard_full']!r} {suffix}"
        )

    print("--- randomized property cases ---")
    print(
        f"seed {seed} | cases {cases} | pattern cycle {'/'.join(PATTERNS)} | "
        "shuffled every 4th case | per-side rows <= 30"
    )
    rng = np.random.default_rng(seed)
    rand_pass = 0
    for index in range(cases):
        a_rows, b_rows, pattern, shuffled = gen_random_case(rng, index)
        with tempfile.TemporaryDirectory() as tmp:
            result = compare_case(
                str(index),
                a_rows,
                b_rows,
                bedtools_bin,
                Path(tmp),
                failures_root,
                state,
                shuffled=shuffled,
                seed=seed,
                pattern=pattern,
            )
        account(result)
        if result["outcome"] == "pass":
            rand_pass += 1
        else:
            records.append({"kind": "randomized", **result})
    mismatches = cases - rand_pass
    print(f"randomized: compared {cases} | pass {rand_pass} | mismatch {mismatches}")
    print(f"max |delta jaccard| vs live full precision: {maxima['jac']!r}")
    print(
        f"max |delta jaccard| vs bedtools 6-dp print (informational): "
        f"{maxima['jac_print']!r}"
    )
    print(
        f"max |delta intersection|: {maxima['inter']} bases | "
        f"max |delta union|: {maxima['union']} bases"
    )
    shuffled_total = sum(1 for i in range(cases) if i % 4 == 3)
    print(
        f"shuffled cases: {shuffled_total} | raw probed {state['raw_probed']} | "
        f"raw rejected {state['raw_errored']} | raw differed {state['raw_differed']} | "
        f"pre-sorted by policy {shuffled_total - int(state['raw_probed'])}"
    )
    if state["rejection_note"] is not None:
        print(
            "sort handling: bedtools jaccard rejected non-lexicographically-sorted "
            f"input (observed: {state['rejection_note']}); after the first rejection "
            "the oracle ran on pre-sorted copies for shuffled comparisons; the numpy "
            "side always sorts internally"
        )
    elif int(state["raw_differed"]) > 0:
        print(
            "sort handling: bedtools accepted raw shuffled input but reported values "
            "differing from the sorted oracle; sorted copies used thereafter; the "
            "numpy side always sorts internally"
        )
    else:
        print(
            "sort handling: bedtools accepted every probed raw file; the numpy side "
            "always sorts internally"
        )
    if int(state["nonshuffled_rejections"]) > 0:
        print(
            f"ANOMALY: bedtools rejected {state['nonshuffled_rejections']} "
            "canonically-sorted (non-shuffled) inputs; sorted copies were used"
        )
    print(f"mismatches: {'none' if not records else len(records)}")
    for record in records:
        if record["kind"] == "randomized":
            context = record["context"]  # type: ignore[index]
            deltas = record["deltas"]  # type: ignore[index]
            print(
                f"  mismatch case {record['label']}: seed {context['seed']} pattern "
                f"{context['pattern']} shuffled {context['shuffled']} "
                f"|A| {context['n_a']} |B| {context['n_b']} d_int {deltas[0]} "
                f"d_union {deltas[1]} d_jaccard {deltas[2]!r} reproduction "
                f"{record['repro']}"
            )
        else:
            deltas = record["deltas"]  # type: ignore[index]
            print(
                f"  mismatch {record['name']}: d_int {deltas[0]} d_union {deltas[1]} "
                f"d_jaccard {deltas[2]!r} reproduction {record['repro']}"
            )
    return {
        "handcrafted_compared": hand_compared,
        "handcrafted_pass": hand_pass,
        "handcrafted_dropped": hand_dropped,
        "randomized_compared": cases,
        "randomized_pass": rand_pass,
        "mismatch_count": len(records),
        "records": records,
        "maxima": maxima,
        "state": state,
    }


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
    corpus_ok = True
    corpus: dict[str, object] | None = None
    if args.frozen_only:
        print("randomized harness: skipped (--frozen-only)")
    else:
        corpus = run_corpus(bedtools_bin, args.cases, args.seed)
        corpus_ok = (
            corpus["mismatch_count"] == 0
            and corpus["handcrafted_pass"] == corpus["handcrafted_compared"]
        )
    elapsed = time.perf_counter() - started
    print(f"RUNTIME: {elapsed:.1f}s")
    parity = frozen_ok and corpus_ok
    if parity:
        print("VERDICT: PARITY")
    else:
        print("VERDICT: NO-PARITY")
        print(f"  reason: frozen gates {'PASS' if frozen_ok else 'FAIL'}")
        if corpus is not None:
            print(
                f"  reason: handcrafted {corpus['handcrafted_pass']}/"
                f"{corpus['handcrafted_compared']} pass "
                f"({corpus['handcrafted_dropped']} dropped); randomized "
                f"{corpus['randomized_pass']}/{corpus['randomized_compared']} pass; "
                f"{corpus['mismatch_count']} mismatch(es) — see the reproduction "
                "lines above"
            )
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
