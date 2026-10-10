#!/usr/bin/env python3
"""
Spike: zero-dependency numpy left-outer-join interval intersect parity vs bedtools.

Evidence-only artifact for quick task 261011-0a3 (windows-ledger id 19, the
NER half of the platform split). It proves — or refutes — that a pure numpy
left-outer-join reproduces `bedtools intersect -a <A> -b <B> -loj` exactly:
row-for-row, multiplicity-for-multiplicity, order-for-order and
null-for-null. This de-risks the v1.3 plan to give
example/notebooks/finetune_NER_task/generate_bpe_dataset.py (and the
matching notebook cell) a Windows numpy path while Linux keeps pybedtools.
This module provides:

- BED3+ parsing (arbitrary trailing columns preserved verbatim) grouped by
  exact chromosome-string key, with loud ValueError naming the offending
  file and line on malformed rows
- The loj join itself: every A row survives, one output line per A x B
  overlap pair (B hits emitted in B-file order), no-hit A rows null-filled
  per the oracle-pinned per-column map — a JOIN, never a merge or coalesce
- Live `bedtools intersect -loj` oracle invocation via a fixed-argv
  subprocess (plus a -sorted variant for the sort-requirement pin)
- Four empirical pin probes — null-B fill per B column count, output order
  including an unsorted A, the sorted-input requirement, and duplicate-row
  multiplicity — each live-asserted against the oracle on every run
- Eleven handcrafted edge cases and a seeded randomized property corpus,
  both gated on canonical row-set-with-multiplicity equality AND exact
  output-line-sequence equality

Imports are stdlib + numpy ONLY: the join arithmetic runs on per-chromosome
int64 start/end arrays and every additional BED column is a string payload
carried along by row index, so pandas would add import weight with no
arithmetic to do (same rationale as the 261010-wx2 jaccard spike). Unlike
that spike's merge semantics, nothing here ever merges or coalesces
intervals: loj is a join, the opposite operation.

Re-run from repo root (deterministic, fixed seed 20261011):
    .venv/bin/python \\
        .planning/quick/261011-0a3-spike-numpy-pandas-loj-intersect-parity-/spike_loj_parity.py

Exit code is 0 iff the verdict line reads VERDICT: PARITY; NO-PARITY is a
successful spike outcome, recorded faithfully and never fixed by loosening
gates.
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
from typing import NamedTuple

import numpy as np

# --- fixed spike constants (never derived from the environment) -----------
SEED_DEFAULT = 20261011
CASES_DEFAULT = 2000

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
FAILURES_DIR = TASK_DIR / "failures"
# Repo-relative display form keeps report lines free of absolute paths.
FAILURES_DISPLAY = os.path.relpath(TASK_DIR, REPO_ROOT).replace(os.sep, "/") + "/failures"

# Null-B fill literals for a no-hit A row, PINNED LIVE by pin_null_fill on
# every run (never guessed): bedtools pads the missing B feature with a null
# BED6 whose per-column print depends on the B column count — the BED "score"
# slot (B column 5) carries "-1" for B counts 5 (int-parseable score values
# only) and 6 (always), but "." for counts <= 4 and >= 7 — which is exactly
# what makes the NER consumer's `name == "-1"` branch fire on the 6-column
# annotation shape.
NULL_CHROM = "."  # B column 1 (chromosome) under a null join
NULL_START = "-1"  # B column 2 (start) under a null join
NULL_END = "-1"  # B column 3 (end) under a null join
NULL_NAME = "."  # B column 4 (BED name slot) under a null join
NULL_SCORE = "-1"  # B column 5 (BED score slot, numeric) under a null join
NULL_OTHER = "."  # B column 6+ (strand and beyond) under a null join

# Pin probe variants as (B column count, literal placed in B column 5 when
# the count reaches it). The NER surface is exactly 6 columns; 3 is the
# minimal BED form bedtools emits for an empty B file.
NULL_PROBE_VARIANTS = (
    (3, None),
    (4, None),
    (5, "0"),  # int-parseable score value
    (5, "exon"),  # non-numeric score value
    (6, "exon"),  # NER annotation shape (non-numeric feature in the score slot)
    (6, "0"),  # numeric score — proves count-6 content independence
    (7, "0"),
    (7, "exon"),
)


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
                    f"{path}: line {lineno}: invalid 0-based half-open interval "
                    f"({start}, {end})"
                )
            rows.append(BedRow(columns[0], start, end, columns[3:], lineno))
    return rows


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
    (".", -1, -1, ".", "-1", ".") and bedtools emits its first `count`
    columns for B counts <= 6. The score slot (B column 5) is
    content-dependent ONLY at count 5 ("-1" when column 5 parses as int,
    else "."); at count 6 it is ALWAYS "-1" (verified with numeric and
    non-numeric content alike) and at count >= 7 ALWAYS "." (likewise) —
    so the NER 6-column surface is content-independent. pin_null_fill
    live-asserts this table on every run.

    Args:
        count: B column count (>= 3; an empty B file means 3).
        numeric_score: for count 5, whether B's column-5 values parse as
            int (derived from content by loj_intersect_np); ignored for
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


def loj_intersect_np(path_a: str | Path, path_b: str | Path) -> list[str]:
    """
    Compute `bedtools intersect -a A -b B -loj` output with pure numpy.

    Every A row survives in A-file order; for each A row, B rows on the
    exact-same chromosome string with a half-open overlap (b.start < a.end
    AND a.start < b.end — touching boundaries do NOT overlap) are emitted
    one output line per pair, in B-file order; duplicates on either side
    each produce their own lines. A row with no hit is emitted once with
    the B side null-filled per _null_b_fields. Per-chromosome candidate
    search broadcasts int64 arrays (spike scale is <= 30 rows per side —
    correctness and emission order over asymptotics).

    Args:
        path_a: BED3+ file for side A (the preserved side of the join).
        path_b: BED3+ file for side B (hits or nulls).

    Returns:
        list[str]: output lines (no trailing newline), byte-identical in
        intent to the oracle's stdout lines.
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
                out.append(
                    "\t".join((*prefix, b.chrom, str(b.start), str(b.end), *b.extra))
                )
        else:
            out.append("\t".join((*prefix, *null_fields)))
    return out


def run_bedtools_loj(
    a: str | Path, b: str | Path, bedtools_bin: str, sorted_mode: bool = False
) -> str:
    """
    Run `bedtools intersect -loj` on two BED files and return raw stdout.

    Args:
        a: BED file passed as -a.
        b: BED file passed as -b.
        bedtools_bin: resolved bedtools executable.
        sorted_mode: append -sorted (chrom-sweep mode; demands sorted input).

    Returns:
        str: the raw stdout of the oracle.

    Raises:
        subprocess.CalledProcessError: when bedtools exits nonzero (for
            example -sorted fed unsorted input).
    """
    argv = [bedtools_bin, "intersect", "-a", str(a), "-b", str(b), "-loj"]
    if sorted_mode:
        argv.append("-sorted")
    proc = subprocess.run(argv, text=True, capture_output=True, check=True)
    return proc.stdout


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


def write_case_bed(path: str | Path, rows: list[tuple]) -> None:
    """
    Serialize one case side to a tab-separated BED3+ file.

    Args:
        path: destination file. Generated cases live inside a
            TemporaryDirectory, so they never land in the repo tree and
            their paths never enter report lines.
        rows: (chromosome, start, end, *payload) tuples in the exact write
            order; payload fields are emitted verbatim.
    """
    lines = ["\t".join(str(field) for field in row) + "\n" for row in rows]
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_sorted_bed(path: str | Path, rows: list[tuple]) -> None:
    """
    Serialize a case side in canonical bedtools-acceptable order.

    Args:
        path: destination file.
        rows: (chromosome, start, end, *payload) tuples in any order.

    Orders by (chromosome, start, end); Python's stable sort preserves
    generation order among exact duplicates.
    """
    ordered = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
    write_case_bed(path, ordered)


def _gates(np_lines: list[str], bt_stdout: str) -> tuple[bool, bool]:
    """
    Evaluate the LOCKED per-case gates: canonical row-set-with-multiplicity
    equality and exact output-line-sequence equality. Never loosened — a
    failure is recorded evidence.

    Args:
        np_lines: output lines from loj_intersect_np.
        bt_stdout: raw oracle stdout.

    Returns:
        tuple[bool, bool]: (canonical_ok, sequence_ok).
    """
    bt_lines = bt_stdout.splitlines()
    return sorted(np_lines) == sorted(bt_lines), np_lines == bt_lines


def _dump_failure(
    failures_root: Path,
    label: str,
    a_raw: Path,
    b_raw: Path,
    bt_out: str,
    np_lines: list[str],
    sorted_paths: list[tuple[Path, str]],
) -> str:
    """
    Copy one mismatching case pair plus BOTH outputs into failures/.

    Args:
        failures_root: reproduction-artifact directory for this run.
        label: stable case identifier (directory name case_<label>).
        a_raw: side A file exactly as written.
        b_raw: side B file exactly as written.
        bt_out: raw oracle stdout.
        np_lines: numpy output lines.
        sorted_paths: sorted copies created by the oracle protocol, if any.

    Returns:
        str: repo-relative display path of the reproduction directory.
    """
    repro_dir = failures_root / f"case_{label}"
    repro_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a_raw, repro_dir / "a.bed")
    shutil.copy2(b_raw, repro_dir / "b.bed")
    (repro_dir / "oracle_output.txt").write_text(bt_out, encoding="utf-8")
    (repro_dir / "numpy_output.txt").write_text(
        "".join(line + "\n" for line in np_lines), encoding="utf-8"
    )
    for src, name in sorted_paths:
        shutil.copy2(src, repro_dir / name)
    return f"{FAILURES_DISPLAY}/case_{label}"


def compare_case(  # noqa: PLR0912, PLR0913 — one explicit oracle protocol
    label: str,
    a_rows: list[tuple],
    b_rows: list[tuple],
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
    Compare numpy vs live bedtools -loj on one written case pair.

    Oracle sort-handling protocol: the numpy side always reads the files
    exactly as written. bedtools is probed with plain -loj on the raw files
    first; when it rejects the input (observed behavior recorded in
    `state`), it is re-invoked on sorted copies and later shuffled
    comparisons pre-sort directly. A raw probe that succeeds is the oracle,
    period — a raw-vs-sorted content difference would itself be evidence
    and is never papered over by retrying.

    Args:
        label: stable case identifier (temp filenames, failure dir name).
        a_rows: side A rows (chrom, start, end, *payload) in write order.
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
        oracle mode used, the numpy lines, per-gate flags and, on
        mismatch, the reproduction path and context.
    """
    a_raw = tmp_root / f"{label}_a.bed"
    b_raw = tmp_root / f"{label}_b.bed"
    write_case_bed(a_raw, a_rows)
    write_case_bed(b_raw, b_rows)
    np_lines = loj_intersect_np(a_raw, b_raw)
    sorted_paths: list[tuple[Path, str]] = []

    def sorted_oracle() -> str:
        a_or = tmp_root / f"{label}_a.sorted.bed"
        b_or = tmp_root / f"{label}_b.sorted.bed"
        write_sorted_bed(a_or, a_rows)
        write_sorted_bed(b_or, b_rows)
        sorted_paths.extend([(a_or, "a.sorted.bed"), (b_or, "b.sorted.bed")])
        return run_bedtools_loj(a_or, b_or, bedtools_bin)

    mode = "raw"
    if shuffled and state["presort_policy"]:
        bt_out = sorted_oracle()
        mode = "presorted(policy)"
    else:
        if shuffled:
            state["raw_probed"] = int(state["raw_probed"]) + 1
        try:
            bt_out = run_bedtools_loj(a_raw, b_raw, bedtools_bin)
        except subprocess.CalledProcessError as exc:
            note = _scrub(_first_line(exc.stderr), tmp_root)
            if drop_if_rejected:
                state["dropped_note"] = note
                return {
                    "outcome": "dropped",
                    "mode": "oracle-rejected",
                    "np_lines": np_lines,
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
            bt_out = sorted_oracle()
    canonical_ok, sequence_ok = _gates(np_lines, bt_out)
    if canonical_ok and sequence_ok:
        return {"outcome": "pass", "mode": mode, "np_lines": np_lines, "bt": bt_out}
    repro = _dump_failure(failures_root, label, a_raw, b_raw, bt_out, np_lines, sorted_paths)
    return {
        "outcome": "mismatch",
        "mode": mode,
        "canonical": canonical_ok,
        "sequence": sequence_ok,
        "np_lines": np_lines,
        "bt": bt_out,
        "repro": repro,
        "context": {
            "seed": seed,
            "pattern": pattern,
            "shuffled": shuffled,
            "n_a": len(a_rows),
            "n_b": len(b_rows),
        },
    }


# --- empirical pin probes ---------------------------------------------------


def _pin_null_fill_variant(
    bedtools_bin: str, a_path: Path, b_path: Path, count: int, numeric_score: bool | None
) -> tuple[bool, list[str]]:
    """
    Probe the oracle's no-hit B-side literals for one B variant.

    Args:
        bedtools_bin: resolved bedtools executable.
        a_path: 6-column NER-shaped A file whose rows hit nothing.
        b_path: B file with `count` columns, disjoint from A.
        count: B column count being probed.
        numeric_score: expected int-parseability of B's column 5 (the
            count-5 content dependence); None for other counts.

    Returns:
        tuple[bool, list[str]]: whether BOTH the observed literals match
        _null_b_fields(count, numeric_score) and loj_intersect_np
        reproduces the oracle byte-identically; and the observed literal
        vector.
    """
    bt_out = run_bedtools_loj(a_path, b_path, bedtools_bin)
    bt_lines = bt_out.splitlines()
    observed = bt_lines[0].split("\t")[6:] if bt_lines else []
    np_lines = loj_intersect_np(a_path, b_path)
    literal_ok = observed == _null_b_fields(count, numeric_score)
    emit_ok = np_lines == bt_lines
    return literal_ok and emit_ok, observed


def pin_null_fill(bedtools_bin: str) -> bool:
    """
    Pin the null-B fill encoding for every probed B column variant.

    Writes a 6-column NER-shaped A whose rows overlap nothing, then one B
    per variant in NULL_PROBE_VARIANTS (column counts 3-7, with numeric and
    non-numeric score values at count 5/6/7 to pin the content
    dependence); captures the oracle's raw no-hit rows, extracts the
    per-column literals, live-asserts them against the module constants and
    gates byte-identical numpy emission. An empty B file is probed as well
    (bedtools cannot infer a column count there). Also derives — from the
    pinned literals, not from memory — which branch of the NER consumer's
    `name == "-1"` check fires.

    Args:
        bedtools_bin: resolved bedtools executable.

    Returns:
        bool: True when every probed variant's literals and numpy emission
        match the oracle.
    """
    print("pin null-B fill (live oracle, no-hit A rows, per B column variant):")
    a_rows = [
        ("Chr1", 100, 200, "ATTG", "Os01g1", "+"),
        ("Chr2", 500, 600, "GGCCT", "Os02g1", "-"),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        a_path = tmp_root / "pin_a6.bed"
        write_case_bed(a_path, a_rows)
        for count, score_value in NULL_PROBE_VARIANTS:
            tail = ("bg", score_value if score_value is not None else "exon", "+", "x", "y")
            b_rows = [("Chr1", 1000, 1200, *tail[: count - 3])]
            b_path = tmp_root / f"pin_b{count}_{score_value or 'plain'}.bed"
            write_case_bed(b_path, b_rows)
            numeric = _parses_int(score_value) if count == 5 else None
            variant_ok, observed = _pin_null_fill_variant(
                bedtools_bin, a_path, b_path, count, numeric
            )
            ok = ok and variant_ok
            score_note = f" score col {score_value!r}" if count >= 5 else ""
            print(
                f"  B columns {count}{score_note}: literals {observed!r} "
                f"numpy-emit {'PASS' if variant_ok else 'FAIL'}"
            )
        empty_path = tmp_root / "pin_bempty.bed"
        empty_path.write_text("", encoding="utf-8")
        empty_ok, empty_observed = _pin_null_fill_variant(
            bedtools_bin, a_path, empty_path, 3, None
        )
        ok = ok and empty_ok
        print(
            f"  empty B file: {empty_observed!r} (column count unknowable — "
            f"oracle emits the 3-column null form) numpy-emit {'PASS' if empty_ok else 'FAIL'}"
        )
    print(
        "  pinned per-column-type map: chrom-col "
        f"{NULL_CHROM!r} | start-col {NULL_START!r} | end-col {NULL_END!r} | "
        f"name-col/B4 {NULL_NAME!r} | score-col/B5 {NULL_SCORE!r} or {NULL_OTHER!r} "
        f"(content-dependent at count 5 only) | strand-and-beyond {NULL_OTHER!r}"
    )
    print(
        "  pinned content rule: B count 5 score slot is '-1' iff column 5 parses "
        "as int, else '.'; B count 6 is ALWAYS '-1' and B count >= 7 ALWAYS '.' "
        "(both verified with numeric and non-numeric content) — the NER 6-column "
        "surface is content-independent"
    )
    consumer_value = _null_b_fields(6)[4]
    fires = consumer_value == "-1"
    print(
        "  NER shape (6-col B): iv.fields[10] == B column 5 == "
        f"{consumer_value!r} -> the consumer's `name == '-1'` intergenic branch "
        f"{'FIRES' if fires else 'DOES NOT FIRE'} under the pinned encoding"
    )
    return ok


def pin_output_order(bedtools_bin: str) -> bool:
    """
    Pin the plain -loj output-order rule, including an unsorted A.

    One deliberately unsorted A (known permutation, multi-hit rows, two
    chromosomes) against a sorted B: compares the oracle's raw line
    SEQUENCE against numpy's A-file-order emission, prints the pinned rule
    derived from that evidence, then re-runs the same case as sorted copies
    under -sorted and records whether the sequence differs.

    Args:
        bedtools_bin: resolved bedtools executable.

    Returns:
        bool: True when numpy matches the oracle sequence on the raw files
        (and on the sorted copies under -sorted).
    """
    print("pin output order (unsorted A across 2 chroms, multi-hit rows, sorted B):")
    a_rows = [
        ("Chr2", 900, 1000, "T1", "g1", "+"),
        ("Chr1", 100, 300, "T2", "g2", "-"),
        ("Chr1", 50, 150, "T3", "g3", "+"),
        ("Chr2", 100, 200, "T4", "g4", "-"),
        ("Chr1", 250, 350, "T5", "g5", "+"),
    ]
    b_rows = [
        ("Chr1", 0, 120, "bg1", "exon", "+"),
        ("Chr1", 110, 260, "bg2", "intron", "-"),
        ("Chr1", 255, 400, "bg3", "exon", "+"),
        ("Chr2", 80, 180, "bg4", "exon", "-"),
        ("Chr2", 850, 950, "bg5", "intron", "+"),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        a_raw = tmp_root / "pinord_a.bed"
        b_raw = tmp_root / "pinord_b.bed"
        write_case_bed(a_raw, a_rows)
        write_case_bed(b_raw, b_rows)
        np_lines = loj_intersect_np(a_raw, b_raw)
        bt_raw_lines = run_bedtools_loj(a_raw, b_raw, bedtools_bin).splitlines()
        raw_ok = np_lines == bt_raw_lines
        ok = ok and raw_ok
        print(
            f"  gate numpy-sequence vs plain -loj raw: {'PASS' if raw_ok else 'FAIL'} "
            f"({len(a_rows)} A rows, {len(bt_raw_lines)} output lines)"
        )
        if not raw_ok:
            for i, (want, got) in enumerate(zip(bt_raw_lines, np_lines)):
                if want != got:
                    print(f"    first divergence at line {i}: oracle {want!r} numpy {got!r}")
                    break
        a_sorted = tmp_root / "pinord_a.sorted.bed"
        b_sorted = tmp_root / "pinord_b.sorted.bed"
        write_sorted_bed(a_sorted, a_rows)
        write_sorted_bed(b_sorted, b_rows)
        bt_sorted_lines = run_bedtools_loj(
            a_sorted, b_sorted, bedtools_bin, sorted_mode=True
        ).splitlines()
        same = bt_sorted_lines == bt_raw_lines
        np_sorted = loj_intersect_np(a_sorted, b_sorted)
        sorted_ok = np_sorted == bt_sorted_lines
        ok = ok and sorted_ok
        print(
            f"  -sorted variant on sorted copies: sequence {'identical to' if same else 'DIFFERS from'} "
            f"plain -loj (same {len(bt_sorted_lines)} lines, {'same' if sorted(bt_sorted_lines) == sorted(bt_raw_lines) else 'DIFFERENT'} multiset)"
        )
        print(f"  gate numpy-sequence vs -sorted oracle on sorted copies: {'PASS' if sorted_ok else 'FAIL'}")
    if raw_ok:
        print(
            "  pinned rule: plain -loj emits output lines in A-file order; per A row "
            "the B hits appear in B-file order (verified on this probe: unsorted A, "
            "B-file order == start order here; unsorted-B order is exercised by the corpus)"
        )
    else:
        print("  FINDING: assumed A-file-order / B-file-order rule does NOT hold — see divergence above")
    return ok


def pin_sorted_requirement(bedtools_bin: str) -> None:
    """
    Record (gate nothing) the sorted-input requirement for both modes.

    Probes plain -loj on unsorted input (accept or reject, recorded);
    -loj -sorted on unsorted input (expect rejection — first stderr line
    captured, scrubbed); and -loj -sorted on sorted copies (accept,
    recorded). bedtools' own rejections are never gated on.

    Args:
        bedtools_bin: resolved bedtools executable.
    """
    print("pin sorted-input requirement (observed, gated on nothing per plan):")
    a_rows = [
        ("Chr2", 900, 1000, "T1", "g1", "+"),
        ("Chr1", 100, 300, "T2", "g2", "-"),
        ("Chr1", 50, 150, "T3", "g3", "+"),
    ]
    b_rows = [
        ("Chr1", 0, 120, "bg1", "exon", "+"),
        ("Chr1", 110, 260, "bg2", "intron", "-"),
        ("Chr2", 850, 950, "bg5", "intron", "+"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        a_raw = tmp_root / "pinsort_a.bed"
        b_raw = tmp_root / "pinsort_b.bed"
        write_case_bed(a_raw, a_rows)
        write_case_bed(b_raw, b_rows)
        try:
            run_bedtools_loj(a_raw, b_raw, bedtools_bin)
            print("  plain -loj on unsorted input: ACCEPTED (exit 0)")
        except subprocess.CalledProcessError:
            print("  plain -loj on unsorted input: REJECTED")
        try:
            run_bedtools_loj(a_raw, b_raw, bedtools_bin, sorted_mode=True)
            print("  -loj -sorted on unsorted input: ACCEPTED (unexpected)")
        except subprocess.CalledProcessError as exc:
            print(f"  -loj -sorted on unsorted input: REJECTED (first stderr: {_scrub(_first_line(exc.stderr), tmp_root)})")
        a_sorted = tmp_root / "pinsort_a.sorted.bed"
        b_sorted = tmp_root / "pinsort_b.sorted.bed"
        write_sorted_bed(a_sorted, a_rows)
        write_sorted_bed(b_sorted, b_rows)
        try:
            run_bedtools_loj(a_sorted, b_sorted, bedtools_bin, sorted_mode=True)
            print("  -loj -sorted on sorted copies: ACCEPTED (exit 0)")
        except subprocess.CalledProcessError as exc:
            print(f"  -loj -sorted on sorted copies: REJECTED ({_scrub(_first_line(exc.stderr), tmp_root)})")
    print(
        "  finding: plain -loj needs no sorted input (the NER consumer's unsorted "
        "tokens_bed is legal); -sorted demands sorted input"
    )


def pin_multiplicity(bedtools_bin: str) -> bool:
    """
    Pin duplicate-row multiplicity: a join, never a merge.

    One A row overlapping 3 distinct B rows plus an exact-duplicate B row
    must yield 4 lines for that A row; the A row itself is duplicated, so
    8 lines total. Both canonical and sequence gates are enforced against
    the oracle, and the structural counts are asserted on the oracle's own
    output (proving bedtools multiplies rather than coalesces).

    Args:
        bedtools_bin: resolved bedtools executable.

    Returns:
        bool: True when both gates pass and the structural counts hold.
    """
    print("pin multiplicity (1 A row x 3 distinct B rows + 1 exact-duplicate B row, A row duplicated):")
    a_row = ("Chr1", 100, 300, "T1", "g1", "+")
    a_rows = [a_row, a_row]
    b_rows = [
        ("Chr1", 50, 150, "b1", "exon", "+"),
        ("Chr1", 120, 220, "b2", "intron", "-"),
        ("Chr1", 250, 350, "b3", "exon", "+"),
        ("Chr1", 250, 350, "b3", "exon", "+"),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        a_raw = tmp_root / "pinmult_a.bed"
        b_raw = tmp_root / "pinmult_b.bed"
        write_case_bed(a_raw, a_rows)
        write_case_bed(b_raw, b_rows)
        np_lines = loj_intersect_np(a_raw, b_raw)
        bt_out = run_bedtools_loj(a_raw, b_raw, bedtools_bin)
        bt_lines = bt_out.splitlines()
        canonical_ok, sequence_ok = _gates(np_lines, bt_out)
        prefix = "\t".join(str(field) for field in a_row)
        hit_lines = [line for line in bt_lines if line.startswith(prefix + "\t")]
        distinct_b = {tuple(line.split("\t")[6:]) for line in hit_lines}
        # Both A rows are byte-identical, so lines cannot be attributed per row;
        # the join arithmetic is asserted instead: 2 A copies x 4 B hits.
        copies_a, hits_per_a = 2, 4
        total_ok = (
            len(bt_lines) == copies_a * hits_per_a
            and len(hit_lines) == copies_a * hits_per_a
            and len(distinct_b) == 3
        )
        ok = ok and canonical_ok and sequence_ok and total_ok
        print(
            f"  oracle lines total {len(bt_lines)} == {copies_a} duplicate-A copies x "
            f"{hits_per_a} B hits each (3 distinct B rows + 1 exact-duplicate copy)"
        )
        print(
            f"  gate canonical {'PASS' if canonical_ok else 'FAIL'} | "
            f"gate sequence {'PASS' if sequence_ok else 'FAIL'} | "
            f"structural counts {'PASS' if total_ok else 'FAIL'}"
        )
    print("  pinned: exact-duplicate rows on either side join per copy — never merged, never coalesced")
    return ok


def run_pins(bedtools_bin: str) -> bool:
    """
    Run all four empirical pin probes and print the pins verdict.

    Args:
        bedtools_bin: resolved bedtools executable.

    Returns:
        bool: True when every gated pin passes (the sorted-requirement
        probe records only and never gates).
    """
    print("--- empirical pins ---")
    ok_null = pin_null_fill(bedtools_bin)
    ok_order = pin_output_order(bedtools_bin)
    pin_sorted_requirement(bedtools_bin)
    ok_mult = pin_multiplicity(bedtools_bin)
    ok = ok_null and ok_order and ok_mult
    print(f"pins: {'PASS' if ok else 'FAIL'} (sorted-requirement probe records only, gated on nothing)")
    return ok


# --- handcrafted edge cases ---------------------------------------------------


def handcrafted_cases() -> list[dict[str, object]]:
    """
    Build the eleven deterministic handcrafted edge cases (NER 6+6 shape).

    Covers: (1) empty A (drop-if-rejected protocol); (2) empty B — every A
    row null-filled; (3) all-overlap; (4) no-overlap (null-representation
    driver); (5) strict containment of A in B; (6) multiplicity (one A row
    hitting several B rows); (7) boundary touching — A.end==B.start and
    B.end==A.start (zero-width overlap: no hit); (8) identical coordinates
    under different chromosome names (must not join — exact string key);
    (9) exact duplicate rows in A AND in B (join, not merge); (10)
    deliberately unsorted A with a B multi-hit (order pin driver); (11)
    NER-shaped realistic case — 6+6 columns with token/gene/strand payloads
    drawn from small fixed pools, A in per-gene iteration order.

    Returns:
        list[dict[str, object]]: one dict per case with name, a, b,
        shuffled and drop_if_rejected keys.
    """
    return [
        {
            "name": "empty-A",
            "a": [],
            "b": [("Chr1", 1000, 1200, "Os01g2", "exon", "+"), ("Chr2", 50, 90, "Os02g3", "intron", "-")],
            "shuffled": False,
            "drop_if_rejected": True,
        },
        {
            "name": "empty-B",
            "a": [
                ("Chr1", 100, 200, "ATTG", "Os01g1", "+"),
                ("Chr1", 300, 450, "GGCCT", "Os01g1", "+"),
                ("Chr2", 500, 600, "T", "Os02g4", "-"),
            ],
            "b": [],
            "shuffled": False,
            "drop_if_rejected": True,
        },
        {
            "name": "all-overlap",
            "a": [
                ("Chr1", 100, 300, "ATT", "g1", "+"),
                ("Chr1", 200, 400, "GGC", "g1", "-"),
                ("Chr1", 250, 350, "T", "g2", "+"),
            ],
            "b": [
                ("Chr1", 50, 450, "g1", "exon", "+"),
                ("Chr1", 120, 280, "g2", "intron", "-"),
                ("Chr1", 260, 500, "g1", "exon", "+"),
            ],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "no-overlap",
            "a": [
                ("Chr1", 100, 200, "ATTG", "g1", "+"),
                ("Chr1", 300, 400, "GGCCT", "g2", "-"),
            ],
            "b": [
                ("Chr1", 1000, 1200, "g3", "exon", "+"),
                ("Chr1", 2000, 2100, "g4", "intron", "-"),
            ],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "strict-containment-A-in-B",
            "a": [("Chr1", 200, 800, "ATTG", "g1", "+"), ("Chr1", 2100, 2200, "GGCCT", "g2", "-")],
            "b": [("Chr1", 100, 1000, "g1", "exon", "+"), ("Chr1", 2000, 3000, "g2", "intron", "-")],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "multiplicity-one-A-multi-B",
            "a": [("Chr1", 100, 300, "ATTG", "g1", "+")],
            "b": [
                ("Chr1", 50, 150, "gA", "exon", "+"),
                ("Chr1", 120, 220, "gB", "intron", "-"),
                ("Chr1", 250, 350, "gC", "exon", "+"),
            ],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "boundary-touching-no-hit",
            "a": [("Chr1", 100, 200, "AA", "g1", "+"), ("Chr1", 400, 500, "CC", "g1", "-")],
            "b": [("Chr1", 200, 300, "b1", "exon", "+"), ("Chr1", 300, 400, "b2", "intron", "-")],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "same-coords-different-chrom-names",
            "a": [("Chr1", 100, 200, "ATTG", "g1", "+")],
            "b": [("chr1", 100, 200, "g1", "exon", "+")],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "duplicate-rows-both-sides",
            "a": [("Chr1", 100, 300, "T1", "g1", "+"), ("Chr1", 100, 300, "T1", "g1", "+")],
            "b": [
                ("Chr1", 50, 150, "b1", "exon", "+"),
                ("Chr1", 120, 220, "b2", "intron", "-"),
                ("Chr1", 250, 350, "b3", "exon", "+"),
                ("Chr1", 250, 350, "b3", "exon", "+"),
            ],
            "shuffled": False,
            "drop_if_rejected": False,
        },
        {
            "name": "unsorted-A-B-multi-hit",
            "a": [
                ("Chr2", 900, 1000, "T1", "g1", "+"),
                ("Chr1", 100, 300, "T2", "g2", "-"),
                ("Chr1", 50, 150, "T3", "g3", "+"),
                ("Chr2", 100, 200, "T4", "g4", "-"),
                ("Chr1", 250, 350, "T5", "g5", "+"),
            ],
            "b": [
                ("Chr1", 0, 120, "bg1", "exon", "+"),
                ("Chr1", 110, 260, "bg2", "intron", "-"),
                ("Chr1", 255, 400, "bg3", "exon", "+"),
                ("Chr2", 80, 180, "bg4", "exon", "-"),
                ("Chr2", 850, 950, "bg5", "intron", "+"),
            ],
            "shuffled": True,
            "drop_if_rejected": False,
        },
        {
            "name": "ner-realistic-per-gene-iteration-order",
            "a": [
                ("Chr1", 1200, 1210, "AT", "Os01g0100100", "+"),
                ("Chr1", 1210, 1215, "G", "Os01g0100100", "+"),
                ("Chr1", 900, 912, "CCGGATTA", "Os01g0202200", "-"),
                ("Chr2", 500, 530, "GGCCT", "Os02g0100300", "+"),
                ("Chr2", 530, 545, "TTGA", "Os02g0100300", "+"),
                ("Chr1", 1150, 1160, "C", "Os01g0100100", "+"),
                ("Chr2", 700, 760, "ATGCA", "Os02g0116600", "-"),
                ("Chr1", 2000, 2050, "AATT", "Os01g0202200", "-"),
            ],
            "b": [
                ("Chr1", 800, 1000, "Os01g0202200", "intron", "-"),
                ("Chr1", 1000, 1300, "Os01g0100100", "exon", "+"),
                ("Chr1", 1350, 1500, "Os01g0100100", "intron", "+"),
                ("Chr1", 1900, 2100, "Os01g0202200", "exon", "-"),
                ("Chr2", 400, 550, "Os02g0100300", "exon", "+"),
                ("Chr2", 600, 620, "Os02g0100300", "intron", "+"),
                ("Chr2", 690, 900, "Os02g0116600", "exon", "-"),
            ],
            "shuffled": True,
            "drop_if_rejected": False,
        },
    ]


# --- randomized property corpus ------------------------------------------------


WINDOW = 8000
CHROM_POOL = ("Chr1", "Chr2", "Chr3", "Chr4", "chr1", "1", "scaffoldA")
PATTERNS = ("mixed", "dense", "containment", "adjacency", "disjoint", "duplicates")
TOKEN_POOL = ("A", "CG", "ATT", "AA", "GGCCT", "T", "CCGGATTA", "GCA", "AT", "ATGCA", "TTGA", "C")
GENE_POOL = ("Os01g0100100", "Os01g0202200", "Os02g0100300", "Os03g0116600", "gene7")
FEATURE_POOL = ("exon", "intron")
STRAND_POOL = ("+", "-")


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
    """Exact duplicate rows and staggered self-overlaps on BOTH sides.

    The join must multiply duplicates on either side, never coalesce them.
    """
    def with_duplicates(base: list[tuple[int, int]]) -> list[tuple[int, int]]:
        rows = list(base)
        for start, end in base:
            roll = rng.random()
            if roll < 0.35:
                rows.append((start, end))  # exact duplicate row
            elif roll < 0.70:
                shift = int(rng.integers(1, 16))
                if rng.random() < 0.5 and start > 0:
                    shift = -min(shift, start)  # keep the staggered start >= 0
                rows.append((start + shift, end + shift))  # staggered overlap
        return rows

    base_a = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    base_b = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    return with_duplicates(base_a), with_duplicates(base_b)


_GENERATORS = {
    "mixed": _gen_mixed,
    "dense": _gen_dense,
    "containment": _gen_containment,
    "adjacency": _gen_adjacency,
    "disjoint": _gen_disjoint,
    "duplicates": _gen_duplicates,
}


def _pick(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    """
    Draw one payload string from a fixed pool.

    Args:
        rng: seeded generator.
        pool: deterministic pool of payload literals.

    Returns:
        str: the drawn literal.
    """
    return pool[int(rng.integers(0, len(pool)))]


def gen_random_case(
    rng: np.random.Generator, index: int
) -> tuple[list[tuple], list[tuple], str, bool]:
    """
    Generate one seeded randomized NER-shaped (6+6 column) case.

    Chromosomes come from a fixed pool mixing naming styles (Chr1..Chr4,
    chr1, 1, scaffoldA) to keep proving exact-string grouping; the pattern
    cycles deterministically; a fixed quarter of cases (index % 4 == 3) is
    emitted in shuffled non-coordinate order while the rest are written
    canonically sorted. A rows carry (token_str, gene, strand) payloads and
    B rows (gene, feature_name, strand), drawn from small fixed pools so
    field-for-field string comparison is meaningful. Side totals are capped
    at 30 rows and both sides are never simultaneously empty (the empty
    sides live in handcrafted cases 1-2).

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
    a_rows: list[tuple] = []
    b_rows: list[tuple] = []
    for chrom in chroms:
        side_a, side_b = generator(rng, cap)
        a_rows.extend(
            (chrom, start, end, _pick(rng, TOKEN_POOL), _pick(rng, GENE_POOL), _pick(rng, STRAND_POOL))
            for start, end in side_a
        )
        b_rows.extend(
            (chrom, start, end, _pick(rng, GENE_POOL), _pick(rng, FEATURE_POOL), _pick(rng, STRAND_POOL))
            for start, end in side_b
        )
    if not a_rows and not b_rows:
        b_rows.append((chroms[0], 0, 10, _pick(rng, GENE_POOL), "exon", "+"))
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


def _fresh_state() -> dict[str, object]:
    """
    Build the harness bookkeeping state for one run.

    Returns:
        dict[str, object]: sort-policy flags and counters consumed by
        compare_case and the report's sort-handling finding.
    """
    return {
        "presort_policy": False,
        "raw_probed": 0,
        "raw_errored": 0,
        "nonshuffled_rejections": 0,
        "rejection_note": None,
        "dropped_note": None,
    }


def run_corpus(bedtools_bin: str, cases: int, seed: int) -> dict[str, object]:
    """
    Run the handcrafted edge cases and the seeded randomized corpus.

    Args:
        bedtools_bin: resolved bedtools executable.
        cases: number of randomized cases (0 = handcrafted only).
        seed: corpus seed (np.random.default_rng).

    Returns:
        dict[str, object]: per-gate comparison counts, mismatch records
        and the sort-handling state, all consumed by the report and
        verdict.
    """
    failures_root = FAILURES_DIR
    # Artifacts must reflect THIS run exactly, so any previous run's
    # failures directory is removed first.
    shutil.rmtree(failures_root, ignore_errors=True)
    state = _fresh_state()
    records: list[dict[str, object]] = []

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
        if result["outcome"] == "dropped":
            hand_dropped += 1
            print(
                f"case {number} {case['name']}: DROPPED from gating — bedtools rejected "
                f"the input (observed: {state['dropped_note']}); numpy side emits "
                f"{len(result['np_lines'])} line(s)"
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
                    **{key: result[key] for key in ("mode", "canonical", "sequence", "repro", "context")},
                }
            )
        canonical_ok = bool(result.get("canonical", True))
        sequence_ok = bool(result.get("sequence", True))
        suffix = "" if ok else f" MISMATCH reproduction {result['repro']}"
        print(
            f"case {number} {case['name']}: mode={result['mode']} lines "
            f"{len(result['np_lines'])} canonical {'PASS' if canonical_ok else 'FAIL'} "
            f"sequence {'PASS' if sequence_ok else 'FAIL'}{suffix}"
        )

    if cases <= 0:
        print("randomized harness: skipped (--cases 0)")
        return {
            "handcrafted_compared": hand_compared,
            "handcrafted_pass": hand_pass,
            "handcrafted_dropped": hand_dropped,
            "randomized_compared": 0,
            "randomized_pass": 0,
            "canonical_pass": 0,
            "sequence_pass": 0,
            "mismatch_count": len(records),
            "records": records,
            "state": state,
        }

    print("--- randomized property cases ---")
    print(
        f"seed {seed} | cases {cases} | pattern cycle {'/'.join(PATTERNS)} | "
        "shuffled every 4th case | per-side rows <= 30 | NER shape 6+6 columns"
    )
    rng = np.random.default_rng(seed)
    rand_pass = 0
    canonical_pass = 0
    sequence_pass = 0
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
        if result["outcome"] == "pass":
            rand_pass += 1
            canonical_pass += 1
            sequence_pass += 1
        elif result["outcome"] == "mismatch":
            if result["canonical"]:
                canonical_pass += 1
            if result["sequence"]:
                sequence_pass += 1
            records.append({"kind": "randomized", **result})
    mismatches = cases - rand_pass
    print(f"randomized: compared {cases} | pass {rand_pass} | mismatch {mismatches}")
    print(f"GATE-canonical (sorted full-row equality incl. multiplicity): {canonical_pass}/{cases} pass")
    print(f"GATE-sequence (exact output line sequence): {sequence_pass}/{cases} pass")
    shuffled_total = sum(1 for i in range(cases) if i % 4 == 3)
    print(
        f"shuffled cases: {shuffled_total} | raw probed {state['raw_probed']} | "
        f"raw rejected {state['raw_errored']} | "
        f"pre-sorted by policy {shuffled_total - int(state['raw_probed'])}"
    )
    if state["rejection_note"] is not None:
        print(
            "sort handling: bedtools rejected a raw probed file (observed: "
            f"{state['rejection_note']}); after the first rejection the oracle ran on "
            "pre-sorted copies for shuffled comparisons; the numpy side always reads "
            "files exactly as written"
        )
    else:
        print(
            "sort handling: plain -loj accepted every probed raw file (unsorted A and "
            "unsorted B included — see pins); the numpy side always reads files "
            "exactly as written"
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
            print(
                f"  mismatch case {record['label']}: seed {context['seed']} pattern "
                f"{context['pattern']} shuffled {context['shuffled']} "
                f"|A| {context['n_a']} |B| {context['n_b']} gates canonical="
                f"{'PASS' if record['canonical'] else 'FAIL'} sequence="
                f"{'PASS' if record['sequence'] else 'FAIL'} reproduction "
                f"{record['repro']}"
            )
        else:
            print(
                f"  mismatch {record['name']}: mode={record['mode']} gates canonical="
                f"{'PASS' if record['canonical'] else 'FAIL'} sequence="
                f"{'PASS' if record['sequence'] else 'FAIL'} reproduction {record['repro']}"
            )
    return {
        "handcrafted_compared": hand_compared,
        "handcrafted_pass": hand_pass,
        "handcrafted_dropped": hand_dropped,
        "randomized_compared": cases,
        "randomized_pass": rand_pass,
        "canonical_pass": canonical_pass,
        "sequence_pass": sequence_pass,
        "mismatch_count": len(records),
        "records": records,
        "state": state,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (argparse, not click: the artifact runs outside the package)."""
    parser = argparse.ArgumentParser(
        description=(
            "Spike: numpy loj intersect parity vs bedtools "
            "(quick-261011-0a3, windows-ledger id 19 NER half)."
        )
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=CASES_DEFAULT,
        help=f"number of randomized property cases (default {CASES_DEFAULT}; 0 = pins + handcrafted only)",
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
    print("spike: numpy loj intersect parity vs bedtools (quick-261011-0a3)")
    print("=" * 80)
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | seed {args.seed} | cases {args.cases}")
    bedtools_bin = resolve_bedtools(args.bedtools)
    source = "override" if args.bedtools is not None else "PATH"
    version = subprocess.run(
        [bedtools_bin, "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    print(f"bedtools binary: {bedtools_bin} (source: {source})")
    print(f"bedtools version: {version}")
    pins_ok = run_pins(bedtools_bin)
    corpus = run_corpus(bedtools_bin, args.cases, args.seed)
    corpus_ok = (
        corpus["handcrafted_pass"] == corpus["handcrafted_compared"]
        and corpus["mismatch_count"] == 0
    )
    elapsed = time.perf_counter() - started
    print(f"RUNTIME: {elapsed:.1f}s")
    parity = pins_ok and corpus_ok
    if parity:
        print("VERDICT: PARITY")
    else:
        print("VERDICT: NO-PARITY")
        print(f"  reason: pins {'PASS' if pins_ok else 'FAIL'}")
        print(
            f"  reason: handcrafted {corpus['handcrafted_pass']}/"
            f"{corpus['handcrafted_compared']} pass "
            f"({corpus['handcrafted_dropped']} dropped); randomized "
            f"{corpus['randomized_pass']}/{corpus['randomized_compared']} pass; "
            f"GATE-canonical {corpus['canonical_pass']}/{corpus['randomized_compared']} | "
            f"GATE-sequence {corpus['sequence_pass']}/{corpus['randomized_compared']}; "
            f"{corpus['mismatch_count']} mismatch(es) — see the reproduction "
            "lines above"
        )
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
