"""Tests for dnallm.utils.genomic_intervals — bedtools-parity interval evaluators.

Oracle literals are baked verbatim from the two parity spikes (quick tasks
261010-wx2 jaccard and 261011-0a3 loj-intersect), which compared 5008 cases
against live bedtools v2.31.1 with zero mismatches. No bedtools binary, no
subprocess, no skip machinery and no custom markers appear anywhere in this
file: the tests are platform-neutral by construction (the point of the
module under test).

Corpus note (AS-2): the randomized corpora regenerate from the ported
seeded generators (seeds 20261010 / 20261011). Property invariants are the
primary layer (no baked per-case oracle); one full 12-cycle subset per
spike (every pattern x shuffle state) carries fully baked outputs so any
numpy Generator stream divergence on another numpy version fails loudly
on the first CI run instead of diffing hundreds of cases. Fallback on a
CI-leg divergence: drop the baked subset to invariants only.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from dnallm.utils.genomic_intervals import jaccard, loj_intersect, parse_bed


def _write_bed(tmp_path: Path, name: str, rows: list[tuple]) -> Path:
    """Write tab-joined BED rows to a utf-8, LF-terminated tmp_path file."""
    lines = ["\t".join(str(field) for field in row) for row in rows]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# jaccard — handcrafted oracle cases (spike 261010-wx2, report.txt:21-28)
# ─────────────────────────────────────────────────────────────────────────────

# Spike handcrafted case 2 input: identical interval sets on both sides.
IDENTICAL_SET = [("Chr1", 100, 250), ("Chr1", 400, 600), ("Chr2", 50, 150)]


def test_jaccard_identical_sets_is_one(tmp_path):
    # Case 2: A == B over three rows -> intersection == union == 450, jaccard 1.0.
    a = _write_bed(tmp_path, "a.bed", IDENTICAL_SET)
    b = _write_bed(tmp_path, "b.bed", IDENTICAL_SET)
    assert jaccard(a, b) == {"intersection": 450, "union": 450, "jaccard": 1.0}


def test_jaccard_bookended_adjacent_touching_coalesces_union(tmp_path):
    # Case 4: A ends exactly where B starts — zero intersection (half-open),
    # but the union coalesces the bookended pair into 200 bases, not 2x100.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 0, 100)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 100, 200)])
    assert jaccard(a, b) == {"intersection": 0, "union": 200, "jaccard": 0.0}


def test_jaccard_duplicates_and_staggered_overlaps(tmp_path):
    # Case 7: duplicate rows and staggered self-overlaps within side A merge
    # to (100, 400); B (150, 350) nests inside -> intersection 200, union 300.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 100, 300), ("Chr1", 100, 300), ("Chr1", 200, 400), ("Chr1", 205, 210)],
    )
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 150, 350)])
    assert jaccard(a, b) == {"intersection": 200, "union": 300, "jaccard": 0.6666666666666666}


def test_jaccard_empty_side_a_counts_toward_union(tmp_path):
    # Case 1: an empty A file is legal; one-sided chromosomes still count
    # their bases toward the union (1000 + 500), and jaccard is 0.0.
    a = _write_bed(tmp_path, "a.bed", [])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 1000, 2000), ("Chr1", 3000, 3500)])
    assert jaccard(a, b) == {"intersection": 0, "union": 1500, "jaccard": 0.0}


def test_jaccard_returns_exactly_three_keys(tmp_path):
    # The behavioral lock for the WPAR-01 exclusion: the return dict carries
    # intersection / union / jaccard and nothing else — no fourth key.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 100, 250)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 200, 400)])
    assert set(jaccard(a, b)) == {"intersection", "union", "jaccard"}


# ─────────────────────────────────────────────────────────────────────────────
# Package surface (re-export wiring)
# ─────────────────────────────────────────────────────────────────────────────


def test_package_surface_reexports_all_three_entry_points(tmp_path):
    # The dnallm.utils package surface (not the submodule) re-exports the
    # three public entry points; prove the wiring end-to-end by calling
    # through it.
    import dnallm.utils

    from dnallm.utils import jaccard as pkg_jaccard
    from dnallm.utils import loj_intersect as pkg_loj_intersect
    from dnallm.utils import parse_bed as pkg_parse_bed

    assert pkg_jaccard is jaccard
    assert pkg_loj_intersect is loj_intersect
    assert pkg_parse_bed is parse_bed
    assert "jaccard" in dnallm.utils.__all__
    assert "loj_intersect" in dnallm.utils.__all__
    assert "parse_bed" in dnallm.utils.__all__
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 0, 100)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 100, 200)])
    assert pkg_jaccard(a, b)["union"] == 200
    assert len(pkg_parse_bed(a)) == 1
    assert pkg_loj_intersect(a, b) == ["Chr1\t0\t100\t.\t-1\t-1"]


# ─────────────────────────────────────────────────────────────────────────────
# loj_intersect — null-B fill pin table (spike 261011-0a3, report.txt:8-17)
# ─────────────────────────────────────────────────────────────────────────────

# (variant name, B-row payload columns after (chrom, start, end), pinned
# null fill). The payload's first literal lands in B column 4 and its
# second in the score slot (B column 5) — the only content-dependent
# slot, and only at count 5. Do NOT re-derive this table from BED
# convention: the obvious reading is wrong at counts 5 and 7.
PIN_TABLE = [
    ("B-cols-3", [], [".", "-1", "-1"]),
    ("B-cols-4", ["name1"], [".", "-1", "-1", "."]),
    ("B-cols-5-int-score", ["name1", "0"], [".", "-1", "-1", ".", "-1"]),
    ("B-cols-5-non-int-score", ["name1", "exon"], [".", "-1", "-1", ".", "."]),
    ("B-cols-6-int-score", ["name1", "0", "+"], [".", "-1", "-1", ".", "-1", "."]),
    ("B-cols-6-non-int-score", ["name1", "exon", "+"], [".", "-1", "-1", ".", "-1", "."]),
    ("B-cols-7-int-score", ["name1", "0", "+", "c7"], [".", "-1", "-1", ".", ".", ".", "."]),
    (
        "B-cols-7-non-int-score",
        ["name1", "exon", "+", "c7"],
        [".", "-1", "-1", ".", ".", ".", "."],
    ),
]


def test_loj_null_fill_pin_table_all_nine_variants(tmp_path):
    # All nine pinned variants, driven through the public path: one no-hit
    # A row per variant; the B side of the emitted line must equal the
    # pinned literals exactly. The 6-column NER vector ('.', '-1', '-1',
    # '.', '-1', '.') is content-independent — both 6-column rows pin it.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 100, 200, "tokA", "geneA", "+")])
    for name, payload, pinned in PIN_TABLE:
        b = _write_bed(tmp_path, "b.bed", [("Chr1", 5000, 5100, *payload)])
        lines = loj_intersect(a, b)
        assert len(lines) == 1, name
        assert lines[0].split("\t")[6:] == pinned, (name, lines[0])
    # Ninth variant: an empty B file (column count unknowable) emits the
    # 3-column null form.
    b = _write_bed(tmp_path, "b_empty.bed", [])
    lines = loj_intersect(a, b)
    assert len(lines) == 1
    assert lines[0].split("\t")[6:] == [".", "-1", "-1"]


# ─────────────────────────────────────────────────────────────────────────────
# loj_intersect — handcrafted oracle cases (spike 261011-0a3, report.txt:37-47)
# ─────────────────────────────────────────────────────────────────────────────


def test_loj_empty_b_emits_three_column_null_form(tmp_path):
    # Case 2: empty B file -> every A row survives once with the 3-column
    # null form; the A-side columns ride through verbatim.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [
            ("Chr1", 100, 200, "ATTG", "Os01g1", "+"),
            ("Chr1", 300, 450, "GGCCT", "Os01g1", "+"),
            ("Chr2", 500, 600, "T", "Os02g4", "-"),
        ],
    )
    b = _write_bed(tmp_path, "b.bed", [])
    assert loj_intersect(a, b) == [
        "Chr1\t100\t200\tATTG\tOs01g1\t+\t.\t-1\t-1",
        "Chr1\t300\t450\tGGCCT\tOs01g1\t+\t.\t-1\t-1",
        "Chr2\t500\t600\tT\tOs02g4\t-\t.\t-1\t-1",
    ]


def test_loj_multiplicity_one_a_multi_b(tmp_path):
    # Case 6: one A row overlapping three B rows -> one line per pair, in
    # B-file order.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 100, 300, "ATTG", "g1", "+")])
    b = _write_bed(
        tmp_path,
        "b.bed",
        [
            ("Chr1", 50, 150, "gA", "exon", "+"),
            ("Chr1", 120, 220, "gB", "intron", "-"),
            ("Chr1", 250, 350, "gC", "exon", "+"),
        ],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t100\t300\tATTG\tg1\t+\tChr1\t50\t150\tgA\texon\t+",
        "Chr1\t100\t300\tATTG\tg1\t+\tChr1\t120\t220\tgB\tintron\t-",
        "Chr1\t100\t300\tATTG\tg1\t+\tChr1\t250\t350\tgC\texon\t+",
    ]


def test_loj_boundary_touching_rows_null_fill(tmp_path):
    # Case 7: A.end == B.start and B.end == A.start both MISS (strict
    # half-open inequalities); both A rows null-fill with the 6-column NER
    # vector '.', '-1', '-1', '.', '-1', '.' exactly — the -1 score slot
    # is load-bearing for the downstream consumer's name == "-1" branch.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 100, 200, "AA", "g1", "+"), ("Chr1", 400, 500, "CC", "g1", "-")],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [("Chr1", 200, 300, "b1", "exon", "+"), ("Chr1", 300, 400, "b2", "intron", "-")],
    )
    lines = loj_intersect(a, b)
    assert len(lines) == 2
    assert lines[0] == "Chr1\t100\t200\tAA\tg1\t+\t.\t-1\t-1\t.\t-1\t."
    assert lines[1] == "Chr1\t400\t500\tCC\tg1\t-\t.\t-1\t-1\t.\t-1\t."


def test_loj_duplicate_rows_both_sides_multiply_per_copy(tmp_path):
    # Case 9: 2 duplicate A rows x 4 B hits (3 distinct + 1 exact-duplicate
    # B row) = 8 lines — a join, never merged or coalesced. Each A copy
    # emits its full B-hit block consecutively (A-file order).
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 100, 300, "T1", "g1", "+"), ("Chr1", 100, 300, "T1", "g1", "+")],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [
            ("Chr1", 50, 150, "b1", "exon", "+"),
            ("Chr1", 120, 220, "b2", "intron", "-"),
            ("Chr1", 250, 350, "b3", "exon", "+"),
            ("Chr1", 250, 350, "b3", "exon", "+"),
        ],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t50\t150\tb1\texon\t+",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t120\t220\tb2\tintron\t-",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t250\t350\tb3\texon\t+",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t250\t350\tb3\texon\t+",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t50\t150\tb1\texon\t+",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t120\t220\tb2\tintron\t-",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t250\t350\tb3\texon\t+",
        "Chr1\t100\t300\tT1\tg1\t+\tChr1\t250\t350\tb3\texon\t+",
    ]


def test_loj_unsorted_input_emits_a_file_order(tmp_path):
    # Case 10: deliberately unsorted A and B — output follows A-file order
    # (Chr2 first), hits per A row in B-file order.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [
            ("Chr2", 900, 1000, "T1", "g1", "+"),
            ("Chr1", 100, 300, "T2", "g2", "-"),
            ("Chr1", 50, 150, "T3", "g3", "+"),
            ("Chr2", 100, 200, "T4", "g4", "-"),
            ("Chr1", 250, 350, "T5", "g5", "+"),
        ],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [
            ("Chr1", 0, 120, "bg1", "exon", "+"),
            ("Chr1", 110, 260, "bg2", "intron", "-"),
            ("Chr1", 255, 400, "bg3", "exon", "+"),
            ("Chr2", 80, 180, "bg4", "exon", "-"),
            ("Chr2", 850, 950, "bg5", "intron", "+"),
        ],
    )
    assert loj_intersect(a, b) == [
        "Chr2\t900\t1000\tT1\tg1\t+\tChr2\t850\t950\tbg5\tintron\t+",
        "Chr1\t100\t300\tT2\tg2\t-\tChr1\t0\t120\tbg1\texon\t+",
        "Chr1\t100\t300\tT2\tg2\t-\tChr1\t110\t260\tbg2\tintron\t-",
        "Chr1\t100\t300\tT2\tg2\t-\tChr1\t255\t400\tbg3\texon\t+",
        "Chr1\t50\t150\tT3\tg3\t+\tChr1\t0\t120\tbg1\texon\t+",
        "Chr1\t50\t150\tT3\tg3\t+\tChr1\t110\t260\tbg2\tintron\t-",
        "Chr2\t100\t200\tT4\tg4\t-\tChr2\t80\t180\tbg4\texon\t-",
        "Chr1\t250\t350\tT5\tg5\t+\tChr1\t110\t260\tbg2\tintron\t-",
        "Chr1\t250\t350\tT5\tg5\t+\tChr1\t255\t400\tbg3\texon\t+",
    ]


def test_loj_case11_ner_realistic_six_plus_six(tmp_path):
    # Case 11: the NER-shaped 6+6-column driver — A rows in per-gene
    # iteration order with token/gene/strand payloads against a 6-column
    # gene annotation B. All 8 A rows hit exactly one B row each.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [
            ("Chr1", 1200, 1210, "AT", "Os01g0100100", "+"),
            ("Chr1", 1210, 1215, "G", "Os01g0100100", "+"),
            ("Chr1", 900, 912, "CCGGATTA", "Os01g0202200", "-"),
            ("Chr2", 500, 530, "GGCCT", "Os02g0100300", "+"),
            ("Chr2", 530, 545, "TTGA", "Os02g0100300", "+"),
            ("Chr1", 1150, 1160, "C", "Os01g0100100", "+"),
            ("Chr2", 700, 760, "ATGCA", "Os02g0116600", "-"),
            ("Chr1", 2000, 2050, "AATT", "Os01g0202200", "-"),
        ],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [
            ("Chr1", 800, 1000, "Os01g0202200", "intron", "-"),
            ("Chr1", 1000, 1300, "Os01g0100100", "exon", "+"),
            ("Chr1", 1350, 1500, "Os01g0100100", "intron", "+"),
            ("Chr1", 1900, 2100, "Os01g0202200", "exon", "-"),
            ("Chr2", 400, 550, "Os02g0100300", "exon", "+"),
            ("Chr2", 600, 620, "Os02g0100300", "intron", "+"),
            ("Chr2", 690, 900, "Os02g0116600", "exon", "-"),
        ],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t1200\t1210\tAT\tOs01g0100100\t+\tChr1\t1000\t1300\tOs01g0100100\texon\t+",
        "Chr1\t1210\t1215\tG\tOs01g0100100\t+\tChr1\t1000\t1300\tOs01g0100100\texon\t+",
        "Chr1\t900\t912\tCCGGATTA\tOs01g0202200\t-\tChr1\t800\t1000\tOs01g0202200\tintron\t-",
        "Chr2\t500\t530\tGGCCT\tOs02g0100300\t+\tChr2\t400\t550\tOs02g0100300\texon\t+",
        "Chr2\t530\t545\tTTGA\tOs02g0100300\t+\tChr2\t400\t550\tOs02g0100300\texon\t+",
        "Chr1\t1150\t1160\tC\tOs01g0100100\t+\tChr1\t1000\t1300\tOs01g0100100\texon\t+",
        "Chr2\t700\t760\tATGCA\tOs02g0116600\t-\tChr2\t690\t900\tOs02g0116600\texon\t-",
        "Chr1\t2000\t2050\tAATT\tOs01g0202200\t-\tChr1\t1900\t2100\tOs01g0202200\texon\t-",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# jaccard — remaining spike handcrafted cases + both-sides-empty guard
# ─────────────────────────────────────────────────────────────────────────────


def test_jaccard_strict_containment_a_in_b(tmp_path):
    # Case 3: A nested strictly inside B on both rows — intersection is the
    # A bases (600 + 100), union the B bases (900 + 1000).
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 200, 800), ("Chr1", 2100, 2200)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 100, 1000), ("Chr1", 2000, 3000)])
    assert jaccard(a, b) == {"intersection": 700, "union": 1900, "jaccard": 0.3684210526315789}


def test_jaccard_fully_disjoint(tmp_path):
    # Case 5: no overlap, no adjacency — union is the plain sum of both sides.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 0, 100)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 500, 600)])
    assert jaccard(a, b) == {"intersection": 0, "union": 200, "jaccard": 0.0}


def test_jaccard_same_coords_different_chrom_names(tmp_path):
    # Case 6: identical coordinates under "Chr1" vs "chr1" — chromosome
    # grouping is by exact string key, so the sides never meet.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 100, 200)])
    b = _write_bed(tmp_path, "b.bed", [("chr1", 100, 200)])
    assert jaccard(a, b) == {"intersection": 0, "union": 200, "jaccard": 0.0}


def test_jaccard_shuffled_row_order(tmp_path):
    # Case 8: deliberately shuffled rows on both sides — the evaluator sorts
    # internally per chromosome, so row order never affects the result.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 500, 700), ("Chr1", 100, 300), ("Chr2", 50, 120), ("Chr1", 650, 900)],
    )
    b = _write_bed(tmp_path, "b.bed", [("Chr2", 100, 150), ("Chr1", 200, 600)])
    assert jaccard(a, b) == {"intersection": 220, "union": 900, "jaccard": 0.24444444444444444}


def test_jaccard_both_sides_empty_union_zero(tmp_path):
    # Extra non-spike guard: union == 0 must yield jaccard 0.0, never a
    # ZeroDivisionError (the single division is guarded).
    a = _write_bed(tmp_path, "a.bed", [])
    b = _write_bed(tmp_path, "b.bed", [])
    assert jaccard(a, b) == {"intersection": 0, "union": 0, "jaccard": 0.0}


def test_sweep_stats_both_sides_empty_returns_zeros():
    # Direct branch lock: the sweep's both-empty early return (unreachable
    # through jaccard, where every chromosome in the key union carries rows
    # on at least one side) — pinned so the branch is contract-locked, not
    # incidental.
    from dnallm.utils.genomic_intervals import _merge_intervals, _sweep_stats

    empty = _merge_intervals([])
    assert empty.shape == (0, 2)
    assert empty.dtype == np.int64
    assert _sweep_stats(empty, empty) == (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# loj_intersect — remaining spike handcrafted cases (full line literals)
# ─────────────────────────────────────────────────────────────────────────────


def test_loj_empty_a_emits_nothing(tmp_path):
    # Case 1: an empty A file is a legal empty join — zero output lines.
    a = _write_bed(tmp_path, "a.bed", [])
    b = _write_bed(
        tmp_path,
        "b.bed",
        [("Chr1", 1000, 1200, "Os01g2", "exon", "+"), ("Chr2", 50, 90, "Os02g3", "intron", "-")],
    )
    assert loj_intersect(a, b) == []


def test_loj_all_overlap_three_by_three(tmp_path):
    # Case 3: every A row overlaps every B row -> 3 x 3 = 9 lines, A-file
    # order outer, B-file order inner.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [
            ("Chr1", 100, 300, "ATT", "g1", "+"),
            ("Chr1", 200, 400, "GGC", "g1", "-"),
            ("Chr1", 250, 350, "T", "g2", "+"),
        ],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [
            ("Chr1", 50, 450, "g1", "exon", "+"),
            ("Chr1", 120, 280, "g2", "intron", "-"),
            ("Chr1", 260, 500, "g1", "exon", "+"),
        ],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t100\t300\tATT\tg1\t+\tChr1\t50\t450\tg1\texon\t+",
        "Chr1\t100\t300\tATT\tg1\t+\tChr1\t120\t280\tg2\tintron\t-",
        "Chr1\t100\t300\tATT\tg1\t+\tChr1\t260\t500\tg1\texon\t+",
        "Chr1\t200\t400\tGGC\tg1\t-\tChr1\t50\t450\tg1\texon\t+",
        "Chr1\t200\t400\tGGC\tg1\t-\tChr1\t120\t280\tg2\tintron\t-",
        "Chr1\t200\t400\tGGC\tg1\t-\tChr1\t260\t500\tg1\texon\t+",
        "Chr1\t250\t350\tT\tg2\t+\tChr1\t50\t450\tg1\texon\t+",
        "Chr1\t250\t350\tT\tg2\t+\tChr1\t120\t280\tg2\tintron\t-",
        "Chr1\t250\t350\tT\tg2\t+\tChr1\t260\t500\tg1\texon\t+",
    ]


def test_loj_no_overlap_null_representation(tmp_path):
    # Case 4: no A row overlaps any B row — every line carries the 6-column
    # null vector verbatim.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 100, 200, "ATTG", "g1", "+"), ("Chr1", 300, 400, "GGCCT", "g2", "-")],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [("Chr1", 1000, 1200, "g3", "exon", "+"), ("Chr1", 2000, 2100, "g4", "intron", "-")],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t100\t200\tATTG\tg1\t+\t.\t-1\t-1\t.\t-1\t.",
        "Chr1\t300\t400\tGGCCT\tg2\t-\t.\t-1\t-1\t.\t-1\t.",
    ]


def test_loj_strict_containment_hits_containing_row(tmp_path):
    # Case 5: A strictly inside B on both rows — the containing B row is the
    # hit; no null fill anywhere.
    a = _write_bed(
        tmp_path,
        "a.bed",
        [("Chr1", 200, 800, "ATTG", "g1", "+"), ("Chr1", 2100, 2200, "GGCCT", "g2", "-")],
    )
    b = _write_bed(
        tmp_path,
        "b.bed",
        [("Chr1", 100, 1000, "g1", "exon", "+"), ("Chr1", 2000, 3000, "g2", "intron", "-")],
    )
    assert loj_intersect(a, b) == [
        "Chr1\t200\t800\tATTG\tg1\t+\tChr1\t100\t1000\tg1\texon\t+",
        "Chr1\t2100\t2200\tGGCCT\tg2\t-\tChr1\t2000\t3000\tg2\tintron\t-",
    ]


def test_loj_same_coords_different_chrom_names_no_join(tmp_path):
    # Case 8: identical coordinates under "Chr1" vs "chr1" never join
    # (exact string key) — the single A row null-fills.
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 100, 200, "ATTG", "g1", "+")])
    b = _write_bed(tmp_path, "b.bed", [("chr1", 100, 200, "g1", "exon", "+")])
    assert loj_intersect(a, b) == ["Chr1\t100\t200\tATTG\tg1\t+\t.\t-1\t-1\t.\t-1\t."]


# ─────────────────────────────────────────────────────────────────────────────
# parse_bed — guard battery (loud ValueError naming file and line)
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_bed_rejects_fewer_than_three_columns(tmp_path):
    bad = _write_bed(tmp_path, "bad.bed", [("Chr1", "300")])
    with pytest.raises(ValueError, match=r"bad\.bed: line 1: expected at least 3 tab-separated"):
        parse_bed(bad)


def test_parse_bed_rejects_non_integer_coordinates(tmp_path):
    bad = _write_bed(tmp_path, "bad.bed", [("Chr1", "abc", "200")])
    with pytest.raises(ValueError, match=r"bad\.bed: line 1: non-integer coordinate"):
        parse_bed(bad)


def test_parse_bed_rejects_negative_start(tmp_path):
    bad = _write_bed(tmp_path, "bad.bed", [("Chr1", "-5", "200")])
    with pytest.raises(
        ValueError, match=r"bad\.bed: line 1: invalid 0-based half-open interval \(-5, 200\)"
    ):
        parse_bed(bad)


def test_parse_bed_rejects_empty_interval(tmp_path):
    # start == end is a zero-width interval: empty, not width-1.
    bad = _write_bed(tmp_path, "bad.bed", [("Chr1", "200", "200")])
    with pytest.raises(
        ValueError, match=r"bad\.bed: line 1: invalid 0-based half-open interval \(200, 200\)"
    ):
        parse_bed(bad)


def test_parse_bed_line_numbers_count_skipped_lines(tmp_path):
    # Blank and '#' comment lines are skipped but still advance the line
    # counter — the error must name the physical line number (here 4).
    path = tmp_path / "bad.bed"
    path.write_text(
        "Chr1\t100\t200\n\n# a comment line\nChr1\t300\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.bed: line 4: expected at least 3"):
        parse_bed(path)


def test_parse_bed_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "mixed.bed"
    path.write_text(
        "# header comment\n\n   \nChr1\t100\t200\tname1\t0\t+\n# trailing comment\nChr2\t50\t90\n",
        encoding="utf-8",
    )
    rows = parse_bed(path)
    # Only the two data rows survive, in file order, never sorted; trailing
    # columns ride through verbatim and line_index keeps physical numbering.
    assert len(rows) == 2
    assert rows[0].chrom == "Chr1"
    assert rows[0].start == 100
    assert rows[0].end == 200
    assert rows[0].extra == ["name1", "0", "+"]
    assert rows[0].line_index == 4
    assert rows[1].chrom == "Chr2"
    assert (rows[1].start, rows[1].end) == (50, 90)
    assert rows[1].extra == []
    assert rows[1].line_index == 6


def test_parse_bed_crlf_file_parses_identically(tmp_path):
    # A CRLF-terminated file (Windows-written fixture shape) parses to the
    # same rows as its LF twin — terminators are stripped per line.
    lf = _write_bed(tmp_path, "lf.bed", [("Chr1", 100, 200), ("Chr1", 300, 450)])
    crlf = tmp_path / "crlf.bed"
    crlf.write_bytes(b"Chr1\t100\t200\r\nChr1\t300\t450\r\n")
    assert parse_bed(crlf) == parse_bed(lf)


# ─────────────────────────────────────────────────────────────────────────────
# Frozen-pair aggregates (documentation anchors; see module of record)
# ─────────────────────────────────────────────────────────────────────────────

# Quick-261010-wx2 recorded oracle: the 51/94-row CRE-selection pair lived
# at example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed,
# which is gitignored (.gitignore .scratch/) and absent from disk — the row
# corpora are unrecoverable in-repo, so only the aggregates survive. The
# real-pair in-band gate is Phase 16 / WPAR-05 and is NOT recomputed here.
FROZEN_PAIR_INTERSECTION = 15872
FROZEN_PAIR_UNION = 48878
FROZEN_PAIR_JACCARD = 0.32472687098490116  # numpy full precision; bedtools prints 6 dp


def test_frozen_pair_anchor_rounds_to_selection_md_value():
    # selection.md anchors jaccard=0.324727; the recorded full-precision
    # value must round to exactly that at 6 decimal places.
    assert round(FROZEN_PAIR_JACCARD, 6) == 0.324727
    assert FROZEN_PAIR_INTERSECTION == 15872
    assert FROZEN_PAIR_UNION == 48878


# ─────────────────────────────────────────────────────────────────────────────
# Seeded corpus generators (ported verbatim from the spikes; draw order is
# load-bearing — seeds 20261010 jaccard / 20261011 loj)
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = 8000
CHROM_POOL = ("Chr1", "Chr2", "Chr3", "Chr4", "chr1", "1", "scaffoldA")
PATTERNS = ("mixed", "dense", "containment", "adjacency", "disjoint", "duplicates")
TOKEN_POOL = (
    "A",
    "CG",
    "ATT",
    "AA",
    "GGCCT",
    "T",
    "CCGGATTA",
    "GCA",
    "AT",
    "ATGCA",
    "TTGA",
    "C",
)
GENE_POOL = ("Os01g0100100", "Os01g0202200", "Os02g0100300", "Os03g0116600", "gene7")
FEATURE_POOL = ("exon", "intron")
STRAND_POOL = ("+", "-")
JACCARD_SEED = 20261010
LOJ_SEED = 20261011


def _draw_intervals(rng, count, lo, hi, min_len, max_len):
    out = []
    for _ in range(count):
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, hi - lo - length))
        out.append((lo + start, lo + start + length))
    return out


def _gen_mixed(rng, cap):
    a = _draw_intervals(rng, int(rng.integers(0, cap)), 0, WINDOW, 20, 1200)
    b = _draw_intervals(rng, int(rng.integers(0, cap)), 0, WINDOW, 20, 1200)
    return a, b


def _gen_dense(rng, cap):
    a = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 60, 900)
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 60, 900)
    return a, b


def _gen_containment(rng, cap):
    n_b = int(rng.integers(2, max(3, cap + 1)))
    b = []
    pos = 0
    for _ in range(n_b):
        gap = int(rng.integers(10, 80))
        length = int(rng.integers(200, 500))
        b.append((pos + gap, pos + gap + length))
        pos += gap + length
    picks = rng.choice(len(b), size=int(rng.integers(1, len(b) + 1)), replace=False)
    a = []
    for idx in picks:
        start, end = b[int(idx)]
        pad_left = int(rng.integers(0, 40))
        pad_right = int(rng.integers(0, 40))
        if pad_left == 0 and pad_right == 0:
            pad_right = 1  # keep containment strict on at least one boundary
        a.append((start + pad_left, end - pad_right))
    return a, b


def _gen_adjacency(rng, cap):
    n_cuts = int(rng.integers(3, 11))
    cuts = sorted({int(x) for x in rng.integers(1, WINDOW, size=n_cuts)})
    bounds = [0, *cuts, WINDOW]
    a = []
    b = []
    for j in range(len(bounds) - 1):
        if rng.random() < 0.15:
            continue  # segment owned by neither side (a gap)
        segment = (bounds[j], bounds[j + 1])
        (a if j % 3 != 2 else b).append(segment)
    return a, b


def _gen_disjoint(rng, cap):
    mid = int(rng.integers(WINDOW // 4, 3 * WINDOW // 4))
    a = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, mid, 30, 300)
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), mid, WINDOW, 30, 300)
    return a, b


def _stagger(rng, rows):
    # Exact-duplicate and staggered-overlap appendages (the duplicates
    # pattern): duplicates multiply per copy on whichever side this runs.
    out = list(rows)
    for start, end in rows:
        roll = rng.random()
        if roll < 0.35:
            out.append((start, end))  # exact duplicate row
        elif roll < 0.70:
            shift = int(rng.integers(1, 16))
            if rng.random() < 0.5 and start > 0:
                shift = -min(shift, start)  # keep the staggered start >= 0
            out.append((start + shift, end + shift))  # staggered overlap
    return out


def _gen_duplicates_jaccard(rng, cap):
    # wx2 spike variant: duplicates and staggers on side A only.
    base = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    a = _stagger(rng, base)
    b = _draw_intervals(rng, int(rng.integers(1, cap + 1)), 0, WINDOW, 80, 500)
    return a, b


def _gen_duplicates_loj(rng, cap):
    # 0a3 spike variant: duplicates and staggers on BOTH sides — the join
    # must multiply them per copy, never coalesce.
    base_a = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    base_b = _draw_intervals(rng, int(rng.integers(2, max(3, cap + 1))), 0, WINDOW, 80, 500)
    return _stagger(rng, base_a), _stagger(rng, base_b)


_JACCARD_GENERATORS = {
    "mixed": _gen_mixed,
    "dense": _gen_dense,
    "containment": _gen_containment,
    "adjacency": _gen_adjacency,
    "disjoint": _gen_disjoint,
    "duplicates": _gen_duplicates_jaccard,
}
_LOJ_GENERATORS = dict(_JACCARD_GENERATORS, duplicates=_gen_duplicates_loj)


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def _gen_jaccard_case(rng, index):
    # One seeded BED3 case (wx2 spike gen_random_case, seed 20261010):
    # pattern cycles over PATTERNS, chroms from CHROM_POOL (exact string
    # keys mix naming styles), per-side cap 30 rows, never both sides empty,
    # every 4th case shuffled else canonically sorted.
    pattern = PATTERNS[index % len(PATTERNS)]
    n_chroms = int(rng.integers(1, 5))
    picked = rng.choice(len(CHROM_POOL), size=n_chroms, replace=False)
    chroms = [CHROM_POOL[int(i)] for i in picked]
    cap = max(2, 30 // n_chroms)
    generator = _JACCARD_GENERATORS[pattern]
    a_rows = []
    b_rows = []
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
    return a_rows, b_rows


def _gen_loj_case(rng, index):
    # One seeded NER-shaped 6+6 column case (0a3 spike gen_random_case,
    # seed 20261011): A rows carry (token, gene, strand) payloads, B rows
    # (gene, feature, strand), drawn per row from the fixed pools.
    pattern = PATTERNS[index % len(PATTERNS)]
    n_chroms = int(rng.integers(1, 5))
    picked = rng.choice(len(CHROM_POOL), size=n_chroms, replace=False)
    chroms = [CHROM_POOL[int(i)] for i in picked]
    cap = max(2, 30 // n_chroms)
    generator = _LOJ_GENERATORS[pattern]
    a_rows = []
    b_rows = []
    for chrom in chroms:
        side_a, side_b = generator(rng, cap)
        a_rows.extend(
            (
                chrom,
                start,
                end,
                _pick(rng, TOKEN_POOL),
                _pick(rng, GENE_POOL),
                _pick(rng, STRAND_POOL),
            )
            for start, end in side_a
        )
        b_rows.extend(
            (
                chrom,
                start,
                end,
                _pick(rng, GENE_POOL),
                _pick(rng, FEATURE_POOL),
                _pick(rng, STRAND_POOL),
            )
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
    return a_rows, b_rows


# ─────────────────────────────────────────────────────────────────────────────
# Seeded corpora — property invariants (primary layer, no baked oracles)
# ─────────────────────────────────────────────────────────────────────────────


def test_jaccard_seeded_corpus_property_invariants(tmp_path):
    # >=300 regenerated cases from the ported seed-20261010 generator: the
    # parity-proven invariants must hold for every case, exactly.
    rng = np.random.default_rng(JACCARD_SEED)
    for index in range(300):
        a_rows, b_rows = _gen_jaccard_case(rng, index)
        a = _write_bed(tmp_path, f"a{index}.bed", a_rows)
        b = _write_bed(tmp_path, f"b{index}.bed", b_rows)
        result = jaccard(a, b)
        assert set(result) == {"intersection", "union", "jaccard"}, index
        assert 0.0 <= result["jaccard"] <= 1.0, index
        assert result["intersection"] <= result["union"], index
        if result["union"] > 0:
            assert result["jaccard"] == result["intersection"] / result["union"], index
        else:
            assert result["jaccard"] == 0.0, index
        # Symmetry: swapping the sides changes nothing.
        assert jaccard(b, a) == result, index
        # Identity: a non-empty side joined with itself is exactly 1.0.
        if a_rows:
            self_result = jaccard(a, a)
            assert self_result["jaccard"] == 1.0, index
            assert self_result["intersection"] == self_result["union"], index


def test_loj_seeded_corpus_property_invariants(tmp_path):
    # >=200 regenerated NER-shaped cases from the ported seed-20261011
    # generator: emission counts, ordering and null fills recomputed
    # independently (plain python double loop — no numpy in the oracle).
    rng = np.random.default_rng(LOJ_SEED)
    for index in range(200):
        a_rows, b_rows = _gen_loj_case(rng, index)
        a = _write_bed(tmp_path, f"a{index}.bed", a_rows)
        b = _write_bed(tmp_path, f"b{index}.bed", b_rows)
        lines = loj_intersect(a, b)
        a_parsed = parse_bed(a)
        b_parsed = parse_bed(b)
        # Null fill for THIS case's B column count: corpus B rows are
        # 6-column when present; an empty B file pins the 3-column form.
        null_fill = [".", "-1", "-1"] if not b_rows else [".", "-1", "-1", ".", "-1", "."]
        # Every A row yields at least one line.
        assert len(lines) >= len(a_parsed), index
        # Walk the emitted lines against per-A-row expectations.
        cursor = 0
        for arow in a_parsed:
            hits = [
                brow
                for brow in b_parsed
                if brow.chrom == arow.chrom and brow.start < arow.end and brow.end > arow.start
            ]
            split_at = 3 + len(arow.extra)
            if hits:
                for brow in hits:
                    fields = lines[cursor].split("\t")
                    cursor += 1
                    assert fields[:split_at] == [
                        arow.chrom,
                        str(arow.start),
                        str(arow.end),
                        *arow.extra,
                    ], (index, cursor)
                    # The hit's B interval satisfies the strict half-open
                    # inequalities recomputed from parse_bed.
                    assert fields[split_at : split_at + 3] == [
                        brow.chrom,
                        str(brow.start),
                        str(brow.end),
                    ], (index, cursor)
                    assert int(fields[split_at + 1]) < arow.end, (index, cursor)
                    assert int(fields[split_at + 2]) > arow.start, (index, cursor)
                    assert fields[split_at + 3 :] == brow.extra, (index, cursor)
            else:
                fields = lines[cursor].split("\t")
                cursor += 1
                # Miss lines carry the exact null fill for the 6-column B.
                assert fields[:split_at] == [
                    arow.chrom,
                    str(arow.start),
                    str(arow.end),
                    *arow.extra,
                ], (index, cursor)
                assert fields[split_at:] == null_fill, (index, cursor)
        # Total line count == sum over A rows of (hit count or 1).
        assert cursor == len(lines), index


def test_seeded_corpus_generators_are_deterministic():
    # Two consecutive runs from the same seed produce identical corpora —
    # reruns are byte-for-byte reproducible (TRUTH-8).
    rng_first = np.random.default_rng(JACCARD_SEED)
    rng_second = np.random.default_rng(JACCARD_SEED)
    for index in range(24):
        assert _gen_jaccard_case(rng_first, index) == _gen_jaccard_case(rng_second, index)
    rng_first = np.random.default_rng(LOJ_SEED)
    rng_second = np.random.default_rng(LOJ_SEED)
    for index in range(24):
        assert _gen_loj_case(rng_first, index) == _gen_loj_case(rng_second, index)


# ─────────────────────────────────────────────────────────────────────────────
# Import purity (numpy + stdlib only; pandas/pybedtools/torch sentinel-blocked)
# ─────────────────────────────────────────────────────────────────────────────


def test_module_import_is_pandas_pybedtools_torch_free(tmp_path):
    # The submodule must import with its heavy siblings sentinel-blocked.
    # Targeting dnallm.utils.genomic_intervals DIRECTLY — never the
    # dnallm.utils package, whose __init__ imports torch via support.py
    # (package-level purity is impossible and not required).
    import dnallm.utils

    blocked = ("pandas", "pybedtools", "torch")
    saved_blocked = {name: sys.modules.get(name) for name in blocked}
    saved_module = sys.modules.get("dnallm.utils.genomic_intervals")
    saved_pkg_attr = getattr(dnallm.utils, "genomic_intervals", None)
    for name in blocked:
        sys.modules.pop(name, None)
        sys.modules[name] = None  # any import attempt now raises ImportError
    sys.modules.pop("dnallm.utils.genomic_intervals", None)
    try:
        mod = importlib.import_module("dnallm.utils.genomic_intervals")
        a = _write_bed(tmp_path, "a.bed", [("Chr1", 0, 100)])
        b = _write_bed(tmp_path, "b.bed", [("Chr1", 100, 200)])
        assert mod.jaccard(a, b) == {"intersection": 0, "union": 200, "jaccard": 0.0}
    finally:
        for name, saved_entry in saved_blocked.items():
            if saved_entry is not None:
                sys.modules[name] = saved_entry
            else:
                del sys.modules[name]
        if saved_module is not None:
            sys.modules["dnallm.utils.genomic_intervals"] = saved_module
        # import_module rebinds the package attribute to the re-executed
        # module object; restore it so no second live module object
        # survives the purity check.
        if saved_pkg_attr is not None:
            dnallm.utils.genomic_intervals = saved_pkg_attr
        else:
            del dnallm.utils.genomic_intervals


def test_import_purity_leaves_single_live_module(tmp_path):
    # After the purity test runs there must be exactly one live module —
    # the package attribute and the sys.modules entry point at one object.
    test_module_import_is_pandas_pybedtools_torch_free(tmp_path)
    import dnallm.utils

    assert dnallm.utils.genomic_intervals is sys.modules["dnallm.utils.genomic_intervals"]


# ─────────────────────────────────────────────────────────────────────────────
# Seeded corpora — fully-baked subset literals (one full 12-cycle per spike:
# every pattern x shuffle state the 6-pattern/4-shuffle cycle produces)
# ─────────────────────────────────────────────────────────────────────────────


def test_jaccard_seeded_subset_baked_literals(tmp_path):
    # Indices 0-11 of the seed-20261010 corpus with fully baked outputs —
    # any numpy Generator stream divergence on another numpy version fails
    # loudly here instead of diffing hundreds of property cases.
    rng = np.random.default_rng(JACCARD_SEED)
    baked = [
        (0, 1276, 6288, 0.2029262086513995),
        (1, 2879, 9498, 0.3031164455674879),
        (2, 1081, 1659, 0.6515973477998794),
        (3, 0, 12942, 0.0),
        (4, 0, 4421, 0.0),
        (5, 520, 5346, 0.09726898615787505),
        (6, 1978, 10533, 0.18779075287192631),
        (7, 1214, 14375, 0.08445217391304348),
        (8, 2683, 6353, 0.42232016370218795),
        (9, 0, 8000, 0.0),
        (10, 0, 4669, 0.0),
        (11, 2085, 8932, 0.23343036274070758),
    ]
    for index, intersection, union, value in baked:
        a_rows, b_rows = _gen_jaccard_case(rng, index)
        a = _write_bed(tmp_path, f"a{index}.bed", a_rows)
        b = _write_bed(tmp_path, f"b{index}.bed", b_rows)
        assert jaccard(a, b) == {
            "intersection": intersection,
            "union": union,
            "jaccard": value,
        }, index


def test_loj_seeded_subset_baked_literals(tmp_path):
    # Indices 0-11 of the seed-20261011 corpus with fully baked line lists.
    rng = np.random.default_rng(LOJ_SEED)
    baked = [
        (
            0,
            [
                "1\t2440\t2981\tATGCA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "1\t3199\t4076\tATT\tOs03g0116600\t-\t1\t3245\t3770\tOs02g0100300\texon\t+",
                "1\t3621\t3790\tGCA\tOs01g0202200\t+\t1\t3245\t3770\tOs02g0100300\texon\t+",
                "Chr1\t87\t1011\tCG\tOs02g0100300\t+\tChr1\t434\t1192\tOs03g0116600\tintron\t+",
                "Chr1\t87\t1011\tCG\tOs02g0100300\t+\tChr1\t717\t1504\tOs02g0100300\tintron\t+",
                "Chr1\t1088\t1344\tT\tOs03g0116600\t-\tChr1\t434\t1192\tOs03g0116600\tintron\t+",
                "Chr1\t1088\t1344\tT\tOs03g0116600\t-\tChr1\t717\t1504\tOs02g0100300\tintron\t+",
                "Chr1\t1232\t2241\tATT\tOs03g0116600\t+\tChr1\t717\t1504\tOs02g0100300\tintron\t+",
                "Chr1\t1344\t1964\tTTGA\tOs02g0100300\t-\tChr1\t717\t1504\tOs02g0100300\tintron\t+",
                "Chr1\t1399\t1668\tGGCCT\tOs01g0202200\t+\tChr1\t717\t1504\tOs02g0100300\tintron\t+",
                "Chr1\t1532\t2110\tGCA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t6120\t6183\tAT\tOs03g0116600\t-\tChr1\t5229\t6239\tOs01g0100100\texon\t-",
                "Chr1\t6120\t6183\tAT\tOs03g0116600\t-\tChr1\t6019\t7187\tOs03g0116600\tintron\t+",
                "Chr3\t790\t1490\tT\tOs03g0116600\t+\tChr3\t32\t818\tOs03g0116600\tintron\t-",
                "Chr3\t790\t1490\tT\tOs03g0116600\t+\tChr3\t133\t1226\tOs02g0100300\texon\t+",
                "Chr3\t3739\t4462\tC\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t3997\t4799\tTTGA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t4049\t5205\tAT\tOs03g0116600\t-\tChr3\t4971\t5735\tOs03g0116600\texon\t+",
                "Chr3\t5526\t6465\tATGCA\tgene7\t-\tChr3\t4971\t5735\tOs03g0116600\texon\t+",
                "Chr3\t5526\t6465\tATGCA\tgene7\t-\tChr3\t5331\t5756\tOs01g0202200\texon\t-",
                "Chr3\t5526\t6465\tATGCA\tgene7\t-\tChr3\t6074\t6780\tOs03g0116600\texon\t+",
                "Chr3\t5526\t6465\tATGCA\tgene7\t-\tChr3\t6118\t6409\tOs02g0100300\texon\t+",
                "Chr3\t5526\t6465\tATGCA\tgene7\t-\tChr3\t6282\t6703\tOs03g0116600\tintron\t-",
                "Chr3\t5621\t6146\tTTGA\tgene7\t+\tChr3\t4971\t5735\tOs03g0116600\texon\t+",
                "Chr3\t5621\t6146\tTTGA\tgene7\t+\tChr3\t5331\t5756\tOs01g0202200\texon\t-",
                "Chr3\t5621\t6146\tTTGA\tgene7\t+\tChr3\t6074\t6780\tOs03g0116600\texon\t+",
                "Chr3\t5621\t6146\tTTGA\tgene7\t+\tChr3\t6118\t6409\tOs02g0100300\texon\t+",
                "Chr3\t6287\t6870\tCCGGATTA\tOs01g0202200\t-\tChr3\t6074\t6780\tOs03g0116600\texon\t+",
                "Chr3\t6287\t6870\tCCGGATTA\tOs01g0202200\t-\tChr3\t6118\t6409\tOs02g0100300\texon\t+",
                "Chr3\t6287\t6870\tCCGGATTA\tOs01g0202200\t-\tChr3\t6282\t6703\tOs03g0116600\tintron\t-",
            ],
        ),
        (
            1,
            [
                "Chr3\t6811\t7231\tC\tOs01g0100100\t+\tChr3\t6954\t7626\tOs01g0100100\tintron\t-",
                "Chr4\t6706\t7271\tCCGGATTA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t942\t1670\tAT\tOs01g0100100\t+\tchr1\t707\t1239\tOs03g0116600\tintron\t+",
                "chr1\t942\t1670\tAT\tOs01g0100100\t+\tchr1\t1086\t1642\tOs01g0100100\tintron\t-",
                "chr1\t942\t1670\tAT\tOs01g0100100\t+\tchr1\t1086\t1861\tOs02g0100300\tintron\t-",
                "chr1\t1866\t2222\tCCGGATTA\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t2919\t3334\tC\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t6506\t6767\tATT\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1277\t1401\tTTGA\tOs03g0116600\t-\tscaffoldA\t1332\t1655\tOs01g0202200\tintron\t-",
                "scaffoldA\t1316\t2138\tGGCCT\tOs01g0100100\t+\tscaffoldA\t1332\t1655\tOs01g0202200\tintron\t-",
                "scaffoldA\t6034\t6503\tC\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t6760\t7376\tATT\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            2,
            [
                "1\t73\t289\tA\tOs03g0116600\t-\t1\t61\t292\tOs02g0100300\texon\t+",
                "1\t982\t1364\tAA\tOs02g0100300\t+\t1\t982\t1386\tOs03g0116600\texon\t+",
                "Chr4\t65\t212\tAT\tOs01g0202200\t-\tChr4\t37\t237\tgene7\texon\t+",
                "Chr4\t335\t704\tATGCA\tOs03g0116600\t+\tChr4\t308\t713\tOs02g0100300\texon\t+",
                "chr1\t503\t788\tA\tOs01g0202200\t-\tchr1\t467\t807\tOs03g0116600\texon\t-",
            ],
        ),
        (
            3,
            [
                "Chr4\t7327\t8000\tATT\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t0\t805\tTTGA\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t805\t1928\tTTGA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t6712\t8000\tATT\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t1655\t6886\tT\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            4,
            [
                "Chr1\t111\t314\tATGCA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t1150\t1234\tATGCA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t1311\t1353\tCCGGATTA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t1719\t1940\tT\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2611\t2826\tC\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2925\t3049\tC\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t3219\t3458\tCG\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t3801\t4077\tA\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t4121\t4324\tA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t3013\t3201\tCCGGATTA\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr3\t3111\t3251\tAT\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t819\t1019\tC\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t877\t999\tT\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1270\t1346\tC\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1347\t1528\tT\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1405\t1533\tT\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1454\t1586\tGGCCT\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1494\t1668\tTTGA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2133\t2344\tT\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2198\t2387\tA\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2546\t2824\tC\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            5,
            [
                "Chr1\t2302\t2777\tATT\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2928\t3183\tGCA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2929\t3184\tCCGGATTA\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t4865\t5149\tAA\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t4957\t5365\tA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t5184\t5631\tATGCA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t5195\t5642\tGCA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t5609\t5699\tGCA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t5609\t5699\tGCA\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t5814\t6181\tOs02g0100300\tintron\t-",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t5814\t6181\tOs03g0116600\tintron\t-",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t5978\t6194\tgene7\texon\t+",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t5978\t6194\tgene7\texon\t+",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t6138\t6387\tOs02g0100300\texon\t-",
                "Chr1\t6077\t6490\tCCGGATTA\tOs02g0100300\t-\tChr1\t6143\t6392\tgene7\texon\t+",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t5814\t6181\tOs02g0100300\tintron\t-",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t5814\t6181\tOs03g0116600\tintron\t-",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t5978\t6194\tgene7\texon\t+",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t5978\t6194\tgene7\texon\t+",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t6138\t6387\tOs02g0100300\texon\t-",
                "Chr1\t6077\t6490\tA\tgene7\t+\tChr1\t6143\t6392\tgene7\texon\t+",
                "Chr1\t7057\t7138\tCG\tgene7\t-\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7057\t7138\tCG\tgene7\t-\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7057\t7138\tATGCA\tOs03g0116600\t-\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7057\t7138\tATGCA\tOs03g0116600\t-\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7190\t7669\tATGCA\tOs01g0100100\t-\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7190\t7669\tATGCA\tOs01g0100100\t-\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7202\t7681\tGCA\tOs01g0100100\t+\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7202\t7681\tGCA\tOs01g0100100\t+\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7210\t7599\tC\tOs03g0116600\t+\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7210\t7599\tC\tOs03g0116600\t+\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7216\t7605\tA\tgene7\t+\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7216\t7605\tA\tgene7\t+\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7353\t7824\tATGCA\tgene7\t+\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7353\t7824\tATGCA\tgene7\t+\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7353\t7824\tATGCA\tgene7\t+\tChr1\t7681\t7830\tOs02g0100300\texon\t-",
                "Chr1\t7353\t7824\tATGCA\tgene7\t+\tChr1\t7683\t7832\tOs01g0100100\tintron\t+",
                "Chr1\t7359\t7830\tATT\tOs02g0100300\t-\tChr1\t7073\t7487\tgene7\tintron\t+",
                "Chr1\t7359\t7830\tATT\tOs02g0100300\t-\tChr1\t7088\t7502\tOs01g0202200\texon\t+",
                "Chr1\t7359\t7830\tATT\tOs02g0100300\t-\tChr1\t7681\t7830\tOs02g0100300\texon\t-",
                "Chr1\t7359\t7830\tATT\tOs02g0100300\t-\tChr1\t7683\t7832\tOs01g0100100\tintron\t+",
                "scaffoldA\t1809\t2235\tTTGA\tgene7\t-\tscaffoldA\t1940\t2325\tOs01g0202200\texon\t-",
                "scaffoldA\t1809\t2235\tTTGA\tgene7\t-\tscaffoldA\t2055\t2204\tOs02g0100300\texon\t+",
                "scaffoldA\t1809\t2235\tTTGA\tgene7\t-\tscaffoldA\t2055\t2204\tOs01g0202200\texon\t+",
                "scaffoldA\t3838\t4020\tATGCA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3838\t4020\tATGCA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            6,
            [
                "1\t2013\t3171\tA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "1\t2125\t2501\tATT\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "1\t2253\t3172\tGGCCT\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "1\t6745\t6867\tTTGA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t428\t867\tCCGGATTA\tOs02g0100300\t+\tChr1\t861\t1329\tgene7\tintron\t-",
                "Chr4\t3457\t4334\tT\tOs01g0100100\t-\tChr4\t2845\t3936\tgene7\texon\t-",
                "Chr4\t3457\t4334\tT\tOs01g0100100\t-\tChr4\t4005\t4445\tgene7\tintron\t-",
                "Chr4\t3457\t4334\tT\tOs01g0100100\t-\tChr4\t4155\t4680\tOs01g0202200\tintron\t+",
                "scaffoldA\t2878\t4003\tTTGA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3686\t3824\tC\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t4831\t5925\tAT\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t4864\t5882\tATGCA\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t5023\t5324\tT\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t6180\t7041\tCG\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            7,
            [
                "Chr1\t5412\t5742\tAT\tgene7\t-\tChr1\t5686\t5991\tOs01g0100100\texon\t-",
                "chr1\t3221\t3882\tAT\tOs01g0100100\t-\tchr1\t2741\t3293\tOs01g0202200\texon\t-",
                "Chr1\t1601\t1787\tCCGGATTA\tOs01g0100100\t+\tChr1\t1692\t2132\tOs01g0202200\tintron\t+",
                "Chr1\t1601\t1787\tCCGGATTA\tOs01g0100100\t+\tChr1\t1566\t1968\tOs01g0100100\tintron\t+",
                "scaffoldA\t2825\t3243\tCG\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1624\t2496\tC\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2920\t3251\tA\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t6071\t6568\tATT\tgene7\t+\tChr1\t6286\t6543\tOs01g0100100\tintron\t+",
                "Chr1\t2775\t3567\tA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t6824\t7191\tA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t2050\t2889\tGGCCT\tgene7\t+\tchr1\t2741\t3293\tOs01g0202200\texon\t-",
                "chr1\t2050\t2889\tGGCCT\tgene7\t+\tchr1\t2161\t3039\tgene7\texon\t-",
                "Chr1\t3422\t3805\tATGCA\tgene7\t-\tChr1\t3658\t4366\tOs01g0202200\texon\t+",
                "chr1\t5381\t5537\tGCA\tOs01g0202200\t-\tchr1\t4883\t5710\tOs03g0116600\tintron\t+",
                "scaffoldA\t3262\t4103\tTTGA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            8,
            [
                "Chr2\t40\t234\tC\tOs01g0100100\t+\tChr2\t10\t260\tOs01g0100100\tintron\t+",
                "Chr3\t89\t391\tT\tOs01g0100100\t+\tChr3\t59\t417\tOs01g0202200\tintron\t-",
                "Chr3\t466\t817\tAA\tOs01g0202200\t+\tChr3\t446\t821\tOs03g0116600\tintron\t+",
                "Chr3\t882\t1272\tAA\tOs02g0100300\t-\tChr3\t872\t1289\tOs02g0100300\texon\t+",
                "Chr3\t1368\t1637\tCG\tOs01g0202200\t-\tChr3\t1355\t1676\tgene7\tintron\t-",
                "Chr3\t1715\t1913\tATGCA\tOs03g0116600\t-\tChr3\t1705\t1928\tgene7\tintron\t-",
                "Chr3\t1974\t2338\tATGCA\tOs02g0100300\t+\tChr3\t1956\t2360\tOs02g0100300\texon\t+",
                "Chr3\t2399\t2831\tAA\tOs01g0202200\t-\tChr3\t2372\t2844\tOs01g0202200\texon\t+",
                "Chr3\t3313\t3625\tATGCA\tOs02g0100300\t+\tChr3\t3292\t3649\tOs03g0116600\tintron\t+",
            ],
        ),
        (
            9,
            [
                "1\t0\t1722\tGCA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "1\t1722\t2215\tCG\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "1\t2345\t3837\tC\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "1\t3837\t3972\tC\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t0\t650\tA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t650\t694\tC\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t722\t2098\tTTGA\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t2098\t3286\tCCGGATTA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t4200\t4276\tTTGA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t4276\t5125\tTTGA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t5333\t6255\tT\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t6255\t8000\tCCGGATTA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t0\t2114\tAA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2114\t2812\tGCA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3258\t3578\tAA\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t4605\t4757\tA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            10,
            [
                "Chr1\t58\t304\tCG\tgene7\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t1681\t1758\tA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t1851\t1887\tC\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2260\t2442\tT\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t2411\t2549\tATT\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t3761\t3855\tAA\tOs03g0116600\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t3971\t4125\tC\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr1\t4173\t4313\tCCGGATTA\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t361\t401\tGCA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t748\t994\tATGCA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t925\t1094\tAT\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1063\t1226\tTTGA\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1258\t1510\tAA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1265\t1351\tCG\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1671\t1848\tGGCCT\tOs01g0100100\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1861\t2103\tGCA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2492\t2727\tAT\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2667\t2804\tTTGA\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t2952\t3139\tATT\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3163\t3196\tA\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3175\t3373\tAT\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t3533\t3582\tGCA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
            ],
        ),
        (
            11,
            [
                "scaffoldA\t4420\t4585\tA\tOs01g0100100\t+\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t2715\t3101\tCG\tOs03g0116600\t-\tchr1\t2802\t2897\tOs01g0100100\texon\t-",
                "chr1\t2715\t3101\tCG\tOs03g0116600\t-\tchr1\t2802\t2897\tgene7\texon\t+",
                "Chr4\t2066\t2507\tA\tgene7\t+\tChr4\t2035\t2532\tOs03g0116600\tintron\t-",
                "Chr4\t2066\t2507\tA\tgene7\t+\tChr4\t2035\t2532\tOs01g0202200\tintron\t-",
                "scaffoldA\t4739\t5136\tAT\tOs01g0100100\t+\tscaffoldA\t4963\t5215\tOs03g0116600\tintron\t+",
                "scaffoldA\t1328\t1770\tT\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t6801\t6932\tCG\tOs02g0100300\t+\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t2715\t3101\tC\tOs01g0202200\t+\tchr1\t2802\t2897\tOs01g0100100\texon\t-",
                "chr1\t2715\t3101\tC\tOs01g0202200\t+\tchr1\t2802\t2897\tgene7\texon\t+",
                "Chr2\t2970\t3236\tAT\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t5393\t5611\tGCA\tOs03g0116600\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1810\t2129\tATGCA\tOs01g0202200\t-\tscaffoldA\t1869\t2230\tOs02g0100300\texon\t+",
                "scaffoldA\t4753\t5150\tAA\tOs03g0116600\t-\tscaffoldA\t4963\t5215\tOs03g0116600\tintron\t+",
                "Chr2\t537\t970\tT\tgene7\t+\tChr2\t301\t775\tOs03g0116600\texon\t-",
                "Chr4\t3330\t3500\tATT\tOs02g0100300\t+\tChr4\t3281\t3424\tOs01g0100100\texon\t+",
                "Chr2\t6430\t6879\tCCGGATTA\tOs01g0100100\t+\tChr2\t6695\t6845\tgene7\tintron\t+",
                "Chr2\t6430\t6879\tCCGGATTA\tOs01g0100100\t+\tChr2\t6695\t6845\tOs03g0116600\texon\t+",
                "scaffoldA\t6271\t6456\tATGCA\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t3320\t3490\tA\tOs01g0202200\t+\tChr4\t3281\t3424\tOs01g0100100\texon\t+",
                "Chr2\t522\t955\tA\tOs01g0202200\t+\tChr2\t301\t775\tOs03g0116600\texon\t-",
                "Chr4\t6787\t6918\tC\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t3021\t3218\tTTGA\tOs01g0202200\t+\t.\t-1\t-1\t.\t-1\t.",
                "Chr2\t6424\t6873\tATT\tgene7\t+\tChr2\t6695\t6845\tgene7\tintron\t+",
                "Chr2\t6424\t6873\tATT\tgene7\t+\tChr2\t6695\t6845\tOs03g0116600\texon\t+",
                "chr1\t5649\t5808\tAT\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t1825\t2144\tATGCA\tOs01g0202200\t-\tscaffoldA\t1869\t2230\tOs02g0100300\texon\t+",
                "chr1\t5658\t5817\tGGCCT\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t3006\t3203\tC\tOs02g0100300\t-\t.\t-1\t-1\t.\t-1\t.",
                "scaffoldA\t6871\t7176\tCG\tgene7\t-\t.\t-1\t-1\t.\t-1\t.",
                "chr1\t446\t910\tCCGGATTA\tOs01g0202200\t-\t.\t-1\t-1\t.\t-1\t.",
                "Chr4\t3295\t3560\tAT\tOs02g0100300\t-\tChr4\t3281\t3424\tOs01g0100100\texon\t+",
            ],
        ),
    ]
    for index, expected in baked:
        a_rows, b_rows = _gen_loj_case(rng, index)
        a = _write_bed(tmp_path, f"a{index}.bed", a_rows)
        b = _write_bed(tmp_path, f"b{index}.bed", b_rows)
        assert loj_intersect(a, b) == expected, index
