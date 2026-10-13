"""Tests for dnallm.utils.genomic_intervals — bedtools-parity interval evaluators.

Oracle literals are baked verbatim from the two parity spikes (quick tasks
261010-wx2 jaccard and 261011-0a3 loj-intersect), which compared 5008 cases
against live bedtools v2.31.1 with zero mismatches. No bedtools binary, no
subprocess, no skip machinery and no custom markers appear anywhere in this
file: the tests are platform-neutral by construction (the point of the
module under test).
"""

from pathlib import Path

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
