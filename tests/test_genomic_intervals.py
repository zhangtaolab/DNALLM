"""Tests for dnallm.utils.genomic_intervals — bedtools-parity interval evaluators.

Oracle literals are baked verbatim from the two parity spikes (quick tasks
261010-wx2 jaccard and 261011-0a3 loj-intersect), which compared 5008 cases
against live bedtools v2.31.1 with zero mismatches. No bedtools binary, no
subprocess, no skip machinery and no custom markers appear anywhere in this
file: the tests are platform-neutral by construction (the point of the
module under test).
"""

from pathlib import Path

from dnallm.utils.genomic_intervals import jaccard, parse_bed


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


def test_package_surface_reexports_jaccard_and_parse_bed(tmp_path):
    # The dnallm.utils package surface (not the submodule) re-exports the
    # public entry points; prove the wiring end-to-end by calling through it.
    import dnallm.utils

    from dnallm.utils import jaccard as pkg_jaccard
    from dnallm.utils import parse_bed as pkg_parse_bed

    assert pkg_jaccard is jaccard
    assert pkg_parse_bed is parse_bed
    assert "jaccard" in dnallm.utils.__all__
    assert "parse_bed" in dnallm.utils.__all__
    a = _write_bed(tmp_path, "a.bed", [("Chr1", 0, 100)])
    b = _write_bed(tmp_path, "b.bed", [("Chr1", 100, 200)])
    assert pkg_jaccard(a, b)["union"] == 200
    assert len(pkg_parse_bed(a)) == 1
