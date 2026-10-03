"""Tests for dnallm.utils.genomic_coords — shared coordinate/chrom-name helpers."""

import importlib
import os
import sys
from pathlib import Path

import pytest

from dnallm.utils.genomic_coords import (
    fetch_sequence,
    gff1_to_half_open,
    half_open_to_gff1,
    normalize_chrom,
    parse_gff_attributes,
    slice_gff_rows,
)

# 180 bp mini chromosome (uppercase ASCII, TAIR10-style >Chr1 header).
FASTA_SEQ = (
    "ACGTTGCAAGGCTTACGATCGATCGGATTACAGCATCGACTAGCGATTAAGGCTAGCTAGCTAGGCTTAGGCATCGATTCGATCGTAGCTAGCT"
    "AGCTAGCTAGGCTTAGGCATCGATTCGATCGTAGCTAGCTAGCTAACGTTGCAAGGCTTACGATCGATCGGATTACAGCATCGACTAGCGATT"
)

# Inline GFF3 rows (tab-separated): TAIR10 + PlantDHS shapes, including the
# trailing-';' comma-Parent CDS row quirk (197,160 rows like it in TAIR10).
GFF_ROWS = [
    "##gff-version 3",
    None,  # blank/None rows are skipped, never crash the filter
    "",
    "Chr1\tjianglab\tDHSs\t3064\t3280\t.\t.\t.\tName=TAIR10_Chr1:3064-3280",  # idx 3
    "Chr1\tTAIR10\tgene\t3631\t5899\t.\t+\t.\tID=AT1G01010",  # idx 4
    "Chr1\tTAIR10\tmRNA\t3631\t5899\t.\t+\t.\tID=AT1G01010.1;Parent=AT1G01010",  # idx 5
    "Chr1\tTAIR10\tCDS\t3760\t3913\t.\t+\t0\tParent=AT1G01010.1,AT1G01010.1-Protein;",  # idx 6
    "Chr1\tTAIR10\tgene\t6788\t9130\t.\t-\t.\tID=AT1G01020",  # idx 7
    "Chr2\tTAIR10\tgene\t100\t200\t.\t+\t.\tID=AT2G01010",  # idx 8 — other chrom
]


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome-name normalization
# ─────────────────────────────────────────────────────────────────────────────


def test_normalize_chrom_tair_style():
    # Bare numeric (Ensembl-style) and chr-prefixed inputs all become TAIR ChrN;
    # the chr prefix is case-insensitive, the token is preserved verbatim.
    assert normalize_chrom("1", style="tair") == "Chr1"
    assert normalize_chrom("Chr1") == "Chr1"  # default style is tair
    assert normalize_chrom("chr1") == "Chr1"
    assert normalize_chrom("CHR1") == "Chr1"
    # Organelle names pass through untouched in both styles
    assert normalize_chrom("ChrC") == "ChrC"
    assert normalize_chrom("ChrM") == "ChrM"


def test_normalize_chrom_ensembl_style():
    assert normalize_chrom("Chr1", style="ensembl") == "1"
    assert normalize_chrom("chr1", style="ensembl") == "1"
    assert normalize_chrom("1", style="ensembl") == "1"
    # Organelle tokens have no numeric Ensembl counterpart: pass through
    assert normalize_chrom("ChrC", style="ensembl") == "ChrC"
    assert normalize_chrom("ChrM", style="ensembl") == "ChrM"


def test_normalize_chrom_rejects_unknown_forms():
    # Empty and non-string names raise with a matchable message
    with pytest.raises(ValueError, match=r"non-empty"):
        normalize_chrom("")
    with pytest.raises(ValueError, match=r"non-empty"):
        normalize_chrom(None)  # type: ignore[arg-type]
    # Undocumented name forms are never best-effort renamed (Pitfall-11 failure
    # mode is silence; guessing recreates it)
    for bad in ("scaffold_1", "chromosome1", "AT1G01010", "Chr", "Chr-1", "C", "M", "-1", "1 "):
        with pytest.raises(ValueError, match=r"Unrecognized chromosome"):
            normalize_chrom(bad)
    # Undocumented styles are rejected too
    with pytest.raises(ValueError, match=r"style"):
        normalize_chrom("Chr1", style="ucsc")


def test_normalize_chrom_rejects_non_ascii_digits():
    # IN-02: full-width/superscript/Arabic-Indic digits satisfy str.isdigit()
    # but are NOT bare Ensembl numerics — accepting them would silently rename
    # to a lookalike chromosome. They must raise in BOTH styles, never be renamed.
    for bad in ("１", "２", "²", "٣"):  # ruff: ignore[ambiguous-unicode-character-string] — the lookalikes ARE the fixture
        with pytest.raises(ValueError, match=r"Unrecognized chromosome"):
            normalize_chrom(bad)
        with pytest.raises(ValueError, match=r"Unrecognized chromosome"):
            normalize_chrom(bad, style="ensembl")
    # Positive control: ASCII bare numerics keep working in both styles
    assert normalize_chrom("1") == "Chr1"
    assert normalize_chrom("1", style="ensembl") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversions (GFF3 1-based closed <-> BED 0-based half-open)
# ─────────────────────────────────────────────────────────────────────────────


def test_gff1_to_half_open_boundaries():
    # Length-1 GFF3 feature (start == end) becomes a width-1 half-open interval
    assert gff1_to_half_open(1, 1) == (0, 1)
    assert gff1_to_half_open(10, 20) == (9, 20)


def test_touching_features_map_to_adjacent_intervals():
    # Exactly-touching 1-based closed features map to adjacent, non-overlapping
    # half-open intervals — they never merge, collide, or overlap
    first = gff1_to_half_open(10, 20)
    second = gff1_to_half_open(21, 30)
    assert first == (9, 20)
    assert second == (20, 30)
    assert first[1] == second[0]  # adjacency
    assert min(first[1], second[1]) - max(first[0], second[0]) <= 0  # no overlap


def test_gff1_to_half_open_rejects_invalid_coordinates():
    for start, end in [(0, 5), (-3, 10), (20, 10)]:
        with pytest.raises(ValueError, match=r"coordinate"):
            gff1_to_half_open(start, end)
    with pytest.raises(ValueError, match=r"integer"):
        gff1_to_half_open(1.5, 10)
    with pytest.raises(ValueError, match=r"integer"):
        gff1_to_half_open("1", 10)  # type: ignore[arg-type]


def test_half_open_to_gff1_round_trip():
    assert half_open_to_gff1(0, 1) == (1, 1)
    assert half_open_to_gff1(9, 20) == (10, 20)
    # Round-trip property over a small vector, touching features included
    for start, end in [(1, 1), (10, 20), (21, 30), (100, 5000), (30_427_671, 30_427_671)]:
        assert half_open_to_gff1(*gff1_to_half_open(start, end)) == (start, end)


def test_half_open_to_gff1_rejects_invalid_coordinates():
    # Zero-width intervals have no 1-based closed representation
    with pytest.raises(ValueError, match=r"empty"):
        half_open_to_gff1(5, 5)
    with pytest.raises(ValueError, match=r"coordinate"):
        half_open_to_gff1(-1, 10)
    with pytest.raises(ValueError, match=r"integer"):
        half_open_to_gff1(0, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# GFF3 column-9 attribute parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_gff_attributes_tair10_cds_row():
    # Trailing ';' after a comma-joined double Parent — the TAIR10 CDS quirk
    attrs = parse_gff_attributes("ID=AT1G01010.1;Parent=AT1G01010,AT1G01010-Protein;")
    assert attrs == {
        "ID": ["AT1G01010.1"],
        "Parent": ["AT1G01010", "AT1G01010-Protein"],
    }
    # No ghost empty keys survive the trailing separator
    assert all(attrs.keys())
    # Attribute insertion order is preserved (never re-sorted)
    assert list(attrs) == ["ID", "Parent"]


def test_parse_gff_attributes_whitespace_order_and_empty():
    attrs = parse_gff_attributes("Name=TAIR10_Chr1:3064-3280; ID = spaced ; Note=a,b,c")
    assert list(attrs) == ["Name", "ID", "Note"]
    assert attrs["ID"] == ["spaced"]
    assert attrs["Note"] == ["a", "b", "c"]
    assert parse_gff_attributes("") == {}
    assert parse_gff_attributes(".") == {}


def test_parse_gff_attributes_rejects_malformed_segment():
    with pytest.raises(ValueError, match=r"Malformed GFF3 attribute"):
        parse_gff_attributes("ID=x;noequals")


# ─────────────────────────────────────────────────────────────────────────────
# FASTA sequence fetching (pyfastx, dev extra)
# ─────────────────────────────────────────────────────────────────────────────


def _write_fasta(tmp_path, name="mini.fas"):
    fa_path = tmp_path / name
    body = "\n".join(FASTA_SEQ[i : i + 80] for i in range(0, len(FASTA_SEQ), 80))
    fa_path.write_text(f">Chr1 test chrom\n{body}\n")
    return fa_path


def _open(fa_path):
    import pyfastx

    return pyfastx.Fasta(str(fa_path))


def test_fetch_sequence_one_based_inclusive(tmp_path):
    # pyfastx fetch is 1-based inclusive, identical to GFF3: (10, 20) == seq[9:20]
    fa_path = _write_fasta(tmp_path)
    assert fetch_sequence(_open(fa_path), "Chr1", 10, 20) == FASTA_SEQ[9:20]
    # Ensembl-style query chrom is normalized before the lookup
    assert fetch_sequence(_open(fa_path), "1", 10, 20) == FASTA_SEQ[9:20]
    # A path is accepted too (pyfastx imported lazily inside the helper)
    assert fetch_sequence(str(fa_path), "Chr1", 10, 20) == FASTA_SEQ[9:20]


def test_fetch_sequence_uppercase_default(tmp_path):
    lower = tmp_path / "lower.fas"
    lower.write_text(">Chr1\n" + "acgt" * 10 + "\n")
    fa = _open(lower)
    assert fetch_sequence(fa, "Chr1", 1, 8) == "ACGTACGT"
    assert fetch_sequence(fa, "Chr1", 1, 8, uppercase=False) == "acgtacgt"


def test_fetch_sequence_guards(tmp_path):
    fa = _open(_write_fasta(tmp_path))
    # Unknown chromosome in the index
    with pytest.raises(ValueError, match=r"Unknown chromosome"):
        fetch_sequence(fa, "Chr9", 10, 20)
    # Invalid coordinates never reach the index
    with pytest.raises(ValueError, match=r"coordinate"):
        fetch_sequence(fa, "Chr1", 20, 10)


def test_fetch_sequence_empty_result_guard():
    class _EmptyFetch:
        def fetch(self, chrom, span):
            return ""

    # SHOW-02: a silently-empty fetch is structurally impossible
    with pytest.raises(ValueError, match=r"[Ee]mpty sequence"):
        fetch_sequence(_EmptyFetch(), "Chr1", 10, 20)


def test_fetch_sequence_path_branch_creates_no_fxi_sidecar(tmp_path):
    # IN-01: the path branch is read-only — after fetching from a fresh FASTA
    # no pyfastx .fxi index sidecar is left beside it, and the FASTA itself
    # is untouched.
    fa_path = _write_fasta(tmp_path)
    original = fa_path.read_bytes()
    assert fetch_sequence(str(fa_path), "Chr1", 10, 20) == FASTA_SEQ[9:20]
    assert not (tmp_path / "mini.fas.fxi").exists()
    assert fa_path.read_bytes() == original


def test_fetch_sequence_path_branch_preserves_preexisting_fxi(tmp_path):
    # IN-01: a .fxi sidecar that existed before the call belongs to the
    # caller (or the user); it is reused and never deleted by the path branch.
    import pyfastx

    fa_path = _write_fasta(tmp_path)
    index = pyfastx.Fasta(str(fa_path))  # builds mini.fas.fxi
    del index  # release the handle; the sidecar stays on disk (caller-owned)
    sidecar = tmp_path / "mini.fas.fxi"
    assert sidecar.exists()
    assert fetch_sequence(str(fa_path), "Chr1", 10, 20) == FASTA_SEQ[9:20]
    assert sidecar.exists()


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="fd accounting needs /proc (Linux only)"
)
def test_fetch_sequence_path_branch_leaks_no_file_descriptors(tmp_path):
    # IN-01: pyfastx.Fasta has no close()/context manager; the path branch
    # must drop its reference so the held file descriptors are released
    # before the function returns.
    fa_path = _write_fasta(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    assert fetch_sequence(str(fa_path), "Chr1", 10, 20) == FASTA_SEQ[9:20]
    assert len(os.listdir("/proc/self/fd")) == before


# ─────────────────────────────────────────────────────────────────────────────
# GFF3 row slicing
# ─────────────────────────────────────────────────────────────────────────────


def test_slice_gff_rows_filters_and_preserves_order():
    rows = slice_gff_rows(GFF_ROWS, "Chr1", 3000, 6000)
    # Rows overlapping the 1-based closed locus, input order preserved,
    # comment/blank rows skipped, other chromosomes excluded
    assert rows == [GFF_ROWS[3], GFF_ROWS[4], GFF_ROWS[5], GFF_ROWS[6]]
    # Ensembl-style query chrom matches TAIR-named rows
    assert slice_gff_rows(GFF_ROWS, "1", 3000, 6000) == rows


def test_slice_gff_rows_closed_interval_boundaries():
    # Closed-interval edges: features merely touching the locus are included;
    # an interior CDS (3760-3913) outside the locus is not
    assert slice_gff_rows(GFF_ROWS, "Chr1", 5899, 6788) == [GFF_ROWS[4], GFF_ROWS[5], GFF_ROWS[7]]


def test_slice_gff_rows_require_nonempty():
    # Without the flag an empty locus result is legitimate
    assert slice_gff_rows(GFF_ROWS, "Chr3", 1, 100) == []
    # With require_nonempty the silent-empty path raises (SHOW-02 guard)
    with pytest.raises(ValueError, match=r"No GFF3 rows"):
        slice_gff_rows(GFF_ROWS, "Chr3", 1, 100, require_nonempty=True)


def test_slice_gff_rows_rejects_invalid_input():
    with pytest.raises(ValueError, match=r"coordinate"):
        slice_gff_rows(GFF_ROWS, "Chr1", 20, 10)
    with pytest.raises(ValueError, match=r"Malformed GFF3 row"):
        slice_gff_rows(["Chr1\ttoo\tfew"], "Chr1", 1, 10)


def test_slice_gff_rows_crlf_and_cr_terminators_match_lf_results():
    # Windows (\r\n) and old-Mac (\r) GFF files carry no information in their
    # line endings: terminator twins must slice identically to LF input, and
    # no terminator may survive into the returned rows.
    expected = [GFF_ROWS[3], GFF_ROWS[4], GFF_ROWS[5], GFF_ROWS[6]]
    assert slice_gff_rows(GFF_ROWS, "Chr1", 3000, 6000) == expected  # LF twin
    for term in ("\n", "\r\n", "\r"):
        twin = [row + term if isinstance(row, str) and row else row for row in GFF_ROWS]
        rows = slice_gff_rows(twin, "Chr1", 3000, 6000)
        assert rows == expected
        assert all("\r" not in row for row in rows)
    # Column-9 attribute values from a returned CRLF row parse clean
    crlf_twin = [row + "\r\n" if isinstance(row, str) and row else row for row in GFF_ROWS]
    cds = slice_gff_rows(crlf_twin, "Chr1", 3000, 6000)[3]
    attrs = parse_gff_attributes(cds.split("\t")[8])
    assert attrs["Parent"] == ["AT1G01010.1", "AT1G01010.1-Protein"]
    assert all("\r" not in value for values in attrs.values() for value in values)


def test_slice_gff_rows_rejects_embedded_carriage_return():
    # A mid-row \r is not a terminator — it would silently corrupt whichever
    # column contains it, so the parser raises instead (loud-error contract),
    # regardless of whether the row's chromosome matches the query.
    poisoned = "Chr1\tTAIR10\tgene\t3631\t5899\t.\t+\t.\tID=AT1G01010\rNote=corrupt\n"
    with pytest.raises(ValueError, match=r"Carriage return embedded"):
        slice_gff_rows([poisoned], "Chr1", 3000, 6000)
    with pytest.raises(ValueError, match=r"Carriage return embedded"):
        slice_gff_rows([poisoned], "Chr2", 3000, 6000)


# ─────────────────────────────────────────────────────────────────────────────
# Import purity (pyfastx is a dev extra)
# ─────────────────────────────────────────────────────────────────────────────


def test_module_import_is_pyfastx_free():
    # pyfastx is imported lazily inside fetch_sequence only; importing the
    # helper module (and therefore dnallm.utils) must never import it.
    saved_pyfastx = sys.modules.get("pyfastx")
    saved_module = sys.modules.get("dnallm.utils.genomic_coords")
    sys.modules.pop("pyfastx", None)
    sys.modules.pop("dnallm.utils.genomic_coords", None)
    sys.modules["pyfastx"] = None  # any import attempt now raises ImportError
    try:
        mod = importlib.import_module("dnallm.utils.genomic_coords")
        assert mod.normalize_chrom("1") == "Chr1"
    finally:
        del sys.modules["pyfastx"]
        if saved_pyfastx is not None:
            sys.modules["pyfastx"] = saved_pyfastx
        if saved_module is not None:
            sys.modules["dnallm.utils.genomic_coords"] = saved_module
