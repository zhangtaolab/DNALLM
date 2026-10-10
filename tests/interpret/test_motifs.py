"""Fast-lane tests for ``dnallm.interpret.motifs`` (MOTIF-01 / REV-10).

Coverage lanes (one class per function under test):
- MEME parsing: strict parse-or-reject against the committed JASPAR-shaped
  fixture plus malformed variants
- Log-odds matrix: FIMO pseudocount formula (counts and nsites==0 branches)
- P-value table: hand-computed exact-DP anchors (uniform-PWM guard, 1-column
  analytic table, 2-column analytic convolution under two backgrounds)
- Threshold inversion: distribution-edge semantics (tail exactly at 1e-4,
  everything-significant, nothing-passes)
- Single-strand scan core slice
- (Task 2 lanes: strands, GC background, BH/E-values, edges, CIS-BP)
- (Task 3 lanes: JASPAR client, mocked + live)
- (Task 4 lane: HBG1/BCL11A golden harness)
"""

import math
from pathlib import Path

import pytest

from dnallm.interpret import motifs
from dnallm.interpret.motifs import (
    Motif,
    log_odds_matrix,
    parse_meme,
    pvalue_table,
    scan_single_strand,
    threshold_bits,
)

FIXTURES = Path(__file__).parent / "fixtures"
UNIFORM_BG = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


def _sharp_motif(consensus: str, *, motif_id: str = "M1", nsites: int = 20000) -> Motif:
    """PWM whose consensus letter carries 0.4 and the others 0.2 per row."""
    rows = []
    for letter in consensus:
        row = [0.2, 0.2, 0.2, 0.2]
        row["ACGT".index(letter)] = 0.4
        rows.append(row)
    return Motif(motif_id=motif_id, name=motif_id, freq_rows=rows, nsites=nsites)


def _minimal_meme_text(
    *,
    header_fields: dict[str, str] | None = None,
    rows: list[str] | None = None,
) -> str:
    """Composable strict-grammar MEME document for malformed-input tests."""
    fields = {"alength": "4", "w": "2", "nsites": "10"}
    fields.update(header_fields or {})
    if rows is None:
        rows = ["0.1 0.2 0.3 0.4", "0.4 0.3 0.2 0.1"]
    lines = [
        "MEME version 4",
        "",
        "ALPHABET= ACGT",
        "strands: + -",
        "",
        "MOTIF MA0001.1 TESTTF",
        (
            f"letter-probability matrix: alength= {fields['alength']} "
            f"w= {fields['w']} nsites= {fields['nsites']} E= 0"
        ),
        *[f"  {row}" for row in rows],
        "",
        "URL https://jaspar.elixir.no/matrix/MA0001.1",
    ]
    return "\n".join(lines) + "\n"


class TestModuleDocstring:
    """REQUIREMENTS acceptance: the D-01 calibration choice is documented honestly."""

    def test_docstring_documents_calibration_choices(self):
        doc = motifs.__doc__.lower()
        for term in (
            "exact",
            "dynamic programming",
            "pssm_range",
            "pseudocount",
            "zero-order",
            "benjamini-hochberg",
        ):
            assert term in doc, f"module docstring must document '{term}' (D-01 honesty)"


class TestParseMeme:
    """Strict MEME grammar: committed JASPAR-shaped fixture + parse-or-reject."""

    def test_parse_meme_fixture_matches_live_probed_shape(self):
        text = (FIXTURES / "meme_motif.txt").read_text(encoding="utf-8")
        parsed = parse_meme(text)
        assert len(parsed) == 1
        motif = parsed[0]
        assert motif.motif_id == "MA2324.1"
        assert motif.name == "BCL11A"
        assert motif.width == 7
        assert motif.nsites == 6265
        assert motif.freq_rows[0] == pytest.approx([0.054749, 0.801915, 0.094334, 0.049002])
        for row in motif.freq_rows:
            assert sum(row) == pytest.approx(1.0, abs=1e-6)

    def test_parse_meme_embedded_uniform_background_line_is_parsed_and_ignored(self):
        text = (FIXTURES / "meme_motif.txt").read_text(encoding="utf-8")
        assert "Background letter frequencies" in text
        # The Motif record carries no background: the scan background is
        # GC-matched from the target windows, never JASPAR's stock 0.25 line.
        motif = parse_meme(text)[0]
        assert not hasattr(motif, "background")
        assert motif.width == 7

    def test_parse_meme_two_motif_blocks(self):
        text = _minimal_meme_text() + _minimal_meme_text().replace("MA0001.1", "MA0002.1")
        parsed = parse_meme(text)
        assert [m.motif_id for m in parsed] == ["MA0001.1", "MA0002.1"]

    def test_parse_meme_name_defaults_to_id(self):
        text = _minimal_meme_text().replace("MOTIF MA0001.1 TESTTF", "MOTIF MA0001.1")
        motif = parse_meme(text)[0]
        assert motif.name == "MA0001.1"

    def test_parse_meme_rejects_empty_text(self):
        with pytest.raises(ValueError, match=r"MEME motif text is empty"):
            parse_meme("   \n  ")

    def test_parse_meme_rejects_bad_alength(self):
        text = _minimal_meme_text(header_fields={"alength": "5"})
        with pytest.raises(ValueError, match=r"alength must be 4 .* got 5"):
            parse_meme(text)

    def test_parse_meme_rejects_missing_w_field(self):
        text = _minimal_meme_text().replace("w= 2 ", "")
        with pytest.raises(ValueError, match=r"matrix header must carry alength/w/nsites"):
            parse_meme(text)

    def test_parse_meme_rejects_width_over_cap(self):
        text = _minimal_meme_text(header_fields={"w": "101"})
        with pytest.raises(ValueError, match=r"motif width w=101 outside"):
            parse_meme(text)

    def test_parse_meme_rejects_negative_nsites(self):
        text = _minimal_meme_text(header_fields={"nsites": "-5"})
        with pytest.raises(ValueError, match=r"nsites must be >= 0, got -5"):
            parse_meme(text)

    def test_parse_meme_rejects_non_numeric_row(self):
        text = _minimal_meme_text(rows=["0.1 abc 0.3 0.4", "0.4 0.3 0.2 0.1"])
        with pytest.raises(ValueError, match=r"non-numeric probability row"):
            parse_meme(text)

    def test_parse_meme_rejects_out_of_range_row(self):
        text = _minimal_meme_text(rows=["1.5 0.0 0.0 0.0", "0.4 0.3 0.2 0.1"])
        with pytest.raises(ValueError, match=r"values must be in \[0, 1\]"):
            parse_meme(text)

    def test_parse_meme_rejects_row_with_wrong_arity(self):
        text = _minimal_meme_text(rows=["0.1 0.2 0.3", "0.4 0.3 0.2 0.1"])
        with pytest.raises(ValueError, match=r"probability row must have 4 values"):
            parse_meme(text)

    def test_parse_meme_rejects_row_count_shortfall(self):
        text = _minimal_meme_text(
            header_fields={"w": "3"},
            rows=["0.1 0.2 0.3 0.4", "0.4 0.3 0.2 0.1", ""],
        ).replace("0.4 0.3 0.2 0.1\n\n", "0.4 0.3 0.2 0.1\n")
        with pytest.raises(ValueError, match=r"incomplete MOTIF block"):
            parse_meme(text)

    def test_parse_meme_rejects_matrix_without_motif_line(self):
        text = _minimal_meme_text().replace("MOTIF MA0001.1 TESTTF\n", "")
        with pytest.raises(ValueError, match=r"without a preceding MOTIF line"):
            parse_meme(text)

    def test_parse_meme_rejects_motif_block_without_matrix(self):
        text = _minimal_meme_text().replace(
            "MOTIF MA0001.1 TESTTF", "MOTIF MA0001.1 TESTTF\nMOTIF MA0002.1 OTHER"
        )
        with pytest.raises(ValueError, match=r"has no\s+letter-probability matrix"):
            parse_meme(text)

    def test_parse_meme_rejects_unrecognized_line(self):
        text = _minimal_meme_text() + "some random junk\n"
        with pytest.raises(ValueError, match=r"unrecognized content"):
            parse_meme(text)

    def test_parse_meme_rejects_non_acgt_alphabet(self):
        text = _minimal_meme_text().replace("ALPHABET= ACGT", "ALPHABET= ACGTN")
        with pytest.raises(ValueError, match=r"alphabet must be ACGT"):
            parse_meme(text)

    def test_parse_meme_rejects_malformed_background_line(self):
        text = _minimal_meme_text().replace(
            "MOTIF MA0001.1 TESTTF",
            "Background letter frequencies\nA 0.25 C 0.25 G\nMOTIF MA0001.1 TESTTF",
        )
        with pytest.raises(ValueError, match=r"letter/frequency\s+pairs"):
            parse_meme(text)

    def test_parse_meme_rejects_trailing_background_header(self):
        text = _minimal_meme_text() + "Background letter frequencies\n"
        with pytest.raises(ValueError, match=r"ends after 'Background letter frequencies'"):
            parse_meme(text)


class TestLogOddsMatrix:
    """FIMO pseudocount formula: counts branch and nsites==0 probability branch."""

    def test_log_odds_counts_branch_formula(self):
        matrix = log_odds_matrix([[1.0, 0.0, 0.0, 0.0]], nsites=1000, bg=UNIFORM_BG)
        p_a = (1.0 * 1000 + 0.1 * 0.25) / 1000.1
        p_c = (0.0 * 1000 + 0.1 * 0.25) / 1000.1
        assert matrix[0][0] == pytest.approx(math.log2(p_a / 0.25))
        assert matrix[0][1] == pytest.approx(math.log2(p_c / 0.25))
        assert matrix[0][0] > 0.0 > matrix[0][1]

    def test_log_odds_nsites_zero_probability_branch(self):
        matrix = log_odds_matrix([[0.25, 0.25, 0.25, 0.25]], nsites=0, bg=UNIFORM_BG)
        assert matrix[0] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)


class TestPvalueTable:
    """Hand-computed exact-DP anchors for the FIMO null distribution."""

    def test_pvalue_table_uniform_pwm_raises(self):
        scores = [[0.0, 0.0, 0.0, 0.0]]
        with pytest.raises(ValueError, match=r"no score variation \(uniform PWM\)"):
            pvalue_table(scores, UNIFORM_BG)

    def test_pvalue_table_one_column_anchor(self):
        scores = log_odds_matrix([[1.0, 0.0, 0.0, 0.0]], nsites=1000, bg=UNIFORM_BG)
        pv, scale, offset, max_scaled = pvalue_table(scores, UNIFORM_BG)
        assert max_scaled == 100
        # Only the A column scores the max scaled value 100: p(A)=0.25.
        assert pv[100] == pytest.approx(0.25)
        # Everything else lands on scaled 0 with mass 0.75; the minimum score
        # has p exactly 1.0 (Pr(score >= min) == 1).
        assert pv[0] == pytest.approx(1.0)
        assert pv[1] == pytest.approx(0.25)
        assert pv[50] == pytest.approx(0.25)
        # Nothing passes p<1e-4: the threshold sits one past the maximum.
        bits = threshold_bits(pv, scale, offset, 1)
        assert bits == pytest.approx(101 / scale + offset)

    def test_pvalue_table_two_column_convolution_uniform_bg(self):
        scores = [[0.0, 100.0, 0.0, 100.0], [0.0, 0.0, 100.0, 100.0]]
        pv, _scale, _offset, max_scaled = pvalue_table(scores, UNIFORM_BG)
        assert max_scaled == 200
        # Hand-computed convolution: pdf {0: .25, 100: .5, 200: .25}.
        assert pv[0] == pytest.approx(1.0)
        assert pv[1] == pytest.approx(0.75)
        assert pv[100] == pytest.approx(0.75)
        assert pv[101] == pytest.approx(0.25)
        assert pv[200] == pytest.approx(0.25)

    def test_pvalue_table_two_column_convolution_skewed_bg(self):
        bg = {"A": 0.1, "C": 0.2, "G": 0.3, "T": 0.4}
        scores = [[0.0, 100.0, 0.0, 100.0], [0.0, 0.0, 100.0, 100.0]]
        pv, _scale, _offset, _max = pvalue_table(scores, bg)
        # Hand-computed: pdf {0: .12, 100: .46, 200: .42}.
        assert pv[0] == pytest.approx(1.0)
        assert pv[100] == pytest.approx(0.88)
        assert pv[101] == pytest.approx(0.42)
        assert pv[200] == pytest.approx(0.42)


class TestThresholdBits:
    """Inversion ``x/scale + w*offset`` at the distribution edges."""

    def test_threshold_bits_tail_exactly_at_threshold_boundary(self):
        # pv[1] == 1e-4 exactly is NOT under the strict p < 1e-4 threshold.
        pv = [1.0, 1e-4, 1e-5]
        assert threshold_bits(pv, 4.0, 0.5, 2) == pytest.approx(2 / 4.0 + 2 * 0.5)

    def test_threshold_bits_everything_significant_edge(self):
        pv = [1.0, 1e-5, 1e-6]
        # Minimal passing scaled score is 1 (pv[0] == 1.0 can never pass).
        assert threshold_bits(pv, 4.0, 0.5, 2) == pytest.approx(1 / 4.0 + 2 * 0.5)

    def test_threshold_bits_nothing_passes_edge(self):
        pv = [1.0, 0.5, 0.2]
        # No achievable score passes: threshold lands one past the maximum.
        assert threshold_bits(pv, 4.0, 0.5, 2) == pytest.approx(3 / 4.0 + 2 * 0.5)

    def test_threshold_bits_mid_distribution_inversion(self):
        pv = [1.0, 0.5, 1e-5]
        assert threshold_bits(pv, 4.0, 0.5, 2) == pytest.approx(2 / 4.0 + 2 * 0.5)


class TestScanSingleStrand:
    """Core scoring slice: one window, one motif, '+' strand, p threshold only."""

    def test_single_strand_scan_pvalue_threshold_hits(self):
        motif = _sharp_motif("CCGGGCC")
        window = "AAAACCGGGCCAAAA"
        hits = scan_single_strand(window, motif, background=UNIFORM_BG)
        assert len(hits) == 1
        hit = hits[0]
        assert hit["motif_id"] == "M1"
        assert hit["start"] == 4
        assert hit["end"] == 11
        assert hit["strand"] == "+"
        # Exact-consensus p is the product of uniform background frequencies.
        assert hit["p"] == pytest.approx(0.25**7)
        # Reported score uses the FIMO inversion over the scaled DP score.
        scores = log_odds_matrix(motif.freq_rows, motif.nsites, UNIFORM_BG)
        _pv, scale, offset, _max = pvalue_table(scores, UNIFORM_BG)
        assert hit["score_bits"] == pytest.approx(700 / scale + 7 * offset)

    def test_single_strand_scan_threshold_bits_consistency(self):
        motif = _sharp_motif("CCGGGCC")
        hits = scan_single_strand("AAAACCGGGCCAAAA", motif, background=UNIFORM_BG)
        scores = log_odds_matrix(motif.freq_rows, motif.nsites, UNIFORM_BG)
        pv, scale, offset, _max = pvalue_table(scores, UNIFORM_BG)
        reported_threshold = threshold_bits(pv, scale, offset, 7)
        # Every reported hit sits at or above the reporting threshold in bits.
        assert all(hit["score_bits"] >= reported_threshold for hit in hits)

    def test_single_strand_scan_returns_empty_when_no_pvalue_passes(self):
        # A weak w=4 motif cannot reach p < 1e-4 anywhere (min p = 0.25**4).
        rows = [[0.3, 0.25, 0.25, 0.2]] * 4
        weak = Motif(motif_id="W", name="W", freq_rows=rows, nsites=20000)
        hits = scan_single_strand("ACGTACGTACGTACGTACGT", weak, background=UNIFORM_BG)
        assert hits == []

    def test_single_strand_scan_gc_matched_default_background(self):
        # No explicit background: GC-matched from the window (never uniform).
        motif = _sharp_motif("CCGGGCC")
        window = "AAAATAATACCGGGCCAAATAAT"
        hits = scan_single_strand(window, motif)
        site_hits = [h for h in hits if h["start"] == 9 and h["end"] == 16]
        assert site_hits, "consensus site at [9, 16) must be reported"
