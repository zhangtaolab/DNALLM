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
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from dnallm.interpret import motifs
from dnallm.interpret.motifs import (
    Motif,
    fetch_meme_motif,
    gc_background,
    log_odds_matrix,
    parse_cisbp,
    parse_meme,
    pvalue_table,
    scan,
    scan_single_strand,
    search_motifs,
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


class TestStrandScanning:
    """Both-strand mechanics: palindromic adjacency + non-palindromic canary."""

    def test_strand_palindromic_motif_hits_both_strands_same_coordinates(self):
        # "AGTATACT" is its own reverse complement; its PWM built from
        # _sharp_motif is therefore column-symmetric.
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        result = scan(["GGGAGTATACTGGG"], [motif], background=UNIFORM_BG)
        site_rows = [h for h in result.hits if h["start"] == 3 and h["end"] == 11]
        # FIMO convention (adjacency edge predicate): BOTH strand rows are
        # reported at the same coordinates -- never deduplicated or merged.
        assert {(h["strand"]) for h in site_rows} == {"+", "-"}
        assert len(site_rows) == 2
        assert all(h["p"] == pytest.approx(0.25**8) for h in site_rows)

    def test_strand_palindromic_tied_p_gets_equal_q(self):
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        result = scan(["GGGAGTATACTGGG"], [motif], background=UNIFORM_BG)
        site_rows = [h for h in result.hits if h["start"] == 3]
        assert site_rows[0]["q"] == site_rows[1]["q"]

    def test_strand_non_palindromic_forward_site_only(self):
        # Revcomp-bug canary: "CCGGGCC" is not palindromic, so a window
        # carrying the forward site must yield exactly one '+' row and no
        # '-' rows.
        motif = _sharp_motif("CCGGGCC", motif_id="FWD")
        result = scan(["AAAACCGGGCCAAAA"], [motif], background=UNIFORM_BG)
        assert result.hits, "forward site must be reported"
        assert {h["strand"] for h in result.hits} == {"+"}
        assert result.hits[0]["start"] == 4
        assert result.hits[0]["end"] == 11

    def test_strand_non_palindromic_reverse_site_maps_to_forward_frame(self):
        # The same motif against the reverse-complement window: the '-' hit
        # must land at the SAME forward-window coordinates [4, 11).
        motif = _sharp_motif("CCGGGCC", motif_id="FWD")
        result = scan(["TTTTGGCCCGGTTTT"], [motif], background=UNIFORM_BG)
        assert len(result.hits) == 1
        hit = result.hits[0]
        assert hit["strand"] == "-"
        assert hit["start"] == 4
        assert hit["end"] == 11


class TestGcBackground:
    """Zero-order GC-matched background over ALL target-window bases."""

    def test_background_gc_matched_letter_frequencies(self):
        bg = gc_background(["ACGT", "ACGT"])
        assert bg == pytest.approx(UNIFORM_BG)

    def test_background_ignores_non_acgt_characters(self):
        bg = gc_background(["ACGTNnn-X"])
        assert bg == pytest.approx(UNIFORM_BG)

    def test_background_aggregates_across_all_windows(self):
        # Only the absent letter (T) hits the floor: raw A=.5, C=.25, G=.25,
        # T=BG_FLOOR -> denominator = 1 + BG_FLOOR after renormalization.
        bg = gc_background(["AAAA", "CCGG"])
        assert bg["A"] == pytest.approx(0.5 / (1.0 + motifs.BG_FLOOR))
        assert bg["C"] == pytest.approx(0.25 / (1.0 + motifs.BG_FLOOR))
        assert bg["T"] == pytest.approx(motifs.BG_FLOOR / (1.0 + motifs.BG_FLOOR))
        assert sum(bg.values()) == pytest.approx(1.0)

    def test_background_floors_absent_letters(self):
        bg = gc_background(["GCGCGCGC"])
        assert bg["A"] > 0.0
        assert bg["T"] > 0.0
        assert sum(bg.values()) == pytest.approx(1.0)
        assert bg["G"] == pytest.approx(0.5 / 1.002)

    def test_background_empty_acgt_raises(self):
        with pytest.raises(ValueError, match=r"windows contain no A/C/G/T bases"):
            gc_background(["NNNN"])

    def test_background_changes_hit_set_on_gc_skewed_windows(self):
        # GC-heavy consensus in a GC-skewed window: significant under a
        # uniform background, NOT significant under the GC-matched
        # background that the scan computes from the windows themselves.
        motif = _sharp_motif("GCGCGCG", motif_id="GC")
        window = "GGGGCGCGCGGGGG"
        uniform = scan([window], [motif], background=UNIFORM_BG)
        matched = scan([window], [motif])
        uniform_site = [h for h in uniform.hits if h["start"] == 3 and h["end"] == 10]
        matched_site = [h for h in matched.hits if h["start"] == 3 and h["end"] == 10]
        assert uniform_site, "site must be reported under the uniform background"
        assert not matched_site, "GC-heavy site must lose significance under GC match"
        assert uniform.hits != matched.hits


class TestBhFdr:
    """One BH call over the FULL window x motif x strand p-vector (D-03)."""

    def test_bh_q_values_match_full_set_single_call_semantics(self):
        motif_a = _sharp_motif("AGTATACT", motif_id="A8")  # w=8, palindromic
        motif_b = _sharp_motif("CCGGGCC", motif_id="B7")  # w=7, non-palindromic
        window = "AGTATACTCCGGGCC"
        result = scan([window], [motif_a, motif_b], background=UNIFORM_BG)
        # Tests: A -> 2 strands x (15-7)=8 positions = 16; B -> 2 x 9 = 18.
        assert result.n_tested_positions == 34
        hits_a = sorted(
            (h for h in result.hits if h["motif_id"] == "A8"), key=lambda h: h["strand"]
        )
        hits_b = [h for h in result.hits if h["motif_id"] == "B7"]
        assert len(hits_a) == 2
        assert len(hits_b) == 1
        # Sorted ps: [pA, pA, pB]. scipy BH: tied pair at ranks 1-2 -> q = p*n/2;
        # pB at rank 3 -> q = p*n/3. These exact values hold ONLY for the
        # full-set single call (a per-motif correction would give pA*16/2 and
        # pB*18/1 -- different numbers).
        n = 34
        assert hits_a[0]["q"] == pytest.approx(0.25**8 * n / 2)
        assert hits_a[1]["q"] == pytest.approx(0.25**8 * n / 2)
        assert hits_b[0]["q"] == pytest.approx(0.25**7 * n / 3)
        # Ordering: the stronger (smaller-p) hits get the smaller q.
        assert hits_a[0]["q"] < hits_b[0]["q"]


class TestEValue:
    """E-value = p x total tested positions (FIMO-paper convention, A4)."""

    def test_evalue_equals_p_times_tested_positions(self):
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        result = scan(["GGGAGTATACTGGG"], [motif], background=UNIFORM_BG)
        assert result.n_tested_positions == 14
        for hit in result.hits:
            assert hit["e_value"] == pytest.approx(hit["p"] * result.n_tested_positions)
        # And numerically: p = 0.25**8 over 14 tested positions.
        assert all(hit["e_value"] == pytest.approx(0.25**8 * 14) for hit in result.hits)


class TestScanEdges:
    """Specless edge predicates: short-window exclusion + empty-but-counted."""

    def test_edge_short_window_excluded_with_count(self):
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        result = scan(["AT", "GGGAGTATACTGGG"], [motif], background=UNIFORM_BG)
        assert result.n_excluded_short_windows == 1
        assert result.hits
        assert all(hit["window"] == 1 for hit in result.hits)
        assert result.n_tested_positions == 14

    def test_edge_all_windows_short_returns_empty_counted_table(self):
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        result = scan(["A"], [motif], background=UNIFORM_BG)
        assert result.hits == []
        assert result.n_excluded_short_windows == 1
        assert result.n_tested_positions == 0
        assert result.n_motifs == 1

    def test_edge_zero_pass_threshold_empty_table_reports_counts(self):
        # Weak w=4 motif: no position anywhere reaches p < 1e-4.
        rows = [[0.3, 0.25, 0.25, 0.2]] * 4
        weak = Motif(motif_id="W", name="W", freq_rows=rows, nsites=20000)
        result = scan(["ACGTACGTACGTACGTACGT"], [weak], background=UNIFORM_BG)
        assert result.hits == []
        assert result.n_tested_positions == 2 * (20 - 4 + 1)
        assert result.n_motifs == 1
        assert result.n_excluded_short_windows == 0

    def test_edge_zero_survive_bh_empty_table_reports_counts(self):
        # Loose p threshold admits every position; every BH q stays above
        # 0.05, so the table is empty but still counted.
        rows = [[0.3, 0.25, 0.25, 0.2]] * 4
        weak = Motif(motif_id="W", name="W", freq_rows=rows, nsites=20000)
        result = scan(
            ["ACGTACGTACGTACGTACGT"],
            [weak],
            background=UNIFORM_BG,
            p_threshold=1.0,
        )
        assert result.hits == []
        assert result.n_tested_positions == 34
        assert result.n_motifs == 1

    def test_edge_window_with_n_skips_unscorable_positions(self):
        motif = _sharp_motif("AGTATACT", motif_id="PAL")
        window = "GGGNAGTATACTGGG"
        result = scan([window], [motif], background=UNIFORM_BG)
        site = [h for h in result.hits if h["start"] == 4 and h["end"] == 12]
        assert site, "site downstream of N must still be reported"
        # 2 strands x 8 positions minus the 4 N-overlapping positions per
        # strand = 8 tested positions.
        assert result.n_tested_positions == 8

    def test_edge_wide_motif_at_width_cap_runs_dp_without_overflow(self):
        consensus = ("AGTATACT" * 13)[:100]
        motif = _sharp_motif(consensus, motif_id="WIDE")
        window = "GGG" + consensus + "GGG"
        result = scan([window], [motif], background=UNIFORM_BG)
        site = [
            h for h in result.hits if h["start"] == 3 and h["end"] == 103 and h["strand"] == "+"
        ]
        assert site
        assert site[0]["p"] == pytest.approx(0.25**100, rel=1e-6)


class TestParseCisbp:
    """CIS-BP local PWM table parsing: fixture + parse-or-reject."""

    def test_cisbp_fixture_parses_with_metadata(self):
        text = (FIXTURES / "cisbp_motif.txt").read_text(encoding="utf-8")
        motif = parse_cisbp(text)
        assert motif.motif_id == "M5085_1.02"
        assert motif.name == "BCL11A"
        assert motif.width == 7
        assert motif.nsites == 0
        assert motif.freq_rows[0] == pytest.approx([0.20, 0.40, 0.20, 0.20])

    def test_cisbp_explicit_id_overrides_metadata(self):
        text = (FIXTURES / "cisbp_motif.txt").read_text(encoding="utf-8")
        motif = parse_cisbp(text, motif_id="CUSTOM.1", name="CustomName")
        assert motif.motif_id == "CUSTOM.1"
        assert motif.name == "CustomName"

    def test_cisbp_name_defaults_to_id_when_absent(self):
        text = "Pos\tA\tC\tG\tT\n1\t0.25\t0.25\t0.25\t0.25\n"
        motif = parse_cisbp(text, motif_id="X")
        assert motif.name == "X"

    def test_cisbp_missing_id_raises(self):
        text = "Pos\tA\tC\tG\tT\n1\t0.25\t0.25\t0.25\t0.25\n"
        with pytest.raises(ValueError, match=r"CIS-BP table provides no motif id"):
            parse_cisbp(text)

    def test_cisbp_rejects_missing_header(self):
        text = "TF Name\tBCL11A\n1\t0.25\t0.25\t0.25\t0.25\n"
        with pytest.raises(ValueError, match=r"expected the 'Pos A C G T' header"):
            parse_cisbp(text)

    def test_cisbp_rejects_non_numeric_row(self):
        text = "Pos\tA\tC\tG\tT\n1\t0.25\tabc\t0.25\t0.25\n"
        with pytest.raises(ValueError, match=r"non-numeric probability row"):
            parse_cisbp(text)

    def test_cisbp_rejects_out_of_range_row(self):
        text = "Pos\tA\tC\tG\tT\n1\t1.5\t0.0\t0.0\t0.0\n"
        with pytest.raises(ValueError, match=r"values must be in \[0, 1\]"):
            parse_cisbp(text)

    def test_cisbp_rejects_wrong_arity_row(self):
        text = "Pos\tA\tC\tG\tT\n1\t0.25\t0.25\t0.25\n"
        with pytest.raises(ValueError, match=r"probability row must be"):
            parse_cisbp(text)

    def test_cisbp_rejects_header_only_table(self):
        text = "Pos\tA\tC\tG\tT\n"
        with pytest.raises(ValueError, match=r"no probability rows"):
            parse_cisbp(text)

    def test_cisbp_rejects_empty_text(self):
        with pytest.raises(ValueError, match=r"CIS-BP table text is empty"):
            parse_cisbp("  \n")

    def test_cisbp_motif_scans_through_the_same_engine(self):
        # nsites == 0 selects the probability-form pseudocount branch; the
        # parsed CIS-BP motif flows through the identical scan path.
        text = (FIXTURES / "cisbp_motif.txt").read_text(encoding="utf-8")
        motif = parse_cisbp(text)
        hits = scan_single_strand("AAAACCGGGCCAAAA", motif, background=UNIFORM_BG)
        site = [h for h in hits if h["start"] == 4 and h["end"] == 11]
        assert site
        assert site[0]["p"] == pytest.approx(0.25**7)


class _FakeResponse:
    """Context-manager stand-in for a urlopen response (records read sizes)."""

    def __init__(self, body: bytes = b"", status: int = 200):
        self.status = status
        self._body = body
        self.read_args: list[int] = []

    def read(self, n: int = -1) -> bytes:
        self.read_args.append(n)
        return self._body if n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """Scripted opener: pops one outcome (response or exception) per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, float | None]] = []

    def open(self, url, timeout=None):
        self.calls.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _jaspar_page(items: list[dict], *, next_url: str | None = None) -> bytes:
    import json

    return json.dumps({"count": len(items), "next": next_url, "results": items}).encode()


class TestJasparClientFetch:
    """Mocked fetch_meme_motif: retry/backoff, error surface, SSRF guards."""

    def test_client_fetch_success_first_try(self):
        body = (FIXTURES / "meme_motif.txt").read_bytes()
        opener = _FakeOpener([_FakeResponse(body)])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            text = fetch_meme_motif("MA2324.1")
        assert text.startswith("MEME version 4")
        assert len(opener.calls) == 1
        url, timeout = opener.calls[0]
        assert url == "https://jaspar.elixir.no/api/v1/matrix/MA2324.1/?format=meme"
        assert timeout == 30.0

    def test_client_fetch_retry_then_success_sleeps_between_attempts(self):
        body = (FIXTURES / "meme_motif.txt").read_bytes()
        opener = _FakeOpener([urllib.error.URLError("connection reset"), _FakeResponse(body)])
        with (
            patch("dnallm.interpret.motifs._JASPAR_OPENER", opener),
            patch("dnallm.interpret.motifs.time.sleep") as sleep,
        ):
            text = fetch_meme_motif("MA2324.1")
        assert text.startswith("MEME version 4")
        assert len(opener.calls) == 2
        sleep.assert_called_once_with(1)  # 2 ** (attempt - 1) backoff

    def test_client_fetch_exhausted_retries_raises_matchable_error(self):
        opener = _FakeOpener([urllib.error.URLError("down")] * 3)
        with (
            patch("dnallm.interpret.motifs._JASPAR_OPENER", opener),
            patch("dnallm.interpret.motifs.time.sleep"),
        ):
            with pytest.raises(ValueError, match=r"JASPAR fetch failed for .*MA2324.1"):
                fetch_meme_motif("MA2324.1")
        assert len(opener.calls) == 3

    def test_client_fetch_non_200_raises_without_retry(self):
        opener = _FakeOpener([_FakeResponse(b"", status=404)])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"JASPAR returned HTTP 404"):
                fetch_meme_motif("MA2324.1")
        assert len(opener.calls) == 1

    def test_client_fetch_invalid_matrix_id_rejected_before_network(self):
        opener = _FakeOpener([])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            # The plan-locked grammar is ^MA\d{4}\.\d+$ -- multi-digit
            # versions (MA0001.10) are VALID by that grammar and absent here.
            for bad_id in ("MA2324", "ma2324.1", "MA23241.1", "../evil", ""):
                with pytest.raises(ValueError, match=r"Invalid JASPAR matrix id"):
                    fetch_meme_motif(bad_id)
        assert opener.calls == []

    def test_client_fetch_non_https_base_url_rejected(self):
        opener = _FakeOpener([])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"must be an https URL"):
                fetch_meme_motif("MA2324.1", base_url="http://jaspar.elixir.no/api/v1")
        assert opener.calls == []

    def test_client_fetch_size_caps_reads(self):
        body = (FIXTURES / "meme_motif.txt").read_bytes()
        response = _FakeResponse(body)
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", _FakeOpener([response])):
            fetch_meme_motif("MA2324.1")
        assert response.read_args == [motifs.MAX_RESPONSE_BYTES]

    def test_client_fetch_output_feeds_parse_meme(self):
        body = (FIXTURES / "meme_motif.txt").read_bytes()
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", _FakeOpener([_FakeResponse(body)])):
            motif = parse_meme(fetch_meme_motif("MA2324.1"))[0]
        assert motif.motif_id == "MA2324.1"


class TestJasparClientSearch:
    """Mocked search_motifs: records, pagination, closed-set params, guards."""

    def test_client_search_parses_matrix_records_with_pinned_query(self):
        opener = _FakeOpener([
            _FakeResponse(
                _jaspar_page([
                    {"matrix_id": "MA2324.1", "name": "BCL11A"},
                    {"matrix_id": "MA2504.1", "name": "BCL11A"},
                ])
            )
        ])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            records = search_motifs("BCL11A")
        assert records == [
            {"matrix_id": "MA2324.1", "name": "BCL11A"},
            {"matrix_id": "MA2504.1", "name": "BCL11A"},
        ]
        url, _timeout = opener.calls[0]
        assert url == (
            "https://jaspar.elixir.no/api/v1/matrix/?name=BCL11A&collection=CORE"
            "&release=2024&version=latest&page=1&page_size=100"
        )

    def test_client_search_paginates_until_next_is_null(self):
        # page_size=1 keeps the partial-page break condition consistent: each
        # full page returns exactly page_size results, page 2 ends on next=null.
        page1 = _FakeResponse(
            _jaspar_page([{"matrix_id": "MA2324.1", "name": "BCL11A"}], next_url="x")
        )
        page2 = _FakeResponse(_jaspar_page([{"matrix_id": "MA2504.1", "name": "BCL11A"}]))
        opener = _FakeOpener([page1, page2])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            records = search_motifs("BCL11A", page_size=1)
        assert [r["matrix_id"] for r in records] == ["MA2324.1", "MA2504.1"]
        assert len(opener.calls) == 2
        assert "page=2&" in opener.calls[1][0] + "&"

    def test_client_search_invalid_collection_rejected_before_network(self):
        opener = _FakeOpener([])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"Invalid JASPAR collection 'BOGUS'"):
                search_motifs("BCL11A", collection="BOGUS")
        assert opener.calls == []

    def test_client_search_empty_name_rejected(self):
        opener = _FakeOpener([])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"non-empty motif name"):
                search_motifs("   ")
        assert opener.calls == []

    def test_client_search_malformed_json_raises(self):
        opener = _FakeOpener([_FakeResponse(b"<html>not json</html>")])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"malformed JSON"):
                search_motifs("BCL11A")
        assert len(opener.calls) == 1

    def test_client_search_missing_results_list_raises(self):
        opener = _FakeOpener([_FakeResponse(b'{"count": 0}')])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"missing its 'results' list"):
                search_motifs("BCL11A")

    def test_client_search_result_item_missing_fields_raises(self):
        opener = _FakeOpener([_FakeResponse(_jaspar_page([{"matrix_id": 1}]))])
        with patch("dnallm.interpret.motifs._JASPAR_OPENER", opener):
            with pytest.raises(ValueError, match=r"missing matrix_id/name strings"):
                search_motifs("BCL11A")


def _skip_if_jaspar_unreachable() -> None:
    """Typed skip when the canonical JASPAR host is unreachable (slow leg)."""
    probe_url = motifs.JASPAR_BASE + "/releases/"
    try:
        # Connectivity probe only: the host is the module constant (https),
        # never caller input.
        request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
            probe_url, method="GET"
        )
        with urllib.request.urlopen(request, timeout=10.0):  # ruff: ignore[suspicious-url-open-usage]
            pass
    except Exception as error:
        pytest.skip(
            f"jaspar-unreachable: {motifs.JASPAR_BASE} unreachable ({type(error).__name__})"
        )


@pytest.mark.slow
class TestJasparLive:
    """Live JASPAR round trips (slow lane; typed jaspar-unreachable: skips)."""

    def test_jaspar_live_fetch_meme_motif_round_trip(self):
        _skip_if_jaspar_unreachable()
        motif = parse_meme(fetch_meme_motif("MA2324.1"))[0]
        assert motif.motif_id == "MA2324.1"
        assert motif.name == "BCL11A"
        assert motif.width == 7

    def test_jaspar_live_search_bcl11a_core_release(self):
        _skip_if_jaspar_unreachable()
        records = search_motifs("BCL11A")
        ids = {record["matrix_id"] for record in records}
        assert {"MA2324.1", "MA2504.1"} <= ids
