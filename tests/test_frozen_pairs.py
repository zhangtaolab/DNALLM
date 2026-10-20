"""Frozen-pair tests — committed real pairs, same-session oracles, parsed bands.

Executes the phase-15 numpy evaluators over the committed frozen interval
pairs (tests/fixtures/genomic_intervals/, regenerated + oracle-measured in
one session on 2026-10-13 — see that directory's PROVENANCE.md) and asserts:

* CRE pair — jaccard: 6-dp round-equality with the same-session bedtools
  value, trailing-float agreement with the same-session numpy value, and
  membership in the parsed selection.md CRE band [0.3, 1.00].
* Anno pair — exon_f1: the local greedy reciprocal-overlap matcher (a port
  of Anno notebook cell 17) recomputes the exact tp/fp/fn integer triple;
  f1 equals the 2tp/(2tp+fp+fn) rational and lies inside the parsed TIGHT
  band from selection.md's frozen-pair row.
* loj pair — byte-identical line equality against the committed bedtools
  v2.31.1 left-outer-join oracle, with the consecutive no-hit A rows
  surfacing the null sextet verbatim in consecutive output lines.

Posture (phase-15, tests/test_genomic_intervals.py): platform-agnostic,
NO markers, NO skip machinery, NO bedtools binary, NO process spawning --
numpy is core on every leg, so this file runs on test-windows and every
Linux lane automatically. Census untouched by root placement.

Band note (Pitfall 6): the TIGHT exon_f1 band parsed here is consumed by
THIS FILE ONLY. The nightly model re-execution keeps the four existing
drift bands plus the exon_f1 evidence key -- a ~1e-9 band against live
model output would false-red under transformers drift.

Every recorded constant carries a provenance pointer: the comparison-layer
constants below were measured in the regeneration session recorded in
tests/fixtures/genomic_intervals/PROVENANCE.md (owner decision: tolerance
lives in the comparison layer only).
"""

import re
import sys
from itertools import groupby
from pathlib import Path

import pytest

from dnallm.utils.genomic_intervals import jaccard, loj_intersect, parse_bed

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "genomic_intervals"
SELECTION_MD = (
    REPO_ROOT / "example" / "notebooks" / "plant_helixseek_shared" / "data" / "selection.md"
)

# ─────────────────────────────────────────────────────────────────────────────
# Comparison-layer constants (PROVENANCE.md, 2026-10-13 session)
# ─────────────────────────────────────────────────────────────────────────────
# bedtools v2.31.1 printed precision for the CRE pair, measured same-session.
CRE_BEDTOOLS_JACCARD_6DP = 0.324727
# numpy full precision for the CRE pair over the exact committed bytes; the
# session reproduced the lost original's aggregate bit-identically (this is
# the same double as tests/test_genomic_intervals.py FROZEN_PAIR_JACCARD).
CRE_NUMPY_JACCARD_FULL = 0.32472687098490116
# Trailing-float tolerance for numpy-vs-recorded-full-precision comparison:
# sized for the ~1e-9-class cross-numpy-version delta (session measured 0.0
# on numpy 2.5.3; CI pins 2.2.0). A single-base intersection change moves
# the jaccard by >= 1/48878 ~ 2e-5, an order of magnitude above this.
CRE_TRAILING_FLOAT_TOL = 1e-9
# Anno pair integer counts: the session's provenance triple, identical to
# the historical selection.md anno_locus_detail keys (526/394/346/48/180) —
# AS-1's anchor branch is ACTIVE, so the historical counts are pinned as
# baked literals too (PROVENANCE.md "Drift vs the historical" table).
ANNO_PROVENANCE_COUNTS = (526, 394, 346, 48, 180)  # n_truth, n_pred, tp, fp, fn
# bedtools loj null-B fill for a 6-column B side (phase-15 pinned literals).
NULL_SEXTET = (".", "-1", "-1", ".", "-1", ".")

# Row-lookup fragments parsed from selection.md (D-07: parsed, never copied).
CRE_BAND_LABEL = "CRE jaccard"
FROZEN_PAIR_BAND_LABEL = "frozen pair"


def _parse_band(label_fragment: str) -> tuple[float, float]:
    """Parse one '[a, b]' tolerance interval from selection.md's band table.

    Duplicated from tests/examples/test_plant_helixseek_showcase.py
    ``_parse_bands`` by chosen coupling posture (16-PATTERNS): the helper is
    NOT imported across the tests/examples boundary. Fail-red parse guards
    (D-06): a missing section, a missing/ambiguous row, or an unparseable
    interval fails the test naming the row -- never a skip.
    """
    lines = SELECTION_MD.read_text(encoding="utf-8").splitlines()
    try:
        header_idx = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("## Thresholds and tolerance bands")
        )
    except StopIteration:
        pytest.fail(
            "selection.md is missing the '## Thresholds and tolerance bands' section "
            "(parse guard, D-06)"
        )
    rows: list[str] = []
    for line in lines[header_idx + 1 :]:
        if line.startswith("## "):  # the band table ends at the next section
            break
        if line.startswith("|"):
            rows.append(line)
    hits = [line for line in rows if label_fragment in line.split("|")[1]]
    if len(hits) != 1:
        pytest.fail(
            f"selection.md band table must carry exactly one row matching "
            f"'{label_fragment}' (parse guard, D-06); found {len(hits)}"
        )
    interval = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", hits[0])
    if interval is None:
        pytest.fail(
            f"selection.md band row '{label_fragment}' carries no '[a, b]' "
            f"tolerance interval (parse guard, D-06)"
        )
    return float(interval.group(1)), float(interval.group(2))


def _pooled_exon_counts() -> tuple[int, int, int, int, int]:
    """Recompute (n_truth, n_pred, tp, fp, fn) over the committed Anno pair.

    Local port of the Anno notebook's greedy reciprocal-overlap matcher
    (cell 17, MATCH_RECIP = 0.5): iteration in predicted-start ascending
    order, first unmatched truth row wins, per strand; tp counts matched
    truth rows, fp = unmatched predicted segments, fn = unmatched truth
    rows, pooled across strands.
    """
    match_recip = 0.5

    def greedy_match(preds, truths):
        matches = []
        used = set()
        for pi, (ps, pe) in sorted(enumerate(preds), key=lambda item: item[1]):
            for ti, (ts, te) in enumerate(truths):
                if ti in used:
                    continue
                overlap = min(pe, te) - max(ps, ts)
                if overlap <= 0:
                    continue
                if overlap / (pe - ps) >= match_recip and overlap / (te - ts) >= match_recip:
                    matches.append((pi, ti))
                    used.add(ti)
                    break
        return matches

    pred_by_strand = {"+": [], "-": []}
    for row in parse_bed(FIXTURES / "anno_pred.bed"):
        pred_by_strand[row.extra[2]].append((row.start, row.end))
    truth_by_strand = {"+": [], "-": []}
    for row in parse_bed(FIXTURES / "anno_truth.bed"):
        truth_by_strand[row.extra[2]].append((row.start, row.end))
    tp = sum(
        len(greedy_match(pred_by_strand[strand], truth_by_strand[strand])) for strand in ("+", "-")
    )
    n_pred = sum(len(segments) for segments in pred_by_strand.values())
    n_truth = sum(len(rows) for rows in truth_by_strand.values())
    return n_truth, n_pred, tp, n_pred - tp, n_truth - tp


# ─────────────────────────────────────────────────────────────────────────────
# CRE pair — dual-oracle jaccard + parsed loose band
# ─────────────────────────────────────────────────────────────────────────────


def test_frozen_cre_pair_jaccard_dual_oracle_and_band() -> None:
    result = jaccard(FIXTURES / "cre_pred.bed", FIXTURES / "cre_truth.bed")
    value = result["jaccard"]
    # (a) 6-decimal round-equality with the same-session bedtools value.
    assert round(value, 6) == CRE_BEDTOOLS_JACCARD_6DP
    # (b) trailing-float agreement with the recorded full-precision value.
    assert abs(value - CRE_NUMPY_JACCARD_FULL) <= CRE_TRAILING_FLOAT_TOL
    # The pair's intersection/union aggregates match the provenance record.
    assert result["intersection"] == 15872
    assert result["union"] == 48878
    # (c) membership in the parsed selection.md CRE band (the loose band
    # that governs the nightly re-execution on both platforms).
    low, high = _parse_band(CRE_BAND_LABEL)
    assert low <= value <= high, (value, low, high)


# ─────────────────────────────────────────────────────────────────────────────
# Anno pair — integer counts, exact rational, parsed TIGHT band (HERE ONLY)
# ─────────────────────────────────────────────────────────────────────────────


def test_frozen_anno_pair_exon_counts_rational_and_tight_band() -> None:
    n_truth, n_pred, tp, fp, fn = _pooled_exon_counts()
    # Exact integer equality with the provenance triple; the session
    # reproduced the historical anchors exactly, so the historical baked
    # literals pin the same numbers (AS-1 anchor branch).
    assert (n_truth, n_pred, tp, fp, fn) == ANNO_PROVENANCE_COUNTS
    # f1 as the exact rational recomputed from those counts...
    pooled_denom = 2 * tp + fp + fn
    f1 = 2 * tp / pooled_denom
    # ...equals the 692/920 rational (same expression, exact double equality)
    assert f1 == 692 / 920
    # ...and lies inside the parsed TIGHT band from the frozen-pair row
    # (selection.md 'Anno pooled exon-F1 (frozen pair)' -- consumed by this
    # file ONLY, Pitfall 6).
    low, high = _parse_band(FROZEN_PAIR_BAND_LABEL)
    assert low <= f1 <= high, (f1, low, high)


# ─────────────────────────────────────────────────────────────────────────────
# loj pair — byte parity vs the committed bedtools oracle + null sextet
# ─────────────────────────────────────────────────────────────────────────────


def test_frozen_loj_pair_byte_parity_and_consecutive_null_sextet() -> None:
    lines = loj_intersect(FIXTURES / "loj_a.bed", FIXTURES / "loj_b.bed")
    oracle = (FIXTURES / "loj_bedtools_oracle.tsv").read_text(encoding="utf-8").splitlines()
    # Same count, same order, exact string equality -- any byte drift in
    # either the pair or the oracle fails here.
    assert lines == oracle
    a_rows = parse_bed(FIXTURES / "loj_a.bed")
    # Join multiplicity: at least one A row is overlapped by 2+ B rows, so
    # the oracle carries MORE lines than A has rows.
    assert len(oracle) > len(a_rows)
    # The consecutive no-hit A rows surface the null sextet VERBATIM in
    # consecutive output lines (the B-side six fields of each line).
    sextet = list(NULL_SEXTET)
    consecutive_null_pairs = sum(
        1
        for i in range(len(oracle) - 1)
        if oracle[i].split("\t")[6:12] == sextet and oracle[i + 1].split("\t")[6:12] == sextet
    )
    assert consecutive_null_pairs >= 1, "expected >=2 consecutive no-hit A rows"
    # Gene-grouped A order (the consumer's gene-contiguity flush): every
    # gene label forms exactly one contiguous run in the A file.
    genes = [row.extra[1] for row in a_rows]
    n_runs = sum(1 for _ in groupby(genes))
    assert n_runs == len(set(genes)), "loj A gene labels are not contiguous runs"


# ─────────────────────────────────────────────────────────────────────────────
# Parse guards — a missing band row fails red (D-06), never a skip
# ─────────────────────────────────────────────────────────────────────────────


def test_band_parse_guard_fails_red_on_missing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection.md without the frozen-pair row must FAIL naming the row."""
    text = SELECTION_MD.read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in text.splitlines() if "Anno pooled exon-F1 (frozen pair)" not in line
    )
    assert stripped != text, "probe row unexpectedly absent from selection.md"
    probe = tmp_path / "selection.md"
    probe.write_text(stripped + "\n", encoding="utf-8")
    # Patch THIS module's global (tests/ carries no __init__.py -- the
    # module's import name is interpreter-path dependent, so patch the
    # already-imported module object directly).
    monkeypatch.setattr(sys.modules[__name__], "SELECTION_MD", probe)
    # Same import shape as the showcase gate-contract precedent
    # (tests/examples/test_plant_helixseek_showcase.py imports Skipped this way).
    from _pytest.outcomes import Failed

    with pytest.raises(Failed, match="frozen pair"):
        _parse_band(FROZEN_PAIR_BAND_LABEL)
