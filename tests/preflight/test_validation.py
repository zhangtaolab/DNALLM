"""Tracer suite for the pre-flight core: validate_data + check_compatibility (PFC-01, 23-01 Task 1).

Fast-leg, no network: these tests walk the tracer slice end to end — a plain
``list[str]`` dataset through ``validate_data`` into a collect-all
``DataReport``, then through ``check_compatibility`` into per-model
``ModelVerdict`` objects driven exclusively by the Phase 22 registry
(``try_resolve`` + ``effective_max_length``). Task 2 extends this file with
the full format-surface reader, column autodetect, and split behavior.
"""

import pytest
from pydantic import ValidationError

from dnallm.preflight import (
    CompatReasonCode,
    DataReasonCode,
    DataReport,
    DimensionVerdict,
    ModelVerdict,
    Severity,
    VerdictStatus,
    check_compatibility,
    validate_data,
)

# Pinned registry fixture (Phase 22 lane): the promoter finetuned entry with
# effective_max_length 512 inherited from its base zhangtaolab/plant-dnabert-BPE.
PROMOTER_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"
PROMOTER_BUDGET = 512

# Plan fixture with the too-short member corrected to 18 chars (the plan's
# literal "ACGTACGTACGTACGTACGT" is 5 x ACGT = 20 chars, which is NOT shorter
# than the default min_length 20; the intended behavior needs an 18-char row).
TRACER_SEQUENCES = ["ACGT" * 10, "ACGTN" * 8, "ACGTACGTACGTACGTAC"]  # 40 / 40 / 18


def _reason_codes(report: DataReport) -> set[str]:
    """Collect every reason code appearing in any split or top-level finding."""
    codes = {f.reason_code for f in report.findings}
    for split in report.splits:
        codes |= {f.reason_code for f in split.findings}
    return codes


def _finding(report: DataReport, code: str):
    """Return the first finding carrying the given reason code, or None."""
    findings = list(report.findings) + [f for s in report.splits for f in s.findings]
    return next((f for f in findings if f.reason_code == code), None)


# ─────────────────────────────────────────────────────────────────────────────
# validate_data over list[str]: collect-all census (PFC-01, D-09)
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_data_list_input_returns_collect_all_report() -> None:
    """list[str] flows into one SplitReport with the full census and no row loss."""
    report = validate_data(TRACER_SEQUENCES)
    assert isinstance(report, DataReport)
    assert len(report.splits) == 1
    split = report.splits[0]
    assert split.name == "data"
    assert split.row_count == 3  # nothing dropped — findings annotate, rows stay
    assert report.min_length == 20
    assert report.seq_col is None  # raw sequences carry no column identity
    assert report.label_col is None


def test_validate_data_charset_census_counts_and_sorts() -> None:
    """Census counts uppercased characters (ACGTN*8 contributes N:8) sorted by char."""
    report = validate_data(TRACER_SEQUENCES)
    census = report.splits[0].charset_census
    counts = {entry.char: entry.count for entry in census}
    assert counts["N"] == 8
    assert counts["A"] == 23  # 10 + 8 + 5 (the 18-char seq ends ..AC)
    assert [entry.char for entry in census] == sorted(counts)  # alphabetical order


def test_validate_data_too_short_finding_but_never_drops_rows() -> None:
    """The 18-char sequence gets an error TOO_SHORT finding; row_count stays 3."""
    report = validate_data(TRACER_SEQUENCES)
    too_short = _finding(report, DataReasonCode.TOO_SHORT)
    assert too_short is not None
    assert too_short.severity == Severity.ERROR
    assert too_short.count == 1
    assert report.splits[0].row_count == 3


def test_validate_data_no_all_n_finding_when_no_sequence_is_all_n() -> None:
    """No ALL_N finding unless some sequence consists entirely of N."""
    report = validate_data(TRACER_SEQUENCES)
    assert DataReasonCode.ALL_N not in _reason_codes(report)


def test_validate_data_all_n_sequence_warns() -> None:
    """A sequence of only N gets a warn ALL_N finding and stays a row."""
    report = validate_data(["N" * 30, "ACGT" * 10])
    all_n = _finding(report, DataReasonCode.ALL_N)
    assert all_n is not None
    assert all_n.severity == Severity.WARN
    assert all_n.count == 1
    assert report.splits[0].row_count == 2


def test_validate_data_empty_sequence_error_finding() -> None:
    """A zero-length sequence gets an error EMPTY_SEQUENCE finding."""
    report = validate_data(["ACGT" * 10, ""])
    empty = _finding(report, DataReasonCode.EMPTY_SEQUENCE)
    assert empty is not None
    assert empty.severity == Severity.ERROR
    assert empty.count == 1


def test_validate_data_exact_duplicates_counted_info_level() -> None:
    """Exact duplicate sequences are counted (occurrences - distinct), info level."""
    report = validate_data(["ACGT" * 5, "ACGT" * 5, "AAAA" * 5])
    dup = _finding(report, DataReasonCode.DUPLICATE_SEQUENCES)
    assert dup is not None
    assert dup.severity == Severity.INFO
    assert dup.count == 1  # 3 occurrences - 2 distinct
    assert report.splits[0].row_count == 3


def test_validate_data_length_stats_quantiles_and_bounds() -> None:
    """p50/p95/p99 are linear-interpolation quantiles; min/max are exact ints."""
    report = validate_data(TRACER_SEQUENCES)
    stats = report.splits[0].length_stats
    assert stats is not None
    assert stats.n == 3
    assert stats.min == 18
    assert stats.max == 40
    assert isinstance(stats.min, int)
    assert isinstance(stats.max, int)
    # sorted lengths [18, 40, 40]; numpy linear interpolation: p50 == p95 == p99 == 40.0
    assert stats.p50 == pytest.approx(40.0)
    assert stats.p95 == pytest.approx(40.0)
    assert stats.p99 == pytest.approx(40.0)


def test_validate_data_gc_and_lowercase_fraction_normalized_via_upper() -> None:
    """Charset/GC normalize via upper(); lowercase fraction is reported separately."""
    report = validate_data(["acgt" * 10, "ACGT" * 10])
    split = report.splits[0]
    assert split.mean_gc_content == pytest.approx(0.5)
    assert split.lowercase_fraction == pytest.approx(0.5)  # 40 of 80 chars lowercase


def test_validate_data_empty_list_yields_empty_dataset_info() -> None:
    """An empty list returns a zero-row report with an info EMPTY_DATASET finding."""
    report = validate_data([])
    split = report.splits[0]
    assert split.row_count == 0
    assert split.length_stats is None
    assert split.charset_census == []
    empty = _finding(report, DataReasonCode.EMPTY_DATASET)
    assert empty is not None
    assert empty.severity == Severity.INFO


def test_validate_data_none_source_raises_matchable_value_error() -> None:
    """A None source raises a descriptive ValueError, never an empty report."""
    with pytest.raises(ValueError, match=r"source"):
        validate_data(None)  # type: ignore[arg-type]


def test_report_objects_are_frozen() -> None:
    """DataReport and its members are frozen: assignment raises ValidationError."""
    report = validate_data(TRACER_SEQUENCES)
    with pytest.raises(ValidationError):
        report.min_length = 99  # type: ignore[misc]
    with pytest.raises(ValidationError):
        report.splits[0].row_count = 0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# check_compatibility tracer: registry-driven length verdicts
# ─────────────────────────────────────────────────────────────────────────────


def _length_dimension(verdict: ModelVerdict) -> DimensionVerdict:
    dims = [d for d in verdict.dimensions if d.dimension == "length"]
    assert len(dims) == 1
    return dims[0]


def test_check_compatibility_real_model_length_ok() -> None:
    """A real registry entry yields a length-ok verdict for short sequences."""
    report = validate_data(["ACGT" * 10])
    result = check_compatibility(report, models=[PROMOTER_REPO], task_type="binary")
    verdict = result.verdicts[0]
    assert isinstance(verdict, ModelVerdict)
    assert verdict.model == PROMOTER_REPO
    assert verdict.overall in (
        VerdictStatus.OK,
        VerdictStatus.WARN,
        VerdictStatus.FAIL,
        VerdictStatus.UNKNOWN,
    )
    length = _length_dimension(verdict)
    assert length.status == VerdictStatus.OK
    assert length.reason_code is None
    assert length.details["budget_chars"] == PROMOTER_BUDGET
    assert length.details["truncated_row_count"] == 0
    assert length.details["truncated_fraction"] == 0.0


def test_check_compatibility_truncation_warns_with_exact_counts() -> None:
    """Rows longer than the char budget warn with exact count and fraction."""
    report = validate_data(["A" * 600, "C" * 100])
    result = check_compatibility(report, models=[PROMOTER_REPO])
    length = _length_dimension(result.verdicts[0])
    assert length.status == VerdictStatus.WARN
    assert length.reason_code == CompatReasonCode.TRUNCATION_ABOVE_THRESHOLD
    assert length.details["budget_chars"] == PROMOTER_BUDGET
    assert length.details["truncated_row_count"] == 1  # exact integer, never rounded
    assert length.details["truncated_fraction"] == pytest.approx(0.5)
    assert result.verdicts[0].overall == VerdictStatus.WARN


def test_check_compatibility_unknown_model_fails_isolated() -> None:
    """An unknown model yields a fail MODEL_NOT_IN_REGISTRY verdict, no exception."""
    report = validate_data(["ACGT" * 10])
    result = check_compatibility(report, models=["no-such-model/repo", PROMOTER_REPO])
    assert len(result.verdicts) == 2  # per-model isolation: the miss never breaks the hit
    miss = result.verdicts[0]
    assert miss.model == "no-such-model/repo"
    assert miss.entry_name is None
    assert miss.is_lab is False
    assert miss.overall == VerdictStatus.FAIL
    assert miss.dimensions[0].reason_code == CompatReasonCode.MODEL_NOT_IN_REGISTRY
    assert result.verdicts[1].model == PROMOTER_REPO


def test_check_compatibility_no_data_length_unknown() -> None:
    """Without a DataReport the length dimension is unknown with reason NO_DATA."""
    result = check_compatibility(None, models=[PROMOTER_REPO], task_type="binary")
    length = _length_dimension(result.verdicts[0])
    assert length.status == VerdictStatus.UNKNOWN
    assert length.reason_code == CompatReasonCode.NO_DATA
    assert result.verdicts[0].overall == VerdictStatus.UNKNOWN


def test_check_compatibility_empty_dataset_length_unknown() -> None:
    """A zero-row report degrades the length dimension to unknown EMPTY_DATASET_DIM."""
    report = validate_data([])
    result = check_compatibility(report, models=[PROMOTER_REPO])
    length = _length_dimension(result.verdicts[0])
    assert length.status == VerdictStatus.UNKNOWN
    assert length.reason_code == CompatReasonCode.EMPTY_DATASET_DIM


def test_check_compatibility_lab_entry_flag_and_registry_metadata() -> None:
    """The verdict carries registry metadata: entry name and the lab flag."""
    report = validate_data(["ACGT" * 10])
    result = check_compatibility(report, models=[PROMOTER_REPO])
    verdict = result.verdicts[0]
    assert verdict.is_lab is True
    assert verdict.entry_name == "Plant DNABERT BPE promoter"
    assert result.listing == []  # listings land in 23-02


# ─────────────────────────────────────────────────────────────────────────────
# Facade
# ─────────────────────────────────────────────────────────────────────────────


def test_preflight_facade_exports_core_api() -> None:
    """The package facade exposes the tracer API and report types via __all__."""
    import dnallm.preflight as preflight

    for name in ("validate_data", "check_compatibility", "DataReport", "ModelVerdict"):
        assert name in preflight.__all__
        assert getattr(preflight, name) is not None


def test_end_to_end_tracer_flow_over_real_registry_entry() -> None:
    """list[str] -> validate_data -> DataReport -> check_compatibility -> ModelVerdict."""
    report = validate_data(["ACGT" * 25, "ACGTN" * 8, "A" * 30])
    assert report.splits[0].row_count == 3
    result = check_compatibility(report, models=[PROMOTER_REPO], task_type="Binary ")
    verdict = result.verdicts[0]
    assert verdict.model == PROMOTER_REPO
    assert verdict.overall == VerdictStatus.OK  # all lengths <= 512-char budget
    length = _length_dimension(verdict)
    assert length.details["budget_chars"] == PROMOTER_BUDGET
    assert length.details["truncated_row_count"] == 0
