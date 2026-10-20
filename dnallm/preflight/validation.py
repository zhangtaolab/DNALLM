"""validate_data: read-only, collect-all data validation (PFC-01, D-01, D-09).

This module owns the data half of the pre-flight core. It turns any
supported data source into a :class:`~dnallm.preflight.report.DataReport`
without ever dropping, mutating, or rewriting a row or the source itself —
findings annotate, reports are new objects, and nothing is persisted.

Census conventions (D-09): length statistics count Unicode code points of
the raw sequence string; the charset census and GC computation normalize via
``upper()`` (lowercase soft-mask counts as the same base) with the lowercase
fraction reported separately; empty and too-short sequences get error-level
reason codes; all-N sequences get a warn; exact duplicates are counted at
info level. Rows are counted, never filtered.

Import discipline (D-03): this module never imports datasets/torch/
transformers — the reader rides on stdlib + pandas + pyarrow only, and
imports nothing from ``dnallm.datahandling`` (parity with
``DNADataset.load_local_data`` is enforced by the double-read structure test
in ``tests/preflight/test_reader_parity.py``, not by sharing code).
"""

from collections import Counter
from pathlib import Path


import numpy as np

from ..utils import get_logger
from ..utils.sequence import calc_gc_content
from .report import (
    CharsetCensusEntry,
    DataReasonCode,
    DataReport,
    Finding,
    LengthStats,
    Severity,
    SplitReport,
)

logger = get_logger("dnallm.preflight.validation")

__all__ = ["validate_data"]

# Split name for unsplit input (dict-of-splits keys become split names).
_UNSPLIT_NAME = "data"

ValidatedSource = str | Path | dict | list


def validate_data(
    source: ValidatedSource,
    *,
    seq_col: str | None = None,
    label_col: str | None = None,
    min_length: int = 20,
) -> DataReport:
    """Validate a dataset read-only and return a collect-all report (PFC-01).

    Args:
        source: The data to validate — a plain ``list[str]`` of raw
            sequences, a file path (``str`` or ``Path``), or a dict of
            splits mapping split names to paths/sequence lists (D-01). The
            full file-format surface (csv/tsv/json/parquet/arrow/fasta/txt)
            arrives with the Task 2 reader; this slice accepts ``list[str]``.
        seq_col: Explicit sequence column name; when None the reader
            auto-detects (Task 2) or the input has no column identity.
        label_col: Explicit label column name; when None the reader may
            guess and report a LABEL_COLUMN_GUESSED finding (Task 2).
        min_length: Minimum sequence length in code points; shorter
            non-empty sequences get an error TOO_SHORT finding (default 20).

    Returns:
        The frozen collect-all report: one split census plus findings. Rows
        are counted, never dropped or mutated.

    Raises:
        ValueError: If the source is None or of an unsupported type.
    """
    if source is None:
        raise ValueError("validate_data source must not be None.")
    if isinstance(source, list):
        return _validate_sequence_list(
            source, seq_col=seq_col, label_col=label_col, min_length=min_length
        )
    raise ValueError(
        f"Unsupported validate_data source type {type(source).__name__!r}; "
        "supported: list[str], file path, dict of splits."
    )


def _validate_sequence_list(
    sequences: list,
    *,
    seq_col: str | None,
    label_col: str | None,
    min_length: int,
) -> DataReport:
    """Build the single-split report for a raw list[str] of sequences."""
    for position, item in enumerate(sequences):
        if not isinstance(item, str):
            raise ValueError(
                f"validate_data list input must contain only strings; "
                f"element {position} is {type(item).__name__!r}."
            )
    split = _census_split(_UNSPLIT_NAME, sequences, min_length=min_length)
    _log_split_summary(split)
    return DataReport(
        splits=[split],
        findings=[],
        seq_col=seq_col,
        label_col=label_col,
        min_length=min_length,
    )


def _census_split(name: str, sequences: list[str], *, min_length: int) -> SplitReport:
    """Compute the full census of one split over its raw sequences (D-09).

    Rows are counted, never filtered: every finding carries an exact count
    and the row count always equals ``len(sequences)``.
    """
    findings: list[Finding] = []
    row_count = len(sequences)
    if row_count == 0:
        findings.append(
            Finding(
                severity=Severity.INFO,
                reason_code=DataReasonCode.EMPTY_DATASET.value,
                message=f"Split {name!r} is empty (0 rows).",
                count=0,
            )
        )
        return SplitReport(
            name=name,
            row_count=0,
            length_stats=None,
            charset_census=[],
            findings=findings,
            lowercase_fraction=0.0,
            mean_gc_content=0.0,
            length_counts={},
        )

    lengths = [len(s) for s in sequences]
    empty_count = sum(1 for length in lengths if length == 0)
    short_count = sum(1 for length in lengths if 0 < length < min_length)
    all_n_count = sum(1 for s in sequences if s and set(s.upper()) == {"N"})
    duplicate_count = row_count - len(set(sequences))

    if empty_count:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                reason_code=DataReasonCode.EMPTY_SEQUENCE.value,
                message=f"{empty_count} of {row_count} sequences are empty.",
                count=empty_count,
            )
        )
    if short_count:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                reason_code=DataReasonCode.TOO_SHORT.value,
                message=f"{short_count} of {row_count} sequences are shorter "
                f"than min_length {min_length}.",
                count=short_count,
            )
        )
    if all_n_count:
        findings.append(
            Finding(
                severity=Severity.WARN,
                reason_code=DataReasonCode.ALL_N.value,
                message=f"{all_n_count} of {row_count} sequences contain only N.",
                count=all_n_count,
            )
        )
    if duplicate_count:
        findings.append(
            Finding(
                severity=Severity.INFO,
                reason_code=DataReasonCode.DUPLICATE_SEQUENCES.value,
                message=f"{duplicate_count} duplicate sequence occurrence(s) "
                f"across {len(set(sequences))} distinct sequences.",
                count=duplicate_count,
            )
        )

    quantiles = np.quantile(lengths, [0.5, 0.95, 0.99])
    length_stats = LengthStats(
        n=row_count,
        p50=float(quantiles[0]),
        p95=float(quantiles[1]),
        p99=float(quantiles[2]),
        min=min(lengths),
        max=max(lengths),
    )

    upper_chars = Counter("".join(s.upper() for s in sequences))
    charset_census = [
        CharsetCensusEntry(char=char, count=count) for char, count in sorted(upper_chars.items())
    ]
    total_chars = sum(lengths)
    lower_chars = sum(1 for s in sequences for ch in s if ch.islower())
    lowercase_fraction = lower_chars / total_chars if total_chars else 0.0
    mean_gc_content = sum(calc_gc_content(s) for s in sequences) / row_count

    return SplitReport(
        name=name,
        row_count=row_count,
        length_stats=length_stats,
        charset_census=charset_census,
        findings=findings,
        lowercase_fraction=lowercase_fraction,
        mean_gc_content=mean_gc_content,
        length_counts=dict(sorted(Counter(lengths).items())),
    )


def _log_split_summary(split: SplitReport) -> None:
    """Log summary counts and reason codes only — never raw sequences (T-23-02)."""
    codes = [f.reason_code for f in split.findings]
    logger.info(
        f"validate_data: split {split.name!r} rows={split.row_count} findings={codes or 'none'}"
    )
