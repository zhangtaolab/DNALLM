"""validate_data: read-only, collect-all data validation (PFC-01, D-01..D-04, D-09).

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

Reader (D-03/D-04): a standalone reader over stdlib + pandas + pyarrow that
replicates the observable read behavior of
``DNADataset.load_local_data`` (``dnallm/datahandling/data.py:82-334``) —
path normalization, suffix dispatch, the headerless-csv -> txt demotion
rule, the fasta label rule with multi-line accumulation, both txt branches,
and the list-of-files rules — pinned by the double-read parity structure
test in ``tests/preflight/test_reader_parity.py``. ``pkl``/``pickle``
inputs are refused with an UNSUPPORTED_FORMAT_PKL error finding and are
never opened as pickle (D-04: ``pickle.load`` on user input is an
arbitrary-code-execution vector). This module never imports
datasets/torch/transformers and never imports ``dnallm.datahandling``.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

# Canonical column names for formats whose layout is structural (fasta/txt).
_SEQ_NAME = "sequence"
_LABEL_NAME = "label"

# Suffix dispatch (parity with data.py:189-190 plus the fasta aliases).
_STRUCTURED_SUFFIXES = ("csv", "tsv", "json", "parquet", "arrow")
_FASTA_SUFFIXES = ("fa", "fna", "fas", "fasta")
_REFUSED_SUFFIXES = ("pkl", "pickle")
_SUPPORTED_SUFFIXES = _STRUCTURED_SUFFIXES + _FASTA_SUFFIXES + ("txt",) + _REFUSED_SUFFIXES
_SUFFIX_NEEDLES = tuple(f".{s}" for s in _SUPPORTED_SUFFIXES)

# D-02 column autodetect: IUPAC DNA alphabet for scoring candidate columns.
_DNA_ALPHABET = frozenset("ACGTNRYKMSWKBHDVU")
_DNA_SCORE_THRESHOLD = 0.9
_AUTODETECT_CELL_LIMIT = 1000

# Label-column guess: candidate names first, then the two-column fallback.
_LABEL_NAME_CANDIDATES = ("label", "labels", "y", "target", "class", "category")
_MULTILABEL_SEPARATORS = (",", ";", "|")

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
        source: The data to validate (D-01): a file path (``str`` or
            ``Path``; csv/tsv/json/parquet/arrow/fasta/txt — a list of paths
            is accepted for the structured formats), a dict of splits
            mapping split names to paths or raw sequence lists, or a plain
            ``list[str]`` of raw sequences. ``pkl``/``pickle`` paths are
            refused with an UNSUPPORTED_FORMAT_PKL error finding and are
            never unpickled.
        seq_col: Explicit sequence column name; when None the reader
            auto-detects the ACGT-dominant column (D-02).
        label_col: Explicit label column name; when None the reader may
            guess it and report a LABEL_COLUMN_GUESSED finding.
        min_length: Minimum sequence length in code points; shorter
            non-empty sequences get an error TOO_SHORT finding (default 20).

    Returns:
        The frozen collect-all report: one split census per split plus
        column-level findings. Rows are counted, never dropped or mutated.

    Raises:
        ValueError: If the source is None, has an unsupported type, names a
            nonexistent file, or carries an unsupported file suffix.
    """
    if source is None:
        raise ValueError("validate_data source must not be None.")
    if isinstance(source, dict):
        if not source:
            raise ValueError("validate_data dict of splits must not be empty.")
        return _validate_splits_dict(
            source, seq_col=seq_col, label_col=label_col, min_length=min_length
        )
    if isinstance(source, (str, Path)):
        return _validate_file_source(
            source, seq_col=seq_col, label_col=label_col, min_length=min_length
        )
    if isinstance(source, list):
        if _looks_like_file_paths(source):
            return _validate_file_source(
                source, seq_col=seq_col, label_col=label_col, min_length=min_length
            )
        return _validate_sequence_list(
            source, seq_col=seq_col, label_col=label_col, min_length=min_length
        )
    raise ValueError(
        f"Unsupported validate_data source type {type(source).__name__!r}; "
        "supported: list[str], file path, dict of splits."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source pipelines
# ─────────────────────────────────────────────────────────────────────────────


def _validate_sequence_list(
    sequences: list,
    *,
    seq_col: str | None,
    label_col: str | None,
    min_length: int,
) -> DataReport:
    """Build the single-split report for a raw list[str] of sequences."""
    _ensure_string_list(sequences)
    split = _census_split(_UNSPLIT_NAME, sequences, min_length=min_length)
    _log_split_summary(split)
    return DataReport(
        splits=[split],
        findings=[],
        seq_col=seq_col,
        label_col=label_col,
        min_length=min_length,
    )


def _validate_file_source(
    file_path, *, seq_col: str | None, label_col: str | None, min_length: int
) -> DataReport:
    """Read one file source (path or list of structured paths) into a report."""
    read = _read_source(file_path, seq_col=seq_col, label_col=label_col)
    if read.file_type in _REFUSED_SUFFIXES:
        # Refused format: zero-row split with NO census findings — the
        # refusal finding (report-level) is the whole story.
        split = SplitReport(name=_UNSPLIT_NAME, row_count=0)
        return DataReport(
            splits=[split],
            findings=list(read.findings),
            seq_col=None,
            label_col=None,
            min_length=min_length,
        )
    return _report_from_read(
        [(_UNSPLIT_NAME, read)], report_findings=read.findings, min_length=min_length
    )


def _validate_splits_dict(
    splits: dict, *, seq_col: str | None, label_col: str | None, min_length: int
) -> DataReport:
    """Read a dict of splits: one census per key, findings combined (deduped)."""
    entries = []
    report_findings: list[Finding] = []
    for name, value in splits.items():
        if isinstance(value, (str, Path)) or (
            isinstance(value, list) and _looks_like_file_paths(value)
        ):
            read = _read_source(value, seq_col=seq_col, label_col=label_col)
            entries.append((name, read))
            report_findings.extend(read.findings)
        elif isinstance(value, list):
            _ensure_string_list(value)
            entries.append((name, value))
        else:
            raise ValueError(
                f"Unsupported source type for split {name!r}: "
                f"{type(value).__name__!r}; expected a file path or a list."
            )
    return _report_from_read(entries, report_findings=report_findings, min_length=min_length)


def _report_from_read(entries, *, report_findings, min_length) -> DataReport:
    """Assemble the DataReport from per-split reads or raw sequence lists.

    ``entries`` pairs (split name, _SourceRead or list[str]); column-level
    findings are deduplicated at report level, label dtype/separator
    metadata is guessed from the first split that carries labels.
    """
    split_reports = []
    label_dtype: str | None = None
    multilabel_sep: str | None = None
    seq_names: set = set()
    label_names: set = set()
    for name, entry in entries:
        if isinstance(entry, list):
            split_reports.append(_census_split(name, entry, min_length=min_length))
            continue
        if entry.labels is not None and label_dtype is None:
            label_dtype, multilabel_sep = _guess_label_metadata(entry.labels)
        seq_names.add(entry.seq_name)
        label_names.add(entry.label_name)
        if entry.file_type in _REFUSED_SUFFIXES:
            split_reports.append(SplitReport(name=name, row_count=0))
            continue
        split_reports.append(_census_split(name, entry.sequences, min_length=min_length))
    for split in split_reports:
        _log_split_summary(split)

    combined: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for finding in report_findings:
        key = (finding.reason_code, finding.message)
        if key not in seen:
            seen.add(key)
            combined.append(finding)
    return DataReport(
        splits=split_reports,
        findings=combined,
        seq_col=next(iter(seq_names)) if len(seq_names) == 1 else None,
        label_col=next(iter(label_names)) if len(label_names) == 1 else None,
        min_length=min_length,
        label_dtype=label_dtype,
        multilabel_sep=multilabel_sep,
    )


def _ensure_string_list(sequences: list) -> None:
    """Raise a matchable ValueError when a raw sequence list holds non-strings."""
    for position, item in enumerate(sequences):
        if not isinstance(item, str):
            raise ValueError(
                f"validate_data list input must contain only strings; "
                f"element {position} is {type(item).__name__!r}."
            )


def _looks_like_file_paths(items: list) -> bool:
    """A list whose every element is a path with a supported data suffix."""
    if not items:
        return False
    return all(
        isinstance(item, (str, Path)) and str(item).lower().endswith(_SUFFIX_NEEDLES)
        for item in items
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone reader (D-03) — parity-locked against DNADataset.load_local_data
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SourceRead:
    """Internal read result of one file source."""

    sequences: list[str]
    labels: list[str] | None
    seq_name: str | None
    label_name: str | None
    file_type: str
    findings: list[Finding]


def _read_source(file_path, *, seq_col: str | None, label_col: str | None) -> _SourceRead:
    """Full read pipeline for one file source: dispatch + column detection."""
    paths, file_type = _normalize_paths(file_path)
    if file_type in _REFUSED_SUFFIXES:
        return _SourceRead(
            sequences=[],
            labels=None,
            seq_name=None,
            label_name=None,
            file_type=file_type,
            findings=[
                Finding(
                    severity=Severity.ERROR,
                    reason_code=DataReasonCode.UNSUPPORTED_FORMAT_PKL.value,
                    message=(
                        f"pickle inputs are refused and never unpickled (source: {paths[0]})."
                    ),
                )
            ],
        )
    sep = _determine_separator(file_type, None)
    effective_type = _check_header_and_type(
        file_path, paths, file_type, seq_col or _SEQ_NAME, label_col or _LABEL_NAME
    )
    if effective_type in _STRUCTURED_SUFFIXES:
        frame = _load_table(paths, effective_type, sep)
        findings: list[Finding] = []
        detected_seq, detected_label = _detect_columns(frame, seq_col, label_col, findings)
        if detected_seq is None:
            # No usable sequence column: report the error finding, census nothing.
            sequences, labels = [], None
        else:
            sequences, labels = _frame_columns(frame, detected_seq, detected_label)
        return _SourceRead(
            sequences, labels, detected_seq, detected_label, effective_type, findings
        )
    if effective_type in _FASTA_SUFFIXES:
        if len(paths) > 1:
            raise ValueError("FASTA files must be single files, not lists")
        sequences, labels = _load_fasta(paths[0], "|")
        return _SourceRead(sequences, labels, _SEQ_NAME, _LABEL_NAME, effective_type, [])
    if effective_type == "txt":
        if len(paths) > 1:
            raise ValueError("TXT files must be single files, not lists")
        sequences, labels = _load_txt(paths[0], seq_col or _SEQ_NAME, label_col or _LABEL_NAME, sep)
        return _SourceRead(sequences, labels, _SEQ_NAME, _LABEL_NAME, "txt", [])
    raise ValueError(f"Unsupported file type: {file_type}.")


def _read_local_values(
    file_path, *, seq_col: str = _SEQ_NAME, label_col: str = _LABEL_NAME
) -> tuple[list[str], list[str] | None]:
    """Parity seam: raw ``(sequences, labels)`` exactly as the reader resolves them.

    This is the function ``tests/preflight/test_reader_parity.py`` double-
    reads against ``DNADataset.load_local_data`` — same dispatch, same
    demotion rule, same parsers; no column autodetect (explicit names win,
    mirroring the parity reference).
    """
    paths, file_type = _normalize_paths(file_path)
    if file_type in _REFUSED_SUFFIXES:
        raise ValueError(
            f"pickle inputs are refused ({DataReasonCode.UNSUPPORTED_FORMAT_PKL.value})."
        )
    sep = _determine_separator(file_type, None)
    effective_type = _check_header_and_type(file_path, paths, file_type, seq_col, label_col)
    if effective_type in _STRUCTURED_SUFFIXES:
        frame = _load_table(paths, effective_type, sep)
        return _frame_columns(frame, seq_col, label_col)
    if effective_type in _FASTA_SUFFIXES:
        if len(paths) > 1:
            raise ValueError("FASTA files must be single files, not lists")
        return _load_fasta(paths[0], "|")
    if effective_type == "txt":
        if len(paths) > 1:
            raise ValueError("TXT files must be single files, not lists")
        return _load_txt(paths[0], seq_col, label_col, sep)
    raise ValueError(f"Unsupported file type: {file_type}.")


def _normalize_paths(file_path) -> tuple[list[str], str]:
    """Expand user paths, verify existence, and derive the suffix file type."""
    if isinstance(file_path, (list, tuple)):
        if not file_path:
            raise ValueError("Empty list of data file paths.")
        paths = [os.path.expanduser(os.fspath(p)) for p in file_path]
    else:
        paths = [os.path.expanduser(os.fspath(file_path))]
    for path in paths:
        if not os.path.isfile(path):
            raise ValueError(f"Data file not found: {path}.")
    file_type = os.path.basename(paths[0]).split(".")[-1].lower()
    return paths, file_type


def _determine_separator(file_type: str, sep: str | None) -> str | None:
    """Separator defaults: csv -> comma, tsv -> tab, else the caller's (D-03 parity)."""
    if file_type == "csv":
        return sep if sep else ","
    if file_type == "tsv":
        return sep if sep else "\t"
    return sep


def _check_header_and_type(
    original, paths: list[str], file_type: str, seq_col: str, label_col: str
) -> str:
    """Demote headerless csv/tsv to the txt rule (data.py:220-235, parity).

    A single csv/tsv/txt file whose first line contains neither the sequence
    nor the label column name is parsed by the txt rule instead.
    """
    if file_type in ("csv", "tsv", "txt") and not isinstance(original, (list, tuple)):
        with open(paths[0]) as f:
            header = f.readline().strip()
        if not header or (seq_col not in header and label_col not in header):
            return "txt"
    return file_type


def _load_table(paths: list[str], file_type: str, sep: str | None):
    """Load structured formats via pandas / pyarrow (no datasets import)."""
    try:
        if file_type in ("csv", "tsv"):
            frames = [pd.read_csv(path, sep=sep) for path in paths]
        elif file_type == "json":
            frames = []
            for path in paths:
                try:
                    frames.append(pd.read_json(path))
                except ValueError:
                    frames.append(pd.read_json(path, lines=True))
        else:  # parquet / arrow
            import pyarrow.parquet as pq

            return pq.read_table(paths).to_pandas()
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    except ValueError as e:
        raise ValueError(f"Failed to read data file {paths[0]}: {e}") from e


def _load_fasta(path: str, fasta_sep: str) -> tuple[list[str], list[str]]:
    """Parse fasta: label = header.split(fasta_sep)[-1] if present else header.

    Sequences accumulate across lines until the next '>' line
    (data.py:259-279, replicated verbatim).
    """
    sequences, labels = [], []
    with open(path) as f:
        seq = ""
        lab = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq and lab is not None:
                    sequences.append(seq)
                    labels.append(lab)
                header = line[1:].strip()
                lab = header.split(fasta_sep)[-1] if fasta_sep in header else header
                seq = ""
            else:
                seq += line.strip()
        if seq and lab is not None:
            sequences.append(seq)
            labels.append(lab)
    return sequences, labels


def _load_txt(path: str, seq_col: str, label_col: str, sep: str | None):
    """Parse txt: csv re-parse when the first line names both columns, else split.

    Plain lines split on ``sep`` (or whitespace), keeping the first two
    fields; lines with fewer than two fields are skipped (data.py:282-300).
    """
    sequences, labels = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and seq_col in line and label_col in line:
                frame = _load_table([path], "csv", sep if sep else ",")
                seq_effective = seq_col if seq_col in frame.columns else frame.columns[0]
                label_effective = label_col if label_col in frame.columns else frame.columns[1]
                return (
                    [_cell_str(v) for v in frame[seq_effective].tolist()],
                    [_cell_str(v) for v in frame[label_effective].tolist()],
                )
            record = line.strip().split(sep) if sep else line.strip().split()
            if len(record) >= 2:
                sequences.append(record[0])
                labels.append(record[1])
    return sequences, labels


def _frame_columns(frame, seq_col: str | None, label_col: str | None):
    """Extract the (sequence, label) column values from a loaded frame."""
    if seq_col is None or seq_col not in frame.columns:
        raise ValueError(f"Sequence column {seq_col!r} not found in tabular data.")
    sequences = [_cell_str(v) for v in frame[seq_col].tolist()]
    labels = None
    if label_col is not None and label_col in frame.columns:
        labels = [_cell_str(v) for v in frame[label_col].tolist()]
    return sequences, labels


def _cell_is_null(value) -> bool:
    """Scalar-safe null check: None/NaN/NA scalars are null; never a crash.

    ``bool(pd.isna(v))`` raises ValueError on non-scalar cells (list/dict
    values — routine in JSON columns, legal in parquet) because ``pd.isna``
    returns an array. Such cells are not null: callers stringify them
    instead of crashing (CR-02).
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell_str(value) -> str:
    """Stringify a cell; null cells (NaN/NA/None) become empty strings."""
    if _cell_is_null(value):
        return ""
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Column autodetect (D-02)
# ─────────────────────────────────────────────────────────────────────────────


def _detect_columns(frame, seq_col: str | None, label_col: str | None, findings) -> tuple:
    """Resolve the sequence/label columns: explicit parameters always win."""
    columns = list(frame.columns)
    if seq_col is not None:
        detected_seq = seq_col if seq_col in columns else None
        if detected_seq is None:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    reason_code=DataReasonCode.SEQUENCE_COLUMN_NOT_FOUND.value,
                    message=f"Sequence column {seq_col!r} not found in data columns {columns}.",
                )
            )
    else:
        detected_seq = _autodetect_sequence_column(frame)
        if detected_seq is None:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    reason_code=DataReasonCode.SEQUENCE_COLUMN_NOT_FOUND.value,
                    message=(
                        f"No column qualifies as a sequence column "
                        f"(dna score >= {_DNA_SCORE_THRESHOLD} required over columns {columns})."
                    ),
                )
            )

    if label_col is not None:
        detected_label = label_col if label_col in columns else None
        if detected_label is None:
            findings.append(
                Finding(
                    severity=Severity.WARN,
                    reason_code=DataReasonCode.UNREADABLE_COLUMN.value,
                    message=f"Label column {label_col!r} not found in data columns {columns}.",
                )
            )
    else:
        detected_label = _guess_label_column(frame, detected_seq)
        if detected_label is not None:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    reason_code=DataReasonCode.LABEL_COLUMN_GUESSED.value,
                    message=f"Label column {detected_label!r} inferred.",
                )
            )
    return detected_seq, detected_label


def _autodetect_sequence_column(frame) -> str | None:
    """Pick the ACGT-dominant column: best (dna_score, median_length), leftmost on ties."""
    best_column = None
    best_key: tuple[float, float] | None = None
    for column in frame.columns:
        score, median_length = _dna_column_score(frame[column])
        if score < _DNA_SCORE_THRESHOLD:
            continue
        key = (score, median_length)
        if best_key is None or key > best_key:
            best_key = key
            best_column = column
    return best_column


def _dna_column_score(series) -> tuple[float, float]:
    """Score one column: mean per-cell DNA-char fraction and median cell length.

    Scoring runs over up to the first 1000 non-null cells; a cell's score is
    the fraction of its uppercased characters inside the IUPAC DNA alphabet
    (empty cells score 0.0).
    """
    cells = [v for v in series.tolist() if not _cell_is_null(v)][:_AUTODETECT_CELL_LIMIT]
    if not cells:
        return (0.0, 0.0)
    scores = []
    lengths = []
    for value in cells:
        text = str(value)
        upper = text.upper()
        lengths.append(len(text))
        if upper:
            scores.append(sum(1 for ch in upper if ch in _DNA_ALPHABET) / len(upper))
        else:
            scores.append(0.0)
    return (float(np.mean(scores)), float(np.median(lengths)))


def _guess_label_column(frame, seq_column: str | None) -> str | None:
    """Guess the label column: candidate names first, then the 2-column fallback."""
    columns = [c for c in frame.columns if c != seq_column]
    lowered = {str(c).lower(): c for c in columns}
    for candidate in _LABEL_NAME_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    if len(list(frame.columns)) == 2 and len(columns) == 1:
        return columns[0]
    return None


def _guess_label_metadata(labels) -> tuple[str | None, str | None]:
    """Guess label dtype and multilabel separator — reported, never applied."""
    values = [v for v in labels if v != ""]
    if not values:
        return (None, None)
    texts = [str(v) for v in values]
    for separator in _MULTILABEL_SEPARATORS:
        if len(texts) >= 2 and all(separator in text for text in texts):
            return ("multilabel", separator)
    try:
        [float(text) for text in texts]
        return ("float", None)
    except ValueError:
        return ("str", None)


# ─────────────────────────────────────────────────────────────────────────────
# Census (D-09)
# ─────────────────────────────────────────────────────────────────────────────


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
    distinct_count = len(set(sequences))
    duplicate_count = row_count - distinct_count

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
                f"across {distinct_count} distinct sequences.",
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
