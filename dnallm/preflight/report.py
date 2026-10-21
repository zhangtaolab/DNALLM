"""Typed report and verdict objects for the pre-flight core (PFC-01..PFC-07).

1. Reports are facts, not state: every model in this module is a frozen
   Pydantic object (``ConfigDict(frozen=True)``) — once built, a report can
   never be mutated by a consumer, matching the Phase 22 registry-entry
   precedent (``dnallm/models/registry.py``).
2. This file is the complete type inventory for the whole phase: plan 23-02
   (full compatibility dimensions + listings), 23-03 (availability probing),
   and 23-04 (config generation) import these types and never edit this
   module — types are born here, interface-first, before their producers.
3. Reason codes are machine-readable UPPER_SNAKE strings (``str`` enums) —
   agent-friendly by design: every finding and every dimension verdict
   carries one, so shells (Phase 25 TUI, Phase 26 MCP) can route on codes
   without owning verdict logic.
4. Import-light discipline, enforced at the SOURCE level: this module's
   own imports are nothing beyond the stdlib, pydantic, and the package
   logger — never torch/transformers/datasets (AST-enforced by
   ``tests/preflight/test_structure.py``). Runtime ``sys.modules`` purity
   is closed as well: the ``dnallm.utils`` and ``dnallm.models`` parent
   package facades resolve their heavy members lazily, so the logger and
   registry imports this module performs pull no part of the training
   stack — pinned by the subprocess marker test in
   ``tests/test_package_facade.py``.
5. Reports are return values only: nothing here reads files, touches the
   network, or mutates user data (the pure/read-only contract of
   ``dnallm.preflight``).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..utils import get_logger

logger = get_logger("dnallm.preflight.report")

__all__ = [
    "AvailabilityReasonCode",
    "AvailabilityReport",
    "AvailabilityState",
    "CharsetCensusEntry",
    "CompatReasonCode",
    "CompatibilityReport",
    "ConfigReasonCode",
    "DataReasonCode",
    "DataReport",
    "DimensionVerdict",
    "Finding",
    "GeneratedConfig",
    "Hub",
    "LengthStats",
    "ListingEntry",
    "ModelVerdict",
    "SectionError",
    "Severity",
    "SplitReport",
    "VerdictStatus",
]


class Severity(StrEnum):
    """Severity of a data finding: annotations only, never a blocker."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class VerdictStatus(StrEnum):
    """Status of a compatibility verdict.

    FAIL is reserved for hard incompatibility (decision D-05): anything
    mitigable by config or data cleanup is at most WARN.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class AvailabilityState(StrEnum):
    """Outcome of an availability probe (PFC-03): never 'unavailable'."""

    CACHED = "cached"
    REMOTE_CONFIRMED = "remote_confirmed"
    UNKNOWN = "unknown"


class Hub(StrEnum):
    """A hub or local source a model can live on (ModelScope-first, D-11)."""

    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    NONE = "none"


class DataReasonCode(StrEnum):
    """Reason codes for data-validation findings (census conventions D-09)."""

    EMPTY_DATASET = "EMPTY_DATASET"
    EMPTY_SEQUENCE = "EMPTY_SEQUENCE"
    TOO_SHORT = "TOO_SHORT"
    ALL_N = "ALL_N"
    DUPLICATE_SEQUENCES = "DUPLICATE_SEQUENCES"
    UNSUPPORTED_FORMAT_PKL = "UNSUPPORTED_FORMAT_PKL"
    SEQUENCE_COLUMN_NOT_FOUND = "SEQUENCE_COLUMN_NOT_FOUND"
    LABEL_COLUMN_GUESSED = "LABEL_COLUMN_GUESSED"
    UNREADABLE_COLUMN = "UNREADABLE_COLUMN"
    READ_ERROR = "READ_ERROR"


class CompatReasonCode(StrEnum):
    """Reason codes for compatibility dimension verdicts (D-05..D-08)."""

    MODEL_NOT_IN_REGISTRY = "MODEL_NOT_IN_REGISTRY"
    TASK_TYPE_MISMATCH = "TASK_TYPE_MISMATCH"
    HEAD_SUPPORT_UNCERTAIN = "HEAD_SUPPORT_UNCERTAIN"
    TRUNCATION_ABOVE_THRESHOLD = "TRUNCATION_ABOVE_THRESHOLD"
    CHARSET_REJECTED_CHARS = "CHARSET_REJECTED_CHARS"
    CHARSET_AMBIGUITY_CHARS = "CHARSET_AMBIGUITY_CHARS"
    CHARSET_NONSTANDARD_CHARS = "CHARSET_NONSTANDARD_CHARS"
    TOKENIZER_UNKNOWN = "TOKENIZER_UNKNOWN"
    NO_DATA = "NO_DATA"
    EMPTY_DATASET_DIM = "EMPTY_DATASET_DIM"


class AvailabilityReasonCode(StrEnum):
    """Reason codes for availability degradation (PFC-06: failure is UNKNOWN)."""

    NOT_FOUND_ON_EITHER_HUB = "NOT_FOUND_ON_EITHER_HUB"
    RATE_LIMITED = "RATE_LIMITED"
    PROBE_TIMEOUT = "PROBE_TIMEOUT"
    OFFLINE = "OFFLINE"
    PROBE_ERROR = "PROBE_ERROR"


class ConfigReasonCode(StrEnum):
    """Reason codes for config-generation section errors (PFC-04)."""

    MODEL_NOT_IN_REGISTRY_CFG = "MODEL_NOT_IN_REGISTRY_CFG"
    TASK_TYPE_REQUIRED = "TASK_TYPE_REQUIRED"
    UNKNOWN_SECTION = "UNKNOWN_SECTION"
    NO_SECTIONS = "NO_SECTIONS"


class Finding(BaseModel):
    """One annotation about the validated data.

    Attributes:
        severity: How severe the finding is (info/warn/error — none of them
            block; the report counts, it never drops rows).
        reason_code: Machine-readable UPPER_SNAKE code (a
            :class:`DataReasonCode` value).
        message: Human-readable description (never contains raw sequences).
        count: Exact number of affected rows when countable, else None.
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity
    reason_code: str
    message: str
    count: int | None = None


class LengthStats(BaseModel):
    """Sequence-length statistics over one split.

    Attributes:
        n: Number of sequences (equals the split's row count).
        p50: Median length (numpy linear-interpolation quantile).
        p95: 95th-percentile length (numpy linear-interpolation quantile).
        p99: 99th-percentile length (numpy linear-interpolation quantile).
        min: Exact minimum length in Unicode code points.
        max: Exact maximum length in Unicode code points.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    p99: float = Field(ge=0.0)
    min: int = Field(ge=0)
    max: int = Field(ge=0)


class CharsetCensusEntry(BaseModel):
    """One character's occurrence count in the uppercased sequences of a split.

    Attributes:
        char: The (uppercased) character.
        count: How many times it occurs across all sequences of the split.
    """

    model_config = ConfigDict(frozen=True)

    char: str = Field(min_length=1, max_length=1)
    count: int = Field(ge=0)


class SplitReport(BaseModel):
    """Census of one data split (a named slice of the validated data).

    Attributes:
        name: Split name (``"data"`` for unsplit input, the dict key for
            pre-split input).
        row_count: Number of rows counted — never filtered.
        length_stats: Length statistics, or None when the split is empty.
        charset_census: Per-character counts over uppercased sequences,
            entries sorted alphabetically by character.
        findings: Split-level findings (empty sequences, too-short, all-N,
            duplicates, ...).
        lowercase_fraction: Lowercase-character count / total characters.
        mean_gc_content: Mean GC content over all rows (empty rows
            contribute 0.0; normalization via ``calc_gc_content``).
        length_counts: Exact length histogram (length -> row count) — the
            exact-count channel for truncation verdicts, so
            ``check_compatibility`` never needs the raw rows.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    length_stats: LengthStats | None = None
    charset_census: list[CharsetCensusEntry] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    lowercase_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_gc_content: float = Field(default=0.0, ge=0.0, le=1.0)
    length_counts: dict[int, int] = Field(default_factory=dict)


class DataReport(BaseModel):
    """Read-only, collect-all report over the validated data (PFC-01).

    Attributes:
        splits: One :class:`SplitReport` per split.
        findings: Report-level findings applying across splits (column
            detection, unsupported formats, ...).
        seq_col: Sequence column in use (None for raw ``list[str]`` input).
        label_col: Label column in use, when one exists or was guessed.
        min_length: The minimum-length threshold the census was run with.
        label_dtype: Guessed label column dtype (``"float"``/``"int"``/
            ``"str"``/``"multilabel"``), reported as metadata, never applied.
        multilabel_sep: Separator detected inside multi-label string labels,
            when one is detected (metadata only; values are never mutated).
    """

    model_config = ConfigDict(frozen=True)

    splits: list[SplitReport] = Field(min_length=1)
    findings: list[Finding] = Field(default_factory=list)
    seq_col: str | None = None
    label_col: str | None = None
    min_length: int = Field(gt=0)
    label_dtype: str | None = None
    multilabel_sep: str | None = None


class DimensionVerdict(BaseModel):
    """One compatibility dimension's verdict for one model.

    Attributes:
        dimension: Dimension name (``"length"``, ``"task"``, ``"charset"``,
            ``"registry"``).
        status: The verdict status for this dimension.
        reason_code: Machine-readable code when status is not ok, else None.
        details: Quantitative payload (budgets, counts, fractions).
    """

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(min_length=1)
    status: VerdictStatus
    reason_code: str | None = None
    details: dict = Field(default_factory=dict)


class ModelVerdict(BaseModel):
    """Suitability verdict for one model against the validated data.

    Attributes:
        model: Canonical repo id of the resolved registry entry (or the
            requested name when resolution failed).
        entry_name: Registry entry name, or None when unresolved.
        is_lab: Whether the entry is a lab model (``zhangtaolab/`` prefix).
        dimensions: Per-dimension verdicts.
        overall: Worst dimension status (fail > warn > unknown > ok).
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    entry_name: str | None = None
    is_lab: bool = False
    dimensions: list[DimensionVerdict] = Field(min_length=1)
    overall: VerdictStatus


class AvailabilityReport(BaseModel):
    """Availability metadata for one model (PFC-03, produced by 23-03).

    Attributes:
        state: cached / remote_confirmed / unknown — never "unavailable".
        size_bytes: Total repo size in bytes, when known.
        revision: Commit revision, when known.
        source_hub: The hub the model was confirmed on (ModelScope reported
            first when both hubs carry it, D-11).
        reason: Degradation reason when state is unknown.
    """

    model_config = ConfigDict(frozen=True)

    state: AvailabilityState
    size_bytes: int | None = Field(default=None, ge=0)
    revision: str | None = None
    source_hub: Hub | None = None
    reason: str | None = None


class ListingEntry(BaseModel):
    """One row of an intent-routed model listing (PFC-05/PFC-07, 23-02).

    Attributes:
        model: Repo id of the registry entry.
        entry_name: Registry entry name.
        is_lab: Lab-model flag (lab entries rank first, PFC-07).
        task_type: The entry's task type.
        num_labels: Number of labels when the entry ships a head.
        label_names: Ordered label names when the entry ships a head.
        availability: Availability metadata (an :class:`AvailabilityReport`
            for shortlisted entries, else None) — consumed by 23-03.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    entry_name: str = Field(min_length=1)
    is_lab: bool
    task_type: str | None = None
    num_labels: int | None = Field(default=None, gt=0)
    label_names: list[str] | None = None
    availability: AvailabilityReport | None = None


class CompatibilityReport(BaseModel):
    """Result of a compatibility check: per-model verdicts plus listings.

    Attributes:
        verdicts: One :class:`ModelVerdict` per requested model, in request
            order — a registry miss is a fail verdict, never an exception.
        listing: Intent-routed listing rows (populated by 23-02; always
            empty in this tracer slice).
    """

    model_config = ConfigDict(frozen=True)

    verdicts: list[ModelVerdict] = Field(default_factory=list)
    listing: list[ListingEntry] = Field(default_factory=list)


class SectionError(BaseModel):
    """One field-level config validation error (PFC-04, consumed by 23-04).

    Attributes:
        section: Config section the error belongs to (``task``, ``model``...).
        field: Field path inside the section (or the section itself).
        message: Human-readable error message.
    """

    model_config = ConfigDict(frozen=True)

    section: str = Field(min_length=1)
    field: str = Field(min_length=1)
    message: str = Field(min_length=1)


class GeneratedConfig(BaseModel):
    """A generated config plus its collect-all validation errors (PFC-04, 23-04).

    Attributes:
        config: The validated config mapping (sections that validated).
        errors: Field-level errors for every section that failed — one
            section's failure never hides another's.
        raw: The exact mapping that would be written to disk — the assembled
            (``generate_config``) or patched (``edit_config``) raw config,
            unknown sections included. A shell edits and previews this
            mapping, so nothing present in the input file is ever lost
            (D-11); :func:`dnallm.preflight.config_gen.render_yaml` turns it
            into the bytes the writer emits.
    """

    model_config = ConfigDict(frozen=True)

    config: dict = Field(default_factory=dict)
    errors: list[SectionError] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)
