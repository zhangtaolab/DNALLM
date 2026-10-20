"""dnallm.preflight — pure, read-only pre-flight validation API.

The pre-flight core takes a user from their own data to a validated,
model-compatible verdict/config without ever mutating the data, the source
files, or any cache:

1. Pure functions: ``validate_data`` and ``check_compatibility`` are
   synchronous, read-only, and offline-safe — reports are return values
   only; nothing is written, dropped, or rewritten.
2. The registry is the sole model-metadata source (REG-01): every verdict
   fact flows through ``dnallm.models.registry`` — this package never
   parses the packaged registry YAML and never probes the network on this path.
3. Import-light: modules under ``dnallm/preflight/`` never import
   torch/transformers/datasets (source-level discipline, enforced by
   ``tests/preflight/test_structure.py``); hub clients are confined to
   ``availability.py`` (23-03).
4. Not a root-facade member: import from ``dnallm.preflight`` directly —
   the root ``dnallm`` package stays torch-heavy by design.

Example:
    from dnallm.preflight import validate_data, check_compatibility

    report = validate_data(["ACGT" * 10, "ACGTN" * 8])
    result = check_compatibility(
        report, models=["zhangtaolab/plant-dnabert-BPE-promoter"], task_type="binary"
    )
    verdict = result.verdicts[0]
"""

from .compatibility import check_compatibility
from .report import (
    AvailabilityReasonCode,
    AvailabilityReport,
    AvailabilityState,
    CharsetCensusEntry,
    CompatReasonCode,
    CompatibilityReport,
    ConfigReasonCode,
    DataReasonCode,
    DataReport,
    DimensionVerdict,
    Finding,
    GeneratedConfig,
    Hub,
    LengthStats,
    ListingEntry,
    ModelVerdict,
    SectionError,
    Severity,
    SplitReport,
    VerdictStatus,
)
from .validation import validate_data

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
    "check_compatibility",
    "validate_data",
]
