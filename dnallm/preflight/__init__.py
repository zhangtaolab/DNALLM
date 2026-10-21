"""dnallm.preflight — pure, read-only pre-flight validation API.

The pre-flight core takes a user from their own data to a validated,
model-compatible verdict/config without ever mutating the data, the source
files, or any cache:

1. Pure functions: ``validate_data``, ``check_compatibility``,
   ``generate_config``, and ``edit_config`` are synchronous, read-only,
   and offline-safe — reports are return values only; nothing is written,
   dropped, or rewritten (a config is written only when the caller passes
   ``output_path`` explicitly).
2. Availability (``probe_availability``) is the one network-capable
   path: opt-in, existence-and-metadata only, never downloading; every
   probe failure degrades to an unknown state with a reason — never an
   exception, never an "unavailable" verdict. ``effective_endpoints``
   reports the library-resolved endpoints/cache roots and
   ``clear_probe_memo`` resets the in-process probe memo.
3. The registry is the sole model-metadata source (REG-01): every verdict
   fact flows through ``dnallm.models.registry`` — this package never
   parses the packaged registry YAML and never probes the network on the
   validation/compatibility/config paths.
4. Import-light at SOURCE and RUNTIME level: modules under
   ``dnallm/preflight/`` never import torch/transformers/datasets
   (AST-enforced by ``tests/preflight/test_structure.py``); hub clients are
   confined to ``availability.py`` (23-03). Runtime ``sys.modules`` purity
   is closed: the parent package facades resolve their heavy members
   lazily, so importing this package leaves torch, transformers, datasets,
   and matplotlib out of ``sys.modules`` — enforced by the subprocess
   marker test in ``tests/test_package_facade.py``.
5. Not a root-facade member: import from ``dnallm.preflight`` directly —
   the root ``dnallm`` package stays torch-heavy by design.

Example:
    from dnallm.preflight import check_compatibility, generate_config, validate_data

    report = validate_data(["ACGT" * 10, "ACGTN" * 8])
    result = check_compatibility(
        report, models=["zhangtaolab/plant-dnabert-BPE-promoter"], task_type="binary"
    )
    verdict = result.verdicts[0]
    generated = generate_config("zhangtaolab/plant-dnabert-BPE-promoter")
"""

from .availability import clear_probe_memo, effective_endpoints, probe_availability
from .compatibility import check_compatibility
from .config_gen import edit_config, generate_config
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
    "clear_probe_memo",
    "edit_config",
    "effective_endpoints",
    "generate_config",
    "probe_availability",
    "validate_data",
]
