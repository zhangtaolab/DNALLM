"""check_compatibility: registry-driven model-suitability verdicts (PFC-02 tracer).

This module turns a :class:`~dnallm.preflight.report.DataReport` into
per-model :class:`~dnallm.preflight.report.ModelVerdict` objects. The
registry (``dnallm.models.registry``, Phase 22) is the sole metadata
source: every verdict fact resolves through ``try_resolve`` /
``effective_max_length`` — this module never opens the packaged registry file
itself and never touches the network.

This tracer slice implements the length dimension only, at char-level 1:1
against the entry's effective max context length. The tokenizer-kind budget
table (k-mer ÷ k, BPE conservative ratio), task-dimension verdicts, and
intent-routed listings land in 23-02. Verdict semantics follow D-05/D-06:
truncation warns with exact counts and never fails; a registry miss is a
per-model fail verdict, never an exception.
"""

from collections.abc import Iterable

from ..models.registry import effective_max_length, is_lab_model, try_resolve
from ..utils import get_logger
from .report import (
    CompatReasonCode,
    CompatibilityReport,
    DimensionVerdict,
    ModelVerdict,
    VerdictStatus,
)

logger = get_logger("dnallm.preflight.compatibility")

__all__ = ["check_compatibility"]

# Dimension names (23-02 extends this vocabulary with task/charset).
DIMENSION_LENGTH = "length"
DIMENSION_REGISTRY = "registry"

# Worst-of aggregation rank: fail > warn > unknown > ok.
_STATUS_RANK = {
    VerdictStatus.OK: 0,
    VerdictStatus.UNKNOWN: 1,
    VerdictStatus.WARN: 2,
    VerdictStatus.FAIL: 3,
}


def check_compatibility(
    data=None,
    *,
    task_type: str | None = None,
    models: Iterable[str] | None = None,
) -> CompatibilityReport:
    """Check data suitability per requested model (tracer: length dimension).

    Args:
        data: The :class:`~dnallm.preflight.report.DataReport` to check
            against, or None when no data was validated yet.
        task_type: Intended task type (e.g. ``"binary"``). Normalized via
            ``lower().strip()``; task-dimension verdicts land in 23-02 —
            the argument is accepted and normalized here so the signature
            is stable for shells.
        models: Model names / repo ids to check. Each resolves through the
            registry; a miss becomes a fail MODEL_NOT_IN_REGISTRY verdict
            (per-model isolation, never an exception).

    Returns:
        The compatibility report: one verdict per requested model in request
        order. Overall status per model is the worst dimension status
        (fail > warn > unknown > ok). ``listing`` stays empty in this slice.
    """
    normalized_task = task_type.lower().strip() if task_type else None
    if normalized_task is not None:
        logger.debug(
            f"check_compatibility: task_type {normalized_task!r} accepted; "
            "task-dimension verdicts land in 23-02"
        )

    verdicts: list[ModelVerdict] = []
    for name in models or []:
        entry = try_resolve(name)
        if entry is None:
            verdicts.append(
                ModelVerdict(
                    model=name,
                    entry_name=None,
                    is_lab=False,
                    dimensions=[
                        DimensionVerdict(
                            dimension=DIMENSION_REGISTRY,
                            status=VerdictStatus.FAIL,
                            reason_code=CompatReasonCode.MODEL_NOT_IN_REGISTRY.value,
                            details={"model": name},
                        )
                    ],
                    overall=VerdictStatus.FAIL,
                )
            )
            logger.info(
                f"check_compatibility: {name!r} not in registry "
                f"({CompatReasonCode.MODEL_NOT_IN_REGISTRY.value})"
            )
            continue
        dimensions = [_length_dimension(entry, data)]
        overall = max(dimensions, key=lambda d: _STATUS_RANK[d.status]).status
        verdicts.append(
            ModelVerdict(
                model=entry.model,
                entry_name=entry.name,
                is_lab=is_lab_model(entry),
                dimensions=dimensions,
                overall=overall,
            )
        )
    return CompatibilityReport(verdicts=verdicts, listing=[])


def _length_dimension(entry, data) -> DimensionVerdict:
    """Compute the length dimension: truncation counts against the char budget.

    Tracer budget: the entry's effective max context length used 1:1 as a
    character budget (the tokenizer-kind budget table lands in 23-02).
    Truncation warns with exact counts — it never fails (D-06).
    """
    if data is None:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.NO_DATA.value,
            details={"model": entry.model},
        )
    total_rows = sum(split.row_count for split in data.splits)
    if total_rows == 0:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.EMPTY_DATASET_DIM.value,
            details={"model": entry.model, "row_count": 0},
        )
    budget = effective_max_length(entry)
    if budget is None:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.TOKENIZER_UNKNOWN.value,
            details={"model": entry.model},
        )
    truncated_row_count = sum(
        count
        for split in data.splits
        for length, count in split.length_counts.items()
        if length > budget
    )
    if truncated_row_count:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.WARN,
            reason_code=CompatReasonCode.TRUNCATION_ABOVE_THRESHOLD.value,
            details={
                "budget_chars": budget,
                "truncated_row_count": truncated_row_count,
                "truncated_fraction": truncated_row_count / total_rows,
            },
        )
    return DimensionVerdict(
        dimension=DIMENSION_LENGTH,
        status=VerdictStatus.OK,
        reason_code=None,
        details={
            "budget_chars": budget,
            "truncated_row_count": 0,
            "truncated_fraction": 0.0,
        },
    )
