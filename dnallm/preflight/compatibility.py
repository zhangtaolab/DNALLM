"""check_compatibility: registry-driven model-suitability verdicts (PFC-02).

This module turns a :class:`~dnallm.preflight.report.DataReport` into
per-model :class:`~dnallm.preflight.report.ModelVerdict` objects over three
dimensions:

- **task** (D-08): finetuned entries hard-match the requested task type
  (a mismatch is a hard fail); base models are ok for any supervised task;
  mask/generation/embedding backbones asked for a supervised task warn
  ``HEAD_SUPPORT_UNCERTAIN``.
- **length** (D-06): a tokenizer-kind char budget (k-mer sliding window,
  char 1:1, BPE conservative ratio) with exact truncated-row counts.
  Truncation warns with counts and NEVER fails — the k-mer trap (every
  ``-Nmer`` variant shares one tokenizer class that carries no k) is
  neutralized by classifying from the entry name first.
- **charset** (D-07): char-level tokenizer classes are strict (data
  characters outside the DNA core + ambiguity codes fail THAT model only);
  BPE classes never fail charset (anything is encodable); U counts as
  T-equivalent; ambiguity codes warn with per-character counts.

The registry (``dnallm.models.registry``, Phase 22) is the sole metadata
source: every verdict fact resolves through ``try_resolve`` /
``effective_max_length`` / ``effective_tokenizer_type`` — this module never
opens the packaged registry data file itself and never touches the network.
FAIL is reserved for hard incompatibility (D-05): anything mitigable by
config or data cleanup stays warn with counts.
"""

import re
from collections.abc import Callable, Iterable

import numpy as np

from ..models.registry import (
    RegistryEntry,
    all_entries,
    effective_max_length,
    effective_tokenizer_type,
    get_by_task_type,
    is_lab_model,
    sort_key_lab_first,
    try_resolve,
)
from ..utils import get_logger
from .report import (
    AvailabilityReasonCode,
    AvailabilityReport,
    AvailabilityState,
    CompatReasonCode,
    CompatibilityReport,
    DimensionVerdict,
    ListingEntry,
    ModelVerdict,
    VerdictStatus,
)

logger = get_logger("dnallm.preflight.compatibility")

__all__ = ["check_compatibility"]

# Dimension names.
DIMENSION_TASK = "task"
DIMENSION_LENGTH = "length"
DIMENSION_CHARSET = "charset"
DIMENSION_REGISTRY = "registry"

# Worst-of aggregation rank: fail > warn > unknown > ok.
_STATUS_RANK = {
    VerdictStatus.OK: 0,
    VerdictStatus.UNKNOWN: 1,
    VerdictStatus.WARN: 2,
    VerdictStatus.FAIL: 3,
}

# Budget kinds (D-06). The k-mer kind carries the window size k separately.
BUDGET_KIND_KMER = "kmer"
BUDGET_KIND_BPE = "bpe"
BUDGET_KIND_CHAR = "char_1to1"
BUDGET_KIND_UNKNOWN = "unknown"

# BPE-class tokenizers: conservative 3 chars-per-token estimate, erring toward
# over-warning (never blocking) — real BPE merges make sequences SHORTER.
_BPE_TOKENIZER_CLASSES = frozenset({
    "DebertaV2Tokenizer",
    "GemmaTokenizer",
    "GPT2Tokenizer",
    "GPTNeoXTokenizer",
    "PreTrainedTokenizerFast",
})
BPE_CONSERVATIVE_CHARS_PER_TOKEN = 3

# k-mer window size extractor: matches "6mer", "3 mer", ... in entry/model names.
_KMER_NAME_RE = r"(\d+)\s*mer"

# Task-dimension vocabulary (D-08): supervised tasks a head can be built for,
# plus the alias forms normalized into the registry's canonical vocabulary.
_SUPERVISED_TASK_TYPES = frozenset({
    "binary",
    "binary_classification",
    "multiclass",
    "multi_class_classification",
    "multilabel",
    "multi_label_classification",
    "regression",
    "token",
    "token_classification",
})
_TASK_TYPE_ALIASES = {
    "binary_classification": "binary",
    "multi_class_classification": "multiclass",
    "multi_label_classification": "multilabel",
    "token_classification": "token",
}
_HEAD_UNCERTAIN_BACKBONE_TYPES = frozenset({"embedding", "mask", "generation"})

# Charset vocabulary (D-07): core bases, IUPAC ambiguity codes, U (T-equivalent).
_CORE_CHARS = frozenset("ACGT")
_AMBIGUITY_CHARS = frozenset("NRYKMSWBDHV")
_U_CHAR = "U"

_CHARSET_STANCE_STRICT = "strict"
_CHARSET_STANCE_BPE = "bpe"

# Intent routing (PFC-05) over the base<->finetuned axis, with the prober
# shortlist cap (D-10): labs get probed first, never all 236 entries.
INTENT_FINETUNE = "finetune"
INTENT_PREDICT = "predict"
_VALID_INTENTS = frozenset({INTENT_FINETUNE, INTENT_PREDICT})
DEFAULT_PROBE_SHORTLIST = 20


def check_compatibility(
    data=None,
    *,
    task_type: str | None = None,
    models: Iterable[str] | None = None,
    intent: str | None = None,
    prober: Callable[[str], AvailabilityReport] | None = None,
    probe_shortlist: int | None = None,
) -> CompatibilityReport:
    """Check data suitability per model, optionally with an intent listing.

    Args:
        data: The :class:`~dnallm.preflight.report.DataReport` to check
            against, or None when no data was validated yet (length/charset
            degrade to unknown NO_DATA; the task dimension still computes).
        task_type: Intended task type (e.g. ``"binary"``). Normalized via
            ``lower().strip()`` plus alias folding into the registry's
            vocabulary (``binary_classification`` -> ``binary``).
        models: Model names / repo ids to check. Each resolves through the
            registry; a miss becomes a fail MODEL_NOT_IN_REGISTRY verdict
            (per-model isolation, never an exception). When given, no
            listing is built.
        intent: Listing route over the base<->finetuned axis (PFC-05):
            ``"finetune"`` lists pretrained/base entries (optionally
            filtered to warn-or-ok task compatibility when ``task_type`` is
            given); ``"predict"`` lists finetuned entries of
            ``task_type`` that ship task heads (``num_labels``).
            Normalized via ``lower().strip()``; any other value raises.
        prober: Optional availability probe callable (D-10). Without it the
            listing is network-free and every ``availability`` stays None;
            with it, only the first ``probe_shortlist`` listing entries are
            probed (labs first — the listing is lab-first sorted). A raising
            prober degrades that entry to an unknown availability report,
            never an exception (PFC-06).
        probe_shortlist: How many listing entries to probe when a prober is
            given (default cap 20; capped at the listing length).

    Returns:
        The compatibility report: one verdict per requested model in request
        order, each carrying task/length/charset dimension verdicts. Overall
        status per model is the worst dimension status (fail > warn >
        unknown > ok). The intent-routed ``listing`` is lab-first sorted
        (``zhangtaolab/*`` first, then name — PFC-07).

    Raises:
        ValueError: If ``intent`` is neither ``"finetune"`` nor
            ``"predict"`` (after normalization).
    """
    normalized_task = _normalize_task_type(task_type)
    normalized_intent = intent.lower().strip() if intent else None
    if normalized_intent is not None and normalized_intent not in _VALID_INTENTS:
        raise ValueError(
            f"check_compatibility intent must be 'finetune' or 'predict', got {intent!r}."
        )

    model_names = list(models or [])
    verdicts = _verdicts_for_models(model_names, data=data, task_type=task_type)
    listing: list[ListingEntry] = []
    if models is None and (normalized_intent is not None or normalized_task is not None):
        effective_intent = normalized_intent or INTENT_FINETUNE
        listing = _build_listing(_intent_listing(normalized_task, effective_intent))
        if prober is not None:
            listing = _attach_probes(listing, prober, probe_shortlist)
        logger.debug(
            f"check_compatibility: {len(listing)} listing entr(ies) for intent "
            f"{effective_intent!r} task {normalized_task!r}"
        )
    elif model_names:
        logger.debug(
            f"check_compatibility: {len(verdicts)} model verdict(s) over "
            "task/length/charset dimensions"
        )
    return CompatibilityReport(verdicts=verdicts, listing=listing)


def _verdicts_for_models(
    model_names: list[str], *, data, task_type: str | None
) -> list[ModelVerdict]:
    """Build one ModelVerdict per requested model name, in request order."""
    verdicts: list[ModelVerdict] = []
    for name in model_names:
        entry = try_resolve(name)
        if entry is None:
            verdicts.append(_registry_miss_verdict(name))
            logger.info(
                f"check_compatibility: {name!r} not in registry "
                f"({CompatReasonCode.MODEL_NOT_IN_REGISTRY.value})"
            )
            continue
        dimensions = [
            _task_verdict(entry, task_type),
            _length_verdict(data, entry),
            _charset_verdict(data, entry),
        ]
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
    return verdicts


def _registry_miss_verdict(name: str) -> ModelVerdict:
    """The fail MODEL_NOT_IN_REGISTRY verdict for an unresolvable model."""
    return ModelVerdict(
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


# ─────────────────────────────────────────────────────────────────────────────
# Task dimension (D-08)
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_task_type(task_type: str | None) -> str | None:
    """Normalize user task input: lower().strip() plus alias folding."""
    if task_type is None:
        return None
    lowered = task_type.lower().strip()
    return _TASK_TYPE_ALIASES.get(lowered, lowered)


def _task_verdict(entry: RegistryEntry, task_type: str | None) -> DimensionVerdict:
    """Compute the task dimension: hard match for finetuned entries (D-08).

    Finetuned entries hard-match: a different requested task type is a hard
    fail (TASK_TYPE_MISMATCH). Base/pretrained entries are ok for any
    supervised task; embedding/mask/generation backbones asked for a
    supervised task warn HEAD_SUPPORT_UNCERTAIN (a head must be bolted on,
    support is not guaranteed). With no task type requested there is nothing
    to contradict: the dimension is ok.
    """
    details = {
        "model": entry.model,
        "entry_kind": entry.kind,
        "entry_task_type": entry.task.task_type,
        "requested_task_type": _normalize_task_type(task_type),
    }
    if task_type is None:
        return DimensionVerdict(
            dimension=DIMENSION_TASK, status=VerdictStatus.OK, reason_code=None, details=details
        )
    requested = _normalize_task_type(task_type)
    own = entry.task.task_type.lower().strip()
    if entry.kind == "finetuned":
        if requested != own:
            return DimensionVerdict(
                dimension=DIMENSION_TASK,
                status=VerdictStatus.FAIL,
                reason_code=CompatReasonCode.TASK_TYPE_MISMATCH.value,
                details=details,
            )
        return DimensionVerdict(
            dimension=DIMENSION_TASK, status=VerdictStatus.OK, reason_code=None, details=details
        )
    if requested in _SUPERVISED_TASK_TYPES:
        if own in _HEAD_UNCERTAIN_BACKBONE_TYPES:
            return DimensionVerdict(
                dimension=DIMENSION_TASK,
                status=VerdictStatus.WARN,
                reason_code=CompatReasonCode.HEAD_SUPPORT_UNCERTAIN.value,
                details=details,
            )
        return DimensionVerdict(
            dimension=DIMENSION_TASK, status=VerdictStatus.OK, reason_code=None, details=details
        )
    # Non-supervised request on a base model (e.g. mask on a mask backbone):
    # no conflict is provable — ok.
    return DimensionVerdict(
        dimension=DIMENSION_TASK, status=VerdictStatus.OK, reason_code=None, details=details
    )


# ─────────────────────────────────────────────────────────────────────────────
# Length dimension (D-06): tokenizer-kind char budgets
# ─────────────────────────────────────────────────────────────────────────────


def _k_from_name(entry: RegistryEntry) -> int | None:
    """Extract the k-mer window size from the entry name, then the repo id.

    The tokenizer class never carries k (every ``-Nmer`` variant is typed as
    the same char-level class), so the name is the only k source.
    """
    match = re.search(_KMER_NAME_RE, entry.name) or re.search(_KMER_NAME_RE, entry.model)
    return int(match.group(1)) if match else None


def _budget_facts(entry: RegistryEntry) -> tuple[str, int | None, int | None]:
    """Classify the entry's char budget: ``(kind, k, budget_chars)``.

    Classification order (D-06):

    1. k-mer (k from the name): ``budget_chars = max_length + k - 1`` — the
       sliding-window inversion (a length-L sequence yields L - k + 1 k-mers).
    2. BPE-class tokenizer: ``budget_chars = max_length x 3`` — the
       conservative chars-per-token estimate (over-warn, never block).
    3. Any other known tokenizer class: char 1:1, ``budget_chars = max_length``.
    4. Unknown tokenizer or missing max length: kind unknown, budget None.
    """
    max_length = effective_max_length(entry)
    k = _k_from_name(entry)
    if k is not None and max_length is not None:
        return (BUDGET_KIND_KMER, k, max_length + k - 1)
    tokenizer_type = effective_tokenizer_type(entry)
    if tokenizer_type is None or max_length is None:
        return (BUDGET_KIND_UNKNOWN, k, None)
    if tokenizer_type in _BPE_TOKENIZER_CLASSES:
        return (BUDGET_KIND_BPE, None, max_length * BPE_CONSERVATIVE_CHARS_PER_TOKEN)
    return (BUDGET_KIND_CHAR, None, max_length)


def _budget_kind(entry: RegistryEntry) -> str:
    """The entry's budget kind: kmer | bpe | char_1to1 | unknown."""
    return _budget_facts(entry)[0]


def _char_budget(entry: RegistryEntry) -> int | None:
    """The entry's character budget, or None when the kind is unknown."""
    return _budget_facts(entry)[2]


def _length_verdict(data, entry: RegistryEntry) -> DimensionVerdict:
    """Compute the length dimension: truncation counts against the budget.

    Census comes from the report's exact length histograms — raw rows are
    never needed. Truncation warns with exact counts; it never fails (D-06),
    at any fraction — even 100% of rows.
    """
    if data is None:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.NO_DATA.value,
            details={"model": entry.model},
        )
    rows, lengths = _combined_length_histogram(data)
    if rows == 0 or not lengths:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.EMPTY_DATASET_DIM.value,
            details={"model": entry.model, "row_count": rows},
        )
    kind, k, budget = _budget_facts(entry)
    if budget is None:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.TOKENIZER_UNKNOWN.value,
            details={
                "model": entry.model,
                "kind": kind,
                "tokenizer_type": effective_tokenizer_type(entry),
            },
        )
    expanded = [length for length, count in lengths.items() for _ in range(count)]
    quantiles = np.quantile(expanded, [0.5, 0.95, 0.99])
    truncated_row_count = sum(count for length, count in lengths.items() if length > budget)
    details = {
        "budget_chars": budget,
        "kind": kind,
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": max(lengths),
        "truncated_row_count": truncated_row_count,
        "truncated_fraction": truncated_row_count / rows,
    }
    if k is not None:
        details["k"] = k
    if truncated_row_count:
        return DimensionVerdict(
            dimension=DIMENSION_LENGTH,
            status=VerdictStatus.WARN,
            reason_code=CompatReasonCode.TRUNCATION_ABOVE_THRESHOLD.value,
            details=details,
        )
    return DimensionVerdict(
        dimension=DIMENSION_LENGTH, status=VerdictStatus.OK, reason_code=None, details=details
    )


def _combined_length_histogram(data) -> tuple[int, dict[int, int]]:
    """Sum row counts and merge exact length histograms across all splits."""
    rows = 0
    histogram: dict[int, int] = {}
    for split in data.splits:
        rows += split.row_count
        for length, count in split.length_counts.items():
            histogram[length] = histogram.get(length, 0) + count
    return rows, histogram


# ─────────────────────────────────────────────────────────────────────────────
# Charset dimension (D-07)
# ─────────────────────────────────────────────────────────────────────────────


def _charset_stance(tokenizer_type: str | None) -> str | None:
    """The entry's charset stance: strict | bpe | None (unknown).

    Char-level tokenizer classes are strict (characters outside core +
    ambiguity + U are rejected by the model's vocabulary); BPE classes never
    fail charset (anything is encodable, possibly into unknown tokens); a
    tokenizer gap cannot prove hard incompatibility, so it never fails.
    """
    if tokenizer_type is None:
        return None
    if tokenizer_type in _BPE_TOKENIZER_CLASSES:
        return _CHARSET_STANCE_BPE
    return _CHARSET_STANCE_STRICT


def _charset_verdict(data, entry: RegistryEntry) -> DimensionVerdict:
    """Compute the charset dimension with per-model isolation (D-07).

    Core {A,C,G,T} is ok; IUPAC ambiguity codes warn with per-character
    counts; U counts as T-equivalent with a warn carrying the U count; any
    other character fails STRICT classes (CHARSET_REJECTED_CHARS, count in
    details) and warns BPE classes (CHARSET_NONSTANDARD_CHARS — encodable).
    """
    if data is None:
        return DimensionVerdict(
            dimension=DIMENSION_CHARSET,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.NO_DATA.value,
            details={"model": entry.model},
        )
    rows = sum(split.row_count for split in data.splits)
    counts: dict[str, int] = {}
    for split in data.splits:
        for census_entry in split.charset_census:
            counts[census_entry.char] = counts.get(census_entry.char, 0) + census_entry.count
    if rows == 0 or not counts:
        return DimensionVerdict(
            dimension=DIMENSION_CHARSET,
            status=VerdictStatus.UNKNOWN,
            reason_code=CompatReasonCode.EMPTY_DATASET_DIM.value,
            details={"model": entry.model, "row_count": rows},
        )
    rejected = {
        char: count
        for char, count in counts.items()
        if char not in _CORE_CHARS and char not in _AMBIGUITY_CHARS and char != _U_CHAR
    }
    ambiguity = {char: count for char, count in counts.items() if char in _AMBIGUITY_CHARS}
    u_count = counts.get(_U_CHAR, 0)
    stance = _charset_stance(effective_tokenizer_type(entry))
    details: dict = {"model": entry.model, "stance": stance}
    if rejected:
        details["rejected_chars"] = rejected
        if stance == _CHARSET_STANCE_STRICT:
            return DimensionVerdict(
                dimension=DIMENSION_CHARSET,
                status=VerdictStatus.FAIL,
                reason_code=CompatReasonCode.CHARSET_REJECTED_CHARS.value,
                details=details,
            )
        return DimensionVerdict(
            dimension=DIMENSION_CHARSET,
            status=VerdictStatus.WARN,
            reason_code=CompatReasonCode.CHARSET_NONSTANDARD_CHARS.value,
            details=details,
        )
    if ambiguity or u_count:
        if ambiguity:
            details["ambiguity_chars"] = ambiguity
        if u_count:
            details["u_count"] = u_count
        return DimensionVerdict(
            dimension=DIMENSION_CHARSET,
            status=VerdictStatus.WARN,
            reason_code=CompatReasonCode.CHARSET_AMBIGUITY_CHARS.value,
            details=details,
        )
    return DimensionVerdict(
        dimension=DIMENSION_CHARSET, status=VerdictStatus.OK, reason_code=None, details=details
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent-routed listings (PFC-05) with lab-first ordering (PFC-07) and the
# prober shortlist seam (D-10)
# ─────────────────────────────────────────────────────────────────────────────


def _intent_listing(task_type: str | None, intent: str) -> list[RegistryEntry]:
    """Route the listing over the base<->finetuned axis, lab-first sorted.

    ``finetune`` lists pretrained/base entries — every entry when no task is
    requested, otherwise those whose task-dimension verdict is warn-or-ok
    (hard task mismatches drop out). ``predict`` lists finetuned entries of
    the requested task type that ship task heads (``num_labels``); a task
    with no such entries yields an empty listing, never an error.
    """
    if intent == INTENT_FINETUNE:
        entries = [e for e in all_entries() if e.kind == "pretrained"]
        if task_type is not None:
            entries = [
                e
                for e in entries
                if _task_verdict(e, task_type).status in (VerdictStatus.OK, VerdictStatus.WARN)
            ]
    elif intent == INTENT_PREDICT:
        if task_type is None:
            logger.debug("check_compatibility: predict intent without task_type -> no listing")
            return []
        entries = [
            e
            for e in get_by_task_type(task_type)
            if e.kind == "finetuned" and e.task.num_labels is not None
        ]
    else:  # pragma: no cover — guarded by check_compatibility's ValueError
        raise ValueError(f"Unknown intent {intent!r}.")
    return sorted(entries, key=sort_key_lab_first)


def _build_listing(entries: list[RegistryEntry]) -> list[ListingEntry]:
    """Turn registry entries into listing rows carrying task-head metadata."""
    return [
        ListingEntry(
            model=entry.model,
            entry_name=entry.name,
            is_lab=is_lab_model(entry),
            task_type=entry.task.task_type,
            num_labels=entry.task.num_labels,
            label_names=entry.task.label_names,
        )
        for entry in entries
    ]


def _attach_probes(
    listing: list[ListingEntry],
    prober: Callable[[str], AvailabilityReport],
    probe_shortlist: int | None,
) -> list[ListingEntry]:
    """Probe the first N listing entries (labs first — the listing is sorted).

    Never probes beyond ``min(probe_shortlist or 20, len(listing))`` (D-10:
    never all 236 entries). Each probe is isolated: a raising prober
    degrades that one entry to an unknown availability report (PFC-06),
    never an exception.
    """
    cap = DEFAULT_PROBE_SHORTLIST if probe_shortlist is None else max(0, probe_shortlist)
    probed = listing
    for position in range(min(cap, len(listing))):
        entry = listing[position]
        try:
            availability = prober(entry.model)
        except Exception as e:  # PFC-06: any probe failure degrades to unknown
            logger.info(
                f"check_compatibility: probe for {entry.model!r} failed "
                f"({AvailabilityReasonCode.PROBE_ERROR.value}): {e}"
            )
            availability = AvailabilityReport(
                state=AvailabilityState.UNKNOWN,
                reason=AvailabilityReasonCode.PROBE_ERROR.value,
            )
        probed[position] = entry.model_copy(update={"availability": availability})
    return probed
