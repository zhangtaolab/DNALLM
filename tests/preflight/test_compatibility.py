"""Behavior suite for check_compatibility: the three verdict dimensions (23-02).

Task 1 covers the full suitability engine over the live registry (PFC-02):

- Task dimension (D-08): finetuned entries hard-match, base models are ok for
  supervised tasks, mask/generation backbones warn HEAD_SUPPORT_UNCERTAIN.
- Length dimension (D-06): the tokenizer budget classifier (k-mer sliding
  window, char 1:1, BPE conservative ratio) with truncation never failing.
- Charset dimension (D-07): strict char-level classes fail on rejected
  characters, BPE classes never fail charset, ambiguity/U warn with counts.

Registry pins are live (Phase 22 data): budget 517 = 512 + 6 - 1 for the
6-mer entry, 1536 = 512 x 3 for the BPE entry, 512 for the char-level entry.
"""

from unittest.mock import Mock

import pytest

from dnallm.models.registry import (
    RegistryEntry,
    TaskInfo,
    all_entries,
    sort_key_lab_first,
    try_resolve,
)
from dnallm.preflight import check_compatibility, validate_data
from dnallm.preflight.report import (
    AvailabilityReport,
    AvailabilityState,
    CompatReasonCode,
    Hub,
    VerdictStatus,
)

# Live-registry fixtures (Phase 22 lane): ids, tokenizer classes, and max
# lengths are pinned facts of the packaged registry data file.
KMER6_REPO = "zhangtaolab/plant-dnabert-6mer"  # EsmTokenizer, ml 512, k 6 -> 517
CHAR_REPO = "zhangtaolab/plant-dnabert-singlebase"  # EsmTokenizer non-kmer, ml 512 -> 512
PROMOTER_REPO = "zhangtaolab/plant-dnabert-BPE-promoter"  # DebertaV2 BPE, ml 512 -> 1536
BPE_BASE_REPO = "zhangtaolab/plant-dnabert-BPE"  # pretrained mask, DebertaV2 BPE
GEN_BASE_REPO = "zhangtaolab/plant-dnagemma-singlebase"  # pretrained generation, ml 1024
NO_TK_REPO = "zhihan1996/DNA_bert_6"  # third-party tokenizer gap -> unknown budget

OK = VerdictStatus.OK
WARN = VerdictStatus.WARN
FAIL = VerdictStatus.FAIL
UNKNOWN = VerdictStatus.UNKNOWN


def _dims(verdict):
    """Map dimension name -> DimensionVerdict for one model verdict."""
    return {d.dimension: d for d in verdict.dimensions}


def _dim(verdict, name):
    """Return one dimension verdict, asserting it was computed."""
    dims = _dims(verdict)
    assert name in dims, f"dimension {name!r} missing from {dims.keys()}"
    return dims[name]


# ─────────────────────────────────────────────────────────────────────────────
# Task 1a: tokenizer budget classifier (D-06) over the live registry
# ─────────────────────────────────────────────────────────────────────────────


def test_kmer_entry_budget_uses_sliding_window_inversion() -> None:
    """6-mer entry: char budget 517 = 512 tokens + k - 1 (the k-mer trap pinned)."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[KMER6_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.details["budget_chars"] == 517
    assert length.details["kind"] == "kmer"
    assert length.details["k"] == 6
    assert length.status == OK


def test_bpe_entry_budget_uses_conservative_ratio() -> None:
    """BPE-class entry: char budget 1536 = 512 x 3 chars-per-token (over-warn)."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[PROMOTER_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.details["budget_chars"] == 1536
    assert length.details["kind"] == "bpe"
    assert length.status == OK


def test_char_level_entry_budget_equals_max_length() -> None:
    """Char-level EsmTokenizer on a non-kmer name: budget = max_length exactly."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[CHAR_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.details["budget_chars"] == 512
    assert length.details["kind"] == "char_1to1"
    assert length.status == OK


def test_unknown_tokenizer_entry_length_unknown() -> None:
    """Tokenizer-gap entries (17 today) degrade: length unknown TOKENIZER_UNKNOWN."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[NO_TK_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.status == UNKNOWN
    assert length.reason_code == CompatReasonCode.TOKENIZER_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Task 1b: truncation boundary semantics (D-06) — warn with exact counts, never fail
# ─────────────────────────────────────────────────────────────────────────────


def _seq_of_length(length: int) -> str:
    """An ACGT sequence of exactly `length` characters."""
    base = "ACGT" * (length // 4)
    return base + "A" * (length % 4)


def test_sequence_exactly_at_budget_is_not_truncated() -> None:
    """100 sequences of exactly 517 chars -> truncated_row_count 0, status ok."""
    report = validate_data([_seq_of_length(517)] * 100)
    result = check_compatibility(report, models=[KMER6_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.status == OK
    assert length.reason_code is None
    assert length.details["truncated_row_count"] == 0
    assert length.details["truncated_fraction"] == 0.0


def test_sequence_one_char_over_budget_truncates_exactly() -> None:
    """One sequence of budget+1 chars in 100 rows -> count 1, fraction exactly 0.01."""
    report = validate_data([_seq_of_length(517)] * 99 + [_seq_of_length(518)])
    result = check_compatibility(report, models=[KMER6_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.status == WARN
    assert length.reason_code == CompatReasonCode.TRUNCATION_ABOVE_THRESHOLD
    assert length.details["truncated_row_count"] == 1
    assert length.details["truncated_fraction"] == 0.01


def test_truncation_at_100_percent_of_rows_stays_warn() -> None:
    """Every row over budget still yields warn — truncation NEVER fails (D-06)."""
    report = validate_data(["A" * 2000] * 10)
    result = check_compatibility(report, models=[PROMOTER_REPO], task_type="binary")
    verdict = result.verdicts[0]
    length = _dim(verdict, "length")
    assert length.status == WARN
    assert length.details["truncated_row_count"] == 10
    assert length.details["truncated_fraction"] == 1.0
    assert verdict.overall == WARN  # never fail from truncation
    assert all(d.status != FAIL for d in verdict.dimensions)


def test_length_details_carry_linear_interpolation_quantiles() -> None:
    """Details carry p50/p95/p99/max; quantiles use linear interpolation."""
    report = validate_data(["A" * 10, "A" * 20, "A" * 30])
    result = check_compatibility(report, models=[CHAR_REPO])
    length = _dim(result.verdicts[0], "length")
    assert length.details["p50"] == pytest.approx(20.0)
    assert length.details["p95"] == pytest.approx(29.0)  # numpy linear interp 0.95
    assert length.details["p99"] == pytest.approx(29.8)
    assert length.details["max"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Task 1c: task dimension (D-08)
# ─────────────────────────────────────────────────────────────────────────────


def test_finetuned_task_type_mismatch_fails() -> None:
    """Finetuned binary entry evaluated with task_type='Regression' -> fail."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[PROMOTER_REPO], task_type="Regression"
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == FAIL
    assert task.reason_code == CompatReasonCode.TASK_TYPE_MISMATCH


def test_finetuned_task_type_match_ok_with_normalization() -> None:
    """Finetuned binary entry with ' Binary ' (case/whitespace) -> task ok."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[PROMOTER_REPO], task_type=" Binary "
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == OK
    assert task.reason_code is None


def test_mask_backbone_supervised_request_warns_head_uncertain() -> None:
    """Pretrained mask backbone + supervised request -> warn HEAD_SUPPORT_UNCERTAIN."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[BPE_BASE_REPO], task_type="binary"
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == WARN
    assert task.reason_code == CompatReasonCode.HEAD_SUPPORT_UNCERTAIN


def test_generation_backbone_supervised_request_warns_head_uncertain() -> None:
    """Pretrained generation backbone + supervised request -> warn uncertain."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[GEN_BASE_REPO], task_type="regression"
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == WARN
    assert task.reason_code == CompatReasonCode.HEAD_SUPPORT_UNCERTAIN


def test_pretrained_backbone_own_task_request_ok() -> None:
    """Pretrained mask entry with a mask request -> task ok (no conflict)."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[BPE_BASE_REPO], task_type="mask"
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == OK
    assert task.reason_code is None


def test_supervised_typed_base_entry_ok_for_supervised_task() -> None:
    """A supervised-typed base entry is ok for its supervised task (D-08).

    The live registry carries only mask/generation pretrained entries today,
    so the ok-branch is pinned through the task-verdict function directly on
    a synthetic entry — the rule must hold as the registry grows.
    """
    from dnallm.preflight.compatibility import _task_verdict

    entry = RegistryEntry(
        name="synthetic binary base",
        model="synthetic/binary-base",
        task=TaskInfo(describe="synthetic", task_type="binary"),
        kind="pretrained",
        tokenizer_type="EsmTokenizer",
        max_length=512,
    )
    verdict = _task_verdict(entry, "binary")
    assert verdict.status == OK
    assert verdict.reason_code is None


def test_task_alias_normalization_matches_registry_vocabulary() -> None:
    """'multi_label_classification' behaves as 'multilabel' (alias normalization)."""
    multilabel_repo = next(
        e.model for e in _finetuned_entries("multilabel") if e.task.num_labels is not None
    )
    result = check_compatibility(
        validate_data(["ACGT" * 10]),
        models=[multilabel_repo],
        task_type="multi_label_classification",
    )
    task = _dim(result.verdicts[0], "task")
    assert task.status == OK


def _finetuned_entries(task_type: str):
    from dnallm.models.registry import get_by_task_type

    return [e for e in get_by_task_type(task_type) if e.kind == "finetuned"]


# ─────────────────────────────────────────────────────────────────────────────
# Task 1d: charset dimension (D-07) with per-model isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_char_fails_strict_entry_and_isolates_per_model() -> None:
    """'X' fails the char-level strict entry only; the BPE sibling stays warn."""
    report = validate_data(["X" + "ACGT" * 10, "ACGT" * 10])
    result = check_compatibility(report, models=[KMER6_REPO, PROMOTER_REPO])
    strict, bpe = result.verdicts
    strict_charset = _dim(strict, "charset")
    assert strict_charset.status == FAIL
    assert strict_charset.reason_code == CompatReasonCode.CHARSET_REJECTED_CHARS
    assert strict_charset.details["rejected_chars"] == {"X": 1}
    assert strict.overall == FAIL
    # Per-model isolation (D-07): the BPE entry keeps its own verdict.
    bpe_charset = _dim(bpe, "charset")
    assert bpe_charset.status == WARN
    assert bpe_charset.reason_code == CompatReasonCode.CHARSET_NONSTANDARD_CHARS
    assert bpe.overall == WARN


def test_ambiguity_codes_warn_with_per_character_counts() -> None:
    """N/R/Y ambiguity codes warn with exact per-character counts in details."""
    report = validate_data(["NRY" + "ACGT" * 10])
    result = check_compatibility(report, models=[CHAR_REPO])
    charset = _dim(result.verdicts[0], "charset")
    assert charset.status == WARN
    assert charset.reason_code == CompatReasonCode.CHARSET_AMBIGUITY_CHARS
    assert charset.details["ambiguity_chars"] == {"N": 1, "R": 1, "Y": 1}


def test_u_chars_are_t_equivalent_and_never_fail() -> None:
    """U counts as T-equivalent: warn carrying the U count, even on strict classes."""
    report = validate_data(["U" * 5 + "ACGT" * 10])
    result = check_compatibility(report, models=[CHAR_REPO])
    charset = _dim(result.verdicts[0], "charset")
    assert charset.status == WARN
    assert charset.reason_code == CompatReasonCode.CHARSET_AMBIGUITY_CHARS
    assert charset.details["u_count"] == 5
    assert result.verdicts[0].overall == WARN  # never fail


def test_unknown_tokenizer_charset_never_fails_on_rejected_chars() -> None:
    """Tokenizer-gap entries cannot prove hard charset incompatibility -> warn."""
    report = validate_data(["X" + "ACGT" * 10])
    result = check_compatibility(report, models=[NO_TK_REPO])
    charset = _dim(result.verdicts[0], "charset")
    assert charset.status == WARN
    assert charset.reason_code == CompatReasonCode.CHARSET_NONSTANDARD_CHARS


def test_pure_acgt_charset_ok() -> None:
    """Data inside {A,C,G,T} yields an ok charset dimension with no reason code."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[PROMOTER_REPO])
    charset = _dim(result.verdicts[0], "charset")
    assert charset.status == OK
    assert charset.reason_code is None


# ─────────────────────────────────────────────────────────────────────────────
# Task 1e: degradation and overall aggregation
# ─────────────────────────────────────────────────────────────────────────────


def test_no_data_degrades_length_and_charset_but_not_task() -> None:
    """data=None -> length/charset unknown NO_DATA; task still computed."""
    result = check_compatibility(None, models=[PROMOTER_REPO], task_type="binary")
    verdict = result.verdicts[0]
    assert _dim(verdict, "length").reason_code == CompatReasonCode.NO_DATA
    assert _dim(verdict, "charset").reason_code == CompatReasonCode.NO_DATA
    assert _dim(verdict, "task").status == OK
    assert verdict.overall == UNKNOWN


def test_empty_dataset_degrades_length_and_charset_dimensions() -> None:
    """Zero-row data -> length/charset unknown EMPTY_DATASET_DIM, task computed."""
    result = check_compatibility(validate_data([]), models=[PROMOTER_REPO], task_type="binary")
    verdict = result.verdicts[0]
    assert _dim(verdict, "length").reason_code == CompatReasonCode.EMPTY_DATASET_DIM
    assert _dim(verdict, "charset").reason_code == CompatReasonCode.EMPTY_DATASET_DIM
    assert _dim(verdict, "task").status == OK


def test_overall_is_worst_of_computed_dimensions() -> None:
    """A hard task mismatch dominates warn/unknown dimensions: overall fail."""
    report = validate_data(["N" + "ACGT" * 10])
    result = check_compatibility(report, models=[PROMOTER_REPO], task_type="regression")
    verdict = result.verdicts[0]
    assert _dim(verdict, "task").status == FAIL
    assert _dim(verdict, "charset").status == WARN
    assert verdict.overall == FAIL


def test_dimension_vocabulary_is_task_length_charset() -> None:
    """A resolved entry carries exactly the task/length/charset dimensions."""
    result = check_compatibility(validate_data(["ACGT" * 10]), models=[PROMOTER_REPO])
    dims = _dims(result.verdicts[0])
    assert set(dims) == {"task", "length", "charset"}
    # Registry metadata still rides on the verdict (23-01 contract).
    entry = try_resolve(PROMOTER_REPO)
    assert result.verdicts[0].entry_name == entry.name
    assert sort_key_lab_first(entry) == (0, entry.name)  # lab partition pins


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: intent-routed listings (PFC-05), lab-first ordering (PFC-07),
# and the prober shortlist seam (D-10)
# ─────────────────────────────────────────────────────────────────────────────


def test_intent_finetune_lists_pretrained_entries_only() -> None:
    """intent='finetune' lists pretrained/base entries (warn-or-ok task compat)."""
    result = check_compatibility(task_type="binary", intent="finetune")
    assert result.listing
    for listing_entry in result.listing:
        assert try_resolve(listing_entry.model).kind == "pretrained"
    # With a supervised request every mask/generation base warns (uncertain
    # head support) and warn-or-ok keeps all of them.
    pretrained_models = {e.model for e in all_entries() if e.kind == "pretrained"}
    assert {e.model for e in result.listing} == pretrained_models


def test_intent_finetune_without_task_type_lists_all_pretrained() -> None:
    """intent='finetune' without a task filter lists every pretrained entry."""
    result = check_compatibility(intent="finetune")
    pretrained_models = {e.model for e in all_entries() if e.kind == "pretrained"}
    assert {e.model for e in result.listing} == pretrained_models


def test_intent_predict_lists_finetuned_entries_with_heads() -> None:
    """intent='predict' lists finetuned entries shipping task heads + labels."""
    result = check_compatibility(task_type="binary", intent="predict")
    assert result.listing
    for listing_entry in result.listing:
        assert listing_entry.task_type == "binary"
        assert listing_entry.num_labels is not None
        assert listing_entry.label_names is not None
        assert try_resolve(listing_entry.model).kind == "finetuned"


def test_listing_is_sorted_lab_first() -> None:
    """zhangtaolab/* entries rank before third-party, then by name (PFC-07)."""
    result = check_compatibility(task_type="binary", intent="predict")
    assert any(not e.is_lab for e in result.listing)  # mixed listing (6 third-party)
    keys = [sort_key_lab_first(try_resolve(e.model)) for e in result.listing]
    assert keys == sorted(keys)  # equals registry.sort_key_lab_first output
    assert result.listing[0].is_lab is True
    # Partition boundary: no third-party entry appears before any lab entry.
    first_third_party = next(i for i, e in enumerate(result.listing) if not e.is_lab)
    assert all(e.is_lab for e in result.listing[:first_third_party])


def test_listing_task_type_normalization_case_and_whitespace() -> None:
    """' Binary ' routes the predict listing exactly like 'binary'."""
    loose = check_compatibility(task_type=" Binary ", intent="predict")
    tight = check_compatibility(task_type="binary", intent="predict")
    assert [e.model for e in loose.listing] == [e.model for e in tight.listing]


def test_intent_zero_matches_returns_empty_listing_not_error() -> None:
    """intent='predict' with a task that has no finetuned entries -> empty listing."""
    # Census-checked: 'generation' exists only as pretrained task type today.
    result = check_compatibility(task_type="generation", intent="predict")
    assert result.listing == []
    assert result.verdicts == []


def test_invalid_intent_raises_matchable_value_error() -> None:
    """An intent outside {finetune, predict} raises a descriptive ValueError."""
    with pytest.raises(ValueError, match=r"intent"):
        check_compatibility(task_type="binary", intent="explore")


def test_intent_value_is_normalized_before_validation() -> None:
    """' Predict ' behaves as 'predict' (case/whitespace normalization)."""
    result = check_compatibility(task_type="binary", intent=" Predict ")
    assert result.listing  # same routing as predict


def _mock_prober():
    """A fake prober returning one shared remote-confirmed report."""
    return Mock(
        return_value=AvailabilityReport(
            state=AvailabilityState.REMOTE_CONFIRMED,
            size_bytes=1,
            revision="abc",
            source_hub=Hub.MODELSCOPE,
        )
    )


def test_prober_shortlist_probes_first_n_in_lab_order() -> None:
    """prober + probe_shortlist=3 -> exactly 3 calls on the first 3 entries."""
    prober = _mock_prober()
    result = check_compatibility(
        task_type="binary", intent="predict", prober=prober, probe_shortlist=3
    )
    assert prober.call_count == 3
    assert [call.args[0] for call in prober.call_args_list] == [
        e.model for e in result.listing[:3]
    ]
    assert all(result.listing[i].availability is prober.return_value for i in range(3))
    # Beyond the shortlist: untouched, availability stays None.
    assert all(e.availability is None for e in result.listing[3:])


def test_without_prober_no_calls_and_availability_none() -> None:
    """Default is network-free: no prober -> availability None on every entry."""
    result = check_compatibility(task_type="binary", intent="predict")
    assert result.listing
    assert all(e.availability is None for e in result.listing)


def test_probe_shortlist_larger_than_listing_caps_at_listing_length() -> None:
    """probe_shortlist=10 on a 3-entry listing probes only 3 times."""
    prober = _mock_prober()
    result = check_compatibility(
        task_type="multilabel", intent="predict", prober=prober, probe_shortlist=10
    )
    assert len(result.listing) == 3  # census-checked: 3 multilabel finetuned entries
    assert prober.call_count == len(result.listing)


def test_prober_without_n_defaults_to_shortlist_20() -> None:
    """prober given without probe_shortlist probes the default cap of 20."""
    prober = Mock(return_value=AvailabilityReport(state=AvailabilityState.UNKNOWN))
    result = check_compatibility(task_type="binary", intent="predict", prober=prober)
    assert len(result.listing) > 20  # 115 binary entries — the cap is exercised
    assert prober.call_count == 20


def test_models_argument_suppresses_listing() -> None:
    """Explicit models still return verdicts only — the listing stays empty."""
    result = check_compatibility(
        validate_data(["ACGT" * 10]), models=[PROMOTER_REPO], task_type="binary"
    )
    assert result.listing == []
    assert len(result.verdicts) == 1
