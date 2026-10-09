"""Behavior tests for the zero-shot VEP scoring core (vep module).

Fast-lane strategy: the same-slot evaluability rule runs on the real
character-level tokenizer from tests/conftest.py so alignment outcomes are
real tokenizations (single-character substitutions are same-slot by
construction); the multi-slot skip path — unreachable with any char-level
vocabulary — runs on a minimal stub tokenizer defined below. The scoring
kernels run on the real tiny per-position torch module so every score is a
real model output. The evaluate_vcf driver runs end-to-end on the committed
synthetic VCF fixture (tests/inference/data/) with real kernels for
structural semantics and with mocked kernels where deterministic score
magnitudes are asserted. No network, no model downloads, no skips.
"""

from types import SimpleNamespace
from typing import ClassVar
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dnallm.inference.vep import (
    VariantAlignment,
    align_variant,
    clm_log_likelihood,
    evaluate_vcf,
    get_model_device,
    mlm_slot_log_prob,
    score_variant,
)

DATA_DIR = Path(__file__).parent / "data"
FIXTURE_VCF = DATA_DIR / "synthetic_variants.vcf"
# The sidecar is FASTA by content but .txt by name: .gitignore excludes
# genome extensions (*.fa/*.fna/*.fasta) wholesale, and this lane does not
# own .gitignore.
FIXTURE_FA = DATA_DIR / "synthetic_reference.txt"

# Token ids of the shared SimpleDNATokenizer vocabulary (tests/conftest.py).
_ID_A, _ID_C, _ID_G, _ID_T = 5, 6, 7, 8

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chrT,length=160>\n"
    '##INFO=<ID=CLNSIG,Number=.,Type=String,Description="cs">\n'
    '##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="rs">\n'
    '##INFO=<ID=CLNVC,Number=1,Type=String,Description="vt">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def _write_vcf(path, rows):
    """Write a minimal ClinVar-style VCF; rows are (chrom, pos, ref, alt,
    clnsig, clnrevstat, clnvc) tuples."""
    lines = [_VCF_HEADER]
    for i, (chrom, pos, ref, alt, sig, rev, vc) in enumerate(rows, 1):
        lines.append(
            f"{chrom}\t{pos}\trsX{i}\t{ref}\t{alt}\t.\t.\t"
            f"CLNSIG={sig};CLNREVSTAT={rev};CLNVC={vc}\n"
        )
    Path(path).write_text("".join(lines), encoding="utf-8")
    return str(path)


def _load_fixture_reference():
    """Parse the committed fixture FASTA into {name: sequence}."""
    sequences, name, chunks = {}, None, []
    for line in FIXTURE_FA.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                sequences[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def _nth_position(sequence, base, n, lo=1, hi=None):
    """1-based position of the n-th `base` (case-insensitive) in [lo, hi]."""
    hi = hi or len(sequence)
    hits = [i + 1 for i, ch in enumerate(sequence) if ch.upper() == base and lo <= i + 1 <= hi]
    return hits[n]


class _TensorEncoding(dict):
    """Dict encoding with a no-op ``.to`` — the minimum kernel contract."""

    def to(self, device):
        return self


class _CaseSensitiveTokenizer:
    """Char-level tokenizer WITHOUT case folding: lowercase bases map to
    UNK, reproducing the empirically verified real-tokenizer behavior that
    makes un-uppercased soft-masked windows silently skip as 'no change'."""

    base_ids: ClassVar[dict[str, int]] = {"A": 5, "C": 6, "G": 7, "T": 8}
    mask_token_id = 4
    vocab_size = 9

    def __call__(self, seq, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = [self.base_ids.get(ch, 1) for ch in seq]
        return _TensorEncoding({"input_ids": torch.tensor([ids], dtype=torch.long)})


def _clm(model, tokenizer, sequence):
    """Score one sequence with the causal kernel."""
    return clm_log_likelihood(model, tokenizer, sequence)


def _mlm(model, tokenizer, sequence, slot_index, token_id):
    """Score one masked slot with the MLM kernel."""
    return mlm_slot_log_prob(model, tokenizer, sequence, slot_index, token_id)


class _ChecksumTokenizer:
    """Stub tokenizer that appends a checksum token after the char ids.

    A single base substitution changes BOTH the substituted base slot and
    the trailing checksum slot, so ref/alt tokenizations differ at exactly
    two indices — the multi-slot outcome a char-level vocabulary can never
    produce.
    """

    base_ids: ClassVar[dict[str, int]] = {"A": 5, "C": 6, "G": 7, "T": 8}

    def __call__(self, seq, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = [self.base_ids.get(ch, 1) for ch in seq]
        ids.append(sum(ids) % 9)
        return {"input_ids": [ids]}


class TestAlignVariant:
    """The same-slot evaluability rule (reviewer R1-3e-1 protocol answer)."""

    def test_snp_mid_sequence_is_evaluatable(self, simple_dna_tokenizer):
        """A passing SNP reports the differing slot plus ref/alt token ids."""
        sequence = "ACGTTGCA"
        result = align_variant(sequence, 3, "T", "A", simple_dna_tokenizer)

        assert result.evaluatable is True
        assert result.slot_index == 3
        assert result.ref_token_id == simple_dna_tokenizer.convert_tokens_to_ids("T")
        assert result.alt_token_id == simple_dna_tokenizer.convert_tokens_to_ids("A")
        assert result.skip_reason is None

    def test_passing_variant_differs_at_exactly_one_token_slot(self, simple_dna_tokenizer):
        """PITFALLS slot-misalignment guard: the FULL id lists are compared
        and must differ at exactly one index — the assertion IS the R1-3e-1
        protocol answer, not a spot-check of the reported slot alone."""
        sequence = "ACGTTGCA"
        alt_sequence = sequence[:3] + "A" + sequence[4:]
        assert alt_sequence == "ACGATGCA"
        result = align_variant(sequence, 3, "T", "A", simple_dna_tokenizer)

        ref_ids = simple_dna_tokenizer(sequence, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ][0].tolist()
        alt_ids = simple_dna_tokenizer(alt_sequence, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ][0].tolist()

        diff_indices = [
            i
            for i, (ref_id, alt_id) in enumerate(zip(ref_ids, alt_ids, strict=True))
            if ref_id != alt_id
        ]
        assert diff_indices == [result.slot_index]
        assert len(diff_indices) == 1

    def test_length_changing_allele_skips_with_reason(self, simple_dna_tokenizer):
        """An insertion/deletion-length allele returns a skip record, never raises."""
        result = align_variant("ACGTTGCA", 3, "T", "TG", simple_dna_tokenizer)

        assert result.evaluatable is False
        assert result.skip_reason == "length-changing allele"
        assert result.slot_index is None
        assert result.ref_token_id is None
        assert result.alt_token_id is None

    def test_identical_ref_alt_is_no_change_skip(self, simple_dna_tokenizer):
        """ref == alt tokenizes to identical ids and skips as no change."""
        result = align_variant("ACGTTGCA", 2, "G", "G", simple_dna_tokenizer)

        assert result.evaluatable is False
        assert result.skip_reason == "no change"
        assert result.slot_index is None
        assert result.ref_token_id is None
        assert result.alt_token_id is None

    def test_ref_mismatch_raises_value_error(self, simple_dna_tokenizer):
        """A ref allele that contradicts the sequence is an input-contract
        violation — a ValueError, distinct from the skip path."""
        with pytest.raises(ValueError, match="does not match"):
            align_variant("ACGTTGCA", 3, "A", "T", simple_dna_tokenizer)

    def test_multi_slot_token_difference_skips(self):
        """A context-sensitive tokenizer whose ref/alt ids differ at two
        slots is rejected as a multi-slot skip (PITFALLS countermeasure)."""
        tokenizer = _ChecksumTokenizer()
        result = align_variant("AAA", 1, "A", "C", tokenizer)

        assert result.evaluatable is False
        assert result.skip_reason == "multi-slot token difference"
        assert result.slot_index is None

    def test_variant_at_sequence_start(self, simple_dna_tokenizer):
        """A boundary SNP at pos 0 aligns at token slot 0."""
        result = align_variant("ACGTTGCA", 0, "A", "T", simple_dna_tokenizer)

        assert result.evaluatable is True
        assert result.slot_index == 0
        assert result.ref_token_id == simple_dna_tokenizer.convert_tokens_to_ids("A")

    def test_variant_at_sequence_end(self, simple_dna_tokenizer):
        """A boundary SNP at the last base aligns at the last token slot."""
        sequence = "ACGTTGCA"
        result = align_variant(sequence, len(sequence) - 1, "A", "C", simple_dna_tokenizer)

        assert result.evaluatable is True
        assert result.slot_index == len(sequence) - 1
        assert result.alt_token_id == simple_dna_tokenizer.convert_tokens_to_ids("C")

    def test_alignment_is_frozen_dataclass(self, simple_dna_tokenizer):
        """The result type is immutable — alignments are shared facts."""
        result = align_variant("ACGTTGCA", 3, "T", "A", simple_dna_tokenizer)

        assert isinstance(result, VariantAlignment)
        # FrozenInstanceError subclasses AttributeError.
        with pytest.raises(AttributeError):
            result.evaluatable = False


class TestClmLogLikelihood:
    """clm_log_likelihood on the real per-position tiny model."""

    def test_returns_finite_nonpositive_float(self, tiny_model_factory, simple_dna_tokenizer):
        """A valid sequence scores a finite log-likelihood <= 0."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        score = _clm(model, simple_dna_tokenizer, "ACGTA")

        assert isinstance(score, float)
        assert math.isfinite(score)
        assert score <= 0.0

    def test_deterministic_repeated_calls(self, tiny_model_factory, simple_dna_tokenizer):
        """Two calls on identical input give bit-identical scores."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        first = _clm(model, simple_dna_tokenizer, "ACGTTGCA")
        second = _clm(model, simple_dna_tokenizer, "ACGTTGCA")

        assert first == second

    def test_different_sequence_scores_differently(self, tiny_model_factory, simple_dna_tokenizer):
        """A changed base changes the causal log-likelihood (the signal the
        delta-log-likelihood paradigm depends on)."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        ref_score = _clm(model, simple_dna_tokenizer, "ACGTTGCA")
        alt_score = _clm(model, simple_dna_tokenizer, "ACGATGCA")

        assert ref_score != alt_score


class TestMlmSlotLogProb:
    """mlm_slot_log_prob on the real per-position tiny model."""

    def test_returns_finite_nonpositive_float(self, tiny_model_factory, simple_dna_tokenizer):
        """A valid slot/token pair scores a finite log-prob <= 0."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        score = _mlm(model, simple_dna_tokenizer, "ACGTA", 2, 7)

        assert isinstance(score, float)
        assert math.isfinite(score)
        assert score <= 0.0

    def test_deterministic_repeated_calls(self, tiny_model_factory, simple_dna_tokenizer):
        """Two calls on identical input give bit-identical scores."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        first = _mlm(model, simple_dna_tokenizer, "ACGTA", 2, 7)
        second = _mlm(model, simple_dna_tokenizer, "ACGTA", 2, 7)

        assert first == second

    def test_competing_token_ids_discriminate(self, tiny_model_factory, simple_dna_tokenizer):
        """Ref and alt token ids at the same slot get different log-probs —
        the ref/alt discrimination the log-odds paradigm depends on."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        # G (id 7) occupies slot 2 of "ACGTA"; T (id 8) is the alt base.
        ref_logp = _mlm(model, simple_dna_tokenizer, "ACGTA", 2, 7)
        alt_logp = _mlm(model, simple_dna_tokenizer, "ACGTA", 2, 8)

        assert ref_logp != alt_logp

    def test_masking_touches_only_target_slot(self, tiny_model_factory, simple_dna_tokenizer):
        """The masked input the model receives equals the original ids with
        ONLY the target slot replaced by the mask id (clone check)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        captured = []
        original_forward = model.forward

        def spy_forward(*args, **kwargs):
            captured.append(kwargs["input_ids"].clone())
            return original_forward(*args, **kwargs)

        model.forward = spy_forward

        sequence = "ACGTA"
        original_ids = simple_dna_tokenizer(sequence, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ]
        _mlm(model, simple_dna_tokenizer, sequence, 2, 7)

        assert len(captured) == 1
        masked = captured[0]
        assert masked[0, 2].item() == simple_dna_tokenizer.mask_token_id
        assert torch.equal(masked[0, :2], original_ids[0, :2])
        assert torch.equal(masked[0, 3:], original_ids[0, 3:])


class TestGetModelDevice:
    """get_model_device resolution order (mirrors mutagenesis.py:241-255)."""

    def test_device_attribute_wins(self):
        """A model-like object exposing .device resolves to that device."""
        model = SimpleNamespace(device=torch.device("meta"))

        assert get_model_device(model) == torch.device("meta")

    def test_parameters_fallback(self, tiny_model_factory):
        """A plain nn.Module (no .device attr) resolves via its parameters."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        assert get_model_device(model) == torch.device("cpu")

    def test_plain_object_falls_back_to_cpu(self):
        """An object with neither .device nor .parameters assumes CPU."""
        assert get_model_device(object()) == torch.device("cpu")


class TestEvaluateVcf:
    """The VCF-level driver on the committed fixture and purpose-built VCFs.

    Real kernels prove structure (counts, skips, conventions); mocked
    kernels prove deterministic score magnitudes (AUROC shape, coordinate
    plumbing) — the D-09 tier-1 split.
    """

    def test_fixture_end_to_end_real_mlm_kernels(self, tiny_model_factory, simple_dna_tokenizer):
        """The committed fixture flows through evaluate_vcf into per-variant
        deltas + skip counts + registry metrics with real model scores."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            FIXTURE_VCF,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=12,
        )

        # 8 single-SNV rows + 4 ALTs of the 4-allelic row + 1 mis-annotated
        # indel row (CLNVC says SNV) = 13 considered per-allele records.
        assert len(result.records) == 13
        assert result.evaluated == 11
        assert result.skipped == 2
        assert result.skip_fraction == pytest.approx(2 / 13)
        assert result.skip_counts == {
            "length-changing allele": 2,
            "multi-slot token difference": 0,
            "no change": 0,
        }
        # Convention exclusions (D-17): honest indel, VUS + conflicting
        # labels, 0-star review status — each counted in its own bucket.
        assert result.convention["exclusion_counts"] == {
            "non_snv_clnvc": 1,
            "unlabeled_clnsig": 2,
            "below_star_floor": 1,
        }
        assert result.metrics is not None
        assert set(result.metrics) == {"AUROC", "AUPRC"}
        for value in result.metrics.values():
            assert 0.0 <= value <= 1.0
        assert result.convention["star_floor"] == 1
        assert result.convention["rows_read"] == 14

    def test_perfect_separation_mocked_mlm_gives_unit_auroc(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Controlled kernel outputs with clean label separation produce
        AUROC == AUPRC == 1.0 — the metric plumbing is registry-driven."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        seq = _load_fixture_reference()["chrT"]
        # Positives G->C; negatives C->A / T->G / G->T (never alt=C).
        rows = [
            (
                "chrT",
                _nth_position(seq, "G", 0, 97),
                "G",
                "C",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "G", 1, 97),
                "G",
                "C",
                "Likely_pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "G", 2, 97),
                "G",
                "C",
                "Pathogenic",
                "reviewed_by_expert_panel",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "C", 0, 97),
                "C",
                "A",
                "Benign",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "T", 0, 97),
                "T",
                "G",
                "Benign",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "G", 3, 97),
                "G",
                "T",
                "Likely_benign",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
        ]
        vcf = _write_vcf(tmp_path / "sep.vcf", rows)
        # alt(C) logp -0.1, everything else -2.0: every positive delta is
        # +1.9, every negative delta <= 0.0 — perfect separation.
        fake = {_ID_A: -2.0, _ID_C: -0.1, _ID_G: -2.0, _ID_T: -2.0}

        def fake_mlm(model, tokenizer, sequence, slot_index, token_id):
            return fake[token_id]

        with patch("dnallm.inference.vep.mlm_slot_log_prob", side_effect=fake_mlm):
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                vcf,
                FIXTURE_FA,
                paradigm="mlm",
                context_window=6,
            )

        assert result.evaluated == 6
        assert result.metrics == {"AUROC": 1.0, "AUPRC": 1.0}

    def test_clm_delta_comes_from_the_clm_kernel_on_one_window(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """CLM scoring substitutes the alt INSIDE the one reference window
        (identical left context) and reports alt - ref log-likelihood."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        model.config.is_decoder = True  # declare causal-ness for the D-11 guard
        seq = _load_fixture_reference()["chrT"]
        pos1 = _nth_position(seq, "G", 0, 97)
        rows = [
            (
                "chrT",
                pos1,
                "G",
                "C",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "clm.vcf", rows)
        context = 12
        pos0 = pos1 - 1
        ref_window = seq[pos0 - context : pos0 + 1 + context].upper()
        local = context  # interior position: local index equals the context
        alt_window = ref_window[:local] + "C" + ref_window[local + 1 :]
        calls = []

        def fake_clm(model, tokenizer, sequence):
            calls.append(sequence)
            return -3.0 if sequence == ref_window else -5.0

        with patch("dnallm.inference.vep.clm_log_likelihood", side_effect=fake_clm):
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                vcf,
                FIXTURE_FA,
                paradigm="clm",
                context_window=context,
            )

        assert sorted(calls) == sorted([ref_window, alt_window])
        assert result.evaluated == 1
        assert result.records[0].delta == pytest.approx(-2.0)

    def test_pos1_converts_to_zero_based_first_base(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A POS=1 record scores the FIRST base of the window (VCF 1-based
        -> align_variant 0-based), proven by the kernel's received args."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        rows = [
            (
                "chrT",
                1,
                "A",
                "G",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "pos1.vcf", rows)
        calls = []

        def fake_mlm(model, tokenizer, sequence, slot_index, token_id):
            calls.append((sequence, slot_index, token_id))
            return -1.0

        with patch("dnallm.inference.vep.mlm_slot_log_prob", side_effect=fake_mlm):
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                vcf,
                FIXTURE_FA,
                paradigm="mlm",
                context_window=6,
            )

        seq = _load_fixture_reference()["chrT"]
        assert calls[0] == (seq[:7], 0, _ID_G)  # alt call: first base, slot 0
        assert calls[1][2] == _ID_A  # ref call: reference token id
        assert result.records[0].delta == pytest.approx(0.0)  # -1.0 - (-1.0)

    def test_lowercase_context_scored_not_skipped(self, tiny_model_factory):
        """The soft-masked record (lowercase reference context) is SCORED:
        the driver uppercases every window, so a case-sensitive tokenizer
        (lowercase -> <unk>) never sees the trap (Pitfall 5)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        tokenizer = _CaseSensitiveTokenizer()

        result = evaluate_vcf(
            model,
            tokenizer,
            FIXTURE_VCF,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=12,
        )

        # rsT4 of the committed fixture: pos 68, REF=G ALT=C over lowercase
        # reference context — must carry a real delta, not a skip reason.
        record = next(r for r in result.records if r.pos == 68)
        assert record.delta is not None
        assert record.skip_reason is None
        assert result.skip_counts["no change"] == 0

    def test_indel_skip_accounting_and_channel_separation(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """Length-changing alleles land in the alignment skip channel; the
        honestly-annotated indel is a CLNVC convention exclusion instead —
        the two channels never mix."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            FIXTURE_VCF,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=12,
        )

        skipped = [r for r in result.records if r.skip_reason == "length-changing allele"]
        # The mis-annotated row (REF=AT ALT=A under an SNV CLNVC) and the
        # insertion ALT of the multi-allelic row.
        assert {(r.ref, r.alt) for r in skipped} == {("AT", "A"), ("A", "AT")}
        # The honest indel row (REF=ATG, CLNVC=Deletion) never reaches
        # alignment: it is counted in the convention block only.
        assert all(r.ref != "ATG" for r in result.records)
        assert result.convention["exclusion_counts"]["non_snv_clnvc"] == 1

    def test_multiallelic_row_expands_per_alt(self, tiny_model_factory, simple_dna_tokenizer):
        """The 4-allelic row yields one record per ALT: three scored SNV
        alts plus the insertion alt skipped as length-changing."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            FIXTURE_VCF,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=12,
        )

        by_pos: dict[int, list] = {}
        for record in result.records:
            by_pos.setdefault(record.pos, []).append(record)
        multi = next(records for pos, records in by_pos.items() if len(records) == 4)
        assert {r.alt for r in multi} == {"C", "G", "T", "AT"}
        assert sum(1 for r in multi if r.delta is not None) == 3

    def test_all_rows_excluded_returns_none_metrics_and_full_skip_fraction(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A VCF with zero scorable variants reports evaluated=0, the full
        skip-count table, skip_fraction 1.0 and metrics None — no crash, no
        fabricated AUROC (the empty edge)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        rows = [
            (
                "chrT",
                6,
                "G",
                "C",
                "Uncertain_significance",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                10,
                "G",
                "A",
                "Conflicting_classifications_of_pathogenicity",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
        ]
        vcf = _write_vcf(tmp_path / "vus.vcf", rows)

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            vcf,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=6,
        )

        assert result.evaluated == 0
        assert result.metrics is None
        assert result.skip_fraction == 1.0
        assert result.records == []
        assert result.skip_counts == {
            "length-changing allele": 0,
            "multi-slot token difference": 0,
            "no change": 0,
        }

    def test_read_vcf_called_with_deliberately_sized_alt_number(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """allel.read_vcf receives alt_number >= 4 (the default 3 silently
        truncates 4+-allelic rows) and the ClinVar INFO field list."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        fake_callset = {
            "variants/CHROM": np.array(["chrT"], dtype=object),
            "variants/POS": np.array([1], dtype=np.int32),
            "variants/ID": np.array(["rsX"], dtype=object),
            "variants/REF": np.array(["A"], dtype=object),
            "variants/ALT": np.array([["G", "", "", ""]], dtype=object),
            "variants/CLNSIG": np.array(["Pathogenic"], dtype=object),
            "variants/CLNREVSTAT": np.array(["criteria_provided"], dtype=object),
            "variants/CLNVC": np.array(["single_nucleotide_variant"], dtype=object),
        }

        with patch("allel.read_vcf", return_value=fake_callset) as read_vcf:
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                "unused.vcf",
                {"chrT": "ACGTTGCA"},
                paradigm="mlm",
                context_window=6,
            )

        kwargs = read_vcf.call_args.kwargs
        assert kwargs["alt_number"] >= 4
        for field in ("variants/CLNSIG", "variants/CLNREVSTAT", "variants/CLNVC"):
            assert field in kwargs["fields"]
        assert result.evaluated == 1

    def test_chrom_prefix_fallback_resolves_clinvar_style_names(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A CHROM value without the 'chr' prefix resolves against a
        'chr'-prefixed reference (ClinVar '22' vs UCSC 'chr22')."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        rows = [
            (
                "T",
                6,
                "G",
                "C",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "chrom.vcf", rows)

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            vcf,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=6,
        )

        assert result.evaluated == 1
        assert result.records[0].chrom == "T"

    def test_missing_chromosome_raises_value_error(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A VCF chromosome absent from the reference is surfaced as a
        dnallm ValueError naming the available chromosomes."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        rows = [
            (
                "chrX",
                6,
                "A",
                "G",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "chrX.vcf", rows)

        with pytest.raises(ValueError, match="not found in the reference"):
            evaluate_vcf(
                model,
                simple_dna_tokenizer,
                vcf,
                FIXTURE_FA,
                paradigm="mlm",
                context_window=6,
            )

    def test_output_dir_writes_deterministic_result_json(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """output_dir yields a deterministic vep_result.json with the full
        result shape — no path is derived from VCF record fields."""
        import json as jsonlib

        model = tiny_model_factory(n_classes=9, pooled=False)
        out_dir = tmp_path / "vep_out"

        result = evaluate_vcf(
            model,
            simple_dna_tokenizer,
            FIXTURE_VCF,
            FIXTURE_FA,
            paradigm="mlm",
            context_window=12,
            output_dir=out_dir,
        )

        out_path = out_dir / "vep_result.json"
        assert out_path.is_file()
        payload = jsonlib.loads(out_path.read_text(encoding="utf-8"))
        assert payload.keys() == result.to_dict().keys()
        assert payload["evaluated"] == result.evaluated
        assert payload["convention"]["star_floor"] == 1

    def test_missing_scikit_allel_raises_helpful_value_error(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """Without the dependency the driver raises dnallm's own ValueError
        naming the extra — never a bare ImportError traceback."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        with patch.dict(sys.modules, {"allel": None}):
            with pytest.raises(ValueError, match="scikit-allel"):
                evaluate_vcf(
                    model,
                    simple_dna_tokenizer,
                    FIXTURE_VCF,
                    FIXTURE_FA,
                    paradigm="mlm",
                )


class TestScoreVariant:
    """score_variant: both paradigms, the D-11 paradigm↔architecture guard,
    and the skip-as-data passthrough."""

    def test_clm_on_bidirectional_model_raises_value_error(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """The guard fires BEFORE scoring: CLM paradigm on a config without
        decoder/architecture evidence raises a matchable dnallm ValueError
        (never a silent skip that would masquerade as a near-random
        finding)."""
        model = tiny_model_factory(n_classes=9, pooled=False)  # no decoder markers

        with pytest.raises(ValueError, match="Paradigm 'clm' requires a causal"):
            score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "A", paradigm="clm")

    def test_mlm_without_mask_token_raises_value_error(self, tiny_model_factory):
        """MLM paradigm with a tokenizer lacking mask_token_id raises a
        matchable dnallm ValueError."""

        class _MasklessTokenizer(_CaseSensitiveTokenizer):
            mask_token_id = None

        model = tiny_model_factory(n_classes=9, pooled=False)

        with pytest.raises(ValueError, match="mask_token_id"):
            score_variant(model, _MasklessTokenizer(), "ACGTTGCA", 3, "T", "A", paradigm="mlm")

    def test_unknown_paradigm_raises_value_error(self, tiny_model_factory, simple_dna_tokenizer):
        """An unrecognized paradigm is rejected before anything runs."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        with pytest.raises(ValueError, match="Unknown paradigm"):
            score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "A", paradigm="plm")

    def test_mlm_happy_path_matches_manual_kernel_computation(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """MLM score equals the manual slot log-prob difference of the
        landed kernel (log-odds of alt against ref)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        sequence, pos, ref, alt = "ACGTTGCA", 3, "T", "A"

        result = score_variant(model, simple_dna_tokenizer, sequence, pos, ref, alt, paradigm="mlm")

        alignment = align_variant(sequence, pos, ref, alt, simple_dna_tokenizer)
        expected = mlm_slot_log_prob(
            model, simple_dna_tokenizer, sequence, alignment.slot_index, alignment.alt_token_id
        ) - mlm_slot_log_prob(
            model, simple_dna_tokenizer, sequence, alignment.slot_index, alignment.ref_token_id
        )
        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    def test_clm_happy_path_matches_manual_kernel_computation(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """CLM score equals the manual full-sequence delta log-likelihood
        of the landed kernel on the substituted window."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        model.config.is_decoder = True  # declare causal-ness for the guard
        sequence, pos, ref, alt = "ACGTTGCA", 3, "T", "A"

        result = score_variant(model, simple_dna_tokenizer, sequence, pos, ref, alt, paradigm="clm")

        alt_sequence = sequence[:pos] + alt + sequence[pos + len(ref) :]
        expected = clm_log_likelihood(
            model, simple_dna_tokenizer, alt_sequence
        ) - clm_log_likelihood(model, simple_dna_tokenizer, sequence)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    def test_length_changing_allele_returns_skip_record_as_data(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """A length-changing allele returns the VariantAlignment skip
        record (reason preserved) — the module's skip-as-data channel."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        result = score_variant(
            model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "TG", paradigm="mlm"
        )

        assert isinstance(result, VariantAlignment)
        assert result.evaluatable is False
        assert result.skip_reason == "length-changing allele"

    def test_no_change_returns_skip_record_as_data(self, tiny_model_factory, simple_dna_tokenizer):
        """ref == alt tokenizes identically and returns a 'no change' skip
        record rather than a fabricated zero delta."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        result = score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 2, "G", "G", paradigm="mlm")

        assert isinstance(result, VariantAlignment)
        assert result.skip_reason == "no change"
