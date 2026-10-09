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
import json as jsonlib
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dnallm.inference.vep import (
    RefMismatchError,
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
        violation — a ValueError (as the dedicated RefMismatchError
        subclass), distinct from the skip path."""
        with pytest.raises(RefMismatchError, match="does not match"):
            align_variant("ACGTTGCA", 3, "A", "T", simple_dna_tokenizer)
        # The subclass IS a ValueError: legacy except-clauses keep working.
        with pytest.raises(ValueError, match="does not match sequence at position 3"):
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
        AUROC == AUPRC == 1.0 — the metric plumbing is registry-driven over
        the deleteriousness score (higher = more pathogenic)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        seq = _load_fixture_reference()["chrT"]
        # Positives G->(A/C/T): ref token likely, alt unlikely -> very
        # negative delta -> high deleteriousness. Negatives (A/C/T)->G:
        # alt at least as likely as ref -> non-negative delta -> low
        # deleteriousness. Perfect separation either way.
        rows = [
            (
                "chrT",
                _nth_position(seq, "G", 0, 97),
                "G",
                "A",
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
                "T",
                "Pathogenic",
                "reviewed_by_expert_panel",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                _nth_position(seq, "C", 0, 97),
                "C",
                "G",
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
                _nth_position(seq, "A", 0, 97),
                "A",
                "G",
                "Likely_benign",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
        ]
        vcf = _write_vcf(tmp_path / "sep.vcf", rows)
        # ref(G) logp -0.1, everything else -2.0: every positive delta is
        # -1.9 (deleteriousness +1.9), every negative delta is +1.9
        # (deleteriousness -1.9) — perfect separation.
        fake = {_ID_A: -2.0, _ID_C: -2.0, _ID_G: -0.1, _ID_T: -2.0}

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


# ---------------------------------------------------------------------------
# Slow lane: real ClinVar acceptance (D-09 tier 2). Real data stays OUT of
# the repo; the fast lane above is network-free by construction.
# ---------------------------------------------------------------------------

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
CHR22_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
DOWNLOAD_TIMEOUT_S = 120
SAMPLE_SEED = 42
PER_CLASS_TARGET = 500  # 1k-sample acceptance target (D-09 tier 2)
CONTEXT_WINDOW = 200  # the VepConfig protocol default

# Within-paradigm/within-convention literature anchors (RESEARCH "State of
# the Art"); small plant DNA models are expected BELOW the big-model anchors
# on human ClinVar — the comparison is a recorded finding, never a hard band.
_ANCHORS = {
    "mlm": {
        "anchor": "Nucleotide Transformer 2.5B MLM ClinVar AUC 0.80 (NT paper A.5.2)",
        "range": (0.70, 0.80),
    },
    "clm": {
        "anchor": "Evo2-40B CLM ~0.98 (evo2-clinvar, >=2-star convention)",
        "range": (0.85, 0.98),
    },
}

# Pinned scoring models (models.lock rows; CLM/MLM mix, paradigm matched to
# each architecture via the Task 2 guard).
_CLINVAR_MODELS = [
    ("zhangtaolab/plant-dnabert-BPE", "modelscope", "mlm"),
    ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "huggingface", "mlm"),
    ("zhangtaolab/plant-dnagpt-6mer", "modelscope", "clm"),
    ("zhangtaolab/plant-dnamamba-BPE-open_chromatin", "modelscope", "clm"),
    ("zhangtaolab/plant-dnagpt-BPE-promoter", "modelscope", "clm"),
]


def _probe(url):
    """HEAD-probe one canonical host; return an evidence string or None."""
    try:
        # Callers pass hardcoded https constants only (no file:/custom schemes).
        request = urllib.request.Request(url, method="HEAD")  # ruff: ignore[suspicious-url-open-usage]
        with urllib.request.urlopen(request, timeout=30):  # ruff: ignore[suspicious-url-open-usage]
            return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"{url} -> {exc}"


def _download(url, destination):
    """Stream one download into the isolated session tmp dir."""
    # Callers pass hardcoded https constants only (no file:/custom schemes).
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response,  # ruff: ignore[suspicious-url-open-usage]
        open(destination, "wb") as handle,
    ):
        while chunk := response.read(1 << 20):
            handle.write(chunk)


@pytest.fixture(scope="class")
def clinvar_cohort(tmp_path_factory):
    """Download ClinVar + chr22, build the D-17 cohort slice under tmp.

    Yields the slice VCF path (downloads live only under the session tmp
    dir; nothing is committed). Typed-skips with the ``clinvar-unavailable:``
    prefix when either canonical host is unreachable.
    """
    evidence = _probe(CLINVAR_URL) or _probe(CHR22_URL)
    if evidence:
        pytest.skip(f"clinvar-unavailable: canonical host unreachable ({evidence})")

    workdir = tmp_path_factory.mktemp("clinvar")
    vcf_gz = workdir / "clinvar.vcf.gz"
    chr22_gz = workdir / "chr22.fa.gz"
    _download(CLINVAR_URL, vcf_gz)
    _download(CHR22_URL, chr22_gz)

    import allel

    callset = allel.read_vcf(
        str(vcf_gz),
        fields=[
            "variants/CHROM",
            "variants/POS",
            "variants/ID",
            "variants/REF",
            "variants/ALT",
            "variants/CLNSIG",
            "variants/CLNREVSTAT",
            "variants/CLNVC",
        ],
        alt_number=2,
    )
    from dnallm.inference.vep import _load_reference

    chr22 = _load_reference(chr22_gz)["chr22"]

    star_tokens = {"criteria_provided", "reviewed_by_expert_panel", "practice_guideline"}
    labels_map = {
        "Pathogenic": 1,
        "Likely_pathogenic": 1,
        "Benign": 0,
        "Likely_benign": 0,
    }
    cohort = {"pos": [], "ref": [], "alt": [], "label": [], "revstat": []}
    mismatches = 0
    for chrom, pos, ref, alt_row, sig, rev, vc in zip(
        callset["variants/CHROM"],
        callset["variants/POS"],
        callset["variants/REF"],
        callset["variants/ALT"],
        callset["variants/CLNSIG"],
        callset["variants/CLNREVSTAT"],
        callset["variants/CLNVC"],
        strict=True,
    ):
        if str(chrom) != "22" or str(vc) != "single_nucleotide_variant":
            continue
        label = labels_map.get(str(sig))
        if label is None or str(rev) not in star_tokens:
            continue
        alt = str(alt_row[0])
        if len(str(ref)) != 1 or len(alt) != 1 or (alt_row[1] != "" and str(alt_row[1]) != "."):
            continue  # keep single-ALT SNVs only
        pos0 = int(pos) - 1
        if chr22[pos0].upper() != str(ref).upper():
            mismatches += 1  # liftover drift: excluded, counted
            continue
        cohort["pos"].append(int(pos))
        cohort["ref"].append(str(ref).upper())
        cohort["alt"].append(alt.upper())
        cohort["label"].append(label)
        cohort["revstat"].append(str(rev))  # preserve the star-level token

    rng = np.random.default_rng(SAMPLE_SEED)
    slice_path = workdir / "clinvar_chr22_slice.vcf"
    chosen = []
    for class_label in (1, 0):
        idx = [i for i, lab in enumerate(cohort["label"]) if lab == class_label]
        take = rng.permutation(idx)[:PER_CLASS_TARGET]
        chosen.extend(take.tolist())
    with open(slice_path, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##INFO=<ID=CLNSIG,Number=.,Type=String,Description="cs">\n')
        handle.write('##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="rs">\n')
        handle.write('##INFO=<ID=CLNVC,Number=1,Type=String,Description="vt">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for i in chosen:
            sig = "Pathogenic" if cohort["label"][i] == 1 else "Benign"
            handle.write(
                f"22\t{cohort['pos'][i]}\t.\t{cohort['ref'][i]}\t{cohort['alt'][i]}\t.\t.\t"
                f"CLNSIG={sig};CLNREVSTAT={cohort['revstat'][i]};"
                f"CLNVC=single_nucleotide_variant\n"
            )
    return {
        "slice_vcf": str(slice_path),
        "reference": {"chr22": chr22},
        "sampled": len(chosen),
        "available": len(cohort["label"]),
        "ref_mismatches_excluded": mismatches,
    }


@pytest.mark.slow
class TestClinVarAcceptance:
    """D-09 tier-2 acceptance: real ClinVar GRCh38 chr22 cohort, D-17
    convention, >= 5 pinned models with a CLM/MLM mix.

    Downloads (documented canonical hosts):
      - ClinVar VCF: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
      - GRCh38 chr22: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz
        (UCSC hg38 chromosomes are soft-masked — the run exercises the
        uppercase-window path on real data)
    """

    @pytest.mark.timeout(2400)
    @pytest.mark.parametrize(("model_name", "source", "paradigm"), _CLINVAR_MODELS)
    def test_clinvar_auroc_with_convention_block(
        self, clinvar_cohort, model_name, source, paradigm
    ):
        """Each model's AUROC is computed on the same 1k D-17 cohort, at or
        above the random floor, with the convention block and per-reason
        skip fractions recorded, and compared within-paradigm against the
        RESEARCH anchor table as a documented finding."""
        from dnallm.configuration.configs import TaskConfig
        from dnallm.models import load_model_and_tokenizer

        task_config = TaskConfig(task_type="generation" if paradigm == "clm" else "mask")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, task_config=task_config, source=source
        )

        result = evaluate_vcf(
            model,
            tokenizer,
            clinvar_cohort["slice_vcf"],
            clinvar_cohort["reference"],
            paradigm=paradigm,
            context_window=CONTEXT_WINDOW,
        )

        assert clinvar_cohort["sampled"] >= 500  # 1k target, hard floor at 500
        assert result.evaluated > 0
        assert result.metrics is not None
        auroc = result.metrics["AUROC"]
        # Sanity floor, not a performance claim: small models are expected
        # NEAR the random floor on this convention, and the null standard
        # error at 500/500 sampling is ~0.018 — a hard 0.5 cutoff would
        # fail statistically-at-chance models half the time (empirically:
        # plant-dnagpt-BPE-promoter measured 0.4904 +/- noise). 0.45 sits
        # ~2.7 SE below chance: systematic score/label inversion (~0.2-0.35
        # for signal-bearing models) and broken wiring (~0 / NaN) still
        # fail loudly; honest within-floor results are recorded below.
        assert auroc >= 0.45, f"{model_name}: AUROC {auroc} below the sanity floor"

        # The convention block ships beside every AUROC (never bare).
        assert result.convention["star_floor"] == 1
        assert result.convention["variant_type"] == "single_nucleotide_variant"
        assert result.convention["clnrevstat_counts"], "per-star counts recorded"

        # Skip fractions per reason per model — the tokenizer-class finding
        # (Pitfall 6): the fraction IS the data about each tokenizer.
        assert sum(result.skip_counts.values()) == result.skipped
        skip_fractions = {
            reason: count / max(result.evaluated + result.skipped, 1)
            for reason, count in result.skip_counts.items()
        }

        # Within-paradigm anchor comparison (recorded finding, not a
        # threshold): small plant DNA models are expected below the
        # big-model anchors on human ClinVar.
        anchor = _ANCHORS[paradigm]
        comparison = {
            "model": model_name,
            "paradigm": paradigm,
            "auroc": auroc,
            "auprc": result.metrics["AUPRC"],
            "evaluated": result.evaluated,
            "skip_fraction": result.skip_fraction,
            "skip_fractions_by_reason": skip_fractions,
            "anchor": anchor["anchor"],
            "anchor_range": anchor["range"],
            "within_anchor_range": anchor["range"][0] <= auroc <= anchor["range"][1],
            "cohort": clinvar_cohort["sampled"],
            "convention": result.convention,
        }
        print(jsonlib.dumps(comparison, indent=2, sort_keys=True))
        assert set(comparison) >= {
            "auroc",
            "anchor_range",
            "within_anchor_range",
            "skip_fractions_by_reason",
        }


class TestEvaluateVcfEdgeCases:
    """Input-boundary behaviors of the driver (V5 untrusted-input discipline)."""

    def test_header_only_vcf_returns_empty_result(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A VCF with zero records yields the empty edge (metrics None,
        skip_fraction 1.0) without touching the kernels."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        path = tmp_path / "empty_records.vcf"
        path.write_text(_VCF_HEADER, encoding="utf-8")

        result = evaluate_vcf(model, simple_dna_tokenizer, str(path), FIXTURE_FA, paradigm="mlm")

        assert result.records == []
        assert result.metrics is None
        assert result.skip_fraction == 1.0
        assert result.convention["rows_read"] == 0

    def test_vcf_without_info_fields_excludes_all_rows(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A VCF carrying no CLNSIG/CLNREVSTAT/CLNVC INFO at all labels every
        row unlabeled — the absent-field branch of the convention filter."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        path = tmp_path / "noinfo.vcf"
        path.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chrT\t6\t.\tG\tC\t.\t.\t.\n",
            encoding="utf-8",
        )

        result = evaluate_vcf(model, simple_dna_tokenizer, str(path), FIXTURE_FA, paradigm="mlm")

        assert result.records == []
        # The CLNVC gate runs first: an absent CLNVC ("" cell) cannot equal
        # single_nucleotide_variant, so the row lands in the non-SNV bucket.
        assert result.convention["exclusion_counts"]["non_snv_clnvc"] == 1
        assert result.convention["exclusion_counts"]["unlabeled_clnsig"] == 0

    def test_bytes_cells_from_allel_are_coerced(self, tiny_model_factory, simple_dna_tokenizer):
        """allel may hand back bytes cells; the driver coerces them (the
        fast fake-callset proves the coercion path)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        fake_callset = {
            "variants/CHROM": np.array([b"chrT"], dtype=object),
            "variants/POS": np.array([1], dtype=np.int32),
            "variants/ID": np.array([b"rsX"], dtype=object),
            "variants/REF": np.array([b"A"], dtype=object),
            "variants/ALT": np.array([[b"G", "", "", ""]], dtype=object),
            "variants/CLNSIG": np.array([b"Pathogenic"], dtype=object),
            "variants/CLNREVSTAT": np.array([b"criteria_provided"], dtype=object),
            "variants/CLNVC": np.array([b"single_nucleotide_variant"], dtype=object),
        }

        with patch("allel.read_vcf", return_value=fake_callset):
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                "unused.vcf",
                {"chrT": "ACGTTGCA"},
                paradigm="mlm",
            )

        assert result.evaluated == 1
        assert result.records[0].chrom == "chrT"
        assert result.records[0].alt == "G"

    def test_missing_reference_file_raises_value_error(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """A reference path that does not exist is an input-contract error."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        with pytest.raises(ValueError, match="Reference FASTA not found"):
            evaluate_vcf(
                model,
                simple_dna_tokenizer,
                FIXTURE_VCF,
                "/nonexistent/ref.fa",
                paradigm="mlm",
            )

    def test_empty_fasta_raises_value_error(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A FASTA with no records is rejected, not silently treated as an
        empty genome."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="contains no sequences"):
            evaluate_vcf(model, simple_dna_tokenizer, FIXTURE_VCF, str(path), paradigm="mlm")

    def test_multirecord_fasta_loads_every_chromosome(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A multi-record FASTA parses every record (in-loop flush), and
        each CHROM resolves against its own sequence."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        path = tmp_path / "multi.txt"
        path.write_text(">chrA\nACGT\n>chrT\nACGTTGCAAGCTTAGGCATGCCTAGGTTACAGG\n", encoding="utf-8")
        rows = [
            (
                "chrA",
                2,
                "C",
                "G",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
            (
                "chrT",
                6,
                "G",
                "C",
                "Benign",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            ),
        ]
        vcf = _write_vcf(tmp_path / "multi.vcf", rows)

        result = evaluate_vcf(
            model, simple_dna_tokenizer, vcf, str(path), paradigm="mlm", context_window=6
        )

        assert result.evaluated == 2
        assert {r.chrom for r in result.records} == {"chrA", "chrT"}

    def test_unknown_paradigm_rejected_before_any_work(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """An unknown paradigm raises immediately (no VCF read)."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        with pytest.raises(ValueError, match="Unknown paradigm"):
            evaluate_vcf(model, simple_dna_tokenizer, FIXTURE_VCF, FIXTURE_FA, paradigm="plm")

    def test_read_vcf_failure_wrapped_as_dnallm_value_error(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """allel parse errors surface as a wrapped dnallm ValueError at the
        driver boundary — never a bare foreign traceback (T-11-10)."""
        model = tiny_model_factory(n_classes=9, pooled=False)

        with patch("allel.read_vcf", side_effect=OSError("gzip: bad magic")):
            with pytest.raises(ValueError, match="Failed to read VCF"):
                evaluate_vcf(model, simple_dna_tokenizer, FIXTURE_VCF, FIXTURE_FA, paradigm="mlm")

    def test_ref_contradicting_reference_raises_with_position_context(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """A REF allele that contradicts the reference sequence raises with
        chrom:pos context (input-contract violation, distinct from skips)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        rows = [
            (
                "chrT",
                6,
                "A",
                "T",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "badref.vcf", rows)  # position 6 is a G

        with pytest.raises(ValueError, match=r"REF/reference mismatch at chrT:6"):
            evaluate_vcf(model, simple_dna_tokenizer, vcf, FIXTURE_FA, paradigm="mlm")

    def test_tokenizer_error_not_relabelled_as_ref_mismatch(self, tiny_model_factory, tmp_path):
        """A plain ValueError raised by the tokenizer mid-scoring surfaces as
        'Scoring failed at <coord>' with its original text preserved — never
        a confidently-wrong REF-mismatch diagnosis (WR-04)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        # Position 6 of chrT is a G in the committed reference: REF matches,
        # so the failure below is purely the tokenizer's.
        rows = [
            (
                "chrT",
                6,
                "G",
                "C",
                "Pathogenic",
                "criteria_provided,_single_submitter",
                "single_nucleotide_variant",
            )
        ]
        vcf = _write_vcf(tmp_path / "tokfail.vcf", rows)

        class _ExplodingTokenizer:
            """Passes the mlm paradigm guard, explodes at tokenization."""

            mask_token_id = 4

            def __call__(self, *args, **kwargs):
                raise ValueError("tokenizer exploded: sequence too long")

        with pytest.raises(
            ValueError, match=r"Scoring failed at chrT:6.*tokenizer exploded"
        ) as exc_info:
            evaluate_vcf(model, _ExplodingTokenizer(), vcf, FIXTURE_FA, paradigm="mlm")

        assert "REF/reference mismatch" not in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert "tokenizer exploded" in str(exc_info.value.__cause__)

    def test_clm_guard_on_configless_model_object(self, simple_dna_tokenizer):
        """A model-like object with no config at all counts as bidirectional
        for the guard (conservative default)."""
        with pytest.raises(ValueError, match="Paradigm 'clm' requires a causal"):
            score_variant(
                SimpleNamespace(),
                simple_dna_tokenizer,
                "ACGTTGCA",
                3,
                "T",
                "A",
                paradigm="clm",
            )

    def test_clm_guard_accepts_causal_model_type_without_causal_architecture(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """A decoder-only family checkpoint whose architectures list was
        rewritten by fine-tuning (e.g. GPT2ForSequenceClassification with
        model_type='gpt2', loaded via AutoModelForCausalLM) is still causal:
        the model_type heuristic must accept it (real plant-dnagpt case)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        model.config.model_type = "gpt2"
        model.config.architectures = ["GPT2ForSequenceClassification"]

        result = score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "A", paradigm="clm")

        assert isinstance(result, float)

    def test_clm_guard_still_rejects_bidirectional_model_type(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """A genuinely bidirectional family (model_type='bert'-style,
        MaskedLM architectures) never passes the CLM guard — the
        model_type heuristic only admits decoder-only families."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        model.config.model_type = "bert"
        model.config.architectures = ["BertForSequenceClassification"]

        with pytest.raises(ValueError, match="Paradigm 'clm' requires a causal"):
            score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "A", paradigm="clm")

    def test_callset_missing_info_keys_counts_rows_unlabeled(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """A callset that lacks the INFO keys entirely (the absent-field
        branch) excludes its rows through the empty-string cell."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        fake_callset = {
            "variants/CHROM": np.array(["chrT"], dtype=object),
            "variants/POS": np.array([1], dtype=np.int32),
            "variants/ID": np.array(["rsX"], dtype=object),
            "variants/REF": np.array(["A"], dtype=object),
            "variants/ALT": np.array([["G", "", "", ""]], dtype=object),
        }

        with patch("allel.read_vcf", return_value=fake_callset):
            result = evaluate_vcf(
                model,
                simple_dna_tokenizer,
                "unused.vcf",
                {"chrT": "ACGTTGCA"},
                paradigm="mlm",
            )

        assert result.records == []
        assert result.convention["exclusion_counts"]["non_snv_clnvc"] == 1

    def test_clm_guard_accepts_causal_architecture_marker(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """An architectures entry carrying a causal marker (GPT2LMHeadModel)
        passes the guard even with is_decoder unset (branch 2 of the
        heuristic)."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        model.config.architectures = ["GPT2LMHeadModel"]

        result = score_variant(model, simple_dna_tokenizer, "ACGTTGCA", 3, "T", "A", paradigm="clm")

        assert isinstance(result, float)
