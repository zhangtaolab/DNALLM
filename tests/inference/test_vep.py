"""Behavior tests for the zero-shot VEP scoring core (vep module).

Fast-lane strategy: the same-slot evaluability rule runs on the real
character-level tokenizer from tests/conftest.py so alignment outcomes are
real tokenizations (single-character substitutions are same-slot by
construction); the multi-slot skip path — unreachable with any char-level
vocabulary — runs on a minimal stub tokenizer defined below. The scoring
kernels run on the real tiny per-position torch module so every score is a
real model output. No network, no model downloads, no skips.
"""

from typing import ClassVar

import pytest

from dnallm.inference.vep import VariantAlignment, align_variant


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
