import pytest

from dnallm.utils.sequence import (
    calc_gc_content,
    check_sequence,
    random_generate_sequences,
    reverse_complement,
    seq2kmer,
)


def test_calc_gc_content():
    # Test normal DNA sequence
    assert calc_gc_content("ATGC") == 0.5
    assert calc_gc_content("AAAA") == 0.0
    assert calc_gc_content("CCGG") == 1.0

    # Test with lowercase
    assert calc_gc_content("atgc") == 0.5

    # Test with U (RNA)
    assert calc_gc_content("AUGC") == 0.5

    # Test with N
    assert calc_gc_content("ATGCN") == 0.5

    # Test empty sequence
    assert calc_gc_content("") == 0.0


def test_reverse_complement():
    # Test normal DNA sequence
    assert reverse_complement("ATGC") == "GCAT"

    # Test with lowercase
    assert reverse_complement("atgc") == "gcat"

    # Test with N
    assert reverse_complement("ATGCN") == "NGCAT"

    # Test only reverse
    assert reverse_complement("ATGC", complement=False) == "CGTA"

    # Test only complement
    assert reverse_complement("ATGC", reverse=False) == "TACG"


def test_seq2kmer():
    # Test single sequence
    assert seq2kmer(["ATGC"], 2) == ["AT TG GC"]

    # Test multiple sequences
    assert seq2kmer(["ATGC", "GCTA"], 2) == ["AT TG GC", "GC CT TA"]

    # Test with k=3
    assert seq2kmer(["ATGCTA"], 3) == ["ATG TGC GCT CTA"]


def test_check_sequence():
    # Test valid sequence
    assert check_sequence("ATGC" * 10, minl=20, maxl=100)

    # Test sequence too short
    assert not check_sequence("ATGC", minl=20)

    # Test sequence too long
    assert not check_sequence("ATGC" * 2000, maxl=100)

    # Test invalid characters
    assert not check_sequence("ATGCX", valid_chars="ACGTN")

    # Test GC content
    assert check_sequence("ATGC", gc=(0.4, 0.6), minl=4)
    assert not check_sequence("AAAA", gc=(0.4, 0.6))


def test_random_generate_sequences():
    # Test basic sequence generation
    seqs = random_generate_sequences(minl=20, samples=5, seed=42)
    assert len(seqs) == 5
    assert all(len(seq) == 20 for seq in seqs)

    # Test with length range
    seqs = random_generate_sequences(minl=20, maxl=30, samples=5, seed=42)
    assert len(seqs) == 5
    assert all(20 <= len(seq) <= 30 for seq in seqs)

    # Test with GC content constraint
    seqs = random_generate_sequences(
        minl=20, samples=5, gc=(0.4, 0.6), seed=42
    )
    assert len(seqs) == 5
    assert all(0.4 <= calc_gc_content(seq) <= 0.6 for seq in seqs)

    # Test with N ratio
    seqs = random_generate_sequences(minl=20, samples=5, n_ratio=0.1, seed=42)
    assert len(seqs) == 5
    assert any("N" in seq for seq in seqs)

    # Test with padding
    seqs = random_generate_sequences(
        minl=20, maxl=30, samples=5, padding_size=5, seed=42
    )
    assert len(seqs) == 5
    assert all(len(seq) % 5 == 0 for seq in seqs)
