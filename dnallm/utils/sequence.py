"""
Sequence utility functions for DNA sequence analysis and generation.

This module provides functions for:
- Calculating GC content
- Generating reverse complements
- Converting sequences to k-mers
- Validating DNA sequences
- Randomly generating DNA sequences with constraints

All functions are designed for use in DNA language modeling and
bioinformatics pipelines.
"""

import random
from tqdm.auto import tqdm


def calc_gc_content(seq: str) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        seq (str): DNA sequence (A/C/G/T/U/N, case-insensitive).

    Returns:
        float: GC content (0.0 ~ 1.0). Returns 0.0 if sequence is empty.
    """
    seq = seq.upper().replace("U", "T").replace("N", "")
    if len(seq) == 0:
        gc = 0.0
    else:
        gc = (seq.count("G") + seq.count("C")) / len(seq)
    return gc


def reverse_complement(
    seq: str, reverse: bool = True, complement: bool = True
) -> str:
    """Compute the reverse complement of a DNA sequence.

    Args:
        seq (str): DNA sequence.
        reverse (bool, optional): Whether to reverse the sequence.
            Defaults to True.
        complement (bool, optional): Whether to complement the sequence.
            Defaults to True.

    Returns:
        str: The reverse complement (or as specified) of the input sequence.
    """
    mapping = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "a": "t",
        "t": "a",
        "c": "g",
        "g": "c",
        "N": "N",
        "n": "n",
    }
    if reverse:
        seq = seq[::-1]
    if complement:
        seq = "".join(mapping.get(base, base) for base in seq)
    return seq


def seq2kmer(seqs: list[str], k: int) -> list[str]:
    """Convert a list of DNA sequences to k-mers (overlapping k-mer
    tokenization).

    Args:
        seqs (list[str]): List of DNA sequences.
        k (int): k-mer length.

    Returns:
        list[str]: List of k-mer tokenized sequences (space-separated
            k-mers).
    """
    all_kmers = []
    for seq in seqs:
        kmer = [seq[x : x + k].upper() for x in range(len(seq) + 1 - k)]
        kmers = " ".join(kmer)
        all_kmers.append(kmers)
    return all_kmers


def check_sequence(
    seq: str,
    minl: int = 1,
    maxl: int = 500000000,
    gc: tuple = (0, 1),
    valid_chars: str = "ACGTN",
) -> bool:
    """Check if a DNA sequence is valid based on length, GC content, and
    allowed characters.

    Args:
        seq (str): DNA sequence.
        minl (int, optional): Minimum length. Defaults to 1.
        maxl (int, optional): Maximum length. Defaults to 500000000.
        gc (tuple, optional): GC content range (min, max). Defaults to (0, 1).
        valid_chars (str, optional): Allowed characters. Defaults to "ACGTN".

    Returns:
        bool: True if valid, False otherwise.
    """
    if len(seq) < minl or len(seq) > maxl:
        return False  # 序列长度不在有效范围内
    elif gc[0] > calc_gc_content(seq) or gc[1] < calc_gc_content(seq):
        return False  # GC含量不在有效范围内
    elif set(seq.upper()) - set(valid_chars) != set():
        return False  # 序列包含不支持的字符
    else:
        return True  # 序列有效


def random_generate_sequences(
    minl: int,
    maxl: int = 0,
    samples: int = 1,
    gc: tuple = (0, 1),
    n_ratio: float = 0.0,
    padding_size: int = 0,
    seed: int | None = None,
) -> list[str]:
    """Randomly generate DNA sequences with specified length, GC content,
    and N ratio.

    Args:
        minl (int): Minimum sequence length.
        maxl (int, optional): Maximum sequence length. If 0, use minl as
            fixed length. Defaults to 0.
        samples (int, optional): Number of sequences to generate.
            Defaults to 1.
        gc (tuple, optional): GC content range (min, max).
            Defaults to (0, 1).
        N_ratio (float, optional): Proportion of 'N' bases (0.0 ~ 1.0).
            Defaults to 0.0.
        padding_size (int, optional): Pad length to nearest multiple.
            Defaults to 0.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        list[str]: List of generated DNA sequences.
    """
    sequences = []
    basemap = ["A", "C", "G", "T"]
    if 0.0 < n_ratio <= 1.0:
        basemap.append("N")
        weights = [(1 - n_ratio) / 4] * 4 + [n_ratio]
    elif n_ratio > 1.0:
        basemap.append("N")
        weights = [(100 - n_ratio) / 4] * 4 + [n_ratio]
    else:
        weights = None
    calc_gc = (
        False if gc == (0, 1) else True
    )  # Guanqing Please check this line!
    if seed:
        random.seed(seed)
    # progress bar
    progress_bar = tqdm(total=samples, desc="Generating sequences")
    # generate sequences
    if maxl:
        # generate sequences with random length
        while True:
            if len(sequences) >= samples:
                break
            length = random.randint(minl, maxl)  # noqa: S311
            if padding_size:
                length = (
                    (length // padding_size + 1) * padding_size
                    if length % padding_size
                    else length
                )
                if length > maxl:
                    length -= padding_size
            seq = "".join(
                random.choices(  # noqa: S311
                    basemap,
                    weights=weights,
                    k=length,
                )
            )
            if calc_gc:
                gc_content = calc_gc_content(seq)
                if gc[0] <= gc_content <= gc[1]:
                    sequences.append(seq)
                    progress_bar.update(1)
            else:
                sequences.append(seq)
                progress_bar.update(1)
    # generate sequences with fixed length
    else:
        maxl = minl
        length = minl
        while True:
            if len(sequences) >= samples:
                break
            seq = "".join(
                random.choices(  # noqa: S311
                    basemap,
                    weights=weights,
                    k=length,
                )
            )
            # calculate GC content
            if calc_gc:
                gc_content = calc_gc_content(seq)
                if gc[0] <= gc_content <= gc[1]:
                    sequences.append(seq)
                    progress_bar.update(1)
            else:
                sequences.append(seq)
                progress_bar.update(1)
    return sequences
