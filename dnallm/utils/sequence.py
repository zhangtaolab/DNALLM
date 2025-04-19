import random
from tqdm.auto import tqdm


def calc_gc_content(seq: str):
    '''
    Calculate the GC content of a DNA sequence.
    
    input:
        seq: str, DNA sequence
    output:
        float, GC content
    '''
    seq = seq.upper().replace("U", "T").replace("N", "")
    try:
        gc = (seq.count("G") + seq.count("C")) / len(seq)
    except ZeroDivisionError as e:
        print(e)
        gc = 0.0
    return gc


def reverse_complement(seq: str, reverse=True, complement=True):
    '''
    Compute the reverse complement of a DNA sequence.
    '''
    mapping = {
        "A": "T", "T": "A", "C": "G", "G": "C",
        "a": "t", "t": "a", "c": "g", "g": "c",
        "N": "N", "n": "n"
    }
    if reverse:
        seq = seq[::-1]
    if complement:
        seq = "".join(mapping.get(base, base) for base in seq)
    return seq


def seq2kmer(seqs: str, k: int):
    '''
    Convert DNA sequences to overlaped k-mers, seq2kmer(["ATGC"], 2) should return ['AT TG GC'] 
    
    input:
        seqs: list, DNA sequences
        k: int, k-mer length
    output:
        list of str, overlaped k-mers, 
    '''
    all_kmers = []
    for seq in seqs:
        kmer = [seq[x:x+k].upper() for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        all_kmers.append(kmers)
    return all_kmers


def check_sequence(seq: str, minl: int = 20, maxl: int = 6000, gc: tuple = (0,1), valid_chars: str = "ACGTN"):
    '''
    Check if a DNA sequence is valid.
    
    input:
        seq: str, DNA sequence
        minl: int, minimum length of the sequence, default 20
        maxl: int, maximum length of the sequence, default 6000
        valid_chars: str, valid characters in the sequence, default "ACGTN"
    output:
        bool, whether the sequence is valid
    '''
    if len(seq) < minl or len(seq) > maxl:
        return False  # 序列长度不在有效范围内
    elif gc[0] > calc_gc_content(seq) or gc[1] < calc_gc_content(seq):
        return False  # GC含量不在有效范围内
    elif set(seq.upper()) - set(valid_chars) != set():
        return False  # 序列包含不支持的字符
    else:
        return True  # 序列有效


def random_generate_sequences(minl: int, maxl: int = 0, samples: int = 1,
                              gc: tuple = (0,1), N_ratio: float = 0.0,
                              padding_size: int = 0, seed: int = None):
    '''
    Randomly generate DNA sequences with specified length and GC content.
    
    input:
        minl: int, minimum length of the sequences
        maxl: int, maximum length of the sequences, default is the same as minl
        samples: int, number of sequences to generate, default 1
        with_N: bool, whether to include N in the base map, default False
        gc: tuple, GC content range, default (0,1)
        padding_size: int, padding size for sequence length, default 0
        seed: int, random seed, default None
    output:
        list, generated DNA sequences
    '''
    sequences = []
    basemap = ["A", "C", "G", "T"]
    if 0.0 < N_ratio <= 1.0:
        basemap.append("N")
        weights = [(1 - N_ratio) / 4] * 4 + [N_ratio]
    elif N_ratio > 1.0:
        basemap.append("N")
        weights = [(100 - N_ratio) / 4] * 4 + [N_ratio]
    else:
        weights = None
    calc_gc = False if gc == (0,1) else True
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
            length = random.randint(minl, maxl)
            if padding_size:
                # padding the length to the nearest multiple of padding_size
                length = (length // padding_size + 1) * padding_size if length % padding_size else length
                if length > maxl:
                    length -= padding_size
            seq = "".join(random.choices(basemap, weights=weights, k=length))
            # calculate GC content
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
            seq = "".join(random.choices(basemap, weights=weights, k=length))
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

