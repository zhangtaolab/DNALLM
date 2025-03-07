import random


def calc_gc_content(seq):
    '''
    Calculate the GC content of a DNA sequence.
    
    input:
        seq: str, DNA sequence
    output:
        float, GC content
    '''
    seq = seq.upper().replace("U", "T").replace("N", "")
    return (seq.count("G") + seq.count("C")) / len(seq)


def check_sequence(seq, minl=20, maxl=6000, valid_chars="ACGTN"):
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
    elif set(seq.upper()) - set(valid_chars) != set():
        return False  # 序列包含不支持的字符
    else:
        return True  # 序列有效


def random_generate_sequences(minl, maxl=0, samples=1, with_N=False, gc=(0,1), padding_size=0, seed=None):
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
    if with_N:
        basemap.append("N")
    calc_gc = False if gc == (0,1) else True
    baseidx = len(basemap) - 1
    if seed:
        random.seed(seed)
    if maxl:
        while True:
            if len(sequences) >= samples:
                break
            length = random.randint(minl, maxl)
            if padding_size:
                # padding the length to the nearest multiple of padding_size
                length = (length // padding_size + 1) * padding_size if length % padding_size else length
                if length > maxl:
                    length -= padding_size
            seq = "".join([basemap[random.randint(0,baseidx)] for _ in range(length)])
            # calculate GC content
            if calc_gc:
                gc_content = calc_gc_content(seq)
                if gc[0] <= gc_content <= gc[1]:
                    sequences.append(seq)
            else:
                sequences.append(seq)
    else:
        while True:
            if len(sequences) >= samples:
                break
            seq = "".join([basemap[random.randint(0,baseidx)] for _ in range(minl)])
            # calculate GC content
            if calc_gc:
                gc_content = calc_gc_content(seq)
                if gc[0] <= gc_content <= gc[1]:
                    sequences.append(seq)
            else:
                sequences.append(seq)
    
    return sequences
