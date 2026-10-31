"""Double-read parity structure test: preflight reader vs DNADataset (D-03, 23-01 Task 2).

Every supported format is written to a tmp fixture once and read TWICE:
through the preflight standalone reader (internal seam, no datasets import)
and through ``DNADataset.load_local_data`` (the parity reference from
``dnallm/datahandling/data.py:82-334``). The test pins agreement on row
count and the sequence/label column VALUES, plus the behavioral lock points
catalogued in 23-PATTERNS.md: the headerless-csv -> txt demotion rule
(data.py:220-235), the fasta label rule and multi-line accumulation
(data.py:259-279), both txt branches (data.py:282-300), and the
list-of-files rules (data.py:175-190). This file MAY import
``dnallm.datahandling`` — it is test-only; preflight never does.
"""

import json as jsonlib

import pandas as pd
import pytest

from dnallm.datahandling.data import DNADataset
from dnallm.preflight import validation as pv

SEQ_A = "ACGT" * 12  # 48 chars
SEQ_B = "TTTTACGGCCATTTACGGAT"  # 20 chars
SEQ_C = "GGGN" * 7  # 28 chars with Ns

# DNADataset renames to sequence/labels; preflight uses the caller's names.
SEQ_COL = "sequence"
LABEL_COL = "label"


def _dnadataset_values(path, seq_col=SEQ_COL, label_col=LABEL_COL):
    """Read the fixture through the parity reference and return raw columns."""
    ds = DNADataset.load_local_data(str(path), seq_col=seq_col, label_col=label_col)
    return list(ds.dataset["sequence"]), list(ds.dataset["labels"])


def _preflight_values(path, seq_col=SEQ_COL, label_col=LABEL_COL):
    """Read the fixture through the preflight standalone reader seam."""
    return pv._read_local_values(str(path), seq_col=seq_col, label_col=label_col)


def _norm_label(value):
    """Normalize per DNADataset._format_labels semantics: numeric strings become floats."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _assert_parity(path, seq_col=SEQ_COL, label_col=LABEL_COL):
    """Assert the two readers agree on every sequence and label value."""
    pf_seqs, pf_labels = _preflight_values(path, seq_col=seq_col, label_col=label_col)
    ds_seqs, ds_labels = _dnadataset_values(path, seq_col=seq_col, label_col=label_col)
    assert pf_seqs == [str(s) for s in ds_seqs]
    assert len(pf_seqs) == len(ds_seqs)  # row-count agreement, stated explicitly
    if pf_labels is None:
        assert ds_labels == [] or list(ds_labels) == []
    else:
        assert [_norm_label(v) for v in pf_labels] == [_norm_label(v) for v in ds_labels]


# ─────────────────────────────────────────────────────────────────────────────
# Structured formats
# ─────────────────────────────────────────────────────────────────────────────


def test_parity_csv_with_header(tmp_path) -> None:
    """Headered csv: both readers yield identical sequence/label values."""
    path = tmp_path / "with_header.csv"
    path.write_text(f"{SEQ_COL},{LABEL_COL}\n{SEQ_A},1\n{SEQ_B},0\n{SEQ_C},1\n", encoding="utf-8")
    _assert_parity(path)


def test_parity_csv_headerless_demotes_to_txt(tmp_path) -> None:
    """Headerless csv demotes to the txt rule in BOTH readers (data.py:220-235)."""
    path = tmp_path / "headerless.csv"
    path.write_text(f"{SEQ_A},1\n{SEQ_B},0\n", encoding="utf-8")
    _assert_parity(path)


def test_parity_tsv_with_header(tmp_path) -> None:
    """Headered tsv: tab-delimited values agree across both readers."""
    path = tmp_path / "data.tsv"
    path.write_text(f"{SEQ_COL}\t{LABEL_COL}\n{SEQ_A}\t1\n{SEQ_B}\t0\n", encoding="utf-8")
    _assert_parity(path)


def test_parity_json_records(tmp_path) -> None:
    """JSON records: values agree across both readers."""
    path = tmp_path / "data.json"
    path.write_text(
        jsonlib.dumps([
            {"sequence": SEQ_A, "label": 1},
            {"sequence": SEQ_B, "label": 0},
        ]),
        encoding="utf-8",
    )
    _assert_parity(path)


def test_parity_parquet(tmp_path) -> None:
    """Parquet: values agree across both readers (pyarrow vs datasets loaders)."""
    path = tmp_path / "data.parquet"
    pd.DataFrame({"sequence": [SEQ_A, SEQ_B], "label": [1, 0]}).to_parquet(path)
    _assert_parity(path)


def test_parity_list_of_files_concatenates(tmp_path) -> None:
    """Structured formats accept a list of file paths, concatenated, in both readers."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text(f"{SEQ_COL},{LABEL_COL}\n{SEQ_A},1\n", encoding="utf-8")
    second.write_text(f"{SEQ_COL},{LABEL_COL}\n{SEQ_B},0\n", encoding="utf-8")
    pf_seqs, _ = pv._read_local_values(
        [str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL
    )
    ds = DNADataset.load_local_data([str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL)
    assert pf_seqs == [str(s) for s in ds.dataset["sequence"]]
    assert len(pf_seqs) == 2


# ─────────────────────────────────────────────────────────────────────────────
# FASTA: label rule and multi-line accumulation (data.py:259-279)
# ─────────────────────────────────────────────────────────────────────────────


def test_parity_fasta_label_rule_and_multiline_accumulation(tmp_path) -> None:
    """Fasta label = header.split('|')[-1] when '|' present else whole header;
    sequences accumulate across lines until the next '>' line."""
    path = tmp_path / "data.fasta"
    path.write_text(
        ">seq1|species one\nACGT\nACGTAC\n>seq2\nTTTTAAAACCCC\n>seq3|x|final\nGG\nCC\n",
        encoding="utf-8",
    )
    _assert_parity(path)
    # Pin the derived semantics explicitly (not just reader-vs-reader agreement):
    pf_seqs, pf_labels = _preflight_values(path)
    assert pf_seqs == ["ACGTACGTAC", "TTTTAAAACCCC", "GGCC"]
    assert pf_labels == ["species one", "seq2", "final"]


def test_parity_fasta_list_refusal_matches(tmp_path) -> None:
    """A list of fasta paths is refused with the same ValueError in both readers."""
    first = tmp_path / "one.fa"
    second = tmp_path / "two.fa"
    first.write_text(f">a|x\n{SEQ_A}\n", encoding="utf-8")
    second.write_text(f">b|y\n{SEQ_B}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"FASTA files must be single files"):
        pv._read_local_values([str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL)
    with pytest.raises(ValueError, match=r"FASTA files must be single files"):
        DNADataset.load_local_data([str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL)


# ─────────────────────────────────────────────────────────────────────────────
# TXT: both branches (data.py:282-300)
# ─────────────────────────────────────────────────────────────────────────────


def test_parity_txt_header_line_triggers_csv_reparse(tmp_path) -> None:
    """A .txt whose first line contains both column names is re-parsed as csv."""
    path = tmp_path / "headerish.txt"
    path.write_text(f"{SEQ_COL},{LABEL_COL}\n{SEQ_A},1\n{SEQ_B},0\n", encoding="utf-8")
    _assert_parity(path)


def test_parity_txt_plain_split(tmp_path) -> None:
    """Plain txt split: first two whitespace-separated fields; 1-field lines skipped.

    Labels stay consistently string-typed: DNADataset's Arrow cast rejects
    mixed numeric/string label columns, so a numeric/string mix is outside
    the parity reference's own domain.
    """
    path = tmp_path / "plain.txt"
    path.write_text(f"{SEQ_A} one\n{SEQ_B} zero\nONLYONEFIELD\n{SEQ_C} extra\n", encoding="utf-8")
    _assert_parity(path)
    # Pin the skipping/first-two-fields semantics explicitly:
    pf_seqs, pf_labels = _preflight_values(path)
    assert pf_seqs == [SEQ_A, SEQ_B, SEQ_C]
    assert pf_labels == ["one", "zero", "extra"]


def test_parity_txt_list_refusal_matches(tmp_path) -> None:
    """A list of txt paths is refused with the same ValueError in both readers."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text(f"{SEQ_A} 1\n", encoding="utf-8")
    second.write_text(f"{SEQ_B} 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"TXT files must be single files"):
        pv._read_local_values([str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL)
    with pytest.raises(ValueError, match=r"TXT files must be single files"):
        DNADataset.load_local_data([str(first), str(second)], seq_col=SEQ_COL, label_col=LABEL_COL)
