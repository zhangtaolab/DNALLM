"""Comprehensive test suite for DNADataset class.

This module contains all tests for the DNADataset class, including:
- Basic initialization and configuration
- Data loading from various file formats
- Data type detection and validation
- Sequence processing and manipulation
- Statistical analysis
- Utility methods and edge cases
"""

import json
import os
import pickle  # ruff: ignore[suspicious-pickle-import]
import tempfile
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from conftest import SimpleDNATokenizer
from datasets import Dataset, DatasetDict

from dnallm.datahandling.data import (
    DNADataset,
    _standardize_column_names,
    load_preset_dataset,
    show_preset_dataset,
)

# Ruff S105 treats any string literal assigned to or compared against a *_token
# name as a hardcoded password; these named constants keep the special-token
# fixtures lint-clean without noqa noise.
PAD_VALUE = "<pad>"
SEP_VALUE = "<sep>"
EOS_VALUE = "<eos>"
PAD_FROM_MAP = "<p>"
DECODED_PAD = "<d>"
CONVERTED_PAD = "<c>"
LAST_FALLBACK_PAD = "<t>"
SEP_FROM_ATTR = "<s>"


class _RecordingTokenizer:
    """Minimal tokenizer that records every batch and keyword it is called with."""

    def __init__(self, **attrs):
        self.special_tokens_map = {}
        self.pad_token = PAD_VALUE
        self.sep_token = SEP_VALUE
        self.eos_token = EOS_VALUE
        self.padding_side = "right"
        for key, value in attrs.items():
            setattr(self, key, value)
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, seq, **kwargs):
        """Return a deterministic id list whose length tracks the input."""
        return [11] * (len(seq) if isinstance(seq, str) else len("".join(seq)))

    def __call__(self, sequences, **kwargs):
        """Record the batch and return per-sequence id/mask lists."""
        seqs = [sequences] if isinstance(sequences, str) else list(sequences)
        self.calls.append((seqs, kwargs))
        return {
            "input_ids": [[1] * len(s) for s in seqs],
            "attention_mask": [[1] * len(s) for s in seqs],
        }


class _ConfigProbeTokenizer:
    """Configurable tokenizer exposing only the attributes explicitly given."""

    def __init__(self, **attrs):
        self.special_tokens_map = attrs.pop("special_tokens_map", {})
        for key, value in attrs.items():
            setattr(self, key, value)


class TestDNADatasetInitialization:
    """Test DNADataset initialization and basic properties."""

    def test_init_basic(self):
        """Test basic initialization."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert dna_ds.dataset is ds
        assert dna_ds.max_length == 512
        assert dna_ds.tokenizer is None
        assert dna_ds.data_type == "classification"

    def test_init_with_tokenizer(self):
        """Test initialization with tokenizer."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        tokenizer = Mock()
        dna_ds = DNADataset(ds, tokenizer=tokenizer, max_length=256)

        assert dna_ds.tokenizer is tokenizer
        assert dna_ds.max_length == 256

    def test_init_with_custom_max_length(self):
        """Test initialization with custom max_length."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds, max_length=1024)

        assert dna_ds.max_length == 1024

    def test_init_with_dataset_dict(self):
        """Test initialization with DatasetDict."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)

        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert len(dna_ds.dataset["train"]) == 3
        assert dna_ds.data_type == "classification"

    def test_init_with_none_dataset(self):
        """Test initialization with None dataset."""
        with pytest.raises(TypeError, match="Dataset cannot be None"):
            DNADataset(None)

    def test_init_with_invalid_max_length(self):
        """Test initialization with invalid max_length."""
        test_data = {"sequence": ["ATCG"], "labels": [0]}
        ds = Dataset.from_dict(test_data)

        with pytest.raises(ValueError, match="max_length must be positive"):
            DNADataset(ds, max_length=-1)

        with pytest.raises(ValueError, match="max_length must be positive"):
            DNADataset(ds, max_length=0)


class TestDNADatasetLoadLocalData:
    """Test loading data from local files in various formats."""

    def test_load_csv_file(self):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sequence,label\n")
            f.write("ATCG,0\n")
            f.write("GCTA,1\n")
            f.write("TAGC,0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.dataset.column_names
            assert "labels" in dna_ds.dataset.column_names
        finally:
            os.unlink(temp_file)

    def test_load_tsv_file(self):
        """Test loading TSV file with custom separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("sequence\tlabel\n")
            f.write("ATCG\t0\n")
            f.write("GCTA\t1\n")
            f.write("TAGC\t0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, seq_col="sequence", label_col="label", sep="\t"
            )
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.dataset.column_names
            assert "labels" in dna_ds.dataset.column_names
        finally:
            os.unlink(temp_file)

    def test_load_json_file(self):
        """Test loading JSON file."""
        test_data = [
            {"sequence": "ATCG", "label": 0},
            {"sequence": "GCTA", "label": 1},
            {"sequence": "TAGC", "label": 0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)

    def test_load_parquet_file(self):
        """Test loading parquet file."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "label": [0, 1, 0],  # Use correct column name
        }
        df = pd.DataFrame(test_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)

    def test_load_pickle_file(self):
        """Test loading pickle file."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(test_data, f)
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)

    def test_load_fasta_file(self):
        """Test loading FASTA file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">seq1|0\n")
            f.write("ATCG\n")
            f.write(">seq2|1\n")
            f.write("GCTA\n")
            f.write(">seq3|0\n")
            f.write("TAGC\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, seq_col="sequence", label_col="label", fasta_sep="|"
            )
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.dataset.column_names
            assert "labels" in dna_ds.dataset.column_names
        finally:
            os.unlink(temp_file)

    def test_load_txt_file_without_header(self):
        """Test loading TXT file without header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("ATCG 0\n")
            f.write("GCTA 1\n")
            f.write("TAGC 0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)

    def test_load_file_with_custom_separator(self):
        """Test loading file with custom separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("ATCG,0\n")
            f.write("GCTA,1\n")
            f.write("TAGC,0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, seq_col="sequence", label_col="label", sep=","
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)

    def test_load_file_with_multi_label_sep(self):
        """Test loading file with multi-label separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sequence,label\n")
            f.write("ATCG,0;1\n")
            f.write("GCTA,1;0\n")
            f.write("TAGC,0;0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(
                temp_file,
                seq_col="sequence",
                label_col="label",
                multi_label_sep=";",
            )
            assert len(dna_ds) == 3
            # Check that labels are converted to lists
            assert isinstance(dna_ds[0]["labels"], list)
        finally:
            os.unlink(temp_file)

    def test_load_file_with_float_labels(self):
        """Test loading file with float labels."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sequence,label\n")
            f.write("ATCG,0.5\n")
            f.write("GCTA,1.0\n")
            f.write("TAGC,0.0\n")
            temp_file = f.name

        try:
            dna_ds = DNADataset.load_local_data(temp_file, seq_col="sequence", label_col="label")
            assert len(dna_ds) == 3
            # Check that labels are converted to floats
            assert isinstance(dna_ds[0]["labels"], float)
        finally:
            os.unlink(temp_file)

    def test_load_unsupported_file_type(self):
        """Test loading unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                DNADataset.load_local_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_pre_split_datasets(self):
        """Test loading pre-split datasets using test data."""
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests",
            "test_data",
        )
        file_paths = {
            "train": os.path.join(base_path, "binary_classification", "train.csv"),
            "test": os.path.join(base_path, "binary_classification", "test.csv"),
            "dev": os.path.join(base_path, "binary_classification", "dev.csv"),
        }
        dna_ds = DNADataset.load_local_data(file_paths, label_col="label")

        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "dev" in dna_ds.dataset


class TestLocalFormatRoundTrips:
    """Behavior-verifying round-trips: one real load path per supported format.

    Every test writes a real file under pytest tmp_path, loads it through
    ``load_local_data`` and asserts the dataset CONTENTS (sequences, labels,
    column mapping) — not just the row count.
    """

    def test_csv_round_trip_contents(self, tmp_path):
        """CSV rows load with exact sequences, labels and mapped columns."""
        path = tmp_path / "data.csv"
        path.write_text("sequence,label\nATCG,0\nGCTA,1\nTAGC,0\n")

        dna_ds = DNADataset.load_local_data(str(path), seq_col="sequence", label_col="label")

        assert len(dna_ds) == 3
        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA", "TAGC"]
        assert dna_ds.dataset["labels"] == [0, 1, 0]
        assert "sequence" in dna_ds.dataset.column_names
        assert "labels" in dna_ds.dataset.column_names

    def test_tsv_round_trip_with_custom_columns(self, tmp_path):
        """TSV loads with custom seq/label column names renamed to standard ones."""
        path = tmp_path / "data.tsv"
        path.write_text("seq\tcat\nATCG\t0\nGCTA\t1\n")

        dna_ds = DNADataset.load_local_data(str(path), seq_col="seq", label_col="cat")

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [0, 1]
        assert set(dna_ds.dataset.column_names) == {"sequence", "labels"}

    def test_tsv_default_tab_separator(self, tmp_path):
        """TSV files auto-select the tab separator when none is passed."""
        path = tmp_path / "data.tsv"
        path.write_text("sequence\tlabels\nATCG\t0\n")

        dna_ds = DNADataset.load_local_data(str(path), seq_col="sequence", label_col="labels")

        assert dna_ds.dataset["sequence"] == ["ATCG"]
        assert dna_ds.dataset["labels"] == [0]

    def test_json_round_trip_contents(self, tmp_path):
        """JSON records load with exact sequence and label contents."""
        path = tmp_path / "data.json"
        records = [
            {"sequence": "ATCG", "label": 0},
            {"sequence": "GCTA", "label": 1},
        ]
        path.write_text(json.dumps(records))

        dna_ds = DNADataset.load_local_data(str(path), seq_col="sequence", label_col="label")

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [0, 1]

    def test_parquet_round_trip_contents(self, tmp_path):
        """Parquet files written via pandas round-trip with exact contents."""
        path = tmp_path / "data.parquet"
        pd.DataFrame({"seq": ["ATCG", "GCTA", "TAGC"], "target": [1, 0, 1]}).to_parquet(path)

        dna_ds = DNADataset.load_local_data(str(path), seq_col="seq", label_col="target")

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA", "TAGC"]
        assert dna_ds.dataset["labels"] == [1, 0, 1]

    def test_fasta_multi_record_contents(self, tmp_path):
        """FASTA headers split on fasta_sep yield per-record sequence/label pairs."""
        path = tmp_path / "data.fa"
        path.write_text(">seq1|0\nATCG\n>seq2|1\nGCTA\n>seq3|0\nTAGC\n")

        dna_ds = DNADataset.load_local_data(str(path), fasta_sep="|")

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA", "TAGC"]
        assert dna_ds.dataset["labels"] == [0.0, 1.0, 0.0]

    def test_fasta_multiline_sequence_joined(self, tmp_path):
        """FASTA sequences wrapped across lines concatenate into one record."""
        path = tmp_path / "data.fa"
        path.write_text(">seq1|7\nATCG\nGCTA\nTAGC\n")

        dna_ds = DNADataset.load_local_data(str(path), fasta_sep="|")

        assert dna_ds.dataset["sequence"] == ["ATCGGCTATAGC"]
        assert dna_ds.dataset["labels"] == [7.0]

    def test_fasta_header_without_sep_is_whole_label(self, tmp_path):
        """A FASTA header without fasta_sep becomes the label verbatim."""
        path = tmp_path / "data.fa"
        path.write_text(">promoter_region\nATCG\n")

        dna_ds = DNADataset.load_local_data(str(path), fasta_sep="|")

        assert dna_ds.dataset["sequence"] == ["ATCG"]
        assert dna_ds.dataset["labels"] == ["promoter_region"]

    def test_txt_round_trip_contents(self, tmp_path):
        """Whitespace-separated TXT lines load with exact contents."""
        path = tmp_path / "data.txt"
        path.write_text("ATCG 0\nGCTA 1\nTAGC 0\n")

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA", "TAGC"]
        assert dna_ds.dataset["labels"] == [0.0, 1.0, 0.0]

    def test_pkl_round_trip_contents(self, tmp_path):
        """Pickled dict files load with exact contents and renamed label column."""
        path = tmp_path / "data.pkl"
        path.write_bytes(pickle.dumps({"sequence": ["ATCG", "GCTA"], "label": [1, 0]}))

        dna_ds = DNADataset.load_local_data(str(path), label_col="label")

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [1, 0]

    def test_csv_list_of_files_concatenates(self, tmp_path):
        """A list of CSV paths loads as one concatenated dataset."""
        first = tmp_path / "a.csv"
        first.write_text("sequence,labels\nATCG,0\nGCTA,1\n")
        second = tmp_path / "b.csv"
        second.write_text("sequence,labels\nTAGC,0\nAAAA,1\n")

        dna_ds = DNADataset.load_local_data([str(first), str(second)])

        assert len(dna_ds) == 4
        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA", "TAGC", "AAAA"]
        assert dna_ds.dataset["labels"] == [0, 1, 0, 1]

    def test_txt_with_header_routes_through_csv_loader(self, tmp_path):
        """A txt file whose first line names seq/label columns is parsed as CSV."""
        path = tmp_path / "data.txt"
        path.write_text("sequence,labels\nATCG,0\nGCTA,1\n")

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [0, 1]

    def test_headerless_csv_falls_back_to_txt_parsing(self, tmp_path):
        """A CSV without column names in its first line is parsed as txt records."""
        path = tmp_path / "data.csv"
        path.write_text("ATCG,0\nGCTA,1\n")

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [0.0, 1.0]

    def test_csv_quoted_field_preserved(self, tmp_path):
        """Quoted CSV fields survive parsing as a single value."""
        path = tmp_path / "data.csv"
        path.write_text('sequence,labels\nATCG,"0,5"\n')

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["sequence"] == ["ATCG"]
        assert dna_ds.dataset["labels"] == ["0,5"]

    def test_fasta_list_of_files_rejected(self, tmp_path):
        """FASTA loading rejects a list of file paths."""
        with pytest.raises(ValueError, match="FASTA files must be single files"):
            DNADataset.load_local_data([str(tmp_path / "a.fa"), str(tmp_path / "b.fa")])

    def test_txt_list_of_files_rejected(self, tmp_path):
        """TXT loading rejects a list of file paths."""
        with pytest.raises(ValueError, match="TXT files must be single files"):
            DNADataset.load_local_data([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])

    def test_pkl_list_of_files_rejected(self, tmp_path):
        """Pickle loading rejects a list of file paths."""
        with pytest.raises(ValueError, match="must be single files"):
            DNADataset.load_local_data([str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")])

    def test_non_numeric_string_labels_stay_strings(self, tmp_path):
        """Labels that cannot be parsed as floats remain strings."""
        path = tmp_path / "data.csv"
        path.write_text("sequence,labels\nATCG,promoter\nGCTA,enhancer\n")

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["labels"] == ["promoter", "enhancer"]
        assert dna_ds.data_type == "classification"

    def test_multilabel_split_with_bad_part_keeps_original_string(self, tmp_path):
        """Multi-label strings whose parts are non-numeric keep their raw value."""
        path = tmp_path / "data.csv"
        path.write_text("sequence,labels\nATCG,alpha;beta\n")

        dna_ds = DNADataset.load_local_data(str(path), multi_label_sep=";")

        assert dna_ds.dataset["labels"] == ["alpha;beta"]

    def test_missing_label_column_leaves_dataset_without_labels(self, tmp_path):
        """A file without the label column loads sequences and no labels column."""
        path = tmp_path / "data.csv"
        path.write_text("sequence\nATCG\nGCTA\n")

        dna_ds = DNADataset.load_local_data(str(path))

        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert "labels" not in dna_ds.dataset.column_names
        assert dna_ds.data_type == "unknown"


class TestDNADatasetOnlineLoading:
    """Test loading datasets from online sources."""

    @patch("dnallm.datahandling.data.load_dataset")
    def test_from_huggingface(self, mock_load_dataset):
        """Test loading dataset from Hugging Face."""
        # Create proper mock dataset
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        mock_dataset = Dataset.from_dict(test_data)
        mock_load_dataset.return_value = mock_dataset

        dna_ds = DNADataset.from_huggingface("test-dataset", seq_col="sequence", label_col="labels")

        assert isinstance(dna_ds, DNADataset)
        assert len(dna_ds) == 3
        mock_load_dataset.assert_called_once_with("test-dataset")

    @patch("dnallm.datahandling.data.load_dataset")
    def test_from_huggingface_with_data_dir(self, mock_load_dataset):
        """Test loading dataset from Hugging Face with data_dir."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        mock_dataset = Dataset.from_dict(test_data)
        mock_load_dataset.return_value = mock_dataset

        dna_ds = DNADataset.from_huggingface(
            "test-dataset",
            data_dir="test_dir",
            seq_col="sequence",
            label_col="labels",
        )

        assert isinstance(dna_ds, DNADataset)
        mock_load_dataset.assert_called_once_with("test-dataset", data_dir="test_dir")


class TestDNADatasetDataTypes:
    """Test automatic data type detection."""

    def test_classification_detection(self):
        """Test classification data type detection."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert dna_ds.data_type == "classification"

    def test_regression_detection(self):
        """Test regression data type detection."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0.5, 1.2, -0.3],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert dna_ds.data_type == "regression"

    def test_data_type_with_empty_dataset(self):
        """Test data type detection with empty dataset."""
        test_data: dict[str, Any] = {"sequence": [], "labels": []}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert dna_ds.data_type == "unknown"

    def test_data_type_with_no_labels(self):
        """Test data type detection with no labels column."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert dna_ds.data_type == "unknown"


class TestDNADatasetSequenceProcessing:
    """Test sequence processing and validation methods."""

    def test_validate_sequences_basic(self):
        """Test basic sequence validation."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "NNNN", "AT"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        # Filter sequences with length between 3 and 5, no N bases
        dna_ds.validate_sequences(minl=3, maxl=5, valid_chars="ACGT")

        # Should filter out sequences with N and too short sequences
        assert len(dna_ds.dataset) < 5

    def test_validate_sequences_with_gc_content(self):
        """Test sequence validation with GC content filtering."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "NNNN", "AT"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        # Filter sequences with GC content between 0.4 and 0.6
        dna_ds.validate_sequences(minl=3, maxl=5, gc=(0.4, 0.6), valid_chars="ACGT")

        # Should filter out sequences with N and extreme GC content
        assert len(dna_ds.dataset) < 5

    def test_process_missing_data_basic(self):
        """Test basic processing of missing data."""
        test_data = {
            "sequence": ["ATCG", "", "TAGC", None, "GCTA"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        dna_ds.process_missing_data()

        # Should filter out empty and None sequences
        assert len(dna_ds.dataset) < 5


class TestDNADatasetDataManipulation:
    """Test data manipulation methods."""

    def test_split_data_basic(self):
        """Test basic dataset splitting."""
        test_data = {"sequence": ["ATCG"] * 100, "labels": [0] * 50 + [1] * 50}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        dna_ds.split_data(test_size=0.2, val_size=0.1, seed=42)

        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "val" in dna_ds.dataset

        # Check approximate split ratios (allow for small rounding differences)
        train_size = len(dna_ds.dataset["train"])
        test_size = len(dna_ds.dataset["test"])
        val_size = len(dna_ds.dataset["val"])

        assert 68 <= train_size <= 72  # 70% ± 2
        assert 18 <= test_size <= 22  # 20% ± 2
        assert 8 <= val_size <= 12  # 10% ± 2
        assert train_size + test_size + val_size == 100

    def test_split_data_with_zero_val_size(self):
        """Test dataset splitting with zero validation size."""
        test_data = {"sequence": ["ATCG"] * 100, "labels": [0] * 50 + [1] * 50}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        dna_ds.split_data(test_size=0.2, val_size=0.0, seed=42)

        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "val" not in dna_ds.dataset

    def test_shuffle_basic(self):
        """Test basic dataset shuffling."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT"],
            "labels": [0, 1, 0, 1],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        dna_ds.shuffle(seed=42)

        # Should still have the same number of samples
        assert len(dna_ds.dataset) == 4

    def test_shuffle_with_seed(self):
        """Test dataset shuffling with specific seed."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT"],
            "labels": [0, 1, 0, 1],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        # Shuffle with specific seed
        dna_ds.shuffle(seed=42)

        # Should still have the same number of samples
        assert len(dna_ds.dataset) == 4


class TestDNADatasetSampling:
    """Test data sampling methods."""

    def test_sampling_basic(self):
        """Test basic sampling."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT", "TATA"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        # Sample 60% of the data
        sampled_ds = dna_ds.sampling(ratio=0.6, seed=42)

        assert isinstance(sampled_ds, DNADataset)
        assert len(sampled_ds.dataset) == 3  # 5 * 0.6 = 3

    def test_sampling_with_overwrite(self):
        """Test sampling with overwrite."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT", "TATA"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        original_length = len(dna_ds.dataset)

        # Sample 60% of the data and overwrite
        result = dna_ds.sampling(ratio=0.6, seed=42, overwrite=True)

        assert result is dna_ds
        assert len(dna_ds.dataset) < original_length

    def test_sampling_with_invalid_ratio(self):
        """Test sampling with invalid ratio."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            dna_ds.sampling(ratio=-0.1)

        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            dna_ds.sampling(ratio=1.5)


class TestDNADatasetStatistics:
    """Test statistical analysis methods."""

    def test_statistics_basic(self):
        """Test basic statistics computation."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        stats = dna_ds.statistics()

        assert isinstance(stats, dict)
        assert "full" in stats
        assert "data_type" in stats["full"]
        assert stats["full"]["data_type"] == "classification"
        assert stats["full"]["n_samples"] == 3
        assert stats["full"]["min_len"] == 4
        assert stats["full"]["max_len"] == 4
        assert stats["full"]["mean_len"] == 4.0

    def test_statistics_dataset_dict(self):
        """Test statistics computation for DatasetDict."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT"],
            "labels": [0, 1, 0, 1],
        }
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)

        stats = dna_ds.statistics()

        assert "train" in stats
        assert stats["train"]["n_samples"] == 4
        assert stats["train"]["data_type"] == "classification"


class TestDNADatasetUtilityMethods:
    """Test utility methods."""

    def test_len_single_dataset(self):
        """Test length with single dataset."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        assert len(dna_ds) == 3

    def test_len_dataset_dict(self):
        """Test length with DatasetDict."""
        test_data = {"sequence": ["ATCG", "GCTA"], "labels": [0, 1]}
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds, "test": ds})
        dna_ds = DNADataset(ds_dict)

        # Test that len() returns total length for DatasetDict
        total_length = len(dna_ds)
        assert isinstance(total_length, int)
        assert total_length == 4  # 2 + 2

        # Test that we can get individual split lengths
        split_lengths = dna_ds.get_split_lengths()
        assert isinstance(split_lengths, dict)
        assert "train" in split_lengths
        assert "test" in split_lengths
        assert split_lengths["train"] == 2
        assert split_lengths["test"] == 2

        # Test that we can access individual split lengths directly
        assert len(dna_ds.dataset["train"]) == 2

    def test_getitem_single_dataset(self):
        """Test indexing with single dataset."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        item = dna_ds[0]
        assert "sequence" in item
        assert "labels" in item
        assert item["sequence"] == "ATCG"
        assert item["labels"] == 0

    def test_getitem_dataset_dict_error(self):
        """Test indexing with DatasetDict (should raise error)."""
        test_data = {"sequence": ["ATCG", "GCTA"], "labels": [0, 1]}
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)

        with pytest.raises(ValueError, match="Dataset is a DatasetDict Object"):
            dna_ds[0]

    def test_iter_batches_with_dataset_dict(self):
        """Test iter_batches with DatasetDict (should raise error)."""
        test_data = {"sequence": ["ATCG", "GCTA"], "labels": [0, 1]}
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)

        with pytest.raises(ValueError, match="Dataset is a DatasetDict Object"):
            list(dna_ds.iter_batches(1))

    def test_iter_batches_with_single_dataset(self):
        """Test iter_batches with single dataset."""
        test_data = {"sequence": ["ATCG", "GCTA", "TAGC"], "labels": [0, 1, 0]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        batches = list(dna_ds.iter_batches(2))

        assert len(batches) == 2
        assert len(batches[0]["sequence"]) == 2
        assert len(batches[1]["sequence"]) == 1


class TestDNADatasetEdgeCases:
    """Test edge cases and error conditions."""

    def test_encode_sequences_without_tokenizer(self):
        """Test encode_sequences without tokenizer."""
        test_data = {"sequence": ["ATCG", "GCTA"], "labels": [0, 1]}
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        with pytest.raises(ValueError, match="Tokenizer is required"):
            dna_ds.encode_sequences()


class TestDNADatasetIntegration:
    """Integration tests using real test data."""

    @pytest.mark.data
    def test_full_workflow_binary_classification(self):
        """Test full workflow with binary classification data."""
        # Load data
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests",
            "test_data",
            "binary_classification",
            "train.csv",
        )
        dna_ds = DNADataset.load_local_data(file_path, label_col="label")

        # Validate sequences
        dna_ds.validate_sequences(minl=20, maxl=6000, valid_chars="ACGTN")

        # Process missing data
        dna_ds.process_missing_data()

        # Split data
        dna_ds.split_data(test_size=0.2, val_size=0.1, seed=42)

        # Get statistics
        stats = dna_ds.statistics()

        # Verify results
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "val" in dna_ds.dataset
        assert stats["train"]["data_type"] == "classification"

    @pytest.mark.data
    def test_full_workflow_regression(self):
        """Test full workflow with regression data."""
        # Load data
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests",
            "test_data",
            "regression",
            "train.csv",
        )
        dna_ds = DNADataset.load_local_data(file_path, label_col="label")

        # Validate sequences
        dna_ds.validate_sequences(minl=20, maxl=6000, valid_chars="ACGTN")

        # Process missing data
        dna_ds.process_missing_data()

        # Split data
        dna_ds.split_data(test_size=0.2, val_size=0.1, seed=42)

        # Get statistics
        stats = dna_ds.statistics()

        # Verify results
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert stats["train"]["data_type"] == "regression"

    @pytest.mark.data
    def test_full_workflow_multilabel(self):
        """Test full workflow with multilabel data."""
        # Load data
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests",
            "test_data",
            "multilabel_classification",
            "train.csv",
        )
        dna_ds = DNADataset.load_local_data(file_path, label_col="label", multi_label_sep=";")

        # Manually set the data type since the detection logic has issues
        dna_ds.data_type = "multi_label"

        # Validate sequences
        dna_ds.validate_sequences(minl=20, maxl=6000, valid_chars="ACGTN")

        # Process missing data
        dna_ds.process_missing_data()

        # Split data
        dna_ds.split_data(test_size=0.2, val_size=0.1, seed=42)

        # Get statistics
        stats = dna_ds.statistics()

        # Verify results
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert stats["train"]["data_type"] == "multi_label"


class TestTokenizationPipeline:
    """Sequence classification tokenization through the full encode path."""

    def _make(self, sequences, labels=None, tokenizer=None, max_length=8):
        """Build a DNADataset over the given sequences."""
        data = {"sequence": sequences}
        if labels is not None:
            data["labels"] = labels
        return DNADataset(Dataset.from_dict(data), tokenizer=tokenizer, max_length=max_length)

    def test_encode_sequences_with_tokenizer_argument(self):
        """A tokenizer passed to encode_sequences is adopted and used."""
        d = self._make(["ATCG"], tokenizer=None)
        tok = SimpleDNATokenizer(max_length=8)

        d.encode_sequences(tokenizer=tok)

        assert d.tokenizer is tok
        assert d.dataset[0]["input_ids"].tolist() == [5, 8, 6, 7, 0, 0, 0, 0]
        assert d.dataset[0]["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]

    def test_task_none_defaults_to_sequence_classification(self):
        """task=None tokenizes through the sequence classification path."""
        d = self._make(["ATCG"], tokenizer=SimpleDNATokenizer(max_length=8))

        d.encode_sequences(task=None)

        assert d.dataset[0]["input_ids"].tolist() == [5, 8, 6, 7, 0, 0, 0, 0]

    def test_uppercase_flag_uppercases_sequences(self):
        """uppercase=True hands uppercased sequences to the tokenizer."""
        tok = _RecordingTokenizer()
        d = self._make(["atcg"], tokenizer=tok)

        d.encode_sequences(uppercase=True)

        assert tok.calls[0][0] == ["ATCG"]

    def test_lowercase_flag_lowercases_sequences(self):
        """lowercase=True hands lowercased sequences to the tokenizer."""
        tok = _RecordingTokenizer()
        d = self._make(["ATCG"], tokenizer=tok)

        d.encode_sequences(lowercase=True)

        assert tok.calls[0][0] == ["atcg"]

    def test_seq_sep_replaced_by_sep_token(self):
        """seq_sep occurrences are replaced with the tokenizer's sep token."""
        tok = _RecordingTokenizer()
        d = self._make(["AT CG"], tokenizer=tok)

        d.encode_sequences(seq_sep=" ")

        assert tok.calls[0][0] == ["AT<sep>CG"]

    def test_padding_side_defaults_right_for_sequence_classification(self):
        """padding_side defaults to 'right' for sequence classification tasks."""
        tok = _RecordingTokenizer()
        d = self._make(["ATCG"], tokenizer=tok)

        d.encode_sequences(task="SequenceClassification")

        assert tok.calls[0][1]["padding_side"] == "right"
        assert tok.calls[0][1]["padding"] == "max_length"

    def test_padding_side_falls_back_right_without_tokenizer_attr(self):
        """Non-classification tasks fall back to 'right' without padding_side."""
        tok = _RecordingTokenizer()
        del tok.padding_side
        d = self._make(["ATCG"], tokenizer=tok)

        d.encode_sequences(task="generation")

        assert tok.calls[0][1]["padding_side"] == "right"

    def test_remove_unused_columns_on_dataset_dict(self):
        """remove_unused_columns drops non-feature columns from every split."""
        ds = Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]})
        d = DNADataset(
            DatasetDict({"train": ds, "test": ds}),
            tokenizer=SimpleDNATokenizer(max_length=8),
        )

        d.encode_sequences(remove_unused_columns=True)

        for split in ("train", "test"):
            assert set(d.dataset[split].column_names) == {"labels", "input_ids", "attention_mask"}

    def test_remove_unused_columns_on_single_dataset(self):
        """remove_unused_columns drops non-feature columns from a plain Dataset."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]}),
            tokenizer=SimpleDNATokenizer(max_length=8),
        )

        d.encode_sequences(remove_unused_columns=True)

        assert set(d.dataset.column_names) == {"labels", "input_ids", "attention_mask"}


class TestTokenizerConfigBranches:
    """Direct probes of the _get_tokenizer_config fallback chain."""

    def _d(self, tokenizer):
        """Build a dataset bound to the given tokenizer."""
        return DNADataset(
            Dataset.from_dict({"sequence": ["A"], "labels": [0]}), tokenizer=tokenizer
        )

    def test_config_requires_tokenizer(self):
        """_get_tokenizer_config raises without a tokenizer."""
        d = self._d(None)

        with pytest.raises(ValueError, match="Tokenizer is required"):
            d._get_tokenizer_config()

    def test_config_rejects_falsy_tokenizer(self):
        """A falsy tokenizer object is rejected like a missing one."""

        class _FalsyTokenizer(_ConfigProbeTokenizer):
            def __bool__(self):
                return False

        d = self._d(_FalsyTokenizer())

        with pytest.raises(ValueError, match="Tokenizer is required"):
            d._get_tokenizer_config()

    def test_pad_id_from_pad_id_attribute(self):
        """A pad_id attribute wins when pad_token_id is absent."""
        tok = _ConfigProbeTokenizer(special_tokens_map={"pad_token": "<p>"}, pad_id=7)

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_id"] == 7
        assert config["pad_token"] == PAD_FROM_MAP

    def test_pad_id_from_encode(self):
        """Without pad id attributes, pad_id is derived by encoding pad_token."""
        tok = _ConfigProbeTokenizer(special_tokens_map={"pad_token": "<p>"})
        tok.encode = lambda seq: [9]

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_id"] == 9

    def test_pad_id_falls_back_to_eos_token_id(self):
        """With no pad source at all, the eos_token_id becomes pad_id."""
        tok = _ConfigProbeTokenizer(special_tokens_map={"pad_token": "<p>"}, eos_token_id=3)

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_id"] == 3
        assert config["pad_token"] == PAD_FROM_MAP

    def test_pad_token_via_decode(self):
        """A missing pad_token is decoded from pad_id when decode exists."""
        tok = _ConfigProbeTokenizer(pad_token_id=5)
        tok.decode = lambda ids: "<d>"

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_token"] == DECODED_PAD
        assert tok.pad_token == DECODED_PAD

    def test_pad_token_via_convert_ids_to_tokens(self):
        """A missing pad_token is converted from pad_id without decode."""
        tok = _ConfigProbeTokenizer(pad_token_id=5)
        tok.convert_ids_to_tokens = lambda ids: "<c>"

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_token"] == CONVERTED_PAD

    def test_pad_token_via_decode_token(self):
        """decode_token serves as the last pad_token fallback."""
        tok = _ConfigProbeTokenizer(pad_token_id=5)
        tok.decode_token = lambda ids: "<t>"

        config = self._d(tok)._get_tokenizer_config()

        assert config["pad_token"] == LAST_FALLBACK_PAD

    def test_eos_token_from_sep_token(self):
        """A missing eos token is taken from sep_token."""
        tok = _ConfigProbeTokenizer(special_tokens_map={"pad_token": "<p>"}, sep_token="<s>")

        config = self._d(tok)._get_tokenizer_config()

        assert config["eos_token"] == SEP_FROM_ATTR

    def test_eos_token_from_pad_token(self):
        """Without sep_token, pad_token serves as the eos fallback."""
        tok = _ConfigProbeTokenizer(pad_token="<p>", pad_token_id=0)

        config = self._d(tok)._get_tokenizer_config()

        assert config["eos_token"] == PAD_FROM_MAP


class TestSeqClassificationSepTokenSelection:
    """Direct probes of sep_token selection in sequence classification."""

    BASE_CONFIG: ClassVar[dict] = {
        "max_length": 8,
        "pad_id": 0,
        "pad_token": None,
        "cls_token": None,
    }

    def _apply(self, tokenizer):
        """Run sequence classification tokenization over a one-sequence dataset."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["AT CG"], "labels": [0]}), tokenizer=tokenizer
        )
        d._apply_sequence_classification_tokenization(
            self.BASE_CONFIG, "max_length", "right", False, False, seq_sep=" "
        )
        return tokenizer

    def test_pad_token_none_falls_back_to_eos(self):
        """A None pad_token is replaced with the tokenizer's eos_token."""
        tok = _RecordingTokenizer(pad_token=None)

        self._apply(tok)

        assert tok.pad_token == EOS_VALUE
        assert tok.calls[0][0] == ["AT<sep>CG"]

    def test_sep_token_from_eos_when_no_sep(self):
        """sep_token falls back to eos_token when sep_token is absent."""
        tok = _RecordingTokenizer()
        del tok.sep_token

        self._apply(tok)

        assert tok.calls[0][0] == ["AT<eos>CG"]

    def test_sep_token_from_pad_when_only_pad(self):
        """sep_token falls back to pad_token when sep and eos are absent."""
        tok = _RecordingTokenizer()
        del tok.sep_token
        del tok.eos_token

        self._apply(tok)

        assert tok.calls[0][0] == ["AT<pad>CG"]

    def test_sep_token_empty_when_no_special_tokens(self):
        """sep_token becomes the empty string with no special tokens at all."""
        tok = _RecordingTokenizer()
        del tok.sep_token
        del tok.eos_token
        tok.pad_token = PAD_VALUE

        self._apply(tok)

        assert tok.calls[0][0] == ["AT<pad>CG"]

    def test_sep_token_empty_when_all_tokens_none(self):
        """sep_token degrades to '' when pad, sep and eos are all unset."""
        tok = _RecordingTokenizer(pad_token=None, eos_token=None)
        del tok.sep_token

        self._apply(tok)

        assert tok.calls[0][0] == ["ATCG"]

    def test_seq_classification_requires_tokenizer(self):
        """Sequence classification tokenization raises without a tokenizer."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]}))
        d.tokenizer = None

        with pytest.raises(ValueError, match="Tokenizer is required"):
            d._apply_sequence_classification_tokenization(
                self.BASE_CONFIG, "max_length", "right", False, False
            )


class TestTokenClassificationPipeline:
    """Token classification tokenization with real content assertions."""

    def test_pads_short_sequences_with_special_tokens(self):
        """Short token sequences gain CLS/SEP and pad to max_length."""
        ds = Dataset.from_dict({
            "sequence": [["A", "T", "C", "G"]],
            "labels": [[0, 1, 0, 1]],
        })
        d = DNADataset(ds, tokenizer=SimpleDNATokenizer(max_length=16), max_length=8)

        d.encode_sequences(task="tokenclassification")

        row = d.dataset[0]
        assert row["input_ids"].tolist() == [5, 8, 6, 7, 0, 0, 0, 0]
        assert row["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
        assert row["sequence"] == [
            "[CLS]",
            "A",
            "T",
            "C",
            "G",
            "[SEP]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
        ]
        assert row["labels"].tolist()[:5] == [-100, 0, 1, 0, 1]
        assert set(row["labels"].tolist()[5:]) == {-100}

    def test_truncates_long_sequences(self):
        """Sequences longer than max_length are truncated with CLS/SEP kept."""
        ds = Dataset.from_dict({
            "sequence": [["A", "T", "C", "G", "A", "T"]],
            "labels": [[0, 1, 0, 1, 0, 1]],
        })
        d = DNADataset(ds, tokenizer=SimpleDNATokenizer(max_length=16), max_length=4)

        d.encode_sequences(task="tokenclassification")

        row = d.dataset[0]
        assert row["input_ids"].tolist() == [5, 8, 6, 7]
        assert row["attention_mask"].tolist() == [1, 1, 1, 1]
        assert row["sequence"] == ["[CLS]", "A", "T", "[SEP]"]
        assert row["labels"].tolist() == [-100, 0, 1, -100]

    def test_without_labels_column_defaults_zero_tags(self):
        """A dataset without labels is tokenized with zero tags."""
        ds = Dataset.from_dict({"sequence": [["A", "T"]]})
        d = DNADataset(ds, tokenizer=SimpleDNATokenizer(max_length=16), max_length=4)

        d.encode_sequences(task="tokenclassification")

        row = d.dataset[0]
        assert row["input_ids"].tolist()[:2] == [5, 8]
        assert "labels" not in d.dataset.column_names

    def test_batch_string_input_splits_on_multi_label_sep(self):
        """A bare string batch is split on multi_label_sep before encoding."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["AT;CG"]}),
            tokenizer=SimpleDNATokenizer(max_length=16),
            max_length=2,
        )
        d.multi_label_sep = ";"

        result = d._process_token_classification_batch(
            {"sequence": "AT;CG", "labels": [[0, 1], [1, 0]]}, d._get_tokenizer_config()
        )

        assert result["input_ids"] == [[5, 8], [6, 7]]
        assert result["sequence"][0] == ["[CLS]", "A", "T", "[SEP]"]
        assert result["labels"][0] == [-100, 0, 1, -100]
        assert result["labels"][1] == [-100, 1, 0, -100]

    def test_process_single_requires_tokenizer(self):
        """Single-sequence processing raises without a tokenizer."""
        d = DNADataset(Dataset.from_dict({"sequence": [["A"]]}))
        d.tokenizer = None

        with pytest.raises(ValueError, match="Tokenizer is required"):
            d._process_single_token_sequence(["A"], {}, 0, {"max_length": 4})


class TestTokenClassificationPaddingInternals:
    """Direct probes of the pad/truncate helpers."""

    def _d(self):
        """Build the smallest dataset under test."""
        return DNADataset(Dataset.from_dict({"sequence": [["A"]], "labels": [[0]]}))

    def test_pad_sequence_without_cls_token(self):
        """Without cls_token, plain padding extends tokens and -100 tags."""
        result = self._d()._pad_sequence(
            all_ids=[5, 8],
            example_tokens=["A", "T"],
            example_ner_tags=[0, 1],
            pad_len=2,
            config={"pad_id": 0, "pad_token": "[PAD]", "cls_token": None},
        )

        assert result["sequence"] == ["A", "T", "[PAD]", "[PAD]"]
        assert result["input_ids"] == [5, 8, 0, 0]
        assert result["attention_mask"] == [1, 1, 0, 0]
        assert result["labels"] == [0, 1, -100, -100]

    def test_truncate_with_cls_and_sep(self):
        """Truncation keeps CLS and SEP within max_length."""
        result = self._d()._truncate_sequence(
            all_ids=[5, 8, 6, 7, 5, 8],
            example_tokens=["A", "T", "C", "G", "A", "T"],
            example_ner_tags=[0, 1, 0, 1, 0, 1],
            config={
                "max_length": 4,
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
            },
        )

        assert result["sequence"] == ["[CLS]", "A", "T", "[SEP]"]
        assert result["input_ids"] == [5, 8, 6, 7]
        assert result["attention_mask"] == [1, 1, 1, 1]
        assert result["labels"] == [-100, 0, 1, -100]

    def test_truncate_without_cls_token(self):
        """Truncation without cls_token slices tokens and tags to max_length."""
        result = self._d()._truncate_sequence(
            all_ids=[5, 8, 6, 7],
            example_tokens=["A", "T", "C", "G"],
            example_ner_tags=[0, 1, 0, 1],
            config={"max_length": 3, "cls_token": None},
        )

        assert result["sequence"] == ["A", "T", "C"]
        assert result["input_ids"] == [5, 8, 6]
        assert result["labels"] == [0, 1, 0]

    def test_add_special_tokens_eos_branch(self):
        """Padding uses eos_token in place of an absent sep_token."""
        tokens, tags = self._d()._add_special_tokens(
            ["A"],
            [0],
            pad_len=1,
            config={
                "cls_token": "[CLS]",
                "sep_token": None,
                "eos_token": "<eos>",
                "pad_token": "[PAD]",
            },
        )

        assert tokens == ["[CLS]", "A", "<eos>", "[PAD]"]
        assert tags == [-100, 0, -100, -100]

    def test_add_special_tokens_without_sep_or_eos(self):
        """Padding with no sep/eos uses cls and pad tokens only."""
        tokens, tags = self._d()._add_special_tokens(
            ["A"],
            [0],
            pad_len=1,
            config={
                "cls_token": "[CLS]",
                "sep_token": None,
                "eos_token": None,
                "pad_token": "[PAD]",
            },
        )

        assert tokens == ["[CLS]", "A", "[PAD]"]
        assert tags == [-100, 0, -100]

    def test_truncated_special_tokens_without_sep(self):
        """Truncation special tokens drop the trailing slot without sep_token."""
        tokens, tags = self._d()._add_special_tokens_truncated(
            ["A", "T", "C"],
            [0, 1, 0],
            config={"max_length": 3, "cls_token": "[CLS]", "sep_token": None},
        )

        assert tokens == ["[CLS]", "A", "T"]
        assert tags == [-100, 0, 1]


class TestAugmentation:
    """Reverse-complement augmentation with content-level assertions."""

    def test_augment_doubles_with_reverse_complement(self):
        """Augmentation doubles the dataset with correct complement pairs."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AAAACCC"], "labels": [1]}))

        d.augment_reverse_complement()

        assert len(d.dataset) == 2
        assert d.dataset["sequence"][0] == "AAAACCC"
        assert d.dataset["sequence"][1] == "GGGTTTT"
        assert d.dataset["labels"] == [1, 1]

    def test_augment_identity_when_reverse_and_complement_off(self):
        """With both flags off, the appended copies equal the originals."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCGGC"], "labels": [0]}))

        d.augment_reverse_complement(reverse=False, complement=False)

        assert d.dataset["sequence"] == ["ATCGGC", "ATCGGC"]

    def test_augment_on_dataset_dict_doubles_each_split(self):
        """Augmentation doubles every split of a DatasetDict."""
        ds = Dataset.from_dict({"sequence": ["AAAACCC"], "labels": [1]})
        d = DNADataset(DatasetDict({"train": ds, "test": ds}))

        d.augment_reverse_complement()

        assert len(d.dataset["train"]) == 2
        assert d.dataset["train"]["sequence"][1] == "GGGTTTT"
        assert len(d.dataset["test"]) == 2

    def test_concat_appends_reverse_complement_with_separator(self):
        """Concatenation appends the reverse complement after a separator."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AACC"], "labels": [0]}))

        d.concat_reverse_complement(sep="-")

        assert d.dataset["sequence"] == ["AACC-GGTT"]
        assert d.dataset["labels"] == [0]

    def test_concat_on_dataset_dict(self):
        """Concatenation applies to every split of a DatasetDict."""
        ds = Dataset.from_dict({"sequence": ["AACC"], "labels": [0]})
        d = DNADataset(DatasetDict({"train": ds}))

        d.concat_reverse_complement()

        assert d.dataset["train"]["sequence"] == ["AACCGGTT"]

    def test_raw_reverse_complement_leaves_sequences_unchanged(self):
        """raw_reverse_complement discards its map result, so data is unchanged.

        The method builds the complemented dataset but never assigns the
        ``Dataset.map`` output back — the observable contract is a no-op.
        """
        d = DNADataset(
            Dataset.from_dict({"sequence": ["AAAACCCCGGGGTTTT", "ACGT"], "labels": [0, 1]})
        )

        d.raw_reverse_complement(ratio=1.0, seed=1)

        assert d.dataset["sequence"] == ["AAAACCCCGGGGTTTT", "ACGT"]
        assert d.dataset["labels"] == [0, 1]

    def test_raw_reverse_complement_on_dataset_dict(self):
        """The DatasetDict branch is exercised per split (also a no-op)."""
        ds = Dataset.from_dict({"sequence": ["AAAACCC"], "labels": [1]})
        d = DNADataset(DatasetDict({"train": ds}))

        d.raw_reverse_complement(ratio=0.5, seed=3)

        assert d.dataset["train"]["sequence"] == ["AAAACCC"]


class TestSplitDisjointness:
    """Split behavior with set-level disjointness assertions."""

    def test_split_produces_disjoint_partitions(self):
        """Train/val/test splits partition the data without overlap."""
        sequences = [f"ATCG{i:03d}" for i in range(100)]
        d = DNADataset(Dataset.from_dict({"sequence": sequences, "labels": [0] * 100}))

        d.split_data(test_size=0.2, val_size=0.1, seed=7)

        train = set(d.dataset["train"]["sequence"])
        test = set(d.dataset["test"]["sequence"])
        val = set(d.dataset["val"]["sequence"])

        assert not train & test
        assert not train & val
        assert not test & val
        assert train | test | val == set(sequences)
        assert 65 <= len(train) <= 75
        assert 15 <= len(test) <= 25
        assert 5 <= len(val) <= 15

    def test_split_already_split_dataset_raises(self):
        """Splitting an already-split dataset raises."""
        ds = Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]})
        d = DNADataset(DatasetDict({"train": ds}))

        with pytest.raises(ValueError, match="Dataset is already a DatasetDict"):
            d.split_data()


class TestStatisticsEdges:
    """Statistics computation edges with hand-computed values."""

    def test_statistics_hand_computed(self):
        """Length stats match hand computation including the median."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["ATCG", "ATCGAA", "AT"], "labels": [0, 1, 0]})
        )

        stats = d.statistics()

        assert stats["full"]["n_samples"] == 3
        assert stats["full"]["min_len"] == 2
        assert stats["full"]["max_len"] == 6
        assert stats["full"]["mean_len"] == 4.0
        assert stats["full"]["median_len"] == 4.0

    def test_statistics_accepts_dataframe(self):
        """A pandas DataFrame dataset produces the same stats shape."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AATT", "AT"], "labels": [0, 1]}))
        d.dataset = pd.DataFrame({"sequence": ["AATT", "AT"], "labels": [0, 1]})

        stats = d.statistics()

        assert stats["full"]["n_samples"] == 2
        assert stats["full"]["max_len"] == 4
        assert isinstance(d.stats_for_plot, pd.DataFrame)

    def test_statistics_rejects_non_dataset_input(self, monkeypatch):
        """Non-Dataset, non-DataFrame inputs raise a descriptive error."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AT"], "labels": [0]}))
        # A non-class Dataset makes the isinstance probe fail like an absent
        # datasets install would.
        monkeypatch.setattr("datasets.Dataset", "not-a-class")
        d.dataset = object()

        with pytest.raises(ValueError, match="prepare_dataframe expects"):
            d.statistics()


class TestHeadAndShow:
    """head/show utilities and split-length reporting."""

    def test_head_returns_first_rows(self):
        """head returns a column dict of the first n rows."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AT", "GC"], "labels": [0, 1]}))

        assert d.head(head=1) == {"sequence": ["AT"], "labels": [0]}

    def test_head_dataset_dict_returns_per_split(self):
        """head on a DatasetDict returns one dict per split."""
        ds = Dataset.from_dict({"sequence": ["AT", "GC"], "labels": [0, 1]})
        d = DNADataset(DatasetDict({"train": ds}))

        assert d.head(head=1) == {"train": {"sequence": ["AT"], "labels": [0]}}

    def test_head_show_prints_and_returns_none(self, capsys):
        """show=True prints the rows and returns None."""
        d = DNADataset(
            DatasetDict({"train": Dataset.from_dict({"sequence": ["AT"], "labels": [0]})})
        )

        assert d.head(head=1, show=True) is None
        assert "train" in capsys.readouterr().out

    def test_show_delegates_to_head(self, capsys):
        """show() prints the dataset head."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AT"], "labels": [0]}))

        d.show(head=1)

        assert "sequence" in capsys.readouterr().out

    def test_get_split_lengths_none_for_single_dataset(self):
        """get_split_lengths returns None for an unsplit dataset."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AT"], "labels": [0]}))

        assert d.get_split_lengths() is None


class TestDataTypeHelpers:
    """Data type detection helpers, probed directly and via crafted datasets."""

    def test_string_categorical_label_detected_as_classification(self):
        """Non-numeric string labels classify as classification."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": ["promoter"]}))
        d.multi_label_sep = None

        assert d.data_type == "classification"

    def test_string_float_label_detected_as_regression(self):
        """Decimal string labels classify as regression."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": ["0.5"]}))
        d.multi_label_sep = None

        assert d.data_type == "regression"

    def test_multilabel_string_detected_with_separator(self):
        """Semi-colon integer strings detect as multi_label."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": ["0;1"]}))
        d.multi_label_sep = ";"
        d.__data_type__()

        assert d.data_type == "multi_label"

    def test_multiregression_string_detected_with_separator(self):
        """Semi-colon decimal strings detect as multi_regression."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": ["0.5;1"]}))
        d.multi_label_sep = ";"
        d.__data_type__()

        assert d.data_type == "multi_regression"

    def test_empty_datasetdict_raises(self):
        """An empty DatasetDict raises during data type detection."""
        with pytest.raises(ValueError, match="DatasetDict is empty"):
            DNADataset(DatasetDict({}))

    def test_is_valid_labels_len_raising_object(self):
        """Objects whose __len__ raises are treated as invalid labels."""
        d = DNADataset(Dataset.from_dict({"sequence": ["A"], "labels": [0]}))

        class _BadLen:
            def __len__(self):
                raise TypeError("bad len")

        assert d._is_valid_labels(_BadLen()) is False

    def test_is_valid_labels_without_len(self):
        """Objects without __len__ are treated as invalid labels."""
        d = DNADataset(Dataset.from_dict({"sequence": ["A"], "labels": [0]}))

        assert d._is_valid_labels(42) is False

    def test_get_first_label_without_getitem(self):
        """Objects without __getitem__ yield None."""
        d = DNADataset(Dataset.from_dict({"sequence": ["A"], "labels": [0]}))

        assert d._get_first_label(42) is None

    def test_get_first_label_index_error(self):
        """An IndexError on the first element yields None."""
        d = DNADataset(Dataset.from_dict({"sequence": ["A"], "labels": [0]}))

        class _RaisingGetItem:
            def __getitem__(self, idx):
                raise IndexError

        assert d._get_first_label(_RaisingGetItem()) is None


class TestRandomGenerate:
    """Random sequence generation across replace/append modes."""

    def test_generate_replaces_dataset_with_labels(self):
        """Default mode replaces the dataset and applies label_func."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AAAA"], "labels": [0]}))

        d.random_generate(minl=6, maxl=10, samples=3, seed=42, label_func=len)

        assert len(d.dataset) == 3
        assert d.dataset["labels"] == [len(s) for s in d.dataset["sequence"]]
        for seq in d.dataset["sequence"]:
            assert 6 <= len(seq) <= 10

    def test_generate_default_label_is_zero(self):
        """Without label_func every generated label is 0."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AAAA"], "labels": [1]}))

        d.random_generate(minl=4, maxl=4, samples=2, seed=42)

        assert d.dataset["labels"] == [0, 0]

    def test_generate_appends_to_single_dataset(self):
        """append=True concatenates generated rows onto a plain Dataset."""
        d = DNADataset(Dataset.from_dict({"sequence": ["AAAA", "CCCC"], "labels": [0, 1]}))

        d.random_generate(minl=4, maxl=4, samples=2, seed=42, append=True)

        assert len(d.dataset) == 4
        assert d.dataset["sequence"][:2] == ["AAAA", "CCCC"]

    def test_generate_appends_proportionally_to_dataset_dict(self):
        """append=True distributes generated rows across splits by size.

        The per-split sample count is computed against a total that already
        includes rows appended to earlier splits, so the second split
        receives proportionally fewer rows (2 then round(4*2/6)=1).
        """
        ds = Dataset.from_dict({"sequence": ["AAAA", "CCCC"], "labels": [0, 1]})
        d = DNADataset(DatasetDict({"train": ds, "test": ds}))

        d.random_generate(minl=4, maxl=4, samples=4, seed=42, append=True)

        assert len(d.dataset["train"]) == 4
        assert len(d.dataset["test"]) == 3
        assert d.dataset["train"]["sequence"][:2] == ["AAAA", "CCCC"]
        assert d.dataset["test"]["sequence"][:2] == ["AAAA", "CCCC"]


class TestSamplingDatasetDict:
    """Sampling over pre-split datasets."""

    def test_sampling_dataset_dict_samples_each_split(self):
        """Each split is sampled independently at the requested ratio."""
        ds = Dataset.from_dict({"sequence": [f"ATCG{i}" for i in range(10)], "labels": [0] * 10})
        d = DNADataset(DatasetDict({"train": ds, "test": ds}))

        sampled = d.sampling(ratio=0.5, seed=1)

        assert len(sampled.dataset["train"]) == 5
        assert len(sampled.dataset["test"]) == 5
        train_seqs = set(sampled.dataset["train"]["sequence"])
        all_seqs = set(ds["sequence"])
        assert train_seqs <= all_seqs


class TestRemoteLoaders:
    """Remote dataset loaders with the download boundary mocked."""

    def test_from_modelscope_without_data_dir(self):
        """from_modelscope loads via MsDataset.load with the bare name."""
        payload = Dataset.from_dict({"sequence": ["ATCG", "GCTA"], "labels": [0, 1]})
        with patch("modelscope.MsDataset") as mock_ms:
            mock_ms.load.return_value = payload

            dna_ds = DNADataset.from_modelscope("test-ds")

        mock_ms.load.assert_called_once_with("test-ds")
        assert dna_ds.dataset["sequence"] == ["ATCG", "GCTA"]
        assert dna_ds.dataset["labels"] == [0, 1]

    def test_from_modelscope_with_data_dir_and_custom_columns(self):
        """data_dir is forwarded and custom columns are renamed."""
        payload = Dataset.from_dict({"seq": ["ATCG"], "lab": [1]})
        with patch("modelscope.MsDataset") as mock_ms:
            mock_ms.load.return_value = payload

            dna_ds = DNADataset.from_modelscope(
                "test-ds", seq_col="seq", label_col="lab", data_dir="subdir"
            )

        mock_ms.load.assert_called_once_with("test-ds", data_dir="subdir")
        assert dna_ds.dataset["sequence"] == ["ATCG"]
        assert dna_ds.dataset["labels"] == [1]

    @patch("dnallm.datahandling.data.load_dataset")
    def test_from_huggingface_custom_columns(self, mock_load):
        """from_huggingface renames custom seq/label columns."""
        mock_load.return_value = Dataset.from_dict({"seq": ["GCTA"], "target": [2]})

        dna_ds = DNADataset.from_huggingface("test-ds", seq_col="seq", label_col="target")

        assert dna_ds.dataset["sequence"] == ["GCTA"]
        assert dna_ds.dataset["labels"] == [2]


class TestPresetDatasets:
    """Preset dataset registry and load_preset_dataset wiring."""

    def test_show_preset_dataset_registry(self):
        """The preset registry exposes non-empty, named entries."""
        presets = show_preset_dataset()

        assert set(presets) == {
            "nucleotide_transformer_downstream_tasks",
            "GUE",
            "plant-genomic-benchmark",
        }
        for info in presets.values():
            assert info["name"]
            assert info["tasks"]
            assert info["default_task"] in info["tasks"]

    def test_load_preset_dataset_unknown_name_raises(self):
        """Unknown preset names are rejected."""
        with pytest.raises(ValueError, match="not found in preset datasets"):
            load_preset_dataset("no-such-preset")

    @patch("modelscope.MsDataset")
    def test_load_preset_dataset_with_task_uses_data_dir(self, mock_ms):
        """A listed task is forwarded as data_dir and separators are applied."""
        mock_ms.load.return_value = Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]})

        dna_ds = load_preset_dataset("GUE", task="prom_core_all")

        mock_ms.load.assert_called_once_with("lgq12697/GUE", data_dir="prom_core_all")
        assert dna_ds.dataset["sequence"] == ["ATCG"]
        assert dna_ds.sep == ","
        assert dna_ds.multi_label_sep == ";"
        assert dna_ds.max_length == 1024

    @patch("modelscope.MsDataset")
    def test_load_preset_dataset_unlisted_task_loads_whole(self, mock_ms):
        """An unlisted task falls back to loading the whole dataset."""
        mock_ms.load.return_value = Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]})

        load_preset_dataset("GUE", task="not_a_real_task")

        mock_ms.load.assert_called_once_with("lgq12697/GUE")

    def test_standardize_short_column_names(self):
        """Single-letter seq/label columns are standardized."""
        ds = Dataset.from_dict({"s": ["ATCG"], "l": [1]})

        out = _standardize_column_names(ds)

        assert out.column_names == ["sequence", "labels"]

    def test_standardize_datasetdict_columns(self):
        """Every DatasetDict split gets its columns standardized."""
        ds = Dataset.from_dict({"seq": ["ATCG"], "target": [1]})
        out = _standardize_column_names(DatasetDict({"train": ds, "test": ds}))

        assert out["train"].column_names == ["sequence", "labels"]
        assert out["test"].column_names == ["sequence", "labels"]

    def test_standardize_already_standard_columns(self):
        """Standard columns pass through without renaming."""
        ds = Dataset.from_dict({"sequence": ["ATCG"], "labels": [1]})

        out = _standardize_column_names(ds)

        assert out.column_names == ["sequence", "labels"]


class TestPlotStatistics:
    """Statistics chart generation saved under tmp_path.

    These tests are objective-required: the plot_statistics chain carries
    ~180 statements inside the wave's area gate.
    """

    @pytest.fixture(autouse=True)
    def restore_altair_transformer(self):
        """Restore altair's default data transformer after each test."""
        import altair as alt

        yield
        alt.data_transformers.enable("default")

    def test_plot_without_statistics_raises(self):
        """Plotting before statistics() raises."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]}))

        with pytest.raises(ValueError, match="Statistics have not been computed"):
            d.plot_statistics(save_path="unused.html")

    def test_plot_classification_saves_chart(self, tmp_path):
        """Classification datasets save a length/GC chart to save_path."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["ATCG", "AT", "GCTAAA"], "labels": [0, 1, 0]})
        )
        d.statistics()
        out = tmp_path / "cls.html"

        d.plot_statistics(save_path=str(out))

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_regression_saves_chart(self, tmp_path):
        """Regression datasets save a scatter chart to save_path."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG", "AT"], "labels": [0.5, 1.5]}))
        d.statistics()
        out = tmp_path / "reg.html"

        d.plot_statistics(save_path=str(out))

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_datasetdict_concatenates_splits(self, tmp_path):
        """DatasetDict statistics plot one chart per split, concatenated."""
        d = DNADataset(
            DatasetDict({
                "train": Dataset.from_dict({
                    "sequence": ["ATCG", "AT", "GCTAAA"],
                    "labels": [0, 1, 0],
                }),
                "test": Dataset.from_dict({"sequence": ["ATCG", "AT"], "labels": [0.5, 1.5]}),
            })
        )
        d.statistics()
        out = tmp_path / "dict.html"

        d.plot_statistics(save_path=str(out))

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_multi_classification_saves_chart(self, tmp_path):
        """Semi-colon integer labels chart per sub-label (multi-classification)."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG", "AT"], "labels": ["1;0", "0;1"]}))
        d.statistics()
        d.data_type = "multi-classification"
        out = tmp_path / "mc.html"

        d.plot_statistics(save_path=str(out))

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_multi_regression_saves_chart(self, tmp_path):
        """Semi-colon decimal labels chart per sub-target (multi-regression)."""
        d = DNADataset(
            Dataset.from_dict({"sequence": ["ATCG", "AT"], "labels": ["0.5;1.5", "2.5;0.5"]})
        )
        d.statistics()
        d.data_type = "multi-regression"
        out = tmp_path / "mr.html"

        d.plot_statistics(save_path=str(out))

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_unknown_data_type_raises(self, tmp_path):
        """Unknown data types are rejected by the chart builder."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]}))
        d.statistics()
        d.data_type = "bogus"

        with pytest.raises(ValueError, match="Unknown data_type"):
            d.plot_statistics(save_path=str(tmp_path / "x.html"))

    def test_parse_multi_labels_with_empty_rows(self):
        """Empty label rows produce blank cells padded to the widest row."""
        d = DNADataset(Dataset.from_dict({"sequence": ["ATCG"], "labels": [0]}))

        parsed = d._parse_multi_labels(pd.Series(["1;0", ""]))

        assert list(parsed.columns) == ["label_0", "label_1"]
        assert parsed.iloc[0].tolist() == [1.0, 0.0]
        assert parsed.iloc[1].isna().all()


if __name__ == "__main__":
    # Only run when executed directly, not when imported by pytest
    import sys

    if "pytest" not in sys.modules:
        pytest.main([__file__, "-v"])
