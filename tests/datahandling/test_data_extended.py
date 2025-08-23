"""Extended test suite for DNADataset class to improve code coverage."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from dnallm.datahandling.data import DNADataset, show_preset_dataset, load_preset_dataset


class TestDNADatasetExtendedInitialization:
    """Extended tests for DNADataset initialization."""
    
    def test_init_with_tokenizer(self):
        """Test initialization with tokenizer."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        tokenizer = Mock()
        dna_ds = DNADataset(ds, tokenizer=tokenizer, max_length=256)
        
        assert dna_ds.tokenizer is tokenizer
        assert dna_ds.max_length == 256
    
    def test_init_with_custom_max_length(self):
        """Test initialization with custom max_length."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds, max_length=1024)
        
        assert dna_ds.max_length == 1024


class TestDNADatasetLoadLocalDataExtended:
    """Extended tests for loading local data files."""
    
    def test_load_txt_file(self):
        """Test loading TXT file with custom separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("sequence\tlabel\n")
            f.write("ATCG\t0\n")
            f.write("GCTA\t1\n")
            f.write("TAGC\t0\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label", 
                sep="\t"
            )
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.column_names
            assert "labels" in dna_ds.column_names
        finally:
            os.unlink(temp_file)
    
    def test_load_fasta_file(self):
        """Test loading FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(">seq1|label:0\n")
            f.write("ATCG\n")
            f.write(">seq2|label:1\n")
            f.write("GCTA\n")
            f.write(">seq3|label:0\n")
            f.write("TAGC\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label", 
                fasta_sep="|"
            )
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.column_names
            assert "labels" in dna_ds.column_names
        finally:
            os.unlink(temp_file)
    
    def test_load_pickle_file(self):
        """Test loading pickle file."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(test_data, f)
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)
    
    def test_load_parquet_file(self):
        """Test loading parquet file."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            df.to_parquet(f.name)
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)
    
    def test_load_json_file(self):
        """Test loading JSON file."""
        test_data = [
            {"sequence": "ATCG", "label": 0},
            {"sequence": "GCTA", "label": 1},
            {"sequence": "TAGC", "label": 0}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)
    
    def test_load_file_without_header(self):
        """Test loading file without header."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("ATCG 0\n")
            f.write("GCTA 1\n")
            f.write("TAGC 0\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)
    
    def test_load_file_with_custom_separator(self):
        """Test loading file with custom separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("ATCG,0\n")
            f.write("GCTA,1\n")
            f.write("TAGC,0\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label",
                sep=","
            )
            assert len(dna_ds) == 3
        finally:
            os.unlink(temp_file)
    
    def test_load_file_with_multi_label_sep(self):
        """Test loading file with multi-label separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
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
                multi_label_sep=";"
            )
            assert len(dna_ds) == 3
            # Check that labels are converted to lists
            assert isinstance(dna_ds[0]["labels"], list)
        finally:
            os.unlink(temp_file)
    
    def test_load_file_with_float_labels(self):
        """Test loading file with float labels."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("sequence,label\n")
            f.write("ATCG,0.5\n")
            f.write("GCTA,1.0\n")
            f.write("TAGC,0.0\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
            # Check that labels are converted to floats
            assert isinstance(dna_ds[0]["labels"], float)
        finally:
            os.unlink(temp_file)
    
    def test_load_unsupported_file_type(self):
        """Test loading unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                DNADataset.load_local_data(temp_file)
        finally:
            os.unlink(temp_file)


class TestDNADatasetOnlineLoading:
    """Tests for loading datasets from online sources."""
    
    @patch('dnallm.datahandling.data.load_dataset')
    def test_from_huggingface(self, mock_load_dataset):
        """Test loading dataset from Hugging Face."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        dna_ds = DNADataset.from_huggingface(
            "test-dataset",
            seq_col="seq",
            label_col="lab"
        )
        
        assert isinstance(dna_ds, DNADataset)
        mock_load_dataset.assert_called_once_with("test-dataset")
    
    @patch('dnallm.datahandling.data.MsDataset')
    def test_from_modelscope(self, mock_ms_dataset):
        """Test loading dataset from ModelScope."""
        mock_dataset = Mock()
        mock_ms_dataset.load.return_value = mock_dataset
        
        dna_ds = DNADataset.from_modelscope(
            "test-dataset",
            seq_col="seq",
            label_col="lab"
        )
        
        assert isinstance(dna_ds, DNADataset)
        mock_ms_dataset.load.assert_called_once_with("test-dataset")
    
    @patch('dnallm.datahandling.data.load_dataset')
    def test_from_huggingface_with_data_dir(self, mock_load_dataset):
        """Test loading dataset from Hugging Face with data_dir."""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        dna_ds = DNADataset.from_huggingface(
            "test-dataset",
            data_dir="test_dir",
            seq_col="seq",
            label_col="lab"
        )
        
        assert isinstance(dna_ds, DNADataset)
        mock_load_dataset.assert_called_once_with("test-dataset", data_dir="test_dir")


class TestDNADatasetSequenceProcessingExtended:
    """Extended tests for sequence processing methods."""
    
    def test_validate_sequences_with_gc_content(self):
        """Test sequence validation with GC content filtering."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "NNNN", "AT"],
            "labels": [0, 1, 0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        # Filter sequences with GC content between 0.4 and 0.6
        dna_ds.validate_sequences(minl=3, maxl=5, gc=(0.4, 0.6), valid_chars="ACGT")
        
        # Should filter out sequences with N and extreme GC content
        assert len(dna_ds.dataset) < 5
    
    def test_process_missing_data_with_none_values(self):
        """Test processing of missing data with None values."""
        test_data = {
            "sequence": ["ATCG", "", "TAGC", None, "GCTA"],
            "labels": [0, 1, 0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        dna_ds.process_missing_data()
        
        # Should filter out empty and None sequences
        assert len(dna_ds.dataset) < 5
    
    def test_process_missing_data_with_empty_labels(self):
        """Test processing of missing data with empty labels."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, "", None]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        dna_ds.process_missing_data()
        
        # Should filter out entries with empty or None labels
        assert len(dna_ds.dataset) < 3


class TestDNADatasetDataManipulationExtended:
    """Extended tests for data manipulation methods."""
    
    def test_split_data_with_zero_val_size(self):
        """Test dataset splitting with zero validation size."""
        test_data = {
            "sequence": ["ATCG"] * 100,
            "labels": [0] * 50 + [1] * 50
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        dna_ds.split_data(test_size=0.2, val_size=0.0, seed=42)
        
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "val" not in dna_ds.dataset
    
    def test_shuffle_with_seed(self):
        """Test dataset shuffling with specific seed."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT"],
            "labels": [0, 1, 0, 1]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        # Shuffle with specific seed
        dna_ds.shuffle(seed=42)
        
        # Should still have the same number of samples
        assert len(dna_ds.dataset) == 4


class TestDNADatasetStatisticsExtended:
    """Extended tests for statistical analysis methods."""
    
    def test_statistics_with_empty_dataset(self):
        """Test statistics with empty dataset."""
        test_data = {
            "sequence": [],
            "labels": []
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        stats = dna_ds.statistics()
        
        assert isinstance(stats, dict)
        assert "data_type" in stats
    
    def test_statistics_with_single_sample(self):
        """Test statistics with single sample."""
        test_data = {
            "sequence": ["ATCG"],
            "labels": [0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        stats = dna_ds.statistics()
        
        assert isinstance(stats, dict)
        assert "data_type" in stats
        assert stats["data_type"] == "classification"


class TestDNADatasetUtilityMethodsExtended:
    """Extended tests for utility methods."""
    
    def test_iter_batches_with_dataset_dict(self):
        """Test iter_batches with DatasetDict (should raise error)."""
        test_data = {
            "sequence": ["ATCG", "GCTA"],
            "labels": [0, 1]
        }
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)
        
        with pytest.raises(ValueError, match="Dataset is a DatasetDict Object"):
            list(dna_ds.iter_batches(1))
    
    def test_iter_batches_with_single_dataset(self):
        """Test iter_batches with single dataset."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        batches = list(dna_ds.iter_batches(2))
        
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1


class TestDNADatasetEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_init_with_none_dataset(self):
        """Test initialization with None dataset."""
        with pytest.raises(TypeError):
            DNADataset(None)
    
    def test_init_with_invalid_max_length(self):
        """Test initialization with invalid max_length."""
        test_data = {
            "sequence": ["ATCG"],
            "labels": [0]
        }
        ds = Dataset.from_dict(test_data)
        
        with pytest.raises(ValueError):
            DNADataset(ds, max_length=-1)
    
    def test_encode_sequences_without_tokenizer(self):
        """Test encode_sequences without tokenizer."""
        test_data = {
            "sequence": ["ATCG", "GCTA"],
            "labels": [0, 1]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        with pytest.raises(ValueError, match="Tokenizer is required"):
            dna_ds.encode_sequences()
    
    def test_sampling_with_invalid_ratio(self):
        """Test sampling with invalid ratio."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        with pytest.raises(ValueError):
            dna_ds.sampling(ratio=-0.1)
        
        with pytest.raises(ValueError):
            dna_ds.sampling(ratio=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
