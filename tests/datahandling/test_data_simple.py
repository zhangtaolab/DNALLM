"""Simple test suite for DNADataset class to improve code coverage."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import Mock, patch
from datasets import Dataset, DatasetDict

from dnallm.datahandling.data import DNADataset


class TestDNADatasetSimple:
    """Simple tests for DNADataset class."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        assert dna_ds.dataset is ds
        assert dna_ds.max_length == 512
        assert dna_ds.tokenizer is None
        assert dna_ds.data_type == "classification"
    
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
    
    def test_init_with_dataset_dict(self):
        """Test initialization with DatasetDict."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)
        
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert len(dna_ds.dataset["train"]) == 3
    
    def test_init_with_none_dataset(self):
        """Test initialization with None dataset."""
        with pytest.raises(TypeError, match="Dataset cannot be None"):
            DNADataset(None)
    
    def test_init_with_invalid_max_length(self):
        """Test initialization with invalid max_length."""
        test_data = {
            "sequence": ["ATCG"],
            "labels": [0]
        }
        ds = Dataset.from_dict(test_data)
        
        with pytest.raises(ValueError, match="max_length must be positive"):
            DNADataset(ds, max_length=-1)
        
        with pytest.raises(ValueError, match="max_length must be positive"):
            DNADataset(ds, max_length=0)


class TestDNADatasetLoadLocalDataSimple:
    """Simple tests for loading local data files."""
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sequence,label\n")
            f.write("ATCG,0\n")
            f.write("GCTA,1\n")
            f.write("TAGC,0\n")
            temp_file = f.name
        
        try:
            dna_ds = DNADataset.load_local_data(
                temp_file, 
                seq_col="sequence", 
                label_col="label"
            )
            assert len(dna_ds) == 3
            assert "sequence" in dna_ds.dataset.column_names
            assert "labels" in dna_ds.dataset.column_names
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
    
    def test_load_unsupported_file_type(self):
        """Test loading unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                DNADataset.load_local_data(temp_file)
        finally:
            os.unlink(temp_file)


class TestDNADatasetSequenceProcessingSimple:
    """Simple tests for sequence processing methods."""
    
    def test_validate_sequences_basic(self):
        """Test basic sequence validation."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "NNNN", "AT"],
            "labels": [0, 1, 0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        # Filter sequences with length between 3 and 5, no N bases
        dna_ds.validate_sequences(minl=3, maxl=5, valid_chars="ACGT")
        
        # Should filter out sequences with N and too short sequences
        assert len(dna_ds.dataset) < 5
    
    def test_process_missing_data_basic(self):
        """Test basic processing of missing data."""
        test_data = {
            "sequence": ["ATCG", "", "TAGC", None, "GCTA"],
            "labels": [0, 1, 0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        dna_ds.process_missing_data()
        
        # Should filter out empty and None sequences
        assert len(dna_ds.dataset) < 5


class TestDNADatasetDataManipulationSimple:
    """Simple tests for data manipulation methods."""
    
    def test_split_data_basic(self):
        """Test basic dataset splitting."""
        test_data = {
            "sequence": ["ATCG"] * 100,
            "labels": [0] * 50 + [1] * 50
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        dna_ds.split_data(test_size=0.2, val_size=0.1, seed=42)
        
        assert isinstance(dna_ds.dataset, DatasetDict)
        assert "train" in dna_ds.dataset
        assert "test" in dna_ds.dataset
        assert "val" in dna_ds.dataset
    
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
    
    def test_shuffle_basic(self):
        """Test basic dataset shuffling."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT"],
            "labels": [0, 1, 0, 1]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        original_order = [seq for seq in dna_ds.dataset["sequence"]]
        dna_ds.shuffle(seed=42)
        
        # Should still have the same number of samples
        assert len(dna_ds.dataset) == 4


class TestDNADatasetSamplingSimple:
    """Simple tests for sampling methods."""
    
    def test_sampling_basic(self):
        """Test basic sampling."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC", "CGAT", "TATA"],
            "labels": [0, 1, 0, 1, 0]
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
            "labels": [0, 1, 0, 1, 0]
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
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            dna_ds.sampling(ratio=-0.1)
        
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            dna_ds.sampling(ratio=1.5)


class TestDNADatasetStatisticsSimple:
    """Simple tests for statistical analysis methods."""
    
    def test_statistics_basic(self):
        """Test basic statistics."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        stats = dna_ds.statistics()
        
        assert isinstance(stats, dict)
        assert "full" in stats
        assert "data_type" in stats["full"]
        assert stats["full"]["data_type"] == "classification"
        assert stats["full"]["n_samples"] == 3


class TestDNADatasetUtilityMethodsSimple:
    """Simple tests for utility methods."""
    
    def test_len_single_dataset(self):
        """Test length with single dataset."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        assert len(dna_ds) == 3
    
    def test_len_dataset_dict(self):
        """Test length with DatasetDict."""
        test_data = {
            "sequence": ["ATCG", "GCTA"],
            "labels": [0, 1]
        }
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)
        
        # Test that len() returns total length for DatasetDict
        total_length = len(dna_ds)
        assert isinstance(total_length, int)
        assert total_length == 2
        
        # Test that we can get individual split lengths
        split_lengths = dna_ds.get_split_lengths()
        assert isinstance(split_lengths, dict)
        assert "train" in split_lengths
        assert split_lengths["train"] == 2
        
        # Test that we can access individual split lengths directly
        assert len(dna_ds.dataset["train"]) == 2
    
    def test_getitem_single_dataset(self):
        """Test indexing with single dataset."""
        test_data = {
            "sequence": ["ATCG", "GCTA", "TAGC"],
            "labels": [0, 1, 0]
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)
        
        item = dna_ds[0]
        assert "sequence" in item
        assert "labels" in item
        assert item["sequence"] == "ATCG"
        assert item["labels"] == 0
    
    def test_getitem_dataset_dict_error(self):
        """Test indexing with DatasetDict (should raise error)."""
        test_data = {
            "sequence": ["ATCG", "GCTA"],
            "labels": [0, 1]
        }
        ds = Dataset.from_dict(test_data)
        ds_dict = DatasetDict({"train": ds})
        dna_ds = DNADataset(ds_dict)
        
        with pytest.raises(ValueError, match="Dataset is a DatasetDict Object"):
            dna_ds[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
