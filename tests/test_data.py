import pytest
from dnallm.finetune.data import DNADataset

def test_dataset_initialization(sample_fasta_file):
    """Test dataset initialization from FASTA file"""
    dataset = DNADataset(sample_fasta_file)
    assert len(dataset) > 0

def test_dataset_getitem(sample_fasta_file):
    """Test dataset item access"""
    dataset = DNADataset(sample_fasta_file)
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "labels" in item

def test_invalid_file_path():
    """Test handling of invalid file path"""
    with pytest.raises(FileNotFoundError):
        DNADataset("nonexistent_file.fasta")

def test_empty_file(temp_dir):
    """Test handling of empty file"""
    empty_file = f"{temp_dir}/empty.fasta"
    with open(empty_file, "w") as f:
        pass
    
    with pytest.raises(ValueError):
        DNADataset(empty_file) 