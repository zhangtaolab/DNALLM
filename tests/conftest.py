import pytest
import tempfile
import os
from dnallm.finetune.config import TrainingConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def sample_dna_sequences():
    """Sample DNA sequences for testing"""
    return [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "TTTTAAAACCCC"
    ]

@pytest.fixture
def training_config(temp_dir):
    """Create a test training configuration"""
    return TrainingConfig(
        output_dir=temp_dir,
        num_epochs=1,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-4
    )

@pytest.fixture
def sample_fasta_file(temp_dir, sample_dna_sequences):
    """Create a temporary FASTA file for testing"""
    fasta_path = os.path.join(temp_dir, "test.fasta")
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sample_dna_sequences):
            f.write(f">seq{i+1}\n{seq}\n")
    return fasta_path 