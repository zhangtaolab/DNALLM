import pytest
from dnallm.finetune.trainer import DNALLMTrainer
from dnallm.finetune.models import get_model
from dnallm.finetune.data import DNADataset

def test_trainer_initialization(training_config, sample_fasta_file):
    """Test trainer initialization"""
    # Load model and data
    model = get_model("plant_dna", "zhangtaolab/plant-dnabert-BPE")
    dataset = DNADataset(sample_fasta_file)
    
    # Initialize trainer
    trainer = DNALLMTrainer(
        model=model,
        config=training_config,
        train_dataset=dataset,
        eval_dataset=dataset
    )
    
    assert trainer.model is not None
    assert trainer.config == training_config
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None

def test_trainer_training(training_config, sample_fasta_file):
    """Test training process"""
    model = get_model("plant_dna", "zhangtaolab/plant-dnabert-BPE")
    dataset = DNADataset(sample_fasta_file)
    
    trainer = DNALLMTrainer(
        model=model,
        config=training_config,
        train_dataset=dataset,
        eval_dataset=dataset
    )
    
    metrics = trainer.train()
    assert isinstance(metrics, dict)
    assert "loss" in metrics

def test_trainer_evaluation(training_config, sample_fasta_file):
    """Test model evaluation"""
    model = get_model("plant_dna", "zhangtaolab/plant-dnabert-BPE")
    dataset = DNADataset(sample_fasta_file)
    
    trainer = DNALLMTrainer(
        model=model,
        config=training_config,
        train_dataset=dataset,
        eval_dataset=dataset
    )
    
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    assert "eval_loss" in metrics 