from dnallm.finetune.trainer import DNALLMTrainer
from dnallm.finetune.config import TrainingConfig
from dnallm.finetune.models import get_model
from dnallm.finetune.data import DNADataset

def main():
    # Initialize configuration
    config = TrainingConfig(
        output_dir="outputs/dnabert",
        num_epochs=3,
        batch_size=32,
        learning_rate=5e-5
    )
    
    # Load model
    model = get_model("dnabert", "dnabert-v2-117m")
    
    # Load datasets
    train_dataset = DNADataset("example/data/train.fasta")
    eval_dataset = DNADataset("example/data/eval.fasta")
    
    # Initialize trainer
    trainer = DNALLMTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train model
    metrics = trainer.train()
    print(f"Training completed. Metrics: {metrics}")

if __name__ == "__main__":
    main() 