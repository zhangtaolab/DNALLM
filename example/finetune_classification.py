from dnallm.finetune.trainer import DNALLMTrainer
from dnallm.finetune.config import TrainingConfig
from dnallm.finetune.tasks import TaskType, TaskConfig
from dnallm.finetune.models import get_model
from dnallm.finetune.data import DNADataset

def main():
    # Task configuration for binary classification
    task_config = TaskConfig(
        task_type=TaskType.BINARY,
        num_labels=2,
        label_names=["negative", "positive"]
    )
    
    # Training configuration
    config = TrainingConfig(
        output_dir="outputs/classification",
        num_epochs=3,
        batch_size=32
    )
    
    # Load model and data
    model = get_model("plant_dna", "zhangtaolab/plant-dnabert-BPE")
    train_dataset = DNADataset(
        "data/train.csv",
        task_config=task_config,
        label_column="label"
    )
    eval_dataset = DNADataset(
        "data/eval.csv",
        task_config=task_config,
        label_column="label"
    )
    
    # Initialize trainer
    trainer = DNALLMTrainer(
        model=model,
        config=config,
        task_config=task_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train model
    metrics = trainer.train()
    print(f"Training metrics: {metrics}")

if __name__ == "__main__":
    main() 