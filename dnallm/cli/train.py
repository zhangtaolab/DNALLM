import click
from ..finetune.config import TrainingConfig
from ..finetune.trainer import DNALLMTrainer
from ..finetune.models import get_model
from ..finetune.data import DNADataset

@click.command()
@click.option("--model-type", type=str, required=True, 
              help="Model type: dnabert/plant_dna/nucleotide")
@click.option("--model-name", type=str, required=True,
              help="Specific model name or path")
@click.option("--train-file", type=str, required=True,
              help="Training data file path")
@click.option("--eval-file", type=str, required=True,
              help="Evaluation data file path")
@click.option("--output-dir", type=str, required=True,
              help="Output directory for model and logs")
@click.option("--num-epochs", type=int, default=3,
              help="Number of training epochs")
@click.option("--batch-size", type=int, default=32,
              help="Training batch size")
@click.option("--learning-rate", type=float, default=5e-5,
              help="Learning rate")
def main(model_type: str, model_name: str, train_file: str, eval_file: str,
         output_dir: str, num_epochs: int, batch_size: int, learning_rate: float):
    """Fine-tune a DNA Language Model"""
    
    # Create config
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Initialize model
    model = get_model(model_type, model_name)
    
    # Load datasets
    train_dataset = DNADataset(train_file)
    eval_dataset = DNADataset(eval_file)
    
    # Initialize trainer
    trainer = DNALLMTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train model
    metrics = trainer.train()
    click.echo(f"Training completed. Metrics: {metrics}")

if __name__ == "__main__":
    main() 