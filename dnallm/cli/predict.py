import click
from ..inference.predictor import DNAPredictor
from ..inference.config import InferenceConfig

@click.command()
@click.option("--model-type", type=str, required=True,
              help="Model type: dnabert/plant_dna/nucleotide")
@click.option("--model-path", type=str, required=True,
              help="Path to fine-tuned model")
@click.option("--input-file", type=str, required=True,
              help="Input FASTA file path")
@click.option("--output-dir", type=str, required=True,
              help="Output directory for predictions")
@click.option("--batch-size", type=int, default=32,
              help="Batch size for inference")
@click.option("--device", type=str, default="cuda",
              help="Device to run inference on (cuda/cpu)")
@click.option("--use-fp16", is_flag=True,
              help="Use half precision for inference")
def main(model_type: str, model_path: str, input_file: str, output_dir: str,
         batch_size: int, device: str, use_fp16: bool):
    """Run inference with fine-tuned DNA Language Model"""
    
    # Create config
    config = InferenceConfig(
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        use_fp16=use_fp16,
        output_dir=output_dir
    )
    
    # Initialize predictor
    predictor = DNAPredictor(model_type, model_path, config)
    
    # Load sequences
    with open(input_file) as f:
        sequences = [line.strip() for line in f if not line.startswith(">")]
    
    # Run prediction
    predictions = predictor.predict(sequences, save_to_file=True)
    click.echo(f"Predictions saved to {output_dir}") 