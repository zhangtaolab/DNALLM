from dnallm.inference.predictor import DNAPredictor
from dnallm.inference.config import InferenceConfig

def main():
    # Initialize configuration
    config = InferenceConfig(
        model_path="outputs/plant_dna/checkpoint-final",
        batch_size=32,
        device="cuda",
        output_dir="outputs/predictions"
    )
    
    # Initialize predictor
    predictor = DNAPredictor(
        model_type="plant_dna",
        model_path=config.model_path,
        config=config
    )
    
    # Example sequences
    sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "TTTTAAAACCCC"
    ]
    
    # Get predictions
    predictions = predictor.predict(sequences, save_to_file=True)
    print(f"Predictions shape: {predictions['logits'].shape}")

if __name__ == "__main__":
    main() 