from typing import List, Dict, Iterator
from pathlib import Path
import torch
import json

def batch_sequences(sequences: List[str], batch_size: int) -> Iterator[List[str]]:
    """Yield batches of sequences"""
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i + batch_size]

def save_predictions(predictions: Dict[str, torch.Tensor], output_dir: Path) -> None:
    """Save predictions to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    json_predictions = {
        k: v.cpu().tolist() for k, v in predictions.items()
    }
    
    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(json_predictions, f) 