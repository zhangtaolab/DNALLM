from typing import Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

def load_model_and_tokenizer(model_name: str, model_type: str = "plant_dna") -> tuple:
    """
    Load model and tokenizer from either HuggingFace or ModelScope
    
    Args:
        model_name: Model name or path
        model_type: Type of model ("plant_dna", "dnabert", etc.)
        
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # Try HuggingFace first
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        # Fall back to ModelScope
        try:
            model_dir = snapshot_download(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        except Exception as ms_e:
            raise ValueError(f"Failed to load model from both HuggingFace and ModelScope: {e}, {ms_e}")
            
    return model, tokenizer 