from abc import ABC, abstractmethod
from transformers import PreTrainedModel

class BaseDNAModel(ABC):
    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        """Return the underlying transformer model"""
        pass
    
    @abstractmethod
    def preprocess(self, sequences: list[str]) -> dict:
        """Preprocess DNA sequences"""
        pass 