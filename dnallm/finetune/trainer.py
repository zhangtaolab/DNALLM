import os
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from .models.base import BaseDNAModel
from .data import DNADataset
from .config import TrainingConfig

class DNALLMTrainer:
    """DNA Language Model Trainer class that supports multiple model types"""
    
    def __init__(
        self,
        model: BaseDNAModel,
        config: TrainingConfig,
        train_dataset: Optional[DNADataset] = None,
        eval_dataset: Optional[DNADataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model.get_model(),
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
    
    def train(self) -> Dict[str, float]:
        """Train the model and return metrics"""
        train_result = self.trainer.train()
        metrics = train_result.metrics
        
        # Save the model
        self.trainer.save_model()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model and return metrics"""
        return self.trainer.evaluate()
    
    def compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        # Implement your metric computation here
        metrics = {
            "accuracy": 0.0,  # Replace with actual metric computation
        }
        return metrics 