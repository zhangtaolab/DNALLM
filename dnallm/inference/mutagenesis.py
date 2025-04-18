import os
import warnings
import json
from typing import Optional, List, Dict, Union
from pathlib import Path
import numpy as np
from scipy.special import softmax, expit
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import Dataset

from ..models import *
from ..datasets.data import DNADataset
from ..tasks.metrics import compute_metrics as Metrics
from .predictor import DNAPredictor
from .plot import plot_muts

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class Mutagenesis:
    """
    Class for evaluating in silico mutagenesis.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: dict
    ):
        """
        Initialize Mutagenesis class.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer for the model
            config: Configuration object containing task settings and inference parameters
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sequences = None

    def get_predictor(self, model, tokenizer) -> DNAPredictor:
        """
        Create a predictor object for the model.
        Args:
            model: The model to be used for prediction.
            tokenizer: The tokenizer to be used for encoding sequences.
        Returns:
            DNAPredictor: The predictor object.
        """
        
        predictor = DNAPredictor(
            model=model,
            tokenizer=tokenizer,
            config=self.config
        )
        return predictor

    def mutate_sequence(self, sequence, batch_size: int=1,
                        replace_mut: bool=True, include_n: bool=False,
                        delete_size: int=0, fill_gap: bool=False,
                        insert_seq: str=None, lowercase: bool=False,
                        do_encode: bool=True):
        """
        Generate dataset from sequences
        Args:
            sequence: Single sequence for mutagenesis
            batch_size: Batch size for DataLoader
            do_encode: Whether to encode sequences
        Returns:
            Dataset object
            DataLoader object
        """
        # Get the inference config
        pred_config = self.config['inference']
        # Define the dataset
        sequences = {'name': ['raw'], 'sequence': [sequence]}
        # Create mutated sequences
        if replace_mut:
            if include_n:
                base_map = ['A', 'C', 'G', 'T', 'N']
            else:
                base_map = ['A', 'C', 'G', 'T']
            # Mutate sequence
            for i, base in enumerate(sequence):
                for mut_base in base_map:
                    if base != mut_base:
                        name = f"mut_{i}_{base}_{mut_base}"
                        mutated_sequence = sequence[:i] + mut_base + sequence[i+1:]
                        sequences['name'].append(name)
                        sequences['sequence'].append(mutated_sequence)
        # Delete mutations
        if delete_size > 0:
            for i in range(len(sequence)-delete_size+1):
                name = f"del_{i}_{delete_size}"
                if fill_gap:
                    mutated_sequence = sequence[:i] + "N" * delete_size + sequence[i+delete_size:]
                else:
                    mutated_sequence = sequence[:i] + sequence[i+delete_size:]
                sequences['name'].append(name)
                sequences['sequence'].append(mutated_sequence)
        # Insert mutations
        if insert_seq is not None:
            for i in range(len(sequence)+1):
                name = f"ins_{i}_{insert_seq}"
                mutated_sequence = sequence[:i] + insert_seq + sequence[i:]
                sequences['name'].append(name)
                sequences['sequence'].append(mutated_sequence)
        # Lowercase sequences
        if lowercase:
            sequences['sequence'] = [seq.lower() for seq in sequences['sequence']]
        # Create dataset
        if len(sequences['sequence']) > 0:
            ds = Dataset.from_dict(sequences)
            dataset = DNADataset(ds, self.tokenizer, max_length=pred_config.max_length)
            self.sequences = sequences
        # Encode sequences
        if do_encode:
            dataset.encode_sequences(remove_unused_columns=True)
        # Create DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=pred_config.num_workers
        )

    def pred_comparison(self, raw_pred, mut_pred):
        """
        Compare raw and mutated predictions.
        
        Args:
            raw_pred: Raw predictions
            mut_pred: Mutated predictions
            mut_name: Name of the mutated sequence
        Returns:
            Comparison results
        """
        # Get the task config
        task_config = self.config['task']
        # Get the predictions
        if task_config.task_type == "binary":
            raw_score = expit(raw_pred)
            mut_score = expit(mut_pred)
        elif task_config.task_type == "multiclass":
            raw_score = softmax(raw_pred)
            mut_score = softmax(mut_pred)
        elif task_config.task_type == "multilabel":
            raw_score = expit(raw_pred)
            mut_score = expit(mut_pred)
        elif task_config.task_type == "regression":
            raw_score = raw_pred
            mut_score = mut_pred
        elif task_config.task_type == "token":
            raw_score = np.argmax(raw_pred, dim=-1)
            mut_score = np.argmax(mut_pred, dim=-1)
        else:
            raise ValueError(f"Unknown task type: {task_config.task_type}")
        
        logfc = np.log2(mut_score / raw_score)
        
        return raw_score, mut_score, logfc
        

    def evaluate(self, strategy: Union[str, int]="last") -> List[Dict]:
        """
        Predict using the model.
        
        Args:
            strategy: Strategy for selecting the score from the log fold change
                - "first": Use the first log fold change
                - "last": Use the last log fold change
                - "sum": Use the sum of log fold changes
                - "mean": Use the mean of log fold changes
                - "max": Use the index of the maximum raw score to select the log fold change
                - int: Use the log fold change at the specified index
            
        Returns:
            Dictionaries containing predictions and metadata
        """
        # Load predictor
        predictor = self.get_predictor(self.model, self.tokenizer)
        # Do prediction
        logits, _, _ = predictor.batch_predict(self.dataloader, do_pred=False)
        logits = logits[0] if isinstance(logits, tuple) else logits
        all_predictions = {}
        # Get the raw predictions
        raw_pred = logits[0].numpy()
        # Get the mutated predictions
        mut_preds = logits[1:].numpy()
        for i, mut_pred in tqdm(enumerate(mut_preds), desc="Evaluating mutations"):
            # Get the mutated name
            mut_name = self.sequences['name'][i+1]
            # Get the mutated sequence
            mut_seq = self.sequences['sequence'][i+1]
            # Compare the predictions
            raw_score, mut_score, logfc = self.pred_comparison(raw_pred, mut_pred)
            # Store the results
            if 'raw' not in all_predictions:
                all_predictions['raw'] = {
                    'sequence': self.sequences['sequence'][0],
                    'pred': raw_score,
                    'logfc': np.zeros(len(raw_score)),
                    'score': 0.0
                }
            all_predictions[mut_name] = {
                'sequence': mut_seq,
                'pred': mut_score,
                'logfc': logfc,
            }
            # Get final score
            if strategy == "first":
                score = logfc[0]
            elif strategy == "last":
                score = logfc[-1]
            elif strategy == "sum":
                score = np.sum(logfc)
            elif strategy == "mean":
                score = np.mean(logfc)
            elif strategy == "max":
                idx = raw_score.index(max(raw_score))
                score = logfc[idx]
            elif isinstance(strategy, int):
                score = logfc[strategy]
            all_predictions[mut_name]['score'] = score
        
        return all_predictions


    def plot(self, preds: dict,
             show_score: bool = False,
             save_path: Optional[str] = None) -> None:
        """
        Plot the benchmark results.
        Args:
            preds (dict): Dictionary containing model predicted scores.
            show_score (bool): Whether to show the score on the plot.
            save_path (Optional[str]): Path to save the plot.
        Returns:
            None
        """
        if save_path:
            suffix = os.path.splitext(save_path)[-1]
            if suffix:
                heatmap = save_path.replace(suffix, "_heatmap" + suffix)
            else:
                heatmap = os.path.join(save_path, "heatmap.pdf")
        else:
            heatmap = None
        # Plot heatmap
        pheat = plot_muts(preds, show_score=show_score,
                          save_path=heatmap)
        return pheat
            
