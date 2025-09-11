"""In Silico Mutagenesis Analysis Module.

This module provides tools for evaluating the impact of sequence mutations on model predictions,
including single nucleotide polymorphisms (SNPs), deletions, insertions, and other sequence variations.
"""

import os
import numpy as np
from scipy.special import softmax, expit
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import Dataset

from ..datahandling.data import DNADataset
from .inference import DNAInference
from .plot import plot_muts

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Mutagenesis:
    """Class for evaluating in silico mutagenesis.

    This class provides methods to analyze how sequence mutations affect model predictions,
    including single base substitutions, deletions, and insertions. It can be used to
    identify important positions in DNA sequences and understand model interpretability.

    Attributes:
        model: Fine-tuned model for prediction
        tokenizer: Tokenizer for the model
        config: Configuration object containing task settings and inference parameters
        sequences: Dictionary containing original and mutated sequences
        dataloader: DataLoader for batch processing of sequences
    """

    def __init__(self, model, tokenizer, config: dict):
        """Initialize Mutagenesis class.

        Args:
            model: Fine-tuned model for making predictions
            tokenizer: Tokenizer for encoding DNA sequences
            config: Configuration object containing task settings and inference parameters
        """

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sequences = None

    def get_inference_engine(self, model, tokenizer) -> DNAInference:
        """Create an inference engine object for the model.

        Args:
            model: The model to be used for inference
            tokenizer: The tokenizer to be used for encoding sequences

        Returns:
            DNAInference: The inference engine object configured with the given model and tokenizer
        """

        inference_engine = DNAInference(
            model=model, tokenizer=tokenizer, config=self.config
        )

        return inference_engine

    def mutate_sequence(
        self,
        sequence,
        batch_size: int = 1,
        replace_mut: bool = True,
        include_n: bool = False,
        delete_size: int = 0,
        fill_gap: bool = False,
        insert_seq: str | None = None,
        lowercase: bool = False,
        do_encode: bool = True,
    ):
        """Generate dataset from sequences with various mutation types.

        This method creates mutated versions of the input sequence including:
        - Single base substitutions (A, C, G, T, optionally N)
        - Deletions of specified size
        - Insertions of specified sequences
        - Case transformations

        Args:
            sequence: Single sequence for mutagenesis
            batch_size: Batch size for DataLoader
            replace_mut: Whether to perform single base substitutions
            include_n: Whether to include N base in substitutions
            delete_size: Size of deletions to create (0 for no deletions)
            fill_gap: Whether to fill deletion gaps with N bases
            insert_seq: Sequence to insert at various positions
            lowercase: Whether to convert sequences to lowercase
            do_encode: Whether to encode sequences for the model

        Returns:
            None (modifies internal state)
        """
        # Get the inference config
        pred_config = self.config["inference"]
        # Define the dataset
        sequences = {"name": ["raw"], "sequence": [sequence]}
        # Create mutated sequences
        if replace_mut:
            if include_n:
                base_map = ["A", "C", "G", "T", "N"]
            else:
                base_map = ["A", "C", "G", "T"]
            # Mutate sequence
            for i, base in enumerate(sequence):
                for mut_base in base_map:
                    if base != mut_base:
                        name = f"mut_{i}_{base}_{mut_base}"
                        mutated_sequence = (
                            sequence[:i] + mut_base + sequence[i + 1 :]
                        )
                        sequences["name"].append(name)
                        sequences["sequence"].append(mutated_sequence)
        # Delete mutations
        if delete_size > 0:
            for i in range(len(sequence) - delete_size + 1):
                name = f"del_{i}_{delete_size}"
                if fill_gap:
                    mutated_sequence = (
                        sequence[:i]
                        + "N" * delete_size
                        + sequence[i + delete_size :]
                    )
                else:
                    mutated_sequence = (
                        sequence[:i] + sequence[i + delete_size :]
                    )
                sequences["name"].append(name)
                sequences["sequence"].append(mutated_sequence)
        # Insert mutations
        if insert_seq is not None:
            for i in range(len(sequence) + 1):
                name = f"ins_{i}_{insert_seq}"
                mutated_sequence = sequence[:i] + insert_seq + sequence[i:]
                sequences["name"].append(name)
                sequences["sequence"].append(mutated_sequence)
        # Lowercase sequences
        if lowercase:
            sequences["sequence"] = [
                seq.lower() for seq in sequences["sequence"]
            ]
        # Create dataset
        if len(sequences["sequence"]) > 0:
            ds = Dataset.from_dict(sequences)
            dataset = DNADataset(
                ds, self.tokenizer, max_length=pred_config.max_length
            )
            self.sequences = sequences
        # Encode sequences
        if do_encode:
            dataset.encode_sequences(remove_unused_columns=True)
        # Create DataLoader
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=pred_config.num_workers
        )

    def pred_comparison(self, raw_pred, mut_pred):
        """Compare raw and mutated predictions.

        This method calculates the difference between predictions on the original sequence
        and mutated sequences, providing insights into mutation effects.

        Args:
            raw_pred: Raw predictions from the original sequence
            mut_pred: Predictions from the mutated sequence

        Returns:
            Tuple containing (raw_score, mut_score, logfc):
            - raw_score: Processed scores from original sequence
            - mut_score: Processed scores from mutated sequence
            - logfc: Log fold change between mutated and original scores

        Raises:
            ValueError: If task type is not supported
        """
        # Get the task config
        task_config = self.config["task"]
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

    def evaluate(self, strategy: str | int = "last") -> list[dict]:
        """Evaluate the impact of mutations on model predictions.

        This method runs predictions on all mutated sequences and compares them
        with the original sequence to calculate mutation effects.

        Args:
            strategy: Strategy for selecting the score from the log fold change
                - "first": Use the first log fold change
                - "last": Use the last log fold change
                - "sum": Use the sum of log fold changes
                - "mean": Use the mean of log fold changes
                - "max": Use the index of the maximum raw score to select the log fold change
                - int: Use the log fold change at the specified index

        Returns:
            Dictionary containing predictions and metadata for all sequences:
            - 'raw': Original sequence predictions and metadata
            - mutation names: Individual mutation results with scores and log fold changes
        """
        # Load predictor
        inference_engine = self.get_inference_engine(
            self.model, self.tokenizer
        )
        # Do prediction
        logits, _, _ = inference_engine.batch_infer(
            self.dataloader, do_pred=False
        )
        logits = logits[0] if isinstance(logits, tuple) else logits
        all_predictions = {}
        # Get the raw predictions
        raw_pred = logits[0].numpy()
        # Get the mutated predictions
        mut_preds = logits[1:].numpy()
        for i, mut_pred in tqdm(
            enumerate(mut_preds), desc="Evaluating mutations"
        ):
            # Get the mutated name
            mut_name = self.sequences["name"][i + 1]
            # Get the mutated sequence
            mut_seq = self.sequences["sequence"][i + 1]
            # Compare the predictions
            raw_score, mut_score, logfc = self.pred_comparison(
                raw_pred, mut_pred
            )
            # Store the results
            if "raw" not in all_predictions:
                all_predictions["raw"] = {
                    "sequence": self.sequences["sequence"][0],
                    "pred": raw_score,
                    "logfc": np.zeros(len(raw_score)),
                    "score": 0.0,
                }
            all_predictions[mut_name] = {
                "sequence": mut_seq,
                "pred": mut_score,
                "logfc": logfc,
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
            all_predictions[mut_name]["score"] = score

        return all_predictions

    def plot(
        self,
        preds: dict,
        show_score: bool = False,
        save_path: str | None = None,
    ) -> None:
        """Plot the mutagenesis analysis results.

        This method generates visualizations of mutation effects, typically as heatmaps,
        bar charts and line plots showing how different mutations affect model predictions
        at various positions.

        Args:
            preds: Dictionary containing model predicted scores and metadata
            show_score: Whether to show the score values on the plot
            save_path: Path to save the plot. If None, plot will be shown interactively

        Returns:
            Plot object
        """
        if save_path:
            suffix = os.path.splitext(save_path)[-1]
            if suffix:
                outfile = save_path
            else:
                outfile = os.path.join(save_path, ".pdf")
        else:
            outfile = None
        # Plot heatmap
        pmut = plot_muts(preds, show_score=show_score, save_path=outfile)
        return pmut
