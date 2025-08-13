"""DNA Language Model Inference Module.

This module implements core model inference functionality, including:

1. DNAPredictor class
   - Model loading and initialization
   - Batch sequence prediction
   - Result post-processing
   - Device management
   - Half-precision inference support

2. Core features:
   - Model state management
   - Batch prediction
   - Result merging
   - Prediction result saving
   - Memory optimization

3. Inference optimization:
   - Batch parallelization
   - GPU acceleration
   - Half-precision computation
   - Memory efficiency optimization

Example:
    ```python
    predictor = DNAPredictor(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    results = predictor.predict(sequences)
    ```
"""

import os
import warnings
import json
from typing import Optional, List, Dict, Union, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from ..datasets.data import DNADataset
from ..tasks.metrics import compute_metrics as Metrics
from .plot import *

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class DNAPredictor:
    """DNA sequence predictor using fine-tuned models.

    This class provides comprehensive functionality for making predictions using DNA language models.
    It handles model loading, inference, result processing, and various output formats including
    hidden states and attention weights for model interpretability.

    Attributes:
        model: Fine-tuned model instance for inference
        tokenizer: Tokenizer for encoding DNA sequences
        task_config: Configuration object containing task settings
        pred_config: Configuration object containing inference parameters
        device: Device (CPU/GPU/MPS) for model inference
        sequences: List of input sequences
        labels: List of true labels (if available)
        embeddings: Dictionary containing hidden states and attention weights
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict
    ):
        """Initialize the predictor.

        Args:
            model: Fine-tuned model instance for inference
            tokenizer: Tokenizer for encoding DNA sequences
            config: Configuration dictionary containing task settings and inference parameters
        """

        self.model = model
        self.tokenizer = tokenizer
        self.task_config = config['task']
        self.pred_config = config['inference']
        self.device = self._get_device()
        if model:
            self.model.to(self.device)
            print(f"Use device: {self.device}")
        self.sequences = []
        self.labels = []

    def _get_device(self) -> torch.device:
        """Get the appropriate device for model inference.

        This method automatically detects and selects the best available device for inference,
        supporting CPU, CUDA (NVIDIA), MPS (Apple Silicon), ROCm (AMD), TPU, and XPU (Intel).

        Returns:
            torch.device: The device to use for model inference

        Raises:
            ValueError: If the specified device type is not supported
        """
        # Get the device type
        device = self.pred_config.device.lower()
        if device == 'cpu':
            return torch.device('cpu')
        elif device in ['cuda', 'nvidia']:
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('cuda')
        elif device in ['mps', 'apple', 'mac']:
            if not torch.backends.mps.is_available():
                warnings.warn("MPS is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('mps')
        elif device in ['rocm', 'amd']:
            if not torch.cuda.is_available():
                warnings.warn("ROCm is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('cuda')
        elif device == ['tpu', 'xla', 'google']:
            try:
                import torch_xla.core.xla_model as xm
                return torch.device('xla')
            except:
                warnings.warn("TPU is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
        elif device == ['xpu', 'intel']:
            if not torch.xpu.is_available():
                warnings.warn("XPU is not available. Please check your installation. Use CPU instead.")
                return torch.device('cpu')
            else:
                return torch.device('xpu')
        elif device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.xpu.is_available():
                return torch.device('xpu')
            else:
                return torch.device('cpu')
        else:
            raise ValueError(f"Unsupported device type: {device}")

    def generate_dataset(self, seq_or_path: Union[str, List[str]], batch_size: int = 1,
                         seq_col: str = "sequence", label_col: str = "labels",
                         sep: str = None, fasta_sep: str = "|",
                         multi_label_sep: Union[str, None] = None,
                         uppercase: bool = False, lowercase: bool = False,
                         keep_seqs: bool = True, do_encode: bool = True) -> Tuple[DNADataset, DataLoader]:
        """Generate dataset from sequences or file path.

        This method creates a DNADataset and DataLoader from either a list of sequences
        or a file path, supporting various file formats and preprocessing options.

        Args:
            seq_or_path: Single sequence, list of sequences, or path to a file containing sequences
            batch_size: Batch size for DataLoader
            seq_col: Column name for sequences in the file
            label_col: Column name for labels in the file
            sep: Delimiter for CSV, TSV, or TXT files
            fasta_sep: Delimiter for FASTA files
            multi_label_sep: Delimiter for multi-label sequences
            uppercase: Whether to convert sequences to uppercase
            lowercase: Whether to convert sequences to lowercase
            keep_seqs: Whether to keep sequences in the dataset for later use
            do_encode: Whether to encode sequences for the model

        Returns:
            Tuple containing:
                - DNADataset: Dataset object with sequences and labels
                - DataLoader: DataLoader object for batch processing

        Raises:
            ValueError: If input is neither a file path nor a list of sequences
        """
        if isinstance(seq_or_path, str):
            suffix = seq_or_path.split(".")[-1]
            if suffix and os.path.isfile(seq_or_path):
                sequences = []
                dataset = DNADataset.load_local_data(seq_or_path, seq_col=seq_col, label_col=label_col,
                                                     sep=sep, fasta_sep=fasta_sep, multi_label_sep=multi_label_sep,
                                                     tokenizer=self.tokenizer, max_length=self.pred_config.max_length)
            else:
                sequences = [seq_or_path]
        elif isinstance(seq_or_path, list):
            sequences = seq_or_path
        else:
            raise ValueError("Input should be a file path or a list of sequences.")
        if len(sequences) > 0:
            ds = Dataset.from_dict({"sequence": sequences})
            dataset = DNADataset(ds, self.tokenizer, max_length=self.pred_config.max_length)
        # If labels are provided, keep labels
        if keep_seqs:
            self.sequences = dataset.dataset["sequence"]
        # Encode sequences
        if do_encode:
            task_type = self.task_config.task_type
            dataset.encode_sequences(remove_unused_columns=True, task=task_type, uppercase=uppercase, lowercase=lowercase)
        if "labels" in dataset.dataset.features:
            self.labels = dataset.dataset["labels"]
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.pred_config.num_workers
        )

        return dataset, dataloader

    def logits_to_preds(self, logits: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """Convert model logits to predictions and human-readable labels.

        This method processes raw model outputs based on the task type to generate
        appropriate predictions and convert them to human-readable labels.

        Args:
            logits: Model output logits tensor

        Returns:
            Tuple containing:
                - torch.Tensor: Model predictions (probabilities or raw values)
                - List: Human-readable labels corresponding to predictions

        Raises:
            ValueError: If task type is not supported
        """
        # Get task type and threshold from config
        task_type = self.task_config.task_type
        threshold = self.task_config.threshold
        label_names = self.task_config.label_names
        # Convert logits to predictions based on task type
        if task_type == "binary":
            probs = torch.softmax(logits, dim=-1)
            preds = (probs[:, 1] > threshold).long()
            labels = [label_names[pred] for pred in preds]
        elif task_type == "multiclass":
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            labels = [label_names[pred] for pred in preds]
        elif task_type == "multilabel":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            labels = []
            for pred in preds:
                label = [label_names[i] for i in range(len(pred)) if pred[i] == 1]
                labels.append(label)
        elif task_type == "regression":
            preds = logits.squeeze(-1)
            probs = preds
            labels = preds.tolist()
        elif task_type == "token":
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            labels = []
            for pred in preds:
                label = [label_names[pred[i]] for i in range(len(pred))]
                labels.append(label)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        return probs, labels

    def format_output(self, predictions: Tuple[torch.Tensor, List]) -> Dict:
        """Format output predictions into a structured dictionary.

        This method converts raw predictions into a user-friendly format with
        sequences, labels, and confidence scores.

        Args:
            predictions: Tuple containing (probabilities, labels)

        Returns:
            Dictionary containing formatted predictions with structure:
            {index: {'sequence': str, 'label': str/list, 'scores': dict/list}}
        """
        # Get task type from config
        task_type = self.task_config.task_type
        formatted_predictions = {}
        probs, labels = predictions
        probs = probs.numpy().tolist()
        keep_seqs = True if len(self.sequences) else False
        label_names = self.task_config.label_names
        for i, label in enumerate(labels):
            prob = probs[i]
            formatted_predictions[i] = {
                'sequence': self.sequences[i] if keep_seqs else '',
                'label': label,
                'scores': {label_names[j]: p for j, p in enumerate(prob)} if task_type != "token"
                          else [max(x) for x in prob],
            }
        return formatted_predictions

    @torch.no_grad()
    def batch_predict(self, dataloader: DataLoader, do_pred: bool = True,
                      output_hidden_states: bool = False,
                      output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Dict], Dict]:
        """Perform batch prediction on sequences.

        This method runs inference on batches of sequences and optionally extracts
        hidden states and attention weights for model interpretability.

        Args:
            dataloader: DataLoader object containing sequences for inference
            do_pred: Whether to convert logits to predictions
            output_hidden_states: Whether to output hidden states from all layers
            output_attentions: Whether to output attention weights from all layers

        Returns:
            Tuple containing:
                - torch.Tensor: All logits from the model
                - Optional[Dict]: Predictions dictionary if do_pred=True, otherwise None
                - Dict: Embeddings dictionary containing hidden states and/or attention weights

        Note:
            Setting output_hidden_states or output_attentions to True will consume
            significant memory, especially for long sequences or large models.
        """
        # Set model to evaluation mode
        self.model.eval()
        all_logits = []
        # Whether or not to output hidden states
        params = None
        embeddings = {}
        if output_hidden_states:
            import inspect
            sig = inspect.signature(self.model.forward)
            params = sig.parameters
            if 'output_hidden_states' in params:
                self.model.config.output_hidden_states = True
            embeddings['hidden_states'] = None
            embeddings['attention_mask'] = []
            embeddings['labels'] = []
        if output_attentions:
            if not params:
                import inspect
                sig = inspect.signature(self.model.forward)
                params = sig.parameters
            if 'output_attentions' in params:
                self.model.config.output_attentions = True
            embeddings['attentions'] = None
        # Iterate over batches
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            if self.pred_config.use_fp16:
                self.model = self.model.half()
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            # Get logits
            logits = outputs.logits.cpu().detach()
            all_logits.append(logits)
            # Get hidden states
            if output_hidden_states:
                hidden_states = [h.cpu().detach() for h in outputs.hidden_states] if hasattr(outputs, 'hidden_states') else None
                if embeddings['hidden_states'] is None:
                    embeddings['hidden_states'] = [[h] for h in hidden_states]
                else:
                    for i, h in enumerate(hidden_states):
                        embeddings['hidden_states'][i].append(h)
                attention_mask = inputs['attention_mask'].cpu().detach() if 'attention_mask' in inputs else None
                embeddings['attention_mask'].append(attention_mask)
                labels = inputs['labels'].cpu().detach() if 'labels' in inputs else None
                embeddings['labels'].append(labels)
            # Get attentions
            if output_attentions:
                attentions = [a.cpu().detach() for a in outputs.attentions] if hasattr(outputs, 'attentions') else None
                if attentions:
                    if embeddings['attentions'] is None:
                        embeddings['attentions'] = [[a] for a in attentions]
                    else:
                        for i, a in enumerate(attentions):
                            embeddings['attentions'][i].append(a)
        # Concatenate logits
        all_logits = torch.cat(all_logits, dim=0)
        if output_hidden_states:
            if embeddings['hidden_states']:
                embeddings['hidden_states'] = tuple(torch.cat(lst, dim=0) for lst in embeddings['hidden_states'])
            if embeddings['attention_mask']:
                embeddings['attention_mask'] = torch.cat(embeddings['attention_mask'], dim=0)
            if embeddings['labels']:
                embeddings['labels'] = torch.cat(embeddings['labels'], dim=0)
        if output_attentions:
            if embeddings['attentions']:
                embeddings['attentions'] = tuple(torch.cat(lst, dim=0) for lst in embeddings['attentions'])
        # Get predictions
        predictions = None
        if do_pred:
            predictions = self.logits_to_preds(all_logits)
            predictions = self.format_output(predictions)
        return all_logits, predictions, embeddings

    def predict_seqs(self, sequences: Union[str, List[str]],
                     evaluate: bool = False,
                     output_hidden_states: bool = False,
                     output_attentions: bool = False,
                     save_to_file: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """Predict for a list of sequences.

        This method provides a convenient interface for predicting on sequences,
        with optional evaluation and saving capabilities.

        Args:
            sequences: Single sequence or list of sequences for prediction
            evaluate: Whether to evaluate predictions against true labels
            output_hidden_states: Whether to output hidden states for visualization
            output_attentions: Whether to output attention weights for visualization
            save_to_file: Whether to save predictions to output directory

        Returns:
            Either:
                - Dict: Dictionary containing predictions
                - Tuple[Dict, Dict]: (predictions, metrics) if evaluate=True

        Note:
            Evaluation requires that labels are available in the dataset
        """
        # Get dataset and dataloader from sequences
        _, dataloader = self.generate_dataset(sequences, batch_size=self.pred_config.batch_size)
        # Do batch prediction
        logits, predictions, embeddings = self.batch_predict(dataloader,
                                                             output_hidden_states=output_hidden_states,
                                                             output_attentions=output_attentions)
        # Keep hidden states
        if output_hidden_states or output_attentions:
            self.embeddings = embeddings
        # Save predictions
        if save_to_file and self.pred_config.output_dir:
            save_predictions(predictions, Path(self.pred_config.output_dir))
        # Do evaluation
        if len(self.labels) == len(logits) and evaluate:
            metrics = self.calculate_metrics(logits, self.labels)
            metrics_save = dict(metrics)
            if 'curve' in metrics_save:
                del metrics_save['curve']
            if 'scatter' in metrics_save:
                del metrics_save['scatter']
            if save_to_file and self.pred_config.output_dir:
                save_metrics(metrics_save, Path(self.pred_config.output_dir))
            return predictions, metrics

        return predictions


    def predict_file(self, file_path: str, evaluate: bool = False,
                     output_hidden_states: bool = False,
                     output_attentions: bool = False,
                     seq_col: str = "sequence", label_col: str = "labels",
                     sep: str = None, fasta_sep: str = "|",
                     multi_label_sep: Union[str, None] = None,
                     uppercase: bool = False, lowercase: bool = False,
                     save_to_file: bool = False, plot_metrics: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """Predict from a file containing sequences.

        This method loads sequences from a file and performs prediction,
        with optional evaluation, visualization, and saving capabilities.

        Args:
            file_path: Path to the file containing sequences
            evaluate: Whether to evaluate predictions against true labels
            output_hidden_states: Whether to output hidden states for visualization
            output_attentions: Whether to output attention weights for visualization
            seq_col: Column name for sequences in the file
            label_col: Column name for labels in the file
            sep: Delimiter for CSV, TSV, or TXT files
            fasta_sep: Delimiter for FASTA files
            multi_label_sep: Delimiter for multi-label sequences
            uppercase: Whether to convert sequences to uppercase
            lowercase: Whether to convert sequences to lowercase
            save_to_file: Whether to save predictions and metrics to output directory
            plot_metrics: Whether to generate metric plots

        Returns:
            Either:
                - Dict: Dictionary containing predictions
                - Tuple[Dict, Dict]: (predictions, metrics) if evaluate=True

        Note:
            Setting output_attentions=True may consume significant memory
        """
        # Get dataset and dataloader from file
        _, dataloader = self.generate_dataset(file_path, seq_col=seq_col, label_col=label_col,
                                              sep=sep, fasta_sep=fasta_sep, multi_label_sep=multi_label_sep,
                                              uppercase=uppercase, lowercase=lowercase,
                                              batch_size=self.pred_config.batch_size)
        # Do batch prediction
        if output_attentions:
            warnings.warn("Cautions: output_attentions may consume a lot of memory.\n")
        logits, predictions, embeddings = self.batch_predict(dataloader,
                                                             output_hidden_states=output_hidden_states,
                                                             output_attentions=output_attentions)
        # Keep hidden states
        if output_hidden_states or output_attentions:
            self.embeddings = embeddings
        # Save predictions
        if save_to_file and self.pred_config.output_dir:
            save_predictions(predictions, Path(self.pred_config.output_dir))
        # Do evaluation
        if len(self.labels) == len(logits) and evaluate:
            metrics = self.calculate_metrics(logits, self.labels, plot=plot_metrics)
            metrics_save = dict(metrics)
            if 'curve' in metrics_save:
                del metrics_save['curve']
            if 'scatter' in metrics_save:
                del metrics_save['scatter']
            if save_to_file and self.pred_config.output_dir:
                save_metrics(metrics, Path(self.pred_config.output_dir))
            # Whether to plot metrics
            if plot_metrics:
                return predictions, metrics
            else:
                return predictions, metrics_save

        return predictions

    def calculate_metrics(self, logits: Union[List, torch.Tensor],
                          labels: Union[List, torch.Tensor], plot: bool = False) -> Dict:
        """Calculate evaluation metrics for model predictions.

        This method computes task-specific evaluation metrics using the configured
        metrics computation module.

        Args:
            logits: Model predictions (logits or probabilities)
            labels: True labels for evaluation
            plot: Whether to generate metric plots

        Returns:
            Dictionary containing evaluation metrics for the task
        """
        # Calculate metrics based on task type
        compute_metrics = Metrics(self.task_config, plot=plot)
        metrics = compute_metrics((logits, labels))

        return metrics

    def plot_attentions(self, seq_idx: int = 0, layer: int = -1, head: int = -1,
                        width: int = 800, height: int = 800,
                        save_path: Optional[str] = None) -> Optional[Any]:
        """Plot attention map visualization.

        This method creates a heatmap visualization of attention weights between tokens
        in a sequence, showing how the model attends to different parts of the input.

        Args:
            seq_idx: Index of the sequence to plot, default 0
            layer: Layer index to visualize, default -1 (last layer)
            head: Attention head index to visualize, default -1 (last head)
            width: Width of the plot
            height: Height of the plot
            save_path: Path to save the plot. If None, plot will be shown interactively

        Returns:
            Attention map visualization if available, otherwise None

        Note:
            This method requires that attention weights were collected during inference
            by setting output_attentions=True in prediction methods
        """
        if hasattr(self, 'embeddings'):
            attentions = self.embeddings['attentions']
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    heatmap = save_path.replace(suffix, "_heatmap" + suffix)
                else:
                    heatmap = os.path.join(save_path, "heatmap.pdf")
            else:
                heatmap = None
            # Plot attention map
            attn_map = plot_attention_map(attentions, self.sequences, self.tokenizer,
                                          seq_idx=seq_idx, layer=layer, head=head,
                                          width=width, height=height,
                                          save_path=heatmap)
            return attn_map
        else:
            print("No attention weights available to plot.")

    def plot_hidden_states(self, reducer: str = "t-SNE",
                           ncols: int = 4, width: int = 300, height: int = 300,
                           save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize embeddings using dimensionality reduction.

        This method creates 2D visualizations of high-dimensional embeddings from
        different model layers using PCA, t-SNE, or UMAP dimensionality reduction.

        Args:
            reducer: Dimensionality reduction method to use ('PCA', 't-SNE', 'UMAP')
            ncols: Number of columns in the plot grid
            width: Width of each plot
            height: Height of each plot
            save_path: Path to save the plot. If None, plot will be shown interactively

        Returns:
            Embedding visualization if available, otherwise None

        Note:
            This method requires that hidden states were collected during inference
            by setting output_hidden_states=True in prediction methods
        """
        if hasattr(self, 'embeddings'):
            hidden_states = self.embeddings['hidden_states']
            attention_mask = torch.unsqueeze(self.embeddings['attention_mask'], dim=-1)
            labels = self.embeddings['labels']
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    embedding = save_path.replace(suffix, "_embedding" + suffix)
                else:
                    embedding = os.path.join(save_path, "embedding.pdf")
            else:
                embedding = None
            # Plot hidden states
            label_names = self.task_config.label_names
            embeddings_vis = plot_embeddings(hidden_states, attention_mask, reducer=reducer,
                                             labels=labels, label_names=label_names,
                                             ncols=ncols, width=width, height=height,
                                             save_path=embedding)
            return embeddings_vis
        else:
            print("No hidden states available to plot.")


def save_predictions(predictions: Dict, output_dir: Path) -> None:
    """Save predictions to JSON file.

    This function saves model predictions in JSON format to the specified output directory.

    Args:
        predictions: Dictionary containing predictions to save
        output_dir: Directory path where predictions will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """Save evaluation metrics to JSON file.

    This function saves computed evaluation metrics in JSON format to the specified output directory.

    Args:
        metrics: Dictionary containing metrics to save
        output_dir: Directory path where metrics will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def generate(self, dataloader: DataLoader, n_tokens: int = 400, temperature: float = 1.0,
             top_k: int = 4) -> Dict:
    """Generate DNA sequences using the model.

    This function performs sequence generation tasks using the loaded model,
    currently supporting EVO2 models for DNA sequence generation.

    Args:
        dataloader: DataLoader containing prompt sequences
        n_tokens: Number of tokens to generate, default 400
        temperature: Sampling temperature for generation, default 1.0
        top_k: Top-k sampling parameter, default 4

    Returns:
        Dictionary containing generated sequences

    Note:
        Currently only supports EVO2 models for sequence generation
    """
    if "evo2" in str(self.model):
        for data in tqdm(dataloader, desc="Generating"):
            prompt_seqs = data['sequence']
            if isinstance(prompt_seqs, list):
                prompt_seqs = [seq for seq in prompt_seqs if seq]
            if not prompt_seqs:
                continue
            # Generate sequences
            output = self.model.generate(prompt_seqs=prompt_seqs, n_tokens=n_tokens,
                                         temperature=temperature, top_k=top_k)
            return output
