"""DNA Language Model Inference Module.

This module implements core model inference functionality, including:

1. DNAInference class
   - Model loading and initialization
   - Batch sequence inference
   - Result post-processing
   - Device management
   - Half-precision inference support

2. Core features:
   - Model state management
   - Batch inference
   - Result merging
   - Inference result saving
   - Memory optimization

3. Inference optimization:
   - Batch parallelization
   - GPU acceleration
   - Half-precision computation
   - Memory efficiency optimization

Example:
    ```python
    inference_engine = DNAInference(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    results = inference_engine.infer(sequences)
    ```
"""

import os
import warnings
import json
from typing import Any
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from ..datahandling.data import DNADataset
from ..tasks.metrics import compute_metrics
from ..utils import get_logger
from .plot import plot_attention_map, plot_embeddings

logger = get_logger("dnallm.inference.inference")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DNAInference:
    """DNA sequence inference engine using fine-tuned models.

    This class provides comprehensive functionality for performing inference
    using DNA language models. It handles model loading, inference, result
    processing, and various output formats including hidden states and
    attention weights for model interpretability.

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

    def __init__(self, model: Any, tokenizer: Any, config: dict):
        """Initialize the inference engine.

        Args:
            model: Fine-tuned model instance for inference
            tokenizer: Tokenizer for encoding DNA sequences
            config: Configuration dictionary containing task settings and
            inference parameters
        """

        self.model = model
        self.tokenizer = tokenizer
        self.task_config = config["task"]
        self.pred_config = config["inference"]
        self.device = self._get_device()
        if model:
            if "CustomEvo" in str(type(self.model)):
                self.model.model.to(self.device)
            else:
                self.model.to(self.device)
            # mamba only support cuda and cpu, and only allow fp32
            if "mamba" in str(type(self.model)).lower():
                if self.device.type != "cuda":
                    self.device = torch.device("cpu")
                if self.pred_config.use_fp16:
                    self.pred_config.use_fp16 = False
            logger.info(f"Using device: {self.device}")
        self.sequences: list[str] = []
        self.labels: list[Any] = []

    def _get_device(self) -> torch.device:
        """Get the appropriate device for model inference.

        This method automatically detects and selects the best available
        device for inference, supporting CPU, CUDA (NVIDIA), MPS (Apple
        Silicon), ROCm (AMD), TPU, and XPU (Intel).

        Returns:
            torch.device: The device to use for model inference

        Raises:
            ValueError: If the specified device type is not supported
        """
        device_str = self.pred_config.device.lower()

        if device_str == "auto":
            return self._get_auto_device()

        device_map = {
            "cpu": ("cpu", lambda: True, "CPU"),
            "cuda": ("cuda", torch.cuda.is_available, "CUDA"),
            "nvidia": ("cuda", torch.cuda.is_available, "CUDA"),
            "mps": ("mps", torch.backends.mps.is_available, "MPS"),
            "apple": ("mps", torch.backends.mps.is_available, "MPS"),
            "mac": ("mps", torch.backends.mps.is_available, "MPS"),
            "rocm": ("cuda", torch.cuda.is_available, "ROCm"),
            "amd": ("cuda", torch.cuda.is_available, "ROCm"),
            "tpu": ("xla", lambda: True, "TPU"),
            "xla": ("xla", lambda: True, "TPU"),
            "google": ("xla", lambda: True, "TPU"),
            "xpu": (
                "xpu",
                lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
                "XPU",
            ),
            "intel": (
                "xpu",
                lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
                "XPU",
            ),
        }

        if device_str not in device_map:
            raise ValueError(f"Unsupported device type: {device_str}")

        torch_device, is_available, device_name = device_map[device_str]

        if torch_device == "xla":
            try:
                return torch.device("xla")
            except Exception:
                warnings.warn(
                    f"{device_name} is not available. Please check your "
                    "installation. Use CPU instead.",
                    stacklevel=2,
                )
                return torch.device("cpu")

        if not is_available():
            warnings.warn(
                f"{device_name} is not available. Please check your "
                "installation. Use CPU instead.",
                stacklevel=2,
            )
            return torch.device("cpu")

        return torch.device(torch_device)

    def _get_auto_device(self) -> torch.device:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    def _has_model_config_attr(self, attr_name: str) -> bool:
        """Check if model has config attribute.

        Args:
            attr_name: Name of the attribute to check

        Returns:
            bool: True if model has the config attribute
        """
        return hasattr(self.model, "config") and hasattr(
            self.model.config, attr_name
        )

    def _try_set_attention_output(self) -> bool:
        """Try to temporarily set output_attentions to test support.

        Returns:
            bool: True if setting succeeded
        """
        try:
            if self._has_model_config_attr("output_attentions"):
                original_value = self.model.config.output_attentions
                self.model.config.output_attentions = True
                self.model.config.output_attentions = original_value
                return True
        except (ValueError, AttributeError):
            pass
        return False

    def _try_enable_eager_attention(self) -> bool:
        """Try to enable eager attention implementation.

        Returns:
            bool: True if eager attention was enabled successfully
        """
        try:
            if self._has_model_config_attr("attn_implementation"):
                self.model.config.attn_implementation = "eager"
                self.model.config.output_attentions = True
                return True
        except Exception as e:
            logger.debug(f"Failed to enable output_attentions: {e}")
        return False

    def _handle_attention_error(self, error: Exception) -> bool:
        """Handle attention configuration errors.

        Args:
            error: The exception that occurred

        Returns:
            bool: True if error was handled and attention enabled
        """
        error_str = str(error)
        if "attn_implementation" in error_str and "sdpa" in error_str:
            return self._try_enable_eager_attention()
        else:
            # For other errors, try to switch to eager implementation
            # as a fallback
            return self._try_enable_eager_attention()

    def _check_attention_support(self) -> bool:
        """Check if the current model supports attention output.

        This method checks whether the model can output attention weights
        based on its current configuration and attention implementation.

        Returns:
            bool: True if attention output is supported, False otherwise
        """
        # First try to set output_attentions normally
        if self._try_set_attention_output():
            return True

        # If that fails, try to handle the error by switching to eager
        # attention
        try:
            # This will raise an exception if there are issues
            if self._has_model_config_attr("output_attentions"):
                self.model.config.output_attentions = True
        except (ValueError, AttributeError) as e:
            return self._handle_attention_error(e)

        return False

    def force_eager_attention(self) -> bool:
        """Force the model to use eager attention implementation.

        This method attempts to switch the model from SDPA to eager attention
        implementation to ensure compatibility with output_attentions=True.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "attn_implementation"
            ):
                self.model.config.attn_implementation = "eager"
                logger.success("Switched to eager attention implementation")
                return True
        except Exception as e:
            logger.failure(f"Failed to switch to eager attention: {e}")
        return False

    def _check_hidden_states_support(self) -> bool:
        """Check if the current model supports hidden states output.

        This method checks whether the model can output hidden states
        based on its current configuration.

        Returns:
            bool: True if hidden states output is supported, False otherwise
        """
        try:
            # Check if the model supports output_hidden_states
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "output_hidden_states"
            ):
                # Try to set it temporarily to see if it works
                original_value = self.model.config.output_hidden_states
                self.model.config.output_hidden_states = True
                self.model.config.output_hidden_states = original_value
                return True
        except (ValueError, AttributeError) as e:
            warnings.warn(
                f"Cannot enable output_hidden_states: {e}", stacklevel=2
            )
        return False

    def generate_dataset(
        self,
        seq_or_path: str | list[str],
        batch_size: int = 1,
        seq_col: str = "sequence",
        label_col: str = "labels",
        sep: str | None = None,
        fasta_sep: str = "|",
        multi_label_sep: str | None = None,
        uppercase: bool = False,
        lowercase: bool = False,
        keep_seqs: bool = True,
        do_encode: bool = True,
    ) -> tuple[DNADataset, DataLoader]:
        """Generate dataset from sequences or file path.

        This method creates a DNADataset and DataLoader from either a list
        of sequences or a file path, supporting various file formats and
        preprocessing options.

        Args:
            seq_or_path: Single sequence, list of sequences, or path to a
                file containing sequences
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
        # Initialize dataset to None to avoid unbound variable issues
        dataset = None

        if isinstance(seq_or_path, str):
            suffix = seq_or_path.split(".")[-1]
            if suffix and os.path.isfile(seq_or_path):
                sequences = []
                dataset = DNADataset.load_local_data(
                    seq_or_path,
                    seq_col=seq_col,
                    label_col=label_col,
                    sep=sep,
                    fasta_sep=fasta_sep,
                    multi_label_sep=multi_label_sep,
                    tokenizer=self.tokenizer,
                    max_length=self.pred_config.max_length,
                )
            else:
                sequences = [seq_or_path]
        elif isinstance(seq_or_path, list):
            sequences = seq_or_path
        else:
            raise ValueError(
                "Input should be a file path or a list of sequences."
            )

        # Create dataset from sequences if we have any and no dataset was
        # loaded from file
        if len(sequences) > 0 and dataset is None:
            ds = Dataset.from_dict({"sequence": sequences})
            dataset = DNADataset(
                ds, self.tokenizer, max_length=self.pred_config.max_length
            )

        # Ensure dataset is not None before proceeding
        if dataset is None:
            raise ValueError(
                "No valid dataset could be created from the input."
            )
        # If labels are provided, keep labels
        if keep_seqs:
            self.sequences = dataset.dataset["sequence"]
        # Encode sequences
        if do_encode:
            task_type = self.task_config.task_type
            dataset.encode_sequences(
                remove_unused_columns=True,
                task=task_type,
                uppercase=uppercase,
                lowercase=lowercase,
            )
        # Check for labels in dataset - handle both Dataset and
        # DatasetDict cases
        if isinstance(dataset.dataset, DatasetDict):
            # For DatasetDict, check the first available split
            keys = list(dataset.dataset.keys())
            if keys and "labels" in dataset.dataset[keys[0]].features:
                self.labels = dataset.dataset[keys[0]]["labels"]
        else:
            # For single Dataset
            if "labels" in dataset.dataset.features:
                self.labels = dataset.dataset["labels"]
        # Create DataLoader
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.pred_config.num_workers,
        )

        return dataset, dataloader

    def logits_to_preds(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """Convert model logits to predictions and human-readable labels.

        This method processes raw model outputs based on the task type to
        generate appropriate predictions and convert them to human-readable
        labels.

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
                label = [
                    label_names[i] for i in range(len(pred)) if pred[i] == 1
                ]
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

    def format_output(self, predictions: tuple[torch.Tensor, list]) -> dict:
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
                "sequence": self.sequences[i] if keep_seqs else "",
                "label": label,
                "scores": {label_names[j]: p for j, p in enumerate(prob)}
                if task_type != "token"
                else [max(x) for x in prob],
            }
        return formatted_predictions

    def _setup_hidden_states_config(
        self, output_hidden_states: bool
    ) -> tuple[bool, dict, dict | None]:
        """Setup configuration for hidden states output.

        Args:
            output_hidden_states: Whether to output hidden states

        Returns:
            Tuple of (enabled, embeddings_dict, params)
        """
        if not output_hidden_states:
            return False, {}, None

        import inspect

        sig = inspect.signature(self.model.forward)
        params = sig.parameters

        embeddings: dict[str, Any] = {
            "hidden_states": None,
            "attention_mask": [],
            "labels": [],
        }

        if "output_hidden_states" in params:
            try:
                self.model.config.output_hidden_states = True
            except ValueError as e:
                warnings.warn(
                    f"Cannot enable output_hidden_states: {e}",
                    stacklevel=2,
                )
                return False, {}, params

        return True, embeddings, params

    def _setup_attentions_config(
        self, output_attentions: bool, params: dict | None
    ) -> tuple[bool, dict]:
        """Setup configuration for attention output.

        Args:
            output_attentions: Whether to output attentions
            params: Model forward parameters (if already computed)

        Returns:
            Tuple of (enabled, embeddings_dict)
        """
        if not output_attentions:
            return False, {}

        if not params:
            import inspect

            sig = inspect.signature(self.model.forward)
            params = sig.parameters

        embeddings = {"attentions": None}

        if "output_attentions" in params:
            try:
                self.model.config.output_attentions = True
            except ValueError as e:
                enabled = self._handle_attention_config_error(e)
                if not enabled:
                    return False, {}

        return True, embeddings

    def _handle_attention_config_error(self, error: Exception) -> bool:
        """Handle attention configuration errors in batch inference.

        Args:
            error: The configuration error

        Returns:
            bool: True if error was resolved
        """
        error_str = str(error)
        if "attn_implementation" in error_str and "sdpa" in error_str:
            try:
                self.model.config.attn_implementation = "eager"
                self.model.config.output_attentions = True
                warnings.warn(
                    "Switched to 'eager' attention implementation to "
                    "support output_attentions",
                    stacklevel=2,
                )
                return True
            except Exception:
                warnings.warn(
                    "Cannot enable output_attentions with current "
                    "attention implementation. Attention weights will "
                    "not be available.",
                    stacklevel=2,
                )
                return False
        else:
            warnings.warn(
                f"Cannot enable output_attentions: {error}",
                stacklevel=2,
            )
            return False

    def _process_batch_outputs(
        self,
        outputs: Any,
        inputs: dict,
        output_hidden_states: bool,
        output_attentions: bool,
        embeddings: dict,
    ) -> torch.Tensor:
        """Process outputs from a single batch.

        Args:
            outputs: Model outputs
            inputs: Model inputs
            output_hidden_states: Whether hidden states are enabled
            output_attentions: Whether attentions are enabled
            embeddings: Embeddings dictionary to update

        Returns:
            torch.Tensor: Batch logits
        """
        logits: torch.Tensor = outputs.logits.cpu().detach()

        if output_hidden_states:
            self._process_hidden_states(outputs, inputs, embeddings)

        if output_attentions:
            self._process_attentions(outputs, embeddings)

        return logits

    def _process_hidden_states(
        self, outputs: Any, inputs: dict, embeddings: dict
    ) -> None:
        """Process hidden states from model outputs.

        Args:
            outputs: Model outputs
            inputs: Model inputs
            embeddings: Embeddings dictionary to update
        """
        hidden_states = (
            [h.cpu().detach() for h in outputs.hidden_states]
            if hasattr(outputs, "hidden_states")
            else None
        )

        if hidden_states:
            if embeddings["hidden_states"] is None:
                embeddings["hidden_states"] = [[h] for h in hidden_states]
            else:
                for i, h in enumerate(hidden_states):
                    embeddings["hidden_states"][i].append(h)

        attention_mask = (
            inputs["attention_mask"].cpu().detach()
            if "attention_mask" in inputs
            else None
        )
        embeddings["attention_mask"].append(attention_mask)

        labels = (
            inputs["labels"].cpu().detach() if "labels" in inputs else None
        )
        embeddings["labels"].append(labels)

    def _process_attentions(self, outputs: Any, embeddings: dict) -> None:
        """Process attention weights from model outputs.

        Args:
            outputs: Model outputs
            embeddings: Embeddings dictionary to update
        """
        attentions = (
            [a.cpu().detach() for a in outputs.attentions]
            if hasattr(outputs, "attentions")
            else None
        )

        if attentions:
            if embeddings["attentions"] is None:
                embeddings["attentions"] = [[a] for a in attentions]
            else:
                for i, a in enumerate(attentions):
                    embeddings["attentions"][i].append(a)

    def _finalize_embeddings(
        self,
        embeddings: dict,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> None:
        """Finalize embeddings by concatenating tensors.

        Args:
            embeddings: Embeddings dictionary to finalize
            output_hidden_states: Whether hidden states were collected
            output_attentions: Whether attentions were collected
        """
        if output_hidden_states:
            if embeddings.get("hidden_states"):
                embeddings["hidden_states"] = tuple(
                    torch.cat(lst, dim=0)
                    for lst in embeddings["hidden_states"]
                )
            if embeddings.get("attention_mask"):
                embeddings["attention_mask"] = torch.cat(
                    embeddings["attention_mask"], dim=0
                )
            if embeddings.get("labels"):
                embeddings["labels"] = torch.cat(embeddings["labels"], dim=0)

        if output_attentions:
            if embeddings.get("attentions"):
                embeddings["attentions"] = tuple(
                    torch.cat(lst, dim=0) for lst in embeddings["attentions"]
                )

    @torch.no_grad()
    def batch_infer(
        self,
        dataloader: DataLoader,
        do_pred: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, dict | None, dict]:
        """Perform batch inference on sequences.

        This method runs inference on batches of sequences and optionally
        extracts hidden states and attention weights for model
        interpretability.

        Args:
            dataloader: DataLoader object containing sequences for inference
            do_pred: Whether to convert logits to predictions
            output_hidden_states: Whether to output hidden states from
                all layers
            output_attentions: Whether to output attention weights from
                all layers

        Returns:
            Tuple containing:
                - torch.Tensor: All logits from the model
                - Optional[Dict]: Predictions dictionary if do_pred=True,
                  otherwise None
                - Dict: Embeddings dictionary containing hidden states
                  and/or attention weights

        Note:
            Setting output_hidden_states or output_attentions to True will
            consume significant memory, especially for long sequences or
            large models.
        """
        # Set model to evaluation mode
        self.model.eval()
        all_logits = []

        # Setup configurations for outputs
        output_hidden_states, hidden_embeddings, params = (
            self._setup_hidden_states_config(output_hidden_states)
        )
        output_attentions, attention_embeddings = (
            self._setup_attentions_config(output_attentions, params)
        )

        # Combine embeddings dictionaries
        embeddings = {**hidden_embeddings, **attention_embeddings}

        # Iterate over batches
        for batch in tqdm(dataloader, desc="Inferring"):
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # Run model inference
            if self.pred_config.use_fp16:
                self.model = self.model.half()
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            # Process batch outputs
            logits = self._process_batch_outputs(
                outputs,
                inputs,
                output_hidden_states,
                output_attentions,
                embeddings,
            )
            all_logits.append(logits)

        # Concatenate all logits
        all_logits = torch.cat(all_logits, dim=0)

        # Finalize embeddings
        self._finalize_embeddings(
            embeddings, output_hidden_states, output_attentions
        )

        # Get predictions if requested
        predictions = None
        if do_pred:
            predictions = self.logits_to_preds(all_logits)
            predictions = self.format_output(predictions)

        return all_logits, predictions, embeddings

    def infer_seqs(
        self,
        sequences: str | list[str],
        evaluate: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        save_to_file: bool = False,
    ) -> dict | tuple[dict, dict]:
        """Infer for a list of sequences.

        This method provides a convenient interface for performing
        inference on sequences, with optional evaluation and saving
        capabilities.

        Args:
            sequences: Single sequence or list of sequences for inference
            evaluate: Whether to evaluate predictions against true labels
            output_hidden_states: Whether to output hidden states for
                visualization
            output_attentions: Whether to output attention weights for
                visualization
            save_to_file: Whether to save predictions to output directory

        Returns:
            Either:
                - Dict: Dictionary containing predictions
                - Tuple[Dict, Dict]: (predictions, metrics) if evaluate=True

        Note:
            Evaluation requires that labels are available in the dataset
        """
        # Get dataset and dataloader from sequences
        _, dataloader = self.generate_dataset(
            sequences, batch_size=self.pred_config.batch_size
        )
        # Do batch inference
        logits, predictions, embeddings = self.batch_infer(
            dataloader,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
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
            metrics_save.pop("curve", None)
            metrics_save.pop("scatter", None)
            if save_to_file and self.pred_config.output_dir:
                save_metrics(metrics_save, Path(self.pred_config.output_dir))
            return predictions, metrics

        return predictions

    def infer_file(
        self,
        file_path: str,
        evaluate: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        seq_col: str = "sequence",
        label_col: str = "labels",
        sep: str | None = None,
        fasta_sep: str = "|",
        multi_label_sep: str | None = None,
        uppercase: bool = False,
        lowercase: bool = False,
        save_to_file: bool = False,
        plot_metrics: bool = False,
    ) -> dict | tuple[dict, dict]:
        """Infer from a file containing sequences.

        This method loads sequences from a file and performs inference,
        with optional evaluation, visualization, and saving capabilities.

        Args:
            file_path: Path to the file containing sequences
            evaluate: Whether to evaluate predictions against true labels
            output_hidden_states: Whether to output hidden states for
                visualization
            output_attentions: Whether to output attention weights for
                visualization
            seq_col: Column name for sequences in the file
            label_col: Column name for labels in the file
            sep: Delimiter for CSV, TSV, or TXT files
            fasta_sep: Delimiter for FASTA files
            multi_label_sep: Delimiter for multi-label sequences
            uppercase: Whether to convert sequences to uppercase
            lowercase: Whether to convert sequences to lowercase
            save_to_file: Whether to save predictions and metrics to
                output directory
            plot_metrics: Whether to generate metric plots

        Returns:
            Either:
                - Dict: Dictionary containing predictions
                - Tuple[Dict, Dict]: (predictions, metrics) if evaluate=True

        Note:
            Setting output_attentions=True may consume significant memory
        """
        # Get dataset and dataloader from file
        _, dataloader = self.generate_dataset(
            file_path,
            seq_col=seq_col,
            label_col=label_col,
            sep=sep,
            fasta_sep=fasta_sep,
            multi_label_sep=multi_label_sep,
            uppercase=uppercase,
            lowercase=lowercase,
            batch_size=self.pred_config.batch_size,
        )
        # Do batch inference
        if output_attentions:
            warnings.warn(
                "Cautions: output_attentions may consume a lot of memory.\n",
                stacklevel=2,
            )
        logits, predictions, embeddings = self.batch_infer(
            dataloader,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # Keep hidden states
        if output_hidden_states or output_attentions:
            self.embeddings = embeddings
        # Save predictions
        if save_to_file and self.pred_config.output_dir:
            save_predictions(predictions, Path(self.pred_config.output_dir))
        # Do evaluation
        if len(self.labels) == len(logits) and evaluate:
            metrics = self.calculate_metrics(
                logits, self.labels, plot=plot_metrics
            )
            metrics_save = dict(metrics)
            metrics_save.pop("curve", None)
            metrics_save.pop("scatter", None)
            if save_to_file and self.pred_config.output_dir:
                save_metrics(metrics, Path(self.pred_config.output_dir))
            # Whether to plot metrics
            if plot_metrics:
                return predictions, metrics
            else:
                return predictions, metrics_save

        return predictions

    def infer(
        self,
        sequences: str | list[str] | None = None,
        file_path: str | None = None,
        evaluate: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        save_to_file: bool = False,
        **kwargs,
    ) -> dict | tuple[dict, dict]:
        """Main inference method for sequences or files.

        This is the primary entry point for performing inference. It
        automatically determines whether to process sequences directly or
        load from a file.

        Args:
            sequences: Single sequence or list of sequences for inference
            file_path: Path to file containing sequences for inference
            evaluate: Whether to evaluate predictions against true labels
            output_hidden_states: Whether to output hidden states for
                visualization
            output_attentions: Whether to output attention weights for
                visualization
            save_to_file: Whether to save predictions to output directory
            **kwargs: Additional arguments passed to specific inference methods

        Returns:
            Either:
                - Dict: Dictionary containing predictions
                - Tuple[Dict, Dict]: (predictions, metrics) if evaluate=True

        Raises:
            ValueError: If neither sequences nor file_path is provided
        """
        if sequences is not None:
            return self.infer_seqs(
                sequences=sequences,
                evaluate=evaluate,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                save_to_file=save_to_file,
            )
        elif file_path is not None:
            return self.infer_file(
                file_path=file_path,
                evaluate=evaluate,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                save_to_file=save_to_file,
                **kwargs,
            )
        else:
            raise ValueError("Either sequences or file_path must be provided")

    def calculate_metrics(
        self,
        logits: list | torch.Tensor,
        labels: list | torch.Tensor,
        plot: bool = False,
    ) -> dict[Any, Any]:
        """Calculate evaluation metrics for model predictions.

        This method computes task-specific evaluation metrics using the
        configured metrics computation module.

        Args:
            logits: Model predictions (logits or probabilities)
            labels: True labels for evaluation
            plot: Whether to generate metric plots

        Returns:
            Dictionary containing evaluation metrics for the task
        """
        # Calculate metrics based on task type
        compute_metrics_func = compute_metrics(self.task_config, plot=plot)
        metrics: dict[Any, Any] = compute_metrics_func((logits, labels))

        return metrics

    def plot_attentions(
        self,
        seq_idx: int = 0,
        layer: int = -1,
        head: int = -1,
        width: int = 800,
        height: int = 800,
        save_path: str | None = None,
    ) -> Any | None:
        """Plot attention map visualization.

        This method creates a heatmap visualization of attention weights
        between tokens in a sequence, showing how the model attends to
        different parts of the input.

        Args:
            seq_idx: Index of the sequence to plot, default 0
            layer: Layer index to visualize, default -1 (last layer)
            head: Attention head index to visualize, default -1 (last head)
            width: Width of the plot
            height: Height of the plot
            save_path: Path to save the plot. If None, plot will be shown
                interactively

        Returns:
            Attention map visualization if available, otherwise None

        Note:
            This method requires that attention weights were collected
            during inference by setting output_attentions=True in prediction
            methods
        """
        if hasattr(self, "embeddings"):
            attentions = self.embeddings["attentions"]
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    heatmap = save_path.replace(suffix, "_heatmap" + suffix)
                else:
                    heatmap = os.path.join(save_path, "heatmap.pdf")
            else:
                heatmap = None
            # Plot attention map
            attn_map = plot_attention_map(
                attentions,
                self.sequences,
                self.tokenizer,
                seq_idx=seq_idx,
                layer=layer,
                head=head,
                width=width,
                height=height,
                save_path=heatmap,
            )
            return attn_map
        else:
            logger.warning("No attention weights available to plot.")
            return None

    def plot_hidden_states(
        self,
        reducer: str = "t-SNE",
        ncols: int = 4,
        width: int = 300,
        height: int = 300,
        save_path: str | None = None,
    ) -> Any | None:
        """Visualize embeddings using dimensionality reduction.

        This method creates 2D visualizations of high-dimensional
        embeddings from different model layers using PCA, t-SNE, or UMAP
        dimensionality reduction.

        Args:
            reducer: Dimensionality reduction method to use
                ('PCA', 't-SNE', 'UMAP')
            ncols: Number of columns in the plot grid
            width: Width of each plot
            height: Height of each plot
            save_path: Path to save the plot. If None, plot will be shown
                interactively

        Returns:
            Embedding visualization if available, otherwise None

        Note:
            This method requires that hidden states were collected during
            inference by setting output_hidden_states=True in prediction
            methods
        """
        if hasattr(self, "embeddings"):
            hidden_states = self.embeddings["hidden_states"]
            attention_mask = torch.unsqueeze(
                self.embeddings["attention_mask"], dim=-1
            )
            labels = self.embeddings["labels"]
            if save_path:
                suffix = os.path.splitext(save_path)[-1]
                if suffix:
                    embedding = save_path.replace(
                        suffix, "_embedding" + suffix
                    )
                else:
                    embedding = os.path.join(save_path, "embedding.pdf")
            else:
                embedding = None
            # Plot hidden states
            label_names = self.task_config.label_names
            embeddings_vis = plot_embeddings(
                hidden_states,
                attention_mask,
                reducer=reducer,
                labels=labels,
                label_names=label_names,
                ncols=ncols,
                width=width,
                height=height,
                save_path=embedding,
            )
            return embeddings_vis
        else:
            logger.warning("No hidden states available to plot.")

    def _get_basic_model_info(self) -> dict[str, Any]:
        """Get basic model information.

        Returns:
            Dict containing basic model information
        """
        return {
            "model_type": type(self.model).__name__,
            "device": str(self.device),
            "attention_supported": self._check_attention_support(),
            "hidden_states_supported": self._check_hidden_states_support(),
            "dtype": str(next(self.model.parameters()).dtype)
            if self.model.parameters()
            else "Unknown",
            "current_attn_implementation": getattr(
                self.model.config, "attn_implementation", "Not set"
            )
            if hasattr(self.model, "config")
            else "Unknown",
        }

    def _get_model_config_info(self) -> dict[str, Any]:
        """Get model configuration information.

        Returns:
            Dict containing model configuration details
        """
        config_info: dict[str, Any] = {}

        if not hasattr(self.model, "config"):
            return config_info

        config = self.model.config
        config_attrs = [
            "model_type",
            "attn_implementation",
            "num_attention_heads",
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
        ]

        for attr in config_attrs:
            if hasattr(config, attr):
                config_info[attr] = getattr(config, attr)

        return config_info

    def _get_model_parameters_info(self) -> dict[str, Any]:
        """Get model parameters information.

        Returns:
            Dict containing parameter information or error
        """
        try:
            return self.get_model_parameters()
        except Exception:
            return {"error": "Could not retrieve parameter information"}

    def _get_model_config_dict(self) -> dict[str, Any]:
        """Get model configuration as dictionary.

        Returns:
            Dict containing configuration or error
        """
        if not hasattr(self.model, "config"):
            return {"error": "Model has no config"}

        try:
            config_dict = {}
            for key, value in self.model.config.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    config_dict[key] = value
            return config_dict
        except Exception:
            return {"error": "Could not retrieve configuration"}

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dict containing model information including type, device, and
            attention support
        """
        # Get basic model information
        info = self._get_basic_model_info()

        # Add model-specific configuration information
        info.update(self._get_model_config_info())

        # Add parameter information
        info["num_parameters"] = self._get_model_parameters_info()

        # Add configuration as a dictionary
        info["config"] = self._get_model_config_dict()

        return info

    def get_model_parameters(self) -> dict[str, int]:
        """Get information about model parameters.

        Returns:
            Dict containing parameter counts
        """
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            frozen_params = total_params - trainable_params

            return {
                "total": total_params,
                "trainable": trainable_params,
                "frozen": frozen_params,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_available_outputs(self) -> dict[str, Any]:
        """Get information about available model outputs.

        Returns:
            Dict containing information about what outputs are available and
            collected
        """
        capabilities = {
            "hidden_states_available": self._check_hidden_states_support(),
            "attentions_available": self._check_attention_support(),
            "hidden_states_collected": hasattr(self, "embeddings")
            and "hidden_states" in self.embeddings
            and self.embeddings["hidden_states"] is not None,
            "attentions_collected": hasattr(self, "embeddings")
            and "attentions" in self.embeddings
            and self.embeddings["attentions"] is not None,
        }
        return capabilities

    def estimate_memory_usage(
        self, batch_size: int = 1, sequence_length: int = 1000
    ) -> dict[str, Any]:
        """Estimate memory usage for inference.

        Args:
            batch_size: Batch size for inference
            sequence_length: Maximum sequence length

        Returns:
            Dict containing memory usage estimates
        """
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            param_memory_mb = (total_params * 4) / (
                1024 * 1024
            )  # Assuming float32

            # Estimate activation memory (rough approximation)
            if hasattr(self.model, "config"):
                config = self.model.config
                hidden_size = getattr(config, "hidden_size", 768)
                num_layers = getattr(config, "num_hidden_layers", 12)
            else:
                hidden_size, num_layers = 768, 12

            # Rough estimate for activations
            activation_memory_mb = (
                batch_size * sequence_length * hidden_size * num_layers * 2
            ) / (1024 * 1024)

            total_memory_mb = param_memory_mb + activation_memory_mb

            return {
                "total_estimated_mb": f"{total_memory_mb:.1f}",
                "parameter_memory_mb": f"{param_memory_mb:.1f}",
                "activation_memory_mb": f"{activation_memory_mb:.1f}",
                "note": "Estimates are approximate and may vary based on "
                "actual usage",
            }
        except Exception as e:
            return {"error": str(e)}

    def generate(
        self,
        inputs: DataLoader | list[str],
        n_tokens: int = 400,
        n_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
    ) -> dict[Any, Any]:
        """Generate DNA sequences using the model.

        This function performs sequence generation tasks using the loaded
        model, currently supporting CausalLM and EVO2 models for
        DNA sequence generation.

        Args:
            inputs: DataLoader or List containing prompt sequences
            n_tokens: Number of tokens to generate, default 400
            n_samples: Do samples n times
            temperature: Sampling temperature for generation, default 1.0
            top_k: Top-k sampling parameter, default 4
            top_p: Top-p sampling paramether, default 1
            batched: Do batched generation

        Returns:
            Dictionary containing generated sequences

        Note:
            Currently only supports EVO2 models for sequence generation
        """
        # Prepare prompt sequences
        prompt_seqs = []
        if isinstance(inputs, DataLoader):
            for data in tqdm(inputs, desc="Generating"):
                seqs = data["sequence"]
                if isinstance(prompt_seqs, list):
                    seqs.extend([seq for seq in seqs if seq])
                if not seqs:
                    continue
        else:
            prompt_seqs = inputs
        # Check if model supports generation
        if "evo2" in str(self.model).lower():
            # Generate sequences
            outputs = self.model.generate(
                prompt_seqs=prompt_seqs,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=True,
            )
            formatted_outputs = []
            for i, seq in enumerate(prompt_seqs):
                generated_seqs = outputs.sequences[i]
                scores = outputs.logprobs_mean[i]
                formatted_outputs.append({
                    "Prompt": seq,
                    "Output": generated_seqs,
                    "Score": scores,
                })
            return formatted_outputs
        elif "evo1" in str(self.model).lower():
            from evo import generate

            model = self.model.model
            tokenizer = self.tokenizer
            # Generate sequences
            outputs = generate(
                prompt_seqs * n_samples,
                model=model,
                tokenizer=tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cached_generation=True,
                batched=batched,
                device=self.device,
                verbose=1,
            )
            formatted_outputs = []
            for i, seq in enumerate(prompt_seqs):
                generated_seqs = outputs[0][i]
                scores = outputs[1][i]
                formatted_outputs.append({
                    "Prompt": seq,
                    "Output": generated_seqs,
                    "Score": scores,
                })
            return formatted_outputs
        elif (
            "causallm" in str(self.model).lower()
            or "lmhead" in str(self.model).lower()
        ):
            outputs = []
            # Tokenize prompt sequences
            for seq in prompt_seqs:
                inputs = self.tokenizer(seq, return_tensors="pt").to(
                    self.device
                )
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=n_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                )
                decoded = self.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                outputs.append({"Prompt": seq, "Output": decoded})
            return outputs
        else:
            raise ValueError(
                "This model is not supported for sequence generation."
            )

        return {}

    def scoring(
        self,
        inputs: DataLoader | list[str],
        reduce_method: str = "mean",
    ) -> dict[Any, Any]:
        # Prepare score sequences
        score_seqs = []
        if isinstance(inputs, DataLoader):
            for data in tqdm(inputs, desc="Scoring"):
                seqs = data["sequence"]
                if isinstance(score_seqs, list):
                    score_seqs.extend([seq for seq in seqs if seq])
                if not seqs:
                    continue
        else:
            score_seqs = inputs
        # Check if model supports scoring
        if "evo2" in str(self.model).lower():
            outputs = self.model.score_sequences(
                score_seqs, reduce_method=reduce_method
            )
            outputs = [
                {"Input": score_seqs[i], "Score": s}
                for i, s in enumerate(outputs)
            ]
            return outputs
        elif "evo1" in str(self.model).lower():
            from evo import score_sequences

            model = self.model.model
            tokenizer = self.tokenizer
            outputs = score_sequences(
                score_seqs,
                model=model,
                tokenizer=tokenizer,
                reduce_method=reduce_method,
                device=self.device,
            )
            outputs = [
                {"Input": score_seqs[i], "Score": s}
                for i, s in enumerate(outputs)
            ]
            return outputs
        return {}


def save_predictions(predictions: dict, output_dir: Path) -> None:
    """Save predictions to JSON file.

    This function saves model predictions in JSON format to the specified
    output directory.

    Args:
        predictions: Dictionary containing predictions to save
        output_dir: Directory path where predictions will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Save evaluation metrics to JSON file.

    This function saves computed evaluation metrics in JSON format to the
    specified output directory.

    Args:
        metrics: Dictionary containing metrics to save
        output_dir: Directory path where metrics will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
