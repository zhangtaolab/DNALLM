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
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from ..datahandling.data import DNADataset
from ..models.model import _get_model_path_and_imports
from ..tasks.metrics import compute_metrics
from ..utils import get_logger
from .plot import plot_attention_map, plot_embeddings, _compute_mean_embeddings

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

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict,
        lora_adapter: str | None = None,
    ) -> None:
        """Initialize the inference engine.

        Args:
            model: Fine-tuned model instance for inference
            tokenizer: Tokenizer for encoding DNA sequences
            config: Configuration dictionary containing task settings\
                    and inference parameters
            lora_adapter: Optional path to LoRA adapter for model
        """

        default_forward_args = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "inputs_embeds",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "past_key_values",
            "use_cache",
        }

        if lora_adapter:
            from peft import PeftModel
            from ..models.model import peft_forward_compatiable

            if os.path.isdir(lora_adapter):
                source = "local"
            else:
                source = (
                    model.source if hasattr(model, "source") else "huggingface"
                )
            try:
                lora_adapter_path, _ = _get_model_path_and_imports(
                    lora_adapter, source
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load LoRA adapter from {lora_adapter}: {e}"
                ) from e

            if model is not None:
                self.accepted_args = self._get_accepted_forward_args(model)
            else:
                self.accepted_args = set(default_forward_args)

            model = peft_forward_compatiable(model)
            self.model = PeftModel.from_pretrained(model, lora_adapter_path)
            logger.info(f"Loaded LoRA adapter from {lora_adapter}")
        else:
            self.model = model
            if model is not None:
                self.accepted_args = self._get_accepted_forward_args(model)
            else:
                self.accepted_args = set(default_forward_args)
        self.tokenizer = tokenizer
        self.pad_id = self._get_pad_id()
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
            "npu": (
                "npu",
                lambda: (hasattr(torch, "npu") and torch.npu.is_available()),
                "NPU",
            ),
            "ascend": (
                "npu",
                lambda: (hasattr(torch, "npu") and torch.npu.is_available()),
                "NPU",
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
        if hasattr(torch, "xpu"):
            if torch.xpu.is_available():
                return torch.device("xpu")
        if hasattr(torch, "npu"):
            if torch.npu.is_available():
                return torch.device("npu")
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

    def _get_accepted_forward_args(self, model) -> set:
        """Get the set of argument names the model's forward method accepts.

        Returns:
            set: Set of accepted argument names
        """
        import inspect

        sig = inspect.signature(model.forward)
        accepted = set(sig.parameters.keys())
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            accepted.add("**kwargs")

        return accepted

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
        sampling: float | None = None,
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
            sampling: Fraction of data to randomly sample for inference
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

        # If sampling is specified, randomly sample the sequences
        if sampling:
            dataset = dataset.sampling(sampling) if dataset else None

        # Create dataset from sequences if we have any and no dataset was
        # loaded from file
        if len(sequences) > 0 and dataset is None:
            ds = Dataset.from_dict({"sequence": sequences})
            dataset = DNADataset(
                ds, self.tokenizer, max_length=self.pred_config.max_length
            )

        # Ensure dataset is not None before proceeding
        if not dataset:
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
            all_cols = dataset.dataset.features
            cols_drop = [c for c in all_cols if c not in self.accepted_args]
            dataset.dataset = dataset.dataset.remove_columns(cols_drop)
        else:
            all_cols = dataset.dataset.features
            # check if dataset is already encoded
            if "sequence" in all_cols:
                check_seq = dataset.dataset["sequence"][0]
                if isinstance(check_seq, torch.Tensor):
                    # already encoded
                    if "input_ids" not in all_cols:
                        dataset.dataset = dataset.dataset.rename_column(
                            "sequence", "input_ids"
                        )
                    dataset.dataset.set_format(type="torch")
                    cols_drop = [
                        c
                        for c in dataset.dataset.features
                        if c not in self.accepted_args
                    ]
                    dataset.dataset = dataset.dataset.remove_columns(cols_drop)
                else:
                    cols_drop = [
                        c
                        for c in dataset.dataset.features
                        if c not in self.accepted_args
                    ]
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
            labels = label_names
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
            if task_type == "regression":
                scores = {label_names[0]: prob}
            elif task_type == "token":
                scores = [max(x) for x in prob]
            else:
                scores = {label_names[j]: p for j, p in enumerate(prob)}
            formatted_predictions[i] = {
                "sequence": self.sequences[i] if keep_seqs else "",
                "label": label,
                "scores": scores,
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
        reduce_hidden_states: bool = False,
        reduce_strategy: str | int = "mean",
    ) -> torch.Tensor | None:
        """Process outputs from a single batch.

        Args:
            outputs: Model outputs
            inputs: Model inputs
            output_hidden_states: Whether hidden states are enabled
            output_attentions: Whether attentions are enabled
            embeddings: Embeddings dictionary to update
            reduce_hidden_states: Whether to average hidden states across
                layers

        Returns:
            torch.Tensor: Batch logits
        """
        # Process logits
        if self.task_config.task_type == "embedding":
            logits: None = None
        else:
            if hasattr(outputs, "logits"):
                logits: torch.Tensor = outputs.logits.detach().float().cpu()
            elif "logits" in outputs:
                logits: torch.Tensor = outputs["logits"].detach().float().cpu()
            elif isinstance(outputs, tuple) or isinstance(outputs, list):
                # Assume logits are in outputs if outputs is a tuple or list
                # index 0 is usually hidden states or last hidden state
                # index 1 is usually logits
                # indexes beyond 1 are usually other outputs like attentions
                if len(outputs) > 1:
                    logits: torch.Tensor = outputs[1].detach().float().cpu()
                else:
                    logits: None = None
            else:
                logits: None = None
        # Process hidden states
        if output_hidden_states:
            self._process_hidden_states(
                outputs,
                inputs,
                embeddings,
                reduce_hidden_states,
                reduce_strategy,
            )
        # Process attentions
        if output_attentions:
            self._process_attentions(outputs, embeddings)

        return logits

    def _get_pad_id(self) -> int | None:
        """Get padding token ID from tokenizer.

        Returns:
            int | None: Padding token ID if available, else None
        """
        # Get padding token if available
        pad_id: int | None = None
        if hasattr(self.tokenizer, "pad_token_id"):
            pad_id = self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, "pad_token"):
            pad_token = self.tokenizer.pad_token
            try:
                pad_id = self.tokenizer.convert_tokens_to_ids(pad_token)
                if pad_id is None:
                    raise ValueError("No convert_tokens_to_ids")
            except Exception:
                try:
                    pad_id = self.tokenizer.encode(pad_token)[0]
                except Exception:
                    pad_id = None
        if pad_id is None and hasattr(self.tokenizer, "eos_token_id"):
            pad_id = self.tokenizer.eos_token_id
        return pad_id

    def _create_attention_mask(self, inputs: dict) -> torch.Tensor | None:
        """Create attention mask from inputs.

        Args:
            inputs: Model inputs

        Returns:
            torch.Tensor | None: Attention mask tensor if padding ID is
                available, else None
        """
        attention_mask: torch.Tensor | None = (
            inputs["attention_mask"].long().cpu().detach()
            if "attention_mask" in inputs
            else None
        )
        if attention_mask is None:
            if self.pad_id is not None and "input_ids" in inputs:
                try:
                    input_ids = inputs["input_ids"]
                    attention_mask = (
                        (input_ids != self.pad_id).long().cpu().detach()
                    )
                except Exception:
                    attention_mask = None
        return attention_mask

    def _process_hidden_states(
        self,
        outputs: Any,
        inputs: dict,
        embeddings: dict,
        reduce_hidden_states: bool = False,
        reduce_strategy: str | int = "mean",
    ) -> None:
        """Process hidden states from model outputs.

        Args:
            outputs: Model outputs
            inputs: Model inputs
            embeddings: Embeddings dictionary to update
            reduce_hidden_states: Whether to average hidden states across
                layers
        """
        attention_mask = self._create_attention_mask(inputs)
        hiddens = None
        hidden_states = []
        if hasattr(outputs, "hidden_states"):
            hiddens = outputs.hidden_states
        elif "hidden_states" in outputs:
            hiddens = outputs["hidden_states"]
        if hiddens is None:
            if hasattr(outputs, "last_hidden_state"):
                hiddens = outputs.last_hidden_state
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Assume hidden states are in outputs[0] if outputs
                # is a list or tuple
                hiddens = outputs[0]
        if hiddens is not None:
            if isinstance(hiddens, (list, tuple)):
                # Multiple layers of hidden states
                for h in hiddens:
                    h = h.cpu().detach()
                    hidden_states.append(
                        torch.tensor(
                            _compute_mean_embeddings(
                                h, attention_mask, reduce_strategy
                            )
                        )
                        if reduce_hidden_states
                        else h
                    )
            else:
                # Single layer of hidden states
                h = hiddens.cpu().detach()
                hidden_states.append(
                    torch.tensor(
                        _compute_mean_embeddings(
                            h, attention_mask, reduce_strategy
                        )
                    )
                    if reduce_hidden_states
                    else h
                )

        if hidden_states:
            if embeddings["hidden_states"] is None:
                embeddings["hidden_states"] = [[h] for h in hidden_states]
            else:
                for i, h in enumerate(hidden_states):
                    embeddings["hidden_states"][i].append(h)
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
                if embeddings["attention_mask"][0] is not None:
                    embeddings["attention_mask"] = torch.cat(
                        embeddings["attention_mask"], dim=0
                    )
            if embeddings.get("labels"):
                if embeddings["labels"][0]:
                    embeddings["labels"] = torch.cat(
                        embeddings["labels"], dim=0
                    )

        if output_attentions:
            if embeddings.get("attentions"):
                embeddings["attentions"] = tuple(
                    torch.cat(lst, dim=0) for lst in embeddings["attentions"]
                )

    @torch.inference_mode()
    def batch_infer(
        self,
        dataloader: DataLoader,
        do_pred: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        reduce_hidden_states: bool = False,
        reduce_strategy: str | int = "mean",
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
            reduce_hidden_states: Whether to average hidden states across
                layers

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

        # Check model precision settings
        if self.pred_config.use_fp16:
            dtype = torch.float16
        elif self.pred_config.use_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Iterate over batches
        for batch in tqdm(dataloader, desc="Inferring"):
            inputs = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
            # Add output flags if supported
            # In case model config does not recognize these args
            if output_attentions:
                if "output_attentions" in self.accepted_args:
                    inputs["output_attentions"] = True
                elif "**kwargs" in self.accepted_args:
                    inputs["output_attentions"] = True
            if output_hidden_states:
                if "output_hidden_states" in self.accepted_args:
                    inputs["output_hidden_states"] = True
                elif "**kwargs" in self.accepted_args:
                    inputs["output_hidden_states"] = True

            # Run model inference
            # check accepted forward method
            args = inputs.keys()
            accepted_inputs = {}
            for arg in args:
                if (
                    arg in self.accepted_args
                    or "**kwargs" in self.accepted_args
                ):
                    accepted_inputs[arg] = inputs[arg]

            # Use autocast for mixed precision if enabled
            if self.pred_config.use_fp16 or self.pred_config.use_bf16:
                with torch.amp.autocast("cuda", dtype=dtype):
                    outputs = self.model(**accepted_inputs)
            else:
                outputs = self.model(**accepted_inputs)

            # Process batch outputs
            logits = self._process_batch_outputs(
                outputs,
                inputs,
                output_hidden_states,
                output_attentions,
                embeddings,
                reduce_hidden_states,
                reduce_strategy,
            )
            all_logits.append(logits)

        # Concatenate all logits
        if all_logits and all_logits[0] is not None:
            all_logits = torch.cat(all_logits, dim=0)

        # Finalize embeddings
        self._finalize_embeddings(
            embeddings, output_hidden_states, output_attentions
        )

        # Get predictions if requested
        predictions = None
        if do_pred and len(all_logits) > 0:
            predictions = self.logits_to_preds(all_logits)
            predictions = self.format_output(predictions)

        return all_logits, predictions, embeddings

    def infer_seqs(
        self,
        sequences: str | list[str],
        do_pred: bool = True,
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
            do_pred=do_pred,
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
        file_path: str | Dataset,
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
        sampling: float | None = None,
        do_encode: bool = True,
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
            sampling: Fraction of data to randomly sample for inference
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
        if isinstance(file_path, Dataset):
            dataloader = DataLoader(
                file_path,
                batch_size=self.pred_config.batch_size,
                num_workers=self.pred_config.num_workers,
            )
            self.labels = (
                file_path["labels"] if "labels" in file_path.features else []
            )
        else:
            _, dataloader = self.generate_dataset(
                file_path,
                seq_col=seq_col,
                label_col=label_col,
                sep=sep,
                fasta_sep=fasta_sep,
                multi_label_sep=multi_label_sep,
                uppercase=uppercase,
                lowercase=lowercase,
                sampling=sampling,
                do_encode=do_encode,
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
        **kwargs: Any,
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
        norm_method: str | None = None,
        skip_cls=True,
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
                norm_method=norm_method,
                skip_cls=skip_cls,
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
        reduced: bool = False,
        quality: str = "fast",
        ncols: int = 4,
        width: int = 300,
        height: int = 300,
        point_size: int = 10,
        save_path: str | None = None,
    ) -> Any | None:
        """Visualize embeddings using dimensionality reduction.

        This method creates 2D visualizations of high-dimensional
        embeddings from different model layers using PCA, t-SNE, or UMAP
        dimensionality reduction.

        Args:
            reducer: Dimensionality reduction method to use
                ('PCA', 't-SNE', 'UMAP')
            reduced: Whether to use already reduced embeddings if available
            quality: Quality/speed trade-off for reduction
            ncols: Number of columns in the plot grid
            width: Width of each plot
            height: Height of each plot
            point_size: Size of points in the plot
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
            # attention_mask = torch.unsqueeze(
            #     self.embeddings["attention_mask"], dim=-1
            # )
            attention_mask = self.embeddings["attention_mask"]
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
                quality=quality,
                labels=labels,
                label_names=label_names,
                ncols=ncols,
                width=width,
                height=height,
                point_size=point_size,
                save_path=embedding,
                reduced=reduced,
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
            Currently only supports Causal language models
            for sequence generation
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
        elif "megadna" in str(self.model).lower():
            model = self.model
            tokenizer = self.tokenizer
            formatted_outputs = []
            for seq in prompt_seqs:
                for _ in range(n_samples):
                    input_ids = tokenizer(seq, return_tensors="pt").to(
                        self.device
                    )["input_ids"]
                    output = model.generate(
                        input_ids,
                        seq_len=n_tokens,
                        temperature=temperature,
                        filter_thres=top_p,
                    )
                    decoded = tokenizer.decode(output.squeeze().cpu().int())
                    formatted_outputs.append({
                        "Prompt": seq,
                        "Output": decoded.replace(" ", ""),
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
                outputs.append({
                    "Prompt": seq,
                    "Output": decoded.replace(" ", ""),
                })
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
        score_type: str = "embedding",
    ) -> dict[Any, Any]:
        # Prepare score sequences
        score_seqs = []
        if isinstance(inputs, DataLoader):
            for data in tqdm(inputs, desc="Scoring"):
                seqs = (
                    data.get("sequence", None)
                    if isinstance(data, dict)
                    else getattr(data, "sequence", None)
                )
                if not seqs:
                    continue
                score_seqs.extend([s for s in seqs if s])
        else:
            score_seqs = inputs
        # Check if model supports scoring
        model_name = str(self.model).lower()
        if "evo2" in model_name:
            outputs = self.model.score_sequences(
                score_seqs, reduce_method=reduce_method
            )
            outputs = [
                {"Input": score_seqs[i], "Score": s}
                for i, s in enumerate(outputs)
            ]
            return outputs
        elif "evo1" in model_name:
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
        elif "megadna" in model_name:
            model = self.model
            tokenizer = self.tokenizer
            outputs = []
            for seq in score_seqs:
                input_ids = tokenizer(seq, return_tensors="pt").to(
                    self.device
                )["input_ids"]
                with torch.no_grad():
                    loss = model(input_ids, return_value="loss")
                outputs.append({"Input": seq, "Score": loss})
            return outputs

        # General scoring for other base models (No classification head)
        # Use batch_infer to get embeddings and compute scores
        if isinstance(inputs, list):
            _, dataloader = self.generate_dataset(
                inputs, batch_size=self.pred_config.batch_size
            )
        else:
            dataloader = inputs
        all_logits, _, embeddings = self.batch_infer(
            dataloader,
            output_hidden_states=True if score_type == "embedding" else False,
            do_pred=False,
        )
        # Prepare logits list for scoring
        logits_list = []
        if isinstance(all_logits, torch.Tensor):
            # assume shape (N, L, V)
            for i in range(all_logits.size(0)):
                logits_list.append(all_logits[i].detach().cpu())
        elif isinstance(all_logits, (list, tuple)):
            for item in all_logits:
                logits_list.append(
                    item.detach().cpu()
                    if isinstance(item, torch.Tensor)
                    else torch.tensor(item)
                )
        else:
            logits_list = []
        # Compute scores
        scores = []
        if "hidden_states" in embeddings and score_type == "embedding":
            hidden_states = embeddings["hidden_states"]
            for i in range(len(score_seqs)):
                # (layers, seq_len, dim)
                seq_hidden = torch.stack([h[i] for h in hidden_states], dim=0)
                if reduce_method == "mean":
                    score = seq_hidden.mean().item()
                elif reduce_method == "max":
                    score = seq_hidden.max().item()
                elif reduce_method == "min":
                    score = seq_hidden.min().item()
                else:
                    score = seq_hidden.mean().item()
                scores.append({"Input": score_seqs[i], "Score": score})
        elif logits_list and score_type == "logits":
            # use logits as scores
            for i in range(len(score_seqs)):
                logits = logits_list[i]
                if logits.dim() == 3:
                    logits = logits.squeeze(0)
                if reduce_method == "mean":
                    score = logits.mean().item()
                elif reduce_method == "max":
                    score = logits.max().item()
                elif reduce_method == "min":
                    score = logits.min().item()
                else:
                    score = logits.mean().item()
                scores.append({"Input": score_seqs[i], "Score": score})
        elif logits_list and score_type == "probability":
            tokenizer = self.tokenizer
            for i, seq in enumerate(score_seqs):
                logits = logits_list[i].to(self.device)  # (L, V) or (1,L,V)
                if logits.dim() == 3 and logits.size(0) == 1:
                    logits = logits.squeeze(0)
                # Re-tokenize the sequence to obtain input_ids & attention_mask
                enc = tokenizer(seq, return_tensors="pt", padding=False)
                input_ids = enc["input_ids"].to(self.device)  # (1, L)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                # Decide model type by presence of mask token
                mask_token_id = getattr(tokenizer, "mask_token_id", None)
                is_mlm = False
                if mask_token_id is not None:
                    if (input_ids == mask_token_id).any():
                        is_mlm = True
                with torch.no_grad():
                    # (L, V)
                    logprobs = torch.log_softmax(
                        logits.to(self.device), dim=-1
                    )
                tgt_ids = input_ids.squeeze(0)
                if is_mlm:
                    # MLM: logits[t] predicts token at t
                    # Gather token logprobs at each position
                    gathered = logprobs.gather(
                        1, tgt_ids.unsqueeze(-1)
                    ).squeeze(-1)  # (L,)
                    # only keep positions where mask appears
                    mask_positions = tgt_ids == mask_token_id
                    if mask_positions.any():
                        token_logprobs = gathered[mask_positions]
                    else:
                        # No explicit mask found; fall back to using all tokens
                        token_logprobs = gathered
                else:
                    # Causal LM: logits[t] predicts token at t+1
                    # Align: drop last logit, drop first input id
                    if logprobs.size(0) >= 2 and tgt_ids.size(0) >= 2:
                        lp = logprobs[:-1, :]  # (L-1, V)
                        tgt = tgt_ids[1:]  # (L-1,)
                        gathered = lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
                        # apply attention_mask if available (exclude padding)
                        if attention_mask is not None:
                            attn = attention_mask.squeeze(0)[1:].to(torch.bool)
                            if attn.any():
                                token_logprobs = gathered[attn]
                            else:
                                token_logprobs = gathered
                        else:
                            token_logprobs = gathered
                    else:
                        # fallback: gather directly (if short)
                        gathered = logprobs.gather(
                            1, tgt_ids.unsqueeze(-1)
                        ).squeeze(-1)
                        token_logprobs = gathered
                if token_logprobs.numel() == 0:
                    score = float("nan")  # no tokens to score
                else:
                    if reduce_method == "mean":
                        score = float(token_logprobs.mean().item())
                    elif reduce_method == "sum":
                        score = float(token_logprobs.sum().item())
                    elif reduce_method == "max":
                        score = float(token_logprobs.max().item())
                    elif reduce_method == "min":
                        score = float(token_logprobs.min().item())
                    else:
                        score = float(token_logprobs.mean().item())

                scores.append({"Input": seq, "Score": score})

            return scores

        return {}

    def get_embeddings(
        self,
        inputs: DataLoader | list[str] | str,
        do_reduce: bool = False,
        reduce_strategy: str | int = "mean",
        force: bool = False,
    ):
        """Get embeddings from the last inference.
        This method performs inference on the provided inputs and extracts
        embeddings from the model's hidden states.

        Args:
            inputs: DataLoader or list of sequences for inference
            do_reduce: Whether to reduce hidden states to 2D using PCA
            reduce_strategy: Strategy to reduce hidden states
                ('mean', 'max', 'min', or int for center window size)
            force: Whether to force re-computation of embeddings

        Returns:
            Dict containing embeddings from the last inference
        """
        # Initialize embeddings
        if force or not hasattr(self, "embeddings"):
            self.embeddings = {"hidden_states": None, "attention_mask": None}
        # Check specific models
        is_special = ""
        special_list = ["CustomEvo", "MEGADNA"]
        for name in special_list:
            if name in str(self.model):
                sequences = inputs
                is_special = name
                break
        if not is_special:
            if isinstance(inputs, list):
                _, dataloader = self.generate_dataset(
                    inputs, batch_size=self.pred_config.batch_size
                )
            elif isinstance(inputs, str):
                # Assume it's a file path
                if os.path.isfile(inputs):
                    file_path = os.path.abspath(inputs)
                else:
                    raise ValueError(
                        f"Input {inputs} is not a valid file path. "
                        "Please provide a valid file path "
                        "or a list contains valid sequences."
                    )
                _, dataloader = self.generate_dataset(
                    file_path,
                    do_encode=True,
                    batch_size=self.pred_config.batch_size,
                )
            else:
                dataloader = inputs

        # Check if model supports generation
        if is_special.startswith("CustomEvo"):
            # Get model and tokenizer
            model = self.model.model
            tokenizer = self.tokenizer
            # Get layer names
            layers = []
            layer_prefix = "blocks"
            for name, _ in model.named_parameters():
                if name.startswith(layer_prefix):
                    layer = layer_prefix + "." + name.split(".")[1]
                    if layer not in layers:
                        layers.append(layer)
            # Get embeddings
            all_embeddings = [[] for _ in layers]
            for sequence in tqdm(sequences):
                input_ids = (
                    torch.tensor(
                        tokenizer.tokenize(sequence),
                        dtype=torch.int,
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                _, embeddings = self.model(
                    input_ids, return_embeddings=True, layer_names=layers
                )
                for i, n in enumerate(layers):
                    tmp = embeddings[n].detach().cpu().to(torch.float32)
                    if do_reduce:
                        mean_emb = _compute_mean_embeddings(tmp, None).squeeze(
                            0
                        )
                        all_embeddings[i].append(mean_emb)
                    else:
                        all_embeddings[i].append(tmp)
            for i, _ in enumerate(layers):
                all_embeddings[i] = np.stack(all_embeddings[i], axis=0)
            if self.embeddings["hidden_states"] is None:
                self.embeddings["hidden_states"] = all_embeddings
            return all_embeddings

        elif is_special == "MEGADNA":
            model = self.model
            tokenizer = self.tokenizer
            all_embeddings = [None] * 3
            out_embeddings = []
            for sequence in tqdm(sequences):
                input_ids = tokenizer(sequence, return_tensors="pt").to(
                    self.device
                )["input_ids"]
                if not isinstance(input_ids, torch.LongTensor):
                    input_ids = input_ids.long()
                with torch.no_grad():
                    embeddings = model(input_ids, return_value="embedding")
                for i in range(len(embeddings)):
                    if all_embeddings[i] is None:
                        all_embeddings[i] = []
                    all_embeddings[i].append(embeddings[i].detach().cpu())
                out_embeddings = all_embeddings
            for i in range(len(all_embeddings)):
                emb = (
                    np.stack(all_embeddings[i], axis=0)
                    if i > 0
                    else np.concatenate(all_embeddings[i], axis=0)
                )
                reshaped_emb = emb.reshape(emb.shape[0], -1, emb.shape[-1])
                if do_reduce:
                    mean_emb = _compute_mean_embeddings(reshaped_emb, None)
                else:
                    mean_emb = reshaped_emb
                # proj_emb = torch.nn.Linear(mean_emb.shape[-1], 128)
                all_embeddings[i] = mean_emb
            # Save embeddings
            if self.embeddings["hidden_states"] is None:
                self.embeddings["hidden_states"] = all_embeddings
            return out_embeddings

        if (
            "hidden_states" not in self.embeddings
            or self.embeddings["hidden_states"] is None
        ):
            _, _, embeddings = self.batch_infer(
                dataloader,
                do_pred=False,
                output_hidden_states=True,
                reduce_hidden_states=do_reduce,
                reduce_strategy=reduce_strategy,
            )
            self.embeddings = embeddings
        return self.embeddings["hidden_states"]


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
