"""DNA Model loading and management utilities.

This module provides functions for downloading, loading, and
    managing DNA language models
from various sources including Hugging Face Hub, ModelScope, and local storage.
"""
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import os
import time
from glob import glob
from typing import Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from ..configuration.configs import TaskConfig
from ..utils import get_logger
from .special import (
    _handle_dnabert2_models,
    _handle_evo1_models,
    _handle_evo2_models,
    _handle_gpn_models,
    _handle_lucaone_models,
    _handle_megadna_models,
    _handle_mutbert_tokenizer,
    _handle_omnidna_models,
    _handle_enformer_models,
    _handle_space_models,
    _handle_borzoi_models,
    _handle_basenji2_tokenizer,
)
from .head import (
    BasicMLPHead,
    BasicCNNHead,
    BasicLSTMHead,
    BasicUNet1DHead,
    MegaDNAMultiScaleHead,
    EVOForSeqClsHead,
)


logger = get_logger("dnallm.models.model")


class DNALLMforSequenceClassification(PreTrainedModel):
    """
    An automated wrapper that selects an appropriate pooling strategy
    based on the underlying model architecture and appends a customizable
    MLP head for sequence classification or regression tasks.
    """

    config_class = AutoConfig

    def __init__(self, config, custom_model=None):
        super().__init__(config)
        from transformers import AutoModel

        if self.config.head_config.get("head", "").lower() == "megadna":
            self.backbone = custom_model
            self.score = MegaDNAMultiScaleHead(**self.config.head_config)
        elif self.config.head_config.get("head", "").lower().startswith("evo"):
            self.backbone = custom_model
            self.score = EVOForSeqClsHead(
                **self.config.head_config,
                base_model=custom_model,
            )
        elif "lucaone" in self.config.head_config.get("head", "").lower():
            from lucagplm import LucaGPLMModel

            self.backbone = LucaGPLMModel(config)
            transformer_output_dim = self.config.hidden_size
            classifier = self._determine_classifier()
            self.score = classifier(
                input_dim=transformer_output_dim, **self.config.head_config
            )
        else:
            import inspect

            self.backbone = AutoModel.from_config(
                config, trust_remote_code=True
            )
            forward_signature = inspect.signature(self.backbone.forward)
            self._backbone_supported_args = set(
                forward_signature.parameters.keys()
            )
            if hasattr(self.backbone.config, "hidden_size"):
                transformer_output_dim = self.backbone.config.hidden_size
            elif hasattr(self.backbone.config, "d_model"):
                transformer_output_dim = self.backbone.config.d_model
            else:
                raise ValueError(
                    "Cannot determine transformer output dimension. "
                    "Please specify 'input_dim' in head_config."
                )
            classifier = self._determine_classifier()
            self.score = classifier(
                input_dim=transformer_output_dim, **self.config.head_config
            )
        self.num_labels = self.config.num_labels
        # determine pooling strategy if not set
        self.pooling_strategy = self._determine_pooling_strategy()
        logger.info(f"Using {self.pooling_strategy} pooling strategy.")

        if self.config.head_config.get("frozen", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.post_init()

    @classmethod
    def from_base_model(cls, model_name_or_path: str, config, module=None):
        """
        Handles weights diffusion when loading a model from
        a pre-trained base model.
        """
        from transformers import AutoModel

        # 1. Use config to create an instance of our custom class.
        model = cls(config)
        # 2. Load the base pre-trained model separately.
        if module is not None:
            base_model = module.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        # 3. Assign the loaded weights to our backbone.
        model.backbone.load_state_dict(base_model.state_dict())

        return model

    def _determine_classifier(self):
        if (
            hasattr(self.config.head_config, "custom_head")
            and self.config.head_config["custom_head"] is not None
        ):
            # Use the custom head class provided in the config
            classifier = self.config.head_config["custom_head"]
        elif self.config.head_config.get("head", "").lower().endswith("mlp"):
            classifier = BasicMLPHead
        elif self.config.head_config.get("head", "").lower().endswith("cnn"):
            classifier = BasicCNNHead
        elif self.config.head_config.get("head", "").lower().endswith("lstm"):
            classifier = BasicLSTMHead
        elif self.config.head_config.get("head", "").lower().endswith("unet"):
            classifier = BasicUNet1DHead
        return classifier

    def _determine_pooling_strategy(self):
        if self.config.head_config.get("pooling_strategy") is not None:
            return self.config.head_config["pooling_strategy"]
        if getattr(self.backbone.config, "is_decoder", False):
            return "last"
        if hasattr(self.config, "cls_token_id"):
            if self.config.cls_token_id is not None:
                return "cls"
        if hasattr(self.config, "cls_idx"):
            if self.config.cls_idx is not None:
                return "cls"
        logger.warning(
            "Warning: Could not determine model type, "
            "falling back to 'mean' pooling."
        )
        return "mean"

    def _get_sentence_embedding(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            expanded_mask = attention_mask.unsqueeze(-1).expand(
                last_hidden_state.size()
            )
            masked_sum = torch.sum(last_hidden_state * expanded_mask, 1)
            actual_lengths = torch.clamp(expanded_mask.sum(1), min=1e-9)
            return masked_sum / actual_lengths
        elif self.pooling_strategy == "max":
            masked_hidden_state = last_hidden_state.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), -float("inf")
            )
            return masked_hidden_state.max(dim=1).values
        elif self.pooling_strategy == "last":
            batch_size = last_hidden_state.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                batch_size, device=last_hidden_state.device
            )
            return last_hidden_state[batch_indices, sequence_lengths, :]
        elif self.pooling_strategy == "first":
            return last_hidden_state[:, 0, :]
        else:
            raise ValueError(
                "Internal error: "
                f"Unsupported pooling strategy '{self.pooling_strategy}'"
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        if kwargs.get("attention_mask") is not None:
            attention_mask = kwargs.get("attention_mask")
        else:
            pad_token_id = self.backbone.config.pad_token_id
            attention_mask = (input_ids != pad_token_id).long()
        if self.config.head_config.get("head", "").lower() == "megadna":
            # convert input_ids to torch.longtensor if not already
            if not isinstance(input_ids, torch.LongTensor):
                input_ids = input_ids.long()
            outputs = self.backbone(input_ids, return_value="embedding")
            last_hidden_state = outputs
        elif self.config.head_config.get("head", "").lower().startswith("evo"):
            outputs = self.backbone(
                input_ids,
                return_embeddings=True,
                layer_names=self.score.target_layers,
            )
            last_hidden_state = outputs[1]
        else:
            # Keep kwargs in the backbone's forward method
            backbone_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in self._backbone_supported_args
            }
            outputs = self.backbone(
                input_ids=input_ids,
                **backbone_kwargs,
            )
            if isinstance(outputs, dict) or hasattr(
                outputs, "last_hidden_state"
            ):
                last_hidden_state = outputs.last_hidden_state
            elif "last_hidden_state" in outputs:
                last_hidden_state = outputs["last_hidden_state"]
            else:
                last_hidden_state = outputs[0]
                if isinstance(last_hidden_state, tuple):
                    last_hidden_state = last_hidden_state[-1]
        # Get sentence embedding if needed
        if self.config.head_config.get("head", "").lower().endswith("mlp"):
            sentence_embedding = self._get_sentence_embedding(
                last_hidden_state, attention_mask
            )
        else:
            sentence_embedding = last_hidden_state
        logits = self.score(sentence_embedding)

        loss = None
        if labels is not None:
            loss_fct = None
            # Allow other loss functions that user selected or provided
            if self.config.head_config.get("loss_function") is not None:
                loss_fct = self.config.head_config["loss_function"]

                class FocalLoss(nn.Module):
                    def __init__(
                        self,
                        alpha=0.25,
                        gamma=2.0,
                        reduction="mean",
                    ):
                        super().__init__()
                        self.alpha = alpha  # controls class imbalance
                        self.gamma = gamma  # focuses on hard examples
                        self.reduction = reduction

                    def forward(self, inputs, targets):
                        # Calculate Binary Cross-Entropy Loss for each sample
                        bce_loss = (
                            nn.functional.binary_cross_entropy_with_logits(
                                inputs, targets, reduction="none"
                            )
                        )
                        # Compute pt (model confidence on true class)
                        pt = torch.exp(-bce_loss)
                        # Apply the focal adjustment
                        focal_loss = (
                            self.alpha * (1 - pt) ** self.gamma * bce_loss
                        )
                        # Apply reduction (mean, sum, or no reduction)
                        if self.reduction == "mean":
                            return focal_loss.mean()
                        elif self.reduction == "sum":
                            return focal_loss.sum()
                        else:
                            return focal_loss

                if isinstance(loss_fct, str):
                    loss_fn_kwargs = self.config.head_config.get(
                        "loss_function_kwargs", {}
                    )
                    if loss_fct.lower() == "mse":
                        loss_fct = nn.MSELoss()
                    elif loss_fct.lower() == "crossentropy":
                        loss_fct = nn.CrossEntropyLoss()
                    elif loss_fct.lower() == "bce":
                        loss_fct = nn.BCELoss()
                    elif loss_fct.lower() == "bcewithlogits":
                        loss_fct = nn.BCEWithLogitsLoss()
                    elif loss_fct.lower() == "focal":
                        loss_fct = FocalLoss(**loss_fn_kwargs)
                    elif loss_fct.lower() == "poisson":
                        loss_fct = nn.PoissonNLLLoss(**loss_fn_kwargs)
                    elif loss_fct.lower() == "cosine_similarity":
                        # Cosine Similarity Loss
                        loss_fct = nn.CosineEmbeddingLoss(**loss_fn_kwargs)
                    else:
                        raise ValueError(
                            f"Unsupported loss function: {loss_fct}"
                        )
                elif isinstance(loss_fct, nn.Module):
                    pass
                else:
                    raise ValueError(
                        "Loss function must be a string or "
                        "an nn.Module instance."
                    )
            if self.score.task_type == "regression":
                loss_fct = nn.MSELoss() if loss_fct is None else loss_fct
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.score.task_type in ["binary", "multiclass"]:
                loss_fct = (
                    nn.CrossEntropyLoss() if loss_fct is None else loss_fct
                )
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.score.task_type == "multilabel":
                loss_fct = (
                    nn.BCEWithLogitsLoss() if loss_fct is None else loss_fct
                )
                loss = loss_fct(logits, labels)

        # Expected output format for Trainer
        if output_hidden_states:
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, "encoder_hidden_states"):
                hidden_states = outputs.encoder_hidden_states
            elif hasattr(outputs, "decoder_hidden_states"):
                hidden_states = outputs.decoder_hidden_states
            elif len(outputs) > 1:
                hidden_states = outputs[1]
            else:
                hidden_states = None
        else:
            hidden_states = None
        if output_attentions:
            if hasattr(outputs, "attentions"):
                attentions = outputs.attentions
            else:
                attentions = None
        else:
            attentions = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


def download_model(
    model_name: str,
    downloader: Any,
    revision: str | None = None,
    max_try: int = 10,
) -> str:
    """Download a model with retry mechanism for network issues.

    In case of network issues, this function will attempt to download the model
    multiple times before giving up.

    Args:
        model_name: Name of the model to download
        downloader: Download function to use (e.g., snapshot_download)
        max_try: Maximum number of download attempts, default 10

    Returns:
        Path where the model files are stored

    Raises:
        ValueError: If model download fails after all attempts
    """
    # In case network issue, try to download multi-times
    cnt = 0
    # init download status
    status = "incomplete"
    while True:
        if cnt >= max_try:
            break
        cnt += 1
        try:
            status = downloader(model_name, revision=revision)
            if status != "incomplete":
                logger.info(f"Model files are stored in {status}")
                break
        # track the error
        except Exception as e:
            # network issue
            if "connection" in str(e):
                reason = "unstable network connection."
            # model not found in HuggingFace
            elif "not found" in str(e).lower():
                reason = "repo is not found."
                logger.debug(e)
                break
            # model not exist in ModelScope
            elif "response [404]" in str(e).lower():
                reason = "repo is not existed."
                logger.debug(e)
                break
            else:
                reason = str(e)
                if "no revision" in reason.lower():
                    revision = None
            logger.warning(f"Retry: {cnt}, Status: {status}, Reason: {reason}")
            time.sleep(1)

    if status == "incomplete":
        raise ValueError(f"Model {model_name} download failed.")

    return status


def _setup_huggingface_mirror(use_mirror: bool) -> None:
    """Configure HuggingFace mirror settings.

    Args:
        use_mirror: Whether to use HuggingFace mirror
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Using HuggingFace mirror at hf-mirror.com")
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def _get_model_path_and_imports(
    model_name: str, source: str, revision: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Get model path and import the required libraries based on source.

    Args:
        model_name: Model name or path
        source: Source to load model from (
                'local',
                'huggingface',
                'modelscope')
        revision: Specific model revision (branch, tag, commit),
                  default None

    Returns:
        Tuple of (model_path, imported_modules_dict)

    Raises:
        ValueError: If local model not found or unsupported source
    """
    source_lower = source.lower()

    if source_lower == "local":
        if not os.path.exists(model_name):
            raise ValueError(f"Model {model_name} not found locally.")
        model_path = model_name

    elif source_lower == "huggingface":
        from huggingface_hub import snapshot_download as hf_snapshot_download

        model_path = download_model(
            model_name, downloader=hf_snapshot_download, revision=revision
        )

    elif source_lower == "modelscope":
        from modelscope.hub.snapshot_download import (
            snapshot_download as ms_snapshot_download,
        )

        model_path = download_model(
            model_name, downloader=ms_snapshot_download, revision=revision
        )

        # Import ModelScope modules
        try:
            from modelscope import (
                AutoConfig,
                AutoModel,
                AutoModelForMaskedLM,
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "ModelScope is required but not available. "
                "Please install it with 'pip install modelscope'."
            ) from e

        modules = {
            "AutoConfig": AutoConfig,
            "AutoModel": AutoModel,
            "AutoModelForMaskedLM": AutoModelForMaskedLM,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSequenceClassification": (
                AutoModelForSequenceClassification
            ),
            "AutoModelForTokenClassification": AutoModelForTokenClassification,
            "AutoTokenizer": AutoTokenizer,
        }

        return model_path, modules

    else:
        raise ValueError(f"Unsupported source: {source}")

    # Import transformers modules for local and huggingface sources
    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )
    except ImportError as e:
        raise ImportError(
            "Transformers is required but not available. "
            "Please install it with 'pip install transformers'."
        ) from e

    modules = {
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": (
            AutoModelForSequenceClassification
        ),
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
    }

    return model_path, modules


def _create_label_mappings(
    task_config: TaskConfig,
) -> tuple[dict[int, str], dict[str, int]]:
    """Create label mappings for classification tasks.

    Args:
        task_config: Task configuration object

    Returns:
        Tuple of (id2label, label2id) mappings
    """
    label_names = task_config.label_names
    if label_names is None:
        # Default empty mappings for tasks without labels
        return {}, {}
    id2label = dict(enumerate(label_names))
    label2id = {label: i for i, label in enumerate(label_names)}
    return id2label, label2id


def _load_model_by_task_type(
    task_type: str,
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    modules: dict[str, Any],
    head_config: dict | None = None,
    custom_tokenizer: Any = None,
) -> tuple[Any, Any]:
    """Load model and tokenizer based on task type.

    Args:
        task_type: Type of task (mask, generation, binary, etc.)
        model_name: Model name or path
        num_labels: Number of labels for classification tasks
        id2label: ID to label mapping
        label2id: Label to ID mapping
        modules: Dictionary of imported model classes
        head_config: Additional head configuration (if any)

    Returns:
        Tuple of (model, tokenizer)
    """
    auto_tokenizer = modules["AutoTokenizer"]

    # Common tokenizer loading
    if custom_tokenizer is None:
        try:
            if task_type == "token":
                tokenizer = auto_tokenizer.from_pretrained(
                    model_name, trust_remote_code=True, add_prefix_space=True
                )
            else:
                tokenizer = auto_tokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
        except Exception:
            logger.warning(
                "Failed to load tokenizer from pretrained model. "
                "Falling back to custom DNAOneHotTokenizer."
            )
            from .tokenizer import DNAOneHotTokenizer

            tokenizer = DNAOneHotTokenizer()
    else:
        tokenizer = custom_tokenizer()

    # Custom model with specific head
    if head_config is not None:
        head_config = head_config.__dict__
        base_config = modules["AutoConfig"].from_pretrained(
            model_name, trust_remote_code=True
        )
        model_config = base_config
        model_config.head_config = head_config
        if hasattr(tokenizer, "cls_token_id"):
            model_config.cls_token_id = tokenizer.cls_token_id
        if hasattr(tokenizer, "cls_idx"):
            model_config.cls_idx = tokenizer.cls_idx
        model = DNALLMforSequenceClassification.from_base_model(
            model_name, config=model_config
        )
        return model, tokenizer

    # Model loading based on task type
    if task_type == "mask":
        model = modules["AutoModelForMaskedLM"].from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
    elif task_type == "generation":
        model = modules["AutoModelForCausalLM"].from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
        )
    elif task_type in ["binary", "multiclass"]:
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "multilabel":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "regression":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif task_type == "token":
        model = modules["AutoModelForTokenClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        try:
            model = modules["AutoModel"].from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception:
            model = modules["AutoModel"].from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                ignore_mismatched_sizes=True,
            )

    return model, tokenizer


def _configure_model_padding(model, tokenizer) -> None:
    """Configure model padding token if not set.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _get_device() -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _safe_num_labels(num_labels: int | None, task_type: str) -> int:
    """Ensure num_labels is at least 2 for classification tasks.

    Args:
        num_labels: Original number of labels

    Returns:
        Safe number of labels (at least 2)
    """
    # Ensure num_labels is not None for classification tasks
    if num_labels is None and task_type in [
        "binary",
        "multiclass",
        "multilabel",
        "regression",
        "token",
    ]:
        raise ValueError(
            f"num_labels is required for task type '{task_type}' but is None"
        )

    # Use default value if num_labels is None for other tasks
    safe_num_labels = num_labels if num_labels is not None else 1
    # num_labels check for non-binary classification tasks
    if task_type == "regression" and safe_num_labels != 1:
        logger.warning(
            f"Regression task typically has num_labels=1, "
            f"but got {safe_num_labels}. Maybe multi-regression task."
        )
    elif task_type == "generation" and safe_num_labels != 0:
        logger.warning(
            f"Generation task does not require num_labels, "
            f"but got {safe_num_labels}. Setting to 0."
        )
        safe_num_labels = 0
    elif task_type == "mask" and safe_num_labels != 0:
        logger.warning(
            f"Mask task does not require num_labels, "
            f"but got {safe_num_labels}. Setting to 0."
        )
        safe_num_labels = 0
    elif task_type == "embedding" and safe_num_labels != 0:
        logger.warning(
            f"Embedding task does not require num_labels, "
            f"but got {safe_num_labels}. Setting to 0."
        )
        safe_num_labels = 0
    if task_type not in [
        "binary",
        "regression",
        "generation",
        "mask",
        "embedding",
    ]:
        if safe_num_labels < 2:
            raise ValueError(
                f"num_labels should be at least 2 for task type "
                f"'{task_type}', but got {safe_num_labels}."
            )

    return safe_num_labels


def load_model_and_tokenizer(
    model_name: str,
    task_config: TaskConfig,
    source: str = "local",
    use_mirror: bool = False,
    revision: str | None = None,
    custom_tokenizer: Any = None,
) -> tuple[Any, Any]:
    """Load model and tokenizer from either HuggingFace or ModelScope.

    This function handles loading of various model types based on the task
        configuration,
            including sequence classification, token classification,
            masked language modeling,
        and causal language modeling.

        Args:
            model_name: Model name or path
            task_config: Task configuration object containing task type and
                label information
                    source: Source to load model and tokenizer from (
                'local',
                'huggingface',
                'modelscope'),
                default 'local'
                    use_mirror: Whether to use HuggingFace mirror (
                hf-mirror.com),
                default False

        Returns:
            Tuple containing (model, tokenizer)

        Raises:
            ValueError: If model is not found locally or loading fails
    """
    # Setup HuggingFace mirror if needed
    _setup_huggingface_mirror(use_mirror)

    # Extract task configuration
    task_type = task_config.task_type
    if hasattr(task_config, "head_config"):
        head_config = task_config.head_config
    else:
        head_config = None
    num_labels = task_config.num_labels
    safe_num_labels = _safe_num_labels(num_labels, task_type)

    # Handle special case for EVO2 models
    evo2_result = _handle_evo2_models(model_name, source, head_config)
    if evo2_result is not None:
        return evo2_result

    # Handle special case for EVO1 models
    evo1_result = _handle_evo1_models(model_name, source, head_config)
    if evo1_result is not None:
        return evo1_result

    # Handle special case for GPN models
    _ = _handle_gpn_models(model_name)

    # Handle special case for megaDNA models
    megadna_result = _handle_megadna_models(model_name, source, head_config)
    if megadna_result is not None:
        return megadna_result

    # Handle special case for LucaOne models
    lucaone_result = _handle_lucaone_models(model_name, source, head_config)
    if lucaone_result is not None:
        return lucaone_result

    # Handle special case for Omni-DNA models
    _ = _handle_omnidna_models(model_name)

    # Handle special case for Enformer models
    enformer_result = _handle_enformer_models(
        model_name,
        source,
        task_type,
        safe_num_labels,
        extra=model_name if "enformer" in model_name.lower() else None,
    )
    if enformer_result is not None:
        return enformer_result

    # Handle special case for SPACE models
    space_result = _handle_space_models(
        model_name,
        source,
        task_type,
        safe_num_labels,
        extra=model_name if "space" in model_name.lower() else None,
    )
    if space_result is not None:
        return space_result

    # Handle special case for Borzoi models
    borzoi_result = _handle_borzoi_models(
        model_name,
        source,
        task_type,
        safe_num_labels,
        extra=model_name if "borzoi" in model_name.lower() else None,
    )
    if borzoi_result is not None:
        return borzoi_result

    # TODO: Add more special cases if needed

    # Get model path and import required modules
    downloaded_model_path, modules = _get_model_path_and_imports(
        model_name, source, revision=revision
    )
    if hasattr(task_config, "head_config"):
        model_name = downloaded_model_path

    # Create label mappings
    id2label, label2id = _create_label_mappings(task_config)

    # Load model and tokenizer based on task type
    try:
        load_args = [
            task_type,
            model_name,
            safe_num_labels,
            id2label,
            label2id,
            modules,
            head_config,
            custom_tokenizer,
        ]
        model, tokenizer = _handle_dnabert2_models(
            downloaded_model_path, load_args
        )
        if model is None or tokenizer is None:
            model, tokenizer = _load_model_by_task_type(*load_args)
        # Process model with custom tokenizer if needed
        if "mutbert" in downloaded_model_path.lower():
            tokenizer = _handle_mutbert_tokenizer(tokenizer)
        if "basenji2" in downloaded_model_path.lower():
            tokenizer = _handle_basenji2_tokenizer(tokenizer)
        # Set model path and source attributes
        model._model_path = downloaded_model_path
        model.source = source
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}") from e

    # Configure model padding
    _configure_model_padding(model, tokenizer)
    model = model.to(_get_device())

    return model, tokenizer


def peft_forward_compatiable(model: Any) -> Any:
    """Convert base model forward to be compatiable with HF

    Args:
        model: Base model

    Returns:
        model with changed forward function
    """
    import inspect

    sig = inspect.signature(model.forward)
    accepted_forward_args = set(sig.parameters.keys())
    original_forward = model.forward

    def forward_hf(*args, **kwargs):
        return original_forward(**{
            k: v for k, v in kwargs.items() if k in accepted_forward_args
        })

    model.forward = forward_hf
    return model


def clear_model_cache(source: str = "huggingface"):
    """Remove all the cached models

    Args:
        source: Source to clear model cache from (
                'huggingface',
                'modelscope'),
            default 'huggingface'
    """
    source_lower = source.lower()
    if source_lower == "huggingface":
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/hub"
        )
    elif source_lower == "modelscope":
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache/modelscope/hub"
        )
    else:
        logger.warning(f"Unsupported source: {source}. No action taken.")
        return

    if os.path.exists(cache_dir):
        files = glob(os.path.join(cache_dir, "*"))
        for f in files:
            try:
                if os.path.isdir(f):
                    import shutil

                    shutil.rmtree(f)
                else:
                    os.remove(f)
                logger.info(f"Removed cached file/directory: {f}")
            except Exception as e:
                logger.warning(f"Failed to remove {f}: {e}")
    else:
        logger.info(
            f"No cache directory found at {cache_dir}. Nothing to clear."
        )


def load_preset_model(
    model_name: str, task_config: TaskConfig
) -> tuple[Any, Any] | int:
    """Load a preset model and tokenizer based on the task configuration.

    This function loads models from the preset model registry, which contains
    pre-configured models for various DNA analysis tasks.

    Args:
        model_name: Name or path of the model
                task_config: Task configuration object containing task type and
            label information

    Returns:
        Tuple containing (model, tokenizer) if successful, 0 if model not found

    Note:
                If the model is not found in preset models,
            the function will print a warning
                and
            return 0. Use `load_model_and_tokenizer` function for custom model
            loading.
    """
    from .modeling_auto import MODEL_INFO

    source = "modelscope"
    use_mirror = False

    # Load model and tokenizer
    try:
        preset_models = [
            preset
            for model in MODEL_INFO
            for preset in MODEL_INFO[model].get("preset", [])
        ]
    except (KeyError, TypeError):
        preset_models = []
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info["default"]
    elif model_name in preset_models:
        pass
    else:
        logger.debug(
            f"Model {model_name} not found in preset models. "
            "Please check the model name or use "
            "`load_model_and_tokenizer` function."
        )
        return 0
    return load_model_and_tokenizer(
        model_name, task_config, source, use_mirror
    )
