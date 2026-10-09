"""DNA Model loading and management utilities.

This module provides functions for downloading, loading, and
    managing DNA large language models
from various sources including Hugging Face Hub, ModelScope, and local storage.
"""
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import hashlib
import os
import time
from glob import glob
from typing import Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, BitsAndBytesConfig  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
from transformers.modeling_outputs import SequenceClassifierOutput

from ..configuration.configs import TaskConfig
from ..utils import get_logger
from .special import (
    _handle_dnabert2_models,
    _handle_evo1_models,
    _handle_evo2_models,
    _handle_gpn_models,
    # _handle_lucaone_models,
    _handle_megadna_models,
    _handle_mutbert_tokenizer,
    _handle_omnidna_models,
    _handle_enformer_models,
    _handle_space_models,
    _handle_borzoi_models,
    _handle_basenji2_tokenizer,
    _handle_crossdna_models,
)
from .head import (
    BasicMLPHead,
    BasicCNNHead,
    BasicLSTMHead,
    BasicUNet1DHead,
    MegaDNAMultiScaleHead,
    EVOForSeqClsHead,
)
from .losses import FocalLoss


logger = get_logger("dnallm.models.model")


class DNALLMforSequenceClassification(PreTrainedModel):
    """
    An automated wrapper that selects an appropriate pooling strategy
    based on the underlying model architecture and appends a customizable
    MLP head for sequence classification or regression tasks.
    """

    config_class = AutoConfig  # type: ignore[assignment]

    def __init__(self, config, custom_model=None):
        super().__init__(config)
        from transformers import AutoModel  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live

        if self.config.head_config.get("head", "").lower() == "megadna":
            self.backbone = custom_model
            self.score = MegaDNAMultiScaleHead(**self.config.head_config)
        elif self.config.head_config.get("head", "").lower().startswith("evo"):
            self.backbone = custom_model
            self.score = EVOForSeqClsHead(  # type: ignore[assignment]
                **self.config.head_config,
                base_model=custom_model,
            )
        elif "lucaone" in self.config.head_config.get("head", "").lower():
            from lucagplm import LucaGPLMModel  # ty: ignore[unresolved-import]  # optional dep, raise-on-use

            self.backbone = LucaGPLMModel(config)
            transformer_output_dim = self.config.hidden_size
            classifier = self._determine_classifier()
            self.score = classifier(input_dim=transformer_output_dim, **self.config.head_config)
        else:
            import inspect

            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
            forward_signature = inspect.signature(self.backbone.forward)
            self._backbone_supported_args = set(forward_signature.parameters.keys())
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
            self.score = classifier(input_dim=transformer_output_dim, **self.config.head_config)
        self.num_labels = self.config.num_labels
        # determine pooling strategy if not set
        self.pooling_strategy = self._determine_pooling_strategy()
        logger.info(f"Using {self.pooling_strategy} pooling strategy.")

        if self.config.head_config.get("frozen", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.post_init()

    @classmethod
    def from_base_model(
        cls, model_name_or_path: str, config, module=None, quantization_config=None
    ):
        """
        Handles weights diffusion when loading a model from
        a pre-trained base model.
        """
        from transformers import AutoModel  # type: ignore[attr-defined]  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live

        # 1. Use config to create an instance of our custom class.
        model = cls(config)
        # 2. Load the base pre-trained model separately.
        load_kwargs = {"trust_remote_code": True}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"  # type: ignore[assignment]
        if module is not None:
            base_model = module.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            base_model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
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
        else:
            raise ValueError(
                f"Unknown head type {self.config.head_config.get('head')!r}: "
                "expected a name ending in mlp/cnn/lstm/unet or a custom_head class."
            )
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
        logger.warning("Warning: Could not determine model type, falling back to 'mean' pooling.")
        return "mean"

    def _get_sentence_embedding(self, last_hidden_state, attention_mask):
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
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
            batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
            return last_hidden_state[batch_indices, sequence_lengths, :]
        elif self.pooling_strategy == "first":
            return last_hidden_state[:, 0, :]
        else:
            raise ValueError(
                f"Internal error: Unsupported pooling strategy '{self.pooling_strategy}'"
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
            if hasattr(self.backbone.config, "pad_token_id"):
                pad_token_id = self.backbone.config.pad_token_id
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)  # type: ignore
        if self.config.head_config.get("head", "").lower() == "megadna":
            # convert input_ids to torch.longtensor if not already
            if not isinstance(input_ids, torch.LongTensor):
                input_ids = input_ids.long()  # type: ignore
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
                k: v for k, v in kwargs.items() if k in self._backbone_supported_args
            }
            outputs = self.backbone(
                input_ids=input_ids,
                **backbone_kwargs,
            )
            if isinstance(outputs, dict) or hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            elif "last_hidden_state" in outputs:
                last_hidden_state = outputs["last_hidden_state"]
            else:
                last_hidden_state = outputs[0]
                if isinstance(last_hidden_state, tuple):
                    last_hidden_state = last_hidden_state[-1]
        # Get sentence embedding if needed
        if self.config.head_config.get("head", "").lower().endswith("mlp"):
            sentence_embedding = self._get_sentence_embedding(last_hidden_state, attention_mask)
        else:
            sentence_embedding = last_hidden_state
        logits = self.score(sentence_embedding)
        if self.num_labels != logits.size(-1):
            self.num_labels = self.config.head_config["num_classes"]

        loss = None
        if labels is not None:
            loss_fct = None
            # Allow other loss functions that user selected or provided
            if self.config.head_config.get("loss_function") is not None:
                loss_fct = self.config.head_config["loss_function"]

                if isinstance(loss_fct, str):
                    loss_fn_kwargs = self.config.head_config.get("loss_function_kwargs", {})
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
                        raise ValueError(f"Unsupported loss function: {loss_fct}")
                elif isinstance(loss_fct, nn.Module):
                    pass
                else:
                    raise ValueError("Loss function must be a string or an nn.Module instance.")
            if self.score.task_type == "regression":
                loss_fct = nn.MSELoss() if loss_fct is None else loss_fct
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.score.task_type in ["binary", "multiclass"]:
                loss_fct = nn.CrossEntropyLoss() if loss_fct is None else loss_fct
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.score.task_type == "multilabel":
                loss_fct = nn.BCEWithLogitsLoss() if loss_fct is None else loss_fct
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
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a model with retry mechanism for network issues.

    In case of network issues, this function will attempt to download the model
    multiple times before giving up.

    Args:
        model_name: Name of the model to download
        downloader: Download function to use (e.g., snapshot_download)
        max_try: Maximum number of download attempts, default 10
        allow_patterns: Optional glob patterns restricting which snapshot
            files are fetched (e.g. ``["*.safetensors", "*.json"]``). When
            ``None`` (the default) the downloader is called exactly as
            before -- no ``allow_patterns`` key is forwarded, so every
            existing caller behaves byte-identically.

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
            # Conditional forwarding, rebuilt per attempt: the downloader
            # kwarg exists ONLY when the caller explicitly passed a pattern
            # set (CI-05 -- no family may silently inherit another's
            # patterns), and a no-revision retry must observe the reset
            # revision below, not the value bound at loop entry.
            download_kwargs: dict[str, Any] = {"revision": revision}
            if allow_patterns is not None:
                download_kwargs["allow_patterns"] = allow_patterns
            status = downloader(model_name, **download_kwargs)
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
                logger.debug(e)  # type: ignore
                break
            # model not exist in ModelScope
            elif "response [404]" in str(e).lower():
                reason = "repo is not existed."
                logger.debug(e)  # type: ignore
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
    model_name: str,
    source: str,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
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
        allow_patterns: Optional glob patterns restricting which snapshot
            files the hub download fetches. When ``None`` (the default) the
            downloader is called exactly as before; only callers that
            explicitly pass a pattern set (the evo-1 giants family) get a
            restricted snapshot.

    Returns:
        Tuple of (model_path, imported_modules_dict)

    Raises:
        ValueError: If local model not found or unsupported source
    """
    source_lower = source.lower()

    # Conditional forwarding, mirroring download_model: the kwarg reaches the
    # downloader ONLY when the caller explicitly passed a pattern set, so the
    # call signature stays byte-identical for every existing caller.
    hub_kwargs: dict[str, Any] = {"revision": revision}
    if allow_patterns is not None:
        hub_kwargs["allow_patterns"] = allow_patterns

    if source_lower == "local":
        if not os.path.exists(model_name):
            raise ValueError(f"Model {model_name} not found locally.")
        model_path = model_name

    elif source_lower == "huggingface":
        from huggingface_hub import snapshot_download as hf_snapshot_download

        model_path = download_model(model_name, downloader=hf_snapshot_download, **hub_kwargs)

    elif source_lower == "modelscope":
        from modelscope.hub.snapshot_download import (
            snapshot_download as ms_snapshot_download,
        )

        model_path = download_model(model_name, downloader=ms_snapshot_download, **hub_kwargs)

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
            "AutoModelForSequenceClassification": (AutoModelForSequenceClassification),
            "AutoModelForTokenClassification": AutoModelForTokenClassification,
            "AutoTokenizer": AutoTokenizer,
        }

        return model_path, modules

    else:
        raise ValueError(f"Unsupported source: {source}")

    # Import transformers modules for local and huggingface sources
    try:
        from transformers import (  # type: ignore[attr-defined]
            AutoConfig,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoModel,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoModelForMaskedLM,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoModelForCausalLM,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoModelForSequenceClassification,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoModelForTokenClassification,  # ty: ignore[unresolved-import]  # lazy export, resolves live
            AutoTokenizer,  # ty: ignore[unresolved-import]  # lazy export, resolves live
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
        "AutoModelForSequenceClassification": (AutoModelForSequenceClassification),
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
    bnb_config: Any = None,
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
        bnb_config: BitsAndBytesConfig for 4-bit quantization (if any)

    Returns:
        Tuple of (model, tokenizer)
    """
    auto_tokenizer = modules["AutoTokenizer"]

    # Common tokenizer loading
    if custom_tokenizer is None:
        from .tokenizer import load_tokenizer_with_fallback

        tokenizer = load_tokenizer_with_fallback(
            model_name,
            auto_tokenizer_cls=auto_tokenizer,
            add_prefix_space=(task_type == "token"),
        )
    else:
        tokenizer = custom_tokenizer()

    # Build common kwargs for model loading
    model_load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if bnb_config is not None:
        model_load_kwargs["quantization_config"] = bnb_config
        model_load_kwargs["device_map"] = "auto"

    # Custom model with specific head
    if head_config is not None:
        head_config = head_config.__dict__
        base_config = modules["AutoConfig"].from_pretrained(model_name, trust_remote_code=True)
        model_config = base_config
        model_config.head_config = head_config
        if hasattr(tokenizer, "cls_token_id"):
            model_config.cls_token_id = tokenizer.cls_token_id
        if hasattr(tokenizer, "cls_idx"):
            model_config.cls_idx = tokenizer.cls_idx
        model = DNALLMforSequenceClassification.from_base_model(
            model_name,
            config=model_config,
            module=modules["AutoModel"],
            quantization_config=bnb_config,
        )
        return model, tokenizer

    # Model loading based on task type
    if task_type == "mask":
        model = modules["AutoModelForMaskedLM"].from_pretrained(model_name, **model_load_kwargs)
    elif task_type == "generation":
        model = modules["AutoModelForCausalLM"].from_pretrained(model_name, **model_load_kwargs)
    elif task_type in ["binary", "multiclass"]:
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            **model_load_kwargs,
        )
    elif task_type == "multilabel":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            **model_load_kwargs,
        )
    elif task_type == "regression":
        model = modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="regression",
            **model_load_kwargs,
        )
    elif task_type == "token":
        model = modules["AutoModelForTokenClassification"].from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **model_load_kwargs,
        )
    else:
        try:
            model = modules["AutoModel"].from_pretrained(
                model_name,
                **model_load_kwargs,
            )
        except Exception:
            model = modules["AutoModel"].from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                **model_load_kwargs,
            )

    return model, tokenizer


def _configure_model_padding(model, tokenizer) -> None:
    """Configure model padding token if not set.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    if model.config.pad_token_id is None:
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, "pad_token_type_id") and tokenizer.pad_token_type_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_type_id
        else:
            model.config.pad_token_id = tokenizer.eos_token_id


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
        raise ValueError(f"num_labels is required for task type '{task_type}' but is None")

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
            f"Generation task does not require num_labels, but got {safe_num_labels}. Setting to 0."
        )
        safe_num_labels = 0
    elif task_type == "mask" and safe_num_labels != 0:
        logger.warning(
            f"Mask task does not require num_labels, but got {safe_num_labels}. Setting to 0."
        )
        safe_num_labels = 0
    elif task_type == "embedding" and safe_num_labels != 0:
        logger.warning(
            f"Embedding task does not require num_labels, but got {safe_num_labels}. Setting to 0."
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


# ───────────────────────── random-init (BASE-01) ─────────────────────────
# From-scratch baseline loading: load_model_and_tokenizer(..., random_init=True)
# builds a genuinely randomly initialized model through AutoConfig.from_pretrained
# + Auto*.from_config and proves it via logged per-tensor hashes.

RANDOM_INIT_SUPPORTED_FAMILIES: frozenset[str] = frozenset({"mamba"})
"""Special model families sanctioned for ``random_init=True`` (D-07).

The allowlist gates the special ``_handle_*`` families only: a model whose
name marks it as one of the special families (evo2, gpn, enformer, ...) is
rejected with a ``ValueError`` unless its family appears here, because the
special handlers construct models through bespoke paths with no
``from_config`` equivalent. Generic ``Auto*``-loadable models (including
BERT-style and remote-code Mamba checkpoints) are always allowed and need
no entry. ``"mamba"`` is the proven trust_remote_code ``from_config``
member (D-06/A6): Plant DNAMamba models load through the generic
``AutoModelForCausalLM`` path, so the entry documents the sanctioned
remote-code architecture rather than unlocking a special handler.
"""

RANDOM_INIT_DEFAULT_SEED = 42
"""CPU-canonical default seed for from-scratch initialization.

Applied via ``torch.manual_seed`` immediately before ``from_config``
construction; two loads with the same seed produce identical hash tables.
"""

# Name markers mirroring the special handlers' own family detection,
# matched case-insensitively as substrings of the model name. Intentionally
# a superset of the handlers' (partly case-sensitive) matching: a false
# "special" verdict only produces a clear error, never wrong weights.
_SPECIAL_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "evo2": ("evo2",),
    "evo1": ("evo-1", "evo1"),
    "gpn": ("gpn",),
    "megadna": ("megadna",),
    "omnidna": ("omni-dna",),
    "enformer": ("enformer",),
    "space": ("space",),
    "borzoi": ("borzoi",),
    "crossdna": ("crossdna",),
    "dnabert2": ("dnabert-2", "dnabert-s"),
}

# Which Auto* class the pretrained task-type loader selects per task type;
# the random path selects the same class. Task types absent from the mapping
# (embedding and unknown) fall back to the plain AutoModel backbone.
_TASK_AUTO_CLASS_KEYS: dict[str, str] = {
    "mask": "AutoModelForMaskedLM",
    "generation": "AutoModelForCausalLM",
    "binary": "AutoModelForSequenceClassification",
    "multiclass": "AutoModelForSequenceClassification",
    "multilabel": "AutoModelForSequenceClassification",
    "regression": "AutoModelForSequenceClassification",
    "token": "AutoModelForTokenClassification",
}


def _detect_special_family(model_name: str) -> str | None:
    """Return the special-handler family that would claim ``model_name``.

    Mirrors the name matching the ``_handle_*`` chain itself uses (see
    ``_SPECIAL_FAMILY_MARKERS``); ``None`` means the model loads through
    the generic task-type path.

    Args:
        model_name: Model name or path

    Returns:
        Family key (e.g. ``"evo2"``) or ``None`` for generic models.
    """
    lowered = model_name.lower()
    for family, markers in _SPECIAL_FAMILY_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return family
    return None


def _gate_random_init(
    model_name: str,
    quantization_config: dict | None,
    head_config: Any,
) -> None:
    """Reject incoherent random_init loads before any handler runs (D-06/D-07).

    Args:
        model_name: Model name or path
        quantization_config: The caller's quantization config, if any
        head_config: The task config's head config, if any

    Raises:
        ValueError: If the model belongs to a special family outside
            RANDOM_INIT_SUPPORTED_FAMILIES, or if random_init is combined
            with quantization_config or a custom head_config (combinations
            with no from_config equivalent).
    """
    family = _detect_special_family(model_name)
    if family is not None and family not in RANDOM_INIT_SUPPORTED_FAMILIES:
        raise ValueError(
            f"random_init=True is not supported for special model family "
            f"'{family}' (model '{model_name}'): from-scratch initialization "
            f"is only implemented for generic Auto* models and the families "
            f"listed in RANDOM_INIT_SUPPORTED_FAMILIES="
            f"{sorted(RANDOM_INIT_SUPPORTED_FAMILIES)}."
        )
    if quantization_config is not None:
        raise ValueError(
            "random_init=True cannot be combined with quantization_config: "
            "from-scratch initialization builds fresh unquantized weights, "
            "so there is no checkpoint to quantize."
        )
    if head_config is not None:
        raise ValueError(
            "random_init=True is not supported with a custom head_config: "
            "the from-scratch path builds plain Auto* task models. Load "
            "without random_init to use DNALLMforSequenceClassification heads."
        )


def _get_auto_modules_for_source(source: str) -> dict[str, Any]:
    """Import the Auto* class bundle for ``source`` without downloading.

    Mirrors the import half of ``_get_model_path_and_imports`` (same class
    set: transformers for local/huggingface, modelscope for modelscope)
    while skipping the snapshot download entirely. The random_init path
    only ever fetches config.json + tokenizer files via ``from_pretrained``
    on these classes, never weight files.

    Args:
        source: Source to import classes for ('local', 'huggingface',
            'modelscope')

    Returns:
        Dictionary of Auto* classes, same keys as
        ``_get_model_path_and_imports`` returns.

    Raises:
        ValueError: If the source is unsupported.
        ImportError: If the required hub library is not installed.
    """
    if source.lower() == "modelscope":
        try:
            from modelscope import (  # ty: ignore[unresolved-import]  # optional at runtime, guarded
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
    elif source.lower() in ("local", "huggingface"):
        try:
            from transformers import (  # type: ignore[attr-defined]
                AutoConfig,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoModel,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoModelForMaskedLM,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoModelForCausalLM,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoModelForSequenceClassification,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoModelForTokenClassification,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
                AutoTokenizer,  # ty: ignore[unresolved-import]  # transformers lazy export, resolves live
            )
        except ImportError as e:
            raise ImportError(
                "Transformers is required but not available. "
                "Please install it with 'pip install transformers'."
            ) from e
    else:
        raise ValueError(f"Unsupported source: {source}")

    return {
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
    }


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Short sha256 hex over the raw bytes of ``tensor`` (BASE-01 proof).

    The digest is computed over ``detach().cpu().contiguous()`` bytes so it
    is independent of device and autograd state; bfloat16 tensors are
    upcast to float32 first (numpy has no bf16 dtype), which preserves
    byte-level equality comparisons between identical tensors.

    Args:
        tensor: Parameter or buffer tensor to fingerprint.

    Returns:
        First 10 hex characters of the sha256 digest.
    """
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:10]


def _log_random_init_fingerprint(
    model: Any,
    model_name: str,
    auto_class_name: str,
    seed: int,
) -> dict[str, str]:
    """Log the from-scratch banner and per-tensor hash table (D-05).

    One loud INFO banner line followed by one INFO line per parameter
    tensor (tied aliases included via ``remove_duplicate=False``, so
    shared storage is visible as identical hashes — expected, not a
    failure) plus one line per buffer that does not shadow a parameter.
    Non-float buffers are deterministic constants and may match a
    pretrained load; that is documented expected behavior.

    Args:
        model: The freshly constructed random-init model (still on CPU).
        model_name: Model name or path, for the banner.
        auto_class_name: Logical Auto* class name used via from_config.
        seed: The seed that drove the initialization.

    Returns:
        Mapping of tensor name -> 10-hex digest (parameters first, then
        non-parameter buffers), for hash-table comparisons in tests.
    """
    logger.info(
        f"[random-init] Model '{model_name}' randomly initialized from scratch "
        f"via {auto_class_name}.from_config (seed={seed}); no pretrained "
        f"weights were fetched. Per-tensor hashes below are sha256[:10] "
        f"over raw tensor bytes."
    )
    digests: dict[str, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        digests[name] = _tensor_digest(param)
        logger.info(
            f"[random-init] param {name} sha={digests[name]} "
            f"shape={tuple(param.shape)} dtype={param.dtype}"
        )
    for name, buf in model.named_buffers():
        if name in digests:
            continue
        digests[name] = _tensor_digest(buf)
        logger.info(
            f"[random-init] buffer {name} sha={digests[name]} "
            f"shape={tuple(buf.shape)} dtype={buf.dtype}"
        )
    return digests


def _load_random_init_model(
    model_name: str,
    task_type: str,
    safe_num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    source: str,
    revision: str | None,
    seed: int,
    custom_tokenizer: Any = None,
) -> tuple[Any, Any]:
    """Build a genuinely from-scratch model via ``Auto*.from_config`` (BASE-01).

    Only ``config.json`` (and tokenizer files) are ever fetched: the
    weight-download seam ``_get_model_path_and_imports`` is never invoked
    on this path, so a random-init baseline is provably free of pretrained
    weights (the config.json fetch through AutoConfig.from_pretrained is
    explicitly allowed — it carries no weight values).

    Args:
        model_name: Model name or local path (config source)
        task_type: Task type driving the Auto* class selection
        safe_num_labels: Sanitized label count for classification heads
        id2label: ID to label mapping
        label2id: Label to ID mapping
        source: Source to resolve the config/tokenizer from ('local',
            'huggingface', 'modelscope')
        revision: Specific model revision, default None
        seed: CPU-canonical torch seed applied BEFORE construction
        custom_tokenizer: Optional tokenizer factory; None loads the
            model's own tokenizer files

    Returns:
        Tuple of (model, tokenizer) with the model still on CPU (device
        placement happens in the caller, after hashing).
    """
    modules = _get_auto_modules_for_source(source)

    # config.json-only fetch: AutoConfig.from_pretrained downloads the model
    # config (plus remote-code *.py files when trust_remote_code resolves
    # them), never weight files.
    config_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision is not None:
        config_kwargs["revision"] = revision
    config = modules["AutoConfig"].from_pretrained(model_name, **config_kwargs)

    # Head shaping via config attributes, not ctor kwargs (transformers
    # from_config convention) — the same fields the pretrained path passes
    # to from_pretrained for classification task types.
    if task_type in ("binary", "multiclass", "multilabel", "regression", "token"):
        config.num_labels = safe_num_labels
        config.id2label = id2label
        config.label2id = label2id
        config.problem_type = {
            "multilabel": "multi_label_classification",
            "regression": "regression",
        }.get(task_type, "single_label_classification")

    # CPU-canonical seeding BEFORE construction: the model is built on CPU,
    # so the CPU RNG stream fully determines the sampled weights.
    torch.manual_seed(seed)

    auto_class_key = _TASK_AUTO_CLASS_KEYS.get(task_type, "AutoModel")
    auto_class = modules[auto_class_key]
    # from_config only — never from_pretrained on this branch.
    model = auto_class.from_config(config, trust_remote_code=True)

    # Tokenizer loads exactly as on the pretrained path (config/vocab files
    # only). The fingerprint table is logged while the model is still on
    # CPU, before any device move.
    if custom_tokenizer is None:
        from .tokenizer import load_tokenizer_with_fallback

        tokenizer = load_tokenizer_with_fallback(
            model_name,
            auto_tokenizer_cls=modules["AutoTokenizer"],
            add_prefix_space=(task_type == "token"),
        )
    else:
        tokenizer = custom_tokenizer()

    _log_random_init_fingerprint(model, model_name, auto_class_key, seed)
    return model, tokenizer


def load_model_and_tokenizer(
    model_name: str,
    task_config: TaskConfig,
    source: str = "local",
    use_mirror: bool = False,
    revision: str | None = None,
    custom_tokenizer: Any = None,
    quantization_config: dict | None = None,
    random_init: bool = False,
    random_init_seed: int = RANDOM_INIT_DEFAULT_SEED,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
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
            random_init: Build the model from scratch with randomly
                initialized weights (``AutoConfig.from_pretrained`` +
                ``Auto*.from_config``) instead of loading a pretrained
                checkpoint. Only ``config.json`` and tokenizer files are
                fetched — the weight-download path is never invoked. A loud
                "randomly initialized" banner and a per-tensor parameter
                hash table are logged as proof (BASE-01).
            random_init_seed: Seed applied via ``torch.manual_seed``
                immediately before from-scratch construction
                (default RANDOM_INIT_DEFAULT_SEED = 42); identical seeds
                reproduce identical per-tensor hash tables.

        Returns:
            Tuple containing (model, tokenizer)

        Raises:
            ValueError: If model is not found locally or loading fails;
                if ``random_init=True`` is requested for a special model
                family outside RANDOM_INIT_SUPPORTED_FAMILIES, or combined
                with ``quantization_config`` or a custom ``head_config``
    """
    # Setup HuggingFace mirror if needed
    _setup_huggingface_mirror(use_mirror)

    # Prepare quantization config if provided
    bnb_config = None
    if quantization_config is not None:
        bnb_config = BitsAndBytesConfig(**quantization_config)

    # Extract task configuration
    task_type = task_config.task_type
    if hasattr(task_config, "head_config"):
        head_config = task_config.head_config
    else:
        head_config = None
    num_labels = task_config.num_labels
    safe_num_labels = _safe_num_labels(num_labels, task_type)

    # From-scratch gate (BASE-01, D-06/D-07): reject unsupported special
    # families BEFORE any handler claims the load, plus incoherent
    # combinations that have no from_config equivalent.
    if random_init:
        _gate_random_init(model_name, quantization_config, head_config)

    # Handle special case for EVO2 models
    evo2_result = _handle_evo2_models(model_name, source, head_config)  # type: ignore
    if evo2_result is not None:
        return evo2_result

    # Handle special case for EVO1 models
    evo1_result = _handle_evo1_models(model_name, source, head_config)  # type: ignore
    if evo1_result is not None:
        return evo1_result

    # Handle special case for GPN models
    _ = _handle_gpn_models(model_name)

    # Handle special case for megaDNA models
    megadna_result = _handle_megadna_models(model_name, source, head_config)  # type: ignore
    if megadna_result is not None:
        return megadna_result

    # Handle special case for LucaOne models
    # lucaone_result = _handle_lucaone_models(model_name, source, head_config)
    # if lucaone_result is not None:
    #     return lucaone_result

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
        return borzoi_result  # type: ignore

    # TODO: Add more special cases if needed

    if random_init:
        # From-scratch path (BASE-01): never touches the weight-download
        # seam below — config.json + tokenizer files only. The gate above
        # already proved no off-list special family would claim this load,
        # so the special handlers above returned None for it.
        id2label, label2id = _create_label_mappings(task_config)
        try:
            model, tokenizer = _load_random_init_model(
                model_name,
                task_type,
                safe_num_labels,
                id2label,
                label2id,
                source,
                revision,
                random_init_seed,
                custom_tokenizer,
            )
            # Tokenizer post-processing parity with the pretrained chain
            # (the chain's must-not-return-early contract applies here too).
            if "mutbert" in model_name.lower():
                tokenizer = _handle_mutbert_tokenizer(tokenizer)
            if "basenji2" in model_name.lower():
                tokenizer = _handle_basenji2_tokenizer(tokenizer)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e
        model._model_path = model_name
        model.source = source
        _configure_model_padding(model, tokenizer)
        # The model was constructed (and hashed) on CPU; move to the target
        # device only now (tied-weight/meta-device safety).
        model = model.to(_get_device())
        return model, tokenizer

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
            bnb_config,
        ]
        # Guarded dispatch chain (first resolved stage wins): crossdna ->
        # dnabert2 -> generic task-type loader. Each stage runs only when the
        # previous stage left model or tokenizer None, and stages MERGE per
        # half: a stage's None half never overwrites an earlier stage's
        # resolved half, so a handler's partial result is never discarded by
        # a later stage. The chain must NOT return early here: the tokenizer
        # post-processing and attribute/device placement below must still
        # run for a resolved handler result.
        model, tokenizer = None, None
        if "crossdna" in downloaded_model_path.lower():
            model, tokenizer = _handle_crossdna_models(
                task_type,
                downloaded_model_path,
                safe_num_labels,
                id2label,
                label2id,
                modules,
                head_config,
                custom_tokenizer,
                bnb_config,
            )
        if model is None or tokenizer is None:
            stage_model, stage_tokenizer = _handle_dnabert2_models(downloaded_model_path, load_args)
            model = stage_model if stage_model is not None else model
            tokenizer = stage_tokenizer if stage_tokenizer is not None else tokenizer
        if model is None or tokenizer is None:
            stage_model, stage_tokenizer = _load_model_by_task_type(*load_args)
            model = stage_model if stage_model is not None else model
            tokenizer = stage_tokenizer if stage_tokenizer is not None else tokenizer
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
    # Skip device placement for 4-bit models (device_map="auto" handles it)
    if bnb_config is None:
        model = model.to(_get_device())

    # Fix improperly quantized layers (e.g., pooler.dense in BERT)
    # that were newly initialized and not properly packed by bitsandbytes
    if bnb_config is not None:
        _fix_bnb_quantized_layers(model)

    return model, tokenizer


def _fix_bnb_quantized_layers(model: Any) -> None:
    """Fix bitsandbytes quantization issues on newly initialized layers.

    When a model is loaded with 4-bit quantization, some layers that are
    newly initialized (not present in the checkpoint) may be wrapped as
    Linear4bit but with unpacked weights. This causes an AssertionError
    during forward pass. This function detects and replaces such layers
    with standard nn.Linear in float16.

    Args:
        model: The loaded model to fix.
    """
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Linear4bit" in module_type or "Linear8bitLt" in module_type:
            if hasattr(module, "weight") and module.weight is not None:
                weight_shape = module.weight.shape
                # Properly quantized 4-bit weights have shape [N, 1]
                # Unpacked weights retain their original 2D shape
                if len(weight_shape) == 2 and weight_shape[1] != 1:
                    # Only fix layers that are truly uninitialized (no quant_state)
                    quant_state = getattr(module, "quant_state", None)
                    if quant_state is not None:
                        continue
                    logger.warning(
                        f"Fixing improperly quantized layer {name}: "
                        f"{module_type} with shape {weight_shape}"
                    )
                    device = module.weight.device
                    in_features = weight_shape[1]
                    out_features = weight_shape[0]
                    # Replace with a standard nn.Linear in the compute dtype.
                    # These layers are newly initialized heads (pooler, classifier)
                    # that PEFT may mark trainable via modules_to_save — they must
                    # stay in floating point, a fresh Linear4bit would quantize its
                    # weight to uint8 on device and break requires_grad.
                    compute_dtype = getattr(module, "compute_dtype", torch.float16)
                    replacement = nn.Linear(
                        in_features,
                        out_features,
                        dtype=compute_dtype,  # type: ignore[assignment]
                    ).to(device)
                    # Copy existing weight data if available
                    with torch.no_grad():
                        if hasattr(module, "weight") and module.weight is not None:
                            try:
                                replacement.weight.copy_(module.weight)
                            except Exception:
                                logger.warning(
                                    f"Could not copy weight for {name}: shapes may be incompatible"
                                )
                    # Navigate to parent and replace
                    parts = name.split(".")
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], replacement)
                    logger.info(
                        f"Replaced {name} with {type(replacement).__name__} "
                        f"({out_features}, {in_features})"
                    )


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
        return original_forward(**{k: v for k, v in kwargs.items() if k in accepted_forward_args})

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
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
    elif source_lower == "modelscope":
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/modelscope/hub")
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
        logger.info(f"No cache directory found at {cache_dir}. Nothing to clear.")


def load_preset_model(model_name: str, task_config: TaskConfig) -> tuple[Any, Any] | int:
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
            for preset in MODEL_INFO[model].get("preset", [])  # type: ignore
        ]
    except (KeyError, TypeError):
        preset_models = []
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info["default"]  # type: ignore[index]
    elif model_name in preset_models:
        pass
    else:
        logger.debug(
            f"Model {model_name} not found in preset models. "
            "Please check the model name or use "
            "`load_model_and_tokenizer` function."
        )
        return 0
    return load_model_and_tokenizer(model_name, task_config, source, use_mirror)


# Backward compatibility: FocalLoss was previously defined inside forward()
# Re-export for existing code that imports from this module
__all__ = [
    "DNALLMforSequenceClassification",
    "FocalLoss",
    "download_model",
    "load_model_and_tokenizer",
    "load_preset_model",
]
