"""DNA Model loading and management utilities.

This module provides functions for downloading, loading, and managing DNA language models
from various sources including Hugging Face Hub, ModelScope, and local storage.
"""
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import os
import time
from glob import glob
from typing import Any
from ..configuration.configs import TaskConfig
import torch

# Type checking imports - removed as they're not used in the actual code


# class BaseDNAModel(ABC):
#     @abstractmethod
#     def get_model(self) -> PreTrainedModel:
#         """Return the underlying transformer model"""
#         pass

#     @abstractmethod
#     def preprocess(self, sequences: list[str]) -> dict:
#         """Preprocess DNA sequences"""
#         pass


def download_model(model_name: str, downloader, max_try: int = 10) -> str:
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
            status = downloader(model_name)
            if status != "incomplete":
                print(f"Model files are stored in {status}")
                break
        # track the error
        except Exception as e:
            # network issue
            if "connection" in str(e):
                reason = "unstable network connection."
            # model not found in HuggingFace
            elif "not found" in str(e).lower():
                reason = "repo is not found."
                print(e)
                break
            # model not exist in ModelScope
            elif "response [404]" in str(e).lower():
                reason = "repo is not existed."
                print(e)
                break
            else:
                reason = e
            print(f"Retry: {cnt}, Status: {status}, Reason: {reason}")
            time.sleep(1)

    if status == "incomplete":
        raise ValueError(f"Model {model_name} download failed.")

    return status


def is_fp8_capable():
    """Check if the current CUDA device supports FP8 precision.

    Returns:
        True if the device supports FP8 (compute capability >= 9.0), False otherwise
    """
    major, minor = torch.cuda.get_device_capability()
    # Hopper (H100) has compute capability 9.0
    return (major, minor) >= (9, 0)


def _handle_evo2_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO2 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO2 model, None otherwise
    """
    evo_models = [
        "evo2_1b_base",
        "evo2_7b_base",
        "evo2_40b_base",
        "evo2_7b",
        "evo2_40b",
    ]

    for m in evo_models:
        if m in model_name.lower():
            try:
                from evo2 import Evo2  # pyright: ignore[reportMissingImports]
            except ImportError as e:
                raise ImportError(
                    f"EVO2 package is required for {model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/ArcInstitute/evo2"
                ) from e

            model_path = (
                glob(model_name + "/*.pt")[0]
                if os.path.isdir(model_name)
                else model_name
            )
            if source.lower() == "local":
                model = Evo2(
                    m, local_path=model_path, use_fp8=is_fp8_capable()
                )
            else:
                model = Evo2(m, use_fp8=is_fp8_capable())
            tokenizer = model.tokenizer
            return model, tokenizer

    return None


def _setup_huggingface_mirror(use_mirror: bool) -> None:
    """Configure HuggingFace mirror settings.

    Args:
        use_mirror: Whether to use HuggingFace mirror
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def _get_model_path_and_imports(
    model_name: str, source: str
) -> tuple[str, dict[str, Any]]:
    """Get model path and import the required libraries based on source.

    Args:
        model_name: Model name or path
        source: Source to load model from ('local', 'huggingface', 'modelscope')

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
        from huggingface_hub import snapshot_download

        model_path = download_model(model_name, downloader=snapshot_download)

    elif source_lower == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        model_path = download_model(model_name, downloader=snapshot_download)

        # Import ModelScope modules
        try:
            from modelscope import (  # pyright: ignore[reportAttributeAccessIssue]
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
            "AutoModel": AutoModel,
            "AutoModelForMaskedLM": AutoModelForMaskedLM,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "AutoModelForTokenClassification": AutoModelForTokenClassification,
            "AutoTokenizer": AutoTokenizer,
        }

        return model_path, modules

    else:
        raise ValueError(f"Unsupported source: {source}")

    # Import transformers modules for local and huggingface sources
    try:
        from transformers import (  # pyright: ignore[reportAttributeAccessIssue]
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
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
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
) -> tuple[Any, Any]:
    """Load model and tokenizer based on task type.

    Args:
        task_type: Type of task (mask, generation, binary, etc.)
        model_name: Model name or path
        num_labels: Number of labels for classification tasks
        id2label: ID to label mapping
        label2id: Label to ID mapping
        modules: Dictionary of imported model classes

    Returns:
        Tuple of (model, tokenizer)
    """
    auto_tokenizer = modules["AutoTokenizer"]

    # Common tokenizer loading
    if task_type == "token":
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, trust_remote_code=True, add_prefix_space=True
        )
    else:
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

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
        model = modules["AutoModel"].from_pretrained(
            model_name, trust_remote_code=True, attn_implementation="eager"
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


def load_model_and_tokenizer(
    model_name: str,
    task_config: TaskConfig,
    source: str = "local",
    use_mirror: bool = False,
) -> tuple[Any, Any]:
    """Load model and tokenizer from either HuggingFace or ModelScope.

    This function handles loading of various model types based on the task configuration,
    including sequence classification, token classification, masked language modeling,
    and causal language modeling.

    Args:
        model_name: Model name or path
        task_config: Task configuration object containing task type and label information
        source: Source to load model and tokenizer from ('local', 'huggingface', 'modelscope'), default 'local'
        use_mirror: Whether to use HuggingFace mirror (hf-mirror.com), default False

    Returns:
        Tuple containing (model, tokenizer)

    Raises:
        ValueError: If model is not found locally or loading fails
    """
    # Handle special case for EVO2 models
    evo_result = _handle_evo2_models(model_name, source)
    if evo_result is not None:
        return evo_result

    # Setup HuggingFace mirror if needed
    _setup_huggingface_mirror(use_mirror)

    # Get model path and import required modules
    _model_path, modules = _get_model_path_and_imports(model_name, source)

    # Extract task configuration
    task_type = task_config.task_type
    num_labels = task_config.num_labels

    # Create label mappings
    id2label, label2id = _create_label_mappings(task_config)

    # Load model and tokenizer based on task type
    try:
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

        model, tokenizer = _load_model_by_task_type(
            task_type, model_name, safe_num_labels, id2label, label2id, modules
        )
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}") from e

    # Configure model padding
    _configure_model_padding(model, tokenizer)

    return model, tokenizer


def load_preset_model(
    model_name: str, task_config: TaskConfig
) -> tuple[Any, Any] | int:
    """Load a preset model and tokenizer based on the task configuration.

    This function loads models from the preset model registry, which contains
    pre-configured models for various DNA analysis tasks.

    Args:
        model_name: Name or path of the model
        task_config: Task configuration object containing task type and label information

    Returns:
        Tuple containing (model, tokenizer) if successful, 0 if model not found

    Note:
        If the model is not found in preset models, the function will print a warning
        and return 0. Use `load_model_and_tokenizer` function for custom model loading.
    """
    from .modeling_auto import MODEL_INFO

    source = "modelscope"
    use_mirror = False

    # Load model and tokenizer
    try:
        preset_models = [
            preset
            for model in MODEL_INFO
            for preset in model.get("preset", [])
        ]
    except (KeyError, TypeError):
        preset_models = []
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info["model_name"]["default"]
    elif model_name in preset_models:
        pass
    else:
        print(
            f"Model {model_name} not found in preset models. Please check the model name or use `load_model_and_tokenizer` function."
        )
        return 0
    return load_model_and_tokenizer(
        model_name, task_config, source, use_mirror
    )
