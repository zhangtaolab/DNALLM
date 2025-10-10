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
from ..configuration.configs import TaskConfig
from ..utils import get_logger
import torch

logger = get_logger("dnallm.models.model")

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


def download_model(
    model_name: str, downloader, revision: str | None = None, max_try: int = 10
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
            logger.warning(f"Retry: {cnt}, Status: {status}, Reason: {reason}")
            time.sleep(1)

    if status == "incomplete":
        raise ValueError(f"Model {model_name} download failed.")

    return status


def is_flash_attention_capable():
    """Check if Flash Attention has been installed.
    Returns:
                True if Flash Attention is installed and the device supports it
            False otherwise
    """
    try:
        import flash_attn  # pyright: ignore[reportMissingImports]

        _ = flash_attn
        return True
    except Exception as e:
        logger.warning(f"Cannot find supported Flash Attention: {e}")
        return False


def is_fp8_capable():
    """Check if the current CUDA device supports FP8 precision.

    Returns:
                True if the device supports FP8 (
            compute capability >= 9.0),
            False otherwise
    """
    major, minor = torch.cuda.get_device_capability()
    # Hopper (H100) has compute capability 9.0
    if (major, minor) >= (9, 0):
        return True
    else:
        logger.warning(
            f"Current device compute capability is {major}.{minor}, "
            "which does not support FP8."
        )
        return False


def _handle_evo2_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO2 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO2 model, None otherwise
    """
    evo_models = {
        "evo2_1b_base": "evo2-1b-8k",
        "evo2_7b_base": "evo2-7b-8k",
        "evo2_7b": "evo2-7b-1m",
        "evo2_40b_base": "evo2-40b-8k",
        "evo2_40b": "evo2-40b-1m",
    }

    for m in evo_models:
        if m in model_name.lower():
            try:
                from evo2 import Evo2  # pyright: ignore[reportMissingImports]
                from vortex.model.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo2(Evo2):
                    def __init__(self):
                        pass

                    load_evo2_model = Evo2.load_evo2_model

            except ImportError as e:
                raise ImportError(
                    f"EVO2 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/ArcInstitute/evo2"
                ) from e

            model_path = (
                glob(model_name + "/*.pt")[0]
                if os.path.isdir(model_name)
                else model_name
            )
            # Check the dependencies and find the correct config file
            is_fp8 = is_fp8_capable()
            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            if is_fp8:
                suffix2 = ""
            else:
                suffix2 = "-noFP8"
            suffix = suffix1 + suffix2 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            evo2_model = CustomEvo2()
            if source.lower() == "local":
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=model_path,
                    config_path=config_path,
                )
            else:
                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                downloaded_model_path = os.path.join(
                    downloaded_model_path, m + ".pt"
                )
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=downloaded_model_path,
                    config_path=config_path,
                )
            tokenizer = CharLevelTokenizer(512)
            evo2_model.tokenizer = tokenizer
            evo2_model._model_path = downloaded_model_path
            return evo2_model, tokenizer

    return None


def _handle_evo1_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for EVO1 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO1 model, None otherwise
    """
    evo_models = {
        "evo-1.5-8k-base": "evo-1-131k-base",
        "evo-1-8k-base": "evo-1-8k-base",
        "evo-1-131k-base": "evo-1-131k-base",
        "evo-1-8k-crispr": "evo-1-8k-base",
        "evo-1-8k-transposon": "evo-1-8k-base",
    }

    for m in evo_models:
        if m in model_name.lower():
            try:
                import yaml
                from evo import Evo  # pyright: ignore[reportMissingImports]
                from stripedhyena.utils import dotdict
                from stripedhyena.model import StripedHyena
                from stripedhyena.tokenizer import CharLevelTokenizer

                # Overwrite Evo2 to avoid init errors
                class CustomEvo1(Evo):
                    def __init__(self):
                        self.device = None
                        self.model = None

                    def load_checkpoint(
                        self,
                        model_name: str = "evo-1-8k-base",
                        revision: str = "main",
                        config_path: str | None = None,
                        modules: dict | None = None,
                    ) -> StripedHyena:
                        autoconfig = modules["AutoConfig"]
                        automodelforcausallm = modules["AutoModelForCausalLM"]
                        # Load the model configuration and weights
                        model_config = autoconfig.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        model_config.use_cache = True
                        model = automodelforcausallm.from_pretrained(
                            model_name,
                            config=model_config,
                            trust_remote_code=True,
                            revision=revision,
                        )
                        state_dict = model.backbone.state_dict()
                        del model
                        del model_config
                        global_config = dotdict(
                            yaml.safe_load(open(config_path))
                        )
                        model = StripedHyena(global_config)
                        model.load_state_dict(state_dict, strict=True)
                        model.to_bfloat16_except_poles_residues()
                        return model

            except ImportError as e:
                raise ImportError(
                    f"EVO-1 package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/evo-design/evo"
                ) from e

            has_flash_attention = is_flash_attention_capable()
            config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            suffix = suffix1 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            evo_model = CustomEvo1()
            revision = (
                "1.1_fix"
                if "." in model_name and source == "huggingface"
                else "main"
            )
            _, modules = _get_model_path_and_imports(
                model_name, source, revision=revision
            )
            evo_model.model = evo_model.load_checkpoint(
                model_name=model_name,
                revision=revision,
                config_path=config_path,
                modules=modules,
            )
            tokenizer = CharLevelTokenizer(512)
            evo_model.tokenizer = tokenizer
            return evo_model, tokenizer

    return None


def _handle_gpn_models(model_name: str) -> str | None:
    gpn_models = [
        "gpn-brassicales",
        "gpn-animal-promoter",
        "gpn-msa-sapiens",
        "PhyloGPN",
        "gpn-brassicales-gxa-sorghum-v1",
    ]
    for m in gpn_models:
        if m in model_name:
            try:
                import gpn.model

                _ = gpn.model
            except ImportError as e:
                raise ImportError(
                    f"gpn package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/songlab-cal/gpn"
                ) from e
            return m
    return None


def _handle_lucaone_models(model_name: str) -> str | None:
    pass


def _handle_megadna_models(model_name: str, source: str) -> tuple | None:
    """Handle special case for megaDNA models."""
    megadna_models = [
        "megaDNA_updated",
        "megaDNA_variants",
        "megaDNA_finetuned",
        "megaDNA_phage_145M",
        "megaDNA_phage_78M",
        "megaDNA_phage_277M",
        "megaDNA_phage_ecoli_finetuned",
    ]
    for m in megadna_models:
        if m in model_name:
            from torch.nn.utils.rnn import pad_sequence

            class DNATokenizer:
                DEFAULT_TOKENS = ("**", "#")

                def __init__(
                    self,
                    vocab=None,
                    pad_token=DEFAULT_TOKENS[0],
                    eos_token=DEFAULT_TOKENS[1],
                ):
                    self.vocab = vocab or [
                        pad_token,
                        "A",
                        "T",
                        "C",
                        "G",
                        eos_token,
                    ]
                    self.token_to_id = {
                        tok: idx for idx, tok in enumerate(self.vocab)
                    }
                    self.id_to_token = dict(enumerate(self.vocab))
                    self.pad_token = pad_token
                    self.pad_id = self.token_to_id[pad_token]
                    self.eos_token = eos_token
                    self.eos_id = self.token_to_id[eos_token]

                def encode(self, text):
                    """text: 'ATCGAT...' → [1,2,3,4,...]"""
                    return [
                        self.token_to_id.get(ch, self.pad_id) for ch in text
                    ]

                def decode(self, ids):
                    """tensor/list ids → 'ATCG...'"""
                    if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                    return "".join(self.id_to_token.get(i, "N") for i in ids)

                def __call__(self, text, return_tensors=None):
                    """
                    text: str or list[str]
                    return_tensors: tensor[batch, seq_len] if is 'pt'
                    """
                    # Single text
                    if isinstance(text, str):
                        ids = torch.tensor(
                            self.encode(text), dtype=torch.long
                        ).unsqueeze(0)
                        return ids

                    # Batch texts (list[str])
                    elif isinstance(text, list):
                        seqs = [
                            torch.tensor(self.encode(t), dtype=torch.long)
                            for t in text
                        ]
                        padded = pad_sequence(
                            seqs, batch_first=True, padding_value=self.pad_id
                        )
                        return padded

                    else:
                        raise ValueError(
                            "Tokenizer input must be str or List[str]"
                        )

            try:
                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                if m in "megaDNA_updated".lower():
                    full_model_name = "megaDNA_phage_145M.pt"
                elif m in "megaDNA_variants".lower():
                    full_model_name = "megaDNA_phage_78M.pt"
                elif m in "megaDNA_finetuned".lower():
                    full_model_name = "megaDNA_phage_ecoli_finetuned.pt"
                else:
                    full_model_name = "megaDNA_phage_145M.pt"
                downloaded_model_path = os.path.join(
                    downloaded_model_path, full_model_name
                )
                megadna_model = torch.load(
                    downloaded_model_path, weights_only=False
                )

            except ImportError as e:
                raise ImportError(
                    f"megaDNA package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/lingxusb/megaDNA"
                ) from e

            megadna_tokenizer = DNATokenizer()
            return megadna_model, megadna_tokenizer

    return None


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
    revision: str | None = None,
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

    # Handle special case for EVO2 models
    evo2_result = _handle_evo2_models(model_name, source)
    if evo2_result is not None:
        return evo2_result

    # Handle special case for EVO1 models
    evo1_result = _handle_evo1_models(model_name, source)
    if evo1_result is not None:
        return evo1_result

    # Handle special case for GPN models
    _ = _handle_gpn_models(model_name)

    # Handle special case for megaDNA models
    megadna_result = _handle_megadna_models(model_name, source)
    if megadna_result is not None:
        return megadna_result

    # Handle other models such as LucaOne, Omni-DNA, etc.
    # TODO: Add more special cases if needed

    # Get model path and import required modules
    downloaded_model_path, modules = _get_model_path_and_imports(
        model_name, source, revision=revision
    )

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
                f"num_labels is required for task type "
                f"'{task_type}' but is None"
            )

        # Use default value if num_labels is None for other tasks
        safe_num_labels = num_labels if num_labels is not None else 1
        # num_labels check for non-binary classification tasks
        if task_type == "regression" and safe_num_labels != 1:
            logger.warning(
                f"Regression task typically has num_labels=1, "
                f"but got {safe_num_labels}."
            )
            safe_num_labels = 1
        elif task_type == "generation" and safe_num_labels != 0:
            logger.warning(
                f"Generation task does not require num_labels, "
                f"but got {safe_num_labels}. Setting to 0."
            )
            safe_num_labels = 0
        elif task_type != "binary":
            if safe_num_labels < 2:
                raise ValueError(
                    f"num_labels should be at least 2 for task type "
                    f"'{task_type}', but got {safe_num_labels}."
                )

        model, tokenizer = _load_model_by_task_type(
            task_type, model_name, safe_num_labels, id2label, label2id, modules
        )
        # Set model path and source attributes
        model._model_path = downloaded_model_path
        model.source = source
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}") from e

    # Configure model padding
    _configure_model_padding(model, tokenizer)

    return model, tokenizer


def peft_forward_compatiable(model):
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
