import os
import time
from glob import glob
from abc import ABC, abstractmethod
from typing import Optional
from ..configuration.configs import TaskConfig
import torch


# class BaseDNAModel(ABC):
#     @abstractmethod
#     def get_model(self) -> PreTrainedModel:
#         """Return the underlying transformer model"""
#         pass

#     @abstractmethod
#     def preprocess(self, sequences: list[str]) -> dict:
#         """Preprocess DNA sequences"""
#         pass


def download_model(model_name: str, downloader, max_try: int=10):
    # In case network issue, try to download multi-times
    cnt = 0
    while True:
        if cnt >= max_try:
            break
        cnt += 1
        # init download status
        status = "incomplete"
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
    major, minor = torch.cuda.get_device_capability()
    # Hopper (H100) has compute capability 9.0
    return (major, minor) >= (9, 0)


def load_model_and_tokenizer(model_name: str, task_config: TaskConfig, source: str="local", use_mirror: bool=False) -> tuple:
    """
    Load model and tokenizer from either HuggingFace or ModelScope

    Args:
        model_name: Model name or path
        task_config: Task configuration object
        source: Source to load model and tokenizer from (local, huggingface, modelscope), default "local"
        use_mirror: Whether to use HuggingFace mirror (hf-mirror.com), default False

    Returns:
        tuple: (model, tokenizer)
    """
    # Special case for models
    # EVO2 models
    evo_models = ["evo2_1b_base", "evo2_7b_base", "evo2_40b_base", "evo2_7b", "evo2_40b"]
    for m in evo_models:
        if m in model_name.lower():
            from evo2 import Evo2
            model_path = glob(model_name + "/*.pt")[0] if os.path.isdir(model_name) else model_name
            if source.lower() == "local":
                model = Evo2(m, local_path=model_path, use_fp8=is_fp8_capable())
            else:
                model = Evo2(m, use_fp8=is_fp8_capable())
            tokenizer = model.tokenizer
            return model, tokenizer
    
    # Define huggingface mirror
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    else:
        os.unsetenv('HF_ENDPOINT')

    # Load model from local disk
    if source.lower() == "local":
        if not os.path.exists(model_name):
            raise ValueError(f"Model {model_name} not found locally.")
        from transformers import (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
                                  AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer)
        model_path = model_name
    elif source.lower() == "huggingface":
        # Load HuggingFace SDK
        from huggingface_hub import snapshot_download
        from transformers import (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
                                  AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer)
        model_path = download_model(model_name, downloader=snapshot_download)
    elif source.lower() == "modelscope":
        # Load ModelScope SDK
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope import (AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
                                AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer)
        model_path = download_model(model_name, downloader=snapshot_download)
    else:
        pass # do nothing
    # Get task type
    task_type = task_config.task_type
    label_names = task_config.label_names
    num_labels = task_config.num_labels
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if task_type == "mask":
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        elif task_type == "generation":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        elif task_type == "binary":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification",
                trust_remote_code=True
            )
        elif task_type == "multiclass":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification",
                trust_remote_code=True
            )
        elif task_type == "multilabel":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="multi_label_classification",
                trust_remote_code=True
            )
        elif task_type == "regression":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="regression",
                trust_remote_code=True
            )
        elif task_type == "token":
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                trust_remote_code=True
            )
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # check if pad_token_id is None
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_preset_model(
    model_name: str,
    task_config: TaskConfig
) -> tuple:
    """
    Load a preset model and tokenizer based on the task configuration.

    Args:
        model_name: Name or path of the model.
        task_config: Task configuration object.

    Returns:
        tuple: (model, tokenizer)
    """
    from modeling_auto import MODEL_INFO

    source = "modelscope"
    use_mirror = False

    # Load model and tokenizer
    preset_models = sum([model['preset'] for model in MODEL_INFO], [])
    if model_name in MODEL_INFO:
        model_info = MODEL_INFO[model_name]
        model_name = model_info['model_name']['default']
    elif model_name in preset_models:
        pass
    else:
        print(f"Model {model_name} not found in preset models. Please check the model name or use `load_model_and_tokenizer` function.")
        return 0
    return load_model_and_tokenizer(model_name, task_config, source, use_mirror)
