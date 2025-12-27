import os
import json
from glob import glob
import torch
from ...utils import is_flash_attention_capable, is_fp8_capable


evo2_models = {
    "evo2_1b_base": "evo2-1b-8k",
    "evo2_7b_base": "evo2-7b-8k",
    "evo2_7b_262k": "evo2-7b-262k",
    "evo2_7b": "evo2-7b-1m",
    "evo2_40b_base": "evo2-40b-8k",
    "evo2_40b": "evo2-40b-1m",
    "evo2_7b_microviridae": "evo2-7b-8k",
}

evo_models = {
    "evo-1.5-8k-base": "evo-1-131k-base",
    "evo-1-8k-base": "evo-1-8k-base",
    "evo-1-131k-base": "evo-1-131k-base",
    "evo-1-8k-crispr": "evo-1-8k-base",
    "evo-1-8k-transposon": "evo-1-8k-base",
}


class EvoTokenizerWrapper:
    def __init__(self, raw_tokenizer, model_max_length=8192, **kwargs):
        """
        raw_tokenizer: Raw CharLevelTokenizer instance from EVO2 package
        pad_token_id: Token ID used for padding (usually 1 for EVO2)
        model_max_length: Maximum context length of the model
        """

        self.raw_tokenizer = raw_tokenizer
        self.model_max_length = model_max_length
        for attr in [
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "pad_token_id",
            "pad_id",
            "eos_id",
            "eod_id",
        ]:
            if hasattr(raw_tokenizer, attr):
                setattr(self, attr, getattr(raw_tokenizer, attr))
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = self.raw_tokenizer.pad_id
        self.pad_token = raw_tokenizer.decode_token(self.pad_token_id)
        self.padding_side = "right"
        self.init_kwargs = kwargs

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        """
        __call__ method to tokenize inputs with padding and truncation.
        """
        if isinstance(text, str):
            text = [text]
            is_batched = False
        else:
            is_batched = True

        input_ids_list = [self.raw_tokenizer.tokenize(seq) for seq in text]

        if truncation:
            limit = (
                max_length if max_length is not None else self.model_max_length
            )
            input_ids_list = [ids[:limit] for ids in input_ids_list]

        if padding:
            if padding == "max_length":
                target_len = (
                    max_length
                    if max_length is not None
                    else self.model_max_length
                )
            elif padding is True or padding == "longest":
                target_len = max(len(ids) for ids in input_ids_list)
            else:
                target_len = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            attention_masks = []

            for ids in input_ids_list:
                current_len = len(ids)
                pad_len = target_len - current_len

                if pad_len < 0:
                    ids = ids[:target_len]
                    pad_len = 0
                    current_len = target_len

                new_ids = ids + [self.pad_token_id] * pad_len
                padded_input_ids.append(new_ids)

                mask = [1] * current_len + [0] * pad_len
                attention_masks.append(mask)
        else:
            padded_input_ids = input_ids_list
            attention_masks = [[1] * len(ids) for ids in input_ids_list]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(
                    attention_masks, dtype=torch.long
                ),
            }

        result = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_masks,
        }

        if not is_batched and return_tensors is None:
            return {k: v[0] for k, v in result.items()}

        return result

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, "
                "not a file."
            )

        os.makedirs(save_directory, exist_ok=True)

        tokenizer_config = {
            "pad_token_id": self.pad_token_id,
            "model_max_length": self.model_max_length,
            "tokenizer_class": "Evo2TokenizerWrapper",
            **self.init_kwargs,
        }

        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        return [config_file]

    @classmethod
    def from_pretrained(cls, save_directory, raw_tokenizer, **kwargs):
        config_file = os.path.join(save_directory, "tokenizer_config.json")

        if not os.path.exists(config_file):
            print(
                "Warning: tokenizer_config.json not found "
                f"in {save_directory}. Using default config."
            )
            return cls(raw_tokenizer, **kwargs)

        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        config.pop("tokenizer_class", None)

        config.update(kwargs)

        return cls(raw_tokenizer=raw_tokenizer, **config)

    @property
    def model_max_length(self):
        return self._model_max_length

    @model_max_length.setter
    def model_max_length(self, value):
        self._model_max_length = value


def _handle_evo2_models(
    model_name: str,
    source: str,
    head_config: dict | None = None,
) -> tuple | None:
    """Handle special case for EVO2 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO2 model, None otherwise
    """

    for m in evo2_models:
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
                    os.path.dirname(__file__), "..", "..", "configuration/evo"
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
            config_path = os.path.join(config_dir, evo2_models[m] + suffix)
            # Load the model with the built-in method
            evo2_model = CustomEvo2()
            if source.lower() == "local":
                evo2_model.model = evo2_model.load_evo2_model(
                    None,
                    local_path=model_path,
                    config_path=config_path,
                )
                downloaded_model_path = model_path
            else:
                from ..model import _get_model_path_and_imports

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
            if head_config is not None:
                from transformers import PretrainedConfig
                from ..model import DNALLMforSequenceClassification

                class Evo2Config(PretrainedConfig):
                    model_type = "evo2"

                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)

                        for key, value in kwargs.items():
                            setattr(self, key, value)

                model_config = Evo2Config(**evo2_model.model.config)
                head_config = head_config.__dict__
                model_config.head_config = head_config
                evo2_model.config = model_config
                evo2_model.config.pad_token_id = evo2_model.tokenizer.pad_id
                tokenizer = EvoTokenizerWrapper(
                    raw_tokenizer=evo2_model.tokenizer,
                    model_max_length=model_config.max_seqlen,
                )
                evo2_model = DNALLMforSequenceClassification(
                    config=model_config,
                    custom_model=evo2_model,
                )
            return evo2_model, tokenizer

    return None


def _handle_evo1_models(
    model_name: str,
    source: str,
    head_config: dict | None = None,
) -> tuple | None:
    """Handle special case for EVO1 models.

    Args:
        model_name: Model name or path
        source: Source to load model from

    Returns:
        Tuple of (model, tokenizer) if EVO1 model, None otherwise
    """

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
                    os.path.dirname(__file__), "..", "..", "configuration/evo"
                )
            )
            if has_flash_attention:
                suffix1 = ""
            else:
                suffix1 = "-noFA"
            suffix = suffix1 + ".yml"
            config_path = os.path.join(config_dir, evo_models[m] + suffix)
            # Load the model with the built-in method
            from ..model import _get_model_path_and_imports

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
            if head_config is not None:
                from transformers import PretrainedConfig
                from ..model import DNALLMforSequenceClassification

                class EvoConfig(PretrainedConfig):
                    model_type = "evo"

                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)

                        for key, value in kwargs.items():
                            setattr(self, key, value)

                model_config = EvoConfig(**evo_model.model.config)
                head_config = head_config.__dict__
                model_config.head_config = head_config
                evo_model.config = model_config
                evo_model.config.pad_token_id = evo_model.tokenizer.pad_id
                tokenizer = EvoTokenizerWrapper(
                    raw_tokenizer=evo_model.tokenizer,
                    model_max_length=model_config.max_seqlen,
                )
                evo_model = DNALLMforSequenceClassification(
                    config=model_config,
                    custom_model=evo_model,
                )
            return evo_model, tokenizer

    return None
