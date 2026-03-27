lucaone_models = [
    "LucaOne-default-step5.6M",
    "LucaOne-default-step17.6M",
    "LucaOne-default-step36M",
    "LucaOne-gene-step36.8M",
]


def _handle_lucaone_models(
    model_name: str,
    source: str,
    head_config: dict | None,
    extra: str | None = None,
) -> tuple | None:
    """Handle special case for LucaOne models."""

    if extra:
        lucaone_models.append(extra)

    for m in lucaone_models:
        if m in model_name:
            try:
                from lucagplm import (
                    LucaGPLMModel,
                    LucaGPLMTokenizer,
                    LucaGPLMConfig,
                )
                from ..model import _get_model_path_and_imports

                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                lucaone_tokenizer = LucaGPLMTokenizer.from_pretrained(
                    downloaded_model_path
                )
                lucaone_model = LucaGPLMModel.from_pretrained(
                    downloaded_model_path
                )
                if head_config is not None:
                    from ..model import DNALLMforSequenceClassification

                    head_config = head_config.__dict__
                    head_config["pooling_strategy"] = "cls"
                    model_config = LucaGPLMConfig.from_pretrained(
                        downloaded_model_path
                    )
                    model_config.head_config = head_config
                    lucaone_model.config = model_config
                    lucaone_model.config.pad_token_id = (
                        lucaone_tokenizer.pad_token_id
                    )
                    lucaone_model = (
                        DNALLMforSequenceClassification.from_base_model(
                            downloaded_model_path,
                            config=model_config,
                            module=LucaGPLMModel,
                        )
                    )
            except ImportError as e:
                raise ImportError(
                    f"lucagplm package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/LucaOne/LucaOne"
                ) from e
            return lucaone_model, lucaone_tokenizer

    return None
