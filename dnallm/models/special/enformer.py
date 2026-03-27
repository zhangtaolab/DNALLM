import os


enformer_models = [
    "enformer-official-rough",
    "enformer-191k",
    "enformer-corr_coef_obj",
    "enformer-191k_corr_coef_obj",
]


def _handle_enformer_models(
    model_name: str,
    source: str,
    task_type: str,
    num_labels: int,
    extra: str | None = None,
) -> tuple | None:
    """Handle special case for Enformer models."""

    if extra:
        enformer_models.append(extra)

    for m in enformer_models:
        if m in model_name:
            from ..tokenizer import DNAOneHotTokenizer
            from ..model import _get_model_path_and_imports
            from .enformer_model.configuration_enformer import EnformerConfig
            from .enformer_model.modeling_enformer import (
                from_pretrained,
                EnformerForSequenceClassification,
            )

            downloaded_model_path, _ = _get_model_path_and_imports(
                model_name, source
            )
            config_path = os.path.join(downloaded_model_path, "config.json")
            config = EnformerConfig.from_pretrained(config_path)
            config.num_labels = num_labels
            if task_type in [
                "binary",
                "regression",
                "multiclass",
                "multilabel",
            ]:
                model = EnformerForSequenceClassification.from_pretrained(
                    downloaded_model_path, config=config
                )
            else:
                model = from_pretrained(downloaded_model_path, config=config)
            tokenizer = DNAOneHotTokenizer()
            return model, tokenizer

    return None
