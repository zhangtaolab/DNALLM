import os


space_models = [
    "SPACE",
]


def _handle_space_models(
    model_name: str,
    source: str,
    task_type: str,
    num_labels: int,
    extra: str | None = None,
) -> tuple | None:
    """Handle special case for Space models."""

    if extra:
        space_models.append(extra)

    for m in space_models:
        if m in model_name:
            from ..tokenizer import DNAOneHotTokenizer
            from ..model import _get_model_path_and_imports
            from .enformer_model.configuration_space import SpaceConfig
            from .enformer_model.modeling_space import (
                Space,
                SpaceForSequenceClassification,
            )

            downloaded_model_path, _ = _get_model_path_and_imports(
                model_name, source
            )
            config_path = os.path.join(downloaded_model_path, "config.json")
            config = SpaceConfig.from_pretrained(config_path)
            config.num_labels = num_labels
            if task_type in [
                "binary",
                "regression",
                "multiclass",
                "multilabel",
            ]:
                model = SpaceForSequenceClassification.from_pretrained(
                    downloaded_model_path, config=config
                )
            else:
                model = Space.from_pretrained(
                    downloaded_model_path, config=config
                )
            tokenizer = DNAOneHotTokenizer()
            return model, tokenizer

    return None
