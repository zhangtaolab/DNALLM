import os


borzoi_models = [
    "borzoi-replicate-0",
    "borzoi-replicate-0-mouse",
    "borzoi-replicate-1",
    "borzoi-replicate-1-mouse",
    "borzoi-replicate-2",
    "borzoi-replicate-2-mouse",
    "borzoi-replicate-3",
    "borzoi-replicate-3-mouse",
    "flashzoi-replicate-0",
    "flashzoi-replicate-1",
    "flashzoi-replicate-2",
    "flashzoi-replicate-3",
]


def _handle_borzoi_models(
    model_name: str,
    source: str,
    task_type: str,
    num_labels: int,
    extra: str | None = None,
) -> str | None:
    """Handle special case for Borzoi models."""

    if extra:
        borzoi_models.append(extra)

    for m in borzoi_models:
        if m not in model_name:
            continue
        try:
            from borzoi_pytorch import Borzoi
            from borzoi_pytorch.config_borzoi import BorzoiConfig
            from ..tokenizer import DNAOneHotTokenizer
            from ..model import _get_model_path_and_imports

            downloaded_model_path, _ = _get_model_path_and_imports(
                model_name, source
            )
            config_path = os.path.join(downloaded_model_path, "config.json")
            config = BorzoiConfig.from_pretrained(config_path)
            config.num_labels = num_labels
            if task_type in [
                "binary",
                "regression",
                "multiclass",
                "multilabel",
            ]:
                import torch
                import torch.nn as nn
                from transformers.modeling_outputs import (
                    SequenceClassifierOutput,
                )

                class BorzoiForSequenceClassification(Borzoi):
                    def __init__(self, config, **kwargs):
                        super().__init__(config)
                        self.num_labels = kwargs.get(
                            "num_labels", config.num_labels
                        )
                        self.config = config
                        self._target_length = self.config.bins_to_return
                        self.resolution = 32

                        self.score = nn.Linear(1920, self.num_labels)

                        # Initialize weights and apply final processing
                        self.post_init()

                    @property
                    def target_length(self):
                        return self._target_length

                    @target_length.setter
                    def target_length(self, value):
                        self._target_length = value
                        self.crop.target_length = value

                    def forward(
                        self,
                        inputs_embeds=None,
                        labels=None,
                        return_embeddings=True,
                        pooling: str = "mean",
                        output_hidden_states: bool | None = None,
                        return_dict: bool | None = None,
                        **kwargs,
                    ):
                        return_dict = (
                            return_dict
                            if return_dict is not None
                            else self.config.use_return_dict
                        )
                        embeddings = self.get_embs_after_crop(inputs_embeds)
                        embeddings = self.final_joined_convs(embeddings)
                        if pooling == "mean":
                            pooled_output = torch.mean(embeddings, dim=2)
                        else:
                            pooled_output = torch.sum(embeddings, dim=2)
                        logits = self.score(pooled_output)
                        loss = None
                        if labels is not None:
                            if self.config.problem_type is None:
                                if self.num_labels == 1:
                                    self.config.problem_type = "regression"
                                elif self.num_labels > 1 and (
                                    labels.dtype == torch.long
                                    or labels.dtype == torch.int
                                ):
                                    self.config.problem_type = (
                                        "single_label_classification"
                                    )
                                else:
                                    self.config.problem_type = (
                                        "multi_label_classification"
                                    )

                            if self.config.problem_type == "regression":
                                loss_fct = nn.MSELoss()
                                if self.num_labels == 1:
                                    loss = loss_fct(
                                        logits.squeeze(), labels.squeeze()
                                    )
                                else:
                                    loss = loss_fct(logits, labels)
                            elif (
                                self.config.problem_type
                                == "single_label_classification"
                            ):
                                loss_fct = nn.CrossEntropyLoss()
                                loss = loss_fct(
                                    logits.view(-1, self.num_labels),
                                    labels.view(-1),
                                )
                            elif (
                                self.config.problem_type
                                == "multi_label_classification"
                            ):
                                loss_fct = nn.BCEWithLogitsLoss()
                                loss = loss_fct(logits, labels)
                            else:
                                raise NotImplementedError(
                                    self.config.problem_type
                                )

                        if not return_dict:
                            output = (
                                (
                                    logits,
                                    embeddings,
                                )
                                if output_hidden_states
                                else (logits,)
                            )
                            return (
                                ((loss, *output))
                                if loss is not None
                                else output
                            )

                        return SequenceClassifierOutput(
                            loss=loss,
                            logits=logits,
                            hidden_states=(
                                embeddings if output_hidden_states else None
                            ),
                            attentions=None,
                        )

                model = BorzoiForSequenceClassification.from_pretrained(
                    downloaded_model_path, config=config
                )
            else:
                model = Borzoi.from_pretrained(
                    downloaded_model_path, config=config
                )
            tokenizer = DNAOneHotTokenizer(
                return_embeds=True, embeds_transpose=True
            )

        except ImportError as e:
            raise ImportError(
                "borzoi_pytorch package is required for "
                f"{model_name} but not installed. "
                "Please install it following the instructions at: "
                "https://github.com/johahi/borzoi-pytorch"
            ) from e

        return model, tokenizer

    return None
