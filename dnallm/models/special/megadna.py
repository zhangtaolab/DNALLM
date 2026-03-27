import os
from typing import Any
import torch


megadna_models = [
    "megaDNA_updated",
    "megaDNA_variants",
    "megaDNA_finetuned",
    "megaDNA_phage_145M",
    "megaDNA_phage_78M",
    "megaDNA_phage_277M",
    "megaDNA_phage_ecoli_finetuned",
]


def _handle_megadna_models(
    model_name: str,
    source: str,
    head_config: dict | None,
    extra: str | None = None,
) -> tuple | None:
    """Handle special case for megaDNA models."""

    if extra:
        megadna_models.append(extra)

    for m in megadna_models:
        if m in model_name:
            from transformers import PretrainedConfig, PreTrainedTokenizer

            class MegaDNAConfig(PretrainedConfig):
                model_type = "megadna"

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

            class DNATokenizer(PreTrainedTokenizer):
                """
                This tokenizer treats each nucleotide (A, T, C, G)
                as a separate token, along with special tokens for padding,
                end-of-sequence, and unknown tokens.
                """

                # Define the vocabulary file names for saving
                vocab_files_names = {  # noqa: RUF012
                    "vocab_file": "vocab.txt"
                }
                DEFAULT_TOKENS = ("**", "#")

                def __init__(
                    self,
                    vocab_file=None,  # from_pretrained will handle this
                    pad_token=DEFAULT_TOKENS[0],
                    eos_token=DEFAULT_TOKENS[1],
                    unk_token=None,
                    **kwargs,
                ):
                    # 1. Initialize your vocabulary and mappings
                    self.vocab = [pad_token, "A", "T", "C", "G", eos_token]
                    self.token_to_id = {
                        tok: idx for idx, tok in enumerate(self.vocab)
                    }
                    self.id_to_token = dict(enumerate(self.vocab))

                    # 2. Initialize the parent class
                    super().__init__(
                        pad_token=pad_token,
                        eos_token=eos_token,
                        unk_token=unk_token,
                        **kwargs,
                    )

                @property
                def vocab_size(self) -> int:
                    return len(self.vocab)

                def get_vocab(self) -> dict[str, int]:
                    return self.token_to_id.copy()

                def _tokenize(self, text: Any, **kwargs: Any) -> list[str]:
                    return list(text)

                def _convert_token_to_id(self, token: str) -> int:
                    return self.token_to_id.get(token, self.unk_token_id)

                def _convert_id_to_token(self, index: int):
                    return self.id_to_token.get(index, self.unk_token)

                def save_vocabulary(
                    self,
                    save_directory: str,
                    filename_prefix: str | None = None,
                ) -> tuple:
                    """
                    Implement how to save your vocabulary to a file.
                    The parent class's save_pretrained method will
                    automatically call this function.
                    """
                    vocab_file_path = os.path.join(
                        save_directory, self.vocab_files_names["vocab_file"]
                    )

                    with open(
                        vocab_file_path, "w", encoding="utf-8"
                    ) as writer:
                        writer.write("\n".join(self.vocab))

                    return (vocab_file_path,)

            try:
                from ..model import _get_model_path_and_imports

                downloaded_model_path, _ = _get_model_path_and_imports(
                    model_name, source
                )
                if m in "megaDNA_updated":
                    full_model_name = "megaDNA_phage_145M.pt"
                elif m in "megaDNA_variants":
                    full_model_name = "megaDNA_phage_78M.pt"
                elif m in "megaDNA_finetuned":
                    full_model_name = "megaDNA_phage_ecoli_finetuned.pt"
                else:
                    full_model_name = "megaDNA_phage_145M.pt"
                downloaded_model_path = os.path.join(
                    downloaded_model_path, full_model_name
                )
                megadna_model = torch.load(
                    downloaded_model_path, weights_only=False
                )
                megadna_tokenizer = DNATokenizer()
                if head_config is not None:
                    from ..model import DNALLMforSequenceClassification

                    head_config = head_config.__dict__
                    head_config["embedding_dims"] = [512, 256, 196]
                    model_config = MegaDNAConfig()
                    model_config.head_config = head_config
                    megadna_model.config = model_config
                    pad_token_id = megadna_tokenizer.pad_token_id
                    megadna_model.config.pad_token_id = pad_token_id
                    megadna_model = DNALLMforSequenceClassification(
                        config=model_config,
                        custom_model=megadna_model,
                    )
            except ImportError as e:
                raise ImportError(
                    f"megaDNA package is required for "
                    f"{model_name} but not installed. "
                    "Please install it following the instructions at: "
                    "https://github.com/lingxusb/megaDNA"
                ) from e

            return megadna_model, megadna_tokenizer

    return None
