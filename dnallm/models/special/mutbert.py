from typing import Any
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


def _handle_mutbert_tokenizer(tokenizer: Any) -> Any:
    """Handle special case for MutBERT tokenizer.

    Args:
        tokenizer: Original tokenizer

    Returns:
        MutBERT tokenizer
    """

    class OneHotTokenizerWrapper:
        def __init__(self, tokenizer: PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
            self.vocab_size = len(tokenizer)

        def __call__(self, text, **kwargs):
            kwargs["return_tensors"] = "pt"
            encoding = self.tokenizer(text, **kwargs)
            input_ids = encoding["input_ids"]
            # input_ids are (Batch, Seq_Len)
            # convert to (Batch, Seq_Len, Vocab_Size) after one-hot encoding
            one_hot_inputs = nn.functional.one_hot(
                input_ids, num_classes=self.vocab_size
            ).float()
            encoding["input_ids"] = one_hot_inputs

            return encoding

        def __getattr__(self, name):
            return getattr(self.tokenizer, name)

    return OneHotTokenizerWrapper(tokenizer)
