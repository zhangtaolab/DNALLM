from typing import Any


basenji2_models = [
    "basenji2",
]


def _handle_basenji2_tokenizer(tokenizer: Any) -> Any:
    """Handle special case for MutBERT tokenizer.

    Args:
        tokenizer: Original tokenizer

    Returns:
        MutBERT tokenizer
    """
    from ..tokenizer import DNAOneHotTokenizer

    tokenizer = DNAOneHotTokenizer(return_embeds=True, embeds_transpose=True)

    return tokenizer
