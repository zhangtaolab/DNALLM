"""Shared pytest fixtures for DNALLM test suite."""

import pytest
from unittest.mock import Mock
import torch
import pandas as pd


@pytest.fixture(scope="session", autouse=True)
def global_cleanup():
    """Session-scoped cleanup fixture."""
    yield
    # Cleanup after all tests complete


@pytest.fixture(scope="function")
def mock_model(request):
    """Return a Mock configured as a transformers PreTrainedModel.

    Args:
        request: Pytest request object for indirect parameterization.

    Returns:
        Mock object configured as a PreTrainedModel.
    """
    mock_model = Mock()

    mock_config = Mock()
    mock_config.output_attentions = False
    mock_config.output_hidden_states = False
    mock_config.attn_implementation = "eager"
    mock_config.num_attention_heads = 12
    mock_config.num_hidden_layers = 6
    mock_config.vocab_size = 1000
    mock_config.model_type = "dna_gpt"

    # Support optional architecture parameter via indirect=True
    if hasattr(request, "param") and request.param == "mamba":
        mock_config.model_type = "mamba"
        mock_config.d_model = 768
    else:
        mock_config.hidden_size = 768

    mock_model.config = mock_config
    mock_model.parameters.return_value = iter([torch.randn(100, 100)])

    def mock_forward(input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        mock_output = Mock()
        mock_output.logits = torch.randn(batch_size, 2)
        return mock_output

    mock_model.forward = mock_forward
    mock_model.eval = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    return mock_model


@pytest.fixture(scope="function")
def mock_tokenizer():
    """Return a Mock configured as a PreTrainedTokenizer.

    Returns:
        Mock object configured as a tokenizer.
    """
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    mock_tokenizer.encode_plus = Mock(
        return_value={
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0, 0],
        }
    )
    mock_tokenizer.__call__ = Mock(
        return_value={
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
    )
    mock_tokenizer.special_tokens_map = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
    mock_tokenizer.pad_token = "[PAD]"
    mock_tokenizer.unk_token = "[UNK]"
    mock_tokenizer.cls_token = "[CLS]"
    mock_tokenizer.sep_token = "[SEP]"
    mock_tokenizer.mask_token = "[MASK]"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.unk_token_id = 1
    mock_tokenizer.vocab_size = 1000
    mock_tokenizer.decode = Mock(return_value="ATGC")
    mock_tokenizer.batch_decode = Mock(return_value=["ATGC", "CGTA"])

    return mock_tokenizer


@pytest.fixture(scope="function")
def mock_config():
    """Return a Mock configured as a task configuration.

    Returns:
        Mock object configured with task settings.
    """
    mock_cfg = Mock()
    mock_cfg.task_type = "binary"
    mock_cfg.num_labels = 2
    mock_cfg.label_names = ["negative", "positive"]
    mock_cfg.threshold = 0.5
    mock_cfg.max_length = 512
    mock_cfg.batch_size = 8
    mock_cfg.head_config = {
        "head": "mlp",
        "num_classes": 2,
        "hidden_sizes": [256, 128],
    }

    return mock_cfg


@pytest.fixture(scope="function")
def sample_dna_sequence():
    """Return a valid DNA sequence string.

    Returns:
        36 bp DNA sequence containing only valid DNA characters.
    """
    return "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGC"


@pytest.fixture(scope="function")
def mock_inference_engine(mock_model, mock_tokenizer):
    """Return a Mock configured as DNAInference.

    Args:
        mock_model: The mock_model fixture.
        mock_tokenizer: The mock_tokenizer fixture.

    Returns:
        Mock object configured as a DNAInference engine.
    """
    mock_engine = Mock()
    mock_engine.predict = Mock(
        return_value=[{"label": "positive", "score": 0.95}]
    )
    mock_engine.predict_batch = Mock(
        return_value=[
            [{"label": "positive", "score": 0.95}],
            [{"label": "negative", "score": 0.88}],
        ]
    )
    mock_engine.embed = Mock(return_value=torch.randn(1, 768))
    mock_engine.model = mock_model
    mock_engine.tokenizer = mock_tokenizer

    return mock_engine


@pytest.fixture(scope="function")
def mock_dataset():
    """Return a Mock configured as a HuggingFace Dataset.

    Returns:
        Mock object configured as a Dataset.
    """
    mock_ds = Mock()
    mock_ds.__len__ = Mock(return_value=10)

    def mock_getitem(index):
        if index == 0:
            return {"sequence": "ATGC" * 10, "label": 1}
        else:
            return {"sequence": "CGTA" * 10, "label": 0}

    mock_ds.__getitem__ = mock_getitem
    mock_ds.to_pandas = Mock(
        return_value=pd.DataFrame(
            {"sequence": ["ATGC" * 10] * 10, "label": [0, 1] * 5}
        )
    )
    mock_ds.features = {
        "sequence": {"dtype": "string"},
        "label": {"dtype": "int64"},
    }

    return mock_ds
