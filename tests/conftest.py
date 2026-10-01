"""Shared pytest fixtures for DNALLM test suite."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock
import uuid

import pandas as pd
import pytest
import torch
import yaml


class SimpleDNATokenizer:
    """Deterministic character-level DNA tokenizer with real encode/decode behavior.

    Callable like a Hugging Face tokenizer (single or batched), pads and
    truncates to ``max_length``, and returns ``transformers.BatchEncoding``
    objects when ``return_tensors="pt"`` so ``.to(device)`` and dict
    unpacking work exactly like the real API. Maps ``N`` to the mask token id
    so masked-language-model scoring paths can be exercised deterministically.
    """

    vocab: ClassVar[list[str]] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "C", "G", "T"]
    char_overrides: ClassVar[dict[str, int]] = {"N": 4}

    def __init__(self, max_length: int = 32) -> None:
        self.max_length = max_length
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.vocab_size = len(self.vocab)
        self.padding_side = "right"
        self.all_special_ids = [0, 1, 2, 3, 4]
        self.special_tokens_map = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }

    def _encode(self, seq: str) -> list[int]:
        table = {c: i for i, c in enumerate(self.vocab)}
        return [self.char_overrides.get(ch.upper(), table.get(ch.upper(), 1)) for ch in seq]

    def __call__(
        self,
        sequences,
        truncation=True,
        padding=True,
        padding_side=None,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        seqs = [sequences] if isinstance(sequences, str) else list(sequences)
        ml = max_length or self.max_length
        ids = [self._encode(s)[:ml] if truncation else self._encode(s) for s in seqs]
        if padding in (True, "max_length", "longest"):
            if padding == "max_length":
                width = ml
            else:
                width = max((len(x) for x in ids), default=0)
            ids = [x + [self.pad_token_id] * (width - len(x)) for x in ids]
        masks = [[1 if i != self.pad_token_id else 0 for i in x] for x in ids]
        if return_tensors == "pt":
            from transformers import BatchEncoding

            return BatchEncoding({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            })
        return {"input_ids": ids, "attention_mask": masks}

    def encode(self, seq, **kwargs) -> list[int]:
        """Encode one sequence to a list of token ids."""
        return self._encode(seq)

    def decode(self, ids, skip_special_tokens=False, **kwargs) -> str:
        """Decode token ids (tensor, list, or nested list) back to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        toks = [self.vocab[i] for i in ids if 0 <= i < len(self.vocab)]
        if skip_special_tokens:
            toks = [t for t in toks if not t.startswith("[")]
        return "".join(toks)

    def batch_decode(self, batch, **kwargs) -> list[str]:
        """Decode a batch of id sequences to strings."""
        return [
            self.decode(x, skip_special_tokens=kwargs.get("skip_special_tokens", False))
            for x in batch
        ]

    def convert_ids_to_tokens(self, ids):
        """Convert ids (int or list) to token strings."""
        if isinstance(ids, int):
            return self.vocab[ids]
        return [self.vocab[i] for i in ids]

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens (str or list) to ids."""
        if isinstance(tokens, str):
            return self.vocab.index(tokens) if tokens in self.vocab else 1
        return [self.vocab.index(t) if t in self.vocab else 1 for t in tokens]

    def tokenize(self, seq: str) -> list[int]:
        """Tokenize a sequence to ids (CustomEvo embedding path contract)."""
        return self._encode(seq)


class TinyDNAModel(torch.nn.Module):
    """Real embedding+linear torch backbone producing autograd-capable outputs.

    Weights are built from a local ``torch.Generator`` seed so instances are
    deterministic without touching the global RNG. ``pooled=True`` produces
    ``(batch, n_classes)`` logits (sequence-level tasks); ``pooled=False``
    produces ``(batch, seq_len, n_classes)`` (token-level tasks and per-position
    scoring). Outputs carry ``.logits`` and ``.hidden_states`` like HF models.
    """

    def __init__(self, vocab_size=9, d_model=16, n_classes=2, pooled=True):
        super().__init__()
        gen = torch.Generator().manual_seed(42)
        emb_weight = torch.empty(vocab_size, d_model).uniform_(-1, 1, generator=gen)
        head_weight = torch.empty(n_classes, d_model).uniform_(-1, 1, generator=gen)
        self.pooled = pooled
        self.embedding = torch.nn.Embedding.from_pretrained(emb_weight, freeze=False)
        self.head = torch.nn.Linear(d_model, n_classes)
        self.head.weight.data = head_weight
        self.head.bias.data = torch.zeros(n_classes)
        self.config = SimpleNamespace(
            model_type="tiny_dna",
            hidden_size=d_model,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=vocab_size,
            output_attentions=False,
            output_hidden_states=False,
            attn_implementation="eager",
        )

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs
    ):
        if inputs_embeds is not None:
            emb = inputs_embeds
        else:
            emb = self.embedding(input_ids)
        if self.pooled:
            logits = self.head(emb.mean(dim=1))
        else:
            logits = self.head(emb)
        return SimpleNamespace(logits=logits, hidden_states=[emb])


@pytest.fixture
def simple_dna_tokenizer():
    """Return a real deterministic character-level DNA tokenizer."""
    return SimpleDNATokenizer()


@pytest.fixture
def tiny_model_factory():
    """Return the TinyDNAModel class for building shaped variants in tests."""
    return TinyDNAModel


@pytest.fixture
def tiny_real_model():
    """Return a default deterministic real torch DNA model (pooled, 2 classes)."""
    return TinyDNAModel()


@pytest.fixture
def inference_config_factory(tmp_path):
    """Return a factory building real loaded inference configs under tmp_path.

    The factory writes a YAML file and loads it via ``load_config`` so every
    test exercises the real Pydantic validation and alias normalization.
    """
    from dnallm.configuration.configs import load_config

    def _make(
        task_type="binary",
        num_labels=None,
        label_names=None,
        threshold=0.5,
        batch_size=2,
        device="cpu",
        max_length=32,
        num_workers=0,
        output_dir=None,
        use_fp16=False,
        use_bf16=False,
    ):
        task = {"task_type": task_type, "threshold": threshold}
        if num_labels is not None:
            task["num_labels"] = num_labels
        if label_names is not None:
            task["label_names"] = label_names
        inference = {
            "batch_size": batch_size,
            "device": device,
            "max_length": max_length,
            "num_workers": num_workers,
            "use_fp16": use_fp16,
            "use_bf16": use_bf16,
        }
        if output_dir is not None:
            inference["output_dir"] = str(output_dir)
        path = tmp_path / f"config-{task_type}-{uuid.uuid4().hex[:8]}.yaml"
        path.write_text(yaml.safe_dump({"task": task, "inference": inference}))
        return load_config(path)

    return _make


@pytest.fixture(scope="session", autouse=True)
def global_cleanup():
    """Session-scoped cleanup fixture."""
    return
    # Cleanup after all tests complete


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
def sample_dna_sequence():
    """Return a valid DNA sequence string.

    Returns:
        36 bp DNA sequence containing only valid DNA characters.
    """
    return "ATGCGTACGTTAGCTAGCTAGCTAGCTAGCTAGC"


@pytest.fixture
def mock_inference_engine(mock_model, mock_tokenizer):
    """Return a Mock configured as DNAInference.

    Args:
        mock_model: The mock_model fixture.
        mock_tokenizer: The mock_tokenizer fixture.

    Returns:
        Mock object configured as a DNAInference engine.
    """
    mock_engine = Mock()
    mock_engine.predict = Mock(return_value=[{"label": "positive", "score": 0.95}])
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


@pytest.fixture
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
        return_value=pd.DataFrame({"sequence": ["ATGC" * 10] * 10, "label": [0, 1] * 5})
    )
    mock_ds.features = {
        "sequence": {"dtype": "string"},
        "label": {"dtype": "int64"},
    }

    return mock_ds
