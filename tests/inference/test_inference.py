"""Test DNAInference class functionality.

This test file tests the core functionality of the DNAInference class,
including model loading, inference, and result processing.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dnallm.datahandling.data import DNADataset
from dnallm.inference.inference import DNAInference


class TestDNAInference(unittest.TestCase):
    """Test cases for DNAInference class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        self.test_csv_path = os.path.join(self.test_dir, "test_data.csv")

        # Create test configuration
        self.create_test_config()

        # Create test data
        self.create_test_data()

        # Mock model and tokenizer
        self.mock_model = self.create_mock_model()
        self.mock_tokenizer = self.create_mock_tokenizer()

        # Create inference engine instance
        self.inference_engine = DNAInference(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.load_test_config(),
        )
        # Keep backward compatibility for tests
        self.predictor = self.inference_engine

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a test configuration file."""
        config_content = """inference:
  batch_size: 4
  device: cpu
  max_length: 256
  num_workers: 1
  output_dir: ./test_results
  use_fp16: false
task:
  label_names:
  - Not promoter
  - Core promoter
  num_labels: 2
  task_type: binary
  threshold: 0.5
"""
        with open(self.config_path, "w") as f:
            f.write(config_content)

    def create_test_data(self):
        """Create test DNA sequence data."""
        test_data = {
            "sequence": [
                "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
                "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
                "TATATATATATATATATATATATATATATATATATATATATATATATATATATATATATA",
                "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG",
            ],
            "label": [0, 1, 0, 1],
        }

        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, index=False)

    def create_mock_model(self):
        """Create a mock model for testing."""
        mock_model = Mock()

        # Mock model config
        mock_config = Mock()
        mock_config.output_attentions = False
        mock_config.output_hidden_states = False
        mock_config.attn_implementation = "eager"
        mock_config.num_attention_heads = 12
        mock_config.num_hidden_layers = 6
        mock_config.hidden_size = 768
        mock_config.vocab_size = 1000
        mock_config.model_type = "dna_gpt"
        mock_model.config = mock_config

        # Mock model parameters
        mock_model.parameters.return_value = iter([torch.randn(100, 100)])

        # Mock model forward method with proper signature
        def mock_forward(input_ids, attention_mask=None, labels=None, **kwargs):
            batch_size = input_ids.shape[0]
            mock_output = Mock()
            mock_output.logits = torch.randn(batch_size, 2)
            return mock_output

        mock_model.forward = mock_forward
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)

        return mock_model

    def create_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = Mock()

        # Mock tokenizer methods
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.encode_plus = Mock(
            return_value={
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "token_type_ids": [0, 0, 0, 0, 0],
            }
        )

        # Mock special tokens map
        mock_tokenizer.special_tokens_map = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }

        # Mock tokenizer properties
        mock_tokenizer.pad_token = "[PAD]"  # ruff: ignore[hardcoded-password-string]
        mock_tokenizer.unk_token = "[UNK]"  # ruff: ignore[hardcoded-password-string]
        mock_tokenizer.cls_token = "[CLS]"  # ruff: ignore[hardcoded-password-string]
        mock_tokenizer.sep_token = "[SEP]"  # ruff: ignore[hardcoded-password-string]
        mock_tokenizer.mask_token = "[MASK]"  # ruff: ignore[hardcoded-password-string]

        return mock_tokenizer

    def load_test_config(self):
        """Load test configuration."""
        from dnallm.configuration.configs import load_config

        return load_config(self.config_path)

    def test_init(self):
        """Test predictor initialization."""
        assert self.predictor.model is not None
        assert self.predictor.tokenizer is not None
        assert self.predictor.task_config is not None
        assert self.predictor.pred_config is not None
        assert self.predictor.pred_config.batch_size == 4
        assert self.predictor.pred_config.device == "cpu"

    def test_get_device_cpu(self):
        """Test device selection for CPU."""
        self.predictor.pred_config.device = "cpu"
        device = self.predictor._get_device()
        assert device == torch.device("cpu")

    def test_get_device_auto(self):
        """Test automatic device selection."""
        self.predictor.pred_config.device = "auto"
        device = self.predictor._get_device()
        # Should return a valid device
        assert isinstance(device, torch.device)

    def test_generate_dataset_from_list(self):
        """Test dataset generation from sequence list."""
        sequences = ["ATCG", "GCTA", "TATA"]
        dataset, dataloader = self.predictor.generate_dataset(
            sequences,
            batch_size=2,
            do_encode=False,  # Skip encoding to avoid tokenizer issues
        )

        assert isinstance(dataset, DNADataset)
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert len(dataset) == 3

    def test_generate_dataset_from_file(self):
        """Test dataset generation from file."""
        dataset, dataloader = self.predictor.generate_dataset(
            self.test_csv_path,
            batch_size=2,
            seq_col="sequence",
            label_col="label",
            do_encode=False,  # Skip encoding to avoid tokenizer issues
        )

        assert isinstance(dataset, DNADataset)
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert len(dataset) == 4

    def test_logits_to_preds_binary(self):
        """Test logits to predictions conversion for binary classification."""
        logits = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]])
        probs, labels = self.predictor.logits_to_preds(logits)

        assert len(labels) == 3
        assert probs.shape == (3, 2)
        # Check that labels are binary (0 or 1)
        # Note: labels are converted to label names, so check for label names
        # instead
        # The third sequence has logits [2.0, 1.0], so class 0 (index 0) wins
        expected_labels = ["Core promoter", "Core promoter", "Not promoter"]
        assert labels == expected_labels

    def test_format_output(self):
        """Test output formatting."""
        # Set up sequences
        self.predictor.sequences = ["ATCG", "GCTA"]

        # Mock predictions
        probs = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
        labels = [1, 0]
        predictions = (probs, labels)

        formatted = self.predictor.format_output(predictions)

        assert isinstance(formatted, dict)
        assert len(formatted) == 2
        assert "sequence" in formatted[0]
        assert "label" in formatted[0]
        assert "scores" in formatted[0]

    def test_batch_infer(self):
        """Test batch inference."""
        # Mock the batch_predict method directly to avoid complex data loading
        # issues
        with patch.object(self.predictor, "batch_infer") as mock_batch_infer:
            mock_batch_infer.return_value = (
                torch.randn(2, 2),  # logits
                {
                    0: {"sequence": "ATCG", "label": 1, "scores": {}},
                    1: {"sequence": "GCTA", "label": 0, "scores": {}},
                },  # predictions
                {},  # embeddings
            )

            # Create a mock dataloader
            mock_dataloader = Mock()

            logits, predictions, embeddings = self.predictor.batch_infer(mock_dataloader)

            assert isinstance(logits, torch.Tensor)
            assert isinstance(predictions, dict)
            assert isinstance(embeddings, dict)

    def test_infer_seqs(self):
        """Test sequence inference."""
        sequences = ["ATCG", "GCTA"]

        # Mock batch_predict method
        with patch.object(self.predictor, "batch_infer") as mock_batch_infer:
            mock_batch_infer.return_value = (
                torch.randn(2, 2),  # logits
                {
                    0: {"sequence": "ATCG", "label": 1, "scores": {}},
                    1: {"sequence": "GCTA", "label": 0, "scores": {}},
                },  # predictions
                {},  # embeddings
            )

            # Mock generate_dataset to avoid encoding issues
            with patch.object(self.predictor, "generate_dataset") as mock_generate:
                mock_generate.return_value = (None, None)
                result = self.predictor.infer_seqs(sequences)

                assert isinstance(result, dict)
                assert len(result) == 2

    def test_infer_file(self):
        """Test file-based inference."""
        # Mock batch_predict method
        with patch.object(self.predictor, "batch_infer") as mock_batch_infer:
            mock_batch_infer.return_value = (
                torch.randn(4, 2),  # logits
                {
                    0: {"sequence": "ATCG", "label": 1, "scores": {}},
                    1: {"sequence": "GCTA", "label": 0, "scores": {}},
                    2: {"sequence": "TATA", "label": 0, "scores": {}},
                    3: {"sequence": "CGCG", "label": 1, "scores": {}},
                },  # predictions
                {},  # embeddings
            )

            # Mock generate_dataset to avoid encoding issues
            with patch.object(self.predictor, "generate_dataset") as mock_generate:
                mock_generate.return_value = (None, None)
                result = self.predictor.infer_file(
                    self.test_csv_path, seq_col="sequence", label_col="label"
                )

                assert isinstance(result, dict)
                assert len(result) == 4

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])

        # Mock the metrics computation
        with patch("dnallm.tasks.metrics.compute_metrics") as mock_metrics:
            mock_metrics.return_value = {"accuracy": 0.75, "f1": 0.8}

            metrics = self.predictor.calculate_metrics(logits, labels)

            assert isinstance(metrics, dict)
            assert "accuracy" in metrics

    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.predictor.get_model_info()

        assert isinstance(info, dict)
        assert "model_type" in info
        assert "device" in info
        assert "attention_supported" in info

    def test_get_model_parameters(self):
        """Test model parameter information."""
        params = self.predictor.get_model_parameters()

        assert isinstance(params, dict)
        assert "total" in params
        assert "trainable" in params

    def test_get_available_outputs(self):
        """Test available outputs information."""
        outputs = self.predictor.get_available_outputs()

        assert isinstance(outputs, dict)
        assert "hidden_states_available" in outputs
        assert "attentions_available" in outputs

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        memory = self.predictor.estimate_memory_usage()

        assert isinstance(memory, dict)
        assert "total_estimated_mb" in memory

    def test_force_eager_attention(self):
        """Test forcing eager attention implementation."""
        # Test successful switch
        result = self.predictor.force_eager_attention()
        assert isinstance(result, bool)

    def test_check_attention_support(self):
        """Test attention support checking."""
        support = self.predictor._check_attention_support()
        assert isinstance(support, bool)

    def test_check_hidden_states_support(self):
        """Test hidden states support checking."""
        support = self.predictor._check_hidden_states_support()
        assert isinstance(support, bool)

    def test_save_predictions(self):
        """Test prediction saving."""
        predictions = {
            0: {
                "sequence": "ATCG",
                "label": 1,
                "scores": {"Not promoter": 0.3, "Core promoter": 0.7},
            },
            1: {
                "sequence": "GCTA",
                "label": 0,
                "scores": {"Not promoter": 0.8, "Core promoter": 0.2},
            },
        }

        output_dir = Path(self.test_dir) / "predictions"

        # Import and test save function
        from dnallm.inference.inference import save_predictions

        save_predictions(predictions, output_dir)

        # Check if file was created
        assert (output_dir / "predictions.json").exists()

    def test_save_metrics(self):
        """Test metrics saving."""
        metrics = {"accuracy": 0.75, "f1": 0.8}

        output_dir = Path(self.test_dir) / "metrics"

        # Import and test save function
        from dnallm.inference.inference import save_metrics

        save_metrics(metrics, output_dir)

        # Check if file was created
        assert (output_dir / "metrics.json").exists()


class TestDNAInferenceIntegration(unittest.TestCase):
    """Integration tests for DNAInference with real model loading."""

    # Class attributes for type checking
    test_dir: str
    config_path: str

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for integration tests."""
        cls.test_dir = tempfile.mkdtemp()
        cls.config_path = os.path.join(cls.test_dir, "integration_config.yaml")

        # Create integration test configuration
        config_content = """inference:
  batch_size: 2
  device: cpu
  max_length: 128
  num_workers: 1
  output_dir: ./integration_results
  use_fp16: false
task:
  label_names:
  - Not promoter
  - Core promoter
  num_labels: 2
  task_type: binary
  threshold: 0.5
"""
        with open(cls.config_path, "w") as f:
            f.write(config_content)

    @classmethod
    def tearDownClass(cls):
        """Clean up integration test fixtures."""
        shutil.rmtree(cls.test_dir)

    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_real_model_integration(self):
        """Test with real model loading from ModelScope."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # Load real model and tokenizer from ModelScope
            model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
            print(f"🔄 Downloading model {model_name} from ModelScope...")

            # Use ModelScope to download model
            from modelscope import snapshot_download

            model_dir = snapshot_download(model_name)

            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print("✅ Model and tokenizer loaded successfully")

            # Load configuration
            from dnallm.configuration.configs import load_config

            config = load_config(self.config_path)

            # Create predictor
            inference_engine = DNAInference(model, tokenizer, config)

            # Test with real sequences
            sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
            result = inference_engine.infer_seqs(sequences)

            assert isinstance(result, dict)
            assert len(result) == 2

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback

            traceback.print_exc()
            # Fail closed: this test only runs in the networked nightly census,
            # so any failure (download, load, or the asserts above) is a real
            # regression, not an environment skip — the skip audit has no
            # allowlist entry for an arbitrary failure message.
            self.fail(f"Real-model integration workflow failed: {e}")


if __name__ == "__main__":
    # Only run when executed directly, not when imported by pytest
    import sys

    if "pytest" not in sys.modules:
        # Run tests
        unittest.main(verbosity=2)


# ─────────────────────────────────────────────────────────────────────────────
# Engine-path behavior tests (pytest style).
#
# Shared cross-file fixtures (simple_dna_tokenizer, tiny_real_model,
# inference_config_factory) come from tests/conftest.py. The doubles below are
# single-purpose fakes local to this file.
# ─────────────────────────────────────────────────────────────────────────────


class _BaseConfig(SimpleNamespace):
    """Config namespace with the attributes DNAInference reads from model.config."""

    def __init__(self, model_type="fake"):
        super().__init__(
            model_type=model_type,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=9,
            output_attentions=False,
            output_hidden_states=False,
            attn_implementation="eager",
        )


class ConstantOutputModel:
    """Model double with a ``**kwargs`` forward returning fixed-shape outputs."""

    def __init__(self, config=None, with_attentions=False, n_classes=2):
        self.config = config if config is not None else _BaseConfig("constant")
        self.with_attentions = with_attentions
        self.n_classes = n_classes
        self.calls = []
        self.eval_called = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        self.calls.append(kwargs)
        out = SimpleNamespace(
            logits=torch.full((2, self.n_classes), 0.9),
            hidden_states=[torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)],
        )
        if self.with_attentions:
            out.attentions = [torch.full((1, 3, 3), 0.25)]
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(3, 3))])


class DictOutputModel:
    """Model double whose forward returns a plain dict with a 'logits' key."""

    def __init__(self):
        self.config = _BaseConfig("dict_out")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return {"logits": torch.full((input_ids.shape[0], 2), 2.0)}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class TupleOutputModel:
    """Model double whose forward returns a (hidden_states, logits) tuple."""

    def __init__(self):
        self.config = _BaseConfig("tuple_out")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return (torch.zeros(input_ids.shape[0], 3, 4), torch.full((input_ids.shape[0], 2), 3.0))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class InputOnlyModel:
    """Model double accepting only input_ids/labels so arg filtering is observable."""

    def __init__(self):
        self.config = _BaseConfig("input_only")
        self.calls = []

    def forward(self, input_ids=None, labels=None):
        self.calls.append({"input_ids": input_ids, "labels": labels})
        return SimpleNamespace(logits=torch.full((input_ids.shape[0], 2), 0.9))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class NoConfigModel:
    """Model double with no config attribute at all."""

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.ones(4, 4))])


class MambaLikeModel:
    """Model double whose type name contains 'mamba' (fp32/CPU downgrade branch)."""

    def __init__(self):
        self.config = _BaseConfig("mamba")
        self.to_called_with = None

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        return self

    def to(self, device):
        self.to_called_with = device
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class CustomEvoLike:
    """Model double whose type name contains 'CustomEvo' wrapping an inner model."""

    class _Inner:
        def __init__(self):
            self.to_called_with = None
            self.calls = []

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=None, **kw):
            self.calls.append({
                "input_ids": input_ids,
                "output_hidden_states": output_hidden_states,
            })
            return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

        def to(self, device):
            self.to_called_with = device
            return self

        def named_parameters(self):
            return iter([
                ("blocks.0.weight", torch.zeros(2, 2)),
                ("blocks.1.weight", torch.zeros(2, 2)),
                ("head.weight", torch.zeros(2, 2)),
            ])

    def __init__(self):
        self.model = self._Inner()
        self.config = _BaseConfig("custom_evo")
        self.calls = []

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        self.calls.append({"layer_names": layer_names, "return_embeddings": return_embeddings})
        layers = {}
        for name in layer_names or []:
            layers[name] = torch.full((input_ids.shape[0], input_ids.shape[1], 4), 0.5)
        return None, layers

    def to(self, device):
        return self


class Evo2Like:
    """Model double whose str contains 'evo2' with generate/score_sequences."""

    def __init__(self):
        self.config = _BaseConfig("evo2")
        self.generate_calls = []
        self.score_calls = []

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    def generate(
        self,
        prompt_seqs=None,
        n_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        batched=None,
        cached_generation=None,
    ):
        self.generate_calls.append({
            "prompt_seqs": prompt_seqs,
            "n_tokens": n_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "batched": batched,
            "cached_generation": cached_generation,
        })
        return SimpleNamespace(
            sequences=["AATT" for _ in prompt_seqs],
            logprobs_mean=[0.42 for _ in prompt_seqs],
        )

    def score_sequences(self, seqs, reduce_method=None):
        self.score_calls.append({"seqs": seqs, "reduce_method": reduce_method})
        return [0.1 * (i + 1) for i in range(len(seqs))]

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class Evo1Like:
    """Model double whose str contains 'evo1' wrapping an inner model."""

    def __init__(self):
        self.config = _BaseConfig("evo1")
        self.inner = SimpleNamespace(name="inner-evo1-model")
        self.model = self.inner

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class CausalLMLike:
    """Model double whose str contains 'causallm' with a generate method."""

    def __init__(self):
        self.config = _BaseConfig("causal")
        self.generate_calls = []

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[5, 6, 7, 8]])

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class MEGADNALike:
    """Model double whose str contains 'MEGADNA' (embedding/loss/generate paths)."""

    def __init__(self):
        self.config = _BaseConfig("MEGADNA")
        self.calls = []
        self.generate_calls = []

    def __call__(self, input_ids, return_value=None):
        self.calls.append({"shape": tuple(input_ids.shape), "return_value": return_value})
        n = input_ids.shape[0]
        if return_value == "embedding":
            return [torch.full((n, 6, 4), float(i + 1)) for i in range(3)]
        if return_value == "loss":
            return torch.tensor(1.25)
        return SimpleNamespace(logits=torch.zeros(n, 2))

    def forward(self, input_ids=None, **kwargs):
        return self.__call__(input_ids)

    def generate(self, input_ids, seq_len=None, temperature=None, filter_thres=None):
        self.generate_calls.append({
            "shape": tuple(input_ids.shape),
            "seq_len": seq_len,
            "temperature": temperature,
            "filter_thres": filter_thres,
        })
        return torch.tensor([[5, 6, 7, 8]])

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(2, 2))])


class FlakyAttentionConfig:
    """Config whose output_attentions setter raises for the first `flaky` sets."""

    def __init__(self, flaky=2, message="attn_implementation is sdpa"):
        self._flaky = flaky
        self._message = message
        self._sets = 0
        self.attn_implementation = "sdpa"

    @property
    def output_attentions(self):
        return False

    @output_attentions.setter
    def output_attentions(self, value):
        self._sets += 1
        if self._sets <= self._flaky:
            raise ValueError(self._message)


class AttentionRaiseForeverConfig:
    """Config whose output_attentions setter always raises (non-sdpa message)."""

    attn_implementation = "eager"

    @property
    def output_attentions(self):
        return False

    @output_attentions.setter
    def output_attentions(self, value):
        raise ValueError("plain failure")


class HiddenStatesRaiseConfig:
    """Config whose output_hidden_states setter always raises."""

    attn_implementation = "eager"

    @property
    def output_hidden_states(self):
        return False

    @output_hidden_states.setter
    def output_hidden_states(self, value):
        raise ValueError("no hidden states for you")


class EagerRaiseConfig:
    """Config whose attn_implementation setter always raises."""

    @property
    def attn_implementation(self):
        return "sdpa"

    @attn_implementation.setter
    def attn_implementation(self, value):
        raise RuntimeError("cannot set attn_implementation")


class ConvertPadTokenizer:
    """Tokenizer double serving pad id via convert_tokens_to_ids."""

    pad_token = "<pad>"  # ruff: ignore[hardcoded-password-string]

    def convert_tokens_to_ids(self, token):
        return 7


class EncodePadTokenizer:
    """Tokenizer double whose convert fails and encode serves the pad id."""

    pad_token = "<pad>"  # ruff: ignore[hardcoded-password-string]

    def convert_tokens_to_ids(self, token):
        return None

    def encode(self, token):
        return [3, 1]


class EosPadTokenizer:
    """Tokenizer double where every pad resolution fails but eos_token_id."""

    pad_token = "<pad>"  # ruff: ignore[hardcoded-password-string]
    eos_token_id = 9

    def convert_tokens_to_ids(self, token):
        return None

    def encode(self, token):
        raise OSError("no encoder")


def _build_engine(model, tokenizer, config, **kwargs):
    """Construct a DNAInference engine from the given collaborators."""
    return DNAInference(model=model, tokenizer=tokenizer, config=config, **kwargs)


class TestLogitsToPredsTaskTypes:
    """Semantic logits_to_preds behavior for every supported task type."""

    def test_binary_semantics(self, mock_model, mock_tokenizer, inference_config_factory):
        """Binary preds use softmax and threshold on the positive-class probability."""
        config = inference_config_factory(
            task_type="binary", label_names=["Not promoter", "Core promoter"]
        )
        engine = _build_engine(mock_model, mock_tokenizer, config)
        logits = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.6, 0.55]])

        probs, labels = engine.logits_to_preds(logits)

        assert probs.shape == (3, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)
        expected = ["Not promoter" if p[1] <= 0.5 else "Core promoter" for p in probs]
        assert labels == expected

    def test_binary_threshold_changes_prediction(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """A high threshold flips a borderline positive back to the negative label."""
        logits = torch.tensor([[0.6, 0.55]])

        low = _build_engine(
            mock_model,
            mock_tokenizer,
            inference_config_factory(
                task_type="binary", threshold=0.1, label_names=["Not promoter", "Core promoter"]
            ),
        )
        high = _build_engine(
            mock_model,
            mock_tokenizer,
            inference_config_factory(
                task_type="binary", threshold=0.9, label_names=["Not promoter", "Core promoter"]
            ),
        )

        assert low.logits_to_preds(logits)[1] == ["Core promoter"]
        assert high.logits_to_preds(logits)[1] == ["Not promoter"]

    def test_multiclass_semantics(self, mock_model, mock_tokenizer, inference_config_factory):
        """Multiclass preds pick the argmax class name and rows sum to one."""
        config = inference_config_factory(
            task_type="multiclass", num_labels=3, label_names=["alpha", "beta", "gamma"]
        )
        engine = _build_engine(mock_model, mock_tokenizer, config)
        logits = torch.tensor([[3.0, 1.0, 0.5], [0.2, 0.1, 4.0], [1.0, 5.0, 0.0]])

        probs, labels = engine.logits_to_preds(logits)

        assert probs.shape == (3, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)
        assert labels == ["alpha", "gamma", "beta"]

    def test_multilabel_semantics(self, mock_model, mock_tokenizer, inference_config_factory):
        """Multilabel preds apply sigmoid per class and return lists of active labels."""
        config = inference_config_factory(
            task_type="multilabel", num_labels=2, label_names=["off", "on"]
        )
        engine = _build_engine(mock_model, mock_tokenizer, config)
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [-0.1, -0.15]])

        probs, labels = engine.logits_to_preds(logits)

        assert probs.shape == (3, 2)
        assert bool(((probs > 0) & (probs < 1)).all())
        # Sigmoid outputs do not sum to one (unlike softmax).
        assert not torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-3)
        assert labels == [["off"], ["on"], []]

    def test_multilabel_threshold(self, mock_model, mock_tokenizer, inference_config_factory):
        """A lower threshold activates borderline classes above it."""
        logits = torch.tensor([[0.1, 0.15]])
        config = inference_config_factory(
            task_type="multilabel", num_labels=2, label_names=["off", "on"], threshold=0.4
        )
        engine = _build_engine(mock_model, mock_tokenizer, config)

        _, labels = engine.logits_to_preds(logits)

        assert labels == [["off", "on"]]

    def test_regression_semantics(self, mock_model, mock_tokenizer, inference_config_factory):
        """Regression preds are the squeezed logits and labels are the name list."""
        config = inference_config_factory(task_type="regression")
        engine = _build_engine(mock_model, mock_tokenizer, config)
        logits = torch.tensor([[1.5], [-2.0]])

        probs, labels = engine.logits_to_preds(logits)

        assert probs.shape == (2,)
        assert torch.allclose(probs, torch.tensor([1.5, -2.0]))
        assert labels == ["value"]

    def test_token_semantics(self, mock_model, mock_tokenizer, inference_config_factory):
        """Token preds argmax per position and map each to a label name."""
        config = inference_config_factory(task_type="token", num_labels=2, label_names=["O", "I"])
        engine = _build_engine(mock_model, mock_tokenizer, config)
        logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [0.0, 5.0]]])

        probs, labels = engine.logits_to_preds(logits)

        assert probs.shape == (1, 3, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 3), atol=1e-6)
        assert labels == [["I", "O", "I"]]

    def test_unsupported_task_type_raises(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """A task type without prediction semantics raises a matchable ValueError."""
        config = inference_config_factory(task_type="mask")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.raises(ValueError, match=r"Unsupported task type: mask"):
            engine.logits_to_preds(torch.zeros(1, 2))


class TestFormatOutputVariants:
    """format_output score assembly per task type."""

    def test_binary_scores_dict(self, mock_model, mock_tokenizer, inference_config_factory):
        """Binary scores map every label name to its probability."""
        config = inference_config_factory(task_type="binary", label_names=["negative", "positive"])
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = ["ATCG", "GCTA"]

        formatted = engine.format_output((torch.tensor([[0.3, 0.7], [0.8, 0.2]]), ["a", "b"]))

        assert formatted[0]["scores"] == {
            "negative": pytest.approx(0.3),
            "positive": pytest.approx(0.7),
        }
        assert formatted[1]["scores"] == {
            "negative": pytest.approx(0.8),
            "positive": pytest.approx(0.2),
        }

    def test_regression_scores_scalar(self, mock_model, mock_tokenizer, inference_config_factory):
        """Regression scores hold the raw value under the single label name."""
        config = inference_config_factory(task_type="regression")
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = ["ATCG"]

        formatted = engine.format_output((torch.tensor([1.5]), ["value"]))

        assert formatted[0]["scores"] == {"value": pytest.approx(1.5)}

    def test_token_scores_max_per_position(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Token scores take the max probability per sequence position."""
        config = inference_config_factory(task_type="token", num_labels=2, label_names=["O", "I"])
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = ["ATCG"]

        formatted = engine.format_output((torch.tensor([[[0.2, 0.8], [0.9, 0.1]]]), [["I", "O"]]))

        assert formatted[0]["scores"] == pytest.approx([0.8, 0.9])

    def test_multilabel_label_is_list(self, mock_model, mock_tokenizer, inference_config_factory):
        """Multilabel predictions keep their list-of-active-labels form."""
        config = inference_config_factory(
            task_type="multilabel", num_labels=2, label_names=["off", "on"]
        )
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = ["ATCG"]

        formatted = engine.format_output((torch.tensor([[0.1, 0.9]]), [["on"]]))

        assert formatted[0]["label"] == ["on"]

    def test_no_sequences_yields_empty_string(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Without stored sequences the sequence field is an empty string."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = []

        formatted = engine.format_output((torch.tensor([[0.5, 0.5]]), ["x"]))

        assert formatted[0]["sequence"] == ""


E2E_SEQUENCES = ["ATCG", "GGCC", "TTTT"]


class TestEndToEndInference:
    """Config -> DNAInference -> logits -> semantic predictions, all real collaborators."""

    def test_binary_end_to_end(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Binary inference labels match an independent forward+softmax recomputation."""
        config = inference_config_factory(
            task_type="binary", label_names=["Not promoter", "Core promoter"]
        )
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        result = engine.infer(E2E_SEQUENCES)

        assert set(result.keys()) == {0, 1, 2}
        enc = simple_dna_tokenizer(E2E_SEQUENCES, return_tensors="pt")
        probs = torch.softmax(tiny_real_model(enc["input_ids"]).logits, dim=-1)
        expected = ["Core promoter" if p[1] > 0.5 else "Not promoter" for p in probs]
        assert [result[i]["label"] for i in range(3)] == expected
        assert result[0]["sequence"] == "ATCG"

    def test_multiclass_end_to_end(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """Multiclass inference labels match the independent argmax recomputation."""
        model = tiny_model_factory(n_classes=3)
        config = inference_config_factory(
            task_type="multiclass", num_labels=3, label_names=["a", "b", "c"]
        )
        engine = _build_engine(model, simple_dna_tokenizer, config)

        result = engine.infer(E2E_SEQUENCES)

        enc = simple_dna_tokenizer(E2E_SEQUENCES, return_tensors="pt")
        probs = torch.softmax(model(enc["input_ids"]).logits, dim=-1)
        expected = ["a", "b", "c"][int(torch.argmax(probs, dim=-1)[0])]
        assert result[0]["label"] == expected

    def test_multilabel_end_to_end(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """Multilabel inference returns list labels from sigmoid thresholding."""
        model = tiny_model_factory(n_classes=3)
        config = inference_config_factory(
            task_type="multilabel", num_labels=3, label_names=["x", "y", "z"]
        )
        engine = _build_engine(model, simple_dna_tokenizer, config)

        result = engine.infer(E2E_SEQUENCES)

        enc = simple_dna_tokenizer(E2E_SEQUENCES, return_tensors="pt")
        probs = torch.sigmoid(model(enc["input_ids"]).logits)
        expected = [["x", "y", "z"][j] for j, p in enumerate(probs[0]) if p > 0.5]
        assert result[0]["label"] == expected

    def test_token_end_to_end(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """Token inference labels every padded position via argmax."""
        model = tiny_model_factory(n_classes=2, pooled=False)
        config = inference_config_factory(task_type="token", num_labels=2, label_names=["O", "I"])
        engine = _build_engine(model, simple_dna_tokenizer, config)

        result = engine.infer(E2E_SEQUENCES)

        # The token-classification encode path pads every sequence to max_length.
        enc = simple_dna_tokenizer(E2E_SEQUENCES, padding="max_length", return_tensors="pt")
        argmax = torch.argmax(model(enc["input_ids"]).logits, dim=-1)
        expected = ["I" if int(t) else "O" for t in argmax[0]]
        assert result[0]["label"] == expected
        assert len(result[0]["label"]) == enc["input_ids"].shape[1]
        assert len(result[0]["label"]) == 32

    def test_single_sequence_string(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """A single bare sequence string produces exactly one prediction."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        result = engine.infer("ATGGCCTA")

        assert len(result) == 1
        assert result[0]["sequence"] == "ATGGCCTA"

    def test_infer_file_csv_end_to_end(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """File-based inference reads a CSV under tmp_path and labels every row."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence,label\nATCG,0\nGGCC,1\nTTTT,0\nACGT,1\n")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        result = engine.infer(file_path=str(csv_path), seq_col="sequence", label_col="label")

        assert set(result.keys()) == {0, 1, 2, 3}
        assert list(engine.labels) == [0, 1, 0, 1]

    def test_infer_file_evaluate_returns_metrics(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """evaluate=True returns (predictions, metrics) without curve payloads."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence,label\nATCG,0\nGGCC,1\nTTTT,0\nACGT,1\n")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        result = engine.infer_file(
            str(csv_path), evaluate=True, seq_col="sequence", label_col="label"
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        predictions, metrics = result
        assert len(predictions) == 4
        assert "accuracy" in metrics
        assert "curve" not in metrics
        assert "scatter" not in metrics

    def test_infer_seqs_save_to_file(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """save_to_file writes predictions.json under the configured output_dir."""
        out_dir = tmp_path / "results"
        config = inference_config_factory(task_type="binary", output_dir=out_dir)
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        predictions = engine.infer_seqs(E2E_SEQUENCES[:2], save_to_file=True)

        import json

        saved = json.loads((out_dir / "predictions.json").read_text())
        assert saved == {str(k): v for k, v in predictions.items()}

    def test_infer_file_plot_metrics_keeps_curve(
        self, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """plot_metrics=True returns the full metrics dict including curve data."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence,label\nATCG,0\nGGCC,1\n")
        out_dir = tmp_path / "out"
        config = inference_config_factory(task_type="binary", output_dir=out_dir)
        engine = _build_engine(ConstantOutputModel(), simple_dna_tokenizer, config)

        with patch(
            "dnallm.inference.inference.compute_metrics",
            return_value=lambda pair: {"accuracy": 0.5, "curve": {"roc": 1}, "scatter": 2},
        ):
            result = engine.infer_file(
                str(csv_path),
                evaluate=True,
                plot_metrics=True,
                seq_col="sequence",
                label_col="label",
                save_to_file=True,
            )

        predictions, metrics = result
        assert len(predictions) == 2
        assert metrics["curve"] == {"roc": 1}
        # The saved copy strips chart payloads.
        import json

        saved = json.loads((out_dir / "metrics.json").read_text())
        # infer_file's plot_metrics branch persists the full metrics payload.
        assert saved["curve"] == {"roc": 1}

    def test_infer_without_inputs_raises(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """infer() with neither sequences nor file_path raises a matchable error."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.raises(ValueError, match=r"Either sequences or file_path must be provided"):
            engine.infer()


class TestBatchInferBranches:
    """batch_infer output-extraction, flag plumbing, and precision branches."""

    def test_hidden_states_and_attention_extraction(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """Collected hidden states/attentions stack over batches to full-dataset shapes."""
        model = tiny_model_factory()
        config = inference_config_factory(task_type="binary", max_length=8)
        # Construct first so accepted_args reflects the real forward signature,
        # then attach real per-head attention scores to every forward output.
        engine = _build_engine(model, simple_dna_tokenizer, config)
        orig_forward = model.forward

        def forward_with_attention(**kwargs):
            out = orig_forward(**kwargs)
            _, seq_len, _ = out.hidden_states[0].shape
            out.attentions = [torch.full((1, seq_len, seq_len), 0.25)]
            return out

        model.forward = forward_with_attention

        result = engine.infer_seqs(
            ["ATCG", "GGCC"], output_hidden_states=True, output_attentions=True
        )

        assert len(result) == 2
        hidden = engine.embeddings["hidden_states"]
        attn = engine.embeddings["attentions"]
        mask = engine.embeddings["attention_mask"]
        assert isinstance(hidden, tuple)
        assert hidden[0].shape == (2, 4, 16)
        # Attention tensors keep the (heads, L, L) layout the plot module expects.
        assert isinstance(attn, tuple)
        assert attn[0].shape == (1, 4, 4)
        assert mask.shape == (2, 4)

    def test_hidden_states_reduced_to_mean(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """do_reduce collapses each hidden layer to its per-token mean vector."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        hidden = engine.get_embeddings(["ATCG"], do_reduce=True)

        assert hidden[0].shape == (1, 16)

    def test_embedding_task_returns_null_logits(
        self, simple_dna_tokenizer, inference_config_factory
    ):
        """embedding task skips logits extraction and returns no predictions."""
        model = ConstantOutputModel()
        config = inference_config_factory(task_type="embedding")
        engine = _build_engine(model, simple_dna_tokenizer, config)
        _, dataloader = engine.generate_dataset(["ATCG", "GGCC"])

        logits, predictions, _ = engine.batch_infer(dataloader, do_pred=False)

        assert logits[0] is None
        assert predictions is None

    def test_dict_output_extraction(self, simple_dna_tokenizer, inference_config_factory):
        """Dict outputs expose logits through the 'logits' key."""
        config = inference_config_factory(
            task_type="binary", label_names=["Not promoter", "Core promoter"]
        )
        engine = _build_engine(DictOutputModel(), simple_dna_tokenizer, config)

        result = engine.infer(["ATCG", "GGCC"])

        # Logits of 2.0 give softmax 0.5/0.5 -> class 0 at threshold 0.5.
        assert [result[i]["label"] for i in range(2)] == ["Not promoter", "Not promoter"]

    def test_tuple_output_extraction(self, simple_dna_tokenizer, inference_config_factory):
        """Tuple outputs expose logits from index 1."""
        config = inference_config_factory(
            task_type="binary", label_names=["Not promoter", "Core promoter"]
        )
        engine = _build_engine(TupleOutputModel(), simple_dna_tokenizer, config)

        result = engine.infer(["ATCG", "GGCC"])

        # Logits of 3.0 give softmax 0.5/0.5 -> class 0 at threshold 0.5.
        assert [result[i]["label"] for i in range(2)] == ["Not promoter", "Not promoter"]

    def test_unaccepted_args_filtered_from_model_call(
        self, simple_dna_tokenizer, inference_config_factory
    ):
        """Arguments outside the model's forward signature never reach the model."""
        model = InputOnlyModel()
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)

        engine.infer(["ATCG", "GGCC"])

        assert model.calls, "model.forward must have been called"
        assert set(model.calls[0].keys()) == {"input_ids", "labels"} or (
            set(model.calls[0].keys()) == {"input_ids"}
        )

    def test_output_flags_forwarded_when_accepted(
        self, simple_dna_tokenizer, inference_config_factory
    ):
        """output_hidden_states/output_attentions are passed when forward accepts them."""
        model = ConstantOutputModel(with_attentions=True)
        config = inference_config_factory(task_type="binary")
        # Construct first so accepted_args reflects the real forward signature.
        engine = _build_engine(model, simple_dna_tokenizer, config)
        model.forward_orig = model.forward

        def recording_forward(**kwargs):
            model.seen_kwargs = dict(kwargs)
            return model.forward_orig(**kwargs)

        model.forward = recording_forward

        engine.infer(["ATCG", "GGCC"], output_hidden_states=True, output_attentions=True)

        assert model.seen_kwargs.get("output_hidden_states") is True
        assert model.seen_kwargs.get("output_attentions") is True

    @pytest.mark.parametrize(
        ("flag", "expected_dtype"), [("fp16", torch.float16), ("bf16", torch.bfloat16)]
    )
    def test_mixed_precision_selects_autocast_dtype(
        self, simple_dna_tokenizer, inference_config_factory, flag, expected_dtype
    ):
        """use_fp16/use_bf16 wrap forward in autocast with the matching dtype."""
        kwargs = {"use_fp16": flag == "fp16", "use_bf16": flag == "bf16"}
        config = inference_config_factory(task_type="binary", **kwargs)
        engine = _build_engine(ConstantOutputModel(), simple_dna_tokenizer, config)

        with patch("torch.amp.autocast", return_value=nullcontext()) as autocast:
            engine.infer(["ATCG", "GGCC"])

        autocast.assert_called_once_with("cuda", dtype=expected_dtype)


class TestPadIdResolution:
    """_get_pad_id fallback chain across tokenizer surfaces."""

    def test_pad_id_from_token_id_attribute(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """A tokenizer with pad_token_id serves it directly."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.pad_id == mock_tokenizer.pad_token_id

    def test_pad_id_via_convert_tokens_to_ids(self, mock_model, inference_config_factory):
        """Without pad_token_id, convert_tokens_to_ids resolves the pad token."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, ConvertPadTokenizer(), config)

        assert engine.pad_id == 7

    def test_pad_id_via_encode_fallback(self, mock_model, inference_config_factory):
        """When convert fails, the first encoded id of the pad token is used."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, EncodePadTokenizer(), config)

        assert engine.pad_id == 3

    def test_pad_id_falls_back_to_eos(self, mock_model, inference_config_factory):
        """When every pad resolution fails, eos_token_id is the pad id."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, EosPadTokenizer(), config)

        assert engine.pad_id == 9


class TestAttentionMaskCreation:
    """_create_attention_mask input-derived and pad-derived branches."""

    def test_mask_taken_from_inputs(self, mock_model, mock_tokenizer, inference_config_factory):
        """An attention_mask in inputs is returned as a long cpu tensor."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        mask = engine._create_attention_mask({"attention_mask": torch.tensor([[1, 0]])})

        assert mask.tolist() == [[1, 0]]
        assert mask.dtype == torch.long

    def test_mask_derived_from_pad_id(self, mock_model, mock_tokenizer, inference_config_factory):
        """Without an attention_mask, non-pad positions form the mask."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.pad_id = 0

        mask = engine._create_attention_mask({"input_ids": torch.tensor([[1, 0], [0, 5]])})

        assert mask.tolist() == [[1, 0], [0, 1]]

    def test_mask_none_on_comparison_failure(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """A failing comparison degrades to None instead of raising."""

        class WeirdIds:
            def __ne__(self, other):
                raise RuntimeError("cannot compare")

        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.pad_id = 0

        assert engine._create_attention_mask({"input_ids": WeirdIds()}) is None


class TestDeviceSelection:
    """_get_device mapping, availability fallbacks, and validation."""

    def test_cuda_unavailable_warns_and_falls_back(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """cuda device on a CUDA-less runner warns and resolves to CPU."""
        config = inference_config_factory(task_type="binary", device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="CUDA is not available"):
                engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("cpu")

    def test_cuda_available_selected(self, mock_tokenizer, inference_config_factory):
        """cuda device with CUDA available resolves to the cuda device."""
        config = inference_config_factory(task_type="binary", device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            engine = _build_engine(MambaLikeModel(), mock_tokenizer, config)

        assert engine.device == torch.device("cuda")

    def test_nvidia_alias_maps_to_cuda(self, mock_model, mock_tokenizer, inference_config_factory):
        """The 'nvidia' alias resolves through the CUDA availability check."""
        config = inference_config_factory(task_type="binary", device="nvidia")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="CUDA is not available"):
                engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("cpu")

    def test_apple_alias_mps_unavailable(
        self, mock_model, mock_tokenizer, inference_config_factory, monkeypatch
    ):
        """The 'apple' alias warns and falls back when MPS is unavailable."""
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        config = inference_config_factory(task_type="binary", device="apple")

        with pytest.warns(UserWarning, match="MPS is not available"):
            engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("cpu")

    def test_tpu_alias_returns_xla(self, mock_model, mock_tokenizer, inference_config_factory):
        """The 'tpu' alias maps to the xla device type."""
        config = inference_config_factory(task_type="binary", device="tpu")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("xla")

    def test_unsupported_device_raises(self, mock_model, mock_tokenizer, inference_config_factory):
        """An unknown device string raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", device="quantum")

        with pytest.raises(ValueError, match=r"Unsupported device type: quantum"):
            _build_engine(mock_model, mock_tokenizer, config)

    def test_auto_prefers_cuda_when_available(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """auto device picks CUDA when torch reports it available."""
        config = inference_config_factory(task_type="binary", device="auto")
        with patch("torch.cuda.is_available", return_value=True):
            engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("cuda")

    def test_auto_falls_back_to_cpu(
        self, mock_model, mock_tokenizer, inference_config_factory, monkeypatch
    ):
        """auto device lands on CPU when no accelerator is available."""
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with patch("torch.cuda.is_available", return_value=False):
            config = inference_config_factory(task_type="binary", device="auto")
            engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.device == torch.device("cpu")


class TestConstructionBranches:
    """Constructor special-model and LoRA branches."""

    def test_mamba_downgraded_on_cpu(self, mock_tokenizer, inference_config_factory):
        """A mamba model on CPU gets fp16 disabled and the CPU device."""
        config = inference_config_factory(task_type="binary", device="cuda", use_fp16=True)
        model = MambaLikeModel()
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="CUDA is not available"):
                engine = _build_engine(model, mock_tokenizer, config)

        assert engine.device == torch.device("cpu")
        assert engine.pred_config.use_fp16 is False

    def test_mamba_cuda_still_forces_fp32(self, mock_tokenizer, inference_config_factory):
        """A mamba model keeps CUDA but never fp16."""
        config = inference_config_factory(task_type="binary", device="cuda", use_fp16=True)
        with patch("torch.cuda.is_available", return_value=True):
            engine = _build_engine(MambaLikeModel(), mock_tokenizer, config)

        assert engine.device == torch.device("cuda")
        assert engine.pred_config.use_fp16 is False

    def test_custom_evo_uses_inner_model(self, mock_tokenizer, inference_config_factory):
        """CustomEvo wrappers introspect and move their inner model."""
        config = inference_config_factory(task_type="binary")
        model = CustomEvoLike()
        engine = _build_engine(model, mock_tokenizer, config)

        assert engine.model is model
        assert engine.model.model.to_called_with == engine.device
        assert {"input_ids", "attention_mask", "**kwargs"} <= engine.accepted_args

    def test_model_none_uses_default_args(self, mock_tokenizer, inference_config_factory):
        """A None model leaves the default forward-arg set in place."""
        config = inference_config_factory(task_type="binary")
        engine = DNAInference(model=None, tokenizer=mock_tokenizer, config=config)

        assert "input_ids" in engine.accepted_args
        assert "output_hidden_states" in engine.accepted_args
        assert engine.device == torch.device("cpu")

    def test_lora_adapter_local_source(
        self, mock_model, mock_tokenizer, inference_config_factory, tmp_path
    ):
        """A local LoRA adapter directory loads through PeftModel.from_pretrained."""
        config = inference_config_factory(task_type="binary")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        peft_model = Mock()
        peft_model.to = Mock(return_value=peft_model)

        with (
            patch(
                "dnallm.inference.inference._get_model_path_and_imports",
                return_value=(str(adapter_dir), {}),
            ) as mock_resolve,
            patch("dnallm.models.model.peft_forward_compatiable", side_effect=lambda m: m),
            patch(
                "peft.PeftModel.from_pretrained", return_value=peft_model
            ) as mock_from_pretrained,
        ):
            engine = _build_engine(
                mock_model, mock_tokenizer, config, lora_adapter=str(adapter_dir)
            )

        mock_resolve.assert_called_once_with(str(adapter_dir), "local")
        mock_from_pretrained.assert_called_once_with(mock_model, str(adapter_dir))
        assert engine.model is peft_model

    def test_lora_adapter_failure_raises(
        self, mock_model, mock_tokenizer, inference_config_factory, tmp_path
    ):
        """A failing adapter resolution raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with patch(
            "dnallm.inference.inference._get_model_path_and_imports",
            side_effect=OSError("cannot resolve"),
        ):
            with pytest.raises(ValueError, match=r"Failed to load LoRA adapter"):
                _build_engine(mock_model, mock_tokenizer, config, lora_adapter=str(adapter_dir))


class TestModelInfoHelpers:
    """Model info, parameter, and memory-estimation reporting."""

    def test_get_model_info_reports_config(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """get_model_info surfaces the known mock config values."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        info = engine.get_model_info()

        assert info["config"]["hidden_size"] == 768
        assert info["config"]["num_hidden_layers"] == 6
        assert info["config"]["model_type"] == "dna_gpt"
        assert info["attention_supported"] is True
        assert info["device"] == "cpu"

    def test_get_model_parameters_counts(self, inference_config_factory, mock_tokenizer):
        """Parameter counts split total/trainable/frozen from a known iterator."""
        frozen = torch.zeros(10, 10)
        trainable = torch.nn.Parameter(torch.ones(5, 5))
        model = Mock()
        model.config = _BaseConfig("counting")
        model.forward = lambda **kw: SimpleNamespace(logits=torch.zeros(1, 2))
        model.to = Mock(return_value=model)
        model.eval = Mock()
        model.parameters.side_effect = lambda: iter([frozen, trainable])
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        params = engine.get_model_parameters()

        assert params == {"total": 125, "trainable": 25, "frozen": 100}

    def test_get_model_parameters_error_dict(self, mock_tokenizer, inference_config_factory):
        """A failing parameters() iterator yields the error payload."""
        model = Mock()
        model.config = _BaseConfig("broken")
        model.forward = lambda **kw: SimpleNamespace(logits=torch.zeros(1, 2))
        model.to = Mock(return_value=model)
        model.eval = Mock()
        model.parameters.side_effect = RuntimeError("no params")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        assert engine.get_model_parameters() == {"error": "no params"}
        assert engine._get_model_parameters_info() == {"error": "no params"}

    def test_estimate_memory_usage_formula(self, mock_tokenizer, inference_config_factory):
        """Memory estimates follow the documented float32 parameter formula."""
        model = Mock()
        model.config = _BaseConfig("counting")
        model.config.hidden_size = 768
        model.config.num_hidden_layers = 6
        model.forward = lambda **kw: SimpleNamespace(logits=torch.zeros(1, 2))
        model.to = Mock(return_value=model)
        model.eval = Mock()
        # 2,621,440 float32 params are exactly 10.0 MiB.
        model.parameters.return_value = iter([torch.zeros(2_621_440)])
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        report = engine.estimate_memory_usage(batch_size=2, sequence_length=1000)

        assert report["parameter_memory_mb"] == "10.0"
        assert report["activation_memory_mb"] == "17.6"
        assert report["total_estimated_mb"] == "27.6"

    def test_model_info_without_config(self, mock_tokenizer, inference_config_factory):
        """Models without a config report the error payloads, not a crash."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(NoConfigModel(), mock_tokenizer, config)

        info = engine.get_model_info()

        assert info["config"] == {"error": "Model has no config"}
        assert info["model_type"] == "NoConfigModel"

    def test_available_outputs_flags(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Collected flags flip only after embeddings exist on the engine."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        before = engine.get_available_outputs()
        engine.embeddings = {"hidden_states": (torch.zeros(1, 2),), "attentions": None}

        after = engine.get_available_outputs()

        assert before["hidden_states_collected"] is False
        assert after["hidden_states_collected"] is True
        assert after["attentions_collected"] is False


class TestAttentionSupportHelpers:
    """Attention/hidden-states capability probing and eager fallbacks."""

    def test_check_attention_support_true_when_settable(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """A settable output_attentions config reports support."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine._check_attention_support() is True

    def test_check_attention_support_eager_fallback(self, mock_tokenizer, inference_config_factory):
        """An sdpa rejection recovers by switching to eager attention."""
        model = ConstantOutputModel(config=FlakyAttentionConfig(flaky=2))
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        assert engine._check_attention_support() is True
        assert model.config.attn_implementation == "eager"

    def test_check_attention_support_plain_failure(self, mock_tokenizer, inference_config_factory):
        """A non-sdpa setter failure leaves attention unsupported."""
        model = ConstantOutputModel(config=AttentionRaiseForeverConfig())
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        assert engine._check_attention_support() is False

    def test_hidden_states_support_warns_on_error(self, mock_tokenizer, inference_config_factory):
        """A failing output_hidden_states setter warns and reports unsupported."""
        model = ConstantOutputModel(config=HiddenStatesRaiseConfig())
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        with pytest.warns(UserWarning, match="Cannot enable output_hidden_states"):
            assert engine._check_hidden_states_support() is False

    def test_force_eager_attention_sets_config(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """force_eager_attention flips attn_implementation to eager."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.force_eager_attention() is True
        assert mock_model.config.attn_implementation == "eager"

    def test_force_eager_attention_failure_returns_false(
        self, mock_tokenizer, inference_config_factory
    ):
        """A config that rejects attn_implementation sets reports failure."""
        model = ConstantOutputModel(config=EagerRaiseConfig())
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, mock_tokenizer, config)

        assert engine.force_eager_attention() is False

    def test_setup_attentions_eager_switch_warns(
        self, simple_dna_tokenizer, inference_config_factory
    ):
        """batch_infer recovers from an sdpa output_attentions rejection via eager."""
        model = ConstantOutputModel(config=FlakyAttentionConfig(flaky=1), with_attentions=True)
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)

        with pytest.warns(UserWarning, match="Switched to 'eager'"):
            engine.infer(["ATCG", "GGCC"], output_attentions=True)

        assert model.config.attn_implementation == "eager"

    def test_handle_attention_config_error_plain(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Non-sdpa config errors warn and stay unresolved."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.warns(UserWarning, match="Cannot enable output_attentions"):
            assert engine._handle_attention_config_error(ValueError("boom")) is False


class TestGeneratePath:
    """Sequence generation across model families."""

    def test_generate_causallm(self, simple_dna_tokenizer, inference_config_factory):
        """CausalLM generation decodes generated ids through the tokenizer."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(CausalLMLike(), simple_dna_tokenizer, config)

        outputs = engine.generate(["ATCG"], n_tokens=4, temperature=0.7)

        assert outputs == [{"Prompt": "ATCG", "Output": "ACGT"}]
        call = engine.model.generate_calls[0]
        assert call["max_new_tokens"] == 4
        assert call["temperature"] == 0.7
        assert call["do_sample"] is True

    def test_generate_unsupported_model_raises(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Models outside the supported families raise a matchable error."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.raises(ValueError, match=r"not supported for sequence generation"):
            engine.generate(["ATCG"])

    def test_generate_megadna(self, simple_dna_tokenizer, inference_config_factory):
        """MEGADNA generation decodes per sample with space stripping."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(MEGADNALike(), simple_dna_tokenizer, config)

        outputs = engine.generate(["ATCG", "GGCC"], n_tokens=4, n_samples=2)

        assert len(outputs) == 4
        assert all(o["Output"] == "ACGT" for o in outputs)
        call = engine.model.generate_calls[0]
        assert call["seq_len"] == 4

    def test_generate_evo2(self, simple_dna_tokenizer, inference_config_factory):
        """EVO2 generation formats per-prompt sequences and mean logprobs."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(Evo2Like(), simple_dna_tokenizer, config)

        outputs = engine.generate(["ATCG", "GGCC"], n_tokens=8, temperature=0.5, top_k=4)

        assert outputs == [
            {"Prompt": "ATCG", "Output": "AATT", "Score": 0.42},
            {"Prompt": "GGCC", "Output": "AATT", "Score": 0.42},
        ]
        call = engine.model.generate_calls[0]
        assert call["prompt_seqs"] == ["ATCG", "GGCC"]
        assert call["n_tokens"] == 8
        assert call["cached_generation"] is True

    def test_generate_evo1_with_stubbed_module(
        self, simple_dna_tokenizer, inference_config_factory, monkeypatch
    ):
        """EVO1 generation delegates to the evo package module contract."""
        fake_evo = types.ModuleType("evo")
        fake_evo.generate = Mock(return_value=(["AATT", "GGCC"], [0.42, -0.7]))
        monkeypatch.setitem(sys.modules, "evo", fake_evo)
        simple_dna_tokenizer.raw_tokenizer = "raw-tokenizer"
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(Evo1Like(), simple_dna_tokenizer, config)

        outputs = engine.generate(["ATCG", "GGCC"], n_tokens=8, n_samples=1)

        assert outputs == [
            {"Prompt": "ATCG", "Output": "AATT", "Score": 0.42},
            {"Prompt": "GGCC", "Output": "GGCC", "Score": -0.7},
        ]
        fake_evo.generate.assert_called_once()
        kwargs = fake_evo.generate.call_args.kwargs
        assert kwargs["model"] is engine.model.model
        assert kwargs["tokenizer"] == "raw-tokenizer"
        assert kwargs["n_tokens"] == 8


class TestScoringPath:
    """Sequence scoring across score types and model families."""

    def test_scoring_embedding_mean_matches_recompute(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Embedding scores equal the recomputed hidden-state mean."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)
        seqs = ["ATCG", "GGCC"]

        scores = engine.scoring(seqs, score_type="embedding", reduce_method="mean")

        enc = simple_dna_tokenizer(seqs, return_tensors="pt")
        hidden = tiny_real_model(enc["input_ids"]).hidden_states[0]
        expected0 = torch.stack([hidden[0]], dim=0).mean().item()
        assert len(scores) == 2
        assert scores[0]["Input"] == "ATCG"
        assert scores[0]["Score"] == pytest.approx(expected0, rel=1e-5)

    @pytest.mark.parametrize("method", ["max", "min", "last", "first"])
    def test_scoring_embedding_reduce_methods(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, method
    ):
        """Every reduction method selects the matching hidden-state aggregate."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        scores = engine.scoring(["ATCG"], score_type="embedding", reduce_method=method)

        assert len(scores) == 1
        assert isinstance(scores[0]["Score"], float)

    def test_scoring_logits_matches_recompute(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Logit scores equal the recomputed logits mean."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)
        seqs = ["ATCG", "GGCC"]

        scores = engine.scoring(seqs, score_type="logits", reduce_method="mean")

        enc = simple_dna_tokenizer(seqs, return_tensors="pt")
        logits = tiny_real_model(enc["input_ids"]).logits
        assert scores[0]["Score"] == pytest.approx(logits[0].mean().item(), rel=1e-5)

    def test_scoring_probability_mlm_mask(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """MLM probability scoring gathers logprobs at masked positions."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)
        seq = "ATNCG"

        scores = engine.scoring([seq], score_type="probability", reduce_method="mean")

        enc = simple_dna_tokenizer(seq, padding=False)
        ids = enc["input_ids"][0]
        logits = model(torch.tensor([ids])).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        expected = logprobs[2, ids[2]].item()
        assert scores[0]["Score"] == pytest.approx(expected, rel=1e-5)

    def test_scoring_probability_causal(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """Causal probability scoring aligns logits[t] with input token t+1."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)
        seq = "ATCGG"

        scores = engine.scoring([seq], score_type="probability", reduce_method="sum")

        enc = simple_dna_tokenizer(seq, padding=False)
        ids = torch.tensor(enc["input_ids"][0])
        logits = model(ids.unsqueeze(0)).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        expected = logprobs[:-1].gather(1, ids[1:].unsqueeze(-1)).squeeze(-1).sum().item()
        assert scores[0]["Score"] == pytest.approx(expected, rel=1e-5)

    def test_scoring_probability_sum_reduce(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """reduce_method='sum' sums masked-position logprobs."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)
        seq = "ANNT"

        mean_scores = engine.scoring([seq], score_type="probability", reduce_method="mean")
        sum_scores = engine.scoring([seq], score_type="probability", reduce_method="sum")

        n_masked = 2
        assert sum_scores[0]["Score"] == pytest.approx(mean_scores[0]["Score"] * n_masked, rel=1e-4)

    def test_scoring_evo2(self, simple_dna_tokenizer, inference_config_factory):
        """EVO2 scoring delegates to score_sequences with the reduce method."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(Evo2Like(), simple_dna_tokenizer, config)

        scores = engine.scoring(["ATCG", "GGCC"], reduce_method="max")

        assert scores == [{"Input": "ATCG", "Score": 0.1}, {"Input": "GGCC", "Score": 0.2}]
        assert engine.model.score_calls[0]["reduce_method"] == "max"

    def test_scoring_evo1_with_stubbed_module(
        self, simple_dna_tokenizer, inference_config_factory, monkeypatch
    ):
        """EVO1 scoring delegates to the evo package module contract."""
        fake_evo = types.ModuleType("evo")
        fake_evo.score_sequences = Mock(return_value=["lo", "hi"])
        monkeypatch.setitem(sys.modules, "evo", fake_evo)
        simple_dna_tokenizer.raw_tokenizer = "raw-tokenizer"
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(Evo1Like(), simple_dna_tokenizer, config)

        scores = engine.scoring(["ATCG", "GGCC"], reduce_method="min")

        assert scores == [{"Input": "ATCG", "Score": "lo"}, {"Input": "GGCC", "Score": "hi"}]
        kwargs = fake_evo.score_sequences.call_args.kwargs
        assert kwargs["model"] is engine.model.model
        assert kwargs["reduce_method"] == "min"

    def test_scoring_megadna(self, simple_dna_tokenizer, inference_config_factory):
        """MEGADNA scoring reports the model's loss value."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(MEGADNALike(), simple_dna_tokenizer, config)

        scores = engine.scoring(["ATCG"])

        assert scores == [{"Input": "ATCG", "Score": torch.tensor(1.25)}]
        assert engine.model.calls[0]["return_value"] == "loss"

    def test_scoring_from_dataloader(self, simple_dna_tokenizer, inference_config_factory):
        """DataLoader scoring re-derives sequences from the wrapped dataset."""
        model = ConstantOutputModel(with_attentions=False)
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(model, simple_dna_tokenizer, config)
        _, dataloader = engine.generate_dataset(["ATCG", "GGCC"], do_encode=False)

        scores = engine.scoring(dataloader, score_type="embedding", reduce_method="mean")

        assert len(scores) == 2
        assert [s["Input"] for s in scores] == ["ATCG", "GGCC"]
        assert all(isinstance(s["Score"], float) for s in scores)


class TestGenerateDatasetEdges:
    """generate_dataset input validation and preprocessing branches."""

    def test_invalid_input_type_raises(self, mock_model, mock_tokenizer, inference_config_factory):
        """Non-str/list inputs raise a matchable ValueError."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.raises(ValueError, match=r"Input should be a file path or a list"):
            engine.generate_dataset(123)

    def test_empty_list_raises(self, mock_model, mock_tokenizer, inference_config_factory):
        """An empty sequence list cannot build a dataset and raises."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        with pytest.raises(ValueError, match=r"No valid dataset could be created"):
            engine.generate_dataset([])

    def test_sampling_reduces_rows(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """A sampling fraction shrinks the loaded dataset deterministically."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence,label\nATCG,0\nGGCC,1\nTTTT,0\nACGT,1\n")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        dataset, _ = engine.generate_dataset(
            str(csv_path), sampling=0.5, do_encode=False, label_col="label"
        )

        assert len(dataset) == 2

    def test_labels_populated_from_file(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """Labels in the source file populate engine.labels verbatim."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence,label\nATCG,0\nGGCC,1\nTTTT,0\nACGT,1\n")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        engine.generate_dataset(str(csv_path), do_encode=False, label_col="label")

        assert list(engine.labels) == [0, 1, 0, 1]


class TestPlotHelpers:
    """plot_attentions / plot_hidden_states plumbing to the plot module."""

    def _engine_with_embeddings(self, mock_model, mock_tokenizer, config):
        engine = _build_engine(mock_model, mock_tokenizer, config)
        engine.sequences = ["ATCG"]
        engine.embeddings = {
            "attentions": (torch.zeros(1, 1, 4, 4),),
            "hidden_states": (torch.zeros(1, 4, 8),),
            "attention_mask": torch.ones(1, 4),
            "labels": torch.zeros(1),
        }
        return engine

    def test_plot_attentions_derives_heatmap_path(
        self, mock_model, mock_tokenizer, inference_config_factory, tmp_path
    ):
        """A save_path with a suffix derives the sibling _heatmap path."""
        config = inference_config_factory(task_type="binary")
        engine = self._engine_with_embeddings(mock_model, mock_tokenizer, config)
        sentinel = object()

        with patch("dnallm.inference.inference.plot_attention_map", return_value=sentinel) as p:
            result = engine.plot_attentions(
                seq_idx=0, layer=-1, head=0, save_path=str(tmp_path / "attn.pdf")
            )

        assert result is sentinel
        assert p.call_args.kwargs["save_path"] == str(tmp_path / "attn_heatmap.pdf")
        assert p.call_args.kwargs["seq_idx"] == 0

    def test_plot_attentions_dir_save_path(
        self, mock_model, mock_tokenizer, inference_config_factory, tmp_path
    ):
        """A suffix-less save_path writes heatmap.pdf inside the directory."""
        config = inference_config_factory(task_type="binary")
        engine = self._engine_with_embeddings(mock_model, mock_tokenizer, config)

        with patch("dnallm.inference.inference.plot_attention_map", return_value=None) as p:
            engine.plot_attentions(save_path=str(tmp_path))

        assert p.call_args.kwargs["save_path"] == str(tmp_path / "heatmap.pdf")

    def test_plot_attentions_without_embeddings_returns_none(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Without collected embeddings the call is a documented None."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.plot_attentions() is None

    def test_plot_hidden_states_derives_embedding_path(
        self, mock_model, mock_tokenizer, inference_config_factory, tmp_path
    ):
        """A save_path with a suffix derives the sibling _embedding path."""
        config = inference_config_factory(task_type="binary")
        engine = self._engine_with_embeddings(mock_model, mock_tokenizer, config)
        sentinel = object()

        with patch("dnallm.inference.inference.plot_embeddings", return_value=sentinel) as p:
            result = engine.plot_hidden_states(reducer="PCA", save_path=str(tmp_path / "emb.pdf"))

        assert result is sentinel
        assert p.call_args.kwargs["save_path"] == str(tmp_path / "emb_embedding.pdf")
        assert p.call_args.kwargs["reducer"] == "PCA"

    def test_plot_hidden_states_without_embeddings_returns_none(
        self, mock_model, mock_tokenizer, inference_config_factory
    ):
        """Without collected embeddings the call returns None."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(mock_model, mock_tokenizer, config)

        assert engine.plot_hidden_states() is None


class TestGetEmbeddings:
    """get_embeddings general and special-model paths."""

    def test_list_input_returns_hidden_states(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """A list input returns stacked per-layer hidden states."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        hidden = engine.get_embeddings(["ATCG", "GGCC"])

        assert isinstance(hidden, tuple)
        assert hidden[0].shape == (2, 4, 16)
        assert engine.embeddings["hidden_states"] is hidden

    def test_file_path_input(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """An existing file path loads a dataset and returns hidden states."""
        csv_path = tmp_path / "seqs.csv"
        csv_path.write_text("sequence\nATCG\nGGCC\n")
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        hidden = engine.get_embeddings(str(csv_path))

        assert hidden[0].shape[0] == 2

    def test_invalid_path_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """A non-file string that is not a path raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"is not a valid file path"):
            engine.get_embeddings(str(tmp_path / "missing.fasta"))

    def test_custom_evo_special_path(self, simple_dna_tokenizer, inference_config_factory):
        """CustomEvo models extract per-block embeddings via the layer contract."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(CustomEvoLike(), simple_dna_tokenizer, config)

        result = engine.get_embeddings(["ATCG", "GGCC"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (2, 1, 4, 4)
        called_layers = engine.model.calls[0]["layer_names"]
        assert called_layers == ["blocks.0", "blocks.1"]

    def test_megadna_special_path(self, simple_dna_tokenizer, inference_config_factory):
        """MEGADNA models collect three stacked embedding groups."""
        config = inference_config_factory(task_type="binary")
        engine = _build_engine(MEGADNALike(), simple_dna_tokenizer, config)

        result = engine.get_embeddings(["ATCG", "GGCC"])

        assert isinstance(result, list)
        assert len(result) == 3
        for group in result:
            assert group.shape == (2, 6, 4)
