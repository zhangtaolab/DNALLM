"""
Mock injections for documentation verification.

This module is exec'd by verify_docs.py into each verification subprocess
to provide lightweight mocks for heavy operations (model loading, training,
benchmarking, etc.) so that code examples can be syntax-checked and run
without downloading multi-GB models or requiring GPUs.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# --- torch / numpy / pandas stubs ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import numpy as np
except Exception:
    np = None

# --- model / tokenizer mocks ---
class _MockModelOutput:
    def __init__(self, batch_size=2, num_labels=2):
        import torch
        self.logits = torch.ones(batch_size, num_labels) * 0.5
        self.last_hidden_state = torch.ones(batch_size, 10, 128)
        self.hidden_states = (torch.ones(batch_size, 10, 128),)

class _MockModel:
    def parameters(self):
        return iter([])

    def to(self, *a, **k):
        return self

    def eval(self):
        pass

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if hasattr(input_ids, "shape"):
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1
        return _MockModelOutput(batch_size=batch_size)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

_mock_model = _MockModel()

_mock_tokenizer = MagicMock()

def _mock_tok_fn(sequences, **kwargs):
    if isinstance(sequences, list):
        n = len(sequences)
    else:
        n = 1
    return {
        "input_ids": [[1, 2, 3, 4, 5]] * n,
        "attention_mask": [[1, 1, 1, 1, 1]] * n,
    }

_mock_tokenizer.side_effect = _mock_tok_fn
_mock_tokenizer.pad.side_effect = _mock_tok_fn
_mock_tokenizer.pad_token = "[PAD]"
_mock_tokenizer.eos_token = "[EOS]"
_mock_tokenizer.sep_token = "[SEP]"
_mock_tokenizer.pad_token_id = 0
_mock_tokenizer.eos_token_id = 1
_mock_tokenizer.special_tokens_map = {"pad_token": "[PAD]", "eos_token": "[EOS]"}

# --- config mock ---
class MockNamespace(SimpleNamespace):
    """SimpleNamespace that also supports dict-style access."""
    def __getitem__(self, key):
        return getattr(self, key)

class _MockConfig(dict):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.update({
            "task": MockNamespace(
                task_type="binary",
                num_labels=2,
                label_names=["negative", "positive"],
                threshold=0.5,
                evaluation=MockNamespace(metrics=["accuracy", "f1"]),
            ),
            "inference": MockNamespace(
                batch_size=8,
                device="cpu",
                use_fp16=False,
                use_bf16=False,
                max_length=512,
                truncation=True,
                num_workers=0,
                pin_memory=False,
                evaluation=MockNamespace(metrics=["accuracy", "f1"]),
            ),
            "finetune": MockNamespace(
                learning_rate=2e-5,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                logging_steps=10,
                bf16=False,
                eval_strategy="no",
                save_strategy="epoch",
                seed=42,
                evaluation=MockNamespace(metrics=["accuracy", "f1"]),
            ),
            "benchmark": MockNamespace(
                models=[
                    MockNamespace(
                        name="test", path="test", source="huggingface", task_type="classification"
                    )
                ],
                config_path=config_path if isinstance(config_path, str) else "benchmark_config.yaml",
                evaluation=MockNamespace(metrics=["accuracy", "f1"]),
            ),
            "config_path": config_path if isinstance(config_path, str) else "config.yaml",
        })

    def model_dump(self):
        return {k: v for k, v in self.items() if not k.startswith("_")}

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        return MockNamespace()

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return MockNamespace()


def _mock_load_config(path, *a, **k):
    return _MockConfig(config_path=path)


# --- DNADataset mock factory ---
def _make_mock_dataset():
    """Return a mock DNADataset-like object with common methods."""
    m = MagicMock()
    m.train_data = MagicMock()
    m.val_data = MagicMock()
    m.test_data = MagicMock()
    m.is_split = True
    m.split_data = lambda *a, **k: None
    m.encode_sequences = lambda *a, **k: None
    m.validate_sequences = lambda *a, **k: None
    m.process_missing_data = lambda *a, **k: None
    m.shuffle = lambda *a, **k: None
    m.sampling = lambda *a, **k: m
    m.head = lambda *a, **k: m
    m.show = lambda *a, **k: None
    m.statistics = lambda *a, **k: None
    m.plot_statistics = lambda *a, **k: None
    m.raw_reverse_complement = lambda *a, **k: None
    m.augment_reverse_complement = lambda *a, **k: None
    m.random_generate = lambda *a, **k: None
    m.label_counts = {0: 50, 1: 50}
    m.stats = {"total": 100, "mean_length": 500}
    m.dataset = {"train": [1, 2], "validation": [3, 4]}
    m.__len__ = lambda self: 100
    return m


# --- inference mocks ---
class _MockDNAInference:
    def __init__(self, *a, **k):
        pass

    def infer(self, sequence, *a, **k):
        return {"prediction": "positive", "score": 0.85}

    def batch_infer(self, sequences, *a, **k):
        return [{"prediction": "positive", "score": 0.85} for _ in sequences]


class _MockBenchmark:
    def __init__(self, *a, **k):
        pass

    def run(self, *a, **k):
        return {
            "dataset1": {
                "model1": {
                    "accuracy": 0.95,
                    "f1": 0.94,
                }
            }
        }

    def plot(self, results, *a, **k):
        return (MagicMock(), MagicMock())

    def run_without_config(self, *a, **k):
        return {
            "dataset1": {
                "model1": {
                    "accuracy": 0.95,
                    "f1": 0.94,
                }
            }
        }


class _MockMutagenesis:
    def __init__(self, *a, **k):
        pass

    def mutate_sequence(self, sequence, *a, **k):
        return MagicMock()

    def evaluate(self, strategy="mean", *a, **k):
        if np is not None:
            return np.zeros((len("ATGC"), 100))
        return [[0.0] * 100 for _ in range(4)]

    def plot(self, predictions, *a, **k):
        return MagicMock()

    def get_important_positions(self, *a, **k):
        return [1, 2, 3]


class _MockDNAInterpret:
    def __init__(self, *a, **k):
        pass

    def interpret(self, sequence, *a, **k):
        return {"attention": MagicMock()}

    def batch_interpret(self, sequences, *a, **k):
        return [{"attention": MagicMock()} for _ in sequences]

    def plot_attributions(self, *a, **k):
        return MagicMock()

    def get_attention(self, sequence, *a, **k):
        return [[0.1, 0.2, 0.3, 0.4] for _ in range(len(sequence))]

    def get_embedding(self, sequence, *a, **k):
        return [0.1] * 128

    def generate_report(self, sequence, *a, **k):
        return {"summary": "mock report"}


# --- trainer mock ---
class _MockDNATrainer:
    def __init__(self, *a, **k):
        pass

    def train(self, *a, **k):
        return {
            "eval_hamming_loss": 0.05,
            "eval_accuracy": 0.95,
            "eval_f1": 0.94,
            "eval_precision": 0.95,
            "eval_recall": 0.94,
            "eval_loss": 0.1,
        }

    def evaluate(self, *a, **k):
        return {
            "accuracy": 0.95,
            "eval_accuracy": 0.95,
            "eval_f1": 0.94,
            "eval_f1_macro": 0.94,
            "eval_f1_micro": 0.93,
            "eval_precision": 0.95,
            "eval_recall": 0.94,
            "eval_rmse": 0.1,
            "eval_mae": 0.08,
            "eval_r2": 0.92,
            "eval_loss": 0.1,
        }

    def save_model(self, *a, **k):
        pass

    def save_evaluation_report(self, *a, **k):
        pass

    def get_trainable_parameters_ratio(self, *a, **k):
        return 0.85

    def save_lora_adapter(self, *a, **k):
        pass

    def save_pretrained(self, *a, **k):
        pass


# --- patch everything ---
try:
    import dnallm

    dnallm.load_model_and_tokenizer = lambda *a, **k: (_mock_model, _mock_tokenizer)
    dnallm.load_config = _mock_load_config

    # DNADataset
    try:
        from dnallm.datahandling import DNADataset

        DNADataset.from_huggingface = classmethod(lambda cls, *a, **k: _make_mock_dataset())
        DNADataset.from_modelscope = classmethod(lambda cls, *a, **k: _make_mock_dataset())
        DNADataset.load_local_data = classmethod(lambda cls, *a, **k: _make_mock_dataset())
    except Exception:
        pass

    # preset datasets
    try:
        import dnallm.datahandling.data as _dna_data

        _orig_show = getattr(_dna_data, "show_preset_dataset", None)
        if _orig_show is not None:
            _dna_data.show_preset_dataset = lambda *a, **k: print("Available presets: plant-genomic-benchmark")

        _orig_load_preset = getattr(_dna_data, "load_preset_dataset", None)
        if _orig_load_preset is not None:
            _dna_data.load_preset_dataset = lambda *a, **k: _make_mock_dataset()
    except Exception:
        pass

    # inference
    try:
        from dnallm.inference import DNAInference

        DNAInference.__init__ = _MockDNAInference.__init__
        DNAInference.infer = _MockDNAInference.infer
        DNAInference.batch_infer = _MockDNAInference.batch_infer
    except Exception:
        pass

    try:
        from dnallm.inference import Benchmark

        Benchmark.__init__ = _MockBenchmark.__init__
        Benchmark.run = _MockBenchmark.run
        Benchmark.plot = _MockBenchmark.plot
    except Exception:
        pass

    try:
        from dnallm.inference import Mutagenesis

        Mutagenesis.__init__ = _MockMutagenesis.__init__
        Mutagenesis.mutate_sequence = _MockMutagenesis.mutate_sequence
        Mutagenesis.evaluate = _MockMutagenesis.evaluate
        Mutagenesis.plot = _MockMutagenesis.plot
        Mutagenesis.get_important_positions = _MockMutagenesis.get_important_positions
    except Exception:
        pass

    try:
        from dnallm.inference import DNAInterpret

        DNAInterpret.__init__ = _MockDNAInterpret.__init__
        DNAInterpret.interpret = _MockDNAInterpret.interpret
        DNAInterpret.batch_interpret = _MockDNAInterpret.batch_interpret
        DNAInterpret.plot_attributions = _MockDNAInterpret.plot_attributions
    except Exception:
        pass
    try:
        import dnallm.inference
        if not hasattr(dnallm.inference, 'DNAInterpreter'):
            dnallm.inference.DNAInterpreter = DNAInterpret
    except Exception:
        pass

    # trainer
    try:
        from dnallm.finetune import DNATrainer

        DNATrainer.__init__ = _MockDNATrainer.__init__
        DNATrainer.train = _MockDNATrainer.train
        DNATrainer.evaluate = _MockDNATrainer.evaluate
        DNATrainer.save_model = _MockDNATrainer.save_model
        DNATrainer.save_pretrained = _MockDNATrainer.save_pretrained
        DNATrainer.save_evaluation_report = _MockDNATrainer.save_evaluation_report
        DNATrainer.get_trainable_parameters_ratio = _MockDNATrainer.get_trainable_parameters_ratio
        DNATrainer.save_lora_adapter = _MockDNATrainer.save_lora_adapter
    except Exception:
        pass

except Exception:
    pass

# --- transformers mocks ---
try:
    import transformers

    _orig_AutoModel = getattr(transformers, "AutoModel", None)
    if _orig_AutoModel is not None:
        _orig_AutoModel.from_pretrained = classmethod(lambda cls, *a, **k: _mock_model)

    _orig_AutoModelSC = getattr(transformers, "AutoModelForSequenceClassification", None)
    if _orig_AutoModelSC is not None:
        _orig_AutoModelSC.from_pretrained = classmethod(lambda cls, *a, **k: _mock_model)

    _orig_AutoTok = getattr(transformers, "AutoTokenizer", None)
    if _orig_AutoTok is not None:
        _orig_AutoTok.from_pretrained = classmethod(lambda cls, *a, **k: _mock_tokenizer)
except Exception:
    pass
