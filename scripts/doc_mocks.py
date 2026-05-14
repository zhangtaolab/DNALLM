"""
Mock injections for documentation verification.

This module is exec'd by verify_docs.py into each verification subprocess
to provide lightweight mocks for heavy operations (model loading, training,
benchmarking, etc.) so that code examples can be syntax-checked and run
without downloading multi-GB models or requiring GPUs.
"""

import os
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
    def __init__(self, batch_size=2, seq_len=10, num_labels=2):
        import torch
        self.logits = torch.ones(batch_size, seq_len, num_labels) * 0.5
        self.last_hidden_state = torch.ones(batch_size, seq_len, 128)
        self.hidden_states = (torch.ones(batch_size, seq_len, 128),)
        self.loss = torch.tensor(0.5, requires_grad=True)
        self.attentions = None

class _MockModel:
    device = "cpu"

    def parameters(self):
        import torch
        return iter([torch.nn.Parameter(torch.ones(1))])

    def to(self, *a, **k):
        return self

    def eval(self):
        pass

    def train(self, mode=True):
        pass

    def __getattr__(self, name):
        if name == "device":
            return "cpu"
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        import torch
        if hasattr(input_ids, "shape"):
            batch_size = input_ids.shape[0]
            if len(input_ids.shape) >= 2:
                seq_len = input_ids.shape[1]
            else:
                seq_len = input_ids.shape[0]
        else:
            batch_size = 1
            seq_len = 10
        return _MockModelOutput(batch_size=batch_size, seq_len=seq_len)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def generate(self, input_ids, max_length=512, num_beams=1, early_stopping=False, *a, **k):
        import torch
        if hasattr(input_ids, 'shape'):
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1
        return torch.ones(batch_size, min(max_length, 10), dtype=torch.long)

    def resize_token_embeddings(self, new_num_tokens, *a, **k):
        return None

    def merge_and_unload(self, *a, **k):
        return self

_mock_model = _MockModel()

class _MockTensorDict(dict):
    """Dict that supports .to() like PyTorch tensors."""
    def to(self, *a, **k):
        return self

_mock_tokenizer = MagicMock()

def _mock_tok_fn(sequences, **kwargs):
    import torch
    if isinstance(sequences, list):
        n = len(sequences)
    else:
        n = 1
    return _MockTensorDict({
        "input_ids": torch.ones(n, 5, dtype=torch.long),
        "attention_mask": torch.ones(n, 5, dtype=torch.long),
    })

_mock_tokenizer.side_effect = _mock_tok_fn
_mock_tokenizer.pad.side_effect = _mock_tok_fn
_mock_tokenizer.__call__ = _mock_tok_fn
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
class _MockDataset:
    """Mock DNADataset that works with sklearn and common operations."""
    def __init__(self):
        self._items = [{"sequence": "ATGCATGCATGC", "label": 0} for _ in range(10)]
        self.train_data = self
        self.val_data = self
        self.test_data = self
        self.test = self
        self.is_split = True
        self.label_counts = {0: 50, 1: 50}
        self.stats = {"total": 100, "mean_length": 500}
        self.dataset = {"train": [1, 2], "validation": [3, 4]}

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        import torch
        return {
            "sequence": "ATGC",
            "label": 1,
            "input_ids": torch.ones(5, dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
        }

    def __iter__(self):
        import torch
        return iter([{
            "sequence": "ATGC",
            "label": 1,
            "input_ids": torch.ones(5, dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
        } for _ in range(100)])

    def split_data(self, *a, **k):
        return None

    def encode_sequences(self, *a, **k):
        return None

    def validate_sequences(self, *a, **k):
        return None

    def process_missing_data(self, *a, **k):
        return None

    def shuffle(self, *a, **k):
        return None

    def sampling(self, *a, **k):
        return self

    def head(self, *a, **k):
        return self

    def show(self, *a, **k):
        return None

    def statistics(self, *a, **k):
        return None

    def plot_statistics(self, *a, **k):
        return None

    def raw_reverse_complement(self, *a, **k):
        return None

    def augment_reverse_complement(self, *a, **k):
        return None

    def random_generate(self, *a, **k):
        return None

    def augment_random_mutation(self, *a, **k):
        return self

    def augment_kmer(self, *a, **k):
        return self

    def select(self, idx):
        return self

    def get_dataloader(self, *a, **k):
        import torch
        return [{"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long), "labels": torch.tensor([1])}]


def _make_mock_dataset():
    """Return a mock DNADataset-like object with common methods."""
    return _MockDataset()


# --- inference mocks ---
class _MockDNAInference:
    def __init__(self, *a, **k):
        self.model = _mock_model
        self.tokenizer = _mock_tokenizer
        self.pred_config = MockNamespace(
            batch_size=8, device="cpu", max_length=512,
            truncation=True, num_workers=0, pin_memory=False,
        )

    def infer(self, sequence=None, sequences=None, file_path=None, evaluate=False, *a, **k):
        result = {
            "prediction": "positive",
            "score": 0.85,
            "probabilities": [0.15, 0.85],
            "predicted_label": "positive",
        }
        if file_path is not None:
            if evaluate:
                return ({"seq_0": result}, {"accuracy": 0.95})
            return {"seq_0": result}
        if sequences is not None:
            return [result for _ in sequences]
        if sequence is not None:
            return result
        return result

    def batch_infer(self, data, do_pred=True, output_hidden_states=False, output_attentions=False, *a, **k):
        import torch
        # If called with a list of sequences, return predictions directly
        if isinstance(data, list) and data and isinstance(data[0], str):
            return [{"prediction": "positive", "score": 0.85} for _ in data]
        n = 2
        logits = torch.ones(n, 2) * 0.5
        predictions = None
        if do_pred:
            predictions = [{"prediction": "positive", "score": 0.85} for _ in range(n)]
        embeddings = {}
        if output_hidden_states:
            embeddings["hidden_states"] = [torch.ones(1, 10, 128)]
        if output_attentions:
            embeddings["attentions"] = [torch.ones(1, 4, 10, 10)]
        return logits, predictions, embeddings

    def generate_dataset(self, sequences, batch_size=8, *a, **k):
        return sequences, [{"input_ids": [[1,2,3]]} for _ in sequences]

    def generate(self, prompts, *a, **k):
        return [
            {"Prompt": p, "Output": "ATGCATGCATGC", "Score": 0.95}
            for p in (prompts if isinstance(prompts, list) else [prompts])
        ]

    def scoring(self, sequences, *a, **k):
        return [
            {"Input": seq, "Score": 0.95}
            for seq in (sequences if isinstance(sequences, list) else [sequences])
        ]

    def estimate_memory_usage(self, batch_size=8, sequence_length=512, *a, **k):
        return {"memory_mb": 1024, "peak_mb": 2048}

    def plot_attentions(self, *a, **k):
        return MagicMock()

    def plot_hidden_states(self, *a, **k):
        return MagicMock()

    def force_eager_attention(self, *a, **k):
        pass


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

    def evaluate_single_model(self, *a, **k):
        return {
            "accuracy": 0.95,
            "f1_score": 0.94,
            "precision": 0.95,
            "recall": 0.94,
        }


class _MockMutagenesis:
    def __init__(self, *a, **k):
        self.sequences = {"name": ["wt", "mut1"], "sequence": ["ATGCATGC", "CGTACGTA"]}

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
        self.model = _mock_model
        self.data_split = {"train": 80, "test": 20}
        self.config = _MockConfig()
        self.trainer = self  # Allow trainer.trainer = ... pattern

    def __call__(self, *a, **k):
        return _MockModelOutput()

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

    def infer(self, data=None, *a, **k):
        import torch
        from types import SimpleNamespace
        n = len(data) if data is not None and hasattr(data, '__len__') else 10
        preds = torch.tensor([[0.1, 0.9]] * n)
        return SimpleNamespace(predictions=preds, label_ids=["1"] * n)

    def predict(self, data=None, *a, **k):
        import torch
        from types import SimpleNamespace
        n = len(data) if data is not None and hasattr(data, '__len__') else 10
        preds = torch.tensor([[0.1, 0.9]] * n)
        return SimpleNamespace(predictions=preds, label_ids=["1"] * n)

    def load_pretrained(self, *a, **k):
        pass

    def merge_and_unload(self, *a, **k):
        return _mock_model


# --- CustomMetric mock ---
class _MockCustomMetric:
    def __init__(self):
        self.name = "mock_metric"

    def compute(self, predictions, targets, sequences=None, **kwargs):
        return {self.name: 0.95}


# --- MCP client mock (must be defined before patch block) ---
class _MockMCPClient:
    """Mock DNALLMMCPClient that returns results without connecting."""
    def __init__(self, transport="stdio", url=None, command=None, args=None, env=None):
        self.transport = transport
        self.url = url
        self.command = command or "dnallm-mcp-server"
        self.args = args or []
        self.env = env or {}

    def dna_sequence_predict(self, sequence, model_name):
        return {"prediction": "positive", "score": 0.85, "model": model_name}

    def dna_batch_predict(self, sequences, model_name):
        return [{"prediction": "positive", "score": 0.85} for _ in sequences]

    def dna_multi_model_predict(self, sequence, model_names=None):
        return {"predictions": {"model1": {"prediction": "positive", "score": 0.85}}}

    def dna_stream_predict(self, sequence, model_name, stream_progress=True):
        return {"prediction": "positive", "score": 0.85, "streamed": True}

    def dna_stream_batch_predict(self, sequences, model_name, stream_progress=True):
        return {"results": [{"prediction": "positive", "score": 0.85} for _ in sequences]}

    def dna_stream_multi_model_predict(self, sequence, model_names=None, stream_progress=True):
        return {"predictions": {"model1": {"prediction": "positive", "score": 0.85}}}

    def dna_mutagenesis(self, sequence=None, sequences=None, mutation_type="single_base_substitution", positions=None, model_name=""):
        return {"mutations": [{"position": 1, "score": 0.5}]}

    def dna_interpret(self, sequence, model_name, method="lig", target_class=None, max_length=None):
        return {"attribution": [0.1] * len(sequence)}

    def list_loaded_models(self):
        return {"models": ["dnabert-2", "plant-dnabert"]}

    def get_model_info(self, model_name):
        return {"name": model_name, "task_type": "binary"}

    def list_models_by_task_type(self, task_type):
        return {"models": ["dnabert-2"]}

    def get_all_available_models(self):
        return {"models": ["dnabert-2", "plant-dnabert"]}

    def health_check(self):
        return {"status": "healthy"}

    def call(self, tool_name, arguments):
        return {"tool": tool_name, "arguments": arguments}


# --- torch lr_scheduler mock ---
try:
    import torch
    if not hasattr(torch.optim.lr_scheduler, 'get_scheduler'):
        def _mock_get_scheduler(name, optimizer, num_warmup_steps=0, num_training_steps=0, **kwargs):
            return torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_training_steps)
        torch.optim.lr_scheduler.get_scheduler = _mock_get_scheduler
except Exception:
    pass


# --- patch everything ---
try:
    import dnallm

    dnallm.load_model_and_tokenizer = lambda *a, **k: (_mock_model, _mock_tokenizer)
    dnallm.load_config = _mock_load_config
    # Also expose on inference subpackage for docs that import from there
    try:
        import dnallm.inference as _inf_mod
        if not hasattr(_inf_mod, "load_model_and_tokenizer"):
            _inf_mod.load_model_and_tokenizer = lambda *a, **k: (_mock_model, _mock_tokenizer)
    except Exception:
        pass

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

    # CustomMetric mock
    try:
        import dnallm.tasks.metrics as _metrics_mod
        if not hasattr(_metrics_mod, "CustomMetric"):
            _metrics_mod.CustomMetric = _MockCustomMetric
    except Exception:
        pass

    # inference
    try:
        from dnallm.inference import DNAInference

        DNAInference.__init__ = _MockDNAInference.__init__
        DNAInference.infer = _MockDNAInference.infer
        DNAInference.batch_infer = _MockDNAInference.batch_infer
        DNAInference.generate_dataset = _MockDNAInference.generate_dataset
        DNAInference.generate = _MockDNAInference.generate
        DNAInference.scoring = _MockDNAInference.scoring
        DNAInference.plot_attentions = _MockDNAInference.plot_attentions
        DNAInference.plot_hidden_states = _MockDNAInference.plot_hidden_states
        DNAInference.force_eager_attention = _MockDNAInference.force_eager_attention
    except Exception:
        pass

    try:
        from dnallm.inference import Benchmark

        Benchmark.__init__ = _MockBenchmark.__init__
        Benchmark.run = _MockBenchmark.run
        Benchmark.run_without_config = _MockBenchmark.run_without_config
        Benchmark.plot = _MockBenchmark.plot
        Benchmark.evaluate_single_model = _MockBenchmark.evaluate_single_model
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
        DNATrainer.infer = _MockDNATrainer.infer
        DNATrainer.predict = _MockDNATrainer.predict
    except Exception:
        pass

    # MCP client
    try:
        from dnallm.mcp.client import DNALLMMCPClient

        DNALLMMCPClient.__init__ = _MockMCPClient.__init__
        DNALLMMCPClient.dna_sequence_predict = _MockMCPClient.dna_sequence_predict
        DNALLMMCPClient.dna_batch_predict = _MockMCPClient.dna_batch_predict
        DNALLMMCPClient.dna_multi_model_predict = _MockMCPClient.dna_multi_model_predict
        DNALLMMCPClient.dna_stream_predict = _MockMCPClient.dna_stream_predict
        DNALLMMCPClient.dna_stream_batch_predict = _MockMCPClient.dna_stream_batch_predict
        DNALLMMCPClient.dna_stream_multi_model_predict = _MockMCPClient.dna_stream_multi_model_predict
        DNALLMMCPClient.dna_mutagenesis = _MockMCPClient.dna_mutagenesis
        DNALLMMCPClient.dna_interpret = _MockMCPClient.dna_interpret
        DNALLMMCPClient.list_loaded_models = _MockMCPClient.list_loaded_models
        DNALLMMCPClient.get_model_info = _MockMCPClient.get_model_info
        DNALLMMCPClient.list_models_by_task_type = _MockMCPClient.list_models_by_task_type
        DNALLMMCPClient.get_all_available_models = _MockMCPClient.get_all_available_models
        DNALLMMCPClient.health_check = _MockMCPClient.health_check
        DNALLMMCPClient.call = _MockMCPClient.call
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

# --- create temp files for doc examples ---
# Monkey-patch pandas to serve mock data for specific filenames used in docs.
_sample_df = None
_labels_df = None
try:
    if pd is not None:
        _sample_df = pd.DataFrame({
            "sequence": ["ATGCATGCATGC", "CGTACGTACGTA"],
            "label": [0, 1],
        })
        _labels_df = pd.DataFrame({
            "name": ["seq1", "seq2"],
            "label": [0, 1],
        })
        _your_dataset_df = pd.DataFrame({
            "sequence": ["ATGCATGCATGC", "CGTACGTACGTA"],
            "label": [0, 1],
        })
        _test_csv_df = pd.DataFrame({
            "sequence": ["ATGCATGCATGC", "CGTACGTACGTA", "TATATATATATA"],
            "label": [0, 1, 0],
        })

        _orig_read_csv = pd.read_csv
        def _mock_read_csv(filepath_or_buffer, *args, **kwargs):
            if isinstance(filepath_or_buffer, str):
                if filepath_or_buffer in ("labels.csv", "train_dataset.csv"):
                    return _labels_df.copy()
                if filepath_or_buffer == "your_dataset.csv":
                    return _your_dataset_df.copy()
                if filepath_or_buffer == "./data/test.csv":
                    return _test_csv_df.copy()
            return _orig_read_csv(filepath_or_buffer, *args, **kwargs)
        pd.read_csv = _mock_read_csv

        _orig_read_pickle = pd.read_pickle
        def _mock_read_pickle(filepath_or_buffer, *args, **kwargs):
            if isinstance(filepath_or_buffer, str) and filepath_or_buffer == "my_dataset.pkl":
                return _sample_df.copy()
            return _orig_read_pickle(filepath_or_buffer, *args, **kwargs)
        pd.read_pickle = _mock_read_pickle
except Exception:
    pass


# --- matplotlib mock ---
try:
    import matplotlib.pyplot as _plt
    _plt.show = lambda *a, **k: None
except Exception:
    pass


# --- peft mock ---
try:
    import peft as _peft_mod
    if hasattr(_peft_mod, "PeftModel"):
        _orig_PeftModel = _peft_mod.PeftModel
        class _MockPeftModel:
            @classmethod
            def from_pretrained(cls, model, model_id, *a, **k):
                return model
            def merge_and_unload(self, *a, **k):
                return _mock_model
        _peft_mod.PeftModel = _MockPeftModel
except Exception:
    pass


# --- nltk mock ---
try:
    import nltk.translate.bleu_score as _bleu_mod
    _bleu_mod.sentence_bleu = lambda *a, **k: 0.5
except Exception:
    pass


# --- os.path.exists mock for doc example paths ---
try:
    import os as _os_mod
    _orig_exists = _os_mod.path.exists
    def _mock_exists(path):
        if isinstance(path, str) and path in ("path/to/your/dna_sequences.csv",):
            return True
        return _orig_exists(path)
    _os_mod.path.exists = _mock_exists
except Exception:
    pass


# --- builtins exit mock (prevents subprocess termination) ---
try:
    import builtins as _builtins_mod
    _builtins_mod.exit = lambda *a, **k: None
    _builtins_mod.quit = lambda *a, **k: None
except Exception:
    pass


# --- common variables for doc blocks ---
try:
    loaded_models = {
        "model1": {"model": _mock_model, "tokenizer": _mock_tokenizer},
        "dnabert": {"model": _mock_model, "tokenizer": _mock_tokenizer},
        "plantcad2": {"model": _mock_model, "tokenizer": _mock_tokenizer},
    }
    datasets = {
        "promoter_strength": _make_mock_dataset(),
        "test": _make_mock_dataset(),
    }
    inference_engine = _MockDNAInference()
    benchmark = _MockBenchmark()
    mutagenesis = _MockMutagenesis()
    interpreter = _MockDNAInterpret()
    model = _mock_model
    tokenizer = _mock_tokenizer
    configs = _MockConfig()
    config = _MockConfig()
    dataset = _make_mock_dataset()
    my_datasets = _make_mock_dataset()
    sampled_datasets = _make_mock_dataset()
    # Mock optimizer for docs
    import torch as _torch
    optimizer = _torch.optim.AdamW(_mock_model.parameters(), lr=2e-5)
    sequence = "ATGCATGCATGC"
    mut_analyzer = _MockMutagenesis()
    dataloader = [{"input_ids": _torch.ones(1, 5, dtype=_torch.long), "attention_mask": _torch.ones(1, 5, dtype=_torch.long)}]
except Exception:
    pass

# --- inject common names into builtins for blocks that omit imports ---
try:
    import builtins as _builtins
    _builtins.load_model_and_tokenizer = lambda *a, **k: (_mock_model, _mock_tokenizer)
    _builtins.load_config = _mock_load_config
    _builtins.DNADataset = _make_mock_dataset()
    _builtins.DNATrainer = _MockDNATrainer
    _builtins.DNAInference = _MockDNAInference
    _builtins.Benchmark = _MockBenchmark
    _builtins.Mutagenesis = _MockMutagenesis
    _builtins.DNAInterpret = _MockDNAInterpret
    _builtins.DNAInterpreter = _MockDNAInterpret
    import random as _random
    _builtins.random = _random
    _builtins.reverse_complement = lambda seq: "".join({"A": "T", "T": "A", "G": "C", "C": "G"}.get(c, c) for c in reversed(seq))
    _builtins.apply_random_mutations = lambda seq, rate=0.1: seq
    # typing helpers for doc examples with type annotations
    from typing import Dict as _Dict, Any as _Any, List as _List, Optional as _Optional, Tuple as _Tuple
    _builtins.Dict = _Dict
    _builtins.Any = _Any
    _builtins.List = _List
    _builtins.Optional = _Optional
    _builtins.Tuple = _Tuple
    # other common names used without import
    import math as _math
    _builtins.math = _math
    _builtins.show_preset_dataset = lambda *a, **k: print("Available presets: plant-genomic-benchmark")
    _builtins.load_preset_dataset = lambda *a, **k: _make_mock_dataset()
    _builtins.custom_collate_fn = lambda batch: batch
    _builtins.DNALLMMCPClient = _MockMCPClient
    # fasta_to_df mock for format_conversion.md
    def _mock_fasta_to_df(path):
        if pd is not None:
            return pd.DataFrame({"name": ["seq1", "seq2"], "sequence": ["ATGCATGCATGC", "CGTACGTACGTA"]})
        return {"name": ["seq1", "seq2"], "sequence": ["ATGCATGCATGC", "CGTACGTACGTA"]}
    _builtins.fasta_to_df = _mock_fasta_to_df
    try:
        import dnallm.datahandling.data as _dna_data_mod
        _dna_data_mod.fasta_to_df = _mock_fasta_to_df
    except Exception:
        pass
except Exception:
    pass
