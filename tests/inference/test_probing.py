"""Tests for dnallm.inference.probing (frozen-embedding probing).

Fast lane: tiny real torch model (``tiny_model_factory`` /
``simple_dna_tokenizer`` from the shared conftest) with synthetic
AT-rich/GC-rich binary labels. The slow-lane real-model acceptance lives at
the bottom of this file behind the ``slow`` marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnallm.inference.probing import (
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    MLP_EARLY_STOP,
    MLP_HIDDEN,
    EmbeddingResult,
    ProbeResult,
    extract_embeddings,
    fit_probe,
)
from dnallm.tasks.metric_registry import registered_names


def _synthetic_binary_data(n: int = 16, seed: int = 0) -> tuple[list[str], list[int]]:
    """Build AT-rich (label 1) vs GC-rich (label 0) sequences and labels."""
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for i in range(n):
        label = i % 2
        motif = "AT" if label == 1 else "GC"
        noise = "".join(rng.choice(list("ACGT"), size=8))
        sequences.append(motif * 6 + noise)
        labels.append(label)
    return sequences, labels


class TestExtractEmbeddings:
    """Fast-lane probing tests on the tiny model."""

    def test_end_to_end_probe_logistic_registry_metrics(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """extract -> fit('logistic') -> metrics dict has registry-canonical keys."""
        sequences, labels = _synthetic_binary_data()
        result = extract_embeddings(
            tiny_model_factory(),
            simple_dna_tokenizer,
            sequences,
            labels,
        )
        probe = fit_probe(
            result.embeddings[:10],
            result.labels[:10],
            result.embeddings[10:],
            result.labels[10:],
            kind="logistic",
            layer=result.layer,
            pooling=result.pooling,
        )
        canonical = set(registered_names())
        assert set(probe.metrics) <= canonical
        assert {"AUROC", "AUPRC", "accuracy"} <= set(probe.metrics)
        assert all(isinstance(v, float) for v in probe.metrics.values())
        assert all(0.0 <= v <= 1.0 for v in probe.metrics.values())
        assert probe.layer == -1
        assert probe.pooling == "mean"
        assert probe.kind == "logistic"

    def test_shapes_and_float32_dtype(self, tiny_model_factory, simple_dna_tokenizer):
        """Embeddings are (n, hidden) float32 with aligned labels."""
        sequences, labels = _synthetic_binary_data(n=12)
        result = extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, sequences, labels)
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (12, 16)
        assert result.embeddings.dtype == np.float32
        assert result.dtype == "float32"
        assert result.labels.shape == (12,)
        assert result.cache_hit is False
        assert result.cache_path is None

    def test_cache_hit_on_second_identical_call(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Second extract with the identical 4-tuple key hits the npz cache."""
        model = tiny_model_factory()
        sequences, labels = _synthetic_binary_data()
        kwargs = dict(
            model_name="tiny-model",
            dataset_name="synthetic",
            layer=-1,
            pooling="mean",
            output_dir=tmp_path,
        )
        first = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, **kwargs)
        assert first.cache_hit is False
        cache_dir = tmp_path / "probe_cache"
        assert cache_dir.is_dir()
        cached_files = list(cache_dir.iterdir())
        assert len(cached_files) == 1
        assert cached_files[0].name.startswith("probe_")
        assert cached_files[0].suffix == ".npz"

        second = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, **kwargs)
        assert second.cache_hit is True
        assert second.cache_path == str(cached_files[0])
        assert np.allclose(first.embeddings, second.embeddings)
        assert np.array_equal(first.labels, second.labels)
        # Exactly one file: the hit must not have written another entry.
        assert len(list(cache_dir.iterdir())) == 1

    def test_cache_miss_on_pooling_change(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Altering pooling MUST miss the cache (Pitfall 8 same-change test)."""
        model = tiny_model_factory()
        sequences, labels = _synthetic_binary_data()
        extract_embeddings(
            model,
            simple_dna_tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="mean",
            model_name="tiny-model",
            dataset_name="synthetic",
            output_dir=tmp_path,
        )
        changed = extract_embeddings(
            model,
            simple_dna_tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="cls",
            model_name="tiny-model",
            dataset_name="synthetic",
            output_dir=tmp_path,
        )
        assert changed.cache_hit is False
        assert changed.pooling == "cls"
        assert len(list((tmp_path / "probe_cache").iterdir())) == 2

    def test_cache_miss_on_layer_change(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Altering layer MUST miss the cache even when values coincide."""
        model = tiny_model_factory()
        sequences, labels = _synthetic_binary_data()
        common = dict(
            model_name="tiny-model",
            dataset_name="synthetic",
            pooling="mean",
            output_dir=tmp_path,
        )
        first = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, layer=-1, **common)
        assert first.cache_hit is False
        second = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, layer=0, **common)
        assert second.cache_hit is False
        assert second.layer == 0

    def test_cls_pooling_differs_from_mean(self, tiny_model_factory, simple_dna_tokenizer):
        """cls (first token) and mean pooling produce different embeddings."""
        sequences, labels = _synthetic_binary_data()
        model = tiny_model_factory()
        mean = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, pooling="mean")
        cls = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, pooling="cls")
        assert not np.allclose(mean.embeddings, cls.embeddings)

    def test_f4_row_schema(self, tiny_model_factory, simple_dna_tokenizer):
        """to_row() carries the documented F4 keys including layer/pooling/kind."""
        sequences, labels = _synthetic_binary_data()
        result = extract_embeddings(
            tiny_model_factory(),
            simple_dna_tokenizer,
            sequences,
            labels,
            model_name="tiny-model",
            dataset_name="synthetic",
        )
        probe = fit_probe(
            result.embeddings[:10],
            result.labels[:10],
            result.embeddings[10:],
            result.labels[10:],
            layer=result.layer,
            pooling=result.pooling,
            model_name=result.model_name,
            dataset_name=result.dataset_name,
            cache_hit=result.cache_hit,
        )
        assert isinstance(probe, ProbeResult)
        row = probe.to_row()
        assert set(row) == {
            "model",
            "dataset",
            "layer",
            "pooling",
            "kind",
            "metrics",
            "n_train",
            "n_test",
            "cache_hit",
        }
        assert row["model"] == "tiny-model"
        assert row["dataset"] == "synthetic"
        assert row["layer"] == -1
        assert row["pooling"] == "mean"
        assert row["kind"] == "logistic"
        assert row["n_train"] == 10
        assert row["n_test"] == 6

    def test_constants_hold_locked_values(self):
        """D-13 fixed hyperparameters keep their locked values."""
        assert LOGISTIC_MAX_ITER == 1000
        assert LOGISTIC_SOLVER == "lbfgs"
        assert MLP_HIDDEN == (256,)
        assert MLP_EARLY_STOP is True
