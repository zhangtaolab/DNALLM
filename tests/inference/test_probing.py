"""Tests for dnallm.inference.probing (frozen-embedding probing).

Fast lane: tiny real torch model (``tiny_model_factory`` /
``simple_dna_tokenizer`` from the shared conftest) with synthetic
AT-rich/GC-rich binary labels. The slow-lane real-model acceptance lives at
the bottom of this file behind the ``slow`` marker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
        kwargs = {
            "model_name": "tiny-model",
            "dataset_name": "synthetic",
            "layer": -1,
            "pooling": "mean",
            "output_dir": tmp_path,
        }
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

    def test_cache_miss_on_pooling_change(self, tiny_model_factory, simple_dna_tokenizer, tmp_path):
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

    def test_cache_miss_on_layer_change(self, tiny_model_factory, simple_dna_tokenizer, tmp_path):
        """Altering layer MUST miss the cache even when values coincide."""
        model = tiny_model_factory()
        sequences, labels = _synthetic_binary_data()
        common = {
            "model_name": "tiny-model",
            "dataset_name": "synthetic",
            "pooling": "mean",
            "output_dir": tmp_path,
        }
        first = extract_embeddings(
            model, simple_dna_tokenizer, sequences, labels, layer=-1, **common
        )
        assert first.cache_hit is False
        second = extract_embeddings(
            model, simple_dna_tokenizer, sequences, labels, layer=0, **common
        )
        assert second.cache_hit is False
        assert second.layer == 0

    def test_cls_pooling_differs_from_mean(self, tiny_model_factory, simple_dna_tokenizer):
        """cls (first token) and mean pooling produce different embeddings."""
        sequences, labels = _synthetic_binary_data()
        model = tiny_model_factory()
        mean = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, pooling="mean")
        cls = extract_embeddings(model, simple_dna_tokenizer, sequences, labels, pooling="cls")
        assert not np.allclose(mean.embeddings, cls.embeddings)

    def test_extraction_restores_output_hidden_states_flag(
        self, tiny_model_factory, simple_dna_tokenizer
    ):
        """The config flip is temporary: the externally owned model is handed
        back with its prior output_hidden_states value in both directions
        (IN-04) — later forwards must not keep materializing the stack."""
        sequences, labels = _synthetic_binary_data()

        model = tiny_model_factory()
        model.config.output_hidden_states = False
        extract_embeddings(model, simple_dna_tokenizer, sequences, labels)
        assert model.config.output_hidden_states is False

        model = tiny_model_factory()
        model.config.output_hidden_states = True
        extract_embeddings(model, simple_dna_tokenizer, sequences, labels)
        assert model.config.output_hidden_states is True

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


class TestProbeKinds:
    """Both probe estimators run end-to-end on synthetic embeddings."""

    def test_mlp_and_logistic_both_run_and_differ(self, tiny_model_factory, simple_dna_tokenizer):
        """mlp and logistic both run end-to-end; their estimator types differ."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        sequences, labels = _synthetic_binary_data(n=24)
        result = extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, sequences, labels)
        split = 16
        args = (
            result.embeddings[:split],
            result.labels[:split],
            result.embeddings[split:],
            result.labels[split:],
        )
        logistic = fit_probe(*args, kind="logistic")
        mlp = fit_probe(*args, kind="mlp")
        assert isinstance(logistic.estimator, LogisticRegression)
        assert isinstance(mlp.estimator, MLPClassifier)
        for probe in (logistic, mlp):
            assert {"AUROC", "AUPRC", "accuracy"} <= set(probe.metrics)
            assert all(0.0 <= v <= 1.0 for v in probe.metrics.values())

    def test_probe_constants_are_actually_consumed(self, tiny_model_factory, simple_dna_tokenizer):
        """The estimators carry exactly the locked constants (D-13)."""
        sequences, labels = _synthetic_binary_data(n=24)
        result = extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, sequences, labels)
        split = 16
        args = (
            result.embeddings[:split],
            result.labels[:split],
            result.embeddings[split:],
            result.labels[split:],
        )
        logistic = fit_probe(*args, kind="logistic")
        mlp = fit_probe(*args, kind="mlp")
        assert logistic.estimator.max_iter == LOGISTIC_MAX_ITER
        assert logistic.estimator.solver == LOGISTIC_SOLVER
        assert mlp.estimator.hidden_layer_sizes == MLP_HIDDEN
        assert mlp.estimator.early_stopping == MLP_EARLY_STOP

    def test_multiclass_scores_take_matrix_branch(self):
        """Non-binary label sets exercise the full-probability-matrix path."""
        rng = np.random.default_rng(3)
        centers = np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0]], dtype=np.float32)
        x_train = (rng.normal(size=(30, 2)) + centers[np.arange(30) % 3]).astype(np.float32)
        y_train = np.arange(30) % 3
        x_test = (rng.normal(size=(15, 2)) + centers[np.arange(15) % 3]).astype(np.float32)
        y_test = np.arange(15) % 3
        probe = fit_probe(x_train, y_train, x_test, y_test, kind="logistic")
        assert {"AUROC", "AUPRC", "accuracy"} <= set(probe.metrics)
        assert probe.metrics["accuracy"] > 0.9


class TestLeakageDiscipline:
    """Pitfall 8: probe metrics must never encode test-split statistics."""

    def test_scaler_fit_on_train_split_only(self, monkeypatch):
        """The ONLY StandardScaler.fit call sees exactly the train rows."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(7)
        n_train, n_test, dim = 40, 20, 5
        x_train = rng.normal(0.0, 1.0, size=(n_train, dim)).astype(np.float32)
        # Heavily shifted test distribution: fitting on anything containing
        # these rows changes the recorded statistics measurably.
        x_test = (rng.normal(0.0, 1.0, size=(n_test, dim)) + 50.0).astype(np.float32)
        y_train = np.arange(n_train) % 2
        y_test = np.arange(n_test) % 2

        original_fit = StandardScaler.fit
        fit_rows: list[int] = []

        def recording_fit(self, x, y=None):
            fit_rows.append(np.asarray(x).shape[0])
            return original_fit(self, x, y)

        monkeypatch.setattr(StandardScaler, "fit", recording_fit)

        result = fit_probe(x_train, y_train, x_test, y_test, kind="logistic")

        assert fit_rows == [n_train]
        # Recorded statistics are the train-split statistics, never the
        # combined ones — this fails if anyone fits on all data.
        assert np.allclose(result.scaler.mean_, x_train.mean(axis=0), atol=1e-5)
        combined_mean = np.concatenate([x_train, x_test]).mean(axis=0)
        assert not np.allclose(result.scaler.mean_, combined_mean)


class TestEdgeBattery:
    """Matchable ValueErrors, cache path edges, dtype and write-atomicity."""

    def test_fit_probe_single_class_train_split_raises_matchable(self):
        """A train split with one label class is rejected with dnallm's own
        matchable message, not sklearn's foreign solver error (IN-07)."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 4)).astype(np.float32)
        y_train = np.zeros(4, dtype=int)
        y_test = np.array([0, 1, 0, 1])

        with pytest.raises(
            ValueError, match=r"fit_probe requires both label classes in the train split"
        ):
            fit_probe(x[:4], y_train, x[4:], y_test)

    def test_empty_sequences_raise_value_error(self, tiny_model_factory, simple_dna_tokenizer):
        """0 rows after filtering raises; no cache file is written."""
        with pytest.raises(ValueError, match=r"after filtering"):
            extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, ["", "   "], [1, 0])

    def test_unknown_pooling_raises(self, tiny_model_factory, simple_dna_tokenizer):
        with pytest.raises(ValueError, match=r"Unknown pooling strategy 'max'"):
            extract_embeddings(
                tiny_model_factory(), simple_dna_tokenizer, ["ACGT"], [0], pooling="max"
            )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match=r"Unknown probe kind 'ridge'"):
            fit_probe([[0.0], [1.0]], [0, 1], [[0.5]], [1], kind="ridge")

    def test_layer_out_of_range_raises(self, tiny_model_factory, simple_dna_tokenizer):
        with pytest.raises(ValueError, match=r"layer 5 is out of range"):
            extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, ["ACGT"], [0], layer=5)

    def test_bool_layer_rejected(self, tiny_model_factory, simple_dna_tokenizer):
        with pytest.raises(ValueError, match=r"layer must be an int"):
            extract_embeddings(
                tiny_model_factory(), simple_dna_tokenizer, ["ACGT"], [0], layer=True
            )

    def test_numpy_integer_layer_accepted(self, tiny_model_factory, simple_dna_tokenizer):
        result = extract_embeddings(
            tiny_model_factory(),
            simple_dna_tokenizer,
            ["ACGTAC", "TTTGGG"],
            [0, 1],
            layer=np.int64(0),
        )
        assert result.layer == 0
        assert result.embeddings.shape == (2, 16)

    def test_sequence_label_length_mismatch_raises(self, tiny_model_factory, simple_dna_tokenizer):
        with pytest.raises(ValueError, match=r"lengths differ"):
            extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, ["ACGT", "TTTT"], [0])

    def test_model_without_hidden_states_raises(self, simple_dna_tokenizer):
        """A forward that returns no hidden states fails with a matchable error."""

        class _LogitOnlyModel:
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

            __call__ = forward

        with pytest.raises(ValueError, match=r"no hidden states"):
            extract_embeddings(_LogitOnlyModel(), simple_dna_tokenizer, ["ACGT"], [0])

    def test_strict_signature_without_kwargs_still_probes(self, simple_dna_tokenizer):
        """A forward accepting neither output_hidden_states nor **kwargs works."""

        class _StrictSigModel:
            def forward(self, input_ids=None, attention_mask=None):
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 5)
                return SimpleNamespace(hidden_states=[hidden])

            __call__ = forward

        result = extract_embeddings(_StrictSigModel(), simple_dna_tokenizer, ["ACGT"], [0])
        assert result.embeddings.shape == (1, 5)

    def test_readonly_config_still_probes(self, simple_dna_tokenizer):
        """A config that rejects the output_hidden_states flag degrades with a warning."""

        class _RaisingConfig:
            def __setattr__(self, name, value):
                raise ValueError("read-only config")

        class _StubModel:
            def __init__(self):
                self.config = _RaisingConfig()

            def forward(
                self, input_ids=None, attention_mask=None, output_hidden_states=None, **kwargs
            ):
                hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 4)
                return SimpleNamespace(hidden_states=[hidden])

            __call__ = forward

        with pytest.warns(UserWarning, match=r"Cannot enable output_hidden_states"):
            result = extract_embeddings(_StubModel(), simple_dna_tokenizer, ["ACGT"], [0])
        assert result.embeddings.shape == (1, 4)

    def test_attention_mask_derived_from_pad_id(self, tiny_model_factory, simple_dna_tokenizer):
        """Mask-less encodings fall back to the pad-id mask and agree with it."""
        from tests.conftest import SimpleDNATokenizer

        class _MasklessTokenizer(SimpleDNATokenizer):
            def __call__(self, sequences, **kwargs):
                enc = super().__call__(
                    sequences, return_tensors="pt", padding=True, truncation=True
                )
                return {"input_ids": enc["input_ids"]}

        sequences, labels = _synthetic_binary_data(n=8)
        model = tiny_model_factory()
        masked = extract_embeddings(model, simple_dna_tokenizer, sequences, labels)
        derived = extract_embeddings(model, _MasklessTokenizer(), sequences, labels)
        assert np.allclose(masked.embeddings, derived.embeddings)

    def test_attention_mask_none_falls_back_to_ones(self, simple_dna_tokenizer):
        from tests.conftest import SimpleDNATokenizer

        class _BareTokenizer(SimpleDNATokenizer):
            def __call__(self, sequences, **kwargs):
                enc = super().__call__(
                    sequences, return_tensors="pt", padding=True, truncation=True
                )
                return {"input_ids": enc["input_ids"]}

        class _StubModel:
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 3)
                return SimpleNamespace(hidden_states=[hidden])

            __call__ = forward

        bare = _BareTokenizer()
        bare.pad_token_id = None
        result = extract_embeddings(_StubModel(), bare, ["ACG", "TTA"], [0, 1])
        # ones-pooled over all positions of ones -> exactly 1.0 everywhere
        assert result.embeddings.shape == (2, 3)
        assert np.allclose(result.embeddings, 1.0)

    def test_cache_dir_created_with_missing_parents(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """output_dir with non-existent ancestors is created (parents=True)."""
        out = tmp_path / "a" / "b" / "c"
        extract_embeddings(
            tiny_model_factory(),
            simple_dna_tokenizer,
            ["ACGT", "TTGG"],
            [0, 1],
            model_name="m",
            dataset_name="d",
            output_dir=out,
        )
        assert (out / "probe_cache").is_dir()
        assert len(list((out / "probe_cache").iterdir())) == 1

    def test_float32_roundtrip_preserves_dtype_and_values(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Cached reload preserves float32 dtype and values within allclose."""
        sequences, labels = _synthetic_binary_data(n=10)
        kwargs = {"model_name": "m", "dataset_name": "d", "output_dir": tmp_path}
        first = extract_embeddings(
            tiny_model_factory(), simple_dna_tokenizer, sequences, labels, **kwargs
        )
        assert first.embeddings.dtype == np.float32
        second = extract_embeddings(
            tiny_model_factory(), simple_dna_tokenizer, sequences, labels, **kwargs
        )
        assert second.cache_hit is True
        assert second.embeddings.dtype == np.float32
        assert second.dtype == "float32"
        assert np.allclose(first.embeddings, second.embeddings, rtol=0, atol=0)
        # The stored metadata records the dtype.
        import json

        with np.load(second.cache_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            stored_dtype = data["embeddings"].dtype
        assert meta["dtype"] == "float32"
        assert stored_dtype == np.float32

    def test_concurrent_same_key_writes_leave_one_valid_entry(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Simulated concurrent double-write: one valid winner, no temp leftovers."""
        from concurrent.futures import ThreadPoolExecutor

        sequences, labels = _synthetic_binary_data(n=10)
        model = tiny_model_factory()
        out = tmp_path / "out"

        def run(_):
            return extract_embeddings(
                model,
                simple_dna_tokenizer,
                sequences,
                labels,
                model_name="m",
                dataset_name="d",
                layer=-1,
                pooling="mean",
                output_dir=out,
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, range(4)))

        cache_files = list((out / "probe_cache").iterdir())
        assert len(cache_files) == 1
        assert cache_files[0].suffix == ".npz"
        assert ".tmp-" not in cache_files[0].name

        final = run(0)
        assert final.cache_hit is True
        assert np.allclose(final.embeddings, results[0].embeddings)
        assert len(list((out / "probe_cache").iterdir())) == 1

    def test_failed_replace_cleans_temp_file(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path, monkeypatch
    ):
        """A failed atomic write removes its temp file and surfaces the error."""
        import os as os_module

        def _boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os_module, "replace", _boom)
        with pytest.raises(OSError, match=r"disk full"):
            extract_embeddings(
                tiny_model_factory(),
                simple_dna_tokenizer,
                ["ACGT", "TTGG"],
                [0, 1],
                model_name="m",
                dataset_name="d",
                output_dir=tmp_path,
            )
        leftovers = list((tmp_path / "probe_cache").iterdir())
        assert leftovers == []

    def test_corrupt_cache_entry_is_treated_as_miss(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Unreadable cache bytes warn and trigger a clean recompute."""
        sequences, labels = _synthetic_binary_data(n=8)
        kwargs = {"model_name": "m", "dataset_name": "d", "output_dir": tmp_path}
        extract_embeddings(tiny_model_factory(), simple_dna_tokenizer, sequences, labels, **kwargs)
        cache_files = list((tmp_path / "probe_cache").iterdir())
        cache_files[0].write_text("not an npz at all")

        with pytest.warns(UserWarning, match=r"unreadable probe cache"):
            recomputed = extract_embeddings(
                tiny_model_factory(), simple_dna_tokenizer, sequences, labels, **kwargs
            )
        assert recomputed.cache_hit is False
        # The recompute replaces the corrupt entry with a valid one.
        third = extract_embeddings(
            tiny_model_factory(), simple_dna_tokenizer, sequences, labels, **kwargs
        )
        assert third.cache_hit is True

    def test_key_mismatched_cache_entry_is_treated_as_miss(
        self, tiny_model_factory, simple_dna_tokenizer, tmp_path
    ):
        """Bytes parked at a key's filename but describing another key miss."""
        import json

        from dnallm.inference.probing import _cache_filename, _cache_key

        sequences, labels = _synthetic_binary_data(n=8)
        model = tiny_model_factory()
        mean_result = extract_embeddings(
            model,
            simple_dna_tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="mean",
            model_name="m",
            dataset_name="d",
            output_dir=tmp_path,
        )
        # Park mean-keyed content at the cls key's filename.
        wrong_path = tmp_path / "probe_cache" / _cache_filename(_cache_key("m", "d", -1, "cls"))
        wrong_path.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array(
            json.dumps({
                "model": "m",
                "dataset": "d",
                "layer": -1,
                "pooling": "mean",
                "dtype": "float32",
            })
        )
        with open(wrong_path, "wb") as handle:
            np.savez(
                handle,
                embeddings=mean_result.embeddings,
                labels=mean_result.labels,
                meta=meta,
            )

        cls_result = extract_embeddings(
            model,
            simple_dna_tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="cls",
            model_name="m",
            dataset_name="d",
            output_dir=tmp_path,
        )
        assert cls_result.cache_hit is False
        assert cls_result.pooling == "cls"

    def test_output_rows_record_layer_pooling_kind(self, tiny_model_factory, simple_dna_tokenizer):
        """Every result/row records layer + pooling + kind (comparability)."""
        sequences, labels = _synthetic_binary_data(n=20)
        result = extract_embeddings(
            tiny_model_factory(),
            simple_dna_tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="cls",
        )
        probe = fit_probe(
            result.embeddings[:14],
            result.labels[:14],
            result.embeddings[14:],
            result.labels[14:],
            kind="mlp",
            layer=result.layer,
            pooling=result.pooling,
        )
        assert probe.layer == -1
        assert probe.pooling == "cls"
        assert probe.kind == "mlp"
        row = probe.to_row()
        assert (row["layer"], row["pooling"], row["kind"]) == (-1, "cls", "mlp")


class TestHiddenStateExtractionPrecedence:
    """The output-shape precedence chain in _hidden_states_from_outputs."""

    def test_dict_output_with_hidden_states_key(self):
        from dnallm.inference.probing import _hidden_states_from_outputs

        hidden = torch.zeros(2, 3, 4)
        outputs = {"hidden_states": [hidden], "logits": torch.zeros(2, 2)}
        assert _hidden_states_from_outputs(outputs) == [hidden]

    def test_tuple_output_takes_first_element(self):
        from dnallm.inference.probing import _hidden_states_from_outputs

        hidden = torch.zeros(2, 3, 4)
        assert _hidden_states_from_outputs((hidden, torch.zeros(2, 2))) == [hidden]

    def test_single_tensor_becomes_one_layer_list(self):
        from dnallm.inference.probing import _hidden_states_from_outputs

        hidden = torch.zeros(2, 3, 4)
        result = _hidden_states_from_outputs(SimpleNamespace(last_hidden_state=hidden))
        assert result == [hidden]


class TestRealModelAcceptance:
    """PROB-01 slow-lane acceptance on the models.lock-pinned model + dataset."""

    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_probe_end_to_end_real_model_two_layers(self, tmp_path):
        """Any model x any binary task: pinned plant-dnabert-BPE x core promoters.

        Extracts frozen embeddings at TWO layers (last + one intermediate —
        the NT-paper layer finding makes the intermediate layer a real case),
        fits both probe kinds, and asserts registry-canonical metrics, cache
        hit on the second call, and layer/pooling/kind on every output row.
        """
        from dnallm import DNADataset, load_config, load_model_and_tokenizer

        config = load_config(Path(__file__).parent / "inference_config.yaml")
        model_name = "zhangtaolab/plant-dnabert-BPE"
        data_name = "zhangtaolab/plant-multi-species-core-promoters"
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_name, task_config=config["task"], source="modelscope"
            )
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )
        except Exception as e:
            pytest.skip(f"network-unavailable: {model_name} / {data_name}: {e}")

        raw = datasets.dataset
        if hasattr(raw, "keys") and "train" in list(raw.keys()):
            split = raw["train"]
        elif hasattr(raw, "train"):
            split = raw.train
        else:
            split = raw
        all_sequences = list(split["sequence"])
        all_labels = [int(label) for label in split["labels"]]

        # Balanced, seeded subsample: probing is a frozen-backbone evaluation,
        # so 60 rows are enough evidence that the path works end-to-end.
        positives = [i for i, label in enumerate(all_labels) if label == 1][:30]
        negatives = [i for i, label in enumerate(all_labels) if label == 0][:30]
        rng = np.random.default_rng(42)
        chosen = rng.permutation(positives + negatives).tolist()
        sequences = [all_sequences[i] for i in chosen]
        labels = [all_labels[i] for i in chosen]
        train_slice, test_slice = slice(0, 36), slice(36, 60)

        n_layers = int(getattr(model.config, "num_hidden_layers", 6) or 6)
        layers = [-1, max(1, n_layers // 2)]

        rows = []
        for layer in layers:
            result = extract_embeddings(
                model,
                tokenizer,
                sequences,
                labels,
                layer=layer,
                pooling="mean",
                model_name=model_name,
                dataset_name=data_name,
                output_dir=tmp_path,
            )
            assert result.cache_hit is False
            assert result.embeddings.dtype == np.float32
            for kind in ("logistic", "mlp"):
                probe = fit_probe(
                    result.embeddings[train_slice],
                    result.labels[train_slice],
                    result.embeddings[test_slice],
                    result.labels[test_slice],
                    kind=kind,
                    layer=layer,
                    pooling="mean",
                    model_name=model_name,
                    dataset_name=data_name,
                    cache_hit=result.cache_hit,
                )
                assert {"AUROC", "AUPRC", "accuracy"} <= set(probe.metrics)
                assert all(
                    isinstance(v, float) and np.isfinite(v) and 0.0 <= v <= 1.0
                    for v in probe.metrics.values()
                )
                row = probe.to_row()
                assert row["layer"] == layer
                assert row["pooling"] == "mean"
                assert row["kind"] == kind
                rows.append(row)

        # Two distinct (layer) keys -> two cache entries.
        cache_files = list((tmp_path / "probe_cache").iterdir())
        assert len(cache_files) == 2

        # Second extract with the identical 4-tuple key hits the cache.
        cached = extract_embeddings(
            model,
            tokenizer,
            sequences,
            labels,
            layer=-1,
            pooling="mean",
            model_name=model_name,
            dataset_name=data_name,
            output_dir=tmp_path,
        )
        assert cached.cache_hit is True
        assert cached.layer == -1
        assert len(rows) == 4
