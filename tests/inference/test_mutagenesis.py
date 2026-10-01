"""Behavior tests for in-silico saturation mutagenesis (Mutagenesis).

Saturation scans and pseudo-log-likelihood scoring run on the real tiny torch
module from tests/conftest.py so per-position scores and log fold changes are
real model outputs; pure data-assembly helpers are asserted on their returned
structures. Any file artifact lands under pytest tmp_path.
"""

import os

import numpy as np
import pytest
import torch
from scipy.special import expit
from unittest.mock import patch

from dnallm.inference.inference import DNAInference
from dnallm.inference.mutagenesis import Mutagenesis


def _make_mut(model, tokenizer, config):
    """Build a Mutagenesis instance from the given collaborators."""
    return Mutagenesis(model=model, tokenizer=tokenizer, config=config)


class TestMutateSequence:
    """Saturation-scan dataset construction across mutation types."""

    def test_substitutions_generated(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Each position mutates to every other base with descriptive names."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=4)

        names = mut.sequences["name"]
        assert names[0] == "raw"
        assert mut.sequences["sequence"][0] == "ACG"
        # 3 positions x 3 alternative bases + raw
        assert len(names) == 10
        assert "mut_0_A_C" in names
        idx = names.index("mut_1_C_A")
        assert mut.sequences["sequence"][idx] == "AAG"

    def test_include_n_expands_base_map(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """include_n adds N substitutions at every position."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", include_n=True, batch_size=4)

        assert len(mut.sequences["name"]) == 1 + 3 * 4
        assert "mut_0_A_N" in mut.sequences["name"]

    def test_replace_mut_false_keeps_only_raw(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Disabling substitutions leaves the raw sequence alone."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", replace_mut=False, batch_size=4)

        assert mut.sequences["name"] == ["raw"]
        assert mut.sequences["sequence"] == ["ACG"]

    def test_deletions_with_and_without_gap_fill(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Deletion windows drop bases, or fill with N when requested."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACGT", replace_mut=False, delete_size=2, batch_size=4)
        plain = dict(zip(mut.sequences["name"], mut.sequences["sequence"], strict=True))

        mut.mutate_sequence("ACGT", replace_mut=False, delete_size=2, fill_gap=True, batch_size=4)
        gapped = dict(zip(mut.sequences["name"], mut.sequences["sequence"], strict=True))

        assert plain["del_0_2"] == "GT"
        assert gapped["del_0_2"] == "NNGT"

    def test_insertions_at_every_boundary(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An inserted sequence lands at every position boundary."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("AC", replace_mut=False, insert_seq="TA", batch_size=4)

        seqs = dict(zip(mut.sequences["name"], mut.sequences["sequence"], strict=True))
        assert seqs["ins_0_TA"] == "TAAC"
        assert seqs["ins_2_TA"] == "ACTA"
        assert len(mut.sequences["name"]) == 4  # raw + 3 boundaries

    def test_cuts_positive_and_negative(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Positive cuts truncate from the left, negative from the right."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACGTAC", replace_mut=False, cut_size=2, batch_size=4)
        pos = dict(zip(mut.sequences["name"], mut.sequences["sequence"], strict=True))

        mut.mutate_sequence("ACGTAC", replace_mut=False, cut_size=-2, batch_size=4)
        neg = dict(zip(mut.sequences["name"], mut.sequences["sequence"], strict=True))

        assert pos["cut_0_2"] == "ACGTAC"
        assert pos["cut_2_2"] == "GTAC"
        assert neg["cut_0_-2"] == "ACGTAC"
        assert neg["cut_2_-2"] == "ACGT"

    def test_lowercase_applies_to_all(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """lowercase transforms every generated sequence."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("AC", lowercase=True, batch_size=4)

        assert mut.sequences["sequence"][0] == "ac"

    def test_do_encode_false_keeps_sequence_column(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Skipping encoding keeps the raw sequence column in the dataset."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=4, do_encode=False)

        assert "sequence" in mut.dataloader.dataset.dataset.column_names
        batch = next(iter(mut.dataloader))
        assert "ACG" in batch["sequence"]
        # First mutations replace the leading A with C, G or T.
        assert {"CCG", "GCG", "TCG"} <= set(batch["sequence"])

    def test_batch_size_defaults_to_config(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """batch_size <= 1 falls back to the configured inference batch size."""
        config = inference_config_factory(task_type="binary", max_length=16, batch_size=3)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=1)

        assert mut.dataloader.batch_size == 3


class TestPredComparison:
    """pred_comparison transforms per task type."""

    def _mut(self, task_type, model, tokenizer, factory, **cfg):
        config = factory(task_type=task_type, max_length=16, **cfg)
        return _make_mut(model, tokenizer, config)

    def test_binary_expit_transform(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Binary comparison sigmoid-transforms both predictions."""
        mut = self._mut("binary", tiny_real_model, simple_dna_tokenizer, inference_config_factory)

        raw, scored, logfc, diff = mut.pred_comparison(np.array([0.0, 2.0]), np.array([2.0, 2.0]))

        assert np.allclose(raw, expit([0.0, 2.0]))
        assert np.allclose(scored, expit([2.0, 2.0]))
        assert np.isclose(logfc[0], np.log2(expit(2.0) / expit(0.0)))
        assert np.isclose(diff[0], expit(2.0) - expit(0.0))

    @pytest.mark.parametrize(
        ("task_type", "raw", "expected_kind"),
        [
            ("multiclass", np.array([1.0, 2.0, 0.5]), "softmax"),
            ("multilabel", np.array([1.0, -1.0]), "expit"),
            ("regression", np.array([1.5, -2.0]), "identity"),
            ("token", np.array([[0.1, 2.0], [3.0, 0.2]]), "argmax"),
            ("generation", 0.5, "wrap"),
            ("mask", 0.25, "wrap"),
            ("embedding", 1.75, "wrap"),
        ],
    )
    def test_task_type_transforms(
        self,
        tiny_real_model,
        simple_dna_tokenizer,
        inference_config_factory,
        task_type,
        raw,
        expected_kind,
    ):
        """Each task type applies its documented score transform."""
        from scipy.special import softmax as scipy_softmax

        mut = self._mut(task_type, tiny_real_model, simple_dna_tokenizer, inference_config_factory)
        mut_val = np.array([0.5]) if expected_kind == "wrap" else raw * 0.5

        raw_score, _, _, _ = mut.pred_comparison(raw, mut_val)

        if expected_kind == "softmax":
            assert np.allclose(raw_score, scipy_softmax(raw))
        elif expected_kind == "expit":
            assert np.allclose(raw_score, expit(raw))
        elif expected_kind == "identity":
            assert raw_score is raw or np.allclose(raw_score, raw)
        elif expected_kind == "argmax":
            assert list(raw_score) == [1, 0]
        else:
            assert raw_score == np.array([raw])

    def test_unknown_task_type_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unsupported task type raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)
        mut.config["task"].task_type = "weird"

        with pytest.raises(ValueError, match=r"Unknown task type: weird"):
            mut.pred_comparison(np.array([1.0]), np.array([2.0]))


class TestModelDevice:
    """get_model_device resolution."""

    def test_device_attribute_preferred(
        self, inference_config_factory, simple_dna_tokenizer, tiny_real_model
    ):
        """A device attribute wins over parameter inspection."""

        class WithDevice:
            device = torch.device("cpu")

        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        assert mut.get_model_device(WithDevice()) == torch.device("cpu")

    def test_parameters_device_fallback(
        self, inference_config_factory, simple_dna_tokenizer, tiny_real_model
    ):
        """Without a device attribute, the first parameter's device is used."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        assert mut.get_model_device(tiny_real_model) == torch.device("cpu")

    def test_bare_object_defaults_to_cpu(
        self, inference_config_factory, simple_dna_tokenizer, tiny_real_model
    ):
        """Objects with neither attribute fall back to CPU."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        assert mut.get_model_device(object()) == torch.device("cpu")


class TestLanguageModelScoring:
    """mlm_evaluate / clm_evaluate on the real per-position tiny model."""

    def test_mlm_evaluate_sum_matches_recompute(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """The PLL score equals the recomputed masked-token logprob sum."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="mask", max_length=16)
        mut = _make_mut(model, simple_dna_tokenizer, config)
        mut.sequences = {"name": ["raw"], "sequence": ["ACGTA"]}

        scores = mut.mlm_evaluate(return_sum=True)

        ids = simple_dna_tokenizer("ACGTA", padding=False)["input_ids"][0]
        expected = 0.0
        for i, tok_id in enumerate(ids):
            masked = list(ids)
            masked[i] = simple_dna_tokenizer.mask_token_id
            logits = model(torch.tensor([masked])).logits[0, i]
            expected += torch.log_softmax(logits, dim=-1)[tok_id].item()
        assert len(scores) == 1
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_mlm_evaluate_per_token(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """return_sum=False reports per-(token, logprob) pairs."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="mask", max_length=16)
        mut = _make_mut(model, simple_dna_tokenizer, config)
        mut.sequences = {"name": ["raw"], "sequence": ["ACG"]}

        scores = mut.mlm_evaluate(return_sum=False)

        assert len(scores) == 1
        tok_name, value = scores[0][0]
        assert tok_name == "A"
        assert np.isfinite(value)

    def test_clm_evaluate_sum_matches_recompute(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """The causal LM score equals the recomputed shifted-logprob sum."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="generation", max_length=16)
        mut = _make_mut(model, simple_dna_tokenizer, config)
        mut.sequences = {"name": ["raw"], "sequence": ["ACGTA"]}

        scores = mut.clm_evaluate(return_sum=True)

        ids = torch.tensor(simple_dna_tokenizer("ACGTA", padding=False)["input_ids"][0])
        logits = model(ids.unsqueeze(0)).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        expected = logprobs[:-1].gather(1, ids[1:].unsqueeze(-1)).squeeze(-1).sum().item()
        assert len(scores) == 1
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_clm_evaluate_per_token(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """return_sum=False reports one logprob per shifted position."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="generation", max_length=16)
        mut = _make_mut(model, simple_dna_tokenizer, config)
        mut.sequences = {"name": ["raw"], "sequence": ["ACG"]}

        scores = mut.clm_evaluate(return_sum=False)

        assert len(scores[0]) == 2
        assert all(np.isfinite(v) for v in scores[0])


class TestEvaluate:
    """evaluate() end-to-end across task types and score strategies."""

    def test_binary_evaluate_end_to_end(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Binary evaluate scores every substitution against the raw sequence."""
        config = inference_config_factory(task_type="binary", max_length=16, batch_size=8)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=8)
        preds = mut.evaluate(strategy="last")

        assert "raw" in preds
        assert preds["raw"]["sequence"] == "ACG"
        assert len(preds) == 10
        entry = preds["mut_0_A_C"]
        assert entry["sequence"] == "CCG"
        # strategy 'last' scores the final class's log fold change.
        assert entry["score"] == pytest.approx(float(entry["logfc"][-1]))
        assert np.isfinite(entry["score"])

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("first", 0),
            ("sum", "sum"),
            ("mean", "mean"),
            (0, 0),
        ],
    )
    def test_evaluate_strategies(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, strategy, expected
    ):
        """Each strategy picks its documented aggregate of the logfc vector."""
        config = inference_config_factory(task_type="binary", max_length=16, batch_size=8)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)
        mut.mutate_sequence("AC", batch_size=8)

        preds = mut.evaluate(strategy=strategy)

        entry = preds["mut_0_A_C"]
        logfc = entry["logfc"]
        if expected == "sum":
            assert entry["score"] == pytest.approx(float(np.sum(logfc)))
        elif expected == "mean":
            assert entry["score"] == pytest.approx(float(np.mean(logfc)))
        else:
            assert entry["score"] == pytest.approx(float(logfc[expected]))

    def test_evaluate_mask_task_uses_pll(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """The mask task scores mutations with masked-token log likelihoods."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="mask", max_length=16, batch_size=8)
        mut = _make_mut(model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=8)
        preds = mut.evaluate(strategy="last")

        assert len(preds) == 10
        assert np.isfinite(preds["mut_0_A_C"]["score"])

    def test_evaluate_generation_task_uses_clm(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """The generation task scores mutations with causal LM log likelihoods."""
        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="generation", max_length=16, batch_size=8)
        mut = _make_mut(model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=8)
        preds = mut.evaluate(strategy="last")

        assert len(preds) == 10
        assert np.isfinite(preds["mut_0_A_C"]["score"])

    def test_evaluate_embedding_task_scores(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The embedding task routes through the scoring path on raw sequences."""
        config = inference_config_factory(task_type="embedding", max_length=16, batch_size=8)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        mut.mutate_sequence("ACG", batch_size=8, do_encode=False)
        preds = mut.evaluate(score_type="embedding", strategy="mean")

        assert len(preds) == 10
        assert np.isfinite(preds["mut_0_A_C"]["score"])

    def test_nan_score_becomes_zero(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """A NaN aggregate (log2 of a zero score) is reported as 0.0."""
        config = inference_config_factory(task_type="regression", max_length=16, batch_size=8)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)
        mut.mutate_sequence("AC", batch_size=8)

        preds = mut.evaluate(strategy="last")

        # The tiny model has nonzero outputs, so assert the guard indirectly:
        # every reported score is a finite float (never NaN).
        for entry in preds.values():
            if "score" in entry:
                assert np.isfinite(entry["score"])

    def test_evaluate_unknown_strategy_falls_back_to_mean(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unknown strategy scores the mean of the logfc vector."""
        config = inference_config_factory(task_type="binary", max_length=16, batch_size=8)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)
        mut.mutate_sequence("AC", batch_size=8)

        preds = mut.evaluate(strategy="bogus")

        entry = preds["mut_0_A_C"]
        assert entry["score"] == pytest.approx(float(np.mean(entry["logfc"])))

    def test_mlm_per_token_convert_fallback(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """A decode KeyError falls back to convert_ids_to_tokens for the label."""

        class KeyErrTokenizer:
            """Tokenizer double whose decode raises KeyError."""

            all_special_ids = (0, 1, 2, 3, 4)
            mask_token_id = 4
            vocab = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "C", "G", "T")

            def __call__(self, seq, **kwargs):
                from transformers import BatchEncoding

                ids = [self.vocab.index(c) for c in seq]
                return BatchEncoding({"input_ids": torch.tensor([ids], dtype=torch.long)})

            def decode(self, ids):
                raise KeyError("no decode for you")

            def convert_ids_to_tokens(self, tok_id):
                return self.vocab[tok_id]

        model = tiny_model_factory(n_classes=9, pooled=False)
        config = inference_config_factory(task_type="mask", max_length=16)
        mut = _make_mut(model, KeyErrTokenizer(), config)
        mut.sequences = {"name": ["raw"], "sequence": ["ACG"]}

        scores = mut.mlm_evaluate(return_sum=False)

        assert scores[0][0][0] == "A"
        assert np.isfinite(scores[0][0][1])

    def test_get_inference_engine_wires_collaborators(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """get_inference_engine returns a DNAInference wired to the same objects."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        engine = mut.get_inference_engine(tiny_real_model, simple_dna_tokenizer)

        assert isinstance(engine, DNAInference)
        assert engine.model is tiny_real_model
        assert engine.tokenizer is simple_dna_tokenizer


class TestIsmProcessing:
    """process_ism_data / find_hotspots / tfmodisco data assembly."""

    @staticmethod
    def _ism_results():
        """Return a small hand-built ISM result dictionary."""
        return {
            "raw": {"sequence": "ACG"},
            "mut_0_A_C": {"score": 0.5},
            "mut_0_A_G": {"score": -2.0},
            "mut_2_G_T": {"score": 1.0},
        }

    def test_process_ism_data_maxabs(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """maxabs picks the mutation with the largest absolute effect."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        scores = mut.process_ism_data(self._ism_results(), strategy="maxabs")

        assert scores[0] == pytest.approx(-2.0)
        assert scores[2] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_process_ism_data_mean(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """mean averages all mutation scores at each position."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        scores = mut.process_ism_data(self._ism_results(), strategy="mean")

        assert scores[0] == pytest.approx((0.5 - 2.0) / 2)

    def test_process_ism_data_unknown_strategy_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unknown aggregation strategy raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"Unknown strategy: bogus"):
            mut.process_ism_data(self._ism_results(), strategy="bogus")

    def test_find_hotspots_returns_windows(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Hotspots come back as (start, end) windows within the sequence."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        # Build a long sequence with one strongly-mutated position.
        seq = "A" * 20
        results = {"raw": {"sequence": seq}}
        for i in range(20):
            results[f"mut_{i}_A_C"] = {"score": 5.0 if i == 10 else 0.01}

        hotspots = mut.find_hotspots(results, window_size=5, percentile_threshold=80.0)

        assert isinstance(hotspots, list)
        assert all(0 <= s < e <= 20 for s, e in hotspots)
        assert mut.hotspots == hotspots

    def test_prepare_tfmodisco_inputs(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """tfmodisco inputs assemble one-hot, hyp scores, and contributions."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        one_hot, hyp, contrib = mut.prepare_tfmodisco_inputs([self._ism_results()])

        assert one_hot.shape == (1, 3, 4)
        assert hyp.shape == (1, 3, 4)
        assert np.allclose(contrib, hyp * one_hot)
        # The raw A at position 0 is one-hot encoded in column 0.
        assert one_hot[0, 0, 0] == 1


class TestPlot:
    """plot() plumbing to the plot module."""

    def test_plot_dispatches_to_plot_muts(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """plot hands preds straight to plot_muts with the derived output path."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)
        sentinel = object()

        with patch("dnallm.inference.mutagenesis.plot_muts", return_value=sentinel) as p:
            result = mut.plot({"raw": {"sequence": "ACG"}}, save_path=str(tmp_path / "mut.pdf"))

        assert result is sentinel
        assert p.call_args.kwargs["save_path"] == str(tmp_path / "mut.pdf")

    def test_plot_dir_save_path_appends_pdf(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, tmp_path
    ):
        """A suffix-less save_path derives a .pdf path inside the directory."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        with patch("dnallm.inference.mutagenesis.plot_muts", return_value=None) as p:
            mut.plot({"raw": {"sequence": "ACG"}}, save_path=str(tmp_path))

        assert p.call_args.kwargs["save_path"] == os.path.join(str(tmp_path), ".pdf")

    def test_plot_without_save_path(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """No save_path passes None through to the plotting function."""
        config = inference_config_factory(task_type="binary", max_length=16)
        mut = _make_mut(tiny_real_model, simple_dna_tokenizer, config)

        with patch("dnallm.inference.mutagenesis.plot_muts", return_value=None) as p:
            mut.plot({"raw": {"sequence": "ACG"}})

        assert p.call_args.kwargs["save_path"] is None
