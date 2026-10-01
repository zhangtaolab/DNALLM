"""Behavior tests for DNAInterpret (captum attributions).

Attribution math runs on a real tiny torch module (see tests/conftest.py
TinyDNAModel) so autograd and captum semantics are exercised for real; mocked
engines cover only constructor and dispatch branches, mirroring the fake-engine
surface documented in tests/mcp/test_interpret_tool.py.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch

from dnallm.inference.interpret import (
    DNAInterpret,
    _CaptumWrapperInputEmbeds,
    _CaptumWrapperInputIDs,
)

SEQ = "ACGTAG"


class LinearOnlyModel(torch.nn.Module):
    """Real torch module with no nn.Embedding anywhere."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(9, 2)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=9).float()
        return type("Out", (), {"logits": self.proj(one_hot.mean(dim=1))})()


class EosFallbackTokenizer:
    """Tokenizer double with pad_token_id None and an eos_token_id."""

    pad_token_id = None
    eos_token_id = 5

    def __call__(self, seq, **kwargs):
        return {"input_ids": torch.tensor([[5, 6, 7]])}


class ConvertPadOnlyTokenizer:
    """Tokenizer double without pad_token_id, serving it via conversion."""

    pad_token = "[PAD]"  # ruff: ignore[hardcoded-password-string]

    def convert_tokens_to_ids(self, token):
        return 0


class TokenizeOnlyTokenizer:
    """Tokenizer double whose __call__ fails, forcing the tokenize() fallback."""

    def __call__(self, seq, **kwargs):
        raise OSError("tokenizer call unavailable")

    def tokenize(self, seq):
        return [5, 6, 7, 8]

    pad_token_id = 0


class TestConstruction:
    """DNAInterpret constructor and pad-id resolution."""

    def test_init_puts_model_in_eval_mode(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Construction calls model.eval() and resolves the pad token id."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        assert tiny_real_model.training is False
        assert interpreter.pad_token_id == 0

    def test_init_eos_fallback(self, tiny_real_model, inference_config_factory):
        """A None pad_token_id falls back to eos_token_id."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, EosFallbackTokenizer(), config)

        assert interpreter.pad_token_id == 5

    def test_init_pad_via_convert_tokens_to_ids(self, tiny_real_model, inference_config_factory):
        """Without pad_token_id, conversion of pad_token resolves the baseline id."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, ConvertPadOnlyTokenizer(), config)

        assert interpreter.pad_token_id == 0

    def test_find_embedding_layer_auto_detects(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The fallback module scan finds the tiny model's embedding layer."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        layer = interpreter._find_embedding_layer()

        assert layer is tiny_real_model.embedding
        assert interpreter.embedding_layer == "embedding"

    def test_find_embedding_layer_none_raises(self, simple_dna_tokenizer, inference_config_factory):
        """A model without any nn.Embedding raises a matchable RuntimeError."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(LinearOnlyModel(), simple_dna_tokenizer, config)

        with pytest.raises(RuntimeError, match=r"Could not auto-detect nn.Embedding"):
            interpreter._find_embedding_layer()


class TestCaptumWrappers:
    """The two captum forward wrappers' output contracts."""

    def test_input_ids_wrapper_returns_logits(self, tiny_real_model):
        """The input_ids wrapper unwraps .logits from model outputs."""
        wrapper = _CaptumWrapperInputIDs(tiny_real_model)
        out = wrapper(torch.tensor([[5, 6, 7, 8]]))

        assert out.shape == (1, 2)

    def test_wrapper_without_logits_raises(self, tiny_real_model):
        """A base model without a head raises a matchable TypeError."""

        class BaseLikeModel(torch.nn.Module):
            def forward(self, **kwargs):
                return {"last_hidden_state": torch.zeros(1, 2)}

        wrapper = _CaptumWrapperInputIDs(BaseLikeModel())

        with pytest.raises(TypeError, match=r"does not have a 'logits'"):
            wrapper(torch.tensor([[5, 6]]))

    def test_wrappers_dict_outputs(self, tiny_real_model):
        """Dict outputs with a 'logits' key unwrap through both wrappers."""

        class DictModel(torch.nn.Module):
            def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
                return {"logits": torch.zeros(1, 2)}

        model = DictModel()
        ids = torch.tensor([[5, 6]])
        assert _CaptumWrapperInputIDs(model)(ids).shape == (1, 2)
        assert _CaptumWrapperInputEmbeds(model)(torch.zeros(1, 2, 16)).shape == (1, 2)

    def test_embeds_wrapper_returns_logits(self, tiny_real_model):
        """The inputs_embeds wrapper routes through the model's embeds path."""
        wrapper = _CaptumWrapperInputEmbeds(tiny_real_model)
        embeds = tiny_real_model.embedding(torch.tensor([[5, 6, 7]]))

        out = wrapper(embeds)

        assert out.shape == (1, 2)


class TestRealAttributions:
    """Real captum attribution runs on the real tiny module (autograd required)."""

    def test_run_lig(self, tiny_real_model, simple_dna_tokenizer, inference_config_factory):
        """LayerIntegratedGradients returns finite per-token attributions."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        tokens, scores = interpreter.run_lig(SEQ, target=1)

        assert len(tokens) == 8
        assert scores.shape == (8,)
        assert np.isfinite(scores).all()
        assert np.abs(scores).sum() > 0

    def test_run_deeplift(self, tiny_real_model, simple_dna_tokenizer, inference_config_factory):
        """LayerDeepLift returns finite per-token attributions."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        tokens, scores = interpreter.run_deeplift(SEQ, target=1)

        assert len(tokens) == 8
        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_gradshap(self, tiny_real_model, simple_dna_tokenizer, inference_config_factory):
        """GradientShap runs at the embeds level and returns per-token scores."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_gradshap(SEQ, target=1, n_samples=2)

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_occlusion(self, tiny_real_model, simple_dna_tokenizer, inference_config_factory):
        """Oclusion perturbs each token against the pad baseline."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_occlusion(SEQ, target=1)

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_feature_ablation(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """FeatureAblation ablates one feature per token."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_feature_ablation(SEQ, target=1)

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_layer_conductance(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """LayerConductance attributes to an explicit internal layer via embeds."""
        model = tiny_model_factory(pooled=False)
        config = inference_config_factory(
            task_type="token", num_labels=2, label_names=["O", "I"], max_length=8
        )
        interpreter = DNAInterpret(model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_layer_conductance(
            SEQ, target=1, target_layer=model.head, token_index=2
        )

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_noise_tunnel_lig(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """NoiseTunnel over IntegratedGradients smooths embeds-level attributions."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_noise_tunnel(SEQ, target=1, base_method="lig", nt_samples=2)

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_noise_tunnel_unknown_base_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unknown base method raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"Unknown base_method: bogus"):
            interpreter.run_noise_tunnel(SEQ, target=1, base_method="bogus")


class TestTargetFormatting:
    """_format_captum_target per task type."""

    def test_string_target_resolves_to_label_index(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """A string target maps through label_names to its index."""
        config = inference_config_factory(
            task_type="binary", label_names=["negative", "positive"], max_length=8
        )
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        assert interpreter._format_captum_target("positive") == 1

    def test_token_task_requires_token_index(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Token classification without a position raises a matchable error."""
        config = inference_config_factory(
            task_type="token", num_labels=2, label_names=["O", "I"], max_length=8
        )
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"`token_index` must be provided"):
            interpreter._format_captum_target(1)

    def test_token_target_is_position_class_tuple(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Token classification formats the target as (token_index, class)."""
        config = inference_config_factory(
            task_type="token", num_labels=2, label_names=["O", "I"], max_length=8
        )
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        assert interpreter._format_captum_target(1, token_index=3) == (3, 1)

    def test_generation_defaults_to_last_token(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Generation targets default to the last token position."""
        config = inference_config_factory(task_type="generation", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        assert interpreter._format_captum_target(7) == (-1, 7)

    def test_unknown_task_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unexpected task type raises a matchable error."""
        config = inference_config_factory(task_type="mask", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.task_config.task_type = "weird"

        with pytest.raises(ValueError, match=r"Unknown task_type: weird"):
            interpreter._format_captum_target(1)


class TestInputHelpers:
    """Tokenization and token-decoding fallbacks."""

    def test_input_tensors_use_tokenizer_call(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The tokenizer call pads to max_length and returns id/mask tensors."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        input_ids, attention_mask = interpreter._get_input_tensors(SEQ, 8)

        assert input_ids.shape == (1, 8)
        assert attention_mask.shape == (1, 8)
        assert attention_mask[0, : len(SEQ)].all()
        assert not attention_mask[0, len(SEQ) :].any()

    def test_input_tensors_fallback_to_tokenize(self, tiny_real_model, inference_config_factory):
        """A failing tokenizer call falls back to tokenizer.tokenize ids."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, TokenizeOnlyTokenizer(), config)

        input_ids, attention_mask = interpreter._get_input_tensors(SEQ, 8)

        assert input_ids.tolist() == [[5, 6, 7, 8]]
        assert attention_mask.tolist() == [[1, 1, 1, 1]]

    def test_pad_baseline_filled_with_pad_id(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The baseline tensor is filled entirely with the pad token id."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        baseline = interpreter._get_pad_baseline(torch.tensor([[5, 6, 7]]))

        assert baseline.tolist() == [[0, 0, 0]]

    def test_ids_to_tokens_fallback_to_tokenize(self, tiny_real_model, inference_config_factory):
        """Without convert/decode helpers, tokens come from tokenize(input_seq)."""
        config = inference_config_factory(task_type="binary", max_length=8)
        tok = TokenizeOnlyTokenizer()
        interpreter = DNAInterpret(tiny_real_model, tok, config)

        tokens = interpreter._ids_to_tokens(torch.tensor([5, 6]), input_seq="AC")

        assert tokens == [5, 6, 7, 8]

    def test_ids_to_tokens_fallback_to_split(self, tiny_real_model, inference_config_factory):
        """With no tokenizer helpers at all, the sequence is split on spaces."""

        class BareTokenizer:
            pad_token_id = 0

        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, BareTokenizer(), config)

        tokens = interpreter._ids_to_tokens(torch.tensor([5]), input_seq="AA BB")

        assert tokens == ["AA", "BB"]


class TestInterpretDispatch:
    """The unified interpret/batch_interpret/plot_attributions interface."""

    def test_interpret_deeplift_stores_attributions(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """interpret() runs the method and stores results for plotting."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        tokens, scores = interpreter.interpret(SEQ, "deeplift", 1, plot=True)

        assert interpreter.attributions == (tokens, scores)

    def test_interpret_plot_false_clears_attributions(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """plot=False leaves no stored attributions."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        interpreter.interpret(SEQ, "deeplift", 1, plot=False)

        assert interpreter.attributions is None

    @pytest.mark.parametrize("method", ["lig", "gradshap", "occlusion", "feature_ablation"])
    def test_interpret_dispatch_simple_methods(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory, method
    ):
        """interpret() runs each simple attribution method and returns its scores."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        tokens, scores = interpreter.interpret(SEQ, method, 1)

        assert len(tokens) == 8
        assert np.isfinite(scores).all()

    def test_interpret_dispatch_layer_conductance(
        self, tiny_model_factory, simple_dna_tokenizer, inference_config_factory
    ):
        """interpret() forwards target_layer to run_layer_conductance."""
        model = tiny_model_factory(pooled=False)
        config = inference_config_factory(
            task_type="token", num_labels=2, label_names=["O", "I"], max_length=8
        )
        interpreter = DNAInterpret(model, simple_dna_tokenizer, config)

        _, scores = interpreter.interpret(
            SEQ, "layer_conductance", 1, token_index=2, target_layer=model.head
        )

        assert scores.shape == (8,)

    def test_interpret_dispatch_noise_tunnel(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """interpret() forwards base_method to run_noise_tunnel."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.interpret(
            SEQ, "noise_tunnel", 1, base_method="deeplift", nt_samples=2
        )

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_run_noise_tunnel_deeplift_base(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """NoiseTunnel over DeepLift runs at the embeds level."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        _, scores = interpreter.run_noise_tunnel(
            SEQ, target=1, base_method="deeplift", nt_samples=2
        )

        assert scores.shape == (8,)
        assert np.isfinite(scores).all()

    def test_interpret_unknown_method_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unknown method name raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"Unknown method: magic"):
            interpreter.interpret(SEQ, "magic", 1)

    def test_interpret_layer_conductance_requires_layer(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """layer_conductance without target_layer raises a matchable error."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        with pytest.raises(ValueError, match=r"`target_layer` must be provided"):
            interpreter.interpret(SEQ, "layer_conductance", 1)

    def test_batch_interpret_runs_each_sequence(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """batch_interpret returns one result per sequence and stores them all."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)

        results = interpreter.batch_interpret(["ACGT", "TTTT"], method="deeplift", targets=[1, 0])

        assert len(results) == 2
        assert interpreter.attributions == results

    def test_plot_attributions_token_dispatch(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The token plot type routes to plot_attributions_token."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.attributions = (["A", "C"], np.array([0.1, -0.2]))
        sentinel = object()

        with patch(
            "dnallm.inference.interpret.plot_attributions_token", return_value=sentinel
        ) as mock_plot:
            result = interpreter.plot_attributions("token")

        assert result is sentinel
        args, _ = mock_plot.call_args
        assert args[0] == ["A", "C"]
        assert np.allclose(args[1], [0.1, -0.2])

    def test_plot_attributions_line_dispatch(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """The line plot type routes to plot_attributions_line."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.attributions = (["A"], np.array([0.1]))
        sentinel = object()

        with patch(
            "dnallm.inference.interpret.plot_attributions_line", return_value=sentinel
        ) as mock_plot:
            result = interpreter.plot_attributions("line")

        assert result is sentinel
        mock_plot.assert_called_once()

    def test_plot_attributions_multi_for_lists(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """List attributions always route to the multi plot."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.attributions = [(["A"], np.array([0.1])), (["C"], np.array([-0.1]))]
        sentinel = object()

        with patch(
            "dnallm.inference.interpret.plot_attributions_multi", return_value=sentinel
        ) as mock_plot:
            result = interpreter.plot_attributions("token")

        assert result is sentinel
        mock_plot.assert_called_once()

    def test_plot_attributions_none_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """Plotting without stored attributions raises a matchable RuntimeError."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.attributions = None

        with pytest.raises(RuntimeError, match=r"No attributions found"):
            interpreter.plot_attributions()

    def test_plot_attributions_unknown_type_raises(
        self, tiny_real_model, simple_dna_tokenizer, inference_config_factory
    ):
        """An unknown plot type raises a matchable ValueError."""
        config = inference_config_factory(task_type="binary", max_length=8)
        interpreter = DNAInterpret(tiny_real_model, simple_dna_tokenizer, config)
        interpreter.attributions = (["A"], np.array([0.1]))

        with pytest.raises(ValueError, match=r"Unknown plot_type"):
            interpreter.plot_attributions("radar")
