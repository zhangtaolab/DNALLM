"""Real-torch tests for the CrossDNA special handler.

CrossDNA's own imports are torch/transformers only, so everything here runs
against real modules: the checkpoint scanner over tmp directories, the
generated sequence-classification class on a tiny in-repo stand-in for the
upstream MLM, and the handler dispatch contract.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from dnallm.models.special.crossdna import (
    _build_crossdna_sequence_classification_class,
    _handle_crossdna_models,
    _problem_type_from_task,
    _read_crossdna_config,
    _register_crossdna_sequence_classification,
    _resolve_crossdna_checkpoint_dir,
)


class TinyCrossDNABackbone(nn.Module):
    """Stand-in for the upstream CrossDNA backbone (for_representation mode)."""

    def __init__(self, d_model=8, alphabet_size=5):
        super().__init__()
        self.embed = nn.Embedding(alphabet_size, d_model)
        self.pretrain = None
        self.for_representation = None
        self.gate_freeze_steps = None
        self.detach_gate = None
        self.use_ema_teacher = None
        self.auto_update_ema_in_forward = None
        self.use_rc_kl = None
        self.use_barlow = None
        self.use_tv = None

    def forward(self, input_ids):
        return (self.embed(input_ids), None)


class TinyCrossDNAForMaskedLM(nn.Module):
    """Stand-in for the remote CrossDNAForMaskedLM dynamic class."""

    base_model_prefix = "backbone"

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.backbone = TinyCrossDNABackbone(
            d_model=config.d_model, alphabet_size=int(getattr(config, "alphabet_size", 5))
        )

    def post_init(self):
        """No-op stand-in for the HF weight-initialization hook."""


def _make_config(**overrides):
    """Build the config shape the generated class reads."""
    config = SimpleNamespace(
        num_labels=2,
        d_model=8,
        alphabet_size=5,
        classifier_pooling="mean",
        classifier_dropout=0.1,
        classifier_gate_freeze_steps=0,
        classifier_detach_gate=False,
        auto_remap_tokenizer_ids=True,
        tokenizer_base_offset=7,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_classifier(config=None, base_cls=TinyCrossDNAForMaskedLM):
    """Build the generated CrossDNA classifier class and one instance."""
    config = config if config is not None else _make_config()
    cls = _build_crossdna_sequence_classification_class(base_cls, type(config))
    return cls(config)


def _write_checkpoint(directory, model_type="crossdna", auto_map=None):
    """Write a CrossDNA-shaped config.json into *directory*."""
    payload = {"model_type": model_type, "d_model": 8, "alphabet_size": 5}
    if auto_map is not None:
        payload["auto_map"] = auto_map
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(payload))
    return str(directory)


class TestReadCrossdnaConfig:
    """_read_crossdna_config checkpoint detection."""

    def test_missing_config_returns_none(self, tmp_path):
        """A directory without config.json is not a checkpoint."""
        assert _read_crossdna_config(str(tmp_path)) is None

    def test_malformed_json_returns_none(self, tmp_path):
        """A corrupt config.json is treated as no checkpoint."""
        (tmp_path / "config.json").write_text("{not json")

        assert _read_crossdna_config(str(tmp_path)) is None

    def test_model_type_crossdna_returns_config(self, tmp_path):
        """model_type=crossdna marks a checkpoint."""
        _write_checkpoint(tmp_path)

        config = _read_crossdna_config(str(tmp_path))

        assert config["model_type"] == "crossdna"

    def test_auto_map_reference_marks_checkpoint(self, tmp_path):
        """An auto_map naming CrossDNA marks a checkpoint regardless of model_type."""
        _write_checkpoint(
            tmp_path,
            model_type="other",
            auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"},
        )

        config = _read_crossdna_config(str(tmp_path))

        assert config is not None

    def test_unrelated_config_returns_none(self, tmp_path):
        """Unrelated model configs are rejected."""
        _write_checkpoint(tmp_path, model_type="bert")

        assert _read_crossdna_config(str(tmp_path)) is None


class TestResolveCheckpointDir:
    """_resolve_crossdna_checkpoint_dir directory resolution order."""

    def test_direct_directory(self, tmp_path):
        """The path itself wins when it holds a CrossDNA config."""
        _write_checkpoint(tmp_path)

        assert _resolve_crossdna_checkpoint_dir(str(tmp_path)) == str(tmp_path)

    def test_preferred_subdirectory(self, tmp_path):
        """The known 8.1M checkpoint subdirectory is resolved."""
        _write_checkpoint(tmp_path / "8.1M")

        resolved = _resolve_crossdna_checkpoint_dir(str(tmp_path))

        assert resolved == str(tmp_path / "8.1M")

    def test_single_unknown_child(self, tmp_path):
        """Exactly one CrossDNA child directory resolves to it."""
        _write_checkpoint(tmp_path / "custom-size")

        resolved = _resolve_crossdna_checkpoint_dir(str(tmp_path))

        assert resolved == str(tmp_path / "custom-size")

    def test_multiple_children_fail_loudly(self, tmp_path):
        """Multiple candidate checkpoints raise instead of guessing."""
        _write_checkpoint(tmp_path / "size-a")
        _write_checkpoint(tmp_path / "size-b")

        with pytest.raises(ValueError, match="Multiple CrossDNA checkpoint directories"):
            _resolve_crossdna_checkpoint_dir(str(tmp_path))

    def test_no_candidates_returns_none(self, tmp_path):
        """A directory without any CrossDNA config yields None."""
        (tmp_path / "unrelated").mkdir()

        assert _resolve_crossdna_checkpoint_dir(str(tmp_path)) is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        """A nonexistent path yields None without raising."""
        assert _resolve_crossdna_checkpoint_dir(str(tmp_path / "missing")) is None


class TestBuildSequenceClassificationClass:
    """The generated CrossDNAForSequenceClassification class."""

    def test_downstream_flags_applied(self):
        """Construction flips the config and backbone into downstream mode."""
        config = _make_config()

        model = _build_classifier(config)

        assert config.pretrain is False
        assert config.for_representation is True
        assert config.architectures == ["CrossDNAForSequenceClassification"]
        assert model.backbone.pretrain is False
        assert model.backbone.for_representation is True
        assert model.backbone.use_ema_teacher is False
        assert model.backbone.use_barlow is False

    def test_classifier_head_built_from_d_model(self):
        """The classifier maps d_model to num_labels."""
        model = _build_classifier(_make_config(num_labels=3))

        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.out_features == 3
        assert model.classifier.in_features == 8

    def test_ema_modules_deleted(self):
        """EMA teacher attributes are removed from the backbone."""
        config = _make_config()
        instance_holder = {}

        class EmaBase(TinyCrossDNAForMaskedLM):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                instance_holder["self"] = self
                self.backbone.branchA_core_ema = nn.Linear(2, 2)
                self.backbone.bridge_ema = nn.Linear(2, 2)

        model = _build_classifier(config, base_cls=EmaBase)

        assert not hasattr(model.backbone, "branchA_core_ema")
        assert not hasattr(model.backbone, "bridge_ema")

    def test_class_cached_per_key(self):
        """The same (base, config-class) pair yields the identical class object."""
        first = _build_crossdna_sequence_classification_class(
            TinyCrossDNAForMaskedLM, SimpleNamespace
        )
        second = _build_crossdna_sequence_classification_class(
            TinyCrossDNAForMaskedLM, SimpleNamespace
        )

        assert first is second

    def test_num_labels_below_one_raises(self):
        """num_labels < 1 is rejected."""
        with pytest.raises(ValueError, match="num_labels must be >= 1"):
            _build_classifier(_make_config(num_labels=0))

    def test_classifier_dropout_out_of_range_raises(self):
        """classifier_dropout outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="classifier_dropout must be in"):
            _build_classifier(_make_config(classifier_dropout=1.5))

    def test_invalid_pooling_raises(self):
        """An unknown classifier_pooling is rejected."""
        with pytest.raises(ValueError, match="classifier_pooling must be one of"):
            _build_classifier(_make_config(classifier_pooling="median"))


class TestPrepareInputIds:
    """Tokenizer-id remapping in _prepare_input_ids."""

    def test_none_input_raises(self):
        model = _build_classifier()

        with pytest.raises(ValueError, match="input_ids must be provided"):
            model._prepare_input_ids(None)

    def test_1d_input_raises(self):
        model = _build_classifier()

        with pytest.raises(ValueError, match=r"input_ids must have shape \[batch, length\]"):
            model._prepare_input_ids(torch.tensor([1, 2, 3]))

    def test_float_input_raises_typeerror(self):
        model = _build_classifier()

        with pytest.raises(TypeError, match="integer token IDs"):
            model._prepare_input_ids(torch.ones(1, 3))

    def test_mask_shape_mismatch_raises(self):
        model = _build_classifier()

        with pytest.raises(ValueError, match="attention_mask must have the same"):
            model._prepare_input_ids(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 4))

    def test_remap_disabled_passthrough(self):
        """auto_remap_tokenizer_ids=False leaves the ids untouched."""
        model = _build_classifier(_make_config(auto_remap_tokenizer_ids=False))
        ids = torch.tensor([[10, 11]])

        normalized, mask = model._prepare_input_ids(ids)

        assert torch.equal(normalized, ids)
        assert mask is None

    def test_native_ids_below_offset_passthrough(self):
        """Ids already below the offset are not remapped."""
        model = _build_classifier()
        ids = torch.tensor([[0, 1, 4]])

        normalized, _mask = model._prepare_input_ids(ids)

        assert torch.equal(normalized, ids)

    def test_empty_batch_passthrough(self):
        """Empty tensors skip the remap machinery."""
        model = _build_classifier()

        normalized, _mask = model._prepare_input_ids(torch.zeros(0, 3, dtype=torch.long))

        assert normalized.shape == (0, 3)

    def test_tokenizer_ids_remapped(self):
        """Ids in the tokenizer range map back onto the native alphabet."""
        model = _build_classifier()  # offset 7, alphabet_size 5 -> range [7, 12)
        ids = torch.tensor([[7, 8, 9, 11]])
        mask = torch.tensor([[1, 1, 0, 1]])

        normalized, effective_mask = model._prepare_input_ids(ids, mask)

        assert normalized.tolist() == [[0, 1, 2, 4]]
        assert effective_mask.tolist() == [[1, 1, 0, 1]]

    def test_out_of_range_ids_become_unk(self):
        """Ids outside the tokenizer window map to the N id."""
        model = _build_classifier()

        normalized, _ = model._prepare_input_ids(torch.tensor([[7, 99]]))

        assert normalized.tolist() == [[0, 4]]

    def test_mask_derived_from_base_mask_without_attention(self):
        """Without an attention mask the remap window becomes the mask."""
        model = _build_classifier()

        _normalized, effective_mask = model._prepare_input_ids(torch.tensor([[7, 99]]))

        assert effective_mask.tolist() == [[1, 0]]


class TestPoolSequence:
    """Sequence-level pooling in _pool_sequence."""

    def test_2d_hidden_raises(self):
        model = _build_classifier()

        with pytest.raises(ValueError, match=r"hidden_states with shape \[B, L, H\]"):
            model._pool_sequence(torch.randn(2, 8))

    def test_mask_shape_mismatch_raises(self):
        model = _build_classifier()

        with pytest.raises(ValueError, match="attention_mask must match hidden states"):
            model._pool_sequence(torch.randn(2, 3, 8), torch.ones(2, 4))

    @pytest.mark.parametrize("pooling", ["mean", "max", "first", "last"])
    def test_pooling_modes_shapes(self, pooling):
        """All four pooling modes return (batch, hidden)."""
        model = _build_classifier(_make_config(classifier_pooling=pooling))
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

        pooled = model._pool_sequence(hidden, mask)

        assert pooled.shape == (2, 8)

    def test_mean_pooling_respects_mask(self):
        """Mean pooling averages only unmasked positions."""
        model = _build_classifier(_make_config(classifier_pooling="mean"))
        hidden = torch.ones(1, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0]])

        pooled = model._pool_sequence(hidden, mask)

        assert torch.allclose(pooled, torch.ones(1, 8))

    def test_none_mask_uses_all_positions(self):
        """A missing mask pools over every position."""
        model = _build_classifier(_make_config(classifier_pooling="mean"))
        hidden = torch.ones(1, 3, 8)

        pooled = model._pool_sequence(hidden, None)

        assert torch.allclose(pooled, torch.ones(1, 8))

    def test_fully_masked_rows_return_zeros(self):
        """Rows whose mask is entirely zero pool to zeros, not NaNs."""
        model = _build_classifier(_make_config(classifier_pooling="mean"))
        hidden = torch.ones(2, 3, 8)
        mask = torch.tensor([[1, 1, 1], [0, 0, 0]])

        pooled = model._pool_sequence(hidden, mask)

        assert torch.allclose(pooled[1], torch.zeros(8))

    def test_first_pooling_picks_first_unmasked(self):
        """'first' pooling takes each row's first unmasked position."""
        model = _build_classifier(_make_config(classifier_pooling="first"))
        hidden = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        mask = torch.tensor([[0, 1, 1], [1, 0, 0]])

        pooled = model._pool_sequence(hidden, mask)

        assert torch.equal(pooled[0], hidden[0, 1, :])
        assert torch.equal(pooled[1], hidden[1, 0, :])

    def test_last_pooling_picks_last_unmasked(self):
        """'last' pooling takes each row's last unmasked position."""
        model = _build_classifier(_make_config(classifier_pooling="last"))
        hidden = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        pooled = model._pool_sequence(hidden, mask)

        assert torch.equal(pooled[0], hidden[0, 1, :])
        assert torch.equal(pooled[1], hidden[1, 2, :])

    def test_unexpected_pooling_raises_runtimeerror(self):
        """A pooling mode that escapes validation raises RuntimeError."""
        model = _build_classifier()
        model.pooling = "bogus"

        with pytest.raises(RuntimeError, match="Unexpected pooling mode"):
            model._pool_sequence(torch.randn(1, 3, 8), None)


class TestCrossDnaForward:
    """Real-torch forwards of the generated classification model."""

    def _ids(self, offset_ids=False, batch=2, length=4):
        if offset_ids:
            return torch.randint(7, 12, (batch, length))
        return torch.randint(0, 5, (batch, length))

    def test_forward_logits_and_gradient(self):
        """Forward yields (batch, num_labels) logits with live gradients."""
        model = _build_classifier()

        output = model(input_ids=self._ids())

        assert output.logits.shape == (2, 2)
        output.logits.sum().backward()
        assert model.classifier.weight.grad is not None

    def test_forward_single_label_loss(self):
        """Integer labels produce a cross-entropy loss."""
        model = _build_classifier()

        output = model(input_ids=self._ids(), labels=torch.tensor([0, 1]))

        assert output.loss is not None
        expected = nn.functional.cross_entropy(output.logits, torch.tensor([0, 1]))
        assert torch.allclose(output.loss, expected)

    def test_forward_multilabel_loss(self):
        """Float labels produce a multi-label BCE loss."""
        model = _build_classifier()

        output = model(
            input_ids=self._ids(),
            labels=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        )

        expected = nn.functional.binary_cross_entropy_with_logits(
            output.logits, torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        assert torch.allclose(output.loss, expected)

    def test_forward_regression_single_label_loss(self):
        """num_labels=1 defaults to regression MSE."""
        model = _build_classifier(_make_config(num_labels=1))

        output = model(input_ids=self._ids(), labels=torch.tensor([[0.5], [1.5]]))

        expected = nn.functional.mse_loss(output.logits.squeeze(-1), torch.tensor([0.5, 1.5]))
        assert torch.allclose(output.loss, expected)

    def test_forward_multiregression_loss(self):
        """num_labels>1 with float labels defaults to multi-label BCE."""
        model = _build_classifier()

        output = model(input_ids=self._ids(), labels=torch.rand(2, 2))

        assert output.loss is not None

    def test_forward_unsupported_problem_type_raises(self):
        """A bogus configured problem_type raises ValueError."""
        model = _build_classifier()
        model.config.problem_type = "quantum"

        with pytest.raises(ValueError, match="Unsupported problem_type"):
            model(input_ids=self._ids(), labels=torch.tensor([0, 1]))

    def test_forward_remapped_tokenizer_ids(self):
        """Offset-space ids are remapped before the backbone sees them."""
        model = _build_classifier()

        output = model(input_ids=self._ids(offset_ids=True))

        assert output.logits.shape == (2, 2)

    def test_forward_tuple_output(self):
        """return_dict=False returns plain tuples."""
        model = _build_classifier()

        output = model(input_ids=self._ids(), return_dict=False)

        assert isinstance(output, tuple)
        assert output[0].shape == (2, 2)

    def test_forward_tuple_output_with_loss_and_hidden(self):
        """Tuple output carries loss first and hidden states when asked."""
        model = _build_classifier()

        output = model(
            input_ids=self._ids(),
            labels=torch.tensor([0, 1]),
            output_hidden_states=True,
            return_dict=False,
        )

        assert isinstance(output, tuple)
        assert len(output) == 3  # (loss, logits, hidden_output)
        assert output[0] is not None

    def test_forward_hidden_states_in_dict_output(self):
        """Dict output exposes hidden_states only when requested."""
        model = _build_classifier()

        visible = model(input_ids=self._ids(), output_hidden_states=True)
        hidden = model(input_ids=self._ids(), output_hidden_states=False)

        assert visible.hidden_states is not None
        assert hidden.hidden_states is None


class TestRegisterSequenceClassification:
    """_register_crossdna_sequence_classification registration contract."""

    def test_missing_auto_map_raises(self):
        """A config without auto_map raises ValueError."""
        config = SimpleNamespace(auto_map={})

        with pytest.raises(ValueError, match="does not provide auto_map"):
            _register_crossdna_sequence_classification(
                config, "/model", auto_model_for_sequence_classification=Mock()
            )

    def test_non_string_reference_raises(self):
        """A non-string AutoModelForMaskedLM reference raises ValueError."""
        config = SimpleNamespace(auto_map={"AutoModelForMaskedLM": 42})

        with pytest.raises(ValueError, match="does not provide auto_map"):
            _register_crossdna_sequence_classification(
                config, "/model", auto_model_for_sequence_classification=Mock()
            )

    def test_missing_register_api_raises(self):
        """An Auto class without .register raises TypeError."""
        config = SimpleNamespace(auto_map={"AutoModelForMaskedLM": "repo--Cls"})

        with patch(
            "dnallm.models.special.crossdna.get_class_from_dynamic_module",
            return_value=TinyCrossDNAForMaskedLM,
        ):
            with pytest.raises(TypeError, match="requires the Hugging Face"):
                _register_crossdna_sequence_classification(
                    config, "/model", auto_model_for_sequence_classification=object()
                )

    def test_successful_registration(self):
        """Registration binds the generated class via exist_ok=True."""
        config = SimpleNamespace(auto_map={"AutoModelForMaskedLM": "repo--Cls"})
        auto_cls = Mock()
        auto_cls.register = Mock()

        with patch(
            "dnallm.models.special.crossdna.get_class_from_dynamic_module",
            return_value=TinyCrossDNAForMaskedLM,
        ) as mock_dynamic:
            model_class = _register_crossdna_sequence_classification(config, "/model", auto_cls)

        mock_dynamic.assert_called_once_with("repo--Cls", "/model")
        assert model_class.__name__ == "CrossDNAForSequenceClassification"
        auto_cls.register.assert_called_once()
        assert auto_cls.register.call_args.kwargs["exist_ok"] is True


class TestProblemTypeFromTask:
    """_problem_type_from_task mapping."""

    @pytest.mark.parametrize(
        ("task_type", "expected"),
        [
            ("binary", "single_label_classification"),
            ("multiclass", "single_label_classification"),
            ("multilabel", "multi_label_classification"),
            ("regression", "regression"),
        ],
    )
    def test_task_mapping(self, task_type, expected):
        """Each supported task maps to its problem type."""
        assert _problem_type_from_task(task_type) == expected

    def test_unsupported_task_raises(self):
        """Unsupported tasks raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported CrossDNA sequence task"):
            _problem_type_from_task("generation")


def _modules():
    """The modules dict shape the handler consumes."""
    return {
        "AutoConfig": Mock(),
        "AutoTokenizer": Mock(),
        "AutoModelForMaskedLM": Mock(),
        "AutoModelForSequenceClassification": Mock(),
    }


class TestHandleCrossdnaModels:
    """_handle_crossdna_models dispatch contract."""

    def test_non_crossdna_path_returns_none_pair(self, tmp_path):
        """A path that is not a CrossDNA snapshot returns (None, None)."""
        model, tokenizer = _handle_crossdna_models("mask", str(tmp_path), 2, {}, {}, _modules())

        assert model is None
        assert tokenizer is None

    def test_head_config_rejected(self, tmp_path):
        """The adapter refuses a head_config."""
        checkpoint = _write_checkpoint(tmp_path)

        with pytest.raises(ValueError, match=r"Remove task\.head_config"):
            _handle_crossdna_models(
                "binary", checkpoint, 2, {}, {}, _modules(), head_config=SimpleNamespace()
            )

    def test_mask_task_loads_mlm(self, tmp_path):
        """mask tasks load AutoModelForMaskedLM directly."""
        checkpoint = _write_checkpoint(tmp_path)
        modules = _modules()
        mlm_model = Mock()
        modules["AutoModelForMaskedLM"].from_pretrained.return_value = mlm_model
        tokenizer_sentinel = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = tokenizer_sentinel

        model, tokenizer = _handle_crossdna_models("mask", checkpoint, 2, {}, {}, modules)

        assert model is mlm_model
        assert tokenizer is tokenizer_sentinel
        modules["AutoModelForMaskedLM"].from_pretrained.assert_called_once_with(
            checkpoint, trust_remote_code=True
        )

    def test_custom_tokenizer_factory(self, tmp_path):
        """A custom tokenizer callable replaces the fallback chain."""
        checkpoint = _write_checkpoint(tmp_path)
        custom_tokenizer = Mock(return_value="custom-tok")
        modules = _modules()
        modules["AutoModelForMaskedLM"].from_pretrained.return_value = Mock()

        _model, tokenizer = _handle_crossdna_models(
            "mask", checkpoint, 2, {}, {}, modules, custom_tokenizer=custom_tokenizer
        )

        assert tokenizer == "custom-tok"
        custom_tokenizer.assert_called_once_with()

    def test_unsupported_task_raises(self, tmp_path):
        """Non-sequence, non-mask tasks raise ValueError."""
        checkpoint = _write_checkpoint(tmp_path)

        with pytest.raises(ValueError, match="supports task types"):
            _handle_crossdna_models("generation", checkpoint, 2, {}, {}, _modules())

    def test_sequence_task_registers_and_loads(self, tmp_path):
        """Sequence tasks register the classifier and load via the Auto class."""
        checkpoint = _write_checkpoint(
            tmp_path, auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"}
        )
        modules = _modules()
        loaded_model = Mock()
        modules["AutoModelForSequenceClassification"].from_pretrained.return_value = loaded_model
        tokenizer_sentinel = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = tokenizer_sentinel
        loaded_config = _make_config(auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"})
        modules["AutoConfig"].from_pretrained.return_value = loaded_config

        with patch(
            "dnallm.models.special.crossdna.get_class_from_dynamic_module",
            return_value=TinyCrossDNAForMaskedLM,
        ):
            model, tokenizer = _handle_crossdna_models(
                "binary", checkpoint, 3, {0: "a", 1: "b", 2: "c"}, {"a": 0, "b": 1, "c": 2}, modules
            )

        assert model is loaded_model
        assert tokenizer is tokenizer_sentinel
        assert model._crossdna_checkpoint_dir == checkpoint
        # The in-memory config was decorated for the downstream classifier
        assert loaded_config.num_labels == 3
        register_call = modules["AutoModelForSequenceClassification"].register.call_args
        assert register_call.kwargs["exist_ok"] is True

    def test_sequence_task_via_8_1m_subdir(self, tmp_path):
        """A repo-root path resolves through the 8.1M checkpoint subdir."""
        checkpoint = _write_checkpoint(
            tmp_path / "8.1M", auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"}
        )
        modules = _modules()
        loaded_model = Mock()
        modules["AutoModelForSequenceClassification"].from_pretrained.return_value = loaded_model
        modules["AutoTokenizer"].from_pretrained.return_value = Mock()
        modules["AutoConfig"].from_pretrained.return_value = _make_config(
            auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"}
        )

        with patch(
            "dnallm.models.special.crossdna.get_class_from_dynamic_module",
            return_value=TinyCrossDNAForMaskedLM,
        ):
            model, _tokenizer = _handle_crossdna_models(
                "regression", str(tmp_path), 1, {}, {}, modules
            )

        assert model is loaded_model
        assert model._crossdna_checkpoint_dir == checkpoint
