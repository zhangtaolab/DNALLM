"""Grouped coverage for the remaining special-family handlers.

Families whose dependencies are absent get sys.modules stubs injected via
monkeypatch.setitem (auto-restored — never a direct sys.modules assignment);
families importing only installed packages run against real modules.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from dnallm.models.special import __all__ as special_exports
from dnallm.models.special import (
    _handle_basenji2_tokenizer,
    _handle_borzoi_models,
    _handle_dnabert2_models,
    _handle_enformer_models,
    _handle_gpn_models,
    _handle_lucaone_models,
    _handle_megadna_models,
    _handle_mutbert_tokenizer,
    _handle_omnidna_models,
    _handle_space_models,
)
from dnallm.models.head import MegaDNAMultiScaleHead
from dnallm.models.tokenizer import DNAOneHotTokenizer


class TestGpnHandler:
    """GPN import-availability gate."""

    def test_non_matching_name_returns_none(self):
        """Non-GPN names fall through with None."""
        assert _handle_gpn_models("bert-base") is None

    def test_matching_name_without_package_raises(self):
        """A GPN name without the package raises the instructive ImportError."""
        with pytest.raises(ImportError, match="gpn package is required"):
            _handle_gpn_models("gpn-brassicales")

    def test_matching_name_with_package_returns_match(self, monkeypatch):
        """With gpn importable the gate returns the matched registry name."""
        fake_gpn = types.ModuleType("gpn")
        fake_model = types.ModuleType("gpn.model")
        fake_gpn.model = fake_model
        monkeypatch.setitem(sys.modules, "gpn", fake_gpn)
        monkeypatch.setitem(sys.modules, "gpn.model", fake_model)

        assert _handle_gpn_models("gpn-animal-promoter-x") == "gpn-animal-promoter"

    def test_extra_extends_the_registry(self, monkeypatch):
        """The extra kwarg registers an additional known name."""
        fake_gpn = types.ModuleType("gpn")
        fake_model = types.ModuleType("gpn.model")
        fake_gpn.model = fake_model
        monkeypatch.setitem(sys.modules, "gpn", fake_gpn)
        monkeypatch.setitem(sys.modules, "gpn.model", fake_model)

        assert _handle_gpn_models("my-custom-gpn", extra="my-custom-gpn") == "my-custom-gpn"


class TestOmnidnaHandler:
    """Omni-DNA import-availability gate."""

    def test_non_matching_name_returns_none(self):
        """Non-Omni-DNA names fall through with None."""
        assert _handle_omnidna_models("bert-base") is None

    def test_matching_name_without_package_raises(self):
        """An Omni-DNA name without ai2-olmo raises the instructive ImportError."""
        with pytest.raises(ImportError, match="ai2-olmo package is required"):
            _handle_omnidna_models("Omni-DNA-60M")

    def test_matching_name_with_package_returns_match(self, monkeypatch):
        """With olmo importable the gate returns the matched registry name."""
        fake_version = types.ModuleType("olmo.version")
        fake_version.VERSION = "1.2.3"
        fake_olmo = types.ModuleType("olmo")
        fake_olmo.version = fake_version
        monkeypatch.setitem(sys.modules, "olmo", fake_olmo)
        monkeypatch.setitem(sys.modules, "olmo.version", fake_version)

        assert _handle_omnidna_models("Omni-DNA-300M-finetuned") == "Omni-DNA-300M"


class TestBasenji2Tokenizer:
    """The basenji2 tokenizer post-processor."""

    def test_returns_one_hot_embeds_tokenizer(self):
        """The original tokenizer is replaced by the transposed one-hot tokenizer."""
        result = _handle_basenji2_tokenizer(Mock())

        assert isinstance(result, DNAOneHotTokenizer)
        assert result.return_embeds is True
        assert result.embeds_transpose is True


class FakeMutBertInner:
    """Minimal HF-tokenizer stand-in for the MutBERT wrapper."""

    def __len__(self):
        return 7

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[0, 3, 1]])}


class TestMutbertTokenizer:
    """The MutBERT one-hot tokenizer wrapper."""

    def test_vocab_size_from_inner_tokenizer(self):
        """vocab_size mirrors len(inner tokenizer)."""
        wrapper = _handle_mutbert_tokenizer(FakeMutBertInner())

        assert wrapper.vocab_size == 7

    def test_call_returns_one_hot_input_ids(self):
        """Calling the wrapper one-hot encodes input_ids to float."""
        wrapper = _handle_mutbert_tokenizer(FakeMutBertInner())

        encoding = wrapper("ATC", padding=True)

        assert encoding["input_ids"].shape == (1, 3, 7)
        assert encoding["input_ids"].dtype == torch.float32
        assert torch.equal(encoding["input_ids"][0, 0], torch.tensor([1.0, 0, 0, 0, 0, 0, 0]))

    def test_attribute_passthrough(self):
        """Unknown attributes delegate to the wrapped tokenizer."""
        inner = FakeMutBertInner()
        inner.pad_token = "[PAD]"  # ruff: ignore[hardcoded-password-string]
        wrapper = _handle_mutbert_tokenizer(inner)

        assert wrapper.pad_token == "[PAD]"  # ruff: ignore[hardcoded-password-string]


def _patch_model_resolution(downloaded="/downloaded/model"):
    """Patch the shared source resolver the handlers call."""
    return patch(
        "dnallm.models.model._get_model_path_and_imports",
        return_value=(downloaded, None),
    )


class TestEnformerHandler:
    """The enformer handler against the vendored (coverage-omitted) models."""

    def test_non_matching_name_returns_none(self):
        """Non-enformer names fall through."""
        assert _handle_enformer_models("bert", "local", "binary", 2) is None

    def test_sequence_task_loads_classification_model(self):
        """Sequence tasks build EnformerForSequenceClassification."""
        with (
            _patch_model_resolution(),
            patch(
                "dnallm.models.special.enformer_model.configuration_enformer."
                "EnformerConfig.from_pretrained",
                return_value=Mock(),
            ) as mock_config,
            patch(
                "dnallm.models.special.enformer_model.modeling_enformer."
                "EnformerForSequenceClassification.from_pretrained",
                return_value="cls-model",
            ) as mock_cls_load,
        ):
            model, tokenizer = _handle_enformer_models(
                "enformer-official-rough", "local", "binary", 2
            )

        assert model == "cls-model"
        assert isinstance(tokenizer, DNAOneHotTokenizer)
        config_path = mock_config.call_args[0][0]
        assert config_path == "/downloaded/model/config.json"
        assert mock_cls_load.call_args.kwargs["config"] is mock_config.return_value
        assert mock_config.return_value.num_labels == 2

    def test_non_sequence_task_loads_base_model(self):
        """Non-sequence tasks build the base enformer via from_pretrained."""
        with (
            _patch_model_resolution(),
            patch(
                "dnallm.models.special.enformer_model.configuration_enformer."
                "EnformerConfig.from_pretrained",
                return_value=Mock(),
            ),
            patch(
                "dnallm.models.special.enformer_model.modeling_enformer.from_pretrained",
                return_value="base-model",
            ) as mock_base_load,
        ):
            model, tokenizer = _handle_enformer_models("enformer-191k", "local", "mask", 0)

        assert model == "base-model"
        assert isinstance(tokenizer, DNAOneHotTokenizer)
        assert mock_base_load.call_args.kwargs["config"] is not None

    def test_extra_extends_the_registry(self):
        """The extra kwarg registers an additional enformer name."""
        with (
            _patch_model_resolution(),
            patch(
                "dnallm.models.special.enformer_model.configuration_enformer."
                "EnformerConfig.from_pretrained",
                return_value=Mock(),
            ),
            patch(
                "dnallm.models.special.enformer_model.modeling_enformer."
                "EnformerForSequenceClassification.from_pretrained",
                return_value="cls-model",
            ),
        ):
            model, _tokenizer = _handle_enformer_models(
                "my-enformer", "local", "binary", 2, extra="my-enformer"
            )

        assert model == "cls-model"


class TestSpaceHandler:
    """The SPACE handler against the vendored (coverage-omitted) models."""

    def test_non_matching_name_returns_none(self):
        """Non-SPACE names fall through."""
        assert _handle_space_models("bert", "local", "binary", 2) is None

    def test_sequence_task_loads_classification_model(self):
        """Sequence tasks build SpaceForSequenceClassification."""
        with (
            _patch_model_resolution("/downloaded/space"),
            patch(
                "dnallm.models.special.enformer_model.configuration_space."
                "SpaceConfig.from_pretrained",
                return_value=Mock(),
            ) as mock_config,
            patch(
                "dnallm.models.special.enformer_model.modeling_space."
                "SpaceForSequenceClassification.from_pretrained",
                return_value="space-cls",
            ) as mock_cls_load,
        ):
            model, tokenizer = _handle_space_models("SPACE", "local", "multilabel", 3)

        assert model == "space-cls"
        assert isinstance(tokenizer, DNAOneHotTokenizer)
        assert mock_config.return_value.num_labels == 3
        assert mock_cls_load.call_args.kwargs["config"] is mock_config.return_value

    def test_non_sequence_task_loads_base_space(self):
        """Non-sequence tasks build the base Space model."""
        with (
            _patch_model_resolution("/downloaded/space"),
            patch(
                "dnallm.models.special.enformer_model.configuration_space."
                "SpaceConfig.from_pretrained",
                return_value=Mock(),
            ),
            patch(
                "dnallm.models.special.enformer_model.modeling_space.Space.from_pretrained",
                return_value="space-base",
            ) as mock_base_load,
        ):
            model, _tokenizer = _handle_space_models("SPACE", "local", "generation", 0)

        assert model == "space-base"
        assert mock_base_load.call_args.kwargs["config"] is not None


class TestMegadnaHandler:
    """The megaDNA handler (torch.load over a real tmp checkpoint)."""

    def test_non_matching_name_returns_none(self):
        """Non-megaDNA names fall through."""
        assert _handle_megadna_models("bert-base", "local", None) is None

    @pytest.mark.parametrize(
        ("model_name", "expected_file"),
        [
            ("megaDNA_updated", "megaDNA_phage_145M.pt"),
            ("megaDNA_variants", "megaDNA_phage_78M.pt"),
            ("megaDNA_finetuned", "megaDNA_phage_ecoli_finetuned.pt"),
            ("megaDNA_phage_145M", "megaDNA_phage_145M.pt"),
        ],
    )
    def test_checkpoint_file_selection(self, tmp_path, model_name, expected_file):
        """Each registry name maps onto its pretrained checkpoint file."""
        loaded = {"sentinel": model_name}
        (tmp_path / expected_file).write_bytes(b"")

        with (
            patch("torch.load", return_value=loaded) as mock_load,
            _patch_model_resolution(str(tmp_path)),
        ):
            model, tokenizer = _handle_megadna_models(model_name, "local", None)

        assert model is loaded
        assert mock_load.call_args[0][0] == str(tmp_path / expected_file)
        assert tokenizer is not None
        assert tokenizer.vocab_size == 6

    def test_head_config_wraps_in_sequence_classifier(self, tmp_path):
        """A head_config wraps the loaded checkpoint in the classification wrapper."""
        (tmp_path / "megaDNA_phage_145M.pt").write_bytes(b"")

        head_config = SimpleNamespace(head="megadna", num_classes=2, task_type="binary")

        with (
            patch("torch.load", return_value=SimpleNamespace()),
            _patch_model_resolution(str(tmp_path)),
        ):
            model, tokenizer = _handle_megadna_models("megaDNA_phage_145M", "local", head_config)

        from dnallm.models.model import DNALLMforSequenceClassification

        assert isinstance(model, DNALLMforSequenceClassification)
        assert isinstance(model.score, MegaDNAMultiScaleHead)
        assert tokenizer.vocab_size == 6


class TestBorzoiHandler:
    """The borzoi handler with borzoi_pytorch stubbed."""

    def test_non_matching_name_returns_none(self):
        """Non-borzoi names fall through."""
        assert _handle_borzoi_models("bert", "local", "binary", 2) is None

    def test_matching_name_without_package_raises(self):
        """A borzoi name without borzoi_pytorch raises the instructive ImportError."""
        with pytest.raises(ImportError, match="borzoi_pytorch package is required"):
            _handle_borzoi_models("borzoi-replicate-0", "local", "binary", 2)

    def _install_stubs(self, monkeypatch):
        """Inject the borzoi_pytorch stubs (auto-restored)."""
        fake_borzoi = types.ModuleType("borzoi_pytorch")
        fake_config_module = types.ModuleType("borzoi_pytorch.config_borzoi")

        class FakeBorzoi:
            from_pretrained = Mock(return_value="base-borzoi")

        fake_borzoi.Borzoi = FakeBorzoi
        fake_config_module.BorzoiConfig = Mock()
        fake_config_module.BorzoiConfig.from_pretrained = Mock(return_value=Mock())
        monkeypatch.setitem(sys.modules, "borzoi_pytorch", fake_borzoi)
        monkeypatch.setitem(sys.modules, "borzoi_pytorch.config_borzoi", fake_config_module)
        return fake_borzoi, fake_config_module

    def test_sequence_task_builds_classification_class(self, monkeypatch):
        """Sequence tasks subclass Borzoi and load via the classification path."""
        fake_borzoi, fake_config_module = self._install_stubs(monkeypatch)
        fake_cls_load = Mock(return_value="cls-borzoi")
        fake_borzoi.Borzoi.from_pretrained = staticmethod(fake_cls_load)

        with _patch_model_resolution("/downloaded/borzoi"):
            model, tokenizer = _handle_borzoi_models(
                "borzoi-replicate-1-mouse", "local", "binary", 2
            )

        assert model == "cls-borzoi"
        assert fake_config_module.BorzoiConfig.from_pretrained.called
        assert fake_config_module.BorzoiConfig.from_pretrained.return_value.num_labels == 2
        load_args = fake_cls_load.call_args
        assert load_args.args[0] == "/downloaded/borzoi"
        assert isinstance(tokenizer, DNAOneHotTokenizer)
        assert tokenizer.embeds_transpose is True

    def test_non_sequence_task_loads_base_borzoi(self, monkeypatch):
        """Non-sequence tasks load the base Borzoi model."""
        fake_borzoi, _fake_config = self._install_stubs(monkeypatch)

        with _patch_model_resolution("/downloaded/borzoi"):
            model, _tokenizer = _handle_borzoi_models("borzoi-replicate-2", "local", "mask", 0)

        assert model == "base-borzoi"
        fake_borzoi.Borzoi.from_pretrained.assert_called_once()

    def test_flashzoi_names_matched(self, monkeypatch):
        """flashzoi replicate names take the same handler path."""
        _fake_borzoi, _fake_config = self._install_stubs(monkeypatch)

        with _patch_model_resolution("/downloaded/borzoi"):
            model, _tokenizer = _handle_borzoi_models("flashzoi-replicate-3", "local", "mask", 0)

        assert model == "base-borzoi"


class TestDnabert2Handler:
    """The DNABERT-2 triton workaround handler."""

    def test_unrelated_path_returns_none_pair(self):
        """Non-DNABERT-2 paths return (None, None) untouched."""
        model, tokenizer = _handle_dnabert2_models("/models/other-model", [])

        assert model is None
        assert tokenizer is None

    def test_dnabert2_path_loads_and_restores_triton_file(self, tmp_path):
        """The disabled flash_attn_triton.py is restored after loading."""
        model_dir = tmp_path / "dnabert-2"
        model_dir.mkdir()
        original_content = "def attention():\n    pass\n"
        triton_file = model_dir / "flash_attn_triton.py"
        triton_file.write_text(original_content)
        load_args = ["mask", str(model_dir), 0, {}, {}, {"AutoTokenizer": Mock()}]

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            return_value=("db2-model", "db2-tokenizer"),
        ) as mock_load:
            model, tokenizer = _handle_dnabert2_models(str(model_dir), load_args)

        assert model == "db2-model"
        assert tokenizer == "db2-tokenizer"
        mock_load.assert_called_once_with(*load_args)
        # Whatever the triton probe decided, the original file content survives.
        assert triton_file.read_text() == original_content

    def test_dnabert2_without_triton_file(self, tmp_path):
        """A checkpoint without flash_attn_triton.py loads without touching disk."""
        model_dir = tmp_path / "dnabert-s"
        model_dir.mkdir()
        load_args = ["mask", str(model_dir), 0, {}, {}, {"AutoTokenizer": Mock()}]

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            return_value=("dbs-model", "dbs-tokenizer"),
        ):
            model, tokenizer = _handle_dnabert2_models(str(model_dir), load_args)

        assert model == "dbs-model"
        assert tokenizer == "dbs-tokenizer"


class TestLucaoneHandler:
    """The LucaOne handler with lucagplm stubbed."""

    def test_non_matching_name_returns_none(self):
        """Non-LucaOne names fall through."""
        assert _handle_lucaone_models("bert-base", "local", None) is None

    def test_matching_name_without_package_raises(self):
        """A LucaOne name without lucagplm raises the instructive ImportError."""
        with pytest.raises(ImportError, match="lucagplm package is required"):
            _handle_lucaone_models("LucaOne-default-step5.6M", "local", None)

    def _install_stubs(self, monkeypatch):
        """Inject the lucagplm stubs (auto-restored)."""
        fake_module = types.ModuleType("lucagplm")
        fake_module.LucaGPLMModel = Mock()
        fake_module.LucaGPLMModel.from_pretrained.return_value = SimpleNamespace()
        fake_module.LucaGPLMTokenizer = Mock()
        fake_module.LucaGPLMTokenizer.from_pretrained.return_value = Mock(pad_token_id=7)
        fake_module.LucaGPLMConfig = Mock()
        fake_module.LucaGPLMConfig.from_pretrained.return_value = Mock()
        monkeypatch.setitem(sys.modules, "lucagplm", fake_module)
        return fake_module

    def test_loads_model_and_tokenizer(self, monkeypatch):
        """Without a head_config the raw model/tokenizer pair is returned."""
        fake_module = self._install_stubs(monkeypatch)
        luca_model = fake_module.LucaGPLMModel.from_pretrained.return_value
        luca_tokenizer = fake_module.LucaGPLMTokenizer.from_pretrained.return_value

        with _patch_model_resolution("/downloaded/lucaone"):
            model, tokenizer = _handle_lucaone_models("LucaOne-gene-step36.8M", "local", None)

        assert model is luca_model
        assert tokenizer is luca_tokenizer
        fake_module.LucaGPLMModel.from_pretrained.assert_called_once_with("/downloaded/lucaone")

    def test_head_config_uses_from_base_model(self, monkeypatch):
        """A head_config routes through DNALLMforSequenceClassification.from_base_model."""
        fake_module = self._install_stubs(monkeypatch)

        head_config = SimpleNamespace(head="lucaone-mlp", num_classes=2)

        with (
            _patch_model_resolution("/downloaded/lucaone"),
            patch("dnallm.models.model.DNALLMforSequenceClassification") as wrapper_cls,
        ):
            wrapper_cls.from_base_model.return_value = "wrapped-luca"
            model, tokenizer = _handle_lucaone_models(
                "LucaOne-default-step17.6M", "local", head_config
            )

        assert model == "wrapped-luca"
        assert tokenizer is fake_module.LucaGPLMTokenizer.from_pretrained.return_value
        wrapper_cls.from_base_model.assert_called_once_with(
            "/downloaded/lucaone",
            config=fake_module.LucaGPLMConfig.from_pretrained.return_value,
            module=fake_module.LucaGPLMModel,
        )


class TestSpecialExports:
    """The special package re-export surface."""

    def test_all_handlers_importable(self):
        """Every declared export resolves to a callable in the package."""
        import dnallm.models.special as special_package

        for name in special_exports:
            attribute = getattr(special_package, name, None)
            assert callable(attribute), f"{name} must be callable"
