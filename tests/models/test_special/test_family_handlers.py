"""Grouped coverage for the remaining special-family handlers.

Families whose dependencies are absent get sys.modules stubs injected via
monkeypatch.setitem (auto-restored — never a direct sys.modules assignment);
families importing only installed packages run against real modules.
"""

import os
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
        """A GPN name without the package raises the instructive ImportError.

        D-10: the guidance is exact and measured -- the literal install
        command, the inspected wheel's Python and transformers constraints,
        the separate-environment recommendation, the honest warning that the
        published wheel does not even provide the handler's gpn.model
        import, and the optional-dependency guide link.
        """
        with pytest.raises(ImportError, match="gpn package is required") as excinfo:
            _handle_gpn_models("gpn-brassicales")
        message = str(excinfo.value)
        assert "pip install gpn" in message
        assert "3.13" in message  # gpn 0.9.0 wheel: >=3.13,<3.14 only
        assert "transformers==5.15.0" in message  # the wheel's hard pin
        assert "separate environment" in message
        assert "gpn.model" in message  # the published wheel lacks this module
        assert "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/" in message
        assert excinfo.value.__cause__ is not None  # original ImportError preserved

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
        """An Omni-DNA name without ai2-olmo raises the instructive ImportError.

        D-10: the guidance names the literal install command, the measured
        numpy<2 pin (which can downgrade a numpy 2 environment), the
        separate-environment recommendation, and the guide link.
        """
        with pytest.raises(ImportError, match="ai2-olmo package is required") as excinfo:
            _handle_omnidna_models("Omni-DNA-60M")
        message = str(excinfo.value)
        assert "pip install ai2-olmo" in message
        assert "numpy<2" in message  # ai2-olmo 0.6.0 pin
        assert "downgrade" in message  # the hazard of installing into a numpy-2 env
        assert "separate environment" in message
        assert "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/" in message
        assert excinfo.value.__cause__ is not None  # original ImportError preserved

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
        assert config_path == os.path.join("/downloaded/model", "config.json")
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

    def _dnabert2_dir_with_triton_file(self, tmp_path):
        """Build a DNABERT-2 checkpoint dir containing flash_attn_triton.py."""
        model_dir = tmp_path / "dnabert-2"
        model_dir.mkdir()
        original_content = "def attention():\n    pass\n"
        triton_file = model_dir / "flash_attn_triton.py"
        triton_file.write_text(original_content)
        load_args = ["mask", str(model_dir), 0, {}, {}, {"AutoTokenizer": Mock()}]
        return model_dir, triton_file, original_content, load_args

    def test_working_triton_probe_disables_file_and_restores(self, tmp_path, monkeypatch):
        """A triton where tl.dot(trans_b=True) runs takes the disable/restore path."""
        fake_tl = types.ModuleType("triton.language")
        fake_tl.float32 = "f32"
        fake_tl.zeros = lambda shape, dtype=None: (shape, dtype)
        fake_tl.dot = lambda *args, **kwargs: None
        fake_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", fake_triton)
        monkeypatch.setitem(sys.modules, "triton.language", fake_tl)

        _, triton_file, original_content, load_args = self._dnabert2_dir_with_triton_file(tmp_path)
        seen_during_load = {}

        def capture_load(*args):
            seen_during_load["content"] = triton_file.read_text()
            return ("m", "t")

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            side_effect=capture_load,
        ):
            model, tokenizer = _handle_dnabert2_models(str(load_args[1]), load_args)

        assert (model, tokenizer) == ("m", "t")
        # The disable marker was live while the model loaded...
        assert "Temporarily disabled" in seen_during_load["content"]
        # ...and the original content is restored afterwards.
        assert triton_file.read_text() == original_content

    def test_typeerror_triton_probe_leaves_file_untouched(self, tmp_path, monkeypatch):
        """tl.dot raising TypeError (unsupported trans_b) means no disable."""
        fake_tl = types.ModuleType("triton.language")
        fake_tl.float32 = "f32"
        fake_tl.zeros = lambda shape, dtype=None: (shape, dtype)

        def dot_without_trans_b(*args, **kwargs):
            raise TypeError("dot() got an unexpected keyword argument 'trans_b'")

        fake_tl.dot = dot_without_trans_b
        fake_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", fake_triton)
        monkeypatch.setitem(sys.modules, "triton.language", fake_tl)

        _, triton_file, original_content, load_args = self._dnabert2_dir_with_triton_file(tmp_path)
        seen_during_load = {}

        def capture_load(*args):
            seen_during_load["content"] = triton_file.read_text()
            return ("m", "t")

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            side_effect=capture_load,
        ):
            model, tokenizer = _handle_dnabert2_models(str(load_args[1]), load_args)

        assert (model, tokenizer) == ("m", "t")
        # Unsupported trans_b -> no disable: the file is never modified.
        assert seen_during_load["content"] == original_content
        assert triton_file.read_text() == original_content

    def test_failing_triton_probe_treated_as_supported(self, tmp_path, monkeypatch):
        """A non-TypeError triton failure also takes the disable/restore path."""
        fake_tl = types.ModuleType("triton.language")
        fake_tl.float32 = "f32"

        def broken_zeros(shape, dtype=None):
            raise RuntimeError("triton needs an active CUDA context")

        fake_tl.zeros = broken_zeros
        fake_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", fake_triton)
        monkeypatch.setitem(sys.modules, "triton.language", fake_tl)

        _, triton_file, original_content, load_args = self._dnabert2_dir_with_triton_file(tmp_path)

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            return_value=("m", "t"),
        ):
            model, tokenizer = _handle_dnabert2_models(str(load_args[1]), load_args)

        assert (model, tokenizer) == ("m", "t")
        assert triton_file.read_text() == original_content


class TestDnabert2Tf5Shims:
    """D-05: the idempotent pre-construction DNABERT-2 transformers-5 installer.

    The measured defect fires inside ``BertEncoder.__init__`` (the remote
    ``rebuild_alibi_tensor`` multiplies a meta-default-device ``torch.arange``
    by a legacy cpu ``torch.Tensor(list)``), so the repair must be installed
    on the remote class BEFORE any model constructor runs -- the seam is
    ``transformers.dynamic_module_utils.get_class_in_module``, which the
    resolver machinery looks up as a module global at call time.
    """

    # A synthetic remote bert_layers.py: the mosaic shape (BertEncoder with
    # rebuild_alibi_tensor + BertModel) the installer narrowly detects.
    REMOTE_SOURCE = (
        "class BertEncoder:\n"
        "    def rebuild_alibi_tensor(self, size, device=None):\n"
        "        return (size, device)\n"
        "\n"
        "\n"
        "class BertModel:\n"
        "    def __init__(self):\n"
        "        self.encoder = BertEncoder()\n"
    )
    REMOTE_SOURCE_WITH_FLASH = REMOTE_SOURCE + (
        "\nflash_attn_qkvpacked_func = lambda *args, **kwargs: None\n"
    )

    def _resolve_remote(
        self,
        tmp_path,
        monkeypatch,
        sub: str,
        class_name: str = "BertModel",
        source: str | None = None,
    ):
        """Resolve a class from a synthetic remote module under a fake modules cache."""
        import dnallm.models.special.dnabert2 as db2
        import transformers.dynamic_module_utils as dmu

        db2._install_dnabert2_tf5_shims()
        remote_dir = tmp_path / sub
        remote_dir.mkdir()
        (remote_dir / "bert_layers.py").write_text(source or self.REMOTE_SOURCE)
        monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(tmp_path))
        resolved = dmu.get_class_in_module(class_name, f"{sub}/bert_layers.py")
        module = sys.modules[resolved.__module__]
        return resolved, module

    def test_module_level_triton_probe_keeps_the_measured_mapping(self, monkeypatch):
        """The lifted probe preserves the handler's measured semantics:
        TypeError (no trans_b support) -> False; a working tl.dot -> True;
        any other triton failure -> True (the disable path)."""
        import dnallm.models.special.dnabert2 as db2

        fake_tl = types.ModuleType("triton.language")
        fake_tl.float32 = "f32"
        fake_tl.zeros = lambda shape, dtype=None: (shape, dtype)
        fake_tl.dot = lambda *args, **kwargs: None
        fake_triton = types.ModuleType("triton")
        monkeypatch.setitem(sys.modules, "triton", fake_triton)
        monkeypatch.setitem(sys.modules, "triton.language", fake_tl)

        assert db2._triton_supports_trans_b() is True

        def dot_without_trans_b(*args, **kwargs):
            raise TypeError("dot() got an unexpected keyword argument 'trans_b'")

        fake_tl.dot = dot_without_trans_b
        assert db2._triton_supports_trans_b() is False

        def broken_zeros(shape, dtype=None):
            raise RuntimeError("triton needs an active CUDA context")

        fake_tl.dot = lambda *args, **kwargs: None
        fake_tl.zeros = broken_zeros
        assert db2._triton_supports_trans_b() is True

    def test_wrap_pins_cpu_only_when_device_is_missing(self):
        """The ALiBi wrapper changes only device=None; an explicit device is
        forwarded unchanged (positional or keyword)."""
        import dnallm.models.special.dnabert2 as db2

        calls = []

        class Encoder:
            def rebuild_alibi_tensor(self, size, device=None):
                calls.append((size, device))
                return "ok"

        assert db2._wrap_rebuild_alibi_tensor(Encoder) is True

        encoder = Encoder()
        assert encoder.rebuild_alibi_tensor(4) == "ok"
        assert calls == [(4, "cpu")]
        assert encoder.rebuild_alibi_tensor(4, device="cuda") == "ok"
        assert calls[-1] == (4, "cuda")
        assert encoder.rebuild_alibi_tensor(5, "cuda:0") == "ok"
        assert calls[-1] == (5, "cuda:0")

    def test_wrap_survives_the_meta_default_device_context(self):
        """The measured crash shape: torch.arange honors the ambient meta
        default device while the legacy torch.Tensor(list) constructor
        ignores it; the pinned cpu makes both operands agree."""
        import dnallm.models.special.dnabert2 as db2

        class MetaCrashEncoder(torch.nn.Module):
            def rebuild_alibi_tensor(self, size, device=None):
                context = torch.arange(size, device=device)
                slopes = torch.Tensor([0.5]).to(device)
                return slopes.unsqueeze(1) * context.to(slopes.dtype)

        with torch.device("meta"):
            with pytest.raises(RuntimeError):
                MetaCrashEncoder().rebuild_alibi_tensor(4)

        assert db2._wrap_rebuild_alibi_tensor(MetaCrashEncoder) is True
        with torch.device("meta"):
            result = MetaCrashEncoder().rebuild_alibi_tensor(4)
        assert result.device.type == "cpu"

    def test_repeat_wrap_preserves_wrapper_identity(self):
        """A second wrap call keeps the first wrapper object (no double-wrap)."""
        import dnallm.models.special.dnabert2 as db2

        class Encoder:
            def rebuild_alibi_tensor(self, size, device=None):
                return (size, device)

        assert db2._wrap_rebuild_alibi_tensor(Encoder) is True
        wrapped = Encoder.__dict__["rebuild_alibi_tensor"]
        assert db2._wrap_rebuild_alibi_tensor(Encoder) is False
        assert Encoder.__dict__["rebuild_alibi_tensor"] is wrapped

    def test_class_without_or_inheriting_the_method_is_untouched(self):
        """A class without rebuild_alibi_tensor in its own __dict__ (absent or
        only inherited) is never wrapped -- the wrapper never re-wraps a
        binding it cannot capture."""
        import dnallm.models.special.dnabert2 as db2

        class Plain:
            pass

        class Base:
            def rebuild_alibi_tensor(self, size, device=None):
                return (size, device)

        class Child(Base):
            pass

        assert db2._wrap_rebuild_alibi_tensor(Plain) is False
        assert db2._wrap_rebuild_alibi_tensor(Child) is False

    def test_installer_arms_only_on_transformers_5(self, monkeypatch):
        """Below transformers 5 the installer never arms the dynamic-class
        hook (D-08 version gate for the construction-path class)."""
        import transformers

        import dnallm.models.special.dnabert2 as db2

        armed = []
        monkeypatch.setattr(
            db2, "_arm_dnabert2_dynamic_class_shims", lambda dmu: armed.append(dmu) or True
        )
        monkeypatch.setattr(transformers, "__version__", "4.49.0")
        db2._install_dnabert2_tf5_shims()
        assert armed == []

        monkeypatch.setattr(transformers, "__version__", "5.17.0")
        db2._install_dnabert2_tf5_shims()
        assert len(armed) == 1

    def test_unparseable_version_never_arms(self):
        """A version string that does not parse reads as major 0 (below 5):
        the gate fails closed, never arms on an unknown shape."""
        import dnallm.models.special.dnabert2 as db2

        assert db2._transformers_major_version("5.17.0") == 5
        assert db2._transformers_major_version("4.49.0") == 4
        assert db2._transformers_major_version("not-a-version") == 0
        assert db2._transformers_major_version("") == 0
        assert db2._transformers_major_version(None) == 0  # type: ignore[arg-type]

    def test_arm_skips_when_the_lookup_symbol_is_absent(self):
        """A dynamic_module_utils without get_class_in_module is left alone."""
        import dnallm.models.special.dnabert2 as db2

        fake_dmu = types.ModuleType("fake_dynamic_module_utils")

        assert db2._arm_dnabert2_dynamic_class_shims(fake_dmu) is False
        assert not hasattr(fake_dmu, "_dnallm_dnabert2_shim_patch")

    def test_resolver_wraps_encoder_before_construction(self, tmp_path, monkeypatch):
        """A DNABERT-2-shaped remote module resolved through the wrapped
        lookup has its BertEncoder wrapped BEFORE any constructor runs."""
        resolved, module = self._resolve_remote(tmp_path, monkeypatch, "dnabert2_shim_a")

        assert resolved.__name__ == "BertModel"
        encoder_cls = module.BertEncoder
        assert getattr(encoder_cls, "_dnallm_alibi_device_patch", False) is True

        encoder = encoder_cls()
        assert encoder.rebuild_alibi_tensor(4) == (4, "cpu")
        assert encoder.rebuild_alibi_tensor(4, device="cuda") == (4, "cuda")

    def test_unrelated_remote_module_is_untouched(self, tmp_path, monkeypatch):
        """A module without the BertEncoder shape resolves unchanged."""
        source = "class PlainModel:\n    marker = 'untouched'\n"
        resolved, _module = self._resolve_remote(
            tmp_path, monkeypatch, "dnabert2_shim_b", class_name="PlainModel", source=source
        )

        assert resolved.marker == "untouched"
        assert not hasattr(resolved, "_dnallm_alibi_device_patch")

    def test_repeat_install_preserves_the_resolver_wrapper(self):
        """A second installer call keeps the same wrapped resolver."""
        import dnallm.models.special.dnabert2 as db2
        import transformers.dynamic_module_utils as dmu

        db2._install_dnabert2_tf5_shims()
        wrapper = dmu.get_class_in_module
        assert getattr(dmu, "_dnallm_dnabert2_shim_patch", False) is True

        db2._install_dnabert2_tf5_shims()

        assert dmu.get_class_in_module is wrapper

    def test_triton_fallback_nulls_the_remote_flash_symbol(self, tmp_path, monkeypatch):
        """When the capability probe selects the existing fallback, the same
        narrowly detected installer nulls the remote module's
        flash_attn_qkvpacked_func (the remote code's own None branch routes
        attention to its PyTorch path)."""
        import dnallm.models.special.dnabert2 as db2

        monkeypatch.setattr(db2, "_triton_supports_trans_b", lambda: True)
        _resolved, module = self._resolve_remote(
            tmp_path, monkeypatch, "dnabert2_shim_c", source=self.REMOTE_SOURCE_WITH_FLASH
        )

        assert module.flash_attn_qkvpacked_func is None

    def test_working_triton_keeps_the_remote_flash_symbol(self, tmp_path, monkeypatch):
        """When the probe finds a compatible triton, the remote symbol stays."""
        import dnallm.models.special.dnabert2 as db2

        monkeypatch.setattr(db2, "_triton_supports_trans_b", lambda: False)
        _resolved, module = self._resolve_remote(
            tmp_path, monkeypatch, "dnabert2_shim_d", source=self.REMOTE_SOURCE_WITH_FLASH
        )

        assert module.flash_attn_qkvpacked_func is not None

    def test_handler_installs_shims_before_loading(self, tmp_path, monkeypatch):
        """The product path arms the installer before the delegated load."""
        import dnallm.models.special.dnabert2 as db2

        calls = []
        monkeypatch.setattr(db2, "_install_dnabert2_tf5_shims", lambda: calls.append("install"))
        model_dir = tmp_path / "dnabert-2"
        model_dir.mkdir()
        load_args = ["mask", str(model_dir), 0, {}, {}, {"AutoTokenizer": Mock()}]

        def record_load(*args):
            calls.append("load")
            return ("m", "t")

        with patch(
            "dnallm.models.model._load_model_by_task_type",
            side_effect=record_load,
        ):
            model, tokenizer = _handle_dnabert2_models(str(model_dir), load_args)

        assert (model, tokenizer) == ("m", "t")
        assert calls == ["install", "load"], "the installer must run before the load"


class TestHandlerExtraRegistration:
    """The `extra` registration arms of the list-gated handlers."""

    def test_lucaone_extra_appended_to_registry(self, monkeypatch):
        """A truthy extra name is appended to the LucaOne registry list."""
        from dnallm.models.special import lucaone as lucaone_module

        registry = list(lucaone_module.lucaone_models)
        monkeypatch.setattr(lucaone_module, "lucaone_models", registry)

        result = lucaone_module._handle_lucaone_models(
            "bert-base", "local", None, extra="My-LucaOne"
        )

        assert result is None  # non-matching name still falls through
        assert "My-LucaOne" in registry

    def test_omnidna_extra_appended_to_registry(self, monkeypatch):
        """A truthy extra name is appended to the Omni-DNA registry list."""
        from dnallm.models.special import omnidna as omnidna_module

        registry = list(omnidna_module.omnidna_models)
        monkeypatch.setattr(omnidna_module, "omnidna_models", registry)

        result = omnidna_module._handle_omnidna_models("bert-base", extra="My-OmniDNA")

        assert result is None
        assert "My-OmniDNA" in registry

    def test_space_extra_appended_to_registry(self, monkeypatch):
        """A truthy extra name is appended to the Space registry list."""
        from dnallm.models.special import space as space_module

        registry = list(space_module.space_models)
        monkeypatch.setattr(space_module, "space_models", registry)

        result = space_module._handle_space_models("bert", "local", "binary", 2, extra="My-Space")

        assert result is None
        assert "My-Space" in registry


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


class TestMissingExtraGuidance:
    """D-17: missing deepspeed / flash-attn extras become actionable install guidance.

    Every helper and loader symbol is imported inside the test body (RED-run
    discipline): each target test fails on the missing symbol it asserts,
    never on a collection error.
    """

    BIGBIRD = "AIRI-Institute/gena-lm-bigbird-base-sparse"
    GUIDE_URL = "https://zhangtaolab.github.io/DNALLM/user_guide/optional_dependencies/"
    FLASH_ERROR_TEXT = (
        "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
    )

    @staticmethod
    def _missing(name: str) -> ModuleNotFoundError:
        """Return a real ModuleNotFoundError carrying the given module name."""
        try:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        except ModuleNotFoundError as error:
            return error

    @staticmethod
    def _flash_loader(error: Exception) -> Mock:
        """Return a from_pretrained loader boundary that raises *error*."""
        return Mock(**{"from_pretrained.side_effect": error})

    @staticmethod
    def _crossdna_checkpoint(tmp_path) -> "os.PathLike":
        """Write a CrossDNA-shaped 8.1M checkpoint dir under *tmp_path*."""
        import json

        checkpoint = tmp_path / "8.1M"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({
                "model_type": "crossdna",
                "auto_map": {"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"},
            }),
            encoding="utf-8",
        )
        return checkpoint

    # ── _raise_missing_extra_import_error: deepspeed shapes (BigBird) ──────

    def test_direct_deepspeed_missing_raises_actionable_import_error(self):
        """A direct ModuleNotFoundError named deepspeed becomes install guidance."""
        from dnallm.models.model import _raise_missing_extra_import_error

        trigger = self._missing("deepspeed")
        with pytest.raises(ImportError) as exc_info:
            _raise_missing_extra_import_error(trigger, self.BIGBIRD)
        message = str(exc_info.value)
        assert self.BIGBIRD in message
        assert "pip install 'dnallm[deepspeed]'" in message
        assert "0.19.7" in message  # the mandatory band-cap caveat
        assert self.GUIDE_URL in message
        assert exc_info.value.__cause__ is trigger  # original cause retained

    def test_nested_deepspeed_submodule_via_cause_chain_raises_guidance(self):
        """A deepspeed.ops.* ModuleNotFoundError nested under __cause__ is found."""
        from dnallm.models.model import _raise_missing_extra_import_error

        trigger = self._missing("deepspeed.ops.sparse_attention")
        try:
            raise RuntimeError("model construction failed") from trigger
        except RuntimeError as wrapped:
            with pytest.raises(ImportError) as exc_info:
                _raise_missing_extra_import_error(wrapped, self.BIGBIRD)
        assert "pip install 'dnallm[deepspeed]'" in str(exc_info.value)
        assert exc_info.value.__cause__ is trigger

    def test_nested_deepspeed_via_context_chain_raises_guidance(self):
        """A deepspeed ModuleNotFoundError nested under __context__ is found."""
        from dnallm.models.model import _raise_missing_extra_import_error

        trigger = self._missing("deepspeed")
        wrapped = None
        try:
            raise trigger
        except ModuleNotFoundError:
            try:
                raise ValueError("wrapped the import failure")
            except ValueError as error:
                wrapped = error
        assert isinstance(wrapped, ValueError)
        assert wrapped.__context__ is trigger
        with pytest.raises(ImportError) as exc_info:
            _raise_missing_extra_import_error(wrapped, self.BIGBIRD)
        assert exc_info.value.__cause__ is trigger

    def test_deepspeed_inside_a_cyclic_chain_still_matches(self):
        """The cycle guard terminates traversal without losing the deepspeed node."""
        from dnallm.models.model import _raise_missing_extra_import_error

        trigger = self._missing("deepspeed")
        outer = ValueError("outer")
        inner = RuntimeError("inner")
        outer.__context__ = inner
        inner.__cause__ = outer
        inner.__context__ = trigger
        with pytest.raises(ImportError) as exc_info:
            _raise_missing_extra_import_error(outer, self.BIGBIRD)
        assert exc_info.value.__cause__ is trigger

    def test_pure_cycle_of_unrelated_errors_terminates_as_none(self):
        """A cause/context cycle with no deepspeed node returns None, never hangs."""
        from dnallm.models.model import _raise_missing_extra_import_error

        first = ValueError("first")
        second = RuntimeError("second")
        first.__context__ = second
        second.__cause__ = first
        assert _raise_missing_extra_import_error(first, self.BIGBIRD) is None

    def test_unrelated_missing_module_returns_none(self):
        """A ModuleNotFoundError for another package is never relabeled."""
        from dnallm.models.model import _raise_missing_extra_import_error

        assert _raise_missing_extra_import_error(self._missing("left_pad"), self.BIGBIRD) is None

    def test_unrelated_exception_returns_none(self):
        """A plain ValueError is left for the generic wrap, never relabeled."""
        from dnallm.models.model import _raise_missing_extra_import_error

        assert _raise_missing_extra_import_error(ValueError("boom"), self.BIGBIRD) is None

    def test_already_guided_crossdna_error_is_reraised_unchanged(self):
        """The CrossDNA guidance ImportError survives the helper untouched."""
        from dnallm.models.model import _raise_missing_extra_import_error
        from dnallm.models.special.crossdna import (
            _DNALLM_MISSING_EXTRA_GUIDANCE_ATTR,
            _from_pretrained_with_flash_attn_guidance,
        )

        with pytest.raises(ImportError) as guided_info:
            _from_pretrained_with_flash_attn_guidance(
                self._flash_loader(ImportError(self.FLASH_ERROR_TEXT)), "/ckpt", {}
            )
        guided = guided_info.value
        assert getattr(guided, _DNALLM_MISSING_EXTRA_GUIDANCE_ATTR, False) is True
        with pytest.raises(ImportError) as reraised:
            _raise_missing_extra_import_error(guided, "chengCCC/CrossDNA_pretrain")
        assert reraised.value is guided  # the SAME error object, never re-wrapped

    # ── _from_pretrained_with_flash_attn_guidance (CrossDNA) ───────────────

    def test_flash_attention_import_error_becomes_install_guidance(self):
        """fla's verbatim Flash Attention ImportError becomes extra install guidance."""
        from dnallm.models.special.crossdna import _from_pretrained_with_flash_attn_guidance

        original = ImportError(self.FLASH_ERROR_TEXT)
        loader = self._flash_loader(original)
        with pytest.raises(ImportError) as exc_info:
            _from_pretrained_with_flash_attn_guidance(loader, "/ckpt", {"trust_remote_code": True})
        message = str(exc_info.value)
        assert "pip install --no-build-isolation 'dnallm[flash-attn]'" in message
        assert "Linux" in message
        assert "CUDA" in message
        assert self.GUIDE_URL in message
        assert exc_info.value.__cause__ is original
        loader.from_pretrained.assert_called_once_with("/ckpt", trust_remote_code=True)

    def test_unrelated_import_error_passes_through_unchanged(self):
        """An ImportError without the Flash Attention marker is never relabeled."""
        from dnallm.models.special.crossdna import _from_pretrained_with_flash_attn_guidance

        original = ImportError("triton kernels not compiled for this platform")
        with pytest.raises(ImportError) as exc_info:
            _from_pretrained_with_flash_attn_guidance(self._flash_loader(original), "/ckpt", {})
        assert exc_info.value is original

    def test_non_import_error_propagates_unchanged(self):
        """RuntimeErrors and friends never enter the translation at all."""
        from dnallm.models.special.crossdna import _from_pretrained_with_flash_attn_guidance

        original = RuntimeError("cuda out of memory")
        with pytest.raises(RuntimeError) as exc_info:
            _from_pretrained_with_flash_attn_guidance(self._flash_loader(original), "/ckpt", {})
        assert exc_info.value is original

    # ── both CrossDNA construction paths through _handle_crossdna_models ───

    def test_mask_path_translates_flash_attention_error(self, tmp_path, monkeypatch):
        """The mask (AutoModelForMaskedLM) path raises flash-attn guidance."""
        from dnallm.models.special import crossdna

        checkpoint = self._crossdna_checkpoint(tmp_path)
        monkeypatch.setattr(
            "dnallm.models.tokenizer.load_tokenizer_with_fallback", lambda *a, **k: Mock()
        )
        modules = {
            "AutoConfig": Mock(),
            "AutoTokenizer": Mock(),
            "AutoModelForMaskedLM": self._flash_loader(ImportError(self.FLASH_ERROR_TEXT)),
        }
        with pytest.raises(ImportError, match=r"dnallm\[flash-attn\]"):
            crossdna._handle_crossdna_models(
                "mask", str(tmp_path), 2, {0: "0", 1: "1"}, {"0": 0, "1": 1}, modules
            )
        modules["AutoModelForMaskedLM"].from_pretrained.assert_called_once_with(
            str(checkpoint), trust_remote_code=True
        )

    def test_classification_path_translates_flash_attention_error(self, tmp_path, monkeypatch):
        """The sequence-classification path raises the same flash-attn guidance."""
        from types import SimpleNamespace

        from dnallm.models.special import crossdna

        checkpoint = self._crossdna_checkpoint(tmp_path)
        monkeypatch.setattr(
            "dnallm.models.tokenizer.load_tokenizer_with_fallback", lambda *a, **k: Mock()
        )

        class _TinyBase(torch.nn.Module):
            pass

        monkeypatch.setattr(crossdna, "get_class_from_dynamic_module", lambda ref, path: _TinyBase)
        config = SimpleNamespace(auto_map={"AutoModelForMaskedLM": "repo--CrossDNAForMaskedLM"})
        modules = {
            "AutoConfig": Mock(**{"from_pretrained.return_value": config}),
            "AutoTokenizer": Mock(),
            "AutoModelForSequenceClassification": self._flash_loader(
                ImportError(self.FLASH_ERROR_TEXT)
            ),
        }
        with pytest.raises(ImportError, match=r"dnallm\[flash-attn\]"):
            crossdna._handle_crossdna_models(
                "binary", str(tmp_path), 2, {0: "0", 1: "1"}, {"0": 0, "1": 1}, modules
            )
        modules["AutoModelForSequenceClassification"].from_pretrained.assert_called_once_with(
            str(checkpoint), config=config, trust_remote_code=True
        )

    def test_handler_passes_unrelated_import_error_through(self, tmp_path, monkeypatch):
        """A non-flash ImportError from the mask path keeps its original shape."""
        from dnallm.models.special import crossdna

        self._crossdna_checkpoint(tmp_path)
        monkeypatch.setattr(
            "dnallm.models.tokenizer.load_tokenizer_with_fallback", lambda *a, **k: Mock()
        )
        original = ImportError("huggingface_hub token required")
        modules = {
            "AutoConfig": Mock(),
            "AutoTokenizer": Mock(),
            "AutoModelForMaskedLM": self._flash_loader(original),
        }
        with pytest.raises(ImportError) as exc_info:
            crossdna._handle_crossdna_models(
                "mask", str(tmp_path), 2, {0: "0", 1: "1"}, {"0": 0, "1": 1}, modules
            )
        assert exc_info.value is original

    # ── both generic load-chain wrap sites (dnallm/models/model.py) ────────

    def test_generic_pretrained_wrap_translates_nested_deepspeed(self):
        """The pretrained-chain wrap raises guidance instead of a bare ValueError."""
        from dnallm.configuration.configs import TaskConfig
        from dnallm.models.model import load_model_and_tokenizer

        trigger = self._missing("deepspeed")

        def _fail(*args, **kwargs):
            raise RuntimeError("construction failed") from trigger

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded/path", {"AutoConfig": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._load_model_by_task_type", side_effect=_fail),
        ):
            with pytest.raises(ImportError, match=r"dnallm\[deepspeed\]") as exc_info:
                load_model_and_tokenizer(
                    self.BIGBIRD, TaskConfig(task_type="mask", num_labels=None)
                )
        assert exc_info.value.__cause__ is trigger

    def test_generic_wrap_reraises_crossdna_guidance_unchanged(self):
        """The CrossDNA guidance error leaves the generic chain as itself, not a ValueError."""
        from dnallm.configuration.configs import TaskConfig
        from dnallm.models.model import load_model_and_tokenizer
        from dnallm.models.special.crossdna import _from_pretrained_with_flash_attn_guidance

        with pytest.raises(ImportError) as guided_info:
            _from_pretrained_with_flash_attn_guidance(
                self._flash_loader(ImportError(self.FLASH_ERROR_TEXT)), "/ckpt", {}
            )
        guided = guided_info.value
        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded/CrossDNA_snapshot", {"AutoConfig": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_crossdna_models", side_effect=guided),
        ):
            with pytest.raises(ImportError) as exc_info:
                load_model_and_tokenizer(
                    "chengCCC/CrossDNA_pretrain", TaskConfig(task_type="mask", num_labels=None)
                )
        assert exc_info.value is guided
        assert "Failed to load model" not in str(exc_info.value)

    def test_random_init_wrap_translates_nested_deepspeed(self):
        """The random-init chain wrap also raises deepspeed guidance."""
        from dnallm.configuration.configs import TaskConfig
        from dnallm.models.model import load_model_and_tokenizer

        trigger = self._missing("deepspeed")

        def _fail(*args, **kwargs):
            raise RuntimeError("construction failed") from trigger

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._load_random_init_model", side_effect=_fail),
        ):
            with pytest.raises(ImportError, match=r"dnallm\[deepspeed\]"):
                load_model_and_tokenizer(
                    "test-scratch-bigbird",
                    TaskConfig(task_type="mask", num_labels=None),
                    source="huggingface",
                    random_init=True,
                )


class TestSpecialExports:
    """The special package re-export surface."""

    def test_all_handlers_importable(self):
        """Every declared export resolves to a callable in the package."""
        import dnallm.models.special as special_package

        for name in special_exports:
            attribute = getattr(special_package, name, None)
            assert callable(attribute), f"{name} must be callable"
