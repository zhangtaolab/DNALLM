"""Tests for the EVO special handlers.

EvoTokenizerWrapper is covered directly (it has no absent-dep imports); the
evo2/evo1 handler bodies are reached through sys.modules stubs injected via
monkeypatch.setitem so every stub is restored between tests (Pitfall 10 —
never a direct sys.modules assignment).
"""

import json
import os
import sys
import types
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from dnallm.models.special.evo import (
    EvoTokenizerWrapper,
    _EVO1_SAFETENSORS_ONLY_PATTERNS,
    _handle_evo1_models,
    _handle_evo2_models,
)


class RawCharTokenizer:
    """Deterministic stand-in for the EVO package's CharLevelTokenizer."""

    vocab_size = 512
    bos_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    pad_token_id = 1
    pad_id = 1
    eos_id = 1
    eod_id = 1

    def tokenize(self, sequence):
        return [(ord(char) % 8) + 3 for char in sequence]

    def decode_token(self, token_id):
        return "-" if token_id == self.pad_id else "X"


class TestEvoTokenizerWrapperInit:
    """EvoTokenizerWrapper attribute copying."""

    def test_attributes_copied_from_raw_tokenizer(self):
        """Known attributes are forwarded from the raw tokenizer."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=64)

        assert wrapper.vocab_size == 512
        assert wrapper.bos_token_id == 0
        assert wrapper.eos_token_id == 1
        assert wrapper.pad_token_id == 1
        assert wrapper.model_max_length == 64
        assert wrapper.padding_side == "right"

    def test_pad_token_decoded(self):
        """The pad token string comes from decode_token at the pad id."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        assert wrapper.pad_token == "-"  # ruff: ignore[hardcoded-password-string]

    def test_missing_pad_token_id_falls_back_to_pad_id(self):
        """A raw tokenizer without pad_token_id uses pad_id."""

        class Minimal:
            pad_id = 5

            def decode_token(self, token_id):
                return "[pad]"

        wrapper = EvoTokenizerWrapper(Minimal())

        assert wrapper.pad_token_id == 5
        assert wrapper.pad_token == "[pad]"  # ruff: ignore[hardcoded-password-string]

    def test_init_kwargs_stored(self):
        """Extra kwargs are kept for the save round-trip."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), custom_flag=True)

        assert wrapper.init_kwargs == {"custom_flag": True}


class TestEvoTokenizerWrapperCall:
    """EvoTokenizerWrapper __call__ encoding."""

    def test_single_string_unbatched(self):
        """A single string returns unbatched lists without tensors."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper("ACGT")

        assert isinstance(result["input_ids"], list)
        assert len(result["input_ids"]) == 4
        assert result["attention_mask"] == [1, 1, 1, 1]

    def test_batch_without_padding(self):
        """A batch without padding keeps per-sequence lengths."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["ACG", "AC"])

        assert [len(ids) for ids in result["input_ids"]] == [3, 2]
        assert result["attention_mask"] == [[1, 1, 1], [1, 1]]

    def test_batch_padding_longest(self):
        """padding=True pads the batch to its longest sequence."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["ACG", "AC"], padding=True)

        assert len(result["input_ids"][1]) == 3
        assert result["input_ids"][1][2] == wrapper.pad_token_id
        assert result["attention_mask"][1] == [1, 1, 0]

    def test_padding_max_length(self):
        """padding='max_length' pads to the explicit max_length."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["AC"], padding="max_length", max_length=4)

        assert len(result["input_ids"][0]) == 4
        assert result["attention_mask"][0] == [1, 1, 0, 0]

    def test_padding_max_length_defaults_to_model_max(self):
        """padding='max_length' without max_length uses model_max_length."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=5)

        result = wrapper(["AC"], padding="max_length")

        assert len(result["input_ids"][0]) == 5

    def test_truncation_to_max_length(self):
        """truncation=True clips sequences to max_length."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["ACGTACGT"], truncation=True, max_length=3)

        assert len(result["input_ids"][0]) == 3

    def test_truncation_defaults_to_model_max(self):
        """truncation without max_length clips to model_max_length."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=3)

        result = wrapper(["ACGTACGT"], truncation=True)

        assert len(result["input_ids"][0]) == 3

    def test_padding_shrinks_longer_sequences(self):
        """A max_length shorter than the sequence truncates during padding."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["ACGT"], padding="max_length", max_length=2)

        assert len(result["input_ids"][0]) == 2
        assert result["attention_mask"][0] == [1, 1]

    def test_return_tensors_pt(self):
        """return_tensors='pt' yields tensor batches."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["AC", "ACG"], padding=True, return_tensors="pt")

        assert isinstance(result["input_ids"], torch.Tensor)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_batched_result_is_batch_encoding(self):
        """Batched calls return a BatchEncoding mapping."""
        from transformers import BatchEncoding

        wrapper = EvoTokenizerWrapper(RawCharTokenizer())

        result = wrapper(["AC"])

        assert isinstance(result, BatchEncoding)


class TestEvoTokenizerWrapperPersistence:
    """save_pretrained / from_pretrained round-trips."""

    def test_save_round_trip(self, tmp_path):
        """Saved settings are restored."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=33)

        files = wrapper.save_pretrained(str(tmp_path))

        assert files == [str(tmp_path / "tokenizer_config.json")]
        saved = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert saved["model_max_length"] == 33
        assert saved["tokenizer_class"] == "Evo2TokenizerWrapper"

        restored = EvoTokenizerWrapper.from_pretrained(str(tmp_path), RawCharTokenizer())

        assert restored.model_max_length == 33
        assert restored.pad_token_id == wrapper.pad_token_id

    def test_save_to_file_path_raises(self, tmp_path):
        """Saving to a file path raises ValueError."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer())
        target = tmp_path / "target.txt"
        target.write_text("x")

        with pytest.raises(ValueError, match="should be a directory"):
            wrapper.save_pretrained(str(target))

    def test_from_pretrained_missing_config_uses_defaults(self, tmp_path, capsys):
        """A missing config file falls back to defaults with a warning."""
        empty = tmp_path / "empty"
        empty.mkdir()

        restored = EvoTokenizerWrapper.from_pretrained(str(empty), RawCharTokenizer())

        assert restored.model_max_length == 8192
        assert "not found" in capsys.readouterr().out

    def test_from_pretrained_kwargs_override(self, tmp_path):
        """Explicit kwargs override the saved config."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=33)
        wrapper.save_pretrained(str(tmp_path))

        restored = EvoTokenizerWrapper.from_pretrained(
            str(tmp_path), RawCharTokenizer(), model_max_length=100
        )

        assert restored.model_max_length == 100

    def test_model_max_length_setter(self):
        """model_max_length round-trips through its property."""
        wrapper = EvoTokenizerWrapper(RawCharTokenizer(), model_max_length=8)
        wrapper.model_max_length = 99

        assert wrapper.model_max_length == 99


class FakeLoadedEvo2Model(nn.Module):
    """Real module standing in for the loaded evo2 weights object."""

    def __init__(self, hidden=4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
        self.config = {"max_seqlen": 16, "hidden_size": hidden}


class FakeEvo2:
    """Stand-in for the evo2 package's Evo2 class."""

    load_calls: ClassVar[list] = []

    def load_evo2_model(self, *args, **kwargs):
        type(self).load_calls.append(kwargs)
        return FakeLoadedEvo2Model()


class FakeVortexCharLevelTokenizer:
    """Stand-in for vortex.model.tokenizer.CharLevelTokenizer."""

    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.pad_id = 1

    def decode_token(self, token_id):
        return "-"

    def tokenize(self, sequence):
        return [(ord(c) % 8) + 3 for c in sequence]


def _install_evo2_stubs(monkeypatch):
    """Inject evo2 + vortex stubs into sys.modules (auto-restored)."""
    FakeEvo2.load_calls = []
    fake_evo2_module = types.ModuleType("evo2")
    fake_evo2_module.Evo2 = FakeEvo2

    fake_vortex = types.ModuleType("vortex")
    fake_vortex_model = types.ModuleType("vortex.model")
    fake_vortex_tok = types.ModuleType("vortex.model.tokenizer")
    fake_vortex_tok.CharLevelTokenizer = FakeVortexCharLevelTokenizer

    monkeypatch.setitem(sys.modules, "evo2", fake_evo2_module)
    monkeypatch.setitem(sys.modules, "vortex", fake_vortex)
    monkeypatch.setitem(sys.modules, "vortex.model", fake_vortex_model)
    monkeypatch.setitem(sys.modules, "vortex.model.tokenizer", fake_vortex_tok)


def _capability_patchers(flash_attention=True, fp8=True):
    """Patch the capability probes the evo handler consults."""
    return [
        patch(
            "dnallm.models.special.evo.is_flash_attention_capable",
            return_value=flash_attention,
        ),
        patch("dnallm.models.special.evo.is_fp8_capable", return_value=fp8),
    ]


class TestHandleEvo2Models:
    """_handle_evo2_models dispatch with evo2 stubbed."""

    def test_non_matching_name_returns_none(self):
        """A non-EVO2 model name falls through."""
        assert _handle_evo2_models("bert-base", "local") is None

    def test_missing_package_raises_importerror(self):
        """An EVO2 name without the package raises the instructive ImportError."""
        with pytest.raises(ImportError, match="EVO2 package is required"):
            _handle_evo2_models("evo2_1b_base", "local")

    def test_local_dir_without_pt_raises_value_error(self, monkeypatch, tmp_path):
        """An empty local evo2 dir raises a descriptive ValueError (IN-01).

        The glob used to index [0] unguarded, so a directory with no .pt
        files escaped as a bare IndexError.
        """
        _install_evo2_stubs(monkeypatch)
        model_dir = tmp_path / "evo2_1b_base"
        model_dir.mkdir()

        with pytest.raises(ValueError, match=r"No \.pt checkpoint found in"):
            _handle_evo2_models(str(model_dir), "local")

    def test_local_source_loads_checkpoint(self, monkeypatch, tmp_path):
        """Local source resolves the .pt file inside the model directory."""
        _install_evo2_stubs(monkeypatch)
        model_dir = tmp_path / "evo2_1b_base"
        model_dir.mkdir()
        (model_dir / "weights.pt").write_bytes(b"fake")

        from contextlib import ExitStack

        with ExitStack() as stack:
            for patcher in _capability_patchers():
                stack.enter_context(patcher)
            model, tokenizer = _handle_evo2_models(str(model_dir), "local")

        assert isinstance(model, FakeEvo2)
        assert isinstance(tokenizer, EvoTokenizerWrapper)
        assert tokenizer.model_max_length == 16
        load_kwargs = FakeEvo2.load_calls[-1]
        assert load_kwargs["local_path"] == str(model_dir / "weights.pt")
        assert load_kwargs["config_path"].endswith("evo2-1b-8k.yml")

    def test_capability_suffixes_selected(self, monkeypatch, tmp_path):
        """No-FA / no-FP8 environments pick the suffixed architecture config."""
        _install_evo2_stubs(monkeypatch)
        model_dir = tmp_path / "evo2_1b_base"
        model_dir.mkdir()
        (model_dir / "weights.pt").write_bytes(b"fake")

        from contextlib import ExitStack

        with ExitStack() as stack:
            for patcher in _capability_patchers(flash_attention=False, fp8=False):
                stack.enter_context(patcher)
            _handle_evo2_models(str(model_dir), "local")

        assert FakeEvo2.load_calls[-1]["config_path"].endswith("evo2-1b-8k-noFA-noFP8.yml")

    def test_remote_source_joins_model_file(self, monkeypatch):
        """Non-local sources download and join <model>.pt onto the path."""
        _install_evo2_stubs(monkeypatch)

        from contextlib import ExitStack

        with (
            ExitStack() as stack,
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", None),
            ),
        ):
            for patcher in _capability_patchers():
                stack.enter_context(patcher)
            _model, _tokenizer = _handle_evo2_models("evo2_7b_base", "huggingface")

        load_kwargs = FakeEvo2.load_calls[-1]
        assert load_kwargs["local_path"] == os.path.join("/downloaded", "evo2_7b_base.pt")
        assert load_kwargs["config_path"].endswith("evo2-7b-8k.yml")

    def test_head_config_wraps_in_sequence_classifier(self, monkeypatch, tmp_path):
        """A head_config wraps the loaded model in DNALLMforSequenceClassification."""
        _install_evo2_stubs(monkeypatch)
        model_dir = tmp_path / "evo2_1b_base"
        model_dir.mkdir()
        (model_dir / "weights.pt").write_bytes(b"fake")
        head_config = SimpleNamespace(
            head="evo", num_classes=2, task_type="binary", target_layer="blocks.0"
        )

        from contextlib import ExitStack

        with ExitStack() as stack:
            for patcher in _capability_patchers():
                stack.enter_context(patcher)
            result = _handle_evo2_models(str(model_dir), "local", head_config=head_config)

        wrapped, tokenizer = result
        from dnallm.models.model import DNALLMforSequenceClassification

        assert isinstance(wrapped, DNALLMforSequenceClassification)
        assert wrapped.backbone is not None
        # The custom model was handed to the wrapper; the tokenizer survives.
        assert isinstance(tokenizer, EvoTokenizerWrapper)


class FakeStripedHyena:
    """Stand-in for stripedhyena.model.StripedHyena."""

    init_calls: ClassVar[list] = []

    def __init__(self, global_config):
        self.config = {"max_sequence_len": 16}
        type(self).init_calls.append(global_config)

    def load_state_dict(self, state_dict, strict=True):
        self.state_dict = state_dict

    def to_bfloat16_except_poles_residues(self):
        pass


def _install_evo1_stubs(monkeypatch):
    """Inject evo + stripedhyena stubs into sys.modules (auto-restored)."""
    FakeStripedHyena.init_calls = []

    class FakeEvo:
        def __init__(self):
            self.device = None
            self.model = None

    fake_evo_module = types.ModuleType("evo")
    fake_evo_module.Evo = FakeEvo

    fake_sh = types.ModuleType("stripedhyena")
    fake_sh_utils = types.ModuleType("stripedhyena.utils")
    fake_sh_utils.dotdict = dict
    fake_sh_model = types.ModuleType("stripedhyena.model")
    fake_sh_model.StripedHyena = FakeStripedHyena
    fake_sh_tok = types.ModuleType("stripedhyena.tokenizer")
    fake_sh_tok.CharLevelTokenizer = FakeVortexCharLevelTokenizer

    monkeypatch.setitem(sys.modules, "evo", fake_evo_module)
    monkeypatch.setitem(sys.modules, "stripedhyena", fake_sh)
    monkeypatch.setitem(sys.modules, "stripedhyena.utils", fake_sh_utils)
    monkeypatch.setitem(sys.modules, "stripedhyena.model", fake_sh_model)
    monkeypatch.setitem(sys.modules, "stripedhyena.tokenizer", fake_sh_tok)


def _evo1_modules():
    """Modules dict returned by a patched _get_model_path_and_imports."""
    hf_model = Mock()
    hf_model.backbone.state_dict.return_value = {"layer.weight": torch.zeros(2, 2)}
    return {
        "AutoConfig": Mock(**{"from_pretrained.return_value": Mock(use_cache=None)}),
        "AutoModelForCausalLM": Mock(**{"from_pretrained.return_value": hf_model}),
    }


class TestHandleEvo1Models:
    """_handle_evo1_models dispatch with evo/stripedhyena stubbed."""

    def test_non_matching_name_returns_none(self):
        """A non-EVO1 model name falls through."""
        assert _handle_evo1_models("bert-base", "local") is None

    def test_missing_package_raises_importerror(self):
        """An EVO1 name without the package raises the instructive ImportError."""
        with pytest.raises(ImportError, match="EVO-1 package is required"):
            _handle_evo1_models("evo-1-8k-base", "local")

    def test_loads_checkpoint_with_main_revision(self, monkeypatch):
        """A dotted HuggingFace name selects the 1.1_fix revision."""
        _install_evo1_stubs(monkeypatch)
        modules = _evo1_modules()

        from contextlib import ExitStack

        with (
            ExitStack() as stack,
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", modules),
            ) as mock_resolve,
        ):
            stack.enter_context(
                patch(
                    "dnallm.models.special.evo.is_flash_attention_capable",
                    return_value=True,
                )
            )
            _model, tokenizer = _handle_evo1_models("evo-1.5-8k-base", "huggingface")

        assert mock_resolve.call_args.kwargs["revision"] == "1.1_fix"
        assert isinstance(tokenizer, EvoTokenizerWrapper)
        assert tokenizer.model_max_length == 16
        # The checkpoint loader consumed the HF modules to build the model
        modules["AutoConfig"].from_pretrained.assert_called_once()
        assert FakeStripedHyena.init_calls, "StripedHyena must be constructed"

    def test_undotted_name_uses_main_revision(self, monkeypatch):
        """An undotted name keeps the main revision."""
        _install_evo1_stubs(monkeypatch)
        modules = _evo1_modules()

        from contextlib import ExitStack

        with (
            ExitStack() as stack,
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", modules),
            ) as mock_resolve,
        ):
            stack.enter_context(
                patch(
                    "dnallm.models.special.evo.is_flash_attention_capable",
                    return_value=False,
                )
            )
            _model, tokenizer = _handle_evo1_models("evo-1-8k-base", "local")

        assert mock_resolve.call_args.kwargs["revision"] == "main"
        assert isinstance(tokenizer, EvoTokenizerWrapper)

    def test_mixed_case_source_selects_1_1_fix_revision(self, monkeypatch):
        """source='HuggingFace' resolves 1.1_fix like the lowercase form (IN-03)."""
        _install_evo1_stubs(monkeypatch)
        modules = _evo1_modules()

        with (
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", modules),
            ) as mock_resolve,
            patch(
                "dnallm.models.special.evo.is_flash_attention_capable",
                return_value=True,
            ),
        ):
            _handle_evo1_models("evo-1.5-8k-base", "HuggingFace")

        assert mock_resolve.call_args.kwargs["revision"] == "1.1_fix"

    def test_head_config_wraps_in_sequence_classifier(self, monkeypatch):
        """A head_config wraps the loaded model in the DNALLM wrapper."""
        _install_evo1_stubs(monkeypatch)
        modules = _evo1_modules()
        head_config = SimpleNamespace(head="evo", num_classes=2, task_type="binary")

        from contextlib import ExitStack

        with (
            ExitStack() as stack,
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", modules),
            ),
        ):
            stack.enter_context(
                patch(
                    "dnallm.models.special.evo.is_flash_attention_capable",
                    return_value=True,
                )
            )
            with patch("dnallm.models.model.DNALLMforSequenceClassification") as wrapper_cls:
                wrapper_cls.return_value = "wrapped-model"
                result = _handle_evo1_models(
                    "evo-1-8k-base", "huggingface", head_config=head_config
                )

        model, tokenizer = result
        assert model == "wrapped-model"
        assert isinstance(tokenizer, EvoTokenizerWrapper)

    def test_hub_fetch_is_safetensors_only(self, monkeypatch):
        """The evo-1 hub fetch carries the safetensors+code allow_patterns set.

        CI-05 / Pitfall 4: a warm giants dir must never be re-expanded with
        the 16.81GB pytorch_model.pt at load time -- the pattern set keeps
        the snapshot download restricted to safetensors + configs.  CR-01
        (08): "*.py" joins the set because offline trust_remote_code
        resolution in load_checkpoint needs the auto_map code files
        (configuration_hyena.py / modeling_hyena.py) in the hub cache --
        fetched from the repo itself or, for the 8k variants whose auto_map
        redirects there, from the evo-1-131k-base sibling repo; the .pt
        weights stay out (see test_hub_fetch_patterns_exclude_pt_weights).
        """
        _install_evo1_stubs(monkeypatch)
        modules = _evo1_modules()

        with (
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/downloaded", modules),
            ) as mock_resolve,
            patch(
                "dnallm.models.special.evo.is_flash_attention_capable",
                return_value=True,
            ),
        ):
            _handle_evo1_models("evo-1-8k-base", "huggingface")

        assert mock_resolve.call_args.kwargs["allow_patterns"] == [
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.py",
            "README.md",
        ]

    def test_hub_fetch_patterns_exclude_pt_weights(self):
        """No evo-1 fetch pattern can match the pytorch_model.pt weights.

        The Pitfall-4 skip intent (never re-expand a warm giants dir with
        the 16.81GB .pt) survives the CR-01 "*.py" addition by omission:
        the allow list must contain no "*.pt" / "pytorch_model" entry and
        no glob able to match a .pt file at all.
        """
        patterns = _EVO1_SAFETENSORS_ONLY_PATTERNS

        assert "*.pt" not in patterns
        assert not any("pytorch_model" in pattern for pattern in patterns)
        assert not any(pattern.endswith(".pt") for pattern in patterns)
        assert not any(pattern == "*" or pattern == "*.*" for pattern in patterns)
