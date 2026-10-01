"""Tests for the DNA tokenizer module.

Covers the three-tier tokenizer fallback chain (AutoTokenizer ->
PreTrainedTokenizerFast -> DNAOneHotTokenizer) with staged failures that
assert WHICH tier served, plus the real DNAOneHotTokenizer surface.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from dnallm.models.tokenizer import DNAOneHotTokenizer, load_tokenizer_with_fallback


def _warning_texts(mock_logger):
    """Collect the message strings of every warning call on the mock logger."""
    return [str(call.args[0]) for call in mock_logger.warning.call_args_list]


class TestFallbackTierOne:
    """Tier-1 resolution: the AutoTokenizer path succeeds."""

    def test_sentinel_auto_tokenizer_cls_serves(self):
        """An explicit auto_tokenizer_cls sentinel is returned verbatim."""
        sentinel = Mock()
        auto_cls = Mock(**{"from_pretrained.return_value": sentinel})

        result = load_tokenizer_with_fallback("test-model", auto_tokenizer_cls=auto_cls)

        assert result is sentinel
        auto_cls.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)

    def test_default_auto_tokenizer_path_serves(self):
        """Without auto_tokenizer_cls the transformers AutoTokenizer serves."""
        sentinel = Mock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=sentinel):
            result = load_tokenizer_with_fallback("test-model")

        assert result is sentinel

    def test_add_prefix_space_forwarded(self):
        """add_prefix_space=True is forwarded to the tier-1 loader."""
        sentinel = Mock()
        auto_cls = Mock(**{"from_pretrained.return_value": sentinel})

        load_tokenizer_with_fallback(
            "test-model", auto_tokenizer_cls=auto_cls, add_prefix_space=True
        )

        auto_cls.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, add_prefix_space=True
        )


class TestFallbackTierTwo:
    """Tier-2 resolution: AutoTokenizer fails, PreTrainedTokenizerFast succeeds."""

    def test_fast_tokenizer_serves_with_warning(self):
        """A tier-1 failure falls back to PreTrainedTokenizerFast with a warning."""
        sentinel = Mock()
        failing_auto = Mock(**{"from_pretrained.side_effect": Exception("slow class mismatch")})

        with (
            patch("transformers.PreTrainedTokenizerFast.from_pretrained", return_value=sentinel),
            patch("dnallm.models.tokenizer.logger") as mock_logger,
        ):
            result = load_tokenizer_with_fallback("test-model", auto_tokenizer_cls=failing_auto)

        assert result is sentinel
        assert any("loaded fast tokenizer" in text for text in _warning_texts(mock_logger))

    def test_default_path_fast_tokenizer_serves(self):
        """The tier-2 fallback also works from the default AutoTokenizer path."""
        sentinel = Mock()

        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("boom")),
            patch("transformers.PreTrainedTokenizerFast.from_pretrained", return_value=sentinel),
        ):
            result = load_tokenizer_with_fallback("test-model")

        assert result is sentinel

    def test_tier_two_receives_kwargs(self):
        """trust_remote_code and add_prefix_space reach the tier-2 loader."""
        sentinel = Mock()
        failing_auto = Mock(**{"from_pretrained.side_effect": Exception("boom")})

        with patch(
            "transformers.PreTrainedTokenizerFast.from_pretrained", return_value=sentinel
        ) as mock_fast:
            load_tokenizer_with_fallback(
                "test-model",
                auto_tokenizer_cls=failing_auto,
                add_prefix_space=True,
            )

        mock_fast.assert_called_once_with(
            "test-model", trust_remote_code=True, add_prefix_space=True
        )


class TestFallbackTierThree:
    """Tier-3 resolution: both transformers tiers fail -> DNAOneHotTokenizer."""

    def test_one_hot_tokenizer_serves_with_warning(self):
        """When both tiers fail the DNAOneHotTokenizer serves with its warning."""
        failing_auto = Mock(**{"from_pretrained.side_effect": Exception("boom")})

        with (
            patch(
                "transformers.PreTrainedTokenizerFast.from_pretrained",
                side_effect=Exception("still broken"),
            ),
            patch("dnallm.models.tokenizer.logger") as mock_logger,
        ):
            result = load_tokenizer_with_fallback("test-model", auto_tokenizer_cls=failing_auto)

        assert isinstance(result, DNAOneHotTokenizer)
        assert any("using DNAOneHotTokenizer" in text for text in _warning_texts(mock_logger))

    def test_default_path_one_hot_tokenizer_serves(self):
        """The tier-3 fallback also works from the default AutoTokenizer path."""
        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("boom")),
            patch(
                "transformers.PreTrainedTokenizerFast.from_pretrained",
                side_effect=Exception("boom"),
            ),
        ):
            result = load_tokenizer_with_fallback("test-model")

        assert isinstance(result, DNAOneHotTokenizer)


class TestDNAOneHotTokenizerVocab:
    """Vocabulary mapping behavior of DNAOneHotTokenizer."""

    def test_vocab_size_is_six(self):
        """The vocab covers ACGT/N/padding in one-hot form."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.vocab_size == 6

    def test_convert_tokens_to_ids_string_overload(self):
        """A single-character token maps to its id."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.convert_tokens_to_ids("A") == 0
        assert tokenizer.convert_tokens_to_ids("T") == 3
        assert tokenizer.convert_tokens_to_ids("U") == 3  # RNA alias

    def test_convert_tokens_to_ids_list_overload(self):
        """A token list maps element-wise."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.convert_tokens_to_ids(["A", "C", "G", "T"]) == [0, 1, 2, 3]

    def test_unknown_token_maps_to_unk(self):
        """Characters outside the vocabulary map to the N id (4)."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.convert_tokens_to_ids("Z") == 4
        assert tokenizer.convert_tokens_to_ids(["Z", "A"]) == [4, 0]

    def test_case_insensitive_mapping(self):
        """Lower-case nucleotides map like their upper-case forms."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.convert_tokens_to_ids("a") == 0
        assert tokenizer.convert_tokens_to_ids(["a", "c", "g", "t"]) == [0, 1, 2, 3]

    def test_convert_ids_to_tokens_round_trip(self):
        """Ids map back to tokens, including the padding token."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.convert_ids_to_tokens([0, 1, 2, 3, 4, -1]) == [
            "A",
            "C",
            "G",
            "T",
            "N",
            "-",
        ]
        assert tokenizer.convert_ids_to_tokens(0) == "A"
        assert tokenizer.convert_ids_to_tokens(99) == "N"  # unknown id falls back


class TestDNAOneHotTokenizerCall:
    """__call__ encoding behavior."""

    def test_single_sequence_dict_output(self):
        """A single sequence returns a dict with input_ids and attention_mask."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer("ATCG", max_length=4, padding=True)

        assert isinstance(output, dict)
        assert output["input_ids"].tolist() == [[0, 3, 1, 2]]
        assert output["attention_mask"].tolist() == [[1, 1, 1, 1]]

    def test_batch_pads_to_max_length(self):
        """A batch pads shorter sequences with pad id -1 and zero masks."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer(["AT", "ATCG"], max_length=4, padding=True)

        assert output["input_ids"].tolist() == [[0, 3, -1, -1], [0, 3, 1, 2]]
        assert output["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]

    def test_left_padding_side(self):
        """padding_side='left' prepends the padding tokens."""
        tokenizer = DNAOneHotTokenizer(padding_side="left")

        output = tokenizer(["AT", "ATCG"], max_length=4, padding=True)

        assert output["input_ids"].tolist() == [[-1, -1, 0, 3], [0, 3, 1, 2]]
        assert output["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]

    def test_truncation_to_max_length(self):
        """Sequences longer than max_length are truncated."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer("ATATAT", max_length=3, padding=True, truncation=True)

        assert output["input_ids"].tolist() == [[0, 3, 0]]

    def test_no_padding_packs_to_longest(self):
        """Without padding the batch packs to the longest sequence."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer(["AT", "ATCG"], padding=False, truncation=False)

        assert output["input_ids"].tolist() == [[0, 3, -1, -1], [0, 3, 1, 2]]

    def test_tensor_ids_without_dict(self):
        """return_dict=False returns the raw id tensor."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer("ATCG", max_length=4, return_dict=False)

        assert isinstance(output, torch.Tensor)
        assert output.tolist() == [[0, 3, 1, 2]]

    def test_inputs_embeds_shape(self):
        """return_inputs_embeds yields a one-hot (batch, length, 4) tensor."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer("ATCG", max_length=4, return_inputs_embeds=True)

        assert output["inputs_embeds"].shape == (1, 4, 4)
        assert torch.equal(output["inputs_embeds"][0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_inputs_embeds_transposed(self):
        """embeds_transpose flips to (batch, 4, length)."""
        tokenizer = DNAOneHotTokenizer(return_embeds=True, embeds_transpose=True)

        output = tokenizer("ATCG", max_length=4)

        assert output["inputs_embeds"].shape == (1, 4, 4)
        assert torch.equal(output["inputs_embeds"][0, :, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_numpy_tensors_requested(self):
        """return_tensors='np' converts the dict values to numpy arrays."""
        tokenizer = DNAOneHotTokenizer()

        output = tokenizer("ATCG", max_length=4, return_tensors="np")

        assert not isinstance(output["input_ids"], torch.Tensor)
        assert output["input_ids"].tolist() if hasattr(output["input_ids"], "tolist") else True


class TestDNAOneHotTokenizerEncodeDecode:
    """encode/decode/batch_decode round-trips."""

    def test_encode_truncates(self):
        """encode truncates to max_length when asked."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.encode("ATATAT", max_length=2) == [0, 3]
        assert tokenizer.encode("ATATAT") == [0, 3, 0, 3, 0, 3]

    def test_decode_skips_special_tokens(self):
        """decode drops padding tokens by default."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.decode([0, 1, -1, -1]) == "AC"
        assert tokenizer.decode([0, 1, -1, -1], skip_special_tokens=False) == "AC--"

    def test_decode_accepts_1d_tensor(self):
        """decode handles a 1D id tensor."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.decode(torch.tensor([0, 1, 2])) == "ACG"

    def test_batch_decode(self):
        """batch_decode decodes each row."""
        tokenizer = DNAOneHotTokenizer()

        assert tokenizer.batch_decode([[0, 1], [2, 3]]) == ["AC", "GT"]
        assert tokenizer.batch_decode(torch.tensor([[0, 1], [2, 3]])) == ["AC", "GT"]


class TestDNAOneHotTokenizerPersistence:
    """save_pretrained/from_pretrained round-trips."""

    def test_round_trip_preserves_settings(self, tmp_path):
        """Saved settings are restored by from_pretrained."""
        tokenizer = DNAOneHotTokenizer(max_length=32, padding_side="left")

        files = tokenizer.save_pretrained(str(tmp_path))

        assert len(files) == 2
        assert all(path.startswith(str(tmp_path)) for path in files)

        restored = DNAOneHotTokenizer.from_pretrained(str(tmp_path))

        assert restored.max_length == 32
        assert restored.padding_side == "left"

    def test_from_pretrained_missing_config_uses_defaults(self, tmp_path, capsys):
        """A directory without tokenizer_config.json falls back to defaults."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        restored = DNAOneHotTokenizer.from_pretrained(str(empty_dir))

        assert restored.max_length == 196_608
        assert "tokenizer_config.json not found" in capsys.readouterr().out

    def test_from_pretrained_kwargs_override_config(self, tmp_path):
        """Explicit kwargs override the saved config values."""
        tokenizer = DNAOneHotTokenizer(max_length=32)
        tokenizer.save_pretrained(str(tmp_path))

        restored = DNAOneHotTokenizer.from_pretrained(str(tmp_path), max_length=64)

        assert restored.max_length == 64
