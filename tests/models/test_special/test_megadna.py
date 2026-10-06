"""Regression tests for the megaDNA DNATokenizer generate path (08-04, D-19).

The census failure signature (05-06 spike ``megadna-pinned-fallback``,
Phase 8 repair queue): a prompt carrying any character outside the
six-token megaDNA vocabulary encoded to ``None`` ids (``unk_token=None``
made ``unk_token_id`` None) and crashed tensor creation inside
transformers ``_call_one`` -- the megaDNA generate path died before the
model ever ran.  The committed ``finetune_generation`` training corpus
(ath_cds.csv) contains 11 IUPAC ambiguity codes (W/K/S/M/Y), so the
notebook's megaDNA half hit exactly this class.

The tokenizer class is defined inside ``_handle_megadna_models``; every
test reaches the REAL class through the handler with the model download
and ``torch.load`` patched (fakes, never downloads -- fast-lane friendly),
mirroring ``tests/models/test_special/test_family_handlers.py``.
"""

import torch
import pytest
from unittest.mock import patch

from dnallm.models.special.megadna import _handle_megadna_models


def _load_tokenizer():
    """Reach the real DNATokenizer through the handler with fakes."""
    with (
        patch("torch.load", return_value=object()),
        patch(
            "dnallm.models.model._get_model_path_and_imports",
            return_value=("/downloaded/model", None),
        ),
    ):
        _model, tokenizer = _handle_megadna_models("megaDNA_updated", "local", None)
    return tokenizer


class TestDnaTokenizerGeneratePath:
    """The megaDNA generate encode contract (inference.py megadna branch).

    The generate path calls ``tokenizer(seq, return_tensors="pt")`` on a
    single prompt string; every character of that prompt must land on a
    valid in-vocabulary id so the tensor builds.
    """

    def test_acgt_prompt_encodes_to_nucleotide_ids(self):
        """A plain ACGT prompt encodes to the nucleotide ids 1-4."""
        tokenizer = _load_tokenizer()

        encoded = tokenizer("ACGT", return_tensors="pt")["input_ids"]

        assert encoded.tolist() == [[1, 3, 4, 2]]

    def test_non_vocab_characters_encode_to_a_valid_id(self):
        """Unknown characters map to the upstream unknown id (1), never None.

        Upstream's own encoding rule (megaDNA mutagenesis notebook,
        ``encode_sequence``) maps any nucleotide outside the six-token
        vocabulary to id 1; the checkpoint vocabulary is six tokens wide,
        so introducing a new unknown id would break the embedding lookup.
        """
        tokenizer = _load_tokenizer()

        encoded = tokenizer("ACGTRYSWKMNDHVBacgt", return_tensors="pt")["input_ids"]
        ids = encoded.tolist()[0]

        assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)
        assert ids[:4] == [1, 3, 4, 2]
        assert all(token_id == 1 for token_id in ids[4:])

    def test_lowercase_and_ambiguous_prompts_survive_the_generate_encode(self):
        """The exact inference.py generate call shape builds an int tensor.

        ``acgN`` is the crash class from the census signature: lowercase
        soft-masked letters plus an ambiguity code.
        """
        tokenizer = _load_tokenizer()

        input_ids = tokenizer("acgN", return_tensors="pt")["input_ids"]

        assert input_ids.dtype == torch.int64
        assert input_ids.shape == (1, 4)
        assert input_ids.tolist() == [[1, 1, 1, 1]]

    def test_repaired_generate_path_produces_valid_output(self):
        """A fake checkpoint model end-to-end through the generate branch.

        Mirrors inference.py megadna branch verbatim (encode -> model
        .generate(seq_len=..., temperature=..., filter_thres=...) ->
        decode -> strip spaces) over the crash-class prompt.
        """
        tokenizer = _load_tokenizer()

        class _FakeMegaDNA:
            """Stand-in for the torch.loaded checkpoint (plain shapes)."""

            @staticmethod
            def generate(input_ids, seq_len, temperature, filter_thres):
                assert input_ids.dtype == torch.int64
                assert 0 <= int(input_ids.min())
                assert int(input_ids.max()) < 6
                return torch.ones(1, input_ids.shape[1] + seq_len, dtype=torch.int64)

        model = _FakeMegaDNA()
        prompt = "acgN"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        output = model.generate(input_ids, seq_len=8, temperature=0.95, filter_thres=0.1)
        decoded = tokenizer.decode(output.squeeze().cpu().int())

        assert decoded.replace(" ", "") == "A" * output.shape[1]


class TestDnaTokenizerDecodePath:
    """Edge behavior the encode bug corrupted: id range and round-trip."""

    def test_generated_ids_round_trip_through_decode(self):
        """Model-side ids (eos included) decode back to their nucleotides."""
        tokenizer = _load_tokenizer()

        output = torch.tensor([[1, 3, 4, 2, 5]])
        decoded = tokenizer.decode(output.squeeze().int())

        assert decoded.replace(" ", "") == "ACGT#"

    def test_every_vocabulary_id_decodes_to_its_nucleotide(self):
        """Each of the six vocabulary ids decodes to its own token."""
        tokenizer = _load_tokenizer()

        for token, token_id in tokenizer.get_vocab().items():
            assert tokenizer.decode(torch.tensor([token_id])).replace(" ", "") == token


class TestMegadnaCheckpointSelection:
    """WR-02: each family member selects its intended checkpoint file.

    The old branch chain tested ``m in "megaDNA_updated"`` (the loop member
    as a substring of a literal), so every explicit phage member fell
    through to the 145M default -- 78M/277M/ecoli names silently loaded the
    wrong .pt (or hit FileNotFoundError). Expected files verified against
    the live repo listings: lingxusb/megaDNA_updated ships only
    megaDNA_phage_145M.pt; lingxusb/megaDNA_variants ships
    megaDNA_phage_78M.pt + megaDNA_phage_277M.pt;
    lingxusb/megaDNA_finetuned ships megaDNA_phage_ecoli_finetuned.pt.
    """

    @pytest.mark.parametrize(
        ("model_name", "expected_checkpoint"),
        [
            ("megaDNA_updated", "megaDNA_phage_145M.pt"),
            ("lingxusb/megaDNA_updated", "megaDNA_phage_145M.pt"),
            ("megaDNA_variants", "megaDNA_phage_78M.pt"),
            ("lingxusb/megaDNA_variants", "megaDNA_phage_78M.pt"),
            ("megaDNA_finetuned", "megaDNA_phage_ecoli_finetuned.pt"),
            ("megaDNA_phage_145M", "megaDNA_phage_145M.pt"),
            ("megaDNA_phage_78M", "megaDNA_phage_78M.pt"),
            ("megaDNA_phage_277M", "megaDNA_phage_277M.pt"),
            ("megaDNA_phage_ecoli_finetuned", "megaDNA_phage_ecoli_finetuned.pt"),
        ],
    )
    def test_member_selects_its_intended_checkpoint(self, model_name, expected_checkpoint):
        """The handler torch.loads <snapshot>/<intended checkpoint>."""
        loaded_paths = []

        def fake_torch_load(path, *args, **kwargs):
            loaded_paths.append(path)
            return object()

        with (
            patch("torch.load", side_effect=fake_torch_load),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/snapshot", None),
            ),
        ):
            result = _handle_megadna_models(model_name, "huggingface", None)

        assert result is not None, f"family member did not match: {model_name}"
        assert loaded_paths == [f"/snapshot/{expected_checkpoint}"]
