"""Zero-shot variant effect prediction (VEP) scoring core.

This module hosts the scoring core for zero-shot variant effect prediction
with DNA large language models. It answers the reviewer-raised alignment
objection (R1-3e-1) with an explicit same-slot evaluability rule: a variant
is scoreable only when the reference and alternate sequences tokenize to
lists of identical length that differ at exactly one token slot. Variants
that fail the rule come back as structured skip records (data, not
exceptions), so downstream reporting can account for the skipped fraction
instead of silently mixing tokenization-shift noise into variant effects.

Features:

1. ``align_variant`` — the same-slot evaluability rule: aligns a reference
   and alternate allele at a 0-based sequence position and reports the
   single differing token slot, or a machine-readable skip reason.
2. ``clm_log_likelihood`` — the causal-LM scoring kernel: the full-sequence
   causal log-likelihood used by the delta-log-likelihood paradigm.
3. ``mlm_slot_log_prob`` — the masked-LM scoring kernel: the log-probability
   of one target token id at one masked slot, used by the log-odds paradigm.

The scoring kernels are adaptations of the proven mutagenesis kernels
(``Mutagenesis.mlm_evaluate`` / ``Mutagenesis.clm_evaluate`` in
``dnallm/inference/mutagenesis.py``). The VCF-level driver
(``evaluate_vcf``), VCF coordinate conversion, and the CLI entry point
arrive with the next phase; this module deliberately ships only the
protocol rule and the kernels.

Example:
    >>> from dnallm.inference.vep import align_variant
    >>> result = align_variant("ACGTTGCA", 3, "T", "A", tokenizer)
    >>> result.evaluatable
    True
    >>> result.slot_index
    3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VariantAlignment:
    """Result of the same-slot evaluability check for one variant.

    Attributes:
        evaluatable: Whether the ref/alt tokenizations differ at exactly one
            token slot (the only case zero-shot scoring accepts).
        slot_index: Index of the single differing token slot; ``None`` when
            the variant is not evaluatable.
        ref_token_id: Token id of the reference allele at ``slot_index``;
            ``None`` when the variant is not evaluatable.
        alt_token_id: Token id of the alternate allele at ``slot_index``;
            ``None`` when the variant is not evaluatable.
        skip_reason: Machine-readable reason the variant was skipped
            ("length-changing allele", "multi-slot token difference", or
            "no change"); ``None`` when the variant is evaluatable.
    """

    evaluatable: bool
    slot_index: int | None
    ref_token_id: int | None
    alt_token_id: int | None
    skip_reason: str | None


def _tokenize_ids(tokenizer: Any, sequence: str) -> list[int]:
    """Tokenize a sequence to a flat list of token ids.

    Uses the shared tokenizer convention
    ``tokenizer(sequence, return_tensors="pt", add_special_tokens=True)``
    so slot indices line up with every other consumer of the same
    tokenization, and accepts both tensor and list ``input_ids`` returns
    (test stub tokenizers may return plain lists).

    Args:
        tokenizer: Hugging Face-style callable tokenizer.
        sequence: DNA sequence to tokenize.

    Returns:
        Flat list of token ids for the sequence.
    """
    encoding = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    ids = encoding["input_ids"][0]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


def align_variant(sequence: str, pos: int, ref: str, alt: str, tokenizer: Any) -> VariantAlignment:
    """Apply the same-slot evaluability rule to one variant.

    The rule (the protocol answer to reviewer R1-3e-1): a variant is
    scoreable by zero-shot DNA large language model scoring only when the
    tokenized reference and alternate sequences have the same length and
    differ at exactly ONE token slot. Anything else is returned as a skip
    record with a machine-readable reason — skips are reportable data, not
    exceptions.

    ``pos`` is a 0-based index into ``sequence``. The 1-based VCF coordinate
    conversion belongs to the next phase's ``evaluate_vcf`` driver, not to
    this kernel.

    Args:
        sequence: Reference DNA sequence.
        pos: 0-based position of the variant in ``sequence``.
        ref: Reference allele; must equal ``sequence[pos:pos + len(ref)]``.
        alt: Alternate allele.
        tokenizer: Hugging Face-style callable tokenizer.

    Returns:
        VariantAlignment describing the single differing token slot, or the
        skip reason.

    Raises:
        ValueError: If ``ref`` does not match the bases of ``sequence`` at
            ``pos`` (input-contract violation — distinct from the skip path).
    """
    found = sequence[pos : pos + len(ref)]
    if found != ref:
        raise ValueError(
            f"Reference allele '{ref}' does not match sequence at position {pos} (found '{found}')"
        )

    alt_sequence = sequence[:pos] + alt + sequence[pos + len(ref) :]

    ref_ids = _tokenize_ids(tokenizer, sequence)
    alt_ids = _tokenize_ids(tokenizer, alt_sequence)

    if len(ref_ids) != len(alt_ids):
        return VariantAlignment(
            evaluatable=False,
            slot_index=None,
            ref_token_id=None,
            alt_token_id=None,
            skip_reason="length-changing allele",
        )
    if ref_ids == alt_ids:
        return VariantAlignment(
            evaluatable=False,
            slot_index=None,
            ref_token_id=None,
            alt_token_id=None,
            skip_reason="no change",
        )
    diff_indices = [
        i
        for i, (ref_id, alt_id) in enumerate(zip(ref_ids, alt_ids, strict=True))
        if ref_id != alt_id
    ]
    if len(diff_indices) != 1:
        return VariantAlignment(
            evaluatable=False,
            slot_index=None,
            ref_token_id=None,
            alt_token_id=None,
            skip_reason="multi-slot token difference",
        )
    slot_index = diff_indices[0]
    return VariantAlignment(
        evaluatable=True,
        slot_index=slot_index,
        ref_token_id=ref_ids[slot_index],
        alt_token_id=alt_ids[slot_index],
        skip_reason=None,
    )


def get_model_device(model: Any) -> torch.device:
    """Get the device a model lives on.

    Mirrors ``Mutagenesis.get_model_device``
    (``dnallm/inference/mutagenesis.py:241-255``): prefer the model's own
    ``device`` attribute, fall back to the device of the first parameter,
    and finally assume CPU.

    Args:
        model: Torch model (or model-like object) to inspect.

    Returns:
        torch.device the model's tensors live on.
    """
    device: torch.device
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "parameters"):
        device = next(model.parameters()).device
    else:
        device = torch.device("cpu")
    return device


@torch.no_grad()
def clm_log_likelihood(model: Any, tokenizer: Any, sequence: str) -> float:
    """Full-sequence causal log-likelihood.

    Scoring formula::

        log P(sequence) = sum_t log P(token_t | tokens_<t)

    computed by one forward pass, shifting the logits left by one position
    so each position predicts its next token, taking the log-softmax, and
    gathering the log-probabilities of the actual token ids. This is the
    same scoring math as ``Mutagenesis.clm_evaluate``
    (``dnallm/inference/mutagenesis.py:311-347``), adapted into a pure
    single-sequence kernel.

    Causal-LM variant-effect paradigm: a variant is scored as
    ``clm_log_likelihood(model, tokenizer, alt_sequence) -
    clm_log_likelihood(model, tokenizer, ref_sequence)`` — the
    delta-log-likelihood consumed by the next phase's ``score_variant``.

    Args:
        model: Causal DNA large language model returning per-position
            logits (batch, seq_len, vocab) from a forward call.
        tokenizer: Hugging Face-style callable tokenizer.
        sequence: DNA sequence to score.

    Returns:
        The causal log-likelihood of the full sequence (a float <= 0).
    """
    device = get_model_device(model)
    toks = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = toks["input_ids"]
    outputs = model(**toks)
    logits = outputs.logits  # (1, L, V)

    # Shift for causal LM: predict token t given tokens < t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, L-1)
    return float(token_logps.sum().item())


@torch.no_grad()
def mlm_slot_log_prob(
    model: Any, tokenizer: Any, sequence: str, slot_index: int, token_id: int
) -> float:
    """Masked-slot log-probability of one target token.

    Scoring formula::

        log P(token_id | masked context)

    computed by masking position ``slot_index`` of the tokenized sequence
    with the tokenizer's mask token, running one forward pass, taking the
    log-softmax over the vocabulary at that slot, and reading off the
    log-probability of ``token_id``. This is the same mask-and-predict math
    as ``Mutagenesis.mlm_evaluate``
    (``dnallm/inference/mutagenesis.py:257-309``), restricted to the single
    slot a variant occupies.

    Masked-LM variant-effect paradigm: a variant is scored at the
    alignment slot reported by ``align_variant`` as
    ``mlm_slot_log_prob(model, tokenizer, sequence, slot_index, alt_token_id) -
    mlm_slot_log_prob(model, tokenizer, sequence, slot_index, ref_token_id)``
    — the log-odds of the alternate token against the reference token.

    Args:
        model: Masked DNA large language model returning per-position
            logits (batch, seq_len, vocab) from a forward call.
        tokenizer: Hugging Face-style callable tokenizer with a
            ``mask_token_id``.
        sequence: DNA sequence providing the context.
        slot_index: Token slot to mask (from ``align_variant``).
        token_id: Target token id whose log-probability to read.

    Returns:
        The log-probability of ``token_id`` at the masked slot (float <= 0).
    """
    device = get_model_device(model)
    toks = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    masked = toks["input_ids"].clone()
    masked[0, slot_index] = tokenizer.mask_token_id
    outputs = model(**{"input_ids": masked})
    logits = outputs.logits
    logp = torch.nn.functional.log_softmax(logits[0, slot_index], dim=-1)
    return float(logp[token_id].item())
