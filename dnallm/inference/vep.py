"""Zero-shot variant effect prediction (VEP) scoring core.

This module hosts the scoring core and the VCF-level driver for zero-shot
variant effect prediction with DNA large language models. It answers the
reviewer-raised alignment objection (R1-3e-1) with an explicit same-slot
evaluability rule: a variant is scoreable only when the reference and
alternate sequences tokenize to lists of identical length that differ at
exactly one token slot. Variants that fail the rule come back as structured
skip records (data, not exceptions), so downstream reporting can account
for the skipped fraction instead of silently mixing tokenization-shift
noise into variant effects.

Features:

1. ``align_variant`` — the same-slot evaluability rule: aligns a reference
   and alternate allele at a 0-based sequence position and reports the
   single differing token slot, or a machine-readable skip reason.
2. ``clm_log_likelihood`` — the causal-LM scoring kernel: the full-sequence
   causal log-likelihood used by the delta-log-likelihood paradigm.
3. ``mlm_slot_log_prob`` — the masked-LM scoring kernel: the log-probability
   of one target token id at one masked slot, used by the log-odds paradigm.
4. ``score_variant`` — one variant scored through either paradigm, guarded
   by the paradigm↔architecture mismatch check.
5. ``evaluate_vcf`` — the VCF-level driver: reads a (ClinVar-style) VCF
   through scikit-allel, applies the ClinVar label/review-status convention,
   builds uppercased reference windows, scores every evaluatable variant
   through the kernels, and reports per-variant deltas plus skip accounting
   plus AUROC/AUPRC through the metric registry.

The scoring kernels are adaptations of the proven mutagenesis kernels
(``Mutagenesis.mlm_evaluate`` / ``Mutagenesis.clm_evaluate`` in
``dnallm/inference/mutagenesis.py``). The ``dnallm-vep`` CLI entry point
(``dnallm/cli/vep.py``) wraps ``evaluate_vcf``.

Example:
    >>> from dnallm.inference.vep import align_variant
    >>> result = align_variant("ACGTTGCA", 3, "T", "A", tokenizer)
    >>> result.evaluatable
    True
    >>> result.slot_index
    3
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


# ---------------------------------------------------------------------------
# VCF-level driver (evaluate_vcf).
#
# Parsing is delegated wholesale to ``allel.read_vcf`` (gzip, header, INFO
# typing, ALT dimensionality — never a hand-rolled reader); this driver owns
# only the ClinVar label convention, coordinate conversion, window building,
# skip accounting, and metric reporting.
# ---------------------------------------------------------------------------

#: CLNSIG values mapped to the positive class (D-17 label whitelist).
_POSITIVE_LABELS = frozenset({"Pathogenic", "Likely_pathogenic"})

#: CLNSIG values mapped to the negative class (D-17 label whitelist).
_NEGATIVE_LABELS = frozenset({"Benign", "Likely_benign"})

#: First-token CLNREVSTAT values that satisfy the >=1 review-star floor.
#: scikit-allel's ``Number=.`` String parsing keeps only the first
#: comma-separated token of a value, which is exactly the granularity the
#: D-17 >=1-star floor needs: every ``criteria_provided*`` review status
#: carries at least one star and every ``no_assertion*`` carries none.
_STAR_FLOOR_TOKENS = frozenset({
    "criteria_provided",
    "reviewed_by_expert_panel",
    "practice_guideline",
})


@dataclass(frozen=True)
class ClinVarFilter:
    """The ClinVar cohort convention applied by ``evaluate_vcf`` (D-17).

    The applied convention is reported with every result — never a bare
    AUROC without its convention (ascertainment-bias comparability, Pitfall 7).

    Attributes:
        variant_type: CLNVC value kept for scoring (SNVs only; indels are
            excluded here AND independently skipped by the same-slot rule
            when annotations disagree).
        positive_labels: CLNSIG values mapped to label 1.
        negative_labels: CLNSIG values mapped to label 0.
        star_floor: Minimum review stars. With ``allel``'s first-token
            granularity this is exact for the default floor of 1; floors
            above 1 cannot distinguish 1-star from 2-star
            ``criteria_provided`` rows.
    """

    variant_type: str = "single_nucleotide_variant"
    positive_labels: frozenset[str] = _POSITIVE_LABELS
    negative_labels: frozenset[str] = _NEGATIVE_LABELS
    star_floor: int = 1


@dataclass(frozen=True)
class VepVariantRecord:
    """One per-allele scoring outcome.

    Attributes:
        chrom: Chromosome name as written in the VCF.
        pos: 1-based VCF coordinate of the record.
        ref: Reference allele.
        alt: Alternate allele (one record per ALT of a multi-allelic row).
        label: Cohort label (1 positive / 0 negative); ``None`` cannot
            occur for considered records — the convention filter assigns
            labels before scoring.
        delta: The paradigm delta score; ``None`` when skipped.
        skip_reason: The same-slot skip reason; ``None`` when scored.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    label: int | None
    delta: float | None
    skip_reason: str | None


@dataclass(frozen=True)
class VepResult:
    """The full outcome of one ``evaluate_vcf`` run.

    Attributes:
        records: Per-allele records (one per non-empty ALT of every
            convention-passing row).
        skip_counts: Same-slot skip reasons -> counts (the alignment
            channel; convention exclusions live in ``convention``).
        evaluated: Number of records scored.
        skipped: Number of records skipped by the alignment rule.
        skip_fraction: ``skipped / considered`` (1.0 when nothing was
            scorable — reported as a finding, never hidden).
        metrics: ``{"AUROC": ..., "AUPRC": ...}`` through the metric
            registry, or ``None`` when nothing was scorable or only one
            label class is present (no fabricated AUROC).
        convention: The applied ClinVar convention block, including
            exclusion counts, so every AUROC ships with its convention.
    """

    records: list[VepVariantRecord]
    skip_counts: dict[str, int]
    evaluated: int
    skipped: int
    skip_fraction: float
    metrics: dict[str, float] | None
    convention: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return {
            "records": [
                {
                    "chrom": r.chrom,
                    "pos": r.pos,
                    "ref": r.ref,
                    "alt": r.alt,
                    "label": r.label,
                    "delta": r.delta,
                    "skip_reason": r.skip_reason,
                }
                for r in self.records
            ],
            "skip_counts": dict(self.skip_counts),
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "skip_fraction": self.skip_fraction,
            "metrics": dict(self.metrics) if self.metrics is not None else None,
            "convention": self.convention,
        }


def _as_str(value: Any) -> str:
    """Coerce one ``allel`` string cell to ``str`` (bytes-safe)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _field_cell(callset: dict, field: str, i: int) -> str:
    """Return row ``i`` of an ``allel`` string field, ``""`` when absent."""
    values = callset.get(field)
    if values is None or i >= len(values):
        return ""
    return _as_str(values[i])


def _load_reference(reference: str | Path | Mapping[str, str]) -> dict[str, str]:
    """Load the reference genome for window building.

    Args:
        reference: Either a mapping of chromosome name -> sequence (case
            preserved; windows are uppercased later), or a path to a FASTA
            file (plain or gzip-compressed) parsed here. Parsing FASTA is
            deliberately simple record/header concatenation — the heavy
            parsing in this module belongs to VCF, which is delegated to
            scikit-allel.

    Returns:
        Mapping of FASTA header name (up to the first whitespace) ->
        sequence string with case preserved (soft-masked input stays
        lowercase until window building uppercases it).

    Raises:
        ValueError: If ``reference`` is a path that does not exist or a
            FASTA containing no records.
    """
    if isinstance(reference, Mapping):
        return dict(reference)
    path = Path(reference)
    if not path.is_file():
        raise ValueError(f"Reference FASTA not found at '{path}'.")
    opener = gzip.open if str(path).endswith(".gz") else open
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name = line[1:].split()[0] if line[1:].split() else ""
                chunks = []
            elif name is not None:
                chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks)
    if not sequences:
        raise ValueError(f"Reference FASTA '{path}' contains no sequences.")
    return sequences


def _resolve_chromosome(sequences: Mapping[str, str], chrom: str) -> str:
    """Resolve a VCF CHROM name against the reference's naming style.

    Tries the exact name first, then the ``chr``-prefixed and
    ``chr``-stripped variants (ClinVar writes ``22``; UCSC references name
    it ``chr22``).

    Args:
        sequences: Reference chromosome -> sequence mapping.
        chrom: CHROM value from the VCF.

    Returns:
        The reference key matching ``chrom``.

    Raises:
        ValueError: If no reference chromosome matches (untrusted input
            surfaced at the driver boundary, never a bare KeyError).
    """
    candidates = [chrom]
    if not chrom.startswith("chr"):
        candidates.append(f"chr{chrom}")
    elif chrom.startswith("chr"):
        candidates.append(chrom[3:])
    for candidate in candidates:
        if candidate in sequences:
            return candidate
    raise ValueError(
        f"VCF chromosome '{chrom}' not found in the reference "
        f"(available: {sorted(sequences)}). Check the reference assembly."
    )


def _build_window(ref_seq: str, pos0: int, context_window: int) -> tuple[str, int]:
    """Build the scoring window around a variant.

    The window is ALWAYS uppercased before alignment/scoring: lowercase
    soft-masked reference FASTA is common, and lowercase k-mers tokenize
    to ``<unk>`` on real tokenizers, making ref/alt look identical and
    silently skipping variants as "no change" (empirically verified trap).

    Args:
        ref_seq: Full chromosome sequence (case preserved).
        pos0: 0-based position of the variant in ``ref_seq``.
        context_window: Reference bases kept on each side (symmetric;
            clipped at chromosome edges).

    Returns:
        Tuple of (uppercased window, 0-based position of the variant
        inside the window). Ref and alt sequences are always derived from
        this ONE window, so they share identical left context.
    """
    start = max(0, pos0 - context_window)
    end = min(len(ref_seq), pos0 + 1 + context_window)
    window = ref_seq[start:end].upper()
    return window, pos0 - start


def evaluate_vcf(
    model: Any,
    tokenizer: Any,
    vcf_path: str | Path,
    reference: str | Path | Mapping[str, str],
    *,
    paradigm: str = "mlm",
    context_window: int = 200,
    clnsig_filter: ClinVarFilter | None = None,
    alt_number: int = 4,
    output_dir: str | Path | None = None,
) -> VepResult:
    """Score zero-shot variant effects for every cohort variant in a VCF.

    Protocol: the VCF is parsed by ``allel.read_vcf`` (field list includes
    CLNSIG/CLNREVSTAT/CLNVC; ``alt_number`` is deliberately sized >= 4
    because allel's default 3 silently truncates 4+-allelic rows). The
    ClinVar cohort convention (D-17) is applied first — SNVs only
    (CLNVC), P/LP vs B/LB labels (strict CLNSIG whitelist; VUS,
    conflicting, and novel strings are excluded, not guessed), >= 1 review
    star — and every convention-passing allele is scored through the
    landed kernels. 1-based VCF POS converts to the 0-based
    ``align_variant`` contract (``pos0 = POS - 1``). Skip-as-data remains
    reserved for the same-slot alignment rule alone; convention exclusions
    are counted separately in the convention block.

    Scoring by paradigm:

    - ``"mlm"``: ``delta = mlm_slot_log_prob(alt_id) -
      mlm_slot_log_prob(ref_id)`` at the alignment slot (log-odds).
    - ``"clm"``: ``delta = clm_log_likelihood(alt_window) -
      clm_log_likelihood(ref_window)`` (delta-log-likelihood).

    Args:
        model: DNA large language model (per-position logits).
        tokenizer: Hugging Face-style callable tokenizer.
        vcf_path: Path to the (optionally gzipped) VCF.
        reference: FASTA path (plain or ``.gz``) or a chromosome ->
            sequence mapping.
        paradigm: ``"mlm"`` or ``"clm"`` (default mirrors
            ``VepConfig.paradigm``).
        context_window: Reference bases on each side of the variant
            (default mirrors ``VepConfig.context_window``).
        clnsig_filter: The ClinVar cohort convention; ``None`` uses the
            D-17 defaults (see ``ClinVarFilter``).
        alt_number: ALT array width passed to ``allel.read_vcf``; keep
            >= 4 so 4+-allelic rows are not truncated.
        output_dir: Optional directory for a deterministic
            ``vep_result.json`` write (no path is ever derived from VCF
            record fields).

    Returns:
        ``VepResult`` with per-variant records, skip accounting, registry
        AUROC/AUPRC (or ``None`` with ``skip_fraction`` 1.0 when nothing
        was scorable), and the applied convention block.

    Raises:
        ValueError: If scikit-allel is not installed, the VCF cannot be
            read, a VCF chromosome is missing from the reference, or a
            REF allele contradicts the reference sequence.
    """
    try:
        import allel
    except ImportError as exc:
        raise ValueError(
            "evaluate_vcf requires scikit-allel, which is not installed. "
            "Install it with: uv pip install 'scikit-allel>=1.3.13,<2'"
        ) from exc

    if clnsig_filter is None:
        clnsig_filter = ClinVarFilter()
    if paradigm not in ("mlm", "clm"):
        raise ValueError(f"Unknown paradigm '{paradigm}'; expected 'clm' or 'mlm'.")

    try:
        callset = allel.read_vcf(
            str(vcf_path),
            fields=[
                "variants/CHROM",
                "variants/POS",
                "variants/ID",
                "variants/REF",
                "variants/ALT",
                "variants/CLNSIG",
                "variants/CLNREVSTAT",
                "variants/CLNVC",
            ],
            alt_number=alt_number,
        )
    except Exception as exc:
        raise ValueError(f"Failed to read VCF '{vcf_path}' with scikit-allel: {exc}") from exc

    sequences = _load_reference(reference)

    skip_counts = {"length-changing allele": 0, "multi-slot token difference": 0, "no change": 0}
    exclusion_counts = {"non_snv_clnvc": 0, "unlabeled_clnsig": 0, "below_star_floor": 0}
    clnrevstat_counts: dict[str, int] = {}
    records: list[VepVariantRecord] = []

    chroms = callset.get("variants/CHROM") if callset is not None else None
    if chroms is None:
        n_rows = 0
    else:
        n_rows = len(chroms)

    for i in range(n_rows):
        chrom = _as_str(chroms[i])
        clnvc = _field_cell(callset, "variants/CLNVC", i)
        if clnvc != clnsig_filter.variant_type:
            exclusion_counts["non_snv_clnvc"] += 1
            continue
        clnsig = _field_cell(callset, "variants/CLNSIG", i)
        if clnsig in clnsig_filter.positive_labels:
            label = 1
        elif clnsig in clnsig_filter.negative_labels:
            label = 0
        else:
            exclusion_counts["unlabeled_clnsig"] += 1
            continue
        revstat = _field_cell(callset, "variants/CLNREVSTAT", i)
        clnrevstat_counts[revstat] = clnrevstat_counts.get(revstat, 0) + 1
        if revstat not in _STAR_FLOOR_TOKENS:
            exclusion_counts["below_star_floor"] += 1
            continue

        pos1 = int(callset["variants/POS"][i])
        ref = _as_str(callset["variants/REF"][i])
        chrom_key = _resolve_chromosome(sequences, chrom)
        ref_seq = sequences[chrom_key]
        pos0 = pos1 - 1  # 1-based VCF POS -> 0-based align_variant contract
        window, local_pos = _build_window(ref_seq, pos0, context_window)

        alt_row = callset["variants/ALT"][i]
        for alt_cell in alt_row:
            alt = _as_str(alt_cell)
            if not alt or alt == ".":
                continue
            try:
                alignment = align_variant(window, local_pos, ref, alt, tokenizer)
            except ValueError as exc:
                raise ValueError(
                    f"REF/reference mismatch at {chrom}:{pos1} (REF={ref}, ALT={alt}): {exc}"
                ) from exc
            if not alignment.evaluatable:
                skip_counts[alignment.skip_reason] += 1
                records.append(
                    VepVariantRecord(
                        chrom=chrom,
                        pos=pos1,
                        ref=ref,
                        alt=alt,
                        label=label,
                        delta=None,
                        skip_reason=alignment.skip_reason,
                    )
                )
                continue
            if paradigm == "mlm":
                delta = mlm_slot_log_prob(
                    model, tokenizer, window, alignment.slot_index, alignment.alt_token_id
                ) - mlm_slot_log_prob(
                    model, tokenizer, window, alignment.slot_index, alignment.ref_token_id
                )
            else:  # "clm" — delta-log-likelihood over the substituted window
                alt_window = window[:local_pos] + alt + window[local_pos + len(ref) :]
                delta = clm_log_likelihood(model, tokenizer, alt_window) - clm_log_likelihood(
                    model, tokenizer, window
                )
            records.append(
                VepVariantRecord(
                    chrom=chrom,
                    pos=pos1,
                    ref=ref,
                    alt=alt,
                    label=label,
                    delta=float(delta),
                    skip_reason=None,
                )
            )

    considered = len(records)
    evaluated = sum(1 for r in records if r.delta is not None)
    skipped = considered - evaluated
    skip_fraction = 1.0 if considered == 0 else skipped / considered

    labels = [r.label for r in records if r.delta is not None]
    deltas = [r.delta for r in records if r.delta is not None]
    metrics: dict[str, float] | None = None
    if evaluated > 0 and len(set(labels)) == 2:
        from ..tasks.metric_registry import resolve

        metrics = {
            "AUROC": float(resolve("AUROC")(labels, deltas)),
            "AUPRC": float(resolve("AUPRC")(labels, deltas)),
        }

    convention = {
        "cohort": "ClinVar-style labels (CLNSIG/CLNREVSTAT/CLNVC)",
        "variant_type": clnsig_filter.variant_type,
        "labels": (
            f"{sorted(clnsig_filter.positive_labels)}=1 vs "
            f"{sorted(clnsig_filter.negative_labels)}=0"
        ),
        "star_floor": clnsig_filter.star_floor,
        "excluded": (
            "non-SNV CLNVC, CLNSIG outside the label whitelist "
            "(VUS/conflicting/novel), CLNREVSTAT below the star floor"
        ),
        "exclusion_counts": exclusion_counts,
        "clnrevstat_counts": dict(sorted(clnrevstat_counts.items())),
        "rows_read": n_rows,
    }

    result = VepResult(
        records=records,
        skip_counts=skip_counts,
        evaluated=evaluated,
        skipped=skipped,
        skip_fraction=skip_fraction,
        metrics=metrics,
        convention=convention,
    )

    if output_dir is not None:
        out_path = Path(output_dir) / "vep_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)

    return result
