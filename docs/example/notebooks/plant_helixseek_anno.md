---
notebook: example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
sync_check: true
---

# PlantHelixSeek Gene-Structure Annotation

This tutorial demonstrates gene-structure annotation over an *Arabidopsis thaliana* showcase locus with the `zhangtaolab/PlantHelixSeek-Anno` checkpoint: an 8192 bp / 4096 bp both-strand sliding-window scan through the dnallm public API, the upstream B<->L label permutation for minus-strand windows, the frozen middle-region stitching rule, an argmax BILOU decode to predicted CDS segments, GFF3 emission, and exon-level plus nucleotide-level agreement checks against the TAIR10 gene annotation.

All results shown in the executed notebook are computed on one illustrative locus (Chr1:5100001-5300000, plus a 20 kb intergenic negative-control window) selected by a recorded methodology — they demonstrate the workflow on an illustrative locus, not genome-wide accuracy.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb){ .md-button }

The committed notebook carries its executed outputs, including the embedded gene-model diagrams comparing predicted CDS segments against the TAIR10 truth gene models (rendered by the GitHub notebook viewer).

## Prerequisites

Install DNALLM with the flash-linear-attention kernels PlantHelixSeek needs:

```bash
uv pip install -e '.[base,fla]'
```

The scan runs on a CUDA GPU (the eager-attention route through dnallm processes one 8192-token window at a time, ~13 GB peak per forward; see the notebook's provenance cell). Unlike the CRE sibling, no `bedtools` install is needed — every Anno metric is computed in-process.

## Environment Guard

The notebook's first code cell hard-fails with a `RuntimeError` when `flash-linear-attention` is absent — without the fla kernels the checkpoint's remote code silently falls back to a non-KDA path that produces positionally-dead outputs, so absence is an error rather than a degradation. The cell also prints the exact `transformers_version=`, `torch_version=` and `fla_version=` the run used.

## Load the Checkpoint

The repo id and the 17-BILOU label order come from the packaged registry (`dnallm/models/model_info.yaml`); the load goes through the generic dnallm dispatch with ModelScope as the source. The token task asks for per-base logits — one 17-way distribution per sequence position:

```python
task_config = TaskConfig(
    task_type=task["task_type"],
    num_labels=task["num_labels"],
    label_names=task["label_names"],
    threshold=task["threshold"],
)
model, tokenizer = load_model_and_tokenizer(REPO_ID, task_config, source="modelscope")
```

## Load the Showcase Data

The committed FASTA carries its genomic interval in the header (the genomic offset is parsed from it, never hardcoded twice), and the locus sequence is fetched with the validated helper from `dnallm.utils`:

```python
sequence = fetch_sequence(str(fasta_path), "Chr1", 1, locus_length)
```

The TAIR10 truth slice is read directly from the committed GFF3 — verbatim rows in source order, never re-sliced, never merged, never re-sorted. CDS rows whose first `Parent=` value names an mRNA are the truth segments; `mRNA` rows provide the per-gene spans.

## Both-Strand Sliding Scan

```python
window, stride, batch_size = 8192, 4096, 1
margin = (window - stride) // 2

scan_starts, scan_cores = plan_windows(locus_length)
```

Each 8192 bp window is forwarded alone (batch 1) under `torch.no_grad()` — mandatory: an autograd graph over the 8192-token eager-attention window OOMs the box. The checkpoint emits one logit row per input token plus a leading BOS and trailing EOS position (measured shape `(1, 8194, 17)`), so base `i` of the window reads `logits[0, 1 + i, :]`; per-base argmax labels are stitched directly into each window's core slice:

```python
plus_labels = np.zeros(locus_length, dtype=np.int64)
plus_written = np.zeros(locus_length, dtype=bool)
for s, (c0, c1) in zip(scan_starts, scan_cores):
    labels = predict_window_labels(sequence[s : s + window])
    plus_labels[c0:c1] = labels[c0 - s : c1 - s]  # stitch directly into the core
    plus_written[c0:c1] = True
```

## Stitching

The first window contributes `[0, 6144)`; middle windows contribute `[start+2048, start+6144)`; when the locus end is not on the 4096 grid, a tail window anchored at `len-8192` contributes `[start+2048, len)` and wins where its core overlaps the previous core (it is processed last). Predictions are stitched directly — never probability averaging — and full coverage of the locus is asserted.

## The B<->L Permutation on the Minus Strand

The same windows are reverse-complemented (`dnallm.utils.sequence.reverse_complement`) and forwarded again: the model now reads the minus strand in its own 5'-to-3' direction. Mapping the labels back to plus-strand coordinates takes the upstream 17-element permutation (transcribed from PlantHelixSeek `predict_genome_multigpu.py:97-101`) followed by reversing the label vector:

```python
_B_SWAP_L = [0, 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 16]
swap = np.array(_B_SWAP_L, dtype=np.int64)
```

```python
minus_labels = np.zeros(locus_length, dtype=np.int64)
minus_written = np.zeros(locus_length, dtype=bool)
for s, (c0, c1) in zip(scan_starts, scan_cores):
    sub = reverse_complement(sequence[s : s + window])  # both flags default True
    labels = predict_window_labels(sub)
    labels = swap[labels][::-1]  # B<->L swap, then reverse to plus-strand coordinates
    minus_labels[c0:c1] = labels[c0 - s : c1 - s]
    minus_written[c0:c1] = True
```

Plus- and minus-strand label tracks stay separate: a predicted segment's strand is the track it was decoded from.

## BILOU Decode

A predicted CDS segment is a maximal run of consecutive positions whose argmax label is in the CDS family {B-CDS, I-CDS, L-CDS, U-CDS}:

```python
predicted_segments = {strand: decode_cds_segments(label_tracks[strand]) for strand in ("+", "-")}
```

(The upstream viterbi+ORF decoder is deliberately not ported — the recorded agreement values were calibrated on this argmax rule.)

## GFF3 Emission

Segments are lifted to genomic coordinates with the FASTA-header offset (predictions only — the truth rows are already genomic), converted to 1-based closed coordinates, and written as a 9-column GFF3 under `outputs/`; the file is then re-parsed and structurally validated in-notebook (9 columns, integer coordinates inside the locus, strand within the {+, -, .} set):

```python
gff3_path = outputs_dir / "predicted_cds_segments.gff3"
gff3_path.write_text("\n".join(gff3_rows) + "\n", encoding="utf-8")
```

## Exon-Level Agreement with TAIR10

Each predicted segment is matched to a truth CDS row on the same strand with reciprocal overlap >= 0.5, greedily in predicted-start order; the pooled exon-level F1 counts TP = matched truth rows, FP = unmatched predicted segments, FN = unmatched truth rows:

```python
tp = len(matched_truth)
fp = len(predicted_segments["+"]) + len(predicted_segments["-"]) - tp
fn = len(truth_cds) - tp
pooled_denom = 2 * tp + fp + fn
exon_f1_value = (2 * tp / pooled_denom) if pooled_denom else 1.0

print(f"exon_f1={exon_f1_value:.4f}")
print(f"genes_above_floor={genes_above_floor_value}")
```

The notebook also prints per-gene exon F1 (genes = mRNAs with CDS rows in the slice; the floor counts genes with per-gene exon-F1 >= 0.8) and nucleotide-level sensitivity / precision / F1 as `key=value` stream lines, plus a transparent observed-vs-contract comparison table parsed at runtime from the frozen selection contract ([selection.md](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_shared/data/selection.md)). The nightly test suite re-executes the whole notebook and asserts the tolerance bands from that same document:

| Metric | Tolerance band | Observed (selection run) |
|---|---|---|
| Anno gene models with exon-F1 >= 0.8 | >= 3 genes | 59 |
| Negative Anno genic-base fraction (intergenic) | [0.00, 0.1] | 0.0000 |

## Negative Control

A 20 kb intergenic window (Chr1:14953292-14973291, inside the largest fully feature-free TAIR10 gap in the scan region) is re-scanned with the same frozen rules; the expected behavior is near-zero predicted genic signal:

```python
genic_mask = (intergenic_tracks["+"] != 0) | (intergenic_tracks["-"] != 0)
neg_anno_fraction_value = float(genic_mask.mean())
print(f"neg_anno_fraction={neg_anno_fraction_value:.4f}")
```

## Related Tutorials

- [Binary Classification Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [PlantHelixSeek CRE Scan](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb)
