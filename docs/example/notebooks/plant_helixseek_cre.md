---
notebook: example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
sync_check: true
---

# PlantHelixSeek CRE Sliding-Window Scan

This tutorial demonstrates cis-regulatory element (CRE) prediction over an *Arabidopsis thaliana* showcase locus with the `zhangtaolab/PlantHelixSeek-CRE` checkpoint: a 500 bp / 50 bp sliding-window scan through the dnallm public API, per-bin p(CRE) scoring, mean + 1.5 sigma peak calling, and a `bedtools jaccard` agreement check against the PlantDHS open-chromatin truth.

All results shown in the executed notebook are computed on one illustrative locus (Chr1:5100001-5300000, plus a 20 kb flanking negative-control window) selected by a recorded methodology — they demonstrate the workflow on an illustrative locus, not genome-wide accuracy.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb){ .md-button }

The committed notebook carries its executed outputs, including the embedded prediction-vs-truth track figures (rendered by the GitHub notebook viewer). The full-locus figure embeds an image/png copy alongside the compiled vega/vega-lite JSON, so it also renders in local JupyterLab/VS Code, and the notebook adds an owner-chosen illustrative zoom window (`Chr1:5220001-5260000`) — the confident-band p(CRE) track plus the official PlantDHS leaf DNase signal from the committed bedGraph — rendered with pygenometracks for display clarity. For the both-modality view on one aligned axis, see the [PlantHelixSeek Combined View](plant_helixseek_combined.md).

## Prerequisites

Install DNALLM with the flash-linear-attention kernels PlantHelixSeek needs:

```bash
uv pip install -e '.[base,fla]'
```

The jaccard step additionally requires [bedtools](https://bedtools.readthedocs.io) v2.31+ on `PATH`, and the scan runs on a CUDA GPU (the eager-attention route through dnallm has a batch-4 memory ceiling; see the notebook's provenance cell).

## Environment Guard

The notebook's first code cell hard-fails with a `RuntimeError` when `flash-linear-attention` is absent — without the fla kernels the checkpoint's remote code silently falls back to a non-KDA path that produces positionally-dead outputs, so absence is an error rather than a degradation. The cell also prints the exact `transformers_version=`, `torch_version=` and `fla_version=` the run used.

## Load the Checkpoint

The repo id and label order come from the packaged registry (`dnallm/models/model_info.yaml`); the load goes through the generic dnallm dispatch with ModelScope as the source:

```python
from dnallm.configuration.configs import TaskConfig
from dnallm.models.model import load_model_and_tokenizer

task_config = TaskConfig(
    task_type=task["task_type"],
    num_labels=task["num_labels"],
    label_names=task["label_names"],
    threshold=task["threshold"],
)
model, tokenizer = load_model_and_tokenizer(REPO_ID, task_config, source="modelscope")
```

## Load the Showcase Data

The committed FASTA carries its genomic interval in the header (the genomic offset is parsed from it, never hardcoded twice), and sequences are fetched with the validated helper from `dnallm.utils`:

```python
from dnallm.utils import fetch_sequence, gff1_to_half_open

sequence = fetch_sequence(str(fasta_path), "Chr1", 1, locus_length)
```

The PlantDHS truth rows are read directly from the committed GFF slice and converted to 0-based half-open genomic coordinates with `gff1_to_half_open`.

## Sliding-Window Scan

```python
window, stride, bin_width, batch_size = 500, 50, 50, 4

bin_scores = scan_bin_scores(sequence)
```

Each 500 bp window is scored p(CRE) = softmax(logits)[:, 1] under `torch.no_grad()` at batch 4, and each 50 bp bin gets the arithmetic mean of the class-1 probabilities of the windows covering it (the notebook asserts every bin is covered by at least one window).

## Peak Calling

```python
merge_gap, min_length = 50, 50
threshold = bin_scores.mean() + 1.5 * bin_scores.std()

peak_bins = call_peaks(bin_scores, threshold)
```

The threshold is calibrated on this locus's bin scores; above-threshold bin runs merge when separated by at most 50 bp and merged peaks shorter than 50 bp are dropped. Peaks are written as a coordinate-sorted narrowPeak (BED6 form, signalValue = mean bin score) in genomic coordinates under `outputs/`.

## Jaccard Against PlantDHS

```python
import subprocess

result = subprocess.run(
    ["bedtools", "jaccard", "-a", str(predicted_bed_path), "-b", str(truth_bed_path)],
    capture_output=True,
    text=True,
    check=True,
)
jaccard_value = float(result.stdout.strip().splitlines()[1].split("\t")[2])
print(f"jaccard={jaccard_value:.4f}")
```

## Negative Control

A 20 kb flanking low-signal window (Chr1:5351001-5371000) is re-scanned with the same frozen rules and scored at the CRE-locus-calibrated absolute threshold — the expected behavior is near-zero predicted CRE signal:

```python
flank_bin_scores = scan_bin_scores(flank_sequence)
flank_peak_bins = call_peaks(flank_bin_scores, threshold)
neg_cre_fraction_value = flank_peak_bases / flank_length
print(f"neg_cre_fraction={neg_cre_fraction_value:.4f}")
```

## Results and Tolerance Bands

The notebook prints its metrics as `key=value` stream lines (`jaccard=`, `neg_cre_fraction=`) and a transparent observed-vs-contract comparison table parsed at runtime from the frozen selection contract ([selection.md](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_shared/data/selection.md)). The nightly test suite re-executes the whole notebook and asserts the tolerance bands from that same document:

| Metric | Tolerance band | Observed (selection run) |
|---|---|---|
| CRE jaccard | [0.3, 1.00] | 0.3247 |
| Negative CRE peak-base fraction (flanking) | [0.00, 0.05] | 0.0325 |

## Related Tutorials

- [Binary Classification Fine-Tuning](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/finetune_binary/finetune_binary.ipynb)
- [Basic Inference](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/inference/inference.ipynb)
- [In-Silico Mutagenesis](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/in_silico_mutagenesis/in_silico_mutagenesis.ipynb)
