---
notebook: example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
sync_check: true
---

# PlantHelixSeek Combined CRE + Anno View

This showcase notebook puts both PlantHelixSeek modalities on one display window: the CRE model's confident-band p(CRE) prediction, the official PlantDHS leaf DNase signal (committed per-bin bedGraph), the Anno model's predicted transcripts, and the TAIR10 truth gene models (committed pre-converted GTF) — all on one aligned genomic axis rendered with [pygenometracks](https://pygenometracks.readthedocs.io). It is a display view for clarity: the full-locus scans, metrics and tolerance bands in the [CRE scan](plant_helixseek_cre.md) and [gene annotation](plant_helixseek_anno.md) showcase notebooks remain the authoritative claims.

All results shown in the executed notebook are computed on one illustrative display region (an owner-chosen window extended by a 5 kb flank) — they demonstrate the workflow on an illustrative locus, not genome-wide accuracy.

## Full Notebook

[:octicons-book-24: View Full Notebook](https://github.com/zhangtaolab/DNALLM/blob/main/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb){ .md-button }

The committed notebook carries its executed outputs; the six-track combined figure embeds as a PNG rendered with pygenometracks (titles on), so it displays in every viewer.

## Prerequisites

Install DNALLM with the flash-linear-attention kernels PlantHelixSeek needs, plus the notebook extra (jupyter + pygenometracks):

```bash
uv pip install -e '.[base,fla,notebook]'
```

The scans run on a CUDA GPU through the eager-attention dnallm route.

## Display Region

The notebook renders one owner-chosen display window, `Chr1:5220001-5260000`, extended by a 5 kb downstream flank (through `Chr1:5265000`) so edge-crossing genes complete. Both models scan only this 45 kb region: the CRE model runs its frozen 500 bp / 50 stride / 50 bp bin sliding scan at batch 4, and the Anno model runs its frozen 8192 bp / 4096 stride both-strand scan at batch 1 (BOS offset token alignment and the upstream B<->L permutation on the minus strand).

## The Combined Figure

Six titled tracks on one genomic axis, top to bottom: the CRE predicted p(CRE) confident band (only bins with p(CRE) >= 0.5 drawn — a disclosed display filter), the PlantDHS leaf DNase signal, the predicted transcripts on the plus and minus strands (blue), and the TAIR10 truth gene models (plus strand green, minus strand red). Predicted transcripts come from a presentation-only island decode of the per-base labels; transcripts spanning less than 100 bp are not drawn.

## Related Tutorials

- [PlantHelixSeek CRE Scan](plant_helixseek_cre.md)
- [PlantHelixSeek Gene Annotation](plant_helixseek_anno.md)
