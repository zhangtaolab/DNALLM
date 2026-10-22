# 261022-tta — Registry task-type audit (owner-ordered, all 236 entries)

**Scope:** verify every registry model's task_type. Lab models (zhangtaolab/*, 189) checked
against ModelScope model code (config.json architectures/num_labels/id2label/problem_type);
third-party + plantcad checked structurally and by hub probing.

## Findings & fixes (commit f94c50f)

1. Six PlantCAD2 on/off entries: task_type 'classification' (out of vocabulary, unloadable
   by name) → **binary** (their own num_labels=2 / label_names ["off","on"]). Phase-24 OQ-1
   strict pin retired → six-entry loadable pin.
2. Three PlantCAD2 cell-type entries: label_names [] (means zero labels, not unrecorded)
   → **null**; the 92 real names are unmeasurable (plantcad/* repos are HF-gated, 404
   without auth, absent from ModelScope).
3. New permanent guard test (test_registry.py): vocabulary + head-structure + no-empty-
   label-lists + pretrained-foundation-only, over all entries.
4. Everything else clean: all 49 pretrained mask/generation assignments verified correct
   by architecture class; token/regression conventions hold; 42 regression entries all
   single-element-wrapped labels.

## Lab sweep (189 zhangtaolab/* vs ModelScope config.json)

- **task_type: registry correct 189/189.** Only 2 apparent conflicts are hub-side upload
  artifacts: plant-nucleotide-transformer-{6mer,singlebase} BASE repos carry a leftover
  promoter-finetune config (EsmForSequenceClassification, id2label Not_promoter/promoter).
  Registry mask (fine-tune base role) stands — hub-side re-upload recommended.
- num_labels: agree wherever the hub records one.
- label_names: 44 repos carry template junk on the hub (LABEL_0/1/2, "0/1/2",
  Not_modification/modification, Not_OCRs/OCRs, underscore variants); registry curated
  names are better and ORDER-consistent everywhere — registry stands; hub cleanup optional.
- DNAGemma family: hub configs have num_labels/id2label null (upload omitted head info) —
  no conflicting evidence; registry binary(2) stands; hub-side completion recommended.
- Hosting: 4 models absent from ModelScope, present on HF (plant-dnamamba-6mer-{conservation,
  H3K27ac,H3K4me3,H3K27me3}, each HTTP 200 on HF) — real entries, HF-only; loading by name
  with default source=modelscope would fail. Owner decision pending: upload to ModelScope
  or accept HF-only.

## Hub-side action list (lab, outside this repo)

1. Re-upload clean base configs for the two NT base repos.
2. Fill num_labels/id2label on DNAGemma repos.
3. Optional: id2label cleanup on ~44 finetuned repos.
4. Decide ModelScope hosting for the 4 HF-only models.
