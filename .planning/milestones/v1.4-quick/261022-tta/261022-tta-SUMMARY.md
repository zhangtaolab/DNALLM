---
status: complete
quick_id: 261022-tta
date: 2026-10-22
---

# Registry task-type audit (owner-ordered, all 236 entries)

Full record: `task-type-audit.md` (same directory). Fixes: commit f94c50f;
audit record: commit b6c53a5.

## What shipped

- Six PlantCAD2 on/off entries corrected `classification` → **binary** (out-of-vocabulary
  task_type made them unloadable by name; their own num_labels=2 / label_names
  ["off","on"] is the evidence) — Phase-24 OQ-1's strict pin retired in favor of a
  six-entry loadable pin.
- Three PlantCAD2 cell-type entries: `label_names []` → **null** (empty list means zero
  labels, not unrecorded; the 92 real names are unmeasurable — plantcad/* repos are
  HF-gated, 404 without auth, absent from ModelScope).
- New permanent guard test (`tests/models/test_registry.py`): vocabulary +
  head-structure + no-empty-label-lists + pretrained-foundation-only, over all entries.
- Lab sweep 189/189: every zhangtaolab task_type verified correct against ModelScope
  config.json (the 2 apparent conflicts are hub-side upload artifacts); hub-side action
  list recorded (NT base re-uploads, DNAGemma head info, 4 HF-only hosting decision) —
  outside repo scope.
