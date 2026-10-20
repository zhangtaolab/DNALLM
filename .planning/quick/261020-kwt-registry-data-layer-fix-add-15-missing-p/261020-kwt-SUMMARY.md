---
phase: 261020-kwt
plan: 01
type: quick
subsystem: models-registry
tags: [registry, model-info, lineage, data-fix, tests]
status: complete
requirements: [KWT-01, KWT-02, KWT-03, KWT-04]
key-files:
  created:
    - tests/models/test_model_info_registry.py
  modified:
    - dnallm/models/model_info.yaml
    - dnallm/models/modeling_auto.py
decisions:
  - Corrected the plan's joined-title grep gate from bare `aDNA` to `whether aDNA` — bare
    `aDNA` is a substring of HyenaDNA/megaDNA (16 HEAD hits) and could never reach 0.
  - Lineage backfill done as scripted text insertion keyed on unique finetuned
    `model: "<repo>"` lines (164 added / 0 modified-deleted lines), never a yaml.dump
    round-trip — provenance comment block and the 23 pre-existing base_model values
    byte-unchanged.
metrics:
  duration: 6 min
  completed: 2026-10-20
actuals:
  tokens: 5473
  tasks: 2
  commits: 2
plan_head_before: 8e2cf9862df3b95810b796ad685d7a8dbf5d5247
plan_head_after: f0b8302e371c0820e8aaf26d0674f053682e2ebb
---

# Quick Task 261020-kwt: Registry data-layer fix — 15 variant entries, task-class strings, base_model lineage Summary

**One-liner:** Registered the 15 missing PDLLMs tokenizer-variant pretrained checkpoints (34→49), corrected 3 task-class strings (Plant DNAGPT → Causal; EVO-1/2 misspelling) and 2 tRNA title typos, and backfilled base_model lineage for all 164 finetuned entries that lacked it — enforced by 9 new no-network registry-integrity tests.

## What Was Built

### KWT-01 — 15 pretrained variant entries (Task 1, commit 4c3393c)

Inserted after the Plant DNAModernBert entry (six BPE family entries stay grouped first),
mirroring the Plant DNABERT entry style exactly (name / model / task.describe /
task.task_type, double quotes, 2-space indent); each describe names its tokenizer variant:

- mask (5): plant-dnabert-6mer, plant-dnabert-singlebase, plant-nucleotide-transformer-6mer,
  plant-nucleotide-transformer-singlebase, plant-dnamodernbert-singlebase
- generation (10): plant-dnagpt-6mer/singlebase, plant-dnagemma-6mer/singlebase,
  plant-dnamamba-2mer/3mer/4mer/5mer/6mer/singlebase

Pretrained section: 34 → **49 entries**. YAML diff is pure insertion (75 added / 0 deleted).

### KWT-02 — 3 task-class strings (Task 1, commit 4c3393c)

- `PRETRAIN_MODEL_MAPS["Plant DNAGPT"]` second element: `"Masked Language Model"` →
  `"Causal Language Model"` (checkpoint config is GPT2LMHeadModel / model_type gpt2,
  grounded at planning time).
- EVO-1 and EVO-2: misspelled `"Casual Language Model"` → `"Causal Language Model"`.

### KWT-04 — 2 tRNA titles (Task 1, commit 4c3393c)

tRNADetector and tRNAPointer MODEL_INFO titles: `"whether aDNA sequence"` →
`"whether a DNA sequence"` (space restored after the leading article).

### KWT-03 — base_model lineage backfill (Task 2, commit f0b8302)

All 187 finetuned entries now carry `base_model` (23 pre-existing + 164 backfilled).

**Derivation split (measured): 135 prefix / 2 repo-map / 27 name-map — total 164.**

- Tier 1 (135): longest pretrained repo id that is a prefix of the finetuned repo id at a
  `-` boundary, evaluated against the post-Task-1 49-entry set (zero multi-prefix ties;
  e.g. plant-dnabert-BPE-promoter → plant-dnabert-BPE,
  plant-dnamamba-singlebase-lncRNAs → plant-dnamamba-singlebase).
- Tier 2 (2): explicit alias map — zhangtaolab/tRNADetector → zhangtaolab/plant-dnamamba-singlebase,
  zhangtaolab/tRNAPointer → zhangtaolab/plant-nucleotide-transformer-singlebase.
- Tier 3 (27): finetuned-entry-name family prefix for short-alias repos — names starting
  "DNABERT-2 " → zhihan1996/DNABERT-2-117M, "Nucleotide Transformer " →
  InstaDeepAI/nucleotide-transformer-v2-100m-multi-species, "AgroNT " →
  InstaDeepAI/agro-nucleotide-transformer-1b (9 tasks x 3 families; "Plant Nucleotide
  Transformer ..." names correctly do not match this key).

**Underivable entries: none** (empty list — the derivation resolved all 164; no entry was
left without base_model and no value was guessed).

Applied by scripted text insertion keyed on the unique finetuned `model: "<repo>"` lines,
`base_model: "<id>"` placed directly after `model:` and before `task:` (PlantCAD2 /
PlantHelixSeek convention). Diff: 164 added / 0 deleted — the 23 pre-existing base_model
values are byte-unchanged and the PlantHelixSeek provenance comment block survived intact.

### Tests — tests/models/test_model_info_registry.py

Nine fast-leg, no-network tests following the test_plant_helixseek_registry.py pattern
(`REGISTRY_PATH = Path(dnallm.__file__).parent / "models" / "model_info.yaml"`,
yaml.safe_load with pytest.fail on YAMLError): pretrained count 49; all 15 variants pinned
with task_type + variant-mentioning name and describe; Plant DNAGPT → Causal Language
Model; misspelled task class absent from every PRETRAIN_MODEL_MAPS entry; tRNA titles
contain "a DNA sequence" and not the joined form; base_model present on every finetuned
entry against an empty DERIVATION_EXEMPT allowlist asserted equal to the actual missing
set (future entries without lineage fail loudly); every base_model resolves to a pretrained
id or the EXTERNAL_BASE_ALLOWLIST (the two kuleshov-group PlantCAD2 Medium/Large pointers,
pre-existing and untouched); 7 spot-pinned load-bearing mappings; no pretrained entry
carries base_model. The misspelled class and joined title form each live in exactly one
module-level test constant. TDD: both tasks written test-first and observed RED before the
data edits (5 failed → green; 164-entry missing set → green).

## Deviations from Plan

**1. [Rule 3 - blocking verify-command defect] Corrected the joined-title grep gate**

- **Found during:** Task 1 verify
- **Issue:** The plan's gate `[ "$(grep -c aDNA dnallm/models/modeling_auto.py)" = "0" ]`
  (and its stated baseline of 2) is wrong: bare `aDNA` is a substring of **HyenaDNA** and
  **megaDNA** — the pattern already matched 16 lines at HEAD and can never reach 0 while
  those families exist.
- **Fix:** Gate corrected to `grep -c "whether aDNA"` (baseline 2 at HEAD, lines 166/179 —
  exactly the two joined-form titles; 0 after the fix). The guarded invariant is unchanged
  and additionally enforced by test_trna_titles_read_a_dna_sequence, which uses the
  precise joined form `"aDNA sequence"` that cannot match HyenaDNA/megaDNA.
- **Files modified:** none (verification-only correction; documented here)

**2. [Plan-expectation note] Derivation split 135/2/27 vs expected 136/2/26**

- **Found during:** Task 2
- **Issue:** The plan's planning-time simulation expected tier split 136 prefix / 2
  repo-map / 26 name-map; the actual derivation over the real registry yields
  135 / 2 / 27.
- **Analysis:** 27 short-alias entries exist in the file (9 task blocks x 3 families:
  DNABERT-2, Nucleotide Transformer, AgroNT), and none of their repo ids has any
  pretrained prefix (all pretrained bases for those families live under different orgs:
  zhihan1996 / InstaDeepAI), so all 27 resolve via the name map. The planning arithmetic
  moved one entry between tiers; both splits sum to 164 with zero unresolved and zero
  ambiguity. No behavior difference — every derived value is identical either way for
  these families.
- **Files modified:** none (expectation correction only; measured split recorded above)

## Verification Evidence

- `.venv/bin/python -m pytest tests/models/test_model_info_registry.py
  tests/models/test_plant_helixseek_registry.py -v` — **12 passed**, no network.
- Inline invariant check (plan's verify command): pretrained resolves, missing == [],
  bad == [], finetuned == 187 — all assertions passed.
- Grep gates post-fix: `grep -c Casual dnallm/models/modeling_auto.py` = **0** (was 2);
  `grep -c "whether aDNA"` = **0** (was 2; corrected gate per Deviation 1).
- `ruff format --check` + `ruff check` on the new test file — clean (one PT018
  composite-assertion finding fixed during Task 1 by splitting the assertion).
- `tests/models/ --collect-only` — 475 tests collected cleanly (no collection regression).
- Registry invariants: pretrained == 49, finetuned == 187, base_model on 187/187 finetuned
  entries, all values resolve to pretrained ids or the two external PlantCAD2 pointers.

## Self-Check: PASSED

- tests/models/test_model_info_registry.py — FOUND
- Commit 4c3393c (Task 1) — ancestor of HEAD
- Commit f0b8302 (Task 2) — ancestor of HEAD
