---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 01
subsystem: finetune-peft
tags: [peft, ia3, lora, presets, trainer, finetune]
requires:
  - "Phase 10 config stubs (Ia3Config registered in load_config)"
  - "Phase 10 pyproject package-data glob dnallm/configuration/presets/*.yaml"
provides:
  - "TrainingConfig.peft_dry_run field + use_ia3 x use_qlora Pydantic rejection"
  - "Ia3Config at peft-0.21.1 parity (exclude_modules, fan_in_fan_out, task_type)"
  - "DNATrainer IA³ branch symmetric to LoRA (get_peft_model + preset auto-selection)"
  - "Packaged per-family PEFT presets (dnallm/configuration/presets/lora_targets.yaml)"
  - "D-03 dry-run validator and D-04 trainable-ratio guard in the trainer"
  - "Adapter-kind-agnostic reload naming in DNAInference (PeftModel.from_pretrained)"
affects:
  - "dnallm/finetune/trainer.py (LoRA branch shares the preset-resolution path)"
  - "dnallm/configuration/configs.py (HeadConfig custom_head/num_classes fixes)"
tech-stack:
  added: []  # zero new dependencies (peft >=0.14 existing)
  patterns:
    - "presets-table auto-selection (importlib.resources packaged YAML, two-tier family resolution)"
    - "requires_grad-tensor ratio guard (never parse print_trainable_parameters)"
    - "peft IA3Config pass-through filtered to peft's accepted dataclass fields (version-span safe)"
key-files:
  created:
    - dnallm/configuration/presets/lora_targets.yaml
    - tests/configuration/test_peft_presets.py
  modified:
    - dnallm/configuration/configs.py
    - dnallm/finetune/trainer.py
    - dnallm/inference/inference.py
    - tests/configuration/test_configs.py
    - tests/finetune/test_trainer.py
    - tests/finetune/test_trainer_real_model.py
    - CHANGELOG.md
decisions:
  - "LoRA and IA³ get separate target lists per family (lora_target_modules vs ia3_target_modules) — a single list would be wrong for one kind; peft's suffix matching makes union lists harmless but the split keeps each row principled"
  - "out_proj excluded from every Mamba-family LoRA row: peft 0.21.1 hard-rejects it on model_type=mamba (verified live on the pinned plant-dnamamba); IA³ keeps out_proj (verified allowed + measured 1.38e-3)"
  - "Ia3Config.task_type=SEQ_CLS mirrors LoraConfig so the classification head (the wrapper's 'score' / BertForSequenceClassification 'classifier') stays trainable — without it the head silently freezes"
  - "peft kwargs filtered through PEFT_IA3_FIELD_NAMES (dataclasses.fields of the installed peft IA3Config) so the peft 0.14-0.21 span never hits unexpected-kwarg TypeErrors"
  - "Family resolution is two-tier: longest name marker from the table against config._name_or_path, then config.model_type; both fail loud with 'set target_modules explicitly'"
  - "Roundtrip acceptance seeds both base-model loads identically: the pinned checkpoint ships no pooler/classifier weights, so those modules differ across loads unless seeded (verified max-diff 0.0 seeded vs 0.66 unseeded)"
metrics:
  duration: ~1h40m
  completed: 2026-10-09
estimate:
  tokens: 30000
  tasks: 3
actuals:
  tokens: 23000   # 91,284 diff chars / 4 over this lane's files
  tasks: 3
  commits: 3      # this lane's commits (3b644bd, e957e2c, d4e9b68); the on-disk
                  # ledger range spans 16 commits because sibling lanes B2-B5
                  # commit into the same shared tree concurrently
  plan_head_before: 99866ed195712e0233aaf07fd1a3139ff54412f0
  plan_head_after: d4e9b6847d65ea589cdacb3ecf824ab652b7f1a5
status: complete
---

# Phase 11 Plan 01: IA³ PEFT Presets Summary

IA³ fine-tuning landed symmetric to LoRA — real trainer branch with peft injection, per-family target-module presets derived from real model code with measured ratio bands, dry-run validator, requires_grad ratio guard, transformer+Mamba slow-lane acceptance, and an IA³-specific save/reload roundtrip.

## What Was Built

### Task 1 — IA³ end-to-end tracer (commit 3b644bd)

- `TrainingConfig.peft_dry_run: bool` (D-03) and a `model_validator(mode="after")` rejecting `use_ia3 x use_qlora` with a matchable message naming both fields (peft only raises at merge time — Pitfall 1).
- `Ia3Config` extended to peft-0.21.1 parity: `exclude_modules`, `fan_in_fan_out` (+ `task_type` mirroring LoraConfig, see decisions).
- `dnallm/finetune/trainer.py`: Phase-10 interim warn demolished; IA³ branch mirrors the LoRA branch (no k-bit prep — rejected combination); `use_lora x use_ia3` rejected at trainer init (the ctor kwarg is invisible to Pydantic — RESEARCH Q1 resolution); `peft_dry_run` popped from TrainingArguments; `remove_unused_columns` disabled for IA³ like LoRA.
- `dnallm/inference/inference.py`: reload seam strings renamed LoRA-only → "PEFT adapter" (mechanics untouched, diff confined to the 111-131 region).
- Tests: `TestIa3Wiring` (7 mocked tests incl. the config-load-time rejection ordering edge), interim-warn pair demolished, Ia3Config parity tests.

### Task 2 — Presets table + auto-selection + dry-run + ratio guard (commit e957e2c)

- `dnallm/configuration/presets/lora_targets.yaml`: all 35 `PRETRAIN_MODEL_MAPS` families, each with match markers, model_types, lora/ia3 target lists, FFN subsets, lora_r, and ratio bands; per-row derivation provenance (cached-remote / config-fetch / arch-canonical). Bands anchored on empirically measured ratios (peft 0.21.1, requires_grad counts): plant-dnabert-BPE IA³ 3.17e-4 / LoRA-r8 3.22e-3; plant-dnamamba IA³ 1.38e-3 / LoRA-r8 1.40e-2.
- Loader via importlib.resources with load-time structure validation (T-11-03); two-tier family resolution failing loud on no row; shared injection path ahead of both adapter branches (LoRA's silent None-behavior is thereby fixed too); D-03 dry-run (report + zero-match ValueError + no-training skip); D-04 ratio guard computed directly from requires_grad tensors (band-enforced with presets, zero-trainable-enforced without).
- `tests/configuration/test_peft_presets.py`: 22 network-free tests (table regression incl. set-equality with PRETRAIN_MODEL_MAPS, resolution, dry-run, guard, malformed-table load failures).

### Task 3 — Slow-lane acceptance + roundtrip + CHANGELOG (commit d4e9b68)

- Transformer IA³ fine-tune on `zhangtaolab/plant-dnabert-BPE` (modelscope, 20 steps on the pinned core-promoters dataset) with `target_modules=null` — preset auto-selection fired on the real backbone and the ratio landed in band.
- Mamba IA³ fine-tune on `zhangtaolab/plant-dnamamba-BPE-open_chromatin` (wrapper head_config path) — the silent-skip proving ground; in-band ratio asserted.
- IA³-specific save→reload roundtrip through `DNAInference(lora_adapter=...)` asserting logit identity (peft #2429 corruption class), with seed-pinned loads for the checkpoint's randomly-initialized pooler/classifier.
- CHANGELOG: REV-04 + REV-05 entries under `## [Unreleased]` with reviewer id R2-2 (D-09 append-only; sibling entries untouched).
- Per-module scoped coverage: `dnallm/finetune/trainer.py` 99%, `dnallm/configuration/configs.py` 100% (≥96% standard met).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] HeadConfig.custom_head default was a tuple-wrapped FieldInfo**
- **Found during:** Task 3 (Mamba acceptance)
- **Issue:** `custom_head: Any | None = (Field(...),)` — the trailing comma made the default a `(FieldInfo,)` tuple, so `head_config.__dict__` carried a FieldInfo into the model config and transformers' `on_train_begin` config-JSON serialization crashed with `TypeError: Object of type FieldInfo is not JSON serializable`.
- **Fix:** Proper `Field(default=None, ...)` declaration in configs.py; regression test asserts `json.dumps(HeadConfig(head="mlp").__dict__)` succeeds.
- **Files modified:** dnallm/configuration/configs.py, tests/configuration/test_configs.py
- **Commit:** d4e9b68

**2. [Rule 2 - Missing functionality] HeadConfig lacked the num_classes field the wrapper reads**
- **Found during:** Task 3 (Mamba acceptance)
- **Issue:** `DNALLMforSequenceClassification.forward` (model.py:247) reads `head_config["num_classes"]` when the head's logits width disagrees with the checkpoint's num_labels — HeadConfig never defined the field, so any 3-class-checkpoint model fine-tuned binary through the wrapper hit `KeyError: 'num_classes'`.
- **Fix:** `HeadConfig.num_classes: int | None = Field(default=2, ...)` in configs.py (model.py untouched — B2's lane); regression test added.
- **Files modified:** dnallm/configuration/configs.py, tests/configuration/test_configs.py
- **Commit:** d4e9b68

**3. [Rule 2 - Robustness] peft IA3Config kwargs filtered to the installed peft's field set**
- **Found during:** Task 1
- **Issue:** Plan prescribed a bare `IA3Config(**section.model_dump())`; on peft <0.15 (floor is >=0.14) `exclude_modules` would crash with an unexpected-kwarg TypeError (version-span constraint).
- **Fix:** Module-level `PEFT_IA3_FIELD_NAMES` from `dataclasses.fields(IA3Config)`; pass-through filtered through it (research Pitfall 10's feature-detect advice).
- **Files modified:** dnallm/finetune/trainer.py
- **Commit:** 3b644bd

**4. [Plan-sanctioned extension] Preset schema carries separate lora/ia3 target lists**
- The plan's artifact schema named a single `target_modules` per family; LoRA (query/value) and IA³ (key/value + FFN) targets genuinely differ per architecture, and peft rejects `out_proj` for LoRA-on-Mamba but allows it for IA³. The shipped schema (`lora_target_modules` + `ia3_target_modules` + `feedforward_modules`) preserves every plan requirement (never-guessed names, FFN subset, bands, lora_r).

## Evidence

- Fast lane: `uv run --no-sync pytest tests/configuration/test_configs.py tests/configuration/test_peft_presets.py tests/finetune/test_trainer.py -q -m "not slow"` → **198 passed**
- Slow lane: `uv run --no-sync pytest tests/finetune/test_trainer_real_model.py -q -m slow -k "ia3"` → **3 passed** (transformer + mamba + roundtrip)
- Coverage (scoped, cov-crash workaround): trainer.py **99%**, configs.py **100%**
- Invariants: `git diff HEAD -- pyproject.toml` and `-- dnallm/__init__.py` empty; `grep -c "(REV-04,"` = 1 and `grep -c "(REV-05,"` = 1 under `## [Unreleased]`; models.lock delta = none (all three artifacts already pinned); CHANGELOG diff purely additive (2 lines, 0 removed)
- Preset log line proven live on a real backbone: `[Info] IA³ preset 'Plant DNAMamba' selected (matched by name marker 'plant-dnamamba'): target_modules=['in_proj', 'out_proj', 'x_proj', 'dt_proj']`

## Known Stubs

None — no placeholder logic shipped; every acceptance criterion exercised real code paths.

## Auth Gates

None — all models/datasets loaded from the local modelscope/HF caches (models.lock rows pre-existing).

## Self-Check: PASSED

- Files: dnallm/configuration/presets/lora_targets.yaml, tests/configuration/test_peft_presets.py created; configs.py, trainer.py, inference.py, 3 test files, CHANGELOG.md modified — all present in HEAD.
- Commits: 3b644bd, e957e2c, d4e9b68 all ancestors of HEAD (verified below after writing).
