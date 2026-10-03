---
phase: 06-model-registry-showcase-data-curation
plan: 01
subsystem: models
tags: [model_info.yaml, plant-helixseek, registry, label-freeze, smoke-test, pytest, modelscope]

requires:
  - phase: 05-example-execution-harness-honest-gates-runner-feasibility
    provides: typed-skip prefix contract (environment-unavailable in expected_skips.yaml), cross-dir test import pattern, slow/fast leg marker discipline
provides:
  - model_info.yaml finetuned entries for zhangtaolab/PlantHelixSeek-CRE (binary, 2) and PlantHelixSeek-Anno (token, 17 BILOU) with committed provenance
  - Frozen label-order contract (CRE_LABELS/ANNO_LABELS in tests/models/test_plant_helixseek_registry.py) that Phase 7 notebooks and Phase 8 tests import and assert against
  - Real-load smoke proof of both checkpoints through the generic dnallm route on transformers 5.17.0 (REG-03 evidence)
  - Phase scratch home example/notebooks/plant_helixseek_shared/.scratch/ + committed .gitignore, reused by plan 06-03 curation tooling
affects: [06-02, 06-03, phase-07-notebooks, phase-08-execution-tests]

actuals:
  tokens: 3242    # chars/4 over the realized dnallm/+tests/+example diff (2 production commits)
  tasks: 2
  commits: 2      # MEASURED: git rev-list --count 9648a53..695a79c

tech-stack:
  added: []       # no new packages — huggingface_hub/pyyaml/pytest all pre-existing
  patterns:
    - "Registry freeze with provenance: append-only yaml edit (newline prepended, byte-identical-prefix self-check) + provenance comments carrying the upstream label-order source and checkpoint shas"
    - "Silent-permutation guard: post-load id2label asserted against hard-coded upstream constants imported cross-dir from the fast-leg test module, never against the yaml list fed into the load"
    - "Checkpoint-config constraint proof: hub metadata treated as evidence-to-assert (placeholder pattern + head shapes), never transcribed — semantics come from the upstream training script"

key-files:
  created:
    - tests/models/test_plant_helixseek_registry.py
    - tests/models/test_plant_helixseek_smoke.py
    - example/notebooks/plant_helixseek_shared/.gitignore
  modified:
    - dnallm/models/model_info.yaml

key-decisions:
  - "Label order frozen from the upstream training script (train_token_cls.py:78-96), not the checkpoint config — the freeze run re-proved live that Anno config.id2label is the LABEL_i placeholder pattern and CRE carries none"
  - "Smoke-test forwards run under torch.no_grad(): the research memory budget (0.59 s/3.83 GB CRE, 7.9 s/12.99 GB Anno) was measured no-grad; an autograd graph over the 8192 bp Anno window OOM-kills the process (exit 137, reproduced then fixed)"
  - "Version evidence emitted via sys.stdout.write (transformers_version=/torch_version= key=value lines) — T20 forbids bare print in this repo's lint set; observed in run output with -rP"
  - "environment-unavailable skip prefix reused without new registration (already in tests/expected_skips.yaml); the manual transformers-4.57 venv attempt procedure is documented in the smoke file docstring per the CONTEXT fallback contract"

patterns-established:
  - "Per-plan scratch-home .gitignore with explicit .scratch/ entry next to the consumer dir (plan 06-03 reuses example/notebooks/plant_helixseek_shared/.scratch/)"
  - "One-shot freeze tooling pattern: constraint-proof (fail-closed, key=value evidence, distinct exit codes) -> idempotent append -> yaml.safe_load self-check, all under gitignored scratch"

requirements-completed: [REG-01, REG-02, REG-03]

coverage:
  - id: D1
    description: "Two finetuned registry entries (CRE binary/2, Anno token/17) appended with provenance comments; pre-existing content preserved byte-identically"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/models/test_plant_helixseek_registry.py#test_cre_entry_matches_frozen_order"
        status: pass
      - kind: unit
        ref: "tests/models/test_plant_helixseek_registry.py#test_anno_entry_matches_frozen_order"
        status: pass
      - kind: other
        ref: "append-only gate: git show HEAD~1:model_info.yaml byte-prefix cmp — append-only-ok"
        status: pass
    human_judgment: false
  - id: D2
    description: "Frozen label order guarded: fast leg pins the yaml to committed upstream constants; slow leg asserts post-load id2label equality against those constants (index 1 == B-CDS)"
    requirement: REG-02
    verification:
      - kind: unit
        ref: "tests/models/test_plant_helixseek_registry.py (3 tests, no network)"
        status: pass
      - kind: integration
        ref: "tests/models/test_plant_helixseek_smoke.py#test_planthelixseek_anno_smoke_load (id2label == dict(enumerate(ANNO_LABELS)))"
        status: pass
      - kind: integration
        ref: "tests/models/test_plant_helixseek_smoke.py#test_planthelixseek_cre_smoke_load (id2label == {0: Not CRE, 1: CRE})"
        status: pass
    human_judgment: false
  - id: D3
    description: "Both checkpoints smoke-proven loadable through the generic task-type route (source=modelscope, no special handler) on transformers 5.x with version evidence recorded"
    requirement: REG-03
    verification:
      - kind: integration
        ref: "pytest tests/models/test_plant_helixseek_smoke.py -q -rs → 2 passed (transformers 5.17.0, torch 2.11.0+cu130, GB10)"
        status: pass
      - kind: other
        ref: "diff grep: no edits under dnallm/models/special/, no dnallm/models/model.py changes in plan commits"
        status: pass
    human_judgment: false
  - id: D4
    description: "Tooling boundary honored: one-shot freeze script lives uncommitted under the gitignored scratch home; commits contain only yaml + 2 test files + the scratch-home .gitignore"
    verification:
      - kind: other
        ref: "git ls-files example/notebooks/plant_helixseek_shared/.scratch/ → empty; git check-ignore covers freeze_registry.py; plan diff = exactly 4 expected files"
        status: pass
    human_judgment: false

duration: 19 min
completed: 2026-10-03
status: complete
---

# Phase 6 Plan 1: PlantHelixSeek Registry Freeze Summary

**Both PlantHelixSeek checkpoints landed in model_info.yaml with the label order frozen from the upstream training script (17-BILOU Anno / 2-class CRE), proven live by a one-shot gitignored freeze run and guarded by fast-leg structure tests plus slow-leg real ModelScope smoke loads on transformers 5.17.0.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-10-03T06:47:55Z
- **Completed:** 2026-10-03T07:07:14Z
- **Tasks:** 2
- **Files modified:** 4 (1 modified yaml, 3 created)

## Accomplishments

- **REG-01:** `dnallm/models/model_info.yaml` gained exactly two finetuned entries (CRE binary/2, Anno token/17, base_model set) as a byte-identical-prefix append with committed provenance comments; both checkpoints smoke-proven loadable through the generic `load_model_and_tokenizer` route with `source="modelscope"` — no special handler, no dispatch-chain edits.
- **REG-02:** Label order frozen from upstream `scripts/gene_annotation/train_token_cls.py:78-96` (fetched 2026-10-02), NOT the checkpoint config — the freeze run re-proved live that Anno `config.id2label` is the 17-entry `LABEL_i` placeholder pattern (classifier head [17, 512]) and CRE has no id2label at all (score head [2, 512]). Fast leg pins the yaml to the committed `CRE_LABELS`/`ANNO_LABELS` constants; slow leg asserts post-load `model.config.id2label` equality against those constants.
- **REG-03:** Both smoke tests pass on the transformers 5.x dev box — `transformers_version=5.17.0`, `torch_version=2.11.0+cu130`, GB10 CUDA (evidence lines in run output via `-rP`; cold ModelScope download of both ~1.9 GB checkpoints, warm run 33-40 s).
- **Owner constraint honored:** the one-shot freeze tooling ran from and stays in the gitignored `example/notebooks/plant_helixseek_shared/.scratch/`; `git ls-files` on the scratch home is empty; the two commits contain only the yaml, the two test files, and the scratch-home `.gitignore`.
- **Fast leg untouched:** `tests/models -m "not slow"` → 377 passed, 5 deselected, zero new skips (the 2 new slow tests deselect cleanly).

## Task Commits

Each task was committed atomically:

1. **Task 1: Scratch-home scaffold + one-shot freeze run + registry append + fast-leg structure test** - `7ba43f1` (feat)
2. **Task 2: Slow-leg smoke tests — real ModelScope download, generic-route load, id2label equality, forward shapes** - `695a79c` (test)

**Plan metadata:** final docs commit (see below)

## Files Created/Modified

- `dnallm/models/model_info.yaml` - two finetuned entries + provenance comment block (append-only, 34 insertions)
- `tests/models/test_plant_helixseek_registry.py` - fast-leg structure tests; committed frozen constants CRE_LABELS/ANNO_LABELS with provenance
- `tests/models/test_plant_helixseek_smoke.py` - slow-leg smoke: ModelScope-first load ladder, id2label equality vs frozen constants, forward shapes (1, 2) and (1, 8194, 17)
- `example/notebooks/plant_helixseek_shared/.gitignore` - `.scratch/` entry (scratch-home ignore coverage; reused by plan 06-03)
- `example/notebooks/plant_helixseek_shared/.scratch/freeze_registry.py` - intentionally uncommitted one-shot freeze tooling (never git-added)

## Decisions Made

- Label-order semantics taken from the upstream training script, with the checkpoint read used only as fail-closed constraint proof (placeholder pattern + head shapes) — re-confirmed live at execution time.
- Smoke forwards wrapped in `torch.no_grad()` to match the research-measured memory budget (see deviation 1).
- Version evidence emitted via `sys.stdout.write` (repo lint forbids bare `print` under T20, which is not relaxed for tests).
- `environment-unavailable:` typed-skip prefix reused (already registered); no new expected_skips entry needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Smoke-test forwards needed torch.no_grad()**
- **Found during:** Task 2 (slow-leg smoke verification)
- **Issue:** As drafted, the Anno 8192 bp forward built the autograd graph; eager-attention activation storage for backward exceeded the box's 121 GB unified memory — the pytest process was OOM-killed (exit 137) in warm-cache runs and failed with a CUDA/forward error in the cold run. The research memory budget (7.9 s / 12.99 GB peak) was measured on a no-grad forward.
- **Fix:** Wrapped both smoke forwards in `with torch.no_grad():` — matches the measured probe conditions; the asserted behavior (logits shapes, id2label equality) is unchanged.
- **Files modified:** tests/models/test_plant_helixseek_smoke.py
- **Verification:** 3 consecutive green runs of the smoke suite (39.62 s / 33.78 s / 32.63 s, 2 passed each, exit 0)
- **Committed in:** 695a79c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minimal and necessary — no scope creep; the fix aligns the test with the measured conditions the plan itself cites as the memory budget.

## Issues Encountered

- The freeze script's first run failed on an API-shape mismatch: `huggingface_hub` 1.33.0 exposes safetensors tensor info under `get_safetensors_metadata(...).files_metadata[filename].tensors`, not a top-level `.tensors` attribute. Fixed inside the scratch script (uncommitted tooling — no repo change) and re-run clean: all constraint facts proven (`anno_config_placeholder_ok=true`, `anno_head_shape=[17, 512]`, `cre_head_shape=[2, 512]`, `entries_appended=2`, `byte_identical_prefix=true`, finetuned count 185 → 187).
- The plan's `read_first` cites `tests/examples/test_notebook_execution.py:24` for the cross-dir import; the import actually lives at line 46 in the current file. Pattern verified working either way (`from tests.models.test_plant_helixseek_registry import ...` collects and runs).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- REG-01/02/03 closed: registry entries committed, frozen order guarded on both legs, compat gate evidence recorded.
- Plan 06-03's curation tooling reuses the scratch home at `example/notebooks/plant_helixseek_shared/.scratch/` (ignore coverage committed here).
- Phase 7 notebooks can import `ANNO_LABELS`/`CRE_LABELS` from `tests.models.test_plant_helixseek_registry` and load both checkpoints through the generic route with the registry entries as the single frozen source.
- No blockers.

## Self-Check: PASSED

- All 4 key files exist on disk (FOUND).
- Both task commits exist (7ba43f1, 695a79c FOUND).
- Scratch home untracked (git ls-files empty).
- All plan verify commands re-run green post-commit (tracer gate): freeze idempotent, fast-leg 3 passed, ruff clean, append-only prefix ok, no LABEL_ outside comments, scratch ignored.

---
*Phase: 06-model-registry-showcase-data-curation*
*Completed: 2026-10-03*
