---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 03
subsystem: docs/terminology
tags: [docs, terminology, comparability-warning, peft-chapter, changelog, DOCS-01, REV-03]
requires:
  - Phase 10 CONTEXT D-08/D-09 (sweep scope, CHANGELOG evidence chain)
provides:
  - Terminology unified to "DNA large language models" on the full A3 surface (docs/, README, example mirror pair, 13 docstring files, mkdocs.yml, marimo wrapper generator)
  - DNADataset.validate_sequences comparability docstring + dropped-row count log line (no new API, D3 ruling)
  - docs/user_guide/fine_tuning/peft_adapters.md (LoRA/QLoRA grounded, IA³ honest forward pointer) + mkdocs nav entry
  - CHANGELOG REV-03 (Ed-2/Ed-6/R1-3c) entry under ## [Unreleased]
affects:
  - docs-validation workflow (all five steps re-verified green)
  - mkdocstrings-rendered API pages (docstring sweep)
tech-stack:
  added: []
  patterns:
    - byte-exact scoped replacement sweep (plural-before-singular mapping, read_bytes/write_bytes — no newline/encoding drift)
key-files:
  created:
    - docs/user_guide/fine_tuning/peft_adapters.md
  modified:
    - dnallm/datahandling/data.py
    - tests/datahandling/test_dna_dataset.py
    - README.md
    - mkdocs.yml
    - CHANGELOG.md
    - example/notebooks/overview.md (+ docs/example/notebooks/overview.md mirror, byte-identical)
    - scripts/generate_md_from_marimo.py
    - "13 A3-owned dnallm docstring files: cli/cli.py, cli/inference.py, cli/train.py, inference/benchmark.py, inference/inference.py, inference/interpret.py, inference/plot.py, mcp/server.py, models/losses.py, models/model.py, models/modeling_auto.py, tasks/task.py, utils/sequence.py"
    - "31 docs/ non-mirror pages + 12 docs/example generated .md wrappers"
decisions:
  - Dropped-row counting sums DatasetDict splits via a private _row_count helper (plan's literal len(self.dataset) counts splits, not rows — the warning would never fire for multi-split datasets)
  - mkdocs.yml site_description and scripts/generate_md_from_marimo.py template strings swept alongside the plan surface (mkdocs.yml is this plan's file; the generator regenerates the swept marimo wrappers — leaving either would reintroduce the old term)
  - Task 2's commit attribution handled without history rewrite after a shared-index race (see Deviations) — content verified intact, no Co-Authored-By trailers anywhere
metrics:
  duration: ~22 min (2026-10-09T10:49:45Z start)
  completed: 2026-10-09
status: complete
actuals:
  tokens: 17200       # chars/4 over the realized diffs of this plan's files (git show slices: 2e3f8fb + c751df6-restricted-to-58-files + 36d0c74 = 68,898 chars)
  tasks: 3
  commits: 9          # MEASURED: git rev-list --count d782c7cb..HEAD — SHARED TREE: includes concurrent wave agents; attributable to 10-03: 2e3f8fb (Task 1), 36d0c74 (Task 3), Task 2 content inside c751df6
plan_head_before: d782c7cb6108f56b79cf5a1fae2ef5dc6f9bed73
plan_head_after: 2d0d3bdfd665d8e5be1422fa529953f04888eb9d
---

# Phase 10 Plan 03: Docs, Terminology & Changelog Summary

**One-liner:** Full-surface "DNA large language models" terminology unification (61 files), validate_sequences cross-model comparability warning with dropped-row count logging, new LoRA/QLoRA/IA³ chapter, and the REV-03 CHANGELOG evidence entry — all five docs-validation gate steps green.

## What Was Built

### Task 1 — validate_sequences comparability warning + dropped-row count log (tracer slice) — commit 2e3f8fb

- Rewrote the `DNADataset.validate_sequences` docstring (Google style): whole-sequence drop semantics via `check_sequence`, the cross-model `valid_chars` comparability hazard (13 strict-charset families silently evaluate different subsets; comparisons valid only over a common subset; unified prefiltering is pipeline-side per the D3 ruling — no new API), case-sensitive literal charset matching mechanics (`set(seq.upper()) - set(valid_chars)`), and the documented log behavior.
- Added the count log line: captures row count before/after the unchanged `.filter(...)` chain and prints exactly one `[Warning] validate_sequences dropped N of M rows (valid_chars=..., minl=..., maxl=...); cross-model comparisons require a common valid_chars subset.` line when N > 0; silent at zero (documented boundary choice). Counts and filter parameters only — never sequence contents (T-10-05 mitigation).
- Four new tests in `TestDNADatasetSequenceProcessing`: fires-once-with-correct-count, silent-at-zero, all-dropped edge (full count + empty dataset, no raise), and DatasetDict total-across-splits.

### Task 2 — full-surface terminology sweep (61 files, 97 replacements) — content in c751df6 (see Deviations)

- Enumerated via `grep -rlniE "DNA[ -]language[ -]model"`; only 4 spaced variants exist (52+26 lowercase, 10+5 capitalized) — no hyphenated forms found.
- Applied as byte-exact scoped replacements (plural-before-singular, `read_bytes`/`write_bytes`): docs/ 31 non-mirror pages + 12 docs/example wrappers, README.md, 13 A3-owned dnallm docstring files, `example/notebooks/overview.md` and its `docs/example/notebooks/overview.md` mirror as an identical pair (byte-identity proven by `diff` + `check_docs_sync.py`).
- Sweep-to-zero proof: the verify grep over the entire A3 surface returns 0 files.
- No dnallm file owned by agents 10-01/10-02/10-04 was touched; the phase-level whole-tree grep (`dnallm/ docs/ README.md example/`) is already zero with their concurrent sweeps landed.

### Task 3 — PEFT chapter + nav + CHANGELOG — commit 36d0c74

- `docs/user_guide/fine_tuning/peft_adapters.md`: LoRA section grounded in the real `LoraConfig` fields (r, lora_alpha, target_modules, lora_dropout, bias, task_type) with the true `DNADataset.load_local_data → split_data → encode_sequences` pipeline and `DNATrainer(use_lora=True)`; QLoRA section with `finetune.use_qlora: true`, the exact `quantization_config` shape from the trainer docstring (`load_in_4bit`, `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_type`), the bitsandbytes requirement, and the k-bit/gradient-checkpointing behavior the trainer applies; IA³ section as an honest forward pointer — `finetune.use_ia3` + the `ia3` YAML section exist as config surface (verified landed by 10-01's scaffolding commit), trainer branch arrives next release, no fabricated examples.
- mkdocs.yml: `- PEFT Adapters (LoRA / QLoRA / IA³): user_guide/fine_tuning/peft_adapters.md` registered in the Fine Tuning block after Advanced Techniques.
- CHANGELOG.md: REV-03 entry appended under `## [Unreleased] > ### Changed` via unique-anchor insert (D-09 discipline, file re-read immediately before edit) with the inline `(REV-03, Ed-2/Ed-6/R1-3c)` evidence tag.

## Verification Evidence

| Gate | Result |
|------|--------|
| `pytest tests/datahandling/test_dna_dataset.py -q` | 154 passed |
| `pytest tests/examples/test_examples.py tests/configuration/test_yaml_load.py -q` | 127 passed, 1 allowlisted skip |
| `scripts/check_docs_sync.py` | OK (mirror byte-identical, incl. after the pair sweep) |
| `scripts/validate_docs_snippets.py` | 147 files, 350 python blocks, all valid |
| `scripts/validate_yaml.py` | 21/21 YAML configs OK |
| `python scripts/check_code.py` | All required checks pass (ruff format, ruff lint, fast suite + coverage; mypy informational only) |
| Verify grep over A3 surface | 0 files |
| Phase cross-check grep (`dnallm/ docs/ README.md example/`) | 0 hits anywhere |
| `grep -c peft_adapters mkdocs.yml` / `grep -c "(REV-03, Ed-2/Ed-6/R1-3c)" CHANGELOG.md` | 1 / 1 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] DatasetDict dropped-row counting**
- **Found during:** Task 1
- **Issue:** The plan's literal implementation (`len(self.dataset)` before/after `.filter`) never fires the warning for `DatasetDict` inputs — `len()` on a DatasetDict returns the number of splits, so `n_dropped` is always 0 even when rows are filtered, violating the must-have "logs a dropped-row count line when rows are filtered".
- **Fix:** Private `DNADataset._row_count` helper sums `len(split)` across DatasetDict splits (plain `len()` for Dataset); filter chain and public API untouched. Covered by `test_validate_sequences_dataset_dict_logs_total_dropped`.
- **Files:** dnallm/datahandling/data.py, tests/datahandling/test_dna_dataset.py
- **Commit:** 2e3f8fb

**2. [Rule 2 - Completeness] mkdocs.yml site_description + marimo wrapper generator swept**
- **Found during:** Task 2
- **Issue:** `mkdocs.yml` line 2 (`site_description`) and the three template strings in `scripts/generate_md_from_marimo.py` (lines 87-89) carry the old terminology. mkdocs.yml renders the description on the site; the generator script regenerates the exact docs/example/marimo wrapper pages the plan sweeps — leaving it would reintroduce the old term on the next regeneration.
- **Fix:** Same scoped replacement applied to both files (mkdocs.yml is this plan's file per Task 3; the generator is unowned by any wave agent and is the upstream of my swept surface).
- **Files:** mkdocs.yml, scripts/generate_md_from_marimo.py
- **Commit:** inside c751df6 (with the Task 2 sweep, see below)

**3. [Process - Shared-tree concurrency] Task 2 commit absorbed by concurrent agent's commit**
- **Found during:** Task 2 commit
- **Issue:** This wave runs 4 agents in ONE shared working tree (workflow.use_worktrees=false), which means one shared git index. My 58 staged sweep files were committed by agent 10-01's `git commit` (c751df6, "one-pass scaffolding") in the window between my `git add` and my `git commit`; my own commit then found an empty index and aborted.
- **Resolution:** No history rewrite (prohibited; concurrent agents actively building on HEAD). Verified all 58 files: working tree == c751df6 == HEAD, content intact, 63-file commit = my 58 + 10-01's 5 scaffolding files. Task 2 therefore has no separate commit hash; its content is traceable via c751df6 and the sweep-to-zero grep proof. Task 3's commit used a single tight stage+verify-staged-count+commit invocation (3/3 files confirmed) to avoid a repeat.
- **Recommendation to orchestrator:** future shared-tree waves should either serialize commits per agent (file-lock around add+commit) or use `git commit -- <paths>` pathspec commits; the staged-count guard added here is a workable per-commit mitigation.

**4. [Authoring correction - Task 3] Chapter examples grounded against the real API**
- **Issue:** First draft used a non-registry model name and a `DNADataset(...)`/`tokenize_dataset` pipeline that does not exist.
- **Fix:** Model replaced with registry-real `zhangtaolab/plant-dnabert-BPE` (house example model); pipeline corrected to `DNADataset.load_local_data(...)` → `split_data(...)` → `encode_sequences(tokenizer=...)` per the signatures and the house getting-started chapter. All python blocks pass validate_docs_snippets AST checks.
- **Commit:** 36d0c74

## Known Stubs

| File | Line | Reason |
|------|------|--------|
| docs/user_guide/fine_tuning/peft_adapters.md | 163 | IA³ section is an intentional forward pointer (plan-mandated split delivery): config surface exists, trainer branch + working examples complete in Phase 11-12 after PEFT-01; recorded in WINDOWS.md id 21 |

## Out-of-Scope Residue (logged, not fixed)

Old terminology persists outside the phase check surface and outside every wave-1 plan's file list — logged to `deferred-items.md` in this phase directory: pyproject.toml description (10-01's file), test-file docstrings (tests/models/test_model.py, tests/tasks/test_task.py, tests/tasks/test_metrics.py), ui/model_config_generator_app.py, root legacy cli/ files, CHANGELOG historical entries (append-only — must not be rewritten), .planning/ artifacts.

## Notes for the Orchestrator

- gsd_run/gsd-tools CLI is not invokable in this environment (`gsd_run: command not found`; the node script at ~/.claude/gsd-core/bin/gsd_run fails to load). WINDOWS.md id 21 was appended manually matching the exact ledger format (JSON re-validated, 21 entries). STATE.md/ROADMAP.md intentionally NOT updated (orchestrator-owned post-wave, per dispatch instructions).
- The plan's flag about numeric display/rounding contracts (half-even vs half-up) never triggered: Phase 10 introduces no metric-value formatting, exactly as the flagged assumption anticipated.

## Self-Check: PASSED

Files: all 9 spot-checked key files present. Commits: 2e3f8fb, 36d0c74 ancestors of HEAD; c751df6 (Task 2 content) ancestor of HEAD.
