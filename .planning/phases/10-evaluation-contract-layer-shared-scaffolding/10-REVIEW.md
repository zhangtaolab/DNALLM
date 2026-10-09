---
phase: 10-evaluation-contract-layer-shared-scaffolding
reviewed: 2026-10-09T11:45:20Z
depth: standard
files_reviewed: 73
files_reviewed_list:
  - CHANGELOG.md
  - dnallm/cli/cli.py
  - dnallm/cli/inference.py
  - dnallm/cli/train.py
  - dnallm/configuration/configs.py
  - dnallm/datahandling/data.py
  - dnallm/finetune/trainer.py
  - dnallm/inference/benchmark.py
  - dnallm/inference/inference.py
  - dnallm/inference/interpret.py
  - dnallm/inference/plot.py
  - dnallm/inference/vep.py
  - dnallm/mcp/server.py
  - dnallm/models/losses.py
  - dnallm/models/modeling_auto.py
  - dnallm/models/model.py
  - dnallm/tasks/metric_registry.py
  - dnallm/tasks/metrics.py
  - dnallm/tasks/task.py
  - dnallm/utils/sequence.py
  - docs/concepts/architecture/tokenization.md
  - docs/concepts/biology/biological_tasks.md
  - docs/concepts/biology/dna_sequences.md
  - docs/concepts/inference.md
  - docs/concepts/mcp.md
  - docs/concepts/technical/transfer_learning.md
  - docs/concepts/training.md
  - docs/example/marimo/benchmark/benchmark_demo.md
  - docs/example/marimo/finetune/finetune_demo.md
  - docs/example/marimo/inference/inference_demo.md
  - docs/example/notebooks/benchmark.md
  - docs/example/notebooks/data_prepare_finetune.md
  - docs/example/notebooks/finetune_binary.md
  - docs/example/notebooks/finetune_multi_labels.md
  - docs/example/notebooks/finetune_NER_task.md
  - docs/example/notebooks/inference.md
  - docs/example/notebooks/inference_megaDNA.md
  - docs/example/notebooks/overview.md
  - docs/getting_started/installation.md
  - docs/getting_started/quick_start.md
  - docs/index.md
  - docs/resources/model_selection.md
  - docs/resources/model_zoo.md
  - docs/resources/troubleshooting_models.md
  - docs/user_guide/benchmark/configuration.md
  - docs/user_guide/benchmark/getting_started.md
  - docs/user_guide/benchmark/index.md
  - docs/user_guide/cli/config_generator.md
  - docs/user_guide/cli/index.md
  - docs/user_guide/cli/mcp_server.md
  - docs/user_guide/cli/usage.md
  - docs/user_guide/data_processing/data_preparation.md
  - docs/user_guide/fine_tuning/getting_started.md
  - docs/user_guide/fine_tuning/index.md
  - docs/user_guide/fine_tuning/peft_adapters.md
  - docs/user_guide/fine_tuning/task_guides.md
  - docs/user_guide/getting_started.md
  - docs/user_guide/inference/getting_started.md
  - docs/user_guide/models.md
  - docs/user_guide/performance/gpu_optimization.md
  - docs/user_guide/performance/inference_speed.md
  - docs/user_guide/performance/model_quantization.md
  - example/notebooks/overview.md
  - mkdocs.yml
  - pyproject.toml
  - README.md
  - scripts/generate_md_from_marimo.py
  - tests/configuration/test_configs.py
  - tests/datahandling/test_dna_dataset.py
  - tests/finetune/test_trainer.py
  - tests/tasks/test_metric_registry.py
  - tests/inference/test_vep.py
findings:
  critical: 1
  warning: 5
  info: 6
  total: 12
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-10-09T11:45:20Z
**Depth:** standard (diff-driven for the ~53 mechanically-swept docs pages, full analysis for the substantive code targets, per orchestrator scope note)
**Files Reviewed:** 73
**Status:** issues_found

## Summary

The substantive engineering is strong: the EVAL-01 trainer guard (`allow_test_as_eval`, collision `ValueError`s), the `evaluate(split=...)` predict-routed override with result JSON, the frozen metric registry with strict one-directional alias handling, the emission gate in `dnallm/tasks/metrics.py`, and the VEP alignment/kernels are all correctly implemented, well-documented, and well-tested (all 467 tests in the six touched test files pass; `ruff check` clean under project config; `pyproject.toml` dependency lists verified unchanged — only the one package-data line was added; no `dnallm/__init__.py` re-export changes). I verified two suspected hazards empirically and rejected both as false positives (vendored `r_squared` returns a bare float, so the `{"r2": r2}` emission is correct; `Trainer.evaluation_loop` applies `denumpify_detensorize`, so the `evaluate(split=...)` JSON dump is safe).

However, the phase carries one accidental edit that silently deletes an existing regression test, plus a set of correctness/quality defects concentrated in the periphery of the mechanical sweep: a new docs chapter whose runnable examples fail against the real `load_model_and_tokenizer` API, paper titles rewritten (and garbled by string-literal concatenation) inside `MODEL_INFO`, an asymmetry in the new eval-guard, a leftover bare debug print inside a rewritten metrics function, and one broken sentence produced by the terminology replacement.

Scope note: `tests/tasks/test_metrics.py` (+232 lines, registry emission-contract tests) was changed by the same phase commits but is not in the review file list; I reviewed its diff incidentally while cross-checking `metrics.py` and found no defects in it. Downstream consumers of this REVIEW.md should be aware it was outside the declared scope.

## Critical Issues

### CR-01: Botched test insertion deleted `test_process_missing_data_basic` and fused its body into the preceding test

**File:** `tests/datahandling/test_dna_dataset.py:735-747`
**Issue:** When the four new `validate_sequences` tests were inserted, the `def test_process_missing_data_basic(self):` line of the existing test was replaced instead of the insertion happening above it. Its body — the `"""Test basic processing of missing data."""` docstring, the `test_data` fixture with `None`/empty sequences, the `process_missing_data()` call, and the closing assert — was left dangling and now executes as the tail of `test_validate_sequences_dataset_dict_logs_total_dropped` (verified against the current file and the diff hunk). Consequences: (1) a named regression test for `DNADataset.process_missing_data` silently vanishes from the suite — the suite count drops by one with no failure; (2) the orphaned docstring is now a dead string statement mid-function, misdocumenting the fused test; (3) the `process_missing_data` assertions are now conditional on the unrelated `validate_sequences` assertions passing first, so a failure in the dict-split test masks any regression in missing-data handling; (4) the fused test no longer matches its name or docstring. This directly undermines the milestone's core guarantee (a fully accounted-for pytest suite whose coverage cannot silently regress). The suite currently passes (467/467) only because the orphaned body still happens to execute inside the wrong test.
**Fix:** Restore the deleted `def` line between the two tests and re-indent nothing else:

```python
        assert len(dna_ds.dataset["train"]) == 1
        assert len(dna_ds.dataset["test"]) == 1

    def test_process_missing_data_basic(self):
        """Test basic processing of missing data."""
        test_data = {
            "sequence": ["ATCG", "", "TAGC", None, "GCTA"],
            "labels": [0, 1, 0, 1, 0],
        }
        ds = Dataset.from_dict(test_data)
        dna_ds = DNADataset(ds)

        dna_ds.process_missing_data()

        # Should filter out empty and None sequences
        assert len(dna_ds.dataset) < 5
```

## Warnings

### WR-01: New PEFT chapter's `load_model_and_tokenizer` examples fail as written (missing `source=`)

**File:** `docs/user_guide/fine_tuning/peft_adapters.md:66-69, 139-148`
**Issue:** Both runnable examples call `load_model_and_tokenizer("zhangtaolab/plant-dnabert-BPE", task_config=config["task"])` without `source=`. The real signature (`dnallm/models/model.py:753`) defaults to `source="local"`, and `_get_model_path_and_imports` (`dnallm/models/model.py:448-450`) raises `ValueError("Model zhangtaolab/plant-dnabert-BPE not found locally.")` for any hub repo id — no special handler intercepts `plant-dnabert-BPE` before that check (the dnabert2 handler runs after path resolution). Every other doc in the repo passes `source="huggingface"` or `source="modelscope"` (e.g. `docs/example/notebooks/lora_finetune.md:36-41`, `finetune_binary.md:38-42`). The chapter's primary LoRA example and QLoRA example therefore crash on the first model-load step if a user follows them verbatim.
**Fix:** Add `source="huggingface"` to both calls:

```python
model, tokenizer = load_model_and_tokenizer(
    "zhangtaolab/plant-dnabert-BPE",
    task_config=config["task"],
    source="huggingface",
)
```

### WR-02: Mechanical sweep rewrote verbatim paper titles in `MODEL_INFO` and the titles remain garbled by string-literal concatenation

**File:** `dnallm/models/modeling_auto.py:238-241, 316-319, 348-351, 415-418, 427-430`
**Issue:** The terminology replacement was applied to literal citation titles in the model registry: the GPN paper is titled "DNA language models are powerful predictors of genome-wide variant effects" and GROVER's "DNA language model GROVER learns sequence context in the human genome" (similarly GENA-LM x2, PlantCaduceus). These are quoted titles of published papers (GPN: doi 10.1073/pnas.2311219120 — the title visible at that DOI does not contain "large"); rewriting them misattributes the citation in a user-facing registry (titles surface through the model-config generator and model docs). Additionally, the edited GENA-LM/PlantCaduceus entries use adjacent string literals with no separating space, so the runtime title is `"GENA-LM: a family of open-sourcefoundational DNA large language models for long sequences"` and `"...genomes atsingle-nucleotide resolution..."`; the GPN/GROVER single-string titles keep their pre-existing missing-space typos (`arepowerful`, `learnssequence`). This phase edited exactly these lines, so the garbled/misquoted titles ship as-is.
**Fix:** Revert the word "large" inside quoted paper titles (keep the sweep to prose/docstrings), add the missing spaces at the literal boundaries, and fix the in-string typos:

```python
    "GENA-LM": {
        "title": "GENA-LM: a family of open-source "
        "foundational DNA language models for long sequences",
        ...
    "GPN": {
        "title": "DNA language models are powerful predictors of genome-wide variant effects",
```

### WR-03: Eval-semantics guard is asymmetric — `load_best_model_at_end` collision only raises on the test-excluded path

**File:** `dnallm/finetune/trainer.py:266-272, 280-282`
**Issue:** `set_up_trainer` raises the loud `ValueError("load_best_model_at_end requires an evaluation split, ...")` only in the `elif "test" in self.data_split` guarded branch. In the final `else` branch (dataset has splits but neither dev nor test, e.g. `{"train"}` only — and the unsplit-dataset path), `eval_strategy` is set to `"no"` but `load_best_model_at_end=True` passes through untouched. Because the mutation happens after `TrainingArguments.__post_init__` has already run, transformers' own construction-time consistency checks are bypassed, and the user is left with a silently never-selected "best model" (or transformers' opaque downstream error) instead of the phase's own matchable `ValueError`. Early stopping is guarded in all branches (the `eval_dataset is None` check at line 335 is branch-independent), so only the `load_best_model_at_end`-without-early-stopping case leaks. This defeats the EVAL-01 goal ("loud failure instead of silent degradation") for one reachable configuration.
**Fix:** Hoist the collision check so it covers every branch that disables evaluation:

```python
        else:
            eval_dataset = None
            self.training_args.eval_strategy = "no"
        if eval_dataset is None and self.training_args.load_best_model_at_end:
            raise ValueError(
                "load_best_model_at_end requires an evaluation split, but none is "
                "available (no dev split present and the test split is excluded "
                "from evaluation when allow_test_as_eval is False). Provide a "
                "dev/validation split or set finetune.allow_test_as_eval=true."
            )
```

### WR-04: Bare debug print left in `calculate_metric_with_sklearn` inside a function this phase rewrote

**File:** `dnallm/tasks/metrics.py:95`
**Issue:** `print(valid_labels.shape, valid_predictions.shape)` dumps raw shapes to stdout on every token-classification evaluation. The phase's stated convention (plan D-04) sanctions only the house `[Info]`/`[Warning]`-prefixed prints; this is an unprefixed debug leftover. It is pre-existing (a context line in the diff), but commit fad6a33 rewrote this exact function's return path (`return _emit({...})`) without removing it, and the phase's own new test has to `patch("builtins.print")` specifically to "silence the legacy shape print" (`tests/tasks/test_metrics.py`, `_emission_calculate_metric_with_sklearn`) — the codebase is now papering over the defect in tests instead of fixing it. (Ruff does not catch it because `[tool.ruff.lint] ignore` includes `"print"` globally.)
**Fix:** Delete line 95 (and drop the now-unneeded `patch("builtins.print")` from the emission test helper).

### WR-05: Terminology sweep produced a broken sentence: "large DNA large language models"

**File:** `docs/user_guide/performance/gpu_optimization.md:3` (intro paragraph)
**Issue:** The mechanical replacement turned "Training and running large DNA language models" into "Training and running **large DNA large language models**" — a duplicated "large". This is exactly the broken-sentence class the sweep was supposed to be checked for; I grepped all other 64 replacement lines and this is the only double-replacement/grammar casualty (no "large large" or leftover short-form occurrences anywhere else in docs/, README, or dnallm/).
**Fix:** "Training and running large DNA language models can be computationally intensive." — the modifier "large" was already present; either drop the inserted word or rephrase to "large DNA language models".

## Info

### IN-01: pyproject package-data declares a `presets/` directory that does not exist

**File:** `pyproject.toml:266`
**Issue:** `"dnallm.configuration" = ["presets/*.yaml"]` was added, but `dnallm/configuration/presets/` does not exist in the repo (verified), and no preset YAML ships. Harmless at build time (the glob matches nothing), but it is dead configuration today, and `Ia3Config.target_modules`' description ("the trainer's per-model preset (shipped as packaged YAML)") refers to files that are not yet packaged.
**Fix:** Acceptable as forward scaffolding for the IA³ phase — but note it in the phase's deferred-items ledger so the presets actually land with the trainer branch, or defer the package-data line to that phase.

### IN-02: `evaluate(split=...)` result mixes non-metric runtime keys into the "canonical names" metrics dict/JSON

**File:** `dnallm/finetune/trainer.py:600-614`
**Issue:** `key.removeprefix("test_")` also strips `test_loss`, `test_runtime`, `test_samples_per_second`, `test_steps_per_second` into the returned dict as `loss`, `runtime`, `samples_per_second`, `steps_per_second`, which are then written into `eval_{split}_result.json` under the docstring's promise of "canonical, unprefixed metric names". These keys are not registry names and bypass `validate_emission` (which gates only compute paths). Not incorrect behavior, but inconsistent with the emission contract this phase establishes for downstream benchmark tooling.
**Fix:** Either filter to registry names (plus `loss`) before returning, or document in the docstring that timing/runtime keys ride along.

### IN-03: `evaluate(split=...)` writes its result JSON into the CWD when `output_dir` is None

**File:** `dnallm/finetune/trainer.py:603`
**Issue:** `Path(self.train_config.output_dir or ".")` silently falls back to the current working directory, creating `eval_{split}_result.json` wherever the process happens to run. Surprising side effect for a config with no `finetune.output_dir`.
**Fix:** Raise or warn when `output_dir` is None, or pick a deterministic default (e.g. `./dnallm_eval/`) and document it.

### IN-04: `use_ia3: true` is a silent no-op during the interim window

**File:** `dnallm/configuration/configs.py:319-326`, `dnallm/finetune/trainer.py:215`
**Issue:** The field is popped in `set_up_trainer` and does nothing until the next phase's trainer branch. This is deliberately documented (field description, docs chapter, and a pinning test `test_use_ia3_has_no_cross_field_rejection_yet`), so it is a conscious scaffold — but a user setting `use_ia3: true` today gets LoRA/full training with zero feedback.
**Fix:** Consider a one-line `[Warning] use_ia3=true has no effect yet...` print at `DNATrainer.__init__` during the interim window, mirroring the phase's own loud-degradation philosophy.

### IN-05: QLoRA example in the new PEFT chapter uses `datasets` without constructing it

**File:** `docs/user_guide/fine_tuning/peft_adapters.md:150-155`
**Issue:** The QLoRA snippet references `datasets=datasets` but never builds it (the LoRA example above does). Readers copying just the QLoRA block hit a `NameError`. Doc shorthand at worst, but the LoRA block is self-contained, so the asymmetry invites the copy-paste failure.
**Fix:** Add the same `DNADataset.load_local_data(...)` block, or a `# datasets built as in the LoRA example above` comment.

### IN-06: Model-count inconsistency surfaced on lines this phase edited: "150+" vs "200+"

**File:** `docs/index.md:~99`, `README.md:23`
**Issue:** The sweep edited both capability bullets but left their counts divergent: index.md says "150+ pre-trained DNA large language models" while README says "200+ pre-trained DNA large language models" (registry holds 238 entries; CLAUDE.md says 150+). Pre-existing numeric drift, but both lines were touched in this pass.
**Fix:** Pick one number (the registry count) and align both files in a follow-up docs pass.

---

_Reviewed: 2026-10-09T11:45:20Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
