---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 01
subsystem: finetune/configuration (evaluation contract layer)
tags: [eval-guard, evaluate-split, ia3, vep, sweep, scaffolding, EVAL-01, REV-01]
requires: []
provides:
  - "TrainingConfig.allow_test_as_eval (bool, default False, YAML finetune.allow_test_as_eval)"
  - "TrainingConfig.use_ia3 (bool, default False, field-first, no validators — D-07)"
  - "Ia3Config / VepConfig / SweepConfig Pydantic sections + DNALLMConfig keys ia3/vep/sweep + load_config registration (D-06)"
  - "DNATrainer.evaluate(split=None, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval') override; split path routes through trainer.predict and writes output_dir/eval_{split}_result.json {split, timestamp, metrics} (D-01/D-02/D-03)"
  - "pyproject [tool.setuptools.package-data] entry 'dnallm.configuration' = ['presets/*.yaml'] (zero new dependencies)"
affects:
  - dnallm/finetune/trainer.py
  - dnallm/configuration/configs.py
  - pyproject.toml
  - tests/finetune/test_trainer.py
  - tests/configuration/test_configs.py
  - CHANGELOG.md
tech-stack:
  added: []  # stdlib json/datetime only; no new dependencies (invariant held)
  patterns:
    - "print('[Warning] ...') house-style one-line WARN at guard time (10-PATTERNS WARN channel)"
    - "mocked HF boundary: post-init load_best_model_at_end read from args_cls.return_value (collision checks test-observable)"
key-files:
  created:
    - .planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-01-SUMMARY.md
  modified:
    - dnallm/finetune/trainer.py
    - dnallm/configuration/configs.py
    - pyproject.toml
    - tests/finetune/test_trainer.py
    - tests/configuration/test_configs.py
    - CHANGELOG.md
decisions:
  - "Flip/opt-in WARNs use the trainer's existing print('[Warning] ...') style (10-PATTERNS recommendation); flip line carries all three D-04 facts in one call"
  - "evaluate(split=...) accepts any split key present in the DatasetDict (CONTEXT discretion) with a matchable ValueError listing sorted available splits otherwise"
  - "Result JSON written to {output_dir}/eval_{split}_result.json with nested 'metrics' block for Phase 11 aggregate_seeds; metric keys are predict-prefix-stripped canonical spellings (no metric_registry import — cross-lane contract)"
  - "CHANGELOG (REV-01, R1-2c) entry landed in the Task 1 guard commit dae194a (D-09 'same commit as the fix' reading of the must-have); Task 3's grep verifier passes on the cumulative file"
  - "Collision checks read the post-init self.training_args.load_best_model_at_end so both YAML and extra_args routes are caught"
metrics:
  duration: 38m
  completed: 2026-10-09
actuals:
  tokens: 10100      # chars/4 over the realized diff of this plan's 6 files (40434 chars)
  tasks: 3
  commits: 4         # 3 task commits + this summary commit (see Shared-tree note)
status: complete
plan_head_before: 93bd52fb284df9df0a0aa51f53caef0b686a17c1
plan_head_after: e1f76a2bb01c34e89ac018e941599233df19c20a
shared_tree_note: "Single shared working tree with 3 concurrent executor agents: git rev-list --count over the ledger range counts sibling commits too (14 at SUMMARY time, 3 are this plan's). Task commit hashes: dae194a, 64d6f41, c751df6."
---

# Phase 10 Plan 01: Trainer Eval Guard & Shared Scaffolding Summary

**One-liner:** EVAL-01 evaluation-semantics contract in the trainer — the test split can never silently become the eval set (`allow_test_as_eval` opt-in, collision ValueErrors, `evaluate(split=...)` predict-routing with canonical-keyed result JSON) — plus the one-pass Phase 11 scaffolding (`use_ia3` field, Ia3Config/VepConfig/SweepConfig stubs, `load_config` registration, presets package-data).

## What Was Built

### Task 1 — Eval-semantics guard (tracer)
- `TrainingConfig.allow_test_as_eval: bool = Field(default=False, ...)` with the leak consequence in the description (configs.py, copied from the use_qlora field shape).
- `set_up_trainer` pops the field before `TrainingArguments(**kwargs)`; the leak branch (old trainer.py:234-241) is replaced: test-only datasets now set `eval_dataset=None` AND `eval_strategy="no"` atomically and emit exactly ONE `print("[Warning] ...")` line carrying all three D-04 facts (test split excluded / differs from previous dnallm versions / `finetune.allow_test_as_eval: true` opt-in).
- Opt-in restores test-as-eval with one WARN naming the leak risk ("metrics ... are leaked and must not be reported as held-out performance").
- The old leak assertion `test_eval_falls_back_to_test_split` is inverted into `test_eval_excludes_test_split_without_opt_in`; new `TestEvalSemanticsGuard` covers warn-exactly-once, opt-in + leak WARN, and the train-only/unsplit no-WARN edges. `test_eval_prefers_validation_split` untouched and green.

### Task 2 — Neighbor collisions + evaluate(split=...) override
- Guard + user-set `load_best_model_at_end=True` (post-init value: YAML or extra_args) → matchable `ValueError` ("load_best_model_at_end requires an evaluation split ... Provide a dev/validation split or set finetune.allow_test_as_eval=true"). Never silently downgraded.
- Early stopping with `eval_dataset is None` (guarded test-only OR train-only) → matchable `ValueError` naming early stopping and both remedies; the force-enable path at old trainer.py:298-303 can no longer resurrect best-model selection without an eval set (PITFALLS #1). Dev-split force-enable behavior unchanged (`test_load_best_model_forced_on_when_missing` green, unmodified).
- `evaluate(split=None, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")`: `split=None` forwards only non-default legacy kwargs to `self.trainer.evaluate` (no-arg call → no kwargs); split path routes through `self.trainer.predict` (never `trainer.evaluate`), strips the `test_` predict prefix to canonical unprefixed keys, writes `output_dir/eval_{split}_result.json` with `{"split", "timestamp" (UTC ISO), "metrics"}` (D-02 schema for Phase 11 `aggregate_seeds`); evaluates current weights — no checkpoint reload, no checkpoint parameter (D-03).
- `DNATrainer` class docstring gained the "Evaluation semantics" passage with the D-03 model-selection wording.
- New test classes `TestEarlyStoppingCollision` (4 tests incl. the train-only raise and the opt-in adjacency edge) and `TestEvaluateSplit` (4 tests: predict-routing + JSON, unknown-split ValueError, no-args passthrough, legacy-kwarg passthrough).

### Task 3 — One-pass scaffolding
- `TrainingConfig.use_ia3` field-first (default False, "arrives with the next release's trainer branch"); NO cross-field validators — pinned by `test_use_ia3_has_no_cross_field_rejection_yet` (D-07 interim window). Popped beside `allow_test_as_eval`.
- `Ia3Config` (peft IA3Config-mirrored field set), `VepConfig` (`paradigm` pattern `^(clm|mlm)$`, `context_window` ge=1, `output_dir`), `SweepConfig` (`seeds` min_length=1, `n_bootstrap` ge=1, `small_n_ci` pattern `^(t-interval|omit)$` with the SEED-01 n<10 guard policy documented) — all `Field(default=..., description=...)` in LoraConfig style.
- Registered as `ia3`/`vep`/`sweep` in the `DNALLMConfig` TypedDict and `load_config()` (one `if "<section>" in config_dict:` block each, mirroring the lora block).
- `pyproject.toml`: only `[tool.setuptools.package-data]` gained `"dnallm.configuration" = ["presets/*.yaml"]` — every dependency list byte-identical.
- D-08 owner-side terminology sweep of the two A1-owned files: all "DNA language model(s)" / "DNA Language Model(s)" docstring occurrences → "DNA large language model(s)" (7 occurrences; zero remain).
- Tests: `TestTrainingConfigScaffoldFields`, `TestIa3Config`, `TestVepConfig`, `TestSweepConfig`, `TestStubSectionLoadConfig` (defaults, custom values, pattern/ge/min_length rejections, YAML roundtrip present/absent); pop-tuple test extended with `allow_test_as_eval` and `use_ia3`.
- CHANGELOG `## [Unreleased]` → `### Changed` entry tagged `(REV-01, R1-2c)` (landed in the Task 1 guard commit per D-09; idempotent anchor insert above `## [0.7.1] - 2026-10-08`).

## Verification Results

| Verification | Result |
| --- | --- |
| Task 1 `<verify>`: `pytest tests/finetune/test_trainer.py -q -k "TestEvalSemanticsGuard or TestDatasetSplitWiring"` | 10 passed |
| Tracer feedback gate (verify re-run end-to-end, interactive/end-of-phase mode) | 50 passed (full file) |
| Task 2 `<verify>`: `-k "Collision or Evaluate"` | 12 passed |
| Task 3 `<verify>`: `pytest tests/configuration/test_configs.py tests/finetune/test_trainer.py -q` | 157 passed (final re-run at HEAD-of-lane also 157 passed) |
| pyproject package-data grep + diff-stat | grep count 1; only the one `+` package-data line, no dependency changes |
| `grep -c "(REV-01, R1-2c)" CHANGELOG.md` | 1 (under `## [Unreleased]`, same commit as the guard) |
| `grep -rn "DNA language model\|DNA Language Model"` A1 files | 0 matches |
| `grep -n use_ia3 configs.py` | field only — no validator references it (D-07) |
| ruff format + ruff check on all 5 lane files | clean |
| mypy | environmentally blocked repo-wide (installed numpy stubs use py3.12 `type` statements vs the py310 mypy target — fails on unmodified modules too); under a py312 override the only 2 hits in trainer.py are pre-existing lines (warmup-ratio conversion, mask collator tokenizer arg), byte-identical before this plan |
| Repo-wide fast lane (`pytest tests/ -q`) + `scripts/check_code.py` | NOT run — owner directive 2026-10-09 (reduce long-running tests; targeted verifiers suffice) arrived after a bare `pytest tests/` stall (bare invocation collects slow/network tests; no default marker deselection in pyproject). Killed before completion; no result claimed |
| Cross-lane canonical-name contract | Deferred by its own precondition ("run once BOTH lanes have landed") — 10-02's lane was mid-flight at this plan's close; belongs to phase verification |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `test_infer_predicts_on_test_split` needed an explicit mock value**
- **Found during:** Task 1
- **Issue:** The test builds a trainer over `["train", "test"]` with default mocks; a default Mock's `load_best_model_at_end` is truthy, which under the new collision guard would raise the load_best ValueError and break the test.
- **Fix:** Set `args_cls.return_value.load_best_model_at_end = False` with a comment tying it to the guard (same pattern the existing `test_load_best_model_forced_on_when_missing` already uses).
- **Files modified:** tests/finetune/test_trainer.py
- **Commit:** dae194a

**2. [Rule 3 - Blocking] `TestLoadConfigTypedDict` pinned the exact TypedDict key set**
- **Found during:** Task 3
- **Issue:** `test_typed_dict_is_total_false` asserts the exact `__optional_keys__` frozenset; the plan-mandated ia3/vep/sweep registration changes it.
- **Fix:** Extended the frozenset and annotation assertions with the three keys; extended `test_minimal_task_model_yaml_loads` absent-keys tuple to include them.
- **Files modified:** tests/configuration/test_configs.py
- **Commit:** c751df6

### Process Notes (not code deviations)

**3. [Shared-tree hazard] Commit c751df6 swept in 58 foreign staged files**
- Agent 10-03 (docs sweep) had staged its files in the shared index between my `git add` and `git commit`; a plain commit commits the whole index. c751df6 therefore contains my 5 scaffolding files PLUS 10-03's ~58 docs/docstring files (content-preserving terminology changes, 0 deletions, all content intact in history). A soft-reset fix was impossible: agent 10-02's commit fad6a33 had already landed on top, and history surgery under active concurrent committers risks destroying sibling work. Left for orchestrator awareness; agent 10-03 may find its commit empty ("nothing to commit"). All my subsequent commits use pathspec commits (`git commit -- <paths>`) to avoid recurrence.

**4. [Owner directive] Broad verification skipped**
- Repo-wide fast lane and `scripts/check_code.py` full sweep skipped per the owner directive that arrived mid-verification (2026-10-09: "run ONLY your plan's targeted <verify> commands and this lane's own test files"). All targeted verifiers green (table above).

**5. [Sequencing] CHANGELOG entry committed with Task 1 (guard), not Task 3**
- The plan's Task 3 text and the must_haves conflict on which commit carries the entry; the must_have ("REV-01 Unreleased entry in the same commit as the guard") and D-03/D-09 semantics win: entry landed in dae194a. Task 3's grep verifier passes on the cumulative file.

**6. [Method] Terminology sweep applied via scripted replace**
- The four case variants from the plan applied programmatically across the two A1 files (grep-verified 0 remaining); identical outcome to enumerated edits.

## Auth Gates

None — no authentication-gated operations in this plan.

## Known Stubs

None that block this plan's goal. Deliberate scaffolding by design (documented in-field, not defects): `use_ia3` has no trainer branch until Phase 11 B1 (D-07); the `presets/*.yaml` package-data glob intentionally matches nothing until Phase 11 PEFT-02 creates the directory.

## Threat Flags

None — no security-relevant surface beyond the plan's threat model (result-JSON write path and split-key lookups were registered and dispositioned there; no new endpoints/auth/file-access patterns introduced).

## Self-Check: PASSED

All 7 created/modified artifacts exist on disk; task commits dae194a, 64d6f41, c751df6 are ancestors of HEAD. 157/157 lane tests green at close.

