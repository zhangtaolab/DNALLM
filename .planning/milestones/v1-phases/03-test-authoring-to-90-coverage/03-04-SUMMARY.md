---
phase: 03-test-authoring-to-90-coverage
plan: 04
subsystem: testing
tags: [pytest, coverage, datahandling, finetune, huggingface-trainer, tmp_path, optuna, altair]

requires:
  - phase: 03-test-authoring-to-90-coverage
    provides: wave-3 mcp closeout (6/110), coverage-wave3-missing.txt ranked worklist, suite at 85.89%
  - phase: 01-harness-integrity-measured-baseline
    provides: measured baseline + coverage tooling of record
provides:
  - datahandling/finetune area closed from 406 missing lines to 10 (gate ≤ 100 — passed with 90 lines of slack; research residual estimate was 46)
  - 104 new behavior tests (47 → 151 collected in tests/datahandling/test_dna_dataset.py; 46 in the new tests/finetune/test_trainer.py with zero skip calls)
  - Suite crossed the milestone target: 91.24% (6,751/7,399) at the wave-4 census — above the >90.5% landing target, before wave 5
  - coverage-wave4-missing.txt — re-ranked worklist input for wave 5
  - One latent source bug found and recorded (raw_reverse_complement discards its map result — pinned as a no-op, not fixed)
affects: [03-test-authoring-to-90-coverage, 04-coverage-gate-ci]

actuals:
  tokens: 20933  # chars/4 over the realized tests diff (83,734 chars); estimate was 40,000 (confidence: low)
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Content-level local-format round-trips: one tmp_path file per format asserting exact sequences/labels/column mapping — no mocks anywhere in the loader path"
    - "Attribute-shaped tokenizer fakes (_ConfigProbeTokenizer / _RecordingTokenizer) to pin the _get_tokenizer_config fallback chain branch-by-branch without a real tokenizer"
    - "HF Trainer boundary mocking: patch dnallm.finetune.trainer.Trainer/TrainingArguments at their import site, drive real config mapping; concrete numerics injected onto the mocked args for arithmetic paths (warmup conversion)"
    - "transformers-version seam: patch dnallm.finetune.trainer.transformers_version with Version('4.55.0') to execute the pre-v5 save branches on an installed 5.x"
    - "Ruff S105 indirection: named constants (PAD_VALUE, ...) instead of *_token string literals — only tests/conftest.py carries the hardcoded-password-string exemption"

key-files:
  created:
    - tests/finetune/test_trainer.py
    - .planning/phases/03-test-authoring-to-90-coverage/coverage-wave4-missing.txt
  modified:
    - tests/datahandling/test_dna_dataset.py
    - .planning/phases/03-test-authoring-to-90-coverage/deferred-items.md

key-decisions:
  - "raw_reverse_complement is a no-op (ds.map result discarded at data.py:983): pinned as-is by test_raw_reverse_complement_leaves_sequences_unchanged per the plan's 'assert what it does' instruction — recorded as a latent bug in deferred-items.md, not fixed (lines covered either way; fixing would change user-visible behavior outside bug-fix scope)"
  - "The plot_statistics chart chain (~180 statements) is objective-required coverage beyond Task 2's literal text — without it the ≤100 area gate is arithmetically unreachable (03-02/03-03 precedent); charts are saved to tmp_path .html and the altair data transformer is restored to default in an autouse teardown to avoid cross-test spec contamination"
  - "Trainer tests mutate the tracked fixture config per test (output_dir → tmp_path, hyperparameter_search, lora) instead of forking a new YAML — load_config is re-read per test so no state leaks"
  - "The warmup_ratio→warmup_steps conversion is covered by injecting concrete numerics onto the mocked TrainingArguments (max_steps=-1/10, batch=1) because the tracked YAML's warmup_ratio is popped into a private field the mock never computes"

patterns-established:
  - "Attribute-shaped probe tokenizer: __init__(**attrs) + setattr loop exposes exactly the attributes a fallback branch checks, making hasattr-guarded chains testable one branch at a time"
  - "Module-attribute version pinning: runtime version comparisons read a module-level constant, so one patch executes the legacy branch against the installed library"

requirements-completed: [TEST-04]

coverage:
  - id: D1
    description: "tests/datahandling/test_dna_dataset.py TestLocalFormatRoundTrips (19): one real load path per format (csv/tsv/json/parquet/fasta/txt/pkl) through tmp_path files with exact sequence/label/column-map assertions; multiline FASTA join, header-without-sep labels, list-input concatenation + rejections, headerless csv→txt fallback, txt-with-header→csv route, quoted fields, non-numeric label preservation, missing label column"
    requirement: TEST-04
    verification:
      - kind: unit
        ref: "tests/datahandling/test_dna_dataset.py (66 passed at Task 1 verify; tracer gate re-ran the full <automated> block end-to-end green)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Tokenization surface (43 tests): seq+token classification pipelines with exact input_ids/attention_mask/token/label lists via SimpleDNATokenizer; the full _get_tokenizer_config pad_id/pad_token/eos fallback matrix and sep_token selection via attribute-shaped fakes; uppercase/lowercase/seq_sep/padding_side paths; remove_unused_columns on both dataset shapes"
    requirement: TEST-04
    verification:
      - kind: unit
        ref: "tests/datahandling/test_dna_dataset.py (151 passed at Task 2 verify; data.py 324 -> 8 missing)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Transformations and loaders: reverse-complement augmentation (A<->T C<->G on crafted non-palindromic input, doubling, labels preserved), concat variant, raw no-op pin, random_generate (replace/append/dict-append with label_func), split disjointness by set comparison, hand-computed stats incl. median, DataFrame + rejection statistics paths, data-type helper matrix, remote loaders mocked at the modelscope/load_dataset boundary, preset registry + load chain + column standardization"
    requirement: TEST-04
    verification:
      - kind: unit
        ref: "tests/datahandling/test_dna_dataset.py (151 passed; from_modelscope/from_huggingface asserted via MsDataset.load / load_dataset call args — no network)"
        status: pass
    human_judgment: false
  - id: D4
    description: "plot_statistics chart chain (objective-required): classification/regression/multi-classification/multi-regression chart composition + DatasetDict split concatenation saved as .html under tmp_path, pre-statistics and unknown-data-type ValueErrors, empty-row multi-label parsing"
    requirement: TEST-04
    verification:
      - kind: unit
        ref: "tests/datahandling/test_dna_dataset.py TestPlotStatistics (8 tests; artifacts asserted non-empty under tmp_path, altair transformer restored)"
        status: pass
    human_judgment: false
  - id: D5
    description: "tests/finetune/test_trainer.py (46, zero skip calls): TrainingArguments mapping (fixture values land, internal fields popped, extra_args override, wrapper keeps columns), dataset split/eval selection matrix incl. unsplit and missing-train, per-task metrics binding (parametrized) + MLM collators, early stopping (wired/forced-load-best/off), LoRA/QLoRA wrapping at the peft boundary, DataParallel, warmup conversion (epoch + max_steps + explicit-steps), optuna hp_space float/int/step + search (disabled raise, no-optuna raise, happy path), v5 + pre-v5 save paths, torch.save fallback, infer, customize_trainer, plot_history"
    requirement: TEST-04
    verification:
      - kind: unit
        ref: "tests/finetune/test_trainer.py (46 passed; HF Trainer/TrainingArguments patched at dnallm.finetune.trainer.* — no training, no tracker construction, no network)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Wave-4 gate: full census 1531 passed / 7 allowlisted skips / 0 failed / 931s; audit_skips exit 0; area missing 10 ≤ 100 (data.py 8, dataset_auto 0, trainer.py 2); suite 91.24% (6751/7399); coverage-wave4-missing.txt committed; pragma budget still exactly 3; pyproject coverage/pytest config untouched; tree clean"
    requirement: TEST-04
    verification:
      - kind: command
        ref: "pytest --junitxml --cov census (exit 0) -> coverage json -> sum(missing_lines over datahandling/ + finetune/trainer.py) = 10; scripts/audit_skips.py exit 0"
        status: pass
    human_judgment: false

duration: 39 min
completed: 2026-09-30
status: complete
commits: 3
plan_head_before: 7a4699e734633fd849b48b2f23459544a4f26ed1
plan_head_after: 09f905944ab53956e7bf6c026ad7cd7604c465cd
---

# Phase 3 Plan 4: Datahandling/Finetune Wave — Test Authoring Summary

**104 content-level tests closing the datahandling/finetune area from 406 to 10 missing lines (gate ≤ 100) via tmp_path format round-trips, exact-id tokenization pipelines, boundary-mocked trainer wiring — and the suite crossed the milestone line to 91.24%, above the >90.5% target, one wave early**

## Performance

- **Duration:** 39 min (incl. 15.5-min full census)
- **Started:** 2026-09-30T11:42:30Z
- **Completed:** 2026-09-30T12:22:22Z
- **Tasks:** 3/3
- **Files modified:** 3 code/artifact files (1 extended test file, 1 new test file, 1 coverage artifact) + 1 planning doc

## Post-Wave Measurement (full census, both roots, slow included)

- **Census:** 1531 passed / 7 skipped (all allowlisted) / 0 failed / exit 0 — 931s (slowest: real-model trainer workflow 193s, config-file leg 190s)
- **Suite coverage:** **91.24%** (6,751 covered / 648 missing on 7,399 stmts) — was 85.89% after wave 3. The >90.5% landing target is met before wave 5 adds the cli/compat area
- **Area missing sum: 10 (gate ≤ 100 — passed with 90 lines of slack; research residual estimate was 46)**
  - data.py 8 · dataset_auto.py 0 · datahandling/__init__.py 0 · finetune/trainer.py 2
- `scripts/audit_skips.py` exit 0 on the census junit; zero new skips introduced
- Fast legs at task boundaries: 1359 → 1379 passed (Task 1); datahandling file 66 → 151 (Task 2); trainer file 46/46 (Task 3)
- Pragma budget: still exactly 3 under `dnallm/`; `pyproject.toml` coverage/pytest config untouched; tree clean (only the known gitignored import-time `logs/dnallm.log` sink)

## Accomplishments

- Every supported local format is proven end-to-end with real files under tmp_path — exact sequences, labels and column mappings asserted (csv/tsv/json/parquet/fasta/txt/pkl), plus the parsing branches the file walk revealed: multiline FASTA records, headerless-CSV→txt fallback, txt-with-header→csv routing, quoted fields, file-list concatenation and the three list-input rejections
- The tokenizer config fallback chain is pinned branch-by-branch with attribute-shaped fakes (pad_id via attribute/encode/eos fallback; pad_token via decode/convert_ids_to_tokens/decode_token; eos via sep/pad; sep_token degradation to ""), and both classification pipelines assert exact input_ids/attention_mask/token/label lists for known sequences
- Augmentation asserted on crafted non-palindromic input (A↔T, C↔G), doubling and label preservation; splits proven disjoint by set comparison with sizes within tolerance; stats hand-computed including the median
- Remote loaders covered hermetically at the modelscope/load_dataset import sites — which loader ran, its arguments, and the renamed result; the preset registry and load chain (task→data_dir, separators, 1024 max_length) pinned without any download
- DNATrainer wiring fully covered by fast tests with the HF boundary mocked: config→TrainingArguments mapping (including the six popped internal fields), the train/eval split-selection matrix, per-task metrics binding, MLM collators, early stopping, LoRA/QLoRA, DataParallel, warmup_ratio conversion, optuna search, both transformers save contracts, infer, and plot_history — the slow real-model file no longer carries wiring coverage alone
- One latent source bug found: `raw_reverse_complement` discards its `Dataset.map` result (data.py:983) — the method is a no-op; pinned as-is and recorded in deferred-items.md

## Task Commits

1. **Task 1: local file-format round-trips (tracer)** — `52b34d0` (test)
2. **Task 2: tokenization, augmentation, splits, stats, remote loaders, presets** — `39c15bc` (test)
3. **Task 3: fast trainer wiring tests + wave-4 re-measure** — `09f9059` (test + artifact)

**Plan metadata:** this commit (docs)

Tracer feedback gate (Task 1, interactive + end-of-phase + automated-only verify): re-ran the full `<automated>` block end-to-end on the committed HEAD — green; expanded without a checkpoint per the #3299 precedence chain.

## Files Created/Modified

- `tests/datahandling/test_dna_dataset.py` — 47 → 151 collected tests (+104): round-trips, tokenization, augmentation, splits/stats, data-type helpers, remote loaders, presets, plot chain
- `tests/finetune/test_trainer.py` — NEW, 46 tests, zero skip calls
- `.planning/phases/03-test-authoring-to-90-coverage/coverage-wave4-missing.txt` — re-ranked worklist for wave 5
- `.planning/phases/03-test-authoring-to-90-coverage/deferred-items.md` — raw_reverse_complement latent-bug entry

## Decisions Made

- `raw_reverse_complement` pinned as a no-op rather than fixed (latent bug recorded in deferred-items.md — 03-01/03-02 precedent: lines are covered either way, and fixing changes user-visible behavior outside this wave's bug-fix scope)
- The plot_statistics chain landed in Task 2 as objective-required coverage (~180 statements inside the area gate — the plan's task text alone cannot reach ≤100)
- Trainer tests reuse the tracked `test_finetune_config.yaml` through `load_config` per test and mutate the returned Pydantic objects (output_dir → tmp_path; lora/hyperparameter_search injected as config objects) — no fixture YAML forked
- Ruff S105 (hardcoded-password-string) fires on any `*_token` string literal in test files (only `tests/conftest.py` is exempt repo-wide) — resolved with named constants rather than noqa noise or a lint-config edit

## Deviations from Plan

### Verify-command adjustments (no code impact)

**1. [Verify substitution] `grep -c "::"` collect tripwires**
- **Found during:** Tasks 1-3 verify blocks
- **Issue:** pytest 9.1.1's `--collect-only -q` emits no `::` separators (established waves 1-3)
- **Fix:** enforced the equivalent "N tests collected" summary counts — 66 ≥ 35 (Task 1), 151 ≥ 55 (Task 2), 38 ≥ 10 (Task 3); windows-ledger deviation entry appended

### Objective-required coverage beyond the plan's task text (03-02 precedent)

- The `plot_statistics` chart chain (~180 statements) was assigned to no task but is arithmetically required by the ≤100 gate → covered in Task 2 with tmp_path `.html` saves and an autouse teardown restoring altair's default data transformer
- Trainer additions beyond the literal task list: warmup_ratio→warmup_steps conversion, pre-transformers-5 save branches (via the `transformers_version` module seam), the search-side torch.save path, `infer()`, `customize_trainer()`, and `plot_history()` — all inside the plan's own area gate and files

---

**Total deviations:** 2 verify-command substitutions + 2 objective-required coverage placements
**Impact on plan:** No scope creep — every addition maps to uncovered statements inside the plan's area gate; no new dependencies; pragma budget intact at 3; allowlist untouched.

## Accepted-Uncovered Residual Ledger (datahandling/finetune area, documented — never pragma'd)

| File | Missing | Lines | Justification |
|------|---------|-------|---------------|
| dnallm/datahandling/data.py | 8 | 512-513, 581, 1232-1233, 1411, 1420-1421 | 512-513/581/1232-1233/1411 are dead defensive branches (the pad_id attr is unconditionally harvested by an earlier loop; the tokenize-closure None guard runs after the outer None check; `__data_type__`'s first-label-None exit is unreachable because `_extract_labels` always yields a list and empty lists are rejected earlier; `_create_final_chart`'s non-dict stats else runs only if `statistics()` never assigned). 1420-1421 is the `chart.show()` browser branch of `_display_or_save_chart` — deliberately not driven (would open a browser/raise without a renderer) |
| dnallm/finetune/trainer.py | 2 | 56-57 | Module-level `except ImportError: optuna = None` — reachable only without optuna installed; covering it needs module-reload machinery that would mutate shared module state mid-suite |

The research-anticipated residuals ("exotic format/header combos, optuna/megatron-adjacent wiring") did not materialize: every exotic format/header combination the file walk revealed was covered, and the optuna wiring itself is fully covered (only the import guard is not).

## Known Stubs

None — every new test asserts observable behavior (exact contents, call arguments, ordered values, file existence); no placeholder logic introduced.

## Issues Encountered

- Ruff's S105 flags `*_token` string literals in test files (only conftest is exempt) — resolved via named constants; no lint-config change
- `augment_reverse_complement`'s first draft asserted a palindromic sequence (its own reverse complement) — caught while drafting and replaced with a crafted non-palindromic input so the complement is actually observable

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 4 complete at 10/100; the suite is already above the 90.5% milestone target at 91.24% — wave 5 (cli/compat + the tasks/configuration orphans) adds margin rather than necessity
- Remaining ranked gaps for wave 5 from `coverage-wave4-missing.txt`: cli (225: mutagenesis 71, cli.py 108, inference 19, train 13, config_generator 14), utils (transformers_compat 45, logger 39), tasks/metrics 31, configuration 9, plus tail residuals in inference files
- Suite runtime: census 931s (+35s vs wave 3; +151 tests); the fixed ~60s test_timeout cost and the two ~190s real-model trainer legs remain the dominant costs (deferred-items has both)
- Windows ledger: one deviation entry appended this wave (the `::` tripwire substitution, so wave 5's plans do not repeat it)

---
*Phase: 03-test-authoring-to-90-coverage*
*Completed: 2026-09-30*

## Self-Check: PASSED

Created files exist on disk (tests/finetune/test_trainer.py, coverage-wave4-missing.txt); all three task commits (52b34d0, 39c15bc, 09f9059) present in history; commits measured from the plan ledger (7a4699e → 09f9059 = 3).
