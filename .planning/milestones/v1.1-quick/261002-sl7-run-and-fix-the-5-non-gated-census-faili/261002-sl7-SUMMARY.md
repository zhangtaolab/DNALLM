---
phase: quick-261002-sl7
plan: 01
subsystem: testing
tags: [transformers-5-compat, notebook-execution, remote-code, pytest, harness]

requires:
  - phase: 05-execution-harness-honest-gates-runner-feasibility
    provides: nbclient execution harness (_execution.py), NOTEBOOK_EXEC_SPECS budgets, ACTIVE/GATED lanes, census evidence
provides:
  - Five census-failing notebooks green and enforced in the ACTIVE pytest lane (13 active total)
  - Five absence-gated transformers-5.x compat shims (config legacy defaults, vendored MambaCache, DebertaV2 dict-vocab normalization, vendored get_head_mask pair, init_weights post_init bookkeeping) with 34 contract tests
  - seed_sandbox (src, dest-relative-to-sandbox) tuple seeding with tmp_path escape guard (T-sl7-03)
  - finetune_generation data-prep fix + honest megaDNA gating with recorded evidence
affects: [phase-8-repair-queue, example-notebooks, docs-mirror, transformers-compat]

actuals:
  tokens: 84438   # chars/4 over 5e598b1..6073f72 (337754 chars; dominated by the 6 committed regression CSVs)
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - closed-map PretrainedConfig.__getattr__ restoring removed 4.x config defaults (extend the map, never a catch-all)
    - probe-gated legacy init_weights wrapper delegating to post_init (bounded depth-2, no double init)
    - seed_sandbox cross-directory sibling inputs with resolved-destination-under-tmp_path guard

key-files:
  created:
    - example/notebooks/data_prepare/finetune/{train,test,dev}.csv (+ docs mirrors)
  modified:
    - dnallm/utils/transformers_compat.py
    - tests/utils/test_transformers_compat.py
    - example/notebooks/embedding_attention.ipynb (+ docs mirror)
    - example/notebooks/data_prepare/finetune/finetune_data.ipynb (+ docs mirror)
    - example/notebooks/finetune_generation/finetune_generation.ipynb (+ docs mirror)
    - tests/examples/_execution.py
    - tests/examples/test_notebook_execution.py

key-decisions:
  - "finetune_generation straddles the gate boundary: data-prep fixed + DNAGPT half verified green, second half honestly gated on the megaDNA package via _gate_megadna (owner deferral stands; installing megaDNA would force the two deferred megaDNA notebooks to execute - scope explosion). FLAGGED FOR OWNER REVIEW."
  - "The NER is_decoder/add_cross_attention shim SUPERSEDES the 05-04 D-07 rung termination (STATE.md had typed it 'not vendored-pure-helper territory'); owner instruction of 2026-10-02 (fix all non-gated census failures now) overrides D-09's typed-skip disposition. FLAGGED FOR OWNER REVIEW."
  - "Sibling CSVs committed via git add -f (root .gitignore:85 *.csv), matching the tracked example/notebooks/inference/test.csv precedent (no negation rules exist in the root .gitignore)."
  - "Two additional rungs discovered during execution fixed per the established pattern with tests in the same change: vendored get_head_mask + _convert_head_mask_to_5d, and the init_weights post_init-bookkeeping wrapper."
  - "check_docs_sync.py locally reports exactly the 4 pre-existing owner-drift DIFFERs (mcp_example x2, benchmark.ipynb, inference.ipynb) - the same set as session start; all 261002-sl7-touched files are byte-identical both sides. CI on a clean checkout sees none of this drift."

patterns-established:
  - "Residual-rung repair loop: run promoted notebook -> read census-style terminal error -> grep ALL cached remote code for the removed-API surface -> absence-gated shim + contract tests in the same commit"

requirements-completed: []

status: complete

coverage:
  - id: D1
    description: "Three planned transformers-5.x shims (config defaults, vendored MambaCache, DebertaV2 dict vocab) active after import dnallm with RED->GREEN contract tests"
    verification:
      - kind: unit
        ref: "tests/utils/test_transformers_compat.py::TestPretrainedConfigLegacyDefaults / TestVendoredMambaCache / TestDebertaVocabDictNormalization (20 tests, 76 total in file, all pass)"
      - kind: unit
        ref: "live: from transformers import PretrainedConfig, cache_utils; import dnallm -> is_decoder False, add_cross_attention False, cache_utils.MambaCache present; AutoTokenizer.from_pretrained('zhangtaolab/plant-dnabert-BPE') loads (DebertaV2Tokenizer, vocab 8000)"
      - kind: integration
        ref: "notebooks/data_prepare/finetune/finetune_data.ipynb end-to-end PASSED in pytest lane (57.6s)"
      - kind: integration
        ref: "notebooks/inference_for_tRNA/inference.ipynb end-to-end PASSED (20.3s)"
  - id: D2
    description: "Notebook/data unblocks: embedding_attention dnallm import, finetune_data sibling CSVs, finetune_generation pinned Ensembl wget - all byte-identical example/docs pairs"
    verification:
      - kind: unit
        ref: "cmp example/... vs docs/... for all 3 notebooks + 6 CSVs (identical); Ensembl release-62 URL HEAD-checked 200 (Content-Length 14458895)"
      - kind: integration
        ref: "notebooks/embedding_attention.ipynb end-to-end PASSED (21.1s)"
  - id: D3
    description: "Harness sibling-input seeding + lane promotion 8->13 ACTIVE + finetune_generation GATED on megaDNA; full slow lane green"
    verification:
      - kind: unit
        ref: "tests/examples/test_notebook_execution.py::TestSeedSandbox (3 kernel-free tests: bare-path copy, sibling-position copy, escape guard)"
      - kind: e2e
        ref: "pytest tests/examples/test_notebook_execution.py -m slow --timeout 7500: 15 passed, 8 skipped (all audit-matched typed skips incl. optional-dep: finetune_generation megaDNA gate), 0 failed, 1:04:24"
      - kind: e2e
        ref: "notebooks/finetune_NER_task/finetune_NER_task.ipynb PASSED (924.9s real training); notebooks/benchmark/benchmark.ipynb PASSED (49.9s)"
      - kind: other
        ref: ".scratch/sl7/finetune_generation-evidence.json: first failure moved from cell 2 FileExistsError to the megaDNA load cell ImportError naming the megaDNA package; fresh DNAGPT training green (train_loss 1.279986, epoch 2.0, 425s)"

---

# Quick Task 261002-sl7: Run and fix the non-gated census-failing notebooks Summary

Five of the six in-scope census failures now execute green end-to-end inside the enforced
pytest ACTIVE lane (8 -> 13 notebooks); the sixth (finetune_generation) had its data-prep
half fixed and its megaDNA half moved behind an honest probe gate. Getting there took five
transformers-5.x compat shims (three planned + two discovered rungs), three minimal notebook
edits with docs mirrors, six committed sibling CSVs, and a harness upgrade that seeds
cross-directory inputs into the sandbox.

## What Was Built

**Shims (dnallm/utils/transformers_compat.py, all absence-gated + idempotent sentinels):**

1. `_patch_pretrained_config_legacy_defaults` - `PretrainedConfig.__getattr__` answering a
   CLOSED map (`is_decoder: False`, `add_cross_attention: False`) that transformers 5.x
   removed; explicit sets, unknown attributes (still AttributeError) and deepcopy
   round-trips unchanged. Unblocks remote modeling_esm.py:335/584-585 (NER load).
2. Vendored `MambaCache` (verbatim semantics from upstream v4.49.0 cache_utils.py,
   adaptations documented: import locality + warn-once over `warnings`) attached to
   `transformers.cache_utils` where absent. Unblocks tRNADetector remote
   modeling_mamba.py:27 (`MambaCache(config, batch_size, device=..., dtype=...)`).
3. `_patch_deberta_vocab_dict` - the `convert_to_native_format` hook on
   DebertaV2Tokenizer normalizes a dict vocab to `list(vocab.items())` (insertion order =
   rank order); fixes `TypeError: 'dict' object is not an instance of 'Sequence'` for
   plant-dnabert-BPE (8000-entry dict reproduced live with an __init__ spy).
4. Vendored `get_head_mask` + `_convert_head_mask_to_5d` (v4.49.0 modeling_utils.py) on
   PreTrainedModel - 5 live remote call sites inside EsmModel.forward (benchmark notebook).
5. `_patch_legacy_init_weights_bookkeeping` - remote classes ending `__init__` with the bare
   4.x `self.init_weights()` entry skip 5.x `post_init` bookkeeping, so
   `from_pretrained`'s `_move_missing_keys_from_meta_to_device` crashes on the missing
   `all_tied_weights_keys`; the wrapper routes such receivers through the real `post_init`
   first (probe-gated so 4.x stays native, bounded depth-2 so weights init exactly once).

**Notebook/data edits (example/ + docs/ byte-identical pairs):** embedding_attention cell 1
appends `import dnallm` (its kernel previously never activated the shims); finetune_data
cells 14-15 repoint to `./{train,test,dev}.csv` with the three regression CSVs committed as
notebook siblings; finetune_generation cell 1 uncomments the pinned Ensembl Plants
release-62 wget (HEAD-checked 200 this session).

**Harness (tests/examples/_execution.py + test_notebook_execution.py):**
`seed_sandbox` accepts `(src, dest-relative-to-sandbox)` tuples (T-sl7-03 guard: resolved
destination must stay under tmp_path, ValueError on escape); the benchmark notebook is
seeded with `../inference/test.csv` through a per-notebook `_NOTEBOOK_EXTRA_INPUTS` table -
the root-cause fix for the labels KeyError (a missing path string was silently treated as
one "sequence"); ACTIVE_NOTEBOOKS 8 -> 13; finetune_generation joined GATED_NOTEBOOKS under
`_gate_megadna` with the 7200s outer-timeout override.

## Flagged Decisions for Owner Review

1. **finetune_generation gating.** The notebook straddles the gate boundary: its second
   half loads `lingxusb/megaDNA_updated` through `dnallm/models/special/megadna.py`, which
   raises ImportError without the megaDNA GitHub package. Installing megaDNA would flip the
   `_gate_megadna` probes green and force the two owner-deferred megaDNA gated notebooks
   (generation_megaDNA, finetune_custom_head) to execute for real - scope explosion this
   round. Decision executed per plan: data-prep cell fixed, DNAGPT half verified green
   (fresh training, loss 1.28, epoch 2.0), notebook placed in GATED_NOTEBOOKS with
   `_gate_megadna` (typed `optional-dep:` skip, audit-matched). Evidence:
   `.scratch/sl7/finetune_generation-evidence.json` + partial artifact under
   `.scratch/sl7/artifacts/finetune_generation/` - first failure moved from cell 2
   (FileExistsError) to the megaDNA load cell raising ImportError naming the megaDNA
   package. The megaDNA round resolves the second half.
2. **D-07 supersession.** STATE.md recorded the is_decoder/add_cross_attention restoration
   as "not vendored-pure-helper territory" with a typed-skip + owner disposition (D-09).
   The owner instruction of 2026-10-02 (fix all non-gated census failures NOW) supersedes
   that termination for this round: Task 1a restores the removed 4.x config defaults via a
   narrowly-scoped closed-map `PretrainedConfig.__getattr__` (proven live before planning).
   The 05-04 record and this supersession should be reconciled at the next STATE update.

## Commits

- 98c39f8 fix(quick-261002-sl7): three transformers-5.x compat shims + 20 contract tests
- 1cb5994 fix(quick-261002-sl7): notebook/data unblocks + six sibling CSVs (example/docs pairs)
- 7832ed4 fix(quick-261002-sl7): get_head_mask + init_weights bookkeeping rungs + 14 contract tests
- 6073f72 test(quick-261002-sl7): seed_sandbox tuples + lane promotion 8->13 + megaDNA gate + 3 unit tests

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Sibling CSVs rejected by the root .gitignore (`*.csv`, line 85)**
- Found during: Task 2 commit
- Fix: `git add -f` for the six CSVs, matching the tracked
  `example/notebooks/inference/test.csv` sibling precedent (no `!` negation rules exist;
  a new negation would weaken the global data-file protection). Amended into 1cb5994.
- Files: the six train/test/dev.csv under example/ and docs/

**2. [Rule 1/3 - Rungs] Two additional transformers-5.x rungs beyond the plan's three**
- Found during: Task 3(c) iterate-to-green (embedding_attention/tRNA/NER hit
  `all_tied_weights_keys`; benchmark hit `get_head_mask`)
- Fix: both fixed per the plan's established pattern (rung inventory grepped across ALL
  cached remote code first - these were the only remaining `self.get_*`/init surfaces);
  each shim shipped with contract tests in the same change (7832ed4).
- Files: dnallm/utils/transformers_compat.py, tests/utils/test_transformers_compat.py

**3. [Environment] check_docs_sync.py cannot go fully green locally**
- The gate reports exactly the 4 pre-existing owner-drift DIFFERs (mcp_example x2,
  benchmark.ipynb, inference.ipynb) - identical set to session start; every 261002-sl7
  file pair is byte-identical. Staging or discarding the user's residue is forbidden
  (working-tree warning), and CI compares a clean checkout where the drift does not
  exist. Treated as satisfied-with-evidence, not a task failure.

**4. [Environment] First full sweep killed at the 30-min background limit**
- The initial sweep run was SIGKILLed by the execution harness's background time cap while
  finetune_multi_labels was mid-training; the in-flight test was marked F with no traceback
  preserved. The test passes standalone (23:43) AND in the re-run full sweep (verbose, 2h
  budget): final state 15 passed / 8 audit-matched skips / 0 failed in 1:04:24.

## Environment Notes

- `.venv/bin/mypy dnallm/utils/transformers_compat.py` fails inside numpy's own stubs
  (`numpy/__init__.pyi:737: Type statement is only supported in Python 3.12+` vs the
  project's mypy python_version 3.10) before reaching the shim file - pre-existing,
  unrelated to this change; CI runs mypy advisory (`|| true`).
- `pytest tests/examples/_execution.py` collects no tests (private `_`-prefixed helper,
  `python_files = test_*.py`); its contract is covered by `TestSeedSandbox` in
  test_notebook_execution.py (3 passing kernel-free tests).
- The plan's census driver (`.scratch/census_driver.py`) is stale against the current
  `_execution.py` (it reads a removed `test_timeout` spec key); targeted pytest runs were
  used instead, as the plan allows.
- `gsd_run` is not installed in this environment; the optional WINDOWS ledger entry for
  the megaDNA gate deviation could not be appended (best-effort per contract). The
  deviation is fully documented above and in the flagged decisions.
- The committed finetune_generation notebook ships stored author outputs (including a
  stale error at its training cell); nbclient stops at the megaDNA load cell, so cells
  after it in any partial artifact carry the author's outputs, not fresh ones. The
  evidence JSON records which cells ran fresh.

## Self-Check: PASSED

- Files exist: dnallm/utils/transformers_compat.py, tests/utils/test_transformers_compat.py,
  tests/examples/_execution.py, tests/examples/test_notebook_execution.py,
  example/docs notebook + CSV pairs (cmp-verified identical).
- Commits exist on phs: 98c39f8, 1cb5994, 7832ed4, 6073f72 (git log verified).
- No forbidden residue staged: working tree shows only the pre-existing
  mcp_example/benchmark/inference notebook drift and unrelated planning docs.
- Final lanes: tests/utils/test_transformers_compat.py 76 passed; TestSeedSandbox 3 passed;
  full slow notebook lane 15 passed / 8 skipped (audit-matched) / 0 failed.
