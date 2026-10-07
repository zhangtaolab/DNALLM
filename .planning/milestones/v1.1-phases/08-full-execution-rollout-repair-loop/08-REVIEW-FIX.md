---
phase: 08-full-execution-rollout-repair-loop
fixed_at: 2026-10-06T23:21:49Z
review_path: .planning/phases/08-full-execution-rollout-repair-loop/08-REVIEW.md
iteration: 1
findings_in_scope: 12
fixed: 12
skipped: 0
status: all_fixed
---

# Phase 08: Code Review Fix Report

**Fixed at:** 2026-10-06T23:21:49Z
**Source review:** `.planning/phases/08-full-execution-rollout-repair-loop/08-REVIEW.md`
**Iteration:** 1
**Scope:** all (fix_scope=all — Critical + Warning + Info)

**Summary:**
- Findings in scope: 12
- Fixed: 12
- Skipped: 0

## Fixed Issues

### CR-01: EVO1 section of the evo notebook never loads an evo-1 model

**Files modified:** `example/notebooks/generation_evo_models/inference.ipynb`, `docs/example/notebooks/generation_evo_models/inference.ipynb`, `tests/examples/test_notebook_execution.py`
**Commit:** 5dc18c6
**Applied fix:** Restored the `model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source="huggingface")` call into cell `f51b8ed4` (the cell after `model_name = "togethercomputer/evo-1-8k-base"`), byte-close to the pre-repair form recovered from `git show c9df253` (the 8k model name is the deliberate 05-FEASIBILITY choice and was kept). The diff is exactly 1 line → 2 lines in the cell source; **no output blob was touched** (execution_count/outputs unchanged) — the orchestrator's local GPU re-execution regenerates outputs honestly as a follow-up commit. The docs mirror was re-synced byte-identically (`cmp` verified). Added `TestEvoNotebookContentContracts` in `tests/examples/test_notebook_execution.py` mirroring the megaDNA sibling pattern: one test pins the evo-1 load cell (model_name + loader + `source="huggingface"`), the second pins that the load precedes the second `DNAInference(` build and that the rebuild references `model=model`/`tokenizer=tokenizer`. Regression-proofed: both tests FAIL against the pre-fix notebook (verified via `git stash` round-trip) and pass post-fix. `scripts/check_notebook_md_sync.py` passes all 24 md/notebook pairs after the restore (docs md already showed the load call; notebook and md now agree again).

### WR-01: ollama num_ctx state contradictory across unit, README, and harness budgets

**Files modified:** `scripts/runner/ollama.service`, `scripts/runner/README.md`, `tests/examples/_execution.py`
**Commit:** 000c743
**Applied fix:** All three artifacts now tell the ONE owner story (2026-10-06 00:52 CST): num_ctx cut DEFERRED entirely; live ollama service NOT reconfigured; the in-repo `OLLAMA_CONTEXT_LENGTH=8192` pin stays committed but INERT until the deferral is lifted and the re-apply step (daemon-reload + restart) runs; the mcp_example pair keeps default-context behavior, cost accepted. The unit's RUNTIME and Environment comments and the README gained an explicit "Deferral status" paragraph; the `_execution.py` budget comment now cites the same deferral state and cross-references the README. The unit's stale model name was updated: `qwen3.8:latest` → `qwen3.5:4b` (261006 swap), explicitly recording that the Modelfile num_ctx probe has NOT been re-run for the swapped model (the 2026-10-05 no-num_ctx probe covered only the replaced model) and that a Modelfile `PARAMETER num_ctx` would silently override the env — re-probe with `ollama show qwen3.5:4b --modelfile` before relying on it. Prose/comment changes only; no behavior changes (unit Environment values untouched).

### WR-02: megaDNA checkpoint-file selection conditions inverted

**Files modified:** `dnallm/models/special/megadna.py`, `tests/models/test_special/test_megadna.py`
**Commit:** b9990d9
**Applied fix:** Replaced the inverted `if m in "megaDNA_updated"` chain (loop member tested as substring of a literal) with an explicit module-level `_MEGADNA_CHECKPOINTS` member→checkpoint mapping. The mapping was verified against the LIVE repo listings fetched from the HF API during this fix: `lingxusb/megaDNA_updated` ships only `megaDNA_phage_145M.pt`; `lingxusb/megaDNA_variants` ships `megaDNA_phage_78M.pt` + `megaDNA_phage_277M.pt`; `lingxusb/megaDNA_finetuned` ships `megaDNA_phage_ecoli_finetuned.pt`. The three shipped repo names keep their historical selections (updated→145M, variants→78M default, finetuned→ecoli); the explicit phage members now select their own-named checkpoints instead of all silently falling to the 145M default. Added `TestMegadnaCheckpointSelection` — 9 parametrized fast tests covering every family member (plus both repo-id spellings) → expected checkpoint filename, through the real handler with `torch.load`/`_get_model_path_and_imports` patched (existing test idiom).

### WR-03: `_handle_megadna_models` mutates the module-level list via `extra`

**Files modified:** `dnallm/models/special/megadna.py`, `tests/models/test_special/test_megadna.py`
**Commit:** f4e6d69
**Applied fix:** The handler now builds and iterates a local `models = megadna_models + ([extra] if extra else [])` list; the module-level registry is never mutated. Added `TestMegadnaExtraDoesNotMutateModuleList`: two extra-carrying calls (matching only via the extra member) both resolve and the module list is asserted unchanged — fails on the old append-mutating code.

### WR-04: guarded dispatch chain can discard a handler's resolved half

**Files modified:** `dnallm/models/model.py`, `tests/models/test_model.py`
**Commit:** 2f4b8e1
**Applied fix:** Investigation first: `_handle_dnabert2_models` returns `(None, None)` or a complete tuple (never bare None), `_load_model_by_task_type` always returns a complete tuple, and `_handle_crossdna_models` results are complete — so the reassignment was NOT load-bearing for any currently-passing path; the flaw was purely latent. The chain now merges per half (`model = stage_model if stage_model is not None else model`, likewise for tokenizer) so a later stage's None half can never discard an earlier stage's resolved half, honoring the documented "never overwritten" invariant (comment updated to describe the merge semantics). Added `TestDispatchChain::test_partial_handler_result_survives_the_chain` with the stub-handler matrix: dnabert2 returns `(sentinel_model, None)`, generic loader returns `(None, sentinel_tokenizer)`, final result is `(sentinel_model, sentinel_tokenizer)` — crashes with AttributeError on the pre-fix code (verified via `git stash` round-trip: fails pre-fix, passes post-fix).

### WR-05: `weights_only=False` full unpickling of a remotely fetched checkpoint

**Files modified:** `dnallm/models/special/megadna.py`, `tests/models/test_special/test_megadna.py`
**Commit:** a7baff1
**Applied fix:** Minimal hardening, both directions. (1) Revision pin: the hub fetch now passes `revision=_MEGADNA_REVISIONS.get(m)`; `_MEGADNA_REVISIONS` carries the models.lock provenance commit `ed298be539e1667b52a1181a6472528a34dd2ef9` for the matched member `megaDNA_updated` (the only repo the lock records — verified during this fix that the repo's live HEAD sha IS that commit, so pinned fetch is byte-equivalent with today's behavior). (2) Safe-load-first: `torch.load(..., weights_only=True)` is attempted first; on refusal the documented fallback loads with `weights_only=False`, with an inline comment stating the trust decision (upstream ships full pickled model objects, not state dicts; the fetch is revision-pinned for the lock-recorded repo; a future state-dict re-upload takes the safe branch with no code change). Added `TestMegadnaLoadHardening` (4 fast tests): pinned fetch for the lock-recorded member, unpinned for members without lock rows, weights_only=True tried first, and exactly-one retry with weights_only=False on refusal. **Residual risk (documented):** `megaDNA_variants` / `megaDNA_finetuned` have no models.lock rows and remain unpinned (see Notes).

### WR-06: `_determine_classifier` crashes with UnboundLocalError on unrecognized head name

**Files modified:** `dnallm/models/model.py`, `tests/models/test_model.py`
**Commit:** bf10205
**Applied fix:** Added the terminal `else` branch raising the project-convention `ValueError(f"Unknown head type {head!r}: expected a name ending in mlp/cnn/lstm/unet or a custom_head class.")`. Added `TestDNALLMforSequenceClassificationInit::test_unknown_head_name_raises_value_error` using the existing `_build_wrapper` idiom with `pytest.raises(ValueError, match=r"Unknown head type.*'attention'")`.

### IN-01: evo2 local-path resolution crashes with bare IndexError

**Files modified:** `dnallm/models/special/evo.py`, `tests/models/test_special/test_evo.py`
**Commit:** 66650c7
**Applied fix:** The `.pt` glob is guarded: an empty result raises `ValueError(f"No .pt checkpoint found in {model_name}")` instead of indexing `[0]` unguarded. Added `TestHandleEvo2Models::test_local_dir_without_pt_raises_value_error` (evo2 stubs installed, empty local dir, `pytest.raises(ValueError, match=...)`).

### IN-02: unclosed file handle in the evo-1 checkpoint loader

**Files modified:** `dnallm/models/special/evo.py`
**Commit:** 5af5929
**Applied fix:** `dotdict(yaml.safe_load(open(config_path)))` → `with open(config_path) as f: global_config = dotdict(yaml.safe_load(f))` (the `# type: ignore` is kept on the `with` line because `config_path` is `str | None` in the `load_checkpoint` signature). No dedicated behavior test, per the fix direction — the existing `TestHandleEvo1Models` tests exercise this exact line through the real packaged `configuration/evo/*.yml` file (FakeStripedHyena receives the parsed config), and all pass.

### IN-03: evo-1 revision gate is case-sensitive

**Files modified:** `dnallm/models/special/evo.py`, `tests/models/test_special/test_evo.py`
**Commit:** c100267
**Applied fix:** `source == "huggingface"` → `source.lower() == "huggingface"`, matching the normalization used by `_handle_evo2_models` and `_get_model_path_and_imports`. Added `test_mixed_case_source_selects_1_1_fix_revision` asserting `revision == "1.1_fix"` for `source="HuggingFace"`.

### IN-04: docs prerequisites reference extras that do not exist in pyproject

**Files modified:** `docs/example/notebooks/finetune_custom_head.md`, `docs/example/notebooks/finetune_generation.md`, `docs/example/notebooks/data_prepare_finetune.md`, `docs/example/notebooks/inference_evo_models.md`, `docs/example/notebooks/inference_megaDNA.md`
**Commit:** 40280e3
**Applied fix:** `.[base,finetune,cuda124]` and `.[base,inference,cuda124]` → `.[base,cuda124]` in the five review-cited pages (base already pulls dev/test/notebook/mcp; cuda124 is a real extra). **Live check run (as required):** `uv pip install --dry-run --no-deps -e '.[base,cuda124]'` resolves with exit 0 and NO warning, while the negative control `uv pip install --dry-run --no-deps -e '.[base,finetune]'` emits exactly `warning: The package dnallm ... does not have an extra named 'finetune'` — proving the corrected names resolve and the old ones were phantom.

### IN-05: deploy job pins `actions/cache@v3`

**Files modified:** `.github/workflows/ci.yml`, `.github/workflows/publish.yml`
**Commit:** c3d74a6
**Applied fix:** Bumped the docs-deploy job's cache step in `ci.yml:1121` AND the publish job's uv cache in `publish.yml:26` (the other v3 occurrence the finding direction named) to `actions/cache@v4`. No `actions/cache@v3` remains anywhere under `.github/`; both workflows re-validate as YAML.

## Skipped Issues

None — all 12 in-scope findings were fixed.

## Test Evidence

Owner rule (dnallm/ lib changes ship with pytest coverage in the same change) — every dnallm/ change above carries its new tests in the same commit, and the mandated fast suite was run at the final fix HEAD:

```
$ .venv/bin/python -m pytest tests/models/test_special/test_megadna.py tests/models/test_model.py \
    tests/models/test_special/test_evo.py tests/utils/test_transformers_compat.py \
    -m "not slow and not giants" -q --tb=short
308 passed, 2 deselected in 4.28s
```

New/extended test classes added this run (all included above):
- `tests/models/test_special/test_megadna.py::TestMegadnaCheckpointSelection` (9 tests, WR-02), `::TestMegadnaExtraDoesNotMutateModuleList` (WR-03), `::TestMegadnaLoadHardening` (4 tests, WR-05)
- `tests/models/test_model.py::TestDispatchChain::test_partial_handler_result_survives_the_chain` (WR-04), `TestDNALLMforSequenceClassificationInit::test_unknown_head_name_raises_value_error` (WR-06)
- `tests/models/test_special/test_evo.py::TestHandleEvo2Models::test_local_dir_without_pt_raises_value_error` (IN-01), `::TestHandleEvo1Models::test_mixed_case_source_selects_1_1_fix_revision` (IN-03)

CR-01 content-contract module (fast subset):

```
$ .venv/bin/python -m pytest "tests/examples/test_notebook_execution.py::TestEvoNotebookContentContracts" \
    "tests/examples/test_notebook_execution.py::TestMegadnaSiblingContentContracts" -q
5 passed in 0.87s
```

Regression proof for the two contract pins (each verified to FAIL against pre-fix code via `git stash` round-trip, then pass post-fix): `TestEvoNotebookContentContracts` (both tests) and `test_partial_handler_result_survives_the_chain`.

Lint/format (ruff, line length 100): `ruff check dnallm/ tests/` — all checks passed; `ruff format --check` over every touched file — clean.

**Where verification ran:** the MAIN checkout at `/home/forrest/Github/DNALLM` (`workflow.use_worktrees=false` in `.planning/config.json`, so this fixer edited/committed directly on branch `phs` per the documented opt-out; no worktree was created and the numbers above are reproducible from this tree). The giants-lane re-execution of the evo notebook was NOT run here — per the owner decision it is handled separately by the orchestrator after these fixes land.

## Notes and Residuals

- **CR-01 outputs:** intentionally left as-is (stale evo2-derived outputs under the EVO1 heading). The orchestrator's local GPU re-execution regenerates them; do not treat the current output blobs as evo-1 evidence until that lands.
- **WR-05 residual:** only `lingxusb/megaDNA_updated` has a models.lock provenance row, so only that fetch is revision-pinned; `megaDNA_variants` / `megaDNA_finetuned` remain unpinned. Recording lock rows for those repos (or loading a state_dict with `weights_only=True` from the pinned local clone) is the medium-term direction. models.lock was NOT modified this run (owner constraint).
- **IN-04 residual:** 15 further docs pages (e.g. `finetune_binary.md`, `inference.md`, `benchmark.md`, `lora_*.md`) still carry phantom `finetune`/`inference`/`benchmark` extras — they predate the phase (commit 009ff10) and were outside the review's cited scope; flagged here for a future docs pass.
- **WR-01:** the live runner was not touched; the qwen3.5:4b Modelfile re-probe (`ollama show qwen3.5:4b --modelfile`) remains an owner action recorded in the README before the env pin can be trusted to govern.

---

_Fixed: 2026-10-06T23:21:49Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
