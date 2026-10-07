---
phase: 08-full-execution-rollout-repair-loop
reviewed: 2026-10-06T17:22:06Z
depth: standard
files_reviewed: 34
files_reviewed_list:
  - dnallm/models/model.py
  - dnallm/models/special/evo.py
  - dnallm/models/special/megadna.py
  - dnallm/utils/transformers_compat.py
  - docs/example/mcp_langchain.md
  - docs/example/mcp_pydantic_ai.md
  - docs/example/notebooks/data_prepare_finetune.md
  - docs/example/notebooks/finetune_custom_head/finetune.ipynb
  - docs/example/notebooks/finetune_custom_head.md
  - docs/example/notebooks/finetune_generation/finetune_generation.ipynb
  - docs/example/notebooks/finetune_generation.md
  - docs/example/notebooks/generation_evo_models/inference.ipynb
  - docs/example/notebooks/generation_megaDNA/inference.ipynb
  - docs/example/notebooks/inference_evo_models.md
  - docs/example/notebooks/inference_megaDNA.md
  - example/notebooks/finetune_custom_head/finetune.ipynb
  - example/notebooks/finetune_generation/finetune_generation.ipynb
  - example/notebooks/generation_evo_models/inference.ipynb
  - example/notebooks/generation_megaDNA/inference.ipynb
  - .github/workflows/ci.yml
  - models.lock
  - pyproject.toml
  - scripts/runner/ollama.service
  - scripts/runner/README.md
  - tests/examples/_execution.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
  - tests/examples/test_script_execution.py
  - tests/models/test_model.py
  - tests/models/test_special/test_evo.py
  - tests/models/test_special/test_megadna.py
  - tests/test_extras_guard.py
  - tests/utils/test_transformers_compat_np.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 08: Code Review Report

**Reviewed:** 2026-10-06T17:22:06Z
**Depth:** standard
**Files Reviewed:** 34
**Status:** issues_found

## Summary

First review of phase 08's full evaluation scope at current HEAD. The review covered the library changes (model.py, evo.py, megadna.py, transformers_compat.py), the CI/runner wiring (ci.yml, ollama.service, README), the execution harness and its tests, the pyproject/models.lock provenance surface, and the four repaired example-notebook families plus their doc mirrors (mirror equality re-verified byte-for-byte with `cmp` — all four pairs identical).

The compat-shim ladder is in good shape: every installer is absence-gated, sentinel-idempotent, and contract-tested, including the numpy `fromstring` rung (probe-based detection of the numpy-2 raising stub, writable-copy semantics, text-mode refusal). The budget arithmetic the phase's harness contract depends on was re-derived independently and holds everywhere: every `cell_timeout`/`timeout_s` sits strictly below its per-test pytest-timeout mark (max ACTIVE cell 3600 vs class 7200; the `_TIMEOUT_7200_GATED` override set covers exactly the four gated entries with cell 3600 under the 3600 class mark; showcase 1200/3600/1200 vs 2400/5400/2400; marimo 1200/3600 vs 7200; script 3000 vs 3600). The already-dispositioned phase-05/06/07 findings were not re-raised.

However, one critical defect invalidates a headline phase claim, and several cross-file contradictions and latent library bugs were found.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

### Critical Issues

### CR-01: EVO1 section of the evo notebook never loads an evo-1 model — the "evo family repair" execution evidence is evo2 output

**File:** `example/notebooks/generation_evo_models/inference.ipynb` (code cell 13, the cell after the `## EVO1` markdown; identical in the `docs/example/notebooks/generation_evo_models/inference.ipynb` mirror)
**Issue:** Cell 13 only assigns `model_name = "togethercomputer/evo-1-8k-base"` — the `load_model_and_tokenizer(model_name, ...)` call that followed it was dropped in the 08-06 repair commit `a7ed221` (verified: the pre-repair version at `c9df253` contains `model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source="huggingface")`). Cell 14 then rebuilds `DNAInference(model=model, tokenizer=tokenizer, ...)` over the **still-bound evo2 model and tokenizer from cell 6**, so the entire "## EVO1" section generates/scores with `arcinstitute/evo2_1b_base` and the committed output blobs under that heading are evo2 outputs. Consequences:

1. The notebook demonstrably does not do what it claims (and what its own doc page shows — `docs/example/notebooks/inference_evo_models.md:117-130` still documents the load call, so doc and notebook now disagree).
2. The phase's "evo family repairs, first real execution green" claim is not actually proven for evo-1: this execution never exercised `_handle_evo1_models`, the stripedhyena `CharLevelTokenizer` (and therefore the `np.fromstring` shim shipped in `transformers_compat.py` specifically for it), the `1.1_fix`/`main` revision logic, or the `_EVO1_SAFETENSORS_ONLY_PATTERNS` fetch path. The giants lane is dispatch-only, so this notebook is the only evo-1 execution artifact — and its evo-1 leg is dead code.
3. No fast content-contract test pins the evo-1 load cell (the megaDNA siblings got `TestMegadnaSiblingContentContracts` for exactly this regression class; the evo notebook got none, which is why this slipped through a green run).

**Fix:**
```python
# cell 13 (restore the load, per the doc page):
model_name = "togethercomputer/evo-1-8k-base"
model, tokenizer = load_model_and_tokenizer(
    model_name, task_config=configs['task'], source="huggingface"
)
```
Then add a fast JSON-level contract in `tests/examples/test_notebook_execution.py` mirroring the megaDNA pattern ("a cell containing both `evo-1` and `load_model_and_tokenizer` exists and precedes the second `DNAInference(` construction"), re-run the dispatch giants lane, and recommit the executed outputs to both mirrors.

## Warnings

### WR-01: ollama num_ctx state is contradictory across the service unit, the README, and the harness budgets; the unit still names the replaced model

**File:** `scripts/runner/ollama.service:20-23,44`; `scripts/runner/README.md:55-75`; `tests/examples/_execution.py:248-266`
**Issue:** The three artifacts disagree on whether the 8192 context cut is in effect:

- `ollama.service` sets `Environment="OLLAMA_CONTEXT_LENGTH=8192"` and asserts "this env governs" (line 22-23).
- `_execution.py` states "the num_ctx 8k cut is still DEFERRED (owner 2026-10-06 00:52 CST), so the brain serves the pair at its default context" — i.e. the cut is NOT in effect — and that this "un-cut reality" justified raising cell_timeout to 3600s.
- `README.md` says "the num_ctx cut remains DEFERRED ... so the server default env pin above stays exactly as committed" — self-contradictory, since the env pin IS the 8192 cut.

Additionally, the unit's governing-model note (lines 20-23) still names `qwen3.8:latest`, but the model was swapped to `qwen3.5:4b` on 2026-10-06. The unit's own instruction says to re-probe the Modelfile when the model is replaced ("re-probe with `ollama show ... --modelfile`"), because Modelfile `PARAMETER num_ctx` takes precedence over the env — and no re-probe result for qwen3.5:4b is recorded anywhere. If its Modelfile sets num_ctx, the pin is silently inert and the README's latency rationale is false; if it does not, the harness's "un-cut 256k-class latency tail" budget rationale is false. Either way one of the two budget narratives rests on an unverified fact.
**Fix:** Run `ollama show qwen3.5:4b --modelfile`, record the result in the README; reconcile all three files to a single true statement of the effective context window; update the unit comment to name qwen3.5:4b (or make the note model-agnostic).

### WR-02: megaDNA checkpoint-file selection conditions are inverted — non-default family members load the wrong .pt

**File:** `dnallm/models/special/megadna.py:120-127`
**Issue:** The chain reads `if m in "megaDNA_updated": ... elif m in "megaDNA_variants": ... elif m in "megaDNA_finetuned":` — testing whether the loop variable `m` is a **substring of the literal**, not whether the matched model belongs to that variant. For the family members `megaDNA_phage_78M`, `megaDNA_phage_277M`, and `megaDNA_phage_ecoli_finetuned` every branch is False (each is longer than, and not a substring of, the literals) and the `else` joins `megaDNA_phage_145M.pt` onto the downloaded snapshot path. Depending on repo layout this is either a `FileNotFoundError` at `torch.load` or — worse — silent loading of the 145M checkpoint for a 78M/277M/ecoli model. The three shipped names (`megaDNA_updated`, `megaDNA_variants`, `megaDNA_finetuned`) hit their intended branches only by coincidence of the reversed containment.
**Fix:**
```python
if "megaDNA_updated" in m:
    full_model_name = "megaDNA_phage_145M.pt"
elif "megaDNA_variants" in m:
    full_model_name = "megaDNA_phage_78M.pt"
elif "megaDNA_finetuned" in m or "ecoli" in m:
    full_model_name = "megaDNA_phage_ecoli_finetuned.pt"
else:
    full_model_name = "megaDNA_phage_145M.pt"
```
(or an explicit `{member: checkpoint_file}` dict), verified against each repo's actual file listing.

### WR-03: `_handle_megadna_models` mutates the module-level `megadna_models` list via `extra`

**File:** `dnallm/models/special/megadna.py:25-26`
**Issue:** `if extra: megadna_models.append(extra)` appends to a module-level list on every call that passes `extra`. No production caller currently passes it (`model.py:815` calls with three arguments), but the sibling handlers (`_handle_enformer_models` / `_handle_space_models` / `_handle_borzoi_models`) DO receive `extra=model_name` from `model.py` — wiring megaDNA the same way (the obvious future edit) would grow the list unboundedly across calls and permanently alter name matching for the whole process lifetime.
**Fix:** `models = megadna_models + ([extra] if extra else [])` and iterate `models`.

### WR-04: the guarded dispatch chain can discard a handler's resolved half, contradicting its own documented invariant

**File:** `dnallm/models/model.py:891-907`
**Issue:** The inline contract says "Each stage runs only when the previous stage left model or tokenizer None, so a handler's result is never overwritten by a later stage." The code does not guarantee that: each stage assigns the **whole tuple** (`model, tokenizer = _handle_dnabert2_models(...)`). If an earlier stage returned a partial result (model set, tokenizer None), the guard `model is None or tokenizer is None` enters the next stage, whose return — even `(None, tokenizer)` — overwrites BOTH variables, silently discarding the previously resolved model and falling through to the generic loader. Current handlers return full tuples or None, so the flaw is latent, but the guard's stated semantics and its implementation diverge.
**Fix:** merge per half instead of reassigning the pair:
```python
if model is None or tokenizer is None:
    m2, t2 = _handle_dnabert2_models(downloaded_model_path, load_args)
    model = m2 if m2 is not None else model
    tokenizer = t2 if t2 is not None else tokenizer
```
(and likewise around `_load_model_by_task_type`, or assert handlers return `(None, None)` or full tuples).

### WR-05: `weights_only=False` full unpickling of a remotely fetched checkpoint

**File:** `dnallm/models/special/megadna.py:129`
**Issue:** `torch.load(downloaded_model_path, weights_only=False)` fully unpickles an artifact downloaded at load time from a remote hub repo (via HF or the mirror) — arbitrary code execution if the repo account or the mirror serving the bytes is compromised. The format is inherent to the upstream megaDNA distribution (the checkpoint is a pickled model object, not a state dict), but the code-level trust base is unpinned: `models.lock:32` records `lingxusb/megaDNA_updated@ed298be...` as provenance only, and nothing at the call site enforces that revision. Contrast the source-code pin (`MEGADNA_CLONE_COMMIT`) which is exact-pinned in three places.
**Fix:** pass the pinned revision from models.lock through to the snapshot fetch (`revision="ed298be..."`) and document the trust decision next to the `torch.load`; medium-term, load a `state_dict` with `weights_only=True` and reconstruct the module from the pinned local clone.

### WR-06: `_determine_classifier` crashes with UnboundLocalError on an unrecognized head name

**File:** `dnallm/models/model.py:132-147`
**Issue:** If `head_config["head"]` carries a `custom_head` of None and its name does not end with `mlp`/`cnn`/`lstm`/`unet` (e.g. a new head name like `"attention"` reaching the generic or lucaone branch), `classifier` is never bound and the function raises `UnboundLocalError: local variable 'classifier' referenced before assignment` — instead of the project-convention descriptive `ValueError` (CLAUDE.md error-handling rule; 127 of ~170 raises in dnallm/ are ValueError).
**Fix:** add a terminal branch:
```python
else:
    raise ValueError(
        f"Unknown head type {self.config.head_config.get('head')!r}: "
        "expected a name ending in mlp/cnn/lstm/unet or a custom_head class."
    )
```

## Info

### IN-01: evo2 local-path resolution crashes with bare IndexError

**File:** `dnallm/models/special/evo.py:231`
**Issue:** `model_path = glob(model_name + "/*.pt")[0] if os.path.isdir(model_name) else model_name` raises `IndexError` when a local directory contains no `.pt` files — no diagnosis, no matchable error.
**Fix:** guard the empty glob and raise `ValueError(f"No .pt checkpoint found in {model_name}")`.

### IN-02: unclosed file handle in the evo-1 checkpoint loader

**File:** `dnallm/models/special/evo.py:347`
**Issue:** `dotdict(yaml.safe_load(open(config_path)))` leaks the file handle (runs once per matching evo-1 load).
**Fix:** `with open(config_path) as f: global_config = dotdict(yaml.safe_load(f))`.

### IN-03: evo-1 revision gate is case-sensitive while the rest of the handler lowercases

**File:** `dnallm/models/special/evo.py:384`
**Issue:** `revision = "1.1_fix" if "." in model_name and source == "huggingface" else "main"` — a caller passing `source="HuggingFace"` silently gets revision `main`, while `_handle_evo2_models` and `_get_model_path_and_imports` normalize with `.lower()`.
**Fix:** compare against `source.lower()`.

### IN-04: docs prerequisites reference extras that do not exist in pyproject

**File:** `docs/example/notebooks/finetune_custom_head.md:17`, `docs/example/notebooks/finetune_generation.md:17`, `docs/example/notebooks/data_prepare_finetune.md:17` (`.[base,finetune,cuda124]`); `docs/example/notebooks/inference_evo_models.md:17`, `docs/example/notebooks/inference_megaDNA.md:17` (`.[base,inference,cuda124]`)
**Issue:** pyproject.toml defines neither a `finetune` nor an `inference` extra; `uv pip install -e '.[base,finetune]'` warns "does not have an extra named `finetune`" (verified live) and older pips silently ignore it. Lines predate the phase (commit 009ff10) but live in pages the phase rewrote.
**Fix:** change to `.[base,cuda124]` (base already pulls dev/test/notebook/mcp).

### IN-05: deploy job pins `actions/cache@v3` while every other job is on v4

**File:** `.github/workflows/ci.yml:1121`
**Issue:** The docs-deploy job's mkdocs cache still uses the deprecated `actions/cache@v3`; all other cache steps in this workflow use v4. The v3 major is retired/deprecated upstream and will break the deploy lane when removed.
**Fix:** bump to `actions/cache@v4`.

---

_Reviewed: 2026-10-06T17:22:06Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
