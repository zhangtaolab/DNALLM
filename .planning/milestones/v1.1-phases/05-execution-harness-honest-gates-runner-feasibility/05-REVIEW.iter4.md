---
phase: 05-execution-harness-honest-gates-runner-feasibility
reviewed: 2026-10-02T08:50:38Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - .gitignore
  - dnallm/utils/transformers_compat.py
  - example/notebooks/finetune_NER_task/generate_bpe_dataset.py
  - tests/examples/_execution.py
  - tests/examples/test_marimo_execution.py
  - tests/examples/test_notebook_execution.py
  - tests/examples/test_script_execution.py
  - tests/models/test_model_remote_code.py
  - tests/utils/test_transformers_compat.py
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 5 (incremental 05-04/05-05/05-06): Code Review Report

**Reviewed:** 2026-10-02T08:50:38Z
**Depth:** standard
**Files Reviewed:** 9 (diff base `debf813b`)
**Status:** issues_found

## Summary

Incremental review of Phase 5's post-closure gap-closure work: the vendored
pruning-helper shim in `dnallm/utils/transformers_compat.py` (+109 lines of
tests), the marimo/script execution lanes, the generalized notebook-execution
harness (`tests/examples/_execution.py` grew from 1 to 21 notebook specs plus
marimo/script runners), and the one-line `rice_annotation.bed` writer restore
in `generate_bpe_dataset.py`.

Verified against ground truth, not just the diff:

- The vendored `_find_pruneable_heads_and_indices` / `_prune_linear_layer`
  are behavior-identical to upstream transformers v4.49.0 (checked line by
  line against the reference implementation; the `heads` rebinding rename is
  semantics-preserving). Live-probed on transformers 5.17.0: the patch
  attaches both helpers to `transformers.modeling_utils` on dnallm import,
  and the known checkpoint's import site is fixed.
- The restored `rice_annotation.bed` block is verbatim from
  `data_generation_and_inference.ipynb` cell 10 (extracted and compared);
  `gene_info[gene]` is populated for every gene before use, so no KeyError
  path exists.
- All 21 `NOTEBOOK_EXEC_SPECS` keys and all 3 `MARIMO_EXEC_SPECS` keys exist
  on disk; all 8 ACTIVE and 7 GATED notebook ids resolve to spec entries;
  `tests/expected_skips.yaml` registers all three typed-skip prefixes used;
  `MCP_ENDPOINT` port 8000 matches the MCP server config; `ruff check .`
  (CI invocation) passes; the new tests collect cleanly on the fast leg
  (26/31, 5 slow deselected) including the Windows-style import path
  (`base` extra carries `test`+`notebook`, so `nbclient` resolves).
- Coverage rule (dnallm/ change ships with pytest): satisfied —
  `TestRemoteCodePruningHelpers` is fast-lane and covers the helper
  arithmetic, live attachment, idempotence, and 4.x/5.x version awareness;
  `tests/models/test_model_remote_code.py` exercises the real load route.
  Note the real-model smoke currently terminates in the documented D-07
  typed skip on transformers 5.17 (remote code's `config.is_decoder` read),
  so the forward-pass assertions are environment-gated; the unit tests carry
  the shim itself.

No Critical findings. The four Warnings are: (1) the shim patches only
`transformers.modeling_utils` while live-verified transformers 5.17 still
ships a `pytorch_utils` that partially lacks the helpers — the other common
4.x remote-code import site stays broken; (2) the gated
`lora_finetune.ipynb` entry gets an outer pytest timeout equal to its
per-cell timeout, violating the harness's own strictly-below invariant;
(3) the spec dicts' `test_timeout`/`flavor` fields are dead data that
contradict their documented "the test layer must apply" contract — the root
cause that let (2) slip through; (4) a permanent HTTP 4xx on the rice input
URLs converts to an ever-green `network-unavailable` skip.

## Warnings

### WR-01: Pruning shim patches only `modeling_utils`; `transformers.pytorch_utils` still lacks `find_pruneable_heads_and_indices` on 5.x

**File:** `dnallm/utils/transformers_compat.py:293-333`
**Issue:** `_patch_remote_code_pruning_helpers` attaches the vendored helpers
only to `transformers.modeling_utils`. Live-verified on the installed
transformers 5.17.0: `transformers.pytorch_utils` still EXISTS and still
exports `prune_linear_layer`, but `find_pruneable_heads_and_indices` is absent
from it (and remains absent after the dnallm patch — re-probed). 4.x-era
`trust_remote_code` checkpoints canonically copy HF's own 4.x model files,
which import `from transformers.pytorch_utils import
find_pruneable_heads_and_indices, prune_linear_layer`. Any such checkpoint
still crashes with ImportError on transformers 5.x even with dnallm imported,
so the module docstring's general claim ("4.x-era trust_remote_code
checkpoints ... still import both names") is only honored for the
`modeling_utils` import site used by the one known checkpoint.
**Fix:** Extend the patch to also attach the missing name(s) to
`transformers.pytorch_utils` when that module exists, gated per name (on 5.17
only `find_pruneable_heads_and_indices` is missing there — do not overwrite
upstream's own `prune_linear_layer`):

```python
try:
    import transformers.pytorch_utils as _pu
except Exception:
    _pu = None
if _pu is not None and not hasattr(_pu, "find_pruneable_heads_and_indices"):
    setattr(_pu, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)
    if not hasattr(_pu, "prune_linear_layer"):
        setattr(_pu, "prune_linear_layer", _prune_linear_layer)
```

### WR-02: Gated `lora_finetune.ipynb` runs with outer timeout == cell timeout, violating the harness's strictly-below invariant

**File:** `tests/examples/test_notebook_execution.py:311-322, 325-326` (spec at `tests/examples/_execution.py:156-160`)
**Issue:** The spec for `notebooks/lora_finetune_inference/lora_finetune.ipynb`
declares `cell_timeout: 3600, test_timeout: 7200`, but the gated parametrize
applies the 7200 mark override only to
`notebooks/finetune_custom_head/finetune.ipynb`; `lora_finetune.ipynb` falls
through to the class-level `@pytest.mark.timeout(3600)`. Result: a cell may
legitimately run up to 3600 s under nbclient while pytest-timeout kills the
whole test at 3600 s. The outer kill preempts nbclient's clean
`CellTimeoutError` handling (and the harness's partial-failure artifact
capture), converting a budget-managed hang into a hard, artifact-less test
abort. `run_notebook`'s own contract (`_execution.py:280-281`) requires
cell_timeout to stay *strictly below* the per-test mark — this is the only
entry across all three lanes where the invariant is broken (verified against
every ACTIVE/GATED/marimo budget).
**Fix:** Add the override for this entry as well (or generate the marks from
the spec, see WR-03):

```python
_TIMEOUT_7200 = {"notebooks/finetune_custom_head/finetune.ipynb",
                 "notebooks/lora_finetune_inference/lora_finetune.ipynb"}
[
    pytest.param(nb_id, marks=pytest.mark.timeout(7200)) if nb_id in _TIMEOUT_7200 else nb_id
    for nb_id, _gate in GATED_NOTEBOOKS
]
```

### WR-03: `test_timeout` (and marimo `flavor`) spec fields are dead data contradicting their documented contract

**File:** `tests/examples/_execution.py:63-75, 75-181, 183-211`
**Issue:** The `NOTEBOOK_EXEC_SPECS` comment says values carry "the per-test
timeout mark the test layer must apply", and `MARIMO_EXEC_SPECS` likewise
documents `test_timeout`/`flavor` per app. Grep-verified: no consumer reads
`spec["test_timeout"]` or `spec["flavor"]` anywhere in `tests/` or `scripts/`.
Every lane applies static class marks instead (7200 / 3600; marimo gets 7200
where its spec says 1500), and `run_marimo_app` hardcodes `"export html"`
while the spec carries a `flavor` field. Consequences: the documented budgets
mislead (inference spec says 1800 s, actual bound is 7200 s), and budget
mistakes of the WR-02 kind are invisible because the enforcement data exists
but is never consulted.
**Fix:** Either enforce the fields — parametrize with
`pytest.param(..., marks=pytest.mark.timeout(spec["test_timeout"]))`
generated from the spec dicts and pass `spec["flavor"]` into
`run_marimo_app` — or amend the spec-dict docstrings to state the fields are
advisory documentation only and delete `flavor` until a second flavor exists.

### WR-04: Permanent HTTP 4xx on the rice input URLs converts to an ever-green `network-unavailable` skip

**File:** `tests/examples/test_script_execution.py:70-78`
**Issue:** `except (urllib.error.URLError, TimeoutError)` also catches
`urllib.error.HTTPError` (its subclass). If rice.uga.edu ever reorganizes the
download URLs (permanent 404/410), every run of the script lane records a
`network-unavailable:` typed skip — a prefix `scripts/audit_skips.py`
unconditionally allows — so the lane goes green-forever while the script is
actually unrunnable and the input contract is broken. A 4xx is not network
unavailability; the harness's own philosophy ("non-qualifying execution
failures always re-raise") argues for loud failure on permanent client
errors.
**Fix:** Skip only on transport failures and 5xx; fail loudly on 4xx:

```python
except urllib.error.HTTPError as exc:
    if exc.code >= 500:
        pytest.skip(f"network-unavailable: fetch {url} (HTTP {exc.code})")
    raise
except (urllib.error.URLError, TimeoutError) as exc:
    pytest.skip(f"network-unavailable: fetch {url} ({type(exc).__name__}: {exc})")
```

## Info

### IN-01: Patch gate checks only one of the two helper names

**File:** `dnallm/utils/transformers_compat.py:313-315`
**Issue:** The 4.x no-op gate is
`hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices")`
alone. If a future transformers version kept that name but dropped
`prune_linear_layer` (or the reverse — note upstream 5.17 already removed
them asymmetrically from `pytorch_utils`), the patch silently no-ops and
remote code importing the missing name still crashes. Both names were
removed together from `modeling_utils` so this is hypothetical today.
**Fix:** Gate on both names being present:
`if hasattr(mu, "find_pruneable_heads_and_indices") and hasattr(mu, "prune_linear_layer"): return`.

### IN-02: Direct `pytest.skip("network-unavailable: ...")` bypasses the `network_unavailable_skip` helper

**File:** `tests/examples/test_script_execution.py:74-78`
**Issue:** The harness defines `network_unavailable_skip` precisely to emit
the registered stable prefix, and this file already imports
`environment_unavailable_skip` from the same module — but the rice-download
path hand-builds the prefix inline. If the prefix ever changes in
`_execution.py`, this site silently diverges (caught only later by
`audit_skips.py` failing the job).
**Fix:** Import and call
`network_unavailable_skip(f"fetch {url} for generate_bpe_dataset.py", evidence=f"{type(exc).__name__}: {exc}")`.

### IN-03: Hardcoded dev-box `sys.path` in the script the new lane executes

**File:** `example/notebooks/finetune_NER_task/generate_bpe_dataset.py:13-14`
**Issue:** `sys.path.insert(0, "/home/forrest/Github/DNALLM")` (pre-existing,
not introduced by this diff — but the new test lane now executes this script
in CI-shaped environments where that path does not exist; it survives only
because dnallm is pip-installed in the venv). On a box without the editable
install the script crashes at import.
**Fix:** Drop the `sys.path` hack (the installed package suffices), or derive
the root from `Path(__file__).resolve().parents[3]`.

### IN-04: No isolated test for the pruning patch's attach branch (unlike the sibling patches)

**File:** `tests/utils/test_transformers_compat.py:418-496`
**Issue:** `_patch_get_parameter_or_buffer` /
`_patch_initialize_weights_for_quantized_missing` have
`test_patch_skips_when_target_method_absent` exercising their early-return
arms against a synthetic class. `_patch_remote_code_pruning_helpers` has no
synthetic-module equivalent: its attach path is only proven via the
already-applied live-module state (import-time), and its 4.x `hasattr` no-op
branch only implicitly through the identity test. The vendored functions
themselves are well covered (arithmetic, shapes, idempotence, identity), so
this is a minor gap against the "dnallm changes ship with tests" rule, not a
hole.
**Fix:** Add a test that monkeypatches
`sys.modules["transformers.modeling_utils"]` with a `types.ModuleType`
lacking both names and the sentinel, calls `_patch_remote_code_pruning_helpers()`,
and asserts both names were attached (restore via `monkeypatch` undo — the
live class patches are never touched).

### IN-05: Timeout path skips the documented "always/failed" run artifacts

**File:** `tests/examples/_execution.py:324-414, 417-482`
**Issue:** `run_example_script`'s docstring says the run log is "ALWAYS
written" and `run_marimo_app`'s says the error artifact is written "on
failure" — but on `subprocess.TimeoutExpired` the exception propagates before
either artifact write, so the census evidence for a timed-out run is missing
exactly when the run was most anomalous (`TimeoutExpired` does carry
stdout/stderr attributes that could be persisted).
**Fix:** Wrap the `subprocess.run` call in try/except TimeoutExpired, write
the log/error artifact from `exc.stdout`/`exc.stderr`, then re-raise.

---

_Reviewed: 2026-10-02T08:50:38Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
