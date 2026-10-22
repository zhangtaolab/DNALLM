---
phase: 23-pre-flight-core-pure-validation-suitability-config-api
reviewed: 2026-10-20T17:53:59Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - dnallm/configuration/configs.py
  - dnallm/preflight/__init__.py
  - dnallm/preflight/availability.py
  - dnallm/preflight/compatibility.py
  - dnallm/preflight/config_gen.py
  - dnallm/preflight/report.py
  - dnallm/preflight/validation.py
  - tests/preflight/test_availability.py
  - tests/preflight/test_compatibility.py
  - tests/preflight/test_config_gen.py
  - tests/preflight/test_integration.py
  - tests/preflight/test_reader_parity.py
  - tests/preflight/test_structure.py
  - tests/preflight/test_validation.py
findings:
  critical: 2
  warning: 5
  info: 8
  total: 15
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-10-20T17:53:59Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Reviewed the new `dnallm/preflight/` package (validation, compatibility, availability, config generation, report types) plus the one sanctioned `validate_sections` factor-out in `dnallm/configuration/configs.py`. The `configs.py` refactor is verbatim (diff-verified against 397eb18; body moved, `load_config` delegates, behavior pinned by equivalence tests). Overall design quality is high: frozen Pydantic reports, safe YAML loaders, pickle refusal, per-model error isolation, and a genuinely thorough behavior test suite.

Two blockers survive scrutiny, both empirically reproduced in the repo venv:

1. The package's headline **import-light invariant is false**: `import dnallm.preflight` transitively imports torch, transformers, AND datasets (verified via `sys.modules`). The AST structure test cannot see this.
2. `validate_data` **crashes with a raw pandas traceback** on any supported-format file carrying a non-scalar cell (e.g., a JSON column of lists) — a direct violation of the collect-all "report, never traceback" contract.

Five warnings cover the zero-traceback gaps in `edit_config`/`generate_config` output handling, probe-memo poisoning across the injectable prober seam, a test-reliability defect (undeclared `httpx` + real network I/O at pytest collection time), and a weak `availability: object` annotation. Eight info items cover convention drift and edge-case robustness.

## Critical Issues

### CR-01: Import-light invariant is empirically false — `import dnallm.preflight` pulls in torch, transformers, and datasets

**File:** `dnallm/preflight/__init__.py:40-65` (via `dnallm/preflight/report.py:27`, `dnallm/preflight/availability.py:51-52`, `dnallm/preflight/compatibility.py:34-44`, `dnallm/preflight/config_gen.py:52-53`)
**Issue:** The package docstring (`__init__.py:22-25`), `report.py:15-17` ("this module imports nothing beyond the stdlib, pydantic, and the package logger — never torch/transformers/datasets"), and `config_gen.py:19-21` ("any shell can generate a config without the training stack") all claim import-lightness. Empirically it does not hold:

```python
import sys; import dnallm.preflight
'torch' in sys.modules        # True
'transformers' in sys.modules # True
'datasets' in sys.modules     # True
```

Two transitive chains break it:
- `from ..utils import get_logger` executes `dnallm/utils/__init__.py`, which eagerly imports `transformers_compat` (top-level `import torch`, `dnallm/utils/transformers_compat.py:19`) and `support` (`from torch.cuda import get_device_capability`).
- `from ..models.registry import ...` executes `dnallm/models/__init__.py`, which eagerly imports `.model` / `.modeling_auto` / `.tokenizer` (transformers, datasets).

`tests/preflight/test_structure.py:68-79` only AST-scans the files under `dnallm/preflight/` for direct heavy imports, so it green-lights a broken invariant. Consequence: every planned shell (TUI, MCP tool, notebook) pays the full training-stack import cost — exactly what the invariant exists to prevent — and three module docstrings plus a guard test document something false.
**Fix:** Make the parent package inits lazy so importing `dnallm.models.registry` / `dnallm.utils`'s logger does not execute the heavy modules: add PEP 562 `__getattr__` to `dnallm/models/__init__.py` (defer `.model`/`.modeling_auto`/`.tokenizer`/`.losses` behind attribute access) and defer the compat-shim imports in `dnallm/utils/__init__.py`. Then strengthen `test_structure.py` with a runtime check in a fresh subprocess:

```python
def test_preflight_import_is_actually_light() -> None:
    code = "import sys, dnallm.preflight; print(any(m in sys.modules for m in ('torch','transformers','datasets')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
```

### CR-02: `validate_data` crashes with a raw pandas ValueError on non-scalar cells (nested-array columns)

**File:** `dnallm/preflight/validation.py:575` (and `dnallm/preflight/validation.py:493`)
**Issue:** `_dna_column_score` filters cells with `not bool(pd.isna(v))`. For a list/dict cell (routine in JSON, legal in parquet), `pd.isna(v)` returns an ndarray and `bool()` raises. `_autodetect_sequence_column` (validation.py:557) scores **every** column, so one nested column anywhere crashes the entire validation with `ValueError: The truth value of an array with more than one element is ambiguous` — an unmatchable traceback on a supported input format, violating the module's collect-all "findings annotate, reports are return values" contract and not covered by the docstring's `Raises` list. Reproduced:

```python
# JSON: [{"sequence": "ACGT...", "label": 1}, {"sequence": "ACGT...", "label": 0, "tags": ["a","b"]}]
validate_data(path)  # ValueError: truth value of an array ... ambiguous
```

The same pattern in `_cell_str` (line 493) crashes when an explicit `seq_col`/`label_col` names a list-valued column.
**Fix:** Handle non-scalar cells explicitly in both helpers:

```python
def _cell_is_null(value) -> bool:
    if value is None or isinstance(value, (str, bytes)):
        return value is None or value == ""
    try:
        return bool(pd.isna(value))
    except ValueError:  # list/dict cells: pd.isna returns an array
        return False
```

and use it in `_dna_column_score` (line 575) and `_cell_str` (lines 491-495); for scoring, stringify non-scalar cells (they will score below the DNA threshold) instead of crashing.

## Warnings

### WR-01: `edit_config` on a nonexistent path raises raw `FileNotFoundError`, breaking the zero-traceback contract

**File:** `dnallm/preflight/config_gen.py:310` (propagated through `edit_config`, `config_gen.py:395-400`)
**Issue:** The module contract (config_gen.py:3-8) says user-input error paths return a `GeneratedConfig` with `SectionError` records, never a raised exception. `edit_config("/nonexistent/config.yaml", {})` raises `FileNotFoundError` (reproduced). This is also inconsistent with `validate_data`, which raises a matchable `ValueError("Data file not found: ...")` for the same user mistake, and with the project convention of `ValueError` with matchable messages.
**Fix:** In `edit_config`, catch `OSError` from `_load_existing` and return it as a record (or re-raise as `ValueError`): `except FileNotFoundError as e: return GeneratedConfig(config={}, errors=[SectionError(section="config", field="existing", message=f"FILE_NOT_FOUND: {e}")])`.

### WR-02: `generate_config(..., output_path=...)` writes an invalid config to disk even when generation produced errors

**File:** `dnallm/preflight/config_gen.py:284-285`
**Issue:** `output_path` is dumped unconditionally after `_validate_collect_all`, without consulting `errors`. Reproduced: `generate_config("zhangtaolab/plant-dnabert-BPE", output_path=...)` (base model, no `task_type`) writes a YAML file that is missing the task section while returning `TASK_TYPE_REQUIRED` — the on-disk artifact is silently invalid with no marker, though the API is titled "validated config generation" and the caller may only check file existence. (Same pattern in `edit_config` at lines 405-406, where patch errors exist.)
**Fix:** Only dump when the run is clean, or make the condition explicit:

```python
if output_path is not None and not (errors + validation_errors):
    _dump_yaml(config_dict, output_path)
```

(or document + emit a top-of-file `# generated with errors: ...` comment when errors exist).

### WR-03: Probe memo is poisoned by injected-prober results — later default probes return the fake report

**File:** `dnallm/preflight/availability.py:133-148`
**Issue:** `probe_availability(model, prober=fake)` memoizes the injected result under both `model` and `repo_id` (lines 146-148). A subsequent `probe_availability(model)` **without** a prober returns the memoized fake report instead of running the real default probe. Reproduced: a fake `REMOTE_CONFIRMED, size_bytes=999` was returned by a later default-probe call. Two different injected probers across calls likewise return the first one's result forever. This makes the "injectable seam" (wired in by the compatibility layer and any shell-provided prober) a source of stale/incorrect availability data for the rest of the process. `clear_probe_memo()` exists but nothing in the prober contract tells callers their injection is cached globally.
**Fix:** Do not memoize when a custom `prober` is supplied (memoize only default-probe results), or key the memo on prober identity:

```python
if prober is None:
    _MEMO[repo_id] = report
    if model != repo_id:
        _MEMO[model] = report
```

Document the caching behavior in the `prober` arg description.

### WR-04: `test_availability.py` performs real network I/O at pytest collection time and depends on undeclared `httpx`

**File:** `tests/preflight/test_availability.py:774-791` (decorator at 787-791, `import httpx` at 776)
**Issue:** `@pytest.mark.skipif(not _any_model_hub_reachable(), ...)` is evaluated at module import, so **every** pytest run — including `-m "not slow"` — fires real HTTP GETs at modelscope.cn and huggingface.co (5 s timeout each; ~10 s stall on a blackholed network during collection). It also contradicts the suite's own "network-free" claim (availability.py:31-32). Additionally, `httpx` is not declared in any extra (`test` extra is pytest-only); it resolves only transitively via `mcp` in `.[full]`, so a minimal `-e ".[test]"` install fails the entire module at collection (~40 tests error) with `ModuleNotFoundError`.
**Fix:** Move the reachability check inside the test body and use a stdlib check (or `urllib.request` with a short timeout), skipping at runtime:

```python
@pytest.mark.slow
def test_slow_real_network_dual_homed_lab_model() -> None:
    if not _any_model_hub_reachable():
        pytest.skip("network-unavailable: no model hub reachable")
```

(or gate the check behind an env var), and drop the `httpx` import in favor of `urllib.request`.

### WR-05: `ListingEntry.availability` typed as bare `object | None` instead of `AvailabilityReport | None`

**File:** `dnallm/preflight/report.py:319`
**Issue:** The field's docstring says it holds "an :class:`AvailabilityReport`" and every producer stores exactly that, but the annotation `object | None` gives downstream shells (Phase 25 TUI / 26 MCP) zero type checking and lets pydantic validate anything (a misbehaving prober returning a dict or `None`-adjacent junk is accepted silently). `AvailabilityReport` is defined in the same module, so there is no circular-import reason for the weakening.
**Fix:** `availability: AvailabilityReport | None = None`. If a future non-pydantic payload is genuinely planned, introduce a union type rather than `object`.

## Info

### IN-01: `label_dtype` value `"int"` is documented but unreachable — integer labels are reported as `"float"`

**File:** `dnallm/preflight/validation.py:603-616`; `dnallm/preflight/report.py:240-241`
**Issue:** `DataReport.label_dtype` documents `("float"/"int"/"str"/"multilabel")`, but `_guess_label_metadata` only ever returns `"multilabel"`, `"float"` (integers parse via `float()`), or `"str"` — verified `_guess_label_metadata(['1','2','3']) == ('float', None)`.
**Fix:** Either detect all-integer texts (`all(re.fullmatch(r"[+-]?\d+", t) for t in texts)`) and return `"int"`, or fix the docstring to the reachable vocabulary.

### IN-02: Default label column name diverges from the parity reference (`"label"` vs `"labels"`)

**File:** `dnallm/preflight/validation.py:57`
**Issue:** `_LABEL_NAME = "label"` while `DNADataset.load_local_data` defaults to `label_col="labels"` (`dnallm/datahandling/data.py:145`). The parity tests always pass explicit names, so the divergence is unpinned; in narrow default-parameter cases (e.g., a `.txt` first line containing `sequence` + `label` but not `labels` — the csv-reparse branch at validation.py:465) the two readers take different branches, despite the module docstring claiming replication of the reference's observable read behavior.
**Fix:** Align the default (use `"labels"`) or document the intentional divergence in `_LABEL_NAME`'s comment.

### IN-03: Reader opens files without `encoding="utf-8"` (locale-dependent decoding)

**File:** `dnallm/preflight/validation.py:401` (`_check_header_and_type`), `:436` (`_load_fasta`), `:463` (`_load_txt`)
**Issue:** All three use bare `open(path)`, so decoding follows the process locale; on a non-UTF-8 default encoding, UTF-8 DNA files with plain ASCII headers still parse but any non-ASCII byte path/label misdecodes or raises. This mirrors `data.py` (parity-consistent) and the rest of the new module correctly uses `encoding="utf-8"` (`config_gen.py:196,310`), so this is an inconsistency within the phase's own code.
**Fix:** Add `encoding="utf-8"` (optionally `errors="replace"` for the header peek) to the three `open()` calls.

### IN-04: One 115-character line exceeds the project's ruff line-length 100

**File:** `dnallm/preflight/config_gen.py:311`
**Issue:** The `yaml.load(...)` line with its trailing `# ruff: ignore[...]` pragma is 115 chars. It escapes enforcement because E501 is not in the selected rule set and the trailing comment blocks the formatter. (The pragma itself is honored under the project's `preview = true` config — verified — but the length still breaches the stated convention.)
**Fix:** Wrap the loader selection into a local (`loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)`) before the `yaml.load` call.

### IN-05: `validate_data(min_length=0)` raises an undocumented raw pydantic `ValidationError`

**File:** `dnallm/preflight/validation.py:78-109` (docstring `Raises`); `dnallm/preflight/report.py:252` (`min_length: int = Field(gt=0)`)
**Issue:** A non-positive `min_length` surfaces only when the `DataReport` is constructed, as a pydantic field error not mentioned in the docstring's `Raises` list (which covers only source-type issues).
**Fix:** Validate at entry: `if min_length <= 0: raise ValueError(f"validate_data min_length must be positive, got {min_length}.")`.

### IN-06: Local-cache detection runs `os.path.isdir` on the raw input alias

**File:** `dnallm/preflight/availability.py:254`
**Issue:** `_probe_cached_local` checks the **raw** `model` string (which may be a registry entry *name* when the caller used an alias) against the filesystem, relative to CWD. A registry name that collides with a relative directory (e.g., a `plant-dnabert` folder in the working directory) yields a spurious `cached/local` verdict; conversely the resolved `repo_id` is not path-checked when it differs from the input.
**Fix:** Check `os.path.isdir` on the raw input only when `try_resolve` missed (i.e., `entry is None`, a genuine path), or check both raw input and repo-id explicitly and prefer the unambiguous one.

### IN-07: Shared bounded-wait executor can report spurious PROBE_TIMEOUT while saturated

**File:** `dnallm/preflight/availability.py:85-88, 340-347`
**Issue:** `_EXECUTOR` has `max_workers=2` and timed-out calls are abandoned, not cancelled — they keep occupying workers. Two abandoned slow hub calls saturate the pool, so a *third*, healthy probe's `future.result(timeout=...)` can expire while merely queued, degrading a probeable model to `unknown/PROBE_TIMEOUT` (contract-safe, but incorrect availability info). Non-daemon worker threads can also delay interpreter shutdown on hung sockets.
**Fix:** Size the pool from the expected concurrency (`max_workers=4+`), or track in-flight abandoned futures and spin a temporary thread when the pool is saturated; at minimum note the saturation mode in the `_bounded_wait` docstring.

### IN-08: New `validate_sections` docstring uses the legacy typed-param style

**File:** `dnallm/configuration/configs.py:708-710`
**Issue:** The project convention says new code uses untyped-param style (`name: description`; types live in the signature) and must not copy the older `name (dict): ...` style. The newly written docstring uses `config_dict (dict): The raw config mapping.`
**Fix:** Change to `config_dict: The raw config mapping.`

---

_Reviewed: 2026-10-20T17:53:59Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
