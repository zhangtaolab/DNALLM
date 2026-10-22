# Phase 23: Code Review Fix Report

**Fixed at:** 2026-10-20T18:22:34Z
**Source review:** `.planning/phases/23-pre-flight-core-pure-validation-suitability-config-api/23-REVIEW.md`
**Iteration:** 1
**Branch:** v1.4 (main checkout; `workflow.use_worktrees=false` — no worktree)

**Summary:**
- Findings in scope: 10 (CR-01 docstring-only, CR-02, WR-01..WR-05, IN-03, IN-04, IN-08)
- Fixed: 10
- Skipped: 0
- Deferred by disposition (no code changes, per fix instructions): IN-01, IN-02, IN-05, IN-06, IN-07, and CR-01's structural root (Phase 25 owner call)

Every behavioral fix landed test-first (new/rewritten tests confirmed red against the pre-fix code, then green with the fix), one atomic commit per finding, message format `fix(23): <id> <description>`.

## Fixed Issues

### CR-02: `validate_data` crashes with a raw pandas ValueError on non-scalar cells (nested-array columns)

**Files modified:** `dnallm/preflight/validation.py`, `tests/preflight/test_validation.py`
**Commit:** 8037647
**What changed:** Added the scalar-safe `_cell_is_null()` helper (None → null; `bool(pd.isna(v))` under `try/except (TypeError, ValueError)` so list/dict/array cells return False instead of raising) and used it in both crash sites: `_cell_str()` (validation.py:491) and `_dna_column_score()`'s cell filter (validation.py:575). Non-scalar cells now stringify and score below the DNA threshold — `validate_data` returns its collect-all report on any supported format, never a traceback.
**Test added:** `test_validate_data_json_list_valued_column_returns_report` (JSON with a `tags` list column — autodetect path) and `test_validate_data_explicit_list_valued_label_column_stringified` (explicit `label_col` naming a list-valued column). Both reproduced the exact `ValueError: The truth value of an array ...` from the review before the fix.

### WR-01: `edit_config` on a nonexistent path raises raw `FileNotFoundError`

**Files modified:** `dnallm/preflight/config_gen.py`, `tests/preflight/test_config_gen.py`
**Commit:** 720acb6
**What changed:** `_load_existing()` wraps its read in `except FileNotFoundError → ValueError("Existing config not found: {path}.")` and `except OSError → ValueError("Existing config not readable: ...")`, consistent with `validate_data`'s matchable not-found contract. `yaml.YAMLError` propagation is untouched (safe-loader pin, T-23-08). Docstrings updated (`_load_existing` Raises, `edit_config` Raises, module invariant 1 notes the one sanctioned exception).
**Test added:** `test_nonexistent_existing_path_raises_matchable_value_error` (`pytest.raises(ValueError, match=r"Existing config not found")`) and `test_unreadable_existing_path_raises_matchable_value_error` (directory path). Both raised raw `FileNotFoundError`/`IsADirectoryError` before the fix.

### WR-02: `generate_config(..., output_path=...)` writes an invalid config even when generation produced errors

**Files modified:** `dnallm/preflight/config_gen.py`, `tests/preflight/test_config_gen.py`
**Commit:** 9c98994
**What changed:** Both dump sites (`generate_config` and the same pattern in `edit_config` flagged by the review) now gate on an error-free run: `if output_path is not None and not (errors + validation_errors): _dump_yaml(...)`. File existence now always implies a valid config; `output_path` arg docs updated. The single-write-site structure (T-23-09) is unchanged — this only refuses writes.
**Test added:** `test_generate_config_with_errors_writes_no_file` (base model, no `task_type` → `TASK_TYPE_REQUIRED`, `out_path.exists()` is False) and `test_edit_config_with_errors_writes_no_file` (`UNKNOWN_SECTION` patch → no file). Both wrote the invalid file before the fix.

### WR-03: Probe memo is poisoned by injected-prober results

**Files modified:** `dnallm/preflight/availability.py`, `tests/preflight/test_availability.py`
**Commit:** 41c503f
**What changed:** The memo now serves default probes only: both memo reads and the memo write are guarded by `prober is None`. An injected prober never reads from and never writes to the memo, so a fake report cannot be served to a later default probe (or to a different injected prober). The lock-guarded concurrency dedup contract (D-12) is preserved for the default flow it was designed for. Caching behavior documented in the `prober` arg description and the `_MEMO` comment, as the review requested.
**Test added:** `test_injected_prober_result_never_poisons_default_probe` (the review's pin: fake prober → default probe of the same model is NOT the fake report) and `test_two_injected_probers_each_get_their_own_report`; `test_memo_concurrent_probes_resolve_once` rewritten to the default seam (`_resolve_model_ref` + `_probe_remote` patched) because the old form deduped through the very memo-poisoning being fixed.

### WR-04: `test_availability.py` performs real network I/O at pytest collection time and depends on undeclared `httpx`

**Files modified:** `tests/preflight/test_availability.py`
**Commit:** f47fc7f
**What changed:** Removed the `httpx` import and the collection-time `@pytest.mark.skipif(not _any_model_hub_reachable(), ...)` evaluation. `_any_model_hub_reachable()` now uses stdlib `urllib.request`/`urllib.error` (any HTTP response — including 4xx/5xx — counts as reachable; repo-style `# ruff: ignore[suspicious-url-open-usage]` pragma on fixed https constants), and the check moved inside the slow test body as a lazy `pytest.skip("network-unavailable: ...")` whose prefix still matches `tests/expected_skips.yaml`'s network entry. The module now collects with zero network in a minimal `.[test]` install (verified: `--collect-only` runs in seconds, no httpx references outside comments).
**Test added:** none needed (the fix is removal of collection-time I/O); the rewritten slow test was executed once for real and passed (report well-formed, 4 s).

### WR-05: `ListingEntry.availability` typed as bare `object | None` instead of `AvailabilityReport | None`

**Files modified:** `dnallm/preflight/report.py`, `tests/preflight/test_compatibility.py`
**Commit:** 7bc3aaf
**What changed:** `availability: AvailabilityReport | None = None` in `report.py:319`; the `AvailabilityReport` class definition moved above `ListingEntry` in the same module (its only dependencies, `AvailabilityState`/`Hub`, are defined near the top), so no forward reference or import is needed and no cycle exists. All producers already store `AvailabilityReport` or None.
**Test added:** `test_listing_entry_availability_field_is_typed` — pydantic now rejects a junk payload at construction (`ValidationError`), a real report and None still round-trip. Was silently accepted before the fix.

### CR-01 (docstring-only): module docstrings claimed runtime import purity

**Files modified:** `dnallm/preflight/__init__.py`, `dnallm/preflight/report.py`, `dnallm/preflight/config_gen.py` (the three REVIEW-flagged docstrings), plus `dnallm/preflight/validation.py`, `dnallm/preflight/availability.py`, `tests/preflight/test_structure.py` (named in the fix instructions)
**Commit:** da4e44d
**What changed:** Docstring-only rewording — no code, no package `__init__` refactor (explicitly out of scope). Every import-lightness claim now states the truth: the discipline is enforced at SOURCE level (AST structure test), and runtime/`sys.modules` purity is a known open constraint deferred to Phase 25 because importing `dnallm.preflight` transitively executes the `dnallm.utils` and `dnallm.models` package inits (torch/transformers/datasets). The guard test's own docstring (`test_structure.py`) was corrected the same way.
**Test added:** none (no behavior change; diff verified docstring-text-only).

### IN-03: Reader opens files without `encoding="utf-8"` (locale-dependent decoding)

**Files modified:** `dnallm/preflight/validation.py`, `tests/preflight/test_validation.py`
**Commit:** 5486913
**What changed:** Added `encoding="utf-8"` to the three bare `open()` calls — `_check_header_and_type` (header peek), `_load_fasta`, `_load_txt` — matching the module's other call sites and `config_gen.py`'s reader.
**Test added:** `test_validate_data_reader_pins_utf8_under_ascii_locale` — runs `validate_data` on a UTF-8 fasta with a non-ASCII label in a subprocess under a genuinely ASCII preferred encoding (`-X utf8=0` + `LC_ALL=C` + `PYTHONCOERCECLOCALE=0`; plain env manipulation is defeated by PEP 538/UTF-8 mode on this box, verified while writing the test). Before the fix the subprocess died with `'ascii' codec can't decode byte 0xc3`; after it, `ROWCOUNT=1`.

### IN-04: One 115-character line exceeds the project's ruff line-length 100

**Files modified:** `dnallm/preflight/config_gen.py`
**Commit:** ea2a3ca
**What changed:** Wrapped the loader selection into a local (`loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)`) before the `yaml.load` call in `_load_existing`, exactly as the review suggested. `awk` confirms no line over 100 chars remains; the `# ruff: ignore[unsafe-yaml-load]` pragma is still honored (no RUF100) and the python/object-tag refusal pin (`test_python_object_tag_refused`) still passes — the safe-loader mitigation T-23-08 is intact.
**Test added:** none (length/formatting only; existing safe-loader tests pin the behavior).

### IN-08: New `validate_sections` docstring uses the legacy typed-param style

**Files modified:** `dnallm/configuration/configs.py`
**Commit:** f13819e
**What changed:** `config_dict (dict): The raw config mapping.` → `config_dict: The raw config mapping.` (untyped-param style per project convention; types live in the signature).
**Test added:** none (docstring style only).

## Skipped Issues

None — all 10 in-scope findings were fixed.

Deferred without code changes (per fix instructions / review disposition): IN-01 (`label_dtype "int"` unreachable), IN-02 (default label name `"label"` vs `"labels"`), IN-05 (`min_length<=0` pydantic error), IN-06 (raw-alias `isdir` check), IN-07 (executor saturation note), and CR-01's root facade/lazy-init refactor (Phase 25 owner call).

## Verification

All gates ran in the MAIN CHECKOUT on branch v1.4 (`workflow.use_worktrees=false`; no worktree was created, so the numbers below are reproducible directly from this tree):

- `.venv/bin/python -m pytest tests/preflight tests/models/test_registry.py tests/configuration -q -m "not slow"` → **388 passed, 1 deselected** (the slow-lane network test), fully green.
- `.venv/bin/ruff format --check` and `.venv/bin/ruff check` on all 11 touched files → clean.
- Per-module coverage (config-driven `--cov` + scoped `coverage report --fail-under=96`; the `--cov=<pkg>` form trips a numpy/coverage `source=` interaction — `ImportError: cannot load module more than once per process` — on this box, so the equivalent `source_pkgs`-config route was used; pyproject omit list has no preflight/configuration entries, per the test_structure omit guard):
  - `dnallm/preflight/` total **99%** — `__init__` 100%, `availability` 99%, `compatibility` 100%, `config_gen` 100%, `report` 100%, `validation` 99%.
  - `dnallm/configuration/configs.py` **100%**.
- Security mitigations re-checked and unweakened: safe loaders only (T-23-08, refusal pin still green), pkl refusal untouched (T-23-01), single write site + output_path-only writes — now refusing more, not fewer (T-23-09), fixed endpoints (T-23-05, stdlib URLs are compile-time constants), no coverage-gate bypass (T-23-10).

---

_Fixed: 2026-10-20T18:22:34Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
