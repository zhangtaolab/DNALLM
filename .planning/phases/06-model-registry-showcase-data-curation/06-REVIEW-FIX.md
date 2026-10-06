---
phase: 06-model-registry-showcase-data-curation
fixed_at: 2026-10-06T16:22:08Z
review_path: .planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 06: Code Review Fix Report

**Fixed at:** 2026-10-06T16:22:08Z
**Source review:** `.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 3 (WR-01, WR-02, IN-01 — fix_scope `all`)
- Fixed: 3
- Skipped: 0

All changes are confined to `tests/models/test_plant_helixseek_smoke.py` (plus its
docstring/comments), as scoped. No production code touched. Commits are on branch `phs`.

## Fixed Issues

### WR-01: `_is_environment_error` docstring cites stale `model.py` line numbers for every load-ladder anchor

**Files modified:** `tests/models/test_plant_helixseek_smoke.py`
**Commit:** `ad6a920`
**Status:** fixed
**Applied fix:** All four anchors re-pinned to current lines and re-anchored by symbol so
future drift cannot silently invalidate the contract:

| Anchor (symbol-first) | Old pin | New pin (verified at HEAD) |
|---|---|---|
| terminal raise of `download_model`'s retry loop, ``Model {name} download failed.`` | model.py:375 | model.py:389 |
| boundary wrap ``raise ValueError(f"Failed to load model: {e}") from e`` in `load_model_and_tokenizer` | model.py:887-888 | model.py:917 |
| `_get_model_path_and_imports` call site outside the boundary ``try`` | model.py:834 | model.py:863 (try opens at 873-917) |
| hf/modelscope function-local imports + modelscope guard in `_get_model_path_and_imports` | model.py:444-448 / 476-480 | model.py:451, 456-458, 474-477 |

The docstring now opens with "Anchors cite the symbol first so the contract survives future
edits; the line numbers are courtesy pins, current at the 2026-10-06 review". The two test
comments citing model.py:375 and model.py:887-888 were re-pinned the same way.

### WR-02: type-based classification still whitelists dnallm-originating `ImportError`/`OSError` as environment-class

**Files modified:** `tests/models/test_plant_helixseek_smoke.py`
**Commit:** `7134aa6`
**Status:** fixed: requires human verification (classification-logic change; behavior is
pinned by the new same-change tests below, but the semantics deserve a human read)
**Applied fix:** `_is_environment_error` now origin-checks `OSError`/`ImportError` nodes via a
new `_raise_site_module(exc)` helper that walks `exc.__traceback__` to the innermost frame
(the raise site / failed import) and returns its module name. An `OSError`/`ImportError`
whose raise site is `dnallm`/`dnallm.*` no longer classifies as environmental (the chain
walk continues to deeper causes and the test fails loud); raise sites in
huggingface_hub/modelscope/requests/urllib3/socket/ssl/remote-code frames keep the typed
skip. `ConnectionError`/`TimeoutError` and the terminal-download message-shape match remain
unconditional, checked before the `OSError` branch (both subclass `OSError`).

**Deliberate adaptation of the suggested fix:** the review's literal suggestion ("only treat
the node as environmental when NO frame's module starts with `dnallm`") is unsatisfiable at
runtime: any exception caught at the test boundary carries dnallm wrapper frames above the
true origin (`load_model_and_tokenizer` called into the failing library), so an all-frames
check would classify every genuine network error as a regression and kill the typed skip
entirely. The innermost-frame (raise-site) check implements the stated intent — "a frame
whose module *originates* inside `dnallm/`" — and was validated empirically before applying:
a failed function-local import's innermost frame is the frame executing the import (importlib
strips its own frames), an env-origin OSError's innermost frame is the library module, and a
dnallm-frame raise is discriminated cleanly.

**Documented consequence (now in the docstring):** huggingface-hub and modelscope are *base*
dependencies (`pyproject.toml` deps), so their absence fails the function-local imports in
`_get_model_path_and_imports` at a dnallm raise site and now fails loud instead of skipping —
a missing base install is a broken environment, not an outage. Genuinely optional deps (e.g.
fla imported by the checkpoint's remote code) raise in non-dnallm frames and keep the skip.

**New same-change classification tests** (fast, no network):
- `test_dnallm_origin_import_error_is_not_environmental` — broken function-local import
  raised in a `dnallm.*`-named frame, boundary-wrapped → NOT environmental.
- `test_dnallm_origin_file_not_found_is_not_environmental` — `FileNotFoundError` (OSError
  subclass) raised in a `dnallm.*`-named frame, boundary-wrapped → NOT environmental.
- `test_env_origin_os_error_keeps_the_typed_skip` — same OSError type raised under a
  `huggingface_hub.*`-named frame → still environmental (mirror of the origin check).

The probes use a synthetic module (`types.ModuleType` + `exec`, suppressed with the
repo-convention `# ruff: ignore[exec-builtin]`) because the classifier keys on
`frame.f_globals["__name__"]`; all 7 pre-existing classification tests stay green unchanged.

### IN-01: fla `importorskip` guard fires before `_emit_env()`, so a fla-missing typed skip carries no version evidence

**Files modified:** `tests/models/test_plant_helixseek_smoke.py`
**Commit:** `8669164`
**Status:** fixed
**Applied fix:** In both slow smokes, `_emit_env()` now runs BEFORE the `importorskip` guard,
and the guard reason interpolates the env versions the same way `_load_with_fallback` does:
`environment-unavailable: flash-linear-attention not installed (transformers
{transformers.__version__}, torch {torch.__version__}) — …`. The registered
`environment-unavailable:` prefix is byte-identical, so the `tests/expected_skips.yaml`
prefix gate is unaffected.

**End-to-end proof (simulated fla-missing leg):** with `fla` blocked at a meta-path finder
(raising `ModuleNotFoundError`), the CRE smoke emits `transformers_version=5.17.0` /
`torch_version=2.11.0+cu130` before the guard and then SKIPS with the full versioned reason.
Side observation (pre-existing, out of scope, unchanged): pytest 9's `importorskip` defaults
to catching only `ModuleNotFoundError`, so a *present-but-broken* fla install (plain
`ImportError` from `fla/__init__`) fails loud rather than skipping — reasonable and not
part of this finding.

## Skipped Issues

None — all three in-scope findings were fixed.

## Verification

**Where verification ran:** main checkout at branch `phs` HEAD `8669164`
(`workflow.use_worktrees=false` — no isolated worktree; results are reproducible directly
from this tree).

Per-fix: `ast.parse` clean; `ruff format --check` clean; `ruff check` clean
(repo-convention suppression comment used for the one `exec` probe); targeted
`pytest -m "not slow"` runs green after each fix.

**Owner-rule test evidence** (same change, main checkout):

```
$ .venv/bin/python -m pytest tests/models/test_plant_helixseek_smoke.py \
    tests/models/test_plant_helixseek_registry.py \
    tests/models/test_plant_helixseek_fla_kernels.py \
    -m "not slow and not giants" -q --tb=short

tests/models/test_plant_helixseek_smoke.py ..........                    [ 52%]
tests/models/test_plant_helixseek_registry.py ...                        [ 68%]
tests/models/test_plant_helixseek_fla_kernels.py ......                  [100%]
======================= 19 passed, 2 deselected in 5.82s =======================
```

19 passed (10 smoke-file tests = 7 pre-existing + 3 new WR-02 classification tests; 3
registry; 6 fla-kernels). The 2 deselected are the two `slow`-marked real-checkpoint loads,
correctly excluded by the `-m "not slow and not giants"` filter per instructions.

---

_Fixed: 2026-10-06T16:22:08Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
