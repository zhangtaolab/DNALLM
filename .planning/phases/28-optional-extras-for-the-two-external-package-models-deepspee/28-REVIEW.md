---
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
reviewed: 2026-10-29T17:12:50Z
depth: standard
files_reviewed: 32
files_reviewed_list:
  - dnallm/models/model_info.yaml
  - dnallm/models/model.py
  - dnallm/models/registry.py
  - dnallm/models/special/crossdna.py
  - dnallm/models/support.py
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_combined.md
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
  - docs/faq/models_troubleshooting.md
  - docs/getting_started/installation.md
  - docs/getting_started/install_honesty.md
  - docs/getting_started/quick_start.md
  - docs/user_guide/model_compatibility.md
  - docs/user_guide/optional_dependencies.md
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - example/notebooks/plant_helixseek_shared/plant_helixseek_combined.ipynb
  - README.md
  - scripts/census_compat.py
  - scripts/install_flash_attn.sh
  - tests/models/compat_census.json
  - tests/models/test_census_compat.py
  - tests/models/test_compat_docs_consistency.py
  - tests/models/test_model_info_registry.py
  - tests/models/test_registry.py
  - tests/models/test_special/test_family_handlers.py
  - tests/models/test_support.py
  - tests/test_extras_guard.py
  - WINDOWS-KDA-VERIFY.md
findings:
  critical: 0
  warning: 2
  info: 3
  total: 5
status: issues_found
---

# Phase 28: Code Review Report

**Reviewed:** 2026-10-29T17:12:50Z
**Depth:** standard
**Files Reviewed:** 32
**Status:** issues_found

## Summary

Reviewed the Phase 28 optional-extras surface at standard depth: the curated
`deepspeed`/`flash-attn` extras and their fail-closed guards, the CrossDNA
`extra-required-tf5` registry flip and its propagation through
`registry.py` / `support.py` / `model_info.yaml` / `compat_census.json`, the
missing-extra loader guidance (`_raise_missing_extra_import_error`,
`_from_pretrained_with_flash_attn_guidance`), the census tool's new
`extra` disposition branch and docs generator changes, the sha256-pinned
flash-attn installer script, and the rewritten/retired docs and notebooks.

The runtime code is sound. The exception-chain traversal in
`_raise_missing_extra_import_error` is cycle-safe and correctly wired into
both load-chain wrap sites (model.py:1424, model.py:1495); the CrossDNA
guidance marker survives the generic `ValueError` wrap; the
`compat_verdict` `extra`-before-skip-and-green ordering is correct and
symmetric for both the extras-green and default-env-failure shapes; the
`support.py` vocabulary additions flow automatically to both consumers
(`dnallm/mcp/server.py:862`, `dnallm/preflight/compatibility.py:584`). The
installer's patch assertions are occurrence-exact on both sides of every
edit and backstopped by an `ast.parse` fail-closed check, and the sha256
gate runs before any extraction or execution. Verified green by execution:
208 tests across `test_support` / `test_registry` / `test_model_info_registry`
/ `test_compat_docs_consistency` / `test_extras_guard`, 17 in
`TestMissingExtraGuidance`, 157 in `test_census_compat`, offline
`census_compat.py --verify` (all 49 rows matched, including
`chengCCC/CrossDNA_pretrain extra-required-tf5`), and a byte-identical
`--emit-docs` regeneration of `model_compatibility.md`.

No critical findings. Two warnings: a stale `base` group left on a line this
phase edited in `quick_start.md`, and unbounded `mktemp` leakage in the
nightly-run installer script.

## Warnings

### WR-01: Edited quick_start line still advertises the removed `base` dependency group

**File:** `docs/getting_started/quick_start.md:347`
**Issue:** This phase's fla-retirement sweep edited this exact line (dropping
`fla`) but left `base` in the list: "All available dependency groups (`all`,
`base`, `dev`, `test`, `notebook`, `docs`, `ui`, `mcp`)". The `base` extra
was removed at v1.4 (PKG-01, asserted by
`tests/test_extras_guard.py::TestFullExtraAndBaseAlias::test_base_alias_is_removed`);
pyproject's extras table carries no `base` key, and the referenced target
page (`docs/getting_started/installation.md`) no longer documents one. A
user following this list runs `pip install 'dnallm[base]'` and hits exactly
the warn-and-install-core-only outcome `install_honesty.md:16` documents as
the missing-extra hazard. The 18-01 residue gate did not catch it because
`SANCTIONED_BASE_BRACKET_RESIDUE` only matches bracket-install forms
(`.[base` / `dnallm[base`), not a bare `` `base` `` mention.
**Fix:**
```markdown
-  - All available dependency groups (`all`, `dev`, `test`, `notebook`, `docs`, `ui`, `mcp`, `tui`)
```
(Match the actual `installation.md` group table, which includes `tui` and
omits `base`.)

### WR-02: flash-attn installer never cleans its temp directories; nightly lane leaks them indefinitely

**File:** `scripts/install_flash_attn.sh:207,228,327-331`
**Issue:** The script creates three fresh `mktemp -d` directories
(`DOWNLOAD_DIR` for the sdist, `EXTRACT_DIR` for the patched tree, and the
default `WHEELHOUSE_DIR` holding the built wheel — tens to a couple hundred
MB for a single-arch CUDA build) and never removes any of them on any exit
path. The nightly extras lane invokes it without `--wheelhouse`
(`.github/workflows/extras_compat.yml:125`:
`bash scripts/install_flash_attn.sh --python .venv/bin/python --index-url ...`),
so the single self-hosted `dnallm-nightly` runner — which also serves the
03:00 and 05:30 lanes — accumulates hundreds of MB per night in `/tmp` with
no bound. Left unattended this eventually exhausts the runner's disk (or
tmpfs RAM) and reds every lane on that box. Note the no-routine-cleanup
owner rule covers model caches, not throwaway build scratch.
**Fix:**
```bash
DOWNLOAD_DIR="$(mktemp -d)"
EXTRACT_DIR="$(mktemp -d)"
CREATED_WHEELHOUSE=""
...
if [ -z "$WHEELHOUSE_DIR" ]; then
    WHEELHOUSE_DIR="$(mktemp -d)"
    CREATED_WHEELHOUSE="$WHEELHOUSE_DIR"
else
    mkdir -p "$WHEELHOUSE_DIR"
fi
# Remove only directories this run created; a caller-passed --wheelhouse
# is theirs to keep.
trap 'rm -rf "$DOWNLOAD_DIR" "$EXTRACT_DIR" ${CREATED_WHEELHOUSE:+"$CREATED_WHEELHOUSE"}' EXIT
```

## Info

### IN-01: `TF_COMPAT_STATUSES` is a dead constant that this phase updated to look load-bearing

**File:** `scripts/census_compat.py:1424-1431`
**Issue:** The frozenset was extended with `extra-required-tf5` in this
window, but a repo-wide search shows it is referenced nowhere else — not by
`verify_registry`, `docs_page_text`, or any test. A future vocabulary change
can therefore drift from whatever it was meant to pin, while its maintained
appearance suggests it is enforced.
**Fix:** Either delete it, or wire it in where the vocabulary is actually
consumed, e.g. in `docs_status_cell`: reject any status not in
`TF_COMPAT_STATUSES` before the per-family branches.

### IN-02: Stale count in the `docs_status_cell` docstring

**File:** `scripts/census_compat.py:2046`
**Issue:** "The exact status-cell wording (one of the five pinned strings,
...)" — `DOCS_STATUS_CELLS` now holds four keys (this phase added
`extra-required-tf5`). The count was already wrong at the diff base (three
cells, said "five") and this phase's sibling docstring fixes
(`docs_page_text` "five-value" → "six-value") left it behind.
**Fix:** Change "five pinned strings" to "four pinned strings".

### IN-03: BigBird compatibility row's lead sentence contradicts its own technical suffix on the load half

**File:** `docs/user_guide/model_compatibility.md:42` (generator:
`scripts/census_compat.py` `DOCS_BROKEN_LEADS["library-change"]`, gate copy:
`tests/models/test_compat_docs_consistency.py:67`)
**Issue:** The row reads "...so it cannot be loaded until the model authors
update it (technical: load works with the deepspeed extra but forward is
blocked by an upstream-dead Triton API)". The owner-ruled class is
library-change and the actionable verdict (unusable for predictions) is
right, but the generic lead's "cannot be loaded" is contradicted by both the
suffix and the guide section, which state loading works with the extra — a
biologist cross-reading the two pages gets opposite claims about the load
half. Wording-level only; the class assignment itself is owner-approved.
**Fix:** If touched, change the library-change lead's "so it cannot be
loaded until the model authors update it" to "so it cannot be run until the
model authors update it" in lockstep (generator `DOCS_BROKEN_LEADS` and the
gate's `BROKEN_LEADS` copy), then regenerate the page.

## Out-of-scope observations (glaring, pre-existing, unchanged in this window)

- `docs/guides/development.md:79` — the `all` extras row still lists `fla`
  (`full + dev + test + notebook + docs + ui + mcp + fla + tui`), which this
  phase's retirement removed from the real `all` bracket. The file is
  outside both the review list and the phase's diff; recommend a follow-up
  sweep so the retirement is complete repo-wide (no gate covers it:
  `test_curated_and_retired_extras_absent_from_meta_extras` checks
  pyproject, not prose).
- `dnallm/models/special/crossdna.py:20` — the module logger is named
  `"dnallm.models.model"` (copy-paste from the loader), misattributing this
  module's log records to the loader's channel. Pre-existing, untouched by
  this phase's edits to the same file.

---

_Reviewed: 2026-10-29T17:12:50Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
