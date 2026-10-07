---
phase: 06-model-registry-showcase-data-curation
reviewed: 2026-10-03T18:20:00Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - .gitignore
  - dnallm/models/model_info.yaml
  - dnallm/utils/genomic_coords.py
  - dnallm/utils/__init__.py
  - example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
  - example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
  - example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
  - example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
  - example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
  - example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
  - example/notebooks/plant_helixseek_shared/data/selection.md
  - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff
  - example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
  - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3
  - example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
  - example/notebooks/plant_helixseek_shared/.gitignore
  - pyproject.toml
  - tests/models/test_plant_helixseek_fla_kernels.py
  - tests/models/test_plant_helixseek_registry.py
  - tests/models/test_plant_helixseek_smoke.py
  - tests/utils/test_genomic_coords.py
findings:
  critical: 0
  warning: 5
  info: 6
  total: 11
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-10-03T18:20:00Z
**Depth:** standard
**Files Reviewed:** 22
**Status:** issues_found

## Summary

Reviewed the Phase 06 delta: two PlantHelixSeek registry entries, the new
`dnallm/utils/genomic_coords.py` helper plus its re-export, four new test files,
the `fla` extra in `pyproject.toml`, ten committed showcase data artifacts plus
`selection.md`, and the scratch-home `.gitignore`.

Verification performed (not just read):

- **Data integrity: clean.** Every `.fas` is exactly the interval length (200000/20000 bp),
  uppercase ACGT, uniform 80-col wrapping, header `>Chr1 TAIR10 fragment [start, end] 1-based closed`
  (pyfastx index name `Chr1` matches `fetch_sequence` lookups). All 1,696 GFF/GFF3 rows are
  9-column, Chr1-only, fully contained in their locus intervals, LF-only. Row counts reproduce
  `selection.md`'s claims exactly: 94 DHS rows (= ranking-table `dhs` for the tile), 526 CDS rows
  (`n_truth_cds`), 91 mRNAs (`n_genes`), tp/fp/fn arithmetic consistent (346+180=526, 346+48=394),
  intergenic slices zero bytes as documented, all byte totals match the artifacts table
  (569,825 data bytes; `audit_bytes=580204` additionally included `selection.md` at its pre-audit-append
  10,379 bytes — coherent). The five ranking-table scores are internally consistent to 6 decimals
  with `max_dhs=106`, `max_complete_mRNA=87`. The slice contains 6 CDS rows (gene AT1G14800.1)
  whose parent mRNA row sits partly outside the locus and is therefore absent — correct per the
  "rows fully contained, verbatim" rule, but Phase 7 per-gene metric code must not assume every
  truth CDS row's Parent is resolvable inside the slice.
- **Tests/lint: pass locally.** 26 fast tests pass (registry, fla kernels, genomic_coords) on the
  dev venv (fla installed); `ruff format --check` and `ruff check` clean on all six Python files;
  smoke module collects. Registry YAML parses; entries unique; `TaskConfig` accepts
  `binary`/`token` and preserves explicit `label_names` (verified against
  `dnallm/configuration/configs.py:100-150`). Registry has no runtime consumer, as the test
  docstring claims (only a comment in `dnallm/mcp/server.py:686` references the yaml).
- **No security issues:** no eval/shell/injection surface; `genomic_coords` validates all inputs
  and raises `ValueError` per project convention.

The defects are in CI wiring and test/robustness details, led by one systemic gap: **no CI leg
installs the `fla` extra**, so the phase's own kernel-degradation guard can never fire in CI, and
the nightly slow leg runs the checkpoint smoke tests on exactly the silent non-KDA fallback path
the phase documents as producing wrong outputs (WR-01).

Note (not a finding): `uv run` currently fails to resolve in this repo (pre-existing `mamba` extra
vs torch cu121 wheel conflict on the darwin/py3.14 split) — unrelated to this diff; venv binaries
were used instead.

## Warnings

### WR-01: `fla` extra is installed by no CI leg — nightly smoke tests run on the silent non-KDA fallback path

**File:** `pyproject.toml:127-129,156-161`; `tests/models/test_plant_helixseek_fla_kernels.py:43-51`; `tests/models/test_plant_helixseek_smoke.py:78-133`
**Issue:** `fla` was wired into `[all]` only, but every CI leg installs `.[base]` (or `.[base,cudaNNN]`/`.[mamba]` — `.github/workflows/ci.yml` lines 68, 157, 252-254, 319-320, 386, 470; no leg installs `.[all]` or `.[fla]`). Consequences, all verified against the workflow file:
1. `test_chunk_kda_importable_when_fla_installed` (not slow-marked) runs on the fast legs and **skips** via `importorskip`; its skip reason starts with `environment-unavailable:`, which is a whitelisted prefix in `tests/expected_skips.yaml` — so the skip audit passes and CI stays green with flash-linear-attention never exercised anywhere.
2. The coverage-nightly leg (line 404, "full suite incl. slow", `pytest -ra` with no `-m` filter) installs `.[base]` (line 470) and therefore loads both real ~1.9 GB checkpoints **without fla** — the remote code silently falls back to the pure-PyTorch path the phase's own docstring proves is not KDA math (probe-set p(CRE) 0.0073 in-DHS vs 0.0087 non-DHS). The smoke tests assert only shapes and `id2label`, both of which pass on the degraded path. Net effect: the only CI environment that touches these models validates the wrong kernel path and reports green.
**Fix:** Add the extra to the legs that run these tests — at minimum the nightly/coverage legs:
```yaml
# .github/workflows/ci.yml, coverage-nightly (and ideally the mamba nightly leg)
- name: Create virtual environment and install dependencies
  run: |
    uv venv
    uv pip install -e ".[base,fla]"
    uv pip install "numpy==2.2.0"
```
Additionally (defense in depth), have the smoke tests skip with distinct evidence when the kernels are absent, instead of running degraded:
```python
# tests/models/test_plant_helixseek_smoke.py, top of each smoke test
pytest.importorskip(
    "fla",
    reason="environment-unavailable: flash-linear-attention not installed — "
    "PlantHelixSeek loads would run the degraded non-KDA fallback",
)
```

### WR-02: `_load_with_fallback` converts any exception — including dnallm code regressions — into a green skip

**File:** `tests/models/test_plant_helixseek_smoke.py:65-75`
**Issue:** `except Exception as exc: errors.append(...)` followed by `pytest.skip(...)` means a `TypeError`/`AttributeError`/`KeyError` introduced by a refactor of `load_model_and_tokenizer` (or of the config plumbing feeding it) is reported as `environment-unavailable:` on both routes — a category the skip audit whitelists. A real load-path regression would therefore present as an environment skip, not a failure. The documented intent is network/hub unavailability, not "any error".
**Fix:** Only treat remote/network failure classes as environment issues; let code errors fail:
```python
NETWORK_ERRORS = (
    ConnectionError, TimeoutError, OSError,  # hub/download failures land here
)
def _load_with_fallback(repo_id, cfg):
    errors = []
    for source in ("modelscope", "huggingface"):
        try:
            return load_model_and_tokenizer(repo_id, cfg, source=source)
        except NETWORK_ERRORS as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
        # non-network exceptions propagate -> test fails -> regression is visible
    pytest.skip("environment-unavailable: ..." )
```
(If `huggingface_hub`/`modelscope` raise typed errors not subclassing these, enumerate those types instead — the point is to stop catching everything.)

### WR-03: `slice_gff_rows` strips only `\n` — CRLF input silently corrupts column-9 values

**File:** `dnallm/utils/genomic_coords.py:227,234`
**Issue:** `row.rstrip("\n")` leaves a trailing `\r` on the last column of CRLF GFF files. Coordinates and chrom matching still work, so such rows are silently matched and returned with `"Name=x\r"` / `"Parent=y\r"` intact — and `parse_gff_attributes` then yields values containing `\r` with no error. This is exactly the silent-corruption failure mode the module docstring promises is "structurally impossible" here. The committed truth slices are LF (verified), so Phase 6 data is unaffected; the defect fires on user-supplied files (Windows-edited GFFs are common).
**Fix:**
```python
columns = row.rstrip("\r\n").split("\t")   # line 227
...
matched.append(row.rstrip("\r\n"))          # line 234
```

### WR-04: `pyfastx` missing from the mypy overrides list — new `import-not-found` error introduced

**File:** `dnallm/utils/genomic_coords.py:153`; `pyproject.toml:413-479`
**Issue:** Confirmed via `mypy dnallm/utils/genomic_coords.py`:
`error: Cannot find implementation or library stub for module named "pyfastx" [import-not-found]`.
The project convention for untyped/compiled third-party modules is an entry in `[[tool.mypy.overrides]]`
(`pyfaidx.*` is already listed; `pyfastx` is the same kind of dev-extra Cython package). Pre-commit runs
`mypy dnallm/` without `|| true` (`.pre-commit-config.yaml:27-32`), so this adds new hook noise; CI mypy is advisory but this is a regression against the documented pattern.
**Fix:** Add to the `[[tool.mypy.overrides]]` module list in `pyproject.toml`:
```toml
    "pyfastx.*",
```

### WR-05: `tomllib` (Python 3.11+) breaks test collection on Python 3.10, still a declared supported version

**File:** `tests/models/test_plant_helixseek_fla_kernels.py:12`
**Issue:** `import tomllib` fails on Python 3.10 (`requires-python = ">=3.10"`; classifiers still list 3.10; ruff `target-version = "py310"`; mypy `python_version = "3.10"`). A collection-time `ImportError` fails the whole pytest run on 3.10, not just this file. This is the first and only `tomllib` use in the repo (verified by grep). CI matrix is 3.11-3.13, so nothing catches it — which is precisely why the declared floor and the suite diverge silently. (If the floor raise to 3.12 lands, this finding evaporates — but it is not true today.)
**Fix:** Either
```python
import sys
if sys.version_info < (3, 11):
    import tomli as tomllib  # dev-extra dependency, or:
```
or parse the two lines without tomllib, or gate the test module with `pytest.importorskip("tomllib")`.

## Info

### IN-01: `fetch_sequence(path)` leaks the pyfastx handle and creates a `.fxi` index beside the user's FASTA

**File:** `dnallm/utils/genomic_coords.py:152-155`
**Issue:** When a path is passed, `pyfastx.Fasta(fa)` is constructed and never closed; pyfastx also builds an index file next to the target by default — an undocumented filesystem side effect for callers.
**Fix:** Document the side effect in the docstring, or accept only open indices from callers and drop the path branch; at minimum set `build_index=False` if the use pattern allows.

### IN-02: `normalize_chrom` accepts non-ASCII digit strings and silently renames them

**File:** `dnallm/utils/genomic_coords.py:72-73`
**Issue:** `name.isdigit()` is True for full-width/superscript digits (e.g. `"２"`), so `normalize_chrom("２")` returns `"Chr２"` — an undocumented form renamed instead of rejected, against the module's own "never renamed heuristically" rule.
**Fix:** `if name.isascii() and name.isdigit():` — likewise consider `match.group(1).isascii()` for the chr-prefixed branch (the regex `[0-9]` already restricts that branch to ASCII, so only the bare-numeric branch needs it).

### IN-03: `test_module_import_is_pyfastx_free` leaves the package attribute pointing at a re-executed module

**File:** `tests/utils/test_genomic_coords.py:258-274`
**Issue:** The test restores `sys.modules` but not the `dnallm.utils.genomic_coords` attribute on the package, which `importlib.import_module` rebinds to the fresh module object. Harmless while the module is stateless; a latent trap if it ever gains module-level state (two live module objects).
**Fix:** Also save/restore `getattr(dnallm.utils, "genomic_coords", None)` in the `finally` block.

### IN-04: Anno `label_names` uses single quotes, rest of the registry uses double quotes

**File:** `dnallm/models/model_info.yaml:1657`
**Issue:** The 17-label flow-style list is single-quoted while every other entry (including the CRE entry two lines up) uses double quotes. Parses identically; cosmetic inconsistency in a 1,600-line hand-maintained registry.
**Fix:** Re-quote to double quotes for uniformity.

### IN-05: `test_fla_reachable_from_all` matches the extra by substring

**File:** `tests/models/test_plant_helixseek_fla_kernels.py:35-37`
**Issue:** `any("fla" in spec for spec in extras["all"])` would also match a future `"flash-attn"`-style spec, passing the guard with the fla extra accidentally dropped.
**Fix:** Assert the exact meta-extra token, e.g. `any(re.search(r"\bfla\b", spec) for spec in extras["all"])` or `"dnallm[...,...,fla]"`-shape check.

### IN-06: Local `.scratch/` ignore is redundant with the root pattern

**File:** `example/notebooks/plant_helixseek_shared/.gitignore:1`
**Issue:** Root `.gitignore:60` already has an unanchored `.scratch/` that matches at any depth (verified with `git check-ignore`), so the local file duplicates it. Harmless defense-in-depth for the documented owner constraint; recorded so a future root cleanup knows it exists.
**Fix:** None required; optionally a comment noting it mirrors the root rule.

---

_Reviewed: 2026-10-03T18:20:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
