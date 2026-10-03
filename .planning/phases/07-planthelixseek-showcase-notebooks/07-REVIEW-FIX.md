---
phase: 07-planthelixseek-showcase-notebooks
fixed_at: 2026-10-03T21:31:12Z
review_path: .planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md
iteration: 2
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 07: Code Review Fix Report

**Fixed at (this iteration):** 2026-10-03T21:31:12Z
**Source review:** `.planning/phases/07-planthelixseek-showcase-notebooks/07-REVIEW.md`
**Iteration:** 2 (cumulative report — see below)

**Summary:**
- Findings in scope (cumulative over both fix passes on this review): 8 (0 Critical, 2 Warning, 6 Info)
- Fixed: 8 (2 in iteration 1, 6 in iteration 2)
- Skipped: 0

**Iteration history:**
- **Iteration 1** (`fix_scope: critical_warning`, 2026-10-03T21:05:59Z): fixed WR-01 (`85f0f23`, `scripts/check_docs_sync.py` ignore set) and WR-02 (`9fd08d7`, CRE notebook cell-18 band-parse guard). Details preserved in the section at the bottom; both are recorded `fixed` in `07-REVIEW-DISPOSITION.md`.
- **Iteration 2** (this run, `fix_scope: all`): fixed the six Info findings IN-01..IN-06. The Warnings were NOT re-attempted (already fixed; per-run instruction).

All verification for this iteration ran in the MAIN CHECKOUT at `/home/forrest/Github/DNALLM`, branch `phs` — `workflow.use_worktrees=false`, so no isolated worktree was created and every number below is reproducible from this tree.

## Fixed Issues

### IN-01: models.lock comment still describes the Anno test as future work

**Files modified:** `models.lock`
**Commit:** `92e584f`
**Applied fix:** Dropped the stale `; Anno test lands with 07-02` clause from the `zhangtaolab/PlantHelixSeek-Anno` comment (line 13); it now matches line 12's phrasing exactly (`# tests/examples/test_plant_helixseek_showcase.py (nightly showcase execution, source=modelscope)`), since 07-02 landed `test_anno_notebook_executes_within_selection_bands`.

### IN-02: Harness docstring budget census is off by the two new entries

**Files modified:** `tests/examples/_execution.py`
**Commit:** `0b4fa07`
**Applied fix:** Rewrote the `NOTEBOOK_EXEC_SPECS` doc-note sentence to "All 21 census example notebooks carry starter budgets (05-05, D-08); the two Phase-7 showcase-lane entries below are budget-only (their tests live in tests/examples/test_plant_helixseek_showcase.py)." Verified the dict really holds 23 `"cell_timeout":` entries (21 census + CRE/Anno showcase) before rewording. AST parse clean.

### IN-03: Anno wrapper claims a per-gene F1 print that the notebook does not emit

**Files modified:** `docs/example/notebooks/plant_helixseek_anno.md`
**Commit:** `c6ab77e`
**Applied fix:** Reworded the sentence to "The notebook also computes per-gene exon F1 (genes = mRNAs with CDS rows in the slice; the floor counts genes with per-gene exon-F1 >= 0.8) and prints the count of genes meeting the 0.8 floor (`genes_above_floor=`) and nucleotide-level sensitivity / precision / F1 as `key=value` stream lines, ...". Confirmed against the committed notebook first: cell 17 computes `gene_f1_scores` but its stream output carries only aggregates (`exon_f1=`, `genes_above_floor=`, `n_genes=`, `nt_*=`) — no per-gene value. Prose-only change; `check_notebook_md_sync.py` failure set unchanged (only the 3 pre-existing stale wrappers documented out of scope in the review; zero `plant_helixseek` entries).

### IN-04: Unused `locus_key` parameter in five of seven parametrized structure tests

**Files modified:** `tests/examples/test_plant_helixseek_showcase.py`
**Commit:** `5f7a164`
**Applied fix:** Took the reviewer's second option (keep the uniform signature, note the intent): extended the `SHOWCASE_NOTEBOOKS` comment to state that every structure test takes the uniform `(nb_path, locus_key)` signature against the one shared list even where it reads only `nb_path` — single-source parametrization beats five bespoke argument lists — and that only `test_provenance_markdown_cell` consumes `locus_key`. Chosen over dropping the parameter because the five tests share the `SHOWCASE_NOTEBOOKS` parametrization; per-test argument lists would fragment that surface. AST parse + ruff clean.

### IN-05: bedtools dependency of the nightly CRE test is an unencoded runner assumption

**Files modified:** `tests/examples/test_plant_helixseek_showcase.py`
**Commit:** `4c9ce59`
**Applied fix:** Added `import shutil` and an upfront guard as the first statement of `test_cre_notebook_executes_within_selection_bands`: `assert shutil.which("bedtools") is not None` with the message "bedtools is not on PATH -- the CRE notebook's jaccard agreement step requires bedtools v2.31+ (see 'Prerequisites' in docs/example/notebooks/plant_helixseek_cre.md)". Still fail-loud per the module's stated design, but self-describing instead of a bare `FileNotFoundError` from inside the kernel on a rebuilt runner. Behaviorally verified both branches of the condition (`shutil.which("bedtools")` returns the runner's `/home/linuxbrew/.linuxbrew/bin/bedtools`; returns `None` with a stripped PATH). AST parse + ruff check/format clean; the fast structure lane (13 tests) passes with the new import.

### IN-06: Anno label-order usage mixes registry-derived and hardcoded indexes

**Files modified:** `docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb`, `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` (byte-identical mirrors; identical edit applied to both)
**Commit:** `2c56c51`
**Status:** fixed: requires human verification (notebook-source edit with committed pre-edit outputs — the nightly lane re-executes and confirms; see verification below)
**Applied fix:** Added a registry label-order assert in cell 11, immediately after the `_B_SWAP_L` transcription and before `swap = np.array(...)`: `assert label_names == ["O", "B-CDS", "I-CDS", "L-CDS", "U-CDS", "B-INTRON", ..., "U-UTR3"], "registry label order drifted from the frozen _B_SWAP_L permutation"`, preceded by a 4-line comment explaining that both the permutation and the intergenic control's `labels != 0` = `O` reading are positional while `label_names.index()` (cell 13) would silently follow a reorder. Strengthened from the reviewer's partial `[0]`/`[1:5]` example to the full 17-element equality because the permutation is itself 17-element positional — a reorder inside the INTRON/UTR families would break it while passing the partial check, and the full assert also covers cell 22's index-0 assumption. Edit applied as a binary-safe exact string replacement in the raw JSON `source` array (single-anchor-match assert, `json.loads` + cell `ast.parse` + <=100-char line checks, then written); the mirrors remain byte-identical (`cmp` clean).

**Verification (iteration 2; all in the main checkout):**
- `python -m pytest tests/examples/test_plant_helixseek_showcase.py -q -m "not slow"`: **13 passed, 2 deselected in 0.82s** — the fast structure lane over both edited notebooks (AST walk, fla-guard, captions, committed outputs, 2 MB budget) and the test module with the new `import shutil`.
- `ruff check` + `ruff format --check` on `tests/examples/_execution.py` and `tests/examples/test_plant_helixseek_showcase.py`: clean.
- `python scripts/check_docs_sync.py`: exit 0 (`OK: docs/example/ is in sync with example/`) — proves the two Anno notebook mirrors stayed byte-identical.
- `python scripts/check_notebook_md_sync.py`: failure set unchanged (the 3 pre-existing stale MCP/data-prepare wrappers; zero `plant_helixseek` entries).
- IN-06 assert truth verified against two independent sources before committing: the live registry (`dnallm/models/model_info.yaml` `label_names` for `zhangtaolab/PlantHelixSeek-Anno`) and the committed runtime outputs of notebook cell 3 (`registry_label_names=[...]`, `model_id2label={0: 'O', ..., 16: 'U-UTR3'}`) — both equal the asserted 17-element order, so the assert is behavior-preserving for the nightly re-execution lane.
- Notebook size after edit: 290,416 bytes per mirror (budget 2,097,152).
- An initial full-module pytest invocation (without `-m "not slow"`) was intentionally stopped: it had started the two nightly GPU re-execution tests (~40-80 min), which are the verifier phase's job, not per-fix verification.

## Fixed Issues

### WR-01: docs-sync gate ignores some sanctioned runtime artifacts, fails red on any tree where the benchmark example ran

**Files modified:** `scripts/check_docs_sync.py`
**Commit:** `85f0f23`
**Applied fix (summary):** Added `"benchmark_results"` to `IGNORE` and `".pdf"` to `IGNORE_SUFFIXES`; `check_docs_sync.py` went from exit 1 (spurious `ONLY in example/` on gitignored runtime artifacts) to exit 0. Re-confirmed exit 0 during iteration 2.

### WR-02: CRE notebook band-table parse fails as a bare KeyError instead of the documented parse guard

**Files modified:** `docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb`, `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb`
**Commit:** `9fd08d7`
**Applied fix (summary):** Cell-18 band loop now raises `RuntimeError` at the parse site when a prefix-matching row carries no `[a, b]` interval (mirroring the Anno sibling), instead of a later opaque `KeyError`. Raw-JSON surgical edit, mirrors kept byte-identical; behavior-preserving for the committed well-formed `selection.md` (verified by simulation in iteration 1).

## Skipped Issues

None — all 8 findings across both iterations were fixed. No findings were skipped in either pass.

---

_Fixed: 2026-10-03T21:31:12Z (iteration 2)_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
