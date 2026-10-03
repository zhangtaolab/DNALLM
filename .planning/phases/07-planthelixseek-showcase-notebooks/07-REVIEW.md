---
phase: 07-planthelixseek-showcase-notebooks
reviewed: 2026-10-03T15:43:21Z
depth: standard
files_reviewed: 24
files_reviewed_list:
  - docs/example/notebooks/plant_helixseek_anno/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_anno/data/TAIR10_GFF3_chr1_5100001_5300000.gff3
  - docs/example/notebooks/plant_helixseek_anno.md
  - docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - docs/example/notebooks/plant_helixseek_cre/data/chr1_5100001_5300000.fas
  - docs/example/notebooks/plant_helixseek_cre/data/TAIR10_DHSs_chr1_5100001_5300000.gff
  - docs/example/notebooks/plant_helixseek_cre.md
  - docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_14953292_14973291.fas
  - docs/example/notebooks/plant_helixseek_shared/data/chr1_5351001_5371000.fas
  - docs/example/notebooks/plant_helixseek_shared/data/selection.md
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_14953292_14973291.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_DHSs_chr1_5351001_5371000.gff
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_14953292_14973291.gff3
  - docs/example/notebooks/plant_helixseek_shared/data/TAIR10_GFF3_chr1_5351001_5371000.gff3
  - example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb
  - example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb
  - mkdocs.yml
  - models.lock
  - scripts/check_docs_sync.py
  - tests/examples/_execution.py
  - tests/examples/test_plant_helixseek_showcase.py
findings:
  critical: 0
  warning: 2
  info: 6
  total: 8
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-10-03T15:43:21Z
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Summary

Reviewed the PlantHelixSeek showcase landing (07-01 CRE + 07-02 Anno): two committed executed notebooks and their docs mirrors, the shared frozen selection contract and committed genomic data slices, the wrapper tutorials, the mkdocs nav additions, the models.lock entries, the `check_docs_sync.py` ignore-list addition, the two new `NOTEBOOK_EXEC_SPECS` budgets in the execution harness, and the new showcase test module (structure tests + slow nightly execution lane).

This review went beyond reading: the exact regex parsing the tests use was simulated against the committed `selection.md` (all four frozen keys and all four band rows parse to the intended values, including the `>= 0.8 | >= 3 genes` first-match behavior); the stream-key extraction was simulated against both committed notebooks (the `^key=` anchors find the freshly computed values in cells 11/16/17/22 and cannot collide with the comparison-table lines of cells 18/24); committed outputs were cross-checked against `selection.md` (94 DHS rows, 526/394/346/48/180, `exon_f1 = 692/920 = 0.7522`, window counts 3991/48/8 match hand-computed tilings); the metric implementations in both notebooks (pooled/per-gene exon F1 with midpoint-scoped FPs, greedy reciprocal-overlap matching, nucleotide masks, minus-strand B<->L permutation vs the registry label order, tail-window stitching) were traced line by line and found correct; coordinate conventions were verified against `dnallm/utils/genomic_coords.py` semantics and the actual data-file contents (FASTA headers, truth-row containment, zero-byte intergenic slices); the docs mirror was byte-compared; and the fast lanes were executed (13 structure tests, 6 `test_examples.py` tests, `check_notebook_md_sync` for the new wrappers, ruff lint/format) — all green. The Anno/CRE budgets satisfy the strict `cell_timeout < pytest-timeout mark` contract (1200<2400, 3600<5400), and the nightly lane's `.[base,fla]` install does provide `pyfastx`/`nbclient`/`vl_convert` transitively (base -> dev/test/notebook; `altair[all]`).

No Critical issues found. Two Warnings (one robustness gap in the exact ignore table this phase edited, one inconsistent parse-guard failure mode in the CRE notebook) and six Info items.

Out-of-scope context, not charged to this phase: `scripts/check_docs_sync.py` and `scripts/check_notebook_md_sync.py` both already fail on this checkout for pre-existing reasons unrelated to the showcase files (gitignored benchmark runtime artifacts; three stale MCP/data-prepare wrappers) — see WR-01 for the sync-script half, which touches the code this phase modified.

## Warnings

### WR-01: docs-sync gate ignores some sanctioned runtime artifacts, fails red on any tree where the benchmark example ran

**File:** `scripts/check_docs_sync.py:8-17`
**Issue:** This phase added `.scratch` to `IGNORE` (correct — the Phase-6 curation tooling lives under `example/notebooks/plant_helixseek_shared/.scratch/`, gitignored). However the ignore set still does not cover the benchmark example's runtime outputs: `benchmark_results/` and `example/notebooks/*/*.pdf` are gitignored (`.gitignore:68,118`) but not in `IGNORE`/`IGNORE_SUFFIXES` (only `.gz`/`.log` suffixes and `logs`/`outputs*`/etc. names are excepted). Verified live on this checkout: `python scripts/check_docs_sync.py` exits 1 with `ONLY in example/: notebooks/benchmark/benchmark_results`, `plot_metrics.pdf`, `plot_roc.pdf`. CI is unaffected (clean checkouts never have these untracked files), but the script's purpose as a local pre-push mirror gate is defeated and the "ONLY in example/" wording misleads — these are gitignored artifacts, not mirror drift. The phase's own artifacts (`logs`, `outputs`, `.scratch`) are correctly covered, so this is a pre-existing gap in the exact table this diff touched rather than a regression.
**Fix:**
```python
IGNORE = {
    "__pycache__",
    "logs",
    "outputs",
    "outputs_multilabel",
    "benchmark_results",   # gitignored runtime output of the benchmark example
    ".ipynb_checkpoints",
    ".gitignore",
    ".scratch",
}
IGNORE_SUFFIXES = (".gz", ".log", ".pdf")  # .pdf: gitignored example/notebooks/*/*.pdf plots
```

### WR-02: CRE notebook band-table parse fails as a bare KeyError instead of the documented parse guard

**File:** `docs/example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` (cell 18; mirrored at `example/notebooks/plant_helixseek_cre/plant_helixseek_cre.ipynb` cell 18)
**Issue:** Cell 18 parses the frozen `selection.md` keys with an explicit `RuntimeError` parse guard, but the band-table loop only populates `bands[metric_label]` when the `[a, b]` interval regex succeeds, and the `break` fires on the first prefix-matching line regardless — so a missing/malformed band row surfaces later as an opaque `KeyError: 'CRE jaccard'` at `low, high = bands[metric_label]`, and a second candidate row would never be consulted. The Anno sibling (cell 24) gets this right (`raise RuntimeError(...)` when `genes_band is None or neg_band is None`). Current committed artifacts parse fine, so this only degrades the nightly diagnostic when `selection.md` changes shape — exactly the scenario the guard exists for.
**Fix:** In the band loop, fail loudly at the parse site, mirroring cell 24:
```python
if line.startswith(f"| {metric_label} |"):
    band_match = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", line)
    if band_match is None:
        raise RuntimeError(
            f"selection.md band row '{metric_label}' carries no '[a, b]' interval (parse guard)"
        )
    bands[metric_label] = (float(band_match.group(1)), float(band_match.group(2)))
    break
```

## Info

### IN-01: models.lock comment still describes the Anno test as future work

**File:** `models.lock:13`
**Issue:** `# ... source=modelscope; Anno test lands with 07-02` — written from 07-01's perspective; 07-02 (this phase) landed `test_anno_notebook_executes_within_selection_bands`, so the future-tense clause is stale.
**Fix:** Drop the clause to match line 12's phrasing: `# tests/examples/test_plant_helixseek_showcase.py (nightly showcase execution, source=modelscope)`.

### IN-02: Harness docstring budget census is off by the two new entries

**File:** `tests/examples/_execution.py:91-92`
**Issue:** The `NOTEBOOK_EXEC_SPECS` doc note says "All 21 example notebooks carry starter budgets (05-05, D-08 census)". This phase appended the CRE and Anno showcase entries, so the dict now holds 23. A reader reconciling the prose with the dict has to rediscover that the showcase lane is deliberately outside the census count.
**Fix:** Update the sentence, e.g. "All 21 census example notebooks carry starter budgets (05-05, D-08); the two Phase-7 showcase-lane entries below are budget-only (their tests live in tests/examples/test_plant_helixseek_showcase.py)."

### IN-03: Anno wrapper claims a per-gene F1 print that the notebook does not emit

**File:** `docs/example/notebooks/plant_helixseek_anno.md:136`
**Issue:** "The notebook also prints per-gene exon F1 (genes = mRNAs with CDS rows in the slice; ...)". Cell 17 computes `gene_f1_scores` but prints only aggregates (`genes_above_floor`, `n_genes`); no per-gene F1 value appears in any stream output of the committed notebook.
**Fix:** Reword to "The notebook also computes per-gene exon F1 and prints the count of genes meeting the 0.8 floor (`genes_above_floor=`)".

### IN-04: Unused `locus_key` parameter in five of seven parametrized structure tests

**File:** `tests/examples/test_plant_helixseek_showcase.py:216,228,261,285,307`
**Issue:** `test_first_code_cell_is_the_fla_guard`, `test_no_import_statement_names_fla`, `test_illustrative_caption_follows_the_metric_figure`, `test_committed_notebook_has_executed_outputs`, and `test_no_genome_wide_claim_phrasing` accept `locus_key` but never read it (only `test_provenance_markdown_cell` does). Presumably a uniform signature for the shared `SHOWCASE_NOTEBOOKS` parametrization; no behavioral risk.
**Fix:** Either drop the parameter from those five signatures (parametrize on `nb_path` only) or keep it and note the uniform-signature intent in a comment.

### IN-05: bedtools dependency of the nightly CRE test is an unencoded runner assumption

**File:** `tests/examples/test_plant_helixseek_showcase.py:323` (test body) / `.github/workflows/ci.yml:468-497` (nightly job)
**Issue:** The CRE slow test re-executes a notebook whose cell 11 runs `subprocess.run(["bedtools", "jaccard", ...], check=True)`. Nothing in the nightly job provisions bedtools; it works because the designated `[self-hosted, dnallm-nightly]` box has bedtools v2.31.1 on PATH (verified on this runner) and the wrapper documents the requirement. Fail-loud (no typed skip) is the module's stated design, so this is acceptable — but if the runner is ever rebuilt the nightly fails with a bare `FileNotFoundError` rather than a diagnosable message.
**Fix:** Optional hardening: add an upfront `shutil.which("bedtools")` assert in the CRE test with a message pointing at the wrapper's prerequisites section (still fail-loud, but self-describing).

### IN-06: Anno label-order usage mixes registry-derived and hardcoded indexes

**File:** `docs/example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb` (cells 11, 13, 22; mirrored at `example/notebooks/plant_helixseek_anno/plant_helixseek_anno.ipynb`)
**Issue:** Cell 13 derives the CDS-family label indexes from the registry (`label_names.index(name)` — drift-safe), while cell 11 hardcodes the 17-element `_B_SWAP_L` permutation against positional indexes and cell 22 hardcodes label `0` as `O` (`genic_mask = (labels != 0)`). All three agree with the current registry order (verified against `dnallm/models/model_info.yaml`), and the permutation is a frozen-contract transcription per `selection.md` — but if the registry order ever changed, cell 13 would silently follow while cells 11/22 would silently break. A one-line assert would pin the coupling.
**Fix:** In cell 11 (or 13), add e.g. `assert label_names[0] == "O" and label_names[1:5] == ["B-CDS", "I-CDS", "L-CDS", "U-CDS"], "registry label order drifted from the frozen permutation"`.

---

_Reviewed: 2026-10-03T15:43:21Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
