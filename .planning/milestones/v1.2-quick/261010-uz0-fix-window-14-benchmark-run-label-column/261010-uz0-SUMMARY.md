---
phase: 261010-uz0
plan: 01
type: execute
subsystem: inference
tags: [benchmark, label-column, windows-ledger, regression-tests]
status: complete
started: 2026-10-10T14:35:58Z
completed: 2026-10-10T14:41:30Z
duration_min: 4
commits:
  - a8e05d5
  - 4db85d0
files:
  created: []
  modified:
    - dnallm/inference/benchmark.py
    - tests/benchmark/test_benchmark.py
    - .planning/WINDOWS.md
windows_closed: [14]
---

# Quick Task 261010-uz0: Benchmark.run() label-column resolution (WINDOWS 14)

One-liner: run() now reads labels through `_extract_labels` (configured
label_column -> 'labels' fallback -> descriptive ValueError naming dataset,
configured column, and available columns) instead of the hardcoded
`self.datasets[di]["labels"]` subscript that crashed with KeyError before any
model loaded.

## What Was Done

**Task 1 — red-first regression tests (commit a8e05d5)**

- `benchmark_yaml_factory` gained a `label_column="labels"` keyword (default
  keeps every existing call site byte-identical in behavior).
- New `TestRunLabelColumnResolution` class (4 tests) inserted after
  `TestRunBranches`, using the established three-patch pattern plus a
  `calculate_metrics` side effect recording its labels argument:
  1. `test_run_reads_configured_label_column` — un-normalized `label` column
     shape; asserts metrics + recorded labels `[0, 1, 0, 1]`.
  2. `test_run_falls_back_to_labels_when_configured_column_absent` — the
     loader-normalized production shape (guard, green before and after).
  3. `test_run_default_labels_path_still_works` — config-less path with
     `get_dataset` default `label_col='labels'` (guard, green before and
     after).
  4. `test_run_without_label_column_raises_value_error` — census
     `['sequence']`-only shape must raise `ValueError` matching
     "has no label column" with the dataset name in the message; no
     model-loader patch on purpose (error must fire before any model loads).

**Red matrix observed before the fix commit (required evidence):**
exactly **2 failed / 2 passed** — Tests 1 and 4 failed with
`KeyError: "Column labels not in the dataset. Current columns in the dataset:
['sequence', 'label']"` (resp. `['sequence']`), Tests 2 and 3 passed.

**Task 2 — fix + ledger closure (commit 4db85d0)**

- `dnallm/inference/benchmark.py`: annotation correction
  `self.datasets: list[str] -> list[Any]` (line 84; `Any` already imported);
  `run()` label read replaced by `labels = self._extract_labels(di, dname)`;
  new private `_extract_labels(di, dataset_name)` placed after `get_dataset`,
  implementing exactly the resolution order configured-column -> `'labels'` ->
  `ValueError`. Nothing else in `run()` or any other method changed;
  `dnallm/datahandling/data.py` and `generate_dataset` untouched (verified via
  `git show --stat`: fix commit touches only benchmark.py and WINDOWS.md).
- `.planning/WINDOWS.md` id 14 closed via `gsd-tools windows fixed 14` (status,
  `resolved_at=2026-10-10T14:39:32.164Z`, counts open 8->7 / fixed 10->11,
  `last_updated` bumped), then the required reason text patched into BOTH the
  table row and the mirrored JSON object, including the out-of-scope note about
  the census's `['sequence']`-only shape stemming from `generate_dataset`'s
  silent path-to-sequence fallback. Ledger re-verified parseable
  (`windows status` reads 7 open / 3 waived / 11 fixed).

## Verification

- `.venv/bin/python -m pytest tests/benchmark/test_benchmark.py -q` —
  **35 passed** (31 pre-existing + 4 new), ~7s.
- `.venv/bin/ruff format --check` and `.venv/bin/ruff check` clean on
  `dnallm/inference/benchmark.py` and `tests/benchmark/test_benchmark.py`.
- Targeted lanes only, per owner directive 2026-10-09 (no repo-wide runs, no
  coverage runs).
- Both commits pushed to `origin/dev` (`e755116..4db85d0`), plain messages,
  no attribution trailers (grep-verified).

## Deviations from Plan

None — plan executed exactly as written. One mechanical adjustment inside
Task 2's sanctioned envelope: the ledger was closed with
`gsd-tools windows fixed 14` (available via
`node ~/.claude/gsd-core/bin/gsd-tools.cjs`; the bare `gsd-tools` name is not
on PATH) followed by direct reason-text edits in both places, which is the
plan's stated fallback path fused with its preferred path.

## Flag for the Owner (candidate new ledger item)

`DNAInference.generate_dataset` silently treats a non-file path string as a
single sequence (`dnallm/inference/inference.py:439-456`,
`sequences = [seq_or_path]` fallback). This is the mechanism behind the
census's `['sequence']`-only dataset that originally surfaced WINDOWS id 14 —
a wrong `path` in a benchmark config now produces this run()'s descriptive
`ValueError` (has no label column) instead of a cryptic KeyError, but the
silent fallback itself remains unfixed by design (owner rule: fixes limited to
what correctness requires). Candidate for a new windows-ledger entry.

## Self-Check: PASSED

- Files: dnallm/inference/benchmark.py (modified, `_extract_labels` present,
  hardcoded subscript gone), tests/benchmark/test_benchmark.py (modified,
  `TestRunLabelColumnResolution` present) — FOUND.
- Commits: a8e05d5 and 4db85d0 are ancestors of origin/dev HEAD — FOUND.
- WINDOWS.md: id 14 `fixed` in table and JSON, frontmatter 7 open / 11 fixed —
  FOUND.
