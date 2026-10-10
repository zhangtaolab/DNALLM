---
phase: 261010-vo6
plan: 01
type: quick
subsystem: inference
tags: [inference, generate_dataset, input-validation, tdd, bugfix]
requires: []
provides:
  - "generate_dataset fails closed on path-shaped strings that do not exist (ValueError naming the input)"
affects:
  - dnallm/inference/inference.py
  - tests/inference/test_inference.py
tech-stack:
  added: []
  patterns:
    - "path-like discriminator (_is_path_like_string) at the str-input trust boundary"
key-files:
  created: []
  modified:
    - dnallm/inference/inference.py
    - tests/inference/test_inference.py
decisions:
  - "Discriminator = separator presence (/ or backslash) OR known data-file extension; separators are unambiguous because the IUPAC nucleic alphabet has neither"
  - "No expanduser, no changes to load_local_data or extension dispatch — missing '~' paths improve from silent garbage to loud error, which is sufficient (plan fact 9)"
metrics:
  duration: 21 min
  completed: 2026-10-10
  tasks: 2
  commits: 2
status: complete
actuals:
  tokens: 6600
  tasks: 2
  commits: 2
---

# Quick Task 261010-vo6: Fix latent bug — DNAInference.generate_dataset path-shaped string silently becomes a one-row sequence dataset

## Summary

`DNAInference.generate_dataset` previously fell back to `sequences = [seq_or_path]` for any
str input failing `os.path.isfile()`, so a path-shaped string pointing at a missing file
silently produced a one-row dataset with only a `sequence` column — the root cause of census
05-06's malformed benchmark dataset (`tests/examples/test_notebook_execution.py:98`). The
str branch now classifies input via a new module-level `_is_path_like_string()` helper:
path-like strings must resolve to an existing file or raise `ValueError` naming the input;
bare sequence strings, lists, and non-str/non-list inputs behave exactly as before.

## What Was Built

**dnallm/inference/inference.py**
- Module-level `_DATA_FILE_EXTENSIONS: frozenset[str]` — exactly the file types
  `DNADataset._load_single_data` (dnallm/datahandling/data.py) dispatches on, minus `dict`
  (a literal-dict input, not a file extension): csv, tsv, json, parquet, arrow, pkl, pickle,
  fa, fna, fas, fasta, txt. Comment requires it stay in sync with that dispatch.
- Private `_is_path_like_string(value: str) -> bool` — True iff the value contains `/` or
  `\`, or its `os.path.splitext` suffix (lowercased, leading dot stripped) is in
  `_DATA_FILE_EXTENSIONS`. Google-style docstring documents the IUPAC-alphabet rationale.
- `generate_dataset` str branch restructured: path-like → `os.path.isfile` required, else
  `ValueError(f"Input {seq_or_path!r} looks like a file path but no such file exists. ...")`
  (style precedent: the `generate()` error at inference.py ~2020); not-path-like →
  `sequences = [seq_or_path]` unchanged. Vacuous `suffix = seq_or_path.split(".")[-1]`
  line deleted. List branch and final else byte-identical. Docstring `Raises` updated.

**tests/inference/test_inference.py** (class `TestDNAInference`, siblings of
`test_generate_dataset_from_file`)
- `test_generate_dataset_missing_path_raises` — subTest over three missing shapes
  (bare `missing_data.csv`, relative `data/missing_seqs.fa`, absolute
  `os.path.join(tempfile.gettempdir(), "vo6_no_such", "x.tsv")` built in-test): expects
  `ValueError` matching `looks like a file path but no such file exists` with the offending
  string in the message.
- `test_generate_dataset_single_sequence_string` — `"ATGGCCTA"` yields a length-1
  `DNADataset` with `dataset["sequence"] == ["ATGGCCTA"]` plus a DataLoader (guard against
  bare-sequence regression).
- `test_is_path_like_string_truth_table` — subTest loops: 12 path-like cases (separators,
  Windows backslash, every supported extension) return True; 7 sequence-like cases
  (`ATCG`, `ATGGCCTA`, `acgtn`, full IUPAC `ACGTRYSWKMBDHVN`, gapped `ACGT-N`, empty
  string, unknown-dot `ATC.G`) return False.

## TDD Evidence

- **RED (task 1, commit 6e82437):** `test_generate_dataset_missing_path_raises` failed on
  all three subtests with `Failed: DID NOT RAISE ValueError` against pre-fix code (the
  silent one-row fallback); `test_generate_dataset_single_sequence_string` passed pre-fix
  (guard green before and after).
- **GREEN (task 2, commit 67953b6):** all three new node IDs pass; full targeted lane
  123 passed (120 pre-existing + 3 new) in ~8s, including `test_single_sequence_string`,
  `test_generate_dataset_from_list`, `test_generate_dataset_from_file`, and the end-to-end
  CSV inference tests.

## Verification

- `.venv/bin/python -m pytest tests/inference/test_inference.py -q` → **123 passed**
- `.venv/bin/ruff format --check` on both touched files → clean
- `.venv/bin/ruff check` on both touched files → All checks passed
- Targeted lane only — no repo-wide runs, no coverage runs (owner directive 2026-10-09)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Truth-table assertion style corrected to satisfy project lint**
- **Found during:** Task 2
- **Issue:** The plan specified `assertTrue/assertFalse(..., message=s)` inside
  `TestDNAInference`. `unittest.TestCase.assertTrue` has no `message=` kwarg (it is
  `msg=`), and the project's ruff `PT` ruleset rejects unittest-style assertions in tests
  outright (`pytest-unittest-assertion`).
- **Fix:** Used plain `assert _is_path_like_string(s), s` / `assert not ...` inside the
  subTest loops — idiomatic for the project, lint-clean, and subTest headers already name
  the failing value.
- **Files modified:** tests/inference/test_inference.py
- **Commit:** 67953b6

No other deviations — plan executed as written (scope guards honored: no expanduser, no
load_local_data changes, nothing outside the str branch).

## Commits

- 6e82437 `test(quick-261010-vo6): red-first regression tests for generate_dataset path-shaped strings`
- 67953b6 `fix(quick-261010-vo6): generate_dataset raises on missing path-shaped strings instead of silent single-sequence fallback`

Both pushed to `origin/dev`. No attribution trailers (repo convention).

## Known Stubs

None.

## Threat Flags

T-vo6-01 mitigated as planned: crafted path-shaped strings at the user-input trust
boundary now fail closed with a descriptive `ValueError` instead of flowing into the
tokenization pipeline as a bogus sequence.

## Self-Check: PASSED

Files present: dnallm/inference/inference.py, tests/inference/test_inference.py,
261010-vo6-SUMMARY.md. Commits 6e82437 and 67953b6 are ancestors of HEAD and pushed to
origin/dev.

