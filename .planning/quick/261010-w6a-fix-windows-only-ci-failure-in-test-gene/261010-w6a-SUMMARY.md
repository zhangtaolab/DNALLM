---
phase: 261010-w6a
plan: 01
subsystem: tests/inference
tags: [ci, windows, test-repair, repr, generate_dataset]
requires:
  - "vo6 fix (67953b6): generate_dataset raises ValueError with repr-embedded input for missing path-shaped strings"
provides:
  - "Platform-independent message assertion for test_generate_dataset_missing_path_raises (all four missing-path shapes, including Windows backslash literal)"
  - "Local Linux red proof reproducing CI run 38061733263 SUBFAILED shape; guard against regression of the repr-embedding contract"
affects: []
tech-stack:
  added: []
  patterns:
    - "assert repr(missing) in str(excinfo.value) — assert against the exact embedded form when the message uses f-string !r"
key-files:
  created: []
  modified:
    - tests/inference/test_inference.py
decisions:
  - "Test-only repair: assertion made repr-based instead of weakening the product error message — the repr embedding is the established convention (generate_dataset f-string !r, same as generate())"
  - "Windows behavior pinned locally via a backslash drive-letter literal rather than platform-skipped: the literal is path-like and missing on every OS, so the repr-doubling mismatch is exercised on Linux CI too"
metrics:
  duration: 2 min
  completed: 2026-10-10
  commits: 2
status: complete
actuals:
  tokens: 358
  tasks: 2
  commits: 2
plan_head_before: c1ba4d9
plan_head_after: 3b99359
---

# Quick Task 261010-w6a: Windows-only CI failure in test_generate_dataset_missing_path_raises Summary

Repaired the Windows-only CI failure by making the missing-path message assertion repr-based (platform-independent) and adding a Windows-style backslash literal that pins the repr-embedding behavior on every platform — test-only, no product change.

## What Was Done

**Failure mechanics (why only Windows was red):** `generate_dataset` embeds the offending input via f-string `!r` (`dnallm/inference/inference.py:484`), which is `repr(x)` for str. On Windows, repr doubles backslashes, so the raw single-backslash tmp path never appears as a substring of the message — the subTest reported SUBFAILED in CI run 38061733263 (job test-windows, py3.12). On Linux tmp paths carry no backslash and repr adds only quotes, so the raw substring was present and only the Windows job failed on 67953b6.

**Task 1 — RED (commit 09a6d39):** Added a fourth `missing_paths` entry, the raw literal `r"C:\no_such_dir\x.tsv"`, with the assertion line left exactly as-is. The literal is path-like on Linux too (backslash counts as a separator in `_is_path_like_string`; `os.path.isfile` is False), so `generate_dataset` raised with the doubled-backslash repr message and the raw-substring assertion failed locally — reproducing the exact CI SUBFAILED shape: `SUBFAILED(missing='C:\\no_such_dir\\x.tsv')` with the message carrying `C:\\\\no_such_dir\\\\x.tsv`. The three original subTests (bare filename, relative path, absolute tmp path) stayed green throughout, as constrained.

**Task 2 — GREEN (commit 3b99359):** Changed the assertion to `assert repr(missing) in str(excinfo.value)` — an f-string `!r` conversion is exactly `repr(x)` for str, so the predicate holds on every platform for every shape (repr adds only quotes for the plain-platform cases; both sides carry the doubled form for the backslash literal). Updated the preceding comment to state why; the `pytest.raises` match regex and all other assertions are unchanged. `dnallm/inference/inference.py` untouched — the vo6 fix (67953b6) and its repr error-message style stay exactly as shipped.

## Verification Evidence

- **Red proof (Task 1):** `.venv/bin/python -m pytest "tests/inference/test_inference.py::TestDNAInference::test_generate_dataset_missing_path_raises" -q` → `1 failed, 1 passed` — the backslash-literal subTest SUBFAILED with the substring mismatch (CI-identical), original three subTests green.
- **Green proof (Task 2):** `.venv/bin/python -m pytest tests/inference/test_inference.py -q` → **123 passed**.
- **Ruff:** `ruff format --check tests/inference/test_inference.py` → `1 file already formatted`; `ruff check` → `All checks passed!`.
- **Scope gate:** `git diff --name-only HEAD~2..HEAD` → exactly `tests/inference/test_inference.py` (plan's verify command passes verbatim).
- **Push:** `git push` → `c1ba4d9..3b99359 dev -> dev`.
- **Windows proof (hand-off, not local):** the true Windows green proof is the CI job test-windows (py3.12) on the pushed run — watched by the orchestrator. Local runs prove the repr predicate and the backslash-literal red-to-green flip; they do not execute Windows path semantics beyond the literal.

## Deviations from Plan

None — plan executed exactly as written. Both red/green gates honored as specified, including the Task 1 local red subTest against the current assertion style before the Task 2 green commit.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 (RED) | 09a6d39 | test(quick-261010-w6a): reproduce Windows repr/backslash CI failure locally |
| 2 (GREEN) | 3b99359 | test(quick-261010-w6a): assert repr-embedded path so the message check is platform-independent |

Both commits pushed to `dev`; no attribution trailers.

## Self-Check: PASSED

- File check: `tests/inference/test_inference.py` — FOUND (modified, contains the four-entry loop and repr assertion at lines 219-239)
- Commit checks: `09a6d39` and `3b99359` — both FOUND as ancestors of HEAD; combined diff is exactly the test file
