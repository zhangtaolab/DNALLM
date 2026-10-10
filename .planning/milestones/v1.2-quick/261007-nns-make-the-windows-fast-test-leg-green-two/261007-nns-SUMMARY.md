---
phase: quick-261007-nns
plan: 01
subsystem: ci-windows
tags: [ci, windows, tests, path-separators, skip-allowlist, audit-skips]
requires:
  - Windows fast leg already install-green (quick 261007-mhz pygenometracks platform gate)
  - scripts/audit_skips.py allowlist contract (one matcher key + category per entry)
  - handler-side os.path.join path construction at dnallm/models/special/megadna.py:155
provides:
  - OS-native megadna checkpoint assertion: all 9 TestMegadnaCheckpointSelection cases pass on Windows (backslash) and Linux (slash) alike
  - allowlisted Linux-only fd-accounting skip (reason_like "fd accounting needs /proc"), so the test-windows skip-audit step exits 0 once fast tests are green
affects:
  - .github/workflows/ci.yml test-windows leg (py3.12) — fast-tests step expected 1930 passed / 0 failed (1921 + 9 repaired) and the subsequent audit step exit 0 on the next phs push; baseline failing run 37593859969
key-files:
  created: []
  modified:
    - tests/models/test_special/test_megadna.py
    - tests/expected_skips.yaml
decisions:
  - Test-side fix only: the handler's backslash torch.load path on Windows is valid local-file IO — the library (dnallm/models/special/megadna.py) is byte-identical; the 9 CI failures were the TEST hardcoding the posix separator
  - Allowlist entry (planner-delegated choice) over rewording the skipif to the typed `environment-unavailable:` prefix — reason_like/category-environment precedent is the in-file SONAME twin (tests/utils/test_cuda_compat.py:26), the typed prefix stays reserved for Phase-8 evidence-backed skip helpers (D-06), and the diff is one data entry with zero test-behavior change
metrics:
  duration: 3 min
  completed: 2026-10-07
status: complete
actuals:
  tokens: 411
  tasks: 2
  commits: 2
plan_head_before: 009c863
plan_head_after: bc426bc
---

# Quick Task 261007-nns: Make the Windows Fast-Test Leg Green (two-file fix) Summary

Made the test-windows fast leg green with a two-file, two-commit change: the megadna
checkpoint-selection assertion now builds its expected path with `os.path.join` (OS-native,
matching the handler's own construction), and the Linux-only fd-accounting skip gained a
documented `reason_like` allowlist entry so the skip-audit step passes on non-Linux legs.

## What Was Done

### Task 1: OS-native megadna checkpoint assertion (commit f3046c2)

- `tests/models/test_special/test_megadna.py`: added `import os` (stdlib group) and replaced
  the hardcoded posix f-string equality with
  `assert loaded_paths == [os.path.join("/snapshot", expected_checkpoint)]`, behind a
  three-line separator-is-an-OS-detail comment (handler joins with `os.path.join`;
  backslash on Windows).
- Both sides of the equality now construct the path the same OS-native way, mirroring
  `dnallm/models/special/megadna.py:155` — identical string on Linux (all 20 file tests
  stay green), correct backslash form on Windows.
- Library file untouched (verified: `git diff` across both commits shows only the two
  test-tree files).

### Task 2: Allowlist the Linux-only fd-accounting skip (commit bc426bc)

- `tests/expected_skips.yaml`: appended `reason_like: "fd accounting needs /proc"` /
  `category: environment` after the SONAME entry, with a two-line comment citing
  `tests/utils/test_genomic_coords.py:261` and the test-windows fast leg, mirroring house
  style. `reason_like` (substring) because junit decorates skipif reasons.
- `tests/utils/test_genomic_coords.py` untouched — the test still runs and passes on
  Linux; it skips only where `/proc/self/fd` is absent.

## Verification Evidence

| Check | Result |
|-------|--------|
| `.venv/bin/python -m pytest tests/models/test_special/test_megadna.py -q` | 20 passed in 3.92s |
| ntpath/posixpath proof | ntpath form `/snapshot\megaDNA_phage_145M.pt` (backslash-joined), posixpath form `/snapshot/megaDNA_phage_145M.pt` |
| `grep -qF 'f"/snapshot/{'` on the test file | no match (hardcoded posix literal gone; remaining "snapshot" hits: docstring prose, `("/snapshot", None)` fixture, os.path.join assert) |
| ruff format --check + ruff check on the test file | both clean |
| `.venv/bin/python -m pytest tests/scripts/test_audit_skips.py -q` | 19 passed in 0.39s |
| fd-accounting test on Linux | 1 passed (runs, not skipped) |
| exactly-one-match proof | single entry `{'reason_like': 'fd accounting needs /proc', 'category': 'environment'}` matches the skip reason |
| synthetic Windows junit (one skipped testcase, bare reason) through `scripts/audit_skips.py` `main()` | exit 0 — "OK: every skip ... matches the allowlist" |
| scope checks | `git diff --name-only` per task shows exactly the one planned file; no deletions in either commit; `dnallm/`, `tests/utils/test_genomic_coords.py`, `.github/` byte-identical |

Remote proof (no local Windows box): next push to phs runs test-windows — expected
fast-tests step 1930 passed / 0 failed and audit step exit 0; baseline failing run
37593859969 (9 failed / 1921 passed, all posix-vs-nt path equality).

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Mitigations Applied

- **T-261007-nns-01 (allowlist breadth):** narrowest practical matcher (`reason_like` on the
  specific substring, tied to one static skipif); load_allowlist structurally validated the
  edited file; exactly-one-match proof confirms the gate widened by precisely one
  deterministic skip.
- **T-261007-nns-02 (assertion platform assumption):** both equality sides now OS-native;
  full 20-test file green pins checkpoint-selection semantics (WR-02) unchanged.

## Self-Check: PASSED

- Files verified modified and committed: `tests/models/test_special/test_megadna.py`
  (f3046c2), `tests/expected_skips.yaml` (bc426bc) — both ancestor-of-HEAD.
- Library and non-scope files verified untouched.
