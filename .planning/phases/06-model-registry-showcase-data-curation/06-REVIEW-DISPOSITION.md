---
phase: 06
review: 06-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "`fla` extra is installed by no CI leg — nightly smoke tests run on the silent non-KDA fallback path"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "_load_with_fallback converts any exception — including dnallm code regressions — into a green skip"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "slice_gff_rows strips only \\n — CRLF input silently corrupts column-9 values"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "pyfastx missing from the mypy overrides list — new import-not-found error introduced"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "tomllib (Python 3.11+) breaks test collection on Python 3.10, still a declared supported version"
  - id: IN-01
    severity: info
    disposition: open
    title: "fetch_sequence(path) leaks the pyfastx handle and creates a .fxi index beside the user's FASTA"
  - id: IN-02
    severity: info
    disposition: open
    title: "normalize_chrom accepts non-ASCII digit strings and silently renames them"
  - id: IN-03
    severity: info
    disposition: open
    title: "test_module_import_is_pyfastx_free leaves the package attribute pointing at a re-executed module"
  - id: IN-04
    severity: info
    disposition: open
    title: "Anno label_names uses single quotes, rest of the registry uses double quotes"
  - id: IN-05
    severity: info
    disposition: open
    title: "test_fla_reachable_from_all matches the extra by substring"
  - id: IN-06
    severity: info
    disposition: open
    title: "Local .scratch/ ignore is redundant with the root pattern"
open: 6
total: 11
recorded: 2026-10-03T18:40:00Z
---

# Phase 06: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | eb85f7e fix(quick-261003-r73): both nightly legs (coverage-nightly + mamba nightly) install .[base,fla]; both slow smokes carry the typed environment-unavailable: importorskip guard |
| WR-02 | warning | fixed | 1219f0f test(quick-261003-r73): _is_environment_error classifier — env-class failures keep the byte-identical typed skip, dnallm regressions propagate and fail; 7 fast regression tests |
| WR-03 | warning | fixed | 8d6bd3b fix(quick-261003-r73): slice_gff_rows strips \n/\r\n/\r terminators and raises ValueError on embedded \r; 2 same-change tests |
| WR-04 | warning | fixed | 5d354c9 (pyproject mypy overrides + pyfaidx precedent; full mypy run still blocked by pre-existing numpy-stubs abort, CI-advisory) |
| WR-05 | warning | fixed | 5d354c9 (tomllib guarded for 3.10; the two pyproject-declaration tests carry typed environment-unavailable skipif; ruff clean, 3 tests pass) |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.

Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.

Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently.
