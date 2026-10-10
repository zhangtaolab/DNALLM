---
phase: 06
review: 06-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "`_is_environment_error` docstring cites stale `model.py` line numbers for every load-ladder anchor"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "type-based classification still whitelists dnallm-originating `ImportError`/`OSError` as environment-class (green skip)"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "fla `importorskip` guard fires before `_emit_env()`, so a fla-missing typed skip carries no version evidence"
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
  - id: IN-02
    severity: info
    disposition: fixed
    title: "normalize_chrom accepts non-ASCII digit strings and silently renames them"
  - id: IN-03
    severity: info
    disposition: fixed
    title: "test_module_import_is_pyfastx_free leaves the package attribute pointing at a re-executed module"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "Anno label_names uses single quotes, rest of the registry uses double quotes"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "test_fla_reachable_from_all matches the extra by substring"
  - id: IN-06
    severity: info
    disposition: fixed
    title: "Local .scratch/ ignore is redundant with the root pattern"
open: 0
total: 11
recorded: 2026-10-06T16:24:33.827Z
---

# Phase 06: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | 06-REVIEW-FIX.md |
| WR-02 | warning | fixed | 7134aa6 — fix-report entry titles the same finding without the "(green skip)" suffix; hand-reconciled (innermost-frame origin check + 3 new classification pins) |
| IN-01 | info | fixed | 06-REVIEW-FIX.md |
| WR-03 | warning | fixed | 8d6bd3b fix(quick-261003-r73): slice_gff_rows strips \n/\r\n/\r terminators and raises ValueError on embedded \r; 2 same-change tests (not in the current review) |
| WR-04 | warning | fixed | 5d354c9 (pyproject mypy overrides + pyfaidx precedent; full mypy run still blocked by pre-existing numpy-stubs abort, CI-advisory) (not in the current review) |
| WR-05 | warning | fixed | 5d354c9 (tomllib guarded for 3.10; the two pyproject-declaration tests carry typed environment-unavailable skipif; ruff clean, 3 tests pass) (not in the current review) |
| IN-02 | info | fixed | c19999a fix(quick-261003-ryz): bare-numeric branch requires name.isascii() and name.isdigit() so non-ASCII digits raise ValueError in both styles; ASCII behavior byte-identical; same-change test (not in the current review) |
| IN-03 | info | fixed | 3662ea5 test(quick-261003-ryz): purity test saves/restores the dnallm.utils.genomic_coords package attribute; new identity guard test proves a single live module object (not in the current review) |
| IN-04 | info | fixed | 7790920 chore(quick-261003-ryz): Anno label_names re-quoted to double quotes, one-line diff gate green, legacy single-quoted line 1447 untouched; registry structure tests re-run green (not in the current review) |
| IN-05 | info | fixed | 2a0ba40 test(quick-261003-ryz): _meta_extra_names exact bracket-member parser replaces the substring match; substring-collision regression tests; tomllib guard untouched (not in the current review) |
| IN-06 | info | fixed | this closure commit (quick-261003-ryz): local .scratch/ ignore removed after live masked/unmasked git check-ignore proof that root .gitignore:60 covers both files and the directory (evidence verbatim in the quick-261003-ryz SUMMARY) (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
