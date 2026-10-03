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
    disposition: fixed
    title: "fetch_sequence(path) leaks the pyfastx handle and creates a .fxi index beside the user's FASTA"
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
| IN-01 | info | fixed | aaf6308 fix(quick-261003-ryz): path branch drops the pyfastx reference in a finally and unlinks only a .fxi it created (contextlib.suppress(OSError) so cleanup never masks results); pre-existing sidecars and caller-owned indices untouched; 3 same-change tests |
| IN-02 | info | fixed | c19999a fix(quick-261003-ryz): bare-numeric branch requires name.isascii() and name.isdigit() so non-ASCII digits raise ValueError in both styles; ASCII behavior byte-identical; same-change test |
| IN-03 | info | fixed | 3662ea5 test(quick-261003-ryz): purity test saves/restores the dnallm.utils.genomic_coords package attribute; new identity guard test proves a single live module object |
| IN-04 | info | fixed | 7790920 chore(quick-261003-ryz): Anno label_names re-quoted to double quotes, one-line diff gate green, legacy single-quoted line 1447 untouched; registry structure tests re-run green |
| IN-05 | info | fixed | 2a0ba40 test(quick-261003-ryz): _meta_extra_names exact bracket-member parser replaces the substring match; substring-collision regression tests; tomllib guard untouched |
| IN-06 | info | fixed | this closure commit (quick-261003-ryz): local .scratch/ ignore removed after live masked/unmasked git check-ignore proof that root .gitignore:60 covers both files and the directory (evidence verbatim in the quick-261003-ryz SUMMARY) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.

Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.

Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently.
