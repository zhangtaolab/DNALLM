---
phase: 03
review: 03-REVIEW.md
titles: json
findings:
  - id: WR-05
    severity: warning
    disposition: open
    title: "The multilabel AUROC/AUPRC guard added in this delta has no test — the guarded (absent-summary) path is never exercised"
  - id: IN-06
    severity: info
    disposition: open
    title: "models.lock keeps a dead entry attributed to the swapped-out open_chromatin config"
  - id: IN-07
    severity: info
    disposition: open
    title: "cuda_compat test indexes `_LIB_PATTERNS[sys.platform]` directly — KeyErrors on platforms the project supports"
  - id: IN-08
    severity: info
    disposition: open
    title: "coverage-nightly timeout notes still argue from the hosted-runner 360-min cap the job no longer runs under"
  - id: IN-01
    severity: info
    disposition: open
    title: "Commented-out assertions leave two legacy plot tests as smoke tests"
  - id: IN-02
    severity: info
    disposition: open
    title: "Placeholder/vacuous assertion in new benchmark test"
  - id: IN-04
    severity: info
    disposition: open
    title: "Direct `from conftest import SimpleDNATokenizer` imports"
  - id: IN-05
    severity: info
    disposition: open
    title: "Dead code in `global_cleanup` fixture"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "`Benchmark.run_without_config()` crashes on its default `k_folds=1`"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "StratifiedKFold fallback passed `y=None` — still raised"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "`test_prepare_data_empty_metrics` was smoke-only with a wrong premise"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "`test_tokenizer_max_length_respected` asserted nothing about the cap"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "New `tempfile.mkdtemp()` calls leaked temp directories"
open: 8
total: 13
recorded: 2026-10-01T11:06:16.311Z
---

# Phase 03: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-05 | warning | open | - |
| IN-06 | info | open | - |
| IN-07 | info | open | - |
| IN-08 | info | open | - |
| IN-01 | info | open | - (not in the current review) |
| IN-02 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| CR-01 | critical | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-01 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-02 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
