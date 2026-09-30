---
phase: 03
review: 03-REVIEW.md
titles: json
findings:
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
open: 4
total: 9
unparsed: 1
recorded: 2026-09-30T13:55:07.454Z
---

# Phase 03: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| CR-01 | critical | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-01 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-02 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 03-REVIEW-FIX.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
