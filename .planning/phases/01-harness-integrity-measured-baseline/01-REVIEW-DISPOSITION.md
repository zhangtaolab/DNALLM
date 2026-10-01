---
phase: 01
review: 01-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: open
    title: "Nightly `test-mamba` job reports green while its tests fail (`continue-on-error`)"
  - id: WR-02
    severity: warning
    disposition: open
    title: "`prepare_data` drops `task_type` — multilabel curve data is silently corrupted through the public API"
  - id: WR-03
    severity: warning
    disposition: open
    title: "Workflow README documents gates and tooling that do not exist"
  - id: IN-01
    severity: info
    disposition: open
    title: "Broken-and-unused `mock_dataset` fixture and no-op `global_cleanup` fixture in conftest"
  - id: IN-02
    severity: info
    disposition: open
    title: "Stale `models.lock` entry after the open-chromatin model swap"
  - id: IN-03
    severity: info
    disposition: open
    title: "Open-chromatin config keeps promoter labels/description after the model swap"
  - id: IN-04
    severity: info
    disposition: open
    title: "Vacuous isinstance assertions via `__class__` swap in benchmark tests"
  - id: IN-05
    severity: info
    disposition: open
    title: "Misleading test names/docstrings pinning non-behavior"
  - id: IN-06
    severity: info
    disposition: open
    title: "`from conftest import ...` relies on pytest's sys.modules side effect"
  - id: IN-07
    severity: info
    disposition: open
    title: "Deprecated `actions/cache@v3` in the deploy job"
  - id: IN-08
    severity: info
    disposition: open
    title: "Code-based `Benchmark.__init__` aliases one task config across all datasets"
  - id: IN-09
    severity: info
    disposition: open
    title: "`metrics_for_dnabert2` regression arm returns a nested `r2` dict — and the new test cements it"
  - id: WR-07
    severity: warning
    disposition: fixed
    title: "`ci_checks.sh` installs uv but never adds it to PATH — auto-setup aborts on fresh hosts"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "Docs still document the `pytest.ini` this phase deleted"
  - id: WR-05
    severity: warning
    disposition: skipped
    title: "`.github/workflows/README.md` remains materially wrong after this phase's edit"
  - id: WR-06
    severity: warning
    disposition: skipped
    title: "Unpinned `curl | sh` installer executed in four CI jobs and the local script"
open: 12
total: 16
recorded: 2026-10-01T09:27:05.766Z
---

# Phase 01: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |
| IN-07 | info | open | - |
| IN-08 | info | open | - |
| IN-09 | info | open | - |
| WR-07 | warning | fixed | 01-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-05 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-06 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
