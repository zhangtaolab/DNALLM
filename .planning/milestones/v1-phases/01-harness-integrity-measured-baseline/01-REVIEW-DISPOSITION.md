---
phase: 01
review: 01-REVIEW.md
titles: json
findings:
  - id: WR-09
    severity: warning
    disposition: open
    title: "README \"Local Testing\" still prescribes the `.[test,dev]` install that CR-03 just proved fails the documented census commands"
  - id: IN-12
    severity: info
    disposition: open
    title: "New ci.yml comment overstates cross-version evidence — \"coverage-nightly proves .[base] resolves green on this exact box\""
  - id: WR-08
    severity: warning
    disposition: open
    title: "The WR-01 fix has no regression test — the guarded path is unreachable from the suite"
  - id: IN-10
    severity: info
    disposition: open
    title: "Dispatch trigger bullet still omits test-mamba — same defect class as the just-fixed IN-02, one line below it"
  - id: IN-11
    severity: info
    disposition: open
    title: "Optional-summary treatment not applied to `plot_radar` in the same module"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "Multilabel branch assumes every per-label curve dict carries AUROC/AUPRC"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "Local-testing comment mislabels the full census as \"what the coverage gate runs\""
  - id: IN-02
    severity: info
    disposition: fixed
    title: "Trigger section omits that the nightly schedule also runs test-mamba"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "`prepare_data` drops `task_type` — multilabel curve data is silently corrupted through the public API"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Workflow README documents gates and tooling that do not exist"
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
total: 21
recorded: 2026-10-01T12:24:07.241Z
---

# Phase 01: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-09 | warning | open | - |
| IN-12 | info | open | - |
| WR-08 | warning | open | - (not in the current review) |
| IN-10 | info | open | - (not in the current review) |
| IN-11 | info | open | - (not in the current review) |
| WR-01 | warning | fixed | 01-REVIEW-FIX.md (not in the current review) |
| IN-01 | info | fixed | 01-REVIEW-FIX.md (not in the current review) |
| IN-02 | info | fixed | 01-REVIEW-FIX.md (not in the current review) |
| WR-02 | warning | fixed | commit 42ada4f (hand-set: fixer re-titled the finding, one-word drift — "data silently" vs "data is silently") (not in the current review) |
| WR-03 | warning | fixed | 01-REVIEW-FIX.md (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| IN-04 | info | open | - (not in the current review) |
| IN-05 | info | open | - (not in the current review) |
| IN-06 | info | open | - (not in the current review) |
| IN-07 | info | open | - (not in the current review) |
| IN-08 | info | open | - (not in the current review) |
| IN-09 | info | open | - (not in the current review) |
| WR-07 | warning | fixed | 01-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-05 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |
| WR-06 | warning | skipped | 01-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
