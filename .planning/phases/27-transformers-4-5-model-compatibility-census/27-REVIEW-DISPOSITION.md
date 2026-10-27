---
phase: 27
review: 27-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "`--run --verify` can never exit 0 — the dispatch lane is permanently red and cannot act as a drift gate"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "`merge_records` silently drops `disposition` and `previous_status` — a default-artifact re-run destroys the frozen 27-02 judgments"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "Loading the census script in tests sets `HF_ENDPOINT=https://hf-mirror.com` process-wide at collection time"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Remote-code probe subprocesses inherit the full runner environment, including `GITHUB_TOKEN`, while executing untrusted repo code"
  - id: IN-01
    severity: info
    disposition: open
    title: "New rung code adds mypy errors under a `ty:`-only ignore convention"
  - id: IN-02
    severity: info
    disposition: open
    title: "Independent artifact-vs-registry checker omits the `fix`-disposition mapping branch"
  - id: IN-03
    severity: info
    disposition: open
    title: "Machine-specific evidence text committed into the packaged registry"
  - id: IN-04
    severity: info
    disposition: open
    title: "`_finalize_model_loading` wrapper skips materialization on a kwargs-only call"
  - id: IN-05
    severity: info
    disposition: open
    title: "Unpinned `curl | sh` for uv on the self-hosted box"
  - id: IN-06
    severity: info
    disposition: open
    title: "`docs_blocked_reason` fails open to the raw first clause for unforeseen blocked shapes"
open: 6
total: 10
recorded: 2026-10-27T10:39:40.435Z
---

# Phase 27: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | fix commit 16363d4 (hand-reconciled: REVIEW-FIX.md re-titled the finding; offline seeded-run regression pins exit 0; live dispatch verification still owed) |
| WR-01 | warning | fixed | fix commit 61d128c (hand-reconciled: REVIEW-FIX.md re-titled the finding; carry runs independent of load carry) |
| WR-02 | warning | fixed | fix commit 10f7ad9 (hand-reconciled: REVIEW-FIX.md re-titled the finding; env snapshot/restore around exec_module) |
| WR-03 | warning | fixed | fix commit 6187404 (hand-reconciled: REVIEW-FIX.md re-titled the finding; sanitized_probe_env blocklist GITHUB_/RUNNER_/ACTIONS_) |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| IN-04 | info | open | - |
| IN-05 | info | open | - |
| IN-06 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
