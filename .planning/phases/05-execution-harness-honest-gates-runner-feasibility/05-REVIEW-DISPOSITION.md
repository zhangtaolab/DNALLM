---
phase: 05
review: 05-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Single-flight inference lock releases on tool timeout while the orphaned infer_seqs thread keeps running"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "dna_interpret runs blocking captum work on the event-loop thread — its timeout wrapper can never fire"
  - id: IN-01
    severity: info
    disposition: open
    title: "Three new patch installers omit the try/except transformers-import guard the module contract promises"
  - id: IN-02
    severity: info
    disposition: open
    title: "Port bind-close-probe race in TestProbeHonesty unbound-port test"
  - id: IN-03
    severity: info
    disposition: open
    title: "langchain notebook ensure-cell spawns a detached MCP server that is never shut down"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "Gated `lora_finetune.ipynb` runs with outer timeout == cell timeout, violating the strictly-below invariant"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "`test_timeout` (and marimo `flavor`) spec fields are dead data contradicting their documented contract"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "Permanent HTTP 4xx on the rice input URLs converts to an ever-green `network-unavailable` skip"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "GPU-absent path reports a green job with zero evidence produced"
  - id: WR-06
    severity: warning
    disposition: fixed
    title: "`feasibility.yml` omits the least-privilege `permissions:` block the repo convention mandates"
  - id: WR-07
    severity: warning
    disposition: fixed
    title: "Workflows README still documents the GPU-absent fail-safe no-op that WR-05's fix replaced"
  - id: WR-08
    severity: warning
    disposition: fixed
    title: "Dispatch comments claim notebook variant first then D-06 fallback — --fallback replaces, not sequences"
  - id: WR-09
    severity: warning
    disposition: fixed
    title: "Two residual clauses in the run-step comment contradict the corrected fallback semantics and the committed evidence"
  - id: IN-04
    severity: info
    disposition: deferred
    title: "Pinned megaDNA clone uses a fixed shared /tmp path (spike-only, throwaway evidence)"
  - id: IN-05
    severity: info
    disposition: deferred
    title: "# ruff: ignore[rule-name] comments are inert — not a ruff directive (harmless under preview config)"
open: 3
total: 15
recorded: 2026-10-03T03:50:22.056Z
---

# Phase 05: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | 032b308 fix(quick-261003-hhj) |
| WR-01 | warning | fixed | 3fe80bf fix(quick-261003-ij4) |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |
| WR-02 | warning | fixed | 05-REVIEW-FIX.md (not in the current review) |
| WR-03 | warning | fixed | 05-REVIEW-FIX.md (not in the current review) |
| WR-04 | warning | fixed | 05-REVIEW-FIX.md (not in the current review) |
| WR-05 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-06 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-07 | warning | fixed | a5f38ac fix(05) iter2 (not in the current review) |
| WR-08 | warning | fixed | 8fe451b fix(05) iter2 (not in the current review) |
| WR-09 | warning | fixed | bb57709 orchestrator post-loop fix (reviewer's verbatim replacement text) (not in the current review) |
| IN-04 | info | deferred | known-deferred: spike-only /tmp path, throwaway evidence context (not in the current review) |
| IN-05 | info | deferred | known-deferred: inert comments; revisit if preview config selects those codes (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
