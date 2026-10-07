---
phase: 05
review: 05-REVIEW.md
titles: json
findings:
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "Model-id swap incomplete — Prerequisites still pulls `qwen3.6:latest`"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "ACTIVE-lane sandbox fixture ignores the spec `yaml_patch` key the gated lane forwards"
  - id: IN-01
    severity: info
    disposition: fixed
    title: "`yaml_overrides` on a non-mapping section raises `AttributeError`, not the documented `ValueError`"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "Prerequisite probes let `subprocess.TimeoutExpired` escape instead of reporting `(False, evidence)`"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Unused `noqa` in spike runner fails `ruff check .` — CI lint gate is red on every push"
  - id: IN-03
    severity: info
    disposition: open
    title: "langchain notebook ensure-cell spawns a detached MCP server that is never shut down"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "New test module imports `nbclient` at module scope, but `nbclient` lives only in the `notebook` extra"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "`|| true` masks the spike runner's exit code, hiding infrastructure crashes as a green step"
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
open: 1
total: 15
recorded: 2026-10-06T15:07:37.134Z
---

# Phase 05: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| WR-01 | warning | fixed | 05-REVIEW-FIX.md |
| WR-02 | warning | fixed | 05-REVIEW-FIX.md |
| IN-01 | info | fixed | 05-REVIEW-FIX.md |
| IN-02 | info | fixed | 05-REVIEW-FIX.md |
| CR-01 | critical | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| IN-03 | info | open | - (not in the current review) |
| WR-03 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-04 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-05 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-06 | warning | fixed | 05-REVIEW-FIX.iter2.md (not in the current review) |
| WR-07 | warning | fixed | a5f38ac fix(05) iter2 (not in the current review) |
| WR-08 | warning | fixed | 8fe451b fix(05) iter2 (not in the current review) |
| WR-09 | warning | fixed | bb57709 orchestrator post-loop fix (reviewer's verbatim replacement text) (not in the current review) |
| IN-04 | info | deferred | known-deferred: spike-only /tmp path, throwaway evidence context (not in the current review) |
| IN-05 | info | deferred | known-deferred: inert comments; revisit if preview config selects those codes (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently.
