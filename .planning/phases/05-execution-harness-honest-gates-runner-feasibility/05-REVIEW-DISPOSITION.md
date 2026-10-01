---
phase: 05
review: 05-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Unused noqa (ANN202) makes ruff check . exit 1 repo-wide — CI lint red on every push"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "assert_tree_clean only asserts on stdout with check=False — a failing git status passes silently"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "filecmp.dircmp defaults to shallow=True — byte-identical mirror contract not byte-verified"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Module-scope nbclient import fails collection under bare .[test] install — add nbclient to test extra"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "|| true greens the spike step unconditionally, masking pre-evidence crashes"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "GPU-absent path skips everything and reports a green job with zero evidence"
  - id: WR-06
    severity: warning
    disposition: fixed
    title: "No permissions block in feasibility.yml — inherits default token scopes"
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
  - id: IN-01
    severity: info
    disposition: deferred
    title: "test_timeout / extra_inputs spec keys are dead config (Phase 8 generalization wires them)"
  - id: IN-02
    severity: info
    disposition: deferred
    title: "assert_tree_clean fails on pre-existing developer WIP under example/ (no baseline/delta)"
  - id: IN-03
    severity: info
    disposition: deferred
    title: "Hardcoded developer home path in the mirrored NER dataset generator (mirror-faithful; Phase 8 per D-03)"
  - id: IN-04
    severity: info
    disposition: deferred
    title: "Pinned megaDNA clone uses a fixed shared /tmp path (spike-only, throwaway evidence)"
  - id: IN-05
    severity: info
    disposition: deferred
    title: "# ruff: ignore[rule-name] comments are inert — not a ruff directive (harmless under preview config)"
open: 0
total: 15
recorded: 2026-10-02T04:35:00+08:00
---

# Phase 05: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | fixed | 89b2f63 fix(05) iter1 (not in the current review) |
| WR-01 | warning | fixed | f8e0f9f fix(05) iter1 (not in the current review) |
| WR-02 | warning | fixed | 9876a0b fix(05) iter1 (not in the current review) |
| WR-03 | warning | fixed | 5c30083 fix(05) iter1 (not in the current review) |
| WR-04 | warning | fixed | 9de82cd fix(05) iter1 (not in the current review) |
| WR-05 | warning | fixed | 898d325 fix(05) iter1 (not in the current review) |
| WR-06 | warning | fixed | 824b901 fix(05) iter1 (not in the current review) |
| WR-07 | warning | fixed | a5f38ac fix(05) iter2 (not in the current review) |
| WR-08 | warning | fixed | 8fe451b fix(05) iter2 (not in the current review) |
| WR-09 | warning | fixed | bb57709 orchestrator post-loop fix (reviewer's verbatim replacement text) |
| IN-01 | info | deferred | known-deferred: Phase 8 fixture generalization wires the spec keys |
| IN-02 | info | deferred | known-deferred: baseline/delta pattern lands with Phase 8 rollout |
| IN-03 | info | deferred | known-deferred: mirror-faithful copy; content repair in Phase 8 per D-03 |
| IN-04 | info | deferred | known-deferred: spike-only /tmp path, throwaway evidence context |
| IN-05 | info | deferred | known-deferred: inert comments; revisit if preview config selects those codes |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.

Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.

Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.

Note: this ledger was rendered by the orchestrator following the canonical schema (the embedded renderer's heading parser does not match the iteration-3 review's `### IN-01 (carried, ...):` heading suffix shape, which would have produced an unparsed-shortfall row set). Every disposition above is grounded in the git commits cited; WR-09's fix (bb57709) applied the reviewer's verbatim replacement text after the 3-iteration --auto cap.
