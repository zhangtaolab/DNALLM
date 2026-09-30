---
status: testing
phase: 04-ci-gate-enforcement
source: [04-VERIFICATION.md]
started: 2026-10-01T00:00:00Z
updated: 2026-10-01T00:00:00Z
---

## Current Test

number: 1
name: Nightly calibration run 36747594207 terminal state (green completion + wall time)
expected: |
  The dispatched coverage-nightly run (full slow census, fail_under=90) completes with
  conclusion success; its wall time feeds the owner runtime decision (accept 4-7.5h /
  larger runner / LANE-01 pull-forward). Note: GitHub hosted runners cap jobs at 360min
  — if the census exceeds that, the run fails at the platform cap and the owner menu
  becomes live.
awaiting: user response

## Tests

### 1. Nightly calibration run 36747594207 terminal state
expected: conclusion success with wall time recorded; platform-cap failure routes to the owner runtime menu
result: [pending]

### 2. Owner: make coverage-gate a required check (branch protection)
expected: gh api -X PUT .../branches/{dev,main}/protection with required check "coverage-gate (py3.12, fast leg)" — payload verbatim in 04-03-SUMMARY.md
result: [pending]

### 3. Optional cleanup: hung test-cuda job on probe run 36749810723
expected: run-level status settles (the coverage-gate FAILURE conclusion is already recorded and unaffected)
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
