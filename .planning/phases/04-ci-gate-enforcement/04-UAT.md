---
status: complete
phase: 04-ci-gate-enforcement
source: [04-VERIFICATION.md]
started: 2026-10-01T00:00:00Z
updated: 2026-10-01T05:00:00Z
---

## Current Test

none — all tests resolved; UAT closed 2026-10-01T05:00Z

## Tests

### 1. Nightly calibration run 36747594207 terminal state
expected: conclusion success with wall time recorded; platform-cap failure routes to the owner runtime menu
result: [pass] — resolved via successor run **36811033498** (workflow_dispatch on dev, self-hosted runner `dnallm-nightly`, post model-swap 95c9ba0): conclusion **success**. Census: 1656 passed / 7 allowlisted skips / 0 failed in 905.93s (15:05); `Required test coverage of 90.0% reached. Total coverage: 96.30%`; skip audit OK. Job wall 1h21m57s (03:33:15Z→04:55:12Z) — census itself 15:05, remainder is the one-time post-job models-cache re-save after the models.lock key changed. History: calibration run 36747594207 died at the hosted 360-min platform cap; census moved to the self-hosted runner per the recorded owner amendment (GATE-02 amended), which removes the cap entirely.

### 2. Owner: make coverage-gate a required check (branch protection)
expected: gh api -X PUT .../branches/{dev,main}/protection with required check "coverage-gate (py3.12, fast leg)" — payload verbatim in 04-03-SUMMARY.md
result: [pass] — verified live 2026-10-01: `gh api .../branches/{dev,main}/protection` returns `required_status_checks.contexts = ["coverage-gate (py3.12, fast leg)"]` on both dev and main (non-strict enforcement)

### 3. Optional cleanup: hung test-cuda job on probe run 36749810723
expected: run-level status settles (the coverage-gate FAILURE conclusion is already recorded and unaffected)
result: [pass] — run 36749810723 settled to `completed` / `failure` (updated 2026-09-30T19:27:08Z); the failure conclusion is the intended GATE-04 red evidence, unaffected

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

none — phase-4 deferred verification closed; see 04-VERIFICATION.md addendum
