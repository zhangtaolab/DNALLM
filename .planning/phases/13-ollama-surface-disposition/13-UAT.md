---
status: testing
phase: 13-ollama-surface-disposition
source: [13-VERIFICATION.md]
started: 2026-10-11T09:50:00Z
updated: 2026-10-11T09:50:00Z
---

## Current Test

number: 1
name: Confirm the first post-merge example-nightly run comes green (next 05:30 UTC scheduled run after merge to the nightly-running branch, or an owner workflow_dispatch — do NOT dispatch casually: a ci.yml workflow_dispatch also enqueues the dispatch-only feas-spike job on the self-hosted box)
expected: |
  Stage 0.5 census gate passes at 216/217 tests collected (1 deselected); stage 1 records 2 typed skips with 'network-unavailable:' messages carrying live probe evidence (not passes, not failures); the stage-4 audit over pytest-junit-example-stage1.xml reports OK; stage-results.txt ledger has zero non-zero lines; job green. A red nightly falsifies OLLM-02 and must be triaged from the uploaded complete failure scene (D-14) before phase close.
awaiting: user response

## Tests

### 1. First post-merge example-nightly run comes green
expected: Stage 0.5 census gate passes at 216/217 tests collected (1 deselected); stage 1 records 2 typed skips with 'network-unavailable:' messages carrying live probe evidence; stage-4 audit over pytest-junit-example-stage1.xml reports OK; stage-results.txt has zero non-zero lines; job green. (WINDOWS.md entry 22, unrun-verify; SUMMARY coverage D5.)
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
