---
status: complete
phase: 13-ollama-surface-disposition
source: [13-VERIFICATION.md]
started: 2026-10-11T09:50:00Z
updated: 2026-10-13T00:36:17.716Z
---

## Current Test

[testing complete]

## Tests

### 1. First post-merge example-nightly run comes green
expected: Stage 0.5 census gate passes at 216/217 tests collected (1 deselected); stage 1 records 2 typed skips with 'network-unavailable:' messages carrying live probe evidence; stage-4 audit over pytest-junit-example-stage1.xml reports OK; stage-results.txt has zero non-zero lines; job green. (WINDOWS.md entry 22, unrun-verify; SUMMARY coverage D5.)
result: pass
evidence: "R8 owner-dispatched run 38308972028, example-nightly job 115009783996, conclusion success (2026-10-13). Log-verified: census measured '216/217 tests collected (1 deselected)'; mcp_example pair (mcp_client_ollama_langchain_agents.ipynb + mcp_client_ollama_pydantic_ai.ipynb) audit-allowed with network-unavailable live-probe reasons; all four stage-4 junit audits exit 0; stage-results.txt eight lines all =0 with 'OK: every recorded stage item exited 0'; pytest 211 passed, 5 skipped, 1 deselected in 1:49:56. Vehicle: local GB10 repo-runner id22 via the blessed dispatch path; shakedown rounds R1-R7 (env leak, cpu-torch, hf-mirror flakiness — infra only, never phase-13 behavior) documented in the verify-work session."

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
