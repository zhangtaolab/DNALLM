---
status: testing
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
source: [28-VERIFICATION.md]
started: 2026-10-30T01:25:00+08:00
updated: 2026-10-30T01:25:00+08:00
---

## Current Test

number: 1
name: Optional-dependency guide prose adequacy (biologist readability)
expected: |
  Read the rewritten docs/user_guide/optional_dependencies.md as a biologist would:
  the "What is an optional extra?" intro, the BigBird deepspeed guide, the CrossDNA
  flash-attn guide, and the PlantHelixSeek built-in section. Every section is
  understandable without engineering background; install commands are copy-pasteable;
  compile-time expectations (hours-scale CUDA build) and Linux+CUDA constraints are clear.
awaiting: user response

## Tests

### 1. Optional-dependency guide prose adequacy (biologist readability)
expected: Every section understandable without engineering background; copy-pasteable install commands; compile-time and platform constraints clear (docs/user_guide/optional_dependencies.md).
result: [pending]

### 2. Cross-page narrative consistency (README + compatibility page)
expected: A biologist cross-reading the README installation journey (extras table, "Built-in kernels", PlantHelixSeek "No extra is needed") and the model_compatibility.md BigBird row gets one consistent story: CrossDNA works after installing its extra; BigBird needs deepspeed to load but cannot run predictions (upstream-dead Triton API).
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
