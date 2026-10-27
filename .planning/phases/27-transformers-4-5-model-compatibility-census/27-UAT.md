---
status: testing
phase: 27-Transformers 4/5 Model Compatibility Census
source: [27-VERIFICATION.md]
started: 2026-10-27T18:57:00+08:00
updated: 2026-10-27T18:57:00+08:00
---

## Current Test

number: 1
name: Live dispatch of the census lane on the self-hosted runner
expected: |
  gh workflow run census_compat.yml --ref v1.4 (or GitHub UI dispatch).
  Expected: lane finishes green — 49 matched entries, --verify exit 0
  (seeded from the committed artifact; VERSION DRIFT only on a real
  transformers bump). A red UNEXPECTED trail on unchanged inputs would
  be a real defect. Offline --verify logic already proven by the
  verifier (matched/exit 0, drift-injected/exit 1, absent/exit 1,
  simulated 5.19.0/exit 1); this item covers the live runner + network
  path only.
awaiting: user response

## Tests

### 1. Live dispatch of the census lane on the self-hosted runner
expected: gh workflow run census_compat.yml --ref v1.4 → 49 matched / exit 0 (or honest red on real drift)
result: [pending]

### 2. Biologist-audience readability sign-off on the compatibility page
expected: Owner reviews docs/user_guide/model_compatibility.md and triages the 3 WARNING-class plain-language findings in 27-UI-REVIEW.md: (a) Failure Details cells leak engineering jargon on 6 "not working" rows, (b) footnote 1/2 lack inline row markers, (c) GENA-LM/MutBERT Details cells exceed 300 chars. Accept as-is or request a follow-up pass.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
