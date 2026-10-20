---
status: testing
phase: 22-Registry Foundation — Runtime Loader & Enrichment
source: [22-VERIFICATION.md]
started: 2026-10-20T12:42:30Z
updated: 2026-10-20T12:42:30Z
---

## Current Test

number: 1
name: Confirm committed enrichment values trace to real remote hub metadata (22-02 no-guessed-values prohibition, D-01)
expected: |
  Remote metadata values equal the committed model_info.yaml values (and the recorded
  .scratch/enrichment_results.json derivations). No committed value turns out to be a
  plausible default rather than a measurement.

  How to check (either works):
  - Re-run the gitignored probe: .scratch/probe_registry_enrichment.py
  - Or spot-check 2-3 load-bearing repos by fetching config.json / tokenizer_config.json
    from ModelScope/HF, e.g.:
    - LongSafari/hyenadna-small-32k-seqlen-hf: config.json max_seq_len == 32770
    - zhangtaolab/plant-dnabert-BPE: tokenizer_class == 'DebertaV2Tokenizer'
awaiting: user response

## Tests

### 1. Enrichment value provenance vs remote hubs
expected: Remote metadata values equal the committed model_info.yaml values; no committed value is a plausible default rather than a measurement. Requires network to the model hubs; the derivation chain lives in gitignored .scratch/ by owner convention, so the committed repo proves committed==recorded (verifier cross-checked 109/109 fields) but not recorded==remote.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
