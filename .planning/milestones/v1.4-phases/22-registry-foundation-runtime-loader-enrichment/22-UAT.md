---
status: complete
phase: 22-Registry Foundation — Runtime Loader & Enrichment
source: [22-VERIFICATION.md]
started: 2026-10-20T12:42:30Z
updated: 2026-10-20T13:36:11.818Z
---

## Current Test

[testing complete]

## Tests

### 1. Enrichment value provenance vs remote hubs
expected: Remote metadata values equal the committed model_info.yaml values; no committed value is a plausible default rather than a measurement. Requires network to the model hubs; the derivation chain lives in gitignored .scratch/ by owner convention, so the committed repo proves committed==recorded (verifier cross-checked 109/109 fields) but not recorded==remote.
result: pass
verified_by: owner delegation ("你来检查", 2026-10-20) — Claude performed the check
evidence: |
  Local chain: committed YAML == .scratch/enrichment_results.json for all 109
  committed fields (46 tokenizer_type + 63 max_length), 0 mismatches.
  Remote chain (direct HTTP, cache-bypassing, artifacts in
  .scratch/uat-spotcheck/): 10 repos fetched live from ModelScope/HF:
  - plant-dnabert-BPE (MS): tokenizer_class=DebertaV2Tokenizer, config
    max_position_embeddings=512 — matches committed
  - FungiHelixSeek (MS): EsmTokenizer, max_position_embeddings=8192 — matches
  - plant-dnamamba-BPE (MS): GPTNeoXTokenizer; config carries NO length key;
    tokenizer_config max_length=1024 is the GPTNeoX class template default with
    model_max_length=1e30 sentinel — exactly as the derivation note states;
    committed 512 verified on all three legs (PDLLMs upstream
    model_pretrain_from_scratch.py:42 default=512 fetched from GitHub; sibling
    plant-dnamamba-6mer tokenizer model_max_length=512)
  - hyenadna-small-32k-seqlen-hf (HF): HyenaDNATokenizer, config
    max_seq_len=32770 (tokenizer agrees) — matches
  - caduceus-ph-131k (HF): CaduceusTokenizer, tokenizer model_max_length=131072
    — matches
  - PlantCAD2-Small (HF): PreTrainedTokenizerFast; config carries NO length key
    (paper-derived 8192 premise holds) — matches
  - GENERanno-1.2b: repo absent from BOTH hubs re-confirmed (MS HTTP 404, HF
    not-found); sibling 0.5b config max_position_embeddings=8192 — matches
  - nucleotide-transformer-v2-100m (HF): EsmTokenizer, config
    max_position_embeddings=2050 — matches
  No committed value is a plausible default rather than a measurement or a
  fully-cited derivation whose premises were spot-verified.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
