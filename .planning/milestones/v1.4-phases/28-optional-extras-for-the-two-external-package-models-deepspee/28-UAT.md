---
status: complete
phase: 28-optional-extras-for-the-two-external-package-models-deepspee
source: [28-VERIFICATION.md]
started: 2026-10-30T01:25:00+08:00
updated: 2026-10-30T16:59:00+08:00
---

## Current Test

[testing complete]

## Tests

### 1. Optional-dependency guide prose adequacy (biologist readability)
expected: Every section understandable without engineering background; copy-pasteable install commands; compile-time and platform constraints clear (docs/user_guide/optional_dependencies.md).
result: pass
reported: "Verified by agent under owner instruction '你来验证' (2026-10-30): full read as biologist. transformers glossed at first mention; block-sparse attention, meta-device, --no-build-isolation all explained by cause; 5 copy-pasteable fenced commands; Linux+CUDA + hours-scale compile expectations stated in three places; honest BigBird guidance ('pick a model that works' for predictions); provenance discloses replica-grade evidence and lanes not yet live."

### 2. Cross-page narrative consistency (README + compatibility page)
expected: A biologist cross-reading the README installation journey and the model_compatibility.md BigBird row gets one consistent story: CrossDNA works after installing its extra; BigBird needs deepspeed to load but cannot run predictions (upstream-dead Triton API).
result: issue
reported: "Verified by agent under owner instruction '你来验证' (2026-10-30): CrossDNA story consistent across guide + compatibility page ('Works after installing an optional extra (tested)') and README built-in/fla statements consistent. BUT the compatibility page's BigBird row contradicts itself and the guide: the class lead reads 'cannot be loaded until the model authors update it' while its own parenthetical and the guide state loading works with the deepspeed extra — a biologist skimming table leads takes away the wrong load conclusion. Secondary omission: README's 'models requiring extra dependencies' table has no BigBird/CrossDNA rows (no contradiction; the two new extras are absent from that surface)."
severity: minor

## Summary

total: 2
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- gap_id: G-28-2
  truth: "A biologist cross-reading the guide and the generated compatibility page gets one consistent BigBird story (needs deepspeed to load; cannot run predictions)"
  status: resolved
  resolved_by: 28-06-PLAN.md
  resolved_at: 2026-10-30
  reason: "Agent-verified (owner-delegated): compatibility page BigBird row's class lead 'cannot be loaded until the model authors update it' contradicts its own split parenthetical and the guide's load-green fact; README extras table also missing BigBird/CrossDNA rows"
  severity: minor
  test: 2
  root_cause: "scripts/census_compat.py DOCS_BROKEN_LEADS (:1855) maps the library-change evidence class to a single generic 'cannot be loaded' lead — BigBird's evidence carries the load-green/forward-red split but the lead is class-keyed, not model-aware; the independent gate mirror BROKEN_LEADS (tests/models/test_compat_docs_consistency.py:55) pins the same wording so both must change in lockstep (code-review finding IN-03, root-caused 2026-10-30)"
  artifacts:
    - path: "scripts/census_compat.py"
      issue: "DOCS_BROKEN_LEADS library-change lead hardcodes 'cannot be loaded until the model authors update it' (:1867)"
    - path: "tests/models/test_compat_docs_consistency.py"
      issue: "BROKEN_LEADS gate mirror requires the same lockstep wording (:55, :217, :455)"
    - path: "README.md"
      issue: "'models requiring extra dependencies' table lacks GENA-LM-BigBird / CrossDNA rows"
  missing:
    - "Split-aware lead for BigBird (or a model-specific override) in DOCS_BROKEN_LEADS + lockstep gate mirror; regenerate compat page byte-check; add the two README table rows linking to the guide"
  debug_session: ""
