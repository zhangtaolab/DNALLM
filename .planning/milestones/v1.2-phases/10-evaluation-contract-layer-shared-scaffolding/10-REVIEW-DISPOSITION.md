---
phase: 10
review: 10-REVIEW.md
titles: json
findings:
  - id: IN-01
    severity: info
    disposition: fixed
    title: "`evaluate(split=...)` guard misses `output_dir=\"\"` — empty string still writes the result JSON to the CWD"
  - id: IN-02
    severity: info
    disposition: fixed
    title: "WR-05 fix restored the pre-phase old terminology in `gpu_optimization.md` — the only swept docs page with zero occurrences of the standard term"
  - id: CR-01
    severity: critical
    disposition: fixed
    title: "Botched test insertion deleted `test_process_missing_data_basic` and fused its body into the preceding test"
  - id: WR-01
    severity: warning
    disposition: fixed
    title: "New PEFT chapter's `load_model_and_tokenizer` examples fail as written (missing `source=`)"
  - id: WR-02
    severity: warning
    disposition: fixed
    title: "Mechanical sweep rewrote verbatim paper titles in `MODEL_INFO` and the titles remain garbled by string-literal concatenation"
  - id: WR-03
    severity: warning
    disposition: fixed
    title: "Eval-semantics guard is asymmetric — `load_best_model_at_end` collision only raises on the test-excluded path"
  - id: WR-04
    severity: warning
    disposition: fixed
    title: "Bare debug print left in `calculate_metric_with_sklearn` inside a function this phase rewrote"
  - id: WR-05
    severity: warning
    disposition: fixed
    title: "Terminology sweep produced a broken sentence: \"large DNA large language models\""
  - id: IN-03
    severity: info
    disposition: fixed
    title: "`evaluate(split=...)` writes its result JSON into the CWD when `output_dir` is None"
  - id: IN-04
    severity: info
    disposition: fixed
    title: "`use_ia3: true` is a silent no-op during the interim window"
  - id: IN-05
    severity: info
    disposition: fixed
    title: "QLoRA example in the new PEFT chapter uses `datasets` without constructing it"
  - id: IN-06
    severity: info
    disposition: fixed
    title: "Model-count inconsistency: \"150+\" vs \"200+\""
open: 0
total: 12
recorded: 2026-10-09T12:24:18.553Z
---

# Phase 10: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| IN-01 | info | fixed | 10-REVIEW-FIX.md (not in the current review) |
| IN-02 | info | fixed | 10-REVIEW-FIX.md (not in the current review) |
| CR-01 | critical | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| WR-01 | warning | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| WR-02 | warning | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| WR-03 | warning | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| WR-04 | warning | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| WR-05 | warning | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| IN-03 | info | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| IN-04 | info | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| IN-05 | info | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |
| IN-06 | info | fixed | 10-REVIEW-FIX.iter2.md (not in the current review) |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
