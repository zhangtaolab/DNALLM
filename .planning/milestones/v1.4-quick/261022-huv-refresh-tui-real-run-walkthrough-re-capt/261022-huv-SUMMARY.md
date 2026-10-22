---
status: complete
quick_id: 261022-huv
date: 2026-10-22
---

# Refresh TUI real-run walkthrough in v1.4-TEST-REPORT.html §7

## What changed

- **§7 rebuilt on the current tree** (post f94c50f): 11 figures from two real wizard
  runs — morning 10:50 first-run (live ~1 GB ModelScope transfer) and 14:34 re-run
  (FungiHelixSeek cached: cache-detection column lights up, download panel verifies
  cache in ~2 s, summary now shows the model path).
- **Owner follow-up (14:58): fool-proofing + data-sample display added.** New
  subsection 「防呆实测」: real captures of bad-data-path rejection (plain-language
  error) and unknown-model lookup (honest `fail / MODEL_NOT_IN_REGISTRY` row), plus
  8 representative fool-proofing tests run live (6+2 passed) and two honestly
  recorded non-blocking boundaries (task-type typo falls back silently; config-edit
  route does not registry-check model names). New subsection 「测试数据长什么样」:
  verbatim truncated CSV head + stats (24 rows, 12/12, 183–260 bp, pure ACGT,
  randomly generated — explains the 0.50 coin-flip downstream). §7 now 13 figures
  + 3 code blocks.
- **Execution loop closed**: added the wizard-generated `dnallm_config.yaml` verbatim
  (9 lines) and a real inference run driven by it (14.1 s total on GB10/CUDA,
  24 sequences). Untrained-head accuracy 0.50 reported as coin-flip, not dressed up.
- **Honest findings recorded**: (a) `dnallm inference -i` declared but never wired
  into `infer()` — added as §10 tech-debt item 13; (b) `dnallm train -c` refuses
  without data with a plain-language error; (c) morning Size line showed 3.7 KB
  because the ModelScope file listing was incomplete that probe — the probe sums
  what the hub returns; explained in figure 7-10 analysis.
- Verified: 11 figures + 2 code blocks render correctly (section-by-section
  Chromium screenshots inspected); sections 1–12 intact; binary base listing
  still 49 (plantcad fix added zero entries — they are finetuned-kind).

## Evidence

- Fresh captures: /tmp/tui-shots/v2-*.svg; morning originals preserved.
- Inference log: /tmp/tui-infer-report.txt (verbatim block embedded in report).
- Pre-edit report backup: /tmp/report-backup-pre-huv.html.

## Commits

- Report + tech-debt row + this task's planning artifacts (see git log, docs(quick-261022-huv)).
