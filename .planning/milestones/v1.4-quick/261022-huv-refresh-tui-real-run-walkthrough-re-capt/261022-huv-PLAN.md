---
phase: 261022-huv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/v1.4-TEST-REPORT.html
  - .planning/STATE.md
autonomous: true
requirements:
  - HUV-01
  - HUV-02
  - HUV-03
user_setup: []

estimate:
  tokens: 60000
  raw_tokens: 30000
  tasks: 3
  confidence: high

must_haves:
  truths:
    - "Section 7 figures are from real wizard runs on the current tree (v1.4 HEAD after f94c50f); no fabricated or mock-rendered screens"
    - "The lead paragraph honestly attributes the live-download figure to the 2026-10-22 morning run and the rest to the afternoon re-run (two real runs, stated as such)"
    - "The post-wizard execution block shows verbatim, real command output from an inference run driven by the wizard-generated dnallm_config.yaml — claims in the analysis match what the output actually shows"
    - "Binary base-model listing count stated in analysis matches the live listing (49; plantcad fix added zero entries to this listing — they are finetuned-kind)"
    - "Figure numbering 7-1..7-9 stays contiguous; new figures extend as 7-10/7-11; later sections do not renumber"
  artifacts:
    - ".planning/v1.4-TEST-REPORT.html — §7 refreshed: fresh captures ①-⑦/⑨ on current tree, download panel now shows cached-model state, retained live-transfer shot, + config-file figure + real inference output block"
    - "/tmp/tui-shots/ — fresh SVG captures (v2-*.svg) + morning run originals preserved"
  key_links:
    - "Fresh captures → dnallm.tui.app.DNALLMPreflightApp.run_test(size=(100,36)) + export_screenshot(), same driver pattern as 2026-10-22 morning run"
    - "Inference block → DNAInference(config=load_config('dnallm_config.yaml')).infer(file_path='/tmp/dnallm-demo-data.csv') — the exact config the wizard saved, unmodified"
---

# Refresh TUI real-run walkthrough in v1.4-TEST-REPORT.html (§7)

**Goal.** §7 「终端向导实跑：逐屏截图与解读」 shows concrete execution on the *current*
tree, closes the story's open loops (download-complete state, generated config file,
real follow-through run), and its analysis matches what the screenshots/output actually
show. Owner chose option B (refresh + close the execution gaps).

**Context.**
- Morning run (2026-10-22 10:50–10:56) captured 9 SVGs → committed in 9ce8793.
- Since then f94c50f changed the registry (6 PlantCAD2 → binary; verified: binary
  finetune listing still 49 entries, plantcad absent — listing shot NOT stale on count)
  and the morning real download left FungiHelixSeek (1006 MB) in
  ~/.cache/modelscope/hub/models/zhangtaolab/FungiHelixSeek — so a fresh run now
  demonstrates the cache-detection column ("已在本机") and the download panel's
  already-downloaded state, which the morning set never showed.
- No dnallm/ code changes in this task → no pytest gate (owner rule applies to lib
  changes only).

## Task 1 — Fresh wizard capture on current tree (no repo writes)
- [ ] Write /tmp/tui-capture-v2.py from the morning driver pattern: full walk
      (env → entry → data w/ /tmp/dnallm-demo-data.csv → suitability binary →
      listing (FungiHelixSeek row visible, cache column) → select → config save →
      download panel incl. cached-state and post-start state → summary).
- [ ] Export SVGs to /tmp/tui-shots/v2-*.svg at 100×36; copy the wizard-saved
      dnallm_config.yaml aside (v2) before any later step overwrites it.
- [ ] Verify each SVG opens (non-empty, <svg root) and row/text states match
      expectations (e.g. availability text on FungiHelixSeek row).
- [ ] Keep morning live-transfer SVG (6b-download-after.svg) — it is the only
      genuine live-download image; do not re-shoot, re-caption.

## Task 2 — Real follow-through execution with the wizard config
- [ ] Run via the venv, exact config the wizard saved:
      DNAInference(model=None, tokenizer=None, config=load_config('<repo>/dnallm_config.yaml'))
      .infer(file_path='/tmp/dnallm-demo-data.csv', ...) — record verbatim console
      output (tee to /tmp/tui-infer-out.txt).
- [ ] If the untrained-head predict path misbehaves, fall back honestly to the
      embedding path and report what a pretrained foundation model actually
      delivers — no cherry-picking, no reruns-until-pretty.
- [ ] Note wall-clock time, device (GB10), and what the output contains.

## Task 3 — Update §7, verify rendering, ship
- [ ] Embed fresh SVGs as data URIs (base64, same inline pattern); keep/extend CSS
      classes already present (tui-fig/tui-shot/tui-analysis). New figures:
      ⑧ becomes two (cached-state + morning live transfer), then ⑨ summary,
      ⑩ generated config file, ⑪ inference output block (terminal-styled <pre>,
      verbatim).
- [ ] Rewrite analysis bullets to match the new evidence: cache column on ⑤/⑥,
      already-downloaded on ⑧, real config fields on ⑩ (max_length 8192 from
      registry enrichment), honest verdict on ⑪ (what a pretrained model gives
      before fine-tuning; next step = dnallm train).
- [ ] Update lead paragraph: two real runs, stated honestly; update counts.
- [ ] Render-verify §7 with the system headless Chromium (extract section →
      screenshot → inspect); confirm no collision/garbling, figures load.
- [ ] Commit (docs(quick-261022-huv)), push origin/v1.4, refresh served copy in
      /tmp/gsd-report-serve, append STATE.md quick-task row.
