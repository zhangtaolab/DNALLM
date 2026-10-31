---
phase: 261020-gk4
plan: 01
subsystem: analysis
tags: [tui, survey, v1.4-planning, ux]
requires: []
provides:
  - "Ranked TUI opportunity report (6 ranked, 7 excluded) feeding v1.4 scope decisions (V14-SCOUT-01)"
affects: []
tech-stack:
  added: []
  patterns: []
key-files:
  created:
    - .planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md
  modified: []
decisions:
  - "Model-registry browser killed to Part 4 despite different underlying catalog — the delivered UX (filterable table + compat verdict + download state) is exactly dnallmmark's selection screen; discovery needs absorbed into R1/R4"
  - "Registry counts recorded from measurement, not docs: model_info.yaml holds 221 entries (34 pretrained + 187 finetuned) and PRETRAIN_MODEL_MAPS has 37 family keys"
  - "Report is a single commit (3028f0c) per Task 3's sole-commit contract; Tasks 1-2 verified by gates between edits without intermediate commits"
metrics:
  duration: 8 min
  completed: 2026-10-20
status: complete
commits: 1
plan_head_before: d86e18f213f28f894650244a0ef3549f7914553e
plan_head_after: 3028f0ca0c0c0e8d18917f43f44e7f7afc71c813
actuals:
  tokens: 5972
  tasks: 3
  commits: 1
---

# Quick Task 261020-gk4: TUI Opportunity Survey Summary

Ranked, evidence-backed survey of terminal-UI opportunities in DNALLM, explicitly excluding everything dnallmmark's Textual TUI already delivers — analysis only, zero source changes.

## What Was Done

- **Task 1 (baseline):** Read all 9 dnallmmark screens and all 16 service modules directly (docstrings + code) and wrote Parts 0-2: method, the exclusion baseline (one-line current function per screen/service), and the full DNALLM user-facing inventory — 6 click commands, the standalone `dnallm-vep` script (confirmed not wired into the click group), all 7 console scripts, 3 Gradio apps, the 17-tool MCP roster (noting load/unload are NOT registered as tools despite existing as `ModelManager` methods), the YAML-config error UX (raw Pydantic traceback), the model registry (measured: 221 entries, 37 families), and training/inference output UX (tqdm + post-hoc PNG/altair files). Gate: TASK1_BASELINE_OK.
- **Task 2 (ranking):** Wrote Parts 3-4. Six ranked opportunities — R1 interactive model-exploration console (amortize per-invocation model loads), R2 live fine-tune dashboard (TrainerCallback stream vs post-hoc PNGs), R3 MCP server control room (launch/observe/load-unload), R4 terminal YAML config validator with live collect-all Pydantic errors, R5 VEP batch runner with progress + hit triage, R6 in-terminal interpretation triage. Seven exclusions in Part 4, each naming its covering dnallmmark screen/service. Gate: TASK2_RANKING_OK.
- **Task 3 (validation + commit):** All gates green — every backticked repo-relative path exists (PATH_GATE_OK), dnallmmark tree clean (EXT_CLEAN_OK), zero diffs under dnallm/ui/tests/configs/scripts (SRC_UNTOUCHED_OK), report committed as the sole change (SURVEY_VALID).

## Verification Results

| Gate | Result |
|------|--------|
| TASK1_BASELINE_OK (9 screens + 7 command surfaces present) | PASS |
| TASK2_RANKING_OK (6 R-blocks; 6 Current UX / 6 Non-overlap lines; Part 4 present) | PASS |
| Path-existence gate (all backticked DNALLM paths exist) | PASS |
| External cleanliness (dnallmmark `git status --porcelain` empty) | PASS |
| Source untouched (`git diff` over dnallm ui tests configs scripts empty) | PASS |
| Commit message + sole-change contract | PASS |

## Key Findings

- The single highest UX pain is structural: every CLI invocation reloads the model (R1); the only warm-session alternative today is a browser Gradio app limited to generation.
- The MCP server cannot load/unload models after startup — `ModelManager.unload_model`/`load_model` exist (`dnallm/mcp/model_manager.py`) but are deliberately absent from the tool roster; a control-room TUI (R3) is the natural consumer.
- Config errors surface as raw Pydantic tracebacks; the click-prompt generator only authors new configs — R4 targets the edit-existing-config + collect-all-preview gap.
- Defect noticed during inventory (out of scope, for v1.4 triage): the `dnallm mutagenesis` click subcommand is a stub that prints success without running anything, while the real flow lives only under the `dnallm-mutagenesis` console script.

## Deviations from Plan

None — plan executed exactly as written. The report enumerates 16 service modules where the plan named 15 (`dnallmmark:tui/services/about.py` also exists on disk and was included for completeness — superset of the must-have list).

## Known Stubs

None (analysis-only task; the survey report is complete with no placeholder content).

## Self-Check: PASSED

- Report exists at .planning/quick/261020-gk4-tui-dnallmmark-home-forrest-github-dnall/261020-gk4-TUI-SURVEY.md — FOUND
- Commit 3028f0c (docs(261020-gk4): TUI opportunity survey for v1.4 planning) — FOUND, ancestor of HEAD
- Measured commits: 1 (3028f0c), matching the sole-commit contract
