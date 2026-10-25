---
phase: 261025-vvv
plan: 01
subsystem: docs
tags: [readme, torch-versions, project-structure, docs-accuracy]
requires:
  - "pyproject.toml torch bounds at <2.15 (quick 261025-e4w, commit f00e90d)"
provides:
  - "README hardware-group table mirroring pyproject resolver truth (ceilings 2.14)"
  - "README Project Structure tree matching all twelve on-disk dnallm/ subpackages"
affects:
  - README.md
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - README.md
decisions:
  - "Ceiling rendered as 2.14 (not 2.15) because pyproject uses exclusive <2.15 bounds — 2.14.x is the highest installable series"
  - "Tree rows inserted alphabetically (interpret after inference, preflight after models, tui after tasks); mcp/ kept as final corner-branch row per existing convention"
metrics:
  duration: 3 min
  completed: 2026-10-25
  commits: 2
status: complete
actuals:
  tasks: 2
  commits: 2
---

# Quick 261025-vvv: Fix stale docs reference in README.md Summary

README hardware-group PyTorch table raised from the stale 2.11 ceiling to 2.14 across all six rows (mirroring pyproject's seven <2.15 torch bounds), and the Project Structure tree now lists all twelve on-disk dnallm/ subpackages including the previously missing interpret/, preflight/, and tui/.

## What Was Done

### Task 1: Hardware-group table torch ceilings 2.11 -> 2.14 (commit 60c5bc8)

Six rows updated, floors preserved exactly:
- `cpu`, `cuda124`: 2.4.0-2.11 -> **2.4.0-2.14**
- `cuda126`, `cuda128`, `mamba`: 2.6.0-2.11 -> **2.6.0-2.14**
- `cuda130`: 2.9.0-2.11 -> **2.9.0-2.14**

Source of truth held: pyproject.toml still shows 7 `<2.15` occurrences (cross-check gate passed unchanged).

### Task 2: Three missing subpackages added to Project Structure tree (commit 967708e)

- `interpret/` after `inference/` — "Motif analysis (FIMO scanner, JASPAR client)"
- `preflight/` after `models/` — "Pre-flight validation core"
- `tui/` after `tasks/` — "Terminal pre-flight wizard (dnallm preflight)"

Rows use the same `│   ├── ` branch character and comment-column alignment as neighbors; `mcp/` remains the final corner-branch row. Comments are plain-language for biologists per owner directive. The tree now matches disk exactly: cli, configuration, datahandling, finetune, inference, interpret, models, preflight, tasks, tui, utils, mcp.

## Verification Results

All five plan gates run against the working tree after both tasks:

1. Stale ceilings gone: `grep -cE '\| 2\.(4|6|9)\.0-2\.11 \|' README.md` -> **0** (baseline 6)
2. New ceilings present: `grep -cE '\| 2\.(4|6|9)\.0-2\.14 \|' README.md` -> **6** (baseline 0)
3. Source-of-truth anchor: `grep -c '<2.15' pyproject.toml` -> **7** (unchanged)
4. Tree completeness: `grep -cE '│   ├── (interpret|preflight|tui)/' README.md` -> **3** (baseline 0)
5. Collateral check: combined README diff touches only the table cells (6+/6-) and three added tree lines (3+/0-); no link targets modified; zero docs/ files touched

Per-task gates: TASK1-PASS and TASK2-PASS both emitted. No pytest run required — README.md is not dnallm/ library code and no test asserts on README content.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

- 60c5bc8: docs(261025-vvv): hardware-group PyTorch table ceilings 2.11 -> 2.14, mirroring pyproject <2.15
- 967708e: docs(261025-vvv): list interpret/, preflight/, tui/ in Project Structure tree

Both on branch v1.4, no Co-Authored-By trailers, staged individually (never `git add .`).

## Self-Check: PASSED

- README.md modified file exists on disk with both regions updated: FOUND
- Commit 60c5bc8 is ancestor of HEAD: FOUND
- Commit 967708e is ancestor of HEAD: FOUND
- No file deletions in either commit; no new untracked files left behind by this item (remaining untracked paths are the orchestrator-owned .planning/ dispatch dirs)
