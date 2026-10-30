---
phase: 261030-rst
plan: 01
subsystem: planning-record
tags: [deferred-items, mkdocs, docs-truth, marimo-mirrors]
requires:
  - "47b9bc2 — docs(19-01): strict-clean — 15 mkdocs warnings to zero (the prior fix this closure cites)"
  - "8b224ea — planning baseline the fixed-base comparison and fresh strict build run against"
provides:
  - "DOCS-12-MKDOCS-STRICT closed with dated evidence: prior-fix commit 47b9bc2 + fresh zero-warning strict build at baseline 8b224ea"
  - "Preservation warning recorded: the three docs/example/marimo mirrors are hand-maintained; regeneration without a redesign would delete enriched prose"
affects:
  - ".planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md (mkdocs section only)"
tech-stack:
  added: []
  patterns:
    - "Evidence-only deferred-item closure: flip one status, cite the prior fix commit and a fresh build, forbid the superseded regeneration approach"
key-files:
  created: []
  modified:
    - .planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md
decisions:
  - "Closure is evidence-only: the 15-warning fix already existed at 47b9bc2; the deferred record was stale, so only the record changed — no source, mirror, or config edits"
  - "Regeneration is explicitly forbidden in the annotation: scripts/generate_md_from_marimo.py is bootstrap scaffolding and a dry run produced extensive non-link deletions; the mirrors' Prerequisites/Overview/See-Also prose lives only in the mirrors"
metrics:
  duration: 6 min
  completed: 2026-10-30
  tasks: 1
actuals:
  tokens: 640
  tasks: 1
  commits: 1
commits: 1
plan_head_before: 8b224eaa6ab0be1b5dd383d31edfbd971671d89f
plan_head_after: aab42607f3a0b3fc671c2c12acee920ebee3bc2a
status: complete
---

# Quick Task 261030-rst: Close stale mkdocs strict deferred item Summary

**One-liner:** Flipped the stale phase-12 mkdocs --strict deferred bullet to resolved with dated dual evidence (prior fix 47b9bc2 + fresh zero-warning strict build at 8b224ea) and a hand-maintained-mirror preservation warning.

## What Was Done

Exactly one scoped edit in
`.planning/milestones/v1.2-phases/12-motif-matching-mcp-tools-milestone-closeout/deferred-items.md`,
in the section `[Out-of-scope discovery — 12-03 Task 1] mkdocs --strict fails on 15 pre-existing warnings`:

- `status: acknowledged` → `status: resolved` (lowercase, bullet indentation preserved).
- A one-paragraph annotation added immediately after the resolved status, recording both evidence
  sources and the preservation constraint.

The historical Found/Issue/Proof/Why/Suggested text of the section is preserved verbatim below the
annotation; the annotation supersedes its stale state.

## Evidence Recorded

1. **Prior fix (evidence source 1):** commit `47b9bc2` — `docs(19-01): strict-clean — 15 mkdocs
   warnings to zero` — already fixed all 15 warnings (14 marimo mirror broken relative links + the
   CONTRIBUTING link).
2. **Fresh build (evidence source 2):** at HEAD `8b224ea` with MkDocs 1.6.1,
   `.venv/bin/mkdocs build --strict` exited 0 with zero WARNING lines.
3. **Preservation constraint:** the three `docs/example/marimo` mirrors are hand-maintained; their
   Prerequisites, Overview, and See-Also prose lives only in the mirrors.
   `scripts/generate_md_from_marimo.py` is bootstrap scaffolding only, and a dry-run regeneration
   produced extensive non-link deletions — do not regenerate without a preservation redesign.

## Verification Results

The full automated gate from the plan ran green end to end:

- **Strict build:** MkDocs 1.6.1 confirmed; `build --strict` exited 0 in 4.86 s with zero lines
  matching `^WARNING\s` — `PASS: strict build exited 0 with zero WARNING lines`.
- **Fixed-base tamper proof:** whole-file section comparison against baseline `8b224ea` proved the
  file still has exactly 4 sections, only the mkdocs section changed, sections 1 (HBG1/BCL11A
  owner-deferral) and 3 (numpy 2.5.x coverage) are byte-identical, and exactly 2
  `status: acknowledged` remain — `PASS: only the mkdocs section changed; HBG1 and numpy sections
  are unchanged`.
- **Scope guards:** `git diff --check` clean (no whitespace errors); `git diff --quiet 8b224ea --
  docs scripts dnallm tests example mkdocs.yml pyproject.toml` confirmed zero product
  code/docs/config changes.
- **Concise changed-line diff:** exactly `-  status: acknowledged` replaced by `+  status:
  resolved` plus the 9-line annotation — nothing else.

## Scope Confirmation

- No mirror regeneration was run against `docs/example/marimo/` (the dry-run hazard is only
  *cited* in the annotation; no generator execution modified any file).
- `scripts/generate_md_from_marimo.py`, all repository code, product documentation, tests,
  examples, and configuration remain unchanged (proved by the fixed-base diffs above).
- No packages installed, no pytest run, no cache cleaning, no ROADMAP.md change.

## Commit

- `aab4260` — `docs(quick-261030-rst): close stale mkdocs strict deferred item`
  (1 file changed, 10 insertions(+), 1 deletion(-); no co-author trailers).

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `deferred-items.md` present on disk and containing the resolved status (FOUND).
- Commit `aab4260` is an ancestor of HEAD (FOUND).

## Known Stubs

None.
