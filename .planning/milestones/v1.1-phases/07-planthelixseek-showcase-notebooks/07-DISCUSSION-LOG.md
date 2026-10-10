# Phase 07: PlantHelixSeek Showcase Notebooks - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-03
**Phase:** 07-PlantHelixSeek Showcase Notebooks
**Areas discussed:** notebook-narrative-honesty, assertion-ownership, docs-mirror-writeback, execution-lanes
**Mode:** default (plain-text numbered questions per the owner's recorded CJK-rendering constraint; one batched 4-question turn per area)

---

## Notebook narrative & honesty

| Option | Description | Selected |
|--------|-------------|----------|
| Centralized provenance first cell | One markdown cell: loci + methodology + selection.md link + floors table + env versions + disclaimer | ✓ |
| Distributed annotations | Context-adjacent notes per computation cell | |

Negative controls in-notebook (show recorded values + lightweight recompute): ✓ / not shown: — selected ✓.
Disclaimer dual coverage (opening statement + per-figure caption): ✓ / opening only: — selected ✓.
Narrative tutorial-style (API walk-through) vs results-first — selected tutorial-style.

**User's choice:** A A A A
**Notes:** —

## Assertion ownership (SHOW-05)

| Option | Description | Selected |
|--------|-------------|----------|
| Two-layer | nb computes+prints comparison table; tests/examples/ asserts bands authoritatively | ✓ |
| In-notebook asserts | Metric cells end in assert; execution test = notebook runs | |
| External only | nb shows numbers; assertion invisible to readers | |

Floors source: parse selection.md at test startup + parse-guard ✓ / copied constants — selected parse.
Drift semantics: bands verbatim ✓ / CI-statistical tolerance — selected verbatim.
Failure presentation: named-cause messages ✓ / bare assert — selected named-cause.

**User's choice:** A A A A
**Notes:** —

## Docs-mirror write-back (SHOW-06)

Write-back timing: local execution + manual commit ✓ / nightly auto-write-back — selected manual.
Figure format: altair embedded vega JSON ✓ / static PNG-SVG — selected altair.

| Option | Description | Selected |
|--------|-------------|----------|
| Wrapper-.md pattern (owner-corrected) | Follow existing docs presentation: wrapper tutorial page + byte-synced executed nb + GitHub full-notebook button; figures render on GitHub | ✓ |
| Activate mkdocs-jupyter rendering | nav points at ipynb; research must prove altair vega rendering under mkdocs-material | |
| Both tracks | Wrapper + rendered notebook page | |

Size budget: ≤2MB per executed nb ✓ / ≤5MB — selected 2MB.

**User's choice:** A A A A (after correcting the orchestrator's stale premise)
**Notes:** Owner challenged "mkdocs-jupyter renders the notebooks" with the live site (https://zhangtaolab.org/DNALLM/example/notebooks/finetune_binary/). Live verification: nav points exclusively to wrapper .md files; mkdocs-jupyter 0.26.3 is configured (execute: False) but renders no pages; the established pattern is wrapper tutorial .md + GitHub link. Premise corrected and recorded in D-11.

## Execution lanes

Lane split: nightly-only execution + fast structure tests ✓ — selected.
Timeouts: CRE 40 min / Anno 90 min (2x measured headroom) ✓ / tighter 20-45 / wider 60-120 — selected 40/90.
Batch sizes: frozen per Phase-6 ceilings (CRE bs=4, Anno bs=1, provenance noted) ✓ / adaptive — selected frozen.
First-cell guard: hard fla assertion + key=value version prints ✓ / version prints only — selected hard assertion.

**User's choice:** A A A A
**Notes:** —

## Claude's Discretion

Cell ordering within the tutorial narrative; altair styling within the 2MB budget; fast-lane structure-test granularity.

## Deferred Ideas

None.
