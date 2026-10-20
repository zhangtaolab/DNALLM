# Phase 19: Docs Accuracy Sweep & Milestone Closeout - Context

**Gathered:** 2026-10-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Every docs/ and example/ page verified accurate against the final v1.3 code state: per-file review with same-pass fixes and a disposition record; install commands re-verified against the final extras surface with cuda130 the recommended default everywhere and uv/pip lane labels per phase 18; mkdocs --strict zero warnings (15 ledgered pre-existing cleared); every example edit mirrored byte-identical; milestone closes honest-green. Requirements PKGH-05, DOCR-01, DOCR-02, DOCR-03.

</domain>

<decisions>
## Implementation Decisions

### 收尾方向 (owner-accepted 2026-10-14)
- test-cuda CI leg re-pointed to **13.0** (cuda130 alignment; the Turing-row exception survives in the docs matrix only)
- Milestone matrix-green vehicle: **temporary ci.yml branch-filter line adding v1.3 to the push filter** (owner declined dev-merge for now) — GitHub-hosted legs only (push never reaches the self-hosted box per the D-19 invariant); stays until actual merge or explicit removal, removal decision recorded at closeout
- Disposition ledger granularity: **~150 logical-file rows** (data/mirror-pair classes collapsed by category)

### Carried constraints (roadmap, phases 16-18)
- Sweep lands AFTER every surface it describes stopped moving (deps 16/17/18 — satisfied; the two owner-deferred acceptance items, WINDOWS #25/#27, do not move docs surfaces)
- cuda130 recommended default everywhere: overview.md's 3 stale `.[base,notebook,cuda124]`-class mentions + all install surfaces, lane-labeled per the phase-18 split
- mkdocs --strict zero warnings with the nav/mirror interplay respected; marimo + PATH requirement recorded by 18-02 for exactly this work
- Automated sync/masking gates stay green throughout; example edits mirrored byte-identical same-commit

### Claude's Discretion
Disposition-record shape (chosen: phase-dir review ledger per the 16-EVIDENCE precedent — one row per file, verdict + fix summary), sweep ordering, warning-clearance mechanics.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- docs/getting_started/ (installation.md lane-split + gpu_support.md + install_honesty.md from 18-02 — accurate as of 399f9f1, the sweep verifies the rest of docs/ + example/ against the same state)
- check_docs_sync / check_notebook_md_sync gates; mkdocs build warning ledger (15 pre-existing, byte-recorded by 18-02)
- The phase-13/16 residue-gate and verbatim-record disciplines

### Established Patterns
- Per-file disposition ledgers; same-commit mirror discipline; census stability (216/217 — docs edits must not move it)

### Integration Points
- overview.md + all example READMEs/notebooks with install mentions; mkdocs.yml nav; the milestone-close CI matrix (fast lane, coverage gate, full matrix)

</code_context>

<specifics>
## Specific Ideas

The sweep's "beyond the automated gates" clause means human-grade accuracy reading per file — stale claims about removed machinery (phase 13's ollama batch, phase 18's retired extras) are the expected catch classes.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
