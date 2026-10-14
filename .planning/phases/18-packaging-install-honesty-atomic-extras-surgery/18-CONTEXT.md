# Phase 18: Packaging & Install Honesty — Atomic Extras Surgery - Context

**Gathered:** 2026-10-14
**Status:** Ready for planning

<domain>
## Phase Boundary

One atomic, grep-guarded extras surgery: base renamed, cuda121 retired, rocm dispositioned, triton-windows declared (WPAR-07), install docs split uv/pip with the GPU support matrix and cross-installer honesty matrix published from pre-verified boundaries, every changed extra resolving clean in a fresh venv validated by phase-17's smoke dispatch against the post-surgery tree. Requirements PKGH-01..04/06/07 + WPAR-07.

</domain>

<decisions>
## Implementation Decisions

### extras 命名与处置 (owner-accepted 2026-10-14)
- base → **`full`** (dev/test/notebook/mcp superset; self-referential `all` chain intact; one-cycle deprecated alias so pip's warn-and-exit-0 cannot silently under-install; grep gate holds stale `.[base]` residue to a sanctioned list)
- rocm: **REMOVED with documented restore path** (no rocm wheel exists for torch>=2.10 — keeping it would be a false promise)
- WPAR-07 triton-windows: **DECLARED** behind `platform_system == 'Windows'` marker in the fla extra (`>=3.7.1.post27,<3.8`, research-recommended); fla KDA smoke re-verified on Windows under the resolved torch
- Validation depth: fresh-venv resolve per changed extra + smoke dispatch against the post-surgery tree (each extras change validated)

### Carried constraints (roadmap)
- cuda121 retired same atomic discipline (extra + uv index/sources/conflicts + docs in one change, index frozen at torch 2.5.1)
- 5 CI/publish call sites + extras guard tests + docs in the same atomic change as the rename
- Install docs: explicit uv vs pip lanes stating the [tool.uv.sources] asymmetry; GPU matrix from pre-verified boundaries (Pascal cpu-only + min-GPU line, 20xx→cu128, 30/40xx all extras, 50xx cu128+); cross-installer honesty matrix page live

### Claude's Discretion
Alias mechanics, grep-gate implementation, docs structure within the locked contracts.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- pyproject.toml extras block (the surgery table); tests/test_extras_guard.py (the guard to extend); .github/workflows call sites (5, mapped by 17-01's work); docs install/GPU pages
- Phase-17 smoke (dispatchable against any tree — the validation vehicle)

### Established Patterns
- Atomic-commit discipline with pickaxe proofs (17-01); grep gates with sanctioned lists (16 residue discipline); WINDOWS ledger (#18 closes here)

### Integration Points
- ci.yml/publish.yml call sites; docs/ install pages; the fla extra; uv index/sources config (cuda121 removal)

</code_context>

<specifics>
## Specific Ideas

The deprecated-alias one-cycle: pip installing `dnallm[base]` on the alias must WARN and exit 0 — exactly the silent-under-install hazard the alias exists to surface loudly, not hide.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
