# Phase 27 — UI Review (Re-audit)

**Audited:** 2026-10-28
**Baseline:** abstract 6-pillar standards + owner plain-language directive (2026-10-21); no UI-SPEC for this phase (frontend surface is the generated docs page `docs/user_guide/model_compatibility.md`)
**Screenshots:** not captured (no dev server on 3000/5173/8080 — Markdown/MkDocs deliverable, docs/code-only audit)
**Interaction captures:** off (workflow.ui_interaction_capture is false)

**Re-audit scope:** prior review's WARNING (a) jargon-led failure cells and (c) >300-char cells were to be closed by plan 27-05; WARNING (b) footnote markers is owner-deferred and is not counted against the phase.

## Verification Evidence

- `python -I scripts/census_compat.py --emit-docs /tmp/mc-audit.md && cmp` → **BYTE-IDENTICAL** (page is generator-produced, never hand-edited)
- `pytest tests/models/test_compat_docs_consistency.py -q` → **68 passed** (baseline 54; extension includes independent lead map, suffix bound, jargon blocklist, 300-char cap)
- Row count: 49; header summary 19+21+1+6+2 = 49, consistent with table
- Max What-we-found cell length: 287 chars (all ≤300)
- All six remote-code-broken rows lead with a plain-language sentence before the ` (technical: ` parenthetical; four distinct honest leads (damaged-files / own-defect / external-package / library-change ×2)

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 4/4 | Six failure rows now lead biologist-readable, honest per failure class; technical cause demoted to a bounded parenthetical (≤100 chars, word boundary) |
| 2. Visuals | 3/4 | Clean table, clear Details column; footnote texts 1–2 exist but no row references them (owner-deferred), and status has no visual weight differentiation for scanning |
| 3. Color | 3/4 | N/A for plain Markdown — no color tokens; status is word-only ("Not usable…" vs "Tested working"), which forces full-cell reading to spot failures |
| 4. Typography | 3/4 | Consistent headings/table; status cells are long (up to 287 chars) relative to other cells, mildly unbalancing column rhythm |
| 5. Spacing | 4/4 | Uniform table structure, 49 rows, no irregular rows or stray artifacts; generated deterministically |
| 6. Experience Design | 4/4 | Every state covered honestly (working / code-verified / shim-fixed / broken-with-cause / unchecked-with-reason); drift gate + fail-closed lead mapping + byte-identical regeneration prevent regression |

**Overall: 21/24**

## Top 3 Priority Fixes

1. **No visual differentiation of failure rows** (Pillars 2/3, WARNING) — a reader scanning 49 rows must parse full sentences to find the 6 broken + 2 unchecked models — bold the status lead ("**Not usable with the current transformers**", "**Could not be checked**") or group broken/unchecked rows after the working ones; trivial generator change + lockstep gate edit.
2. **Footnote markers absent from cells** (Pillar 2, DEFERRED — do not action) — footnotes 1/2 are never referenced in the table; **owner-deferred by explicit decision recorded in 27-UAT.md / 27-05-SUMMARY; recorded here only**.
3. **DNABERT-2 and GENA-LM technical suffixes end mid-clause at the truncation boundary** (Pillar 1, minor) — "(rebuild_alibi_tensor line 399:…)" and "buffer starts with…" read as cut-off; the trailing ellipsis makes this honest, but rounding to the nearest complete sub-clause would read better — cosmetic, bounded by the same 100-char constant.

## Prior-Warning Disposition

- **(a) Engineering jargon leading failure cells — CLOSED.** All six broken rows verified to lead with a plain sentence; the jargon lives only inside the parenthetical and is bounded (e.g. lines 40–55 of the page).
- **(b) Footnote markers — DEFERRED (owner decision, not counted).**
- **(c) Cells exceeding 300 chars — CLOSED.** Max observed 287; page-wide cap pinned by a gate test.

## Detailed Findings

### Pillar 1: Copywriting (4/4)
Leads are class-honest (a version-independent defect is never worded as a library removal — GENA-LM/MutBERT rows confirm). Census-blocked rows ("Could not be checked…") carry plain reasons. Header paragraph explains transformers in plain terms and states the tested/inferred distinction. Bilingual footnote labels per D-05.

### Pillar 2: Visuals (3/4)
Table structure consistent; Details column shows repo ids plus "full run tested on X" provenance. Findings: no per-row anchor to footnotes (deferred); no scannable status emphasis.

### Pillar 3: Color (3/4)
Plain Markdown — no color system applies. Scored on differentiation substitutes: status distinction exists only through wording. Not a defect against the deliverable's medium; bold/grouping would compensate (fix 1).

### Pillar 4: Typography (3/4)
One heading level + table + ordered footnotes; uniform. Status cells up to 287 chars dominate their rows — inherent to the honest-cause approach and capped; acceptable.

### Pillar 5: Spacing (4/4)
Deterministic generator output; byte-identical regeneration verified; no layout irregularities.

### Pillar 6: Experience Design (4/4)
State coverage complete across all five outcome classes; drift gate independently re-derives every cell (checker independence, no generator import); fail-closed ValueError on unmapped failure classes; deliberate-red demonstrations recorded in 27-05-SUMMARY.

## Registry Safety Audit
`components.json` absent — skipped (no shadcn / no third-party registries).

## Files Audited
- docs/user_guide/model_compatibility.md (live page, 49 rows)
- scripts/census_compat.py (generator — via byte-identical regeneration)
- tests/models/test_compat_docs_consistency.py (drift gate, 68 tests)
- 27-01..27-05 PLAN/SUMMARY, 27-CONTEXT.md
