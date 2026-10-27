# Phase 27 — UI Review

**Audited:** 2026-10-27
**Baseline:** Abstract 6-pillar standards + project directives (plain-language-for-biologists, honest-labeling)
**Screenshots:** not captured (no dev server — no web app exists in this phase; the only user-facing surface is a static MkDocs Markdown page)
**Interaction captures:** off (workflow.ui_interaction_capture is false)

**Scope note:** Phase 27 is a backend/CLI/data phase (census tool, compat shims, registry write-back, CI lanes). The only user-facing surface is the generated documentation page `docs/user_guide/model_compatibility.md`. Pillars with no applicable surface are graded N/A rather than invented.

---

## Pillar Scores

| Pillar | Score | Key Finding |
|--------|-------|-------------|
| 1. Copywriting | 3/4 | Status wording is plain and pinned, but failure Details cells carry engineering jargon that violates the plain-language-for-biologists directive |
| 2. Visuals | 3/4 | Clean 3-column table with consistent hierarchy; long failure cells make rows visually unbalanced |
| 3. Color | N/A | No color surface (static Markdown; MkDocs theme provides styling, not this phase's code) |
| 4. Typography | N/A | No typography surface (Markdown; rendered by MkDocs Material theme) |
| 5. Spacing | N/A | No spacing surface |
| 6. Experience Design | 3/4 | Byte-identical regeneration, nav entry, and 54-test drift gate are excellent; page lacks inline footnote markers and per-row navigation aids |

**Scored surface overall: 9/12** (3 applicable pillars; N/A pillars excluded from total)

---

## Top 3 Priority Fixes

1. **Failure Details cells are engineering-internal, not plain language** (`docs/user_guide/model_compatibility.md:40-42,45,53,55`) — a biologist reading "torch.Tensor(list) stays on cpu while torch.arange lands on meta" or "type_vocab_size=2" gets no usable signal, contra the 2026-10-21 owner directive — move the technical cause into a collapsed/parenthetical suffix and lead each failure cell with a plain-language sentence ("This model's published code uses a feature the new library version removed; it cannot be loaded until the model authors update it").
2. **Footnotes 1/2 have no inline markers in the table** (`model_compatibility.md:62-63`) — the two provenance qualifiers (tested-on-5.17 / inferred-4) apply per-row but nothing in the table links them, so a reader scanning one row never sees the inference caveat — add a marker (e.g. `*` on import-verified rows, `†` on the page-wide "tested on 5.17.0" claim) wired to the footnotes; note this requires extending the generator + drift gate in lockstep.
3. **Long failure cells break table scannability** (`model_compatibility.md:41,53`) — the GENA-LM and MutBERT rows are 300+ character single cells, pushing the table layout wide in rendered MkDocs — cap Details at one short sentence and move full technical causes to a separate "Technical notes" section below the table (same generator change, keeps determinism since all content stays registry-derived).

---

## Detailed Findings

### Pillar 1: Copywriting (3/4)

**Good evidence:**
- Five pinned, plain status wordings ("Tested working", "Tested loading its code; full run inferred from the family model", "Works with DNALLM's built-in compatibility fixes (tested)", "Not usable with the current transformers", "Could not be checked") — consistent across all 49 rows.
- Intro introduces transformers in plain terms ("a widely used AI library called *transformers*, which is the software that loads and runs the models") and never assumes census/taxonomy vocabulary.
- Count sentence (line 8) matches the table composition (19+21+1+6+2 = 49).
- Honest labeling honored: no unqualified "compatible with transformers 4 and 5" claim anywhere in the table (prohibited by the drift gate).
- Provenance footnotes include the owner-approved Chinese originals verbatim.

**Failures (WARNING):**
- Failure Details cells leak engineering jargon to the biologist audience: `model_compatibility.md:40` ("builds its ALiBi tensor eagerly in __init__ … torch.arange lands on meta"), `:41` ("persisted bert.embeddings.token_type_ids buffer … type_vocab_size=2"), `:42` ("block-sparse attention … deepspeed package"), `:53` ("input_shape = input_ids.size()[:-1] ('# modified' in the repo's own source)"), `:55` ("imports olmo.config.ModelConfig"). 27-04's key decision deliberately improved the Omni-DNA cell but the fix pattern was not extended to the other five failure rows.
- Footnote language is good but unanchored (see Pillar 6).

### Pillar 2: Visuals (3/4)

**Good evidence:**
- Clear focal structure: title → plain intro → count summary → table → footnotes → provenance line.
- Consistent 3-column schema (Model / What we found / Details); status column reads as the visual hierarchy anchor; repo ids are consistently placed in Details.
- GENERATED header comment (line 1) prevents manual-edit confusion.

**Findings (WARNING):**
- Rows 41 and 53 have very long Details cells; the visual rhythm of the table breaks — status column becomes hard to scan when one cell dominates the row height.
- "Details" column conflates two kinds of content (repo id vs failure cause vs family representative) without typographic separation; backtick code spans help but the mixing is consistent-by-accident, not by design.

### Pillar 6: Experience Design (3/4)

**Good evidence (strong for a docs surface):**
- Byte-identical regeneration verified live during this audit: `python -I scripts/census_compat.py --emit-docs` output `cmp`-identical to the committed page; drift gate `tests/models/test_compat_docs_consistency.py` passes 54/54 in 0.30s.
- MkDocs nav entry present (`mkdocs.yml:68`, under User Guide after Preset Model Loading) — the page is reachable.
- Footer states the source of truth and the overwrite-on-regeneration contract — correct expectation-setting for a reader who might edit.
- Fail-closed everywhere: generator fails on missing verdicts/unknown statuses; gate fails (never skips) on table-structure damage.

**Findings (WARNING):**
- Footnotes 1 and 2 qualify per-row claims ("tested on 5.17.0", "4-side inferred") but carry no inline markers — a reader landing on a single row via search misses the inference caveat entirely. This is the honesty-labeling weakest link on the page.
- No anchors/deep links per model row (MkDocs generates table-row anchors unreliably); a biologist sharing "the GENA-LM result" has no stable link target. Minor for a 49-row page.
- Six "not working" models get no next-step guidance (e.g. "use a tested alternative from this table" or issue link) — the page informs but does not route the user to an action.

### Pillars 3, 4, 5: N/A

Static Markdown rendered by the pre-existing MkDocs Material theme; this phase introduced no color, typography, or spacing surface. No findings invented.

---

## Registry Safety Audit

`components.json` absent — shadcn not initialized. No third-party UI registries in scope. **Registry audit: skipped (no shadcn).**

---

## Files Audited

- `docs/user_guide/model_compatibility.md` (primary audited surface, read in full)
- `scripts/census_compat.py` (`--emit-docs` regeneration executed; byte-identical output confirmed)
- `tests/models/test_compat_docs_consistency.py` (executed: 54 passed)
- `mkdocs.yml` (nav entry, line 68)
- `dnallm/models/model_info.yaml` (tf_compat_status field presence spot-check)
- Phase artifacts: `27-CONTEXT.md`, `27-01..27-04-SUMMARY.md`, `27-01..27-04-PLAN.md`
