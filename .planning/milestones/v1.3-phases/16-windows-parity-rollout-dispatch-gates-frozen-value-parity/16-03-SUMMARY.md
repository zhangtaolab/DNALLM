---
phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity
plan: 03
subsystem: testing
tags: [frozen-pairs, fixtures, bedtools, numpy, jaccard, exon-f1, loj-intersect, selection-md, windows-parity, provenance]

requires:
  - phase: 15-windows-parity-core-numpy-interval-evaluators
    provides: dnallm.utils jaccard / loj_intersect / parse_bed (locked import-pure numpy evaluators)
provides:
  - Committed real frozen interval pairs under tests/fixtures/genomic_intervals/ (CRE 51/94, Anno 394/526, loj 51/94 + 57-line bedtools loj oracle tsv) with same-session dual oracles and PROVENANCE.md
  - selection.md tight frozen-pair exon_f1 tolerance band row [0.752173912, 0.752173914] (= 692/920 pin) + byte-identical docs mirror
  - tests/test_frozen_pairs.py at tests/ root — the platform-agnostic suite that makes test-windows an executor of the numpy evaluators (WPAR-05's recomputable real-pair gate)
affects: [16-04, WPAR-05, WPAR-08, phase-19-docs-sweep]

actuals:
  tokens: 19245     # chars/4 over the realized diff (76982 diff chars)
  tasks: 3
  commits: 3        # MEASURED: git rev-list --count 630c1a0..HEAD

plan_head_before: 630c1a0ca40185edfd9a7814647b805ac08850b7
plan_head_after: bc2326bcfe0daaad404478580fcfb8beaeae7cd4

tech-stack:
  added: []          # no new packages — flash-linear-attention 0.5.2 is the project's own pinned fla extra, installed into the repo .venv to complete the faithful GPU env (see Deviations)
  patterns:
    - "Frozen-fixture same-session provenance: pair bytes + BOTH oracle families + provenance record produced in one dev-box session over the exact committed bytes (Pitfall 7), with the drift-vs-historical table surfacing any anchor movement (AS-1/AS-4)"
    - "Tight-band scoping: a ~1e-9 band row in selection.md is parsed by the frozen-pair tests ONLY (nightly keeps the four drift bands + evidence key — Pitfall 6), encoded as a permanent comment in the consuming test file"

key-files:
  created:
    - tests/fixtures/genomic_intervals/cre_pred.bed
    - tests/fixtures/genomic_intervals/cre_truth.bed
    - tests/fixtures/genomic_intervals/anno_pred.bed
    - tests/fixtures/genomic_intervals/anno_truth.bed
    - tests/fixtures/genomic_intervals/loj_a.bed
    - tests/fixtures/genomic_intervals/loj_b.bed
    - tests/fixtures/genomic_intervals/loj_bedtools_oracle.tsv
    - tests/fixtures/genomic_intervals/PROVENANCE.md
    - tests/test_frozen_pairs.py
  modified:
    - example/notebooks/plant_helixseek_shared/data/selection.md
    - docs/example/notebooks/plant_helixseek_shared/data/selection.md
    - .gitignore

key-decisions:
  - "Regeneration session reproduced ALL historical selection.md anchors exactly (jaccard 0.32472687098490116 / 15872 / 48878; counts 526/394/346/48/180; exon_f1 692/920; genes_above_floor 59) — AS-1's anchor branch is ACTIVE and the frozen suite pins the historical counts as baked literals"
  - "Tight band [0.752173912, 0.752173914] (~2e-9 wide, 9-decimal endpoints) centered on the measured 692/920 rational; consumed by tests/test_frozen_pairs.py ONLY"
  - "Band row label 'Anno pooled exon-F1 (frozen pair)' (OQ1 recommendation) — collision-probed both directions against every existing parser label (showcase _parse_bands substrings, CRE cell 21 / Anno cell 27 startswith)"
  - "Scoped .gitignore exception !tests/fixtures/genomic_intervals/*.tsv — the plan's 'no .gitignore edit' premise was wrong for *.tsv (Rule 3); follows the repo's committed-fixtures-by-exception pattern"
  - "loj A-side gene labels derived from Anno truth mRNA spans (midpoint containment) with contiguous-run renumbering — gene-grouped A order by construction; B side is a byte-copy of the session's cre_truth.bed"

requirements-completed: [WPAR-05, WPAR-08]

coverage:
  - id: D1
    description: "Frozen interval pairs + same-session dual oracles + PROVENANCE.md committed atomically (8 files, one commit) — CRE bedtools 6-dp jaccard 0.324727 + numpy full precision 0.32472687098490116; Anno triple tp=346 fp=48 fn=180 with exon_f1 = 692/920; complete 57-line bedtools intersect -loj oracle; all historical anchors reproduced exactly with the drift table recording it"
    requirement: WPAR-05
    verification:
      - kind: unit
        ref: "tests/test_frozen_pairs.py#test_frozen_cre_pair_jaccard_dual_oracle_and_band"
        status: pass
      - kind: unit
        ref: "tests/test_frozen_pairs.py#test_frozen_anno_pair_exon_counts_rational_and_tight_band"
        status: pass
      - kind: unit
        ref: "tests/test_frozen_pairs.py#test_frozen_loj_pair_byte_parity_and_consecutive_null_sextet"
        status: pass
      - kind: other
        ref: "plan Task-1 verify: git ls-files count == 8, all six BEDs parse via parse_bed (51/94/394/526/51/94), oracle 57 >= 51 A rows, provenance token grep + historical-anchor grep (7 hits)"
        status: pass
    human_judgment: false
  - id: D2
    description: "selection.md gains exactly ONE band-table row (frozen-pair exon_f1 tolerance, >=6-decimal endpoints, centered on measured value) with the docs twin byte-identical in the same commit; every existing parser label still matches exactly one row"
    requirement: WPAR-08
    verification:
      - kind: other
        ref: "plan Task-2 verify (corrected section isolation): 5 rows, band-interval regex, both-direction collision probe (substring + startswith, one hit each)"
        status: pass
      - kind: other
        ref: "scripts/check_docs_sync.py -> OK exit 0; git diff one added line per copy"
        status: pass
    human_judgment: false
  - id: D3
    description: "tests/test_frozen_pairs.py at tests/ root — platform-agnostic (no markers/skips/subprocess/bedtools), bands parsed never copied, tight band consumed here only, parse guards fail red (D-06); census untouched (216/217), full fast lane green"
    requirement: WPAR-05
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest tests/test_frozen_pairs.py -q --no-header -> 4 passed"
        status: pass
      - kind: unit
        ref: "tests/test_frozen_pairs.py#test_band_parse_guard_fails_red_on_missing_row"
        status: pass
      - kind: other
        ref: "negative greps clean (no pybedtools/subprocess import, no pytest.skip/pytest.mark); ci.yml census triple grep = '216/217 tests collected (1 deselected)'"
        status: pass
      - kind: other
        ref: "full fast lane: pytest -m 'not slow' --no-cov -q -> 2539 passed, 1 skipped, 75 deselected, exit 0"
        status: pass
    human_judgment: false

duration: 42 min
completed: 2026-10-13
status: complete
---

# Phase 16 Plan 03: Frozen-Value Parity — Pairs, Band & Suite Summary

**Real frozen interval pairs committed with same-session dual oracles (every historical anchor reproduced exactly — jaccard 0.324727/0.32472687098490116, exon_f1 692/920, tp/fp/fn 346/48/180), the tight exon_f1 band pinned in selection.md, and the platform-agnostic frozen-pair suite live on every CI leg including test-windows.**

## Performance

- **Duration:** 42 min (GPU regeneration session: CRE scan ~10 min, Anno both-strand scan ~11 min, plus model downloads)
- **Started:** 2026-10-13T16:28:21Z
- **Completed:** 2026-10-13T17:10:22Z
- **Tasks:** 3
- **Files modified:** 12 (9 created, 3 modified)

## Accomplishments
- One dev-box session regenerated ALL three pairs from the selection.md pipeline — truth sides deterministic from the committed TAIR10 slices, pred sides from GPU model runs (torch 2.11.0+cu130, transformers 5.17.0, fla 0.5.2, registry-pinned checkpoints) — and measured BOTH oracle families over the exact committed bytes: bedtools v2.31.1 (runner-cached prefix) AND the numpy evaluators
- The session reproduced every historical anchor EXACTLY (AS-1 anchor branch): the regenerated pred sides are oracle-identical to the lost originals; numpy loj output is byte-identical to the bedtools oracle on the real pair (57/57 lines)
- selection.md band table gained exactly one row — `Anno pooled exon-F1 (frozen pair) | = 692/920 | [0.752173912, 0.752173914] | 0.7522` — collision-probed against every existing parser label in both directions; docs twin byte-identical in the same commit
- tests/test_frozen_pairs.py born at tests/ root (4 tests, phase-15 posture): CRE dual-oracle jaccard + parsed loose band; Anno exact integer triple + 692/920 rational + parsed TIGHT band (this file its only consumer, Pitfall 6); loj byte-parity vs the committed oracle with the consecutive null-sextet and gene-contiguity shapes; a permanent D-06 red-capability test that fails naming the row when it is missing
- Zero census movement (216/217 triple unchanged), zero ci.yml/expected_skips.yaml/pyproject edits, phase-15 anchor file byte-identical, full fast lane green (2539 passed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Regeneration session — pairs + dual oracles + PROVENANCE (8 files, ONE atomic commit)** - `a5c07cc` (test)
2. **Task 2: selection.md frozen-pair exon_f1 band row + mirror** - `92ac413` (docs)
3. **Task 3: tests/test_frozen_pairs.py — platform-agnostic frozen-pair suite** - `bc2326b` (test)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified
- `tests/fixtures/genomic_intervals/{cre,anno}_{pred,truth}.bed, loj_a.bed, loj_b.bed` - the committed real pairs (51/94 and 394/526 rows; pred sides GPU-derived, truth sides deterministic)
- `tests/fixtures/genomic_intervals/loj_bedtools_oracle.tsv` - complete bedtools intersect -loj output (57 lines, 12 columns), git-tracked via the scoped .gitignore exception
- `tests/fixtures/genomic_intervals/PROVENANCE.md` - session env (bedtools v2.31.1 + prefix, torch/transformers/fla/numpy, registry-pinned checkpoints, box), per-pair commands, every oracle value, loj construction rule, and the drift-vs-historical table (all REPRODUCED EXACTLY)
- `example/notebooks/plant_helixseek_shared/data/selection.md` (+ `docs/` mirror) - one additive band-table row; nothing else changed
- `tests/test_frozen_pairs.py` - the WPAR-05 suite executing the numpy evaluators over the committed pairs on every leg
- `.gitignore` - `!tests/fixtures/genomic_intervals/*.tsv` exception (Rule 3 deviation)

## Decisions Made
- Recorded in frontmatter key-decisions; additionally: the comparison-layer trailing-float tolerance constant is 1e-9 (session measured 0.0 delta on numpy 2.5.3; CI pins 2.2.0; a single-base intersection change moves jaccard by >= 2e-5, an order of magnitude above the constant)
- loj pair sourced from the same session's CRE peaks (A) and truth DHS rows (B) per AS-2/A6 — no rice.uga.edu access anywhere in the plan

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Scoped .gitignore exception for the oracle tsv**
- **Found during:** Task 1 (staging)
- **Issue:** The plan asserts "BED/GFF data are NOT gitignored — verified .gitignore:70-90; no .gitignore edit", but `.gitignore:90` ignores `*.tsv` — `loj_bedtools_oracle.tsv` could not be tracked, failing the hard 8-tracked-files requirement (the plan's own `<fails_when>` names the untracked-fixture loss)
- **Fix:** Added `!tests/fixtures/genomic_intervals/*.tsv` immediately after the `*.tsv` pattern (gitignore last-match-wins), mirroring the repo's existing `!tests/interpret/fixtures/**/*.fasta` committed-fixtures-by-exception precedent; probe-verified with `git check-ignore`
- **Files modified:** .gitignore
- **Verification:** `git ls-files tests/fixtures/genomic_intervals/ | wc -l` == 8; Task-1 verify green
- **Committed in:** a5c07cc (same atomic fixture commit — the tsv cannot be staged without it)

**2. [Rule 3 - Blocking] flash-linear-attention 0.5.2 installed into the repo .venv**
- **Found during:** Task 1 (session launch precondition)
- **Issue:** The plan precondition ("GPU env up on this box, verified at plan time: repo .venv torch 2.11.0+cu130") was incompletely verified — the repo .venv lacked `fla`, and the D-16 guard makes absence a HARD error (PlantHelixSeek silently falls back to a positionally-dead non-KDA path — the pred sides would have been garbage, silently, the worst failure mode for a frozen-truth regeneration)
- **Fix:** `uv pip install --python .venv/bin/python "flash-linear-attention>=0.5.2,<0.6"` (bare, no backend extras, per pyproject's own warning) — the project's own pinned `fla` extra (pyproject.toml:185-187); version 0.5.2 already proven legitimate on this exact box by the nightly runner venv. Not a package-discovery install (T-16-SC's "zero new packages" holds — no new dependency entered any manifest)
- **Files modified:** .venv only (no repo file)
- **Verification:** `find_spec("fla")` resolves; the session's exact anchor reproduction is itself end-to-end evidence the KDA path ran
- **Committed in:** n/a (environment, not repo state)

**3. [Rule 1 - Bug] Plan Task-2 verify snippet swept the whole file tail**
- **Found during:** Task 2 (verify)
- **Issue:** `text.split('## Thresholds and tolerance bands')[1]` collects every `|` row AFTER the heading — including the committed-artifacts, ranking, and audit tables — so the `len(rows) == 5` assertion fails even against the pre-edit 4-row file (a plan-verify defect, not an implementation defect)
- **Fix:** Corrected the section isolation to cut at the next `## ` heading (measuring exactly what the acceptance criteria state); added a both-direction collision probe (substring + startswith) on top
- **Files modified:** none (verify-command correction)
- **Verification:** Corrected probe prints BAND ROW OK; full collision probe green
- **Committed in:** n/a

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 verify-snippet defect)
**Impact on plan:** Both Rule-3 fixes were required to satisfy the plan's own hard requirements (8 tracked files; faithful pred sides). No scope creep; the third is a measurement correction only.

## Issues Encountered
- The generation process aborted at its own loj gene-contiguity probe (my session script asserted per-row label uniqueness — `len(list) == len(set)` — instead of run contiguity) AFTER all six pair files were written and BEFORE any oracle ran. The A file was verified correctly grouped by inspection (`uniq -c`: 21 labels, one run each); the probe was fixed (itertools.groupby run count) in the completion script, which then measured ALL oracles over the untouched bytes in the same session/env — Pitfall-7 substance intact (PROVENANCE.md discloses the two-process session structure honestly). A second latent bug in the same script's informational block (BedRow `[:2]` slice yields `(chrom, start)`) was fixed on disk and never reached.
- `pytest.Failed` does not exist in this pytest version; switched to `from _pytest.outcomes import Failed` (the showcase gate-contract precedent) before the first green run.

## TDD Note (Task 3, tdd="true")
Classic pre-implementation RED is structurally pre-satisfied by the plan's own sequencing: Tasks 1-2 ARE the implementation (Task 3's precondition names them), so the suite is green on arrival by design. RED capability is permanently embodied in `test_band_parse_guard_fails_red_on_missing_row`, which constructs the missing-row state (row-stripped tmp copy) and asserts the D-06 fail-red path raises `Failed` naming the row. The suite's one real pre-green failure (AttributeError on `pytest.Failed`) was a test-code defect, fixed before commit.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WPAR-05's recomputable real-pair gate is live on every CI leg (test-windows included); WPAR-08's mirror discipline held (check_docs_sync green)
- 16-04 (verification/evidence plan) can consume: the frozen-pair suite as the test-windows executor evidence, PROVENANCE.md as the provenance record, and the unchanged census (216/217)
- REQUIREMENTS.md marking for WPAR-05/WPAR-08 is shared-ID-gated until every declaring plan finishes (16-01 declared WPAR-08 too)
- The nightly lanes re-execute the four drift bands + evidence key unchanged — the tight band is invisible to them by design (Pitfall 6)

---
*Phase: 16-windows-parity-rollout-dispatch-gates-frozen-value-parity*
*Completed: 2026-10-13*

## Self-Check: PASSED

- All 9 key-files exist on disk (FOUND each)
- All 3 task commits (a5c07cc, 92ac413, bc2326b) are ancestors of HEAD on this branch
- `check evaluation-scope --plan 16-03 --commits-only`: resolved, 3 commits on this branch
- Re-ran all acceptance criteria live: Task-1 verify (8 tracked, parse_bed 51/94/394/526/51/94, oracle 57>=51, provenance greps), Task-2 verify corrected probe + check_docs_sync exit 0, Task-3 suite 4 passed + negative greps clean + census triple 216/217 + full fast lane 2539 passed / 1 allowlisted skip; anchor file tests/test_genomic_intervals.py byte-identical to plan base
