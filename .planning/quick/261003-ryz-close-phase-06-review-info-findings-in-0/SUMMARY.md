# Quick Task 261003-ryz: Close Phase-06 Review Info Findings IN-01..06 Summary

**Six surgical info-finding fixes: deterministic pyfastx handle/.fxi lifecycle in fetch_sequence, ASCII-strict chrom digits, contamination-free import-purity test, quote-normalized Anno registry line, exact bracket-member fla assertion, and an evidence-decided .scratch/ ignore removal (dispositions open 6 -> 0)**

## Performance

- **Duration:** 9 min (12:20-12:29 UTC)
- **Started:** 2026-10-03T12:20:23Z
- **Completed:** 2026-10-03T12:29:35Z
- **Tasks:** 6/6
- **Files modified:** 5 code/test files + 2 planning docs

## Task Commits

1. **Task 1 (IN-01): fetch_sequence path branch - handle release + .fxi cleanup** - `aaf6308` (fix)
2. **Task 2 (IN-02): normalize_chrom ASCII-only bare-numeric branch** - `c19999a` (fix)
3. **Task 3 (IN-03): purity test restores the package attribute** - `3662ea5` (test)
4. **Task 4 (IN-04): Anno label_names double-quoted** - `7790920` (chore)
5. **Task 5 (IN-05): exact bracket-member fla assertion** - `2a0ba40` (test)
6. **Task 6 (IN-06 + closure): local .scratch/ ignore removed; dispositions flipped** - `188a4f6` (chore)

Plan commit: `c8feaa7` (planner). Branch `phs` pushed (`0ea9213..188a4f6`) before this docs commit; final push follows it. No attribution trailers anywhere.

## IN-06 Evidence (verbatim, mandatory for either outcome)

The planner's masked/unmasked chain was reproduced live at execution time (2026-10-03, cwd repo root); the DELETE branch was taken because the reproduction confirmed the planner's evidence.

**1. Unmasked control** (local file present):

```
$ git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/foo.py
example/notebooks/plant_helixseek_shared/.gitignore:1:.scratch/	example/notebooks/plant_helixseek_shared/.scratch/foo.py
exit: 0
```

**2. Masked probe** (local `.gitignore` copied to a byte backup outside the repo, then `mv` aside):

```
$ git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/foo.py
.gitignore:60:.scratch/	example/notebooks/plant_helixseek_shared/.scratch/foo.py
exit: 0
$ git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/
.gitignore:60:.scratch/	example/notebooks/plant_helixseek_shared/.scratch/
exit: 0
```

Restore verified non-destructive: `cmp example/notebooks/plant_helixseek_shared/.gitignore /tmp/ryz-gitignore-backup` -> identical.

**3. Decision (DELETE branch):** the unanchored root directory pattern ignores ALL contained files regardless of extension, so the 06-01 plan's claim that root patterns do not cover `.py` files inside `.scratch/` is disproven. The local entry (the file's only line) was removed via `git rm`.

**4. Post-removal re-check:**

```
$ git check-ignore -v example/notebooks/plant_helixseek_shared/.scratch/foo.py
.gitignore:60:.scratch/	example/notebooks/plant_helixseek_shared/.scratch/foo.py
exit: 0
```

`git status --porcelain example/notebooks/plant_helixseek_shared/` showed only the `.gitignore` deletion (no scratch files surfacing as trackable); root `.gitignore` diff count 0 (byte-untouched, line 60 intact).

**06-01 plan-claim correction:** 06-01-PLAN.md / 06-01-SUMMARY.md were NOT rewritten (phase history is immutable). This SUMMARY is the record: the local `.scratch/` ignore was redundant with root `.gitignore:60`, and the root pattern does cover `.py` files inside `.scratch/`.

## Accomplishments

- **IN-01 (`aaf6308`):** `fetch_sequence` path branch records sidecar existence before constructing `pyfastx.Fasta`, drops the reference in a `finally` (no `close()` exists; also covers error paths where a traceback keeps the frame alive), and unlinks only a `.fxi` it created under `contextlib.suppress(OSError)`. Pre-existing sidecars and caller-owned open indices are never touched. RED: the no-sidecar test failed before the change. 3 same-change tests (no sidecar + FASTA untouched; pre-existing sidecar preserved; fd count stable). Docstring documents the read-only contract.
- **IN-02 (`c19999a`):** bare-numeric branch guard is now `name.isascii() and name.isdigit()` - full-width/superscript/Arabic-Indic digits raise `ValueError` in both styles instead of being renamed to a lookalike chromosome. chr-prefixed branch untouched (`_CHROM_RE` token is already ASCII-only). ASCII behavior byte-identical. RED -> GREEN, 26 tests.
- **IN-03 (`3662ea5`):** the import-purity test saves/restores `getattr(dnallm.utils, "genomic_coords")` alongside `sys.modules`; new `test_import_purity_leaves_single_live_module` invokes the purity test directly and asserts the package attribute IS the `sys.modules` entry. RED against the old restore (attribute stayed rebound), green after.
- **IN-04 (`7790920`):** the Anno `label_names` line re-quoted single -> double, content/order verbatim. Tripwires: one-line diff (2 changed lines total), `head -1649` sha256 byte-identical above the Anno block, single trailing newline after `threshold: 0.5`, legacy single-quoted tRNAPointer line 1447 untouched, double-quoted count 146 -> 147, registry structure tests (3) green in the same change, YAML parses.
- **IN-05 (`2a0ba40`):** new `_meta_extra_names` parser (anchored `dnallm[...]` fullmatch, comma-split, whitespace-strip, drop empties; non-meta specs -> `[]`); `test_fla_reachable_from_all` now requires `fla` as an exact parsed member. Substring-collision regressions (`flash-attn`, `fla-core`, `mamba-fla`), whitespace tolerance, and non-meta specs covered by 3 parser tests with no tomllib/skipif; the WR-05 tomllib guard on pyproject-reading tests untouched.
- **IN-06 (`188a4f6`):** decided by the committed evidence above; `06-REVIEW-DISPOSITION.md` rows IN-01..06 flipped `fixed` with per-finding commit hashes, frontmatter dispositions fixed, `open: 0`.

## Files Created/Modified

- `dnallm/utils/genomic_coords.py` - path-branch handle/sidecar lifecycle + ASCII digit guard + docstrings
- `tests/utils/test_genomic_coords.py` - 5 new tests (3 IN-01, 1 IN-02, 1 IN-03) + purity-test restore
- `dnallm/models/model_info.yaml` - one line: Anno label_names quotes
- `tests/models/test_plant_helixseek_fla_kernels.py` - parser helper + rewritten assertion + 3 parser tests
- `example/notebooks/plant_helixseek_shared/.gitignore` - deleted (IN-06 DELETE branch)
- `.planning/phases/06-model-registry-showcase-data-curation/06-REVIEW-DISPOSITION.md` - IN rows fixed, open: 0
- `.planning/STATE.md` - Quick Tasks row

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 4 verify one-liner assumed a wrong YAML top-level key**
- **Found during:** Task 4 (IN-04)
- **Issue:** the plan's parse check indexed `d['models']`; the registry's top-level keys are `pretrained`/`finetuned` -> `KeyError: 'models'`.
- **Fix:** ran the same check iterating both real sections (yaml parses, 1643 label entries). All substantive gates (tests, one-line diff, quote counts, prefix sha, legacy-line survival, tail bytes) passed unchanged.
- **Files modified:** none (verify-command adaptation only).
- **Committed in:** n/a (verification step).

**2. [Rule 1 - Bug] Ruff flagged the IN-02 fixture's fullwidth digits and the suppression form**
- **Found during:** Task 2
- **Issue:** `RUF001 ambiguous-unicode-character-string` fires on the lookalike fixture digits (they are the point of the test); this ruff 0.16.9 preview config also rejects `# noqa:` in favor of `# ruff: ignore[...]` with rule names, not codes.
- **Fix:** reworded the comment to drop the literal fullwidth char and applied ruff's canonical directive `# ruff: ignore[ambiguous-unicode-character-string]` on the fixture line (settled via `ruff check --fix`). Also reformatted the Task 5 long assert message per the formatter.
- **Files modified:** `tests/utils/test_genomic_coords.py`, `tests/models/test_plant_helixseek_fla_kernels.py`
- **Verification:** `ruff format --check` + `ruff check` green on all changed files.
- **Committed in:** part of Task 2 (`c19999a`) and Task 5 (`2a0ba40`) commits.

**Total deviations:** 2 auto-fixed (1 blocking verify-command adaptation, 1 lint-form fix).
**Impact on plan:** None on scope or outcomes - both are mechanical; all plan gates green.

## Issues Encountered

None beyond the deviations above. All per-task verify gates, the final combined suite (36 passed across the three touched test files), ruff format/check, and the IN-06 evidence chain ran green. `tests/models/test_plant_helixseek_smoke.py` was never executed, per the plan constraint.

## Next Steps

- Phase 06 review debt fully closed: all 5 warnings (r73) and all 6 info findings (this task) `fixed`, `open: 0`.
- Phase 7 (PlantHelixSeek Showcase Notebooks) planning can proceed; note for Phase 7 per-gene metric code (from the review summary): the slice contains 6 CDS rows whose parent mRNA sits partly outside the locus - do not assume every truth CDS row's Parent is resolvable inside the slice.
