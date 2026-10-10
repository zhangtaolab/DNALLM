---
phase: 261010-wx2
plan: 01
subsystem: testing
tags: [bedtools, jaccard, numpy, interval-arithmetic, parity-spike, windows-ledger-19, genomic-coords]
requires:
  - "frozen BEDs example/notebooks/plant_helixseek_shared/.scratch/{pred,truth}.bed (read-only, never written or moved)"
  - "recorded ground truth: bedtools v2.31.1 intersection=15872 union=48878 jaccard=0.324727 (6-dp print) n_intersections=51"
provides:
  - "Spike evidence that a zero-dependency numpy interval-jaccard reproduces bedtools v2.31.1 exactly (VERDICT: PARITY on the frozen pair and 3008 corpus cases)"
  - "Re-runnable deterministic spike script (seed 20261010) + captured report.txt for the v1.3 jaccard=0.3247 truth-chain re-freeze decision"
affects:
  - "windows-ledger id 19 platform-split direction (Linux keeps bedtools, Windows numpy evaluator)"
  - "v1.3 re-freeze of the jaccard=0.3247 truth chain"
  - "future replacement of the NER example's pybedtools loj intersect (same interval-arithmetic precedent)"
tech-stack:
  added: []
  patterns:
    - "Per-chromosome int64 boundary sweep (+1 at starts, -1 at ends, starts tie-breaking first) -> coverage-exactly-2 = intersection, coverage>=1 = union"
    - "bedtools merge semantics in numpy: sort by (start, end), coalesce while next start <= current end (overlaps AND bookended adjacency)"
    - "Live full-precision oracle jaccard reconstructed as intersection/union integers — bedtools prints only 6 dp"
key-files:
  created:
    - .planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/spike_jaccard_parity.py
    - .planning/quick/261010-wx2-spike-numpy-interval-jaccard-parity-vs-b/report.txt
  modified: []
decisions:
  - "numpy + stdlib only (no pandas): per-chrom int64 boundary arrays carry the whole computation; argparse not click so the artifact runs outside the package"
  - "Gate jaccard compared against live full precision reconstructed from the live integers (bedtools' 6-dp print cannot resolve 1e-9); the printed value kept as informational anchor"
  - "Non-shuffled corpus cases written canonically (lexicographic chrom, start, end) so bedtools accepts first try; shuffled quarter probed raw, oracle re-invoked on sorted copies after the observed rejection; numpy always sorts internally"
metrics:
  duration: 21 min
  completed: 2026-10-10
  commits: 2
status: complete
actuals:
  tokens: 12974
  tasks: 2
  commits: 2
plan_head_before: bd1486e
plan_head_after: 454623e
---

# Quick Task 261010-wx2: numpy interval-jaccard parity spike vs bedtools Summary

Evidence-only spike proving a zero-dependency numpy interval-jaccard matches bedtools v2.31.1 exactly — VERDICT: PARITY on the frozen CRE-showcase pair (intersection/union integer-exact, jaccard delta 0.0) and across 3000 seeded randomized + 8 handcrafted cases with zero mismatches — so the windows-ledger id 19 zero-dependency direction stands for v1.3.

## What Was Built

`spike_jaccard_parity.py` (1205 lines, task artifact — never library code), stdlib + numpy only. Algorithm sketch:

1. `parse_bed3` — tab-separated BED3 rows, >= 3 columns, integer coords with `0 <= start < end`, grouped per chromosome by exact string key preserving file order; loud `ValueError` naming file + line on malformed rows (the `genomic_coords.py` silent-empty guard philosophy).
2. `merge_intervals` — sort by (start, end), coalesce while next start <= current end: overlaps AND bookended adjacency merge (bedtools merge, default distance 0).
3. `sweep_stats` — combined boundary sweep over concatenated start/end events (+1/-1, starts tie-breaking before ends at equal coordinates); with both sides merged, coverage never exceeds 2, so bases at coverage 2 = |A∩B| and coverage >= 1 = |A∪B|. Third return counts merged-A intervals overlapping merged-B — an n_intersections analogue kept informational only.
4. `jaccard_np` — per-chromosome merge + sweep, summed across chromosomes (one-sided chromosomes still count toward the union); jaccard = intersection / union.
5. `run_bedtools` — fixed argv `[bin, "jaccard", "-a", a, "-b", b]`, `check=True`, never shell; plus `resolve_bedtools` (shutil.which + `--version` echo + `--bedtools` override so a shadowed PATH binary is detectable).

Numpy-only rationale: per-chromosome int64 boundary arrays carry the entire computation, so pandas would add import weight with no work to do. CLI: `--frozen-only`, `--cases` (default 3000), `--seed` (default 20261010), `--bedtools`. Exit 0 iff VERDICT: PARITY.

## Frozen-Pair Table (numpy vs live bedtools vs recorded reference)

Sortedness findings (verified by the script, not assumed): pred.bed 51 rows, 1 chrom (Chr1=51), chrom-grouped=True, start-sorted-within-chrom=True; truth.bed 94 rows, 1 chrom (Chr1=94), chrom-grouped=True, start-sorted-within-chrom=True.

| metric | numpy | bedtools (live) | recorded ref | gate |
|---|---|---|---|---|
| intersection | 15872 | 15872 | 15872 | exact match (delta 0) |
| union | 48878 | 48878 | 48878 | exact match (delta 0) |
| jaccard | 0.32472687098490116 | 0.32472687098490116 (full, reconstructed int/int) | 0.324727 | PASS — delta 0.0 vs live full precision |
| n_intersections (informational) | 45 (merged-A analogue) | 51 (raw rows) | 51 | never gated |

Anchor delta vs the 6-dp reference 0.324727: `1.2901509882645712e-07` — pure bedtools print rounding (bedtools prints 6 dp; the live full-precision value is reconstructed as intersection/union from the live integers, which is what makes the locked 1e-9 gate meaningful). Frozen gates: PASS.

## Corpus Results

- **Handcrafted edge cases: 8 compared, 8 pass, 0 dropped.** Empty-side-A (bedtools accepted the empty file: `0 1500 0 0`), identical sets (jaccard 1.0), strict containment, bookended adjacency (zero intersection, coalesced union), fully disjoint, same coords under different chrom names (`Chr1` vs `chr1` — no intersection, exact string key), duplicate + staggered self-overlaps (merge semantics), shuffled row order — all integer-exact with jaccard delta 0.0.
- **Randomized property cases: 3000 compared, 3000 pass, 0 mismatches** (seed 20261010, default_rng; 6-pattern cycle mixed/dense/containment/adjacency/disjoint/duplicates; chrom pool Chr1..Chr4/chr1/1/scaffoldA; 1–4 chroms per case; per side 0–30 rows, never both empty; every 4th case written in shuffled order).
- **Max deltas:** |Δjaccard| vs live full precision `0.0`; max |Δintersection| 0 bases; max |Δunion| 0 bases. Informational |Δjaccard| vs bedtools' 6-dp print: `5.22047572615314e-07` (display rounding only).
- **Mismatches: none** — no `failures/` reproduction artifacts were produced.
- **Sort handling:** bedtools jaccard rejected non-lexicographically-sorted input (first observed rejection: `Error: Sorted input specified, but the file <tmpdir>/hand8_a.bed has the following out of order record`, rc=1); after that the oracle ran on pre-sorted copies for the remaining 749 shuffled cases (750 total, 1 raw probed, 0 raw-differed). Non-shuffled cases were written canonically sorted and accepted raw. The numpy side always sorts internally.

## Environment and Run Facts

- bedtools binary: `/home/linuxbrew/.linuxbrew/bin/bedtools` (PATH-resolved, symlink into Cellar 2.31.1), version `bedtools v2.31.1` — the exact version the frozen truth was computed with.
- Python 3.13.15 (repo dev venv `.venv`), numpy 2.5.3. Note: bare `python` on PATH is miniconda base (no numpy), so every run used `.venv/bin/python` explicitly — the plan's `<automated>` commands are otherwise identical.
- Full default run: RUNTIME 5.1s (3008 bedtools subprocess invocations).
- Determinism: a second full run is byte-identical to report.txt excluding the RUNTIME line (5.1s vs 5.0s).
- Evidence-only guarantee: `git status --porcelain -- dnallm tests example` empty at every verify gate; no pytest lanes touched (nothing under tests/ changed, so no test commands were run by this spike).

Verdict line, verbatim from report.txt:

```
VERDICT: PARITY
```

## Implications for Windows-Ledger id 19

The zero-dependency direction **stands for v1.3**: numpy interval arithmetic reproduces `bedtools jaccard` exactly — integer-exact intersection and union, jaccard delta 0.0 at full precision — on the frozen pair and across 3008 adversarial cases spanning every property that could plausibly diverge (containment, bookending, duplicates, chrom-name styles, empty sides, shuffled order). The platform split (Linux keeps bedtools, Windows gets the numpy evaluator, gated on this spike's parity) is evidence-backed and the v1.3 re-freeze of the `jaccard=0.324727` truth chain can proceed: numpy's full-precision value 0.32472687098490116 rounds to the recorded 6-dp anchor. Porting notes for the production evaluator: (1) merge must coalesce bookended adjacency and group chromosomes by exact string key — both behaviors are load-bearing for parity; (2) n_intersections is NOT portable as-is — bedtools counts raw A rows (51) while the natural numpy analogue counts merged-A intervals (45) on the frozen pair, so any production contract must either port bedtools' raw-row semantics or exclude that field; (3) bedtools requires lexicographically sorted input while the numpy evaluator does not — a behavioral freedom, not a parity risk.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Negative stagger produced an invalid BED start**
- **Found during:** Task 2 (first full 3000-case run)
- **Issue:** `_gen_duplicates` could shift a staggered interval below coordinate 0 (base start < 16 with a negative shift), producing `(start < 0, end)` rows that `parse_bed3` correctly rejected — the run crashed at case 2717 with `ValueError: invalid 0-based half-open interval (-11, 375)`.
- **Fix:** Bound the negative shift by the interval's start (`shift = -min(shift, start)`); rng draw counts are unchanged, so corpus determinism relative to the intended design is preserved.
- **Files modified:** `spike_jaccard_parity.py` (one line in `_gen_duplicates`)
- **Verification:** Full rerun: 3000/3000 pass, VERDICT: PARITY, exit 0; determinism re-verified byte-identical excluding the RUNTIME line.
- **Committed in:** 454623e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Generator correctness fix only; no gates loosened, no scope creep. The parser's loud guard did exactly what it was designed to do.

## Task Commits

1. **Task 1: numpy jaccard core + frozen-pair parity** — `ef8a0aa` (feat)
2. **Task 2: seeded randomized + edge-case harness, report, verdict** — `454623e` (test)

SUMMARY/STATE/PLAN docs handled by the orchestrator per quick-task flow.

## Self-Check: PASSED

All three artifacts exist on disk (spike_jaccard_parity.py, report.txt, 261010-wx2-SUMMARY.md); both task commits (ef8a0aa, 454623e) are ancestors of HEAD; `git status --porcelain -- dnallm tests example` empty.
