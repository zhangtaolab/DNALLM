---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "01"
subsystem: dnallm/interpret
tags: [motif-scanning, fimo, jaspar, pwm, exact-dp, bh-fdr, rev-10]
requires:
  - scipy.stats.false_discovery_control (scipy >= 1.15.2, already core)
  - dnallm.utils.sequence.reverse_complement (existing)
  - dnallm.inference.vep._load_reference (existing FASTA reader, tests only)
provides:
  - dnallm.interpret.motifs — FIMO-convention scanner (parse_meme/parse_cisbp/
    log_odds_matrix/pvalue_table/threshold_bits/scan/scan_single_strand/
    gc_background) + stdlib JASPAR REST client (fetch_meme_motif/search_motifs)
  - tests/interpret/ — 93-test lane (91 fast + 2 slow live) with committed fixtures
  - tests/expected_skips.yaml — jaspar-unreachable: typed-skip allowlist entry
  - HBG1/BCL11A golden-test harness (synthetic-verified, owner-input-gated)
affects:
  - CHANGELOG.md (## [Unreleased] ### Added — REV-10 bullet, shared surface with 12-02)
  - .gitignore (scoped !tests/interpret/fixtures/**/*.fasta negation)
  - tests/examples census pin — untouched, verified exact (208/217, 9 deselected)
tech-stack:
  added: []  # zero new dependencies (urllib/scipy/numpy only)
  patterns:
    - FIMO exact-DP recipe transcribed from MEME 4.8.1 source (PSSM_RANGE=100
      integer scaling -> column convolution -> reverse cumsum -> threshold
      inversion x/scale + w*offset)
    - retry-with-backoff house pattern (model.py:323-377 analog, time.sleep
      patched in tests)
    - parse-or-reject discipline (vep._load_reference analog) on all three inputs
key-files:
  created:
    - dnallm/interpret/__init__.py
    - dnallm/interpret/motifs.py
    - tests/interpret/test_motifs.py
    - tests/interpret/fixtures/meme_motif.txt
    - tests/interpret/fixtures/cisbp_motif.txt
    - tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml
    - tests/interpret/fixtures/hbg1_bcl11a/synthetic_window.fasta
    - tests/interpret/fixtures/hbg1_bcl11a/synthetic_motif.meme
  modified:
    - tests/expected_skips.yaml
    - CHANGELOG.md
    - .gitignore
decisions:
  - "D-01/D-02 reconciliation implemented and asserted by test: exact-DP null
    distribution with FIMO's own [0..100] integer scaling — the module docstring
    states honestly that 'exact' contrasts with empirical-null sampling, not
    with FIMO's documented quantization"
  - "BH is exactly one scipy.stats.false_discovery_control call per scan over
    the concatenated FULL window x motif x strand p-vector (D-03); analytic
    full-set q-values (p*n/2 for a top tie, p*n/3 at rank 3) are asserted,
    which distinguishes it from any per-window/per-motif correction"
  - "CIS-BP resolves as local-table parse only (no REST exists); nsites=0
    drives the probability-form pseudocount branch"
  - "P-vector clamped to [0,1] before BH — float cumsum drift left pv[0] a hair
    above 1.0 at w=100 and scipy rejects that (Rule 1 fix, regression-tested)"
  - "Background floor BG_FLOOR=1e-3 (renormalized) keeps log-odds finite when a
    letter is absent from the windows — MEME tools require strictly positive
    background frequencies"
  - "JASPAR_DEFAULT_RELEASE=2024 pinned on every search (both 2024 and 2026
    active; matrix sets differ); the paper's release is pending owner input"
metrics:
  duration: 24m
  completed: 2026-10-10
  tasks: 4
  tests: 93
status: complete
actuals:
  tokens: 22000  # chars/4 over the lane's realized diff (~88k chars created + shared-file hunks)
  tasks: 4
  commits: 5     # lane's own pathspec-verified commits (below); raw rev-list from
                # plan_head_before is 8 because the single shared tree also carries
                # concurrent sibling 12-02 commits (c4051c4, 0a4c7f5, fb60af3)
plan_head_before: 3bc072f95385f31d77f362ac2c5f9af6bc864dcf
plan_head_after: 07bd537
commits: 5
---

# Phase 12 Plan 01: FIMO-Convention Motif Scanner + JASPAR Client Summary

FIMO-convention motif scanner at `dnallm/interpret/motifs.py`: exact-DP p-value
calibration transcribed from MEME 4.8.1 (PSSM_RANGE=100 integer scaling, column
convolution, reverse cumsum), strict MEME/CIS-BP parsing, zero-order GC-matched
background, both-strand scanning with palindromic double reporting, p<1e-4 +
full-set BH q<0.05 via one scipy call, E = p x tested positions, and a stdlib
JASPAR REST client (canonical jaspar.elixir.no host, retry-with-backoff,
release-pinned searches) — plus the synthetic-verified, owner-input-gated
HBG1/BCL11A golden-test harness.

## OWNER INPUT REQUIRED (MOTIF-01 Fig 4a acceptance — the phase's tracked blocker)

The paper-exact HBG1/BCL11A golden assertion **cannot close without owner
input**. The harness, loader, and coordinate comparison are committed, green,
and drop-in ready — only fixture values are missing. Recorded as `pending:
true` in `tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml`:

| Input | Detail needed | Candidates / anchors |
|---|---|---|
| Fig 4a window coordinates | locus, flank size, assembly, verbatim | literature anchors only (not substitutes): BCL11A +58 enhancer GRCh38 chr2:60,495,219-60,495,336; HBG1/HBG2 promoter site ~-115 from TSS (chr11 ~5.27 Mb) |
| Motif ID | which matrix the paper used | MA2324.1 (w=7) or MA2504.1 — both JASPAR CORE BCL11A (live-verified 2026-10-10); or a CIS-BP PWM |
| JASPAR release | which release the paper searched | 2024 or 2026 (both active; matrix sets differ) |
| Tolerance policy | exact vs figure-derived | 0 bp when owner-supplied verbatim; +-2 bp if figure-derived |

Activation is fixture-files-only: replace `synthetic_window.fasta` +
`synthetic_motif.meme` with the owner window + frozen JASPAR MEME file, set
`pending: false`, fill `expected_hits`/`tolerance_bp` — zero harness code
changes (proven: the committed test drives manifest -> loader -> scan ->
tolerance comparison generically).

## Task Summary

| Task | Name | Commit | Result |
|---|---|---|---|
| 1 (tracer) | FIMO core slice: MEME parse, log-odds, exact-DP p-table, threshold, single-strand scan | 3d9c97b | 34 tests green; tracer verify re-run end-to-end green; facade byte-stable |
| 2 | Full scan semantics: both strands, GC background, BH full-set, E-values, CIS-BP parser, edges | bbedd25 | 38 verify-filter tests green; all five edge predicates asserted by name |
| 3 | JASPAR REST client + typed live skips + REV-10 CHANGELOG entry | bd165d8 | 17 tests green (mocked + live); yaml allowlist entry same-commit |
| 4 | HBG1/BCL11A golden harness (owner-input-gated, synthetic-verified) | 0380529 | 3 golden tests green; manifest records all pending inputs |
| — | Coverage hardening (parse variants, redirect guard, DP guards) | 07bd537 | motifs.py 97% -> 99% fast-lane |

## Verification Results

- Fast lane: `pytest tests/interpret -q -m "not slow"` — **91 passed, 0 failed**
- Slow live lane: `pytest tests/interpret -q -m slow` — **2 passed** (real
  jaspar.elixir.no round trips: MA2324.1 MEME fetch + BCL11A CORE search
  returning MA2324.1/MA2504.1); typed `jaspar-unreachable:` skips fire only
  when the host is unreachable, allowlisted in tests/expected_skips.yaml
- Per-module coverage: `dnallm/interpret/motifs.py` **99%** (374 stmts, 2 miss
  — the DP normalization invariant guard and the unreachable threshold branch,
  both documented defensive), `dnallm/interpret/__init__.py` **100%**; rows
  VISIBLE (not under any coverage omit glob)
- Facade stability: `git diff <base>..HEAD -- dnallm/__init__.py` — empty
  across all lane commits
- Census safety: `pytest tests/examples --collect-only -q -m "not giants" -k
  "not mcp_example"` — **208/217 tests collected (9 deselected)**, exactly the
  ci.yml Stage 0.5 pin
- Lint: ruff check + ruff format clean on dnallm/interpret/ and tests/interpret/
- Zero new dependencies (pyproject.toml untouched)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] P-vector clamped to [0,1] before the BH call**
- Found during: Task 2 (w=100 edge test)
- Issue: float accumulation in the reverse cumsum left pv[0] a hair above 1.0
  at w=100; scipy's false_discovery_control rejects values outside [0, 1]
  strictly, crashing the wide-motif edge case.
- Fix: `np.clip(all_p, 0.0, 1.0)` before the single BH call (comment in scan);
  the w=100 test is the regression proof.
- Files: dnallm/interpret/motifs.py; commit bbedd25.

**2. [Rule 3 - Blocking] .gitignore blanket `*.fasta` blocked the committed golden fixture**
- Found during: Task 4 commit
- Issue: the plan requires a committed FASTA stand-in; the repo-wide `*.fasta`
  ignore (meant for genomic data dumps) rejected it.
- Fix: scoped negation `!tests/interpret/fixtures/**/*.fasta` with a comment
  (61-byte hand-constructed regression input, not a data dump). A `git add -f`
  was rejected as it would leave the file silently fighting the rule.
- Files: .gitignore; commit 0380529.

**3. [Rule 1 - Bug] CIS-BP parser swallowed numeric pre-header rows as metadata**
- Found during: Task 2 verification
- Issue: a tab-separated data row before the `Pos A C G T` header parsed as a
  metadata entry (key "1") instead of being rejected.
- Fix: metadata keys must be word-like (contain a letter); numeric first fields
  raise the matchable expected-header ValueError.
- Files: dnallm/interpret/motifs.py; commit bbedd25.

### Plan-text interpretation notes (no code impact)

- The plan's census verification line ("`pytest tests/ --collect-only` still
  reports 208/217") is not literally satisfiable — `tests/` collects the whole
  suite. The actual ci.yml Stage 0.5 gate is scoped to `tests/examples` with
  the exact selector flags; that selector was run and matches the pin exactly.
- Test-side index/formula slips caught by the same-change red bar (GC-skew site
  at [3,10) not [4,11); floor denominator 1+BG_FLOOR not 1+2*BG_FLOOR; CIS-BP
  fixture consensus row; `MA0001.10` is VALID under the plan-locked
  `^MA\d{4}\.\d+$` grammar) — fixed within the task that introduced them.

## Auth Gates

None — JASPAR requires no authentication; live tests passed against the
canonical host.

## Known Stubs

None. The only intentionally pending artifact is the owner-input-gated golden
fixture VALUES (manifest `pending: true`) — the harness itself is fully
functional and verified; this is the tracked blocker above, not a stub.

## Threat Surface

No surface beyond the plan's threat_model. T-12-01/T-12-02/T-12-03 mitigations
implemented as specified and regression-tested: strict regex-anchored MEME
grammar rejecting unknown lines; 1 MB read cap; `MA\d{4}\.\d+` validation
before URL construction; https-only base URLs; redirect following disabled
(handler refusal unit-tested); no eval/exec on fetched content; per-call
arguments never compose the host.

## Self-Check: PASSED

- Files: all 8 created files exist on disk (checked).
- Commits: all 5 lane commits are ancestors of HEAD (checked:
  3d9c97b, bbedd25, bd165d8, 0380529, 07bd537).
- Facade diff empty; census pin exact; fast + slow lanes green.
