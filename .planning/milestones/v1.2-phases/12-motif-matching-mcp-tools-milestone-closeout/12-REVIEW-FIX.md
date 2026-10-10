---
phase: 12-motif-matching-mcp-tools-milestone-closeout
fixed_at: 2026-10-10T06:30:06Z
review_path: .planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-REVIEW.md
iteration: 2
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-10-10T06:30:06Z
**Source review:** `.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-REVIEW.md`
**Iteration:** 2
**Scope:** all (2 Warning findings from the iteration-2 review)

**Summary:**
- Findings in scope: 2
- Fixed: 2
- Skipped: 0

**Verification location:** `workflow.use_worktrees=false` — all edits,
commits, and verification ran in the MAIN CHECKOUT on branch `revision`
(no isolated worktree was created; numbers below are reproducible from
this tree at `4f5cb16`/`46b336f`).

**Verification evidence (targeted lane only, per owner directive):**
- `uv run --no-sync pytest tests/mcp/test_server_tools_v12.py
  tests/mcp/test_mutagenesis_tool.py tests/mcp/test_interpret_tool.py -q -m "not slow"`
  → **110 passed** (98 pre-existing + 2 WR-05 regression tests + 12 WR-06
  parametrized acceptance cases; the two named dna-pattern sites' existing
  invalid-character tests in `test_mutagenesis_tool.py` /
  `test_interpret_tool.py` still pass)
- `uv run --no-sync ruff format --check` + `uv run --no-sync ruff check` on
  `dnallm/mcp/server.py` and `tests/mcp/test_server_tools_v12.py` → clean
- Empirical pattern probe: all 12 accepted contig names (incl.
  `HLA-A*01:01:01:01`, `HLA-DRB1*07:01:01:01`) match; `chr1\n`, tab-bearing,
  leading-newline, space, `\r`, `#`-bearing, and empty strings all rejected

## Fixed Issues

### WR-05: `_CHROM_PATTERN` anchors with `$` — a single trailing `\n` is accepted as "whitespace-free"

**Files modified:** `dnallm/mcp/server.py`, `tests/mcp/test_server_tools_v12.py`
**Commit:** `4f5cb16` (`fix(12): WR-05 anchor CHROM/allele/DNA validation patterns with \Z not $`)
**Applied fix:** Replaced the `$` anchor with `\Z` at ALL FOUR sites of the
idiom, exactly as the review mapped them:
- `_CHROM_PATTERN` (`server.py:137`, was 134) — with a comment explaining
  why `\Z` (not `$`): `$` also matches just before a trailing newline,
  defeating the whitespace-free check on exactly the row-splitting character
- `allele_pattern` (`server.py:2532`, was 2527) — with the same rationale
  comment at the site
- `dna_pattern` × 3 (`server.py:1434/1617/1899`, was 1431/1614/1896 — every
  instance, in `_dna_mutagenesis`, `_dna_interpret`, `_ism_scan`)
Regression tests added (kernel `assert_not_called()` on both):
- `test_zero_shot_trailing_newline_chrom_rejected` — `chrom="chr1\n"`
  returns the matchable `whitespace-free` / `variants[0].chrom` error
- `test_zero_shot_trailing_newline_ref_rejected` — `ref="A\n"` returns the
  matchable `variants[0].ref` / `ACGT` error
No committed test relied on `$` accepting a trailing newline; the change is
strictly more restrictive, and the existing invalid-character tests at the
dna-pattern sites still pass.

### WR-06: `_CHROM_PATTERN` omits `*` — legal HLA ALT contig names from hs38DH are over-rejected

**Files modified:** `dnallm/mcp/server.py`, `tests/mcp/test_server_tools_v12.py`
**Commit:** `46b336f` (`fix(12): WR-06 allow * in inline chrom names for hs38DH HLA ALT contigs`)
**Applied fix:** Added `*` to the character class — final combined pattern
`^[A-Za-z0-9_.:<>|()*-]+\Z` (the form the review prescribed for the two
findings together). The pattern comment now documents the HLA ALT contig
coverage (hs38DH: `HLA-A*01:01:01:01`) and why `#` stays rejected (no
mainstream reference uses it in CHROM; a leading `#` renders a data line
header-like to VCF parsers). New regression test
`test_zero_shot_mainstream_chrom_names_accepted` (parametrized, 12 cases)
asserts every mainstream name — conventional (`chr1`, `chrX`, `chrM`,
`chrEBV`), accession (`NC_000001.11`, `GL000220.1`), alt/decoy
(`chrUn_gl000220`, `chrUn_GL000220v1`, `chr1_KI270706v1_random`), plain
(`1`), and the HLA-star contigs — reaches the kernel (`call_count == 1`,
no `isError`).

## Skipped Issues

None — both in-scope findings were fixed.

---

_Fixed: 2026-10-10T06:30:06Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
