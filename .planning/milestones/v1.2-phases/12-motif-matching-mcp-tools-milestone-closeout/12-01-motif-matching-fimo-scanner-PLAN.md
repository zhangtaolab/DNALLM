---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/interpret/__init__.py
  - dnallm/interpret/motifs.py
  - tests/interpret/test_motifs.py
  - tests/interpret/fixtures/meme_motif.txt
  - tests/interpret/fixtures/cisbp_motif.txt
  - tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml
  - tests/interpret/fixtures/hbg1_bcl11a/synthetic_window.fasta
  - tests/interpret/fixtures/hbg1_bcl11a/synthetic_motif.meme
  - tests/expected_skips.yaml
  - CHANGELOG.md
autonomous: true
requirements: [MOTIF-01]
user_setup: []
estimate:
  tokens: 38000
  raw_tokens: 38000
  tasks: 4
  confidence: low
coupling_justified: >-
  CHANGELOG.md is the milestone's single sanctioned shared append surface (Phase 10 D-09
  mechanism): Wave-1 sibling 12-02 appends its own REV-11 bullet to ## [Unreleased] in the
  same window. Discipline that makes concurrent appends safe: re-read the file immediately
  before every edit, insert only a unique-anchor bullet under the existing ### Added heading,
  never edit sibling entries in place.
must_haves:
  truths:
    - "Scanning hotspot windows against a JASPAR/CIS-BP PWM yields a table with motif ID, start/end coordinates (forward-window frame), strand, score (bits, FIMO inversion), p-value, BH q-value (scipy.stats.false_discovery_control over the FULL window x motif p-vector, never per-sequence) and E-value (p x total tested positions)"
    - "The p-value for every hit comes from an exact dynamic-programming null distribution computed per FIMO/MEME convention (PSSM_RANGE=100 integer scaling, column-wise convolution under the GC-matched zero-order background, reverse cumulative sum) — and the module docstring states this calibration choice honestly, including the integer quantization (D-01)"
    - "Both strands are scored; a palindromic motif produces hits on BOTH strands at the same coordinates, both reported with their strand field — never deduplicated, never merged (FIMO convention)"
    - "A GC-matched background computed from the target windows changes the hit set versus a uniform background on GC-skewed synthetic windows; the embedded uniform 'Background letter frequencies' line in JASPAR MEME responses is ignored"
    - "A window shorter than the motif width is excluded cleanly with a count; a scan where zero motif tests survive p<1e-4 or BH q<0.05 returns an empty table that still reports the tested-position and tested-motif counts — never a vacuous error"
    - "A uniform (no score variation) PWM raises a matchable ValueError; a 1-column motif's p-table is analytically exact; a 2-column motif's p-table equals the hand-computed convolution"
    - "The JASPAR client fetches MEME-format matrices from a parameterized base URL defaulting to the canonical host https://jaspar.elixir.no/api/v1 with retry-with-backoff (max_try=3), a response size cap, and a matchable ValueError after exhausted retries"
    - "Live JASPAR tests that skip on unreachable networks use typed 'jaspar-unreachable:' reasons allowlisted in tests/expected_skips.yaml in the SAME change"
    - "The golden-test harness (fixture loader + coordinate-comparison logic) is green against a committed synthetic stand-in fixture; the paper-exact HBG1/BCL11A fixture drops in without harness changes once owner input (Fig 4a coordinates, motif ID, JASPAR release) arrives"
  artifacts:
    - "dnallm/interpret/__init__.py (minimal package init; NO dnallm/__init__.py re-export — facade byte-stable)"
    - "dnallm/interpret/motifs.py (constants with MEME-source provenance comments; parse_meme; CIS-BP table parser; log_odds_matrix; pvalue_table; threshold_bits; strand-aware scan; GC background; BH/E-value assembly; JASPAR client)"
    - "tests/interpret/test_motifs.py (DP anchors, background, strands, parsers, BH, client mocks, edges, golden harness)"
    - "tests/interpret/fixtures/ (meme_motif.txt shaped like the live-probed MA2324.1 response; cisbp_motif.txt; hbg1_bcl11a/ manifest + synthetic stand-ins)"
    - "tests/expected_skips.yaml (+ jaspar-unreachable: prefix entry, same change)"
    - "CHANGELOG.md ## [Unreleased] (+ REV-10 bullet, same commit as the code)"
  key_links:
    - "scan() -> scipy.stats.false_discovery_control(concatenated full-set p-vector, method='bh') — the single BH call (D-03)"
    - "strand '-' scoring -> dnallm.utils.sequence.reverse_complement (reuse, never re-implement)"
    - "fetch_meme_motif() -> parse_meme() -> scan() — client output feeds the same parser as local files"
    - "dnallm/interpret/motifs.py coverage row visible under coverage report (NOT under any omit glob)"
  prohibitions:
    - "No new dependencies (pyproject.toml dependency lists untouched; urllib/scipy/numpy only)"
    - "No re-exports added to dnallm/__init__.py"
    - "No numpy score-discretization or float-tuple DP — the calibration follows FIMO's own integer scaling (D-02 reading per RESEARCH Pitfall 1)"
    - "No per-sequence/per-window FDR — BH is called once over the full window x motif test set (D-03)"
    - "No default uniform background — the scan background is GC-matched from the target windows"
    - "No bare print in library code; comments in English; relative imports inside dnallm/"
    - "No example/ or tests/examples/ changes (census pin 208/217 must stay exact)"
    - "No eval/exec on any fetched or parsed content; no caller-supplied host composition in URLs"
    - "No guessing the paper's Fig 4a coordinates — the golden fixture stays owner-input-gated"
---

<objective>
MOTIF-01 (REV-10): a FIMO-convention motif scanner at `dnallm/interpret/motifs.py` —
JASPAR REST client (stdlib, parameterized base URL, `format=meme`), CIS-BP local-table
parser, exact-DP p-value calibration transcribed from MEME 4.8.1 source, GC-matched
zero-order background, both-strand scanning, p<1e-4 threshold, BH FDR over the full
window x motif set via `scipy.stats.false_discovery_control` — emitting a
motif-ID/coordinates/E-value table, plus the owner-gated HBG1/BCL11A golden-test harness.

Purpose: the paper revision's Fig 4a annotation (HBG1/BCL11A motif hits) must be
reproducible from the package, with the calibration choice (D-01 exact-DP) documented
honestly per REQUIREMENTS.
Output: new `dnallm/interpret/` package, its fast-lane test suite at the >=96% per-module
standard, committed synthetic + golden-harness fixtures, typed JASPAR network skips, and
the REV-10 CHANGELOG entry.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-RESEARCH.md
@.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-PATTERNS.md
@.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-CONTEXT.md
@dnallm/models/model.py
@dnallm/inference/vep.py
@dnallm/utils/sequence.py
@tests/expected_skips.yaml
</context>

<tasks>

<task type="tracer">
  <name>Task 1: FIMO core slice — MEME parse, log-odds, exact-DP p-value table, threshold, one-strand scan</name>
  <files>dnallm/interpret/__init__.py, dnallm/interpret/motifs.py, tests/interpret/test_motifs.py, tests/interpret/fixtures/meme_motif.txt</files>
  <read_first>
  - 12-RESEARCH.md: Pattern 1 (the five FIMO steps, every constant source-cited), Code Examples "FIMO-recipe calibration" (use verbatim as starting reference), Pitfall 1 (do not hand-roll a float DP)
  - 12-PATTERNS.md: "dnallm/interpret/motifs.py" section (Analog A retry shape is Task 3; Analog B parse-or-reject style applies now), "dnallm/interpret/__init__.py" section (datahandling init shape)
  - dnallm/inference/vep.py:451-501 (_load_reference — the parse-or-reject discipline: every malformed input raises a matchable ValueError naming the artifact)
  - .claude/CLAUDE.md (relative imports, Google docstrings, matchable ValueError, no print, ruff line 100)
  </read_first>
  <action>
  Create the new package per the CONTEXT discretion note (small `dnallm/interpret/__init__.py` in the datahandling analog shape — docstring only or minimal re-export list; NEVER referenced from `dnallm/__init__.py`). In `dnallm/interpret/motifs.py`, write the module docstring FIRST and make it the D-01 honesty statement: summary + numbered features + an explicit calibration paragraph naming (a) exact dynamic programming over the log-odds null distribution per the MEME/FIMO documented convention (D-01), (b) FIMO's own integer scaling to [0..100] with `PSSM_RANGE = 100` cited to MEME 4.8.1 `src/pssm.h:13` — "exact" contrasts with empirical-null sampling, not with FIMO's documented quantization (D-02 reconciliation per RESEARCH Pitfall 1), (c) pseudocount 0.1 scaled by background frequency, (d) zero-order GC-matched background, (e) p<1e-4 threshold and BH over the full test set (D-03). Define UPPER_SNAKE constants with provenance comments: PSSM_RANGE, FIMO_PSEUDOCOUNT, FIMO_P_THRESHOLD. Implement the core slice: `parse_meme(text)` — strict regex-anchored grammar (MOTIF line -> id/name, `letter-probability matrix:` header fields alength/w/nsites/E validated: alength==4, w bounded <=100, rows numeric in [0,1], nsites >= 0); the embedded `Background letter frequencies` line is parsed-and-ignored (it is JASPAR's stock uniform background — RESEARCH Pattern 2); malformed input raises matchable ValueError. `log_odds_matrix(freq_rows, nsites, bg)` — FIMO pseudocount formula with the nsites==0 probability-form branch. `pvalue_table(scores, bg)` — pure-Python column-wise DP (D-02): scale to [0..PSSM_RANGE] per RESEARCH Code Examples, convolve `nxt[k+s] += pdf[k]*bg[j]`, reverse-cumsum to `pv[x] = Pr(score >= x)`; uniform PWM (no variation) raises ValueError. `threshold_bits(pv, scale, offset, w)` — minimal x with p under threshold, inverted per `logodds.c:50`. A single-strand `scan` path scoring one window against one motif and emitting hit dicts {motif_id, start, end, strand, score_bits, p}. Commit the MEME fixture shaped like the live-probed MA2324.1 response head quoted in 12-RESEARCH Code Examples ("MEME version 4 / ALPHABET= ACGT / strands: + - / MOTIF MA2324.1 BCL11A / letter-probability matrix: alength= 4 w= 7 nsites= 6265 E= 0"). Tests: the hand-computable anchors from RESEARCH (uniform PWM ValueError; 1-column motif p=[1,0,0,0] vs bg=0.25 gives p exactly 1.0; 2-column motif pv equals the analytic convolution; threshold inversion at distribution edges including the everything-significant case and a p-table whose tail sits exactly at 1e-4); parse tests against the committed fixture plus malformed variants (bad alength, non-numeric row) asserting pytest.raises(ValueError, match=...); a docstring-content test asserting the calibration terms (exact/dynamic programming, PSSM_RANGE, pseudocount, zero-order) appear in __doc__ — this is the REQUIREMENTS "documented honestly" acceptance made checkable.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/interpret/test_motifs.py -q -k "parse or pvalue or threshold or log_odds or docstring"</automated>
    <fails_when>nonzero exit — e.g. the uniform-PWM case fails to raise, a DP anchor p-value deviates from the hand-computed convolution, or a malformed-MEME variant parses instead of raising the matchable ValueError</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: `dnallm/interpret/__init__.py` exists and `dnallm/__init__.py` is byte-identical to its pre-task state (facade stability)
  - Behavior: 1-column motif analytic p == 1.0; 2-column pv == analytic convolution; threshold inversion returns bits via x/scale + w*offset
  - Test: every anchors test passes; parse-or-reject raises ValueError matching a regex naming the artifact
  - Docstring: module __doc__ names exact-DP, PSSM_RANGE scaling, 0.1 pseudocount, zero-order background, BH full-set — asserted by test
  </acceptance_criteria>
  <done>Core calibration slice green: strict MEME parse, log-odds, exact-DP p-table, threshold inversion, single-strand scan producing hit dicts — proven by the hand-computed anchors; D-01/D-02 honesty statement in the docstring and asserted by test; no new dependencies.</done>
  <reversibility rating="reversible">New freestanding package; nothing existing imports it yet.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Full FIMO scan semantics — both strands, GC-matched background, BH full-set FDR, E-values, CIS-BP parser, edges</name>
  <files>dnallm/interpret/motifs.py, tests/interpret/test_motifs.py, tests/interpret/fixtures/cisbp_motif.txt</files>
  <read_first>
  - 12-RESEARCH.md: Pattern 2 (GC-matched background + both-strand mechanics and the exact test list), Pattern 1 step 5 + Assumption A4 (E-value = p x number of scanned positions), specless edge predicates (boundary/adjacency/empty for MOTIF-01)
  - dnallm/utils/sequence.py:37-66 (reverse_complement — reuse; note the lowercase-n passthrough quirk)
  - 12-CONTEXT.md decisions D-03 (BH over the FULL window x motif test set)
  - Task 1's landed motifs.py core slice
  </read_first>
  <action>
  Expand to the full scanner (per D-03): `gc_background(windows)` — per-letter counts over ALL target-window bases, one background per scan set so every motif's DP shares it (this is what makes p-values cross-motif comparable); both-strand scoring using `..utils.sequence.reverse_complement` — the reverse-complement PWM is scanned for strand "-" hits with coordinates mapped back to the forward-window frame; palindromic motifs report hits on BOTH strands at the same coordinates with the strand field distinguishing them — never deduplicated or merged (FIMO convention; adjacency edge predicate). The multi-motif scan entry (motif records x windows -> table): per motif per strand, hits above threshold_bits; p from the motif's pv table; then ONE call to `scipy.stats.false_discovery_control(concatenated_p_vector, method="bh")` over the FULL window x motif x strand test set — never per-sequence/per-window (D-03); q<0.05 reported; E-value column = p x total tested positions with the formula in the docstring (A4). Add the CIS-BP local-table parser (`Pos A C G T` rows) with the same parse-or-reject discipline and a committed `cisbp_motif.txt` fixture (CIS-BP is local-parse only — no REST exists; RESEARCH Pattern 3). Edge behavior (specless probes): windows shorter than the motif width are excluded with an explicit count in the result; a scan where zero tests survive threshold or BH returns an empty table that still reports tested-motif and tested-position counts — never a vacuous error; very wide motifs (w at the cap) run the DP without overflow. Tests: palindromic motif -> identical coordinate pairs on both strands, both rows present; non-palindromic motif -> strand-specific positions (revcomp-bug canary); GC-skewed synthetic windows -> uniform vs GC-matched background CHANGES the hit set; BH constructed p-sets with known q ordering (ties included); E-value arithmetic; short-window exclusion count; zero-pass empty table with counts; CIS-BP parse + malformed rejection.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/interpret/test_motifs.py -q -k "strand or background or bh or fdr or cisbp or evalue or edge"</automated>
    <fails_when>nonzero exit — e.g. a palindromic motif yields only one strand row, the uniform-vs-matched background hit sets are identical on GC-skewed windows, or the zero-pass case raises instead of returning the counted empty table</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: strand "-" coordinates map to the forward-window frame; exactly one false_discovery_control call per scan over the concatenated full-set p-vector
  - Behavior: background is one per scan set computed from all windows; embedded JASPAR background line never used
  - Test: all Pattern-2 test classes pass (palindromic, non-palindromic, GC-skew, BH ordering, E-value, short-window, zero-pass)
  - Edge: adjacency (both strands reported, no dedup) and empty (counted empty table) predicates asserted by name
  </acceptance_criteria>
  <done>The scanner emits the full motif-ID/coordinates/strand/score/p/q/E-value table under FIMO conventions with all five MOTIF-01 edge predicates covered; CIS-BP parses locally; BH is full-set only.</done>
</task>

<task type="auto">
  <name>Task 3: JASPAR REST client (stdlib) + typed live skips + REV-10 CHANGELOG entry</name>
  <files>dnallm/interpret/motifs.py, tests/interpret/test_motifs.py, tests/expected_skips.yaml, CHANGELOG.md</files>
  <read_first>
  - 12-RESEARCH.md: Pattern 3 (live-probed endpoints, verbatim response head, version pinning), Code Examples "JASPAR client", Pitfall 7 (typed skips + pinned release), Security Domain rows for JASPAR tampering/SSRF
  - dnallm/models/model.py:317-377 (download_model — the retry-with-backoff house pattern to copy: attempt counter, max_try, sleep between, final ValueError wrap)
  - tests/expected_skips.yaml (header + existing prefix entries — the shape to extend)
  - COVERAGE.md in this phase directory (the recorded JASPAR capability decisions: search + MEME download + collection filter INTEGRATE; releases/bulk/tax_id/other-formats OPT-OUT)
  - CHANGELOG.md ## [Unreleased] block (entry shape to match; re-read immediately before edit)
  </read_first>
  <action>
  Implement the stdlib JASPAR client in motifs.py per the parameterization decision (assumption-delta recorded: base URL is a parameter with a module-constant default): `JASPAR_BASE = "https://jaspar.elixir.no/api/v1"` (canonical host, provenance comment) and `JASPAR_DEFAULT_RELEASE` module constant with provenance comment (fixture reproducibility — every search passes release=). `fetch_meme_motif(matrix_id, *, base_url=JASPAR_BASE, timeout=30.0, max_try=3) -> str`: matrix_id validated against `^MA\d{4}\.\d+$` BEFORE URL construction; retry-with-backoff exactly per the model.py house pattern (time.sleep between attempts, tests patch time.sleep); response read size-capped; non-200 raises; exhausted retries raise `ValueError(f"JASPAR fetch failed for {matrix_id}: ...")` chained from the cause. `search_motifs(name, *, collection="CORE", release=JASPAR_DEFAULT_RELEASE, base_url=JASPAR_BASE, ...)`: paginated JSON parse returning matrix_id/name records; collection values from a closed set. Base-URL handling per the security seed: parameter must be an https URL; per-call arguments never compose the host (matrix id and query params only) — the SSRF-adjacent guard. Live slow-lane tests (marked `slow` from birth): fetch MA2324.1 MEME text and parse it through Task 1's parser; search name=BCL11A expecting the two CORE matrices; network absence produces typed skips with reason prefix `jaspar-unreachable:` — and tests/expected_skips.yaml gains the matching prefix entry in the SAME change (audit_skips.py runs on every CI leg). Mocked-urlopen fast-lane tests: success first try, fail-then-succeed (retry path, sleep patched and asserted), exhausted retries -> matchable ValueError, non-200 -> ValueError, invalid matrix_id rejected before any network. Append the REV-10 bullet to CHANGELOG.md ## [Unreleased]### Added in the same commit as the code: one dense bullet matching the existing entry style with the inline reviewer tag from the intake plan (.planning/research/261009-paper-revision-suite-plan.md REV-10 mapping); re-read CHANGELOG immediately before the edit; append-only.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/interpret/test_motifs.py -q -k "client or fetch or search or jaspar"</automated>
    <fails_when>nonzero exit — e.g. the exhausted-retries case returns text instead of raising the matchable ValueError, or an invalid matrix id reaches urlopen</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: default base URL is the canonical jaspar.elixir.no host; release pinned by default on every search; no requests/httpx import anywhere in the module
  - Behavior: retry/backoff mirrors model.py:317-377 (max_try=3, sleep patched in tests); non-200 and exhaustion surface as matchable ValueErrors
  - Test: mocked client lane fully green; slow live tests carry `jaspar-unreachable:` typed skips and the yaml allowlist entry exists in the same commit
  - CHANGELOG: REV-10 bullet present under ## [Unreleased] with inline tag, same commit as motifs.py client code
  </acceptance_criteria>
  <done>Stdlib JASPAR client (fetch + search, parameterized base URL, canonical-host default, pinned release) green under mocked tests; live round-trip behind typed allowlisted skips; REV-10 CHANGELOG entry landed same-commit.</done>
</task>

<task type="auto">
  <name>Task 4 (LAST, owner-input-gated): HBG1/BCL11A golden-test harness, synthetic-verified with a drop-in structure</name>
  <files>tests/interpret/test_motifs.py, tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml, tests/interpret/fixtures/hbg1_bcl11a/synthetic_window.fasta, tests/interpret/fixtures/hbg1_bcl11a/synthetic_motif.meme</files>
  <read_first>
  - 12-RESEARCH.md: Pitfall 5 (golden fixture owner-blocked; two BCL11A CORE matrices exist — MA2324.1 w=7, MA2504.1), Open Questions 1-2 (required owner inputs; +-2bp tolerance if figure-derived), Assumptions A2/A5/A7 (loci anchors, release pin, motif candidates)
  - 12-CONTEXT.md specifics ("The HBG1/BCL11A hit coordinates must match the paper's Fig 4a annotation — freeze as a regression fixture (golden test)") and the Claude's-discretion note (fixture construction details)
  - 12-PATTERNS.md: "tests/interpret/fixtures/" section (Phase-11 committed synthetic ClinVar fixture precedent — network-free fast lane)
  - Task 2's landed scan entry (the API the golden test drives)
  </read_first>
  <action>
  Build the golden-test harness as the last task, explicitly gated on owner input for the PAPER-EXACT assertion only. Commit `tests/interpret/fixtures/hbg1_bcl11a/` containing: `manifest.yaml` recording every pending owner input — Fig 4a window coordinates (locus, flank, assembly), motif ID (expected MA2324.1 or MA2504.1 per live-verified JASPAR CORE, A7), JASPAR release (2024 vs 2026, A5), and the coordinate tolerance policy (exact match when owner-supplied verbatim; +-2bp documented if figure-derived, Open Question 2) — plus a `pending: true` flag; a `synthetic_window.fasta` and `synthetic_motif.meme` stand-in pair where the expected hit coordinates are hand-derivable by construction (embed the motif at a known offset in a generated window). The golden test (class TestGoldenHBG1BCL11A): loads manifest + fixtures through a dedicated fixture-loader helper, runs the Task-2 scan entry over the fixture window(s) with the manifest's motif and background settings, extracts the target motif's hit rows, and asserts coordinates within the manifest tolerance — proven GREEN now against the synthetic stand-in (comparison logic and loader verified end-to-end). When owner input arrives, the drop-in is fixture-files-only: replace the stand-ins with the owner-supplied FASTA window + frozen JASPAR MEME file, flip `pending: false`, set the real coordinates/tolerance — ZERO harness code changes. If owner input becomes available during execution (check for a phase handoff note; otherwise do not block), apply it directly and the same test becomes the MOTIF-01 Fig-4a acceptance. The SUMMARY must report the owner-input request explicitly (coordinates, motif ID, JASPAR release) so the phase closeout tracks it.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/interpret/test_motifs.py -q -k "golden"</automated>
    <fails_when>nonzero exit — e.g. the loader cannot parse manifest.yaml, or the comparison logic fails to locate the synthetic motif embedded at its known offset within the manifest tolerance</fails_when>
  </verify>
  <acceptance_criteria>
  - Test: golden harness green against the synthetic stand-in (loader + scan + coordinate comparison all exercised); network-free (committed files only)
  - Fixture: manifest.yaml lists every pending owner input with the tolerance policy; committed stand-in FASTA/MEME files present
  - Traceability: the test docstring names MOTIF-01 Fig 4a as the acceptance and owner-input as the activation condition
  - Report: SUMMARY states the pending owner inputs; no guessed paper coordinates anywhere in the fixture
  </acceptance_criteria>
  <done>Golden-test harness committed and green on the synthetic stand-in with the drop-in point fully specified; the paper-exact Fig 4a assertion activates by fixture replacement once owner input lands — tracked in the SUMMARY as the phase's known pending input (MOTIF-01's coordinate-match acceptance cannot close before it; nothing else in C1 waits on it).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| JASPAR network -> motifs.py parser | Live HTTP responses parsed as motif data (tampered/slop content possible) |
| caller -> base_url / matrix_id parameters | Per-call values must never compose the request host |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-12-01 | Tampering | motifs.py JASPAR client + parse_meme | high | mitigate | Strict regex-anchored MEME grammar parse-or-reject with matchable ValueError; response read size-capped (1 MB class); unknown sections rejected; never eval/exec on fetched content |
| T-12-02 | Information Disclosure | motifs.py base_url parameter (SSRF-adjacent egress) | medium | mitigate | Default host is the module constant (canonical jaspar.elixir.no); base_url must be https; per-call arguments (matrix id regex `^MA\d{4}\.\d+$`, closed-set collection/release params) never form the host; no redirect-following to arbitrary hosts |
| T-12-03 | Tampering | motifs.py CIS-BP/MEME local file parsers | low | mitigate | Parse-or-reject discipline identical to vep._load_reference: validated alength/w/row ranges/nsites; malformed input raises matchable ValueError naming the artifact |
| T-12-SC | Tampering | package installs | high | accept | Zero package installs this phase (zero-new-dependencies milestone invariant; RESEARCH Package Legitimacy Audit is empty by construction) — nothing to gate |
</threat_model>

<verification>
- Targeted lane green: `uv run --no-sync pytest tests/interpret -q -m "not slow"` — 0 failed.
- Per-module coverage standard: `uv run --no-sync coverage run -m pytest tests/interpret -q -m "not slow" && uv run --no-sync coverage report --include="dnallm/interpret/*"` — dnallm/interpret/motifs.py row >= 96% and VISIBLE (not under any omit glob; the same-change proof it is measured).
- Slow live lane (network present): `uv run --no-sync pytest tests/interpret -q -m slow` — live JASPAR round trip passes or typed-skips with `jaspar-unreachable:` prefix; `uv run --no-sync python scripts/audit_skips.py` shape holds (yaml entry present same-change).
- Facade stability: `git diff --stat dnallm/__init__.py` empty across the lane's commits.
- Census safety: `uv run --no-sync pytest tests/ --collect-only -q | tail -3` still reports 208/217 tests collected (9 deselected).
</verification>

<success_criteria>
- MOTIF-01 scanner delivered at the >=96% per-module standard with every FIMO convention implemented per D-01/D-02/D-03 and the calibration honestly documented (docstring + test).
- All five MOTIF-01 edge predicates (threshold inversion at distribution edges; width-1 motif; palindromic both-strand adjacency; short-window exclusion; zero-pass empty table) asserted by named tests.
- JASPAR client stdlib-only, canonical-host default, release-pinned searches, typed allowlisted skips.
- Golden harness green on synthetic stand-in; owner-input request reported; no guessed coordinates.
- REV-10 CHANGELOG entry landed same-commit; zero new dependencies; facade byte-stable.
</success_criteria>

## Artifacts this phase produces
(This plan's share; the phase-level rollup lives in 12-03.)
- `dnallm/interpret/` package (init + motifs.py) — the MOTIF-01 scanner + JASPAR client
- `tests/interpret/` — test_motifs.py + committed fixtures (MEME/CIS-BP samples, golden harness with manifest + synthetic stand-ins)
- `tests/expected_skips.yaml` — `jaspar-unreachable:` typed-skip allowlist entry
- `CHANGELOG.md` — REV-10 entry under ## [Unreleased]

<output>
Create `.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-01-SUMMARY.md` when done.
The SUMMARY must explicitly list the pending owner inputs for the golden fixture (Fig 4a window
coordinates, motif ID, JASPAR release) — this is the phase's tracked blocker for MOTIF-01's
coordinate-match acceptance.
</output>
