---
phase: 12-motif-matching-mcp-tools-milestone-closeout
verified: 2026-10-10T06:48:54Z
status: passed
score: 12/13 must-haves verified
covered_files: [".gitignore", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-01-SUMMARY.md", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-01-motif-matching-fimo-scanner-PLAN.md", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-02-SUMMARY.md", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-02-mcp-tools-host-port-fix-PLAN.md", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-03-SUMMARY.md", ".planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-03-milestone-closeout-docs-changelog-PLAN.md", "CHANGELOG.md", "dnallm/interpret/__init__.py", "dnallm/interpret/motifs.py", "dnallm/mcp/server.py", "dnallm/mcp/start_server.py", "docs/user_guide/continuous_integration.md", "docs/user_guide/fine_tuning/peft_adapters.md", "pyproject.toml", "tests/expected_skips.yaml", "tests/interpret/fixtures/cisbp_motif.txt", "tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml", "tests/interpret/fixtures/hbg1_bcl11a/synthetic_motif.meme", "tests/interpret/fixtures/hbg1_bcl11a/synthetic_window.fasta", "tests/interpret/fixtures/meme_motif.txt", "tests/interpret/test_motifs.py", "tests/mcp/test_server_tools_v12.py", "tests/mcp/test_server_transports.py", "tests/mcp/test_start_server.py"]
covered_digest: "v3:sha256:358943ae3dc820840a4bd9c8fb61e9323d1ddf09b94c16d29b88c3ee40730fa9"
behavior_unverified: 1
behavior_unverified_items:
  - truth: "The HBG1/BCL11A motif hit coordinates match the paper's Fig 4a annotation (MOTIF-01 acceptance clause, ROADMAP SC1)"
    test: "Owner supplies the four pending inputs recorded in tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml (Fig 4a window coordinates verbatim, motif ID MA2324.1 vs MA2504.1 vs CIS-BP PWM, JASPAR release 2024 vs 2026, tolerance policy), then swaps the two stand-in fixture files, flips pending: false, and fills expected_hits/tolerance_bp — zero harness code changes — and runs pytest tests/interpret -k golden"
    expected: "test_golden_manifest_fixture_scan_matches_expected_hits passes against the owner-supplied paper window: the scan's hit rows fall within the manifest tolerance of the paper's Fig 4a annotated coordinates"
    why_human: "The paper is under revision and not publicly indexed; the plan (12-01 Task 4) sequenced this acceptance as owner-input-gated by design. The harness itself is committed, generic, and green on the committed synthetic stand-in (94/94 fast-lane tests including all 3 golden tests) — only the fixture VALUES are missing. This is an external owner dependency (like Phase 11's VEP magnitude override), not a code gap; no verification command can substitute for the owner's manuscript data."
human_verification:
  - test: "Provide the HBG1/BCL11A golden-fixture owner inputs (Fig 4a window coordinates + motif ID + JASPAR release + tolerance policy), perform the fixture-files-only drop-in, and run uv run --no-sync pytest tests/interpret -q -k golden"
    expected: "Golden test passes against the paper-exact fixture; manifest pending flips to false; MOTIF-01's coordinate-match acceptance closes"
    why_human: "Requires reading the owner's unrevised manuscript figure — data that exists only in owner hands and is explicitly recorded as pending in the manifest; no programmatic check is possible"
overrides_applied: 1
overrides:
  - must_have: "The HBG1/BCL11A motif hit coordinates match the paper's Fig 4a annotation (MOTIF-01 acceptance clause, ROADMAP SC1)"
    reason: "Owner deferral (option B, 2026-10-10, same mechanism as the Phase 11 VEP-magnitude override): the four pending inputs (Fig 4a window coordinates, motif ID MA2324.1/MA2504.1/CIS-BP, JASPAR release 2024/2026, tolerance policy) exist only in the unrevised manuscript. The harness is committed, generic, and green on the synthetic stand-in (94/94 fast-lane incl. all 3 golden tests); activation is fixture-files-only (manifest.yaml pending->false + expected_hits/tolerance_bp + two file swaps, zero code changes), tracked in a GitHub issue — deferred without reopening the milestone."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-10T17:46:00+08:00"
---

# Phase 12: Motif Matching, MCP Tools & Milestone Closeout Verification Report

**Phase Goal:** The narrative-facing surface is complete — motif hits reproduce the paper's annotation, LLM agents can drive ISM/hotspot/zero-shot scoring over MCP with honest CLI flags — and the milestone closes fully green with the docs chapter and CHANGELOG evidence chain finished
**Verified:** 2026-10-10T06:48:54Z
**Status:** passed (2026-10-10 — owner deferred the single owner-input-gated truth; override recorded in frontmatter)
**Re-verification:** No — initial verification

## Goal Achievement

Must-haves merged from ROADMAP Success Criteria 1-4 (the contract) plus PLAN frontmatter detail. All three plans' must_haves were checked; none reduce roadmap scope (12-01's 9 truths, 12-02's 9 truths, and 12-03's 6 truths all fold into the 13 consolidated truths below — plan-level specifics are evidenced in-line).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Scanner emits motif-ID/start/end/strand/score-bits/p/q/E-value table under FIMO conventions (GC-matched background, both strands, p<1e-4, single BH call over full set) | ✓ VERIFIED | `dnallm/interpret/motifs.py:678-789` — one `false_discovery_control(..., method="bh")` call over the concatenated full window×motif×strand p-vector (with the [0,1] clamp), both strands via `..utils.sequence.reverse_complement`, E=p×n_tested; behavioral proof: `pytest tests/interpret -m "not slow"` = 94 passed, including analytic full-set q anchors (p·n/2 tie, p·n/3 rank-3 — values that hold ONLY under single-call semantics) and palindrome/E-value anchors |
| 2 | HBG1/BCL11A hit coordinates match the paper's Fig 4a annotation | ⚠️ OWNER-DEFERRED 2026-10-10 | Harness committed, generic, and green on the synthetic stand-in (`TestGoldenHBG1BCL11A`, 3 tests pass in the 94); `manifest.yaml` records `pending: true` + all four owner inputs; no guessed coordinates anywhere. Owner chose deferral (option B) over supplying manuscript data now — activation is fixture-files-only and tracked in a GitHub issue; override recorded in frontmatter. Not a code gap, not silently dropped |
| 3 | stdlib JASPAR client fetches from parameterized canonical host (jaspar.elixir.no, format=meme) with retry/size-cap/matchable errors | ✓ VERIFIED | `motifs.py:88` `JASPAR_BASE="https://jaspar.elixir.no/api/v1"`, `fetch_meme_motif` validates `^MA\d{4}\.\d+$` before URL construction, retry-with-backoff mirroring the model.py house pattern, 1 MB read cap, redirect refusal; client tests green (8 fetch + 7 search); live round trips passed 2026-10-10 with typed `jaspar-unreachable:` skip fallback allowlisted in `tests/expected_skips.yaml:86` (same-change) |
| 4 | p-value calibration choice documented honestly (exact-DP vs empirical-null, with FIMO quantization) | ✓ VERIFIED | Module docstring `motifs.py:12-45` states exact-DP per MEME/FIMO convention, PSSM_RANGE=100 integer scaling (pssm.h:13), pseudocount 0.1, zero-order GC background, BH full-set, E-value formula; asserted by `test_docstring_documents_calibration_choices` (green) |
| 5 | MCP client on a live in-memory server can call ism_scan/hotspots/zero_shot_score and receive valid JSON under timeout-wrapper + error-dict conventions | ✓ VERIFIED | `server.py:424-426` registers all three via `_with_timeout_wrapper`; ASGI round trips for all 3 tools (+ error-dict variants) green in the 16-test targeted run; `list_tools` asserts exact set equality with EXPECTED_TOOLS (16); 307/307 `tests/mcp` fast lane passed under coverage |
| 6 | zero_shot_score wraps the Phase-11 VEP module with skip accounting verbatim (dual-mode) | ✓ VERIFIED | `server.py:2389-2655` — both inline-variants and vcf_path modes converge on ONE `evaluate_vcf` call (executor + `_infer_thread_lock`); response surfaces `skip_counts`/`skipped`/`skip_fraction`/`convention` verbatim from `VepResult.to_dict()`; 28 `test_zero_shot_*` contract tests green incl. temp-VCF content and dual-mode rejection |
| 7 | --host/--port CLI flags take precedence over YAML on BOTH sse and streamable-http | ✓ VERIFIED | `server.py:2706-2751` `_resolve_bind_address` (CLI-explicit > transport YAML > server YAML > 127.0.0.1:8000), resolved ONCE at `start_server:2833`; both override blocks deleted (grep clean); argparse `default=None` in BOTH entry points (`server.py:main`, `start_server.py:main`); flipped `test_config_fields_assembled_from_streamable_http_block` asserts explicit 8123 beats YAML 8124; `TestHostPortPrecedence` covers the 4-case matrix on both transports; `test_defaults_are_forwarded` flipped to None-sentinels — all green |
| 8 | IA³ chapter completes DOCS-01; "coming in the next release" forward pointer gone | ✓ VERIFIED | Phrase absent from `peft_adapters.md` (grep 0 hits); real chapter at :174-300 documents use_ia3, the ia3 section, both rejections, save/reload — every field verified against source: `Ia3Config` fields match `configs.py:426-470` 1:1, use_ia3×use_qlora rejected at Pydantic time (`configs.py:376-384`), LoRA×IA³ at trainer init (`trainer.py:397-401`), `get_peft_model` IA³ branch (`trainer.py:485-495`) |
| 9 | CHANGELOG evidence chain complete: every REV-01..REV-11 entry SHA-linked | ✓ VERIFIED | 11 REV tags present exactly once each; 11 commit links under ## [Unreleased]; spot-checked 4 links resolve to HEAD ancestors with matching subjects (dae194a→REV-01, 58bbf41→REV-02, bd165d8→REV-10, 0a4c7f5→REV-11 — the latter per the 12-02 SUMMARY's documented shared-append-race instruction) |
| 10 | Fast lane fully passing; every new skip typed and allowlisted same-change | ✓ VERIFIED | 12-03 closing gate 2404P/0F recorded (+151 vs 2253 baseline); at HEAD: interpret lane 94P, mcp lane 307P, transports/start_server 24P — 0 failures; only new skip type is `jaspar-unreachable:` (slow lane, allowlisted) |
| 11 | Coverage gate green with every new module at the ≥96% per-module standard | ✓ VERIFIED | Independently re-measured at HEAD via the recorded workaround (`coverage run -m pytest <lane>`): motifs.py 99% (377 stmts, 2 miss — lines 386/406), interpret/__init__.py 100%, mcp/server.py 97% (815 stmts) — all ≥96% |
| 12 | docs-validation green | ✓ VERIFIED | `check_docs_sync.py` OK; `validate_docs_snippets.py` 352/352 blocks pass. Note: `mkdocs build --strict` carries 15 PRE-EXISTING warnings (proven zero-delta in 12-03, deferred with owner actions in `deferred-items.md`); the CI docs-validation workflow does not run mkdocs --strict |
| 13 | Zero new dependencies across the milestone beyond approved scikit-allel | ✓ VERIFIED | `git diff 3bc072f..HEAD -- pyproject.toml`: pyarrow cap (`>=15,<26`) + pydantic-ai floor (`>=1.107.0,<2`) — constraint changes on EXISTING transitive deps (CI-drift fixes), no new dependency entries; `dnallm/__init__.py` byte-stable across the phase (empty diff) |

**Score:** 12/13 truths verified (1 owner-input-gated, behavior not exercisable without owner data)

### Deferred Items

One owner-deferred item (2026-10-10, option B): the MOTIF-01 paper-exact coordinate clause. The golden-fixture harness (`TestGoldenHBG1BCL11A` + `manifest.yaml`) is committed and green on the synthetic stand-in; activation needs only the four owner inputs from the unrevised manuscript (Fig 4a window coordinates, motif ID, JASPAR release, tolerance policy) via a fixture-files-only drop-in — recorded in `deferred-items.md` and tracked as a GitHub issue (filed 2026-10-10). No milestone reopen required. Future v1.3+ requirements (GENOMEWIDE-SCAN etc.) are out-of-milestone and unrelated.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/interpret/__init__.py` | Minimal package init, no facade re-export | ✓ VERIFIED | 27 lines, docstring + 7-name `__all__`; facade untouched |
| `dnallm/interpret/motifs.py` | Full FIMO scanner + JASPAR client | ✓ VERIFIED | 983 lines; all claimed symbols present and substantive (parse_meme/parse_cisbp/log_odds_matrix/pvalue_table/threshold_bits/gc_background/scan/scan_single_strand/fetch_meme_motif/search_motifs) |
| `tests/interpret/test_motifs.py` | DP anchors, background, strands, parsers, BH, client mocks, edges, golden harness | ✓ VERIFIED | 1033 lines, 23 test classes, 94 fast + 2 slow live tests — all claimed test classes present by name |
| `tests/interpret/fixtures/` | MEME/CIS-BP samples + golden manifest + stand-ins | ✓ VERIFIED | All 5 fixture files on disk; manifest carries pending owner inputs |
| `tests/expected_skips.yaml` | `jaspar-unreachable:` prefix entry | ✓ VERIFIED | Line 86 |
| `dnallm/mcp/server.py` | 3 new tools + sentinel host/port resolution | ✓ VERIFIED | 3210 lines; tools at :1801/:2072/:2389; `_resolve_bind_address` at :2706; caps constants at :91-126 |
| `dnallm/mcp/start_server.py` | argparse default=None (deviation, justified) | ✓ VERIFIED | :62-83 + None-forwarding at :132 — the sanctioned Rule-3 fix; without it the flipped defaults test could not hold |
| `tests/mcp/test_server_tools_v12.py` | Per-tool JSON contracts, liveness, security | ✓ VERIFIED | 1249 lines, 5 test classes, 71 tests |
| `tests/mcp/test_server_transports.py` | EXPECTED_TOOLS 16, round trips, precedence matrix | ✓ VERIFIED | :46 EXPECTED_TOOLS, :174 `len(names)==16`, TestHostPortPrecedence both transports |
| `tests/mcp/test_start_server.py` | Flipped None-sentinel defaults test | ✓ VERIFIED | :123 `test_defaults_are_forwarded` asserts host=None/port=None |
| `docs/user_guide/fine_tuning/peft_adapters.md` | Completed IA³ chapter | ✓ VERIFIED | Real chapter, source-accurate (see Truth 8) |
| `docs/user_guide/continuous_integration.md` | Measured coverage expectation | ✓ VERIFIED | 96.72% with dated provenance note ("measured at the v1.2 closeout 2026-10-10; was 96.30% at Phase 3") |
| `pyproject.toml` | Comment-only fail_under annotation | ✓ VERIFIED | fail_under=90 value unchanged; comment carries measured 96.72% |
| `CHANGELOG.md` | REV-01..11 SHA-linked evidence chain | ✓ VERIFIED | See Truth 9 |

All artifacts pass Levels 1-3 (exists, substantive, wired). Data-flow (Level 4): the three MCP tools call the real kernels (`evaluate_vcf`, `find_hotspots` via the engine, `_load_reference`) — no static/mock paths in production code; the scanner's JASPAR client feeds `parse_meme` → `scan` (verified in source and by `test_client_fetch_output_feeds_parse_meme`).

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `scan()` | `scipy.stats.false_discovery_control` | single BH call over concatenated full-set p-vector | ✓ WIRED | `motifs.py:756-762`; exactly one call site; analytic full-set q anchors prove semantics |
| strand `-` scoring | `dnallm.utils.sequence.reverse_complement` | import + call | ✓ WIRED | `motifs.py:61,744` — reused, not re-implemented |
| `fetch_meme_motif()` | `parse_meme()` → `scan()` | client output feeds the same parser | ✓ WIRED | live round-trip test + `test_client_fetch_output_feeds_parse_meme` |
| `_zero_shot_score` | `dnallm.inference.vep.evaluate_vcf` | sole VCF path | ✓ WIRED | `server.py:2613`; scikit-allel never imported in server.py |
| `_hotspots` | engine + `Mutagenesis.find_hotspots` + `vep._load_reference` | D-05 workflow | ✓ WIRED | `server.py:2204,2217`; round trip asserts slice provenance via mutate_sequence call args |
| argparse `--host/--port` | `start_server` → both starters | one resolution point | ✓ WIRED | `server.py:2833` → :2846/:2848 receive final values; no per-starter re-resolution |
| IA³ docs prose | `configs.py` Ia3Config + `trainer.py` branch | field/rejection parity | ✓ WIRED | Verified field-by-field (Truth 8) |
| CHANGELOG SHA links | git commits | link → ancestor of HEAD | ✓ WIRED | 4 spot-checks resolve; subjects match the REV fixes |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Interpret fast lane (incl. DP anchors, BH full-set, palindrome, golden harness) | `uv run --no-sync pytest tests/interpret -q -m "not slow"` | 94 passed, 0 failed (3.8s) | ✓ PASS |
| MCP tool surface + precedence + round trips | `pytest tests/mcp/test_server_transports.py -k "list_tools or precedence or round_trip or assembled"` | 16 passed | ✓ PASS |
| start_server sentinel forwarding | `pytest tests/mcp/test_start_server.py -m "not slow"` | 8 passed | ✓ PASS |
| Liveness + security + happy contracts | `pytest tests/mcp/test_server_tools_v12.py -k "liveness or security or happy or not_loaded"` | 10 passed | ✓ PASS |
| Full mcp lane under coverage | `coverage run -m pytest tests/mcp -m "not slow"` | 307 passed, 0 failed | ✓ PASS |
| Census pin | `pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` | 208/217 collected (9 deselected) | ✓ PASS |
| Docs validation gates | `check_docs_sync.py` / `validate_docs_snippets.py` | OK / 352 blocks pass | ✓ PASS |
| Per-module coverage re-measure | `coverage report --include=...` | motifs 99%, init 100%, server.py 97% | ✓ PASS |
| CHANGELOG SHA ancestry | `git merge-base --is-ancestor <sha> HEAD` ×4 | all ancestors | ✓ PASS |

Per the owner directive (targeted tests only), the full fast lane was not re-run; its green state is evidenced by 12-03's recorded closing gate (2404P/0F) plus the lane-scoped re-runs above at HEAD (post-review-fix).

### Probe Execution

Not applicable — phase declares no `probe-*.sh` scripts (the phase's checks are pytest lanes and the audit_skips/coverage gates, run above).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MOTIF-01 | 12-01 | FIMO-convention scanner, JASPAR client, honest calibration; Fig 4a coordinate acceptance | ✓ SATISFIED (one owner-gated acceptance clause → human) | Truths 1-4; scanner/client/calibration/docs all code-verified and test-proven; only the Fig 4a paper-exact clause awaits owner input (Truth 2) |
| MCPE-01 | 12-02 | ism_scan/hotspots/zero_shot_score tools + host/port CLI-precedence fix on both transports | ✓ SATISFIED | Truths 5-7; handshake round trips, EXPECTED_TOOLS 16, precedence matrix both transports |
| DOCS-01 (completion) | 12-03 | IA³-chapter section completing the Phase-10 split delivery | ✓ SATISFIED | Truths 8-9; REQUIREMENTS.md's split-delivery note maps the IA³ section's completion to Phase 12 — traceable in 12-03, which declares `requirements: [DOCS-01]` |

Orphaned requirements: none — REQUIREMENTS.md maps exactly MOTIF-01 and MCPE-01 to Phase 12; both are claimed by plans. Note (Info): the REQUIREMENTS.md traceability checkboxes for MOTIF-01/MCPE-01 still read "Pending" — that update belongs to milestone completion, not phase verification.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No TBD/FIXME/XXX, no TODO/HACK/PLACEHOLDER, no stub markers in any phase file | — | Clean |

### Test Quality Audit

| Test File | Linked Req | Active | Skipped | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|---------|----------|-----------------|---------|
| tests/interpret/test_motifs.py | MOTIF-01 | 96 | 2 (slow live; typed conditional skips only) | 0 | Value (analytic approx equalities) | OK |
| tests/mcp/test_server_tools_v12.py | MCPE-01 | 71 | 0 | 0 | Value/Behavioral (payload contracts, path-content security) | OK |
| tests/mcp/test_server_transports.py | MCPE-01 | 37 | 0 | 0 | Value (uvicorn kwargs, exact tool set) | OK |

Golden-fixture provenance: expected hits are hand-derived by construction (palindromic consensus embedded at a known offset in a letter-balanced window — documented in the manifest), NOT captured from the scanner; independent-oracle provenance. BH/E-value anchors are analytic (mathematical derivation — which is exactly what "FIMO-convention parity" demands). No circular patterns.

### Decision Coverage

All trackable CONTEXT.md decisions are honored by shipped artifacts (8/8, gate run 2026-10-10; non-blocking informational gate).

### Human Verification Required

### 1. HBG1/BCL11A golden-fixture owner input (MOTIF-01 Fig 4a acceptance)

**Test:** Supply the four pending inputs recorded in `tests/interpret/fixtures/hbg1_bcl11a/manifest.yaml` — (a) Fig 4a window coordinates verbatim (locus, flank, assembly), (b) motif ID (MA2324.1 or MA2504.1, both live-verified JASPAR CORE BCL11A, or a CIS-BP PWM), (c) JASPAR release (2024 or 2026), (d) tolerance policy (0 bp verbatim / ±2 bp figure-derived). Then replace `synthetic_window.fasta` + `synthetic_motif.meme` with the owner window and the frozen JASPAR MEME file, set `pending: false`, fill `expected_hits`/`tolerance_bp`, and run `uv run --no-sync pytest tests/interpret -q -k golden`.
**Expected:** The golden test passes against the paper-exact fixture — the scanner's HBG1/BCL11A hit coordinates fall within tolerance of the paper's Fig 4a annotation. Zero harness code changes are needed (the loader, scan entry, and comparison are proven generic on the synthetic stand-in).
**Why human:** The paper is under revision and not publicly indexed; the manuscript data exists only in owner hands. The plan deliberately gated this clause on owner input (12-01 Task 4) and the harness is complete, committed, and green — this is a designed external dependency, not incomplete work.

### Gaps Summary

No gaps. All code-level must-haves verified against the codebase at HEAD (post-review-chain): the FIMO scanner's conventions, calibration honesty, and JASPAR client; the three MCP tools with timeout/error-dict/caps/security conventions wired to the real VEP/Mutagenesis kernels; the host/port CLI-precedence fix with both codifying tests flipped; the IA³ docs chapter matching shipped source; the complete SHA-linked CHANGELOG evidence chain; the census pin; per-module coverage at/above the 96% standard (re-measured); docs-validation green; zero new dependencies; facade byte-stable.

The single non-closing item is the designed owner gate: MOTIF-01's Fig 4a paper-exact coordinate acceptance, pending four owner inputs that no verification command can synthesize. The harness is drop-in ready. Status is therefore `human_needed` — automated checks passed; awaiting owner input on that one clause.

Observations (Info, no action required for this phase):
1. `dnallm/mcp/server.py` measures 97% at HEAD (815 stmts) vs the 98% (796 stmts) published at the 12-03 measurement — the post-closeout review fixes added 19 statements. Still ≥96% standard; the docs number carries dated provenance per D-07, so the honesty contract (measured in-plan, not copied from an older phase) holds.
2. CI: latest code commit 46b336f is full-matrix green (first fully green matrix on revision). Two subsequent failure runs (fad3ac4, b5766fc) are windows-leg-only failures on `.planning`-docs-only commits — such commits cannot affect code tests (flake); the two newest docs-only commits' runs were not yet concluded at verification time.
3. `mkdocs build --strict` carries 15 pre-existing warnings, proven zero-delta and deferred with suggested owner actions in `deferred-items.md`; the CI docs gate does not run strict mode.

---

_Verified: 2026-10-10T06:48:54Z_
_Verifier: Claude (gsd-verifier)_
