---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "02"
subsystem: mcp
tags: [mcp, fastmcp, ism, mutagenesis, vep, zero-shot, hotspots, cli-precedence, argparse, uvicorn]

requires:
  - phase: 11-clinvar-zero-shot-vep-probing-sweeps
    provides: dnallm/inference/vep.py evaluate_vcf kernel + skip accounting (the sole VCF path wrapped here)
  - phase: 12-motif-matching-mcp-tools-milestone-closeout
    provides: 12-RESEARCH Pattern 4/5/6, 12-PATTERNS tool skeleton, D-04/D-05/D-06 decisions
provides:
  - MCP tools ism_scan, hotspots, zero_shot_score (all timeout-wrapped, error-dict, off-loop)
  - Host/port CLI-precedence fix on BOTH sse and streamable-http (sentinel resolution, one documented default 127.0.0.1:8000)
  - EXPECTED_TOOLS 13 -> 16 with same-change len-assert flips and ASGI round trips
  - Tool-boundary caps (2000 bases / 100 positions / 500 variants / 64 MiB VCF / 10k context window) with error dicts stating them
  - Security path invariant test for the inline temp-VCF materialization (no record-derived string in any path)
  - REV-11 CHANGELOG evidence-chain entry
affects: [12-03-closeout, mcp-tooling, milestone-v1.2-evidence-chain]

actuals:
  tokens: 32884
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Sentinel-based bind-address resolution resolved ONCE before transport dispatch (CLI-explicit > transport YAML > server YAML > documented default)"
    - "ISM/VEP tool bodies run torch work in the executor under ModelManager._infer_thread_lock (fork-safety parity with predict traffic), lock acquired inside the closure"
    - "Dual-mode input convergence: inline variants materialize to a fixed-sanitized-name temp VCF so both modes share one evaluate_vcf kernel with identical skip accounting"
    - "Pass-through CLNSIG/CLNVC/CLNREVSTAT sentinels (not_analyzed / inline_variant / criteria_provided) admit label-less inline variants through the kernel's D-17 convention gates; metrics stay None by construction"

key-files:
  created:
    - tests/mcp/test_server_tools_v12.py
  modified:
    - dnallm/mcp/server.py
    - dnallm/mcp/start_server.py
    - tests/mcp/test_server_transports.py
    - tests/mcp/test_start_server.py
    - CHANGELOG.md

key-decisions:
  - "start_server.py's argparse also moved to default=None (beyond the plan's file list): forwarding a fake-explicit 0.0.0.0:8000 would have defeated CLI precedence, and the mandated test_defaults_are_forwarded flip requires the None sentinels"
  - "Documented default unified to 127.0.0.1:8000 (start_server's old documented default); the argparse-side 0.0.0.0 divergence is disclosed in the REV-11 CHANGELOG bullet and --help"
  - "hotspots coordinates are 0-based half-open (Python slicing convention); windows returned region-relative AND as absolute genomic coordinates"
  - "Inline-mode convention: kernel gates are pass-through sentinels, disclosed in the tool docstring and visible verbatim in the response convention block (labels \"['not_analyzed']=1 vs []=0\", clnrevstat_counts criteria_provided:N)"
  - "Registry gate added before the engine gate (_ism_engine_guard): not-configured dict (T-12-07) then not-loaded dict; both matchable"

patterns-established:
  - "Split-staging a shared CHANGELOG under concurrent single-tree execution: synthesize the staged blob from HEAD + own bullet only (git hash-object + update-index --cacheinfo), never a pathspec commit of the shared file"
  - "Tool test methods named to carry the plan's -k keywords (test_zero_shot_*, ..._security_*) so the verify filters actually select them"

requirements-completed: [MCPE-01]

coverage:
  - id: D1
    description: "--host/--port CLI precedence over YAML on both sse and streamable-http, stdio unregressed, one documented default"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_transports.py#TestHostPortPrecedence (CLI-explicit/yaml-only/no-block/no-config on BOTH transports) + flipped test_config_fields_assembled_from_streamable_http_block (explicit 8123 beats YAML 8124)"
        status: pass
      - kind: unit
        ref: "tests/mcp/test_start_server.py#TestMain::test_defaults_are_forwarded (None sentinels) + test_starts_server_with_parsed_args (explicit flags, unchanged)"
        status: pass
  - id: D2
    description: "ism_scan tool: bounded ISM mirroring dna_mutagenesis with caps before engine access and off-loop execution"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_tools_v12.py#TestIsmScanContracts (happy, caps, not-loaded/not-configured, invalid sequence, exception boundary)"
        status: pass
      - kind: integration
        ref: "tests/mcp/test_server_transports.py#TestInMemoryProtocolRoundTrip::test_round_trip_ism_scan (+ error-dict variant)"
        status: pass
  - id: D3
    description: "hotspots tool: windows from model x coordinates via vep._load_reference + Mutagenesis ISM + find_hotspots over per-call fasta_path"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_tools_v12.py#TestHotspotsContracts (coordinates field validation, caps, chrom/bounds, suffix allowlist, missing fasta)"
        status: pass
      - kind: integration
        ref: "tests/mcp/test_server_transports.py#TestInMemoryProtocolRoundTrip::test_round_trip_hotspots (slice provenance asserted via mutate_sequence call args)"
        status: pass
  - id: D4
    description: "zero_shot_score dual-mode tool: inline variants and vcf_path both through the one evaluate_vcf kernel with skip accounting verbatim; security path invariant"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_tools_v12.py#TestZeroShotScoreContracts (dual-mode, per-field validation, caps, filters, kernel error surfaces, temp-VCF content, security path)"
        status: pass
      - kind: integration
        ref: "tests/mcp/test_server_transports.py#TestInMemoryProtocolRoundTrip::test_round_trip_zero_shot_score_inline (+ both-modes error-dict variant)"
        status: pass
  - id: D5
    description: "Tool surface: EXPECTED_TOOLS 16, exact set equality, all wrapped in _with_timeout_wrapper"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_transports.py#TestInMemoryProtocolRoundTrip::test_list_tools_returns_complete_registered_set"
        status: pass
  - id: D6
    description: "Event loop stays live during a long tool call (executor off-load proven)"
    requirement: MCPE-01
    verification:
      - kind: unit
        ref: "tests/mcp/test_server_tools_v12.py#TestEventLoopLiveness::test_health_check_completes_during_long_ism_call"
        status: pass

metrics:
  duration: 25 min
  completed: 2026-10-10
  commits: 4
  plan_head_before: 283b14a
  plan_head_after: fb60af3

status: complete
---

# Phase 12 Plan 02: MCP Tools (ism_scan / hotspots / zero_shot_score) + Host/Port CLI-Precedence Fix Summary

Three MCP tools wrapping the existing Mutagenesis/VEP surfaces under the house timeout-wrapper + error-dict conventions, plus the v1.1-audit deferred --host/--port silently-overridden-by-YAML bug fixed on both HTTP transports via sentinel resolution (CLI-explicit > transport YAML > server YAML > 127.0.0.1:8000).

## Tasks Completed

1. **Host/port CLI-precedence fix (tracer)** — `start_server(host: str | None, port: int | None)` resolves once in `_resolve_bind_address`; both override blocks deleted (the unconditional YAML override in `start_server`, the always-true conditional in `_start_http_server`); argparse `default=None` in BOTH entry points (`server.py::main` and `start_server.py::main`) with the chain in `--help`; the two codifying tests flipped same-change; precedence matrix (CLI/yaml-only/no-block/no-config) added for sse AND streamable-http. Tracer feedback gate re-ran the verify end-to-end post-commit (39 passed). Commit 3bc072f.
2. **ism_scan + hotspots** — validation-first bodies per the `_dna_mutagenesis` template with caps (2000 bases, 100 positions, 2000-base region) enforced before engine access; `_ism_engine_guard` registry+engine gate; torch work off-loop in the executor under `ModelManager._infer_thread_lock` (fork-safety parity with predict traffic); hotspots derives windows from model x coordinates + per-call fasta_path (vep._load_reference/_resolve_chromosome), never a window file; EXPECTED_TOOLS 13 -> 15 same-change; per-tool JSON contracts + the event-loop liveness proof in the new tests/mcp/test_server_tools_v12.py; ASGI round trips for both tools. Commit c4051c4.
3. **zero_shot_score + REV-11 entry** — dual-mode (D-04): inline {chrom,pos,ref,alt} variants materialize to a temp VCF with the FIXED sanitized name `inline_variants.vcf` (never a record-derived path) or a server-side size-capped .vcf/.vcf.gz; both modes converge on one `vep.evaluate_vcf` call; skip_counts/skipped/skip_fraction/convention surface verbatim; `clnsig_filter` parameter maps to ClinVarFilter (non-ClinVar opt-out); 500-variant cap before the kernel; EXPECTED_TOOLS 16 + round trips; security path test (distinctive chrom/ref/alt strings asserted absent from every path, temp dir cleaned up); REV-11 CHANGELOG bullet with the host/port fix and the 0.0.0.0 -> 127.0.0.1 default-unification note. Commits 0a4c7f5 + fb60af3.

## Verification Results

- Targeted lane: `uv run --no-sync pytest tests/mcp -q -m "not slow"` — **283 passed**.
- Tool-surface proof: `list_tools` — 16 names, exact set equality with EXPECTED_TOOLS.
- Per-module coverage: `coverage run -m pytest tests/mcp -m "not slow"` + `coverage report --include=dnallm/mcp/server.py` — **97%** (798 stmts, 23 miss) against the ≥96% standard.
- Sibling-suite safety: `pytest dnallm/mcp/tests -q -m "not slow"` — **52 passed, 7 deselected** (packaged MCP test root unregressed).
- Census safety: `pytest tests/examples --collect-only -q -m "not giants" -k "not mcp_example"` — **208/217 tests collected (9 deselected)**, pin intact.
- Ruff check + format clean on all lane files.

## Evidence Chain (REV-11 CHANGELOG entry — for 12-03 backfill)

The REV-11 bullet's final authoritative state is at commit **fb60af3** (HEAD at plan completion); the same-change code commit is **0a4c7f5**. NOTE for 12-03: due to the concurrent shared-append race (see Deviation 2 below), the REV-11 line TEXT first entered history inside sibling commit bd165d8, and 0a4c7f5's CHANGELOG hunk nets to zero against it. Backfill **0a4c7f5** as the REV-11 evidence link (the commit that ships the tools the bullet describes, message names the CHANGELOG append) and treat fb60af3 as the both-bullets final state.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `dnallm/mcp/start_server.py` argparse also moved to default=None**
- **Found during:** Task 1
- **Issue:** The plan mandates flipping `tests/mcp/test_start_server.py::test_defaults_are_forwarded` to the None-sentinel chain, but that test drives `dnallm.mcp.start_server.main` — a second argparse entry point not in the plan's files_modified. Leaving its `default="0.0.0.0"`/`8000` would forward fake-explicit values and defeat CLI precedence (an argparse default is indistinguishable from an explicit flag).
- **Fix:** Same sentinel change as `server.py::main` (default=None, resolution chain in --help, None-aware log line). File-disjoint from sibling lane 12-01.
- **Files modified:** dnallm/mcp/start_server.py
- **Commit:** 3bc072f

**2. [Rule 3 - Process] CHANGELOG shared-append race with sibling 12-01 (both lanes' bullets, one file)**
- **Found during:** Task 3 commit
- **Issue:** Under single-tree concurrency, 12-01's commit bd165d8 staged the whole CHANGELOG.md working file and therefore carried BOTH bullets (their REV-10 + my already-written REV-11). My single-writer staging (synthesized blob = HEAD + my REV-11 line only) then deleted their REV-10 line relative to the new HEAD.
- **Fix:** Follow-up commit fb60af3 re-adds the sibling's REV-10 bullet verbatim (no history rewrite — amend would race the concurrent agent). Final state: both bullets present, zero sibling text modified.
- **Files modified:** CHANGELOG.md
- **Commit:** fb60af3
- **Lesson (recorded in patterns-established):** under this concurrency model, a pathspec commit of the shared CHANGELOG by EITHER agent absorbs the other's uncommitted line; the synthesized-blob staging protects only against the case where the sibling has NOT yet committed. The repair commit is the safe endgame.

**3. [Rule 1 - Bug] Test-method names did not match the plan's `-k` verify filters**
- **Found during:** Task 3
- **Issue:** The class name `TestZeroShotScoreContracts` does not contain the substring `zero_shot`, so the plan's `-k "zero_shot or ..."` filter silently deselected the entire contract class (7 selected instead of 31).
- **Fix:** All 18 test methods renamed to carry `test_zero_shot_*` (the security test also carries `security`), matching the verify filters verbatim.
- **Files modified:** tests/mcp/test_server_tools_v12.py
- **Commit:** 0a4c7f5

### Design notes within A8 discretion (not deviations)

- **Inline-mode pass-through convention:** `evaluate_vcf`'s D-17 gates (CLNVC/CLNSIG/CLNREVSTAT) cannot admit label-less rows through ClinVarFilter alone — the star-floor token check is hardcoded. Inline temp-VCF rows therefore carry pass-through sentinels (`CLNSIG=not_analyzed`, `CLNVC=inline_variant`, `CLNREVSTAT=criteria_provided`) admitted under a tool-built sentinel filter. Disclosed in the tool docstring; the response's convention block reports verbatim what was applied (labels "['not_analyzed']=1 vs []=0", clnrevstat_counts criteria_provided:N); metrics stay None by construction (single label class).
- **ISM executor lock:** ISM builds a DataLoader (num_workers may exceed 0) and shares the fork-unsafe window that predicts serialize on, so the ISM/VEP executor closures acquire `ModelManager._infer_thread_lock` (the CR-01 pattern), not a server-private lock.
- **VCF size cap set at 64 MiB** (T-12-04 "size caps"); no FASTA size cap (reference data; suffix allowlist + parse-or-reject via `_load_reference` suffice).

## Auth Gates

None.

## Known Stubs

None — all three tools are fully wired to the Mutagenesis/VEP kernels; no placeholder data paths.

## Commits

- 3bc072f fix(12-02): --host/--port CLI precedence over YAML on both MCP transports
- c4051c4 feat(12-02): ism_scan + hotspots MCP tools with caps, error dicts, loop liveness
- 0a4c7f5 feat(12-02): zero_shot_score dual-mode tool, EXPECTED_TOOLS 16, security path test, REV-11 entry
- fb60af3 fix(12-02): restore sibling REV-10 CHANGELOG bullet lost to the shared-append race

(plan_head_before 283b14a -> plan_head_after fb60af3; the range also contains sibling 12-01's four commits — this lane's commits are exactly the four above)

## Self-Check: PASSED

- Files exist: dnallm/mcp/server.py, dnallm/mcp/start_server.py, tests/mcp/test_server_tools_v12.py, tests/mcp/test_server_transports.py, tests/mcp/test_start_server.py, CHANGELOG.md — all verified on disk post-commit.
- Commits 3bc072f, c4051c4, 0a4c7f5, fb60af3 are ancestors of HEAD (verified via git log).
- CHANGELOG.md at HEAD contains exactly one REV-11 bullet and one REV-10 bullet.
