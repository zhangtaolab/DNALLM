---
phase: 12-motif-matching-mcp-tools-milestone-closeout
plan: "02"
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/mcp/server.py
  - tests/mcp/test_server_transports.py
  - tests/mcp/test_start_server.py
  - tests/mcp/test_server_tools_v12.py
  - CHANGELOG.md
autonomous: true
requirements: [MCPE-01]
user_setup: []
estimate:
  tokens: 30000
  raw_tokens: 30000
  tasks: 3
  confidence: low
coupling_justified: >-
  CHANGELOG.md is the milestone's single sanctioned shared append surface (Phase 10 D-09
  mechanism): Wave-1 sibling 12-01 appends its own REV-10 bullet to ## [Unreleased] in the
  same window. Discipline: re-read immediately before edit, unique-anchor bullet under the
  existing ### Added heading, never edit sibling entries in place.
must_haves:
  truths:
    - "`--host`/`--port` CLI flags take precedence over YAML config on BOTH the sse and streamable-http paths: resolution is CLI-explicit > transport-specific YAML (streamable_http) > server YAML > one documented default — proven by construction tests on both transports via the patched-uvicorn pattern"
    - "An MCP client connected to a live in-memory server can call `ism_scan`, `hotspots`, and `zero_shot_score` and receive valid JSON — list_tools reports all 16 tools (EXPECTED_TOOLS 13 -> 16, same change as registration)"
    - "`zero_shot_score` accepts BOTH input modes — inline `variants` list of {chrom,pos,ref,alt} and server-side `vcf_path` — routing through the same dnallm.inference.vep.evaluate_vcf kernel, with VepResult's skip_counts/skipped/skip_fraction/convention blocks surfaced verbatim in the tool JSON (D-04, locked)"
    - "`hotspots` computes windows from `model` x `coordinates` parameters (server-side inference engine + find_hotspots), not from any precomputed-window-file interface (D-05); reference sequence comes from a per-call `fasta_path` parsed via vep._load_reference"
    - "All three tools are registered through `_with_timeout_wrapper` and never raise across the protocol boundary: validation failures and unknown models return `{\"error\": ..., \"isError\": true}` dicts (D-06); a missing/invalid model name returns the matchable not-loaded error dict"
    - "Input caps (max sequence length, max positions, max variant count) are enforced at the tool boundary BEFORE the engine call, with error dicts stating the caps so the timeout wrapper's suggestion is actionable"
    - "The event loop stays live during a long tool call — a concurrent health_check completes while a mocked engine call is in flight"
    - "The two tests that codified the old YAML-beats-CLI precedence are flipped in the same change as the fix, and the complementary yaml-only and CLI-explicit cases exist for both transports"
    - "No temp-file or output path is ever constructed from VCF record fields — inline variants materialize to a fixed sanitized temp name; a security test asserts no record-derived string appears in any path"
  artifacts:
    - "dnallm/mcp/server.py (sentinel-based host/port resolution in start_server + starters; argparse default=None; three new tools registered and implemented)"
    - "tests/mcp/test_server_tools_v12.py (per-tool JSON contracts with mocked engines, liveness test, security path test)"
    - "tests/mcp/test_server_transports.py (EXPECTED_TOOLS 16; round trips for the 3 new tools; precedence tests on both transports; the flipped streamable-http construction test)"
    - "tests/mcp/test_start_server.py (defaults-forwarded test flipped to the None-sentinel chain)"
    - "CHANGELOG.md ## [Unreleased] (+ REV-11 bullet incl. the host/port fix and the default-divergence note, same commit as the code)"
  key_links:
    - "_zero_shot_score -> dnallm.inference.vep.evaluate_vcf (Phase-11 kernel, sole VCF path — vep is never re-implemented in server.py)"
    - "_hotspots -> ModelManager.get_inference_engine + Mutagenesis.find_hotspots (D-05 workflow parity)"
    - "_ism_scan -> the _dna_mutagenesis body template (validation-first, caps, error dicts, executor-safe engine access)"
    - "argparse --host/--port -> start_server(host: str | None, port: int | None) -> BOTH starters receive final resolved values (one resolution point)"
  prohibitions:
    - "No raising across the MCP protocol boundary in any tool; no bare (unwrapped) tool registration (D-06)"
    - "No new MCP-config schema fields (per-call parameters only — the recorded reference-sourcing decision)"
    - "No re-import of scikit-allel in server.py — VCF reading goes through vep.evaluate_vcf only"
    - "No blocking sync torch calls directly in an async tool body — engines come from get_inference_engine; loads stay out of tools"
    - "No path construction from VCF/variant record fields; temp VCF uses a fixed sanitized name"
    - "No streaming-tool detour for long ISM — caps + timeout suggestion per D-06/Pitfall 4"
    - "No test-file basename collision across tests/ and dnallm/mcp/tests/ (new file is uniquely named test_server_tools_v12.py)"
    - "No example/ or tests/examples/ changes (census pin must stay 208/217)"
    - "No dnallm/__init__.py changes; no new dependencies"
---

<objective>
MCPE-01 (REV-11): the MCP server gains `ism_scan`, `hotspots`, `zero_shot_score` tools
wrapping the existing Mutagenesis/VEP surfaces under the house `_with_timeout_wrapper` +
error-dict conventions, and the deferred `--host/--port` silently-overridden-by-yaml bug
is fixed with CLI precedence on BOTH sse and streamable-http transports.

Purpose: LLM agents must be able to drive ISM, hotspot scanning, and zero-shot scoring
over MCP with honest flags — and operators must be able to bind a host/port from the CLI
without YAML silently winning (v1.1-audit deferred bug).
Output: server.py tool surface + precedence fix, flipped/extended MCP test suite, and the
REV-11 CHANGELOG entry.
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
@dnallm/mcp/server.py
@dnallm/mcp/model_manager.py
@dnallm/inference/vep.py
@dnallm/inference/mutagenesis.py
@tests/mcp/test_server_transports.py
@tests/mcp/test_start_server.py
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Host/port CLI-precedence fix on both transports — sentinel resolution + flipped tests</name>
  <files>dnallm/mcp/server.py, tests/mcp/test_server_transports.py, tests/mcp/test_start_server.py</files>
  <read_first>
  - dnallm/mcp/server.py:1676-1760 (start_server + the unconditional override at 1743-1747), 1849-1899 (_start_http_server + the always-true conditional at 1853-1857), 1990-2010 (argparse --host default="0.0.0.0" / --port default=8000)
  - 12-RESEARCH.md: Pattern 6 (verified bug mechanics + fix shape + the exact tests that must flip), Code Examples "Host/port sentinel resolution", Pitfall 2 (desync between the two transports)
  - 12-PATTERNS.md: "dnallm/mcp/server.py — host/port CLI-precedence fix" section
  - tests/mcp/test_server_transports.py:276-400 (TestStreamableHTTPConstruction + TestSSEConstruction._run_sse_start patched-uvicorn pattern)
  - tests/mcp/test_start_server.py:66-145 (TestMain defaults/parsed-args tests)
  </read_first>
  <action>
  Implement the sentinel-based fix per RESEARCH Pattern 6 (this is the v1.1-audit deferred MCPE-01 bug): argparse `--host`/`--port` become `default=None` with `--help` documenting the resolution chain (CLI-explicit > transport YAML > server YAML > default); `start_server(host: str | None = None, port: int | None = None, ...)` resolves ONCE before dispatch — when None, resolve per transport (streamable_http block on the http path, else server block), falling back to ONE documented default; pick `127.0.0.1:8000` (start_server's current documented default) and note the argparse `0.0.0.0` divergence in the Task-3 CHANGELOG bullet; DELETE both override blocks — the unconditional `if server_config: host = server_config.server.host; port = server_config.server.port` (1743-1747) and the always-true `if server_config.server.host == host:` pair in `_start_http_server` (1853-1857) — both starters receive final values; the stdio path is untouched. Flip the two codifying tests in the SAME change: test_config_fields_assembled_from_streamable_http_block now asserts explicit CLI 8123 beats YAML streamable_http 8124, and gains the complementary yaml-only case (no CLI -> streamable_http block wins); add SSE-side precedence equivalents via the existing _run_sse_start patched-uvicorn pattern; test_defaults_are_forwarded flips from asserting host="0.0.0.0"/port=8000 to the None-sentinel forwarding (or resolved chain if resolution moves into main); test_starts_server_with_parsed_args (explicit flags forwarded) must still pass unchanged. Precedence matrix covered by tests on BOTH transports: CLI-explicit > yaml > default.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/mcp/test_server_transports.py tests/mcp/test_start_server.py -q -m "not slow"</automated>
    <fails_when>nonzero exit — e.g. the flipped construction test still expects YAML 8124 to beat explicit 8123, or a precedence case on either transport forwards the wrong value to uvicorn.Config</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: both override blocks deleted; one resolution point; stdio branch unchanged; no `default="0.0.0.0"` / `default=8000` remains on the two argparse flags
  - Test: precedence matrix (CLI-explicit / yaml-only / no-config) asserted for sse AND streamable-http via patched uvicorn kwargs; the two named tests flipped same-change
  - Behavior: explicit CLI flags win on both transports; yaml-only resolves from the transport-appropriate block; neither transport desyncs (Pitfall 2)
  </acceptance_criteria>
  <done>The --host/--port CLI flags take precedence over YAML on both sse and streamable-http, proven by the flipped + new construction tests; stdio unregressed; zero behavior change when no flags are passed beyond the documented default unification.</done>
  <reversibility rating="reversible">Entry-point argument handling; no persistence, no protocol change; revert restores prior precedence.</reversibility>
</task>

<task type="auto">
  <name>Task 2: `ism_scan` + `hotspots` tools — registration, caps, mocked-engine contracts, liveness, EXPECTED_TOOLS 15</name>
  <files>dnallm/mcp/server.py, tests/mcp/test_server_tools_v12.py, tests/mcp/test_server_transports.py</files>
  <read_first>
  - dnallm/mcp/server.py:1224-1451 (_dna_mutagenesis — the validation-first template: registry check, ^[ACGTacgtNn]+$ content check, combo caps with error dicts stating them, get_inference_engine None-path, response shape, catch-all), 263-292 (registration block), 282+ (_with_timeout_wrapper)
  - 12-PATTERNS.md: "dnallm/mcp/server.py — three new tools" section (the 4-point skeleton to copy exactly)
  - 12-RESEARCH.md: Pattern 4 (per-tool surfaces + caps rationale), Pitfall 3 (EXPECTED_TOOLS same-change), Pitfall 4 (timeout vs long ISM)
  - dnallm/mcp/model_manager.py:224 (get_inference_engine), 99-121 (executor bridge precedent)
  - dnallm/inference/mutagenesis.py:517-581 (find_hotspots signature)
  - 12-CONTEXT.md decisions D-05/D-06
  - tests/mcp/test_server_transports.py:44-58 (EXPECTED_TOOLS), 134-247 (real_server fixture + ASGI round-trip pattern with localhost base_url)
  </read_first>
  <action>
  Register and implement the first two tools per D-06 (all through `self.app.tool()(self._with_timeout_wrapper(self._ism_scan, "ism_scan"))`-style registrations — wire names carry the leading underscore via update_wrapper). `_ism_scan` mirrors the _dna_mutagenesis surface (model_name, sequence/sequences, mutation_type, positions) with the validation-first body: non-empty model_name -> registry allowed-values check -> per-field validation -> input caps (max sequence length, max positions count) enforced BEFORE the engine call, error dicts stating the caps (Pitfall 4 — the timeout suggestion must be actionable); engine from get_inference_engine, None -> the matchable not-loaded error dict; response shape {"content": [...], ..., "model_name": ...}; whole body in try/except logging exc_info=True returning the generic isError dict. `_hotspots` per D-05: parameters model_name + coordinates ({chrom, start, end}) + fasta_path (per-call, server-side file; suffix allowlist .fasta/.fa/.fa.gz/.fna; missing file -> matchable error dict naming the tool) + optional strategy/window_size/percentile_threshold; sequence slice via vep._load_reference; ISM preds through the engine; `Mutagenesis.find_hotspots(preds, strategy, window_size, percentile_threshold)` -> window list returned as JSON; NOT a precomputed-window-file interface. Blocking torch stays off the event loop (already-loaded engines only; executor precedent where sync-heavy). Tests in the NEW tests/mcp/test_server_tools_v12.py (unique basename — no collision with dnallm/mcp/tests/): per-tool JSON contracts with Mock() inference engines (happy path + not-loaded model + invalid sequence + cap-exceeded + missing fasta — the MCPE-01 empty/boundary error-dict predicates); the liveness test (concurrent health_check completes while a long mocked tool call is in flight). In tests/mcp/test_server_transports.py: EXPECTED_TOOLS grows by exactly _ism_scan and _hotspots (13 -> 15) with the `len(names) == 13` assert flipped to 15 in the SAME change (Pitfall 3), plus in-memory ASGI round trips for both tools (error-dict paths first; happy paths via mocked get_inference_engine).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/mcp/test_server_tools_v12.py tests/mcp/test_server_transports.py -q -m "not slow" -k "ism or hotspot or liveness or list_tools or round_trip"</automated>
    <fails_when>nonzero exit — e.g. list_tools still counts 13 names, health_check stalls behind a long tool call, or the not-loaded model case returns a raised exception instead of the isError error dict</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: both tools registered through _with_timeout_wrapper (never bare); caps constants defined and enforced before engine access; no raise crosses the protocol boundary
  - Test: EXPECTED_TOOLS == 15 asserted; JSON contracts green for happy + all four error predicates; liveness green; ASGI round trips green for both tools
  - Behavior: hotspots derives windows from model x coordinates + fasta_path (D-05), never from a window file; ism_scan mirrors the mutagenesis surface (D-06)
  </acceptance_criteria>
  <done>ism_scan and hotspots callable end-to-end through the in-memory ASGI client returning valid JSON; error-dict paths (invalid/missing model, bad sequence, cap exceeded, missing fasta) all return matchable isError dicts; loop liveness proven; EXPECTED_TOOLS at 15.</done>
</task>

<task type="auto">
  <name>Task 3: `zero_shot_score` (dual-mode, skip accounting verbatim) + EXPECTED_TOOLS 16 + security path test + REV-11 CHANGELOG</name>
  <files>dnallm/mcp/server.py, tests/mcp/test_server_tools_v12.py, tests/mcp/test_server_transports.py, CHANGELOG.md</files>
  <read_first>
  - dnallm/inference/vep.py:734-800 (evaluate_vcf full signature + VepResult fields: records, skip_counts, evaluated, skipped, skip_fraction, metrics, convention), 451-501 (_load_reference), 660+ (score_variant paradigm guard)
  - 12-RESEARCH.md: Pattern 4 (_zero_shot_score dual-mode design + Open Question 3 recommendation), Pattern 5 (fasta_path per-call decision), Security Domain (untrusted VCF threat seed), Pitfall 4 (variant-count cap)
  - 12-CONTEXT.md decision D-04 (dual-mode locked) + specifics ("skip accounting visible in the tool response (locked)")
  - tests/inference/data/synthetic_variants.vcf + synthetic_reference.txt (Phase-11 committed fixture precedent — reuse pattern for the tool's mocked tests)
  - CHANGELOG.md ## [Unreleased] block (entry shape; re-read immediately before edit)
  </read_first>
  <action>
  Implement `_zero_shot_score` per D-04 dual-mode: parameter `variants` (inline JSON list of {chrom, pos, ref, alt} — per-field validation: pos int > 0, ref/alt non-empty ACGT strings, chrom non-empty string; matchable error dicts naming the failing field) AND `vcf_path` (server-side file, suffix allowlist .vcf/.vcf.gz, size-capped read). Exactly one of the two modes per call (both/none -> matchable error dict). Inline variants materialize to a server-side temp VCF under tempfile with a FIXED sanitized filename — never a name derived from record fields (RESEARCH Open Question 3 recommendation) — so BOTH modes route through the SAME `vep.evaluate_vcf` kernel with identical skip accounting. Reference from the per-call `fasta_path` parameter parsed via `vep._load_reference` (per-call parameter, no MCP-config schema change — recorded decision); expose `paradigm` and a ClinVar-convention override parameter mapping to evaluate_vcf's clnsig_filter (non-ClinVar VCFs opt out of the D-17 default; A8 discretion) plus context_window. The tool response surfaces VepResult's skip_counts, skipped, skip_fraction, and convention blocks VERBATIM alongside per-record scores (locked specificity). Variant-count cap enforced before the kernel with an error dict stating the cap. EXPECTED_TOOLS -> 16 (three new names) and the len assert flips to 16 in the SAME change; ASGI round trip for zero_shot_score. Security test: after an inline-variants call, assert no path of any produced temp/output artifact contains record-derived strings (chrom values, ref/alt) — the vep.py contract extended to the tool layer. Window-edge variants flow through as vep's skip-as-data (boundary predicate; do not re-implement the edge in the tool). Append the REV-11 bullet to CHANGELOG.md ## [Unreleased] same-commit: dense bullet covering the three tools, the dual-mode kernel reuse, AND the host/port precedence fix with the documented-default change note (0.0.0.0 -> 127.0.0.1 unification); inline reviewer tag per the intake REV-11 mapping; re-read immediately before the append-only edit.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/mcp/test_server_tools_v12.py tests/mcp/test_server_transports.py -q -m "not slow" -k "zero_shot or list_tools or security or round_trip"</automated>
    <fails_when>nonzero exit — e.g. skip_counts is absent from the zero_shot_score response JSON, the list_tools count asserts 15 instead of 16, or the security test finds a record-derived string inside a temp/output path</fails_when>
  </verify>
  <acceptance_criteria>
  - Source: both input modes converge on one evaluate_vcf call; scikit-allel never imported in server.py; skip/convention blocks serialized verbatim into the response
  - Test: dual-mode contracts (inline happy path, vcf_path happy path, both-modes error, none error, per-field validation errors, cap error, missing fasta/model error dicts) green; EXPECTED_TOOLS == 16; security path test green
  - Edge: MCPE-01 empty/boundary predicates — missing/invalid model -> error dict; window-edge variant -> skip-as-data visible in skip accounting; missing fasta -> matchable error dict
  - CHANGELOG: REV-11 bullet present with the host/port fix + default note, same commit as the tool code
  </acceptance_criteria>
  <done>zero_shot_score callable both ways with skip accounting verbatim in the JSON; all 16 tools listed; the protocol boundary never sees a raise; security path invariant tested; REV-11 evidence-chain entry landed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| MCP client -> server tools | LLM-agent-controlled arguments (model_name, sequences, variants, coordinates) cross into server code |
| client-named files -> server filesystem | `vcf_path`/`fasta_path` are operator-trust-boundary server-side reads of client-named files |
| VCF/FASTA file content -> parsers | Untrusted file content reaches allel/vep/_load_reference parsers |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-12-04 | Tampering | _zero_shot_score vcf_path/fasta_path handling | high | mitigate | Suffix allowlist (.vcf/.vcf.gz/.fasta/.fa/.fa.gz/.fna), size caps, strict parse-or-reject surfaced as matchable error dicts; never exec/eval file content |
| T-12-05 | Elevation | temp-VCF materialization from inline variants | high | mitigate | Fixed sanitized temp filename under tempfile (never from record fields); security test asserts no record-derived string appears in any path |
| T-12-06 | DoS | all three tools (result/input explosion into agent context) | medium | mitigate | Caps at the tool boundary (sequence length, positions, variant count) with error dicts stating the caps; timeout wrapper inherited (D-06) |
| T-12-07 | Repudiation | model_name routing in the three tools | medium | mitigate | Preserve the existing ModelManager registry validation (arbitrary model names never reach a downloader); not-in-registry -> matchable error dict |
| T-12-08 | Information Disclosure | per-call fasta_path/vcf_path server-side reads (SSRF-adjacent) | low | mitigate | Operator trust boundary documented in tool docstrings; suffix allowlist + resolve/contain paths server-side; no URL fetching from tool args |
| T-12-SC | Tampering | package installs | high | accept | Zero package installs this phase (zero-new-dependencies milestone invariant) — nothing to gate |
</threat_model>

<verification>
- Targeted lane green: `uv run --no-sync pytest tests/mcp -q -m "not slow"` — 0 failed (both edited files + the new file).
- Tool-surface proof: `uv run --no-sync pytest tests/mcp/test_server_transports.py -q -k "list_tools"` — 16 names, exact set equality with EXPECTED_TOOLS.
- Per-module standard: `uv run --no-sync coverage run -m pytest tests/mcp -q -m "not slow" && uv run --no-sync coverage report --include="dnallm/mcp/server.py"` — row >= 96%.
- Sibling-suite safety: `uv run --no-sync pytest dnallm/mcp/tests -q -m "not slow"` — still green (packaged MCP test root unregressed).
- Census safety: `uv run --no-sync pytest tests/ --collect-only -q | tail -3` still 208/217 collected (9 deselected).
</verification>

<success_criteria>
- MCPE-01 delivered: three tools handshake-clean (server up -> client calls all 3 -> JSON), host/port CLI precedence proven on both transports with the two codifying tests flipped, EXPECTED_TOOLS at 16, error-dict + timeout conventions intact throughout.
- All MCPE-01 edge predicates covered by named tests (missing/invalid model error dict; window-edge variant skip-as-data; missing fasta error dict).
- Zero new dependencies, no config-schema change, facade byte-stable, no raise across the protocol boundary, security path invariant tested.
- REV-11 CHANGELOG entry landed same-commit including the default-unification note.
</success_criteria>

## Artifacts this phase produces
(This plan's share; the phase-level rollup lives in 12-03.)
- `dnallm/mcp/server.py` — three new MCP tools (`ism_scan`, `hotspots`, `zero_shot_score`) + the host/port CLI-precedence fix
- `tests/mcp/test_server_tools_v12.py` — per-tool JSON contracts, liveness, security path tests
- Extended `tests/mcp/test_server_transports.py` (EXPECTED_TOOLS 16, round trips, precedence matrix) and `tests/mcp/test_start_server.py` (sentinel defaults)
- `CHANGELOG.md` — REV-11 entry under ## [Unreleased]

<output>
Create `.planning/phases/12-motif-matching-mcp-tools-milestone-closeout/12-02-SUMMARY.md` when done.
Record the commit SHA of the REV-11 CHANGELOG append in the SUMMARY — 12-03 backfills it as
the evidence-chain link.
</output>
