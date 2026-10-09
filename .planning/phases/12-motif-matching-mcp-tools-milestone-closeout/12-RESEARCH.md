# Phase 12: Motif Matching, MCP Tools & Milestone Closeout - Research

**Researched:** 2026-10-10
**Domain:** PWM motif scanning (FIMO-convention exact-DP calibration), MCP server tool expansion + host/port precedence fix, milestone closeout (docs / CHANGELOG / census / coverage docs)
**Confidence:** HIGH for the FIMO algorithm (pinned from MEME source), JASPAR API (live-probed HTTP 200), and all in-repo facts (every line cited was read this session); MEDIUM for the HBG1/BCL11A golden-fixture coordinates (paper not publicly indexed — owner must pin); flagged per-claim below.

## Summary

Phase 12 is three owner-fixed agents with file-disjoint scopes. **C1 (REV-10 / MOTIF-01)** builds `dnallm/interpret/motifs.py`: a JASPAR REST client (stdlib, base URL parameterized, `format=meme`) plus a FIMO-convention scanner. The flagged plan-time spike — the D-01 exact-DP p-value calibration — is now fully pinned from MEME Suite source: FIMO discretizes each motif's log-odds matrix to integers in `[0..100]` per column (`#define PSSM_RANGE 100`, `pssm.h:13`), runs a column-wise convolution DP of size `w*100+1` under the zero-order background, and reads p-values off a reverse cumulative sum (`pv[x] = Pr(score >= x)`). This is directly implementable in pure Python (≈360k float ops per motif at w=30) — D-02's "pure-Python column-wise scan, no numpy score-discretization approximation" is satisfied by adopting FIMO's *own* integer scaling deliberately (it IS the standard exact form — Staden 1989 convention; "exact" contrasts with empirical-null sampling, not with discretization). Threshold = minimal scaled score with p < 1e-4, inverted back to bits as `score_bit = scaled/scale + w*offset`; BH q-values over the FULL window×motif set via `scipy.stats.false_discovery_control` (verified in the project venv, scipy 1.18.1, signature `(ps, *, axis=0, method='bh')`).

**C2 (REV-11 / MCPE-01)** adds `ism_scan`, `hotspots`, `zero_shot_score` to `dnallm/mcp/server.py` and fixes the `--host/--port` bug. The bug mechanics are verified at source: `start_server` (server.py:~1743-1747) unconditionally overwrites host/port from the YAML `server` block, so CLI flags never win; the streamable-http starter then compares `server_config.server.host == host` — always true after the first override — so the `streamable_http` block *always* wins on that transport. Fix shape: `str | None = None` sentinels on `start_server` + argparse `default=None`, resolution CLI-explicit > transport YAML > default. Two existing tests codify today's buggy precedence and must flip in the same change (`test_server_transports.py::test_config_fields_assembled_from_streamable_http_block` asserts YAML 8124 beats explicit 8123; `test_start_server.py::test_defaults_are_forwarded` asserts `0.0.0.0:8000` forwarded). The handshake regression pattern already exists and extends mechanically: `EXPECTED_TOOLS` (13 → 16) plus in-memory ASGI client sessions calling each new tool. One design gap surfaced: `hotspots` (D-05: model × coordinates) and `zero_shot_score` (reference for `evaluate_vcf`) both need a reference-genome source, and **no FASTA field exists in the MCP config today** — recommend a per-call `fasta_path` parameter (reusing `vep._load_reference`, avoiding the optional pyfastx dep).

**C3 (closeout)**: IA³ chapter completion is a targeted rewrite of `docs/user_guide/fine_tuning/peft_adapters.md:174-186` ("IA³ (coming in the next release)" → real usage; the Phase-11 trainer branch landed). CHANGELOG `## [Unreleased]` carries REV-01..REV-09 with inline `(REV-ID, R#-#)` tags; C1/C2 append REV-10/REV-11 same-change per D-09, C3 backfills commit SHAs as links. Census re-pin (D-07) is verified NOT needed today: the measured collect line is exactly `208/217 tests collected (9 deselected)`, matching the pin at `.github/workflows/ci.yml:865-871` (Phase 11 touched neither `example/` nor `tests/examples/`). Coverage-expectation docs = `docs/user_guide/continuous_integration.md` ("96.30%" at lines 26/30) — C3 updates honestly with the measured post-Phase-12 number.

**Primary recommendation:** Implement the scanner as a literal transcription of FIMO's recipe (PSSM_RANGE=100 scaling → column DP → reverse cumsum → threshold at p<1e-4 → BH via scipy), with provenance comments citing the MEME source lines; wire the three MCP tools through the existing wrapper/executor/error-dict patterns with `EXPECTED_TOOLS` extended and the two precedence tests flipped; get the Fig 4a window coordinates from the owner BEFORE the golden test is written (blocked input, see Open Questions).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** p-value calibration uses **exact dynamic programming over the log-odds null distribution** (the MEME/FIMO documented convention) — cross-motif comparable; provenance comments + honest documentation of the choice per REQUIREMENTS.
- **D-02:** The DP is a **pure-Python column-wise scan over the PWM** (widths ≤~30, alphabet 4 — standard practice, keeps zero-new-dependencies intact); no numpy score-discretization approximation.
- **D-03:** Threshold = FIMO default **p<1e-4** resolved from the DP cumulative distribution; **BH q<0.05 over the FULL window×motif test set** (never per-sequence correction).
- **D-04:** `zero_shot_score` takes **dual-mode input** — `variants` inline JSON list ({chrom,pos,ref,alt}) for light callers AND `vcf_path` server-side file for batch — both routing through the same `evaluate_vcf` kernel with skip accounting.
- **D-05:** `hotspots` computes windows from **`model` × `coordinates` parameters** (server-side inference engine), matching the paper's hotspot-scan workflow — not a precomputed-window-file interface.
- **D-06:** `ism_scan` follows the existing mutagenesis engine surface; all three tools use the existing `_with_timeout_wrapper` + error-dict conventions (locked by REQUIREMENTS).
- **D-07:** Census re-pin is **minimal**: only if Phase 11's added test counts broke a hard-asserted count; plus the honest coverage-expectation docs update. No wholesale count re-freeze.
- **D-08:** IA³ docs chapter section completes the peft_adapters.md forward pointer (finishing DOCS-01's split delivery); CHANGELOG finalization completes the evidence chain (SHA backfill per the D-09/Phase-10 mechanism).

### Claude's Discretion
- HBG1/BCL11A golden-fixture construction details (sequence window sourcing, coordinate rounding tolerance) — planner/researcher pins from the paper's Fig 4a.
- JASPAR client caching/retry specifics (retry-with-backoff house pattern applies).
- MCP tool JSON response field layout beyond the timeout/error-dict contracts.
- `interpret/` module registration point (`dnallm/interpret/` is new — planner decides __init__ shape; no root-facade re-export either way).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MOTIF-01 (REV-10) | `dnallm/interpret/motifs.py` — hotspot windows ↔ JASPAR/CIS-BP PWM similarity scan following FIMO conventions (GC-matched background, both strands, log-odds threshold p<1e-4, BH FDR via `scipy.stats.false_discovery_control`), emitting a motif-ID/coordinates/E-value table; a stdlib JASPAR REST client with the base URL parameterized (canonical host migrated to jaspar.elixir.no; prefer `format=meme`); acceptance: the HBG1/BCL11A motif hit coordinates match the paper's Fig 4a annotation; p-value calibration method documented honestly | FIMO algorithm pinned from MEME 4.8.1 source (scaling constant, DP, threshold inversion — Code Examples); JASPAR endpoints live-verified; CIS-BP has no REST API → local-parse input (Assumptions/Open Questions); golden fixture construction plan + literature loci anchors |
| MCPE-01 (REV-11) | MCP server gains `ism_scan`, `hotspots`, `zero_shot_score` tools wrapping existing classes (existing `_with_timeout_wrapper` + error-dict-not-raise conventions; `zero_shot_score` wraps the VEP-01 module); handshake regression tests: server up → client calls all 3 tools → JSON assertions; the known `--host/--port` silently-overridden-by-yaml bug fixed on BOTH sse and streamable-http paths with CLI-precedence tests | Bug mechanics verified at server.py:1743-1747 + 1854-1857 with exact fix shape; existing test patterns enumerated (EXPECTED_TOOLS, ASGI round trip, construction tests); wrapped-class surfaces verified (Mutagenesis, evaluate_vcf, get_inference_engine); reference-source design gap identified |
| DOCS-01 completion (carried) | IA³ chapter section completes in Phase 12 (after PEFT-01) | Exact section + current text pinned at `docs/user_guide/fine_tuning/peft_adapters.md:174-186`; Phase-11 CHANGELOG entry confirms the trainer branch landed ("the Phase-10 interim warn is demolished") |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Same-change pytest rule (owner rule, all phases):** any `dnallm/` code modification ships with pytest coverage in the same change; working standard is ≥96% per-module mocked fast-lane coverage (the 90% gate is a floor, not the standard — PITFALLS #13).
- **Zero new dependencies:** scipy + numpy + stdlib suffice (REQUIREMENTS out-of-scope list rejects biopython/statsmodels/torchmetrics; scikit-allel was the milestone's only sanctioned addition, already landed in Phase 11). The JASPAR client must be stdlib `urllib.request`, not `requests`/`httpx`.
- **No new test frameworks**; pytest config lives in `pyproject.toml` `[tool.pytest.ini_options]` (asyncio auto mode, timeout 300s, `--strict-markers`).
- **Compatibility:** suite green on Python 3.11/3.12/3.13 (code must remain 3.10-compatible: `requires-python >=3.10`), numpy 1.26.4 & 2.2.0; no pins to a single transformers minor.
- **Facade byte-stable:** no new `dnallm/__init__.py` re-exports — `dnallm/interpret/` gets its own `__init__.py` only (CONTEXT discretion; REQUIREMENTS out-of-scope).
- **Style:** ruff format (line 100), PEP 604 unions (`str | None`), relative imports inside `dnallm/`, absolute in tests; `ValueError` with matchable messages for invalid input; Google-style docstrings; comments in English; no bare `print` in library code (`T20`).
- **Coverage omit / lint exclude globs must not swallow the new module:** `dnallm/interpret/` is measured, linted, and mypy-checked (it is NOT under any exclude glob — `dnallm/tasks/metrics/`, `mamba_npu.py`, `megatron.py` are the only vendored exclusions). Same-change check: `coverage report` must show a measured row for `dnallm/interpret/motifs.py` (PITFALLS #2 class).
- **Skip discipline:** new typed skips (e.g. JASPAR network-unavailable) go into `tests/expected_skips.yaml` in the SAME change (audit gate `scripts/audit_skips.py` runs on all CI legs); `slow` marking from birth for real-model/network tests; `models.lock` rows for any newly-downloaded acceptance model.
- **CHANGELOG D-09 discipline:** entries land in the same commit as their fix, REV-ID + reviewer-comment inline (e.g. `(REV-10, R1-3d)`), unique-anchor insert into `## [Unreleased]`, re-read immediately before edit.
- **GSD workflow enforcement:** phase work runs through `/gsd-plan-phase` → `/gsd-execute-phase`; owner execution directive: targeted verify commands + lane test files only.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PWM parsing (MEME/JASPAR/CIS-BP formats) | API / Backend (`dnallm/interpret/motifs.py`) | — | Pure library function; no I/O policy |
| JASPAR REST fetch | API / Backend (`motifs.py` client) | — | stdlib network client; base URL parameterized; retry-with-backoff house pattern |
| p-value calibration (exact DP) | API / Backend (`motifs.py`) | — | Pure computation; FIMO-recipe transcription |
| FDR correction | API / Backend (`motifs.py` → `scipy.stats`) | — | Single global BH over the full window×motif set |
| Hotspot-window scanning | API / Backend (`motifs.py`) | — | Consumes windows produced by inference-side ISM (via MCP `hotspots` tool or library callers) |
| MCP tool exposure (3 new tools) | API / Backend (`dnallm/mcp/server.py`) | ModelManager (engine resolution, executor bridge) | Tools wrap library classes; blocking torch must go through `run_in_executor` |
| Host/port precedence | API / Backend (`server.py` main/start_server/starters) | — | CLI parsing + bind-address resolution before uvicorn |
| Reference-genome sourcing for tools | API / Backend (per-call `fasta_path` param; `vep._load_reference` reader) | — | No FASTA field in MCP config today; per-call parameter avoids config-schema collision |
| Golden fixture (HBG1/BCL11A) | Test assets (`tests/` fixtures, committed FASTA + MEME files) | — | v1.1 committed-loci precedent; network-free fast lane |
| Closeout docs/CHANGELOG/census | Documentation + CI config | — | docs pages, CHANGELOG.md, ci.yml pin (re-pin only if broken) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy (already a core dep) | `>=1.15.2` (installed 1.18.1) | `scipy.stats.false_discovery_control(ps, method="bh")` for BH q-values | REQUIREMENTS names it explicitly; signature verified live in the project venv: `(ps, *, axis=0, method='bh')` [VERIFIED: venv `python -c "import scipy; ..."` 2026-10-10] |
| numpy (already a core dep) | per pyproject | Sliding-window scan loop (vectorized log-odds over positions) | PITFALLS performance table recommends numpy windows for scanning; D-02 constrains only the DP to pure Python |
| stdlib `urllib.request` | — | JASPAR REST client | REQUIREMENTS: "stdlib JASPAR REST client"; zero-new-deps constraint [VERIFIED: REQUIREMENTS.md MOTIF-01] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-allel (already landed Phase 11) | per pyproject | `evaluate_vcf` VCF reading inside `zero_shot_score` | Only via `dnallm/inference/vep.py`; never re-imported in server.py directly [VERIFIED: vep.py:734-800 read this session] |
| pyfastx (`dev` extra only) | dev extra | `utils/genomic_coords.fetch_sequence` | NOT for MCP tools — optional dep; prefer `vep._load_reference` pure-Python FASTA reader [VERIFIED: genomic_coords.py:130 docstring "pyfastx"; vep.py:451] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure-Python FIMO-recipe DP | MOODS / motifmatchpy | Rejected: MOODS = new C++ dep; motifmatchpy needs Python 3.12+/numba — breaks the 3.11–3.13 matrix (FEATURES.md REV-10) |
| numpy sliding-window scan | Pure-Python scan loop | Pure Python fine for a few kb of hotspot windows; numpy if windows grow (deferred GENOMEWIDE-SCAN is v1.3+) |
| Per-call `fasta_path` MCP parameter | MCP config `reference` field | Config field = config_manager/schema change (collision surface); per-call param = smaller diff, per-request flexibility (recommended; planner decides) |

**Installation:**
```bash
# NOTHING new. Zero new dependencies — scipy/numpy already core deps, verified installed:
.venv/bin/python -c "import scipy, numpy; from scipy.stats import false_discovery_control"   # scipy 1.18.1 OK
```

**Version verification:** performed this session — scipy 1.18.1 in `.venv` with `false_discovery_control` importable and signature-verified; `false_discovery_control` was added in scipy 1.11 (2023), so the `>=1.15.2` floor is safely above it [VERIFIED: venv probe; scipy docs history ASSUMED for the 1.11 introduction].

## Package Legitimacy Audit

> No packages are installed by this phase (zero-new-dependencies constraint, D-02/STATE.md). The audit is therefore empty by construction.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| *(none — no new packages)* | — | — | — | — | — | — |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                    C1: dnallm/interpret/motifs.py (NEW)
┌──────────────────────────────────────────────────────────────────┐
│  JASPAR REST (stdlib)          PWM loading         Scanner        │
│  GET /api/v1/matrix/{id}/  →   MEME-format parse → per-motif:     │
│    ?format=meme                (+CIS-BP local      1. log-odds vs  │
│  GET /api/v1/matrix/?name=...  pwm tables)            GC-matched  │
│    &tax_id&collection&release  pseudocount 0.1×bg      background │
│        │ retry/backoff            │                  2. scale to  │
│        ▼                          ▼                     [0..100]  │
│   motif records ────────────► PFM→PWM (bits)       3. column DP   │
│                                                     (null dist)   │
│  target windows (from ISM hotspots        4. reverse cumsum pv[x] │
│  or library callers) ──► GC background ──► 5. threshold = min x   │
│                                              with pv[x] < 1e-4    │
│        both strands: motif on stated strand + revcomp scan        │
│        ──► hits (id, coords, strand, score, p) ──►                │
│   6. BH over FULL window×motif p-set (scipy) ──► q-values, E=pxN  │
└──────────────────────────────────────────────────────────────────┘

                    C2: dnallm/mcp/server.py (EDIT)
┌──────────────────────────────────────────────────────────────────┐
│ main() argparse ──► start_server(host|None, port|None)           │
│   resolution: CLI-explicit > YAML (server / streamable_http)     │
│             > documented default      [FIX — both transports]    │
│                                                                   │
│ _register_tools(): + _ism_scan, + _hotspots, + _zero_shot_score  │
│   all three: _with_timeout_wrapper(...) ──► asyncio.wait_for     │
│   sync torch ──► ModelManager executor bridge (run_in_executor)  │
│   errors ──► {"error": ..., "isError": True} dicts (never raise) │
│                                                                   │
│ _zero_shot_score: variants[] inline ─┐                           │
│                  vcf_path server-side ┴─► vep.evaluate_vcf       │
│                                          (skip accounting shown) │
│ _hotspots: model × coordinates (+fasta source) ──► ISM ──►       │
│            Mutagenesis.find_hotspots ──► windows                 │
│ _ism_scan: model × sequence/positions ──► Mutagenesis engine     │
└──────────────────────────────────────────────────────────────────┘

                    C3: closeout (docs + evidence chain)
┌──────────────────────────────────────────────────────────────────┐
│ docs/user_guide/fine_tuning/peft_adapters.md:174-186             │
│   "IA³ (coming in the next release)" → real usage chapter        │
│ CHANGELOG.md ## [Unreleased]: REV-10/11 appended same-change;    │
│   C3 backfills commit SHAs as links (D-09)                       │
│ docs/user_guide/continuous_integration.md "96.30%" → measured    │
│ ci.yml:865 census pin 208/217 — verified current, re-pin NOT     │
│   needed unless example/ artifacts are added                      │
└──────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
dnallm/interpret/               # NEW package (C1 sole owner)
├── __init__.py                 # minimal init; NO dnallm/__init__.py re-export (facade byte-stable)
└── motifs.py                   # JASPAR client + MEME/CIS-BP parsing + FIMO-convention scanner
tests/interpret/                # NEW (mirrors package layout)
├── test_motifs.py              # DP calibration, background, strands, parser, BH, golden HBG1
└── fixtures/                   # committed: HBG1/BCL11A FASTA windows + frozen JASPAR MEME files
tests/mcp/
├── test_server_transports.py   # EDIT: EXPECTED_TOOLS 13→16; ASGI round trip for 3 new tools;
│                               #      streamable-http host/port precedence tests (flipped)
├── test_start_server.py        # EDIT: defaults-forwarded test (argparse default change)
└── test_server_tools_v12.py    # NEW (suggested): per-tool JSON contract tests (mocked engines)
dnallm/mcp/server.py            # EDIT (C2 sole owner)
docs/user_guide/fine_tuning/peft_adapters.md   # EDIT (C3)
docs/user_guide/continuous_integration.md      # EDIT (C3)
CHANGELOG.md                                   # EDIT (C1/C2 append, C3 backfill)
```

### Pattern 1: FIMO-recipe exact-DP calibration (the D-01/D-02 spike answer)
**What:** Transcribe FIMO's own pipeline; the discretization is FIMO's documented internal convention, not an approximation layered on top.
**When to use:** Always, for every motif, per background (the DP depends on the background, so a GC-matched background computed from target windows gets its own DP table per motif).
**The five steps, pinned from MEME 4.8.1 source (all read this session):**

1. **PWM → log-odds bits with background-scaled pseudocount.** FIMO default `--motif-pseudo 0.1`, applied per count "after first multiplying by the corresponding background frequency" [CITED: meme-suite.org/meme/doc/fimo.html]. JASPAR MEME rows are probabilities with `nsites` in the header (live-verified: `letter-probability matrix: alength= 4 w= 11 nsites= 2000 E= 0`), so reconstruct counts and apply:
   `p_ij = (freq_ij * nsites + 0.1 * bg_j) / (nsites + 0.1)`; if `nsites == 0`, the probability form `(freq_ij + 0.1 * bg_j) / 1.1`.
   Then `score_ij = log2(p_ij / bg_j)` (bits).
2. **Integer scaling to [0..100].** `PSSM_RANGE = 100` [VERIFIED: MEME 4.8.1 `src/pssm.h:13` — `#define PSSM_RANGE 100`, quoted verbatim]. Per motif: `small`/`large` = min/max matrix entry; `scale = range/(large-small)`, `offset = small`; `scaled_ij = round((score_ij - offset) * scale)` [VERIFIED: `logodds.c:566-630` — `lo->scale = range/(large-small); ... lo->offset = small;` and the header comment `score_bit = (score/scale) + (w*offset)` at `logodds.c:50`].
3. **Column-wise DP over the null distribution.** Array size `w*range+1`; per column `i`, per letter `j`: `new_pdf[k + s_ij] += pdf[k] * bg[j]` [VERIFIED: `pssm.c` `get_pdf_table` — `double new = get_array_item(k+s, pdf_new) + (old * get_array_item(j, background));`].
4. **Reverse cumulative p-value table.** `pv[x] = Pr(score >= x)` [VERIFIED: `pssm.c:531,570` — `pssm->pv[x] = Pr(score >= x)`, quoted verbatim].
5. **Threshold + inversion.** Report matches with p < 1e-4 (FIMO default `--thresh`): threshold = smallest integer scaled score `x` with `pv[x] < 1e-4`. Invert to bits: `score_bits = x/scale + w*offset` (per `logodds.c:50`). For each hit's reported score use the same inversion (FIMO's reported scores carry this quantization — maintainer-confirmed [CITED: groups.google.com/g/meme-suite/c/BXv4pRCacN8]).

**D-02 reconciliation (write into the module docstring):** FIMO itself discretizes (`PSSM_RANGE=100`); "exact" in D-01 means the null distribution is computed exactly by DP (vs. empirical-null sampling, which FEATURES.md had recommended and the owner overrode). D-02's "no numpy score-discretization approximation" bars ad-hoc float-rounding tricks, not FIMO's documented integer scaling. The honest documentation requirement (REQUIREMENTS: "p-value calibration method documented honestly") is met by a module docstring stating: exact-DP per FIMO/MEME convention including the [0..100] integer scaling, pseudocount 0.1×background, zero-order GC-matched background, and BH over the full test set.

**Complexity check (why pure Python is fine):** DP array ≤ 30×100+1 = 3001 entries; work ≈ w × 4 × array-size ≈ 360k float ops per motif×background — trivial in pure Python lists.

### Pattern 2: GC-matched zero-order Markov background + both-strand scan
**What:** Background = per-letter frequencies counted over all target-window bases (`bg[b] = count(b)/total`, zero-order Markov, "0-order" per FIMO docs [CITED: fimo.html]). Ignore the `Background letter frequencies A 0.25 ...` line embedded in JASPAR MEME responses (it is JASPAR's placeholder uniform background — live-verified in both probe responses; FIMO would use a `--bfile` to override it exactly as we do).
**Both strands:** score the window with the motif PWM (strand `+` hits) and with the reverse-complement PWM (strand `−` hits, coordinates mapped back to forward-window coordinates). FIMO: "Both strands are scored if the alphabet is complementable"; `--norc` opts out [CITED: fimo.html]. The motif is scanned "on its stated strand" — JASPAR `strands: + -` declares both searchable.
**When to use:** Every scan; the background is a single per-scan-set input (compute once from all windows, use for every motif's DP — this is what makes p-values cross-motif comparable).
**Tests (CONTEXT established patterns):** synthetic GC-skewed sequence asserting uniform-vs-matched background changes the hit set; palindromic motif → same hits both strands at the same coordinates; non-palindromic motif → strand-specific positions (revcomp-bug canary per PITFALLS #11 warning signs).

### Pattern 3: JASPAR REST client (live-verified endpoints)
**What:** stdlib `urllib.request`, base URL parameterized (default `https://jaspar.elixir.no/api/v1/`), retry-with-backoff house pattern (`download_model` `max_try=3` + `time.sleep` precedent at `dnallm/models/model.py:317`).
**Endpoints (all live-probed 2026-10-10, HTTP 200):**
- Single matrix in MEME format: `GET /api/v1/matrix/MA2324.1/?format=meme` → `MEME version 4` text block [VERIFIED: live probe, response quoted in Code Examples]
- Search: `GET /api/v1/matrix/?name=BCL11A&collection=CORE&version=latest&release=2024&page_size=N` → paginated JSON (`count`, `results[]` with `matrix_id`, `name`) — probed with `name=BCL11A` returning `count: 2`, `MA2324.1 BCL11A CORE`, `MA2504.1 BCL11A CORE` [VERIFIED: live probe]
- Releases: `GET /api/v1/releases/` → JSON, latest 2026 (release 11), 2024 (release 10) also `active: Yes` [VERIFIED: live probe]
- No auth; no documented rate limits [CITED: jaspar.elixir.no/api/v1/docs]
**Version pinning:** pass `release=2024` (or the release the paper used) in searches so the motif set is frozen for the golden fixture's reproducibility; a config/module constant with a provenance comment. **CIS-BP:** no documented REST API exists — access is via static bulk ZIPs at `cisbp.ccbr.utoronto.ca/bulk.php` (`pwms/` per-motif files + `TF_Information.txt`) [CITED: re3data registry; TFutils/universalmotif docs]. ⇒ REQUIREMENTS' "JASPAR/CIS-BP" resolves as: JASPAR via the REST client; CIS-BP as a parseable LOCAL input (its per-motif `Pos A C G T` table). The parser should accept both formats; acceptance is JASPAR-live + CIS-BP-local-parse (owner-confirmed reading in Open Questions).

### Pattern 4: MCP tool registration + the three new tools
**What:** Extend `_register_tools()` (server.py:251-292) with three `self.app.tool()(self._with_timeout_wrapper(self._x, "x"))` registrations — wire names derive from the method `__name__` (leading underscore preserved; `functools.update_wrapper` inside `_with_timeout_wrapper`) [VERIFIED: tests/mcp/test_server_transports.py:50-56 comment + EXPECTED_TOOLS set].
**When to use:** all three tools (D-06 locked). Conventions each tool inherits: `asyncio.wait_for` timeout → structured timeout error dict with `suggestion`; error dicts never raises across the protocol boundary; blocking sync torch calls bridged via `ModelManager` executor pattern (`_load_model_sync` in executor, lock inside the closure — model_manager.py:99-121) [VERIFIED: read this session].
**Surfaces to wrap (all verified this session):**
- `_ism_scan`: mirror `_dna_mutagenesis` (server.py:1224+) — `model_name`, `sequence`/`sequences`, `mutation_type`, `positions`; engine from `model_manager.get_inference_engine(model_name)` (model_manager.py:224). Cap sequence length / positions count (timeout reality: >6000 forward passes for 2kb ISM; the wrapper's timeout dict carries the `suggestion` to reduce inputs — PITFALLS #12a).
- `_hotspots` (D-05: model × coordinates): coordinates → sequence (reference source — see Pattern 5) → ISM via the engine → `Mutagenesis.find_hotspots(preds, strategy, window_size, percentile_threshold)` (mutagenesis.py:517, signature verified) → window list. Return windows (+ optional immediate motif scan if C1's module is importable — planner decides; REQUIREMENTS only demands windows).
- `_zero_shot_score` (D-04 dual-mode): `variants` inline list `{chrom,pos,ref,alt}` AND `vcf_path` server-side file → both route to `vep.evaluate_vcf(model, tokenizer, vcf_path, reference, paradigm=..., ...)` (vep.py:734, full signature verified). Inline variants must be materialized to a temp VCF (server-side tmp, never inside the client's path tree) or scored via `score_variant` per variant while still emitting the same skip-accounting shape. **Skip accounting MUST be visible in the tool response** (CONTEXT specifics: "skip accounting visible in the tool response (locked)") — surface `VepResult`'s skip/convention blocks verbatim in the JSON. Note `evaluate_vcf` applies the D-17 ClinVar convention by default (`clnsig_filter=None` → defaults); for non-ClinVar VCFs expose a parameter mapping to `clnsig_filter` override — planner discretion within D-04.

### Pattern 5: Reference-genome sourcing for `hotspots` / `zero_shot_score` (design gap — planner decides)
**What:** Both tools need reference sequence. `evaluate_vcf` requires `reference: str | Path | Mapping[str,str]` [VERIFIED: vep.py:734-800]; coordinates→sequence needs a FASTA [no FASTA field exists in the MCP config — VERIFIED: grep over `dnallm/mcp/configs/mcp_server_config.yaml` + `config_manager.py`, zero hits].
**Recommendation:** per-call `fasta_path` parameter on both tools (server-side file, same trust boundary as `vcf_path`), parsed via `vep._load_reference` (pure-Python FASTA reader in-package, handles plain/.gz and mappings — vep.py:451) — NOT `utils/genomic_coords.fetch_sequence` (pyfastx is a `dev`-extra optional dep; MCP servers run without dev extras). Alternative: add an optional MCP-config `reference` field (touches config_manager schema — larger diff, more validation surface; only worth it if the owner wants a server-wide default genome).

### Pattern 6: Host/port CLI-precedence fix (MCPE-01, both transports)
**Verified bug mechanics:**
- `start_server(host="127.0.0.1", port=8000, ...)` (server.py:1676-1680): unconditional `if server_config: host = server_config.server.host; port = server_config.server.port` (server.py:~1743-1747) — CLI values are ALWAYS discarded when a YAML server block exists (which is always, given the default `--config` path) [VERIFIED: read this session].
- `_start_http_server` (server.py:~1850-1857): `if server_config.server.host == host: host = streamable_http_config.host` — the comparison is always true because `start_server` just overwrote host with `server_config.server.host`, so the `streamable_http` block ALWAYS wins on this transport; the comment ("Use streamable_http host/port only if server config doesn't specify them (to avoid overriding CLI args...)") expresses an intent the sentinel-less API cannot implement [VERIFIED: read this session].
- `main()` argparse: `--host` default `"0.0.0.0"`, `--port` default `8000` (server.py:1996-2008) — "explicitly passed" is indistinguishable from "default" [VERIFIED: read this session].
**Fix shape (recommended):**
1. argparse: `default=None` for both flags (document the effective resolution chain in `--help`: CLI > config > default).
2. `start_server(host: str | None = None, port: int | None = None)`: when `None`, resolve from config (per transport: `streamable_http.host/port` if the block exists on the http path, else `server.host/port`), falling back to the current documented defaults (`127.0.0.1:8000`; note the argparse default today is `0.0.0.0` vs `start_server`'s `127.0.0.1` — pick one documented default and note the change).
3. Delete both override blocks; resolve once, pass down. stdio unaffected.
**Tests that must flip/change in the same change (they codify today's precedence):**
- `tests/mcp/test_server_transports.py::TestStreamableHTTPConstruction::test_config_fields_assembled_from_streamable_http_block` — asserts `kwargs["port"] == 8124` (YAML streamable_http block beats an explicitly-passed 8123). Under CLI-precedence the explicit 8123 wins → assertion flips; add the complementary yaml-only case.
- `tests/mcp/test_start_server.py::TestMain::test_defaults_are_forwarded` (lines 123-142) — asserts `host="0.0.0.0", port=8000` forwarded without flags; changes to `None` forwarding (or the resolved chain, if resolution moves into `main`).
- New precedence tests for BOTH transports (CLI-explicit > yaml > default), asserted via the patched-`uvicorn.Config`-kwargs construction pattern already in `TestStreamableHTTPConstruction` / `TestSSEConstruction`.
- `test_starts_server_with_parsed_args` (explicit flags → forwarded) continues to hold.

### Pattern 7: Handshake regression tests for the 3 new tools
**What:** Extend the proven in-memory ASGI pattern: `real_server` fixture (zero models, `dnallm.mcp.model_manager.load_model_and_tokenizer` patched) + `streamablehttp_client` over `httpx.ASGITransport` (localhost base_url required — mcp 1.30.0 DNS-rebinding protection rejects non-localhost Host with 421) + `ClientSession.initialize()` then `session.call_tool("_ism_scan", {...})` → `json.loads(result.content[0].text)` assertions [VERIFIED: tests/mcp/test_server_transports.py:155-247 read this session].
**When to use:** one call per new tool (error-dict paths first — mocked engine returning controlled results; happy paths via mocked `get_inference_engine`). **SSE caveat:** the in-memory SSE exchange deadlocks under ASGITransport (module docstring documents this, reproduced twice) — SSE stays construction-only; live SSE remains typed network skips.
**Structure test:** `EXPECTED_TOOLS` (test_server_transports.py:40-56) grows to 16 and `assert len(names) == 13` → `16` [VERIFIED: current set quoted in the test].

### Anti-Patterns to Avoid
- **Registering tools bare (`self.app.tool()(self._ism_scan)`)** — no timeout, no structured error; the registration block at server.py:263-292 is the pattern (PITFALLS #12a).
- **Blocking sync torch calls directly in an async tool body** — stalls the event loop (`health_check` stops responding); bridge via the ModelManager executor pattern (PITFALLS #12b). Add the event-loop liveness test (concurrent `health_check` during a long tool call).
- **Uniform 0.25 background** — inflates hits on GC-rich windows; also: trusting the embedded `Background letter frequencies` line from JASPAR MEME responses (it is a placeholder).
- **Per-sequence FDR** — D-03: BH over the FULL window×motif set, one `false_discovery_control` call over the concatenated p-vector.
- **Placing `interpret/` code under any coverage-omit glob** or forgetting the same-change coverage-row check (PITFALLS #2/#13 class).
- **New test files with duplicate basenames across the two collected roots** (`tests/` + `dnallm/mcp/tests/`, no `__init__.py`) — pytest "import file mismatch" (PITFALLS #8).
- **Editing `dnallm/__init__.py`** to re-export the new package — facade is byte-stable by REQUIREMENTS.
- **Touching `example/` or `tests/examples/`** — would break the census pin (verified current at 208/217) and trigger D-03 re-pin + docs-mirror work outside phase scope.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BH FDR correction | Manual rank/reject loop | `scipy.stats.false_discovery_control(ps, method="bh")` | Named by REQUIREMENTS; signature verified in venv; edge cases (ties, NaN policy) already handled upstream |
| FASTA reading (plain/.gz) | New parser in server.py | `vep._load_reference` (vep.py:451) | Already in-package, handles mapping/FASTA/gz; pyfastx is dev-extra-only |
| Variant↔token alignment / VCF convention | Re-derive in the tool | `vep.evaluate_vcf` / `vep.score_variant` | Phase-11 acceptance-tested kernel; the tool is a wrapper (D-04 locked) |
| Hotspot window extraction | New sliding-window scorer | `Mutagenesis.find_hotspots` (mutagenesis.py:517) | Paper-workflow parity (D-05); percentile/window params already validated in v1.1 |
| Reverse complement | New helper | `utils/sequence.reverse_complement` (sequence.py:37) | Existing; note lowercase-`n` mapping quirk worth a fixture if N appears in windows (PITFALLS #6 note) |
| Retry/backoff on network | ad-hoc loop | House `max_try=3` + sleep pattern (model.py:317 precedent) | CONTEXT discretion explicitly names the house pattern |

**Key insight:** every heavy component this phase wraps already exists and is acceptance-tested (VEP kernel, ISM engine, hotspot finder, timeout wrapper, executor bridge, ASGI test harness). The genuinely new code is (a) the FIMO-recipe calibration + scanner and (b) thin tool adapters — keep it that way.

## Common Pitfalls

### Pitfall 1: Reading D-02 as "no discretization at all" and hand-rolling a float DP
**What goes wrong:** A float-score DP over distinct achievable sums explodes combinatorially (each column multiplies distinct sums by ≤4; no practical collision structure at w≈15+); or a "clever" numpy rounding produces p-values that don't match FIMO's.
**Why it happens:** D-02's wording ("no numpy score-discretization approximation") reads like a discretization ban; but FIMO itself discretizes (`PSSM_RANGE=100`).
**How to avoid:** Adopt FIMO's integer scaling deliberately as the documented convention (Pattern 1, step 2) with a provenance comment citing `pssm.h:13`; document in the module docstring that the quantization is FIMO's own (reported scores approximate raw bits — maintainer-confirmed).
**Warning signs:** a DP keyed by float tuples/dicts; p-values that disagree with a FIMO run on the same motif+background by more than the quantization grain.

### Pitfall 2: The streamable-http "fix" desyncs from the sse path (PITFALLS #12d)
**What goes wrong:** Patching only the `start_server` unconditional override leaves the `_start_http_server` conditional override in place (or vice versa); the two transports end up with different precedence.
**Why it happens:** The bug lives in two places with different shapes (unconditional vs. always-true-conditional).
**How to avoid:** Resolve host/port ONCE (sentinel-based) before dispatch; both starters receive final values; precedence tests for BOTH transports via the patched-uvicorn construction pattern; `stdio` unaffected (assert no regression).
**Warning signs:** precedence tests present for one transport only; the `streamable_http` block still mutating host/port inside `_start_http_server`.

### Pitfall 3: EXPECTED_TOOLS / same-basename test drift (PITFALLS #8)
**What goes wrong:** Adding tools without extending `EXPECTED_TOOLS` (and the `len == 13` assert) leaves the registration structure test green while the new tools are silently unregistered — or a new `tests/mcp/test_server_tools.py` collides with a file of the same name under `dnallm/mcp/tests/`.
**How to avoid:** Same-change `EXPECTED_TOOLS` update (13 → 16); unique test basenames (`test_server_tools_v12.py` or extend existing files).
**Warning signs:** `list_tools` assertions still counting 13; pytest "import file mismatch" collection errors.

### Pitfall 4: Timeout wrapper vs. legitimately long ISM (PITFALLS #12a / performance table)
**What goes wrong:** `ism_scan` on a 2kb sequence (>6000 forward passes) hits `_tool_timeout_seconds` and returns timeout error dicts on valid input; users retry-loop.
**How to avoid:** Cap inputs at the tool boundary (max sequence length, max positions) with matchable error dicts stating the caps; the wrapper's timeout dict already carries a `suggestion`; document the caps in the tool docstring (LLM-facing surface). Do NOT switch to streaming tools (D-06 locks the wrapper surface).
**Warning signs:** handshake tests only exercising tiny inputs; timeout error rate on the nightly MCP probe.

### Pitfall 5: Golden fixture built on the wrong motif/locus before owner input
**What goes wrong:** Two BCL11A CORE matrices exist in JASPAR (`MA2324.1` w=7, `MA2504.1` — live-verified), and the paper's Fig 4a coordinates are not publicly retrievable; guessing either bakes a wrong golden test.
**Why it happens:** The paper is under revision (not indexed); the fixture's authority is the manuscript figure, not the literature.
**How to avoid:** Block the golden test on owner input (exact Fig 4a window coordinates + motif ID + JASPAR release); build everything else (DP, scan, parser, client) against synthetic fixtures meanwhile. Commit the owner-supplied FASTA windows + frozen MEME files so the golden test is network-free (v1.1 committed-loci precedent).
**Warning signs:** a golden test citing literature coordinates with no owner-confirmation record; fixture depending on a live JASPAR fetch.

### Pitfall 6: Census re-pin triggered accidentally / coverage-expectation docs left stale
**What goes wrong:** Any `example/` or `tests/examples/` addition in this phase shifts the `208/217 (9 deselected)` collect count and reds the Stage 0.5 hard gate (ci.yml:865-871); conversely C3 forgetting the honest coverage-docs update leaves "96.30%" stale after new modules shift the number.
**How to avoid:** Keep the phase out of `example/` entirely (nothing in scope requires it — verified: current collect line matches the pin exactly, measured this session). C3 measures the new total and updates `docs/user_guide/continuous_integration.md` (lines 26/30) + the `pyproject.toml:26` comment honestly, per D-07.
**Warning signs:** CI example-nightly Stage 0.5 red; docs claiming a coverage number the last coverage run contradicts.

### Pitfall 7: New network skips without same-change allowlist / unpinned JASPAR release
**What goes wrong:** JASPAR live tests skip on unreachable networks with untyped messages → `audit_skips.py` fails all CI legs; or searches without `release=` float against the live DB (2026 is now latest — matrix sets changed between releases), making the golden fixture non-reproducible.
**How to avoid:** Typed skip reasons (e.g. `jaspar-unreachable:` prefix) + `expected_skips.yaml` entries in the same change; pin `release=` on every fixture-bearing query; default the client's base URL to a module constant with provenance.
**Warning signs:** audit-skips failures mentioning the new tests; fixture assertions that passed locally failing on CI a week later.

### Pitfall 8: IA³ docs chapter contradicts the shipped behavior
**What goes wrong:** The chapter is written from the Phase-10 stub state ("setting `use_ia3: true` does not yet switch the trainer") instead of the Phase-11 reality (real branch, `use_ia3 × use_qlora` rejected at Pydantic time, IA³ save/reload roundtrip).
**How to avoid:** Write from the CHANGELOG Unreleased entry + trainer source truth; fenced code blocks must survive the docs-validation ruff-format gate (0.7.1 precedent: over-long fenced Python blocks tripped CI).
**Warning signs:** docs examples not matching `TrainingConfig` field validation; docs-validation workflow red.

## Code Examples

### FIMO-recipe calibration (pure Python; every constant cited)
```python
# Source: MEME Suite 4.8.1 source, read this session:
#   src/pssm.h:13       #define PSSM_RANGE 100
#   src/logodds.c:566+  scale_lo() — scale = range/(large-small), offset = small,
#                       scaled = round((score - offset) * scale); inversion
#                       "score_bit = (score/scale) + (w*offset)" (logodds.c:50)
#   src/pssm.c get_pdf_table — pdf[k+s] += pdf[k] * bg[j]; get_pv_lookup —
#                       "pssm->pv[x] = Pr(score >= x)" (pssm.c:531)
# Docs: meme-suite.org/meme/doc/fimo.html — pseudocount 0.1 scaled by background,
#                       zero-order background, both strands, p<1e-4 default, BH q-values.

PSSM_RANGE = 100          # FIMO's internal integer-score granularity (pssm.h:13)
FIMO_PSEUDOCOUNT = 0.1    # --motif-pseudo default, scaled by background freq
FIMO_P_THRESHOLD = 1e-4   # --thresh default

def log_odds_matrix(freq_rows: list[list[float]], nsites: int, bg: dict[str, float]):
    """PFM rows (A,C,G,T per position) -> log-odds bits with 0.1*bg pseudocount."""
    # JASPAR MEME headers carry nsites (live-verified: "nsites= 2000");
    # reconstruct counts, apply FIMO's background-weighted pseudocount.
    n = float(nsites) if nsites > 0 else 1.0
    letters = "ACGT"
    scores = []
    for row in freq_rows:
        col = []
        for b, f in zip(letters, row):
            p = (f * n + FIMO_PSEUDOCOUNT * bg[b]) / (n + FIMO_PSEUDOCOUNT) if nsites > 0 \
                else (f + FIMO_PSEUDOCOUNT * bg[b]) / (1.0 + FIMO_PSEUDOCOUNT)
            col.append(math.log2(p / bg[b]))
        scores.append(col)
    return scores

def pvalue_table(scores: list[list[float]], bg: dict[str, float]) -> tuple[list[float], float, float, int]:
    """FIMO exact-DP: scale to [0..PSSM_RANGE], convolve per column, reverse-cumsum.

    Returns (pv, scale, offset, max_scaled) where pv[x] = Pr(scaled score >= x)
    for 0 <= x <= w*range; threshold is the minimal x with pv[x] < 1e-4.
    """
    flat = [s for col in scores for s in col]
    small, large = min(flat), max(flat)
    if large == small:                      # no information — logodds.c skips
        raise ValueError("motif has no score variation (uniform PWM)")
    scale = PSSM_RANGE / (large - small)
    offset = small
    scaled = [[round((s - offset) * scale) for s in col] for col in scores]
    letters = "ACGT"
    pdf = [1.0] + [0.0] * (len(scores) * PSSM_RANGE)
    for col, scol in zip(scores, scaled):   # pure-Python column scan (D-02)
        nxt = [0.0] * len(pdf)
        for b, s in enumerate(scol):
            p = bg[letters[b]]
            for k, mass in enumerate(pdf):
                if mass:
                    nxt[k + s] += mass * p
        pdf = nxt
    total = sum(pdf)                        # ~1.0 (rounding-tolerant assert)
    pv = pdf                                # pv[x] = Pr(score >= x) after cumsum
    for x in range(len(pv) - 2, -1, -1):
        pv[x] += pv[x + 1]
    return pv, scale, offset, len(scores) * PSSM_RANGE

def threshold_bits(pv, scale, offset, w) -> float:
    """Minimal bits score with p < 1e-4; inversion per logodds.c:50."""
    for x in range(len(pv) - 1, -1, -1):    # find the LOWEST x still under threshold
        if pv[x] < FIMO_P_THRESHOLD:
            continue
        x_th = x + 1                        # first score strictly under threshold
        return x_th / scale + w * offset
    return len(pv) / scale + w * offset     # everything significant
```
(Hand-computable unit-test anchors: a uniform PWM must raise the no-variation `ValueError`; a 1-column motif `p=[1,0,0,0]` against `bg=0.25` uniform gives one achievable scaled score with p exactly 1.0; a 2-column motif's pv must equal the analytic convolution. BH: `false_discovery_control(np.array(ps), method="bh")` on constructed p-sets with known q ordering.)

### JASPAR client (stdlib, live-verified response shape)
```python
# Source: live probes 2026-10-10 (HTTP 200):
#   GET https://jaspar.elixir.no/api/v1/matrix/MA2324.1/?format=meme
#   GET https://jaspar.elixir.no/api/v1/matrix/?name=BCL11A&collection=CORE&version=latest
# Response (verbatim head):
#   MEME version 4
#   ALPHABET= ACGT
#   strands: + -
#   Background letter frequencies
#   A 0.25 C 0.25 G 0.25 T 0.25
#   MOTIF MA2324.1 BCL11A
#   letter-probability matrix: alength= 4 w= 7 nsites= 6265 E= 0
#    0.054749  0.801915  0.094334  0.049002
#   ...
#   URL https://jaspar.elixir.no/matrix/MA2324.1

JASPAR_BASE = "https://jaspar.elixir.no/api/v1"   # parameterized per REQUIREMENTS

def fetch_meme_motif(matrix_id: str, *, base_url: str = JASPAR_BASE,
                     timeout: float = 30.0, max_try: int = 3) -> str:
    """Fetch one matrix in MEME format (retry-with-backoff house pattern)."""
    url = f"{base_url}/matrix/{matrix_id}/?format=meme"
    for attempt in range(1, max_try + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                body = resp.read(1_000_000)          # size cap: network input
                if resp.status != 200:
                    raise ValueError(f"JASPAR returned HTTP {resp.status}")
                return body.decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == max_try:
                raise ValueError(f"JASPAR fetch failed for {matrix_id}: {e}") from e
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")
# Parser: MOTIF line -> (id, name); "letter-probability matrix:" -> alength/w/nsites/E
# via regex; rows -> floats. IGNORE the embedded uniform background (JASPAR
# placeholder) — the scan background is GC-matched from target windows.
```

### Tool registration + timeout wrapper extension (server.py pattern, verified)
```python
# Source: dnallm/mcp/server.py:263-292 (read this session)
# Register interpret-adjacent tools (wrapped with timeout)
self.app.tool()(self._with_timeout_wrapper(self._ism_scan, "ism_scan"))
self.app.tool()(self._with_timeout_wrapper(self._hotspots, "hotspots"))
self.app.tool()(self._with_timeout_wrapper(self._zero_shot_score, "zero_shot_score"))
# Wire names derive from the method __name__ via functools.update_wrapper
# (leading underscore preserved) — EXPECTED_TOOLS in tests/mcp/test_server_transports.py:40
# must grow to 16 in the same change.
```

### Host/port sentinel resolution (fix shape)
```python
# main(): argparse
parser.add_argument("--host", type=str, default=None, ...)   # was "0.0.0.0"
parser.add_argument("--port", type=int, default=None, ...)    # was 8000

# start_server(): resolve once, both starters receive final values
def start_server(self, host: str | None = None, port: int | None = None,
                 transport: str = "stdio") -> None:
    server_config = self.config_manager.get_server_config()
    if host is None:
        host = (server_config.server.host if server_config else None) or "127.0.0.1"
    if port is None:
        port = (server_config.server.port if server_config else None) or 8000
    # streamable-http: the dedicated block refines the *config-sourced* value only
    if transport == "streamable-http" and host_was_config_sourced:
        sh = getattr(server_config, "streamable_http", None)
        if sh:
            host, port = sh.host, sh.port
    ...
# Precedence: CLI-explicit > transport-specific YAML > server YAML > default.
# Tests: patch uvicorn.Config, assert kwargs["host"]/["port"] for the
# CLI-explicit / yaml-only / no-config-none cases on BOTH transports.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| JASPAR at jaspar.genereg.net | Canonical host **jaspar.elixir.no** (ELIXIR-hosted) | 2024+ | REQUIREMENTS already mandates the elixir host; base URL parameterized |
| FIMO default p-value selection only | FIMO `--qv-thresh` BH q-value selection; we compute BH directly over the full set via scipy (no `--max-stored-scores` 100k truncation — our BH is exact on every hit) | MEME 4.x era → scipy 1.11+ | D-03's full-set BH is stronger than FIMO's approximate-q path when >100k stored scores |
| FEATURES.md's recommended empirical-null calibration | D-01 exact-DP (owner override) | 2026-10-09 (discuss) | Cross-motif comparable p-values; no null-window sampling cost; honest-documentation requirement instead |
| MCP SSE transport | streamable-http recommended (MCP 2025-11-25 spec, per server.py `--transport` help) | 2024-2025 | Both transports must be precedence-tested (the bug exists on both) |

**Deprecated/outdated:**
- `--score-scaling`/float-score "exact DP" variants: not part of FIMO; the shipped convention is integer scaling (verified from source).
- CIS-BP REST expectations: no API exists; local ZIP/PWM parsing is the only programmatic route [CITED: re3data].

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `PSSM_RANGE = 100` persists in current MEME 5.x (verified in 4.8.1 source only; 5.x docs don't contradict) | Pattern 1 / Code Examples | Low: quantization grain differs slightly from current FIMO binaries; provenance comment names the version; cross-version FIMO outputs already differ by design (maintainer thread) |
| A2 | HBG1 promoter region is chr11 ~5.27 Mb (GRCh38), BCL11A site ~−115 from TSS | Pitfall 5 / fixture | Medium: wrong window → golden test can't reproduce Fig 4a regardless of scanner correctness; mitigated by blocking on owner input (Open Question 1) |
| A3 | CIS-BP acceptance reading = JASPAR-live + CIS-BP-local-parse (no CIS-BP network client) | Pattern 3 | Low: if the owner expects a CIS-BP fetcher, it is bulk-ZIP-only (no REST); scope stays parse-side |
| A4 | E-value column = p × number of scanned positions (both strands × motifs, FIMO-paper convention: expected false positives at that p) | Output table | Low: it is a reporting convenience; q-values are the significance currency (D-03); formula documented in docstring either way |
| A5 | JASPAR release to pin = the one the paper used (2024 vs 2026 both active; live DB now defaults to 2026) | Pattern 3 / Pitfall 7 | Medium: matrix sets differ between releases (MA IDs/version suffixes shift); fixture must record the release; owner confirms which the paper used |
| A6 | `hotspots`/`zero_shot_score` reference source = per-call `fasta_path` (no MCP-config schema change) | Pattern 5 | Low: planner may choose a config field instead; both workable, per-call is the smaller-diff recommendation |
| A7 | BCL11A motif for the fixture is one of MA2324.1 / MA2504.1 (both JASPAR CORE, live-verified) | Pitfall 5 | Medium: paper may use a CIS-BP BCL11A PWM or another JASPAR version; owner pins |
| A8 | `evaluate_vcf`'s D-17 ClinVar convention applies by default in the MCP tool; non-ClinVar callers pass an override | Pattern 4 | Low: parameter surface detail within C2 discretion; documented in the tool docstring |

## Open Questions

1. **The paper's exact Fig 4a coordinates + motif identity (BLOCKING for the golden test only)**
   - What we know: acceptance = "HBG1/BCL11A motif hit coordinates match the paper's Fig 4a annotation"; literature anchors are BCL11A +58 enhancer core GRCh38 chr2:60,495,219-60,495,336 (GATA1/GATAA motif core, Canver 2015 PMC4644101) and the HBG1/HBG2 promoter BCL11A site ~−115 from TSS; JASPAR has MA2324.1 (w=7) and MA2504.1 for BCL11A.
   - What's unclear: the paper's window definitions (locus, flank size, assembly), which motif ID, and which JASPAR release. The DNALLM paper is not publicly indexed (searched 2026-10-10; only docs/DeepWiki surface).
   - Recommendation: ask the owner for the Fig 4a window coordinates + motif ID + release BEFORE writing the golden test; build the DP/scan/parser/client against synthetic fixtures meanwhile. Commit owner windows as FASTA fixtures (network-free golden test).
2. **Coordinate-matching tolerance for the golden test**
   - What we know: CONTEXT gives Claude discretion on "coordinate rounding tolerance"; figure-annotated coordinates are typically read to the nearest shown tick.
   - Recommendation: exact match when the window is owner-supplied verbatim; ±2 bp tolerance documented in the fixture if coordinates are figure-derived.
3. **`zero_shot_score` inline-variant mode plumbing**
   - What we know: D-04 routes BOTH modes "through the same `evaluate_vcf` kernel"; inline variants are `{chrom,pos,ref,alt}` dicts; `evaluate_vcf` reads a VCF path.
   - What's unclear: whether "same kernel" means materializing a temp VCF (exact reuse, incl. convention filtering) vs. looping `score_variant` (no ClinVar convention by construction).
   - Recommendation: temp-VCF materialization (server-side tmpfile, sanitized fixed name — never from record fields) — one code path, skip accounting identical; planner confirms.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| jaspar.elixir.no REST | MOTIF-01 client + slow-lane live tests | ✓ (live-probed HTTP 200, 2026-10-10) | API v1, latest release 2026 (11) | typed network skips + committed fixtures for the fast lane |
| scipy `false_discovery_control` | MOTIF-01 BH | ✓ (venv-verified) | scipy 1.18.1 (`>=1.15.2` floor) | — |
| numpy | MOTIF-01 scan loop | ✓ | per pyproject | pure-Python scan (windows are small) |
| mcp SDK (FastMCP) | MCPE-01 tools + tests | ✓ | pinned `>=1.3.0,<2` (server code read against it) | — |
| pytest / pytest-asyncio / pytest-timeout | all lanes | ✓ | configured in pyproject `[tool.pytest.ini_options]` | — |
| Network + GPU runner (nightly) | slow lane (tiny real models for `ism_scan`/`hotspots` tools) | ✓ (CI matrix) | — | mocked-engine fast lane covers contracts |
| Reference FASTA files | `hotspots`/`zero_shot_score` runtime + fixtures | fixture files ✓ (to be committed); runtime = caller-supplied | — | tools return matchable error dicts when absent |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none (JASPAR network absence falls back to typed skips by design).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (>=8.3.5) + pytest-asyncio (auto mode) + pytest-timeout (300s) + pytest-cov |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (testpaths `tests` + `dnallm/mcp/tests`; `--strict-markers`) |
| Quick run command | `.venv/bin/python -m pytest tests/interpret tests/mcp -m "not slow" --no-cov` |
| Full suite command | `.venv/bin/python -m pytest --cov=dnallm --cov-report=term-missing` (CI-enforced `fail_under=90`; working standard ≥96% per module) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MOTIF-01 | DP calibration: scaling, convolution pv table, threshold inversion (hand-computed anchors) | unit | `pytest tests/interpret/test_motifs.py -k "pvalue or threshold" --no-cov` | ❌ Wave 0 |
| MOTIF-01 | GC-matched vs uniform background changes hit set on GC-skewed synthetic windows | unit | `pytest tests/interpret/test_motifs.py -k background --no-cov` | ❌ Wave 0 |
| MOTIF-01 | Strand mechanics: palindromic (same coords both strands) + non-palindromic (strand-specific) | unit | `pytest tests/interpret/test_motifs.py -k strand --no-cov` | ❌ Wave 0 |
| MOTIF-01 | MEME-format parsing (live-probed shape) + CIS-BP local table parsing; embedded-background ignored | unit | `pytest tests/interpret/test_motifs.py -k "parse or meme" --no-cov` | ❌ Wave 0 |
| MOTIF-01 | BH over full window×motif set via `false_discovery_control` (constructed p-sets, known q ordering) | unit | `pytest tests/interpret/test_motifs.py -k "bh or fdr" --no-cov` | ❌ Wave 0 |
| MOTIF-01 | JASPAR client retry/backoff + error surface (mocked urlopen) | unit | `pytest tests/interpret/test_motifs.py -k client --no-cov` | ❌ Wave 0 |
| MOTIF-01 | HBG1/BCL11A golden: hit coordinates match Fig 4a (committed fixtures, network-free) | regression | `pytest tests/interpret/test_motifs.py -k golden --no-cov` | ❌ Wave 0 (owner-blocked input) |
| MOTIF-01 | JASPAR live fetch round trip | slow/network | `pytest tests/interpret/test_motifs.py -k "live and jaspar" -m slow` (typed `jaspar-unreachable:` skips) | ❌ Wave 0 |
| MCPE-01 | Handshake: server up → client calls all 3 tools → JSON assertions (ASGI in-memory) | integration | `pytest tests/mcp/test_server_transports.py -k "round_trip or tool" --no-cov` | ✅ (extend) |
| MCPE-01 | EXPECTED_TOOLS 16 + registration timeout-wrapper structure | unit | `pytest tests/mcp/test_server_transports.py -k list_tools --no-cov` | ✅ (extend) |
| MCPE-01 | Host/port precedence CLI > yaml > default on BOTH sse + streamable-http (patched uvicorn) | unit | `pytest tests/mcp/test_server_transports.py -k "precedence or construction" --no-cov` | ✅ (extend + flip 2) |
| MCPE-01 | Tool JSON contracts + skip accounting visible in `zero_shot_score` response (mocked engines) | unit | `pytest tests/mcp/test_server_tools_v12.py --no-cov` | ❌ Wave 0 |
| MCPE-01 | Event-loop liveness under a long tool call (concurrent `health_check`) | integration | `pytest tests/mcp/test_server_tools_v12.py -k liveness --no-cov` | ❌ Wave 0 |
| MCPE-01 | Tiny real model through `ism_scan`/`hotspots` (nightly) | slow | `pytest tests/mcp/ -m slow -k "tools_v12"` (models.lock row if a new model id is referenced) | ❌ Wave 0 |
| DOCS-01 | IA³ chapter present + accurate; docs build green | gate | `.github/workflows/docs-validation.yml` (ruff-format over fenced blocks) | ✅ (gate exists) |

### Sampling Rate
- **Per task commit:** quick run command above (targeted lane files, `--no-cov`) — owner execution directive.
- **Per wave merge:** `pytest tests/ tests-dnallm-mcp` full fast lane (`-m "not slow"`) + per-module coverage rows for `dnallm/interpret/motifs.py` and `dnallm/mcp/server.py` (≥96% standard).
- **Phase gate:** full suite green + coverage ≥90 gate (working standard verified per-module) + census collect line still `208/217 (9 deselected)` before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/interpret/__init__.py` + `tests/interpret/test_motifs.py` — covers MOTIF-01 unit lanes
- [ ] `tests/interpret/fixtures/` — HBG1/BCL11A FASTA windows + frozen MEME motif file(s) (owner input)
- [ ] `tests/mcp/test_server_tools_v12.py` — per-tool JSON contracts, liveness test
- [ ] `tests/mcp/test_server_transports.py` — EXPECTED_TOOLS 16, new-tool round trips, precedence tests (flip 2 existing)
- [ ] `tests/expected_skips.yaml` — typed `jaspar-unreachable:` entries, same change as the live tests
- [ ] `models.lock` row — only if the slow tool lane references a model id not already locked

## Security Domain

> `security_enforcement` not explicitly disabled in config → included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Server is unauthenticated by design (operator-run; existing surface unchanged) |
| V3 Session Management | no | MCP session handling owned by the SDK (StreamableHTTPSessionManager), unchanged |
| V4 Access Control | yes | `model_name` restricted to the server-config registry (existing ModelManager validation — arbitrary model names must never reach a downloader); `vcf_path`/`fasta_path`/output paths are operator-trust-boundary server-side reads |
| V5 Input Validation | yes | Strict parsers: MEME regex-validated (alength=4, w cap, numeric rows in [0,1], nsites ≥ 0); CIS-BP table validated; VCF via scikit-allel + `vep` coordinate/REF validation; inline `variants` validated per-field ({chrom,pos,ref,alt} shapes, alt ∈ ACGT, pos int > 0) with matchable `ValueError` error dicts |
| V6 Cryptography | no | No secrets, no crypto |
| V12 File Handling | yes | **Untrusted VCF input reaches the server via `vcf_path`** (threat analysis below); no path is EVER constructed from VCF record fields (vep.py guarantees: "no path is ever derived from VCF record fields" — verified in its docstring); output writes confined to configured `output_dir`; temp VCF for inline variants uses a fixed sanitized name |

### Known Threat Patterns for {MCP server + network motif client}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/corrupted VCF or FASTA (server-side `vcf_path`/`fasta_path` from an LLM client) crashes parsers or pivots paths | Tampering/DoS | Treat downloaded/client-named files as untrusted data: strict parse-or-reject (matchable `ValueError` → error dict), size caps, suffix allowlist (`.vcf`/`.vcf.gz`/`.fasta`/`.fa`/`.fa.gz`/`.fna`), resolve+contain paths server-side; never exec/eval anything from file content |
| Path traversal via record fields (e.g. a CHROM value used in filenames) | Elevation | Already banned by vep.py contract; new tool code inherits — assert in a test that no output path contains record-derived strings |
| JASPAR response tampering / slop content (network input parsed as motifs) | Tampering | Parse strictly (regex-anchored MEME grammar); reject unknown sections; cap response bytes; never `eval`; base-URL allowlist (scheme https, host from config/module constant — NOT from caller input); no redirect-following to arbitrary hosts |
| Uncontrolled egress via parameterized URLs (SSRF-adjacent) | Information disclosure | Base URL is config/module-owned; matrix IDs validated against `^MA\d{4}\.\d+$` + collection query params from a closed set; per-call args never form the host |
| Arbitrary model names in tool args triggering downloads | Repudiation/DoS | Existing ModelManager registry restriction (validate against configured model list before any load — PITFALLS security table); preserve in all three tools |
| Result-size explosion (huge ISM tables into an agent context) | DoS | Cap inputs (sequence length, positions, variant count) at tool boundary; error dicts state the caps (anti-feature: unbounded scans via MCP — FEATURES REV-11) |

## Sources

### Primary (HIGH confidence)
- MEME Suite 4.8.1 source, fetched and read this session: `src/pssm.h` (`#define PSSM_RANGE 100`, line 13), `src/pssm.c` (`get_pdf_table` DP loop lines ~485-516; `get_pv_lookup` `pssm->pv[x] = Pr(score >= x)` lines 531/570; `build_motif_pssm`), `src/logodds.c` (`scale_lo` lines 566-630; inversion comment line 50), `src/fimo.c` (PSSM_RANGE passed at lines 1081/1113; `--pval-lookup` table print) — via http://tdb.ccmb.res.in/meme/meme_4.8.1/src/
- FIMO official docs — https://meme-suite.org/meme/doc/fimo.html (p<1e-4 default, pseudocount 0.1 background-scaled, zero-order background, both strands, BH q-values, `--max-stored-scores` approximation note)
- JASPAR REST API — live probes against https://jaspar.elixir.no/api/v1/ (matrix/MA2324.1/?format=meme; matrix/?name=BCL11A; releases/) — HTTP 200 responses quoted in Code Examples; endpoint docs at /api/v1/docs/
- In-repo source (read this session): `dnallm/mcp/server.py` (registration 251-354, `start_server` 1676-1747, `_start_sse_server`/`_start_http_server` 1749-1899, argparse 1928-2064, `_dna_mutagenesis` 1224+), `dnallm/mcp/model_manager.py` (executor bridge 99-121, `get_inference_engine` 224), `dnallm/inference/vep.py` (`evaluate_vcf` 734-800, `_load_reference` 451), `dnallm/inference/mutagenesis.py` (class 31, `find_hotspots` 517, `prepare_tfmodisco_inputs` 583+), `dnallm/utils/sequence.py` (`reverse_complement` 37, `check_sequence` 89), `dnallm/utils/genomic_coords.py` (`fetch_sequence` 130, pyfastx), `dnallm/mcp/config_manager.py` (`get_streamable_http_config` 170-190), `tests/mcp/test_server_transports.py` (EXPECTED_TOOLS 40-56, ASGI round trip 155-247, construction tests 276-400), `tests/mcp/test_start_server.py` (TestMain 66-145), `.github/workflows/ci.yml` (census pin 857-871), `CHANGELOG.md` (Unreleased block), `docs/user_guide/fine_tuning/peft_adapters.md` (IA³ section 174-186), `docs/user_guide/continuous_integration.md` (96.30% lines 26/30), `pyproject.toml` (pytest/coverage config)
- venv probes: scipy 1.18.1 + `false_discovery_control` signature; census collect `208/217 tests collected (9 deselected)` (2026-10-10)

### Secondary (MEDIUM confidence)
- MEME Suite maintainer thread (cegrant) — https://groups.google.com/g/meme-suite/c/BXv4pRCacN8 — integer scaling of internal scores, reported-score quantization
- Canver et al. 2015, Nature 527:192-197 (PMC4644101) + patent US20200384032A1 — BCL11A +58 enhancer GRCh38 chr2:60,495,219-60,495,336 core, GATA1/GATAA motif
- MOODS (Korhonen et al. 2009, PMC2778336) + universalmotif "dynamic" p-value method — corroborating the Staden-1989 DP convention and adjustable precision
- CIS-BP access reality — re3data.org/repository/r3d100013971 + TFutils/universalmotif manuals (bulk.php ZIPs, no REST API)
- Project research artifacts: `.planning/research/PITFALLS.md` (#11, #12, #13, #15), `.planning/research/FEATURES.md` (REV-10/REV-11), `.planning/research/261009-paper-revision-suite-plan.md` (REV-10/11 acceptance), Phase 10 CONTEXT D-09 + summaries (CHANGELOG mechanism)

### Tertiary (LOW confidence)
- None used without upgrade path (every claim above carries a tag inline where it appears)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — nothing new installed; scipy verified in-venv; every wrapped surface read from source
- Architecture (motifs module + tools): HIGH — FIMO recipe pinned from MEME source line-by-line; MCP patterns extend read-in-session code; one design gap (reference source) documented with a recommendation
- Pitfalls: HIGH for repo-grounded items (all line-cited); MEDIUM for the fixture-coordinate items (owner-blocked by nature)

**Research date:** 2026-10-10
**Valid until:** 2026-11-09 (stable domain; JASPAR live-DB claims are the fastest-moving — re-probe if fixture work slips past a JASPAR release)
