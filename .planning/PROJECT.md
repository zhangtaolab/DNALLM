# DNALLM — Test Suite Audit & Coverage Hardening

## What This Is

DNALLM (`dnallm` v0.5.2) is a Python toolkit for fine-tuning, inference, and benchmarking of DNA language models (150+ pretrained models from HF/ModelScope), plus an MCP server exposing them to LLM agents. Milestone v1 (shipped 2026-10-01) was a quality-engineering cycle on that existing codebase: the pytest suite was audited end to end, test gaps closed, and line coverage driven from 45.92% to 96.30% behind a CI-enforced >90% hard gate.

## Core Value

A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## Current Milestone: v1.1 Example Execution Testing & Repair

**Goal:** Establish real-model execution testing for everything under `example/` and fix every error it surfaces; add PlantHelixSeek-CRE/-Anno Arabidopsis inference example notebooks; repair the CI gate false-green (WR-08/09) and bring example tests under formal gating.

**Target features:**
- Real execution of all existing examples — 20 Jupyter notebooks (nbclient), 3 marimo apps (headless), `generate_bpe_dataset.py`, all YAML configs through real `load_config()` validation — using real models on the nightly self-hosted GPU runner
- Error repair across example/ code, docs/example/ mirror, and any dnallm library bugs the executions expose
- CI: remove `continue-on-error` false-green in docs-validation, add missing `mcp` extra, fix README (WR-08/WR-09); example execution tests marked `slow` join the nightly census gate
- New PlantHelixSeek examples: registry entries for `PlantHelixSeek-CRE` (binary) / `PlantHelixSeek-Anno` (token, 17 BILOU); two notebooks using dnallm API with in-notebook sliding-window scanning and BigWig/GFF3 post-processing (mirroring upstream `scripts/cis_regulatory` / `scripts/gene_annotation` pipelines)
- Arabidopsis showcase data committed in-repo at ≤200kb per region: **guarantee that predictions are substantially consistent with experimental truth on the selected loci** (CRE ↔ PlantDHS TAIR10_DHSs.gff; Anno ↔ TAIR10 GFF3 gene annotation) — agreement verified during loci selection and asserted by the example tests; intermediate full-genome downloads from arabidopsis.org stay gitignored

## Requirements

### Validated

Inferred from the existing codebase (see `.planning/codebase/`):

- ✓ YAML → Pydantic config loading (`dnallm/configuration/configs.py`)
- ✓ Registry-dispatch model loading for 35 model families (`dnallm/models/model.py`, `modeling_auto.py`, `special/`)
- ✓ Dataset handling: local files, HF/ModelScope, tokenization, augmentation, splitting (`dnallm/datahandling/`)
- ✓ Fine-tuning via HF Trainer + LoRA/QLoRA + Optuna (`dnallm/finetune/trainer.py`)
- ✓ Inference engine, interpretability, mutagenesis, benchmarking (`dnallm/inference/`)
- ✓ MCP server with 11 tools over stdio/SSE/streamable-HTTP (`dnallm/mcp/`)
- ✓ Existing pytest suite: 464 tests across `tests/` and `dnallm/mcp/tests/`
- ✓ Published to PyPI; CI matrix Python 3.11–3.13

Shipped in Phase 1 (Harness Integrity & Measured Baseline, 2026-09-30):

- ✓ Full-suite audit report with pass/fail/skip census (625/0/0/9, both roots, `slow` included) — Phase 1 (`01-AUDIT-REPORT.md`)
- ✓ Measured line coverage on the agreed denominator (whole `dnallm/` minus vendored dirs, unimportable adapters, packaged test files; 7-entry omit list in `pyproject.toml`) — Phase 1
- ✓ Per-module coverage gap report (`term-missing` + `coverage.json`, 43-row ranked worklist) — Phase 1
- ✓ Honest harness: single pytest config (`pyproject.toml` only), real exit codes (mask removed, permanent CI canary), measured baseline **45.92%** — Phase 1

Shipped in Phase 2 (Suite Hygiene & Known-Bug Fixes, 2026-09-30):

- ✓ Multiclass AUROC fixed and unskipped — presence-guard ValueError + `labels=expected_classes`, both crash-skips removed (38/0 census) — Phase 2
- ✓ CrossDNA handler result survives dispatch — guarded first-resolved-wins chain, sentinel regression test, 12-handler audit (1 bug) — Phase 2
- ✓ Every skip typed and allowlisted — `network-unavailable:`-prefixed typed network skips, `expected_skips.yaml` + `scripts/audit_skips.py` CI gate, full run 623/7/0 with audit exit 0 — Phase 2
- ✓ PDF tests leave the tree clean — autouse tmp_path rebind, gitignore fixed, 9 strays deleted — Phase 2

### Active

Milestone v1.1 (see Current Milestone section above; formal REQ-IDs in REQUIREMENTS.md):

- Real-model execution tests for every artifact under `example/` (notebooks, marimo apps, helper script, YAML configs)
- All errors found by real execution fixed — example code, docs/example/ mirror, and exposed dnallm library bugs
- CI example gate repaired and enforced (WR-08/WR-09 closed; `slow`-marked execution tests in nightly census)

Shipped in Phase 9 (CI Wiring & Census Verification, 2026-10-06):

- ✓ Nightly census formally gates the execution-test layer, proven end-to-end on the real runner: final run 37432001711 all three legs green (example-nightly census 192P/1S/0F in 1:56:53 with stage-3 mcp pair at 87s; test-mamba 1840P/0F/1S; coverage-nightly 1938P/0F/15S at **96.42%**) — CI-03/CI-06
- ✓ giants-class evo exit as a true deselect (`giants` marker + `-m "not giants"`, never a typed skip) with the D-03 census triple hard-asserted (197/206 collected, 9 deselected; re-pinned 2026-10-07 after post-close repair growth); evo CI steps + models-cache layer deleted; all three nightly crons schedule-aware — CI-03
- ✓ `models.lock` consistency guard live on the fast leg (drift-injection proven; 12 contract tests) — CI-08; coverage-expectation docs page registered in mkdocs nav (AUDIT-04: example execution runs in kernel subprocesses and by design does not move the 96.30%→96.42% gate) — CI-09
- ✓ D-12 measured budgets in ci.yml citing run ids (example 2:09:55, coverage 1:42–1:58, mamba 10:49); D-13 hygiene/memory-floor hard gates; D-14 always-uploaded failure scene; D-08 ty advisory step — CI-07
- ✓ Runtime cuts landed at honest seams: finetune_custom_head epochs 3→1 as TEST-SANDBOX-ONLY yaml patch (committed example content byte-identical, 566s measured vs ~31min); num_ctx cut DEFERRED by owner mid-phase, then superseded by the qwen3.8→qwen3.5:4b agent-model swap (11-file sweep incl. docs mirrors, probe-proven tool-calling, stage-3 latency 35min-timeout → 87s)
- Verifier verdict: passed — 18/18 must-haves, 5/5 requirement IDs (CI-03/06/07/08/09), 0 gaps; regression gate 592P/0F over phases 5–8 test files; repo-wide ruff green after CR-01 fix (8a405fe)

Shipped in Phase 8 (Full Execution Rollout & Repair Loop, 2026-10-05):

- ✓ Entire example/ tree executes for real on the shared GB10 box: final census **196 passed / 1 benign skip / 0 failed** (2:59:34) — all 21 notebooks (incl. the formerly gated evo/megaDNA/lora/mcp families), 3 marimo apps (D-18 quadruple incl. export-html validation), `generate_bpe_dataset.py`, every YAML through real `load_config()`; fast lane 1807P/1S with zero new skips
- ✓ 13 repair classes fixed WITH same-change regression tests (DNATokenizer unk, np.fromstring shim, allow_patterns passthrough, column-drop span, lora mirror endpoint, bedtools rootless, D-13 retry probe, …); every notebook repair carries its byte-synced docs mirror; zero cwd false-repairs
- ✓ `models.lock` grown to 24 rows / 14 sha-pinned new ids, prefix↔source= aligned (CI-04); evo-1 fetched safetensors-only into the `~/models-giants` tier outside every cached path (CI-05); A4 offline-load proof recorded
- ✓ example-nightly job complete (CI-06 pre-authorization exercised): staggered 05:30 UTC, staged-serial D-07 (torch → MCP :8000 probes 3+3 → ollama loopback), fail-soft with nonzero summary exit, uv-wheelhouse/bedtools cached, job-level HF_ENDPOINT=hf-mirror.com; first real dispatch consumed (all 6 failures dispositioned) and the completed job's first full dispatch run landed green on its prereq steps
- ✓ ollama loopback systemd unit + runner README in-repo (MCP-01); mcp_example pair green in the both-up state with 6 live-server probes across both transports (MCP-02)
- Verifier verdict: passed — 8/8 must-haves, 11/11 requirement IDs, 0 gaps (08-VERIFICATION.md, 2e64237)

Shipped in Phase 6 + Phase 7 (PlantHelixSeek Showcase, 2026-10-03/04):

- ✓ Registry entries for `PlantHelixSeek-CRE`/`-Anno` (owner-org ModelScope repos, frozen label order, provenance comments) + `dnallm.utils.genomic_coords` coordinate helpers + committed Arabidopsis showcase loci (≤200kb/set, selection.md frozen contract with observed values and tolerance bands) — Phase 6
- ✓ Two flagship showcase notebooks executed for real on GB10 and committed with embedded vega figures: CRE 500/50/50 scan + mean+1.5σ peaks + `bedtools jaccard` (0.3247, band [0.3, 1.00]); Anno 8192/4096 both-strand + argmax BILOU decode → valid GFF3 (exon_f1=0.7522, 59 genes ≥0.8) — every selection.md value reproduced exactly — Phase 7
- ✓ Nightly execution lane asserting the truth-agreement floors parsed from selection.md (never literals; D-08 named-cause failures; parse-guard) + fast structure tests; docs-mirror write-back with wrapper pages, byte-identical mirrors, Showcase nav, models.lock — Phase 7

Shipped in Phase 5 (Execution Harness, Honest Gates & Runner Feasibility, 2026-10-03):

- ✓ Private nbclient execution harness proven end-to-end, including a deliberate-hang kernel-kill test; typed-skip prefixes (`environment-unavailable:`/`optional-dep:`) behind a zero-caller allowlist gate
- ✓ Both false-green CI gates closed together with the docs-mirror drift they hid (docs-validation masking flags removed over a byte-identical mirror resync; mcp extra installed; README proven; branch protection live on dev+main)
- ✓ GB10 runner feasibility settled in writing with real-forward evidence (evo-1 8k variant, evo2 noFP8 config, megaDNA pinned clone, pyBigWig environment-unavailable, marimo export-html) — 05-FEASIBILITY.md
- ✓ Entire example/ tree executed for real (25-item census: 11 PASS / 12 class-tagged FAIL → Phase-8 repair queue / 2 deferred-owner, since executed green via 261003-csd); durable wiring ACTIVE×13 / GATED×8 / 3 marimo apps; fast lane 1716 passed
- ✓ Reopened 5th success criterion (D-07/D-08/D-09) closed; verification re-passed 25/25 at 80b40a5 (2026-10-03) with the NT remote-code smoke real-green through the shim set

Shipped in Phase 3 (Coverage Waves, 2026-10-01):

- ✓ Write new tests until coverage exceeds 90% on the agreed denominator — **96.30%** (7,131/7,405 stmts, verifier-reproduced at HEAD d152d12; ~1,000 behavior tests across 5 ranked-worklist waves; pragma held at 3; 7 allowlisted skips; 8 latent source bugs fixed en route)

Shipped in Phase 4 (CI Gate Enforcement, 2026-10-01):

- ✓ Enforce the gate in CI: `fail_under=90` in `[tool.coverage.report]` enforced through the pytest exit code; two-job CI (fast-leg `coverage-gate` on push/PR + slow-census `coverage-nightly` on a self-hosted GPU runner, both bare `--cov` against the same pyproject); gate green at 96.27–96.30%, red-proven via probe PR #39 (78.91% < floor, 1261 tests otherwise green); required-check branch protection on dev+main; nightly census verified green end to end (run 36811033498: 1656 passed / 7 allowlisted / 0 failed); re-verification passed 10/10

### Out of Scope

- Vendored code coverage (`dnallm/tasks/metrics/`, `enformer_model/`) — upstream HF `evaluate` / ported Enformer, excluded from lint/mypy by design
- `megatron.py` / `mamba_npu.py` test coverage — require Megatron-LM / Ascend NPU toolchains that cannot import in CI
- Root `cli/` legacy launcher cleanup — packaging concern (CONCERNS.md), not needed for coverage
- mypy `|| true` CI fix, dependency lockfile, other CONCERNS items — separate quality work
- Performance optimization (e.g. `attn_implementation` hardcoding) — record, don't fix

## Context

Shipped v1 on 2026-10-01: 1,657 tests passing (7 allowlisted skips), **96.30% line coverage** (7,133/7,407 stmts) on a denominator byte-stable since Phase 1, `fail_under = 90` enforced through the pytest exit code and required-check branch protection on dev+main.

- Test config lives solely in `pyproject.toml [tool.pytest.ini_options]` (`--asyncio-mode=auto`, `--timeout=300`, `--strict-markers`; markers `slow`, `pdf`, `performance`, `integration`; testpaths `tests/` + `dnallm/mcp/tests/`)
- Enforcement surface is `fail_under = 90` in `[tool.coverage.report]` — a bare `--cov` on any invocation activates it; CI census jobs run exactly that
- CI shape: `coverage-gate` (fast PR leg, push/PR) + `coverage-nightly` (slow census, self-hosted `dnallm-nightly` GPU runner, models.lock-keyed cache) + `test-mamba` (same runner, schedule/dispatch-only); matrix legs + windows leg stay ungated
- Skip discipline: every skip is typed and matched against `tests/expected_skips.yaml` by `scripts/audit_skips.py` in 4 CI jobs — an unexpected skip fails the run
- transformers compatibility spans 4.49–5.x via `dnallm/utils/transformers_compat.py`; installed dev env uses transformers 5.17, torch 2.11 cu130
- Known tech debt (reviewed, dispositioned, non-blocking): see `.planning/v1-MILESTONE-AUDIT.md` tech-debt ledger — 7 warning-tier + ~31 info-tier review findings, 3 acknowledged deferred engineering items (STATE.md), concentrated in the `/gsd-ship` triage path
- Codebase map with full concerns list: `.planning/codebase/` (STACK, ARCHITECTURE, TESTING, CONCERNS)

## Constraints

- **Tech stack**: pytest + pytest-cov; coverage configured via `[tool.coverage.run]` omit list in `pyproject.toml` — no new test frameworks
- **Compatibility**: suite must keep passing on the CI matrix (Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0); tests must not pin to a single transformers minor version
- **CI**: coverage-gated run includes `slow` tests — requires network for model downloads; runtime cost accepted by owner
- **Scope**: bug fixes limited to what correctness/coverage requires; no refactors beyond that

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Coverage denominator: whole `dnallm/` excluding vendored dirs and unimportable adapters | Vendored code is upstream and excluded from lint/mypy; adapters cannot import in CI — including them makes 90% unattainable | ✓ Landed Phase 1 (7-entry omit list; baseline 45.92% on 7,383 stmts) |
| Audit first, then fix | Gap report drives test-writing priorities and surfaces real bugs before mass test authoring | ✓ Landed Phase 1 (43-row ranked worklist from measured artifacts) |
| CI hard gate `--cov-fail-under=90`, run includes slow tests | Prevents coverage regression; owner accepts network downloads and longer CI runs for real coverage | ✓ Landed Phase 4 (fail_under=90 native via pyproject; green 96.27–96.30%; red-proven PR #39; branch protection on dev+main) |
| GATE-02 amended: PR gate = fast leg; slow census = nightly on self-hosted GPU runner | Hosted 360-min cap killed the 6h CPU census; org cannot use larger hosted runners | ✓ Landed Phase 4 (census ~15 min warm on `dnallm-nightly`; dispatch/cron only, fork PRs can't reach it) |
| Fix real code bugs encountered during audit (AUROC, CrossDNA) | Skipped-crash tests hide real defects; unskipping them is required for honest coverage | ✓ Landed Phase 2 (both fixed, regression-tested, unskipped) |
| Subprocess coverage: start minimal, escalate only on canary evidence (Phase 1) | pytest-cov 7 removed `.pth` subprocess auto-measurement; no collected test spawns subprocesses | ✓ Landed Phase 1 (AUDIT-04; escalation trigger recorded) |
| test-mamba on the self-hosted GPU runner at nightly cadence (schedule/dispatch-only), not push/PR | Per-run CUDA kernel source build is too heavy for per-push cadence (GATE-02 amended); PR-authored code (incl. forks) must never execute on the self-hosted box | ✓ Landed v1 closeout (quick task 261001-ith; dispatch run 36821471332 green) |
| Harness: nbclient plain `execute()` + `shutdown_kernel=immediate` in a tmp sandbox; tree-clean check is delta-zero vs an import-time baseline | nbclient 0.11 NotebookClient is not a context manager; the owner's live IDE churn on tracked notebooks is not harness business | ✓ Landed Phase 5 (05-01/05-05) |
| Gated lane is probe-then-execute with live probe results carried in skip messages (both directions proven) | An ever-green skip is the same dishonesty class as the false-green CI gates this milestone closes | ✓ Landed Phase 5 + 261003-csd (ollama/MCP gates) |
| Census before repair: full-tree real execution with class-tagged exact tracebacks ranks the repair queue | Phase 8 repair must be evidence-ranked, not anecdotal | ✓ Landed Phase 5 (05-CENSUS.md: 11 PASS / 12 FAIL / 2 deferred) |
| Showcase honesty: illustrative-loci framing is dual (full provenance-cell disclaimer + per-figure captions) and enforced by a genome-wide denylist structure test | Single-locus results must never read as genome-wide accuracy; SHOW-07 intent is mechanical plus judgment, both covered | ✓ Landed Phase 7 (UAT-verified on GitHub blob rendering and wording) |
| Assertion ownership is two-layer: notebooks print metrics + floors comparison; the authoritative band assertions live in `tests/examples/` and parse selection.md at test startup | Transparent in-notebook numbers without brittle in-kernel asserts; single source of truth for thresholds (D-05/D-06) | ✓ Landed Phase 7 (`_parse_floors`/`_parse_bands`, parse-guard, 4/4 band rows) |
| Repair rollout by model family with per-repair full-census reconciliation (D-01/D-03) | Same root cause benefits the whole family; every step keeps a full-tree baseline, so regressions surface immediately | ✓ Landed Phase 8 (final census identical across 08-08/08-09: 196P/1S/0F) |
| Nightly coexistence is staged-serial with explicit VRAM/process cleanup + assert between stages (D-07 + owner VRAM directive 2026-10-05) | Heavy torch, MCP :8000, and ollama share one box; unserviced leftovers compound into OOM (orphaned 24.5GB server lesson) | ✓ Landed Phase 8 (≥35Gi-available rule in the rollup inheritance notes; census trough 39Gi, floor held) |
| models-cache layer dropped from CI; cold pulls per run (owner decision 2026-10-05) | Lock-only cache ≈15.2GiB > 10GB Actions quota and never actually saved; cold stage 1 measured green at 65 min against a 2700-min budget | ✓ Adopted Phase 8 close; ci.yml edit lands in Phase 9 |
| giants/evo exit the pytest census as a deselect, not a skip (owner directive 2026-10-05) | runner env is AVAILABLE — a typed skip would fake environment-unavailability; committed executed-notebook outputs remain the evidence | ✓ Adopted Phase 9 (D-01, marker + not-giants deselect) |
| mcp_example agent brain swapped qwen3.8→qwen3.5:4b; num_ctx cut stays deferred (owner 2026-10-06) | 17GB/256k-ctx latency tail broke the 1800s cell timeout; 3.3GB 4b model probe-proven for tool-calling, stage-3 87s; server config never touched | ✓ Adopted Phase 9 (11-file sweep; timeout stays 3600s guard) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-06 after Phase 9 — milestone v1.1 100%*
