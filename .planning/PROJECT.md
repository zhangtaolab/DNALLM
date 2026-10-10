# DNALLM — Test Suite Audit & Coverage Hardening

## What This Is

DNALLM (`dnallm` v0.8.0) is a Python toolkit for fine-tuning, inference, and benchmarking of DNA large language models (150+ pretrained models from HF/ModelScope), plus an MCP server exposing them to LLM agents. Milestone v1 (shipped 2026-10-01) was a quality-engineering cycle on that existing codebase: the pytest suite was audited end to end, test gaps closed, and line coverage driven from 45.92% to 96.30% behind a CI-enforced >90% hard gate. Milestone v1.1 (shipped 2026-10-07) made every artifact under `example/` execute for real — 21 notebooks, 3 marimo apps, the helper script, every YAML — on the nightly GPU runner, fixed every error it surfaced with same-change regression tests, delivered the PlantHelixSeek showcase notebooks over committed Arabidopsis loci, and brought the execution tests under formal nightly gating. Milestone v1.2 (shipped 2026-10-10) delivered the paper-revision suite: the eleven reviewer-requested capabilities REV-01..REV-11 (eval-semantics guard, metric registry, IA³ + PEFT presets, from-scratch baselines, probing, zero-shot VEP, multi-seed sweeps, FIMO motif scanning, MCP tool expansion) with the suite, gate, and CI honest-green throughout.

## Core Value

A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## Current Milestone: none active — v1.2 shipped 2026-10-10

**v1.2 delivered (all 11 REQ-IDs, full ledger in `milestones/v1.2-REQUIREMENTS.md` after archival):**
- Evaluation semantics: no silent test-as-eval (REV-01), metric registry contract (REV-02), docs/terminology/comparability warnings (REV-03)
- Adaptation & baselines: IA³ adapter (REV-04), per-model PEFT target presets (REV-05), `random_init=True` from-scratch loading (REV-06)
- New evaluation capabilities: frozen-embedding probing (REV-07), zero-shot VEP module with token-slot alignment rules (REV-08), multi-seed sweep protocol with uncertainty aggregation (REV-09)
- Interpretation & agent surface: JASPAR/CIS-BP PWM matching (REV-10), MCP tools for ISM/hotspots/zero-shot scoring (REV-11)

**Next milestone goals:** not yet defined — `/gsd-new-milestone` when the paper revision's next needs are known (candidate intake: golden-fixture activation inputs (issue #44), dnallmmark re-run findings, v1.3+ future-requirement ledger in `milestones/v1.2-REQUIREMENTS.md`).

## Current State

v1.2 shipped 2026-10-10 (11/11 requirements, 3/3 phases verified at the milestone fixpoint, audit 0 blockers / 2 owner-accepted overrides / tech-debt ledger in `milestones/v1.2-MILESTONE-AUDIT.md`). Package at 0.8.0. Fast lane at close: 2404 passed / 0 failed; coverage 96.72% against `fail_under=90`; full CI matrix green (3.11/3.12/3.13 × numpy 1.26.4/2.2.0 + windows). Zero new dependencies beyond owner-approved scikit-allel; facade byte-stable; zero Co-Authored-By trailers. Known carried debt (owner-tracked): golden-fixture activation inputs (GitHub issue #44 — fixture-files-only when the manuscript values arrive), mkdocs --strict 15 pre-existing warnings, numpy 2.5.x/coverage py3.13 instrumentation incompatibility (numpy ceiling call when the 1.26.4 matrix leg retires), pytest-cov dotted-target env crash (workaround: coverage CLI), old-terminology prose hits outside the check surface, runner-box operational items carried from v1.1 (ollama loopback re-apply, cache-quota, `$HOME` cleanup), giants-lane manual-execution policy (W1).

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

(Next milestone not yet defined — run `/gsd-new-milestone`; candidate inputs listed under Current Milestone above.)

### Validated — v1.2 (Paper Revision Suite Support, shipped 2026-10-10)

Milestone headline — 11/11 REQ-IDs satisfied (full ledger in `milestones/v1.2-REQUIREMENTS.md` after archival; 2 owner-accepted overrides):

- ✓ All eleven reviewer capabilities REV-01..REV-11 shipped and verified end-to-end (per-phase detail in the Validated sections below)
- ✓ Cross-phase integration audit-clean: 9/9 seams, 4/4 E2E flows, one evaluate_vcf kernel feeding CLI + MCP verbatim, metric-name registry validated at every emission, CHANGELOG SHA chain fully verified
- ✓ Quality gates held throughout: 2404 fast-lane tests 0 failed (+151 vs v1.1 close), 96.72% coverage, every new module ≥96% per-module, full CI matrix green, zero new deps beyond scikit-allel

Shipped in Phase 12 (Motif Matching, MCP Tools & Milestone Closeout, 2026-10-10):

- ✓ FIMO-convention motif scanning + JASPAR client (MOTIF-01/REV-10): exact-DP p-values (MEME 4.8.1 recipe), GC-matched background, both strands, single BH call full-set, E-values; stdlib REST client on jaspar.elixir.no with retry/size-cap/matchable errors; paper-exact Fig 4a clause owner-deferred (issue #44) — harness committed + green on synthetic stand-in, activation fixture-files-only
- ✓ MCP analysis tools + host/port fix (MCPE-01/REV-11): ism_scan/hotspots/zero_shot_score under timeout-wrapper + error-dict conventions, one evaluate_vcf kernel with verbatim skip accounting; `--host/--port` CLI>YAML precedence fixed on both transports (single `_resolve_bind_address` consumption point)
- ✓ Milestone closeout: IA³ docs chapter completed (DOCS-01 split delivery closed), CHANGELOG REV-01..11 SHA-linked evidence chain, coverage expectation updated (96.72%)

Shipped in Phase 11 (PEFT Adaptation, Baselines & New Evaluation Capabilities, 2026-10-09):

- ✓ IA³ adapter + per-model PEFT target presets (PEFT-01/PEFT-02/REV-04+05): IA³ fine-tunes exactly as LoRA (transformer AND Mamba slow-acceptance proven), config-time `use_ia3×use_qlora` + trainer-init `lora×ia3` matchable rejections, 35/35-family `lora_targets.yaml` presets with FFN⊆IA³ invariants, `peft_dry_run` validator (loud on no-adapter), `requires_grad` ratio guard, seed-pinned IA³ roundtrip (peft #2429 class covered)
- ✓ `random_init=True` from-scratch loading (BASE-01/REV-06): from_config-only path (no download, proven), loud banner + per-tensor sha256 proof with counted exceptions, allowlist frozenset + matchable ValueError, same-seed reproducibility, two-architecture acceptance (BERT generic + mamba allowlist)
- ✓ Frozen-embedding probing (PROB-01/REV-07): layer/pooling-selectable extraction, fixed-hyperparameter logistic/mlp probes, train-only scaler (spy-tested), registry-only metrics, atomic sha256-keyed npz cache with hit assertions, F4 schema documented; real-model slow acceptance
- ✓ Zero-shot VEP (VEP-01/REV-08): `evaluate_vcf` via scikit-allel (owner-approved dep, bounded `<2`), D-17 ClinVar convention (≥1 star/SNV/P-LP-vs-B-LB reported alongside), uppercase-window soft-mask guard, skip-as-data accounting per reason, paradigm↔architecture ValueError, `dnallm-vep` CLI, formula docstrings + README protocol; ClinVar chr22 1k × 5-model GPU acceptance — magnitude clause owner-overridden 2026-10-09 (small-model near-floor AUROCs 0.490–0.581 honest finding; same-scale BPE anchor matched 0.543 vs 0.538)
- ✓ Multi-seed sweep (SEED-01/REV-09): `run_seeds` {model}/{task}/seed_{s}/ protocol with D-16 same-split semantics, pure `aggregate_seeds` (n<3 omit, 3≤n<10 t-interval, n≥10 seeded bootstrap), boundary matrix n=2/3/9/10, 3-seed end-to-end trial with statistics block
- Review chain: 3 iterations, 15 findings all fixed, disposition 0 open; full fast lane 2253P/0F (CI gate shape); regression gate over Phase-10 files 530P; verifier 8/9 truths + 1 owner override recorded

Shipped in Phase 10 (Evaluation Contract Layer & Shared Scaffolding, 2026-10-09):

- ✓ Evaluation-semantics leak guard + explicit `evaluate(split=...)` (EVAL-01/REV-01): trainer can never silently evaluate on the test split — atomic `eval_strategy="no"` + `eval_dataset=None`, `allow_test_as_eval` loud opt-in, symmetric collision ValueErrors (early-stopping/best-model/train-only/unsplit), predict-path `evaluate(split=...)` with canonical-keyed result JSON (`eval_{split}_result.json`, runtime block separated)
- ✓ Metric registry contract (METR-01/REV-02): `dnallm/tasks/metric_registry.py` — 28 canonical names, aliases recognized-never-emitted, frozen mapping, import-light (AST-proven), 99% coverage, provably outside the vendored omit glob; `metrics.py` emits exclusively through it with byte-identical emitted keys
- ✓ Docs/terminology/comparability honesty (DOCS-01/REV-03): full-surface "DNA large language models" sweep (docs+README+13 docstrings+example pair, 0 residual; verbatim paper titles sanctioned exception), `validate_sequences` docstring + dropped-row count log, PEFT chapter with honest IA³ pointer, CHANGELOG REV-ID-inline evidence chain opened
- ✓ One-pass scaffolding for Phase 11: `use_ia3` field-first, field-complete Ia3Config/VepConfig/SweepConfig + `load_config()` registration, presets package-data; zero new dependencies in Phase 10, facades byte-stable
- Verifier verdict: passed — 24/24 must-haves, 3/3 requirement IDs, 0 gaps; code review converged 12→2→0 findings (14 atomic fixes); coverage proofs metric_registry 99% / metrics.py 100% / vep.py 100%

### Validated — v1.1 (Example Execution Testing & Repair, shipped 2026-10-07)

Milestone headline — 32/32 REQ-IDs satisfied (full ledger in `milestones/v1.1-REQUIREMENTS.md`):

- ✓ Real-model execution for every artifact under `example/`: 21 notebooks, 3 marimo apps, `generate_bpe_dataset.py`, every YAML through real `load_config()` — final census 196P/1S/0F, formally nightly-gated
- ✓ Every surfaced error fixed with a same-change regression test (13 repair classes + all review findings) across example code, the docs mirror, and dnallm library bugs
- ✓ CI false-green repaired and example tests under formal gating (WR-08/WR-09 closed; hard census collection gate; honest docs-validation)
- ✓ PlantHelixSeek-CRE/-Anno showcase: registry route, committed ≤200kb Arabidopsis loci, two executed notebooks reproducing the frozen truth-agreement metrics exactly (jaccard=0.3247, exon_f1=0.7522), byte-identical docs-mirror write-back

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

Shipped v1.2 on 2026-10-10 (145 commits over v1.1; 3 phases, 12 plans): the paper-revision suite (REV-01..REV-11) landed in three sequential waves of file-disjoint parallel agents under pathspec-commit discipline, with every phase verified at the milestone fixpoint (zero code changes between regenerations). Two owner-accepted overrides recorded in phase VERIFICATION frontmatter: VEP-01 literature-magnitude (small ≤50M plant models honestly at the random floor; same-scale BPE anchor matched) and MOTIF-01 Fig 4a coordinates (golden-fixture deferral, issue #44). Fast lane at close: 2404 passed / 0 failed; coverage 96.72%; full CI matrix green including Windows.

Shipped v1.1 on 2026-10-07 (373 commits over v1; 5 phases, 24 plans): the whole `example/` tree executes for real behind a staged example-nightly job (census hard gate pinned at 197/206 collected, hygiene floors ≥35 GiB, fail-soft summary with a hard non-zero exit), the docs mirror is byte-identical under `check_docs_sync.py`, and the showcase notebooks assert truth-agreement floors parsed from `selection.md`. Fast lane at close: 1,933 passed / 1 allowlisted skip; coverage-nightly 96.42% against the unchanged `fail_under=90` gate.

Shipped v1 on 2026-10-01: 1,657 tests passing (7 allowlisted skips), **96.30% line coverage** (7,133/7,407 stmts) on a denominator byte-stable since Phase 1, `fail_under = 90` enforced through the pytest exit code and required-check branch protection on dev+main.

- Test config lives solely in `pyproject.toml [tool.pytest.ini_options]` (`--asyncio-mode=auto`, `--timeout=300`, `--strict-markers`; markers `slow`, `pdf`, `performance`, `integration`; testpaths `tests/` + `dnallm/mcp/tests/`)
- Enforcement surface is `fail_under = 90` in `[tool.coverage.report]` — a bare `--cov` on any invocation activates it; CI census jobs run exactly that
- CI shape: `coverage-gate` (fast PR leg, push/PR) + `coverage-nightly` (slow census, self-hosted `dnallm-nightly` GPU runner, models.lock-keyed cache) + `test-mamba` (same runner, schedule/dispatch-only); matrix legs + windows leg stay ungated
- Skip discipline: every skip is typed and matched against `tests/expected_skips.yaml` by `scripts/audit_skips.py` in 4 CI jobs — an unexpected skip fails the run
- transformers compatibility spans 4.49–5.x via `dnallm/utils/transformers_compat.py`; installed dev env uses transformers 5.17, torch 2.11 cu130
- Known tech debt (reviewed, dispositioned, non-blocking): v1 — see `milestones/v1-MILESTONE-AUDIT.md` (7 warning-tier + ~31 info-tier findings); v1.1 — see `milestones/v1.1-MILESTONE-AUDIT.md` (0 blockers, 13 deferred items: runner ops, MCP flag-override bug, giants-lane manual policy, one stale README sentence)
- Codebase map with full concerns list: `.planning/codebase/` (STACK, ARCHITECTURE, TESTING, CONCERNS)

## Constraints

- **Tech stack**: pytest + pytest-cov; coverage configured via `[tool.coverage.run]` omit list in `pyproject.toml` — no new test frameworks
- **Compatibility**: suite must keep passing on the CI matrix (Python 3.11/3.12/3.13, numpy 1.26.4 & 2.2.0); tests must not pin to a single transformers minor version
- **CI**: coverage-gated run includes `slow` tests — requires network for model downloads; runtime cost accepted by owner
- **Scope**: bug fixes limited to what correctness/coverage requires; no refactors beyond that

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Owner overrides instead of silent scope reduction (P11 VEP magnitude, P12 MOTIF-01 golden fixture) | Plans may add to, never subtract from, roadmap SCs; unmet acceptances must not pass silently — the owner adjudicates with the evidence on record | ✓ 2 overrides recorded in phase VERIFICATION frontmatter with reasons/timestamps; golden fixture tracked as GitHub issue #44 (fixture-files-only activation, no milestone reopen) |
| Stale verifications regenerate at the milestone fixpoint | Phase 11/12 edits invalidated earlier covered-file digests; regenerating with zero code changes between proves the truths still hold at the shipped HEAD | ✓ Phase 10 re-verified 24/24 at 1e77a87; Phase 11 regenerated at the same fixpoint at close |
| Zero new dependencies beyond scikit-allel (milestone invariant) | Reviewer capabilities fit the existing scipy/sklearn/stdlib footprint; the only addition (VCF reading) owner-approved with empirical Windows/numpy verification | ✓ Held — pyproject delta over the milestone is exactly 2 constraint changes on existing transitive deps (pyarrow<26, pydantic-ai>=1.107.0,<2 CI-drift fixes) |
| VCF reading via scikit-allel (supersedes stdlib-reader research decision) | Original premise (VCF libs break Windows CI) disproven empirically 2026-10-09: scikit-allel 1.3.13 has Windows cp310–313 wheels and numpy 1.26.4/2.2.0 verified live; only new required transitive dep is dask[array]; INFO parsing (ClinVar CLNSIG) natively covered | ✓ Owner-approved 2026-10-09; lands Phase 11 B5 (ROADMAP/REQUIREMENTS/STATE amended) |
| Single-tree concurrent wave execution with pathspec commits | Owner-fixed v1.2 mode (v1.1 Phase 8 proven); shared-index commit race in Phase 10 Wave 1 (c751df6 swept a sibling's staged files, no content loss) showed plain `git commit` commits the whole index | ✓ Wave 1 completed 4/4; all subsequent commits pathspec-limited; mandated in Phase 11/12 dispatch prompts |
| CHANGELOG.md as the single sanctioned cross-lane append surface | D-09 same-commit REV-ID entries vs 4-lane file-disjointness resolved via coupling_justified declaration + idempotent unique-anchor append discipline | ✓ Checker-verified; three REV entries coexisted intact across concurrent lanes |
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
*Last updated: 2026-10-10 after v1.2 milestone (paper revision suite shipped 0.8.0: 11/11 REQ-IDs, 2 owner overrides, audit 0 blockers)*
