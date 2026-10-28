# DNALLM — Test Suite Audit & Coverage Hardening

## What This Is

DNALLM (`dnallm` v1.3.3) is a Python toolkit for fine-tuning, inference, and benchmarking of DNA large language models (150+ pretrained models from HF/ModelScope), plus an MCP server exposing them to LLM agents. Milestone v1 (shipped 2026-10-01) was a quality-engineering cycle on that existing codebase: the pytest suite was audited end to end, test gaps closed, and line coverage driven from 45.92% to 96.30% behind a CI-enforced >90% hard gate. Milestone v1.1 (shipped 2026-10-07) made every artifact under `example/` execute for real — 21 notebooks, 3 marimo apps, the helper script, every YAML — on the nightly GPU runner, fixed every error it surfaced with same-change regression tests, delivered the PlantHelixSeek showcase notebooks over committed Arabidopsis loci, and brought the execution tests under formal nightly gating. Milestone v1.2 (shipped 2026-10-10) delivered the paper-revision suite: the eleven reviewer-requested capabilities REV-01..REV-11 (eval-semantics guard, metric registry, IA³ + PEFT presets, from-scratch baselines, probing, zero-shot VEP, multi-seed sweeps, FIMO motif scanning, MCP tool expansion) with the suite, gate, and CI honest-green throughout. Milestone v1.4 (shipped 2026-10-22) delivered the first user-facing onboarding surface: a pure-function pre-flight core (`dnallm.preflight`) plus the `dnallm preflight` Textual wizard that takes a user from their own dataset to a validated, model-compatible config and an optional ModelScope-first model download, with `model_info.yaml` (236 entries) as the runtime single source of truth behind one cached read-only registry loader.

## Core Value

A fully passing pytest suite with >90% line coverage across `dnallm/` (excluding vendored code), enforced by a CI hard gate so coverage cannot regress.

## Current Milestone: none — planning next

v1.4 shipped 2026-10-22 (verified_closeout). Start the next cycle with `/gsd-new-milestone`. Carried future scope from the v1.4 requirements deliberation (owner, 2026-10-20): Track B (dnallmmark TUI substrate merge — merge from downstream, never duplicate) and Track C (R2 training dashboard, R3 MCP control room, R5 VEP runner, R6 interpretation triage); defects #62/#63 remain open on GitHub (#64's registry-consumption half shipped as MCP-02).

## Current State

v1.4 shipped 2026-10-22 (5 phases, 23 plans, 62 tasks; audit `tech_debt`, 0 critical gaps — `milestones/v1.4-MILESTONE-AUDIT.md`; verified_closeout, 10 carried suppressions). Package at 1.3.3 (no product release cut in-cycle; publish.yml stays event-gated). New surfaces: `dnallm.preflight` (99% package coverage, every module ≥96%), `dnallm preflight` TUI wizard (per-module ≥97.4%, no omit entries), `dnallm/models/registry.py` (100%) now the sole parser of `model_info.yaml`, `dnallm/mcp/consistency.py` (98%) with the strict CI gate over committed configs; `tui` extra live, `base` alias retired; `import dnallm.preflight` 0.29 s torch-free. Branch `v1.4` (main frozen all cycle); milestone tag `milestone/v1.4`. Known carried debt (owner-tracked): Phase-25 TUI copy read-through delegated-verified, not human-verified (25-UAT); Phase-24 WR-01 unprobed third-party ModelScope hosting; v1.3 WINDOWS #13/#18/#19/#25/#26/#27 (#25/#26/#27 resume checklists in the WINDOWS ledger).

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

(None — all v1.4 requirements shipped and verified; the next milestone defines fresh requirements via `/gsd-new-milestone`.)

### Validated — v1.4 (Pre-flight UX & Registry-as-Truth-Source, shipped 2026-10-22)

Shipped in Phase 25 (Terminal UI — Pre-flight Wizard & `tui` Extra, 2026-10-22):

- ✓ TUI-01..TUI-08 — `dnallm preflight` Textual wizard over the pure core: `tui` extra + install/CI wiring + thread-worker seam landed before any panel; every blocking core call crosses a worker as a typed message with a proven BaseException cancel sentinel escaping `download_model`'s retry loop; import facade 0.29 s and torch-free (was 3.71 s); env/data/suitability/config panels with lab-first listings, honest UNKNOWN, collect-all field-level errors, byte-faithful YAML preview, save gated on zero problems, real byte-progress download with cooperative stop; proven by one offline Pilot walk; copy carries one-line glosses for all twelve ML terms and all eight task types from one shared mapping (gap plans 25-11..25-13)
- ✓ PKG-01 — `tui` extra (textual + platformdirs) declared and installed on every CI leg; deprecated `base` extras alias retired in the same install-line change

Shipped in Phase 26 (MCP Consistency — Registry-Keyed Check & Honest Listing, 2026-10-22):

- ✓ MCP-01/MCP-03 — `check_config_registry_consistency()`: read-only, repo-id-keyed comparison of every per-model MCP YAML (plus server-config reference integrity) against the packaged registry — tiered findings (four hard fields strict-fatal; source/max_length/unknown-repo WARN; description INFO; unreadable/invalid/dangling ERROR fatal in both modes), collect-all in one pass, CI gate test over the committed `dnallm/mcp/configs/` green on every lane via existing testpaths (98% module coverage)
- ✓ MCP-02 — `get_all_available_models` serves the full 236-entry registry listing lab-first with configured/enabled/loaded overlay, `server_config_key` on configured rows, and `unmatched_configured` visibility (issue #64's registry-consumption half; client surface pinned by 73 unedited SDK tests)

Shipped in Phase 24 (Preset Loading — `load_preset_model` Registry Rewire, 2026-10-21):

- ✓ LOAD-01 — `load_preset_model` resolves registry entry names first (then repo ids) via `try_resolve`, falls back to the legacy `MODEL_INFO` family table, returns 0 on miss; finetuned entries patch `num_labels`/`label_names`/`threshold` verbatim into a `model_copy` of the caller's TaskConfig (never mutated); `task_type` validated with a plain-language ValueError, never patched; base-model lineage logged; dead preset-list branch deleted; ten-family zhangtaolab regression pins unchanged family behavior (verified 10/10 must-haves, 232+41+257 tests green, model.py coverage 93%)

Shipped in Phase 22 (Registry Foundation — Runtime Loader & Enrichment, 2026-10-20):

- ✓ REG-01 — `dnallm/models/registry.py`: cached single-parse read-only loader, sole parser of `model_info.yaml` (typed frozen entries; `resolve`/`try_resolve`/`get_by_task_type`/`get_by_base_model`; `reload()` hook; CSafeLoader with fallback; grep-guard against second parsers)
- ✓ REG-02 — registry enrichment: `tokenizer_type`/`max_length` on all 49 pretrained + 14 external-base finetuned entries (109 committed fields), every value measured from hub metadata or carrying a cited derivation; plausible defaults rejected (GPTNeoX class-default 1024 → upstream-script 512; repo-name 32768 → measured 32770); provenance re-verified live at UAT (2026-10-20, 10-repo remote re-fetch, 0 mismatches)
- ✓ REG-03 — extended integrity tests: section counts pinned 49/187, field-level insertion-only check, spot-pinned load-bearing values (hyenadna 32770, DebertaV2Tokenizer), no-write-API verb/emitter scans, frozen-entry guards, sole-mentioner allowlist

Shipped in Phase 23 (Pre-flight Core — Pure Validation / Suitability / Config API, 2026-10-21):

- ✓ PFC-01 — `dnallm/preflight/` `validate_data()`: read-only collect-all report (stats, charset census, length distribution, severities + reason codes), standalone reader parity-locked against `DNADataset`, 7 formats + jsonl, pkl refused and never unpickled, no row ever dropped or mutated
- ✓ PFC-02 — `check_compatibility()`: three-state verdicts (OK/WARN/FAIL) over task × length × tokenizer/charset; WARN never blocks; k-mer/BPE budget classifier (k from name regex, budget = effective_max_length + k − 1) neutralizes the 140-entry EsmTokenizer k-mer trap
- ✓ PFC-03 — `probe_availability()`: ModelScope-first, no weights ever (cache detection is pure path checks + `try_to_load_from_cache`/`scan_cache_dir`), cached/remote-confirmed/unknown with exact-int size + revision; injectable seam keeps the unit suite network-free
- ✓ PFC-04 — `validate_sections` factored verbatim out of `load_config` (the one sanctioned refactor, equivalence-pinned) + `generate_config`/`edit_config` round-trip with collect-all field-level errors in fixed section order
- ✓ PFC-05 — intent-routed listings over the base↔finetuned axis ("fine-tune" lists bases, "predict now" lists task models with heads/labels)
- ✓ PFC-06 — degraded mode: every probe/network failure maps to UNKNOWN with a reason (RATE_LIMITED/OFFLINE/PROBE_TIMEOUT/NOT_FOUND_ON_EITHER_HUB/PROBE_ERROR); no core function ever raises from unavailability
- ✓ PFC-07 — `zhangtaolab/*` ranks first in every listing the core produces (owner directive 2026-10-20)

### Validated — v1.3 (Platform Reach & Packaging Honesty, shipped 2026-10-16)

- ✓ Ollama surface dispositioned per directive — OLLM-01..03 (census 216/217 re-pinned, sanctioned-residue oracle, nightly stage-3 deleted)
- ✓ GAP-1 zero-rung disposition — GAPC-01/02 (transformers 5.17 green under the existing ladder; dead-branch proof)
- ✓ Windows parity core + rollout — WPAR-01..06/08 (numpy evaluators, dispatch seams, frozen pairs, direction tests; test-windows green)
- ✓ Publish chain — PUBS-01/02/03/05, PKGH-08 (event-gated upload, never-uploading smoke, coverage-CLI fix, mypy cap)
- ✓ Packaging honesty — PKGH-01..04/06/07 (atomic extras surgery, alias, cuda121/rocm retirement, lane-split docs, GPU matrix)
- ✓ Docs accuracy — DOCR-01..03 + PKGH-05 (149-file ledger, cuda130 default everywhere, strict zero warnings)
- ○ PUBS-04 / WPAR-07 — declaration/delivery halves verified; acceptance halves owner-deferred (WINDOWS #25/#27) — Known Gaps at close

### Validated — v1.3 partial (shipped 2026-10-13)

- ✓ Ollama pytest surface skip-marked per owner directive — Phase 13 (stage-3 ollama batch deleted; mcp_example pair probe-gated typed-skips; census re-pinned 216/217; first green example-nightly verified 2026-10-13 run 38308972028)
- ✓ GAP-1 disposition: shim-zero-rung — Phase 14 (NT promoter + BPE leg load real on transformers 5.17 via the 10-02 rung; invert_attention_mask dead-branch; WINDOWS #11/#12/#15/#23 closed; rice.uga.edu external transient owner-accepted)
- ✓ Windows parity core: numpy interval evaluators — Phase 15 (genomic_intervals.py 3-function surface, 39 oracle tests, 100% module coverage, platform-neutral)
- ✓ Windows parity rollout: dispatch seams + platform gates + frozen-value parity — Phase 16 (SC matrix 5/5, nightly record 38441258701, rice caveat owner-accepted)

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
- Track B (dnallmmark TUI substrate merge) and Track C (R2 training dashboard, R3 MCP control room, R5 VEP runner, R6 interpretation triage) — v1.4 owner boundary (2026-10-20); future-milestone scope, Track B merges from downstream rather than duplicating

## Context

Shipped v1.4 on 2026-10-22 (228 commits over v1.3; 5 phases, 23 plans): the pre-flight surface (pure core + Textual wizard) with the registry as runtime single source of truth — one cached read-only loader (236 entries) consumed by `load_preset_model`, the MCP listing, and the CI-gated consistency checker; `tui` extra live, `base` alias retired; per-module coverage ≥96% everywhere new (preflight 99% package, TUI ≥97.4%, registry 100%, consistency 98%); 23/23 requirements, 7/7 integration seams, 3/3 E2E flows proven live at the milestone audit.

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
| [Phase 24] OQ-1 STRICT: registry entries with `task_type='classification'` (6 PlantCAD2, third-party) stay unloadable by name — no aliasing to user binary/multiclass | Smallest reading of locked D-05; all six sit outside every zhangtaolab flow; widening the mismatch definition would silently change the validated contract | ✓ Pinned by a real-entry test; documented known limitation in the docstring; registry data fix out of scope (261020-kwt discipline) |
| [Phase 24] Patched preset configs built with `model_copy(update=...)` only — never model_validate/reconstruction; caller's TaskConfig never mutated | `model_post_init`'s regression branch would clobber registry `label_names` (e.g. `['promoter strength in tobacco leaves']`) to `['value']`; research Pitfall 2 reproduced against live data | ✓ Regression-preservation test pins verbatim labels; non-mutation asserted twice (tracer + dedicated test) |
| [Phase 24] Registry-first resolution order with registry-wins-on-collision disclosed in the docstring (24 third-party family names change resolved repo — intentional, documented) | D-01 order + research A2: the delta is real but every zhangtaolab-default family resolves identically; discoverability beats a silent behavior change | ✓ DNABERT-2 collision pinned to `zhihan1996/*`; ModelScope-hosting assumption carried as review WR-01 (deferred, Phases 25/26 awareness) |
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

| Nightly interpreter honesty (v1.3, fix bdf6ba3) | Unpinned `uv venv` on a persistent self-hosted box resolves uv-managed interpreters first — pin to setup-python's exact toolcache interpreter and py-tag any wheelhouse cache key | ✓ Drift class closed; trio 38695301631 all green under true labels |
| Publish upload guard lands before any trigger exists (v1.3 phase 17 first act) | PyPI filenames are immutable — an unguarded dispatch→upload path is un-recoverable | ✓ Event gate live-proven ×3 (publish always skipped off-release) |
| Owner human-gates halt plans, never improvise past them (17-03/18-03 pattern) | Supply-chain registration and real-hardware acceptance are owner-only acts; the ledger carries the resume checklist | ✓ Both carried as WINDOWS #25/#27 with minute-level resume paths |
| One census-moving edit per phase (D-03 discipline, held all milestone) | The 216/217 triple is the suite's integrity anchor; every test-set change re-pins triple + skips + stages same-change | ✓ Triple byte-stable through 8 phases incl. the phase-20 test additions (measured hold, no blind re-pin) |
| Registry loader born read-only, forever (v1.4 Phase 22) | A parse owner with write power becomes a second mutation surface; registry maintenance = scripted text insertion, never a YAML round-trip | ✓ REG-03 structure guards CI-red on any write API, second parser, or unfrozen entry |
| Enrichment provenance: measured values or cited derivations only — plausible defaults rejected (v1.4 Phase 22) | A plausible default is indistinguishable from a measurement in review but wrong in production (GPTNeoX class-default 1024 vs upstream-script 512; repo-name 32768 vs measured 32770) | ✓ 109 committed fields cross-checked against derivation records; 10-repo live remote re-verification at UAT 2026-10-20, 0 mismatches |
| Preflight import discipline is SOURCE-level (AST), not runtime — sys.modules purity deferred to Phase 25 owner call (v1.4 Phase 23, CR-01 deferral) | `import dnallm.preflight` transitively pulls torch/transformers/datasets via `dnallm.utils`/`dnallm.models` package inits; fixing that means an unsanctioned root-facade refactor | ✓ AST structure test green; docstrings state the truth; TUI fast-start (Phase 25) owns the runtime-purity decision |
| One `validate_sections` path shared by `load_config` and `generate_config`/`edit_config` (v1.4 Phase 23, the one sanctioned refactor) | Two validators drift; the round-trip editor must revalidate through the exact path production loads with | ✓ Verbatim factor-out, equivalence-pinned; load_config raise-first contract unchanged |
| Probes never raise — every failure degrades to UNKNOWN with a machine-readable reason (v1.4 Phase 23) | A 429 masquerading as "unavailable" would wrongly steer model choice; unavailability is a state, not an exception | ✓ Pinned taxonomy tests (RATE_LIMITED/OFFLINE/PROBE_TIMEOUT/NOT_FOUND_ON_EITHER_HUB/PROBE_ERROR); severed-network integration test green |
| PEP 562 lazy facades cut `dnallm.preflight` import 3.71 s → 0.29 s (v1.4 Phase 25) | The TUI must start instantly on CPU-only biologist machines; the compat-shim ordering guarantee moved into every consumer module | ✓ Torch-free fast start pinned by structure tests; no import-order behavior change |
| `DownloadCancelled` is a BaseException sentinel (v1.4 Phase 25) | A regular Exception is swallowed by `download_model`'s retry/backoff loop; cooperative stop must escape ten attempts in one call | ✓ Proven by a dedicated sentinel test |
| Worker seam before panels; Pilot-functional tests only, no snapshot/golden TUI infra (v1.4 Phase 25) | Every blocking core call crossing a thread worker is a structural invariant, not a per-panel choice; snapshots would pin rendering over behavior | ✓ Six probe-red structure guards; offline Pilot E2E walk; every TUI module ≥97.4% coverage with no omit entries |
| MCP consistency checker warn-by-default, strict in CI over the committed configs (v1.4 Phase 26) | Committed-config drift is breakage (CI fatal); user-side configs get WARN-first honesty, never a write | ✓ 98% module coverage; gate test green on every lane via existing testpaths |
| `get_all_available_models` serves all 236 registry entries with configured/enabled/loaded overlay; UNKNOWN shown honestly (v1.4 Phase 26) | The menu must tell the truth about the registry (issue #64 registry half); unavailability is a state to display, not hide | ✓ Client surface pinned by 73 unedited SDK tests; unmatched-configured rows visible |
| Census probes execute each fetched snapshot dir as a synthetic package; census-driven rung scope — only MEASURED failure classes got compat rungs (v1.4 Phase 27) | HF snapshots are extensionless blob symlinks (no path resolution); D-06: the census is the truth, research-predicted failure classes never surfaced and got none | ✓ 49-entry census committed as evidence; two rungs (apply_chunking alias + load-boundary bookkeeping) restored 12 entries; misroutes recovered green |
| Dispositions consume the D-03 verdict, not the failure depth: shim-required-tf5 derives from verdict + rung dependency; remote-code-broken covers import-fail and structurally-failed skips (v1.4 Phase 27) | Literal disposition-type mapping would empty the shim-required bucket and mislabel PlantCaduceus_l32 as verified-tf5 — a false native-load claim | ✓ Registry write-back (49 rows × 3 compat fields) + independent consistency gate; zero unaccounted entries |
| Docs page is generator-only with pinned DOCS_DATA_DATE and byte-identical regeneration; failure rows lead with four honest per-class plain sentences, technical cause demoted to a ≤100-char parenthetical, 300-char cell cap (v1.4 Phase 27) | Wall-clock stamps break byte-identical regeneration; biologists must get an actionable verdict first (owner plain-language directive 2026-10-21); a version-independent model defect is never worded as a library removal | ✓ Drift gate re-derives independently (68 tests); UI re-audit 21/24 with prior warnings (a)/(c) closed; max cell 285 chars |
| Census lane installs mamba kernels only via the sanctioned sha256-pinned scripts/install_mamba.sh (pip bootstrap, post-checkout artifact dir); lane greenness proven by the runner-box replica run (v1.4 Phase 27) | The plain .[mamba] extra cannot build on the box (uv source build fails); actions/checkout clean removes pre-checkout dirs on the persistent workspace; mamba-family remote code imports mamba_ssm at probe time | ✓ Replica run 49 matched / exit 0 / 302 s; 13 offline contract pins (ordering, invocation, no masking, no continue-on-error, posture); one live dispatch owed post-merge |

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
*Last updated: 2026-10-28 after Phase 27 (Transformers 4/5 census — v1.4 line reopened post-archive, gap-closure cycle complete)*
