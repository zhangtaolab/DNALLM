# Roadmap: DNALLM

## Milestones

- ✅ **v1 Test Suite Audit & Coverage Hardening** — Phases 1–4 (shipped 2026-10-01) — [archive](milestones/v1-ROADMAP.md)
- ✅ **v1.1 Example Execution Testing & Repair** — Phases 5–9 (shipped 2026-10-07) — [archive](milestones/v1.1-ROADMAP.md)
- 🚧 **v1.2 Paper Revision Suite Support** — Phases 10–12 (in progress)

## Overview — v1.2 Paper Revision Suite Support

Eleven paper-revision capabilities (REV-01…REV-11) land on the hardened v1.1 codebase in three sequential waves of parallel implementation agents, per the owner-fixed execution strategy (full scope incl. P2 items, 4–5 agents per wave, ~1–1.5 days calendar). Phase 10 lands the contract layer that gates the dnallmmark re-run — the evaluation-semantics leak guard, the shared metric registry, the revision docs — plus the one-pass shared-file scaffolding that makes the later waves collision-free. Phase 11 delivers the adaptation and evaluation capabilities as five file-disjoint agents. Phase 12 completes the narrative surface (JASPAR motifs, MCP tools) and closes the milestone green.

Cross-cutting invariants for every phase: zero new dependencies beyond the single approved scikit-allel addition (owner decision 2026-10-09: VCF reading for VEP-01 — Windows cp310–313 wheels and numpy 1.26.4/2.2.0 compatibility both verified empirically before approval; only new required transitive dep is `dask[array]`; lands in Phase 11 agent B5); every new module reaches the ≥96% per-module coverage standard via mocked fast-lane tests (the 90% global gate alone would not catch an under-tested new module — verified at phase verification); `dnallm/__init__.py` gets no new re-exports (byte-stable facade); every new skip typed and allowlisted same-change; commits carry no Co-Authored-By trailers; wave parallelism runs as concurrent background executor agents in the single working tree (file-disjoint by design per each phase's ownership map) — no per-agent git worktrees and no GSD workstreams for v1.2 (owner decision 2026-10-09).

## Phases

<details>
<summary>✅ v1.1 Example Execution Testing & Repair (Phases 5–9) — SHIPPED 2026-10-07</summary>

- [x] Phase 5: Execution Harness, Honest Gates & Runner Feasibility (6/6 plans) — completed 2026-10-03
- [x] Phase 6: Model Registry & Showcase Data Curation (3/3 plans) — completed 2026-10-03
- [x] Phase 7: PlantHelixSeek Showcase Notebooks (2/2 plans) — completed 2026-10-04
- [x] Phase 8: Full Execution Rollout & Repair Loop (9/9 plans) — completed 2026-10-05
- [x] Phase 9: CI Wiring & Census Verification (4/4 plans) — completed 2026-10-06

Full phase details, requirements mapping, and success criteria: [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1 Test Suite Audit & Coverage Hardening (Phases 1–4) — SHIPPED 2026-10-01</summary>

- [x] Phase 1: Harness Integrity & Measured Baseline (2/2 plans) — completed 2026-09-30
- [x] Phase 2: Suite Hygiene & Known-Bug Fixes (3/3 plans) — completed 2026-09-30
- [x] Phase 3: Test Authoring to >90% Coverage (5/5 plans) — completed 2026-09-30
- [x] Phase 4: CI Gate Enforcement (3/3 plans) — completed 2026-10-01

Full phase details, requirements mapping, and success criteria: [milestones/v1-ROADMAP.md](milestones/v1-ROADMAP.md)

</details>

**Phase Numbering:**
- Integer phases (10, 11, 12): planned v1.2 milestone work (continues v1.1's Phase 9 — numbering never restarts)
- Decimal phases (10.1, 10.2): urgent insertions (marked INSERTED)

- [x] **Phase 10: Evaluation Contract Layer & Shared Scaffolding** - Test-as-eval leak guard, shared metric registry, revision docs, and the one-pass scaffolding (incl. the REV-08 core head start) that keeps the parallel waves collision-free (completed 2026-10-09)
- [x] **Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities** - IA³ + per-model PEFT presets, random-init baselines, frozen probing, zero-shot VEP completion, and multi-seed sweeps as five file-disjoint agents (completed 2026-10-09)
- [ ] **Phase 12: Motif Matching, MCP Tools & Milestone Closeout** - JASPAR/CIS-BP PWM scanning, three new MCP tools with the host/port CLI fix, the IA³ docs chapter, and milestone-wide green closeout

## Phase Details

**Milestone goal (v1.2):** ship the suite-side capabilities the paper revision requires so the dnallmmark full re-run and the reviewer-requested experiments (E1'–E8') can start.

### Phase 10: Evaluation Contract Layer & Shared Scaffolding

**Goal**: The evaluation-semantics contract that gates the benchmark re-run is in place — the trainer can never silently evaluate on the test split, every metric resolves through one shared registry, and the revision docs surface is honest — and all shared files are scaffolded in one pass so Phase 11's parallel agents only fill modules
**Depends on**: Nothing (first phase of v1.2 — milestone v1.1 shipped 2026-10-07)
**Requirements**: EVAL-01, METR-01, DOCS-01
**Success Criteria** (what must be TRUE):
  1. A training run with no dev split never evaluates on the test split — `eval_strategy="no"` and `eval_dataset=None` are set atomically with `load_best_model_at_end` defaulting to False; `allow_test_as_eval=True` opts in with a loud WARN; a collision with the early-stopping neighbor path (trainer.py:298-303) or user-set best-model loading raises a descriptive ValueError; unit tests cover dev+test / test-only / train-only × default/override plus the early-stopping collision case
  2. A user can explicitly evaluate any split via the new `evaluate(split="test"|"dev"|...)` entry point routing through the predict path, and the trainer docstring documents eval-set selection and held-out semantics
  3. `resolve(name)` on the new `dnallm/tasks/metric_registry.py` returns the canonical metric function for every metric key used across the benchmark task set; unknown names raise a matchable ValueError; historical aliases (`eval_auroc`, `eval_spearman_r`, …) are recognized but never emitted; `metrics.py` emits exclusively through the registry; the module imports without torch/sklearn and its coverage row is measured (same-change proof it sits outside the vendored `dnallm/tasks/metrics/` omit glob)
  4. The docs build stays green under the docs-validation gate with terminology unified to "DNA large language models", `validate_sequences` carrying a docstring plus the cross-model `valid_chars` comparability warning with a dropped-row count log line, and one CHANGELOG entry per revision fix each traceable to its commit (rebuttal-letter evidence chain opened; the IA³-chapter section completes in Phase 12 after PEFT-01)
  5. The scaffolding pass is committed as one change — Pydantic config stubs (Ia3Config/VepConfig/SweepConfig skeletons), pyproject package-data entries, any new skip-allowlist rows — and the REV-08 long pole has started (`dnallm/inference/vep.py` core: `align_variant` same-slot rule + CLM/MLM scoring kernels with unit tests); `pyproject.toml` dependency lists are unchanged (zero new dependencies) and `dnallm/__init__.py` carries no new re-exports

**Plans**: 4

Plans:
- [x] 10-01-trainer-eval-guard-scaffolding-PLAN.md — A1: EVAL-01 trainer guard + evaluate(split=) + one-pass config scaffolding (Ia3Config/VepConfig/SweepConfig, use_ia3, pyproject package-data)
- [x] 10-02-metric-registry-contract-PLAN.md — A2: METR-01 registry at dnallm/tasks/metric_registry.py + metrics.py exclusive emission + coverage/import-light proofs
- [x] 10-03-docs-terminology-changelog-PLAN.md — A3: DOCS-01 terminology sweep + validate_sequences comparability warning + LoRA/QLoRA/IA³ chapter + CHANGELOG evidence chain
- [x] 10-04-vep-core-kernels-PLAN.md — A4: REV-08 head start — vep.py core (align_variant same-slot rule + CLM/MLM kernels) at the ≥96% standard

**Wave structure (owner-fixed):** 4 parallel agents with zero file overlap — A1 trainer.py+configs.py (REV-01); A2 tasks/ registry (REV-02); A3 docs (REV-03); A4 new inference/vep.py core (REV-08 start). The plan encodes the per-agent file-ownership map (CHANGELOG.md is the single sanctioned shared append surface per the D-09 same-commit entry mechanism). Standard patterns only — no plan-time research needed.

### Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities

**Goal**: Every reviewer-experiment capability works end-to-end — users can fine-tune with IA³ or preset-directed LoRA targets, load from-scratch baselines, probe frozen embeddings, score variants zero-shot from VCF, and run multi-seed sweeps with uncertainty aggregates
**Depends on**: Phase 10 (metric registry consumed by probing/VEP/sweep metrics; hot files configs.py/trainer.py freed by Wave A; pre-created scaffolding)
**Requirements**: PEFT-01, PEFT-02, BASE-01, PROB-01, VEP-01, SEED-01
**Success Criteria** (what must be TRUE):
  1. A user can fine-tune with IA³ exactly as with LoRA — `Ia3Config` + `TrainingConfig.use_ia3`, one transformer-family AND one Mamba model each fine-tune one task, adapter save/reload roundtrips through the shared PeftModel path (peft #2429 corruption class covered), and `use_ia3 × use_qlora` / `lora × ia3` are rejected at Pydantic config time with matchable ValueErrors; `target_modules=None` auto-selects per-family presets from the packaged `dnallm/configuration/presets/lora_targets.yaml` (targets derived from real `config.json` module names, never guessed; ~44 benchmark models covered with presets-table regression tests), and a dry-run validator plus a runtime trainable-parameter-count guard catch silent peft module-skip on Mamba/hybrid backbones
  2. `load_model_and_tokenizer(..., random_init=True)` produces a genuinely from-scratch model — a loud "randomly initialized" log with per-tensor parameter-hash proof differing from the pretrained path on every tensor (not one global hash), same-seed reproducibility, a no-download assertion, the tokenizer loading normally, and an explicit ValueError from special-family handlers; proven on two architectures
  3. Any model × any binary classification task runs frozen-embedding probing end-to-end — `extract_embeddings` with selectable layer and pooling, `fit_probe(kind='logistic'|'mlp')` with fixed hyperparameters and the scaler fit on the train split only, probe metrics emitted through the Phase 10 registry, and embeddings cached to npz keyed by (model, dataset, layer, pooling) with second-run cache hits asserted; the output schema is documented for the dnallmmark F4 lane
  4. A user can score variants zero-shot from a VCF — `align_variant` enforces the same-slot rule (ref/alt must tokenize into the identical token slot, differing by exactly one, asserted in tests; otherwise the variant is explicitly skipped with reason + count and the skip fraction is itself reported as a finding), `score_variant(paradigm='clm'|'mlm')` reuses the mutagenesis.py kernels behind a paradigm↔architecture mismatch guard, `evaluate_vcf(...)` + CLI entry point yield per-variant scores with VCF coordinate-system fixtures and AUROC/AUPRC via the registry, and the ClinVar 1k-sample × ≥5-model (CLM/MLM mix) acceptance produces literature-magnitude AUROCs with the scoring formulas written into docstrings and README as the protocol declaration
  5. A user can run multi-seed sweeps — `run_seeds` writes the `{model}/{task}/seed_{s}/` directory protocol and `aggregate_seeds` as a pure function returns mean/sd/ci95 via a seeded percentile bootstrap with the n<10 guard (omit CI or t-interval — never a vacuous bootstrap at n=3); aggregation is proven against constructed known arrays and a ≥3-seed trial of one small task completes end-to-end with the result-JSON `statistics` block

**Plans**: 5

Plans:
- [x] 11-01-ia3-peft-presets-PLAN.md — B1: REV-04+REV-05 — IA³ trainer branch (interim warn demolished), cross-field rejections, packaged presets lora_targets.yaml, dry-run validator, trainable-ratio guard, transformer+mamba slow acceptance + IA³ roundtrip
- [x] 11-02-random-init-baselines-PLAN.md — B2: REV-06 — random_init via AutoConfig+from_config, banner + per-tensor hash proof, no-download/reproducibility proofs, RANDOM_INIT_SUPPORTED_FAMILIES allowlist, two-architecture slow acceptance
- [x] 11-03-frozen-embedding-probing-PLAN.md — B3: REV-07 — probing.py: extract_embeddings (layer/pooling) + npz cache keyed by 4-tuple + fit_probe (fixed constants, train-only scaler) + registry metrics + F4 schema
- [x] 11-04-multi-seed-sweep-PLAN.md — B4: REV-09 — sweep.py: aggregate_seeds (n-guarded t/bootstrap) + run_seeds directory protocol with D-16 same-split seed semantics + statistics block + ≥3-seed trial
- [x] 11-05-vep-evaluate-vcf-cli-PLAN.md — B5: REV-08 completion — scikit-allel dep + evaluate_vcf (D-17 ClinVar convention, uppercase windows, skip accounting) + score_variant paradigm guard + dnallm-vep CLI + README protocol + ClinVar 1k × ≥5-model slow acceptance

**Wave structure (owner-fixed):** 5 file-disjoint agents — B1 REV-04+REV-05 as ONE agent (shared hot files configs.py/trainer.py/inference.py:111-131; presets land before IA³ defaults); B2 REV-06 (model.py sole owner); B3 REV-07 probing (new file, registry read-only); B4 REV-09 sweep (new file, pure aggregation first); B5 REV-08 completion (evaluate_vcf, CLI, ClinVar slow tests with typed network skips + models.lock rows; sole owner of cli.py; adds the approved `scikit-allel>=1.3.13` dependency to pyproject and reads VCFs via `allel.read_vcf` incl. INFO parsing for ClinVar filtering — owner decision 2026-10-09, stdlib reader plan superseded). The plan encodes the per-agent file-ownership map. **Research flag:** the REV-08 lane carries the highest flag weight (tokenizer-class alignment semantics across char/k-mer/BPE, ClinVar filtering conventions, split-token alignment) — plan it with `/gsd-plan-phase --research-phase`; the REV-04/05/06/07/09 lanes follow standard patterns.

### Phase 12: Motif Matching, MCP Tools & Milestone Closeout

**Goal**: The narrative-facing surface is complete — motif hits reproduce the paper's annotation, LLM agents can drive ISM/hotspot/zero-shot scoring over MCP with honest CLI flags — and the milestone closes fully green with the docs chapter and CHANGELOG evidence chain finished
**Depends on**: Phase 11 (`zero_shot_score` wraps the completed vep.py; the IA³ docs chapter needs the landed PEFT-01)
**Requirements**: MOTIF-01, MCPE-01
**Success Criteria** (what must be TRUE):
  1. A user can scan hotspot windows against JASPAR/CIS-BP PWMs under FIMO conventions — GC-matched background, both strands, log-odds threshold p<1e-4, BH FDR via `scipy.stats.false_discovery_control` — and receive a motif-ID/coordinates/E-value table; the HBG1/BCL11A hit coordinates match the paper's Fig 4a annotation; the stdlib JASPAR client fetches from the parameterized canonical host (jaspar.elixir.no, preferring `format=meme`); the p-value calibration choice (empirical-null vs exact-DP) is documented honestly
  2. An MCP client connected to a live server can call the three new tools — `ism_scan`, `hotspots`, `zero_shot_score` — and receive valid JSON per the existing `_with_timeout_wrapper` + error-dict conventions (`zero_shot_score` wraps the Phase 11 VEP module with skip accounting); handshake regression tests assert server up → client calls all 3 tools → JSON
  3. The `--host/--port` CLI flags take precedence over yaml config on BOTH sse and streamable-http paths (the v1.1-audit deferred bug), proven by CLI-precedence tests on each transport
  4. The milestone closes green — the IA³/LoRA/QLoRA usage chapter completes the DOCS-01 docs (finalizing the one-entry-per-fix CHANGELOG evidence chain), the fast lane is fully passing with every new skip typed and allowlisted same-change, the coverage gate is green with every new module at the ≥96% per-module standard, docs-validation is green, and zero new dependencies landed across the milestone beyond the approved scikit-allel addition (Phase 11, owner decision 2026-10-09)

**Plans**: 3

Plans:
**Wave 1**
- [x] 12-01-motif-matching-fimo-scanner-PLAN.md — C1: MOTIF-01 — dnallm/interpret/motifs.py FIMO-convention scanner (exact-DP calibration per D-01/D-02, GC background, both strands, BH full-set per D-03), stdlib JASPAR client + CIS-BP local parse, owner-gated HBG1/BCL11A golden harness
- [x] 12-02-mcp-tools-host-port-fix-PLAN.md — C2: MCPE-01 — ism_scan/hotspots/zero_shot_score MCP tools (D-04/D-05/D-06, skip accounting verbatim) + the --host/--port CLI-precedence fix on both transports with flipped tests

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 12-03-milestone-closeout-docs-changelog-PLAN.md — C3: DOCS-01 completion — IA³ chapter (D-08), measured coverage-expectation docs (D-07), CHANGELOG REV-01..11 SHA backfill, census verify

**Wave structure (owner-fixed):** 3 agents — Wave 1: C1 (`dnallm/interpret/` + `tests/interpret/`) and C2 (`dnallm/mcp/server.py` + `tests/mcp/`) run file-disjoint in parallel (CHANGELOG.md is the single sanctioned shared append surface via the D-09 same-commit mechanism, declared coupling_justified in both plans); Wave 2: C3 integration closeout (needs C1/C2 commits for SHA backfill and their modules for the measured coverage number). **Research flag (resolved):** the REV-10 calibration spike is closed — D-01 exact-DP per FIMO/MEME source, transcribed in 12-RESEARCH.md Pattern 1; documented honestly in the module docstring per REQUIREMENTS.

## Progress

**Execution Order:**
Phases execute in numeric order: 10 → 11 → 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Evaluation Contract Layer & Shared Scaffolding | v1.2 | 4/4 | Complete    | 2026-10-09 |
| 11. PEFT Adaptation, Baselines & New Evaluation Capabilities | v1.2 | 5/5 | Complete    | 2026-10-09 |
| 12. Motif Matching, MCP Tools & Milestone Closeout | v1.2 | 2/3 | In planning | - |

---

*Roadmap created 2026-10-09 for milestone v1.2 (Paper Revision Suite Support). Previous milestones archived under `milestones/`.*
