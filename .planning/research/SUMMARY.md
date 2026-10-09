# Project Research Summary

**Project:** DNALLM — v1.2 "Paper Revision Suite Support" (REV-01…REV-11)
**Domain:** Quality/feature milestone on an existing PyPI-published DNA-LM toolkit (pytest-hardened, 96%+ coverage, CI-gated)
**Researched:** 2026-10-09
**Confidence:** HIGH (stack claims verified by direct execution in the installed venv; architecture/pitfall claims verified against the working tree on branch `revision`)

## Executive Summary

This milestone adds eleven paper-revision-driven capabilities (REV-01…REV-11) to an existing, battle-tested codebase — it is integration work, not greenfield. The research converges on four load-bearing conclusions. First, **zero new dependencies**: every REV is satisfiable with the existing pins (peft >=0.14 already covers `IA3Config` by 10 minor versions; scipy >=1.15.2 covers `bootstrap` and `false_discovery_control`; the only apparent gaps — VCF and JASPAR/PWM parsing — resolve to ~100-line stdlib readers because cyvcf2/pysam have no Windows wheels and biopython is the wrong footprint). Second, **the REV-02 metric registry must NOT live at `dnallm/tasks/metrics/registry.py`** (the intake's sketch) — three researchers independently flagged that this directory is the vendored-code glob omitted from coverage, ruff, AND mypy, so the contract layer would ship invisible to every gate. Place it at `dnallm/tasks/metric_registry.py` beside `metrics.py`. Third, **REV-08 (zero-shot VEP) is the long pole** (~2 days), and its same-slot alignment rule (ref/alt must occupy the identical token slot, else explicit skip with reason + count) is the protocol answer to reviewer R1-3e(1) — it must start in Wave 1 despite being P1. Fourth, **the eval-semantics guard (REV-01) must cover its neighbors**, not just trainer.py:234-241: the early-stopping path at trainer.py:298-303 force-enables `load_best_model_at_end`, so the guard must set `eval_strategy="no"` AND `eval_dataset=None` atomically and raise a descriptive `ValueError` when user-set best-model-loading/early-stopping collides with a missing dev split.

The recommended structure is three sequential waves of 4–5 parallel agents (~2 days serial, ~1.5 days pipelined — matching the owner's 1–1.5-day target): **Wave/Phase A** (contract layer: REV-01/02/03 plus shared-file scaffolding and a REV-08 head start), **Wave/Phase B** (adaptation + evaluation: REV-04/05, 06, 07, 08-completion, 09 as five file-disjoint agents), **Wave/Phase C** (narrative: REV-10/11 + docs closeout). Both the architecture and pitfalls researchers independently produced conflict matrices; reconciled, they agree on one rule: **Phase A pre-creates all shared-file scaffolding (Pydantic config stubs, `__init__.py` re-export decision, pyproject changes) in ONE plan so Phase B agents only fill modules**, and `dnallm/__init__.py` gets NO new re-exports in v1.2 (import via full paths; keeps the facade byte-stable and conflict-free). Two hot files (`configs.py`, `trainer.py`) change ownership cleanly from Wave 1 (REV-01) to Wave 2 (REV-04/05, deliberately one agent because they share every hot file).

Key risks and mitigations: silent PEFT failures on Mamba/hybrid backbones (peft silently skips non-matching `target_modules` — presets must be derived from real `config.json` module names with a dry-run validator and a trainable-parameter-count runtime guard); `random_init` accidentally loading pretrained weights (use `from_config`, never `from_pretrained`; per-tensor hash comparison, not one global hash; CPU-canonical seeding); and the 90% gate NOT catching under-tested new modules (a 400-line `vep.py` with zero tests only drops ~96.4% → ~94.9% — the working standard is 96%+ per module, verified at phase verification, plus the owner's same-change-pytest rule).

## Key Findings

### Recommended Stack (STACK.md — HIGH, execution-verified)

**Zero new runtime and zero new dev/test dependencies.** Every feature rides existing pins plus the stdlib. The milestone's "addition" is a what-NOT-to-add list.

**Core technologies (all existing):**
- **peft** (`>=0.14.0` → 0.21.1 installed) — REV-04 IA3; `IA3Config` needs only >=0.4.0; end-to-end smoke-verified on a BERT-style transformer AND a mambapy Mamba backbone under transformers 5.17
- **scipy** (`>=1.15.2`) — REV-09 `stats.bootstrap` (percentile, seeded) and REV-10 `false_discovery_control` (BH) — both verified working
- **scikit-learn** — entire REV-07 probing component (LogisticRegression/MLPClassifier/StandardScaler)
- **transformers** (`>=4.49.0,<6`) — REV-06 via `AutoConfig.from_pretrained` + `AutoModel*.from_config` (stable across the span, exercised under 5.17)
- **mcp SDK** (`>=1.3.0,<2`) — REV-11 tools via the existing `app.tool()` + `_with_timeout_wrapper` pattern; **do NOT upgrade to 2.x in this milestone**
- **Stdlib additions:** ~100-line VCF reader (`gzip` + str-splitting; bgzipped ClinVar verified readable), ~50-line JASPAR fetch/parser (`urllib.request`; **host has moved to `jaspar.elixir.no`** — genereg.net 301s; prefer `format=meme`; parameterize base URL), `hashlib` param hashes

**Explicitly rejected:** cyvcf2/pysam (no Windows wheels — would break the Windows CI leg), biopython, statsmodels, mcp 2.x, any VEP framework, torchmetrics, and **`dnallm/tasks/metrics/registry.py` as a placement** (vendored-dir omit hazard).

### Expected Features (FEATURES.md)

**Must have (P0 / Phase A — gates the benchmark re-run):**
- REV-01: default `eval_strategy="no"` + `load_best_model_at_end=False` when dev absent; explicit `allow_test_as_eval=True` override with WARN; `evaluate(split=...)` via the predict path. Anti-feature: auto-creating a dev split.
- REV-02: `{canonical: (fn, aliases)}` registry + `resolve()` raising matchable ValueError; canonical = CURRENT spellings (`AUROC`, `AUPRC`, `spearmanr`, `pearsonr`); aliases recognize-only, never emitted; import-light (no torch/sklearn at module import) so dnallmmark CI can import it cheaply.
- REV-03: terminology sweep to "DNA large language models", `validate_sequences` comparability warning (+ dropped-row count log line), CHANGELOG-per-fix with commits.

**Must have (P1 / Phase B):**
- REV-05→REV-04 (that order): presets YAML verified from real module names with dry-run validator + explicit "unsupported" family entries; then IA3 as one branch beside LoRA sharing the PeftModel save/reload path; `use_ia3 x use_qlora` rejected at config validation.
- REV-06: `random_init=True` on generic Auto* families only (ValueError + allowlist for special families); loud log + parameter hash; per-tensor proof.
- REV-07: frozen backbone, fixed-hyperparameter LR/MLP probes, scaler fit on train only, layer selection exposed, npz cache keyed (model, dataset, layer, pooling).
- REV-08: `align_variant` same-slot rule + skip accounting; CLM delta-log-lik and MLM log-odds paradigms reusing `mutagenesis.py` kernels; `evaluate_vcf` + CLI; formulas written into docstrings (protocol declaration). **Skip fraction is itself a finding** — report it.
- REV-09: `run_seeds` dir protocol `{model}/{task}/seed_{s}/`; `aggregate_seeds` pure function (mean/sd/seeded-bootstrap-CI); n<10 guard (omit or t-interval, never vacuous bootstrap at n=3).

**Should have (P2 / Phase C):** REV-10 hotspot-window PWM scan with FIMO conventions (GC-matched background, p<1e-4 threshold, BH over the full test set, both strands, HBG1/BCL11A golden test); REV-11 three MCP tools (`ism_scan`, `hotspots`, `zero_shot_score`) with skip accounting in result JSON + the `--host/--port` CLI-precedence fix on BOTH sse and streamable-http paths.

**Defer:** GPN-style tokenizer-less models in VEP (extension point), indels/multi-allelic, significance-testing machinery, TF-MoDISco integration, genome-wide scanning, mcp 2.x, cross-repo version-locking mechanism (intake open question #3 — coordination, not this repo's code).

### Architecture Approach (ARCHITECTURE.md — HIGH, tree-verified)

Four integration shapes: behavioral guards in existing facades (REV-01), a new contract module beside existing dispatch (REV-02), new sibling modules composing existing kernels (REV-05/07/08/09/10 — all NEW FILES, which is what makes waves conflict-free), and config+branch extensions to PEFT/loading seams (REV-04/06). Nothing restructures a layer.

**Major components / placements:**
1. `dnallm/tasks/metric_registry.py` — sibling of `metrics.py`, NOT inside vendored `metrics/` (coverage-omit + ruff + mypy exclusion; verify with `coverage report | grep registry` after merge)
2. `dnallm/inference/vep.py` (REV-08), `probing.py` (REV-07), `dnallm/finetune/sweep.py` (REV-09), `dnallm/finetune/presets.py` (REV-05) — new files composing `mutagenesis` kernels, `get_embeddings`, `extra_args={"seed": s}` read-only
3. `dnallm/configuration/presets/lora_targets.yaml` — packaged data (importlib.resources) + `[tool.setuptools.package-data]` entry; NOT repo-root `configs/` (not in the wheel)
4. New `dnallm/interpret/` subpackage (REV-10); do NOT move `dnallm/inference/interpret.py`
5. `dnallm/models/model.py`: `random_init` implemented inside `_load_model_by_task_type` only; special-family handlers out of scope by design
6. No new `dnallm/__init__.py` re-exports in v1.2 (byte-stable facade; avoids the one file every agent would touch)
7. Test layering: **every new module reaches the coverage bar via mocked fast-lane tests** (`pytest -m "not slow" --cov` is the PR gate); slow tests are acceptance only

### Critical Pitfalls (PITFALLS.md — top items)

1. **REV-01 guard bypassed by its neighbors** (trainer.py:298-303 early stopping force-enables `load_best_model_at_end`; transformers silently defaults `metric_for_best_model="loss"`) — guard atomically sets strategy+dataset, raises descriptive ValueError on collision; the early-stopping neighbor test is the one most likely to be forgotten.
2. **Registry in the vendored dir ships unmeasured** — relocate + same-change coverage-row check; aliases are one-directional (recognize historical, emit canonical only) or drift recurs.
3. **Silent PEFT failures on Mamba/hybrid** (peft skips non-matching target_modules; peft #3554/#2556; IA3x4-bit raises version-dependently) — config-time rejection, per-family dry-run tests, trainable-count runtime guard, IA3-specific roundtrip test (peft #2429 corruption class).
4. **random_init partially loading pretrained weights / lying hash** — `from_config` only, per-tensor comparison, CPU-canonical seed, no-download assertion, tokenizer-loads-normally assertion; tied-weight/meta-tensor traps on 5.x.
5. **VEP slot misalignment fails silently with plausible AUROCs** — alignment asserted to differ in exactly ONE slot; skip-reason fixtures; paradigm guard (causal vs bidirectional); VCF coordinate fixtures; frozen score goldens. Near-random AUROC is an EXPECTED outcome of protocol mismatch — treat as protocol bug first.
6. **Parallel-agent collisions** (configs.py x4 REVs, `__init__.py` x5 modules, `expected_skips.yaml`, pyproject, test collection) — Phase A scaffolding pass + roadmap-level file-ownership map + same-change skip allowlisting.
7. **The 90% gate won't catch under-tested new modules** (~96.4% → ~94.9% stays green) — per-module 96%+ standard verified at phase verification; also: don't pin foreign exception strings (version-span rule); real-model tests are `slow` + models.lock rows from birth.

## Implications for Roadmap

Phases start at Phase 10. Three phases matching the owner-fixed full-scope strategy; each phase's plan encodes its wave's agent/file-ownership map.

### Phase 10 (Phase A): Contract layer + scaffolding (P0, ~0.5 day, 4 agents)
**Rationale:** REV-02 gates the entire benchmark re-run (F3) and REV-07/08 consume its registry; REV-01 prevents reproducing the leak in the re-run; the scaffolding pass is the single highest-leverage structural decision for parallel safety; REV-08's long pole starts here.
**Delivers:** REV-01 guard + `evaluate(split=)` + neighbor tests; REV-02 registry + metrics.py rewire + contract tests; REV-03 docs/terminology/CHANGELOG (IA3 chapter completes in Phase 12); shared-file stubs (Ia3Config/VepConfig/SweepConfig skeletons, pyproject package-data, any skip-allowlist entries); REV-08 core (`align_variant` + scoring kernels + unit tests).
**Avoids:** Pitfalls 1, 2, 8, 9.
**Agent ownership:** W1-A trainer.py+configs.py (REV-01); W1-B tasks/ (REV-02); W1-C docs (REV-03); W1-D new inference/vep.py (REV-08 core). Zero file overlap.

### Phase 11 (Phase B): Adaptation + evaluation (P1, ~1 day, 5 agents)
**Rationale:** Consumes Phase 10's registry and freed hot files; splits into file-disjoint agents per the reconciled conflict matrix.
**Delivers:** B1 = REV-04+REV-05 as ONE agent (they share configs.py/trainer.py/inference.py:111-131: Ia3Config, presets YAML + resolver, IA3 branch, config-time `use_ia3 x use_qlora` rejection); B2 = REV-06 (model.py sole owner); B3 = REV-07 probing (new file, registry read-only, cache keyed with layer+pooling, disjoint split asserts); B4 = REV-09 sweep (new file, pure aggregation first, seed-semantics documented, n-guard); B5 = REV-08 completion (`evaluate_vcf`, CLI, ClinVar slow test with typed network skips + models.lock rows).
**Avoids:** Pitfalls 3, 4, 5, 6, 7, 10, 13, 14, 15.
**Note:** the two B-waves in the researchers' matrices (B1–B4 vs W2-A..E) differ only in whether REV-08 completion rides Phase B or a Phase B.5 — either is safe; REV-08 solely owns cli.py, so no contention exists.

### Phase 12 (Phase C): Narrative surface + closeout (P2, ~0.5 day, 3 agents)
**Rationale:** REV-11's `zero_shot_score` wraps the Phase-11 vep module; REV-10 is freestanding and can be pulled forward into Phase 11 idle capacity if calendar pressure demands (touches nothing Phase B owns).
**Delivers:** C1 = REV-10 (`dnallm/interpret/` motifs, FIMO conventions, HBG1 golden test); C2 = REV-11 (3 MCP tools, timeout-wrapper structure test, event-loop liveness test, host/port fix on BOTH transports); C3 = integration closeout (IA3/LoRA docs chapter completing REV-03, CHANGELOG finalization, census-count re-pin if test counts hard-assert, coverage-expectation docs update).
**Avoids:** Pitfalls 11, 12, 13.

### Phase Ordering Rationale

- REV-02 before everything metric-emitting (REV-07/08/09) — landing them first means never re-touching outputs (intake Phase A gating, confirmed by FEATURES dependency graph).
- REV-05 before REV-04 (IA3 target defaults come from presets), and both in ONE agent because every hot file is shared.
- REV-11 after REV-08/10 — "a tool surface is a contract; wrap working code."
- Hot files change ownership cleanly across waves: configs.py/trainer.py W1→W2; the orchestrator commits between waves.
- Every wave: new skips allowlisted same-change, new models.lock rows, per-module coverage at the 96% working standard, no foreign-exception test matches.

### Research Flags

Phases likely needing deeper research (`/gsd-plan-phase --research-phase`):
- **Phase 11 / REV-08 lane:** highest flag weight — tokenizer-class coverage breadth (char/3-mer/6-mer/BPE alignment semantics), ClinVar filtering conventions for literature-comparable AUROCs, and the split-token alignment scheme details (Mut-BPE).
- **Phase 12 / REV-10:** empirical-null calibration choice (FIMO exact DP vs shuffle-null + BH) deserves a short plan-time spike; must be documented honestly either way.

Phases with standard patterns (skip research-phase):
- **Phase 10 (all of REV-01/02/03 + scaffolding):** verified anchors, house patterns, no unknowns.
- **Phase 11 REV-04/05/06/07/09 lanes:** official docs + execution-verified smoke tests + tree-verified seams cover them.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Headline claims verified by direct execution against the installed venv + live JASPAR API probes (host migration confirmed 2026-10-09) |
| Features | HIGH | PEFT IA3 and FIMO semantics from official docs (fetched directly); multi-source cross-checks elsewhere; LOW only on isolated illustrative numbers (not load-bearing) |
| Architecture | HIGH | Every integration point verified against the working tree on branch `revision` with line anchors |
| Pitfalls | HIGH | Repo-grounded facts verified in source; installed-library behavior (peft 0.21.x, transformers 5.x) source-verified; cross-version-span claims MEDIUM where only the installed version was inspected |

**Overall confidence:** HIGH

### Gaps to Address

- **Cross-repo version locking (intake open question #3):** how dnallmmark pins the dnallm registry/sweep contracts — coordination decision needed at requirements time; recommend dependency-light registry + dnallm-side public-surface test.
- **GPN-class tokenizer-less models in VEP (open question #4):** research says keep v1.2 tokenizer-based; extension point design deferred.
- **Probing output schema for F4:** emitted columns must be agreed when dnallmmark F4 is written — document in probing.py.
- **peft 0.14–0.16 floor behavior:** IA3x4-bit and no-match-raise behaviors verified on 0.21.x only; mitigated by config-time rejection + own-error-surface tests, not version pins.
- **REV-08 alignment under real BPE tokenizers beyond the smoke set:** covered by plan-time research flag, not by stack smoke tests.

## Sources

Aggregated from the four research files (full source lists there):

### Primary (HIGH confidence)
- Direct execution in installed venv (peft 0.21.1/0.21.2, transformers 5.17/5.19, scipy 1.18.1, torch 2.11): IA3Config smoke on transformer + mambapy, `from_config`, bootstrap/FDR calls
- Working tree on branch `revision` (line-anchored): trainer.py, metrics.py, configs.py, model.py, inference.py, mutagenesis.py, data.py, mcp/server.py, pyproject.toml, ci.yml
- Live JASPAR API probe (jaspar.elixir.no, `format=pfm|meme`), 2026-10-09
- Official docs: HF PEFT IA3, FIMO/MEME Suite, FastMCP tools, HF Trainer/training_args source

### Secondary (MEDIUM confidence)
- Mut-BPE, DART-Eval, GPN, BEND, NT, Evo2 calibration studies (zero-shot VEP protocol + alignment pitfall); Varoquaux & Colliot (multi-seed reporting); peft issues #1289/#2429/#2432/#2556/#3554; ssm-peft ICML25; pysam/cyvcf2 Windows-support findings; Geneformer silent-scratch incident

---
*Research completed: 2026-10-09*
*Ready for roadmap: yes*
