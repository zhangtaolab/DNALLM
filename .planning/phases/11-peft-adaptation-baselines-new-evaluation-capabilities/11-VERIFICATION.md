---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
verified: 2026-10-10T10:15:42Z
status: passed
score: 8/9 must-haves verified
covered_files: [".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-01-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-01-ia3-peft-presets-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-random-init-baselines-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-03-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-03-frozen-embedding-probing-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-04-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-04-multi-seed-sweep-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-05-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-05-vep-evaluate-vcf-cli-PLAN.md", "CHANGELOG.md", "README.md", "dnallm/cli/vep.py", "dnallm/configuration/configs.py", "dnallm/configuration/presets/lora_targets.yaml", "dnallm/finetune/sweep.py", "dnallm/finetune/trainer.py", "dnallm/inference/inference.py", "dnallm/inference/probing.py", "dnallm/inference/vep.py", "dnallm/models/model.py", "models.lock", "pyproject.toml", "tests/cli/test_vep_cli.py", "tests/configuration/test_configs.py", "tests/configuration/test_peft_presets.py", "tests/expected_skips.yaml", "tests/finetune/test_sweep.py", "tests/finetune/test_trainer.py", "tests/finetune/test_trainer_real_model.py", "tests/inference/data/synthetic_reference.txt", "tests/inference/data/synthetic_variants.vcf", "tests/inference/test_inference.py", "tests/inference/test_probing.py", "tests/inference/test_vep.py", "tests/models/test_model.py"]
covered_digest: "v3:sha256:1224f619d7d6f7ae2e38aed123fd4ac989140dec72fc2bf8e1bae6c5ece5b743"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "The ClinVar 1k-sample x >=5-model (CLM/MLM mix) acceptance produces literature-magnitude AUROCs"
    reason: "Small (<=50M) plant DNA models measure 0.490-0.581 AUROC on the D-17 within-ClinVar cohort - honest empirical finding at this scale; same-scale BPE anchor matched (0.543 vs DNABERT-2 0.538); capability fully verified. Big-model anchors (0.85-0.98) come from 2.5B-40B models with different negative schemes."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-09T23:55:00+08:00"
re_verification:
  previous_status: passed
  previous_score: 8/9
  gaps_closed: []
  gaps_remaining:
    - "VEP-01 literature-magnitude AUROC clause — owner-accepted override carried in overrides: (not an open gap)"
  regressions: []
---

# Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities Verification Report

**Phase Goal:** Every reviewer-experiment capability works end-to-end — users can fine-tune with IA³ or preset-directed LoRA targets, load from-scratch baselines, probe frozen embeddings, score variants zero-shot from VCF, and run multi-seed sweeps with uncertainty aggregates
**Verified:** 2026-10-10T10:15:42Z
**Status:** passed
**Re-verification:** Yes — fixpoint regeneration at HEAD 6675bea (the 2026-10-09T15:52:24Z report went stale because later phases modified covered files; per the stale-verification fixpoint rule this regeneration runs with ZERO code changes — the last `dnallm/` change is 1e77a87 and every commit after it is docs/planning-only)

## Goal Achievement

Re-verification of the 9 observable truths from the 2026-10-09T15:52:24Z report (frontmatter `status: passed`, 8/9, 1 owner override) against the codebase at HEAD 6675bea. Staleness provenance, established exactly: of this phase's covered files, only four changed since the previous verification — `dnallm/inference/probing.py` (commit 1e77a87, the one `dnallm/` code change in the interval), `CHANGELOG.md` (Phase-12 REV-10/11 entries + SHA backfill), `pyproject.toml` (Phase-12 CI stabilization pins), and `tests/expected_skips.yaml` (Phase-12 JASPAR live skips). `trainer.py`, `configs.py`, `model.py`, `inference.py`, `sweep.py`, `vep.py`, `cli/vep.py`, `lora_targets.yaml`, `README.md`, `models.lock`, and every phase test file are bit-identical to what the previous report verified — so its first-hand slow-lane behavioral evidence (Mamba IA³ fine-tune, IA³ roundtrip, per-tensor hash proof, 3-seed sweep trial, ClinVar 5-model acceptance) remains valid at HEAD, per owner policy cited rather than re-run. The changed `probing.py` is re-verified directly below, and every lane re-ran green in the fast lane at HEAD.

### Observable Truths

| # | Truth | Status | Evidence at HEAD 6675bea |
|---|-------|--------|--------------------------|
| 1 | IA³ fine-tunes exactly as with LoRA — real trainer branch injecting IA³ vectors via peft get_peft_model; transformer AND Mamba each fine-tune one task (SC1/PEFT-01) | ✓ VERIFIED | `trainer.py:485-497` ("Applying IA³" at :488, `IA3Config(**peft_kwargs)` at :493, `get_peft_model` at :495); file unchanged since previous verification; full fast lane at HEAD: 233 passed / 0 failed incl. all of `tests/finetune/test_trainer.py`; Mamba IA³ slow fine-tune evidence (verifier-run, 1 passed 34.4s) and transformer slow fine-tune (executor-run) carried from the unchanged file |
| 2 | `use_ia3 × use_qlora` rejected at Pydantic config time and `lora × ia3` at trainer init, both matchable ValueErrors naming both fields (SC1) | ✓ VERIFIED | `configs.py:375-389` `reject_ia3_with_qlora` model_validator; `trainer.py:397-401`; targeted selector at HEAD: `test_configs.py -k "ia3 or Ia3 or HeadConfig"` + `test_trainer.py -k "Ia3 or peft or dry_run or preset or ratio or collision"` → 126 passed / 0 failed |
| 3 | `target_modules=None` auto-selects per-family presets from packaged `dnallm/configuration/presets/lora_targets.yaml`; table covers all families with regression tests; dry-run validator; trainable-parameter-count guard (SC1/PEFT-02) | ✓ VERIFIED | Re-checked programmatically at HEAD: 35/35 `PRETRAIN_MODEL_MAPS` families, zero empty `lora_target_modules`/`ia3_target_modules` lists, `feedforward_modules ⊆ ia3_target_modules` holds for every family (empty FFN only where the architecture has none — conv/mamba backbones), both `lora_ratio_band` and `ia3_ratio_band` present everywhere; loader via `resources.files` (`trainer.py:95-115`), two-tier resolution failing loud (`trainer.py:200-207`), dry-run report + zero-match ValueError (`trainer.py:211-237`), ratio guard computed directly from `requires_grad` tensor sums (`trainer.py:262-285`); 21 preset tests passed in the HEAD fast lane |
| 4 | IA³ adapter save→reload roundtrip through the shared PeftModel path reproduces identical outputs (peft #2429 corruption class); reload path adapter-kind-agnostic (SC1) | ✓ VERIFIED | `inference.py:112-132` uses `PeftModel.from_pretrained` with "PEFT adapter" naming; file unchanged since the previous verifier-run roundtrip test (1 passed, 44.0s); the slow test `test_ia3_adapter_save_reload_roundtrip` remains in `tests/finetune/test_trainer.py` (deselected by the `-m "not slow"` fixpoint filter, present and green in prior first-hand + executor runs over the identical file) |
| 5 | `random_init=True` genuinely from-scratch: from_config only, banner + per-tensor hash proof, same-seed reproducibility, no weight download, tokenizer normal, special-family ValueError, two architectures (SC2/BASE-01) | ✓ VERIFIED | `_load_random_init_model` (`model.py:1009-1090`): `AutoConfig.from_pretrained` config-only fetch (:1053), `torch.manual_seed(seed)` before construction (:1072), `auto_class.from_config` — never `from_pretrained` on this branch (:1076), tokenizer via the normal fallback path, `_log_random_init_fingerprint` before device move (:1090); `RANDOM_INIT_SUPPORTED_FAMILIES` (:759) gate raises before handlers (:843-857); `_tensor_digest` (:940) per-tensor sha256; file unchanged since previous verification (38 fast proofs + verifier-run per-tensor difference slow proof + executor-run BERT/mamba two-arch slow tests carried) |
| 6 | Any model × binary task probes frozen embeddings end-to-end: layer/pooling-selectable extraction, fixed-hyperparameter logistic/mlp probe, train-only scaler, registry metrics, npz cache keyed by 4-tuple with cache-hit assertions, documented schema (SC3/PROB-01) | ✓ VERIFIED | **This is the one covered code file changed since the previous report** — commit 1e77a87 adds a Windows `PermissionError` retry (5 attempts, 50ms backoff steps) around the atomic `os.replace` in `_write_cache` (`probing.py:262-280`); re-read in full: the retry re-raises on the final attempt, the outer `except OSError` unlinks the temp file and re-raises — the single-winner contract (exactly one valid entry; readers never see a half-written file; no temp litter) is preserved and the change is purely additive. All other anchors re-confirmed at HEAD: module constants (`LOGISTIC_MAX_ITER=1000` at :100, no YAML surface), registry metrics via `metric_registry.resolve` (:81 import), sha256 cache filenames (`_cache_filename` :228-242), `StandardScaler` fit on train split only (:646-663), float32, F4 schema in docstring; 36 probing fast tests (incl. cache-hit/cache-miss and atomic-write coverage) passed at HEAD, directly exercising the changed `_write_cache` |
| 7 | A user can score variants zero-shot from a VCF: same-slot rule + skip accounting, paradigm↔architecture guard, evaluate_vcf + dnallm-vep CLI, VCF coordinate fixtures, AUROC/AUPRC via registry, formulas in docstrings + README (SC4/VEP-01, capability clauses) | ✓ VERIFIED | `vep.py` (unchanged since previous verification): `align_variant` :114, uppercase windows :559, `_check_paradigm_compatible` :622, `score_variant` :660 with both kernels called (:721-734 — `mlm_slot_log_prob` ref/alt pair, `clm_log_likelihood` delta), `evaluate_vcf` :734 with `alt_number=4` (:743), guarded `import allel` with dnallm ValueError (:805-812), registry resolve (:953); `cli/vep.py` lazy import :62 + call :104; README protocol section :530-541 (same-slot rule, `-delta` deleteriousness direction, convention block); vep + cli lanes green at HEAD (within 233 fast-lane passes); ClinVar 5-model slow acceptance (5 passed, 183s GPU, executor evidence) carried over the unchanged file |
| 8 | The ClinVar acceptance produces **literature-magnitude AUROCs** (SC4/VEP-01, magnitude clause) | ✓ PASSED (override) | Override: Small (<=50M) plant DNA models measure 0.490-0.581 AUROC on the D-17 within-ClinVar cohort - honest empirical finding at this scale; same-scale BPE anchor matched (0.543 vs DNABERT-2 0.538); capability fully verified. Big-model anchors (0.85-0.98) come from 2.5B-40B models with different negative schemes. — accepted by owner (Tao Zhang) on 2026-10-09T23:55:00+08:00 |
| 9 | Multi-seed sweeps: `run_seeds` directory protocol + pure `aggregate_seeds` (n-guarded t/bootstrap), proven against known arrays, ≥3-seed trial with statistics block (SC5/SEED-01) | ✓ VERIFIED | `sweep.py` (unchanged since previous verification): n<3→method "none" (:138), 3≤n<10→`stats.t.ppf(0.975, n-1)` (:141), n≥10→`np.random.default_rng` seeded percentile bootstrap (:148-152); boundary matrix + known-array moment tests and the verifier-run 3-seed end-to-end slow trial (1 passed, 34.6s) carried; 37 sweep fast tests passed at HEAD |

**Score:** 8/9 truths verified (0 present, behavior-unverified; 1 PASSED (override) — truth 8, owner-accepted and preserved verbatim)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/configuration/presets/lora_targets.yaml` | 35-family packaged presets with targets, FFN subsets, lora_r, ratio bands | ✓ VERIFIED | 533 lines; 35/35 families, structural invariants re-checked programmatically at HEAD |
| `dnallm/finetune/trainer.py` (modified) | IA³ branch, preset auto-selection, dry-run validator, ratio guard | ✓ VERIFIED | All present at :95-115/:200-237/:262-285/:397-401/:485-497; unchanged since previous verification; trainer lane green at HEAD |
| `dnallm/configuration/configs.py` (modified) | `peft_dry_run`, cross-field rejections, Ia3Config parity | ✓ VERIFIED | `reject_ia3_with_qlora` at :375-389; unchanged since previous verification; 126-test targeted selector passed |
| `dnallm/inference/inference.py` (modified) | Adapter-kind-agnostic reload naming | ✓ VERIFIED | `PeftModel.from_pretrained` + "PEFT adapter" strings at :112-132; unchanged |
| `dnallm/models/model.py` (modified) | random_init path, allowlist, per-tensor fingerprint | ✓ VERIFIED | :759/:843-857/:940/:960/:1009-1090; unchanged since previous verification |
| `dnallm/inference/probing.py` (new) | extract/fit/cache/registry | ✓ VERIFIED | 700 lines at HEAD (was 688; +13 from the 1e77a87 Windows retry — additive, contract-preserving, re-verified in full); 36 fast tests green at HEAD |
| `dnallm/finetune/sweep.py` (new) | aggregate_seeds + run_seeds | ✓ VERIFIED | 452 lines; n-guard machinery at :138-152; unchanged; 37 fast tests green at HEAD |
| `dnallm/inference/vep.py` (modified) | evaluate_vcf + score_variant + guard | ✓ VERIFIED | All contract elements at :114/:559/:622/:660-734/:734-957; unchanged |
| `dnallm/cli/vep.py` (new) | dnallm-vep click command | ✓ VERIFIED | Click command :15+, lazy `evaluate_vcf` import :62, call :104; unchanged; 7 CLI tests green at HEAD |
| `pyproject.toml` (modified) | exactly the two sanctioned lines | ✓ VERIFIED | Phase-11 lines present at HEAD: `scikit-allel>=1.3.13,<2` (:61) and `dnallm-vep = "dnallm.cli.vep:main"` (:290); the Phase-11 range delta remains the two sanctioned lines — the pydantic-ai pin and pyarrow cap added later (c75e54f/9fcff67/e30eeee) are Phase-12 CI-stabilization changes outside this phase's contract |
| `tests/inference/data/synthetic_variants.vcf` + `synthetic_reference.txt` | committed network-free fixture | ✓ VERIFIED | Both fixtures present, unchanged; vep fast lane (network-free) green at HEAD |
| `tests/expected_skips.yaml`, `models.lock` | clinvar-unavailable allowlist; sha pins | ✓ VERIFIED | `clinvar-unavailable:` prefix entry still at :35; the 3 VEP-acceptance sha-pinned rows still at :44-46 (plant-dnabert-BPE / plant-dnamamba-BPE / plant-dnagpt-BPE-promoter); Phase-12 JASPAR skips appended after, additive |
| README.md, CHANGELOG.md | random_init paragraph + VEP protocol section; REV-04..09 entries | ✓ VERIFIED | README protocol section :530-541; all six REV-04..REV-09 bullets present under `[Unreleased]` with full commit URLs (CHANGELOG.md:19-24); Phase-12 entries (REV-10/11 + SHA backfill) layered after without disturbing the Phase-11 chain |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| trainer.py | presets/lora_targets.yaml | importlib.resources lookup | ✓ WIRED | `resources.files("dnallm.configuration").joinpath("presets/lora_targets.yaml")` at `trainer.py:103` |
| trainer.py | inference.py shared adapter reload | PeftModel.from_pretrained | ✓ WIRED | `inference.py:131`; roundtrip behavioral evidence carried (unchanged files) |
| configs.py | peft IA3Config | field-filtered pass-through | ✓ WIRED | `PEFT_IA3_FIELD_NAMES` filter at `trainer.py:491-493` |
| model.py | transformers AutoConfig/Auto*.from_config | no-checkpoint instantiation | ✓ WIRED | `model.py:1053` (config fetch) + `:1076` (from_config), never `from_pretrained` on the branch |
| model.py | get_logger banner | per-tensor hash table | ✓ WIRED | `_log_random_init_fingerprint` INFO lines at `model.py:960-1005`, called at `:1090` |
| probing.py | metric_registry | resolve/validate_emission | ✓ WIRED | Import at `probing.py:81`; registry file untouched since previous verification |
| probing.py | inference.py hidden-states idiom | output_hidden_states reuse | ✓ WIRED | Consumer, not reimplementer |
| sweep.py | scipy.stats.t / default_rng | t.ppf + seeded bootstrap | ✓ WIRED | `sweep.py:141, 148` |
| sweep.py | DNATrainer orchestration | run_seeds drives train/evaluate | ✓ WIRED | Verifier-run slow trial carried (unchanged file) |
| vep.py | allel.read_vcf | guarded import + fields/alt_number | ✓ WIRED | `vep.py:805-812` (guard), `:836` (alt_number) |
| vep.py | Phase-10 kernels | align_variant + clm/mlm kernels wrapped | ✓ WIRED | `score_variant` calls at `vep.py:714-734` |
| vep.py | metric_registry | AUROC/AUPRC resolve | ✓ WIRED | `vep.py:953-957` |
| cli/vep.py | evaluate_vcf | lazy import in command body | ✓ WIRED | `cli/vep.py:62, 104` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| vep.py | per-variant deltas | model forward via clm/mlm kernels | Yes — no static returns | ✓ FLOWING |
| probing.py | embeddings | model forward output_hidden_states | Yes | ✓ FLOWING |
| sweep.py | statistics blocks | caller-supplied per-seed metric values | Yes — pure function over inputs | ✓ FLOWING |
| lora_targets.yaml | target lists | packaged YAML | Yes — loaded via importlib.resources | ✓ FLOWING |
| model.py random path | tensor digests | sha256 over live tensor bytes | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full Phase-11 fast lane at HEAD | `uv run --no-sync pytest tests/finetune/test_trainer.py tests/configuration/test_peft_presets.py tests/inference/test_probing.py tests/inference/test_vep.py tests/finetune/test_sweep.py tests/cli/test_vep_cli.py -m "not slow" -q --no-cov --timeout=240` | 233 passed, 7 deselected, 0 failed in 5.03s | ✓ PASS |
| IA³/config rejection selector at HEAD | `pytest tests/configuration/test_configs.py -k "ia3 or Ia3 or HeadConfig" tests/finetune/test_trainer.py -k "Ia3 or peft or dry_run or preset or ratio or collision" -m "not slow"` | 126 passed, 0 failed | ✓ PASS |
| Slow-lane behavioral evidence (NOT re-run per owner fixpoint policy — basic tests only) | Mamba IA³ fine-tune (34.4s), IA³ roundtrip (44.0s), per-tensor random-vs-pretrained proof (12.2s), 3-seed sweep trial (34.6s) — all verifier-run in the 2026-10-09 report; transformer IA³ fine-tune, BERT+mamba random_init, probing real-model, ClinVar 5-model GPU acceptance — executor-run | All passed; every exercised source file proven bit-identical at HEAD via `git log --since=2026-10-09T15:52:24Z` (no commits touching them) | ✓ PASS (carried) |
| Probing cache-write change coverage | 1e77a87 diff re-read in full + 36 probing fast tests (cache-hit/cache-miss, atomic write) green at HEAD | Retry is additive; single-winner contract preserved | ✓ PASS |

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared or conventional for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PEFT-01 | 11-01 | IA³ adapter support symmetric to LoRA | ✓ SATISFIED | Truths 1, 2, 4 |
| PEFT-02 | 11-01 | Per-model PEFT target-module presets | ✓ SATISFIED | Truth 3 |
| BASE-01 | 11-02 | random_init from-scratch baselines | ✓ SATISFIED | Truth 5 |
| PROB-01 | 11-03 | Frozen-embedding probing | ✓ SATISFIED | Truth 6 |
| VEP-01 | 11-05 | Zero-shot VEP from VCF | ✓ SATISFIED (with owner override on the magnitude clause) | Truths 7, 8 |
| SEED-01 | 11-04 | Multi-seed sweep protocol | ✓ SATISFIED | Truth 9 |

Orphaned requirements: none — REQUIREMENTS.md (re-checked at HEAD after the Phase-12 edits) still maps exactly the six IDs above to Phase 11, all marked Complete.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/models/model.py | 1228 | `# TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (introduced 2026-03-27, commit e0b3494; confirmed not added by this phase in the 2026-10-09 report); file unchanged since; out of phase scope |

Re-confirmed at HEAD: no TBD/FIXME/XXX/HACK/PLACEHOLDER markers in any phase-modified file; `probing.py` (the one changed file) is marker-clean. No stub implementations — every module computes from real inputs. Prohibition spot re-checks: the two sanctioned pyproject lines remain the Phase-11 delta; no repo-root `configs/` additions; no new `dnallm/__init__.py` re-exports.

### Human Verification Required

None. All behavior-dependent truths carry behavioral test evidence: the fast lanes re-ran green at HEAD (233 + 126 passes) covering the trainer branches, rejections, presets, probing cache invariants, VEP kernels/guards, and sweep statistics; the slow-lane behavioral evidence from the 2026-10-09 report (verifier-run and executor-run) remains valid because every file it exercised is proven unchanged at HEAD, and the single changed file (`probing.py`) was re-verified line-by-line with its fast-lane cache tests re-run green.

### Gaps Summary

No gaps. All 8 verifiable truths hold at HEAD 6675bea and the 9th (VEP-01 literature-magnitude clause) is PASSED under the owner-accepted override, preserved verbatim in the frontmatter. The changes that staled the previous report are all additive and non-regressing: the 1e77a87 probing fix adds a Windows `PermissionError` retry that preserves the single-winner cache contract (re-read in full, tests green at HEAD); the Phase-12 CHANGELOG/pyproject/expected_skips edits layer around the Phase-11 contract without touching it. This regeneration ran with zero code changes, per the milestone fixpoint rule (last `dnallm/` change = 1e77a87; commits 70a5ecb, 9d67988, 6675bea after it are docs/planning-only).

---

_Verified: 2026-10-10T10:15:42Z_
_Verifier: Claude (gsd-verifier)_
