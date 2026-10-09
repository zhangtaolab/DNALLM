---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
verified: 2026-10-09T15:52:24Z
status: passed
score: 8/9 must-haves verified
covered_files: [".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-01-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-01-ia3-peft-presets-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-02-random-init-baselines-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-03-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-03-frozen-embedding-probing-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-04-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-04-multi-seed-sweep-PLAN.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-05-SUMMARY.md", ".planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-05-vep-evaluate-vcf-cli-PLAN.md", "CHANGELOG.md", "README.md", "dnallm/cli/vep.py", "dnallm/configuration/configs.py", "dnallm/configuration/presets/lora_targets.yaml", "dnallm/finetune/sweep.py", "dnallm/finetune/trainer.py", "dnallm/inference/inference.py", "dnallm/inference/probing.py", "dnallm/inference/vep.py", "dnallm/models/model.py", "models.lock", "pyproject.toml", "tests/cli/test_vep_cli.py", "tests/configuration/test_configs.py", "tests/configuration/test_peft_presets.py", "tests/expected_skips.yaml", "tests/finetune/test_sweep.py", "tests/finetune/test_trainer.py", "tests/finetune/test_trainer_real_model.py", "tests/inference/data/synthetic_reference.txt", "tests/inference/data/synthetic_variants.vcf", "tests/inference/test_inference.py", "tests/inference/test_probing.py", "tests/inference/test_vep.py", "tests/models/test_model.py"]
covered_digest: "v3:sha256:93918aaa60514e4202cd3cda8ff02f519d6be65b11c05660fc418df549ba28ff"
behavior_unverified: 0
overrides_applied: 1
overrides:
  - must_have: "The ClinVar 1k-sample x >=5-model (CLM/MLM mix) acceptance produces literature-magnitude AUROCs"
    reason: "Small (<=50M) plant DNA models measure 0.490-0.581 AUROC on the D-17 within-ClinVar cohort - honest empirical finding at this scale; same-scale BPE anchor matched (0.543 vs DNABERT-2 0.538); capability fully verified. Big-model anchors (0.85-0.98) come from 2.5B-40B models with different negative schemes."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-09T23:55:00+08:00"
---

# Phase 11: PEFT Adaptation, Baselines & New Evaluation Capabilities Verification Report

**Phase Goal:** Every reviewer-experiment capability works end-to-end — users can fine-tune with IA³ or preset-directed LoRA targets, load from-scratch baselines, probe frozen embeddings, score variants zero-shot from VCF, and run multi-seed sweeps with uncertainty aggregates
**Verified:** 2026-10-09T15:52:24Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

Must-haves merged from the 5 ROADMAP Success Criteria (the contract) with PLAN frontmatter detail. All 6 requirement IDs (PEFT-01, PEFT-02, BASE-01, PROB-01, VEP-01, SEED-01) are claimed by plans; none orphaned (REQUIREMENTS.md maps exactly these six to Phase 11).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | IA³ fine-tunes exactly as with LoRA — real trainer branch injecting IA³ vectors via peft get_peft_model; transformer AND Mamba each fine-tune one task (SC1/PEFT-01) | ✓ VERIFIED | `trainer.py:488-493` ("Applying IA³", `IA3Config(**peft_kwargs)`); 14 IA³ wiring/preset trainer tests + 11 config tests passed first-hand; **Mamba IA³ slow fine-tune run by verifier: 1 passed (34.4s)** — the silent-skip proving ground with in-band ratio; transformer slow fine-tune executor-run (documented, same shared injection code path) |
| 2 | `use_ia3 × use_qlora` rejected at Pydantic config time and `lora × ia3` at trainer init, both matchable ValueErrors naming both fields (SC1) | ✓ VERIFIED | `configs.py:382-386` model_validator; `trainer.py:397-401`; rejection tests passed first-hand in `test_configs.py` (11) and `test_trainer.py` (14) |
| 3 | `target_modules=None` auto-selects per-family presets from packaged `dnallm/configuration/presets/lora_targets.yaml`; table covers all families with regression tests; dry-run validator; trainable-parameter-count guard (SC1/PEFT-02) | ✓ VERIFIED | YAML has 35/35 `PRETRAIN_MODEL_MAPS` families, zero empty target lists, FFN⊆IA³ subset holds (verifier-checked programmatically); loader via `resources.files` (`trainer.py:103`), two-tier resolution failing loud (`trainer.py:204-207`); dry-run report + zero-match ValueError (`trainer.py:211-237`); ratio guard computed directly from `requires_grad` tensors (`trainer.py:270`); 21 preset tests passed first-hand |
| 4 | IA³ adapter save→reload roundtrip through the shared PeftModel path reproduces identical outputs (peft #2429 corruption class); reload path adapter-kind-agnostic (SC1) | ✓ VERIFIED | `test_ia3_adapter_save_reload_roundtrip` **run by verifier: 1 passed (44.0s)**; `inference.py:111-131` uses `PeftModel.from_pretrained` with "PEFT adapter" naming |
| 5 | `random_init=True` genuinely from-scratch: from_config only, banner + per-tensor hash proof, same-seed reproducibility, no weight download, tokenizer normal, special-family ValueError, two architectures (SC2/BASE-01) | ✓ VERIFIED | `_load_random_init_model` (`model.py:1009-1090`): `AutoConfig.from_pretrained` + `Auto*.from_config` only, `torch.manual_seed` before construction, CPU through hashing; `RANDOM_INIT_SUPPORTED_FAMILIES` gate raises before handlers (`model.py:832-869`); `_tensor_digest`/`_log_random_init_fingerprint` per-tensor table; 38 fast proofs passed first-hand; **per-tensor difference-vs-pretrained slow proof run by verifier: 1 passed (12.2s)**; BERT+mamba slow two-arch tests executor-run |
| 6 | Any model × binary task probes frozen embeddings end-to-end: layer/pooling-selectable extraction, fixed-hyperparameter logistic/mlp probe, train-only scaler, registry metrics, npz cache keyed by 4-tuple with cache-hit assertions, documented schema (SC3/PROB-01) | ✓ VERIFIED | `probing.py`: D-13 constants (`LOGISTIC_MAX_ITER=1000` etc., no YAML surface), `StandardScaler` fit on train split, metrics exclusively via `metric_registry.resolve` (line 80 import), sha256 cache filenames + atomic `os.replace` (line 263), float32, F4 schema in docstring; 34 fast tests passed first-hand (incl. cache-hit/cache-miss-on-param-change, leakage spy, edge battery); slow real-model acceptance executor-run (cache-hit asserted) |
| 7 | A user can score variants zero-shot from a VCF: same-slot rule + skip accounting, paradigm↔architecture guard, evaluate_vcf + dnallm-vep CLI, VCF coordinate fixtures, AUROC/AUPRC via registry, formulas in docstrings + README (SC4/VEP-01, capability clauses) | ✓ VERIFIED | `vep.py`: `align_variant` same-slot rule, `score_variant` → `_check_paradigm_compatible` → kernels (all called, verified at `vep.py:714-731`), guarded `import allel` with dnallm ValueError (`vep.py:806-812`), `read_vcf` alt_number=4, D-17 filter chain (CLNVC→CLNSIG→star floor), uppercase windows (`vep.py:559`), `-delta` deleteriousness direction documented, registry resolve (`vep.py:953-957`); `cli/vep.py` click command wired to evaluate_vcf; README protocol section (lines 530-541); 56 fast tests + 7 CLI tests passed first-hand; ClinVar 5-model slow acceptance executor-run (5 passed, 183s GPU) |
| 8 | The ClinVar acceptance produces **literature-magnitude AUROCs** (SC4/VEP-01, magnitude clause) | ✗ FAILED | Measured 0.490–0.581 across all five pinned models vs the phase's own anchors (CLM 0.85–0.98, MLM 0.70–0.80 from 2.5B–40B models); executors recorded `within_anchor_range: false` and the test asserts only a ≥0.45 sanity floor (`test_vep.py:1146`). Honest empirical finding, not a wiring failure — see Gaps Summary and the override suggestion below |
| 9 | Multi-seed sweeps: `run_seeds` directory protocol + pure `aggregate_seeds` (n-guarded t/bootstrap), proven against known arrays, ≥3-seed trial with statistics block (SC5/SEED-01) | ✓ VERIFIED | `sweep.py`: n<3→method "none", 3≤n<10→`stats.t.ppf(0.975, n-1)`, n≥10→`np.random.default_rng` percentile bootstrap; boundary matrix n=2/3/9/10 + known-array moment tests passed first-hand (73 probing+sweep fast tests); **3-seed end-to-end slow trial run by verifier: 1 passed (34.6s)**; `coverage run` spot-check: sweep.py 100% (112/112) |

**Score:** 8/9 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dnallm/configuration/presets/lora_targets.yaml` | 35-family packaged presets with targets, FFN subsets, lora_r, ratio bands | ✓ VERIFIED | 533 lines; 35/35 families, structural invariants hold |
| `dnallm/finetune/trainer.py` (modified) | IA³ branch, preset auto-selection, dry-run validator, ratio guard | ✓ VERIFIED | All present and wired; ruff clean |
| `dnallm/configuration/configs.py` (modified) | `peft_dry_run`, cross-field rejections, Ia3Config parity | ✓ VERIFIED | Fields + validator verified at lines 326-454 |
| `dnallm/inference/inference.py` (modified) | Adapter-kind-agnostic reload naming | ✓ VERIFIED | `PeftModel.from_pretrained` + "PEFT adapter" strings |
| `dnallm/models/model.py` (modified) | random_init path, allowlist, per-tensor fingerprint | ✓ VERIFIED | Full implementation read and verified |
| `dnallm/inference/probing.py` (new) | extract/fit/cache/registry | ✓ VERIFIED | 688 lines, 100% module coverage claimed + methodology validated on sweep.py |
| `dnallm/finetune/sweep.py` (new) | aggregate_seeds + run_seeds | ✓ VERIFIED | 452 lines; 100% coverage (112/112) confirmed first-hand via `coverage run` |
| `dnallm/inference/vep.py` (modified) | evaluate_vcf + score_variant + guard | ✓ VERIFIED | 997 lines; all contract elements present |
| `dnallm/cli/vep.py` (new) | dnallm-vep click command | ✓ VERIFIED | 136 lines, lazy imports, error channels |
| `pyproject.toml` (modified) | exactly the two sanctioned lines | ✓ VERIFIED | `git diff 24d90cb..HEAD` = scikit-allel dep + dnallm-vep script, nothing else |
| `tests/inference/data/synthetic_variants.vcf` + `synthetic_reference.txt` | committed network-free fixture | ✓ VERIFIED | 14-row VCF + mixed-case reference sidecar |
| `tests/expected_skips.yaml`, `models.lock` | clinvar-unavailable allowlist; sha pins | ✓ VERIFIED | Line 35 prefix entry; 3 new sha-pinned rows (lines 44-46) + multi-pin convention header |
| README.md, CHANGELOG.md | random_init paragraph + VEP protocol section; REV-04..09 entries | ✓ VERIFIED | Both sections present; all six CHANGELOG bullets under `[Unreleased]` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| trainer.py | presets/lora_targets.yaml | importlib.resources lookup | ✓ WIRED | `resources.files("dnallm.configuration").joinpath(...)` at trainer.py:103 |
| trainer.py | inference.py shared adapter reload | PeftModel.from_pretrained | ✓ WIRED | Roundtrip test run by verifier passed |
| configs.py | peft IA3Config | field-filtered pass-through | ✓ WIRED | `PEFT_IA3_FIELD_NAMES` from `dataclass_fields` (version-span safe) |
| model.py | transformers AutoConfig/Auto*.from_config | no-checkpoint instantiation | ✓ WIRED | Verified in `_load_random_init_model` |
| model.py | get_logger banner | per-tensor hash table | ✓ WIRED | `_log_random_init_fingerprint` INFO lines |
| probing.py | metric_registry | resolve/validate_emission | ✓ WIRED | Import + calls verified; registry file untouched (git diff empty) |
| probing.py | inference.py hidden-states idiom | output_hidden_states reuse | ✓ WIRED | Consumer, not reimplementer; IN-04 restore fix present (lines 383-412) |
| sweep.py | scipy.stats.t / default_rng | t.ppf + seeded bootstrap | ✓ WIRED | Lines 141, 148 |
| sweep.py | DNATrainer orchestration | run_seeds drives train/evaluate | ✓ WIRED | Proven by the slow trial run first-hand |
| vep.py | allel.read_vcf | guarded import + fields/alt_number | ✓ WIRED | veep.py:806-836 |
| vep.py | Phase-10 kernels | align_variant + clm/mlm kernels wrapped | ✓ WIRED | `score_variant` calls verified at vep.py:714-731 |
| vep.py | metric_registry | AUROC/AUPRC resolve | ✓ WIRED | vep.py:953-957 |
| cli/vep.py | evaluate_vcf | lazy import in command body | ✓ WIRED | cli/vep.py:62, 104 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| vep.py | per-variant deltas | model forward via clm/mlm kernels | Yes — no static returns; perfect-separation test uses mocked kernels explicitly marked as mock | ✓ FLOWING |
| probing.py | embeddings | model forward output_hidden_states | Yes | ✓ FLOWING |
| sweep.py | statistics blocks | caller-supplied per-seed metric values | Yes — pure function over inputs | ✓ FLOWING |
| lora_targets.yaml | target lists | packaged YAML | Yes — loaded via importlib.resources | ✓ FLOWING |
| model.py random path | tensor digests | sha256 over live tensor bytes | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Presets + CLI fast lane | `pytest tests/configuration/test_peft_presets.py tests/cli/test_vep_cli.py -m "not slow"` | 28 passed | ✓ PASS |
| Probing + sweep fast lane | `pytest tests/inference/test_probing.py tests/finetune/test_sweep.py -m "not slow"` | 73 passed, 2 deselected | ✓ PASS |
| VEP fast lane | `pytest tests/inference/test_vep.py -m "not slow"` | 56 passed, 5 deselected | ✓ PASS |
| random_init fast proofs | `pytest tests/models/test_model.py::TestRandomInit -m "not slow"` | 38 passed, 3 deselected | ✓ PASS |
| IA³ wiring + config rejections | `pytest tests/finetune/test_trainer.py -k "Ia3/peft/dry_run/preset/ratio"` + `test_configs.py -k ia3/HeadConfig` | 14 + 11 passed | ✓ PASS |
| 3-seed sweep end-to-end (SEED-01 acceptance) | `pytest tests/finetune/test_sweep.py -m slow` | 1 passed in 34.61s | ✓ PASS |
| IA³ roundtrip (peft #2429 class) | `pytest ...::test_ia3_adapter_save_reload_roundtrip -m slow` | 1 passed in 44.00s | ✓ PASS |
| Mamba IA³ fine-tune (silent-skip ground) | `pytest ...::test_ia3_training_mamba -m slow` | 1 passed in 34.43s | ✓ PASS |
| Per-tensor random-vs-pretrained hash proof | `pytest ...::TestRandomInit::test_random_init_per_tensor_difference_vs_pretrained -m slow` | 1 passed in 12.15s | ✓ PASS |
| sweep.py module coverage | `coverage run -m pytest tests/finetune/test_sweep.py -m "not slow"` + scoped report | 100% (112/112) | ✓ PASS |
| ruff format + check on lane modules | `ruff format --check` + `ruff check` on 7 phase modules | 4 formatted, all checks passed | ✓ PASS |

Full fast-lane gate at HEAD (cross-checked, executor-run ~30 min prior): `pytest tests/ -m "not slow"` = 2253 passed / 1 skipped / 0 failed. Slow acceptances not re-run by the verifier (transformer IA³ fine-tune; random_init BERT+mamba loads; probing real-model; ClinVar 5-model GPU) are covered by executor evidence in the SUMMARYs plus the verifier's first-hand runs of the highest-risk members of each family.

### Probe Execution

Not applicable — no `scripts/*/tests/probe-*.sh` declared or conventional for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PEFT-01 | 11-01 | IA³ adapter support symmetric to LoRA | ✓ SATISFIED | Truths 1, 2, 4 |
| PEFT-02 | 11-01 | Per-model PEFT target-module presets | ✓ SATISFIED | Truth 3 |
| BASE-01 | 11-02 | random_init from-scratch baselines | ✓ SATISFIED | Truth 5 |
| PROB-01 | 11-03 | Frozen-embedding probing | ✓ SATISFIED | Truth 6 |
| VEP-01 | 11-05 | Zero-shot VEP from VCF | ⚠️ PARTIAL | Capability truths 7 verified; "literature-magnitude AUROCs" acceptance not achieved (truth 8) — the single gap |
| SEED-01 | 11-04 | Multi-seed sweep protocol | ✓ SATISFIED | Truth 9 |

Orphaned requirements: none — REQUIREMENTS.md maps exactly the six IDs above to Phase 11 and all six are claimed by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dnallm/models/model.py | 1228 | `# TODO: Add more special cases if needed` | ℹ️ Info | Pre-existing (introduced 2026-03-27, commit e0b3494 — confirmed NOT in the phase diff as an added line); out of phase scope |

No TBD/FIXME/XXX/HACK/PLACEHOLDER markers in any phase-modified file. No stub implementations found — every new module computes from real inputs. Prohibition checks all pass: pyproject delta is exactly the two sanctioned lines; `dnallm/__init__.py` and `dnallm/tasks/metric_registry.py` diffs empty; no repo-root `configs/` additions; no private peft/transformers imports in tests; zero Co-Authored-By trailers in the 40-commit phase range; ratio guard computed from `requires_grad` tensors, never parsed stdout.

### Human Verification Required

None — status is gaps_found; the single failed truth is an owner decision (below), not a human test. All behavior-dependent truths carry behavioral test evidence (verifier-run or executor-run slow-lane plus first-hand fast-lane invariant coverage).

### Gaps Summary

**One gap, one root cause: the VEP-01 magnitude acceptance.**

Everything else in Phase 11 is verified end-to-end, with first-hand behavioral evidence on the highest-risk acceptances (Mamba IA³ fine-tune, IA³ roundtrip, per-tensor hash proof, 3-seed sweep trial) and green fast lanes across all six new/changed test files.

The gap is the magnitude qualifier of ROADMAP SC4 / VEP-01: "the ClinVar 1k-sample × ≥5-model acceptance produces literature-magnitude AUROCs." The acceptance ran (executor evidence: 5 passed on GPU), the capability is demonstrably working — correct formulas (docstrings + README), corrected `-delta` direction so discriminating models score above the floor, skip accounting with per-reason fractions, convention block beside every AUROC — but the measured AUROCs (plant-dnabert-BPE 0.543, NT-v2-50m 0.523, plant-dnagpt-6mer 0.581, plant-dnamamba 0.500, plant-dnagpt-BPE-promoter 0.490) sit at the random floor, below the recorded CLM (0.85–0.98) and MLM (0.70–0.80) anchors. The executors recorded `within_anchor_range: false` honestly and relaxed the plan's floor to a 0.45 statistical sanity bound — the plan itself had already reworded the SC from "produces literature-magnitude" to "compared against the anchors", a scope reduction this verifier does not accept silently (plans may add to, never subtract from, roadmap SCs).

Adjudication weighing: the anchors derive from 2.5B–40B-parameter models with different negative schemes (NT used 1000G-common negatives; D-17 uses within-ClinVar B/LB), while all five acceptance models are ≤~50M-parameter plant DNA models; the phase's own RESEARCH (Pitfall 7) cites Alfisi et al. with normalized-Wilcoxon <0.6 for most DNA models under strict labeling, and the only same-scale anchor (DNABERT-2 BPE embedding-distance 0.538) IS matched by plant-dnabert-BPE's 0.543. On that evidence this is an honest empirical finding about small models, not a capability failure — but the criterion as written is not met, and per the backstop rules it must not pass silently.

**This looks intentional.** To accept this deviation, the owner should add to this file's frontmatter:

```yaml
overrides:
  - must_have: "The ClinVar 1k-sample x >=5-model (CLM/MLM mix) acceptance produces literature-magnitude AUROCs"
    reason: "Small (<=50M) plant DNA models measure 0.490-0.581 AUROC on the D-17 within-ClinVar cohort — at the random floor, honestly recorded (within_anchor_range: false). The 0.70-0.98 literature anchors come from 2.5B-40B models with different negative schemes (1000G-common vs within-ClinVar B/LB); the phase's own research (Alfisi et al. <0.6 for most DNA models under strict labeling; DNABERT-2 BPE 0.538) predicts this band at this scale, and plant-dnabert-BPE 0.543 matches the same-scale anchor. The capability (formulas, direction, skip accounting, convention reporting) is verified working."
    accepted_by: "owner (Tao Zhang)"
    accepted_at: "2026-10-09T23:55:00+08:00"
```

The alternative resolution is an acceptance run on a model large enough to reach the anchor band (none currently models.lock-pinned).

---

_Verified: 2026-10-09T15:52:24Z_
_Verifier: Claude (gsd-verifier)_
