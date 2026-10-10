# Pitfalls Research

**Domain:** Adding paper-revision-driven ML-evaluation capabilities (REV-01…REV-11) to DNALLM — an existing DNA-LM toolkit wrapping HF Trainer/peft/transformers 4.49–5.x, with a 96.42%-coverage CI hard gate, executed by 4–5 parallel implementation agents.
**Researched:** 2026-10-09
**Confidence:** HIGH for repo-grounded facts (verified against source at branch `revision`); MEDIUM for library-behavior claims across version spans (verified against installed peft 0.21.2 / transformers 5.19.0 source plus cross-checked web sources); each pitfall notes its grounding.

**Grounding:** `dnallm/finetune/trainer.py:234-241,298-303,489-518`, `dnallm/inference/inference.py:80-158,1746`, `dnallm/inference/mutagenesis.py:257-347`, `dnallm/models/model.py:753-915,1007-1026`, `dnallm/tasks/metrics.py:113-219`, `dnallm/datahandling/data.py:842-860`, `dnallm/utils/sequence.py:89-124`, `dnallm/mcp/server.py:263-354,1743-1857`, `dnallm/mcp/model_manager.py:99-121`, `dnallm/configuration/configs.py:263-337`, `pyproject.toml` (coverage omit / ruff / mypy excludes), `tests/expected_skips.yaml`, `.planning/research/261009-paper-revision-suite-plan.md`.

**Phase labels** follow the intake's structure: **Phase A** (contract layer, P0: REV-01/02/03), **Phase B waves B1–B4** (adaptation + evaluation, P1: REV-04…09), **Phase C** (narrative, P2: REV-10/11). Wave assignments below are recommendations to the roadmap author.

---

## Critical Pitfalls

### Pitfall 1 (REV-01): The eval-semantics guard is bypassed by its own neighbors — early stopping force-enables `load_best_model_at_end`, and a YAML `load_best_model_at_end: True` crashes or re-leaks after the guard disables eval

**What goes wrong:**
The leak mechanism is confirmed at `trainer.py:234-241`: with no dev/val split, `test` silently becomes `eval_dataset` (`eval_strategy` stays `"steps"`), so per-interval evaluation, `metric_for_best_model` selection, and `load_best_model_at_end` all operate on the test set. The naive fix — "when no dev, set `eval_strategy='no'` and `eval_dataset=None`" — collides with three transformers 5.x behaviors verified in the installed source:

1. `Trainer` raises `ValueError: "You have set args.eval_strategy to {strategy} but you didn't pass an eval_dataset…"` if strategy is active with no dataset — the original crash the silent test-promotion was avoiding.
2. `trainer.py:298-303` (early stopping) **actively re-enables** `load_best_model_at_end=True` when the user did not set it. After the guard nulls eval, this path resurrects best-checkpoint selection with no eval set, producing either a transformers error or, worse, evaluation on some fallback.
3. Benchmark YAMLs (the very configs under revision) ship `load_best_model_at_end: True` + per-task `metric_for_best_model`. Transformers 5.19 silently defaults `metric_for_best_model="loss"` when `load_best=True` and the metric is `None` (verified in `TrainingArguments.__post_init__`) — silent defaults stacking on silent defaults.

**Why it happens:**
The leak is the emergent composition of three individually-reasonable code paths (split detection, early-stopping convenience, YAML passthrough). A guard written only against `trainer.py:234-241` misses the other two.

**How to avoid:**
- Guard must set BOTH `eval_strategy="no"` AND `eval_dataset=None` atomically, and when the guard fires with user-set `load_best_model_at_end=True` or early-stopping enabled, raise a descriptive `ValueError` (matchable message, e.g. `"test split cannot serve as eval set; provide a dev split or pass allow_test_as_eval=True"`) — do not silently downgrade two independent user settings.
- `allow_test_as_eval=True` override must emit a WARN (repo uses `print("[Warning] ...")` in trainer.py today; new code should use `get_logger`).
- New `evaluate(split="test"|"dev"|...)` must route test through `trainer.predict` (the proven-correct `infer()` path at `trainer.py:502-518`), never `trainer.evaluate`.
- Tests (acceptance-grade): 3 split combos (dev+test / test-only / train-only) × default vs explicit override. Assert: (a) test-only + default → `training_args.eval_strategy == "no"` and `load_best_model_at_end` untouched-or-False; (b) test-only + early stopping → the descriptive `pytest.raises(ValueError, match=...)` (this is the neighbor-path test most likely to be forgotten); (c) override path logs the warning and evaluates on test.

**Warning signs:**
- Review: any diff touching `trainer.py` split logic that changes `eval_key` handling without touching the early-stopping block at 298-303.
- Tests: a green suite where the early-stopping + test-only case is absent; `load_best_model_at_end=True` in a test fixture YAML combined with no dev split.
- CI: fast leg passing while benchmark-side F1 (test-predict) and suite-side semantics disagree in a follow-up manual check.

**Phase to address:** Phase A (REV-01 plan). The neighbor-path tests must be in the same change as the guard (owner rule: lib changes ship with same-change pytest coverage).

---

### Pitfall 2 (REV-02): The metric registry is placed inside `dnallm/tasks/metrics/` — a directory excluded from coverage, ruff, AND mypy — so the new contract layer ships unmeasured and unlinted, while alias handling re-creates the drift it was built to prevent

**What goes wrong:**
The intake specifies `dnallm/tasks/metrics/registry.py`. Verified in `pyproject.toml`: `*/dnallm/tasks/metrics/*` is in the coverage omit list (line 542), the ruff exclude (line 304), and the mypy exclude (line 405). The new registry — the single source of truth the whole re-run depends on — would be invisible to all three gates. Separately, the historical drift failure mode ("key-name drift → exporter reads empty string → hand-patched JSON → broken provenance", which produced reviewer finding R1-2d) recurs if aliases are treated as symmetric: `resolve("eval_auroc")` must resolve, but nothing must ever EMIT a historical alias — otherwise a new producer silently resurrects the old spelling and the companion repo pins the wrong key again.

**Why it happens:**
The vendored boundary was drawn at the directory level with globs, and the natural-looking home for a "metrics registry" is the metrics directory. Alias semantics look like a bidirectional map but the contract is directional (recognize historical, emit canonical only).

**How to avoid:**
- Place the registry OUTSIDE the vendored glob: `dnallm/tasks/registry.py` (or keep the intake path but change the omit/exclude globs to enumerate vendored subdirs — more brittle; prefer relocation). Then verify with a same-change test that `coverage report --include=**/registry.py` shows a measured row.
- Design as one-directional: `resolve(name)` recognizes aliases; the emitted dict from `compute_metrics` paths only ever contains canonical keys. Add a test that iterates every registered metric fn and asserts no historical alias (`eval_auroc`, `eval_spearman_r`) appears in any emitted key set.
- Contract test enumerating the full key set used by the 47 benchmark tasks (intake acceptance), plus known-alias resolution tests with exact strings: `pytest` asserts `resolve("eval_AUROC") == "AUROC"` and `resolve("eval_auroc") == "AUROC"` (recognition) — anchoring the CURRENT pipeline spellings (`AUROC`, `AUPRC`, `spearmanr`, `pearsonr`, verified at `metrics.py:134-135,181-219`) as canonical.
- Cross-repo: dnallmmark must pin a dnallm version/commit (its import must not float against dnallm HEAD — open question #3 in the intake). The dnallm-side test asserts the registry's public surface (`resolve`, key set) so a dnallm change that breaks it fails HERE before dnallmmark CI notices.

**Warning signs:**
- CI: `coverage.json` / `coverage report` shows no row for `registry.py` (it was omitted) — that silence is the alarm. A quick check: `coverage report | grep registry` empty after merge.
- Review: any `emit`/`as_dict` API on the registry that returns aliases; any `dnallmmark` PR importing dnallm without a version pin.
- Tests: alias-in-emission test failing; contract key-set test failing when someone adds a metric with a new spelling.

**Phase to address:** Phase A (REV-02 plan). Must land before any dnallmmark F3 exporter work — the dependency chain in the intake makes this the gate-opener for the whole re-run.

---

### Pitfall 3 (REV-04): IA³ is combined with the existing QLoRA 4-bit path and the shared adapter save/load path — peft raises (or worse, merges fail) because IA³ × 4-bit is not fully supported, and a known peft bug corrupts saved `adapter_config.json`

**What goes wrong:**
Three verified failure surfaces:
1. **IA³ × 4-bit:** older peft raises `NotImplementedError("4-bit quantization is not supported for IA3 yet…")` at injection; peft 0.21.2 (installed, source-verified) allows injection but raises `ValueError("Cannot merge ia3 layers when the model is loaded in 4-bit mode")` (and the 8-bit variant). The existing trainer path (`use_qlora` → `prepare_model_for_kbit_training` at `trainer.py:159-163` → `get_peft_model`) makes `use_ia3 + use_qlora` a one-flag-away combination. Behavior differs across the `peft>=0.14` floor the package declares — CI green on peft 0.21 says nothing about peft 0.14–0.16.
2. **Save/reload roundtrip:** peft's target-module list minimization corrupted IA³ configs (issue #2429, fixed by PR #2432) — saved `adapter_config.json` gets simplified `target_modules` but full-path `feedforward_modules`, failing on reload. The intake's acceptance ("adapter save/reload roundtrip consistent") hits exactly this class if the roundtrip test only checks LoRA.
3. **Shared adapter path:** `inference.py:111-130` loads adapters via `PeftModel.from_pretrained(...)` with no adapter-type awareness. An IA³ adapter flows through the same path — fine for forward passes, but anything downstream that calls `merge_and_unload()` (or a user notebook) breaks on quantized models. Also `feedforward_modules` must be a subset of `target_modules` and the subset CHECK IS SKIPPED when either is a regex string — regex-based presets silently skip validation.

**Why it happens:**
IA³ is being added to a codebase whose PEFT scaffolding (kbit prep, adapter save at `trainer.py:401-419`, adapter load at `inference.py:111-130`) was built and battle-tested for LoRA/QLoRA only. "Align with the LoRA branch" (the intake instruction) copies assumptions that do not hold for IA³.

**How to avoid:**
- Reject the incompatible combination at config-validation time, not at peft's mercy: Pydantic `model_validator` on `TrainingConfig` raising `ValueError("use_ia3 cannot be combined with use_qlora (4-bit); IA³ does not support 4-bit quantized training")`. Test with `pytest.raises(ValidationError, match="use_ia3")`. This converts a peft-version-dependent runtime error into a stable, matchable, version-independent one (repo error-handling convention).
- Same-change roundtrip test for IA³ specifically: save adapter → reload via `DNAInference(lora_adapter=...)` → assert outputs identical (tolerance-based) to the pre-save model on a fixed input. Cover one transformer AND one Mamba model (intake acceptance).
- If presets use regex `target_modules`, add an explicit preset self-test that the regex actually matches ≥1 module per family (see Pitfall 4) because peft's subset validation no-op's on regex.
- Pin minimum peft version only if a specific behavior is required; otherwise feature-detect. Do NOT pin a single peft minor in tests (same constraint as transformers).

**Warning signs:**
- Tests: `use_ia3 + use_qlora` test absent; IA³ roundtrip test absent (only LoRA roundtrip).
- CI: nightly real-model leg failing with `Cannot merge ia3 layers` or `NotImplementedError` from peft internals — an unmatchable foreign error means the guard was missing.
- Review: any `merge_and_unload` call in new MCP/inference code paths that can receive a quantized model.

**Phase to address:** Phase B wave B1 (REV-04 + REV-05 together — they share `configs.py`/`trainer.py`/presets; see Pitfall 8).

---

### Pitfall 4 (REV-05): Wrong PEFT `target_modules` do not fail — peft SILENTLY SKIPS non-existent modules on Mamba/hybrid backbones, so "IA³/LoRA fine-tuned" results are a frozen model with near-zero trainable parameters

**What goes wrong:**
Verified peft behavior (source + troubleshooting docs): on hybrid architectures (Mamba, Jamba-like, and DNALLM's registry spans BERT/GPT/Mamba/Gemma/Llama/hybrid), `target_modules` entries that match nothing are silently skipped; only the match-NOHING-at-all case warns (`RuntimeWarning` from `tuners_utils`) or raises depending on version. A preset listing `query,key` against a Mamba backbone (whose projections are `in_proj`/`out_proj`) yields a model where the adapter attached to few or no layers. Training runs, loss decreases slightly (from the head/bias drift), results get exported, and the paper's IA³ baseline is garbage — discovered only when a reviewer asks why IA³ ≈ frozen. `trainer.py:168` already prints trainable parameters (`print_trainable_parameters`), but nobody asserts on it.

**Why it happens:**
Module names differ per architecture family and the intake correctly demands presets "derived from real model configs, not guessed" — but the failure mode is silent, so a wrong preset survives every run that lacks an assertion. The `warning sign` culture (assert on counts) is not yet wired into PEFT setup.

**How to avoid:**
- Generate presets from real `config.json` module names (intake requirement) and add a **dry-run contract test per architecture family**: instantiate (or introspect via `AutoConfig` + module-name listing without full weight load where possible) each family representative and assert every preset `target_modules` entry matches ≥1 module, and that the matched-module count equals the expected per-family count. Families without a representative in the fast leg get a `slow`-marked variant on the nightly GPU runner (models.lock row for any newly downloaded model — repo precedent).
- Runtime guard in the trainer: after `get_peft_model`, parse `print_trainable_parameters` output (or compute trainable/total ratio) and `raise ValueError` when trainable count is 0 or the ratio is implausibly below the family preset's expected ratio band. Message like `"IA3/LoRA attached to 0 modules; target_modules preset {name} does not match this backbone"`.
- Preset-file schema test (`configs/presets/lora_targets.yaml`): every family row has `target_modules` (non-empty list or validated regex) + recommended `r`; unknown family keys rejected — prevents hand-edit drift, mirroring the models.lock consistency-guard pattern (CI-08 precedent).

**Warning signs:**
- Tests/CI: trainable-parameter assertion absent from the IA³/LoRA smoke tests; nightly log line `trainable params: 0` — grep-able, so make the assertion automatic rather than eyeball.
- Review: presets YAML whose module names were typed by hand without a provenance comment citing the source `config.json`/model.
- Science: IA³ results suspiciously identical to frozen-model probing results (REV-07 cross-check).

**Phase to address:** Phase B wave B1 (REV-05, same wave as REV-04; the trainer-side trainable-count guard belongs to whichever plan lands second in the wave, or a shared acceptance).

---

### Pitfall 5 (REV-06): `random_init=True` accidentally loads pretrained weights — via `from_pretrained` side effects, tied weights, or safetensors cache — and the parameter-hash "proof" is device-dependent so the test lies

**What goes wrong:**
Four verified sub-traps:
1. **The load path is `from_pretrained`-shaped.** The generic loader (`model.py:878-915`) selects Auto* classes and calls `from_pretrained`. The natural implementation — call `from_pretrained` then re-`init` weights — still downloads/mmap's pretrained weights and risks stale-tensor leftovers (buffers, non-reinitialized norms). Transformers explicitly documents meta-device + `from_pretrained` as an anti-pattern raising `RuntimeError`; `from_config` is the canonical random-init path. But `from_config` changes the call signature per task-type Auto* class, and the special-family handlers (evo, megadna, enformer — dispatched at `model.py:806-863`) each construct models their own way; random-init semantics are undefined for several of them.
2. **Tied weights / meta tensors:** memory-efficient init in transformers 5.x can leave tied weights (e.g., `lm_head.weight` tied to embeddings) on the meta device → `"Cannot copy out of meta tensor; no data!"` on `.to(device)` (transformers #41038/#30703; partial fix in #43523). Also transformers 5.x `post_init()` tied-weight registration breaks some `trust_remote_code=True` models — and DNALLM loads several remote-code families (DNABERT-2, GPN, …). Verify tying with `data_ptr()` equality after init.
3. **Hash proof is device/RNG-dependent:** CUDA init (philox) and CPU init produce different tensors for the same seed. A hash-inequality test written on CPU passes; the same test on the nightly GPU runner compares different RNG streams and can spuriously pass/fail. Worse: a *weak* hash test (random-init hash ≠ pretrained hash) passes even if only 90% of tensors were re-initialized — one leftover pretrained block still yields a different hash.
4. **Tokenizer must load normally:** reusing the full load path and skipping weights accidentally skips tokenizer load too (the special handlers return `(model, tokenizer)` tuples).

**Why it happens:**
`from_pretrained` is wired into 12 special handlers and the guarded dispatch chain; threading a `random_init` flag through without leaking weight loads requires touching each family's semantics. The hash-proof acceptance criterion invites a naive implementation.

**How to avoid:**
- Implement per-path: generic path → `Auto*Class.from_config(config)` (no `from_pretrained` call at all — this also proves no weight download: assert no new files in the HF cache in a test, or assert `_get_model_path_and_imports`/download was never invoked by patching); special families → explicit allowlist of families supporting random_init, `ValueError("random_init is not supported for {family}")` otherwise (matchable message, documented in README). Do NOT attempt to retrofit all 12 handlers.
- Seed before init (`torch.manual_seed` fixed), initialize on CPU, then `.to(device)` — deterministic hash within a device class; document that hashes are CPU-canonical.
- Proof test design (all same-change): (a) per-tensor comparison, not one global hash — assert EVERY parameter tensor's bytes differ between random-init and pretrained for ≥2 architectures (or assert count of differing tensors == total tensors, allowing documented exceptions like int buffers); (b) same-seed reproducibility: two random-init loads with the same seed → identical per-tensor hashes; (c) tokenizer equality: `random_init=True` tokenizer identical to normal-load tokenizer (vocab/hash); (d) no-download assertion via patched downloader.
- Regression test for the meta/tied-weight class: load a tied-weights generation-family model with random_init on CPU and call `.to("cpu")`→`.to("cuda")` on the nightly leg; `data_ptr()` tie assertion where applicable.

**Warning signs:**
- Tests: a single global-hash inequality test (weak proof); any random-init test that runs the download path (network in fast leg — the typed-skip audit would also flag network skips if used to dodge this).
- CI: `"Cannot copy out of meta tensor"` traceback on nightly; `HF` cache growing during random-init tests.
- Science: from-scratch learning curves that converge suspiciously like pretrained ones (reviewer-facing evidence the intake explicitly wants the hash for).

**Phase to address:** Phase B wave B2 (REV-06, `model.py` cluster — no other B-wave plan owns `model.py`, keeping this wave conflict-free).

---

### Pitfall 6 (REV-08): Variant↔token slot misalignment makes Δlog-likelihood incomparable — the exact challenge the reviewer raised — and it fails silently with plausible-looking AUROCs

**What goes wrong:**
Verified field consensus: a single SNV under BPE or k-mer tokenization changes the tokenization of the whole downstream sequence (DART-Eval arXiv:2412.05430 restricts likelihood eval to fixed-encoding models for this reason; Mut-BPE gives the canonical example: ref `['ATCA','ATGGC','AATT']` vs alt `['ATCA','TAGCA','TT']` — different tokens, different COUNT, no positional correspondence). dnallm's existing kernels inherit the trap: `clm_evaluate` (`mutagenesis.py:312-347`) computes whole-sequence logp; scoring ref vs alt sequences tokenized independently mixes tokenization-shift noise with the variant effect. With non-overlapping 6-mers (NT-style) a substitution shifts every k-mer downstream by the same phase — ref/alt tokens never occupy the same slot. Additional traps stacked on top:
- **Paradigm mismatch:** applying CLM Δlogp scoring to bidirectional MLM models (or pseudo-PLL `mlm_evaluate` masking one token at a time to a causal model) yields numbers with no valid interpretation. One ClinVar evaluation of masked-LM allele probabilities found near-random AUROC (0.345–0.536) — near-random results are an EXPECTED outcome of protocol mismatch, not proof of a code bug, which makes the trap hard to detect from results alone.
- **VCF coordinates:** 0- vs 1-based, `chr` prefixes, GRCh37/38 mismatch, indel left-alignment, REF allele not matching the reference genome. A ±1 offset corrupts the alignment for every variant. dnallm already has `dnallm.utils.genomic_coords` (v1.1) to reuse.
- **ClinVar ascertainment bias:** labels enriched for pathogenic-observed variants; AUROC "consistent with literature magnitudes" (the intake acceptance) must use the same filtering conventions as the reference literature or magnitudes are incomparable.

**Why it happens:**
The pieces exist (`scoring()` at `inference.py:1746`, `mlm_evaluate`/`clm_evaluate`), so assembling a VEP scorer looks like glue work. The alignment requirement — ref and alt tokenizations must be IDENTICAL except at the variant slot (Mut-BPE's split-token approach: prefix/variant-base/suffix) — is invisible in any single test that only checks "it returns scores".

**How to avoid:**
- Make `align_variant(seq, pos, ref, alt, tokenizer)` a pure, heavily-tested function: hand-computed unit tests per tokenizer class — character/3-mer/6-mer (fixed-stride: alignment holds iff pos maps to a token boundary or the substitution stays within one token — assert both cases), BPE (assert the split-token scheme yields ref/alt token lists of equal length differing in exactly one index, else return an explicit `Skip(reason)` record). Assert: aligned(ref) vs aligned(alt) differ in EXACTLY one slot — this assertion IS the R1-3e① response.
- Explicit skip accounting: `evaluate_vcf` returns scored variants + per-reason skip counts (`slot_mismatch`, `paradigm_unsupported`, `coordinate_invalid`); AUROC computed on the scored subset with the skip denominators reported. Test with a synthetic VCF where each skip reason fires a known number of times.
- Paradigm guard: `score_variant(paradigm=...)` raises `ValueError` (matchable: `"paradigm='clm' requires a causal model"`) when the loaded model is bidirectional — introspection via model config `is_decoder`/architectures, consistent with the repo's reflection-based capability detection (`_get_accepted_forward_args` precedent).
- VCF validation: assert REF matches the provided reference sequence at pos before scoring; reject mixed coordinate conventions with a per-record error, not silent best-guess parsing. Unit tests with 0-/1-based and `chr`-prefix fixtures.
- Both-strand option documented (established practice: average forward/revcomp LLRs) — and the reverse-complement helper at `utils/sequence.py` has a lowercase-`n` mapping quirk worth a fixture test if N-containing windows occur.
- Goldens: small frozen VEP fixture (e.g., 20 variants, 2 tokenizer classes) with pinned scores per release — catches silent scoring-kernel drift.

**Warning signs:**
- Tests: alignment function tested only via "returns something"; no test where a variant is REJECTED for slot mismatch; no skip-count assertions.
- Review: any ref/alt scored through independent `tokenizer(seq)` calls — that one line is the reviewer's challenge incarnate.
- Science: AUROC ≈ 0.5 on a known-good dataset (GPN/DNABERT-2 reference magnitudes) — treat as protocol bug first, data second.

**Phase to address:** Phase B wave B3 (REV-08, new `inference/vep.py`). Depends on REV-02 registry (metrics output) from Phase A. Highest research-flag weight of the milestone — the roadmap should mark this phase for deeper plan-time research if tokenizer-class coverage grows.

---

### Pitfall 7 (REV-09): The seed illusion (seed does not control dataloader/ augment/ dataset sampling), invalid statistics on n=3 seeds, and directory-protocol drift against the companion repo

**What goes wrong:**
Three layers:
1. **Seed scope:** `TrainingConfig.seed` (configs.py:291) feeds HF `TrainingArguments.seed` → `set_seed` covers python/numpy/torch main-process RNG — but not necessarily `DNADataset` operations performed BEFORE the trainer runs (`train_test_split(seed=...)` at data.py:823 takes a caller-supplied seed; stratified `sampling` has its own seed; reverse-complement augmentation randomness), nor bitwise GPU determinism (cuDNN autotune/atomics; `use_deterministic_algorithms` off by default). Two runs with the same sweep seed can differ in metric 4th decimal — and two runs with different seeds can accidentally share the same data split, shrinking the effective variance the sweep claims to measure (or the opposite: split changes with seed when it shouldn't, conflating split variance with init variance — pick one semantics and document it).
2. **Aggregation on n=3:** `ci95_bootstrap` on 3 values is statistically vacuous (bootstrap resamples from 3 numbers yield 10 distinct multisets); a normal-approximation CI at n=3 understates uncertainty. The intake itself specifies `mean/sd/ci95_bootstrap` — encode guards rather than shipping it verbatim.
3. **Protocol drift:** `{model}/{task}/seed_{s}/` must match dnallmmark F2 byte-for-byte or the companion's aggregation reads fail — the same cross-repo drift class as the metric-key history (REV-02), but for paths and JSON schema.

**Why it happens:**
"Set a seed" feels sufficient; every individual piece claims seeding support. The composite (dataset ops + trainer + augment + probe) has no single owner of randomness.

**How to avoid:**
- `run_seeds(fn, seeds, out_root)` must thread ONE seed into every stochastic stage and document the semantics: recommended default = same data split across seeds (split seed derived from dataset, not the sweep seed) so seed-to-seed variance measures init/shuffle only. Assert in a same-change test: two `run_seeds` invocations with identical seed → identical metrics on a CPU tiny model (deterministic there); different seeds → different metrics.
- `aggregate_seeds`: pure function, unit tests with constructed known arrays (intake acceptance); n-guard: `ci95` uses t-distribution (or reports `n<10, ci95 omitted`) — encode as `ValueError` or explicit None + documented field, with a test asserting bootstrap CI is refused/flagged at n=3.
- Directory/JSON protocol: write a schema contract test (structure + required `statistics` block) and share the exact spec with dnallmmark F2 via the same mechanism as REV-02 (dnallm is the source of truth; companion pins a version).
- Do NOT claim bitwise reproducibility on GPU — document CPU-deterministic only.

**Warning signs:**
- Tests: no same-seed-identity test; aggregation tests only on n≥10 arrays.
- CI/review: sweep output JSON missing the `statistics` block; seed directories named `seed_1` vs `s1` in different runs; a companion-repo PR hardcoding paths instead of importing the protocol.
- Science: identical metrics across "different" seeds (seed not actually threaded) — a canary assert (different seeds → not-all-identical) catches this cheaply.

**Phase to address:** Phase B wave B4 (REV-09, new `finetune/sweep.py`). If it needs `TrainingConfig`/`Ia3Config` schema additions, take them from the Phase A config-stub pass (see Pitfall 8) rather than editing `configs.py` concurrently with B1.

---

### Pitfall 8 (Milestone-level): 4–5 parallel implementation agents collide on the same files — `configs.py`, `trainer.py`, `dnallm/__init__.py`, `expected_skips.yaml`, `pyproject.toml` — and on test collection

**What goes wrong:**
Concrete collision matrix verified against the intake's file plan:
- `dnallm/configuration/configs.py`: REV-04 (`Ia3Config`, `use_ia3`), REV-05 (preset reference), REV-08 (VEP config), REV-09 (sweep/seed) — four plans editing one 700+-line Pydantic module in parallel waves.
- `dnallm/finetune/trainer.py`: REV-01 (Phase A) and REV-04 (B1) both edit init/setup paths; if wave boundaries blur, both land in one agent's branch.
- `dnallm/__init__.py`: five new modules (`registry`, `probing`, `vep`, `sweep`, `motifs`) each need an import + `__all__` line — N agents appending to the same lines = guaranteed textual conflicts, and mid-wave a partially-re-exported package breaks `from dnallm import ...` for everyone (the facade is the documented API surface).
- `tests/expected_skips.yaml`: any agent adding a typed skip (network-unavailable for a new real-model test) without same-change allowlisting fails `scripts/audit_skips.py` for ALL legs (4 CI jobs) — a shared gate failing for a private reason.
- Test collection: repo collects TWO roots (`tests/` + `dnallm/mcp/tests/`) with no `__init__.py`; two agents creating same-basename test files in different dirs (e.g., `tests/mcp/test_server_tools.py` vs an existing/parallel `dnallm/mcp/tests/test_server_tools.py`) trigger pytest "import file mismatch" collection errors that look mysterious.
- `pyproject.toml`: any coverage-omit or marker edit by one agent rebase-breaks another's.

**Why it happens:**
The compression decision (owner, 2026-10-09: ~1–1.5 days via 4–5 parallel agents) optimizes calendar time; every shared file is a serialization point the wave design must respect or the merge phase eats the saved time.

**How to avoid:**
- **Phase A pre-creates all shared-file scaffolding in ONE plan:** every new Pydantic config class (Ia3Config, VepConfig, SweepConfig) as accepted stubs, `__init__.py` re-exports for all five modules, any `pyproject.toml` changes, and (optionally) empty module files with docstrings. Phase B agents then FILL modules without touching shared files. This is the single highest-leverage structural decision for the milestone.
- **Wave file-ownership map as a roadmap artifact:** B1 owns `configs.py`+`trainer.py` (REV-04+05 together — intake dependency chain agrees); B2 owns `model.py` (REV-06); B3 owns new `inference/vep.py`+`probing.py` (REV-07+08 — different agents, zero shared files, but both consume the registry read-only); B4 owns `finetune/sweep.py` (REV-09); C owns `mcp/server.py`+`interpret/motifs.py` (REV-10/11). Each phase plan's UAT should include "no edits outside owned files" as a review check.
- **Skip allowlist discipline:** any new typed skip must be added to `expected_skips.yaml` in the SAME change (existing audit gate enforces this — make it a checklist line in every plan template so agents don't rediscover it in CI).
- **Unique test basenames:** convention `test_<module>.py` where module names are unique (they are, mirroring the package) — call it out so nobody creates helper-named twins across the two collected roots.

**Warning signs:**
- Process: merge conflicts on `configs.py`/`__init__.py` in more than one wave-PR; a wave PR whose diff touches files outside its ownership map.
- CI: `audit_skips.py` exit 1 with an unexpected skip message from another agent's feature; pytest collection errors mentioning "import file mismatch".
- Review: `__all__` ordering churn or duplicated import lines.

**Phase to address:** Phase A (scaffolding pass) + roadmap-level wave design (the phase planner, not individual plans, owns the file-ownership map).

---

## Moderate Pitfalls

### Pitfall 9 (REV-03): Comparability warnings treated as "just docs" while the validity trap ships — and the docs-mirror gate bites

**What goes wrong:**
Verified: `validate_sequences` (data.py:842-860, via `check_sequence` sequence.py:89-124) drops WHOLE sequences containing any char outside `valid_chars`; 13 models reject N (`"ACGTacgt"` strict charset), so each model silently evaluates a different subset of the same dataset — cross-model tables compare different data. The D3 ruling makes the suite side docs-only, which is right, but two follow-on traps: (a) the warning exists only in a docstring nobody reads while the benchmark JSONs show per-model row-count differences nobody notices; (b) REV-03 edits docs → `scripts/check_docs_sync.py` byte-identical mirror enforcement applies; a hastily edited `docs/` page without the mirror resync fails CI (v1.1 precedent).

**Prevention:** docstring + API-docs warning in the same change; add a one-line runtime affordance — log the dropped-row count when `validate_sequences` filters (a log line, not a new API, respecting the D3 ruling). Docs changes go through the established mirror-resync flow. Warning sign: benchmark JSON row counts differing across models for the same dataset (reviewer-visible); docs-validation CI red.

**Phase to address:** Phase A (REV-03).

---

### Pitfall 10 (REV-07): Probe leakage, embedding-cache staleness, and pooling-choice sensitivity silently flip probing conclusions

**What goes wrong:**
(a) `fit_probe` with "dev early stopping" (intake) leaks if the dev split used for probe early-stopping is the same rows later used to report probe metrics — or worse, if embeddings are extracted from a dataset that was itself filtered per-model (REV-03 interaction). (b) The npz embedding cache keyed only by model+dataset goes stale when pooling/layer parameters change — a second run "hits cache" with the WRONG embeddings (the intake acceptance "cache second-run hit" invites a key that omits the pooling/layer dims). (c) Pooling choice (mean vs last vs CLS) changes probing conclusions; if the default differs from what the finetuning comparison (F4 lane) assumes, probe-vs-finetune tables compare different representations.

**Prevention:** probe split discipline enforced in code (fit on train, early-stop on dev, report on test — assert disjoint row counts in tests); cache key = hash(model_id, dataset_fingerprint, layer, pooling) with a unit test that changing pooling MISSES the cache; pooling recorded in every output row and pinned in the F4 export schema. Warning signs: probe metrics ≥ fine-tuned metrics (leakage smell); cache-hit test passing while a pooling-change test is absent; output tables missing the pooling column.

**Phase to address:** Phase B wave B3 (REV-07, `inference/probing.py`).

---

### Pitfall 11 (REV-10): PWM matching with uniform background, raw log-odds thresholds across motifs, per-sequence FDR, and untested strand conventions

**What goes wrong:**
Verified FIMO/JASPAR conventions that naive implementations violate: (a) uniform 0.25 background inflates hits on GC-rich DNA — background must be zero-order Markov matched to target GC (`fasta-get-markov` practice); (b) log-odds thresholds are NOT comparable across motifs (different score ranges) — p-values via the exact null distribution (dynamic programming) are the cross-motif-comparable currency, BH q-values for FDR over the FULL window×motif test set (millions of tests — per-sequence correction is wrong); (c) strand: default practice scans both strands (`--norc` opts out); an implementation that scans only the given strand halves sensitivity, and a revcomp-convention mixup (motif given on its stated strand) double-counts or misses; (d) pseudocount convention (0.1 scaled by background) affects short-matrix stability.

**Prevention:** implement the FIMO defaults as documented constants with provenance comments; unit tests: a synthetic GC-skewed sequence where uniform-vs-matched background changes the hit set (assert the difference is flagged/logged); strand test with a palindromic and a non-palindromic motif; golden test = the HBG1/BCL11A case matching paper Fig 4a coordinates (the intake acceptance — freeze it as a regression fixture).

**Warning signs:** motif hit counts scaling linearly with window count without FDR control; identical hits on + and − strands for non-palindromic motifs (revcomp bug); HBG1 golden test absent.

**Phase to address:** Phase C (REV-10, `interpret/motifs.py`).

---

### Pitfall 12 (REV-11): New MCP tools skip the timeout wrapper, make blocking sync calls inside the asyncio server, or raise across the protocol boundary — and the `--host/--port` fix breaks the streamable-http case

**What goes wrong:**
Verified server conventions that new tools must inherit, each with a concrete bypass route: (a) `_with_timeout_wrapper` (server.py:295-354) wraps every non-streaming tool via `asyncio.wait_for` and returns an error DICT — a new tool registered bare (`self.app.tool()(self._ism_scan)`) gets no timeout and no structured error; the registration block at 263-292 is the pattern to extend, and its "streaming tools handle timeout internally" comment means long-running ISM scans (>6000 forward passes for 2kb) should probably be streaming/chunk-based like the existing stream tools rather than wait_for'd; (b) ISM/mutagenesis/VEP scoring are heavy SYNCHRONOUS torch calls — calling them directly in an async tool body stalls the event loop so even `health_check` stops responding; the established bridge is `loop.run_in_executor` with the lock held INSIDE the executor closure (model_manager.py:99-121, whose comments document the orphaned-24.5GB-server lesson); (c) the protocol boundary rule is error-dicts-not-raises — a `raise` inside a tool surfaces as an MCP protocol error and breaks the client contract tests; (d) the known `--host/--port` bug: config silently overrides CLI args (server.py:1743-1747), but the streamable-http path has DIFFERENT conditional precedence (1854-1857) — a naive "CLI wins" fix applied to one path desyncs the other. Also: `zero_shot_score` inherits REV-08 skip semantics — skip counts must appear in the tool result payload so agents can interpret partial scoring.

**Prevention:** handshake regression tests per new tool (server up → client calls tool → JSON assertions — the v1.1 live-server-probe precedent, both transports); an event-loop liveness test (concurrent `health_check` during a long tool call must succeed within a bound — catches blocking-in-async without reading the diff); a registration-coverage structure test asserting every tool in a known list is timeout-wrapped; host/port precedence tests for BOTH transports (`stdio` unaffected, `sse` and `streamable-http` asserting CLI > config > default).

**Warning signs:** review diff adding `self.app.tool()(self._x)` without the wrapper; tests calling tool functions directly instead of through a live/fixture server; timeout errors mentioning tools that were supposed to be streaming.

**Phase to address:** Phase C (REV-11).

---

### Pitfall 13 (Milestone-level): The 90% hard gate will NOT catch under-tested new modules — the real bar is the 96%+ working standard plus the owner's same-change rule

**What goes wrong:**
Arithmetic: the denominator is ~7.4k stmts at 96.42%; a 400-line `vep.py` merged with zero tests drops coverage to ~94.9% — still comfortably above `fail_under=90`, so CI stays green while the milestone silently gives back a point and a half of the v1 hardening. The gate is a floor, not the standard. (Also: a module accidentally placed under an omit glob — Pitfall 2 — doesn't even appear in the arithmetic.)

**Prevention:** every plan's UAT includes per-module coverage (`pytest --cov=dnallm.inference.vep ...` scoped run — the repo already documents `--no-cov` for scoped runs, so the inverse pattern is established) plus the owner rule (memory: any dnallm/ change ships with pytest coverage in the same change). The verifier for each phase should reproduce the module-level number, not just the global gate.

**Warning signs:** nightly coverage number trending down across waves; a merged module with no matching `tests/<pkg>/test_<module>.py`.

**Phase to address:** All Phase B/C plans (checklist item per plan); enforced at phase verification.

---

### Pitfall 14 (Milestone-level): Tests accidentally pin peft/transformers behaviors that differ across the supported version span (transformers 4.49–5.x, peft ≥0.14)

**What goes wrong:**
Every new PEFT/model path inherits the version-span constraint ("tests must not pin to a single transformers minor" — pyproject/CI matrix). Concrete verified examples in scope: peft's IA³×4-bit behavior differs by version (older: `NotImplementedError` at injection; 0.21.2: injection OK, merge raises); peft's no-match target-module handling shifted between warn-and-continue and raise; transformers 5.x changed `warmup_ratio` handling (already patched at trainer.py:203-205, 243-254 — proof the class of problem is real here) and tied-weight `post_init` semantics. A test asserting a foreign library's exact exception type/message passes on the dev env (transformers 5.19) and fails the matrix leg or a future bump.

**Prevention:** assert on DNALLM's OWN error surface (`pytest.raises(ValueError, match="<our message>")` from our config-validation guards — Pitfall 3's strategy) rather than peft/transformers internals; feature-detect library capabilities instead of version-gating; where a matrix-leg difference is unavoidable, prefer parametrized expectations over `xfail`-spray. Warning sign: any new test importing a private peft/transformers symbol or matching a foreign exception string.

**Phase to address:** B1/B2 plans primarily (PEFT and model-loading clusters); review checklist for all.

---

### Pitfall 15 (Milestone-level): Nightly-leg realities — new real-model tests need models.lock rows, `slow` marking, and budget/census awareness

**What goes wrong:**
REV-04/05/06/08 acceptances require real models (IA³×Mamba, VEP ×5 models, two-architecture random-init). Dropped into the fast PR leg they pull network downloads into every PR run (and typed network skips into the allowlist); dropped into nightly without `models.lock` rows they break the consistency guard (CI-08 precedent: drift-injection-proven); added in bulk they blow the measured D-12 budgets and the staged-serial D-07 VRAM discipline (≥35Gi floor). Separately, the D-03 example-lane census triple (208/217 hard-asserted) is only disturbed if REV work adds `example/` artifacts — if any REV adds an example notebook or YAML, the triple needs a deliberate same-change re-pin (2026-10-07 precedent).

**Prevention:** every real-model test is `slow`-marked from birth; every newly-referenced model id gets a `models.lock` row (sha-pinned, prefix↔source aligned) in the same change; nightly budget deltas estimated before the wave merges; example/ additions explicitly re-pin D-03. Warning signs: fast leg runtime creeping up; `audit_skips.py` failures on new network skips; nightly OOM at the VRAM floor.

**Phase to address:** B1–B4 and C plans (checklist per plan); CI-facing items at phase verification.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-Term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Emitting metric aliases "temporarily" for backward compat | Old consumers keep working | Re-creates the R1-2d drift chain the registry exists to kill | Never — recognition only (Pitfall 2) |
| Guessing PEFT target_modules from family name conventions | Presets table done in an hour | Silently frozen models invalidating the IA³ baseline (Pitfall 4) | Never — derive from real configs |
| Global single-hash proof for random_init | One quick test | Weak proof; leftover pretrained tensors undetected (Pitfall 5) | Never as sole proof; per-tensor comparison required |
| Scoring ref/alt via independent tokenization | Reuses `clm_evaluate` unchanged | Tokenization noise mixed into variant effects; reviewer challenge unanswered (Pitfall 6) | Never for VEP; fine for whole-sequence scoring unrelated to variants |
| Registering new MCP tools without the timeout wrapper | Tool works in happy path | Event-loop stalls, protocol errors, unbounded runs (Pitfall 12) | Only for streaming tools that own their chunk timers |
| Skipping `expected_skips.yaml` update "until CI tells me" | Faster local loop | 4 CI legs red for everyone; audit gate failure (Pitfall 8) | Never |
| Bootstrap CI on n=3 seeds | Matches intake spec verbatim | Statistically vacuous precision in the paper (Pitfall 7) | Only with explicit n-guard + t-interval fallback |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HF `Trainer` (transformers 5.19) | Active `eval_strategy` with `eval_dataset=None` "should just skip" | Trainer raises `ValueError` — guard must set both atomically; `metric_for_best_model` silently defaults to `"loss"` when `load_best=True` (verified) |
| `Trainer.evaluate` vs `predict` | Using `evaluate(split="test")` via `trainer.evaluate` | Route test through `trainer.predict` (`infer()` path, trainer.py:502) — `evaluate` evaluates the configured eval set only |
| peft `get_peft_model` | Assuming wrong `target_modules` raises | Silent skip on Mamba/hybrid; assert trainable-param count after attach (Pitfall 4) |
| peft `prepare_model_for_kbit_training` | Calling after `get_peft_model` or skipping for IA³ | Call between `from_pretrained` and `get_peft_model` (trainer.py:159-167 order is correct); IA³+4bit rejected at OUR config layer (Pitfall 3) |
| peft adapter save/reload | Roundtrip tested only for LoRA | IA³ roundtrip has a distinct corruption class (#2429/#2432); test per adapter type |
| transformers random init | `from_pretrained` then re-init | `from_config`; never `from_pretrained` under meta context (raises anti-pattern RuntimeError); check `data_ptr()` ties; expect `trust_remote_code` tied-weight quirks on 5.x |
| HF `datasets` splits | Assuming `train_test_split` is deterministic | Default-seeded only when seed passed — sweep must thread it explicitly (Pitfall 7) |
| FastMCP/asyncio | Blocking torch calls in tool body | `run_in_executor` bridge with lock inside the closure (model_manager.py:99-121 pattern) |
| JASPAR/FIMO ecosystem | Hand-rolled threshold logic | Follow FIMO defaults (p<1e-4, BH q-values, zero-order GC background, both strands); JASPAR PFMs are already MEME-format |
| dnallmmark (companion repo) | Importing dnallm at HEAD | Pin a dnallm version/commit; registry + sweep protocol are dnallm-owned contracts |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `mlm_evaluate`-style per-token masking in VEP | O(L) forward passes per variant; 1k variants × 2kb × 5 models never finishes nightly | Batch variants; restrict masking to the variant slot (alignment already constrains it); mark large runs `slow`/nightly | ~100 variants on CPU-only CI; ~10k variants on GPU nightly |
| MCP `ism_scan` under `asyncio.wait_for` | Timeout error dicts for legitimately long scans; users retry-loop | Chunk-based streaming tool (existing stream-tool pattern owns its timer) or documented sub-scan limits in the error dict's `suggestion` | Sequences >~500bp at default `_tool_timeout_seconds` |
| `prepare_model_for_kbit_training` + gradient checkpointing combos in IA³ tests | Nightly VRAM floor (≥35Gi, D-07) breached | Keep tiny models for logic tests; real-model legs staggered per staged-serial D-07 | Parallel wave merges landing multiple heavy test classes same night |
| Embedding cache without fingerprint key | Silent stale-cache reuse producing wrong probe numbers (Pitfall 10) | Hash(model, dataset, layer, pooling) key | Any run after changing pooling defaults |
| PWM scan over millions of windows in pure Python | Multi-hour scans; nightly budget overrun | Vectorized log-odds via numpy sliding windows (altair-adjacent stack already numpy-heavy); FDR on the full set at once | Genomic-scale windows (>10^6) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting downloaded VCF/PWM files (ClinVar snapshots, JASPAR downloads) | Malicious or corrupted scientific data crashing parsers; path traversal in record IDs used as filenames | Parse with strict validators; never use VCF record fields to construct filesystem paths; treat downloaded files as untrusted input (isolate + validate, per environment policy) |
| Network model downloads inside tool/MCP paths triggered by agent input | Uncontrolled egress + cache flooding from arbitrary model names in `zero_shot_score` args | Restrict to models in the server config registry (existing ModelManager pattern); validate against the configured model list before any download |
| `trust_remote_code=True` families exercised by new random_init paths | Arbitrary code execution surface from third-party model repos (pre-existing, but REV-06 widens the set of invocation paths) | random_init allowlist limited to vetted families; document that unlisted families raise `ValueError` |

## UX Pitfalls (API/docs surface for paper-revision users)

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| `random_init=True` on an unsupported family failing deep inside a handler | Opaque foreign traceback; user thinks their model is broken | Family allowlist + `ValueError("random_init is not supported for {family}; supported: …")` at the load boundary |
| VEP skip reasons hidden in logs | Users report AUROC on an unknown subset and compare to literature | Skip counts + reasons in the returned table AND a summary line; docstring protocol statement (intake requirement) |
| Sweep seed semantics undocumented | Users conflate split variance with init variance when comparing runs | One documented rule (recommended: split fixed across seeds) in `run_seeds` docstring + README protocol section |
| New config flags (`use_ia3`, `allow_test_as_eval`) with silent defaults | Users unknowingly reproduce the exact leak the paper revision fixes | WARN on every override path; Pydantic field descriptions state the danger (configs.py `Field(description=...)` convention) |

## "Looks Done But Isn't" Checklist

- [ ] **REV-01 guard:** early-stopping neighbor path (test-only split + early stopping) has an assertion — often missing while the headline guard test exists
- [ ] **REV-02 registry:** `coverage report` actually shows a measured row for the registry file (not omitted); alias-emission test exists
- [ ] **REV-04 IA³:** trainable-parameter count asserted > 0 AND within family band; IA³ (not just LoRA) save/reload roundtrip; `use_ia3+use_qlora` rejected at config time
- [ ] **REV-05 presets:** every family row dry-run-validated against real module names; provenance (source config.json) recorded per row
- [ ] **REV-06 random_init:** per-tensor hash comparison (not one global hash); same-seed reproducibility; tokenizer-loads-normally assertion; no-download proof
- [ ] **REV-07 probing:** cache-key includes pooling+layer; probe fit/early-stop/report splits asserted disjoint
- [ ] **REV-08 VEP:** at least one test where a variant is explicitly SKIPPED for slot mismatch; skip counts asserted; paradigm guard raises on mismatched model class; VCF coordinate fixtures (0-/1-based, chr-prefix)
- [ ] **REV-09 sweep:** same-seed-identity test (CPU); aggregation n-guard; output JSON schema contract test shared with F2
- [ ] **REV-10 motifs:** GC-matched-vs-uniform background difference demonstrated; HBG1/BCL11A golden coordinates match Fig 4a
- [ ] **REV-11 MCP:** every new tool timeout-wrapped (structure test); event-loop liveness under load; host/port precedence tested on BOTH sse and streamable-http
- [ ] **Every wave:** new skips allowlisted same-change; new models.lock rows; new modules have same-change tests at the 96% working standard

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Test-as-eval shipped in a re-run (REV-01 missed) | HIGH (re-run compute + reviewer trust) | Re-run affected tasks via F1 predict path; add guard + neighbor tests; disclose in response letter |
| Registry drift recurrence (REV-02) | MEDIUM | Exporter maps through `resolve()`; regenerate JSONs; alias table grows one row with provenance comment |
| IA³ baseline invalid (frozen model, REV-04/05) | HIGH (paper table) | Detect via trainable-param logs; re-run IA³ lane with corrected presets; presets provenance audit |
| random_init loaded pretrained partially (REV-06) | MEDIUM | Per-tensor diff pinpoints leftover tensors; fix init path; re-run learning curves; hash log updated |
| VEP misalignment discovered post-hoc (REV-08) | HIGH | Skip-recounted re-score; if tokenizers fundamentally unsuited (BPE families), report as model-class limitation with skip stats — the alignment evidence itself is the response |
| Sweep statistics invalid (REV-09) | LOW | `aggregate_seeds` is pure — recompute from raw seed JSONs; only paper numbers regenerate |
| Merge-conflict sprawl across waves (milestone) | MEDIUM | Rebase onto the Phase A scaffold; file-ownership map arbitrates; worst case serialize the offending wave |
| Coverage slide below working standard (milestone) | LOW | Backfill tests per module (no code changes needed — v1 precedent: wave-based test authoring) |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1 REV-01 eval-semantics + neighbors | Phase A (REV-01) | 3-combo × override tests incl. early-stopping neighbor; `pytest.raises(match=...)` |
| P2 REV-02 registry placement/aliases | Phase A (REV-02) | Coverage row visible; key-set contract test; alias-emission test; dnallmmark pinned import |
| P9 REV-03 comparability docs | Phase A (REV-03) | Docstring/API docs warning present; docs-mirror sync green; filter-count log line |
| P8 parallel-agent conflicts | Phase A scaffold + roadmap wave design | File-ownership map in roadmap; no out-of-ownership diffs; audit_skips green all legs |
| P3 REV-04 IA³×kbit + roundtrip | Phase B wave B1 | Config-time rejection test; IA³ roundtrip; transformer+Mamba coverage |
| P4 REV-05 silent target-module skip | Phase B wave B1 | Per-family dry-run contract tests; trainable-count runtime guard; preset schema test |
| P5 REV-06 random_init proofs | Phase B wave B2 | Per-tensor hash tests; same-seed repro; no-download; family allowlist `ValueError` |
| P6 REV-08 VEP token alignment | Phase B wave B3 | Slot-differ-in-exactly-one assertion; skip-reason fixtures; paradigm guard; VCF fixtures; goldens |
| P10 REV-07 probe leakage/cache | Phase B wave B3 | Disjoint-split asserts; cache-key miss test; pooling pinned in export schema |
| P7 REV-09 seed illusion/stats | Phase B wave B4 | Same-seed identity (CPU); aggregation n-guard; schema contract shared with F2 |
| P13 coverage standard | All B/C plans (per-plan UAT) | Per-module coverage reproduced at verification, not just global gate |
| P14 version-span assumptions | B1/B2 primarily (review checklist all) | No foreign-exception matches in tests; feature-detection over version pins |
| P15 nightly/models.lock/skip discipline | All B/C plans | `slow` marking from birth; lock rows same-change; budgets estimated; D-03 re-pin if example/ touched |
| P11 REV-10 PWM conventions | Phase C (REV-10) | Background/strand/FDR unit tests; HBG1 golden vs Fig 4a |
| P12 REV-11 MCP conventions | Phase C (REV-11) | Handshake tests both transports; liveness test; wrapper structure test; host/port precedence both transports |

## Sources

- Repo source (HIGH confidence, read directly): `dnallm/finetune/trainer.py`, `dnallm/inference/inference.py`, `dnallm/inference/mutagenesis.py`, `dnallm/models/model.py`, `dnallm/tasks/metrics.py`, `dnallm/datahandling/data.py`, `dnallm/utils/sequence.py`, `dnallm/mcp/server.py`, `dnallm/mcp/model_manager.py`, `dnallm/configuration/configs.py`, `pyproject.toml`, `tests/expected_skips.yaml`
- Installed-library source (HIGH for installed versions): peft 0.21.2 `tuners/ia3/model.py` (4/8-bit merge ValueError, feedforward checks), transformers 5.19.0 `TrainingArguments.__post_init__` + `trainer.py` (eval_dataset/load_best validation, silent `metric_for_best_model="loss"` default)
- [peft PR #2432 — IA³ target-module minimization bug fix](https://github.com/huggingface/peft/pull/2432) (MEDIUM)
- [peft IA³ model source v0.17.0](https://github.com/huggingface/peft/blob/v0.17.0/src/peft/tuners/ia3/model.py) — merge limitations (MEDIUM)
- [peft tuners_utils — no-match warning behavior](https://github.com/huggingface/peft/blob/v0.19.0/src/peft/tuners/tuners_utils.py) and [DeepWiki peft troubleshooting — Mamba/hybrid silent skip](https://deepwiki.com/huggingface/peft/6.6-troubleshooting) (MEDIUM)
- [DART-Eval, arXiv:2412.05430 — BPE breaks variant likelihood comparability](https://arxiv.org/html/2412.05430) (MEDIUM)
- [Mut-BPE, bioRxiv 2025.12.01.691503 — split-token alignment scheme](https://www.biorxiv.org/cgi/reprint/2025.12.01.691503v1) (MEDIUM)
- [GPN-MSA, Benegas et al. — LLR scoring convention](https://www.zoology.ubc.ca/~otto/veg/Readings/Benegas2025.pdf) (MEDIUM)
- [transformers meta-device anti-pattern error](https://errors.standardbeagle.com/huggingface/transformers/you-are-using-from-pretrained-with-a-meta-device/) and [transformers PR #43523 — tie weights on meta init](https://semanticdiff.com/gh/huggingface/transformers/commit/00f886a9f435fa552bd2ab93f75c10685d1a9e67) (MEDIUM)
- [transformers PR #33913 — tied-weight load breakage](https://github.com/huggingface/transformers/pull/33913), [issai/Qolda tied-weights discussion](https://huggingface.co/issai/Qolda/discussions/1) (MEDIUM)
- [FIMO — MEME Suite documentation](https://meme-suite.org/meme/doc/fimo.html) — thresholds, background, q-values, strands (MEDIUM)
- [prepare_model_for_kbit_training usage](https://theneuralbase.com/lora-qlora/learn/beginner/trainable-parameter-count-check/) (LOW-MEDIUM, corroborating only)
- Project memory & planning artifacts: `.planning/PROJECT.md`, `.planning/research/261009-paper-revision-suite-plan.md`, owner rules (same-change pytest coverage; no routine cache cleanup), v1/v1.1 milestone precedents (typed skips, models.lock guard, D-03 census triple, D-07/D-12 nightly budgets)

---
*Pitfalls research for: DNALLM v1.2 Paper Revision Suite Support (REV-01…REV-11 additions to an existing DNA-LM toolkit)*
*Researched: 2026-10-09*
