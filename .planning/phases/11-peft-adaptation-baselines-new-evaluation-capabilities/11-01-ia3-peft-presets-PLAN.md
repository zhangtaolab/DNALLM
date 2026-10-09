---
phase: 11-peft-adaptation-baselines-new-evaluation-capabilities
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/configuration/configs.py
  - dnallm/finetune/trainer.py
  - dnallm/inference/inference.py
  - dnallm/configuration/presets/lora_targets.yaml
  - tests/configuration/test_configs.py
  - tests/configuration/test_peft_presets.py
  - tests/finetune/test_trainer.py
  - tests/finetune/test_trainer_real_model.py
  - CHANGELOG.md
autonomous: true
requirements: [PEFT-01, PEFT-02]
coupling_justified: >
  CHANGELOG.md is the single sanctioned cross-lane append surface (milestone invariant).
  This plan appends exactly two unique-anchor bullets (REV-04, REV-05) under the idempotent
  ## [Unreleased] anchor using D-09 discipline (re-read-before-edit, reviewer-comment id
  inline, never touching sibling lanes' entries). No other shared file is touched; all
  other files in files_modified are exclusively owned by this lane (B1 hot files
  configs.py/trainer.py/inference.py per the ROADMAP ownership map).

estimate:
  tokens: 30000
  raw_tokens: 30000
  tasks: 3
  confidence: low   # calibration sample_count=0, factor=1

must_haves:
  truths:
    - "A user can fine-tune with IA³ exactly as with LoRA: finetune.use_ia3=true drives a real trainer branch that injects IA³ vectors via peft get_peft_model (D-01, D-18, PEFT-01)"
    - "use_ia3 × use_qlora is rejected at Pydantic config time with a matchable ValueError naming both fields (PEFT-01 adjacency edge)"
    - "use_lora=True (ctor kwarg) × train_config.use_ia3=True is rejected at trainer-init time with a matchable ValueError naming both flags (PEFT-01 adjacency edge; RESEARCH Open Question 1 resolution — the ctor kwarg is invisible to Pydantic)"
    - "Config-time rejection fires before trainer construction (load_config constructs TrainingConfig first); the trainer-init gate is the second, not the only, line of defense (PEFT-01 ordering edge)"
    - "All rejection tests assert dnallm's own ValueError messages only — never peft/transformers foreign exception text, so the suite stays green across peft 0.14–0.21 and transformers 4.49–5.x (PEFT-01 version-span edge)"
    - "target_modules=None auto-selects a per-family preset from the packaged dnallm/configuration/presets/lora_targets.yaml with a log line naming the preset; a family with no preset row raises a matchable ValueError telling the user to set target_modules explicitly — never a silent fallback to guessed defaults (PEFT-02 boundary edge)"
    - "Every preset row's target_modules and feedforward_modules were derived from real config.json module names (D-01/D-18: peft's real subset field feedforward_modules, explicit per-family lists, attention-value + FFN targets) and the presets-table regression tests fail if any row goes empty (PEFT-02 empty-scan edge)"
    - "peft_dry_run=true resolves the adapter config against the live model's named modules, prints a report, errors on zero matches, and performs no training (D-03)"
    - "After adapter attach, a trainable-parameter-count guard computed directly from requires_grad tensors raises a hard ValueError naming the preset, the expected ratio band, and the actual count (D-04); user-supplied target_modules (no preset band) are guarded against a zero-trainable (fully frozen) model"
    - "One transformer-family model AND one Mamba model each fine-tune one task with IA³ on small models.lock-pinned models (D-02), and the IA³ adapter save→reload roundtrip through DNAInference(lora_adapter=...) reproduces identical outputs on a fixed input (peft #2429 corruption class)"
    - "The shared adapter reload path in inference.py is adapter-kind-agnostic (PeftModel.from_pretrained) with naming/logging that no longer claims LoRA-only"
  artifacts:
    - path: dnallm/configuration/configs.py
      provides: "TrainingConfig.peft_dry_run field, use_ia3 × use_qlora model_validator rejection, Ia3Config field parity with peft 0.21.1 (adds exclude_modules + fan_in_fan_out), refreshed use_ia3 description"
      contains: "peft_dry_run"
    - path: dnallm/finetune/trainer.py
      provides: "IA³ branch mirroring the LoRA branch (interim warn demolished), lora × ia3 trainer-init rejection, preset auto-selection, dry-run validator, trainable-ratio guard"
      contains: "Applying IA³"
    - path: dnallm/configuration/presets/lora_targets.yaml
      provides: "Packaged per-family PEFT target-module presets (~44 benchmark models across PRETRAIN_MODEL_MAPS families) with target_modules, feedforward_modules, lora_r, ratio bands"
    - path: tests/configuration/test_peft_presets.py
      provides: "Presets-table regression tests + dry-run validator tests + ratio-guard tests (network-free)"
    - path: tests/finetune/test_trainer.py
      provides: "TestIa3Wiring mocked fast lane replacing the two interim-warn tests"
    - path: tests/finetune/test_trainer_real_model.py
      provides: "slow-marked IA³ acceptance (transformer + mamba) and IA³ save/reload roundtrip"
  key_links:
    - from: dnallm/finetune/trainer.py
      to: dnallm/configuration/presets/lora_targets.yaml
      via: "target_modules=None → per-family preset lookup via importlib.resources (package-data glob landed in Phase 10)"
      pattern: "presets"
    - from: dnallm/finetune/trainer.py
      to: dnallm/inference/inference.py
      via: "IA³ adapter save_pretrained → DNAInference(lora_adapter=...) → PeftModel.from_pretrained reload (shared, adapter-type-agnostic)"
      pattern: "PeftModel.from_pretrained"
    - from: dnallm/configuration/configs.py
      to: peft IA3Config
      via: "Pydantic Ia3Config field-for-field pass-through into peft.tuners.ia3.IA3Config, including feedforward_modules ⊆ target_modules"
      pattern: "feedforward_modules"
  prohibitions:
    - "No pyproject.toml changes (B5 solely owns pyproject this wave; B1's presets ride the already-landed package-data glob)"
    - "No new dnallm/__init__.py re-exports (facade stays byte-stable)"
    - "No parsing of print_trainable_parameters stdout for the ratio guard — compute sum(p.numel() for p in model.parameters() if p.requires_grad) directly"
    - "No test matching peft or transformers foreign exception strings (version-span rule)"
    - "No private peft/transformers symbol imports in tests"
    - "No edits outside this lane's files (configs.py, trainer.py, inference.py adapter-reload region, presets/, own tests, CHANGELOG append)"
    - "No Co-Authored-By trailers; commits are pathspec-limited (git commit -- <paths>) in the shared tree"
    - "No repo-root configs/ additions — presets live only in packaged dnallm/configuration/presets/"
---

<objective>
B1 (REV-04 + REV-05): IA³ fine-tuning exactly as easy as LoRA, plus per-family PEFT
target-module presets, the dry-run validator, and the trainable-parameter-count guard.

Purpose: PEFT-01/PEFT-02 — reviewers must be able to run IA³ fine-tuning on any
benchmark model without hand-picking target modules, with silent module-skip
(the Mamba/hybrid frozen-model trap) made impossible by two countermeasures.
Output: real IA³ trainer branch replacing the Phase-10 interim warn, packaged
presets YAML, validator + guard, mocked fast-lane tests, slow-lane acceptance
(transformer + Mamba) and IA³ roundtrip, CHANGELOG entries.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-CONTEXT.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-RESEARCH.md
@.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-PATTERNS.md
</context>

<assumption_delta_decision>
The assumption-delta scan fired (pluralization signal: this phase adds a second PEFT
adapter kind alongside LoRA). Decision: no-change. The adapter-kind noun is already
generalized in the config layer (Ia3Config sibling of LoraConfig, both registered in
load_config since Phase 10) and the reload path is adapter-type-agnostic
PeftModel.from_pretrained. Promoting an "AdapterConfig" union type now would churn the
just-landed config surface for no identity-model shift. Recorded for the record; not a
work item.
</assumption_delta_decision>

<tasks>

<task type="tracer">
  <name>Task 1: IA³ end-to-end — config rejection → trainer branch → adapter attach (mocked peft boundary)</name>
  <reversibility rating="costly">The trainer adapter-attach path and its save/reload lifecycle are the contract downstream consumers (DNAInference reload, dnallmmark) build on; undoing a wrong branch shape touches both hot files and the reload seam.</reversibility>
  <files>dnallm/configuration/configs.py, dnallm/finetune/trainer.py, dnallm/inference/inference.py, tests/configuration/test_configs.py, tests/finetune/test_trainer.py</files>
  <read_first>
  - dnallm/finetune/trainer.py (lines 100-250: docstring, __init__, interim use_ia3 warn block at ~174-179, LoRA branch at ~181-197, set_up_trainer pop region at ~222-236)
  - dnallm/configuration/configs.py (lines 263-423: TrainingConfig use_ia3/use_qlora fields, allow_test_as_eval field shape to mirror, LoraConfig Field idiom, Ia3Config stub)
  - tests/finetune/test_trainer.py (TestLoraWiring class ~567-620 incl. test_use_lora_wraps_model_via_peft — the wiring-test shape to replicate; the two interim-warn tests at ~570-599 are the demolition site)
  - tests/configuration/test_configs.py (lines ~990-1010: test_use_ia3_defaults_false stays, test_use_ia3_has_no_cross_field_rejection_yet is the demolition site)
  - tests/conftest.py (lines 160-235: mock_hf_boundary, simple_dna_tokenizer, tiny_model_factory fixtures)
  - 11-RESEARCH.md sections: Pattern 1 (LoRA branch template), Pattern 2 (peft IA3Config verbatim surface), Pitfall 1, Pitfall 10
  - 11-PATTERNS.md: trainer.py IA³ branch analog + configs.py validator analog
  </read_first>
  <action>
  One end-to-end path first: a YAML config with finetune.use_ia3=true and an ia3 section
  carrying explicit target_modules feeds load_config → DNATrainer init → the new IA³
  branch attaches a (mocked) IA³ adapter. Everything else in this lane expands from that
  slice.

  configs.py:
  - Add `TrainingConfig.peft_dry_run: bool = Field(default=False, description=...)` per
    D-03 — description states: validates PEFT target-module resolution against the live
    model and exits before training; handled inside DNATrainer.
  - Add a `model_validator(mode="after")` on TrainingConfig rejecting
    `use_ia3=True and use_qlora=True` with a matchable ValueError whose message names
    both `use_ia3` and `use_qlora` and states IA³ cannot be merged on 4-bit quantized
    models (peft raises only at merge time — Pitfall 1). Cite the config field names in
    the message so `pytest.raises(ValidationError, match="use_ia3")` and
    `match="use_qlora"` both anchor.
  - Refresh the `use_ia3` field description (drop the interim "arrives with the next
    release's trainer branch" wording; describe the real branch).
  - Extend `Ia3Config` to full peft-0.21.1 field parity: add `exclude_modules:
    list[str] | None = None` and `fan_in_fan_out: bool = False` beside the existing
    target_modules / feedforward_modules / init_ia3_weights / modules_to_save
    (D-18: the real peft subset field is `feedforward_modules`; there is no
    feedforward-only boolean in peft — explicit lists per family are the D-01 intent).

  trainer.py:
  - Delete the interim warn block (the `if self.train_config.use_ia3:` print at
    ~174-179) and the comment above it.
  - Insert the trainer-init rejection before either adapter branch:
    `if use_lora and self.train_config.use_ia3:` raises ValueError naming both
    `use_lora` and `use_ia3` (matchable; this is the ctor-kwarg combination Pydantic
    cannot see — RESEARCH Open Question 1 resolution).
  - Add the IA³ branch mirroring the LoRA branch shape at 181-197, minus the kbit prep
    (the rejected combination): `print("[Info] Applying IA³ to the model...")` house
    style; read the section via `config.get("ia3", Ia3Config())` (the section is absent
    when the YAML omits it); construct peft's IA3Config by field pass-through
    (`IA3Config(**ia3_section.model_dump())`); `peft_forward_compatiable(model)` then
    `get_peft_model(model, ia3_config)`; keep `print_trainable_parameters()` for logs.
  - Extend the set_up_trainer pop tuple: `training_args.pop("peft_dry_run", None)`
    beside the existing `use_ia3` pop (extend, do not remove, the pop list).
  - Wire the shared reload seam naming in dnallm/inference/inference.py lines 111-131:
    keep the `lora_adapter` kwarg and `PeftModel.from_pretrained` mechanics untouched;
    make the log line and the ValueError wrap message adapter-kind-agnostic (say "PEFT
    adapter" instead of claiming LoRA-only). This is the only inference.py change.

  Tests (same commit — owner rule):
  - tests/configuration/test_configs.py: replace
    `test_use_ia3_has_no_cross_field_rejection_yet` with rejection tests
    (`pytest.raises(ValidationError, match=...)` for use_ia3 × use_qlora; keep
    `test_use_ia3_defaults_false`); add peft_dry_run default-False + pop-tuple test
    extension; add Ia3Config new-field tests (exclude_modules/fan_in_fan_out defaults).
  - tests/finetune/test_trainer.py: demolish `test_use_ia3_warns_no_effect_yet` and
    `test_use_ia3_default_does_not_warn`; add `TestIa3Wiring` replicating the
    `test_use_lora_wraps_model_via_peft` shape — patch `get_peft_model`, fake PeftModel
    with controllable requires_grad tensors, mock_hf_boundary, patch("builtins.print")
    for the [Info] assertion; add the lora × ia3 trainer-init ValueError test; add a
    test asserting the Pydantic rejection fires at load_config time (before trainer
    construction) — the ordering edge.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/configuration/test_configs.py tests/finetune/test_trainer.py -q -k "ia3"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "error" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - `grep -n "peft_dry_run" dnallm/configuration/configs.py` shows the field definition; `grep -c "peft_dry_run" dnallm/finetune/trainer.py` is >= 1 (the pop)
  - `grep -n "Applying IA³" dnallm/finetune/trainer.py` hits the new branch; the interim-warn string "has no effect yet" no longer appears in trainer.py (`grep -c "has no effect yet" dnallm/finetune/trainer.py` returns 0)
  - `uv run --no-sync pytest tests/finetune/test_trainer.py -q -k "Ia3"` passes with at least 4 tests; `grep -c "test_use_ia3_has_no_cross_field_rejection_yet" tests/configuration/test_configs.py` returns 0
  - TrainingConfig(use_ia3=True, use_qlora=True) raises ValidationError matching "use_ia3"; DNATrainer(use_lora=True) with train_config.use_ia3=True raises ValueError matching "use_lora"
  - inference.py reload region logs/messages contain "PEFT adapter" and the diff touches no line outside the 111-131 region's strings
  </acceptance_criteria>
  <done>IA³ branch is real (mocked attach verified), both incompatible combinations raise matchable dnallm ValueErrors at the correct layer, interim warn and its tests are gone, reload seam naming is adapter-agnostic, all targeted tests green.</done>
</task>

<task type="auto">
  <name>Task 2: Presets table lora_targets.yaml + auto-selection + dry-run validator + trainable-ratio guard</name>
  <files>dnallm/configuration/presets/lora_targets.yaml, dnallm/finetune/trainer.py, tests/configuration/test_peft_presets.py</files>
  <read_first>
  - dnallm/models/modeling_auto.py (PRETRAIN_MODEL_MAPS — the family keys the presets table must cover)
  - dnallm/models/model_info.yaml (registry of the ~238 models; the ~44 benchmark-model subset the presets derive from)
  - dnallm/finetune/trainer.py (Task 1's IA³ branch + the LoRA branch — the preset resolution slots ahead of both LoraConfig/IA3Config constructions)
  - 11-RESEARCH.md sections: Pitfall 3 (silent skip evidence, SSM target literature), Assumption A3 (in_proj/out_proj/x_proj/dt_proj are literature-informed starting points, derivation is from real config.json), Assumption A5 (bands pinned at derivation time)
  - 11-PATTERNS.md: lora_targets.yaml analog (model_info.yaml packaged-registry pattern) + the D-04 ratio-guard code example in 11-RESEARCH.md Code Examples
  </read_first>
  <action>
  Derive and package the presets, then wire the three countermeasures.

  Presets derivation (offline, dev-time network for config.json reads is fine — the
  TESTS stay network-free):
  - For each PRETRAIN_MODEL_MAPS family, read a real exemplar's `config.json` module
    names (local HF/ModelScope cache first; hub fetch for uncached families) and record
    per family in `dnallm/configuration/presets/lora_targets.yaml`:
    `target_modules` (attention key/value + FFN linear names as they actually appear,
    e.g. BERT-family query/key/value/dense; Mamba-family in_proj/out_proj/x_proj/dt_proj
    — A3 literature-informed, but the shipped names must come from the real
    config.json/module names, never guessed — D-01), `feedforward_modules` (the FFN
    subset, per D-18), `lora_r` (recommended r, 8 default), `ia3_ratio_band` and
    `lora_ratio_band` (`[lo, hi]` floats pinned from the real module counts — A5).
  - Cover ~44 benchmark models across the families (the model_info.yaml /
    benchmark-config model set); every PRETRAIN_MODEL_MAPS family key gets a row.
  - Keep the YAML hand-reviewable: one top-level `families:` mapping, family keys
    matching PRETRAIN_MODEL_MAPS names.

  Loader + auto-selection (trainer.py):
  - Load the presets once via importlib.resources on
    `dnallm/configuration/presets/lora_targets.yaml` (the package-data glob landed in
    Phase 10 — do NOT touch pyproject).
  - When the effective adapter config has `target_modules=None` (Ia3Config or
    LoraConfig), resolve the model's family (match the model/config against the preset
    keys) and inject the preset's target_modules (+ feedforward_modules for IA³) with a
    `print("[Info] IA³/LoRA preset '<family>' selected ...")` line (PEFT-02 log-line
    requirement). No preset row for the family → matchable ValueError naming the family
    and telling the user to set target_modules explicitly (fail loud, never guessed
    defaults — boundary edge).
  - Apply auto-selection to BOTH adapter kinds (one resolution path before
    LoraConfig/peft-IA3Config construction) — LoRA's current None-behavior (peft
    internal defaults, the silent-skip hazard) is thereby also fixed.

  Dry-run validator (D-03, inside the trainer):
  - When `train_config.peft_dry_run=True` (IA³ or LoRA): resolve the final adapter
    config, match target_modules against the live model's `named_modules()`, print a
    `[Info]` report (matched module names + count), raise a matchable ValueError when
    zero modules match (wrong names — the peft silent-skip countermeasure), and on
    success print "[Info] PEFT dry run complete — no training performed." and skip the
    training loop entirely.
  - Pop `peft_dry_run` before TrainingArguments (Task 1 already added the pop).

  Trainable-ratio guard (D-04, after get_peft_model in both adapter branches):
  - Compute trainable/total directly from requires_grad tensors (never parse
    print_trainable_parameters output). With a preset active, enforce the preset's
    ratio band; with user-supplied target_modules (no band), enforce ratio > 0 (a fully
    frozen model is always wrong). Out of band → hard ValueError whose message names
    the preset, the expected band, and the actual count/ratio, and says a silent
    module-skip is the likely cause (matchable).

  Tests — tests/configuration/test_peft_presets.py (network-free):
  - Presets-table regression: YAML loads via importlib.resources; every
    PRETRAIN_MODEL_MAPS family key present; every row's target_modules non-empty
    (empty-scan edge); feedforward_modules ⊆ target_modules; bands present with
    lo <= hi; lora_r >= 1.
  - Dry-run validator: fake module trees (match / zero-match ValueError / report
    contents).
  - Ratio guard: fake PeftModel with controllable requires_grad tensors — in-band
    passes, below-band raises the D-04 message, zero-trainable raises.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/configuration/test_peft_presets.py tests/finetune/test_trainer.py -q -k "TestPeftPresets or TestPeftDryRun or TestTrainableRatioGuard or Ia3 or Lora"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - `dnallm/configuration/presets/lora_targets.yaml` exists, is valid YAML, loads via importlib.resources inside the package, and its family-key set equals PRETRAIN_MODEL_MAPS' keys
  - `uv run --no-sync pytest tests/configuration/test_peft_presets.py -q` passes with >= 10 tests covering table regression, dry-run, and the guard
  - trainer.py contains the preset-selection [Info] log line and the ratio-guard ValueError; `grep -c "print_trainable_parameters()" dnallm/finetune/trainer.py` is unchanged from Task 1 (print kept for logs, guard computed independently)
  - A family absent from the presets table with target_modules=None raises ValueError (test-proven), and a zero-match dry run raises ValueError (test-proven)
  - No pyproject.toml byte changed by this lane (`git diff --stat HEAD -- pyproject.toml` empty over this lane's commits)
  </acceptance_criteria>
  <done>Presets packaged and regression-tested; target_modules=None auto-selects with a log line or fails loud; dry-run validates against live module names; ratio guard fails hard on silent skip; fast lane fully green.</done>
</task>

<task type="auto">
  <name>Task 3: Slow-lane IA³ acceptance (transformer + Mamba) + IA³ roundtrip + CHANGELOG</name>
  <precondition>models.lock-pinned small models zhangtaolab/plant-dnabert-BPE (ms) and zhangtaolab/plant-dnamamba-BPE-open_chromatin (ms) are cached locally or the slow-lane network route is available (typed network skips guard the tests otherwise).</precondition>
  <files>tests/finetune/test_trainer_real_model.py, CHANGELOG.md</files>
  <read_first>
  - tests/finetune/test_trainer_real_model.py (existing slow-lane LoRA tests — the shape to mirror: markers, model loading, tiny epoch counts)
  - models.lock (pinned rows for plant-dnabert-BPE, plant-dnabert-BPE-promoter, plant-dnamamba-BPE-open_chromatin, dataset plant-multi-species-core-promoters)
  - 11-RESEARCH.md: Pitfall 2 (peft #2429 roundtrip class), D-02 (small pinned models), Validation Architecture lane test strategy
  - .planning/research/261009-paper-revision-suite-plan.md (intake doc — the reviewer-comment ids to cite inline in the CHANGELOG entries for REV-04 and REV-05)
  - CHANGELOG.md (existing ## [Unreleased] anchor and entry format)
  </read_first>
  <action>
  Slow-lane acceptance (all tests `slow`-marked, following the file's existing
  real-model conventions):
  - IA³ fine-tune of one task on the transformer-family model
    `zhangtaolab/plant-dnabert-BPE` (modelscope route, models.lock row exists) with
    finetune.use_ia3=true, ia3.target_modules=None (exercising the preset
    auto-selection on a real backbone), tiny epoch/steps budget on the
    `zhangtaolab/plant-multi-species-core-promoters` dataset (D-02: pinned, fast).
    Assert training completes and the trainable ratio falls inside the preset band.
  - IA³ fine-tune on the Mamba model
    `zhangtaolab/plant-dnamamba-BPE-open_chromatin` (modelscope route) with the same
    shape. This is the guard's proving ground: pre-presets, IA³ on Mamba silently
    froze the backbone (Pitfall 3) — assert the ratio band again.
  - IA³ adapter save→reload roundtrip (peft #2429 corruption class, Pitfall 2):
    save_pretrained the IA³ adapter from the transformer run, reload via
    `DNAInference(..., lora_adapter=<saved path>)`, assert output identity on a fixed
    input sequence (logits equal within tolerance) — the roundtrip must be IA³-specific,
    not the existing LoRA roundtrip.
  - Add models.lock rows only if a new artifact was actually fetched (the two models
    are already pinned; expected delta: none).

  CHANGELOG (D-09 discipline):
  - Re-read CHANGELOG.md, then append exactly two bullets under the idempotent
    `## [Unreleased]` anchor (new `### Added` entries if the section is absent):
    one tagged `(REV-04, <reviewer-comment-id>)` for IA³ support and one tagged
    `(REV-05, <reviewer-comment-id>)` for the presets/validator/guard — reviewer ids
    from the intake doc, cited inline. Unique anchors; never touch sibling lanes'
    entries.
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_trainer_real_model.py -q -m slow -k "ia3"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line, or "failed" in the summary line</fails_when>
    <automated>test "$(grep -c "(REV-04," CHANGELOG.md)" -ge 1 && test "$(grep -c "(REV-05," CHANGELOG.md)" -ge 1 && echo CHANGELOG-OK</automated>
    <fails_when>CHANGELOG-OK absent from output (non-zero exit)</fails_when>
  </verify>
  <acceptance_criteria>
  - One transformer AND one Mamba IA³ fine-tune complete on pinned small models with in-band trainable ratios (slow tests green)
  - IA³ save→reload roundtrip asserts output identity on a fixed input (distinct from the LoRA roundtrip test)
  - `grep -c "(REV-04," CHANGELOG.md` >= 1 and `grep -c "(REV-05," CHANGELOG.md` >= 1, both under ## [Unreleased]
  - Per-module scoped coverage (phase gate): `uv run --no-sync coverage run -m pytest tests/finetune/test_trainer.py tests/configuration/test_peft_presets.py -q -m "not slow"` then `uv run --no-sync coverage report --include="dnallm/finetune/trainer.py,dnallm/configuration/configs.py"` shows both >= 96%
  </acceptance_criteria>
  <done>PEFT-01/PEFT-02 acceptance proven on real models (transformer + Mamba), IA³ roundtrip guarded, CHANGELOG evidence chain extended, lane files at the 96% per-module standard.</done>
</task>

</tasks>

<artifacts_produced>
Symbols this plan creates (B1 lane):
- `TrainingConfig.peft_dry_run: bool` (configs.py)
- `TrainingConfig` model_validator: use_ia3 × use_qlora rejection (configs.py)
- `Ia3Config.exclude_modules`, `Ia3Config.fan_in_fan_out` (configs.py)
- `dnallm/configuration/presets/lora_targets.yaml` — packaged per-family presets (families mapping: target_modules, feedforward_modules, lora_r, ia3_ratio_band, lora_ratio_band)
- Trainer internals (trainer.py, private unless named): IA³ branch, lora × ia3 init rejection, preset auto-selection + loader, dry-run validator, trainable-ratio guard
- Test classes: `TestIa3Wiring` (tests/finetune/test_trainer.py), `TestPeftPresets` / `TestPeftDryRun` / `TestTrainableRatioGuard` (tests/configuration/test_peft_presets.py), IA³ slow acceptance + roundtrip tests (tests/finetune/test_trainer_real_model.py)
- inference.py: adapter-agnostic reload log/error strings (no new symbols)
</artifacts_produced>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| packaged YAML → trainer | presets file is packaged data read at runtime; a corrupted/edited wheel contents file feeds module names into peft |
| user YAML config → Pydantic | untrusted-by-default user input validated at the config boundary (V14) |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-01 | Tampering | IA³/LoRA preset injection in trainer.py | high | mitigate | Wrong or silently non-matching target modules freeze the backbone and produce wrong scientific results undetected: D-04 hard ValueError ratio guard + dry-run zero-match error + presets-table regression tests (three independent countermeasures) |
| T-11-02 | Elevation of Privilege | TrainingConfig combination handling | medium | mitigate | use_ia3 × use_qlora explodes late inside peft with version-dependent foreign errors (merge-time ValueError/NotImplementedError): Pydantic-time matchable rejection (Pitfall 1) |
| T-11-03 | Tampering | presets YAML loading | medium | mitigate | Presets loaded via importlib.resources from the package (not CWD-relative paths); structure validated at load (family keys, non-empty lists, subset property) with regression tests |
| T-11-SC | Tampering | package installs | high | mitigate | This lane installs nothing (peft is an existing dependency; zero pyproject changes). The phase's only sanctioned install (scikit-allel) is owned by plan 11-05 under owner decision D-08 with the bounded range >=1.3.13,<2 |
</threat_model>

<verification>
- Fast lane: `uv run --no-sync pytest tests/configuration/test_configs.py tests/configuration/test_peft_presets.py tests/finetune/test_trainer.py -q -m "not slow"` — all green, no new network dependence
- Slow lane: `uv run --no-sync pytest tests/finetune/test_trainer_real_model.py -q -m slow -k "ia3"` — transformer + Mamba acceptance + roundtrip
- Per-module coverage (cov-crash workaround — never `pytest --cov`): `uv run --no-sync coverage run -m pytest tests/finetune/test_trainer.py tests/configuration/test_peft_presets.py -q` then `uv run --no-sync coverage report --include="dnallm/finetune/trainer.py,dnallm/configuration/configs.py"` — both >= 96%
- Invariants: `git diff HEAD -- pyproject.toml` empty for this lane's commits; `git diff HEAD -- dnallm/__init__.py` empty; `grep -c "(REV-04," CHANGELOG.md` and `grep -c "(REV-05," CHANGELOG.md` each >= 1
- Owner directive 2026-10-09: run ONLY the targeted verifiers above and this lane's test files — no repo-wide lanes, no check_code.py full sweeps
</verification>

<success_criteria>
- PEFT-01: IA³ fine-tuning works symmetrically to LoRA (branch, save/reload shared path, config-time rejections, transformer + Mamba acceptance, IA³ roundtrip) — all requirement clauses test-proven
- PEFT-02: presets table packaged + regression-tested, auto-selection with log line, dry-run validator, ratio guard — all requirement clauses test-proven
- Every dnallm/ behavior change shipped with its tests in the same commit; lane hot files at >= 96% per-module coverage
- No new dependencies, no facade re-exports, no out-of-lane file edits, CHANGELOG entries traceable
</success_criteria>

<output>
Create `.planning/phases/11-peft-adaptation-baselines-new-evaluation-capabilities/11-01-SUMMARY.md` when done
</output>
