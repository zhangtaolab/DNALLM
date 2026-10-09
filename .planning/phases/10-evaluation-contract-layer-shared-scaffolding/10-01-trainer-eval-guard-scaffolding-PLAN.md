---
phase: 10-evaluation-contract-layer-shared-scaffolding
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - dnallm/finetune/trainer.py
  - dnallm/configuration/configs.py
  - pyproject.toml
  - tests/finetune/test_trainer.py
  - tests/configuration/test_configs.py
  - CHANGELOG.md
autonomous: true
requirements: [EVAL-01]
user_setup: []
estimate:
  tokens: 34000
  raw_tokens: 34000
  tasks: 3
  confidence: low   # 0 calibration samples (first phase of v1.2); factor 1 per estimate-calibration

must_haves:
  truths:
    # EVAL-01 core (goal-backward from ROADMAP SC 1-2, 5)
    - "Constructing DNATrainer over a DatasetDict with train+test but NO dev split and default config yields Trainer(eval_dataset=None) with training_args.eval_strategy=='no' — the test split is never the eval set (per D-04, EVAL-01)"
    - "The flip WARN fires exactly ONCE at guard time (set_up_trainer), never per eval step, and its single line states all three facts: test split excluded from evaluation, behavior differs from previous dnallm versions, opt-in via allow_test_as_eval=true"
    - "TrainingConfig.allow_test_as_eval=True (default False, YAML-configurable, consequence documented in the field description) restores test-as-eval with a loud WARN naming the leak risk (per D-05)"
    - "When the guard fires with user-set load_best_model_at_end=True (via finetune YAML or extra_args), construction raises a matchable ValueError; it never silently downgrades a user setting"
    - "When early stopping (callbacks.early_stopping.patience set) is configured and no eval dataset exists (test-only guarded, or train-only), construction raises a matchable ValueError naming both remedies; the trainer.py:298-303 force-enable path can no longer resurrect best-model selection without an eval set (PITFALLS #1 neighbor path)"
    - "allow_test_as_eval=True combined with user-set load_best_model_at_end=True or early stopping does NOT raise — the user opted in, an eval set exists (test), selection proceeds with the WARN"
    - "train-only and unsplit datasets keep today's behavior: eval_strategy='no', eval_dataset=None, no WARN (empty-input edge)"
    - "DNATrainer.evaluate(split=...) accepts any split key present in the DatasetDict (discretion decision), raises matchable ValueError for an absent key listing available splits, and routes an accepted split through self.trainer.predict — never trainer.evaluate — on the held-out data (per D-01)"
    - "evaluate(split=...) returns a metrics dict keyed by unprefixed canonical metric spellings (registry-canonical by construction: 'accuracy', 'AUROC', ...) and writes a result JSON {split, timestamp, metrics} under output_dir (per D-02, schema consumable by Phase 11 aggregate_seeds)"
    - "evaluate(split=...) evaluates the weights the trainer currently holds — no checkpoint reload, no checkpoint parameter (per D-03); the model-selection rule is written in the DNATrainer class docstring"
    - "evaluate() called with legacy HF kwargs (eval_dataset=, ignore_keys=, metric_key_prefix=) forwards them to self.trainer.evaluate unchanged; evaluate() with no args still calls self.trainer.evaluate() with no kwargs (signature compatibility for external HF-ecosystem callers, per D-01)"
    # Scaffolding (SC 5, D-06, D-07)
    - "TrainingConfig.use_ia3 lands field-first (default False, full description) with NO cross-field validators — use_ia3 x use_qlora and lora x ia3 rejections stay in Phase 11 B1 (per D-07 interim-window rule)"
    - "Ia3Config, VepConfig, SweepConfig are field-complete Pydantic sections (final fields, defaults, pattern/ge validations) registered in DNALLMConfig and load_config() so Phase 11 agents B4/B5 never touch configs.py (per D-06)"
    - "training_args.pop covers allow_test_as_eval and use_ia3 before TrainingArguments(**kwargs) — neither field ever reaches TrainingArguments (TypeError guard)"
    - "pyproject.toml gains only the [tool.setuptools.package-data] entry shipping dnallm/configuration/presets/*.yaml; every dependency list is byte-identical — Phase 10 adds zero dependencies (the milestone's single approved addition, scikit-allel for VEP-01 VCF reading, lands in Phase 11 B5 per the updated roadmap invariant)"
    - "dnallm/__init__.py is untouched (byte-stable facade, no new re-exports)"
    - "trainer.py and configs.py docstrings carry no pre-sweep terminology ('DNA language model' variants) — owner-side sweep per D-08 (A1 owns these two files)"
    - "The trainer class docstring documents eval-set selection and held-out semantics including the D-03 model-selection wording"
  artifacts:
    - dnallm/finetune/trainer.py  # guarded split selection, evaluate(split=...) override, docstrings
    - dnallm/configuration/configs.py  # allow_test_as_eval, use_ia3, Ia3Config, VepConfig, SweepConfig, load_config registration
    - pyproject.toml  # package-data presets entry only
    - tests/finetune/test_trainer.py  # inverted leak test + guard/collision/evaluate suites
    - tests/configuration/test_configs.py  # stub-section and field tests
    - CHANGELOG.md  # REV-01 Unreleased entry in the same commit as the guard
  key_links:
    - "YAML finetune.allow_test_as_eval -> TrainingConfig field -> DNATrainer.set_up_trainer guard decision (the leak site trainer.py:234-241)"
    - "DNATrainer.evaluate(split=...) -> self.trainer.predict(dataset[split]) -> prefix-stripped canonical-keyed metrics -> output_dir/eval_{split}_result.json"
    - "TrainingConfig.model_dump() -> pop list (allow_test_as_eval, use_ia3) -> TrainingArguments(**kwargs)"
    - "load_config() section blocks -> DNALLMConfig keys ia3/vep/sweep -> Phase 11 consumers"
  prohibitions:
    - "The guard must NOT auto-create or carve a dev split out of train/test (FEATURES anti-feature for REV-01)"
    - "evaluate(split=...) must NOT reload checkpoints or accept a checkpoint parameter in v1.2 (D-03)"
    - "The guard must NOT silently downgrade user-set load_best_model_at_end or early-stopping settings — collisions raise ValueError (EVAL-01)"
    - "evaluate(split=...) must NOT route the test split through trainer.evaluate — predict path only (PITFALLS #1)"
    - "Phase 10 must NOT add the use_ia3 cross-field rejection validators (D-07 — Phase 11 B1 owns them)"
    - "No new dnallm/__init__.py re-exports and no new skips (no expected_skips.yaml rows expected; if one becomes necessary it is typed + allowlisted in the same change)"
---

<objective>
Agent lane A1 (owner-fixed wave structure): land the EVAL-01 evaluation-semantics contract in the trainer — the test-as-eval leak can never happen silently, an explicit evaluate(split=...) entry point exists — plus the one-pass shared-file scaffolding (allow_test_as_eval, use_ia3 field-first, Ia3Config/VepConfig/SweepConfig stubs, load_config registration, pyproject package-data) so Phase 11's five agents only fill modules.

Purpose: EVAL-01 (REV-01, reviewer R1-2c) gates the dnallmmark re-run; the scaffolding pass is the single highest-leverage structural decision for parallel safety (PITFALLS #6/#8, research SUMMARY).
Output: guarded trainer.py, scaffolded configs.py, pyproject package-data entry, mocked-boundary fast-lane tests, CHANGELOG REV-01 entry.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-CONTEXT.md
@.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-PATTERNS.md
@.planning/research/SUMMARY.md
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Eval-semantics guard — kill the silent test-as-eval flip (trainer.py:234-241) with allow_test_as_eval opt-in</name>
  <files>dnallm/finetune/trainer.py, dnallm/configuration/configs.py, tests/finetune/test_trainer.py</files>
  <read_first>
  - dnallm/finetune/trainer.py (whole file — set_up_trainer at 177-315, leak site 234-241, pop list 197-205, class docstring 67-125)
  - dnallm/configuration/configs.py (TrainingConfig 263-337: use_qlora field at 310-313 is the analog for allow_test_as_eval)
  - tests/finetune/test_trainer.py (fixtures trainer_config/mock_hf_boundary/make_datasets at 36-68; TestDatasetSplitWiring 137-188; the test to INVERT: test_eval_falls_back_to_test_split at 166-173; test_train_only_split_disables_evaluation at 175)
  - .planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-PATTERNS.md (trainer.py section — guard semantics, WARN style, test analogs)
  </read_first>
  <action>
  This is the tracer: config field -> guard -> observable WARN/ValueError -> inverted test, the thinnest full path of REV-01.

  1. configs.py: add `allow_test_as_eval: bool = Field(default=False, description=...)` to TrainingConfig, copying the exact shape of use_qlora (configs.py:310-313). Description states the consequence: enabling lets the test split serve as the eval set for per-step evaluation and best-model selection — metrics are then leaked and must not be reported as held-out performance (per D-05).

  2. trainer.py set_up_trainer: add `training_args.pop("allow_test_as_eval", None)` beside the use_qlora pop (line 200) so the field never reaches TrainingArguments.

  3. Replace the leak branch at 234-241: eval_key selection for dev splits is unchanged. In the `elif "test" in self.data_split:` arm — when `self.train_config.allow_test_as_eval` is False (new default): set eval_dataset=None AND self.training_args.eval_strategy="no" atomically, then emit exactly ONE `print("[Warning] ...")` line (house style per trainer.py:299-302, planner discretion resolved: print style, not get_logger) carrying all three facts (D-04): test split excluded from evaluation; previous dnallm versions silently used it as the eval set; set finetune.allow_test_as_eval: true to opt in. When allow_test_as_eval is True: eval_dataset=test dataset, plus one WARN line naming the leak risk explicitly.

  4. tests/finetune/test_trainer.py: INVERT test_eval_falls_back_to_test_split (line 166) into the guard assertion — eval_dataset kwarg is None, args_cls.return_value.eval_strategy == "no". Add class TestEvalSemanticsGuard: (a) test-only default → guard fires, WARN printed exactly once (patch("builtins.print") and assert the "[Warning]" flip line appears in exactly one call — ordering edge: once at guard time, not per eval step); (b) test-only + trainer_config["finetune"].allow_test_as_eval = True → eval_dataset is datasets.dataset["test"] and the opt-in WARN fires; (c) train-only and unsplit datasets keep today's behavior with NO flip WARN (empty-input edge, extends the two existing tests at 145/175); (d) dev+test unchanged (existing test_eval_prefers_validation_split at 157 stays green untouched).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_trainer.py -q -k "TestEvalSemanticsGuard or TestDatasetSplitWiring"</automated>
    <fails_when>non-zero exit, or the summary line shows "0 passed", or any of the untouched TestDatasetSplitWiring tests fail</fails_when>
  </verify>
  <acceptance_criteria>
  - grep -n "allow_test_as_eval" dnallm/configuration/configs.py dnallm/finetune/trainer.py finds the Field in TrainingConfig, the pop in set_up_trainer, and the guard branch
  - tests/finetune/test_trainer.py contains no assertion that eval_dataset is the test split under default config (the leak assertion is inverted)
  - The flip WARN is one print call containing the substrings "test split", "previous", and "allow_test_as_eval"
  - A test asserts the flip WARN fires exactly once per construction
  </acceptance_criteria>
  <done>Guard + opt-in land with same-change tests; the 3-split x default/override matrix (dev+test / test-only / train-only) is green at the mocked HF boundary.</done>
</task>

<task type="auto">
  <name>Task 2: Neighbor collisions + evaluate(split=...) override with canonical-keyed result JSON</name>
  <files>dnallm/finetune/trainer.py, tests/finetune/test_trainer.py</files>
  <read_first>
  - dnallm/finetune/trainer.py (early-stopping neighbor 286-303; existing evaluate() 489-500; infer() predict path 502-518 — the proven route evaluate(split) must mirror)
  - tests/finetune/test_trainer.py (TestEarlyStopping 247-290 — test_load_best_model_forced_on_when_missing at 269 must stay green for the dev-split case)
  - 10-CONTEXT.md decisions D-01, D-02, D-03
  </read_first>
  <action>
  1. Collision guard (PITFALLS #1): when the guard from Task 1 fires (test present, no dev, allow_test_as_eval False) and `self.training_args.load_best_model_at_end` is True (user set it via finetune YAML or extra_args — check the post-init value so both routes are caught), raise ValueError with a matchable message: load_best_model_at_end requires an evaluation split; provide a dev split or set allow_test_as_eval=true. In the early-stopping block (298-303), when early stopping is configured AND eval_dataset is None (guarded test-only OR train-only), raise ValueError naming early stopping and both remedies — never force-enable load_best_model_at_end without an eval set. The dev+test early-stopping case keeps today's force-enable + WARN behavior unchanged.

  2. Replace DNATrainer.evaluate (489-500) with the signature-compatible override (D-01): `def evaluate(self, split=None, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict[str, float]`. When split is None: legacy passthrough — call self.model.eval() then self.trainer.evaluate(...) forwarding ONLY the non-default legacy kwargs (evaluate() with no args still calls self.trainer.evaluate() with no kwargs; explicit kwargs pass through unchanged). When split is given: raise ValueError(f"Split '{split}' not found in dataset; available splits: {sorted(...)}") unless split is in self.data_split (any present key accepted — discretion decision); else self.model.eval(), route through `self.trainer.predict(self.datasets.dataset[split], ignore_keys=ignore_keys)` (never trainer.evaluate), strip the "test_" predict prefix from predict_result.metrics keys to produce the canonical-keyed dict (canonical = unprefixed current spellings; these are the Phase 10 registry canonical names by the cross-lane contract — no metric_registry import needed), and write output_dir/eval_{split}_result.json containing at least {"split": split, "timestamp": utc isoformat, "metrics": {canonical keys}} (D-02 schema: nested "metrics" block so Phase 11 aggregate_seeds consumes it directly). Evaluates the weights the trainer currently holds — no checkpoint reload, no checkpoint parameter (D-03).

  3. Docstrings: DNATrainer class docstring gains an "Evaluation semantics" passage — eval-set selection rule (dev preferred; test excluded unless allow_test_as_eval), held-out semantics, and the D-03 model-selection wording: "evaluate(split=...) evaluates the model you ended training with — the best checkpoint when load_best_model_at_end or early stopping fired, otherwise final-epoch weights".

  4. Tests: TestEarlyStoppingCollision — (a) test-only + early stopping → pytest.raises(ValueError, match=...); (b) test-only + trainer_config["finetune"].load_best_model_at_end = True → ValueError; (c) test-only + allow_test_as_eval=True + early stopping → no raise (opt-in works, adjacency edge); (d) dev+test + early stopping → existing force-enable behavior stays green. TestEvaluateSplit — (a) split="test" calls trainer.predict with the test dataset and NOT trainer.evaluate; mocked predict returns metrics {"test_accuracy": 0.9, "test_AUROC": 0.8} → evaluate returns {"accuracy": 0.9, "AUROC": 0.8} and eval_test_result.json exists under tmp_path output_dir with split/timestamp/metrics keys; (b) split="nonexistent" → ValueError listing available splits; (c) evaluate() no-args → trainer.evaluate called with no kwargs; (d) evaluate(eval_dataset=obj) → trainer.evaluate called with eval_dataset=obj (signature compatibility per D-01).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/finetune/test_trainer.py -q -k "Collision or Evaluate"</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
  </verify>
  <acceptance_criteria>
  - grep -n "def evaluate" dnallm/finetune/trainer.py shows the split/eval_dataset/ignore_keys/metric_key_prefix signature
  - A test proves trainer.predict (not trainer.evaluate) backs the split path
  - A test proves the result JSON is written with the three required top-level keys and prefix-stripped metric keys
  - The early-stopping + no-dev case raises ValueError (the PITFALLS #1 "most likely to be forgotten" test)
  - The dev-split early-stopping force-enable test (line 269) still passes unmodified
  </acceptance_criteria>
  <reversibility rating="costly">D-01/D-02: the dnallmmark re-run and E1'-E8' experiment scripts consume the evaluate(split=) surface and result-JSON schema; moving them after those exist touches every call site (CONTEXT.md reversibility ratings).</reversibility>
  <done>All EVAL-01 behaviors hold at the mocked boundary: guard, opt-in, both collision ValueErrors, evaluate(split) predict-routing + canonical keys + JSON, signature compatibility.</done>
</task>

<task type="auto">
  <name>Task 3: One-pass scaffolding — use_ia3 field, Ia3Config/VepConfig/SweepConfig stubs, load_config registration, pyproject package-data, CHANGELOG entry</name>
  <files>dnallm/configuration/configs.py, pyproject.toml, tests/configuration/test_configs.py, tests/finetune/test_trainer.py, dnallm/finetune/trainer.py, CHANGELOG.md</files>
  <read_first>
  - dnallm/configuration/configs.py (LoraConfig 340-372 field pattern; TrainingConfig.use_qlora 310-313; validate_report_to 326-337 validator pattern; DNALLMConfig TypedDict 495-510; load_config 513-550)
  - pyproject.toml ([tool.setuptools.package-data] at 264-265 — the entry to extend; confirm no dependency list changes)
  - tests/configuration/test_configs.py (existing validator-test style)
  - tests/finetune/test_trainer.py (test_non_training_arguments_fields_are_popped at 86-101 — the tuple to extend)
  - CHANGELOG.md (header + `## [0.7.1]` section — the Unreleased anchor point)
  </read_first>
  <action>
  1. configs.py, TrainingConfig: add `use_ia3: bool = Field(default=False, description=...)` (D-07 field-first; description notes IA³ training support arrives with the next release's trainer branch). NO cross-field validators (use_ia3 x use_qlora, lora x ia3 stay in Phase 11 B1 — interim window is repo-internal only). trainer.py: `training_args.pop("use_ia3", None)` beside the allow_test_as_eval pop.

  2. configs.py new classes, following LoraConfig field style exactly (Field(default=..., description=...), pattern= for enum-ish strings, ge=/min_length for numerics/lists):
     - Ia3Config mirroring peft.IA3Config's field set (CONTEXT discretion): target_modules: list[str] | None = None, feedforward_modules: list[str] | None = None, init_ia3_weights: bool = True, modules_to_save: list[str] | None = None — each with descriptions.
     - VepConfig (anticipates Phase 11 B5, which never touches configs.py per D-06): paradigm: str = Field(default="mlm", pattern="^(clm|mlm)$"), context_window: int = Field(default=200, ge=1), output_dir: str | None = None.
     - SweepConfig (anticipates Phase 11 B4): seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=1), out_root: str | None = None, n_bootstrap: int = Field(default=2000, ge=1), bootstrap_seed: int = 42, small_n_ci: str = Field(default="t-interval", pattern="^(t-interval|omit)$") — descriptions document the SEED-01 n<10 guard policy.

  3. Register all three: add `ia3: Ia3Config`, `vep: VepConfig`, `sweep: SweepConfig` keys to the DNALLMConfig TypedDict (495-510) and one `if "<section>" in config_dict:` block each in load_config (mirroring the lora block at 542-544).

  4. pyproject.toml [tool.setuptools.package-data]: add `"dnallm.configuration" = ["presets/*.yaml"]` so Phase 11's lora_targets.yaml ships in the wheel. Touch NOTHING else in pyproject.toml (zero-new-dependency invariant; the presets directory itself is Phase 11 PEFT-02 scope — the glob harmlessly matches nothing until then).

  5. Terminology sweep of A1-owned files (D-08 owner-side): replace in dnallm/finetune/trainer.py and dnallm/configuration/configs.py docstrings — "DNA language models" -> "DNA large language models", "DNA language model" -> "DNA large language model", "DNA Language Models" -> "DNA Large Language Models", "DNA Language Model" -> "DNA Large Language Model". No code identifiers change.

  6. Tests: tests/configuration/test_configs.py — Ia3Config/VepConfig/SweepConfig default instantiation (all defaults valid), pattern rejections (VepConfig(paradigm="bogus") and SweepConfig(small_n_ci="bogus") raise ValidationError; SweepConfig(seeds=[]) raises), and load_config tmp_path YAML roundtrips: a YAML with ia3/vep/sweep sections produces configs["ia3"] etc. of the right types; a YAML without them omits the keys. tests/finetune/test_trainer.py: extend the tuple in test_non_training_arguments_fields_are_popped (86-101) with "allow_test_as_eval" and "use_ia3" (TypeError guard proof at the mocked boundary).

  7. CHANGELOG.md (deliberate shared append surface — the one sanctioned cross-lane file, per D-09; re-read immediately before editing): if `## [Unreleased]` is absent, insert it (with `### Added` and `### Changed` subheadings) directly above `## [0.7.1] - 2026-10-08`. Under `### Changed` add one entry: "- Evaluation semantics: the trainer no longer silently evaluates on the test split when no dev split exists; evaluation is disabled with a loud warning unless `finetune.allow_test_as_eval: true` is set, and the new `DNATrainer.evaluate(split=...)` evaluates any split through the predict path writing a result JSON (REV-01, R1-2c)". Commit this entry IN THE SAME COMMIT as the guard work (D-09 mechanism: inline REV-ID + reviewer comment id; SHA backfilled by Phase 12 C3, never same-commit).
  </action>
  <verify>
    <automated>uv run --no-sync pytest tests/configuration/test_configs.py tests/finetune/test_trainer.py -q</automated>
    <fails_when>non-zero exit, or "0 passed" in the summary line</fails_when>
    <automated>grep -c "dnallm.configuration" pyproject.toml && git -C . diff --stat HEAD -- pyproject.toml | tail -1</automated>
    <fails_when>grep count is 0 (package-data entry missing), or the diff stat shows any change outside the package-data block (dependency lists must be untouched)</fails_when>
    <automated>grep -c "(REV-01, R1-2c)" CHANGELOG.md</automated>
    <fails_when>count is 0 — the entry must exist under ## [Unreleased] in the same commit as the guard</fails_when>
  </verify>
  <acceptance_criteria>
  - tests/configuration/test_configs.py contains instantiation, rejection, and load_config roundtrip tests for all three stub sections
  - test_non_training_arguments_fields_are_popped's tuple includes "allow_test_as_eval" and "use_ia3"
  - grep -n "dnallm.configuration" pyproject.toml returns the presets/*.yaml package-data entry; git diff on pyproject.toml touches only that block
  - grep -rn "DNA language model\|DNA Language Model" dnallm/finetune/trainer.py dnallm/configuration/configs.py returns nothing
  - grep -n "use_qlora" dnallm/configuration/configs.py shows NO new validator referencing use_ia3 (D-07)
  - CHANGELOG.md carries the (REV-01, R1-2c) entry under ## [Unreleased]
  </acceptance_criteria>
  <done>Scaffolding lands as one change: field-complete stubs registered in load_config, pyproject package-data, popped config fields, swept docstrings, CHANGELOG REV-01 evidence entry — Phase 11 agents never touch configs.py except B1's IA³ refinements.</done>
</task>

</tasks>

## Artifacts this phase produces (plan 10-01)

- `TrainingConfig.allow_test_as_eval: bool` (Pydantic field, default False)
- `TrainingConfig.use_ia3: bool` (Pydantic field, default False — field-first, no validators)
- `Ia3Config` (Pydantic model: target_modules, feedforward_modules, init_ia3_weights, modules_to_save)
- `VepConfig` (Pydantic model: paradigm, context_window, output_dir)
- `SweepConfig` (Pydantic model: seeds, out_root, n_bootstrap, bootstrap_seed, small_n_ci)
- `DNALLMConfig` keys: `ia3`, `vep`, `sweep`; `load_config()` section blocks for the three
- `DNATrainer.evaluate(split=None, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval")` (override; split path writes `output_dir/eval_{split}_result.json`)
- pyproject `[tool.setuptools.package-data]` entry `"dnallm.configuration" = ["presets/*.yaml"]`
- Test classes: TestEvalSemanticsGuard, TestEarlyStoppingCollision, TestEvaluateSplit (+ inverted test_eval_falls_back_to_test_split)

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| User YAML config -> Pydantic configs | User-supplied finetune YAML feeds TrainingConfig validation and the guard decision |
| Result JSON write path | evaluate(split=...) writes a JSON file under the user-configured output_dir |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-01 | Tampering | evaluate() result-JSON path (user-controlled output_dir) | low | accept | Local CLI/library tool: the user already owns the filesystem; JSON written with json.dump of floats/strings only, no executable content; path comes from validated config |
| T-10-02 | Denial of Service | guard/evaluate split-key lookup | low | accept | Pure dict-key checks on small key sets; no loops over unbounded input; ValueError on absent keys |
| T-10-SC | Tampering | package installs | high | accept | Phase 10 adds no dependencies (the milestone's single approved addition, scikit-allel, lands in Phase 11 B5); the only pyproject change is a package-data glob — no package-manager install tasks exist to gate |

YAML loading already uses yaml.safe_load (existing mitigation, unchanged).
</threat_model>

<verification>
- `uv run --no-sync pytest tests/finetune/test_trainer.py tests/configuration/test_configs.py -q` — all green
- `uv run --no-sync pytest tests/ -x -q` fast lane stays green (no regressions outside the lane)
- `uv run --no-sync python scripts/check_code.py` (ruff + mypy per pre-commit config) green
- `git diff origin..HEAD -- pyproject.toml` touches only the package-data block
- Cross-lane contract note (no import dependency): evaluate(split) canonical keys = current unprefixed metric spellings; plan 10-02's registry anchors exactly those spellings, so the phase verifier can cross-check evaluate() JSON keys against `metric_registry.registered_names()` after both lanes land
</verification>

<success_criteria>
- EVAL-01 fully holds at the mocked boundary: guard, one-time WARN, opt-in WARN, both collision ValueErrors, evaluate(split) predict-routing with canonical-keyed result JSON, signature compatibility, trainer docstring documenting eval-set selection and held-out semantics
- Scaffolding committed as one change (D-06): field-complete stubs, load_config registration, pyproject package-data, no dependency-list changes, no facade re-exports, no new skips
- CHANGELOG carries the (REV-01, R1-2c) entry in the same commit as the guard
</success_criteria>

<output>
Create `.planning/phases/10-evaluation-contract-layer-shared-scaffolding/10-01-SUMMARY.md` when done
</output>
