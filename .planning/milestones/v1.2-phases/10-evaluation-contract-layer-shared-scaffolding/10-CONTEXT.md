# Phase 10: Evaluation Contract Layer & Shared Scaffolding - Context

**Gathered:** 2026-10-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase lands the evaluation-semantics contract that gates the dnallmmark re-run — the trainer can never silently evaluate on the test split (EVAL-01 / REV-01), every metric resolves through one shared registry at `dnallm/tasks/metric_registry.py` (METR-01 / REV-02), and the revision docs surface is honest (DOCS-01 / REV-03) — plus the one-pass shared-file scaffolding (field-complete Ia3Config/VepConfig/SweepConfig stubs, `load_config()` section registration, pyproject package-data) and the REV-08 long-pole head start (`dnallm/inference/vep.py` core: `align_variant` same-slot rule + CLM/MLM scoring kernels + unit tests). Executed as 4 file-disjoint agents: A1 trainer.py+configs.py, A2 tasks/ registry, A3 docs, A4 new inference/vep.py.

Everything else (IA³ trainer branch, presets, probing, sweep, VEP completion, MCP tools) belongs to Phases 11–12.

</domain>

<decisions>
## Implementation Decisions

### evaluate() API design (EVAL-01 entry point)
- **D-01:** The new explicit evaluation entry point is a **`DNATrainer.evaluate(split=...)` override of the HF `Trainer.evaluate` parent method**, signature-compatible: legacy kwargs (`eval_dataset=`, `ignore_keys=`, `metric_key_prefix=`) pass through to the parent unchanged; the new `split=` kwarg routes through the predict path. Signature-compatibility tests are required so external HF-ecosystem callers (wandb sweeps etc. call `evaluate()`) are not broken. — **Reversibility:** costly — the dnallmmark re-run and E1'–E8' experiment scripts consume this surface; moving it after those exist touches every call site.
- **D-02:** `evaluate(split=...)` **returns a metrics dict keyed by metric-registry canonical names AND writes a result JSON** (containing at least: split name, timestamp, metrics). The JSON is the input contract for REV-09 `aggregate_seeds` (Phase 11) — design the schema so multi-seed aggregation can consume it directly. — **Reversibility:** costly — the REV-09 aggregation parser depends on this JSON schema.
- **D-03:** `evaluate(split=...)` evaluates the **weights the trainer currently holds** (post-`train()` semantics: best checkpoint when `load_best_model_at_end`/early stopping fired, else final-epoch weights). No extra checkpoint reloads, no optional checkpoint parameter in v1.2. The model-selection rule ("you evaluate the model you ended training with") is written into the trainer docstring.

### Guard behavior on the silent flip (EVAL-01 default change)
- **D-04:** When the guard **actively flips** a configuration that would have leaked today (test split present, no dev split) into "no evaluation", it emits a **loud WARN at flip time** — one log line covering all three: the test split was excluded from evaluation, behavior differs from previous versions, and how to explicitly opt in (`allow_test_as_eval=True`). The silent default change is never silent.
- **D-05:** `allow_test_as_eval` is a **`TrainingConfig` field** (Pydantic, `dnallm/configuration/configs.py`), YAML-configurable, default `False`, docstring stating the consequence. Landed by agent A1 together with the guard. (The WARN on explicit opt-in and the `ValueError`s on early-stopping/best-model collisions are already locked in REQUIREMENTS.md EVAL-01.)

### Scaffolding stub depth (shared-file pass)
- **D-06:** The Ia3Config/VepConfig/SweepConfig skeletons are **field-complete on landing**: final fields with defaults and known validations, `load_config()` section registration for every YAML-driven section, and pyproject package-data entries — all committed in the one-pass Phase 10 scaffolding change. Consequence: Phase 11 agents B4 (sweep) and B5 (VEP completion) never touch `configs.py`; B1 only makes IA³-specific refinements there.
- **D-07:** `TrainingConfig.use_ia3` **lands field-first in Phase 10** (default `False`, full docstring) so the TrainingConfig YAML surface is final; the cross-field rejection validators (`use_ia3 × use_qlora`, `lora × ia3`) and the trainer IA³ branch stay with agent B1 in Phase 11. The interim window (field exists, validators not yet) is repo-internal only — no release exposes it.

### Docs sweep scope & CHANGELOG evidence chain (DOCS-01)
- **D-08:** The "DNA large language models" terminology sweep is **full-surface**: docs/ non-mirror pages (~30 pages carry the old phrasing), README.md, `dnallm/` docstrings (16 files — the API reference pages are mkdocstrings-rendered, so leaving docstrings unswept keeps the old term visible on the site), and the single example/ file plus its docs/example mirror **changed as a pair** (byte-identity under `check_docs_sync.py` must hold after the sweep). Within Phase 10, the docstring sweep must avoid files owned by A1/A2/A4 in the same wave (trainer.py, configs.py, metrics.py, metric_registry.py, vep.py) — either skipped or swept by their owners.
- **D-09:** CHANGELOG entries follow the **REV-ID-inline, SHA-backfill** mechanism: each entry lands in the same commit as its fix, carrying the REV-ID and reviewer-comment number inline (e.g. `(REV-01, R1-2c)`) so the rebuttal letter can grep; the commit SHA is backfilled as a link by the Phase 12 C3 closeout agent (already roadmap-assigned "CHANGELOG finalization"). No same-commit SHA paradox, no separate evidence table to maintain.

### Claude's Discretion
- `split=` parameter key semantics (accept any split key present in the DatasetDict vs only `test`/`dev`) — recommend accepting any present key with a matchable `ValueError` otherwise.
- vep.py kernel reuse strategy (extract shared kernels from mutagenesis.py:258/312 vs self-contained adaptation inside vep.py) — planner decides under the "no refactors beyond correctness/coverage" constraint; Phase 10 A4 owns vep.py, no intra-wave collision either way.
- WARN channel (`get_logger` vs the trainer's existing `print("[Warning] ...")` style) — follow house conventions.
- Result-JSON location/naming convention; exact Ia3Config field set derivation (mirror peft `IA3Config`).
- Sequencing of the docstring sweep within A3 vs owner-side sweeps.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap (locking scope)
- `.planning/REQUIREMENTS.md` — EVAL-01 / METR-01 / DOCS-01 full acceptance criteria (this phase) plus PEFT/BASE/PROB/VEP/SEED (Phase 11, defines what scaffolding must anticipate); Out of Scope table
- `.planning/ROADMAP.md` §"Phase 10: Evaluation Contract Layer & Shared Scaffolding" — goal, 5 success criteria, 4-agent ownership map, cross-cutting invariants

### Milestone research (basis of every locked decision above)
- `.planning/research/SUMMARY.md` — four-lane synthesis: exec summary (registry placement, REV-08 long pole, neighbor-path guard), per-phase rationale incl. Phase 10 agent ownership, pitfalls index
- `.planning/research/261009-paper-revision-suite-plan.md` — the paper-revision intake (REV-01…REV-11 ↔ reviewer comments R1/R2/Ed mapping)
- `.planning/research/PITFALLS.md` — 15 pitfalls; Phase 10 avoids #1 (guard bypassed by neighbors), #2 (registry in vendored dir), #8/#9
- `.planning/research/ARCHITECTURE.md` — placement decisions (metric_registry.py sibling, new-file strategy, no `__init__.py` re-exports)
- `.planning/research/STACK.md` — zero-new-dependency evidence (peft IA3 coverage, scipy bootstrap/FDR, stdlib VCF/JASPAR readers)

### Codebase maps (v1-era, still accurate)
- `.planning/codebase/ARCHITECTURE.md` — layer responsibilities, config-dict pattern
- `.planning/codebase/TESTING.md` — fast-lane vs slow lane, typed-skip discipline, coverage gate mechanics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- HF `Trainer.evaluate()` parent (signature to stay compatible with: `eval_dataset`, `ignore_keys`, `metric_key_prefix`) — D-01 passthrough target
- `DNATrainer.compute_task_metrics()` + `dnallm/tasks/metrics.py` compute_* factories — the metric-emission surface the registry rewires (METR-01)
- `dnallm/inference/mutagenesis.py:258,312` — CLM/MLM scoring kernels the vep.py core reuses (A4)
- `LoraConfig` in `dnallm/configuration/configs.py` — the Pydantic-section precedent every stub follows; `load_config()` section dispatch is where new sections register
- `dnallm/utils/logger.py get_logger` — WARN channel; note trainer.py currently mixes `print("[Warning] ...")` (house tension, planner picks per D-05/D-04)
- `tests/expected_skips.yaml` + `scripts/audit_skips.py` — typed-skip gate any new skip must satisfy same-change
- `scripts/check_docs_sync.py` — byte-identity guard between example/ and docs/example/ that the D-08 pair-sweep must keep green

### Established Patterns
- All tunables live in Pydantic `BaseModel`s in `dnallm/configuration/configs.py` with `Field(default=..., description=...)`; YAML loads via `load_config()` — stubs and `allow_test_as_eval` follow this exactly
- `ValueError` with descriptive matchable messages; tests assert via `pytest.raises(..., match=...)`
- `[tool.coverage.run] omit` excludes the vendored `dnallm/tasks/metrics/` glob — the registry must sit outside it, with a same-change coverage-row proof (success criterion 3)
- Coverage working standard ≥96% per new module via mocked fast-lane tests; `slow`-marked real-model tests are acceptance-only

### Integration Points
- `dnallm/finetune/trainer.py:236-241` — the leak site (falls back to `dataset["test"]` as eval when no dev split); guard replaces this block
- `dnallm/finetune/trainer.py:298-303` — early-stopping path that force-enables `load_best_model_at_end` (the collision the guard must raise on)
- `dnallm/models/model.py` / `dnallm/inference/inference.py:111-131` — PeftModel save/reload path the IA³ branch will share in Phase 11 (scaffolding must not pre-wire it)
- `dnallm/inference/__init__.py` and `dnallm/tasks/__init__.py` — module registration points for vep.py / metric_registry.py
- `pyproject.toml [tool.setuptools.package-data]` — package-data entry for `dnallm/configuration/presets/` lands in the scaffolding pass
- docs API pages render docstrings via mkdocstrings — the D-08 docstring sweep is what actually fixes the visible site terminology

</code_context>

<specifics>
## Specific Ideas

- Model-selection rule wording for the trainer docstring: "evaluate(split=...) evaluates the model you ended training with" — best checkpoint if early stopping/best-model loading fired, otherwise final-epoch weights.
- The flip-time WARN (D-04) must say three things in one line: test split excluded from evaluation; this differs from previous dnallm behavior; set `allow_test_as_eval: true` to evaluate on test explicitly.
- CHANGELOG entry inline tag format: `(REV-XX, <reviewer-comment-id>)` — greppable for the rebuttal letter.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 10-Evaluation Contract Layer & Shared Scaffolding*
*Context gathered: 2026-10-09*
